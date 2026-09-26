"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below stay `c_int64` for addresses; `c_int` truncates
them and segfaults. Index tables are passed as `float64` buffers (the kernel
reads them through the same pointer type and converts), which keeps every
exported symbol down to a single buffer type.
"""

from __future__ import annotations

import ctypes
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_BAD_BETA = (
    "negative beta in recurrence coefficients: cannot build a Gaussian "
    "quadrature rule"
)

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-chaospy.so"

_A = ctypes.c_int64
_F = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.cp_eval_orthogonal.restype = None
    lib.cp_eval_orthogonal.argtypes = [_A, _A, _A, _A, _A, _A]

    lib.cp_tensor_eval.restype = None
    lib.cp_tensor_eval.argtypes = [_A] * 8

    lib.cp_dot_weighted.restype = None
    lib.cp_dot_weighted.argtypes = [_A, _A, _A, _A, _A]

    lib.cp_moments.restype = None
    lib.cp_moments.argtypes = [_A, _A, _A, _A]

    lib.cp_fit_coefficients.restype = None
    lib.cp_fit_coefficients.argtypes = [_A, _A, _A, _A, _A, _A, _A]

    lib.cp_bisect_eigenvalues.restype = ctypes.c_int
    lib.cp_bisect_eigenvalues.argtypes = [_A] * 7

    lib.cp_golub_weights.restype = ctypes.c_int
    lib.cp_golub_weights.argtypes = [_A, _A, _A, _A, _A, _A, _A, _A]

    lib.cp_ihfft_real.restype = ctypes.c_int
    lib.cp_ihfft_real.argtypes = [_A] * 7

    lib.cp_clenshaw_curtis.restype = ctypes.c_int
    lib.cp_clenshaw_curtis.argtypes = [_A] * 9

    lib.cp_stieltjes_iterate.restype = _F
    lib.cp_stieltjes_iterate.argtypes = [_A] * 10
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def fft_scratch(n: int) -> list[np.ndarray]:
    """Four real buffers of the Bluestein working length for a length-`n` FFT."""
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    return [np.zeros(m, dtype=np.float64) for _ in range(4)]


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def eval_orthogonal(
    alpha: np.ndarray, beta: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Evaluate P_0..P_order at each row of `x`; result shape (order+1, n)."""
    alpha = _f64(alpha)
    beta = _f64(beta)
    x = _f64(x)
    order = alpha.size - 1
    out = np.empty((order + 1, x.size), dtype=np.float64)
    lib.cp_eval_orthogonal(
        _addr(alpha), _addr(beta), order, _addr(x), x.size, _addr(out)
    )
    return out


def tensor_eval(bank: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Contract a `(dim, order+1, n)` bank into a `(terms, n)` expansion.

    `indices` is `(dim, terms)`.
    """
    bank = _f64(bank)
    indices = _f64(indices)
    dim, stride, n = bank.shape
    terms = indices.shape[1]
    out = np.empty((terms, n), dtype=np.float64)
    acc = np.empty(n, dtype=np.float64)
    lib.cp_tensor_eval(
        _addr(bank), _addr(indices), dim, terms, n, stride, _addr(acc),
        _addr(out),
    )
    return out


def dot_weighted(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a = _f64(a)
    b = _f64(b)
    w = _f64(w)
    out = np.zeros(1, dtype=np.float64)
    lib.cp_dot_weighted(_addr(a), _addr(b), _addr(w), a.size, _addr(out))
    return float(out[0])


def moments(psi: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, second moment, variance, total weight) of `psi` under `w`."""
    psi = _f64(psi)
    w = _f64(w)
    out = np.zeros(4, dtype=np.float64)
    lib.cp_moments(_addr(psi), _addr(w), psi.size, _addr(out))
    return tuple(float(v) for v in out)  # type: ignore[return-value]


def fit_coefficients(
    psi: np.ndarray, w: np.ndarray, solves: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral projection norms and Fourier coefficients."""
    psi = _f64(psi)
    w = _f64(w)
    solves = _f64(solves)
    terms, n = psi.shape
    norms = np.empty(terms, dtype=np.float64)
    coeffs = np.empty(terms, dtype=np.float64)
    lib.cp_fit_coefficients(
        _addr(psi), _addr(w), _addr(solves), terms, n, _addr(norms),
        _addr(coeffs),
    )
    return coeffs, norms


def bisect_eigenvalues_range(
    alpha: np.ndarray,
    beta: np.ndarray,
    k0: int,
    k1: int,
    e: np.ndarray,
    vals: np.ndarray,
) -> int:
    """Fill `vals[k0:k1]`, for callers fanning the loop across threads."""
    return lib.cp_bisect_eigenvalues(
        _addr(_f64(alpha)), _addr(_f64(beta)), alpha.size, k0, k1, _addr(e),
        _addr(vals),
    )


def bisect_eigenvalues(
    alpha: np.ndarray, beta: np.ndarray, workers: int = 1,
    worker_threshold: int = 48,
) -> np.ndarray:
    """Ascending eigenvalues of the Jacobi matrix, or raise on an illegal beta."""
    alpha = _f64(alpha)
    beta = _f64(beta)
    n = alpha.size
    vals = np.empty(n, dtype=np.float64)
    if n < worker_threshold or workers <= 1:
        e = np.empty(n, dtype=np.float64)
        rc = bisect_eigenvalues_range(alpha, beta, 0, n, e, vals)
        if rc != 0:
            raise np.linalg.LinAlgError(_BAD_BETA)
        return vals

    step = -(-n // workers)

    def part(k: int) -> None:
        lo = k * step
        hi = min(lo + step, n)
        if lo < hi:
            rc = bisect_eigenvalues_range(
                alpha, beta, lo, hi, np.empty(n, dtype=np.float64), vals
            )
            if rc != 0:
                raise np.linalg.LinAlgError(_BAD_BETA)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(part, range(workers)))
    return vals


def golub_weights(
    alpha: np.ndarray, beta: np.ndarray, vals: np.ndarray
) -> np.ndarray:
    """Gaussian weights for the given Jacobi spectrum."""
    alpha = _f64(alpha)
    beta = _f64(beta)
    vals = _f64(vals)
    n = alpha.size
    scratch = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    rc = lib.cp_golub_weights(
        _addr(alpha), _addr(beta), n, _addr(vals), 0, n, _addr(scratch),
        _addr(weights),
    )
    if rc != 0:
        raise np.linalg.LinAlgError("Golub-Welsch weight step failed")
    return weights


def golub_weights_range(
    alpha: np.ndarray,
    beta: np.ndarray,
    vals: np.ndarray,
    k0: int,
    k1: int,
    scratch: np.ndarray,
    weights: np.ndarray,
) -> int:
    """Fill `weights[k0:k1]`, for callers fanning the loop across threads."""
    return lib.cp_golub_weights(
        _addr(_f64(alpha)), _addr(_f64(beta)), vals.size, _addr(_f64(vals)),
        k0, k1, _addr(scratch), _addr(weights),
    )


def ihfft_real(a: np.ndarray) -> np.ndarray:
    """Real part of the inverse DFT of a real sequence, scaled by 1/n."""
    a = _f64(a)
    n = a.size
    out = np.empty(n, dtype=np.float64)
    re, im, re2, im2 = fft_scratch(n)
    rc = lib.cp_ihfft_real(
        _addr(a), n, _addr(out), _addr(re), _addr(im), _addr(re2), _addr(im2)
    )
    if rc != 0:
        raise RuntimeError("inverse FFT rejected its scratch buffers")
    return out


def clenshaw_curtis(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Clenshaw-Curtis nodes and weights on [0, 1]."""
    if order < 0:
        raise ValueError("order must be non-negative")
    x = np.zeros(order + 1, dtype=np.float64)
    w = np.zeros(order + 1, dtype=np.float64)
    a = np.zeros(max(order, 1), dtype=np.float64)
    spec = np.zeros(order + 1, dtype=np.float64)
    re, im, re2, im2 = fft_scratch(max(order, 1))
    count = lib.cp_clenshaw_curtis(
        order, _addr(x), _addr(w), _addr(a), _addr(spec), _addr(re), _addr(im),
        _addr(re2), _addr(im2),
    )
    if count < 0:
        raise RuntimeError("Clenshaw-Curtis construction failed")
    return x[:count].copy(), w[:count].copy()


def stieltjes_iterate(
    x: np.ndarray,
    w: np.ndarray,
    order: int,
    alpha_in: np.ndarray,
    beta_in: np.ndarray,
    norms: np.ndarray,
    alpha_out: np.ndarray,
    beta_out: np.ndarray,
) -> float:
    """One discretized-Stieltjes sweep; returns max |delta beta|."""
    x = _f64(x)
    w = _f64(w)
    pbuf = np.empty(3 * x.size, dtype=np.float64)
    return lib.cp_stieltjes_iterate(
        _addr(x), _addr(w), x.size, order, _addr(alpha_in), _addr(beta_in),
        _addr(norms), _addr(alpha_out), _addr(beta_out), _addr(pbuf),
    )
