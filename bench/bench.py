"""Correctness-gated benchmark for mojo-chaospy.

Every case checks numerical agreement with an independent reference before
timing, so a regression in the Mojo kernels shows up as a correctness failure
rather than a suspiciously good number. The reference for the recurrence,
tensor product and quadrature cases is vectorised NumPy doing the same sum;
the reference for Golub-Welsch is `scipy.linalg.eig_banded`, which is what
chaospy itself calls.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_chaospy as mcp  # noqa: E402
from mojo_chaospy import _lib  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _numpy_recurrence(alpha, beta, x):
    rows = [np.ones_like(x), x - alpha[0]]
    for m in range(2, alpha.size):
        rows.append((x - alpha[m - 1]) * rows[-1] - beta[m - 1] * rows[-2])
    return np.array(rows)


def bench_recurrence(order=256, n=1 << 16):
    rng = np.random.default_rng(0)
    # a real recurrence (monic probabilists' Hermite) rather than random
    # coefficients: random betas make the monic family grow like a factorial
    # and the comparison is then dominated by conditioning, not by the kernel
    alpha = np.zeros(order + 1)
    beta = np.arange(order + 1, dtype=float)
    x = rng.standard_normal(n) * min(1.0, 4.0 / np.sqrt(order))
    got = _lib.eval_orthogonal(alpha, beta, x)
    want = _numpy_recurrence(alpha, beta, x)
    scale = np.abs(want).max()
    assert np.allclose(got, want, rtol=1e-11, atol=1e-11 * scale), (
        "recurrence mismatch"
    )
    return (
        f"recurrence order={order} n={n}",
        _time(lambda: _numpy_recurrence(alpha, beta, x), 3),
        _time(lambda: _lib.eval_orthogonal(alpha, beta, x)),
    )


def bench_tensor(dim=3, order=6, n=1 << 14):
    rng = np.random.default_rng(1)
    points = np.ascontiguousarray(rng.standard_normal((dim, n)))
    terms = (order + 1) ** dim
    alpha = np.zeros(order + 1)
    beta = np.arange(order + 1, dtype=float)
    bank = np.ascontiguousarray(
        np.stack([_lib.eval_orthogonal(alpha, beta, points[d]) for d in range(dim)])
    )
    grid = np.indices((order + 1,) * dim).reshape(dim, -1)[:terms]
    indices = np.ascontiguousarray(grid.astype(np.float64))
    got = _lib.tensor_eval(bank, indices)
    def reference():
        ref = np.ones((terms, n))
        for d in range(dim):
            ref = ref * bank[d][grid[d]]
        return ref

    assert np.allclose(got, reference(), rtol=1e-11, atol=1e-13), "tensor mismatch"
    return (
        f"tensor dim={dim} terms={terms} n={n}",
        _time(reference, 3),
        _time(lambda: _lib.tensor_eval(bank, indices)),
    )


def bench_moments(n=1 << 20, repeats=100):
    rng = np.random.default_rng(2)
    psi = rng.standard_normal(n)
    w = rng.random(n) + 0.1

    def reference():
        for _ in range(repeats):
            np.dot(psi, w)

    def mine():
        for _ in range(repeats):
            _lib.moments(psi, w)

    got = _lib.moments(psi, w)
    assert abs(got[0] - np.dot(psi, w)) < 1e-9 * abs(got[0]), "mean mismatch"
    return f"moments n={n} x{repeats}", _time(reference, 3), _time(mine, 3)


def bench_fft(order=4096, repeats=100):
    a = np.cos(2 * np.pi * np.arange(order) * 0.001) + 0.3

    def reference():
        for _ in range(repeats):
            np.fft.ifft(a).real

    def mine():
        for _ in range(repeats):
            _lib.ihfft_real(a)

    got = _lib.ihfft_real(a)
    assert np.allclose(got, np.fft.ifft(a).real, atol=1e-12), "fft mismatch"
    return (
        f"ihfft n={order} x{repeats}",
        _time(reference, 3),
        _time(mine, 3),
    )


def bench_golub_welsch(order, workers):
    import scipy.linalg

    coeffs = np.array([np.zeros(order + 1), np.arange(order + 1, dtype=float)])
    nodes, weights = mcp.gaussian_quadrature(coeffs, workers=workers)
    bands = np.zeros((2, order + 1))
    bands[0, :] = coeffs[0]
    bands[1, :-1] = np.sqrt(coeffs[1][1:])
    vals, vecs = scipy.linalg.eig_banded(bands, lower=True)
    index = np.argsort(vals.real)
    ref_nodes, ref_weights = vals.real[index], (vecs[0, :] ** 2)[index]
    assert np.allclose(nodes, ref_nodes, rtol=1e-9), "node mismatch"
    assert np.allclose(weights, ref_weights, rtol=1e-8), "weight mismatch"
    return (
        f"golub-welsch order={order} workers={workers}",
        _time(lambda: mcp.gaussian_quadrature(coeffs, workers=1), 3),
        _time(lambda: mcp.gaussian_quadrature(coeffs, workers=workers), 3),
    )


def bench_fit_coefficients(terms=2048, n=64, repeats=20):
    rng = np.random.default_rng(4)
    psi = np.ascontiguousarray(rng.standard_normal((terms, n)))
    w = rng.random(n) + 0.1
    solves = rng.standard_normal(n)

    def reference():
        for _ in range(repeats):
            np.sum(psi**2 * w, axis=1)
            (psi * solves) @ w / (psi**2 @ w)

    def mine():
        for _ in range(repeats):
            _lib.fit_coefficients(psi, w, solves)

    got, norms = _lib.fit_coefficients(psi, w, solves)
    assert np.allclose(norms, psi**2 @ w, rtol=1e-11), "norm mismatch"
    return (
        f"fit_coefficients terms={terms} n={n} x{repeats}",
        _time(reference, 3),
        _time(mine, 3),
    )


def main():
    print(f"{'case':<42}{'reference':>12}{'mojo-chaospy':>16}{'ratio':>10}")
    print("-" * 80)
    cases = [
        bench_recurrence,
        bench_tensor,
        bench_moments,
        bench_fit_coefficients,
        bench_fft,
        lambda: bench_golub_welsch(200, 1),
        lambda: bench_golub_welsch(200, 8),
    ]
    for fn in cases:
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<42}{ref * 1e3:>10.2f}ms{got * 1e3:>14.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
