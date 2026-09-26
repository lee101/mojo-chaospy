"""Polynomial chaos expansion API built on the Mojo kernels.

Everything with a loop over collocation points runs in `libmojo-chaospy.so`.
This module owns the arrays, the expansion index set, and the two pieces of
policy that are not numeric work: the order-growth/convergence loop of the
discretized Stieltjes procedure, and the `numpy.linalg.lstsq` solve inside
`fit_regression`.

The design mirrors `chaospy` deliberately so results can be compared directly
against the real package.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _lib

__all__ = [
    "FirstOrderSobol",
    "TotalOrderSobol",
    "basis",
    "clenshaw_curtis",
    "construct_recurrence_coefficients",
    "evaluate",
    "expected",
    "expected_expansion",
    "fit_quadrature",
    "fit_regression",
    "gaussian_quadrature",
    "generate_indices",
    "recurrence_evaluate",
    "std",
    "variance",
]


# ---------------------------------------------------------------------------
# expansion index set
# ---------------------------------------------------------------------------


def generate_indices(
    dim: int,
    order: int,
    graded: bool = True,
    reverse: bool = True,
    cross_truncation: float = 1.0,
) -> np.ndarray:
    """Tensor-product / hyperbolic-cross exponent set, shape `(terms, dim)`.

    Matches `chaospy.generate_expansion`'s default ordering: graded by the sum
    of exponents, then lexicographic, which is what the real package's
    `graded=True, reverse=True` produces.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    grid = np.indices((order + 1,) * dim).reshape(dim, -1).T
    if graded:
        keep = grid.sum(axis=1) <= order
        grid = grid[keep]
        if np.isfinite(cross_truncation):
            weight = grid ** (1.0 / cross_truncation)
            grid = grid[weight.sum(axis=1) ** cross_truncation <= order + 1e-12]
    if not reverse:
        grid = grid[:, ::-1]
    # graded by exponent sum, ties broken lexicographically on the exponents
    keys = tuple(grid[:, d] for d in range(dim - 1, -1, -1))
    if graded:
        keys = keys + (grid.sum(axis=1),)
    order_key = np.lexsort(keys)
    return np.ascontiguousarray(grid[order_key], dtype=np.int64)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------




def recurrence_evaluate(
    coeffs: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Evaluate an orthogonal polynomial family at `points`; `(order+1, n)`."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim == 3:
        coeffs = coeffs[0]
    return _lib.eval_orthogonal(coeffs[0], coeffs[1], np.asarray(points, float))


def basis(
    indices: np.ndarray, coeffs: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Chaos expansion basis at `points`; shape `(terms, n)`.

    `coeffs` is the `(2, order+1)` three-term recurrence of each dimension, or
    a list of one such array per dimension.
    """
    indices = np.asarray(indices, dtype=np.int64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    dim, n = points.shape
    coeff_list = _as_coeff_list(coeffs, dim)
    bank = np.ascontiguousarray(
        np.stack([_lib.eval_orthogonal(c[0], c[1], points[d]) for d, c in
                  enumerate(coeff_list)])
    )
    return _lib.tensor_eval(bank, indices.T.astype(np.float64))


def _as_coeff_list(coeffs, dim: int) -> list[np.ndarray]:
    out = []
    for entry in coeffs:
        entry = np.asarray(entry, dtype=np.float64)
        if entry.ndim == 3:
            entry = entry[0]
        if entry.shape[0] != 2:
            raise ValueError("recurrence coefficients must have shape (2, m)")
        out.append(entry)
    if len(out) == 1 and dim > 1:
        out = out * dim
    if len(out) != dim:
        raise ValueError(f"expected {dim} recurrence coefficient sets")
    return out


def evaluate(
    indices: np.ndarray,
    coeffs: np.ndarray,
    points: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate the expansion `sum_m coefficients[m] * Psi_m(points)`.

    `points` is `(dim, n)`, the result is `(n,)`, or `(n, k)` when
    `coefficients` carries one column per model output.
    """
    indices = np.asarray(indices, dtype=np.int64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    psi = basis(indices, coeffs, points)
    if coefficients.ndim == 1:
        return np.einsum("mn,m->n", psi, coefficients)
    return psi.T @ coefficients


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit_regression(
    indices: np.ndarray, coeffs: np.ndarray, abscissas: np.ndarray,
    evals: np.ndarray,
) -> np.ndarray:
    """Least-squares fit of the expansion to model evaluations.

    The design matrix is built by the Mojo kernels; the normal-equation solve
    is left to `numpy.linalg.lstsq`, which is a well-conditioned LAPACK call
    rather than something worth reimplementing.
    """
    indices = np.asarray(indices, dtype=np.int64)
    evals = np.asarray(evals, dtype=np.float64)
    psi = basis(indices, coeffs, np.atleast_2d(abscissas))
    if evals.ndim == 1:
        uhat, *_ = np.linalg.lstsq(psi.T, evals, rcond=None)
    else:
        uhat, *_ = np.linalg.lstsq(psi.T, evals.reshape(len(evals), -1),
                                   rcond=None)
    return uhat


def fit_quadrature(
    indices: np.ndarray,
    coeffs: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
    solves: np.ndarray,
    norms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral projection: exact Fourier coefficients on a quadrature rule.

    Returns `(coefficients, norms)`, matching `chaospy.fit_quadrature` with
    `retall`.
    """
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    solves = np.ascontiguousarray(solves, dtype=np.float64)
    psi = basis(indices, coeffs, np.atleast_2d(nodes))
    if solves.ndim == 1:
        fourier, computed = _lib.fit_coefficients(psi, weights, solves)
    else:
        flat = solves.reshape(len(solves), -1)
        fourier, computed = _lib.fit_coefficients(psi, weights, flat[:, 0])
        fourier = fourier[:, None]
        for column in range(1, flat.shape[1]):
            more, _ = _lib.fit_coefficients(psi, weights, flat[:, column])
            fourier = np.concatenate([fourier, more[:, None]], axis=1)
        fourier = fourier.reshape(-1, *solves.shape[1:])
    norms = computed if norms is None else np.asarray(norms, dtype=np.float64)
    return fourier, norms


# ---------------------------------------------------------------------------
# descriptive quantities
# ---------------------------------------------------------------------------


def expected(values: np.ndarray, weights: np.ndarray) -> float:
    """Quadrature mean of a basis function."""
    return _lib.moments(values, weights)[0]


def variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Quadrature variance of a basis function."""
    return _lib.moments(values, weights)[2]


def std(values: np.ndarray, weights: np.ndarray) -> float:
    """Quadrature standard deviation of a basis function."""
    return float(np.sqrt(_lib.moments(values, weights)[2]))


def expected_expansion(
    indices: np.ndarray, coeffs: np.ndarray, nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Expectation of every basis function, shape `(terms,)`."""
    psi = basis(indices, coeffs, np.atleast_2d(nodes))
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    ones = np.ones(psi.shape[1], dtype=np.float64)
    return np.array(
        [_lib.dot_weighted(row, ones, weights) for row in psi],
        dtype=np.float64,
    )


def _sobol(indices: np.ndarray, coefficients: np.ndarray, total: bool):
    indices = np.asarray(indices, dtype=np.int64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    active = indices.any(axis=1)
    power = coefficients[active] ** 2
    total_variance = power.sum(axis=0)
    out = np.empty((indices.shape[1], coefficients.shape[1]))
    for d in range(indices.shape[1]):
        if total:
            mask = indices[:, d] > 0
        else:
            mask = (indices[:, d] > 0) & (indices.sum(axis=1) == indices[:, d])
        out[d] = coefficients[mask] ** 2
    return out.sum(axis=0) / total_variance


def FirstOrderSobol(
    indices: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """First-order Sobol indices, from a fitted expansion's coefficients."""
    return _sobol(indices, coefficients, total=False)


def TotalOrderSobol(
    indices: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Total-effect Sobol indices, from a fitted expansion's coefficients."""
    return _sobol(indices, coefficients, total=True)


# ---------------------------------------------------------------------------
# quadrature rules
# ---------------------------------------------------------------------------


def clenshaw_curtis(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Clenshaw-Curtis abscissas and weights on [0, 1]."""
    return _lib.clenshaw_curtis(order)


def gaussian_quadrature(
    coeffs: np.ndarray, workers: int = 1, worker_threshold: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian quadrature nodes and weights from recurrence coefficients.

    This is the Golub-Welsch step: build the Jacobi matrix, take its
    eigenvalues, then recover the weight of each eigenvalue from the first
    component of its eigenvector. Both loops are embarrassingly parallel over
    the eigenvalue index, so `workers > 1` fans them out over a thread pool.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim == 3:
        coeffs = coeffs[0]
    alpha = np.ascontiguousarray(coeffs[0])
    beta = np.ascontiguousarray(coeffs[1])
    n = alpha.size
    vals = _lib.bisect_eigenvalues(
        alpha, beta, workers=workers, worker_threshold=worker_threshold
    )
    if n < worker_threshold or workers <= 1:
        return vals, _lib.golub_weights(alpha, beta, vals)

    weights = np.empty(n, dtype=np.float64)
    step = -(-n // workers)

    def part(k: int) -> None:
        lo = k * step
        hi = min(lo + step, n)
        if lo < hi:
            # the eigenvector recurrence scratch is per worker, not shared
            _lib.golub_weights_range(
                alpha, beta, vals, lo, hi, np.empty(n, dtype=np.float64),
                weights,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(part, range(workers)))
    return vals, weights


def _segmented_clenshaw_curtis(
    order: int, segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Clenshaw-Curtis on [0, 1] split into `segments` equal subintervals.

    Reproduces `chaospy.quadrature.hypercube.split_into_segments`: each
    subinterval gets a rule of near-equal order, the shared boundary node is
    collapsed and its two weights added. The per-segment rule itself is the
    Mojo kernel; this is only the bookkeeping.
    """
    if segments == 1 or order <= 2:
        nodes, weights = _lib.clenshaw_curtis(order)
        return nodes, weights
    edges = np.linspace(0.0, 1.0, segments + 1)
    abscissas = []
    weights = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        local = order // segments + (1 if idx < order % segments else 0)
        node, weight = _lib.clenshaw_curtis(int(local))
        weight = weight * (hi - lo)
        node = node * (hi - lo) + lo
        if abscissas and abs(abscissas[-1][-1] - lo) < 1e-14:
            weights[-1][-1] += weight[0]
            node = node[1:]
            weight = weight[1:]
        abscissas.append(node)
        weights.append(weight)
    return np.hstack(abscissas), np.hstack(weights)


def _proxy_rule(
    order: int, lower, upper, pdf, rule: str, segments: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """`chaospy`'s proxy integration rule for one Stieltjes sweep.

    `segments=0` reproduces chaospy's own choice of `int(sqrt(order))`.
    """
    if rule != "clenshaw_curtis":
        raise ValueError(f"unsupported proxy rule {rule!r}")
    lower = float(np.ravel(lower)[0])
    upper = float(np.ravel(upper)[0])
    if not segments:
        segments = int(np.sqrt(order))
    nodes, weights = _segmented_clenshaw_curtis(int(order), int(segments))
    abscissas = (upper - lower) * nodes + lower
    weights = weights * (upper - lower)
    eps = 1e-14 * (upper - lower)
    clipped = np.clip(abscissas, lower + eps, upper - eps)
    weighted = weights * np.asarray(pdf(clipped), dtype=np.float64)
    total = weighted.sum() * (upper - lower)
    if total == 0.0:
        raise ValueError("proxy rule has zero total weight")
    return abscissas, weighted * weights.sum() / total


def construct_recurrence_coefficients(
    order: int,
    lower: float,
    upper: float,
    pdf,
    rule: str = "clenshaw_curtis",
    tolerance: float = 1e-10,
    scaling: int = 3,
    n_max: int = 5000,
) -> np.ndarray:
    """Discretized Stieltjes recurrence coefficients, shape `(2, order+1)`.

    Same policy as `chaospy.recurrence.discretized_stieltjes`: the proxy
    quadrature order grows geometrically until the second recurrence row stops
    moving. All the O(order * nodes) work is in the Mojo kernel.
    """
    order = int(order)
    # Two coefficient buffers, alternated: the kernel measures convergence
    # against its own input row, so the two must not be the same memory.
    state = (
        (np.ones(order + 1, dtype=np.float64), np.ones(order + 1, dtype=np.float64)),
        (np.empty(order + 1, dtype=np.float64), np.empty(order + 1, dtype=np.float64)),
    )
    norms = np.ones(order + 2, dtype=np.float64)

    order_ = (2.0 * order - 1.0) / scaling
    first = True
    while True:
        order_ = max(order_ * scaling, order_ + 1.0)
        if order_ > n_max:
            break
        abscissas, weights = _proxy_rule(int(order_), lower, upper, pdf, rule)
        (prev_alpha, prev_beta), (next_alpha, next_beta) = state
        delta = _lib.stieltjes_iterate(
            abscissas, weights, order, prev_alpha, prev_beta, norms, next_alpha,
            next_beta,
        )
        state = (state[1], state[0])
        if not first and delta < tolerance:
            break
        first = False
    return np.stack(state[0])
