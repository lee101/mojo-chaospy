"""Parity against the real `chaospy`, plus analytic checks on the kernels.

Every tolerance here is justified: Mojo emits FMA, so products inside a
recurrence or a weighted sum are fused and cannot match NumPy bit for bit. The
recurrence evaluation, the Clenshaw-Curtis weights and the Golub-Welsch
eigenvalues are different: the first two are dominated by `cos`/`sin` and one
fused multiply-add, and the last is a bisection, so those are checked tight.
"""

import numpy as np
import pytest

import chaospy

import mojo_chaospy as mcp
from mojo_chaospy import _lib

# Distributions whose three-term recurrence is well conditioned. The
# ill-conditioned ones (a log-normal with a large order, say) overflow inside
# scipy's own eigensolver, so comparing there would measure nothing.
DISTS = [
    chaospy.Uniform(0, 1),
    chaospy.Uniform(-2, 3),
    chaospy.Normal(0, 1),
    chaospy.LogNormal(0, 0.5),
    chaospy.Gamma(1, 1),
    chaospy.Beta(2, 5),
]

# Distributions with no analytic recurrence, so chaospy also has to run the
# discretized Stieltjes procedure.
NO_TTR = [
    chaospy.Uniform(0, 1),
    chaospy.Beta(2, 5),
    chaospy.Gamma(2, 1),
    chaospy.Triangle(-1, 0.5, 2),
    chaospy.Weibull(1.5, 1),
]


def _rows(values, shape):
    """numpoly returns the expansion values transposed in some versions."""
    values = np.asarray(values)
    if values.shape == shape:
        return values
    assert values.T.shape == shape, f"{values.shape} cannot match {shape}"
    return values.T


def _ttr(dist, order):
    return [
        np.asarray(c, dtype=float).reshape(2, order + 1)
        for c in chaospy.construct_recurrence_coefficients(
            order, dist, recurrence_algorithm="stieltjes",
            rule="clenshaw_curtis", tolerance=1e-16,
        )
    ]


# --------------------------------------------------------------------------
# three-term recurrence evaluation
# --------------------------------------------------------------------------


def test_recurrence_matches_numpy_step_by_step():
    """A plausible bug: a wrong starting row or a swapped alpha/beta."""
    alpha = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    beta = np.array([1.0, 1 / 12, 2 / 15, 9 / 56, 4 / 35])
    x = np.linspace(-3.0, 4.0, 33)
    got = _lib.eval_orthogonal(alpha, beta, x)
    ref = [np.ones_like(x), x - alpha[0]]
    for m in range(2, alpha.size):
        ref.append(
            (x - alpha[m - 1]) * ref[-1] - beta[m - 1] * ref[-2]
        )
    np.testing.assert_allclose(got, np.array(ref), rtol=1e-13, atol=1e-13)


def test_recurrence_legendre_is_exact_polynomial():
    """Monic Legendre on [-1, 1]: p_0=1, p_1=x, p_2=x^2-1/3, p_3=x^3-3x/5.

    Checking the closed form catches a wrong beta indexing, which a
    self-consistent recurrence test cannot see.
    """
    order = 3
    alpha = np.zeros(order + 1)
    beta = np.array([1.0, 1.0 / 3.0, 4.0 / 15.0, 9.0 / 35.0])
    x = np.linspace(-0.99, 0.99, 21)
    got = _lib.eval_orthogonal(alpha, beta, x)
    np.testing.assert_allclose(got[2], x**2 - 1.0 / 3.0, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(got[3], x**3 - 0.6 * x, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(got[1], x, rtol=0, atol=0)
    np.testing.assert_allclose(got[0], np.ones_like(x), rtol=0, atol=0)


def test_recurrence_hermite_against_closed_form():
    """Probabilists' Hermite: H_0=1, H_1=x, H_n = x H_{n-1} - (n-1) H_{n-2}."""
    order = 4
    alpha = np.zeros(order + 1)
    beta = np.arange(order + 1, dtype=float)
    x = np.linspace(-2.0, 2.0, 17)
    got = _lib.eval_orthogonal(alpha, beta, x)
    ref = [np.ones_like(x), x]
    for n in range(2, order + 1):
        ref.append(x * ref[-1] - (n - 1) * ref[-2])
    np.testing.assert_allclose(got, np.array(ref), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dist", DISTS)
def test_recurrence_matches_chaospy_expansion(dist):
    """The univariate family must match the polynomials chaospy builds."""
    order = 6
    coeffs = _ttr(dist, order)
    x = np.linspace(
        float(np.ravel(dist.lower)[0]), float(np.ravel(dist.upper)[0]), 19
    )
    got = _lib.eval_orthogonal(coeffs[0][0], coeffs[0][1], x)
    reference = chaospy.generate_expansion(order, dist)
    expected = np.array([np.asarray(poly(x)) for poly in reference])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# tensor product / evaluation / fitting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4])
def test_tensor_eval_matches_naive_product(order):
    """A wrong index transpose or a wrong per-dimension offset shows here."""
    dim, n, terms = 3, 11, 5
    rng = np.random.default_rng(5)
    points = np.ascontiguousarray(rng.standard_normal((dim, n)))
    alpha = rng.standard_normal(order + 1) * 0.3
    beta = np.abs(rng.standard_normal(order + 1)) + 0.5
    bank = np.ascontiguousarray(
        np.stack([_lib.eval_orthogonal(alpha, beta, points[d])
                  for d in range(dim)])
    )
    indices = rng.integers(0, order + 1, size=(dim, terms))
    got = _lib.tensor_eval(bank, indices.astype(np.float64))
    ref = np.ones((terms, n))
    for d in range(dim):
        ref = ref * bank[d][indices[d]]
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_expansion_values_match_chaospy():
    """basis()/evaluate() against numpoly's own evaluation of the expansion."""
    rng = np.random.default_rng(3)
    dist = chaospy.J(chaospy.Uniform(-1, 2), chaospy.Normal(0, 1))
    order = 4
    reference = chaospy.generate_expansion(order, dist)
    indices = np.asarray(reference.exponents, dtype=np.int64)
    coeffs = _ttr(dist, order)
    points = np.ascontiguousarray(rng.uniform(-1, 2, (2, 9)))

    got = mcp.basis(indices, coeffs, points)
    theirs = np.asarray(reference(*points))
    assert got.shape == theirs.shape == (len(indices), points.shape[1])

    # numpoly lists its polynomials in sparse-key order, chaospy's own
    # `fit_*` in graded order, so the two are the same rows in some
    # permutation. Recover it and require it to be a bijection.
    match = [int(np.argmin(np.abs(theirs - row).max(axis=1))) for row in got]
    assert sorted(match) == list(range(len(indices)))
    np.testing.assert_allclose(
        theirs[match], got, rtol=1e-10, atol=1e-12
    )

    term = 3
    coefficients = np.zeros(len(indices))
    coefficients[term] = 1.5
    np.testing.assert_allclose(
        mcp.evaluate(indices, coeffs, points, coefficients),
        1.5 * np.asarray(theirs[match[term]]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_evaluate_multiple_outputs_is_a_matrix_product():
    rng = np.random.default_rng(9)
    dist = chaospy.Normal(0, 1)
    order = 3
    indices = np.asarray(chaospy.generate_expansion(order, dist).exponents, dtype=np.int64)
    coeffs = _ttr(dist, order)
    points = np.ascontiguousarray(rng.standard_normal((1, 6)))
    psi = mcp.basis(indices, coeffs, points)
    coefficients = rng.standard_normal((indices.shape[0], 4))
    got = mcp.evaluate(indices, coeffs, points, coefficients)
    assert got.shape == (6, 4)
    np.testing.assert_allclose(got, psi.T @ coefficients, rtol=1e-12,
                               atol=1e-14)


@pytest.mark.parametrize("order", [3, 5])
def test_fit_regression_matches_chaospy(order):
    """Least squares on an exactly-determined system: both must recover the
    Fourier coefficients of the sampled function."""
    rng = np.random.default_rng(11)
    dist = chaospy.J(chaospy.Uniform(-1, 2), chaospy.Normal(0, 1))
    reference = chaospy.generate_expansion(order, dist)
    indices = np.asarray(reference.exponents, dtype=np.int64)
    coeffs = _ttr(dist, order)
    nodes, weights = chaospy.generate_quadrature(order, dist, rule="gaussian")
    evals = np.sin(nodes[0]) + nodes[1] ** 2

    # chaospy returns the coefficients in its own polynomial order, which is
    # not `reference.exponents`, so compare the fitted models instead.
    got = mcp.fit_regression(indices, coeffs, nodes, evals)
    probe = np.ascontiguousarray(rng.uniform(-1, 2, (2, 5)))
    expected = chaospy.fit_regression(reference, nodes, evals)
    np.testing.assert_allclose(
        mcp.evaluate(indices, coeffs, probe, got),
        _rows(np.asarray(expected(*probe)), (5,)),
        rtol=1e-9, atol=1e-11,
    )


def test_fit_regression_recovers_exact_projection():
    """A regression that projects onto an orthogonal basis is the identity.
    A wrong index row or a transposed design matrix cannot pass this."""
    rng = np.random.default_rng(13)
    dist = chaospy.Normal(0, 1)
    order = 4
    indices = np.asarray(chaospy.generate_expansion(order, dist).exponents, dtype=np.int64)
    coeffs = _ttr(dist, order)
    nodes, _ = chaospy.generate_quadrature(order + 2, dist, rule="gaussian")
    truth = rng.standard_normal(indices.shape[0])
    psi = mcp.basis(indices, coeffs, nodes)
    evals = psi.T @ truth
    np.testing.assert_allclose(
        mcp.fit_regression(indices, coeffs, nodes, evals), truth,
        rtol=1e-9, atol=1e-11,
    )


def test_fit_quadrature_is_its_own_inverse():
    """A spectral projection is its own inverse: projecting a model that is
    already in the span must return its coefficients unchanged. This pins the
    design-matrix layout, the norm diagonal and the division, all of which a
    wrong index or a transposed row would break."""
    rng = np.random.default_rng(19)
    dist = chaospy.Normal(0, 1)
    order = 6
    indices = np.asarray(
        chaospy.generate_expansion(order, dist).exponents, dtype=np.int64
    )
    coeffs = _ttr(dist, order)
    nodes, weights = mcp.gaussian_quadrature(coeffs[0])
    psi = mcp.basis(indices, coeffs, nodes[None, :])
    truth = rng.standard_normal(indices.shape[0])

    got, got_norms = mcp.fit_quadrature(
        indices, coeffs, nodes, weights, psi.T @ truth
    )
    np.testing.assert_allclose(got, truth, rtol=1e-8, atol=1e-10)

    # the mass-matrix diagonal is the univariate norm of the monic polynomial
    _, norms = chaospy.generate_expansion(order, dist, retall=True)
    np.testing.assert_allclose(
        got_norms, np.ravel(np.asarray(norms)), rtol=1e-9, atol=1e-11
    )

    # chaospy's own front end must agree, term for term
    reference = chaospy.generate_expansion(order, dist)
    _, theirs = chaospy.fit_quadrature(
        reference, nodes, weights, psi.T @ truth, retall=1
    )
    np.testing.assert_allclose(
        np.ravel(theirs), truth, rtol=1e-8, atol=1e-10
    )


# --------------------------------------------------------------------------
# descriptive quantities
# --------------------------------------------------------------------------


def test_moments_and_variance_match_numpy_and_chaospy():
    """E, Var and Std of an expansion term on a Gaussian rule."""
    dist = chaospy.Normal(0, 1)
    order = 5
    reference = chaospy.generate_expansion(order, dist)
    indices = np.asarray(reference.exponents, dtype=np.int64)
    coeffs = _ttr(dist, order)
    nodes, weights = chaospy.generate_quadrature(order, dist, rule="gaussian")
    psi = mcp.basis(indices, coeffs, nodes)
    for m in range(psi.shape[0]):
        np.testing.assert_allclose(
            mcp.expected(psi[m], weights), np.dot(psi[m], weights),
            rtol=1e-13, atol=1e-15,
        )
        np.testing.assert_allclose(
            mcp.variance(psi[m], weights),
            np.dot(psi[m] ** 2, weights) - np.dot(psi[m], weights) ** 2,
            rtol=1e-11, atol=1e-14,
        )
    np.testing.assert_allclose(
        mcp.std(psi[2], weights), chaospy.Std(reference[2], dist), rtol=1e-9,
        atol=1e-11,
    )


def test_dot_weighted_matches_numpy():
    rng = np.random.default_rng(17)
    a, b, w = (rng.standard_normal(101) for _ in range(3))
    assert _lib.dot_weighted(a, b, w) == pytest.approx(
        float(np.dot(a * b, w)), rel=1e-12
    )


def test_expected_expansion_is_one_on_the_constant_term():
    """E[1] must be exactly the total weight; every other term vanishes for
    an orthogonal basis. Catches a wrong weight broadcast."""
    dist = chaospy.Normal(0, 1)
    order = 4
    indices = np.asarray(chaospy.generate_expansion(order, dist).exponents, dtype=np.int64)
    coeffs = _ttr(dist, order)
    nodes, weights = chaospy.generate_quadrature(order, dist, rule="gaussian")
    got = mcp.expected_expansion(indices, coeffs, nodes, weights)
    const = int(np.argmin(indices.sum(axis=1)))
    rest = np.delete(got, const)
    assert got[const] == pytest.approx(weights.sum(), rel=1e-13)
    np.testing.assert_allclose(rest, 0.0, atol=1e-13)


# --------------------------------------------------------------------------
# Clenshaw-Curtis
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2, 3, 5, 8, 13, 32, 33, 64])
def test_clenshaw_curtis_matches_chaospy(order):
    """Node and weight parity with `clenshaw_curtis_simple`. The weights come
    from one real inverse DFT, so machine precision is the right bar."""
    from chaospy.quadrature.clenshaw_curtis import clenshaw_curtis_simple

    nodes, weights = mcp.clenshaw_curtis(order)
    ref_nodes, ref_weights = clenshaw_curtis_simple(order)
    assert nodes.size == ref_nodes.size == weights.size
    np.testing.assert_allclose(nodes, ref_nodes, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(weights, ref_weights, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("order", [2, 3, 6, 11, 20])
def test_clenshaw_curtis_is_exact_for_low_degree_polynomials(order):
    """Analytic check independent of chaospy: the rule must integrate
    x^k exactly up to k = order - 1."""
    nodes, weights = mcp.clenshaw_curtis(order)
    for k in range(order):
        assert np.dot(weights, nodes**k) == pytest.approx(
            1.0 / (k + 1), rel=1e-9, abs=1e-12
        )


def test_clenshaw_curtis_weights_sum_to_one():
    for order in [2, 5, 12, 40]:
        _, weights = mcp.clenshaw_curtis(order)
        assert weights.sum() == pytest.approx(1.0, rel=1e-13)


def test_clenshaw_curtis_nodes_are_symmetric_about_one_half():
    nodes, _ = mcp.clenshaw_curtis(17)
    np.testing.assert_allclose(nodes, 1.0 - nodes[::-1], rtol=1e-14, atol=1e-15)
    assert nodes[0] == pytest.approx(0.0, abs=1e-15)
    assert nodes[-1] == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 9, 12, 17, 33, 64, 101])
def test_ihfft_real_matches_numpy(n):
    """The Bluestein path and the radix-2 path must both be exact; an off-by-one
    in the chirp index shows up as an O(1) error, not a rounding."""
    rng = np.random.default_rng(23)
    a = rng.standard_normal(n)
    np.testing.assert_allclose(
        _lib.ihfft_real(a), np.fft.ifft(a).real, rtol=1e-13, atol=1e-14
    )


# --------------------------------------------------------------------------
# Golub-Welsch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("order", [2, 6, 20])
def test_gaussian_quadrature_matches_chaospy(dist, order):
    coeffs = _ttr(dist, order)[0]
    nodes, weights = mcp.gaussian_quadrature(coeffs)
    ref_nodes, ref_weights = chaospy.quadrature.gaussian(order, dist)
    ref_nodes = np.ravel(np.atleast_2d(ref_nodes)[0])
    assert np.all(np.diff(nodes) >= 0.0), "nodes must come back sorted"
    np.testing.assert_allclose(nodes, ref_nodes, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(weights, ref_weights, rtol=1e-9, atol=1e-12)
    assert weights.sum() == pytest.approx(1.0, rel=1e-11)


def test_gaussian_quadrature_is_exact_for_polynomials():
    """A 10-point Gauss rule on the uniform measure integrates x^k exactly
    for k < 20, so every moment must be the analytic one."""
    order = 10
    n = np.arange(1, order + 1, dtype=float)
    coeffs = np.array([np.zeros(order + 1), np.concatenate([[1.0], n**2 / (4 * n**2 - 1)])])
    nodes, weights = mcp.gaussian_quadrature(coeffs)
    assert nodes.size == order + 1
    for k in range(2 * order + 1):
        want = 1.0 / (k + 1) if k % 2 == 0 else 0.0
        assert np.dot(weights, nodes**k) == pytest.approx(want, abs=1e-11)


def test_gaussian_quadrature_threaded_matches_serial():
    coeffs = np.array([np.zeros(200), np.arange(200, dtype=float)])
    serial = mcp.gaussian_quadrature(coeffs, workers=1)
    threaded = mcp.gaussian_quadrature(coeffs, workers=8, worker_threshold=8)
    np.testing.assert_array_equal(serial[0], threaded[0])
    np.testing.assert_array_equal(serial[1], threaded[1])


def test_negative_beta_is_reported_not_silently_wrong():
    coeffs = np.array([np.zeros(4), np.array([1.0, -1.0, 2.0, 3.0])])
    with pytest.raises(np.linalg.LinAlgError):
        mcp.gaussian_quadrature(coeffs)


# --------------------------------------------------------------------------
# discretized Stieltjes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dist", NO_TTR)
@pytest.mark.parametrize("order", [3, 6, 10])
def test_stieltjes_matches_chaospy(dist, order):
    """The whole procedure: proxy rule, recurrence, norms, convergence test.
    A wrong seeding of P_0/P_1 or a stale convergence row fails here."""
    expected = np.asarray(
        chaospy.recurrence.discretized_stieltjes(
            order, dist, rule="clenshaw_curtis", tolerance=1e-16
        )[0],
        dtype=float,
    ).reshape(2, order + 1)
    got = mcp.construct_recurrence_coefficients(
        order, dist.lower, dist.upper, dist.pdf, tolerance=1e-16
    )
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-9, atol=1e-11)


def test_stieltjes_recovers_the_exact_uniform_recurrence():
    """Uniform(-1, 1): alpha_n = 0 and beta_n = n^2 / (4 n^2 - 1). Independent
    of chaospy, so it pins the procedure to the mathematics."""
    order = 8
    got = mcp.construct_recurrence_coefficients(
        order, -1.0, 1.0, lambda x: np.ones_like(x), tolerance=1e-16
    )
    n = np.arange(1, order + 1, dtype=float)
    np.testing.assert_allclose(got[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(got[1][1:], n**2 / (4 * n**2 - 1), rtol=1e-10)
    assert got[1][0] == pytest.approx(1.0, rel=1e-15)


def test_stieltjes_rejects_an_unsupported_rule():
    with pytest.raises(ValueError):
        mcp.construct_recurrence_coefficients(
            3, 0.0, 1.0, lambda x: np.ones_like(x), rule="discrete"
        )


# --------------------------------------------------------------------------
# index set
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim,order", [(1, 4), (2, 4), (3, 3), (4, 2)])
def test_generate_indices_is_a_graded_cross_set(dim, order):
    indices = mcp.generate_indices(dim, order)
    assert indices.shape[1] == dim
    assert indices.min() == 0
    assert indices.max() <= order
    assert indices.sum(axis=1).max() <= order
    # graded: the exponent sums never decrease along the list
    assert np.all(np.diff(indices.sum(axis=1)) >= 0)
    # no duplicates
    assert len({tuple(r) for r in indices}) == len(indices)
    # the constant term is always first
    assert tuple(indices[0]) == (0,) * dim


def test_generate_indices_matches_chaospy_term_set():
    """Same expansion set as chaospy, up to ordering."""
    dist = chaospy.Iid(chaospy.Uniform(0, 1), 3)
    reference = chaospy.generate_expansion(3, dist)
    theirs = sorted(map(tuple, np.asarray(reference.exponents).tolist()))
    ours = sorted(map(tuple, mcp.generate_indices(3, 3).tolist()))
    assert ours == theirs
