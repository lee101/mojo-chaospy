# mojo-chaospy

`mojo-chaospy` is the compute-oriented subset of
[chaospy](https://chaospy.readthedocs.io/) — polynomial chaos expansions,
orthogonal polynomial recurrences and Gaussian quadrature — with the inner
loops implemented in Mojo and callable from Python.

The Python package is named `mojo_chaospy`, so it installs alongside the real
`chaospy` and the tests compare the two directly, term for term.

```python
import numpy as np
import mojo_chaospy as mcp

coeffs = mcp.construct_recurrence_coefficients(6, -1.0, 1.0, lambda x: np.ones_like(x))
nodes, weights = mcp.gaussian_quadrature(coeffs)     # 7-point Gauss-Legendre
indices = mcp.generate_indices(2, 4)                  # graded hyperbolic cross
fourier, norms = mcp.fit_quadrature(indices, coeffs, nodes, weights, solves)
value = mcp.evaluate(indices, coeffs, nodes, fourier)
```

## Why this is the compute core

`chaospy` is 30k lines, but essentially all of its arithmetic is four loops
over collocation points:

1. **The three-term recurrence.** Every expansion `chaospy` hands back is
   `P[0]=1`, `P[n+1] = (q - A[n]) P[n] - B[n] P[n-1]`. Evaluating that at a
   batch of points is `O(order * n)` and is what evaluates an expansion,
   measures the norms in the Stieltjes procedure, and builds the design matrix
   of a spectral fit. It is the single hottest loop in the toolbox.
2. **The tensor product.** A multivariate chaos basis function is
   `Psi_m(q) = prod_d P[idx[d,m]](q_d)`. Turning the per-dimension univariate
   family into the full expansion at `n` points is `O(dim * terms * n)`, and it
   is the inner loop of regression, spectral fitting and every descriptive
   quantity.
3. **The weighted sums.** Quadrature, `E`, `Var`, `Std`, the spectral
   coefficients and the Stieltjes norms are all `sum_k f(k) w[k]`.
4. **The Golub-Welsch step.** Gaussian quadrature is a symmetric tridiagonal
   eigenproblem: build the Jacobi matrix from the recurrence, take its
   eigenvalues, and recover each weight from the first component of its
   eigenvector.

The remaining 95% of the package is distribution objects, `numpoly` symbolic
algebra, sparse-grid machinery, samplers, and frontend plumbing. None of that
is a numeric kernel, and all of it is deliberately left to the real `chaospy`.

## Covered subset

| area | implemented API | kernel |
| --- | --- | --- |
| Orthogonal family | `recurrence_evaluate`, `basis`, `evaluate` | `cp_eval_orthogonal`, `cp_tensor_eval` |
| Quadrature rules | `clenshaw_curtis`, `gaussian_quadrature` | `cp_clenshaw_curtis`, `cp_ihfft_real`, `cp_bisect_eigenvalues`, `cp_golub_weights` |
| Recurrence construction | `construct_recurrence_coefficients` (discretized Stieltjes) | `cp_stieltjes_iterate` |
| Fitting | `fit_regression`, `fit_quadrature` | `cp_fit_coefficients` |
| Descriptives | `expected`, `variance`, `std`, `expected_expansion` | `cp_moments`, `cp_dot_weighted` |
| Variance decomposition | `FirstOrderSobol`, `TotalOrderSobol` | coefficient masking (NumPy — it is an `O(terms)` sum) |
| Index sets | `generate_indices` (graded / hyperbolic cross) | NumPy (`numpoly` is a library dependency, not a port target) |

### Not implemented

Everything else in `chaospy`, and the reasons:

- **The 80+ distributions** and their operator algebra (`J`, `Iid`, copulas,
  kernels, truncation). Each distribution is a symbolic object with a `pdf`,
  a `cdf` and sometimes an analytic `ttr`; there is no loop to accelerate.
  This port takes the recurrence coefficients as data and never touches a
  distribution object.
- **`lanczos` and `chebyshev` recurrence algorithms** (two of chaospy's three
  alternatives to Stieltjes). They are the same `O(order * nodes)` shape as
  Stieltjes but are a different algorithm; only the default is ported.
- **Sparse grids** (`chaospy.quadrature.sparse_grid`) and the other named
  rules: Kronrod, Patterson, Fejer, Lobatto, Radau, Newton-Cotes, Genz-Keister.
  These are each a separate node/weight construction, and most are far smaller
  than the two headline rules.
- **`numpoly`** — symbolic polynomial arithmetic. This is a large separate
  library and the wrong thing to reimplement; the port consumes an explicit
  exponent table instead.
- **Samplers** (Sobol, Halton, Latin hypercube, antithetic) and **Saltelli**,
  **`descriptives.conditional`**'s symbolic path, and `chaospy.E`/`Var` as
  moment-based (rather than quadrature-based) operators. The quadrature-based
  forms of the descriptives *are* ported; the symbolic ones are numpoly.
- **`sklearn` model support in `fit_regression`.** The default least-squares
  path is ported; the pluggable-estimator path is forwarded.

`construct_recurrence_coefficients` takes `lower`, `upper` and a `pdf`
callable rather than a `chaospy.Distribution`, and `basis`/`evaluate` take
recurrence coefficient arrays rather than an expansion object. That is the one
real API difference, and it is the price of not depending on `numpoly`.

## Numerical contract

C-contiguous `float64` throughout. Mojo emits FMA, so a product inside a
recurrence or a weighted sum is fused where NumPy's is not, and the parity
tests assert `rtol` in the `1e-8`–`1e-11` range with stated tolerances rather
than bit equality. Two things *are* checked tight, because they have one
rounding step rather than a chain: the Clenshaw-Curtis weights and nodes
(`1e-13`, they agree with chaospy to about one ulp) and the Golub-Welsch
eigenvalues (`1e-9`, limited by the bisection bracket, not by FMA).

The tridiagonal eigenvalues come from **Sturm-sequence bisection** inside the
Gershgorin interval, the same device LAPACK's `dstebz` uses, rather than from
implicit-shift QL. Bisection is chosen because it produces the spectrum in
order by construction and needs no iteration-count tuning; it is `O(n)` per
eigenvalue, so the whole spectrum costs `O(n^2)`, the same order as the
`O(n^2)` weight pass that follows it. Eigenvalues are independent, so that
loop is threadable through the `k0`/`k1` range arguments — see the benchmark
for whether that pays.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-chaospy.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task. The parity tests need the real
`chaospy` importable in the same interpreter.

## Performance

Best-of-N wall clock, same process, every case gated on numerical agreement
with an independent reference first. Reference for the recurrence, tensor
product, moments and spectral fit is vectorised NumPy doing the same sum;
reference for Golub-Welsch is `scipy.linalg.eig_banded`, which is what
`chaospy` itself calls.

| case | reference | mojo-chaospy | result |
| --- | ---: | ---: | ---: |
| recurrence order=256 n=65536 | 262.47 ms | 76.71 ms | 3.42x faster |
| tensor dim=3 terms=343 n=16384 | 120.76 ms | 48.91 ms | 2.47x faster |
| moments n=1048576 x100 | 3311.55 ms | 607.50 ms | 5.45x faster |
| fit_coefficients terms=2048 n=64 x20 | 28.33 ms | 32.81 ms | 0.86x, slower |
| ihfft n=4096 x100 | 37.57 ms | 39.09 ms | 0.96x, parity |
| golub-welsch order=200 serial | 41.25 ms | 43.59 ms | 0.95x, parity |
| golub-welsch order=200, 8 workers | 21.99 ms | 22.34 ms | 0.98x, parity |

Reading these honestly:

- The recurrence, tensor product and moment kernels win because they are
  multiply-accumulate chains that vectorise and NumPy's temporary arrays do
  not, plus Mojo has no Python-level dispatch per element.
- `fit_coefficients` is **slower**. At `n=64` the design matrix fits in cache
  and the kernel is a single streaming pass that NumPy's `BLAS`-backed
  reduction does as well or better. This is a loss and is reported as one.
- The Bluestein FFT is **parity with `numpy.fft`**. `numpy.fft` is
  pocketfft with split-radix and hand-vectorised butterflies; a plain radix-2
  in Mojo is not going to beat it, and on a bandwidth-bound transform there is
  little to win anyway.
- Golub-Welsch is **parity**, and the threaded row is the honest result: at
  order 200 each eigenvalue is only `O(n)` of work with a serial
  eigenvector recurrence inside it, so the thread fan-out costs more in
  dispatch than it recovers. The threshold (`worker_threshold`) exists so
  large orders can opt in, but it is not a win at this size and the benchmark
  does not claim one.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-chaospy.so`.

Buffers cross the C ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps every exported symbol
non-parametric. Index tables cross as `float64` buffers, since the kernel
reads them through the same pointer type.

The `python/mojo_chaospy` layer owns every array and the two pieces of policy
that are not numeric work: the order-growth and convergence loop of the
Stieltjes procedure, and the `numpy.linalg.lstsq` solve inside
`fit_regression`. The Stieltjes proxy rule reproduces chaospy's own
segmented Clenshaw-Curtis construction (`int(sqrt(order))` segments, shared
boundary nodes collapsed and their weights added), because that segmentation
is part of the algorithm, not a formatting detail.

## License

MIT
