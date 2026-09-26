"""Polynomial-chaos kernels: the numeric core of `chaospy`.

`chaospy` is a polynomial chaos expansion toolbox. Almost everything it does
reduces to four numeric kernels, all of which are loops over sample points and
are ported here:

  * the three-term recurrence that evaluates an orthogonal polynomial family,
  * the tensor product that turns univariate families into a multivariate chaos
    expansion at a batch of collocation points,
  * the weighted sums that quadrature and every descriptive quantity reduce to,
  * the Golub-Welsch step (symmetric tridiagonal eigensolver) that turns
    recurrence coefficients into Gaussian quadrature nodes and weights.

On top of those sit the two node/weight generators that `chaospy` needs:
Clenshaw-Curtis (a real inverse DFT, done with a Bluestein-chirped radix-2 FFT
here) and the discretized Stieltjes procedure that builds recurrence
coefficients from a proxy integration rule.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.math import abs, cos, exp, sin, sqrt

comptime FP = Pointer[Float64, AnyOrigin[mut=True]]
comptime PI = 3.14159265358979323846


def fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# 1. Three-term recurrence: P[n+1] = (q - alpha[n]) P[n] - beta[n] P[n-1]
# ---------------------------------------------------------------------------


@export("cp_eval_orthogonal")
def cp_eval_orthogonal(
    alpha_addr: Int, beta_addr: Int, order: Int, x_addr: Int, n: Int,
    out_addr: Int
) abi("C"):
    """Evaluate P_0 .. P_order at `n` points, stored row-major as `(order+1, n)`.

    `alpha` and `beta` hold `order+1` recurrence coefficients each. This is the
    single hottest loop in the toolbox: it is what evaluates an expansion, what
    measures the norms in the Stieltjes procedure, and what a spectral fit
    multiplies against.
    """
    var alpha = fp(alpha_addr)
    var beta = fp(beta_addr)
    var x = fp(x_addr)
    var out = fp(out_addr)

    var stride = n
    for i in range(n):
        out[unsafe_offset=i] = 1.0
    if order >= 1:
        var a0 = alpha[unsafe_offset=0]
        for i in range(n):
            out[unsafe_offset=stride + i] = x[unsafe_offset=i] - a0
    var m = 2
    while m <= order:
        var a = alpha[unsafe_offset=m - 1]
        var b = beta[unsafe_offset=m - 1]
        var prev = (m - 2) * stride
        var cur = (m - 1) * stride
        var nxt = m * stride
        for i in range(n):
            out[unsafe_offset=nxt + i] = (
                (x[unsafe_offset=i] - a) * out[unsafe_offset=cur + i]
                - b * out[unsafe_offset=prev + i]
            )
        m += 1


# ---------------------------------------------------------------------------
# 2. Tensor product: Psi_m(q) = prod_d P_{idx[d,m]}(q_d)
# ---------------------------------------------------------------------------


@export("cp_tensor_eval")
def cp_tensor_eval(
    bank_addr: Int, idx_addr: Int, dim: Int, terms: Int, n: Int,
    stride: Int, acc_addr: Int, out_addr: Int
) abi("C"):
    """Evaluate a multivariate chaos expansion at `n` points.

    `bank` is the per-dimension univariate family already evaluated by
    `cp_eval_orthogonal`, laid out as `(dim, stride, n)`. `idx` holds the
    univariate order of every (dimension, term) pair, row-major `(dim, terms)`.
    The result is `(terms, n)`.
    """
    var bank = fp(bank_addr)
    var idx = fp(idx_addr)
    var acc = fp(acc_addr)
    var out = fp(out_addr)

    for m in range(terms):
        for i in range(n):
            acc[unsafe_offset=i] = 1.0
        for d in range(dim):
            var base = d * stride * n + Int(idx[unsafe_offset=d * terms + m]) * n
            for i in range(n):
                acc[unsafe_offset=i] = (
                    acc[unsafe_offset=i] * bank[unsafe_offset=base + i]
                )
        var dst = m * n
        for i in range(n):
            out[unsafe_offset=dst + i] = acc[unsafe_offset=i]


# ---------------------------------------------------------------------------
# 3. Quadrature: every descriptive quantity is a weighted sum.
# ---------------------------------------------------------------------------


@export("cp_dot_weighted")
def cp_dot_weighted(a_addr: Int, b_addr: Int, w_addr: Int, n: Int,
                    out_addr: Int) abi("C"):
    """Weighted inner product: out[0] = sum_k a[k] * b[k] * w[k]."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var w = fp(w_addr)
    var out = fp(out_addr)
    var acc = Float64(0.0)
    for k in range(n):
        acc += a[unsafe_offset=k] * b[unsafe_offset=k] * w[unsafe_offset=k]
    out[unsafe_offset=0] = acc


@export("cp_moments")
def cp_moments(psi_addr: Int, w_addr: Int, n: Int, out_addr: Int) abi("C"):
    """First and second quadrature moments of a basis function.

    out[0] = E[psi], out[1] = E[psi^2], out[2] = Var[psi], out[3] = sum(w).
    """
    var psi = fp(psi_addr)
    var w = fp(w_addr)
    var out = fp(out_addr)
    var m0 = Float64(0.0)
    var m1 = Float64(0.0)
    var sw = Float64(0.0)
    for k in range(n):
        var value = psi[unsafe_offset=k]
        var weight = w[unsafe_offset=k]
        m0 += value * weight
        m1 += value * value * weight
        sw += weight
    out[unsafe_offset=0] = m0
    out[unsafe_offset=1] = m1
    out[unsafe_offset=2] = m1 - m0 * m0
    out[unsafe_offset=3] = sw


@export("cp_fit_coefficients")
def cp_fit_coefficients(
    psi_addr: Int, w_addr: Int, solves_addr: Int, terms: Int, n: Int,
    norms_addr: Int, coeffs_addr: Int
) abi("C"):
    """Spectral projection: norms and Fourier coefficients of an expansion.

    norms[m] = sum_k Psi[m,k]^2 w[k] (the diagonal of the mass matrix, which
    is all a spectral fit needs because the basis is orthogonal), and
    coeffs[m] = sum_k Psi[m,k] q[k] w[k] / norms[m]. Both are O(terms * n).
    """
    var psi = fp(psi_addr)
    var w = fp(w_addr)
    var solves = fp(solves_addr)
    var norms = fp(norms_addr)
    var coeffs = fp(coeffs_addr)

    for m in range(terms):
        var base = m * n
        var acc = Float64(0.0)
        for k in range(n):
            var value = psi[unsafe_offset=base + k]
            acc += value * value * w[unsafe_offset=k]
        norms[unsafe_offset=m] = acc
    for m in range(terms):
        var base = m * n
        var acc = Float64(0.0)
        for k in range(n):
            acc += (
                psi[unsafe_offset=base + k] * solves[unsafe_offset=k]
                * w[unsafe_offset=k]
            )
        coeffs[unsafe_offset=m] = acc / norms[unsafe_offset=m]


# ---------------------------------------------------------------------------
# 4. Golub-Welsch: symmetric tridiagonal eigenproblem -> Gaussian quadrature
# ---------------------------------------------------------------------------


def _sturm_below(d: FP, e: FP, n: Int, x: Float64) -> Int:
    """Number of eigenvalues of the tridiagonal (d, e) strictly below x.

    Sturm sequence count, the same device LAPACK's `dstebz` uses.
    """
    var q = d[unsafe_offset=0] - x
    var count = 0
    var sign = 1.0
    if q < 0.0:
        count = 1
        sign = -1.0
    for i in range(1, n):
        var prev = q
        if prev == 0.0:
            prev = sign * 1.0e-300
        var off = e[unsafe_offset=i - 1]
        q = d[unsafe_offset=i] - x - off * off / prev
        if q < 0.0:
            count += 1
            sign = -1.0
        else:
            sign = 1.0
    return count


@export("cp_bisect_eigenvalues")
def cp_bisect_eigenvalues(
    alpha_addr: Int, beta_addr: Int, n: Int, k0: Int, k1: Int, e_addr: Int,
    vals_addr: Int
) abi("C") -> Int:
    """Eigenvalues `k0`..`k1` of the Jacobi matrix, ascending.

    The matrix has diagonal `alpha` and off-diagonal `sqrt(beta[1:n])`. Each
    eigenvalue is isolated by bisection on the Sturm sequence count inside the
    Gershgorin interval, so the result is ordered by construction and does not
    depend on an iteration count. `e_addr` is per-caller scratch of `n` floats
    holding the off-diagonal; eigenvalues are independent, so the caller may
    fan the `k` range out across threads with one buffer each.

    Returns 0 on success and -1 if the recurrence carries a negative beta,
    which is how an illegal three-term recurrence is reported.
    """
    var d = fp(alpha_addr)
    var beta = fp(beta_addr)
    var e = fp(e_addr)
    var vals = fp(vals_addr)

    for i in range(n):
        if i > 0:
            var b = beta[unsafe_offset=i]
            if b < 0.0:
                return -1
            e[unsafe_offset=i - 1] = sqrt(b)
    e[unsafe_offset=n - 1] = 0.0

    var lo = d[unsafe_offset=0]
    var hi = d[unsafe_offset=0]
    for i in range(n):
        var rad = Float64(0.0)
        if i > 0:
            rad += abs(e[unsafe_offset=i - 1])
        if i < n - 1:
            rad += abs(e[unsafe_offset=i])
        var centre = d[unsafe_offset=i]
        if centre - rad < lo:
            lo = centre - rad
        if centre + rad > hi:
            hi = centre + rad

    for k in range(k0, k1):
        var a = lo
        var b = hi
        for _ in range(200):
            var mid = 0.5 * (a + b)
            if mid == a or mid == b:
                break
            if _sturm_below(d, e, n, mid) <= k:
                a = mid
            else:
                b = mid
        vals[unsafe_offset=k] = 0.5 * (a + b)
    return 0


@export("cp_golub_weights")
def cp_golub_weights(
    alpha_addr: Int, beta_addr: Int, n: Int, vals_addr: Int, k0: Int, k1: Int,
    scratch_addr: Int, weights_addr: Int
) abi("C") -> Int:
    """Gaussian weights for eigenvalues `vals[k0:k1]` of the Jacobi matrix.

    The first component of a normalised eigenvector is obtained by fixing
    v[0] = 1 and running the second-order recurrence of (T - lambda I) v = 0
    forwards, so the weight is 1 / ||v||^2. Eigenvalues are independent, so the
    caller may fan this loop out over a thread pool with `k0`/`k1`.
    """
    var alpha = fp(alpha_addr)
    var beta = fp(beta_addr)
    var vals = fp(vals_addr)
    var v = fp(scratch_addr)
    var weights = fp(weights_addr)

    for k in range(k0, k1):
        var lam = vals[unsafe_offset=k]
        v[unsafe_offset=0] = 1.0
        if n == 1:
            weights[unsafe_offset=k] = 1.0
            continue
        var top = 1.0
        v[unsafe_offset=1] = (lam - alpha[unsafe_offset=0]) / sqrt(
            beta[unsafe_offset=1]
        )
        if abs(v[unsafe_offset=1]) > top:
            top = abs(v[unsafe_offset=1])
        for i in range(1, n - 1):
            var value = (
                (lam - alpha[unsafe_offset=i]) * v[unsafe_offset=i]
                - sqrt(beta[unsafe_offset=i]) * v[unsafe_offset=i - 1]
            ) / sqrt(beta[unsafe_offset=i + 1])
            v[unsafe_offset=i + 1] = value
            var av = abs(value)
            if av > top:
                top = av
            if top > 1.0e100:
                # rescale the whole partial vector; exact up to one rounding
                for r in range(i + 2):
                    v[unsafe_offset=r] = v[unsafe_offset=r] * 1.0e-100
                top = top * 1.0e-100
        var acc = Float64(0.0)
        for i in range(n):
            acc += v[unsafe_offset=i] * v[unsafe_offset=i]
        weights[unsafe_offset=k] = 1.0 / acc
    return 0


# ---------------------------------------------------------------------------
# 5. Clenshaw-Curtis nodes and weights
# ---------------------------------------------------------------------------


def _fft(re: FP, im: FP, n: Int, sign: Float64):
    """In-place radix-2 decimation-in-time FFT, unscaled.

    `sign` is +1 for the unnormalised inverse transform e^(+2 pi i j k / n)
    and -1 for the forward transform. `n` must be a power of two.
    """
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit != 0:
            j = j ^ bit
            bit = bit >> 1
        j = j ^ bit
        if i < j:
            var tr = re[unsafe_offset=i]
            re[unsafe_offset=i] = re[unsafe_offset=j]
            re[unsafe_offset=j] = tr
            var ti = im[unsafe_offset=i]
            im[unsafe_offset=i] = im[unsafe_offset=j]
            im[unsafe_offset=j] = ti
    var span = 2
    while span <= n:
        var angle = sign * 2.0 * PI / Float64(span)
        var wr = cos(angle)
        var wi = sin(angle)
        var half = span >> 1
        var base = 0
        while base < n:
            var cr = 1.0
            var ci = 0.0
            for k in range(base, base + half):
                var ar = re[unsafe_offset=k]
                var ai = im[unsafe_offset=k]
                var br = re[unsafe_offset=k + half]
                var bi = im[unsafe_offset=k + half]
                var vr = br * cr - bi * ci
                var vi = br * ci + bi * cr
                re[unsafe_offset=k] = ar + vr
                im[unsafe_offset=k] = ai + vi
                re[unsafe_offset=k + half] = ar - vr
                im[unsafe_offset=k + half] = ai - vi
                var ncr = cr * wr - ci * wi
                ci = cr * wi + ci * wr
                cr = ncr
            base += span
        span = span << 1


@export("cp_ihfft_real")
def cp_ihfft_real(
    a_addr: Int, n: Int, out_addr: Int, re_addr: Int, im_addr: Int,
    re2_addr: Int, im2_addr: Int
) abi("C") -> Int:
    """Real inverse DFT of length `n`, `out[k] = sum_j a[j] cos(2 pi j k / n) / n`.

    This is `numpy.fft.ihfft` of a real sequence, real part only, which is
    exactly what the Clenshaw-Curtis weight formula consumes. Power-of-two
    lengths go straight through the radix-2 kernel; other lengths use
    Bluestein's chirp, so the scratch buffers must hold `m` floats each with
    `m >= 2n - 1` the next power of two.

    Returns 0, or -1 if the scratch is too small.
    """
    var a = fp(a_addr)
    var out = fp(out_addr)
    var re = fp(re_addr)
    var im = fp(im_addr)
    var re2 = fp(re2_addr)
    var im2 = fp(im2_addr)

    if n <= 0:
        return -1
    var m = 1
    while m < 2 * n - 1:
        m = m << 1
    if (n & (n - 1)) == 0:
        for i in range(n):
            re[unsafe_offset=i] = a[unsafe_offset=i]
            im[unsafe_offset=i] = 0.0
        _fft(re, im, n, 1.0)
        var inv = 1.0 / Float64(n)
        for i in range(n):
            out[unsafe_offset=i] = re[unsafe_offset=i] * inv
        return 0

    for i in range(m):
        re[unsafe_offset=i] = 0.0
        im[unsafe_offset=i] = 0.0
        re2[unsafe_offset=i] = 0.0
        im2[unsafe_offset=i] = 0.0
    # b[j] = a[j] * exp(i pi j^2 / n)
    for j in range(n):
        var mm = (j * j) % (2 * n)
        var angle = PI * Float64(mm) / Float64(n)
        var br = cos(angle)
        var bi = sin(angle)
        re[unsafe_offset=j] = a[unsafe_offset=j] * br
        im[unsafe_offset=j] = a[unsafe_offset=j] * bi
        var cj = m - j
        if cj >= m:
            cj = cj % m
        re2[unsafe_offset=j] = re2[unsafe_offset=j] + br
        im2[unsafe_offset=j] = im2[unsafe_offset=j] - bi
        if j != 0:
            re2[unsafe_offset=cj] = re2[unsafe_offset=cj] + br
            im2[unsafe_offset=cj] = im2[unsafe_offset=cj] - bi
    # forward transforms, circular convolution, then the inverse
    _fft(re, im, m, -1.0)
    _fft(re2, im2, m, -1.0)
    for i in range(m):
        var rr = re[unsafe_offset=i]
        var ii = im[unsafe_offset=i]
        re[unsafe_offset=i] = rr * re2[unsafe_offset=i] - ii * im2[unsafe_offset=i]
        im[unsafe_offset=i] = rr * im2[unsafe_offset=i] + ii * re2[unsafe_offset=i]
    _fft(re, im, m, 1.0)
    # X[k] = exp(i pi k^2 / n) * conv[k] / (m * n)
    var inv = 1.0 / (Float64(m) * Float64(n))
    for k in range(n):
        var mm = (k * k) % (2 * n)
        var angle = PI * Float64(mm) / Float64(n)
        out[unsafe_offset=k] = (
            re[unsafe_offset=k] * cos(angle) - im[unsafe_offset=k] * sin(angle)
        ) * inv
    return 0


@export("cp_clenshaw_curtis")
def cp_clenshaw_curtis(
    order: Int, x_addr: Int, w_addr: Int, a_addr: Int, spec_addr: Int,
    re_addr: Int, im_addr: Int, re2_addr: Int, im2_addr: Int
) abi("C") -> Int:
    """Clenshaw-Curtis nodes and weights on [0, 1]; returns the node count.

    Reproduces `chaospy.quadrature.clenshaw_curtis.clenshaw_curtis_simple`: the
    nodes are the Chebyshev-Lobatto points, and the weights are the real
    inverse DFT of the Tiedeman spectral filter, mirrored to full length.
    """
    var x = fp(x_addr)
    var w = fp(w_addr)
    var a = fp(a_addr)
    var spec = fp(spec_addr)
    var re = fp(re_addr)
    var im = fp(im_addr)
    var re2 = fp(re2_addr)
    var im2 = fp(im2_addr)

    if order == 0:
        x[unsafe_offset=0] = 0.5
        w[unsafe_offset=0] = 1.0
        return 1
    if order == 1:
        x[unsafe_offset=0] = 0.0
        x[unsafe_offset=1] = 1.0
        w[unsafe_offset=0] = 0.5
        w[unsafe_offset=1] = 0.5
        return 2

    var count = order // 2
    var remains = order - count
    var parity = order % 2
    var scale = 1.0 / (Float64(order) * Float64(order) - 1.0 + Float64(parity))

    # b0 is length order + 1; the filter is its reflection about the centre.
    for i in range(order + 1):
        spec[unsafe_offset=i] = 0.0
    for i in range(count):
        var step = Float64(2 * i + 1)
        spec[unsafe_offset=i] = 2.0 / (step * (step - 2.0))
    spec[unsafe_offset=count] = 1.0 / (2.0 * Float64(count) - 1.0)

    for i in range(order):
        a[unsafe_offset=i] = (
            -spec[unsafe_offset=i] - spec[unsafe_offset=order - i] - scale
        )
    a[unsafe_offset=count] += Float64(order) * scale
    a[unsafe_offset=remains] += Float64(order) * scale

    if cp_ihfft_real(a_addr, order, spec_addr, re_addr, im_addr, re2_addr,
                     im2_addr) != 0:
        return -1

    var half = order // 2 + 1
    var tail = half - 2 + parity
    for i in range(order + 1):
        x[unsafe_offset=i] = (
            0.5 * cos(Float64(order - i) * PI / Float64(order)) + 0.5
        )
    for i in range(half):
        w[unsafe_offset=i] = 0.5 * spec[unsafe_offset=i]
    for i in range(tail + 1):
        w[unsafe_offset=half + i] = 0.5 * spec[unsafe_offset=tail - i]
    return order + 1


# ---------------------------------------------------------------------------
# 6. Discretized Stieltjes procedure
# ---------------------------------------------------------------------------


@export("cp_stieltjes_iterate")
def cp_stieltjes_iterate(
    x_addr: Int, w_addr: Int, n: Int, order: Int, alpha_in_addr: Int,
    beta_in_addr: Int, norms_addr: Int, alpha_out_addr: Int, beta_out_addr: Int,
    pbuf_addr: Int
) abi("C") -> Float64:
    """One sweep of the discretized Stieltjes procedure.

    Builds P_2 .. P_order+1 on the proxy quadrature rule `(x, w)`, updates the
    norms, and writes the fresh three-term recurrence coefficients into
    `alpha_out`/`beta_out`. Returns `max |beta_out - beta_in|`, which is the
    convergence measure the caller thresholds.

    `pbuf` is scratch of `3 * n` floats: the recurrence only ever needs the
    previous two rows and the one being written.
    """
    var x = fp(x_addr)
    var w = fp(w_addr)
    var alpha_in = fp(alpha_in_addr)
    var beta_in = fp(beta_in_addr)
    var norms = fp(norms_addr)
    var alpha_out = fp(alpha_out_addr)
    var beta_out = fp(beta_out_addr)
    var buf = fp(pbuf_addr)

    var row0 = buf
    var row1 = buf.unsafe_offset(n)
    var row2 = buf.unsafe_offset(2 * n)

    var inner = Float64(0.0)
    for i in range(n):
        inner += x[unsafe_offset=i] * w[unsafe_offset=i]

    # P_0 = 0 and P_1 = 1, as chaospy seeds them; norms[0] and norms[1] stay 1.
    for i in range(n):
        row0[unsafe_offset=i] = 0.0
        row1[unsafe_offset=i] = 1.0


    var maxdiff = Float64(0.0)
    for idx in range(order):
        var alpha_new = inner / norms[unsafe_offset=idx + 1]
        var beta_new = norms[unsafe_offset=idx + 1] / norms[unsafe_offset=idx]
        alpha_out[unsafe_offset=idx] = alpha_new
        beta_out[unsafe_offset=idx] = beta_new
        var diff = abs(beta_new - beta_in[unsafe_offset=idx])
        if diff > maxdiff:
            maxdiff = diff
        var acc = Float64(0.0)
        inner = Float64(0.0)
        for i in range(n):
            var xi = x[unsafe_offset=i]
            var value = (xi - alpha_new) * row1[unsafe_offset=i] - beta_new * row0[
                unsafe_offset=i
            ]
            row2[unsafe_offset=i] = value
            var quad = w[unsafe_offset=i] * value * value
            acc += quad
            inner += xi * quad
        norms[unsafe_offset=idx + 2] = acc
        for i in range(n):
            row0[unsafe_offset=i] = row1[unsafe_offset=i]
            row1[unsafe_offset=i] = row2[unsafe_offset=i]

    alpha_out[unsafe_offset=order] = inner / norms[unsafe_offset=order + 1]
    beta_out[unsafe_offset=order] = norms[unsafe_offset=order + 1] / norms[
        unsafe_offset=order
    ]
    var diff = abs(
        beta_out[unsafe_offset=order] - beta_in[unsafe_offset=order]
    )
    if diff > maxdiff:
        maxdiff = diff
    return maxdiff
