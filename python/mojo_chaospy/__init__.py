"""mojo-chaospy: the compute-oriented subset of `chaospy` with Mojo kernels.

Polynomial chaos expansions, orthogonal polynomial recurrences and Gaussian
quadrature are the numeric core of `chaospy`; they are implemented here in
`dist/libmojo-chaospy.so` and callable from Python. The package installs
alongside the real `chaospy`, which the parity tests compare against.
"""

from .core import (
    FirstOrderSobol,
    TotalOrderSobol,
    basis,
    clenshaw_curtis,
    construct_recurrence_coefficients,
    evaluate,
    expected,
    expected_expansion,
    fit_quadrature,
    fit_regression,
    gaussian_quadrature,
    generate_indices,
    recurrence_evaluate,
    std,
    variance,
)

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
__version__ = "0.1.0"
