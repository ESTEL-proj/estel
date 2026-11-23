"""Derivatives of the statistics of tensor products (task T3.2, issue #23; Part I, §3.2.2;
maths.md M-S-TP, M-LP, M-NORM-TP).

The function differentiated is ``estel.functional.scattering`` on a ``DataTensorProd``, seen as
a real function of the data array: ``to_vector`` of its statistics with ``norm="self"``, so that
the reference, the centred scale with the low-pass filter in every block (OP-42) and the Gram
products of the second layer (M-GRAM) are all in it. ``[2D, 1D]``, 8 x 8 x 4, the 2D block open,
two channels with their cross terms, J = (2, 1). The JVP through the L0 primitives
(``Backend.jvp``) is compared with central differences in the fast tier, and the Hessian-vector
product (``Backend.hvp``) in the full tier, in float64, with the tolerances of
``test_scattering.py``. The parity of the gradient of the synthesis loss with STL is in
``tests/parity/test_tensor_prod.py``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import estel
from estel import Data1D, Data2D, DataTensorProd, STOperatorTensorProd
from estel.backend._protocol import Backend
from tests.differentiation.test_scattering import ATOL, RTOL, _central_difference

pytestmark = pytest.mark.maths("M-S-TP", "M-LP", "M-NORM-TP")


def _function(backend: Backend) -> tuple[Any, Any]:
    field = np.random.default_rng(0).standard_normal((1, 2, 8, 8, 4)) + 0.2
    data = DataTensorProd(field, [Data2D, Data1D], pbc=[False, True])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13
        operator = STOperatorTensorProd(data, [2, 1], norm="self")

    def function(array: Any) -> Any:
        return estel.functional.scattering(
            data.with_array(array),
            operator.plan,
            operator.filters,
        ).to_vector()

    return data.array, function


def _direction(backend: Backend, like: Any, seed: int) -> Any:
    values = np.random.default_rng(seed).standard_normal(tuple(like.shape))
    return backend.asarray(values, "float64")


def test_jvp_matches_finite_differences(backend: Backend) -> None:
    """``Backend.jvp`` of every coefficient against central differences."""
    x, function = _function(backend)
    v = _direction(backend, x, 1)
    _, tangent = backend.jvp(function, (x,), (v,))
    np.testing.assert_allclose(
        backend.to_numpy(tangent),
        _central_difference(backend, function, x, v),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.slow
def test_hvp_matches_finite_differences(backend: Backend) -> None:
    """``Backend.hvp`` of a random linear combination of the coefficients against central
    differences of its gradient: second order through the whole loop (Part I, §3.2.2).

    Slow: second order, which the decisions of issue #23 put in the full tier: 0.2 s of CPU on
    PyTorch, and about 24 s on JAX, which compiles the second derivative of the whole loop."""
    x, function = _function(backend)
    function = backend.jit(function)
    v = _direction(backend, x, 1)
    weights = _direction(backend, function(x), 2)

    def scalar(array: Any) -> Any:
        return backend.xp.sum(weights * function(array))

    def gradient(array: Any) -> Any:
        return backend.value_and_grad(scalar)(array)[1]

    # Richardson's extrapolation of two central differences, (4 D(h/2) - D(h)) / 3, removes
    # their h² term: the curvature of the cross S1 (OP-16) makes it 4e-8 at h = 1e-6 on some
    # entries, above ATOL, while it falls as h², so the derivative is right.
    coarse = _central_difference(backend, gradient, x, v, 2e-5)
    fine = _central_difference(backend, gradient, x, v, 1e-5)
    np.testing.assert_allclose(
        backend.to_numpy(backend.hvp(scalar, x, v)),
        (4 * fine - coarse) / 3,
        rtol=RTOL,
        atol=ATOL,
    )
