"""The derivatives of the functions of the internal ``WaveletOperator``, to second order
(task T2.10; Part I, §2.3, §3.2.2), as ``test_scattering.py`` does for the whole loop (T2.7).

The function is two layers of the pyramid, the wavelet operator alone with the modulus in
between: ψ ⋆ D(|ψ ⋆ x|) and D(x), with ``apply`` at levels 0 and 1 and ``downsample`` of the
field and of first-layer data, on masked data without ``pbc``. The modulus makes it non-linear,
so its Hessian is not zero. Through ``Backend.jvp`` and ``Backend.hvp``, on both backends, in
float64, against central finite differences; step and tolerances as in ``test_scattering.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data, Data1D, Data2D
from estel.backend._protocol import Backend
from estel.data._moments import modulus
from estel.wavelets._operator import WaveletOperator
from tests.differentiation._cases import direction_like

STEP = 1e-6
RTOL, ATOL = 1e-6, 1e-8


def _two_layers(backend: Backend, cls: type[Data]) -> tuple[Any, Callable[[Any], Any]]:
    shape = (1, 2, *(8,) * cls.ndim)
    mask = np.zeros(shape[2:], dtype=bool)
    mask[(slice(3, 5),) * cls.ndim] = True
    field = np.random.default_rng(0).standard_normal(shape) + 0.2
    data = cls(field, mask=mask, pbc=False)
    operator = WaveletOperator(data, 2, 2)
    xp = backend.xp

    def function(x: Any) -> Any:
        u = operator.downsample(modulus(operator.apply(x, 0)), 0, history=(0,))
        v = backend.complex_to_real_pairs(operator.apply(u, 1))
        return xp.concat([xp.reshape(v, (-1,)), xp.reshape(operator.downsample(x, 0), (-1,))])

    return data.array, function


def _central_difference(
    backend: Backend, function: Callable[[Any], Any], x: Any, v: Any
) -> NDArray[np.float64]:
    plus = backend.to_numpy(function(x + STEP * v))
    minus = backend.to_numpy(function(x - STEP * v))
    difference: NDArray[np.float64] = (plus - minus) / (2 * STEP)
    return difference


@pytest.mark.maths("M-W1", "M-DOWN", "M-DOWN-MASK")
@pytest.mark.parametrize("cls", [Data1D, pytest.param(Data2D, marks=pytest.mark.slow)])
def test_jvp_and_hvp_of_apply_and_downsample_match_finite_differences(
    backend: Backend, cls: type[Data]
) -> None:
    """``Backend.jvp`` of every output, and ``Backend.hvp`` of a random linear combination of
    them, against central differences of the function and of its gradient.

    Slow: in 2D only; the 1D case, 8 pixels, is the one of the fast tier (Part I, §5.4)."""
    x, function = _two_layers(backend, cls)
    v = direction_like(backend, x, seed=1)
    _, tangent = backend.jvp(function, (x,), (v,))
    np.testing.assert_allclose(
        backend.to_numpy(tangent),
        _central_difference(backend, function, x, v),
        rtol=RTOL,
        atol=ATOL,
    )
    weights = direction_like(backend, function(x), seed=2)

    def scalar(array: Any) -> Any:
        return backend.xp.sum(weights * function(array))

    def gradient(array: Any) -> Any:
        return backend.value_and_grad(scalar)(array)[1]

    hessian_vector = backend.to_numpy(backend.hvp(scalar, x, v))
    assert np.abs(hessian_vector).max() > 1e-3  # the modulus: a second order that is not zero
    np.testing.assert_allclose(
        hessian_vector, _central_difference(backend, gradient, x, v), rtol=RTOL, atol=ATOL
    )
