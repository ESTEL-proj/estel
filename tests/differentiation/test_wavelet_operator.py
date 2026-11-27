"""The derivatives of the functions of the internal ``WaveletOperatorSingleDomain``, to second order
(task T2.10; Part I, §2.3, §3.2.2), as ``test_scattering.py`` does for the whole loop (T2.7),
and those of the wavelet operator of a tensor product (task T3.1).

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

from estel import Data1D, Data2D, DataHealpix, DataTensorProd
from estel.backend._protocol import Backend
from estel.data._moments import modulus
from estel.wavelets._operator import WaveletOperatorSingleDomain
from estel.wavelets._operator_tensor_prod import WaveletOperatorTensorProd
from tests.differentiation._cases import direction_like

STEP = 1e-6
RTOL, ATOL = 1e-6, 1e-8


def _two_layers(
    backend: Backend, cls: type[Data2D] | type[Data1D]
) -> tuple[Any, Callable[[Any], Any]]:
    shape = (1, 2, *(8,) * cls.ndim)
    mask = np.zeros(shape[2:], dtype=bool)
    mask[(slice(3, 5),) * cls.ndim] = True
    field = np.random.default_rng(0).standard_normal(shape) + 0.2
    data = cls(field, mask=mask, pbc=False)
    operator = WaveletOperatorSingleDomain(data, 2, 2)
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
    backend: Backend, cls: type[Data2D] | type[Data1D]
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


def _layers_tensor_prod(backend: Backend) -> tuple[Any, Callable[[Any], Any]]:
    """Two layers of the pyramid of a ``[2D, 1D]`` field, block by block (maths.md M-W1-TENSORPROD):
    the block convolutions at level (0, 0), the modulus, a downsampling along each block, then
    the block convolutions at level (1, 1); not periodic in block 0, periodic in block 1."""
    field = np.random.default_rng(0).standard_normal((1, 2, 8, 8, 8)) + 0.2
    data = DataTensorProd(field, [Data2D, Data1D], pbc=[False, True])
    operator = WaveletOperatorTensorProd(data, [2, 2])
    xp = backend.xp

    def product(x: Any, j: int, level: tuple[int, int]) -> Any:
        first = operator.apply(x, j, block=0, level=level)
        return operator.apply(first, j, block=1, level=level)

    def function(x: Any) -> Any:
        u = modulus(product(x, 0, (0, 0)))
        u = operator.downsample(operator.downsample(u, (0, 0), block=0), (1, 0), block=1)
        v = backend.complex_to_real_pairs(product(u, 1, (1, 1)))
        down = operator.downsample(x, (0, 0), block=1)
        return xp.concat([xp.reshape(v, (-1,)), xp.reshape(down, (-1,))])

    return data.array, function


@pytest.mark.slow
@pytest.mark.maths("M-W1-TENSORPROD", "M-NOT-TENSORPROD")
def test_jvp_and_hvp_of_the_tensor_prod_operator_match_finite_differences(
    backend: Backend,
) -> None:
    """The hooks of ``DataTensorProd``, through its wavelet operator, differentiated to second
    order: ``Backend.jvp`` and ``Backend.hvp`` against central differences, as for the planar
    operator (task T3.1).

    Slow: second-order derivatives belong to the full tier (AGENTS.md, golden rule 13; the
    decisions of the maintainer on issue #22)."""
    x, function = _layers_tensor_prod(backend)
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


def _layers_healpix(backend: Backend) -> tuple[Any, Callable[[Any], Any]]:
    """Two layers of the pyramid of a HEALPix field, ``nside`` 4 with one missing pixel, the
    smallest map whose first layer keeps valid pixels through a masked decimation (at ``nside``
    2 the stencil covers most of the sphere, and one missing pixel invalidates the whole first
    layer): ψ ⋆ D(|ψ ⋆ x|) and D(x), as :func:`_two_layers` (maths.md M-CONV-HEALPIX,
    M-DOWN-HEALPIX, M-MASK-HEALPIX; task T5.5)."""
    mask = np.zeros(192, dtype=bool)
    mask[70] = True
    field = np.random.default_rng(0).standard_normal((1, 2, 192)) + 0.2
    data = DataHealpix(field, mask=mask)
    operator = WaveletOperatorSingleDomain(data, 2)
    xp = backend.xp

    def function(x: Any) -> Any:
        u = operator.downsample(modulus(operator.apply(x, 0)), 0, history=(0,))
        v = backend.complex_to_real_pairs(operator.apply(u, 1))
        return xp.concat([xp.reshape(v, (-1,)), xp.reshape(operator.downsample(x, 0), (-1,))])

    return data.array, function


@pytest.mark.maths("M-CONV-HEALPIX", "M-DOWN-HEALPIX", "M-MASK-HEALPIX")
def test_jvp_and_hvp_of_the_healpix_operator_match_finite_differences(backend: Backend) -> None:
    """The hooks of ``DataHealpix`` (``gather_weighted``, ``segment_sum``), through its wavelet
    operator, differentiated to second order, as for the planar operator (issue #166,
    decision 5)."""
    x, function = _layers_healpix(backend)
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
