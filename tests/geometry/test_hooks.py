"""The planar geometry hooks ``_conv`` and ``_downsample`` (Part II, §12.4; maths.md M-CONV-2D,
M-W1, M-DOWN, M-DOWN-MASK; task T2.2).

Run on the backend and dtype of the test run, against a NumPy float64 reference written here:
padding (circular if ``pbc``, replicate otherwise), then a correlation without flipping the
kernel. Tolerances: Part I, §5.2 (``docs/agents/testing.md``), float32 rtol 1e-4 / atol 1e-6,
float64 rtol 1e-10 / atol 1e-12, with atol relative to the largest expected modulus.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass
from estel.backend._protocol import Backend, DType
from estel.data._base import DownsampleWeights
from estel.data._masks import build_mask_maps
from estel.wavelets._banks import bump_steerable_1d, bump_steerable_2d, smoothing_kernel

TOLERANCES = {"float32": (1e-4, 1e-6), "float64": (1e-10, 1e-12)}
COMPLEX: dict[str, DType] = {"float32": "complex64", "float64": "complex128"}
GEOMETRIES = {2: Data2D, 1: Data1D}


def _assert_close(backend: Backend, actual: Any, expected: NDArray[Any], dtype: str) -> None:
    rtol, atol = TOLERANCES[dtype]
    got = np.asarray(backend.to_numpy(actual))
    assert got.shape == expected.shape
    scale = float(np.abs(expected).max()) or 1.0
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * scale)


def _geometry(ndim: int, n: int, pbc: bool) -> DataClass:
    return GEOMETRIES[ndim](np.zeros((n,) * ndim), pbc=pbc)


def _field(shape: tuple[int, ...], seed: int = 0, complex_: bool = False) -> NDArray[Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    return x + 1j * rng.standard_normal(shape) if complex_ else x


def _bank(ndim: int) -> NDArray[np.complex128]:
    return bump_steerable_2d(4) if ndim == 2 else bump_steerable_1d(conjugate=True)


def _correlate(
    a: NDArray[Any], k: NDArray[Any], before: int, after: int, pbc: bool, stride: int
) -> NDArray[Any]:
    """NumPy reference: pad every spatial axis (the last k.ndim), correlate, keep one in stride."""
    d = k.ndim
    widths = [(0, 0)] * (a.ndim - d) + [(before, after)] * d
    padded = np.pad(a, widths, mode="wrap" if pbc else "edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k.shape, axis=tuple(range(-d, 0)))
    index: tuple[Any, ...] = (Ellipsis, *[slice(None, None, stride)] * d, *[slice(None)] * d)
    windows = windows[index]
    return np.tensordot(windows, k, axes=(tuple(range(-d, 0)), tuple(range(d))))


@pytest.mark.maths("M-CONV-2D", "M-W1")
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("ndim", [2, 1])
def test_conv_is_padded_correlation(
    backend: Backend, dtype: DType, ndim: int, pbc: bool, complex_input: bool
) -> None:
    data = _geometry(ndim, 16, pbc)
    kernels = _bank(ndim)
    x = _field((2, 3) + (16,) * ndim, complex_=complex_input)
    in_dtype = COMPLEX[dtype] if complex_input else dtype
    out = data._conv(backend.asarray(x, in_dtype), backend.asarray(kernels, COMPLEX[dtype]), 0)
    # One output per orientation, inserted before the spatial axes: (Nb, Nc, L, *spatial).
    expected = np.stack([_correlate(x, k, 2, 2, pbc, 1) for k in kernels], axis=2)
    _assert_close(backend, out, expected, dtype)
    assert np.asarray(backend.to_numpy(out)).dtype == np.dtype(COMPLEX[dtype])


@pytest.mark.maths("M-CONV-2D")
def test_conv_does_not_depend_on_level(backend: Backend, dtype: DType) -> None:
    data = _geometry(2, 8, True)
    x = backend.asarray(_field((1, 1, 8, 8)), dtype)
    k = backend.asarray(_bank(2), COMPLEX[dtype])
    np.testing.assert_array_equal(
        backend.to_numpy(data._conv(x, k, 0)), backend.to_numpy(data._conv(x, k, 3))
    )


def test_conv_rejects_a_real_or_misshaped_bank(backend: Backend, dtype: DType) -> None:
    data = _geometry(2, 8, True)
    x = backend.asarray(_field((1, 1, 8, 8)), dtype)
    with pytest.raises(TypeError, match="complex wavelet bank"):
        data._conv(x, backend.asarray(np.ones((4, 5, 5)), dtype), 0)
    with pytest.raises(TypeError, match="complex wavelet bank"):
        data._conv(x, backend.asarray(_bank(1), COMPLEX[dtype]), 0)
    with pytest.raises(ValueError, match="spatial axis"):
        data._conv(
            backend.asarray(_field((8, 8)), dtype), backend.asarray(_bank(2), "complex128"), 0
        )


@pytest.mark.maths("M-DOWN")
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("ndim", [2, 1])
def test_downsample_is_strided_correlation_with_the_smoothing_kernel(
    backend: Backend, dtype: DType, ndim: int, pbc: bool, complex_input: bool
) -> None:
    data = _geometry(ndim, 16, pbc)
    h = smoothing_kernel(ndim)
    x = _field((2, 3) + (16,) * ndim, complex_=complex_input)
    in_dtype = COMPLEX[dtype] if complex_input else dtype
    out = data._downsample(
        backend.asarray(x, in_dtype), DownsampleWeights(backend.asarray(h, dtype)), 0
    )
    # Padded by 2 before and 3 after, stride 2 (M-DOWN, OP-2).
    _assert_close(backend, out, _correlate(x, h, 2, 3, pbc, 2), dtype)


@pytest.mark.maths("M-DOWN-MASK")
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("ndim", [2, 1])
def test_masked_downsample_zeroes_invalid_inputs_and_reweights(
    backend: Backend, dtype: DType, ndim: int, pbc: bool
) -> None:
    """M-DOWN-MASK with the static maps of ``_masks.py``: NaN at missing pixels are never read,
    in the values or in the gradient (OP-27)."""
    n = 16
    rng = np.random.default_rng(3)
    mask = rng.random((n,) * ndim) < 0.15
    data = GEOMETRIES[ndim](np.zeros((n,) * ndim), mask=mask, pbc=pbc)
    h = smoothing_kernel(ndim)
    maps = build_mask_maps(mask, 2, h, 5, data._nan_threshold, pbc)
    field = _field((2, 1) + (n,) * ndim)
    x = np.where(mask, np.nan, field)
    weights = DownsampleWeights(
        backend.asarray(h, dtype),
        backend.asarray(~mask, "bool"),
        backend.asarray(maps.smooth_reweight[1], dtype),
    )
    out = data._downsample(backend.asarray(x, dtype), weights, 0)
    expected = _correlate(np.where(mask, 0.0, field), h, 2, 3, pbc, 2) * maps.smooth_reweight[1]
    _assert_close(backend, out, expected, dtype)

    xp = backend.xp
    _, grad = backend.value_and_grad(lambda a: xp.sum(data._downsample(a, weights, 0)))(
        backend.asarray(x, dtype)
    )
    grad = np.asarray(backend.to_numpy(grad))
    assert np.isfinite(grad).all()
    assert (grad[..., mask] == 0.0).all()


def test_downsample_rejects_bad_inputs(backend: Backend, dtype: DType) -> None:
    data = _geometry(2, 8, True)
    h = backend.asarray(smoothing_kernel(2), dtype)
    x = backend.asarray(_field((1, 1, 8, 8)), dtype)
    with pytest.raises(ValueError, match="smoothing_kernel"):
        data._downsample(x, DownsampleWeights(backend.asarray(smoothing_kernel(1), dtype)), 0)
    with pytest.raises(ValueError, match="or neither"):
        data._downsample(
            x, DownsampleWeights(h, valid=backend.asarray(np.ones((8, 8), bool), "bool")), 0
        )
    odd = _geometry(2, 6, True)
    with pytest.raises(ValueError, match="divisible by 2"):
        odd._downsample(
            backend.asarray(_field((1, 1, 6, 6))[..., :5, :5], dtype), DownsampleWeights(h), 0
        )


def test_mother_class_hooks_say_to_override() -> None:
    with pytest.raises(NotImplementedError, match="overrides"):
        DataClass._conv(None, None, None, 0)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="overrides"):
        DataClass._downsample(None, None, None, 0)  # type: ignore[arg-type]
