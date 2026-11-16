"""The internal ``WaveletOperator`` (Part I, §2.3; Part II, §12.5; task T2.10): its functions
against the geometry hooks and the maths entries they implement.

The hooks are tested against NumPy references in ``test_hooks.py``, the mask maps in
``test_masks.py`` and the weights of every mean of a plan in ``tests/scattering/test_plan.py``;
here the wavelet operator is checked to call them with the right tables, maps and levels, so
the comparisons are exact. Its derivatives are in ``tests/differentiation/``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data, Data1D, Data2D
from estel.backend._protocol import Backend
from estel.data._base import DownsampleWeights
from estel.data._masks import build_mask_maps
from estel.data._moments import modulus
from estel.wavelets._banks import bump_steerable_1d, bump_steerable_2d, smoothing_kernel
from estel.wavelets._operator import (
    COMPLEX,
    WaveletOperator,
    build_wavelet_filters,
    build_wavelet_plan,
)

GEOMETRIES = {2: Data2D, 1: Data1D}
N = 16


def _data(ndim: int, pbc: bool, masked: bool = False, lead: tuple[int, ...] = (2, 3)) -> Data:
    """Seeded data; the mask, two pixels in a row along each axis, is far from the borders."""
    mask = None
    if masked:
        mask = np.zeros((N,) * ndim, dtype=bool)
        mask[(slice(6, 8),) * ndim] = True
    field = np.random.default_rng(0).standard_normal((*lead, *(N,) * ndim))
    return GEOMETRIES[ndim](field, mask=mask, pbc=pbc)


def _equal(backend: Backend, actual: Any, expected: Any) -> None:
    np.testing.assert_array_equal(backend.to_numpy(actual), backend.to_numpy(expected))


def _dtype(backend: Backend, x: Any) -> Any:
    return "float64" if x.dtype == backend.xp.float64 else "float32"


@pytest.mark.maths("M-W1", "M-WAV-2D", "M-WAV-1D")
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("ndim", [2, 1])
def test_apply_is_the_hook_conv_with_the_bank_of_the_geometry(
    backend: Backend, ndim: int, pbc: bool
) -> None:
    """``apply(x, j)`` is ``_conv`` with the bump-steerable bank, on the field and, with the
    orientation axis of a first convolution kept, on the first layer: W x, then ψ ⋆ |W x|."""
    data = _data(ndim, pbc)
    L = 2  # noqa: N806
    operator = WaveletOperator(data, 2, L)
    x = data.array
    bank = bump_steerable_2d(L) if ndim == 2 else bump_steerable_1d(conjugate=True)
    kernels = backend.asarray(bank, COMPLEX[_dtype(backend, x)])
    w = operator.apply(x, 0)
    assert tuple(w.shape) == (2, 3, L, *(N,) * ndim)
    _equal(backend, w, data._conv(x, kernels, 0))

    u = operator.downsample(modulus(w), 0, convolved_at=0)
    v = operator.apply(u, 1)
    half = (N // 2,) * ndim
    assert tuple(v.shape) == (2, 3, L, L, *half)
    xp = backend.xp
    flat = data._conv(xp.reshape(u, (2, 3 * L, *half)), kernels, 1)
    _equal(backend, v, xp.reshape(flat, (2, 3, L, L, *half)))


@pytest.mark.maths("M-DOWN", "M-DOWN-MASK", "M-MASK")
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("ndim", [2, 1])
def test_downsample_is_the_hook_step_with_the_maps_of_its_level(
    backend: Backend, ndim: int, masked: bool
) -> None:
    """``downsample(x, level)`` is ``_downsample`` with the smoothing kernel and, with missing
    values, the valid pixels and the reweighting of that step: those of the field, or those of
    first-layer data convolved at ``convolved_at``."""
    data = _data(ndim, pbc=False, masked=masked)
    operator = WaveletOperator(data, 3, 1)
    x = data.array
    dtype = _dtype(backend, x)
    s = backend.asarray(smoothing_kernel(ndim), dtype)
    maps = None
    if data.mask is not None:
        maps = build_mask_maps(
            data.mask, 3, smoothing_kernel(ndim), 5, type(data)._nan_threshold, pbc=False
        )

    def weights(invalid: NDArray[np.bool_] | None, reweight: Any) -> DownsampleWeights:
        if invalid is None:
            return DownsampleWeights(s)
        return DownsampleWeights(
            s, backend.asarray(~invalid, "bool"), backend.asarray(reweight, dtype)
        )

    def step(invalid: str, reweight: str, at: Any, key: Any) -> DownsampleWeights:
        if maps is None:
            return weights(None, None)
        found = getattr(maps, invalid)
        return weights(found if at is None else found[at], getattr(maps, reweight)[key])

    x1 = operator.downsample(x, 0)
    _equal(backend, x1, data._downsample(x, step("full_res", "smooth_reweight", None, 1), 0))
    x2 = operator.downsample(x1, 1)
    _equal(backend, x2, data._downsample(x1, step("smooth_invalid", "smooth_reweight", 1, 2), 1))
    assert tuple(x2.shape) == (2, 3, *(N // 4,) * ndim)

    u = modulus(operator.apply(x, 0))[:, :, 0]
    u1 = operator.downsample(u, 0, convolved_at=0)
    _equal(backend, u1, data._downsample(u, step("layer1", "wav_reweight", 0, (1, 0)), 0))
    u2 = operator.downsample(u1, 1, convolved_at=0)
    expected = data._downsample(u1, step("wav_invalid", "wav_reweight", (1, 0), (2, 0)), 1)
    _equal(backend, u2, expected)


@pytest.mark.maths("M-BORDER", "M-MASK", "M-MEAN")
def test_mean_weights_are_the_crop_times_the_union_of_the_masks() -> None:
    """The weights of a covariance: the crop of the larger border of its operands (M-BORDER),
    0 on the union of their masks (M-MASK); ``None`` when no pixel is left (OP-3); and one
    warning for all the borders too large for the grid (OP-13)."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[20:22, 20:22] = True
    large = Data2D(np.zeros((32, 32)), mask=mask, pbc=False)
    operator = WaveletOperator.from_parts(large, build_wavelet_plan(large, 2, 4))
    plan = operator.plan
    assert plan.masks is not None
    assert (plan.border((0, 1), 1), plan.border((1,), 1)) == (3, 2)
    found = operator.mean_weights({"s3": (((0, 1), (1,)), 1), "field": (((),), 0)})
    s3, field = found["s3"], found["field"]
    assert s3 is not None
    assert field is not None
    crop = np.pad(np.ones((10, 10)), 3)
    invalid = plan.masks.layer2[(1, 0)] | plan.masks.layer1[1]
    np.testing.assert_array_equal(s3.weights, np.where(invalid, 0.0, crop))
    assert (s3.level, s3.border, s3.denominator) == (1, 3, s3.weights.sum())
    assert 0 < s3.denominator < 100  # the crop keeps 10 x 10 pixels, the mask removes some
    np.testing.assert_array_equal(field.weights, (~mask).astype(float))

    data = _data(2, pbc=False, masked=True)
    small = WaveletOperator.from_parts(data, build_wavelet_plan(data, 3, 4))
    with pytest.warns(UserWarning, match="OP-13") as caught:
        empty = small.mean_weights({"a": (((0, 2),), 2), "b": (((1, 2),), 2)})
    assert len(caught) == 1
    assert empty == {"a": None, "b": None}  # 4 x 4 pixels, all under the propagated mask


def test_static_part_gives_the_levels_and_is_hashable() -> None:
    """The static part: the shape of each level, the low-pass filter of 1.0.0 (M-LP), and an
    identity by value, so that the ``STPlan`` that holds it can be a static argument."""
    data = _data(2, pbc=True, masked=True)
    plan = build_wavelet_plan(data, 3)
    assert (plan.J, plan.L, plan.kernel_size, plan.levels) == (3, 4, 5, (0, 1, 2))
    assert plan.level_shapes == ((16, 16), (8, 8), (4, 4))
    assert (plan.lowpass_realization, plan.lowpass_level) == ("smoothing_cascade", 3)
    same = build_wavelet_plan(_data(2, pbc=True, masked=True, lead=(1, 1)), 3, 4)
    assert plan == same
    assert hash(plan) == hash(same)
    assert plan != build_wavelet_plan(data, 2)
    assert plan != build_wavelet_plan(_data(2, pbc=True), 3)


def test_wavelet_operator_rejects_bad_arguments(backend: Backend) -> None:
    data = _data(1, pbc=True, lead=(1, 1))
    operator = WaveletOperator(data, 3)
    x = data.array
    with pytest.raises(ValueError, match="level 1 first"):
        operator.apply(x, 1)  # a map at level 0
    with pytest.raises(ValueError, match="scales 0 … 2"):
        operator.apply(x, 3)
    with pytest.raises(ValueError, match="levels 0 … 1"):
        operator.downsample(x, 2)
    with pytest.raises(ValueError, match="convolved_at"):
        operator.downsample(x, 0, convolved_at=1)
    static = WaveletOperator.from_parts(data, operator.plan)
    with pytest.raises(ValueError, match="static part only"):
        static.apply(x, 0)
    with pytest.raises(AttributeError, match="immutable"):
        operator.plan = static.plan  # type: ignore[misc]
    with pytest.raises(ValueError, match="bump_steerable"):
        WaveletOperator(data, 3, wavelet="morlet")
    with pytest.raises(ValueError, match="L = 2"):
        WaveletOperator(data, 3, 3)
    with pytest.raises(ValueError, match=r"2\^\(J-1\)"):
        WaveletOperator(data, 6)
    with pytest.raises(TypeError, match="Data object"):
        build_wavelet_plan(np.zeros(16))  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filters = build_wavelet_filters(operator.plan, data.backend, "float64", None)
    assert tuple(filters.kernels.shape) == (1, 5)
    assert not filters.field_down
