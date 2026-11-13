"""The ``ScatteringPlan``: index sets, levels, borders, masks and mean denominators (task T2.3,
issue #15; maths.md §1.3 to 1.5, M-MEAN).

The plan is NumPy float64 and does not depend on the backend or dtype of the run; the data are
built on the run's backend only because a DataClass holds a backend array. Comparisons are
exact: weights are 0 or 1 and denominators are pixel counts. ``tests/parity/test_plan.py``
compares the same quantities with STL.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass
from estel.data._masks import build_mask_maps
from estel.scattering._plan import MeanWeights, ScatteringPlan, border_width, build_plan
from estel.wavelets._banks import smoothing_kernel

GEOMETRIES: dict[int, type[DataClass]] = {2: Data2D, 1: Data1D}


def _data(ndim: int, n: int, pbc: bool, mask: NDArray[np.bool_] | None = None) -> DataClass:
    return GEOMETRIES[ndim](np.zeros((n,) * ndim), mask=mask, pbc=pbc)


def _plan(data: DataClass, J: int, L: int = 4) -> ScatteringPlan:  # noqa: N803
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13, tested on its own below
        return build_plan(data, J, L)


def _all_means(plan: ScatteringPlan) -> list[MeanWeights]:
    return [plan.moments, *plan.first_layer.values(), *plan.s3.values(), *plan.s4.values()]


def _crop(shape: tuple[int, ...], border: int) -> NDArray[np.float64]:
    """STL's ``_crop`` as weights, with its reduction of a border too large (OP-13)."""
    if border and min(shape) <= 2 * border:
        border = (min(shape) - 1) // 2
    return np.pad(np.ones(tuple(n - 2 * border for n in shape)), border)


def _sparse_mask(ndim: int, n: int, seed: int = 3) -> NDArray[np.bool_]:
    """A few missing pixels, so that every mean of a small plan keeps valid pixels."""
    rng = np.random.default_rng(seed)
    mask: NDArray[np.bool_] = rng.random((n,) * ndim) < (0.002 if ndim == 2 else 0.02)
    return mask


# ---------------------------------------------------------------------- index sets and levels


@pytest.mark.parametrize("J", [1, 2, 4])
def test_index_sets_follow_stls_loops(J: int) -> None:  # noqa: N803
    """for j3: S1, S2 at j3; for j2 ≤ j3: S3 at (j2, j3); for j1 ≤ j2: S4 at (j1, j2, j3)."""
    plan = _plan(_data(2, 32, pbc=True), J)
    loop_s3 = [(j2, j3) for j3 in range(J) for j2 in range(j3 + 1)]
    loop_s4 = [(j1, j2, j3) for j3 in range(J) for j2 in range(j3 + 1) for j1 in range(j2 + 1)]
    assert plan.scales == tuple(range(J))
    assert list(plan.s3_pairs) == loop_s3
    assert list(plan.s4_triples) == loop_s4
    assert len(plan.s3_pairs) == J * (J + 1) // 2
    assert len(plan.s4_triples) == math.comb(J + 2, 3)
    assert list(plan.first_layer) == list(plan.scales)
    assert list(plan.s3) == loop_s3
    assert list(plan.s4) == loop_s4


@pytest.mark.maths("M-DATA-2D", "M-DATA-1D")
@pytest.mark.parametrize("ndim", sorted(GEOMETRIES))
def test_levels_and_their_shapes(ndim: int) -> None:
    plan = _plan(_data(ndim, 32, pbc=False), 4, L=1)
    assert plan.levels == (0, 1, 2, 3)
    assert plan.level_shapes == tuple((32 // 2**d,) * ndim for d in range(4))
    assert (plan.J, plan.L, plan.kernel_size, plan.ndim) == (4, 1, 5, ndim)
    assert plan.spatial_shape == (32,) * ndim
    assert plan.pbc is False
    # Every mean is taken at the level of the last convolution of its operands.
    assert plan.moments.level == 0
    assert all(w.level == j for j, w in plan.first_layer.items())
    assert all(w.level == j3 for (_, j3), w in plan.s3.items())
    assert all(w.level == j3 for (_, _, j3), w in plan.s4.items())
    assert all(w.weights.shape == plan.level_shapes[w.level] for w in _all_means(plan))


def test_lowpass_is_the_cascade_of_smoothing_kernels() -> None:
    """Part II, §12.6 and maths.md M-NOT: in 1.0.0 the low-pass filter φ_J is the data
    downsampled J times."""
    plan = _plan(_data(2, 32, pbc=True), 3)
    assert plan.lowpass_realisation == "smoothing_cascade"
    assert plan.lowpass_level == 3


# ---------------------------------------------------------------------- borders


@pytest.mark.maths("M-BORDER")
def test_border_width_fully_flexible() -> None:
    """maths.md M-BORDER, written out for K = 5 (⌊K/2⌋ = 2), K = 3 and K = 7."""
    assert border_width((), 0, 5, pbc=False) == 0
    assert border_width((2,), 2, 5, pbc=True) == 0
    assert border_width((1, 3), 3, 5, pbc=True) == 0
    # One convolution at j, read at level d: ⌈2^(j-d) · 2⌉.
    assert border_width((0,), 0, 5, pbc=False) == 2
    assert border_width((0,), 1, 5, pbc=False) == 1
    assert border_width((0,), 2, 5, pbc=False) == 1  # ⌈0.5⌉
    assert border_width((0,), 3, 5, pbc=False) == 1  # ⌈0.25⌉
    assert border_width((3,), 3, 5, pbc=False) == 2
    # Two convolutions: β₁ = ⌈2^(j2-j3) · 2⌉, then ⌈2^(j3-d) · (β₁ + 2)⌉.
    assert border_width((2, 2), 2, 5, pbc=False) == 4  # β₁ = 2
    assert border_width((1, 2), 2, 5, pbc=False) == 3  # β₁ = 1
    assert border_width((0, 3), 3, 5, pbc=False) == 3  # β₁ = ⌈0.25⌉ = 1
    assert border_width((0, 1), 2, 5, pbc=False) == 2  # ⌈0.5 · (1 + 2)⌉
    assert border_width((1,), 1, 3, pbc=False) == 1
    assert border_width((1, 1), 1, 7, pbc=False) == 6
    assert border_width((0, 2), 2, 7, pbc=False) == 4  # β₁ = ⌈0.25 · 3⌉ = 1
    with pytest.raises(ValueError, match="at most twice"):
        border_width((0, 1, 2), 2, 5, pbc=False)


@pytest.mark.maths("M-BORDER")
@pytest.mark.parametrize("ndim", sorted(GEOMETRIES))
def test_mean_borders_are_the_larger_of_the_operands(ndim: int) -> None:
    plan = _plan(_data(ndim, 64, pbc=False), 4, L=1)
    assert plan.moments.border == 0
    assert all(w.border == 2 for w in plan.first_layer.values())
    for (j2, j3), w in plan.s3.items():
        assert w.border == max(border_width((j2, j3), j3, 5, False), 2)
        assert w.border == (4 if j2 == j3 else 3)
    for (j1, j2, j3), w in plan.s4.items():
        expected = max(border_width((j1, j3), j3, 5, False), border_width((j2, j3), j3, 5, False))
        assert w.border == expected


@pytest.mark.maths("M-BORDER", "M-MEAN")
@pytest.mark.parametrize("ndim", sorted(GEOMETRIES))
def test_weights_without_mask_are_the_crop(ndim: int) -> None:
    """Without missing values, the weights are the border crop and the denominators its size;
    with ``pbc`` nothing is cropped."""
    open_ = _plan(_data(ndim, 64, pbc=False), 4, L=1)
    periodic = _plan(_data(ndim, 64, pbc=True), 4, L=1)
    assert open_.masks is None
    for w in _all_means(open_):
        expected = _crop(open_.level_shapes[w.level], w.border)
        np.testing.assert_array_equal(w.weights, expected)
        assert w.denominator == expected.sum()
    for w in _all_means(periodic):
        assert w.border == 0
        assert np.all(w.weights == 1.0)
        assert w.denominator == w.weights.size
    assert open_.moments.denominator == 64**ndim
    assert open_.first_layer[0].denominator == 60**ndim


@pytest.mark.maths("M-BORDER")
@pytest.mark.parametrize("ndim", sorted(GEOMETRIES))
def test_a_border_too_large_is_reduced_with_one_warning(ndim: int) -> None:
    """OP-13: at N = 32, J = 4, level 3 has 4 pixels per axis, too few for the borders 2, 3
    and 4 of S1, S3 and S4; STL reduces them to 1 and warns once per operator."""
    data = _data(ndim, 32, pbc=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = build_plan(data, 4, 1)
    assert [str(w.message).count("OP-13") for w in caught] == [1]
    assert plan.first_layer[3].border == 2
    assert plan.first_layer[3].denominator == 2**ndim
    assert plan.s3[(3, 3)].border == 4
    assert plan.s3[(3, 3)].denominator == 2**ndim
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_plan(_data(ndim, 32, pbc=True), 4, 1)  # periodic: no crop, no warning


# ---------------------------------------------------------------------- masks


def _expected_invalid(
    ndim: int,
    mask: NDArray[np.bool_],
    J: int,  # noqa: N803
    pbc: bool,
) -> dict[Any, NDArray[np.bool_]]:
    """The invalid pixels of each mean, from maths.md M-MASK written out with the maps of
    ``build_mask_maps``: the union of the operands' masks."""
    maps = build_mask_maps(mask, J, smoothing_kernel(ndim), 5, GEOMETRIES[ndim]._nan_threshold, pbc)
    expected: dict[Any, NDArray[np.bool_]] = {"moments": maps.full_res}
    for j3 in range(J):
        expected[("S1", j3)] = maps.layer1[j3]
        for j2 in range(j3 + 1):
            expected[("S3", j2, j3)] = maps.layer2[(j3, j2)] | maps.layer1[j3]
            for j1 in range(j2 + 1):
                expected[("S4", j1, j2, j3)] = maps.layer2[(j3, j1)] | maps.layer2[(j3, j2)]
    return expected


@pytest.mark.maths("M-MASK", "M-MEAN", "M-BORDER")
@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("ndim", sorted(GEOMETRIES))
def test_masked_weights_are_crop_times_valid_union(ndim: int, pbc: bool) -> None:
    n = 64 if ndim == 2 else 128
    mask = _sparse_mask(ndim, n)
    plan = _plan(_data(ndim, n, pbc, mask), 3, L=1)
    assert plan.masks is not None
    expected = _expected_invalid(ndim, mask, 3, pbc)
    got: dict[Any, MeanWeights] = {
        "moments": plan.moments,
        **{("S1", j): w for j, w in plan.first_layer.items()},
        **{("S3", *k): w for k, w in plan.s3.items()},
        **{("S4", *k): w for k, w in plan.s4.items()},
    }
    assert set(got) == set(expected)
    for key, w in got.items():
        crop = _crop(plan.level_shapes[w.level], w.border)
        np.testing.assert_array_equal(w.weights, np.where(expected[key], 0.0, crop), str(key))
        assert w.denominator == float(w.weights.sum()) > 0
        assert not w.weights.flags.writeable


@pytest.mark.maths("M-MASK")
def test_s3_uses_the_union_where_the_layer_2_mask_misses_layer_1() -> None:
    """maths.md M-MASK: a covariance uses the union of its operands' masks. STL takes the
    layer-2 mask alone, assuming that it contains the layer-1 mask; on this 1D signal it does
    not, at (j2, j3) = (0, 2), and the union removes more samples (reported on issue #15)."""
    rng = np.random.default_rng(112)
    mask = rng.random(64) < rng.uniform(0.01, 0.15)
    plan = _plan(_data(1, 64, pbc=True, mask=mask), 3, L=1)
    assert plan.masks is not None
    layer1, layer2 = plan.masks.layer1[2], plan.masks.layer2[(2, 0)]
    assert np.any(layer1 & ~layer2)
    np.testing.assert_array_equal(plan.s3[(0, 2)].weights, np.where(layer1 | layer2, 0.0, 1.0))
    assert plan.s3[(0, 2)].denominator == float((~layer2).sum() - (layer1 & ~layer2).sum())


@pytest.mark.maths("M-MEAN")
@pytest.mark.parametrize("ndim", sorted(GEOMETRIES))
def test_a_mean_without_valid_pixel_raises(ndim: int) -> None:
    """OP-3: STL divides by zero; ESTEL refuses to build the plan, naming the mean."""
    n = 32
    mask = np.zeros((n,) * ndim, dtype=bool)
    mask[(slice(8, 24),) * ndim] = True  # a large hole: nothing valid at level 3
    with pytest.raises(ValueError, match=r"no pixel is left for the mean of .*OP-3"):
        _plan(_data(ndim, n, pbc=False, mask=mask), 4, L=1)


@pytest.mark.maths("M-DOWN-MASK")
def test_downsample_maps_are_the_steps_of_stls_downsample() -> None:
    mask = _sparse_mask(2, 64)
    plan = _plan(_data(2, 64, pbc=False, mask=mask), 3)
    maps = plan.masks
    assert maps is not None
    step = plan.downsample_maps(0)
    assert step is not None
    np.testing.assert_array_equal(step[0], ~maps.full_res)
    np.testing.assert_array_equal(step[1], maps.smooth_reweight[1])
    step = plan.downsample_maps(1)
    assert step is not None
    np.testing.assert_array_equal(step[0], ~maps.smooth_invalid[1])
    np.testing.assert_array_equal(step[1], maps.smooth_reweight[2])
    step = plan.downsample_maps(0, convolved_at=0)
    assert step is not None
    np.testing.assert_array_equal(step[0], ~maps.layer1[0])
    np.testing.assert_array_equal(step[1], maps.wav_reweight[(1, 0)])
    step = plan.downsample_maps(1, convolved_at=0)
    assert step is not None
    np.testing.assert_array_equal(step[0], ~maps.wav_invalid[(1, 0)])
    np.testing.assert_array_equal(step[1], maps.wav_reweight[(2, 0)])
    assert _plan(_data(2, 64, pbc=False), 3).downsample_maps(1, convolved_at=1) is None
    with pytest.raises(ValueError, match="levels 0"):
        plan.downsample_maps(2)
    with pytest.raises(ValueError, match="convolved_at"):
        plan.downsample_maps(0, convolved_at=1)


# ---------------------------------------------------------------------- static, hashable


def test_plan_is_hashable_and_compares_by_its_inputs() -> None:
    mask = _sparse_mask(2, 64)
    a = _plan(_data(2, 64, pbc=False, mask=mask), 3)
    b = _plan(_data(2, 64, pbc=False, mask=mask.copy()), 3)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    other_mask = mask.copy()
    other_mask[0, 0] = not other_mask[0, 0]
    assert a != _plan(_data(2, 64, pbc=False, mask=other_mask), 3)
    assert a != _plan(_data(2, 64, pbc=True, mask=mask), 3)
    assert a != _plan(_data(2, 64, pbc=False, mask=mask), 2)
    assert a != _plan(_data(2, 64, pbc=False, mask=mask), 3, L=2)
    assert _plan(_data(1, 64, pbc=True), 3) != _plan(_data(2, 64, pbc=True), 3)


def test_plan_is_immutable() -> None:
    plan = _plan(_data(2, 32, pbc=False), 3)
    with pytest.raises(AttributeError):
        plan.J = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.s3[(0, 0)] = plan.s3[(0, 1)]  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        plan.moments.weights[0, 0] = 0.0


# ---------------------------------------------------------------------- validation


@pytest.mark.maths("M-DATA-2D")
def test_sizes_must_be_divisible_by_two_to_the_j_minus_one() -> None:
    """OP-8: ESTEL rejects J when a size is not divisible by 2^(J-1), with the largest J."""
    data = GEOMETRIES[2](np.zeros((24, 32)), pbc=True)
    _plan(data, 4)  # 24 = 3 · 2^3
    with pytest.raises(ValueError, match=r"2\^\(J-1\) = 16.*J ≤ 4"):
        _plan(data, 5)


def test_build_plan_rejects_bad_arguments() -> None:
    data = _data(2, 32, pbc=True)
    with pytest.raises(TypeError, match="DataClass"):
        build_plan(np.zeros((32, 32)), 3, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        build_plan(data, 2.0, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        build_plan(data, True, 4)
    with pytest.raises(ValueError, match="at least 1"):
        build_plan(data, 0, 4)
    with pytest.raises(ValueError, match="at least 1"):
        build_plan(data, 3, 0)
    with pytest.raises(ValueError, match="odd"):
        build_plan(data, 3, 4, kernel_size=4)
