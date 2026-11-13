"""Unit tests of the static masks and reweighting maps (maths.md M-DOWN-MASK, M-MASK; T2.1).

Small cases computed by hand. Parity with STL is in ``tests/parity/test_masks.py``. The maps are
built in NumPy float64 whatever the run's dtype and backend (tag N of Part II, §12).
"""

from __future__ import annotations

import numpy as np
import pytest

from estel.data._masks import build_mask_maps, downsample_step, propagate_mask_conv

H_AVG_1D = np.array([0.5, 0.5])
H_AVG_2D = np.full((2, 2), 0.25)
# The 1D anti-aliasing kernel of maths.md M-H-1D, and its 2D product (M-H).
H_1D = np.array([-0.04879268, 0.05638086, 0.49241182, 0.49241182, 0.05638086, -0.04879268])
H_1D = H_1D / H_1D.sum()
H_2D = np.outer(H_1D, H_1D)


def test_downsample_step_with_an_average_kernel_averages_pairs() -> None:
    a = np.arange(8.0)
    np.testing.assert_allclose(downsample_step(a, H_AVG_1D, pbc=True), [0.5, 2.5, 4.5, 6.5])
    b = np.arange(16.0).reshape(4, 4)
    expected = [[2.5, 4.5], [10.5, 12.5]]
    np.testing.assert_allclose(downsample_step(b, H_AVG_2D, pbc=False), expected)


def test_downsample_step_pads_two_before_and_three_after_for_a_kernel_of_six() -> None:
    """M-DOWN: x̃ padded by 2 before and 3 after; output n reads x̃[2n … 2n+5]."""
    a = np.zeros(8)
    impulse = np.where(np.arange(8) == 0, 1.0, 0.0)
    # Output n reads inputs 2n-2 … 2n+3: input 0 is read by n = 0 (tap 2) and n = 1 (tap 0).
    out = downsample_step(impulse + a, H_1D, pbc=True)
    np.testing.assert_allclose(out, [H_1D[2], H_1D[0], 0.0, H_1D[4]])


@pytest.mark.parametrize("pbc", [False, True])
def test_downsample_step_keeps_a_constant(pbc: bool) -> None:
    np.testing.assert_allclose(downsample_step(np.full((8, 8), 3.0), H_2D, pbc), 3.0)


def test_downsample_step_rejects_odd_sizes_and_wrong_kernels() -> None:
    with pytest.raises(ValueError, match="divisible by 2"):
        downsample_step(np.zeros(6 + 1), H_1D, pbc=True)
    with pytest.raises(ValueError, match="axes"):
        downsample_step(np.zeros((4, 4)), H_1D, pbc=True)
    with pytest.raises(ValueError, match="wraps more than once"):
        downsample_step(np.zeros(2), H_1D, pbc=True)


def test_propagate_mask_conv_replicate_and_circular_borders() -> None:
    """A missing first sample, K = 5: outputs within 2 samples are invalid; with circular
    padding the footprint also wraps to the end of the axis."""
    mask = np.arange(8) == 0
    replicate = propagate_mask_conv(mask, (5,), pbc=False)
    circular = propagate_mask_conv(mask, (5,), pbc=True)
    np.testing.assert_array_equal(np.flatnonzero(replicate), [0, 1, 2])
    np.testing.assert_array_equal(np.flatnonzero(circular), [0, 1, 2, 6, 7])


def test_propagate_mask_conv_in_2d_is_a_box() -> None:
    mask = np.zeros((9, 9), dtype=bool) | (np.arange(9)[:, None] == 4) & (np.arange(9) == 4)
    out = propagate_mask_conv(mask, (5, 5), pbc=True)
    box = (np.abs(np.arange(9) - 4)[:, None] <= 2) & (np.abs(np.arange(9) - 4) <= 2)
    np.testing.assert_array_equal(out, box)


@pytest.mark.parametrize("pbc", [False, True])
def test_no_missing_value_gives_valid_maps_and_unit_weights(pbc: bool) -> None:
    maps = build_mask_maps(np.zeros((16, 16), dtype=bool), 3, H_2D, 5, 0.33, pbc)
    for invalid in [*maps.smooth_invalid.values(), *maps.wav_invalid.values()]:
        assert not invalid.any()
    for invalid in [*maps.layer1.values(), *maps.layer2.values()]:
        assert not invalid.any()
    for weight in [*maps.smooth_reweight.values(), *maps.wav_reweight.values()]:
        np.testing.assert_array_equal(weight, 1.0)


def test_maps_have_the_shape_of_their_level() -> None:
    maps = build_mask_maps(np.arange(32) == 3, 4, H_1D, 5, 0.5, pbc=False)
    assert maps.full_res.shape == (32,)
    assert {d: m.shape for d, m in maps.smooth_invalid.items()} == {1: (16,), 2: (8,), 3: (4,)}
    assert {k: m.shape[0] for k, m in maps.wav_invalid.items()} == {
        (d, j): 32 >> d for d in range(1, 4) for j in range(d)
    }
    assert {j: m.shape[0] for j, m in maps.layer1.items()} == {j: 32 >> j for j in range(4)}
    assert {k: m.shape[0] for k, m in maps.layer2.items()} == {
        (j3, j2): 32 >> j3 for j3 in range(4) for j2 in range(j3 + 1)
    }


def test_threshold_reweighting_and_invalid_pixels() -> None:
    """With the averaging kernel, the fraction of invalid input is 0, 1/2 or 1: at τ = 0.5 a
    half-missing pair is valid with weight 2 (f ≤ τ), a fully missing pair invalid with weight
    0; at τ = 0.33 the half-missing pair is invalid too."""
    mask = np.array([False, False, True, False, True, True, False, False])
    maps = build_mask_maps(mask, 2, H_AVG_1D, 1, 0.5, pbc=True)
    np.testing.assert_array_equal(maps.smooth_invalid[1], [False, False, True, False])
    np.testing.assert_allclose(maps.smooth_reweight[1], [1.0, 2.0, 0.0, 1.0])
    strict = build_mask_maps(mask, 2, H_AVG_1D, 1, 0.33, pbc=True)
    np.testing.assert_array_equal(strict.smooth_invalid[1], [False, True, True, False])
    np.testing.assert_allclose(strict.smooth_reweight[1], [1.0, 0.0, 0.0, 1.0])


def test_layer_masks_chain_the_levels() -> None:
    """With K = 1 the footprint is one pixel: layer 1 at level j is the layer-0 mask of level
    j, and layer 2 at (j3, j2 < j3) is layer 1 of j2 brought down to j3."""
    mask = np.array([True, True, False, False, False, False, True, False])
    maps = build_mask_maps(mask, 3, H_AVG_1D, 1, 0.5, pbc=True)
    np.testing.assert_array_equal(maps.layer1[0], mask)
    np.testing.assert_array_equal(maps.layer1[1], maps.smooth_invalid[1])
    np.testing.assert_array_equal(maps.layer2[(1, 1)], maps.layer1[1])
    np.testing.assert_array_equal(maps.layer2[(2, 0)], maps.wav_invalid[(2, 0)])
    # STL downsamples the thresholded mask of the previous level, not its fraction.
    np.testing.assert_array_equal(maps.smooth_invalid[1], [True, False, False, False])
    np.testing.assert_array_equal(maps.smooth_invalid[2], [False, False])
    np.testing.assert_allclose(maps.smooth_reweight[2], [2.0, 1.0])


def test_build_mask_maps_rejects_a_kernel_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sums to"):
        build_mask_maps(np.zeros(8, dtype=bool), 2, 2 * H_1D, 5, 0.5, pbc=True)
    with pytest.raises(ValueError, match="at least one level"):
        build_mask_maps(np.zeros(8, dtype=bool), 0, H_1D, 5, 0.5, pbc=True)
