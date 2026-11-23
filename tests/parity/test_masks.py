"""Parity of the static masks and reweighting maps with STL (task T2.1, issue #13).

Compares ``estel.data._masks.build_mask_maps`` with the fixtures ``masks_2d`` (STL `master`) and
``masks_1d`` (STL `2D+1D`), for both padding modes (maths.md M-DOWN-MASK, M-MASK). The masks
must be equal; the reweighting maps equal within the float64 parity tolerances of Part II, §14.3,
with NaN where STL marks a pixel invalid. ESTEL stores a boolean mask and a reweighting of 0
there instead (OP-27), turned back into NaN here for the comparison.

The maps are built with ESTEL's own smoothing kernel, ``smoothing_kernel(ndim)`` (maths.md
M-SMOOTH-2D, M-SMOOTH-1D; task T2.2, issue #14), not with the kernel STL used, stored in the
fixture as ``smooth_kernel``: ``tests/parity/test_kernels.py`` compares the two kernels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel.data._masks import MaskMaps, build_mask_maps
from estel.data._planar import Data1D, Data2D
from estel.wavelets._banks import smoothing_kernel
from tests.parity._compare import assert_parity, load_fixture, read_manifest

CASES: dict[str, type[Data2D] | type[Data1D]] = {"masks_2d": Data2D, "masks_1d": Data1D}
MODES = {"replicate": False, "circular": True}


def _maps(case: str, pbc: bool) -> tuple[MaskMaps, dict[str, NDArray[Any]], dict[str, Any]]:
    fixture = load_fixture(case)
    parameters = read_manifest()[f"{case}.npz"]["parameters"]
    maps = build_mask_maps(
        fixture["mask_full_res"] > 0,
        scale_levels=tuple(range(parameters["J"])),
        s=smoothing_kernel(CASES[case].ndim),
        kernel_size=parameters["K"],
        threshold=CASES[case]._nan_threshold,
        pbc=pbc,
    )
    return maps, fixture, parameters


def _with_nan(invalid: NDArray[np.bool_], reweight: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(invalid, np.nan, reweight)


@pytest.mark.parametrize("case", sorted(CASES))
def test_threshold_is_stl_default(case: str) -> None:
    parameters = read_manifest()[f"{case}.npz"]["parameters"]
    assert CASES[case]._nan_threshold == parameters["threshold"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_mask_from_nan_equals_stl(case: str) -> None:
    """The mask ESTEL reads from the NaN of the data is STL's ``mask_full_res``."""
    expected = load_fixture(case)["mask_full_res"] > 0
    field = np.where(expected, np.nan, np.random.default_rng(0).standard_normal(expected.shape))
    data = CASES[case](field, pbc=False)
    assert data.mask is not None
    np.testing.assert_array_equal(data.mask, expected)


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("case", sorted(CASES))
def test_layer_masks_equal_stl(case: str, mode: str) -> None:
    maps, fixture, parameters = _maps(case, MODES[mode])
    n_scales = parameters["J"]
    for j in range(n_scales):
        np.testing.assert_array_equal(
            maps.layer1[j], fixture[f"{mode}_layer1_{j}"] > 0, err_msg=f"layer1 j={j}"
        )
        for j2 in range(j + 1):
            np.testing.assert_array_equal(
                maps.layer2[(j, j2)],
                fixture[f"{mode}_layer2_{j}_{j2}"] > 0,
                err_msg=f"layer2 j3={j} j2={j2}",
            )


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("case", sorted(CASES))
def test_reweighting_maps_equal_stl(case: str, mode: str) -> None:
    maps, fixture, parameters = _maps(case, MODES[mode])
    # Scale j is computed at level d = j (maths.md M-NOT): levels 0 … J-1.
    scale_levels = tuple(range(parameters["J"]))
    coarser = range(1, max(scale_levels) + 1)  # the levels d reached by a downsampling step

    def finer_scales(d: int) -> list[int]:
        """The scales j computed at a level before d, whose maps are downsampled to d."""
        return [j for j, level in enumerate(scale_levels) if level < d]

    assert sorted(maps.smooth_reweight) == list(coarser)
    assert sorted(maps.wav_reweight) == sorted((d, j) for d in coarser for j in finer_scales(d))
    for d in coarser:
        assert_parity(
            _with_nan(maps.smooth_invalid[d], maps.smooth_reweight[d]),
            fixture[f"{mode}_smooth_{d}"],
        )
        for j in finer_scales(d):
            assert_parity(
                _with_nan(maps.wav_invalid[(d, j)], maps.wav_reweight[(d, j)]),
                fixture[f"{mode}_wav_{d}_{j}"],
            )


@pytest.mark.parametrize("case", sorted(CASES))
def test_fixture_exercises_the_threshold(case: str) -> None:
    """The case is useful only if some downsampled pixels are reweighted and others dropped."""
    maps, _, _ = _maps(case, pbc=False)
    reweighted = any(
        bool(np.any((r != 1.0) & ~i))
        for r, i in zip(maps.smooth_reweight.values(), maps.smooth_invalid.values(), strict=True)
    )
    dropped = any(bool(np.any(i)) for i in maps.smooth_invalid.values())
    assert reweighted
    assert dropped
