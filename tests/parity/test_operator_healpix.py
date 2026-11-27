"""Parity of ``DataHealpix`` and its ``WaveletOperatorSingleDomain`` with STL (task T5.5, issue
#166; maths.md M-CONV-HEALPIX, M-DOWN-HEALPIX, M-MASK-HEALPIX, OP-110).

Through the operator the scattering loop will use (``apply``, ``downsample``, the masks of the
plan), against the fixtures of T5.3: the responses of ``operators_healpix`` to 3 random maps, and
the static masks and reweighting maps of ``statistics_healpix``.
``tests/parity/test_tables_healpix.py`` checks the tables of T5.1 alone, applied in NumPy; this
module checks their application by the hooks, on the backend and in the run's dtype.

The full skies are compared on every row; the discs on every (gauge, pixel) row that is not
ill-conditioned (maths.md OP-110; decision 3 on #165): STL's values there are set by rounding.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import DataHealpix, _config
from estel.backend import _registry
from estel.backend._protocol import DType
from estel.wavelets._operator import WaveletOperatorSingleDomain, build_wavelet_plan
from estel.wavelets._operator_healpix import WaveletPlanHealpix
from tests.parity._compare import assert_parity, load_fixture
from tests.parity.test_tables_healpix import EXCLUDED_ROWS, _well_conditioned_rows

DOMAINS = [
    pytest.param("full_8", marks=pytest.mark.dtype),
    pytest.param("disc_8", marks=pytest.mark.dtype),
    "full_16",
    "disc_16",
]
"""The domains of ``operators_healpix``; the representative cases of ``nside`` 8 also run in
float32 in the fast tier (``dtype``), those of ``nside`` 16 in float64 there."""


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    monkeypatch.setattr(_config, "_dtype", dtype)


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


@functools.cache
def _operators() -> dict[str, NDArray[Any]]:
    return load_fixture("operators_healpix")


@functools.cache
def _statistics() -> dict[str, NDArray[Any]]:
    return load_fixture("statistics_healpix")


def _operator(name: str, J: int) -> WaveletOperatorSingleDomain:  # noqa: N803
    """The operator of a domain of ``operators_healpix``, on its 3 maps as 3 batch elements."""
    fixture = _operators()
    ids = fixture[f"{name}_cell_ids"].astype(np.int64)
    maps = fixture[f"{name}_maps"][:, None]  # (3, 1, Npix)
    if name.startswith("full"):
        return WaveletOperatorSingleDomain(DataHealpix(maps), J)
    depth = int(name.split("_")[1]).bit_length() - 1
    return WaveletOperatorSingleDomain(DataHealpix(maps, depth=depth, cell_ids=ids), J)


@pytest.mark.maths("M-CONV-HEALPIX", "M-WAV-HEALPIX")
@pytest.mark.parametrize("name", DOMAINS)
def test_convolution_equals_stl(name: str, dtype: DType) -> None:
    operator = _operator(name, 1)
    actual = _numpy(operator.apply(operator._data.array, 0))[:, :, 0]  # (3, L, Npix)
    expected = _operators()[f"{name}_conv"]
    if name.startswith("full"):
        assert_parity(actual, expected, dtype)
        return
    rows, _ = _well_conditioned_rows(name)
    assert rows.size - rows.sum() == EXCLUDED_ROWS[name]
    scale = float(np.abs(expected).max())
    assert_parity(actual[:, rows], expected[:, rows], dtype, scale=scale)


@pytest.mark.maths("M-DOWN-HEALPIX")
@pytest.mark.parametrize("name", DOMAINS)
def test_decimation_equals_stl(name: str, dtype: DType) -> None:
    operator = _operator(name, 2)
    actual = _numpy(operator.downsample(operator._data.array, 0))[:, 0]
    assert_parity(actual, _operators()[f"{name}_down"], dtype)
    plan = operator.plan
    assert isinstance(plan, WaveletPlanHealpix)
    coarse = plan.level_cell_ids(1)
    if coarse is not None:
        np.testing.assert_array_equal(coarse, _operators()[f"{name}_down_cell_ids"])


@pytest.mark.maths("M-MASK-HEALPIX", "M-MASK", "M-DOWN-MASK")
@pytest.mark.parametrize("dtype", ["float64"])  # the masks are float64 in every dtype
@pytest.mark.parametrize("name", ["nan_16", "disc_nan_16"])
def test_masks_equal_stl(name: str, dtype: DType) -> None:
    """Every static mask and reweighting map of the two masked configurations, on the pixels of
    their level: the masks exactly, the maps to the float64 parity tolerance (they are built in
    float64 with the plan, whatever the dtype)."""
    fixture = _statistics()
    ids = fixture[f"{name}_cell_ids"].astype(np.int64)
    full = ids.size == 12 * 16 * 16
    field = fixture[f"{name}_field"]
    data = DataHealpix(
        field,
        fixture[f"{name}_mask"].astype(bool),
        depth=None if full else 4,
        cell_ids=None if full else ids,
    )
    plan = build_wavelet_plan(data, 3)
    assert isinstance(plan, WaveletPlanHealpix)
    masks = plan.masks
    assert masks is not None

    def same(mask: Any, key: str) -> None:
        np.testing.assert_array_equal(np.asarray(mask, dtype=np.float64), fixture[f"{name}_{key}"])

    same(masks.full_res, "smooth_0")
    for d in (1, 2):
        if not full:
            expected_ids = fixture[f"{name}_cell_ids_{d}"]
            np.testing.assert_array_equal(plan.level_cell_ids(d), expected_ids)
        same(masks.smooth_invalid[d], f"smooth_{d}")
        assert_parity(masks.smooth_reweight[d], fixture[f"{name}_reweight_smooth_{d}"])
    for j, layer in masks.layer1.items():
        same(layer, f"layer1_{j}")
    assert len(masks.wav_invalid) == 3
    for (d, j), layer in masks.wav_invalid.items():
        same(layer, f"wav_{d}_{j}")
        assert_parity(masks.wav_reweight[d, j], fixture[f"{name}_reweight_wav_{d}_{j}"])
    assert len(masks.layer2) == 6
    for (j3, j2), layer in masks.layer2.items():
        same(layer, f"layer2_{j3}_{j2}")
