"""Spec parity of the wavelet operator of HEALPix data: ``DataHealpix`` and its
``WaveletOperatorSingleDomain`` against the oracle ``estel._spec.healpix`` (task T5.5, issue #166; maths.md
M-CONV-HPX, M-WAV-HPX, M-DOWN-HPX, M-MASK-HPX, M-DEF-HPX; Part I, §4.2).

The oracle has its own geometry, written from maths.md §4 (T5.4); ESTEL applies the tables
ported from ``TorchGeometry`` (T5.1). Both apply the partial-sky rule of M-CONV-HPX with the
threshold of OP-110, so the discs are compared on every row, unlike STL's fixtures. The maps
and the pixels are those of STL's fixtures (T5.3), so the domains are the ones of
``tests/parity/test_healpix_operator.py``. Compared with ``assert_parity`` in the run's dtype.

A disagreement is a finding about ``docs/maths.md``, the oracle or the port: report the
coefficients on an issue, change nothing to make a test pass.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import DataHealpix, _config
from estel._spec import healpix as spec
from estel.backend import _registry
from estel.backend._protocol import DType
from estel.wavelets._operator import WaveletOperatorSingleDomain, build_wavelet_plan
from tests.parity._compare import assert_parity, load_fixture

DOMAINS = [
    "full_8",
    "disc_8",
    pytest.param("full_16", marks=pytest.mark.slow),
    pytest.param("disc_16", marks=pytest.mark.slow),
]
"""The domains of ``operators_hpx``. Slow: ``nside`` 16, the oracle's naive tables and
decimation take about 0.3 s, over the budget with ESTEL's (decision 4 on #165)."""


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    monkeypatch.setattr(_config, "_dtype", dtype)


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


@functools.cache
def _operators() -> dict[str, NDArray[Any]]:
    return load_fixture("operators_hpx")


@functools.cache
def _statistics() -> dict[str, NDArray[Any]]:
    return load_fixture("statistics_hpx")


def _domain(name: str) -> tuple[int, NDArray[np.int64] | None, NDArray[np.float64]]:
    """``nside``, ``cell_ids`` (``None`` on the full sky) and the 3 maps of a domain."""
    nside = int(name.split("_")[1])
    ids = _operators()[f"{name}_cell_ids"].astype(np.int64)
    return nside, (None if name.startswith("full") else ids), _operators()[f"{name}_maps"]


def _data(nside: int, cell_ids: Any, values: Any, mask: Any = None) -> DataHealpix:
    depth = nside.bit_length() - 1
    return DataHealpix(values, mask, depth=None if cell_ids is None else depth, cell_ids=cell_ids)


@pytest.mark.dtype
@pytest.mark.maths("M-CONV-HPX", "M-WAV-HPX")
@pytest.mark.parametrize("name", DOMAINS)
def test_convolution_equals_the_oracle(name: str, dtype: DType) -> None:
    """Every (gauge, pixel) row, the edge of the discs included: ESTEL and the oracle apply the
    same partial-sky rule (OP-50, OP-110).

    Slow: at ``nside`` 16 only, the oracle's naive tables (decision 4 on #165)."""
    nside, ids, maps = _domain(name)
    operator = WaveletOperatorSingleDomain(_data(nside, ids, maps), 1)
    # The 3 maps are channels: (1, L, 3, Npix) -> (3, L, Npix).
    actual = np.swapaxes(_numpy(operator.apply(operator._data.array, 0))[0], 0, 1)
    tables = spec.spec_conv_tables(nside, ids)
    expected = spec.spec_conv_healpix(maps, tables, spec.spec_wavelet_healpix())
    assert_parity(actual, expected, dtype)


@pytest.mark.dtype
@pytest.mark.maths("M-DOWN-HPX")
@pytest.mark.parametrize("name", DOMAINS)
def test_decimation_equals_the_oracle(name: str, dtype: DType) -> None:
    """Slow: at ``nside`` 16 only, the oracle's naive decimation (decision 4 on #165)."""
    nside, ids, maps = _domain(name)
    operator = WaveletOperatorSingleDomain(_data(nside, ids, maps), 2)
    matrix, _ = spec.spec_decimation_healpix(nside, ids)
    actual = _numpy(operator.downsample(operator._data.array, 0))[0]
    assert_parity(actual, maps @ matrix.T, dtype)


def _check_masks(plan: Any, levels: Any, invalid: NDArray[np.bool_]) -> None:
    """The masks and reweighting maps of the plan equal the oracle's (M-MASK-HPX): every
    family, on the pixels of its level; the masks exactly, the maps to the float64 parity
    tolerance (they are static, in float64, whatever the run's dtype)."""
    masks, expected = plan.masks, spec.spec_masks_healpix(levels, invalid)
    assert masks is not None
    np.testing.assert_array_equal(masks.full_res, expected["field"][0])
    for d in range(1, len(levels)):
        np.testing.assert_array_equal(masks.smooth_invalid[d], expected["field"][d])
        assert_parity(masks.smooth_reweight[d], expected["reweight_field"][d])
    for j, layer in masks.layer1.items():
        np.testing.assert_array_equal(layer, expected["layer1"][j])
    assert set(masks.wav_invalid) == set(expected["down1"])
    for key, layer in masks.wav_invalid.items():
        np.testing.assert_array_equal(layer, expected["down1"][key])
        assert_parity(masks.wav_reweight[key], expected["reweight_down1"][key])
    assert {(j2, j3) for j3, j2 in masks.layer2} == set(expected["layer2"])
    for (j3, j2), layer in masks.layer2.items():
        np.testing.assert_array_equal(layer, expected["layer2"][j2, j3])


@pytest.mark.maths("M-MASK-HPX", "M-MASK", "M-DOWN-MASK")
@pytest.mark.parametrize("dtype", ["float64"])  # the masks are float64 in every dtype
@pytest.mark.parametrize("sky", ["full", "disc"])
def test_masks_equal_the_oracle_at_nside_8(sky: str, dtype: DType) -> None:
    """A hole of 16 pixels and a few scattered ones, J = 3, on the full sky and on the disc of
    ``disc_8``: enough to make every family non-trivial."""
    nside, ids, maps = _domain(f"{sky}_8")
    invalid = np.zeros(maps.shape[-1], dtype=bool)
    invalid[: maps.shape[-1] // 3 : 3] = True
    plan = build_wavelet_plan(_data(nside, ids, maps[0], invalid), 3)
    _check_masks(plan, spec.spec_levels_healpix(nside, ids, 3), invalid)


@pytest.mark.slow
@pytest.mark.maths("M-MASK-HPX", "M-MASK", "M-DOWN-MASK")
@pytest.mark.parametrize("dtype", ["float64"])  # the masks are float64 in every dtype
@pytest.mark.parametrize("name", ["nan_16", "disc_nan_16"])
def test_masks_equal_the_oracle_on_the_masked_configurations(name: str, dtype: DType) -> None:
    """The two masked configurations of ``statistics_hpx``, ``nside`` 16, J = 3.

    Slow: ``nside`` 16, the oracle's naive tables (decision 4 on #165)."""
    fixture = _statistics()
    ids = fixture[f"{name}_cell_ids"].astype(np.int64)
    cell_ids = None if ids.size == 3072 else ids
    invalid = fixture[f"{name}_mask"].astype(bool)
    field = fixture[f"{name}_field"]
    plan = build_wavelet_plan(_data(16, cell_ids, field, invalid), 3)
    _check_masks(plan, spec.spec_levels_healpix(16, cell_ids, 3), invalid)


@pytest.mark.maths("M-DEF-HPX", "M-MASK-HPX")
@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16, 32])
def test_defaults_and_no_border(nside: int) -> None:
    """J, L and K of M-DEF-HPX; no border (β = 0) on a full sky and on a partial sky, so the
    weights of every mean are its valid pixels (M-MASK-HPX, OP-50)."""
    defaults = spec.spec_defaults_healpix(nside)
    data = DataHealpix(np.zeros(12 * nside * nside))
    plan = build_wavelet_plan(data)
    assert (plan.J, plan.L, plan.kernel_size) == (defaults["J"], defaults["L"], defaults["K"])
    assert plan.nan_threshold == defaults["nan_threshold"]
    partial = DataHealpix(np.zeros(5), depth=nside.bit_length() - 1, cell_ids=np.arange(5))
    operator = WaveletOperatorSingleDomain.from_parts(partial, build_wavelet_plan(partial, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weights = operator.mean_weights({"s1": (((0,),), 0)})["s1"]
    assert weights is not None
    assert weights.border == defaults["border"] == 0
    np.testing.assert_array_equal(weights.weights, np.ones(5))
    assert weights.kind == "full"
