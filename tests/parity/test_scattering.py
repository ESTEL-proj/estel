"""Parity of the vanilla scattering statistics with STL (task T2.4, issue #16; maths.md M-MOM,
M-W1, M-S1 to M-S4, M-CHAN, M-MEAN; OP-3, OP-26, OP-27, OP-30).

The fixtures ``scattering_2d`` (STL `master`) and ``scattering_1d`` (STL `2D+1D`) hold STL's
statistics, without normalisation and without spectrum, for each configuration of their case
(``tools/stl_parity/cases/``): periodic and open, with and without missing values, two channels
with every pair, a custom ``compute_cross_matrix``, standardised data, L = 2 in 1D, and a deep
open configuration where some means have no valid pixel. ESTEL's internal ``STOperator`` is
built with the same parameters on the stored input, and every family is compared with
``assert_parity``: same full-grid shape, NaN at the same positions (structural, channel pairs not
computed, empty means), values within the tolerances of Part II, §14.3.

Both libraries run in float64 for parity (§14.3), so the operator is built in float64 in every
run, whatever ``ESTEL_TEST_DTYPE``, as the kernels are compared in float64 in ``test_kernels``.
The float32 secondary check of §14.3 is not run on the statistics: its absolute tolerance,
0.01 x 1e-5 x max|STL|, is below the rounding of float32 sums over many pixels; issue #66
asks the maintainers how to check float32 instead.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass, _config
from estel.scattering._loop import ScatteringValues
from estel.scattering._operator import STOperator
from tests.parity._compare import assert_parity, load_fixture, read_manifest

CASES: dict[str, type[DataClass]] = {"scattering_2d": Data2D, "scattering_1d": Data1D}
FAMILIES = ("S1", "S2", "S3", "S4")
pytestmark = pytest.mark.maths("M-MOM", "M-W1", "M-S1", "M-S2", "M-S3", "M-S4", "M-CHAN", "M-MEAN")


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data and operators in float64 for this test only (§14.3)."""
    monkeypatch.setattr(_config, "_dtype", "float64")


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _configs() -> list[tuple[str, str]]:
    return [(case, name) for case in sorted(CASES) for name in _parameters(case)["configs"]]


def _statistics(
    case: str, name: str, fixture: dict[str, NDArray[Any]]
) -> tuple[ScatteringValues, dict[str, Any]]:
    parameters = _parameters(case)
    config = parameters["configs"][name]
    mask = fixture[f"{name}_mask"] > 0 if f"{name}_mask" in fixture else None
    data = CASES[case](fixture[f"{name}_field"], mask=mask, pbc=config["pbc"])
    matrix = config.get("cross_matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3 and OP-13, tested in tests/scattering
        operator = STOperator(
            data,
            config["J"],
            config.get("L", parameters["L"]),
            norm="vanilla",
            standardize=config.get("standardize", False),
            compute_cross_matrix=None if matrix is None else np.array(matrix, dtype=bool),
            compute_PS=False,
        )
    return operator.apply(data), config


@pytest.mark.parametrize(("case", "name"), _configs())
def test_statistics_equal_stl(case: str, name: str) -> None:
    fixture = load_fixture(case)
    values, config = _statistics(case, name, fixture)
    # mean and var, the moments of M-MOM, are compared as one array: the absolute tolerance of
    # §14.3 is relative to the largest value of the compared array, and the mean of standardised
    # data is zero in theory, a whole array of rounding errors (1e-16) on its own.
    moments = np.stack([np.asarray(values.mean), np.asarray(values.var)])
    assert moments.dtype == fixture[f"{name}_mean"].dtype == fixture[f"{name}_var"].dtype
    assert_parity(moments, np.stack([fixture[f"{name}_mean"], fixture[f"{name}_var"]]))
    for family in FAMILIES:
        actual = np.asarray(getattr(values, family))
        assert actual.dtype == fixture[f"{name}_{family}"].dtype, family
        assert_parity(actual, fixture[f"{name}_{family}"])
    if config.get("standardize", False):
        assert_parity(np.asarray(values.mean_pre_std), fixture[f"{name}_mean_pre_std"])
        assert_parity(np.asarray(values.std_pre_std), fixture[f"{name}_std_pre_std"])
    else:
        assert values.mean_pre_std is None
        assert values.std_pre_std is None


def test_fixtures_cover_the_cells_of_t24() -> None:
    """The configurations are useful only if they exercise what they are named for: missing
    values, empty means (NaN beyond the structural ones), the custom channel matrix."""
    for case in CASES:
        configs = _parameters(case)["configs"]
        fixture = load_fixture(case)
        assert any("mask" in c and c["pbc"] for c in configs.values()), case
        assert any("mask" in c and not c["pbc"] for c in configs.values()), case
        assert any(c.get("standardize") for c in configs.values()), case
        matrix = np.array(configs["cross_matrix"]["cross_matrix"])
        assert not matrix[1, 1], case  # channel 1: cross only (OP-30)
        assert matrix[1, 2], case
        # The deep open configuration has NaN S3 within j2 <= j3, where STL divides 0 by 0.
        s3 = fixture["deep_open_S3"][0, 0, 0, :, :, 0, 0]
        upper = np.triu(np.ones(s3.shape, dtype=bool))
        assert np.isnan(s3[upper]).any(), case
