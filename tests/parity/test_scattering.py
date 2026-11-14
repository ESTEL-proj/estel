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

The operator runs in the run's dtype: float64 is the parity check of §14.3 (rtol 1e-10), float32
its secondary check (rtol 1e-5), both with atol = 0.1 x rtol x max|STL| (issue #66). ``mean``,
whose theoretical value may be zero (standardised data), takes the standard deviation of the data
as its scale instead: atol = 0.1 x rtol x sqrt(max(var)) (§14.3).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass, _config
from estel.backend._protocol import DType
from estel.scattering._loop import ScatteringValues
from estel.scattering._operator import STOperator
from tests.parity._compare import assert_parity, load_fixture, read_manifest

CASES: dict[str, type[DataClass]] = {"scattering_2d": Data2D, "scattering_1d": Data1D}
FAMILIES = ("var", "S1", "S2", "S3", "S4")
pytestmark = pytest.mark.maths("M-MOM", "M-W1", "M-S1", "M-S2", "M-S3", "M-S4", "M-CHAN", "M-MEAN")


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


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
def test_statistics_equal_stl(case: str, name: str, dtype: DType) -> None:
    fixture = load_fixture(case)
    values, config = _statistics(case, name, fixture)
    expected_kind = {"float32": ("float32", "complex64"), "float64": ("float64", "complex128")}
    real, complex_ = expected_kind[dtype]
    # mean has the units of the data and may be zero in theory (standardised data), so its scale
    # is their standard deviation, sqrt(max(var)) (§14.3).
    mean = np.asarray(values.mean)
    assert mean.dtype == np.dtype(real)
    scale = float(np.sqrt(np.max(fixture[f"{name}_var"])))
    assert_parity(mean, fixture[f"{name}_mean"], dtype, scale=scale)
    for family in FAMILIES:
        actual = np.asarray(getattr(values, family))
        assert actual.dtype == np.dtype(real if family == "var" else complex_), family
        assert_parity(actual, fixture[f"{name}_{family}"], dtype)
    if config.get("standardize", False):
        for key in ("mean_pre_std", "std_pre_std"):
            assert_parity(np.asarray(getattr(values, key)), fixture[f"{name}_{key}"], dtype)
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
