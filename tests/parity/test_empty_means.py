"""Parity of the NaN coefficients with STL where a mean has no valid pixel (task T2.3, issue #15;
maths.md M-MEAN, OP-3, M-CHAN).

The fixtures ``empty_means_2d`` (STL `master`) and ``empty_means_1d`` (STL `2D+1D`) record where
STL's statistics are NaN, for one map with missing values. With one channel, a coefficient of
STL is NaN exactly when it is structural (outside ``j2 ≤ j3`` for S3, ``j1 ≤ j2 ≤ j3`` for S4) or
when its mean has no valid pixel, where STL divides 0 by 0. ESTEL decides the second kind
statically: the plan lists those means (``empty_scales``, ``empty_s3_pairs``,
``empty_s4_triples``), and the computation places NaN there (task T2.4). The test checks that the
plan's empty means and STL's index sets give STL's NaN positions exactly, for every orientation.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass
from estel.scattering._plan import ScatteringPlan, build_plan
from tests.parity._compare import load_fixture, read_manifest

CASES: dict[str, type[DataClass]] = {"empty_means_2d": Data2D, "empty_means_1d": Data1D}
pytestmark = pytest.mark.maths("M-MEAN", "M-CHAN")


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _configs() -> list[tuple[str, str]]:
    return [(case, name) for case in sorted(CASES) for name in _parameters(case)["configs"]]


def _plan(case: str, name: str, fixture: dict[str, NDArray[Any]]) -> ScatteringPlan:
    parameters = _parameters(case)
    config = parameters["configs"][name]
    cls = CASES[case]
    mask = fixture[f"mask_{config['mask']}"] > 0
    data = cls(np.zeros(mask.shape), mask=mask, pbc=config["pbc"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3 and OP-13, tested in tests/scattering
        return build_plan(data, config["J"], parameters["L"])


def _per_index(nan: NDArray[np.float64], n_scale_axes: int) -> dict[tuple[int, ...], bool]:
    """For each scale index of a family, whether STL's coefficient is NaN; the test requires it
    to be NaN for every orientation or for none."""
    out = {}
    J = nan.shape[0]  # noqa: N806
    for index in itertools.product(range(J), repeat=n_scale_axes):
        values = nan[index] > 0
        assert values.all() or not values.any(), f"partly NaN at {index}"
        out[index] = bool(values.all())
    return out


@pytest.mark.parametrize(("case", "name"), _configs())
def test_nan_coefficients_equal_stl(case: str, name: str) -> None:
    fixture = load_fixture(case)
    plan = _plan(case, name, fixture)
    assert not np.any(fixture[f"{name}_mean_nan"])
    assert not np.any(fixture[f"{name}_var_nan"])
    for family in ("S1", "S2"):
        expected = {(j,): j in plan.empty_scales for j in plan.scales}
        assert _per_index(fixture[f"{name}_{family}_nan"], 1) == expected, family
    s3 = {
        index: index not in plan.s3_pairs or index in plan.empty_s3_pairs
        for index in itertools.product(plan.levels, repeat=2)
    }
    assert _per_index(fixture[f"{name}_S3_nan"], 2) == s3
    s4 = {
        index: index not in plan.s4_triples or index in plan.empty_s4_triples
        for index in itertools.product(plan.levels, repeat=3)
    }
    assert _per_index(fixture[f"{name}_S4_nan"], 3) == s4


def test_fixtures_hold_nan_coefficients_beyond_the_structural_ones() -> None:
    """The open configurations are useful only if STL returns NaN means there, and the periodic
    ones as a control only if it returns none."""
    for case in CASES:
        fixture = load_fixture(case)
        open_plan = _plan(case, "deep_open", fixture)
        periodic_plan = _plan(case, "deep_periodic", fixture)
        assert open_plan.empty_s3_pairs, case
        assert open_plan.empty_s4_triples, case
        assert periodic_plan.empty_s3_pairs == (), case
        assert periodic_plan.empty_s4_triples == (), case
