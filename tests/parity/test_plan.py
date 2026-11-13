"""Parity of the plan's means with STL: borders, crops, masks and denominators (task T2.3,
issue #15; maths.md M-BORDER, M-MASK, M-MEAN).

Compares ``estel.scattering._plan.build_plan`` with the fixtures ``plan_2d`` (STL `master`) and
``plan_1d`` (STL `2D+1D`), which record, for every mean of STL's loop, the border STL's border
method returns, the number of pixels its ``maskmean`` divides by, and (for the configurations
marked ``"weights"``) the pixels it averages, in the order of STL's loops (the cases'
docstrings, ``tools/stl_parity/cases/plan_2d.py``). The comparison is exact: borders and counts
are integers, weights 0 or 1.

Where STL divides by zero (a count of 0), ESTEL raises a clear error instead (maths.md OP-3): the
test checks that the plan raises exactly for the configurations where some STL count is 0,
naming the first such mean in the plan's order (moments, S1 and S2, S3, S4).
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass
from estel.scattering._plan import MeanWeights, ScatteringPlan, border_width, build_plan
from tests.parity._compare import load_fixture, read_manifest

CASES: dict[str, type[DataClass]] = {"plan_2d": Data2D, "plan_1d": Data1D}
pytestmark = pytest.mark.maths("M-BORDER", "M-MASK", "M-MEAN")


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _configs() -> list[tuple[str, str]]:
    return [(case, name) for case in sorted(CASES) for name in _parameters(case)["configs"]]


def _stl_order(J: int) -> list[tuple[str, tuple[tuple[int, ...], ...], int, Any]]:  # noqa: N803
    """The means of the fixtures, in STL's loop order: (label, histories, level, plan key)."""
    order: list[tuple[str, tuple[tuple[int, ...], ...], int, Any]] = [
        ("mean and var", ((),), 0, "moments")
    ]
    for j3 in range(J):
        order.append((f"S1 and S2 (j={j3})", ((j3,),), j3, ("S1", j3)))
        for j2 in range(j3 + 1):
            order.append((f"S3 (j2={j2}, j3={j3})", ((j2, j3), (j3,)), j3, ("S3", j2, j3)))
            for j1 in range(j2 + 1):
                key = ("S4", j1, j2, j3)
                order.append((f"S4 (j1={j1}, j2={j2}, j3={j3})", ((j1, j3), (j2, j3)), j3, key))
    return order


def _family(key: Any) -> str:
    return "moments" if key == "moments" else str(key[0])


def _lookup(plan: ScatteringPlan, key: Any) -> MeanWeights:
    if key == "moments":
        return plan.moments
    family, *index = key
    if family == "S1":
        return plan.first_layer[index[0]]
    if family == "S3":
        return plan.s3[(index[0], index[1])]
    return plan.s4[(index[0], index[1], index[2])]


def _data(case: str, name: str, fixture: dict[str, NDArray[Any]]) -> tuple[DataClass, int]:
    config = _parameters(case)["configs"][name]
    cls = CASES[case]
    shape = (config["N"],) * cls.ndim
    mask = None if config["mask"] is None else fixture[f"mask_{config['mask']}"] > 0
    return cls(np.zeros(shape), mask=mask, pbc=config["pbc"]), config["J"]


def _build(data: DataClass, J: int, L: int) -> ScatteringPlan:  # noqa: N803
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13, checked in tests/scattering
        return build_plan(data, J, L)


@pytest.mark.parametrize(("case", "name"), _configs())
def test_borders_equal_stl(case: str, name: str) -> None:
    """M-BORDER: the border of each mean, before the reduction of OP-13, is STL's."""
    fixture = load_fixture(case)
    data, n_scales = _data(case, name, fixture)
    kernel_size = _parameters(case)["K"]
    expected = fixture[f"{name}_border"]
    got = [
        max(border_width(h, level, kernel_size, data.pbc) for h in histories)
        for _, histories, level, _ in _stl_order(n_scales)
    ]
    np.testing.assert_array_equal(np.asarray(got, dtype=np.float64), expected)


@pytest.mark.parametrize(("case", "name"), _configs())
def test_denominators_and_weights_equal_stl(case: str, name: str) -> None:
    """M-MEAN: every denominator, and the averaged pixels where stored, are STL's; the plan
    raises where STL divides by zero (OP-3)."""
    fixture = load_fixture(case)
    parameters = _parameters(case)
    data, n_scales = _data(case, name, fixture)
    order = _stl_order(n_scales)
    counts = fixture[f"{name}_count"]
    assert len(counts) == len(order)
    empty = {key for (_, _, _, key), count in zip(order, counts, strict=True) if count == 0}
    if empty:
        # The plan builds the means family by family: moments, S1 and S2, S3, then S4.
        families = ("moments", "S1", "S3", "S4")
        plan_order = sorted(order, key=lambda mean: families.index(_family(mean[3])))
        label = next(label for label, _, _, key in plan_order if key in empty)
        with pytest.raises(ValueError, match=rf"mean of {re.escape(label)}, .*OP-3"):
            _build(data, n_scales, parameters["L"])
        return
    plan = _build(data, n_scales, parameters["L"])
    got = [_lookup(plan, key) for _, _, _, key in order]
    np.testing.assert_array_equal(np.asarray([w.denominator for w in got]), counts)
    if not parameters["configs"][name]["weights"]:
        return
    for level in range(n_scales):
        expected = fixture[f"{name}_weights_{level}"]
        at_level = [w.weights for w in got if w.level == level]
        np.testing.assert_array_equal(np.stack(at_level), expected, err_msg=f"level {level}")


def test_fixtures_exercise_the_cases_they_are_for() -> None:
    """The fixtures are useful only if they reduce some borders (OP-13), mask some pixels, and
    hold configurations where STL divides by zero (OP-3) and where it does not."""
    for case in CASES:
        fixture = load_fixture(case)
        configs = _parameters(case)["configs"]
        # Without mask, a reduced border shows as a count above the crop of the border itself.
        open_counts, open_borders = fixture["open_count"], fixture["open_border"]
        order = _stl_order(configs["open"]["J"])
        n, ndim = configs["open"]["N"], CASES[case].ndim
        full = [
            float((n // 2**level - 2 * border) ** ndim) if n // 2**level > 2 * border else -1.0
            for (_, _, level, _), border in zip(order, open_borders, strict=True)
        ]
        assert any(f != c for f, c in zip(full, open_counts, strict=True)), case
        assert any(np.any(fixture[f"{name}_count"] == 0) for name in configs), case
        stored = [name for name in configs if configs[name]["weights"]]
        assert stored, case
        assert all(np.all(fixture[f"{name}_count"] > 0) for name in stored), case
