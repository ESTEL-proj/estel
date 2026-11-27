"""tools/prove_unchanged.py sees a difference (task T7.2, #193): the positive control.

On crafted outputs and dumps, without running the cases: a one-bit change in an output, a
changed ``_key``, a changed function body are each detected, and a rename recorded in the table
is not a difference. The full run (minutes) is not a test of the suite.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prove_unchanged import (
    BEYOND,
    EQUAL,
    WITHIN,
    Job,
    Names,
    Row,
    canonical_dump,
    compare_bodies,
    compare_outputs,
    compare_runs,
    function_bodies,
    report,
)

from .conftest import write

NO_NAMES = Names([])


def _done(old: str, new: str) -> Row:
    return Row(old=old, new=new, kind="rename", task="T7.7", done=True)


def _class(module: str, name: str) -> type:
    return type(name, (), {"__module__": module, "__qualname__": name})


@dataclasses.dataclass(frozen=True)
class _Plan:
    geometry: Any
    J: int
    scale: float
    table: Any


def _plan(**change: Any) -> _Plan:
    values = {
        "geometry": _class("estel.scattering._plan", "PlanarPlan"),
        "J": 2,
        "scale": 0.5,
        "table": np.arange(6.0),
    }
    return _Plan(**{**values, **change})


def _outputs(values: Any) -> dict[str, Any]:
    return {"2d/S1": values, "2d/plan._key": ("dump", canonical_dump(_plan(), NO_NAMES))}


def test_equal_outputs_are_equal() -> None:
    values = np.random.default_rng(0).standard_normal(8)
    assert (
        compare_outputs("2d/torch/float64", _outputs(values), _outputs(values.copy()), "float64")
        == []
    )


def test_one_bit_is_seen_and_found_within_the_tolerance() -> None:
    values = np.random.default_rng(0).standard_normal(8)
    changed = values.copy()
    changed.view(np.int64)[3] ^= 1  # the last bit of one value
    [difference] = compare_outputs("c", _outputs(values), _outputs(changed), "float64")
    assert difference.status == WITHIN
    assert difference.where == "c/2d/S1"
    assert "largest absolute difference" in difference.detail


@pytest.mark.parametrize(
    ("change", "status"),
    [
        pytest.param(lambda v: v * (1 + 1e-6), BEYOND, id="beyond-rtol-float64"),
        pytest.param(lambda v: np.where(np.arange(v.size) == 2, np.nan, v), BEYOND, id="nan"),
        pytest.param(lambda v: v.astype(np.float32), BEYOND, id="dtype"),
        pytest.param(lambda v: v[:-1], BEYOND, id="shape"),
    ],
)
def test_a_larger_difference_is_beyond(change: Any, status: int) -> None:
    values = np.random.default_rng(0).standard_normal(8)
    [difference] = compare_outputs("c", _outputs(values), _outputs(change(values)), "float64")
    assert difference.status == status


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"J": 3}, id="an-integer"),
        pytest.param({"scale": np.nextafter(0.5, 1.0)}, id="one-bit-of-a-float"),
        pytest.param({"table": np.arange(6.0)[::-1].copy()}, id="an-array"),
        pytest.param(
            {"geometry": _class("estel.scattering._plan", "TensorProdPlan")}, id="a-class"
        ),
    ],
)
def test_a_changed_key_is_seen(change: dict[str, Any]) -> None:
    base = {"k": ("dump", canonical_dump(_plan(), NO_NAMES))}
    branch = {"k": ("dump", canonical_dump(_plan(**change), NO_NAMES))}
    [difference] = compare_outputs("c", base, branch, "float64")
    assert difference.status == BEYOND
    assert "the dumps differ" in difference.detail


def test_a_class_renamed_through_the_table_is_the_same() -> None:
    names = Names([_done("estel.scattering._plan:PlanarPlan", "STPlanSingleDomain")])
    renamed = _plan(geometry=_class("estel.scattering._plan", "STPlanSingleDomain"))
    assert canonical_dump(renamed, names) == canonical_dump(_plan(), NO_NAMES)
    assert canonical_dump(renamed, NO_NAMES) != canonical_dump(_plan(), NO_NAMES)


def test_a_module_renamed_through_the_table_is_the_same() -> None:
    names = Names([_done("estel.data._planar", "estel.data._cartesian")])
    assert names.canonical("estel.data._cartesian", "Data2D") == "estel.data._planar:Data2D"
    split = Names(
        [
            Row(
                "estel._spec.planar",
                "estel._spec.common, estel._spec.cartesian",
                "split",
                "T7.3",
                True,
            )
        ]
    )
    assert split.canonical("estel._spec.common", "spec_apply") == "estel._spec.planar:spec_apply"
    assert split.new_modules("estel._spec.planar") == [
        "estel._spec.common",
        "estel._spec.cartesian",
    ]


def test_an_output_or_a_run_on_one_side_only_is_beyond() -> None:
    values = np.zeros(2)
    [difference] = compare_outputs("c", {"a": values, "b": values}, {"a": values}, "float64")
    assert difference.status == BEYOND
    job = Job("2d-periodic", "torch", "float64")
    base = {job: {"outputs": {"a": values}}}
    [crash] = compare_runs(base, {job: {"crash": "Traceback\nImportError: no module"}})
    assert crash.status == BEYOND


SOURCE = '''
class PlanarPlan:
    def size(self):
        """The size."""
        return self.J * 2


def build(plan):
    return PlanarPlan()
'''


def _bodies(root: Path, source: str, rows: list[Row]) -> dict[str, str]:
    write(root, {"src/estel/scattering/_plan.py": source})
    return function_bodies(root / "src", Names(rows))


def test_a_changed_body_is_listed_and_a_renamed_one_is_not(tmp_path: Path) -> None:
    base = _bodies(tmp_path / "main", SOURCE, [])
    renamed = SOURCE.replace("PlanarPlan", "STPlanSingleDomain").replace("The size.", "Its size.")
    rows = [_done("estel.scattering._plan:PlanarPlan", "STPlanSingleDomain")]
    assert compare_bodies(base, _bodies(tmp_path / "renamed", renamed, rows)) == {
        "differ": [],
        "main only": [],
        "branch only": [],
    }
    changed = SOURCE.replace("self.J * 2", "2 * self.J")
    listed = compare_bodies(base, _bodies(tmp_path / "changed", changed, []))
    assert listed["differ"] == ["estel.scattering._plan:PlanarPlan.size"]
    unrecorded = compare_bodies(base, _bodies(tmp_path / "unrecorded", renamed, []))
    assert "estel.scattering._plan:PlanarPlan.size" in unrecorded["main only"]


def test_the_verdict_is_the_worst_difference(capsys: pytest.CaptureFixture[str]) -> None:
    no_bodies: dict[str, list[str]] = {"differ": [], "main only": [], "branch only": []}
    assert report({}, [], no_bodies, 1.0) == EQUAL
    values = np.random.default_rng(0).standard_normal(4)
    near = values.copy()
    near.view(np.int64)[0] ^= 1
    within = compare_outputs("c", {"x": values}, {"x": near}, "float64")
    assert report({}, within, no_bodies, 1.0) == WITHIN
    beyond = compare_outputs("c", {"x": values}, {"x": values + 1}, "float64")
    assert report({}, within + beyond, no_bodies, 1.0) == BEYOND
    assert "Verdict 2" in capsys.readouterr().out
