"""The L0 contract ``estel.backend._protocol`` documents every primitive of Part II, §12.3.

Task T1.1: every primitive has its contract (shapes, dtypes, values, gradients, edge cases),
and value contracts cite IDs that ``docs/maths.md`` defines. The numerical conformance suite
of each backend comes with its implementation (T1.2).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from check_maths_ids import ID_RE, defined_ids
from estel.backend import _protocol
from estel.backend._protocol import Backend

REPO = Path(__file__).resolve().parents[2]

# Part II, §12.3, in the order of its table; ``name`` identifies the backend.
PRIMITIVES = (
    "xp",
    "asarray",
    "to_numpy",
    "pad",
    "conv_axes",
    "gather_weighted",
    "segment_sum",
    "scatter_to_grid",
    "gather_from_grid",
    "complex_to_real_pairs",
    "real_pairs_to_complex",
    "stop_gradient",
    "random_normal",
    "make_key",
    "jit",
    "value_and_grad",
    "vjp",
    "jvp",
    "hvp",
    "lbfgs",
    "deterministic",
)
SECTIONS = ("Shapes:", "Dtypes:", "Values:", "Gradients:", "Edge cases:")

# Primitives whose values enter the statistics, and the maths entries they must cite.
VALUE_CONTRACTS = {
    "pad": {"M-CONV-2D", "M-DOWN"},
    "conv_axes": {"M-CONV-2D", "M-DOWN"},
    "segment_sum": {"M-PS-2D"},
    "scatter_to_grid": {"M-CHAN"},
    "xp": {"M-MEAN", "M-PS-2D", "M-RED-ANG"},
}


def _members() -> set[str]:
    return {name for name in vars(Backend) if not name.startswith("_")}


def _doc(name: str) -> str:
    member = inspect.getattr_static(Backend, name)
    if isinstance(member, property):
        member = member.fget
    return inspect.getdoc(member) or ""


def test_protocol_lists_exactly_the_inventory() -> None:
    members = _members()
    missing = set(PRIMITIVES) - members
    extra = members - set(PRIMITIVES) - {"name"}
    assert not missing, f"Add these primitives of Part II, §12.3 to Backend: {sorted(missing)}"
    assert not extra, (
        f"Backend has members outside Part II, §12.3: {sorted(extra)}. A new primitive starts "
        "in the inventory (maintainer decision); otherwise make it private."
    )


@pytest.mark.parametrize("name", PRIMITIVES)
def test_every_primitive_documents_its_contract(name: str) -> None:
    doc = _doc(name)
    missing = [section for section in SECTIONS if section not in doc]
    assert not missing, f"Backend.{name}: add the sections {missing} to its docstring."


def test_cited_maths_ids_are_defined() -> None:
    defined, _ = defined_ids((REPO / "docs/maths.md").read_text())
    cited = set(ID_RE.findall(Path(_protocol.__file__).read_text()))
    unknown = cited - set(defined)
    assert not unknown, f"_protocol.py cites IDs that docs/maths.md does not define: {unknown}"


@pytest.mark.parametrize(("name", "ids"), sorted(VALUE_CONTRACTS.items()))
def test_value_contracts_cite_maths(name: str, ids: set[str]) -> None:
    cited = set(ID_RE.findall(_doc(name)))
    assert ids <= cited, f"Backend.{name}: cite {sorted(ids - cited)} in its Values section."
