"""Check the mapping between docs/maths.md, the oracle estel._spec and the tests (Part I, §4.1).

* **Definitions.** An ID (``M-...``) is defined by a heading of ``docs/maths.md``
  (``#### M-S1``, ``### 1.4 Masks of the scattering layers — M-MASK``). Each ID is defined
  once, and every ID cited in the text of ``docs/maths.md`` is defined.
* **Oracle.** Every public top-level function of ``src/estel/_spec/`` lists in its docstring
  the IDs it implements, and each of them is defined.
* **Tests.** Tests cite IDs with ``@pytest.mark.maths("M-S1", ...)`` (or a module-level
  ``pytestmark``); each cited ID is defined.
* **Coverage** (the other direction): every defined ID has at least one ``_spec`` function and
  at least one test, except those of the sections without an oracle by decision, which need a
  test and no ``_spec`` function, and those listed below as not yet implemented.

**Without an oracle by decision.** ``SECTIONS_WITHOUT_ORACLE`` lists the sections of
``docs/maths.md`` that have no oracle, each with the decision that says so: section 5, the
syntheses (Part IV, §22, row of 21 November 2026, "No oracle for the syntheses"; task T6.1).
Each of their IDs needs at least one test; a ``_spec`` function that cites one is an error, and
so is an entry whose section defines no ID.

**Not yet implemented.** The oracle covers sections 1 (planar 2D) and 2 (1D) of
``docs/maths.md`` (task T2.6, issue #18). ``NOT_YET_IMPLEMENTED_SECTIONS`` and
``NOT_YET_IMPLEMENTED_IDS`` list what the coverage step accepts without an oracle function or a
test, each with the task that implements it; that task removes its own entries. An entry that
names nothing, or whose IDs all have an oracle function and a test, is an error: remove it.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATHS = Path("docs/maths.md")
SPEC = Path("src/estel/_spec")
TESTS = Path("tests")
# What the coverage step accepts as not yet implemented, with the task that implements it and
# removes the entry (decision of the maintainer on issue #18).
NOT_YET_IMPLEMENTED_SECTIONS = {
    "3": "T3.2",  # tensor products
    "4": "T5.1",  # HEALPix
}
# Sections without an oracle by decision, with the decision: their IDs need a test and no
# _spec function.
SECTIONS_WITHOUT_ORACLE = {
    "5": 'Part IV, §22, row of 21 November 2026, "No oracle for the syntheses"',  # syntheses
}
NOT_YET_IMPLEMENTED_IDS = {
    "M-PS-2D": "T4.1",  # the 2D power spectrum
}
SECTION_RE = re.compile(r"^##\s+(\d+)\.")
ID_RE = re.compile(r"(?<![\w-])M-[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?![\w-])")
HEADING_RE = re.compile(r"^#{1,6}\s")


def defined_ids(text: str) -> tuple[Counter[str], set[str]]:
    """IDs defined by headings (with their counts), and IDs cited outside code fences."""
    defined: Counter[str] = Counter()
    cited: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue  # formulas and code blocks neither define nor cite IDs
        if HEADING_RE.match(line):
            defined.update(ID_RE.findall(line))
        cited.update(ID_RE.findall(line))
    return defined, cited


def id_sections(text: str) -> dict[str, str]:
    """``{id: section}``: the number of the top-level section (``## 3. Tensor products``) whose
    headings define each ID; ``""`` before the first numbered section."""
    sections: dict[str, str] = {}
    section = ""
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not HEADING_RE.match(line):
            continue
        numbered = SECTION_RE.match(line)
        if numbered:
            section = numbered.group(1)
        for mid in ID_RE.findall(line):
            sections.setdefault(mid, section)
    return sections


def not_yet_implemented(
    sections: dict[str, str], covered: set[str]
) -> tuple[dict[str, str], list[str]]:
    """``{id: task}`` for the IDs the lists accept without an oracle function or a test, and the
    errors of the lists themselves: an entry that names nothing, or that is no longer needed."""
    pending: dict[str, str] = {}
    errors = []
    for section, task in NOT_YET_IMPLEMENTED_SECTIONS.items():
        members = sorted(mid for mid, found in sections.items() if found == section)
        if not members:
            errors.append(
                f"NOT_YET_IMPLEMENTED_SECTIONS lists section {section} ({task}), but no ID of "
                f"{MATHS} is defined there. Remove the entry from tools/check_maths_ids.py."
            )
        elif all(mid in covered for mid in members):
            errors.append(
                f"NOT_YET_IMPLEMENTED_SECTIONS lists section {section} ({task}), but every ID of "
                "that section has an oracle function and a test. Remove the entry from "
                "tools/check_maths_ids.py."
            )
        pending.update(dict.fromkeys(members, task))
    for mid, task in NOT_YET_IMPLEMENTED_IDS.items():
        if mid not in sections:
            errors.append(
                f"NOT_YET_IMPLEMENTED_IDS lists {mid} ({task}), which {MATHS} does not define. "
                "Remove the entry from tools/check_maths_ids.py."
            )
        elif mid in covered:
            errors.append(
                f"NOT_YET_IMPLEMENTED_IDS lists {mid} ({task}), which has an oracle function and "
                "a test. Remove the entry from tools/check_maths_ids.py."
            )
        pending[mid] = task
    return pending, errors


def without_oracle(sections: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """``{id: decision}`` for the IDs of the sections without an oracle by decision, and the
    errors of the list itself: an entry whose section defines no ID."""
    found: dict[str, str] = {}
    errors = []
    for section, decision in SECTIONS_WITHOUT_ORACLE.items():
        members = sorted(mid for mid, where in sections.items() if where == section)
        if not members:
            errors.append(
                f"SECTIONS_WITHOUT_ORACLE lists section {section} ({decision}), but no ID of "
                f"{MATHS} is defined there. Remove the entry from tools/check_maths_ids.py."
            )
        found.update(dict.fromkeys(members, decision))
    return found, errors


def spec_ids(spec_dir: Path, root: Path) -> tuple[dict[str, set[str]], list[str]]:
    """``{id: {function, ...}}`` from the docstrings of public ``_spec`` functions."""
    found: dict[str, set[str]] = {}
    errors = []
    for path in sorted(spec_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            where = f"{path.relative_to(root)}:{node.lineno} {node.name}"
            ids = ID_RE.findall(ast.get_docstring(node) or "")
            if not ids:
                errors.append(
                    f"{where} cites no maths ID. List the IDs it implements in its docstring "
                    "(for example 'Implements M-S1.'), or make it private with a leading "
                    "underscore if it is a helper."
                )
            for mid in ids:
                found.setdefault(mid, set()).add(where)
    return found, errors


def _maths_marker_ids(node: ast.expr) -> list[str] | None:
    """The IDs of a ``pytest.mark.maths(...)`` expression, or None if it is not one."""
    if not isinstance(node, ast.Call):
        return None
    if not ast.unparse(node.func).endswith("mark.maths"):
        return None
    ids = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            ids.append(arg.value)
    return ids


def cited_by_tests(tests_dir: Path, root: Path) -> dict[str, set[str]]:
    """``{id: {test, ...}}`` from ``pytest.mark.maths`` markers under ``tests_dir``."""
    found: dict[str, set[str]] = {}
    for path in sorted(tests_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            exprs: list[ast.expr] = []
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                exprs = list(node.decorator_list)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                value = node.value
                exprs = list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
            for expr in exprs:
                ids = _maths_marker_ids(expr)
                if ids is None:
                    continue
                where = f"{path.relative_to(root)}:{getattr(node, 'lineno', '?')}"
                for mid in ids:
                    found.setdefault(mid, set()).add(where)
    return found


def check(root: Path) -> list[str]:
    maths = root / MATHS
    if not maths.exists():
        return [f"{MATHS} is missing. Restore it from main."]
    text = maths.read_text()
    defined_counts, cited = defined_ids(text)
    defined = set(defined_counts)
    errors = []

    for mid, count in sorted(defined_counts.items()):
        if count > 1:
            errors.append(f"{MATHS}: {mid} is defined by {count} headings; an ID is defined once.")
    for mid in sorted(cited - defined):
        errors.append(
            f"{MATHS} cites {mid}, which no heading defines. Fix the reference, or ask the "
            "maintainers to define it (docs/maths.md is protected)."
        )

    spec, spec_errors = spec_ids(root / SPEC, root) if (root / SPEC).exists() else ({}, [])
    errors += spec_errors
    for mid in sorted(spec.keys() - defined):
        for where in sorted(spec[mid]):
            errors.append(f"{where} cites {mid}, which {MATHS} does not define. Use a defined ID.")

    tests = cited_by_tests(root / TESTS, root) if (root / TESTS).exists() else {}
    for mid in sorted(tests.keys() - defined):
        for where in sorted(tests[mid]):
            errors.append(f"{where} marks {mid}, which {MATHS} does not define. Use a defined ID.")

    sections = id_sections(text)
    pending, list_errors = not_yet_implemented(sections, spec.keys() & tests.keys())
    errors += list_errors
    no_oracle, no_oracle_errors = without_oracle(sections)
    errors += no_oracle_errors
    for mid in sorted(spec.keys() & no_oracle.keys()):
        for where in sorted(spec[mid]):
            errors.append(
                f"{where} cites {mid}, whose section has no oracle by decision "
                f"({no_oracle[mid]}). Remove the function, or ask the maintainers to change "
                "the decision."
            )
    for mid in sorted(defined - spec.keys() - pending.keys() - no_oracle.keys()):
        errors.append(
            f"{mid} has no oracle function. Add one to {SPEC}/ whose docstring cites {mid}."
        )
    for mid in sorted(defined - tests.keys() - pending.keys()):
        errors.append(f"{mid} has no test. Mark a test with @pytest.mark.maths({mid!r}).")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    errors = check(args.root)
    for error in errors:
        print(f"check_maths_ids: {error}", file=sys.stderr)
    if not errors:
        text = (args.root / MATHS).read_text()
        defined, _ = defined_ids(text)
        sections = id_sections(text)
        pending, _ = not_yet_implemented(sections, set())
        no_oracle, _ = without_oracle(sections)
        waiting = len(defined.keys() & pending.keys())
        tested = len(defined.keys() & no_oracle.keys())
        print(
            f"check_maths_ids: OK, {len(defined)} IDs: {len(defined) - waiting - tested} with an "
            f"oracle function and a test, {tested} with a test and no oracle by decision, "
            f"{waiting} listed as not yet implemented."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
