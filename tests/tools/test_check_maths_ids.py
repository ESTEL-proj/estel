"""tools/check_maths_ids.py fails when maths.md, the oracle and the tests disagree."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from .conftest import REPO, TOOLS, run_tool, write

MATHS = """# maths
## 1. 2D
### 1.1 Data — M-DATA-2D
Arrays. See M-S1.
#### M-S1
```latex
# M-FENCED is neither a heading nor a citation
```
"""

SPEC = '''
def spec_data(x):
    """Implements M-DATA-2D."""

def spec_s1(x):
    """Implements M-S1 (see also M-DATA-2D)."""

def _helper(x):
    pass
'''

TESTS = """
import pytest

pytestmark = pytest.mark.maths("M-DATA-2D")

@pytest.mark.maths("M-S1")
def test_s1():
    pass
"""

GOOD = {
    "docs/maths.md": MATHS,
    "src/estel/_spec/__init__.py": SPEC,
    "tests/spec/test_s1.py": TESTS,
}


# Section 5 has no oracle by decision (Part IV, §22, "No oracle for the syntheses"): its IDs
# need a test and no _spec function. The tool lists no single ID since M-PS-2D got its oracle
# (issue #149), and no section since section 4 got its own (issue #165): the tests of these lists
# give them one entry each, PENDING_ID and PENDING_SECTION, in the tool's own process.
PENDING = """### 1.9 Power spectrum — M-PS-2D
"""
PENDING_ID = {"M-PS-2D": "T4.1"}
PENDING_SECTION = {"4": "T5.4"}
PENDING_SECTIONS = """## 4. HEALPix
### 4.0 Data — M-DATA-HEALPIX
"""
NO_ORACLE = """## 5. Syntheses
### 5.1 Target — M-SYN-TARGET
"""
MARKED = 'import pytest\npytestmark = pytest.mark.maths("{}")\n'
COMPLETE = {
    **GOOD,
    "docs/maths.md": MATHS + NO_ORACLE,
    "tests/synthesis/test_target.py": MARKED.format("M-SYN-TARGET"),
}


def check(root: Path) -> tuple[int, str]:
    result = run_tool("check_maths_ids.py", "--root", str(root))
    return result.returncode, result.stderr


def check_in_process(root: Path, monkeypatch: pytest.MonkeyPatch, **lists: Any) -> list[str]:
    """The errors of the tool on ``root``, run in this process with its lists replaced by
    ``lists`` (for example ``NOT_YET_IMPLEMENTED_IDS=PENDING_ID``)."""
    spec = importlib.util.spec_from_file_location("check_maths_ids", TOOLS / "check_maths_ids.py")
    assert spec is not None
    assert spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    for name, value in lists.items():
        monkeypatch.setattr(tool, name, value)
    errors: list[str] = tool.check(root)
    return errors


def check_with_listed_id(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The errors of the tool on ``root``, with ``PENDING_ID`` as its list of single IDs not yet
    implemented."""
    return check_in_process(root, monkeypatch, NOT_YET_IMPLEMENTED_IDS=PENDING_ID)


def check_with_listed_section(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The errors of the tool on ``root``, with ``PENDING_SECTION`` as its list of sections not
    yet implemented."""
    return check_in_process(root, monkeypatch, NOT_YET_IMPLEMENTED_SECTIONS=PENDING_SECTION)


def test_repository_passes() -> None:
    code, err = check(REPO)
    assert code == 0, err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"docs/maths.md": MATHS + "#### M-S1 again\n"},
            "M-S1 is defined by 2 headings",
            id="duplicate-definition",
        ),
        pytest.param(
            {"docs/maths.md": MATHS + "Uses M-NOSUCH.\n"},
            "cites M-NOSUCH, which no heading defines",
            id="undefined-reference-in-maths",
        ),
        pytest.param(
            {"src/estel/_spec/s2.py": 'def spec_s2(x):\n    """Second moment."""\n'},
            "spec_s2 cites no maths ID",
            id="spec-function-without-id",
        ),
        pytest.param(
            {"src/estel/_spec/s2.py": 'def spec_s2(x):\n    """Implements M-S2."""\n'},
            "cites M-S2, which docs/maths.md does not define",
            id="spec-cites-undefined-id",
        ),
        pytest.param(
            {
                "tests/test_x.py": 'import pytest\n@pytest.mark.maths("M-S3")\ndef test_x():\n'
                "    pass\n"
            },
            "marks M-S3, which docs/maths.md does not define",
            id="test-cites-undefined-id",
        ),
        pytest.param(
            {"docs/maths.md": MATHS + "#### M-S2\n"},
            "M-S2 has no oracle function",
            id="id-without-oracle",
        ),
        pytest.param(
            {"tests/spec/test_s1.py": TESTS.replace('@pytest.mark.maths("M-S1")\n', "")},
            "M-S1 has no test",
            id="id-without-test",
        ),
    ],
)
def test_crafted_violation_fails(tmp_path: Path, change: dict[str, str], message: str) -> None:
    write(tmp_path, {**COMPLETE, **change})
    code, err = check(tmp_path)
    assert code == 1
    assert message in err


def test_listed_ids_need_no_oracle_and_no_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path, {**COMPLETE, "docs/maths.md": COMPLETE["docs/maths.md"] + PENDING_SECTIONS})
    assert check_with_listed_section(tmp_path, monkeypatch) == []


def test_unlisted_id_of_a_covered_section_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maths = COMPLETE["docs/maths.md"] + PENDING_SECTIONS + "## 2. 1D\n#### M-X\n"
    write(tmp_path, {**COMPLETE, "docs/maths.md": maths})
    errors = check_with_listed_section(tmp_path, monkeypatch)
    assert any("M-X has no oracle function" in error for error in errors)
    assert not any("M-DATA-HEALPIX" in error for error in errors)


def test_unlisted_section_is_not_accepted(tmp_path: Path) -> None:
    """A section that the lists do not name needs its oracle and its tests."""
    write(tmp_path, {**COMPLETE, "docs/maths.md": COMPLETE["docs/maths.md"] + PENDING_SECTIONS})
    code, err = check(tmp_path)
    assert code == 1
    assert "M-DATA-HEALPIX has no oracle function" in err


def test_id_without_oracle_by_decision_still_needs_a_test(tmp_path: Path) -> None:
    """Section 5 needs no oracle, but each of its IDs needs a test."""
    write(tmp_path, {**COMPLETE, "tests/synthesis/test_target.py": "import pytest\n"})
    code, err = check(tmp_path)
    assert code == 1
    assert "M-SYN-TARGET has no test" in err
    assert "M-SYN-TARGET has no oracle function" not in err


def test_oracle_function_of_a_section_without_oracle_fails(tmp_path: Path) -> None:
    spec = 'def spec_target(x):\n    """Implements M-SYN-TARGET."""\n'
    write(tmp_path, {**COMPLETE, "src/estel/_spec/syn.py": spec})
    code, err = check(tmp_path)
    assert code == 1
    assert "cites M-SYN-TARGET, whose section has no oracle by decision" in err


def test_section_without_oracle_and_without_id_fails(tmp_path: Path) -> None:
    """An entry of ``SECTIONS_WITHOUT_ORACLE`` whose section defines no ID must be removed."""
    write(tmp_path, {**COMPLETE, "docs/maths.md": MATHS + PENDING_SECTIONS})
    code, err = check(tmp_path)
    assert code == 1
    assert "SECTIONS_WITHOUT_ORACLE lists section 5" in err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({}, "lists section 4 (T5.4), but no ID", id="section-without-id"),
        pytest.param(
            {
                "docs/maths.md": COMPLETE["docs/maths.md"] + PENDING_SECTIONS,
                "src/estel/_spec/healpix.py": (
                    'def spec_healpix(x):\n    """Implements M-DATA-HEALPIX."""\n'
                ),
                "tests/spec/test_healpix.py": MARKED.format("M-DATA-HEALPIX"),
            },
            "lists section 4 (T5.4), but every ID of that section has an oracle function",
            id="section-implemented",
        ),
    ],
)
def test_stale_listed_section_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, str], message: str
) -> None:
    """An entry of ``NOT_YET_IMPLEMENTED_SECTIONS`` that names nothing, or is no longer needed,
    must be removed."""
    write(tmp_path, {**COMPLETE, **change})
    assert any(message in error for error in check_with_listed_section(tmp_path, monkeypatch))


def test_listed_id_needs_no_oracle_and_no_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ID of ``NOT_YET_IMPLEMENTED_IDS`` is accepted without an oracle function or a test."""
    write(tmp_path, {**COMPLETE, "docs/maths.md": COMPLETE["docs/maths.md"] + PENDING})
    assert check_with_listed_id(tmp_path, monkeypatch) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {},
            "lists M-PS-2D (T4.1), which docs/maths.md does not define",
            id="listed-id-not-defined",
        ),
        pytest.param(
            {
                "docs/maths.md": COMPLETE["docs/maths.md"] + PENDING,
                "src/estel/_spec/ps.py": 'def spec_ps(x):\n    """Implements M-PS-2D."""\n',
                "tests/spec/test_ps.py": MARKED.format("M-PS-2D"),
            },
            "lists M-PS-2D (T4.1), which has an oracle function and a test",
            id="listed-id-implemented",
        ),
    ],
)
def test_stale_listed_id_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, str], message: str
) -> None:
    """An entry of ``NOT_YET_IMPLEMENTED_IDS`` that names nothing, or is no longer needed, must
    be removed."""
    write(tmp_path, {**COMPLETE, **change})
    assert any(message in error for error in check_with_listed_id(tmp_path, monkeypatch))
