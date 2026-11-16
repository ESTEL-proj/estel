"""tools/check_maths_ids.py fails when maths.md, the oracle and the tests disagree."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import REPO, run_tool, write

MATHS = """# maths
## 1. Planar
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


# Sections 3 to 5 and M-PS-2D are listed in the tool as not yet implemented (issue #18).
PENDING = """### 1.9 Power spectrum — M-PS-2D
## 3. Tensor products
### 3.1 Data — M-DATA-TP
"""
PENDING_SECTIONS = """## 4. HEALPix
### 4.0 Data — M-DATA-HPX
## 5. Syntheses
### 5.1 Target — M-SYN-TARGET
"""
MARKED = 'import pytest\npytestmark = pytest.mark.maths("{}")\n'
COMPLETE = {**GOOD, "docs/maths.md": MATHS + PENDING + PENDING_SECTIONS}


def check(root: Path) -> tuple[int, str]:
    result = run_tool("check_maths_ids.py", "--root", str(root))
    return result.returncode, result.stderr


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
            {"docs/maths.md": MATHS + "Uses M-S9.\n"},
            "cites M-S9, which no heading defines",
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


def test_listed_ids_need_no_oracle_and_no_test(tmp_path: Path) -> None:
    write(tmp_path, COMPLETE)
    code, err = check(tmp_path)
    assert code == 0, err


def test_unlisted_id_of_a_covered_section_is_not_accepted(tmp_path: Path) -> None:
    write(
        tmp_path, {**COMPLETE, "docs/maths.md": COMPLETE["docs/maths.md"] + "## 2. 1D\n#### M-X\n"}
    )
    code, err = check(tmp_path)
    assert code == 1
    assert "M-X has no oracle function" in err
    assert "M-DATA-TP" not in err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"docs/maths.md": MATHS + PENDING},
            "lists section 4 (T5.1), but no ID",
            id="section-without-id",
        ),
        pytest.param(
            {"docs/maths.md": MATHS + PENDING_SECTIONS + "## 3. Tensor products\n#### M-TP\n"},
            "lists M-PS-2D (T4.1), which docs/maths.md does not define",
            id="listed-id-not-defined",
        ),
        pytest.param(
            {
                "src/estel/_spec/tp.py": 'def spec_tp(x):\n    """Implements M-DATA-TP."""\n',
                "tests/spec/test_tp.py": MARKED.format("M-DATA-TP"),
            },
            "lists section 3 (T3.2), but every ID of that section has an oracle function",
            id="section-implemented",
        ),
        pytest.param(
            {
                "src/estel/_spec/ps.py": 'def spec_ps(x):\n    """Implements M-PS-2D."""\n',
                "tests/spec/test_ps.py": MARKED.format("M-PS-2D"),
            },
            "lists M-PS-2D (T4.1), which has an oracle function and a test",
            id="listed-id-implemented",
        ),
    ],
)
def test_stale_list_entry_fails(tmp_path: Path, change: dict[str, str], message: str) -> None:
    """An entry of the lists that names nothing, or is no longer needed, must be removed."""
    write(tmp_path, {**COMPLETE, **change})
    code, err = check(tmp_path)
    assert code == 1
    assert message in err
