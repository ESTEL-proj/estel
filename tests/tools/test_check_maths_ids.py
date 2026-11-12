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


def check(root: Path, *flags: str) -> tuple[int, str]:
    result = run_tool("check_maths_ids.py", "--root", str(root), *flags)
    return result.returncode, result.stderr


def test_repository_passes_without_coverage() -> None:
    code, err = check(REPO, "--allow-unimplemented")
    assert code == 0, err


def test_consistent_repository_passes_with_coverage(tmp_path: Path) -> None:
    write(tmp_path, GOOD)
    code, err = check(tmp_path)
    assert code == 0, err


@pytest.mark.parametrize(
    ("change", "message", "flags"),
    [
        pytest.param(
            {"docs/maths.md": MATHS + "#### M-S1 again\n"},
            "M-S1 is defined by 2 headings",
            ("--allow-unimplemented",),
            id="duplicate-definition",
        ),
        pytest.param(
            {"docs/maths.md": MATHS + "Uses M-S9.\n"},
            "cites M-S9, which no heading defines",
            ("--allow-unimplemented",),
            id="undefined-reference-in-maths",
        ),
        pytest.param(
            {"src/estel/_spec/s2.py": 'def spec_s2(x):\n    """Second moment."""\n'},
            "spec_s2 cites no maths ID",
            ("--allow-unimplemented",),
            id="spec-function-without-id",
        ),
        pytest.param(
            {"src/estel/_spec/s2.py": 'def spec_s2(x):\n    """Implements M-S2."""\n'},
            "cites M-S2, which docs/maths.md does not define",
            ("--allow-unimplemented",),
            id="spec-cites-undefined-id",
        ),
        pytest.param(
            {"tests/test_x.py": 'import pytest\n@pytest.mark.maths("M-S3")\ndef test_x():\n'
             "    pass\n"},
            "marks M-S3, which docs/maths.md does not define",
            ("--allow-unimplemented",),
            id="test-cites-undefined-id",
        ),
        pytest.param(
            {"docs/maths.md": MATHS + "#### M-S2\n"},
            "M-S2 has no oracle function",
            (),
            id="id-without-oracle",
        ),
        pytest.param(
            {"tests/spec/test_s1.py": TESTS.replace('@pytest.mark.maths("M-S1")\n', "")},
            "M-S1 has no test",
            (),
            id="id-without-test",
        ),
    ],
)
def test_crafted_violation_fails(
    tmp_path: Path, change: dict[str, str], message: str, flags: tuple[str, ...]
) -> None:
    write(tmp_path, {**GOOD, **change})
    code, err = check(tmp_path, *flags)
    assert code == 1
    assert message in err


def test_coverage_is_skipped_only_on_request(tmp_path: Path) -> None:
    write(tmp_path, {"docs/maths.md": MATHS})
    assert check(tmp_path, "--allow-unimplemented")[0] == 0
    assert check(tmp_path)[0] == 1
