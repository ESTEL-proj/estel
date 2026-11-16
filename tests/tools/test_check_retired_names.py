"""tools/check_retired_names.py fails when a retired public name comes back."""

from __future__ import annotations

from pathlib import Path

import pytest

from check_retired_names import check

from .conftest import REPO, run_tool, write

PART4 = """# Part IV
## 21. Review
Nothing retired here.
## 22. Decision log
| R-A02 | `norm` | `norm_batch_mean` becomes `ref_batch_mean` |
## 23. Points to settle
"""

MATHS = """# maths
### 1.7 Scattering statistics — vanilla ScatCov
- [1.7 Scattering statistics — vanilla ScatCov](#17-scattering-statistics--vanilla-scatcov)
- **Source:** `Base_DataClass`, `ST_Operator.vanilla`.
`norm="self"` (STL's `store_ref`, `load_ref` and `vanilla`); STL's `to_angular_ft`.
"""

PUBLIC = '''
class Data:
    """Data of every geometry."""

    def copy(self):
        """A copy."""


def _private():
    """STL's DataClass and store_ref may be named here."""
'''

GOOD = {
    "docs/transfer/part4.md": PART4,
    "docs/maths.md": MATHS,
    "AGENTS.md": '# AGENTS\nUse `norm="self"`.\n',
    "src/estel/__init__.py": "from ._data import Data\n",
    "src/estel/_data.py": PUBLIC,
}


def test_repository_passes() -> None:
    result = run_tool("check_retired_names.py", "--root", str(REPO))
    assert result.returncode == 0, result.stderr


def test_allowed_uses_pass(tmp_path: Path) -> None:
    write(tmp_path, GOOD)
    assert check(tmp_path) == []


@pytest.mark.parametrize(
    ("change", "where", "found"),
    [
        pytest.param(
            {"docs/maths.md": MATHS + 'With `norm="vanilla"`.\n'},
            "docs/maths.md:6",
            "vanilla",
            id="maths-body",
        ),
        pytest.param(
            {"docs/transfer/part2.md": "Use angular\\_ft.\n"},
            "docs/transfer/part2.md:1",
            "angular\\_ft",
            id="escaped-underscores",
        ),
        pytest.param(
            {"docs/transfer/part4.md": PART4 + "Uses ScatteringPlan.\n"},
            "docs/transfer/part4.md:7",
            "ScatteringPlan",
            id="part4-after-section-22",
        ),
        pytest.param(
            {"AGENTS.md": "STL's `store_ref` is not attributed outside docs/.\n"},
            "AGENTS.md:1",
            "store_ref",
            id="agents-attribution",
        ),
        pytest.param(
            {"src/estel/_data.py": PUBLIC.replace("A copy.", "As STL's `load_ref`.")},
            "src/estel/_data.py:5 (docstring of estel.Data.copy)",
            "load_ref",
            id="public-docstring",
        ),
    ],
)
def test_retired_name_fails(tmp_path: Path, change: dict[str, str], where: str, found: str) -> None:
    write(tmp_path, {**GOOD, **change})
    hits = check(tmp_path)
    assert [(h.where, h.found) for h in hits] == [(where, found)]
    assert "Use `" in hits[0].message()
