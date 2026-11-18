"""tools/check_identifiers.py fails when a cited identifier has no row in its list or in the log.

This file is itself searched for citations, so every identifier written here exists in the
repository; the small trees below define only some of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from check_identifiers import check

from .conftest import REPO, run_tool, write

MATHS = """# maths
| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-1 | Filtering is a correlation | reproduce | reproduced |
"""

PART2 = """# Part II
### 15.2 Work breakdown of step 1
| ID | Phase | Task |
| --- | --- | --- |
| T2.12 | P2 | The Gram form of the second layer |
"""

LF1_ROW = "| LF1 | Structure | JAX backend |\n"
PART3_HEAD = """# Part III
## 16. Register of later features
| ID | Category | Feature |
| --- | --- | --- |
"""
PART3 = PART3_HEAD + LF1_ROW + "| W1 | Work that is not a feature | GPU CI |\n## 17. JAX backend\n"

G1_ROW = "| 4 November 2026 | G1 | Old open points | replaced |\n"
PART4_HEAD = """# Part IV
## 21. Critical review: failure modes
| # | Failure mode |
| --- | --- |
| F1 | References that freeze bugs |
## 22. Decision log
| Date | ID | Point | Decision |
| --- | --- | --- | --- |
| 17 November 2026 | LF14 | Compact layout | a task now; the log may name LF2 |
| 16 November 2026 | R-A07, R-S08 | Names | renamed |
"""
PART4_TAIL = """## 23. Points to settle
| ID | Point to settle |
| --- | --- |
| Q1 | Compact layout |
"""
PART4 = PART4_HEAD + G1_ROW + PART4_TAIL

CITES = (
    "OP-1, LF1, W1, Q1, F1 and T2.12; LF14, closed; R-A07 and R-S08, decided; "
    "R-N01, R-V10 and R-S09, kept in issues; G1.\n"
)

GOOD = {
    "docs/maths.md": MATHS,
    "docs/transfer/part2.md": PART2,
    "docs/transfer/part3.md": PART3,
    "docs/transfer/part4.md": PART4,
    "docs/transfer/part1.md": CITES,
    "AGENTS.md": "# AGENTS\nTask T2.12.\n",
    "src/estel/_module.py": '"""OP-1."""\n',
    "changes/107.minor.md": "History may name LF2.\n",
}


def test_repository_passes() -> None:
    result = run_tool("check_identifiers.py", "--root", str(REPO))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_good_tree_passes(tmp_path: Path) -> None:
    write(tmp_path, GOOD)
    report = check(tmp_path)
    assert report.errors == []
    assert (report.in_lists, report.in_log, report.in_issues) == (6, 4, 3)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"docs/transfer/part1.md": CITES + "OP-2\n"},
            "docs/transfer/part1.md:2 cites OP-2, which has no row in the open points",
            id="open-point",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "LF2\n"},
            "docs/transfer/part1.md:2 cites LF2, which has no row in the register",
            id="later-feature",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "W2\n"},
            "docs/transfer/part1.md:2 cites W2, which has no row in the register",
            id="deferred-work",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "Q2\n"},
            "docs/transfer/part1.md:2 cites Q2, which has no row in the points to settle",
            id="question",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "F2\n"},
            "docs/transfer/part1.md:2 cites F2, which has no row in the risks",
            id="risk",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "T2.13\n"},
            "docs/transfer/part1.md:2 cites T2.13, which has no row in the tasks",
            id="task",
        ),
        pytest.param(
            {"docs/transfer/part2.md": PART2 + "## 16. After the tasks\n| T2.13 | P2 | a row |\n"},
            "docs/transfer/part2.md:7 cites T2.13, which has no row in the tasks",
            id="row-after-the-list",
        ),
        pytest.param(
            {"src/estel/_module.py": '"""OP-1."""\n# R-A09\n'},
            "src/estel/_module.py:2 cites R-A09, a remark of the review of 16 November 2026",
            id="remark",
        ),
        pytest.param(
            {"docs/transfer/part4.md": PART4_HEAD + PART4_TAIL},
            "docs/transfer/part1.md:1 cites G1, an old open point with no row",
            id="old-open-point",
        ),
        pytest.param(
            {"docs/transfer/part3.md": PART3.replace(LF1_ROW, "")},
            "docs/transfer/part1.md:1 cites LF1, which has no row in the register (Part III, §16) "
            "and none in the decision log (Part IV, §22): add its row, or a row to the log",
            id="removed-from-register",
        ),
        pytest.param(
            {"docs/transfer/part3.md": PART3.replace(LF1_ROW, LF1_ROW * 2)},
            "docs/transfer/part3.md:6 defines LF1 again in the register (Part III, §16), already "
            "defined at line 5",
            id="defined-twice",
        ),
        pytest.param(
            {"docs/transfer/part3.md": PART3.replace("## 16.", "## 15.")},
            "has no heading starting with '## 16. '",
            id="list-moved",
        ),
    ],
)
def test_problem_fails(tmp_path: Path, change: dict[str, str], message: str) -> None:
    write(tmp_path, {**GOOD, **change})
    errors = check(tmp_path).errors
    assert len(errors) == 1, errors
    assert message in errors[0]
