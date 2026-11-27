"""tools/check_identifiers.py fails when a cited identifier has no row in its list or in the log.

This file is itself searched for citations, so every identifier written here exists in the
repository; the small trees below define only some of them. Each tree is a git repository, since
the tool searches the Markdown files that git tracks.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
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

CARD = '+++\nid = "{}"\n+++\n'
REGISTER_LOG_HEAD = """# Decision log of the register
| Date | ID | Point | Decision | Where |
| --- | --- | --- | --- | --- |
"""
REGISTER_LOG = REGISTER_LOG_HEAD + "| 20 November 2026 | LF30 | Tried in step 1 | a task | — |\n"

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
| 17 November 2026 | Q14 | The vector of the loss | closed |
"""
PART4 = PART4_HEAD + G1_ROW

CITES = (
    "OP-1, LF1, W1, F1 and T2.12; LF14, LF30 and Q14, closed; R-A07 and R-S08, decided; "
    "R-N01, R-V10 and R-S09, kept in issues; G1.\n"
)

GOOD = {
    "docs/maths.md": MATHS,
    "docs/transfer/part2.md": PART2,
    "docs/register/LF1.md": CARD.format("LF1"),
    "docs/register/W1.md": CARD.format("W1"),
    "docs/register/decisions.md": REGISTER_LOG,
    "docs/transfer/part4.md": PART4,
    "docs/transfer/part1.md": CITES,
    "AGENTS.md": "# AGENTS\nTask T2.12.\n",
    "src/estel/backend/AGENTS.md": "# Backend\nW1.\n",
    "bench/README.md": "# Benchmark\nTask T2.12.\n",
    "src/estel/_module.py": '"""OP-1."""\n',
    "changes/107.minor.md": "History may name LF2.\n",
}


def repository(root: Path, files: Mapping[str, str | None]) -> None:
    """Write ``files`` (None: left out) under ``root`` and track them in a new git repository."""
    write(root, {rel: text for rel, text in files.items() if text is not None})
    for command in (["git", "init", "-q"], ["git", "add", "-A"]):
        subprocess.run(command, cwd=root, check=True, capture_output=True)


@pytest.mark.slow
def test_repository_passes() -> None:
    """Slow: the tool reads every file of the repository (about 1.1 s of CPU, over the budget
    of 1 s); `make check-tools` runs it on every pull request anyway."""
    result = run_tool("check_identifiers.py", "--root", str(REPO))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_good_tree_passes(tmp_path: Path) -> None:
    repository(tmp_path, GOOD)
    write(tmp_path, {"docs/draft.md": "Not tracked by git: may name LF2.\n"})
    report = check(tmp_path)
    assert report.errors == []
    assert (report.in_lists, report.in_log, report.in_issues) == (5, 6, 3)


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
            "docs/transfer/part1.md:2 cites LF2, which has no card in the register",
            id="later-feature",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "W2\n"},
            "docs/transfer/part1.md:2 cites W2, which has no card in the register",
            id="deferred-work",
        ),
        pytest.param(
            {"docs/transfer/part1.md": CITES + "Q7\n"},
            "docs/transfer/part1.md:2 cites Q7, an old point to settle with no row",
            id="old-point-to-settle",
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
            {"src/estel/backend/AGENTS.md": "# Backend\nQ7\n"},
            "src/estel/backend/AGENTS.md:2 cites Q7, an old point to settle with no row",
            id="nested-markdown",
        ),
        pytest.param(
            {"src/estel/_module.py": '"""OP-1."""\n# R-A09\n'},
            "src/estel/_module.py:2 cites R-A09, a remark of the review of 16 November 2026",
            id="remark",
        ),
        pytest.param(
            {"docs/transfer/part4.md": PART4_HEAD},
            "docs/transfer/part1.md:1 cites G1, an old open point with no row",
            id="old-open-point",
        ),
        pytest.param(
            {"docs/register/LF1.md": None},
            "docs/transfer/part1.md:1 cites LF1, which has no card in the register "
            "(docs/register/) and no row in the decision logs (Part IV, §22; "
            "docs/register/decisions.md): add its card, or a row to a log",
            id="removed-from-register",
        ),
        pytest.param(
            {"docs/register/decisions.md": REGISTER_LOG_HEAD},
            "docs/transfer/part1.md:1 cites LF30, which has no card in the register",
            id="removed-from-register-log",
        ),
        pytest.param(
            dict.fromkeys(
                ("docs/register/LF1.md", "docs/register/W1.md", "docs/register/decisions.md")
            ),
            "docs/register/ is missing: it holds the register (docs/register/)",
            id="register-missing",
        ),
        pytest.param(
            {"docs/transfer/part2.md": PART2 + PART2.splitlines(keepends=True)[-1]},
            "docs/transfer/part2.md:6 defines T2.12 again in the tasks (Part II, §15.2), already "
            "defined at line 5",
            id="defined-twice",
        ),
        pytest.param(
            {"docs/transfer/part2.md": PART2.replace("### 15.2", "### 15.3")},
            "has no heading starting with '### 15.2 '",
            id="list-moved",
        ),
    ],
)
def test_problem_fails(tmp_path: Path, change: dict[str, str | None], message: str) -> None:
    repository(tmp_path, {**GOOD, **change})
    errors = check(tmp_path).errors
    assert len(errors) == 1, errors
    assert message in errors[0]
