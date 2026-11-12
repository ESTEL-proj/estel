"""tools/check_agent_mirrors.py fails when an agent context file drifts from AGENTS.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import REPO, run_tool, write

AGENTS = "# AGENTS.md\n\nRule 1.\n"
CLAUDE = "@AGENTS.md\n\n## Notes for Claude Code\n"


def good_repo(tmp_path: Path) -> Path:
    write(
        tmp_path,
        {
            "AGENTS.md": AGENTS,
            "CLAUDE.md": CLAUDE,
            "src/estel/backend/AGENTS.md": "# backend rules\n",
            "src/estel/backend/CLAUDE.md": "@AGENTS.md\n",
        },
    )
    assert run_tool("check_agent_mirrors.py", "--root", str(tmp_path), "--write").returncode == 0
    return tmp_path


def check(root: Path, *flags: str) -> tuple[int, str]:
    result = run_tool("check_agent_mirrors.py", "--root", str(root), *flags)
    return result.returncode, result.stderr


def test_repository_passes() -> None:
    code, err = check(REPO, "--allow-missing-mirror")
    assert code == 0, err


def test_consistent_repository_passes(tmp_path: Path) -> None:
    code, err = check(good_repo(tmp_path))
    assert code == 0, err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"AGENTS.md": AGENTS + "Rule 2.\n"},
            ".github/copilot-instructions.md differs from AGENTS.md",
            id="mirror-stale",
        ),
        pytest.param(
            {".github/copilot-instructions.md": "hand-edited\n"},
            ".github/copilot-instructions.md differs from AGENTS.md",
            id="mirror-edited",
        ),
        pytest.param(
            {"CLAUDE.md": "## Notes\n@AGENTS.md\n"},
            "CLAUDE.md must start with the line `@AGENTS.md`",
            id="claude-without-import",
        ),
        pytest.param(
            {"src/estel/_spec/AGENTS.md": "# oracle rules\n"},
            "src/estel/_spec/AGENTS.md has no CLAUDE.md beside it",
            id="nested-agents-without-claude",
        ),
        pytest.param(
            {"tests/reference/CLAUDE.md": "@AGENTS.md\n"},
            "tests/reference/CLAUDE.md has no AGENTS.md beside it",
            id="nested-claude-without-agents",
        ),
    ],
)
def test_crafted_violation_fails(tmp_path: Path, change: dict[str, str], message: str) -> None:
    root = good_repo(tmp_path)
    write(root, change)
    code, err = check(root, "--allow-missing-mirror")
    assert code == 1
    assert message in err


def test_missing_mirror_fails_unless_allowed(tmp_path: Path) -> None:
    root = good_repo(tmp_path)
    (root / ".github/copilot-instructions.md").unlink()
    code, err = check(root)
    assert code == 1
    assert "copilot-instructions.md is missing" in err
    assert check(root, "--allow-missing-mirror")[0] == 0
