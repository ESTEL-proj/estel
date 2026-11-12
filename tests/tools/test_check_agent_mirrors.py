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
    assert run_tool("check_agent_mirrors.py", "--root", str(tmp_path), "--fix").returncode == 0
    return tmp_path


def check(root: Path, *flags: str) -> tuple[int, str]:
    result = run_tool("check_agent_mirrors.py", "--root", str(root), *flags)
    return result.returncode, result.stderr


def test_repository_passes() -> None:
    code, err = check(REPO)
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
    code, err = check(root)
    assert code == 1
    assert message in err


def test_missing_mirror_fails(tmp_path: Path) -> None:
    root = good_repo(tmp_path)
    (root / ".github/copilot-instructions.md").unlink()
    code, err = check(root)
    assert code == 1
    assert "copilot-instructions.md is missing" in err


@pytest.mark.parametrize("state", ["missing", "stale", "edited"])
def test_fix_regenerates_mirror(tmp_path: Path, state: str) -> None:
    root = good_repo(tmp_path)
    mirror = root / ".github/copilot-instructions.md"
    if state == "missing":
        mirror.unlink()
    elif state == "stale":
        write(root, {"AGENTS.md": AGENTS + "Rule 2.\n"})
    else:
        write(root, {".github/copilot-instructions.md": "hand-edited\n"})
    code, err = check(root, "--fix")
    assert code == 0, err
    assert mirror.read_text().endswith((root / "AGENTS.md").read_text())
    assert check(root)[0] == 0


def test_fix_does_not_hide_other_violations(tmp_path: Path) -> None:
    root = good_repo(tmp_path)
    write(root, {"CLAUDE.md": "## Notes\n@AGENTS.md\n"})
    code, err = check(root, "--fix")
    assert code == 1
    assert "CLAUDE.md must start with the line `@AGENTS.md`" in err
