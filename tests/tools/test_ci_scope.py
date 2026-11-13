"""tools/ci_scope.py skips the test suites only when every changed file is under ``docs/``,
``.github/`` or ``changes/`` (task T0.7)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .conftest import run_tool, write


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with a ``main`` branch and a checked-out feature branch."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ci@example.invalid")
    git(tmp_path, "config", "user.name", "CI")
    write(tmp_path, {"src/estel/a.py": "x = 1\n", "docs/a.md": "a", "README.md": "r"})
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "checkout", "-q", "-b", "feature")
    return tmp_path


def scope(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return run_tool("ci_scope.py", "--root", str(root), *args)


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param({"docs/a.md": "b"}, "tests=false", id="docs"),
        pytest.param({".github/workflows/t.yml": "x"}, "tests=false", id="github"),
        pytest.param({"changes/9.patch.md": "x"}, "tests=false", id="changes"),
        pytest.param(
            {"docs/a.md": "b", ".github/x": "y", "changes/9.patch.md": "z"},
            "tests=false",
            id="all-three",
        ),
        pytest.param({"src/estel/a.py": "x = 2\n"}, "tests=true", id="code"),
        pytest.param({"docs/a.md": "b", "README.md": "s"}, "tests=true", id="docs-and-root"),
        pytest.param({"docs-old/a.md": "b"}, "tests=true", id="prefix-in-name-only"),
        pytest.param({"tests/docs/a.py": "b"}, "tests=true", id="docs-not-at-root"),
    ],
)
def test_scope_of_a_change(repo: Path, files: dict[str, str], expected: str) -> None:
    write(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    result = scope(repo, "--base", "main")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_a_file_moved_into_docs_is_a_code_change(repo: Path) -> None:
    git(repo, "mv", "src/estel/a.py", "docs/a.py")
    git(repo, "commit", "-q", "-m", "move")
    assert scope(repo, "--base", "main").stdout.strip() == "tests=true"


def test_an_empty_diff_runs_the_tests(repo: Path) -> None:
    assert scope(repo, "--base", "main").stdout.strip() == "tests=true"


def test_no_base_runs_the_tests(repo: Path) -> None:
    write(repo, {"docs/a.md": "b"})
    git(repo, "commit", "-qam", "docs")
    assert scope(repo).stdout.strip() == "tests=true"


def test_an_unknown_base_fails_and_says_what_to_do(repo: Path) -> None:
    result = scope(repo, "--base", "origin/nowhere")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "fetch-depth: 0" in result.stderr
