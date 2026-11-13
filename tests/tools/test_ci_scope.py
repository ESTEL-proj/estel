"""tools/ci_scope.py skips the test suites only when every changed file is under ``docs/``,
``.github/`` or ``changes/``, and none is a file the tests read (task T0.7)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ci_scope import docs_read_by_tests

from .conftest import REPO, run_tool, write

# A test of the fake repository that reads a document, as tests/backend/test_protocol.py does.
READS_MATHS = 'REPO = Path(__file__).parents[1]\ntext = (REPO / "docs/maths.md").read_text()\n'


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


def test_the_tests_of_this_repository_read_maths_md() -> None:
    """tests/backend/test_protocol.py reads docs/maths.md: changing it runs the whole suite."""
    assert "docs/maths.md" in docs_read_by_tests(REPO)


def test_a_change_to_maths_md_runs_the_tests(repo: Path) -> None:
    write(repo, {"tests/test_protocol.py": READS_MATHS, "docs/maths.md": "#### M-S1\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test reads maths.md")
    git(repo, "branch", "-f", "main")
    write(repo, {"docs/maths.md": "#### M-S1\n#### M-S2\n"})
    git(repo, "commit", "-qam", "maths only")
    result = scope(repo, "--base", "main")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "tests=true"
    assert "tests read docs/maths.md" in result.stderr


@pytest.mark.parametrize(
    ("test", "change", "expected"),
    [
        pytest.param(READS_MATHS, {"docs/maths.md": "b"}, "tests=true", id="one-string"),
        pytest.param(
            'p = REPO / "docs" / "guide" / "a.md"\n',
            {"docs/guide/a.md": "b"},
            "tests=true",
            id="joined-parts",
        ),
        pytest.param(READS_MATHS, {"docs/other.md": "b"}, "tests=false", id="unread-document"),
    ],
)
def test_documents_named_by_tests(
    repo: Path, test: str, change: dict[str, str], expected: str
) -> None:
    write(repo, {"tests/test_reads.py": test})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test reads a document")
    git(repo, "branch", "-f", "main")
    write(repo, change)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "docs only")
    assert scope(repo, "--base", "main").stdout.strip() == expected


def test_deleting_a_document_the_tests_read_runs_the_tests(repo: Path) -> None:
    write(repo, {"tests/test_protocol.py": READS_MATHS, "docs/maths.md": "#### M-S1\n"})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a test reads maths.md")
    git(repo, "branch", "-f", "main")
    git(repo, "rm", "-q", "docs/maths.md")
    git(repo, "commit", "-q", "-m", "drop maths.md")
    assert scope(repo, "--base", "main").stdout.strip() == "tests=true"
