"""tools/version_impact.py fails when the declared bump is below the required one, and when the
reference counters and the references do not move together (Part I, §7.2)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from .conftest import run_tool, write

API = {"estel.f": {"kind": "function", "signature": "(x)"}}
COUNTERS_FILE = "src/estel/_reference.py"


def counters(statistics: int | str = 3, synthesis: int | str = 1) -> str:
    return f"STATISTICS_REFERENCE = {statistics}\nSYNTHESIS_REFERENCE = {synthesis}\n"


def manifest(public: dict[str, dict[str, str]]) -> str:
    return json.dumps({"schema": 1, "public": public})


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with a ``main`` branch and a checked-out feature branch."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ci@example.invalid")
    git(tmp_path, "config", "user.name", "CI")
    write(
        tmp_path,
        {
            "pyproject.toml": '[project]\nname = "estel"\nversion = "1.2.0"\n',
            "api/manifest.json": manifest(API),
            "tests/reference/s2.npz": "v1",
            COUNTERS_FILE: counters(),
            "changes/.gitkeep": "",
            "README.md": "x",
        },
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "checkout", "-q", "-b", "feature")
    return tmp_path


def commit(root: Path, files: dict[str, str]) -> None:
    write(root, files)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "change")


def impact(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return run_tool("version_impact.py", "--base", "main", "--root", str(root), *args)


def test_docs_change_with_patch_fragment_passes(repo: Path) -> None:
    commit(repo, {"README.md": "y", "changes/7.patch.md": "Docs."})
    result = impact(repo, "--pr", "7")
    assert result.returncode == 0, result.stderr
    assert "required bump: patch" in result.stdout


def test_added_entry_with_minor_fragment_passes(repo: Path) -> None:
    api = {**API, "estel.g": {"kind": "function", "signature": "()"}}
    commit(repo, {"api/manifest.json": manifest(api), "changes/7.minor.md": "Add g."})
    result = impact(repo, "--pr", "7")
    assert result.returncode == 0, result.stderr
    assert "minor: estel.g added" in result.stdout


@pytest.mark.parametrize(
    ("files", "args", "message"),
    [
        pytest.param(
            {"README.md": "y"},
            ("--pr", "7"),
            "No news fragment. Add changes/7.patch.md",
            id="no-fragment",
        ),
        pytest.param(
            {
                "api/manifest.json": manifest(
                    {**API, "estel.g": {"kind": "function", "signature": "()"}}
                ),
                "changes/7.patch.md": "Add g.",
            },
            ("--pr", "7"),
            "declares a patch change, but the change requires minor",
            id="added-entry-declared-patch",
        ),
        pytest.param(
            {"api/manifest.json": manifest({}), "changes/7.minor.md": "Drop f."},
            ("--pr", "7"),
            "declares a minor change, but the change requires major",
            id="removed-entry-declared-minor",
        ),
        pytest.param(
            {
                "api/manifest.json": manifest(
                    {"estel.f": {"kind": "function", "signature": "(x, y)"}}
                ),
                "changes/7.minor.md": "Change f.",
            },
            ("--pr", "7"),
            "declares a minor change, but the change requires major",
            id="changed-signature-declared-minor",
        ),
        pytest.param(
            {"README.md": "y", "changes/7.patch.md": "Faster."},
            ("--pr", "7", "--labels", "bench"),
            "declares a patch change, but the change requires minor",
            id="bench-label-declared-patch",
        ),
        pytest.param(
            {"tests/reference/s2.npz": "v2", "changes/7.patch.md": "Oops."},
            ("--pr", "7"),
            "but STATISTICS_REFERENCE went from 3 to 3; it must increase by exactly one, to 4",
            id="reference-changed-without-bump",
        ),
        pytest.param(
            {"tests/reference/s2.npz": "v2", "changes/7.patch.md": "Oops."},
            ("--pr", "7", "--labels", "major"),
            "but STATISTICS_REFERENCE went from 3 to 3; it must increase by exactly one, to 4",
            id="major-label-does-not-replace-bump",
        ),
        pytest.param(
            {
                "tests/reference/s2.npz": "v2",
                COUNTERS_FILE: counters(statistics=5),
                "changes/7.patch.md": "Jump.",
            },
            ("--pr", "7"),
            "but STATISTICS_REFERENCE went from 3 to 5; it must increase by exactly one, to 4",
            id="reference-changed-bump-by-two",
        ),
        pytest.param(
            {COUNTERS_FILE: counters(statistics=4), "changes/7.patch.md": "Bump."},
            ("--pr", "7"),
            "STATISTICS_REFERENCE changed from 3 to 4, but tests/reference/ did not change",
            id="bump-without-reference-change",
        ),
        pytest.param(
            {COUNTERS_FILE: counters(statistics=2), "changes/7.patch.md": "Down."},
            ("--pr", "7"),
            "STATISTICS_REFERENCE changed from 3 to 2, but tests/reference/ did not change",
            id="counter-decreased",
        ),
        pytest.param(
            {COUNTERS_FILE: counters(synthesis=2), "changes/7.patch.md": "Bump."},
            ("--pr", "7"),
            "SYNTHESIS_REFERENCE changed from 1 to 2, but SYNTHESIS_REFERENCE has no references",
            id="synthesis-bump-without-references",
        ),
        pytest.param(
            {COUNTERS_FILE: counters(statistics="3 + 1"), "changes/7.patch.md": "Expr."},
            ("--pr", "7"),
            "STATISTICS_REFERENCE = 3 + 1 is not an integer literal",
            id="counter-not-a-literal",
        ),
        pytest.param(
            {COUNTERS_FILE: "SYNTHESIS_REFERENCE = 1\n", "changes/7.patch.md": "Drop."},
            ("--pr", "7"),
            "STATISTICS_REFERENCE is missing",
            id="counter-missing",
        ),
        pytest.param(
            {"README.md": "y", "changes/fix.patch.md": "Docs."},
            ("--pr", "7"),
            "changes/fix.patch.md is not a valid news fragment",
            id="misnamed-fragment",
        ),
        pytest.param(
            {"README.md": "y", "changes/8.patch.md": "Docs."},
            ("--pr", "7"),
            "names PR 8, but this is PR 7",
            id="fragment-for-another-pr",
        ),
    ],
)
def test_crafted_violation_fails(
    repo: Path, files: dict[str, str], args: tuple[str, ...], message: str
) -> None:
    commit(repo, files)
    result = impact(repo, *args)
    assert result.returncode == 1
    assert message in result.stderr


def test_counter_file_deleted_fails(repo: Path) -> None:
    git(repo, "rm", "-q", COUNTERS_FILE)
    commit(repo, {"changes/7.patch.md": "Drop counters."})
    result = impact(repo, "--pr", "7")
    assert result.returncode == 1
    assert f"{COUNTERS_FILE} is missing. Restore it from main." in result.stderr


def test_reference_change_with_bump_by_one_passes(repo: Path) -> None:
    commit(
        repo,
        {
            "tests/reference/s2.npz": "v2",
            COUNTERS_FILE: counters(statistics=4),
            "changes/7.patch.md": "New references.",
        },
    )
    result = impact(repo, "--pr", "7")
    assert result.returncode == 0, result.stderr
    assert "STATISTICS_REFERENCE bumped from 3 to 4" in result.stdout


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({"tests/reference/s2.npz": "v2"}, id="reference-changed-without-bump"),
        pytest.param({COUNTERS_FILE: counters(statistics=4)}, id="bump-without-reference"),
    ],
)
def test_mismatch_is_only_reported_before_1_0(repo: Path, files: dict[str, str]) -> None:
    commit(
        repo,
        {
            "pyproject.toml": '[project]\nname = "estel"\nversion = "0.3.0"\n',
            **files,
            "changes/7.patch.md": "Draft references.",
        },
    )
    result = impact(repo, "--pr", "7")
    assert result.returncode == 0, result.stderr
    assert "Before 1.0.0 this is only reported." in result.stdout


def test_counters_absent_on_base_count_as_zero(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ci@example.invalid")
    git(tmp_path, "config", "user.name", "CI")
    write(
        tmp_path,
        {
            "pyproject.toml": '[project]\nname = "estel"\nversion = "1.0.0"\n',
            "tests/reference/s2.npz": "v1",
            "changes/.gitkeep": "",
        },
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "checkout", "-q", "-b", "feature")
    commit(tmp_path, {COUNTERS_FILE: counters(0, 0), "changes/7.patch.md": "Add counters."})
    result = impact(tmp_path, "--pr", "7")
    assert result.returncode == 0, result.stderr
    assert "REFERENCE" not in result.stdout + result.stderr
