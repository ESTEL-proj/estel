"""tools/version_impact.py fails when the declared bump is below the required one."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from .conftest import run_tool, write

API = {"estel.f": {"kind": "function", "signature": "(x)"}}


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
            "tests/reference/ changed",
            id="reference-changed-without-major-label",
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


def test_reference_change_passes_with_major_label(repo: Path) -> None:
    commit(repo, {"tests/reference/s2.npz": "v2", "changes/7.patch.md": "New references."})
    result = impact(repo, "--pr", "7", "--labels", "major")
    assert result.returncode == 0, result.stderr


def test_reference_change_is_only_reported_before_1_0(repo: Path) -> None:
    commit(
        repo,
        {
            "pyproject.toml": '[project]\nname = "estel"\nversion = "0.3.0"\n',
            "tests/reference/s2.npz": "v2",
            "changes/7.patch.md": "Draft references.",
        },
    )
    result = impact(repo, "--pr", "7")
    assert result.returncode == 0, result.stderr
    assert "Before 1.0.0 reference values gate nothing" in result.stdout
