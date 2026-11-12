"""tools/api_manifest.py fails when the public API and api/manifest.json differ."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import REPO, run_tool, write

EMPTY = json.dumps({"schema": 1, "public": {}})

PACKAGE = {
    "src/estel/__init__.py": "from ._config import set_backend\nfrom . import data\n",
    "src/estel/_config.py": "def set_backend(name: str) -> None:\n    pass\n",
    "src/estel/data/__init__.py": "from ._base import DataClass as DataClass\n",
    "src/estel/data/_base.py": (
        "import numpy as np\n"
        "class DataClass:\n"
        "    def __init__(self, array, *, pbc=True):\n        pass\n"
        "    def with_array(self, array):\n        pass\n"
        "    def _hook(self):\n        pass\n"
        "    @property\n    def shape(self):\n        pass\n"
    ),
    "src/estel/spectra/_estimators.py": "def fft_estimator(x):\n    pass\n",
}

MANIFEST = {
    "estel.data.DataClass": {"kind": "class", "signature": "(self, array, *, pbc=True)"},
    "estel.data.DataClass.shape": {"kind": "property"},
    "estel.data.DataClass.with_array": {"kind": "method", "signature": "(self, array)"},
    "estel.set_backend": {"kind": "function", "signature": "(name: str) -> None"},
}


def make_repo(tmp_path: Path, manifest: dict[str, object], extra: dict[str, str]) -> Path:
    files = {**PACKAGE, **extra}
    files["api/manifest.json"] = json.dumps({"schema": 1, "public": manifest})
    write(tmp_path, files)
    return tmp_path


def test_repository_manifest_matches_the_code() -> None:
    result = run_tool("api_manifest.py", "--check", "--root", str(REPO))
    assert result.returncode == 0, result.stderr


def test_matching_manifest_passes(tmp_path: Path) -> None:
    root = make_repo(tmp_path, MANIFEST, {})
    result = run_tool("api_manifest.py", "--check", "--root", str(root))
    assert result.returncode == 0, result.stderr


def test_print_reproduces_the_manifest(tmp_path: Path) -> None:
    root = make_repo(tmp_path, MANIFEST, {})
    result = run_tool("api_manifest.py", "--print", "--root", str(root))
    assert json.loads(result.stdout) == {"schema": 1, "public": MANIFEST}


@pytest.mark.parametrize(
    ("extra", "manifest", "message"),
    [
        pytest.param(
            {"src/estel/data/planar.py": "def helper(x):\n    pass\n"},
            MANIFEST,
            "estel.data.planar.helper is public but not in api/manifest.json",
            id="public-function-in-public-module",
        ),
        pytest.param(
            {"src/estel/data/planar.py": "class Data2D:\n    def crop(self):\n        pass\n"},
            MANIFEST,
            "estel.data.planar.Data2D.crop is public but not in api/manifest.json",
            id="public-method-of-public-class",
        ),
        pytest.param(
            {"src/estel/__init__.py": "from ._config import set_backend\nSTATISTICS_REFERENCE = 0\n"},
            MANIFEST,
            "estel.STATISTICS_REFERENCE is public but not in api/manifest.json",
            id="public-constant",
        ),
        pytest.param(
            {"src/estel/__init__.py": "from .spectra._estimators import fft_estimator\n"
             "from ._config import set_backend\n"},
            MANIFEST,
            "estel.fft_estimator is public but not in api/manifest.json",
            id="private-function-re-exported",
        ),
        pytest.param(
            {},
            {**MANIFEST, "estel.get_backend": {"kind": "function", "signature": "()"}},
            "estel.get_backend is in api/manifest.json but not in the code",
            id="removed-name",
        ),
        pytest.param(
            {"src/estel/_config.py": "def set_backend(name: str, strict: bool) -> None:\n"
             "    pass\n"},
            MANIFEST,
            "estel.set_backend changed",
            id="changed-signature",
        ),
    ],
)
def test_crafted_violation_fails(
    tmp_path: Path, extra: dict[str, str], manifest: dict[str, object], message: str
) -> None:
    root = make_repo(tmp_path, manifest, extra)
    result = run_tool("api_manifest.py", "--check", "--root", str(root))
    assert result.returncode == 1
    assert message in result.stderr


def test_private_names_are_not_api(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        MANIFEST,
        {
            "src/estel/data/planar.py": "def _helper(x):\n    pass\n_TABLE = 1\n",
            "src/estel/data/_masks.py": "def build_masks(x):\n    pass\n",
        },
    )
    result = run_tool("api_manifest.py", "--check", "--root", str(root))
    assert result.returncode == 0, result.stderr


def test_malformed_manifest_says_what_to_do(tmp_path: Path) -> None:
    write(tmp_path, {"src/estel/__init__.py": "", "api/manifest.json": "[]"})
    result = run_tool("api_manifest.py", "--check", "--root", str(tmp_path))
    assert result.returncode != 0
    assert "Restore the file from main" in result.stderr


def test_empty_package_and_empty_manifest_pass(tmp_path: Path) -> None:
    write(tmp_path, {"src/estel/__init__.py": "", "api/manifest.json": EMPTY})
    result = run_tool("api_manifest.py", "--check", "--root", str(tmp_path))
    assert result.returncode == 0, result.stderr
