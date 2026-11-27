"""The import-linter contracts of pyproject.toml fail on crafted violations (Part I, §2.4)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .conftest import REPO, TOOLS, write


def lint_imports(tmp_path: Path, files: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run lint-imports with the repository's contracts on a fake ``estel`` package."""
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    write(tmp_path / "src", {"estel/__init__.py": "", **files})
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path / "src"), str(TOOLS)])}
    exe = shutil.which("lint-imports")
    assert exe is not None, "lint-imports is missing: run `uv sync`"
    return subprocess.run(
        [exe, "--config", str(tmp_path / "pyproject.toml"), "--no-cache"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )


def test_empty_package_keeps_every_contract(tmp_path: Path) -> None:
    result = lint_imports(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr


def test_clean_layered_package_keeps_every_contract(tmp_path: Path) -> None:
    result = lint_imports(
        tmp_path,
        {
            "estel/backend/__init__.py": "",
            "estel/backend/torch/__init__.py": "import torch\n",
            "estel/data/__init__.py": "",
            "estel/data/_base.py": "import estel.backend\n",
            "estel/data/_cartesian.py": "import estel.data._base\n",
            "estel/scattering/__init__.py": "import estel.data._base\n",
            "estel/_spec/__init__.py": "import math\nimport numpy\n",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("files", "broken"),
    [
        pytest.param(
            {"estel/data/__init__.py": "import torch\n"},
            "torch, jax and optax are imported only under estel.backend",
            id="torch-outside-backend",
        ),
        pytest.param(
            {"estel/scattering/__init__.py": "import jax.numpy\n"},
            "torch, jax and optax are imported only under estel.backend",
            id="jax-outside-backend",
        ),
        pytest.param(
            {"estel/data/__init__.py": "import estel_rust\n"},
            "estel_rust is imported only under estel.engines",
            id="rust-outside-engines",
        ),
        pytest.param(
            {
                "estel/backend/__init__.py": "",
                "estel/engines/__init__.py": "",
                "estel/engines/python.py": "import estel.backend\n",
            },
            "The engine layer never imports a backend directly",
            id="engine-imports-backend",
        ),
        pytest.param(
            {
                "estel/data/__init__.py": "",
                "estel/data/_cartesian.py": "",
                "estel/statistics/__init__.py": "",
                "estel/statistics/stats.py": "from estel.data import _cartesian\n",
            },
            "L2 talks to the Data mother class, never to a subclass",
            id="l2-imports-subclass",
        ),
        pytest.param(
            {
                "estel/backend/__init__.py": "import estel.data\n",
                "estel/data/__init__.py": "",
            },
            "Layer order: engines, then L2, then L1, then L0; nothing imports upwards",
            id="l0-imports-l1",
        ),
        pytest.param(
            {
                "estel/wavelets/__init__.py": "",
                "estel/wavelets/_banks_cartesian.py": "",
                "estel/scattering/__init__.py": "",
                "estel/scattering/_filters.py": "from estel.wavelets import _banks_cartesian\n",
            },
            "Outside L1, the banks and mask builders go through WaveletOperator only",
            id="l2-imports-the-banks",
        ),
        pytest.param(
            {
                "estel/data/__init__.py": "",
                "estel/data/_masks.py": "",
                "estel/statistics/__init__.py": "",
                "estel/statistics/_statistics.py": "import estel.data._masks\n",
            },
            "Outside L1, the banks and mask builders go through WaveletOperator only",
            id="l2-imports-the-mask-builders",
        ),
        pytest.param(
            {
                "estel/data/__init__.py": "import estel.wavelets\n",
                "estel/wavelets/__init__.py": "",
            },
            "Layer order: engines, then L2, then L1, then L0; nothing imports upwards",
            id="data-imports-wavelets",
        ),
        pytest.param(
            {"estel/_spec/__init__.py": "import scipy\n"},
            "estel._spec imports NumPy only, and nothing from estel",
            id="oracle-imports-scipy",
        ),
        pytest.param(
            {
                "estel/_spec/__init__.py": "",
                "estel/_spec/wav.py": "from estel import data\n",
                "estel/data/__init__.py": "",
            },
            "estel._spec imports NumPy only, and nothing from estel",
            id="oracle-imports-estel",
        ),
        pytest.param(
            {
                "estel/scattering/__init__.py": "",
                "estel/scattering/_plan.py": "",
                "estel/spectra/__init__.py": "",
                "estel/spectra/_estimators.py": "from estel.scattering import _plan\n",
            },
            "estel.spectra imports nothing from estel.scattering (T7.9)",
            id="spectra-imports-scattering",
        ),
    ],
)
def test_crafted_violation_breaks_its_contract(
    tmp_path: Path, files: dict[str, str], broken: str
) -> None:
    result = lint_imports(tmp_path, files)
    assert result.returncode != 0
    assert f"{broken} BROKEN" in result.stdout, result.stdout
