"""Consistency tests run on PyTorch, in float64 (task T2.16, issue #98, decision 7): they compare
estimates of the same statistics with their statistical error, which neither the backend nor
the dtype changes.

A run on the experimental JAX backend, or restricted to float32 (``ESTEL_TEST_DTYPE=float32``),
collects none of them: the float64 run on PyTorch has them all.
"""

from __future__ import annotations

import os

import pytest

from estel import _config
from tests._backends import backend_name

if backend_name() != "torch" or os.environ.get("ESTEL_TEST_DTYPE") == "float32":
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data and operators in float64, whatever the dtypes of the run."""
    monkeypatch.setattr(_config, "_dtype", "float64")
