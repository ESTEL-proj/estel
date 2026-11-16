"""Invariance tests run in float64 (task T2.8, issue #20): the properties hold to rounding, and
they are compared with the float64 tolerances of Part II, §14.3.

A run restricted to float32 (``ESTEL_TEST_DTYPE=float32``, one of the two JAX jobs of the full
tier) collects none of them: the float64 run has them all.
"""

from __future__ import annotations

import os

import pytest

from estel import _config

if os.environ.get("ESTEL_TEST_DTYPE") == "float32":
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data and operators in float64, whatever the dtypes of the run."""
    monkeypatch.setattr(_config, "_dtype", "float64")
