"""Consistency tests run on PyTorch, in float64 (task T2.16, issue #98, second start, decision
2): they compare estimates of the same statistics with their statistical error, which neither
the backend nor the dtype changes.

A run on the experimental JAX backend, or restricted to float32 (``ESTEL_TEST_DTYPE=float32``),
collects none of them: the float64 run on PyTorch has them all.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from estel import _config
from tests._backends import backend_name

if backend_name() != "torch" or os.environ.get("ESTEL_TEST_DTYPE") == "float32":
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data and operators in float64, whatever the dtypes of the run."""
    monkeypatch.setattr(_config, "_dtype", "float64")


@pytest.fixture(autouse=True)
def _one_thread() -> Iterator[None]:
    """PyTorch on one thread during each test, then the previous number again: the statistics
    of 64 fields gain nothing from more threads, which only add CPU time (30 s against 18 s
    measured in one process), and the budget of these tests is in CPU time. PyTorch is imported
    here: a JAX run, whose environment may have no PyTorch, collects no test of this directory."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
