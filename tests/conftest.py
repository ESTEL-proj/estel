"""Shared test configuration.

``make check`` runs the suite twice, with ``ESTEL_TEST_DTYPE=float32`` and ``float64``
(Part I, §3.1, rule 7). Numerical tests take the ``dtype`` fixture.
"""

from __future__ import annotations

import os

import pytest

DTYPES = ("float32", "float64")


@pytest.fixture
def dtype() -> str:
    """The floating dtype of this run: float32 (default) or float64."""
    value = os.environ.get("ESTEL_TEST_DTYPE", "float32")
    if value not in DTYPES:
        raise pytest.UsageError(f"ESTEL_TEST_DTYPE={value!r}: set it to one of {DTYPES}.")
    return value
