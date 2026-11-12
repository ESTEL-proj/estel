"""Shared test configuration.

``make check`` runs the suite on every backend (``ESTEL_TEST_BACKEND``, see ``tests/_backends.py``)
and, for each, twice: with ``ESTEL_TEST_DTYPE=float32`` and ``float64`` (Part I, §3.1, rule 7).
Numerical tests take the ``dtype`` and ``backend`` fixtures.
"""

from __future__ import annotations

import os

import pytest

from estel.backend._protocol import Backend
from tests._backends import backend_name, make_backend

DTYPES = ("float32", "float64")


def pytest_configure(config: pytest.Config) -> None:
    """Set up the backend of the run, before any test creates an array."""
    try:
        name = backend_name()
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if name == "jax":
        # JAX keeps float64 only in its process-wide 64-bit mode, which the backend never
        # switches itself (issue #10); the JAX run switches it on at start, for both dtypes.
        # Every primitive takes an explicit dtype, so float32 runs stay in float32.
        import jax

        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


@pytest.fixture
def dtype() -> str:
    """The floating dtype of this run: float32 (default) or float64."""
    value = os.environ.get("ESTEL_TEST_DTYPE", "float32")
    if value not in DTYPES:
        raise pytest.UsageError(f"ESTEL_TEST_DTYPE={value!r}: set it to one of {DTYPES}.")
    return value


@pytest.fixture
def backend() -> Backend:
    """The backend of this run (``ESTEL_TEST_BACKEND``)."""
    return make_backend(backend_name())
