"""The backends the test suite runs on (Part I, §3.1 and §6.1).

Each pytest run has one backend, as each tier-0 job does: ``ESTEL_TEST_BACKEND`` names it
(``torch`` by default); both dtypes run in the same process (``tests/conftest.py``). Adding a
backend means adding it to
``BACKENDS`` and passing the conformance suite (``src/estel/backend/AGENTS.md``).
"""

from __future__ import annotations

import importlib
import os

from estel.backend._protocol import Backend

# Name -> "module:class", imported only by the run that uses it.
BACKENDS = {
    "torch": "estel.backend.torch._primitives:TorchBackend",
    "jax": "estel.backend.jax._primitives:JaxBackend",
}


def backend_name() -> str:
    """The backend of this run, from ``ESTEL_TEST_BACKEND``."""
    value = os.environ.get("ESTEL_TEST_BACKEND", "torch")
    if value not in BACKENDS:
        raise ValueError(f"ESTEL_TEST_BACKEND={value!r}: set it to one of {sorted(BACKENDS)}.")
    return value


def make_backend(name: str) -> Backend:
    module, _, cls = BACKENDS[name].partition(":")
    backend: Backend = getattr(importlib.import_module(module), cls)()
    return backend
