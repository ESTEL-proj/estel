"""The backends the test suite runs on (Part I, §3.1 and §6.1).

Tier 0 runs the whole suite once per backend, in parallel jobs: ``ESTEL_TEST_BACKEND`` names
the backend of a run (``torch`` by default), as ``ESTEL_TEST_DTYPE`` names its dtype; ``make
check`` runs every backend in float32 and float64. Adding a backend means adding it to
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
