"""Fixtures of the conformance suite: every backend that exists, checked the same way.

Adding a backend means adding it to ``BACKENDS`` and passing the suite (Part II, §12.3;
``src/estel/backend/AGENTS.md``).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from estel.backend._protocol import Backend
from estel.backend.torch._primitives import TorchBackend

BACKENDS: dict[str, Callable[[], Backend]] = {"torch": TorchBackend}


@pytest.fixture(params=sorted(BACKENDS))
def backend(request: pytest.FixtureRequest) -> Backend:
    """One instance of each backend of this step."""
    return BACKENDS[request.param]()
