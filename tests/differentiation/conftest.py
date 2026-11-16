"""Differentiation tests run in float64 (Part I, §3.2.2); the tests that name a backend's own
checker run only in that backend's run.

``test_torch.py`` (``torch.autograd.gradcheck`` and ``gradgradcheck``) and ``test_jax.py``
(``jax.test_util.check_grads``) are collected only when ``ESTEL_TEST_BACKEND`` names their
backend, as in ``tests/backend/``; ``test_implicit.py`` only on PyTorch, since one backend is
enough for it (issue #19). ``test_scattering.py`` goes through the L0 primitives and runs on
both. A run restricted to float32 (``ESTEL_TEST_DTYPE=float32``, one of the two JAX jobs of the
full tier) collects none of them: the float64 run has them all.
"""

from __future__ import annotations

import os

import pytest

from estel import _config
from tests._backends import BACKENDS, backend_name

collect_ignore = [f"test_{name}.py" for name in sorted(BACKENDS) if name != backend_name()]
if backend_name() != "torch":
    collect_ignore.append("test_implicit.py")
if os.environ.get("ESTEL_TEST_DTYPE") == "float32":
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Data and operators in float64, whatever the dtypes of the run."""
    monkeypatch.setattr(_config, "_dtype", "float64")
