"""Backend-specific tests run only in their own backend's run.

``test_torch.py`` and ``test_jax.py`` pin what the conformance suite cannot say without naming
a backend; each is collected only when ``ESTEL_TEST_BACKEND`` names its backend, so each tier-0
job runs its own (issue #10). The conformance suite runs in every job.
"""

from __future__ import annotations

from tests._backends import BACKENDS, backend_name

collect_ignore = [f"test_{name}.py" for name in sorted(BACKENDS) if name != backend_name()]
