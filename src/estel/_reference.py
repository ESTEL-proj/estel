# AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. Read docs/policy/VERSIONING.md.
"""The two reference counters of ESTEL (Part I, §7.1 and §7.2).

* ``STATISTICS_REFERENCE`` is bumped whenever any statistic value changes: within one value,
  statistics and test results are unchanged up to the tolerances of ``tests/tolerances.toml``.
* ``SYNTHESIS_REFERENCE`` is the same for the reference synthesis trajectories.

Both are integers, never reset, and become 1 at the 1.0.0 freeze. ``tools/version_impact.py``
reads them as integer literals on the base branch and on the PR: a PR that changes
``tests/reference/`` must increase ``STATISTICS_REFERENCE`` by exactly one, and a PR that
changes ``STATISTICS_REFERENCE`` must change ``tests/reference/``. Keep each value a plain
integer literal.
"""

STATISTICS_REFERENCE = 0
SYNTHESIS_REFERENCE = 0
