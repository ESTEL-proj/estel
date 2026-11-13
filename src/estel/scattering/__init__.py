"""Scattering: the static plan and, later, the operators (Part II, §12.6).

``_plan.py`` holds the ``ScatteringPlan``, the static description of a scattering transform,
built once in NumPy when an operator is built (tag N; Part I, §2.3, §3.1.1). ``_loop.py`` holds
the scattering loop, the pure function that computes the statistics, and ``_operator.py`` the
``STOperator``, internal until T2.5 makes it public with ``STStatistics`` (Part II, §15.2).
"""
