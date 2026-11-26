"""Pure functions behind the operators (Part II, §12.6; tasks T2.5, T4.6).

``estel.functional.scattering(data, plan, filters, ref=None)`` is the pure function behind
``STOperator.apply``, the power spectrum included, public for users of ``jit``, ``grad`` and
``vmap`` and for downstream implicit differentiation.
"""

from ._scattering import scattering

__all__ = ["scattering"]
