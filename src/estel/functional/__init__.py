"""Pure functions behind the operators (Part II, §12.6; task T2.5).

``estel.functional.scattering(x, plan, filters, ref=None)`` is the pure function behind
``STOperator.apply``, public for users of ``jit``, ``grad`` and ``vmap`` and for downstream
implicit differentiation.
"""

from ._scattering import scattering

__all__ = ["scattering"]
