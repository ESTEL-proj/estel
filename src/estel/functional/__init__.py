"""Pure functions behind the operators (Part II, §12.6, §12.8; tasks T2.5, T4.1).

``estel.functional.scattering(data, plan, filters, ref=None)`` is the pure function behind
``STOperator.apply``, and ``estel.functional.spectrum(data, plan, filters, ref=None)`` the one
behind ``SpectrumOperator.apply``, public for users of ``jit``, ``grad`` and ``vmap`` and for
downstream implicit differentiation.
"""

from ._scattering import scattering
from ._spectrum import spectrum

__all__ = ["scattering", "spectrum"]
