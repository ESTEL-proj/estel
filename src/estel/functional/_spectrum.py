"""``estel.functional.spectrum``: the pure function behind ``SpectrumOperator.apply`` (Part II,
§12.8; maths.md §1.9, M-PS-2D; task T4.1).

Its type check, then the one path of every operator (``functional._scattering.compute``): the
values of the plan, here the estimator of the power spectrum (``SpectrumOperatorPlan._values``),
compiled where the backend compiles with the plan static, then the normalisation of
``STStatistics`` (maths.md M-NORM). It holds no state: with ``norm="self"`` the reference is
computed from the spectrum and returned with it, and the operator, not this function, keeps it.
"""

from __future__ import annotations

from ..data._base import Data
from ..scattering._filters import STFilters
from ..scattering._plan import STPlan
from ..spectra._plan import SpectrumOperatorFilters, SpectrumOperatorPlan
from ..statistics._statistics import STStatistics
from ._scattering import compute, reference_of


def spectrum(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    ref: STStatistics | None = None,
) -> STStatistics:
    """The power spectrum of ``data`` (maths.md §1.9, M-PS-2D), as ``STStatistics`` holding the
    family ``"PS"`` alone: the pure function behind ``SpectrumOperator.apply``.

    ``plan`` and ``filters`` are those of a ``SpectrumOperator``, ``op.plan`` and ``op.filters``;
    ``data`` has the geometry, spatial shape, ``pbc``, mask and number of channels the plan was
    built for, and the backend and dtype of the filters. The normalisation is the plan's
    (maths.md M-NORM): with ``"none"`` and ``"self"`` pass no ``ref``; with ``"ref"``, ``ref`` is
    required: statistics computed with ``norm="self"`` by a ``SpectrumOperator``, or by an
    ``STOperator`` whose spectrum has the same ``n_bins``, J and ``Jmin``. With ``"self"`` the
    reference is part of the computation and gradients flow through it; with ``"ref"`` it is a
    constant.
    """
    if not (
        isinstance(plan, SpectrumOperatorPlan) and isinstance(filters, SpectrumOperatorFilters)
    ):
        raise TypeError(
            f"expected the plan and the filters of a SpectrumOperator, op.plan and op.filters; "
            f"got {type(plan).__name__} and {type(filters).__name__}. For a scattering "
            "operator, use estel.functional.scattering."
        )
    return compute(data, plan, filters, reference_of(ref, plan))
