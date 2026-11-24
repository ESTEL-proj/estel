"""``estel.functional.spectrum``: the pure function behind ``SpectrumOperator.apply`` (Part II,
§12.8; maths.md §1.9; task T4.1).

It takes the data, the static plan and the filters of a ``SpectrumOperator`` and an optional
reference, and returns ``STStatistics`` holding the family ``"PS"`` alone: the estimator of the
plan (``spectra._estimators``), then the normalisation of the plan (maths.md M-NORM), the one
path after the loop of the scattering statistics (``statistics._statistics.finish``). It holds no
state: with ``norm="self"`` the reference is computed from the spectrum and returned with it,
and the operator, not this function, keeps it.

The estimator is called through ``Backend.jit`` with the plan static, as the scattering loop
(``functional._scattering``): one compiled program per plan, shape and dtype on JAX, the
identity on PyTorch.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from ..backend import _registry
from ..data._base import Data
from ..scattering._filters import STFilters
from ..scattering._plan import STPlan
from ..spectra._estimators import spectrum_floats
from ..spectra._plan import SpectrumOperatorFilters, SpectrumOperatorPlan
from ..statistics._layout import spectrum_pairs
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics, finish
from ._scattering import check_data, reference_of


@functools.cache
def _estimate(backend_name: str) -> Callable[..., Any]:
    """The estimator of a backend, compiled where it compiles (``Backend.jit``, the plan
    static), made once per backend so that its compiled programs are reused."""
    backend = _registry.backend_named(backend_name)

    def estimate(data: Data, plan: SpectrumOperatorPlan, filters: SpectrumOperatorFilters) -> Any:
        return spectrum_floats(
            backend,
            data.array,
            plan.spectrum.method,
            filters.spectrum,
            spectrum_pairs(plan._layout()),
        )

    return backend.jit(estimate, static_argnames=("plan",))


def spectrum(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    ref: STStatistics | None = None,
) -> STStatistics:
    """The power spectrum of ``data`` (maths.md §1.9), as ``STStatistics`` holding the family
    ``"PS"`` alone: the pure function behind ``SpectrumOperator.apply``.

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
            f"got {type(plan).__name__} and {type(filters).__name__}. For an STOperator, use "
            "estel.functional.scattering."
        )
    return compute_spectrum(data, plan, filters, reference_of(ref, plan))  # type: ignore[arg-type]


def compute_spectrum(
    data: Data,
    plan: SpectrumOperatorPlan,
    filters: SpectrumOperatorFilters,
    reference: _NormReference | None,
) -> STStatistics:
    """:func:`spectrum` with the reference already extracted: the data and a ``norm="ref"``
    reference checked, the estimator, then :func:`statistics._statistics.finish`."""
    check_data(data, plan, filters)  # type: ignore[arg-type]
    if plan.norm == "ref":
        if reference is None:
            raise ValueError(
                'norm="ref" needs a reference: pass ref=, statistics computed with '
                'norm="self" (maths.md M-NORM).'
            )
        plan._check_reference(reference, int(data.shape[0]), filters._smoothing.dtype)
    floats = _estimate(data.backend)(data, plan=plan, filters=filters)
    return finish(floats, plan, reference, None)  # type: ignore[arg-type]
