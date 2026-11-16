"""``estel.functional.scattering``: the pure function behind ``STOperator.apply`` (Part II,
§12.6; Part I, §2.3; task T2.5).

It takes the data, the static plan, the filters and an optional reference, and returns the
statistics: the scattering loop (``estel.scattering._loop``), then the normalisation of the plan
(maths.md M-NORM; ``estel.statistics._normalization``). It holds no state and modifies
nothing: with ``norm="self"`` the reference is computed from the statistics and returned
with them, and the operator, not this function, keeps it.
"""

from __future__ import annotations

import numpy as np

from ..backend import _registry
from ..data._base import Data
from ..scattering._filters import STFilters
from ..scattering._loop import scattering_values
from ..scattering._plan import STPlan
from ..statistics._grid import auto_scales, base_grids
from ..statistics._normalization import (
    _NormReference,
    as_constant,
    check_reference,
    normalize,
    self_reference,
)
from ..statistics._statistics import STStatistics


def scattering(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    ref: STStatistics | None = None,
) -> STStatistics:
    """The scattering statistics of ``data`` (maths.md §1.6 to 1.8): the pure function behind
    ``STOperator.apply``.

    ``plan`` and ``filters`` are those of an operator, ``op.plan`` and ``op.filters``; ``data`` has
    the geometry, spatial shape, ``pbc``, mask and number of channels the plan was built for, and
    the backend and dtype of the filters. The normalisation is the plan's: with ``"none"``
    and ``"self"`` pass no ``ref``; with ``"ref"``, ``ref`` is required: statistics
    computed with ``norm="self"``, whose reference normalises these (M-NORM). With
    ``"self"`` the reference is part of the computation and gradients flow through it;
    with ``"ref"`` it is a constant.
    """
    return compute(data, plan, filters, reference_of(ref, plan))


def reference_of(ref: object, plan: STPlan) -> _NormReference | None:
    """The reference held by ``ref``, checked against the normalisation of ``plan``; its J,
    L and channels are checked against the plan by :func:`compute`, before the loop."""
    if ref is None:
        if plan.norm == "ref":
            raise ValueError(
                'norm="ref" needs a reference: pass ref=, statistics computed with '
                'norm="self" (maths.md M-NORM).'
            )
        return None
    if plan.norm != "ref":
        raise ValueError(
            f'norm="{plan.norm}" takes no reference: pass ref only with norm="ref" '
            "(maths.md M-NORM)."
        )
    if not isinstance(ref, STStatistics):
        raise TypeError(
            f'ref: pass STStatistics computed with norm="self", got {type(ref).__name__}.'
        )
    if ref._plan.norm != "self" or ref._reference is None:
        raise ValueError(
            f'ref: these statistics were computed with norm="{ref._plan.norm}"; the reference '
            'comes from statistics computed with norm="self" (maths.md M-NORM).'
        )
    return ref._reference


def check_data(data: object, plan: STPlan, filters: STFilters) -> None:
    """Raise unless ``data`` is data the plan was built for, on the filters' backend and dtype."""
    if not isinstance(data, Data):
        raise TypeError(f"expected a Data object, such as estel.Data2D, got {type(data).__name__}.")
    backend = _registry.backend_of(filters.smoothing) or ""
    _registry.check_backend(backend, data.array, owner="these filters")
    if data.mask is None or plan.mask is None:
        same_mask = data.mask is None and plan.mask is None
    else:
        same_mask = bool(np.array_equal(data.mask, plan.mask))
    if not (
        type(data) is plan.geometry
        and tuple(data.shape[2:]) == plan.spatial_shape
        and data.pbc == plan.pbc
        and same_mask
    ):
        raise ValueError(
            f"the data ({type(data).__name__}, spatial shape {tuple(data.shape[2:])}, "
            f"pbc={data.pbc}) differ from those the plan was built for "
            f"({plan.geometry.__name__}, spatial shape {plan.spatial_shape}, pbc={plan.pbc}, and "
            "the same mask of missing values). The plan is static: build a new STOperator for "
            "these data."
        )
    if data.n_channels != plan.n_channels:
        raise ValueError(
            f"the data have {data.n_channels} channel(s), but the operator was built for "
            f"{plan.n_channels}: the number of channels is fixed at construction. Build a new "
            "STOperator for these data."
        )
    if data.array.dtype != filters.smoothing.dtype:
        raise TypeError(
            f"the data are {data.array.dtype}, but the operator computes in "
            f"{filters.smoothing.dtype}, the default dtype when it was built. Build the data and "
            "the operator with the same default dtype (estel.set_default_dtype)."
        )


def compute(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    reference: _NormReference | None,
) -> STStatistics:
    """:func:`scattering` with the reference already extracted: the loop, then the
    normalisation of the plan."""
    check_data(data, plan, filters)
    if plan.norm == "ref":
        if reference is None:
            raise ValueError(
                'norm="ref" needs a reference: pass ref=, statistics computed with '
                'norm="self" (maths.md M-NORM).'
            )
        check_reference(reference, plan, int(data.shape[0]), filters.smoothing.dtype)
    backend = _registry.backend_named(data.backend)
    found = scattering_values(data, plan, filters)
    values = {name: getattr(found, name) for name in ("mean", "var", "S1", "S2", "S3", "S4")}
    grids = base_grids(plan)
    if plan.norm == "self":
        reference = self_reference(backend, values, auto_scales(plan), plan.ref_batch_mean)
    elif plan.norm == "ref" and reference is not None:  # checked before the loop
        reference = as_constant(backend, reference)
    else:
        reference = None
    if reference is not None:
        values, grids = normalize(backend, values, grids, reference)
    standardization = (found.mean_pre_std, found.std_pre_std) if plan.standardize else None
    return STStatistics._create(
        values,
        grids,
        plan,
        data.backend,
        reference=reference,
        standardization=standardization,
    )
