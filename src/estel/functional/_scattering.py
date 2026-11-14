"""``estel.functional.scattering``: the pure function behind ``STOperator.apply`` (Part II,
§12.6; Part I, §2.3; task T2.5).

It takes the data, the static plan, the filters and an optional reference, and returns the
statistics: the scattering loop (``estel.scattering._loop``), then the normalisation of the plan
(maths.md M-NORM; ``estel.statistics._normalization``). It holds no state and modifies
nothing: with ``norm="store_ref"`` the reference is computed from the statistics and returned
with them, and the operator, not this function, keeps it.
"""

from __future__ import annotations

import numpy as np

from ..backend import _registry
from ..data._base import DataClass
from ..scattering._filters import ScatteringFilters
from ..scattering._loop import scattering_values
from ..scattering._plan import ScatteringPlan
from ..statistics._grid import auto_scales, base_grids
from ..statistics._normalization import (
    _NormReference,
    as_constant,
    check_compatible,
    normalize,
    store_reference,
)
from ..statistics._statistics import STStatistics


def scattering(
    x: DataClass,
    plan: ScatteringPlan,
    filters: ScatteringFilters,
    ref: STStatistics | None = None,
) -> STStatistics:
    """The scattering statistics of ``x`` (maths.md §1.6 to 1.8): the pure function behind
    ``STOperator.apply``.

    ``plan`` and ``filters`` are those of an operator, ``op.plan`` and ``op.filters``; ``x`` has
    the geometry, spatial shape, ``pbc``, mask and number of channels the plan was built for, and
    the backend and dtype of the filters. The normalisation is the plan's: with ``"vanilla"``
    and ``"store_ref"`` pass no ``ref``; with ``"load_ref"``, ``ref`` is required: statistics
    computed with ``norm="store_ref"``, whose reference normalises these (M-NORM). With
    ``"store_ref"`` the reference is part of the computation and gradients flow through it;
    with ``"load_ref"`` it is a constant.
    """
    return compute(x, plan, filters, reference_of(ref, plan))


def reference_of(ref: object, plan: ScatteringPlan) -> _NormReference | None:
    """The reference held by ``ref``, checked against the normalisation of ``plan``."""
    if ref is None:
        if plan.norm == "load_ref":
            raise ValueError(
                'norm="load_ref" needs a reference: pass ref=, statistics computed with '
                'norm="store_ref" (maths.md M-NORM).'
            )
        return None
    if plan.norm != "load_ref":
        raise ValueError(
            f'norm="{plan.norm}" takes no reference: pass ref only with norm="load_ref" '
            "(maths.md M-NORM)."
        )
    if not isinstance(ref, STStatistics):
        raise TypeError(
            f'ref: pass STStatistics computed with norm="store_ref", got {type(ref).__name__}.'
        )
    if ref._plan.norm != "store_ref" or ref._reference is None:
        raise ValueError(
            f'ref: these statistics were computed with norm="{ref._plan.norm}"; the reference '
            'comes from statistics computed with norm="store_ref" (maths.md M-NORM).'
        )
    return ref._reference


def check_data(x: object, plan: ScatteringPlan, filters: ScatteringFilters) -> None:
    """Raise unless ``x`` is data the plan was built for, on the filters' backend and dtype."""
    if not isinstance(x, DataClass):
        raise TypeError(f"expected a DataClass, such as estel.Data2D, got {type(x).__name__}.")
    backend = _registry.backend_of(filters.smoothing) or ""
    _registry.check_backend(backend, x.array, owner="these filters")
    if x.mask is None or plan.mask is None:
        same_mask = x.mask is None and plan.mask is None
    else:
        same_mask = bool(np.array_equal(x.mask, plan.mask))
    if not (
        type(x) is plan.geometry
        and tuple(x.shape[2:]) == plan.spatial_shape
        and x.pbc == plan.pbc
        and same_mask
    ):
        raise ValueError(
            f"the data ({type(x).__name__}, spatial shape {tuple(x.shape[2:])}, pbc={x.pbc}) "
            f"differ from those the plan was built for ({plan.geometry.__name__}, spatial shape "
            f"{plan.spatial_shape}, pbc={plan.pbc}, and the same mask of missing values). The "
            "plan is static: build a new STOperator for these data."
        )
    if x.n_channels != plan.n_channels:
        raise ValueError(
            f"the data have {x.n_channels} channel(s), but the operator was built for "
            f"{plan.n_channels}: the number of channels is fixed at construction. Build a new "
            "STOperator for these data."
        )
    if x.array.dtype != filters.smoothing.dtype:
        raise TypeError(
            f"the data are {x.array.dtype}, but the operator computes in "
            f"{filters.smoothing.dtype}, the default dtype when it was built. Build the data and "
            "the operator with the same default dtype (estel.set_default_dtype)."
        )


def compute(
    x: DataClass,
    plan: ScatteringPlan,
    filters: ScatteringFilters,
    reference: _NormReference | None,
) -> STStatistics:
    """:func:`scattering` with the reference already extracted: the loop, then the
    normalisation of the plan."""
    check_data(x, plan, filters)
    backend = _registry.backend_named(x.backend)
    found = scattering_values(x, plan, filters)
    values = {name: getattr(found, name) for name in ("mean", "var", "S1", "S2", "S3", "S4")}
    grids = base_grids(plan)
    if plan.norm == "store_ref":
        reference = store_reference(backend, values, auto_scales(plan), plan.norm_batch_mean)
    elif plan.norm == "load_ref":
        if reference is None:
            raise ValueError(
                'norm="load_ref" needs a reference: pass ref=, statistics computed with '
                'norm="store_ref" (maths.md M-NORM).'
            )
        check_compatible(reference, values)
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
        x.backend,
        reference=reference,
        standardization=standardization,
    )
