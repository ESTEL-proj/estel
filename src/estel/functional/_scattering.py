"""``estel.functional.scattering``: the pure function behind ``STOperator.apply`` (Part II,
§12.6; Part I, §2.3; task T2.5).

It takes the data, the static plan, the filters and an optional reference, and returns the
statistics: the scattering loop (``estel.scattering._loop``), then the normalisation of the plan
(maths.md M-NORM; ``estel.statistics._normalization``). It holds no state and modifies
nothing: with ``norm="self"`` the reference is computed from the statistics and returned
with them, and the operator, not this function, keeps it.

**Compiled where the backend compiles** (Part I, §2.2.2; task T2.13). The loop is called
through ``Backend.jit`` with the plan as its static argument: on JAX the whole loop is one
compiled program per plan and per shape and dtype of the data, built at the first call and
reused by the later ones; the data and the filters are its arguments, never constants of the
program (Part I, §3.1, rule 4). On PyTorch ``jit`` is the identity. Derivatives and other
transformations pass through the compiled loop unchanged.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from ..backend import _registry
from ..data._base import Data
from ..scattering._filters import PlanarFilters, STFilters, TensorProdFilters
from ..scattering._loop import ScatteringValues, scattering_values
from ..scattering._loop_tp import tensor_prod_values
from ..scattering._plan import PlanarPlan, STPlan, TensorProdPlan
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics, check_reference_of, statistics_of


@functools.cache
def _loop(backend_name: str) -> Callable[..., ScatteringValues]:
    """The scattering loop of a backend, compiled where it compiles (``Backend.jit``, the
    plan static): one function per backend, made once, so that its compiled programs are
    reused from one call to the next. The compiled programs belong to the function made here:
    ``_loop.cache_clear()`` drops them, for a test that replaces a hook of the loop."""
    backend = _registry.backend_named(backend_name)

    def loop(data: Data, plan: PlanarPlan, filters: PlanarFilters) -> ScatteringValues:
        return scattering_values(data, plan, filters)

    return backend.jit(loop, static_argnames=("plan",))


@functools.cache
def _tensor_prod_loop(backend_name: str) -> Callable[..., ScatteringValues]:
    """The scattering loop of tensor products of a backend, compiled as :func:`_loop`."""
    backend = _registry.backend_named(backend_name)

    def loop(data: Data, plan: TensorProdPlan, filters: TensorProdFilters) -> ScatteringValues:
        return tensor_prod_values(data, plan, filters)

    return backend.jit(loop, static_argnames=("plan",))


def scattering(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    ref: STStatistics | None = None,
) -> STStatistics:
    """The scattering statistics of ``data`` (maths.md §1.6 to 1.8, §3): the pure function
    behind ``STOperator.apply`` and ``STOperatorTensorProd.apply``, one function for both.

    ``plan`` and ``filters`` are those of an operator, ``op.plan`` and ``op.filters``; ``data`` has
    the geometry, spatial shape, ``pbc``, mask and number of channels the plan was built for
    (for a tensor product, its blocks, their sizes and their ``pbc``), and the backend and dtype
    of the filters. ``STPlan`` and ``STFilters`` are the types of the plan and filters of
    either operator: those of a tensor product are their subclasses (task T3.2). The
    normalisation is the plan's: with ``"none"``
    and ``"self"`` pass no ``ref``; with ``"ref"``, ``ref`` is required: statistics
    computed with ``norm="self"``, whose reference normalises these (M-NORM). With
    ``"self"`` the reference is part of the computation and gradients flow through it;
    with ``"ref"`` it is a constant.
    """
    found, _ = _parts(plan, filters)
    return compute(data, plan, filters, reference_of(ref, found))


def _parts(
    plan: STPlan, filters: STFilters
) -> tuple[PlanarPlan | TensorProdPlan, PlanarFilters | TensorProdFilters]:
    """The plan and the filters as the subclasses of one geometry, or a clear error."""
    if not (
        isinstance(plan, PlanarPlan | TensorProdPlan)
        and isinstance(filters, PlanarFilters | TensorProdFilters)
    ):
        raise TypeError(
            f"expected the plan and the filters of an operator, op.plan and op.filters; got "
            f"{type(plan).__name__} and {type(filters).__name__}."
        )
    return plan, filters


def reference_of(ref: object, plan: PlanarPlan | TensorProdPlan) -> _NormReference | None:
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


def check_data(
    data: object, plan: PlanarPlan | TensorProdPlan, filters: PlanarFilters | TensorProdFilters
) -> None:
    """Raise unless ``data`` is data the plan was built for, on the filters' backend and dtype."""
    if not isinstance(data, Data):
        raise TypeError(f"expected a Data object, such as estel.Data2D, got {type(data).__name__}.")
    if isinstance(plan, TensorProdPlan) != isinstance(filters, TensorProdFilters):
        raise TypeError(
            "the plan and the filters come from different operators: pass op.plan and "
            "op.filters of one operator."
        )
    backend = _registry.backend_of(_smoothing(filters)) or ""
    _registry.check_backend(backend, data.array, owner="these filters")
    if isinstance(plan, TensorProdPlan):
        _check_tensor_prod_data(data, plan)
    else:
        _check_planar_data(data, plan)
    if data.n_channels != plan.n_channels:
        name = "STOperatorTensorProd" if isinstance(plan, TensorProdPlan) else "STOperator"
        raise ValueError(
            f"the data have {data.n_channels} channel(s), but the operator was built for "
            f"{plan.n_channels}: the number of channels is fixed at construction. Build a new "
            f"{name} for these data."
        )
    if data.array.dtype != _dtype(filters):
        raise TypeError(
            f"the data are {data.array.dtype}, but the operator computes in "
            f"{_dtype(filters)}, the default dtype when it was built. Build the "
            "data and the operator with the same default dtype (estel.set_default_dtype)."
        )


def _check_tensor_prod_data(data: Data, plan: TensorProdPlan) -> None:
    blocks = tuple(getattr(data, "blocks", ()))
    if not (
        type(data) is plan.geometry
        and blocks == plan.block_geometries
        and tuple(data.shape[2:]) == plan.spatial_shape
        and tuple(data.pbc) == plan.pbc  # type: ignore[arg-type]
    ):
        names = ", ".join(g.__name__ for g in plan.block_geometries)
        raise ValueError(
            f"the data ({type(data).__name__}, spatial shape {tuple(data.shape[2:])}, "
            f"pbc={data.pbc}) differ from those the plan was built for (DataTensorProd with the "
            f"blocks [{names}], spatial shape {plan.spatial_shape}, pbc={list(plan.pbc)}). The "
            "plan is static: build a new STOperatorTensorProd for these data."
        )


def _check_planar_data(data: Data, plan: PlanarPlan) -> None:
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


def compute(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    reference: _NormReference | None,
) -> STStatistics:
    """:func:`scattering` with the reference already extracted: the loop of the geometry of
    the plan, a subclass of ``STPlan``, then the one path after the loop (``statistics_of``). A
    ``norm="ref"`` reference is checked before the loop, on planar data and on tensor products
    alike."""
    plan, filters = _parts(plan, filters)
    check_data(data, plan, filters)
    if plan.norm == "ref":
        if reference is None:
            raise ValueError(
                'norm="ref" needs a reference: pass ref=, statistics computed with '
                'norm="self" (maths.md M-NORM).'
            )
        check_reference_of(reference, plan, int(data.shape[0]), _dtype(filters))
    backend = _registry.backend_named(data.backend)
    if isinstance(plan, TensorProdPlan):
        found = _tensor_prod_loop(backend.name)(data, plan=plan, filters=filters)
    else:
        found = _loop(backend.name)(data, plan=plan, filters=filters)
    standardization = (found.mean_pre_std, found.std_pre_std) if plan.standardize else None
    return statistics_of(found.buffer, plan, reference, standardization)


def _smoothing(filters: PlanarFilters | TensorProdFilters) -> Any:
    """An array of the filters, for their backend and dtype: the smoothing kernel (of the
    first block of a tensor product)."""
    if isinstance(filters, TensorProdFilters):
        return filters.wavelets[0].smoothing
    return filters.wavelets.smoothing


def _dtype(filters: PlanarFilters | TensorProdFilters) -> Any:
    return _smoothing(filters).dtype
