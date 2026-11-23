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

from ..backend import _registry
from ..data._base import Data
from ..scattering._filters import PlanarFilters, STFilters, TensorProdFilters
from ..scattering._loop import ScatteringValues
from ..scattering._plan import PlanarPlan, STPlan, TensorProdPlan
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics, finish


@functools.cache
def _loop(backend_name: str) -> Callable[..., ScatteringValues]:
    """The scattering loop of a backend, compiled where it compiles (``Backend.jit``, the
    plan static): one function per backend, made once, so that its compiled programs are
    reused from one call to the next; it runs the loop of the geometry of the plan
    (``STPlan._values``). The compiled programs belong to the function made here:
    ``_loop.cache_clear()`` drops them, for a test that replaces a hook of the loop."""
    backend = _registry.backend_named(backend_name)

    def loop(data: Data, plan: STPlan, filters: STFilters) -> ScatteringValues:
        return plan._values(data, filters)

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
    backend = _registry.backend_of(filters._smoothing) or ""
    _registry.check_backend(backend, data.array, owner="these filters")
    plan._check_data(data, filters)
    if data.n_channels != plan.n_channels:
        raise ValueError(
            f"the data have {data.n_channels} channel(s), but the operator was built for "
            f"{plan.n_channels}: the number of channels is fixed at construction. Build a new "
            f"{plan._operator} for these data."
        )
    dtype = filters._smoothing.dtype
    if data.array.dtype != dtype:
        raise TypeError(
            f"the data are {data.array.dtype}, but the operator computes in "
            f"{dtype}, the default dtype when it was built. Build the "
            "data and the operator with the same default dtype (estel.set_default_dtype)."
        )


def compute(
    data: Data,
    plan: STPlan,
    filters: STFilters,
    reference: _NormReference | None,
) -> STStatistics:
    """:func:`scattering` with the reference already extracted: the loop of the geometry of
    the plan, a subclass of ``STPlan``, then the one path after the loop
    (``statistics._statistics.finish``). A ``norm="ref"`` reference is checked once, before the
    loop, on planar data and on tensor products alike."""
    plan, filters = _parts(plan, filters)
    check_data(data, plan, filters)
    if plan.norm == "ref":
        if reference is None:
            raise ValueError(
                'norm="ref" needs a reference: pass ref=, statistics computed with '
                'norm="self" (maths.md M-NORM).'
            )
        plan._check_reference(reference, int(data.shape[0]), filters._smoothing.dtype)
    backend = _registry.backend_named(data.backend)
    found = _loop(backend.name)(data, plan=plan, filters=filters)
    standardization = (found.mean_pre_std, found.std_pre_std) if plan.standardize else None
    return finish(found.buffer, plan, reference, standardization)
