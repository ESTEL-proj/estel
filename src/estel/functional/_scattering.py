"""``estel.functional.scattering``: the pure function behind ``STOperator.apply`` (Part II,
§12.6; Part I, §2.3; task T2.5).

It takes the data, the static plan, the filters, an optional reference and the normalisation
(``norm``, ``ref_batch_mean``, chosen at each call; task T2.19), and returns the statistics: the
scattering loop (``estel.scattering._loop``), then the normalisation (maths.md M-NORM;
``estel.statistics._normalization``). It holds no state and modifies nothing: with
``norm="self"`` the reference is computed from the statistics and returned with them, and the
operator, not this function, keeps it. ``norm`` and ``ref_batch_mean`` are Python values, static
under ``jit`` as the plan is.

**Compiled where the backend compiles** (Part I, §2.2.2; task T2.13). The loop is called
through ``Backend.jit`` with the plan as its static argument: on JAX the whole loop is one
compiled program per plan and per shape and dtype of the data, built at the first call and
reused by the later ones; the data and the filters are its arguments, never constants of the
program (Part I, §3.1, rule 4). On PyTorch ``jit`` is the identity. Derivatives and other
transformations pass through the compiled loop unchanged.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable

from ..backend import _registry
from ..data._base import Data
from ..scattering._filters import STFilters, STFiltersSingleDomain, STFiltersTensorProd
from ..scattering._loop import ScatteringValues
from ..scattering._plan import STPlan, STPlanSingleDomain, STPlanTensorProd
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics, check_normalization, finish


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
    *,
    norm: str = "self",
    ref_batch_mean: bool = False,
) -> STStatistics:
    """The scattering statistics of ``data`` (maths.md §1.6 to 1.8, §3): the pure function
    behind ``STOperator.apply`` and ``STOperatorTensorProd.apply``, one function for both.

    ``plan`` and ``filters`` are those of an operator, ``op.plan`` and ``op.filters``; ``data`` has
    the geometry, spatial shape, ``pbc``, mask and number of channels the plan was built for
    (for a tensor product, its blocks, their sizes and their ``pbc``), and the backend and dtype
    of the filters. ``STPlan`` and ``STFilters`` are the types of the plan and filters of
    either operator: those of a tensor product are their subclasses (task T3.2).

    ``norm`` is the normalisation (M-NORM): ``"self"`` (the default) normalises the statistics by
    themselves, ``"ref"`` by the reference of ``ref``, statistics computed with
    ``norm="self"``, which it requires, and ``"none"`` not at all; ``ref`` is taken with
    ``norm="ref"`` only. ``ref_batch_mean`` averages the ``norm="self"`` reference over the
    batch, and is taken with ``norm="self"`` only. With ``"self"`` the reference is part of the
    computation and gradients flow through it; with ``"ref"`` it is a constant (OP-18). Both are
    Python values, static under ``jit`` as the plan is.
    """
    found, parts = _parts(plan, filters)
    check_normalization(norm, ref_batch_mean)
    return compute(data, found, parts, reference_of(ref, norm), norm, ref_batch_mean)


def _parts(
    plan: STPlan, filters: STFilters
) -> tuple[STPlanSingleDomain | STPlanTensorProd, STFiltersSingleDomain | STFiltersTensorProd]:
    """The plan and the filters of a scattering operator, or a clear error: the type check of
    :func:`scattering`."""
    if not (
        isinstance(plan, STPlanSingleDomain | STPlanTensorProd)
        and isinstance(filters, STFiltersSingleDomain | STFiltersTensorProd)
    ):
        raise TypeError(
            f"expected the plan and the filters of a scattering operator, op.plan and "
            f"op.filters of an STOperator; got {type(plan).__name__} and "
            f"{type(filters).__name__}."
        )
    return plan, filters


def reference_of(ref: object, norm: str) -> _NormReference | None:
    """The reference held by ``ref``, ``None`` without it, checked against ``norm``, already
    checked by ``check_normalization``; its J, L and channels are checked against the plan by
    :func:`compute`, before the loop."""
    if ref is None:
        return None
    if norm != "ref":
        raise ValueError(
            f'norm="{norm}" takes no reference: pass ref only with norm="ref" (maths.md M-NORM).'
        )
    if not isinstance(ref, STStatistics):
        raise TypeError(
            f'ref: pass STStatistics computed with norm="self", got {type(ref).__name__}.'
        )
    if ref._norm != "self" or ref._reference is None:
        raise ValueError(
            f'ref: these statistics were computed with norm="{ref._norm}"; the reference '
            'comes from statistics computed with norm="self" (maths.md M-NORM).'
        )
    return ref._reference


def check_data(data: object, plan: STPlan, filters: STFilters) -> None:
    """Raise unless ``data`` is data the plan was built for, on the filters' backend and dtype."""
    if not isinstance(data, Data):
        raise TypeError(f"expected a Data object, such as estel.Data2D, got {type(data).__name__}.")
    backend = _registry.backend_of(filters._any_array) or ""
    _registry.check_backend(backend, data.array, owner="these filters")
    plan._check_data(data, filters)
    if data.n_channels != plan.n_channels:
        raise ValueError(
            f"the data have {data.n_channels} channel(s), but the operator was built for "
            f"{plan.n_channels}: the number of channels is fixed at construction. Build a new "
            f"{plan._operator_name} for these data."
        )
    dtype = filters._any_array.dtype
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
    norm: str,
    ref_batch_mean: bool = False,
) -> STStatistics:
    """The one path of every operator, behind :func:`scattering`, with the reference already
    extracted and the normalisation already checked (``check_normalization``): the values of
    the plan (``STPlan._values``: the scattering loop of its geometry, the power spectrum
    included), then the one path after them (``statistics._statistics.finish``). A
    ``norm="ref"`` reference, required, is checked once, before the values, by the plan."""
    check_data(data, plan, filters)
    if norm == "ref":
        if reference is None:
            raise ValueError(
                'norm="ref" needs a reference: pass ref=, statistics computed with '
                'norm="self" (maths.md M-NORM).'
            )
        plan._check_reference(reference, int(data.shape[0]), filters._any_array.dtype)
    backend = _registry.backend_named(data.backend)
    found = _loop(backend.name)(data, plan=plan, filters=filters)
    standardization = (found.mean_pre_std, found.std_pre_std) if plan.standardize else None
    return finish(found.buffer, plan, reference, standardization, norm, ref_batch_mean)


def apply_operator(
    name: str,
    data: Data,
    plan: STPlan,
    filters: STFilters,
    ref: STStatistics | None,
    held: _NormReference | None,
    norm: str,
    ref_batch_mean: bool,
) -> tuple[STStatistics, _NormReference | None]:
    """The ``apply`` of every operator (``STOperator``, ``STOperatorTensorProd``; maths.md
    M-NORM): the statistics of ``data`` normalised by ``norm`` and ``ref_batch_mean``, chosen at
    this call, and the reference the operator holds afterwards. ``ref`` is the reference passed
    to ``apply``; ``held`` the one the operator holds. With ``norm="ref"`` and no ``ref``,
    ``held`` normalises, and its absence is an error; with ``norm="self"`` the operator keeps the
    reference of the result, replacing the one it held with a ``UserWarning`` at the line that
    calls ``apply``; with ``norm="none"`` it keeps the one it held."""
    check_data(data, plan, filters)
    check_normalization(norm, ref_batch_mean)
    reference = reference_of(ref, norm)
    if reference is None and norm == "ref":
        reference = held
        if reference is None:
            raise ValueError(
                f'{name}.apply: norm="ref" needs a reference, and this operator holds none. '
                f'Call {name}.apply(data, norm="self") first, whose reference it keeps, or pass '
                f'ref=, statistics computed with norm="self", or build the operator with '
                f"{name}.from_statistics (maths.md M-NORM)."
            )
    stats = compute(data, plan, filters, reference, norm, ref_batch_mean)
    if norm != "self":
        return stats, held
    if held is not None:
        warnings.warn(
            f'{name}.apply: norm="self" replaces the reference this operator held by that of '
            "the new statistics (maths.md M-NORM).",
            UserWarning,
            stacklevel=3,
        )
    return stats, stats._reference
