"""``STFilters``: the numerical arrays of a scattering transform (Part II, §12.6; task
T2.5).

The arrays of the wavelet operator (``wavelets``, a ``WaveletFilters``; Part I, §2.3): the
wavelet bank, the smoothing kernel and the maps of each downsampling (maths.md M-WAV-2D,
M-WAV-1D, M-SMOOTH-2D, M-SMOOTH-1D, M-DOWN-MASK); and everything derived from the plan to apply
them: the valid pixels, the weights of the means of the field and of the first layer and the
windows of the second-layer maps, with their border crops (M-MEAN, M-BORDER, M-MASK;
maths_step2.md M-GRAM), and the grid positions of the coefficients. They are backend arrays in
the operator's dtype, built once from the plan (:func:`build_filters`) and passed to the
computation as an argument rather than baked in (Part I, §3.1, rules 4 and 7).

``STFilters`` is public as a name only (Part II, §12.6): users get it from
``STOperator.filters`` and pass it to ``estel.functional.scattering``. Its fields are those of
the private base class ``_FilterFields``. It is a named tuple, so JAX takes it as a pytree whose
leaves are the arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend import _registry
from ..wavelets._operator import WaveletFilters, build_wavelet_filters
from ._plan import STPlan


class _FilterFields(NamedTuple):
    """The fields of :class:`STFilters`, internal to ESTEL."""

    wavelets: WaveletFilters
    """The arrays of the wavelet operator: the wavelet bank, the smoothing kernel and the maps
    of each downsampling step (``plan.wavelets``)."""
    valid: Any
    """The valid pixels of the full-resolution field (bool), or ``None`` without missing
    values."""
    moments: Any
    """The weights of ``mean`` and ``var`` (``plan.moments``)."""
    first_layer: Mapping[int, Any]
    """The weights of S1 and S2, per scale of ``plan.first_layer``."""
    second_layer: Mapping[int, Any]
    """The windows of the second-layer maps and of the band map, per level of
    ``plan.second_layer``: the distinct windows of the level stacked, ``(D, *spatial)``, or
    ``None`` when every window is the whole grid (maths_step2.md M-GRAM). S3 and S4 need no
    other weights: each mean is a Gram product of windowed maps, divided by the denominator
    of the plan."""
    s1_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j)`` of the S1 and S2 values (:func:`first_layer_keys`)."""
    s3_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j2, j3)`` of the S3 values (:func:`s3_keys`)."""
    s4_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j1, j2, j3)`` of the S4 values (:func:`s4_keys`)."""


class STFilters(_FilterFields):
    """The filters of a scattering transform and the arrays derived to apply them: an opaque
    object (Part II, §12.6).

    Get it from ``STOperator.filters`` and pass it, with ``STOperator.plan``, to
    ``estel.functional.scattering``. Its arrays are on the operator's backend, dtype and
    device; it has no public field.
    """

    __slots__ = ()


# ---------------------------------------------------------------------- static coefficient keys


def first_layer_keys(plan: STPlan) -> list[tuple[int, int, int]]:
    """The ``(c1, c2, j)`` of S1 (and of S2) the loop computes, in its order: for each scale
    with valid pixels, the auto channels, then the cross pairs."""
    pairs = plan.pairs
    return [
        (c1, c2, j)
        for j in plan.first_layer
        for c1, c2 in (*((c, c) for c in pairs.auto), *pairs.cross)
    ]


def s3_keys(plan: STPlan) -> list[tuple[int, int, int, int]]:
    """The ``(c1, c2, j2, j3)`` of S3 the loop computes, in its order."""
    return [(c1, c2, j2, j3) for j2, j3 in plan.s3 for c1, c2 in plan.pairs.ordered]


def s4_keys(plan: STPlan) -> list[tuple[int, int, int, int, int]]:
    """The ``(c1, c2, j1, j2, j3)`` of S4 the loop computes, in its order."""
    return [(c1, c2, j1, j2, j3) for j1, j2, j3 in plan.s4 for c1, c2 in plan.pairs.ordered]


def coefficient_index(
    keys: Sequence[tuple[int, ...]], n_axes: int
) -> tuple[NDArray[np.int64], ...]:
    """The index of ``scatter_to_grid`` for ``keys``: one coordinate array per grid axis."""
    table = np.asarray(keys, dtype=np.int64).reshape(len(keys), n_axes)
    return tuple(np.ascontiguousarray(table[:, axis]) for axis in range(n_axes))


# ---------------------------------------------------------------------- construction


def build_filters(plan: STPlan, backend_name: str, dtype: FloatDType, device: Any) -> STFilters:
    """The filters of ``plan`` on the backend ``backend_name``, cast once to ``dtype`` on
    ``device`` (Part I, §3.1, rules 4 and 7)."""
    backend = _registry.backend_named(backend_name)

    def real(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a, dtype=np.float64), dtype, device)

    def boolean(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a, dtype=bool), "bool", device)

    def index(keys: Sequence[tuple[int, ...]], n_axes: int) -> tuple[Any, ...]:
        return tuple(
            backend.asarray(c.astype(np.int32), "int32", device)
            for c in coefficient_index(keys, n_axes)
        )

    return STFilters(
        wavelets=build_wavelet_filters(plan.wavelets, backend_name, dtype, device),
        valid=None if plan.mask is None else boolean(~plan.mask),
        moments=real(plan.moments.weights),
        first_layer={j: real(w.weights) for j, w in plan.first_layer.items()},
        second_layer={
            j3: real(np.stack(level.windows)) if level.windows else None
            for j3, level in plan.second_layer.items()
        },
        s1_index=index(first_layer_keys(plan), 3),
        s3_index=index(s3_keys(plan), 4),
        s4_index=index(s4_keys(plan), 5),
    )
