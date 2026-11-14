"""``ScatteringFilters``: the numerical arrays of a scattering transform (Part II, §12.6; task
T2.5).

The filters, the wavelet bank and the smoothing kernel (maths.md M-WAV-2D, M-WAV-1D,
M-SMOOTH-2D, M-SMOOTH-1D), and everything derived from the plan to apply them: the valid
pixels, the propagated masks and reweighting maps of each downsampling (M-DOWN-MASK, M-MASK),
the weights of every mean, with their border windows (M-MEAN, M-BORDER), and the grid positions
of the coefficients. They are backend arrays in the operator's dtype, built once from the plan
(:func:`build_filters`) and passed to the computation as an argument rather than baked in
(Part I, §3.1, rules 4 and 7).

``ScatteringFilters`` is public as a name only (Part II, §12.6): users get it from
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
from ..backend._protocol import DType
from ..wavelets._banks import bump_steerable_1d, bump_steerable_2d, smoothing_kernel
from ._plan import ScatteringPlan

COMPLEX: dict[FloatDType, DType] = {"float32": "complex64", "float64": "complex128"}


class _FilterFields(NamedTuple):
    """The fields of :class:`ScatteringFilters`, internal to ESTEL."""

    kernels: Any
    """The complex wavelet bank, ``(L, *kernel)`` (M-WAV-2D, M-WAV-1D)."""
    smoothing: Any
    """The real smoothing kernel (M-SMOOTH-2D, M-SMOOTH-1D)."""
    valid: Any
    """The valid pixels of the full-resolution field (bool), or ``None`` without missing
    values."""
    moments: Any
    """The weights of ``mean`` and ``var`` (``plan.moments``)."""
    first_layer: Mapping[int, Any]
    """The weights of S1 and S2, per scale of ``plan.first_layer``."""
    s3: Mapping[tuple[int, int], Any]
    """The weights of S3, per pair of ``plan.s3``."""
    s4: Mapping[tuple[int, int, int], Any]
    """The weights of S4, per triple of ``plan.s4``."""
    field_down: Mapping[int, tuple[Any, Any]]
    """Per level d < J-1, the ``(valid, reweight)`` maps of the field from d to d+1, or no
    entry without missing values (``plan.downsample_maps(d)``)."""
    modulus_down: Mapping[tuple[int, int], tuple[Any, Any]]
    """Per ``(d, j)``, j ≤ d < J-1, the maps of U_j from d to d+1, or no entry without missing
    values (``plan.downsample_maps(d, j)``)."""
    s1_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j)`` of the S1 and S2 values (:func:`first_layer_keys`)."""
    s3_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j2, j3)`` of the S3 values (:func:`s3_keys`)."""
    s4_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j1, j2, j3)`` of the S4 values (:func:`s4_keys`)."""


class ScatteringFilters(_FilterFields):
    """The filters of a scattering transform and the arrays derived to apply them: an opaque
    object (Part II, §12.6).

    Get it from ``STOperator.filters`` and pass it, with ``STOperator.plan``, to
    ``estel.functional.scattering``. Its arrays are on the operator's backend, dtype and
    device; it has no public field.
    """

    __slots__ = ()


# ---------------------------------------------------------------------- static coefficient keys


def first_layer_keys(plan: ScatteringPlan) -> list[tuple[int, int, int]]:
    """The ``(c1, c2, j)`` of S1 (and of S2) the loop computes, in its order: for each scale
    with valid pixels, the auto channels, then the cross pairs."""
    pairs = plan.pairs
    return [
        (c1, c2, j)
        for j in plan.first_layer
        for c1, c2 in (*((c, c) for c in pairs.auto), *pairs.cross)
    ]


def s3_keys(plan: ScatteringPlan) -> list[tuple[int, int, int, int]]:
    """The ``(c1, c2, j2, j3)`` of S3 the loop computes, in its order."""
    return [(c1, c2, j2, j3) for j2, j3 in plan.s3 for c1, c2 in plan.pairs.ordered]


def s4_keys(plan: ScatteringPlan) -> list[tuple[int, int, int, int, int]]:
    """The ``(c1, c2, j1, j2, j3)`` of S4 the loop computes, in its order."""
    return [(c1, c2, j1, j2, j3) for j1, j2, j3 in plan.s4 for c1, c2 in plan.pairs.ordered]


def coefficient_index(
    keys: Sequence[tuple[int, ...]], n_axes: int
) -> tuple[NDArray[np.int64], ...]:
    """The index of ``scatter_to_grid`` for ``keys``: one coordinate array per grid axis."""
    table = np.asarray(keys, dtype=np.int64).reshape(len(keys), n_axes)
    return tuple(np.ascontiguousarray(table[:, axis]) for axis in range(n_axes))


# ---------------------------------------------------------------------- construction


def wavelet_kernels(ndim: int, n_orientations: int) -> NDArray[np.complex128]:
    """The wavelet bank of the geometry (maths.md M-WAV-2D, M-WAV-1D)."""
    if ndim == 2:
        return bump_steerable_2d(n_orientations)
    if ndim == 1:
        if n_orientations not in (1, 2):
            raise ValueError(
                f"L={n_orientations}: 1D data have L = 1 (the default) or L = 2, the analytic "
                "wavelet and its conjugate (maths.md M-WAV-1D)."
            )
        return bump_steerable_1d(conjugate=n_orientations == 2)
    raise ValueError(f"STOperator handles 1D and 2D planar data; got {ndim} spatial axes.")


def build_filters(
    plan: ScatteringPlan, backend_name: str, dtype: FloatDType, device: Any
) -> ScatteringFilters:
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

    def maps(level: int, convolved_at: int | None = None) -> tuple[Any, Any] | None:
        found = plan.downsample_maps(level, convolved_at)
        return None if found is None else (boolean(found[0]), real(found[1]))

    down_levels = range(plan.J - 1)
    field_down = {d: maps(d) for d in down_levels}
    modulus_down = {(d, j): maps(d, j) for d in down_levels for j in range(d + 1)}
    return ScatteringFilters(
        kernels=backend.asarray(wavelet_kernels(plan.ndim, plan.L), COMPLEX[dtype], device),
        smoothing=real(smoothing_kernel(plan.ndim)),
        valid=None if plan.masks is None else boolean(~plan.masks.full_res),
        moments=real(plan.moments.weights),
        first_layer={j: real(w.weights) for j, w in plan.first_layer.items()},
        s3={key: real(w.weights) for key, w in plan.s3.items()},
        s4={key: real(w.weights) for key, w in plan.s4.items()},
        field_down={d: m for d, m in field_down.items() if m is not None},
        modulus_down={k: m for k, m in modulus_down.items() if m is not None},
        s1_index=index(first_layer_keys(plan), 3),
        s3_index=index(s3_keys(plan), 4),
        s4_index=index(s4_keys(plan), 5),
    )
