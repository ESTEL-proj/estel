"""``STFilters``: the numerical arrays of a scattering transform (Part II, §12.6; task
T2.5).

The arrays of the wavelet operator (``wavelets``, a ``WaveletFilters``; Part I, §2.3): the
wavelet bank, the smoothing kernel and the maps of each downsampling (maths.md M-WAV-2D,
M-WAV-1D, M-SMOOTH-2D, M-SMOOTH-1D, M-DOWN-MASK); and everything derived from the plan to apply
them: the valid pixels, the weights of the means of the field and of the first layer and the
windows of the second-layer maps, with their border crops, where the data have missing values
(M-MEAN, M-BORDER, M-MASK; maths_step2.md M-GRAM, M-WIN), and the gather from the sums of the
loop to the compact buffer (Part III, §19.9). They are backend arrays in the operator's dtype,
built once from the plan (:func:`build_filters`) and passed to the computation as an argument
rather than baked in (Part I, §3.1, rules 4 and 7).

``STFilters`` is public as a name only (Part II, §12.6): users get it from
``STOperator.filters`` and pass it to ``estel.functional.scattering``. Its fields are those of
the private base class ``_FilterFields``. It is a named tuple, so JAX takes it as a pytree whose
leaves are the arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend import _registry
from ..statistics._layout import loop_gather
from ..wavelets._operator import MeanWeights, WaveletFilters, build_wavelet_filters
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
    """The weights of ``mean`` and ``var`` (``plan.moments``), or ``None`` when their window
    needs none: only a window of kind ``weights``, that of data with missing values, has an
    array (maths_step2.md M-WIN); the loop reads the other kinds from the plan
    (:func:`window`)."""
    first_layer: Mapping[int, Any]
    """The weights of S1 and S2, per scale of ``plan.first_layer``, or ``None`` likewise."""
    second_layer: Mapping[int, Any]
    """The windows of the second-layer maps and of the band map, per scale j3 of
    ``plan.second_layer``: the distinct windows of these maps stacked, ``(D, *spatial)`` at
    the level of j3, or
    ``None`` when every window is the whole grid (maths_step2.md M-GRAM). S3 and S4 need no
    other weights: each mean is a Gram product of windowed maps, divided by the denominator
    of the plan."""
    gather: Any
    """The static gather from the divided sums of the loop, in the order of
    ``_plan.loop_sums``, to the compact buffer of the statistics (int32;
    ``statistics._layout.loop_gather``; Part III, §19.3, §19.9)."""


class STFilters(_FilterFields):
    """The filters of a scattering transform and the arrays derived to apply them: an opaque
    object (Part II, §12.6).

    Get it from ``STOperator.filters`` and pass it, with ``STOperator.plan``, to
    ``estel.functional.scattering``. Its arrays are on the operator's backend, dtype and
    device; it has no public field.
    """

    __slots__ = ()


# ---------------------------------------------------------------------- windows


def window(weights: MeanWeights, array: Any) -> Any:
    """The window of a mean as the sums take it (``sum_valid``; maths_step2.md M-WIN):
    ``None`` for a full window, a tuple of slices for a rectangle, both read from the plan, and
    ``array``, the weights of the filters, otherwise."""
    if weights.box is None:
        return array
    if weights.kind == "full":
        return None
    return tuple(slice(start, stop) for start, stop in weights.box)


# ---------------------------------------------------------------------- construction


def build_filters(plan: STPlan, backend_name: str, dtype: FloatDType, device: Any) -> STFilters:
    """The filters of ``plan`` on the backend ``backend_name``, cast once to ``dtype`` on
    ``device`` (Part I, §3.1, rules 4 and 7)."""
    backend = _registry.backend_named(backend_name)

    def real(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a, dtype=np.float64), dtype, device)

    def weights(found: MeanWeights) -> Any:
        return real(found.weights) if found.box is None else None

    def boolean(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a, dtype=bool), "bool", device)

    return STFilters(
        wavelets=build_wavelet_filters(plan.wavelets, backend_name, dtype, device),
        valid=None if plan.mask is None else boolean(~plan.mask),
        moments=weights(plan.moments),
        first_layer={j: weights(w) for j, w in plan.first_layer.items()},
        second_layer={
            j3: real(np.stack(maps.windows)) if maps.windows else None
            for j3, maps in plan.second_layer.items()
        },
        gather=backend.asarray(loop_gather(plan)[0].astype(np.int32), "int32", device),
    )
