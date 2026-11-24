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

``STFilters`` is public as a name only (Part II, §12.6): users get it from an operator's
``filters`` and pass it to ``estel.functional.scattering``. It is the base class of the filters
of each geometry: :class:`PlanarFilters`, whose fields are those of the private base class
``_FilterFields``, and :class:`TensorProdFilters` (task T3.2). Each is a named tuple, so JAX
takes it as a pytree whose leaves are the arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend import _registry
from ..spectra._estimators import SpectrumFilters, build_spectrum_filters
from ..statistics._layout import ChunkTables, loop_gather, tensor_prod_gather
from ..wavelets._operator import MeanWeights, WaveletFilters
from ..wavelets._tensor_prod import build_tensor_prod_wavelet_filters
from ._tensor_prod import Scale

if TYPE_CHECKING:
    from ._plan import PlanarPlan, TensorProdPlan


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
    gather: tuple[Any, ...]
    """The integer arrays of the static gather from the divided sums of the loop, in the order
    of ``statistics._layout.loop_sums``, to the compact buffer of the statistics: for each kind
    of chunk the loop computes complex and the buffer stores with real entries, the floats kept
    (int32; ``statistics._layout.loop_gather``; Part III, §19.3, §19.9; task T3.7)."""
    spectrum: SpectrumFilters | None = None
    """The static arrays of the 2D power spectrum (maths.md §1.9; task T4.1), those of
    ``SpectrumOperator`` for the same plan of the spectrum, or ``None`` without it."""


class STFilters:
    """The filters of a scattering transform and the arrays derived to apply them: an opaque
    object (Part II, §12.6).

    Get it from ``STOperator.filters`` or ``STOperatorTensorProd.filters`` and pass it, with the
    operator's plan, to ``estel.functional.scattering``. Its arrays are on the operator's
    backend, dtype and device; it has no public field. The filters of each geometry are a
    subclass, internal to ESTEL, and a named tuple: :class:`PlanarFilters` for 1D and 2D data,
    :class:`TensorProdFilters` for tensor products, and the filters of ``SpectrumOperator``
    (``spectra._plan``). Its one hook, :attr:`_any_array`, gives their backend and dtype; the
    plan builds them (``STPlan._build_filters``; issue #144).
    """

    __slots__ = ()

    @property
    def _any_array(self) -> Any:
        """An array of the filters, which gives their backend and dtype: the smoothing kernel
        (of the first block of a tensor product), or a table of the spectrum."""
        raise NotImplementedError


class PlanarFilters(_FilterFields, STFilters):
    """The :class:`STFilters` of 1D and 2D planar data (``STOperator.filters``), built by
    :func:`build_filters`; its fields are those of ``_FilterFields``."""

    __slots__ = ()

    @property
    def _any_array(self) -> Any:
        return self.wavelets._any_array


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


def build_filters(
    plan: PlanarPlan, backend_name: str, dtype: FloatDType, device: Any
) -> PlanarFilters:
    """The filters of ``plan`` on the backend ``backend_name``, cast once to ``dtype`` on
    ``device`` (Part I, §3.1, rules 4 and 7)."""
    backend = _registry.backend_named(backend_name)

    def real(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a, dtype=np.float64), dtype, device)

    def weights(found: MeanWeights) -> Any:
        return real(found.weights) if found.box is None else None

    def boolean(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a, dtype=bool), "bool", device)

    return PlanarFilters(
        wavelets=plan.wavelets._build_filters(backend_name, dtype, device),
        valid=None if plan.mask is None else boolean(~plan.mask),
        moments=weights(plan.moments),
        first_layer={j: weights(w) for j, w in plan.first_layer.items()},
        second_layer={
            j3: real(np.stack(maps.windows)) if maps.windows else None
            for j3, maps in plan.second_layer.items()
        },
        gather=tuple(
            backend.asarray(take.astype(np.int32), "int32", device)
            for take in loop_gather(plan).takes
        ),
        spectrum=None
        if plan.spectrum is None
        else build_spectrum_filters(plan.spectrum, plan.mask, backend, dtype, device),
    )


# ---------------------------------------------------------------------- tensor products


class _TensorProdFilterFields(NamedTuple):
    """The fields of :class:`TensorProdFilters`, internal to ESTEL."""

    wavelets: tuple[WaveletFilters, ...]
    """The arrays of the wavelet operator of each block: its bank and smoothing kernel."""
    arm_crops: dict[Scale, tuple[Any, ...]]
    """Per wavelet scale j3, per block, the crops of the arms along that block,
    ``(n_arms, *shape of the block)``, or ``None`` (``SecondLayerScale.arm_crops``)."""
    band_crops: dict[Scale, tuple[Any, ...]]
    """Per wavelet scale j3, per block, the crop of the band map, or ``None``."""
    s3_rows: dict[Scale, Any]
    """Per j3, the rows of the S3 sums that are the chunks of its segment (int32)."""
    s3_denominators: dict[Scale, Any]
    """Per j3, their denominators, in the operator's dtype."""
    s4_rows: dict[Scale, Any]
    s4_denominators: dict[Scale, Any]
    s4_pack: ChunkTables
    """The tables of the packing of S4 into its floats (``statistics._layout.pack``)."""


class TensorProdFilters(_TensorProdFilterFields, STFilters):
    """The :class:`STFilters` of a tensor product (Part II, §12.6; task T3.2): what
    ``STOperatorTensorProd.filters`` gives. Backend arrays in the operator's dtype, built once
    from the plan (:func:`build_tensor_prod_filters`); a named tuple, so that JAX takes it as a
    pytree whose leaves are the arrays. Its tables grow with the chunks of the segments and
    with the pixels of one block, never with the coefficients."""

    __slots__ = ()

    @property
    def _any_array(self) -> Any:
        return self.wavelets[0].smoothing


def build_tensor_prod_filters(
    plan: TensorProdPlan, backend_name: str, dtype: FloatDType, device: Any
) -> TensorProdFilters:
    """The filters of the tensor-product plan ``plan`` on the backend ``backend_name``, cast
    once to ``dtype`` on ``device`` (Part I, §3.1, rules 4 and 7)."""
    backend = _registry.backend_named(backend_name)
    loop = plan.loop

    def real(a: NDArray[Any] | None) -> Any:
        if a is None:
            return None
        return backend.asarray(np.asarray(a, dtype=np.float64), dtype, device)

    def index(a: NDArray[Any]) -> Any:
        return backend.asarray(np.asarray(a).astype(np.int32), "int32", device)

    second = loop.second_layer
    pack = tensor_prod_gather(plan)
    return TensorProdFilters(
        wavelets=build_tensor_prod_wavelet_filters(plan.wavelets, backend_name, dtype, device),
        arm_crops={s.j3: tuple(real(c) for c in s.arm_crops) for s in second},
        band_crops={s.j3: tuple(real(c) for c in s.band_crops) for s in second},
        s3_rows={s.j3: index(s.s3_rows) for s in second},
        s3_denominators={s.j3: real(s.s3_denominators) for s in second},
        s4_rows={s.j3: index(s.s4_rows) for s in second},
        s4_denominators={s.j3: real(s.s4_denominators) for s in second},
        s4_pack=(index(pack.within), index(pack.kinds), index(pack.starts)),
    )
