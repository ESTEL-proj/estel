"""``WaveletOperator``: the only way to the wavelets and to the geometry of the pyramid
(Part I, §2.3; Part II, §12.5; task T2.10).

Internal: ESTEL computes scattering statistics, and STL's ``data.get_wavelet_op()`` has no
public counterpart (Part IV, §22.5). ``STOperator``, its plan and its loop go through this
module for everything that depends on the wavelets and on the geometry: level by level, the
convolution by the wavelets, the downsampling, the shape of each level, the border rule
(maths.md M-BORDER), the mask rules (M-MASK, M-DOWN-MASK), and the defaults of J and L (M-DEF,
M-DEF-1D). They see levels, maps and weights only.

Like ``STOperator``, the wavelet operator is a static part plus arrays, applied by pure
functions:

* :class:`WaveletPlan`, the static part, held by the ``STPlan``: built once, in NumPy float64,
  never traced (tag N of Part II, §12); immutable and hashable. It holds the levels and their
  shapes, how the low-pass filter is realised, the masks and reweighting maps of the data
  (``estel.data._masks``), and the rules of borders and masks.
* :class:`WaveletFilters`, the arrays, held by the ``STFilters``: the wavelet bank, the
  smoothing kernel and the maps of each downsampling step, on the operator's backend and dtype.
* :class:`WaveletOperator`, which binds the two to data, read for their geometry and backend
  only: :meth:`WaveletOperator.apply` and :meth:`WaveletOperator.downsample` are the pure
  functions the loop calls, on maps and levels, and :meth:`WaveletOperator.mean_weights` gives
  the plan the weights of its means.

The code proper to each geometry stays in the hooks of its ``Data`` subclass and in the NumPy
functions that build its tables. Only this module calls the hooks of filtering and
downsampling, ``_conv(x, tables, level)``, ``_downsample(x, weights, level)`` and
``_crop_weights(level, border)``, and only this module reads the banks of ``estel.wavelets``
and the mask builders of ``estel.data``: CI checks it (Part I, §2.4). The low-pass filter is
described here and applied with its first user, the tensor products (task T3.2).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType, default_device, default_dtype
from ..backend import _registry
from ..backend._protocol import DType
from ..data._base import Data, DownsampleWeights, _StaticMask
from ..data._masks import BoolMap, FloatMap, MaskMaps, build_mask_maps
from ._banks import KERNEL_SIZE as KERNEL_SIZE  # the default K, for the plan (re-exported)
from ._banks import (
    _check_n_orientations,
    _check_size,
    bump_steerable_1d,
    bump_steerable_2d,
    smoothing_kernel,
)

History = tuple[int, ...]
"""The scales of the convolutions applied to an operand, in order: ``()`` for the field,
``(j,)`` for first-layer data, ``(j2, j3)`` for second-layer data (STL's ``conv_history``)."""

LowpassRealization = Literal["smoothing_cascade"]

WAVELETS = ("bump_steerable",)
"""The wavelets of 1.0.0 (maths.md M-WAV-2D, M-WAV-1D)."""

COMPLEX: dict[FloatDType, DType] = {"float32": "complex64", "float64": "complex128"}

_K = TypeVar("_K", bound=Hashable)


# ---------------------------------------------------------------------- rules


def border_width(history: History, level: int, kernel_size: int, pbc: bool) -> int:
    """The border β of maths.md M-BORDER ("fully flexible"), in pixels at ``level``, of data
    convolved at the scales ``history`` (STL ``_get_crop_border_size_fully_flexible``).

    * no convolution, or ``pbc``: 0;
    * one convolution at ``j``: ⌈2^(j-d) · ⌊K/2⌋⌉;
    * two convolutions at ``j2`` then ``j3``: β₁ = ⌈2^(j2-j3) · ⌊K/2⌋⌉, then
      ⌈2^(j3-d) · (β₁ + ⌊K/2⌋)⌉.
    """
    half = kernel_size // 2
    if pbc or not history:
        return 0
    if len(history) == 1:
        return math.ceil(2.0 ** (history[0] - level) * half)
    if len(history) == 2:
        first = math.ceil(2.0 ** (history[0] - history[1]) * half)
        return math.ceil(2.0 ** (history[1] - level) * (first + half))
    raise ValueError(
        f"history={history}: the scattering transform convolves at most twice, so a history "
        "has 0, 1 or 2 scales (maths.md M-BORDER)."
    )


def wavelet_bank(ndim: int, n_orientations: int, size: int = KERNEL_SIZE) -> NDArray[np.complex128]:
    """The wavelet bank of the geometry, ``(L, *kernel)`` (maths.md M-WAV-2D, M-WAV-1D)."""
    if ndim == 2:
        return bump_steerable_2d(n_orientations, size)
    if ndim == 1:
        if n_orientations not in (1, 2):
            raise ValueError(
                f"L={n_orientations}: 1D data have L = 1 (the default) or L = 2, the analytic "
                "wavelet and its conjugate (maths.md M-WAV-1D)."
            )
        return bump_steerable_1d(size=size, conjugate=n_orientations == 2)
    raise ValueError(f"STOperator handles 1D and 2D planar data; got {ndim} spatial axes.")


def default_scales(data: Data) -> int:
    """J = ⌊log₂ min N⌋ - 2 (maths.md M-DEF, M-DEF-1D), at least 1."""
    smallest = min(data.shape[2:])
    n = math.floor(math.log2(smallest)) - 2
    if n < 1:
        raise ValueError(
            f"the default J = ⌊log2 {smallest}⌋ - 2 = {n} is below 1: the data are too small "
            "for the default (maths.md M-DEF). Pass J explicitly, or use larger data."
        )
    return n


def default_orientations(data: Data) -> int:
    """L = 4 in 2D, 1 in 1D (maths.md M-DEF, M-DEF-1D)."""
    return 4 if data.ndim == 2 else 1


@dataclass(frozen=True, eq=False)
class MeanWeights:
    """The static weights of one mean (maths.md M-MEAN, M-BORDER, M-MASK).

    The mean of an array ``a`` at ``level`` is ``sum(weights · a) / denominator`` over the
    spatial axes (Part I, §3.1.1), which is STL's ``maskmean`` of the cropped array.
    """

    level: int
    """The resolution level of the mean."""
    border: int
    """The border width β of M-BORDER, in pixels at ``level``, before the reduction of a
    border too large for the grid (OP-13); the reduction, if any, is in ``weights``."""
    weights: FloatMap
    """1 on the pixels that enter the mean, 0 on the border crop and on invalid pixels; a
    read-only float64 array of the spatial shape at ``level``."""
    denominator: float
    """The number of pixels that enter the mean, ``weights.sum()``, at least 1: a mean with
    no valid pixel has no ``MeanWeights`` (OP-3)."""


# ---------------------------------------------------------------------- the static part


@dataclass(frozen=True, eq=False)
class WaveletPlan:
    """The static part of a wavelet operator, held by the ``STPlan`` (Part II, §12.6).

    Built by :func:`build_wavelet_plan`; NumPy only, immutable and hashable. Two wavelet plans
    are equal, and hash equal, when they were built from the same parameters and the same
    static metadata of the data (geometry, spatial shape, ``pbc``, mask).
    """

    geometry: type[Data]
    """The geometry of the data, the Data subclass the plan was built for."""
    J: int
    """The number of scales."""
    L: int
    """The number of orientations."""
    kernel_size: int
    """K, the size of the wavelet kernels on every spatial axis."""
    ndim: int
    """The number of spatial axes of the geometry."""
    spatial_shape: tuple[int, ...]
    """The spatial shape of the data at full resolution."""
    pbc: bool
    """Whether the boundary conditions are periodic."""
    mask: NDArray[np.bool_] | None
    """The missing values of the data (True = missing), read-only, or ``None``."""
    nan_threshold: float
    """τ of the geometry (M-DOWN-MASK, OP-12)."""
    levels: tuple[int, ...]
    """The resolution levels of the wavelets: 0 … J-1, scale ``j`` at level ``j`` (STL's
    ``j_to_dg``)."""
    level_shapes: tuple[tuple[int, ...], ...]
    """The spatial shape at each level of :attr:`levels`."""
    lowpass_realization: LowpassRealization
    """How the low-pass filter φ_J is realised: in 1.0.0, ``"smoothing_cascade"``, the cascade
    of the smoothing kernels of J downsamplings, as in STL (maths.md M-NOT, M-LP)."""
    lowpass_level: int
    """The level at which the low-pass filter is read: J, the data downsampled J times. The
    planar statistics of 1.0.0 compute no low-pass coefficient; the tensor products apply it
    (task T3.2)."""
    masks: MaskMaps | None
    """The static masks and reweighting maps of the data (M-DOWN-MASK, M-MASK), or ``None``
    when no value is missing."""
    _key: tuple[Any, ...] = field(repr=False)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WaveletPlan):
            return NotImplemented
        return self._key == other._key

    def bank(self) -> NDArray[np.complex128]:
        """The wavelet bank, ``(L, *kernel)``, NumPy complex128 (M-WAV-2D, M-WAV-1D). Raises a
        ``ValueError`` when the geometry has no bank with ``L`` orientations."""
        return wavelet_bank(self.ndim, self.L, self.kernel_size)

    def border(self, history: History, level: int) -> int:
        """The border β, in pixels at ``level``, of data convolved at the scales ``history``
        (M-BORDER; :func:`border_width`)."""
        return border_width(history, level, self.kernel_size, self.pbc)

    def invalid(self, history: History, level: int) -> BoolMap | None:
        """The invalid pixels at ``level`` of data convolved at the scales ``history``
        (M-MASK; STL ``_find_mask``), or ``None`` when no value is missing."""
        return None if self.masks is None else _invalid(self.masks, history, level)

    def check_downsample(self, level: object, convolved_at: object = None) -> None:
        """Raise unless a downsampling step from ``level`` exists for the field
        (``convolved_at=None``) or for first-layer data convolved at ``convolved_at``."""
        if not _is_index(level) or not 0 <= level < self.J - 1:  # type: ignore[operator]
            raise ValueError(
                f"downsample(level={level!r}): the loop downsamples from the levels 0 … "
                f"{self.J - 2} (J = {self.J})."
            )
        if convolved_at is not None and (
            not _is_index(convolved_at) or not 0 <= convolved_at <= level  # type: ignore[operator]
        ):
            raise ValueError(
                f"downsample(level={level}, convolved_at={convolved_at!r}): first-layer "
                f"data convolved at scale j are downsampled from level j on, so pass None (the "
                f"field) or a scale between 0 and {level}."
            )

    def downsample_maps(
        self, level: int, convolved_at: int | None = None
    ) -> tuple[BoolMap, FloatMap] | None:
        """The static maps of one downsampling step, from ``level`` to ``level + 1``
        (maths.md M-DOWN-MASK; STL ``downsample``), or ``None`` when no value is missing.

        ``convolved_at`` is ``None`` for the field, or the scale ``j ≤ level`` at which
        first-layer data were convolved. Returns the valid input pixels, of the spatial shape
        at ``level``, and the reweighting 1 / (1 - f) of the output, of the shape at
        ``level + 1``, 0 at invalid outputs: the ``valid`` and ``reweight`` of
        ``DownsampleWeights``, before they are cast.
        """
        self.check_downsample(level, convolved_at)
        if self.masks is None:
            return None
        masks = self.masks
        if convolved_at is None:
            invalid = masks.full_res if level == 0 else masks.smooth_invalid[level]
            reweight = masks.smooth_reweight[level + 1]
        else:
            invalid = (
                masks.layer1[convolved_at]
                if level == convolved_at
                else masks.wav_invalid[(level, convolved_at)]
            )
            reweight = masks.wav_reweight[(level + 1, convolved_at)]
        return ~invalid, reweight


def build_wavelet_plan(
    data: Data,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    L: int | None = None,  # noqa: N803
    wavelet: str = "bump_steerable",
    *,
    kernel_size: int = KERNEL_SIZE,
) -> WaveletPlan:
    """The :class:`WaveletPlan` of wavelets with ``J`` scales and ``L`` orientations on data
    with the geometry, spatial shape, ``pbc`` and mask of ``data``; its values are not read.

    By default J = ⌊log₂ min N⌋ - 2, and L = 4 in 2D, 1 in 1D (M-DEF, M-DEF-1D). Raises when J
    is not an integer ≥ 1 or the sizes are not divisible by 2^(J-1) (M-DATA-2D, OP-8), and when
    L is not an integer ≥ 1, as STL accepts (Part II, §12.5); :meth:`WaveletPlan.bank` checks L
    against the bank of the geometry.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"the wavelets are built for a Data object, such as estel.Data2D, got "
            f"{type(data).__name__}."
        )
    if wavelet not in WAVELETS:
        raise ValueError(
            f"wavelet={wavelet!r}: ESTEL 1.0.0 has the bump-steerable wavelets only; pass "
            f'"bump_steerable" (maths.md M-WAV-2D, M-WAV-1D).'
        )
    n_scales = _check_scales(default_scales(data) if J is None else J, data)
    n_orientations = _check_n_orientations(default_orientations(data) if L is None else L)
    size = _check_size(kernel_size)
    spatial = data.shape[2:]
    pbc = data.pbc
    levels = tuple(range(n_scales))
    masks = (
        None
        if data.mask is None
        else build_mask_maps(
            data.mask,
            n_levels=n_scales,
            s=smoothing_kernel(data.ndim),
            kernel_size=size,
            threshold=data._nan_threshold,
            pbc=pbc,
        )
    )
    static_mask = None if data.mask is None else _StaticMask(data.mask)
    return WaveletPlan(
        geometry=type(data),
        J=n_scales,
        L=n_orientations,
        kernel_size=size,
        ndim=data.ndim,
        spatial_shape=spatial,
        pbc=pbc,
        mask=None if data.mask is None else _read_only(data.mask),
        nan_threshold=data._nan_threshold,
        levels=levels,
        level_shapes=tuple(tuple(n // 2**d for n in spatial) for d in levels),
        lowpass_realization="smoothing_cascade",
        lowpass_level=n_scales,
        masks=masks,
        _key=(
            type(data),
            n_scales,
            n_orientations,
            size,
            spatial,
            pbc,
            data._nan_threshold,
            static_mask,
            wavelet,
        ),
    )


# ---------------------------------------------------------------------- the arrays


class WaveletFilters(NamedTuple):
    """The arrays of a wavelet operator, held by the ``STFilters`` (Part II, §12.6): backend
    arrays in the operator's dtype, built once by :func:`build_wavelet_filters`."""

    kernels: Any
    """The tables of ``_conv``: for planar data the complex wavelet bank, ``(L, *kernel)``, the
    same at every level (M-WAV-2D, M-WAV-1D)."""
    smoothing: Any
    """The real smoothing kernel of the downsampling (M-SMOOTH-2D, M-SMOOTH-1D)."""
    field_down: Mapping[int, tuple[Any, Any]]
    """Per level d < J-1, the ``(valid, reweight)`` maps of the field from d to d+1, or no
    entry without missing values (``WaveletPlan.downsample_maps(d)``)."""
    modulus_down: Mapping[tuple[int, int], tuple[Any, Any]]
    """Per ``(d, j)``, j ≤ d < J-1, the maps of first-layer data convolved at j from d to d+1,
    or no entry without missing values (``WaveletPlan.downsample_maps(d, j)``)."""


def build_wavelet_filters(
    plan: WaveletPlan, backend_name: str, dtype: FloatDType, device: Any
) -> WaveletFilters:
    """The arrays of ``plan`` on the backend ``backend_name``, cast once to ``dtype`` on
    ``device`` (Part I, §3.1, rules 4 and 7)."""
    backend = _registry.backend_named(backend_name)

    def maps(level: int, convolved_at: int | None = None) -> tuple[Any, Any] | None:
        found = plan.downsample_maps(level, convolved_at)
        if found is None:
            return None
        valid = backend.asarray(np.asarray(found[0], dtype=bool), "bool", device)
        reweight = backend.asarray(np.asarray(found[1], dtype=np.float64), dtype, device)
        return valid, reweight

    down_levels = range(plan.J - 1)
    field_down = {d: maps(d) for d in down_levels}
    modulus_down = {(d, j): maps(d, j) for d in down_levels for j in range(d + 1)}
    return WaveletFilters(
        kernels=backend.asarray(plan.bank(), COMPLEX[dtype], device),
        smoothing=backend.asarray(smoothing_kernel(plan.ndim), dtype, device),
        field_down={d: m for d, m in field_down.items() if m is not None},
        modulus_down={k: m for k, m in modulus_down.items() if m is not None},
    )


# ---------------------------------------------------------------------- the operator


class WaveletOperator:
    """The wavelets of 1D and 2D planar data, level by level: internal (Part II, §12.5).

    ``WaveletOperator(data, J, L)`` builds the static part and, on the backend of ``data`` in
    the default dtype, the arrays. ``STOperator`` holds the two apart, in its plan and its
    filters, and the loop binds them to the data it computes on with :meth:`from_parts`; the
    functions are the same.

    Parameters
    ----------
    data:
        An example of the data: its geometry, spatial shape, ``pbc``, mask and backend fix the
        operator; its values are not read.
    J:
        The number of scales; by default ⌊log₂ min N⌋ - 2 (M-DEF, M-DEF-1D).
    L:
        The number of orientations; by default 4 in 2D and 1 in 1D.
    wavelet:
        ``"bump_steerable"``, the only wavelets of 1.0.0.
    """

    __slots__ = ("_data", "_filters", "_plan")

    _data: Data
    _plan: WaveletPlan
    _filters: WaveletFilters | None

    def __init__(
        self,
        data: Data,
        J: int | None = None,  # noqa: N803 (maths.md notation)
        L: int | None = None,  # noqa: N803
        wavelet: str = "bump_steerable",
        *,
        kernel_size: int = KERNEL_SIZE,
    ) -> None:
        plan = build_wavelet_plan(data, J, L, wavelet, kernel_size=kernel_size)
        filters = build_wavelet_filters(plan, data.backend, default_dtype(), default_device())
        self._bind(data, plan, filters)

    def _bind(self, data: Data, plan: WaveletPlan, filters: WaveletFilters | None) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_filters", filters)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("WaveletOperator is immutable: build a new one instead.")

    @classmethod
    def from_parts(
        cls, data: Data, plan: WaveletPlan, filters: WaveletFilters | None = None
    ) -> WaveletOperator:
        """The operator of a static part and its arrays, bound to ``data``, which must have
        the static metadata ``plan`` was built from: no validation and no conversion, so that
        it works inside traced code. Without ``filters``, the static rules only."""
        new = cls.__new__(cls)
        new._bind(data, plan, filters)
        return new

    @property
    def plan(self) -> WaveletPlan:
        """The static part."""
        return self._plan

    @property
    def filters(self) -> WaveletFilters | None:
        """The arrays, or ``None`` for an operator bound to its static part only."""
        return self._filters

    # ------------------------------------------------------------------ static rules

    def mean_weights(
        self, means: Mapping[_K, tuple[Sequence[History], int]]
    ) -> dict[_K, MeanWeights | None]:
        """The weights of the means ``means``: for each key, the histories of the operands of
        a (co)variance and the level it is taken at (M-MEAN, M-BORDER, M-MASK).

        The border of a mean is the larger of its operands' borders (M-BORDER, "fully
        flexible"); the geometry turns it into crop weights (``Data._crop_weights``), reducing
        a border too large for the grid as STL does, with one warning for all the means
        (OP-13). Its invalid pixels are the union of its operands' masks (M-MASK,
        "Covariances"). A mean with no pixel left has ``None`` (OP-3). NumPy only: called once,
        when a plan is built.
        """
        plan = self._plan
        borders = {
            key: max(plan.border(h, level) for h in histories)
            for key, (histories, level) in means.items()
        }
        needed = sorted({(level, borders[key]) for key, (_, level) in means.items()})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crops = {pair: self._data._crop_weights(*pair) for pair in needed}
        if caught:
            warnings.warn(caught[0].message, stacklevel=3)
        return {
            key: self._mean_weights(histories, level, borders[key], crops[(level, borders[key])])
            for key, (histories, level) in means.items()
        }

    def _mean_weights(
        self, histories: Sequence[History], level: int, border: int, crop: FloatMap
    ) -> MeanWeights | None:
        """The crop of ``border`` times the valid pixels of the union of the operands' masks;
        ``None`` when no pixel is left (OP-3)."""
        masks = self._plan.masks
        if masks is None:
            weights = np.array(crop, dtype=np.float64)
        else:
            invalid = np.logical_or.reduce([_invalid(masks, h, level) for h in histories])
            weights = np.where(invalid, 0.0, crop)
        denominator = float(weights.sum())
        if denominator == 0.0:
            return None
        weights.flags.writeable = False
        return MeanWeights(level=level, border=border, weights=weights, denominator=denominator)

    # ------------------------------------------------------------------ pure functions

    def apply(self, x: Any, j: int) -> Any:
        """ψ_j ⋆ x, on a map at level ``j`` (maths.md M-W1): complex, L orientations.

        ``x`` is an array of the data's backend, ``(Nb, Nc, *O, *spatial)`` with the spatial
        shape of level ``j`` and any axes ``O`` in between (the orientations of a first
        convolution), real or complex; the result is ``(Nb, Nc, *O, L, *spatial)``. Calls the
        hook ``_conv`` with the tables of the level. Pure: nothing is modified, nothing depends
        on array values.
        """
        plan, filters = self._plan, self._arrays("apply")
        if not _is_index(j) or not 0 <= j < plan.J:
            raise ValueError(
                f"apply(j={j!r}): the wavelets have the scales 0 … {plan.J - 1} (J = {plan.J})."
            )
        lead, spatial = self._split(x, j, "apply")
        xp = _registry.backend_named(self._data.backend).xp
        flat = xp.reshape(x, (lead[0], -1, *spatial))
        out = self._data._conv(flat, filters.kernels, j)
        return xp.reshape(out, (*lead, plan.L, *spatial))

    def downsample(self, x: Any, level: int, convolved_at: int | None = None) -> Any:
        """One normalised downsampling step of ``x``, from ``level`` to ``level + 1``, with the
        masks and reweighting maps of the plan (maths.md M-DOWN, M-DOWN-MASK).

        ``x`` is ``(Nb, Nc, *O, *spatial)`` at ``level``; the result has the spatial shape of
        ``level + 1``. ``convolved_at`` says which maps apply when values are missing: ``None``
        for the field, or the scale ``j ≤ level`` at which first-layer data were convolved
        (M-MASK). Calls the hook ``_downsample`` with the weights of the step. Pure.
        """
        plan, filters = self._plan, self._arrays("downsample")
        plan.check_downsample(level, convolved_at)
        lead, spatial = self._split(x, level, "downsample")
        if convolved_at is None:
            maps = filters.field_down.get(level)
        else:
            maps = filters.modulus_down.get((level, convolved_at))
        valid, reweight = (None, None) if maps is None else maps
        xp = _registry.backend_named(self._data.backend).xp
        flat = xp.reshape(x, (lead[0], -1, *spatial))
        weights = DownsampleWeights(filters.smoothing, valid, reweight)
        out = self._data._downsample(flat, weights, level)
        return xp.reshape(out, (*lead, *tuple(out.shape[2:])))

    def _arrays(self, function: str) -> WaveletFilters:
        if self._filters is None:
            raise ValueError(
                f"WaveletOperator.{function}: this operator holds its static part only. Bind "
                "the arrays too: WaveletOperator.from_parts(data, plan, filters), with "
                "filters = build_wavelet_filters(plan, backend, dtype, device)."
            )
        return self._filters

    def _split(self, x: Any, level: int, function: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The leading and the spatial shape of ``x``, a map ``(Nb, Nc, *O, *spatial)`` at
        ``level``; shapes are static, so the check works in traced code."""
        ndim = self._plan.ndim
        shape = tuple(int(n) for n in x.shape)
        expected = self._plan.level_shapes[level]
        if len(shape) < 2 + ndim or shape[-ndim:] != expected:
            raise ValueError(
                f"WaveletOperator.{function}: the array has shape {shape}, but a map at level "
                f"{level} is (Nb, Nc, ..., {', '.join(str(n) for n in expected)}). Bring it to "
                f"level {level} first, one downsample step per level."
            )
        return shape[:-ndim], shape[-ndim:]

    def __repr__(self) -> str:
        plan = self._plan
        return (
            f"WaveletOperator(J={plan.J}, L={plan.L}, geometry={plan.geometry.__name__}, "
            f"spatial_shape={plan.spatial_shape}, pbc={plan.pbc})"
        )


# ---------------------------------------------------------------------- helpers


def _is_index(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _invalid(masks: MaskMaps, history: History, level: int) -> BoolMap:
    """The invalid pixels at ``level`` of data convolved at the scales ``history`` (M-MASK;
    STL ``_find_mask``)."""
    if not history:
        return masks.full_res if level == 0 else masks.smooth_invalid[level]
    if len(history) == 1:
        (j,) = history
        return masks.layer1[j] if level == j else masks.wav_invalid[(level, j)]
    j2, j3 = history
    if level != j3:
        raise ValueError(
            f"second-layer data convolved at {history} are read at level {j3}, not {level}."
        )
    return masks.layer2[(j3, j2)]


def _read_only(array: Any) -> NDArray[Any]:
    copy = np.array(array)
    copy.flags.writeable = False
    return copy


def _check_scales(value: object, data: Data) -> int:
    """J as an int: at least 1, and the sizes divisible by 2^(J-1) (maths.md M-DATA-2D,
    M-DATA-1D, OP-8)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"J={value!r}: the number of scales is an integer ≥ 1, such as J=4.")
    n = int(value)
    if n < 1:
        raise ValueError(f"J={n}: the number of scales is at least 1.")
    spatial = data.shape[2:]
    if any(size % 2 ** (n - 1) for size in spatial):
        largest = min((size & -size).bit_length() for size in spatial)
        raise ValueError(
            f"J={n} needs every spatial size divisible by 2^(J-1) = {2 ** (n - 1)}, since the "
            f"loop downsamples J-1 times, but the spatial shape is {spatial} (maths.md "
            f"M-DATA-2D, OP-8). Use J ≤ {largest}, or crop or pad the data first."
        )
    return n
