"""The ``ScatteringPlan``: the static description of a scattering transform (Part II, §12.6;
Part I, §2.3, §3.1.1; task T2.3).

Built once, in NumPy float64, when an operator is built, from the data's static metadata
(spatial shape, ``pbc``, mask) and the operator's parameters; never traced, never
differentiated (tag N of Part II, §12). It is immutable and hashable, so that JAX can take it
as a static argument. Its arrays are NumPy: the operator casts them to its backend and dtype
once, and passes them to the computation as arguments (Part I, §3.1, rules 4 and 7).

What it holds, for STL's loop structure (maths.md §1.6 to 1.7; STL ``ST_Operator.apply`` at
`master` ``8eaf769``, ``_compute_ST_loop`` at `2D+1D` ``8f51143``):

* **Index sets**, in the order of STL's loops: the scales ``j`` of S1 and S2, the pairs
  ``(j2, j3)``, ``j2 ≤ j3``, of S3 (for j3, then j2), and the triples ``(j1, j2, j3)``,
  ``j1 ≤ j2 ≤ j3``, of S4 (for j3, then j2, then j1).
* **Levels**: 0 … J-1. Scale ``j`` is computed at level ``j`` (STL's ``j_to_dg``), and every
  mean is taken at the level of the last convolution of its operands.
* **The low-pass filter**: how it is realised and at which level. In 1.0.0, as in STL, it is the
  cascade of the smoothing kernels, the data downsampled J times (maths.md M-NOT, "Filters
  and scales"). The planar statistics compute no low-pass coefficient (M-S1 to M-S4); the
  tensor-product operator reads it (task T3.2).
* **The masks and reweighting maps** of the data, ``estel.data._masks.MaskMaps`` (M-DOWN-MASK,
  M-MASK), and the maps of each downsampling step (:meth:`ScatteringPlan.downsample_maps`).
* **The weights and denominator of every mean** (:class:`MeanWeights`; M-MEAN, M-BORDER): 1 on
  the pixels that enter the mean, 0 elsewhere, and their sum.

Each mean of the loop is a (co)variance of operands described by their convolution history,
the scales of the convolutions applied, which in ESTEL is plan data rather than data state
(Part I, §2.3):

* ``mean`` and ``var``: the field, history ``()``, at level 0 (STL ``mean``, ``cov``);
* S1 and S2 at ``j``: ``(j,)``, at level ``j`` (``mean``, ``square_mean``, ``cov``);
* S3 at ``(j2, j3)``: ``(j2, j3)`` and ``(j3,)``, at level ``j3``;
* S4 at ``(j1, j2, j3)``: ``(j1, j3)`` and ``(j2, j3)``, at level ``j3``.

The border of a mean is the larger of its operands' borders (M-BORDER, "fully flexible");
the geometry turns it into weights (``DataClass._crop_weights``), reducing a border too large
for the grid as STL does, with one warning per plan (OP-13). The invalid pixels of a mean are
the union of its operands' masks (M-MASK, "Covariances"). STL takes the mask of the operand
with more convolutions instead, assuming that it contains the other one; it does not always do
so (masked 1D data), and STL's S3 is then NaN, while the union gives a value: reported on
issue #15. Without missing values every pixel is valid. A mean with no pixel left raises a
clear error when the plan is built (M-MEAN, OP-3), where STL divides by zero.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ..data._base import DataClass, _StaticMask
from ..data._masks import BoolMap, FloatMap, MaskMaps, build_mask_maps
from ..wavelets._banks import KERNEL_SIZE, _check_n_orientations, _check_size, smoothing_kernel

History = tuple[int, ...]
"""The scales of the convolutions applied to an operand, in order: ``()`` for the field,
``(j,)`` for first-layer data, ``(j2, j3)`` for second-layer data (STL's ``conv_history``)."""

LowpassRealisation = Literal["smoothing_cascade"]


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
    """The number of pixels that enter the mean, ``weights.sum()``, at least 1."""


@dataclass(frozen=True, eq=False)
class ScatteringPlan:
    """The static description of a scattering transform on one geometry (Part II, §12.6).

    Build it with :func:`build_plan`. Two plans are equal, and hash equal, when they were built
    from the same parameters and the same static metadata of the data (geometry, spatial shape,
    ``pbc``, mask), so that they hold the same values.
    """

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
    levels: tuple[int, ...]
    """The resolution levels of the loop, in execution order: 0 … J-1."""
    level_shapes: tuple[tuple[int, ...], ...]
    """The spatial shape at each level of :attr:`levels`."""
    scales: tuple[int, ...]
    """The index set of S1 and S2: the scales 0 … J-1, scale ``j`` at level ``j``."""
    s3_pairs: tuple[tuple[int, int], ...]
    """The index set of S3: ``(j2, j3)``, j2 ≤ j3, in STL's loop order (M-S3)."""
    s4_triples: tuple[tuple[int, int, int], ...]
    """The index set of S4: ``(j1, j2, j3)``, j1 ≤ j2 ≤ j3, in STL's loop order (M-S4)."""
    lowpass_realisation: LowpassRealisation
    """How the low-pass filter φ_J is realised: in 1.0.0, ``"smoothing_cascade"``, the cascade
    of the smoothing kernels of J downsamplings, as in STL (maths.md M-NOT; Part II, §12.4,
    ``_lowpass``)."""
    lowpass_level: int
    """The level at which the low-pass filter is read: J, the data downsampled J times. The
    planar statistics of 1.0.0 compute no low-pass coefficient."""
    masks: MaskMaps | None
    """The static masks and reweighting maps of the data (M-DOWN-MASK, M-MASK), or ``None``
    when no value is missing."""
    moments: MeanWeights
    """The weights of ``mean`` and ``var``, at level 0 (M-MOM)."""
    first_layer: Mapping[int, MeanWeights]
    """For each scale ``j``: the weights of S1 and S2 at ``j`` (M-S1, M-S2), at level ``j``."""
    s3: Mapping[tuple[int, int], MeanWeights]
    """For each pair of :attr:`s3_pairs`: the weights of S3, at level ``j3`` (M-S3)."""
    s4: Mapping[tuple[int, int, int], MeanWeights]
    """For each triple of :attr:`s4_triples`: the weights of S4, at level ``j3`` (M-S4)."""
    _key: tuple[Any, ...] = field(repr=False)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScatteringPlan):
            return NotImplemented
        return self._key == other._key

    def downsample_maps(
        self, level: int, convolved_at: int | None = None
    ) -> tuple[BoolMap, FloatMap] | None:
        """The static maps of one downsampling step, from ``level`` to ``level + 1``
        (maths.md M-DOWN-MASK; STL ``downsample``), or ``None`` when no value is missing.

        ``convolved_at`` is ``None`` for the field, or the scale ``j ≤ level`` at which
        first-layer data were convolved. Returns the valid input pixels, of the spatial shape
        at ``level``, and the reweighting 1 / (1 - f) of the output, of the shape at
        ``level + 1``, 0 at invalid outputs: the ``valid`` and ``reweight`` of
        ``DownsampleWeights``, before the operator casts them.
        """
        if not _is_index(level) or not 0 <= level < self.J - 1:
            raise ValueError(
                f"downsample_maps(level={level!r}): the loop downsamples from the levels 0 … "
                f"{self.J - 2} (J = {self.J})."
            )
        if convolved_at is not None and (
            not _is_index(convolved_at) or not 0 <= convolved_at <= level
        ):
            raise ValueError(
                f"downsample_maps(level={level}, convolved_at={convolved_at!r}): first-layer "
                f"data convolved at scale j are downsampled from level j on, so pass None (the "
                f"field) or a scale between 0 and {level}."
            )
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


def build_plan(
    data: DataClass,
    J: int,  # noqa: N803 (maths.md notation)
    L: int,  # noqa: N803
    *,
    kernel_size: int = KERNEL_SIZE,
) -> ScatteringPlan:
    """The ``ScatteringPlan`` of a scattering transform with ``J`` scales and ``L``
    orientations on data with the geometry, spatial shape, ``pbc`` and mask of ``data``.

    The values of ``data.array`` are not read. Raises a ``ValueError`` when the sizes are not
    divisible by 2^(J-1) (maths.md M-DATA-2D, OP-8), and when a mean would have no valid pixel
    (M-MEAN, OP-3). Warns once when a border is too large for the grid and is reduced (M-BORDER,
    OP-13).
    """
    if not isinstance(data, DataClass):
        raise TypeError(
            f"build_plan takes a DataClass, such as estel.Data2D, got {type(data).__name__}."
        )
    n_scales = _check_scales(J, data)
    n_orientations = _check_n_orientations(L)
    size = _check_size(kernel_size)
    spatial = data.shape[2:]
    pbc = data.pbc
    levels = tuple(range(n_scales))
    scales = levels
    s3_pairs = tuple((j2, j3) for j3 in levels for j2 in range(j3 + 1))
    s4_triples = tuple(
        (j1, j2, j3) for j3 in levels for j2 in range(j3 + 1) for j1 in range(j2 + 1)
    )
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

    # The means of STL's loop: a label for error messages, the operands' histories, the level.
    means: dict[Any, tuple[str, tuple[History, ...], int]] = {
        "moments": ("mean and var", ((),), 0),
        **{("S1", j): (f"S1 and S2 (j={j})", ((j,),), j) for j in scales},
        **{
            ("S3", j2, j3): (f"S3 (j2={j2}, j3={j3})", ((j2, j3), (j3,)), j3) for j2, j3 in s3_pairs
        },
        **{
            ("S4", j1, j2, j3): (
                f"S4 (j1={j1}, j2={j2}, j3={j3})",
                tuple(dict.fromkeys(((j1, j3), (j2, j3)))),
                j3,
            )
            for j1, j2, j3 in s4_triples
        },
    }
    borders = {
        key: max(border_width(h, level, size, pbc) for h in histories)
        for key, (_, histories, level) in means.items()
    }
    crops = _crop_weights(data, {(means[key][2], border) for key, border in borders.items()})
    weights = {
        key: _mean_weights(label, histories, level, borders[key], crops, masks)
        for key, (label, histories, level) in means.items()
    }

    return ScatteringPlan(
        J=n_scales,
        L=n_orientations,
        kernel_size=size,
        ndim=data.ndim,
        spatial_shape=spatial,
        pbc=pbc,
        levels=levels,
        level_shapes=tuple(tuple(n // 2**d for n in spatial) for d in levels),
        scales=scales,
        s3_pairs=s3_pairs,
        s4_triples=s4_triples,
        lowpass_realisation="smoothing_cascade",
        lowpass_level=n_scales,
        masks=masks,
        moments=weights["moments"],
        first_layer=MappingProxyType({j: weights[("S1", j)] for j in scales}),
        s3=MappingProxyType({(j2, j3): weights[("S3", j2, j3)] for j2, j3 in s3_pairs}),
        s4=MappingProxyType({t: weights[("S4", *t)] for t in s4_triples}),
        _key=(
            type(data),
            n_scales,
            n_orientations,
            size,
            spatial,
            pbc,
            data._nan_threshold,
            None if data.mask is None else _StaticMask(data.mask),
        ),
    )


# ---------------------------------------------------------------------- helpers


def _is_index(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _check_scales(value: object, data: DataClass) -> int:
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


def _crop_weights(
    data: DataClass, needed: Iterable[tuple[int, int]]
) -> dict[tuple[int, int], FloatMap]:
    """The crop weights of every ``(level, border)`` in ``needed``, from the geometry hook.

    STL warns once per operator when it reduces a border too large for the grid (OP-13): the
    warnings of the hook are collected, and the first one alone is shown again.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        crops = {pair: data._crop_weights(*pair) for pair in sorted(needed)}
    if caught:
        warnings.warn(caught[0].message, stacklevel=3)
    return crops


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


def _mean_weights(
    label: str,
    histories: tuple[History, ...],
    level: int,
    border: int,
    crops: Mapping[tuple[int, int], FloatMap],
    masks: MaskMaps | None,
) -> MeanWeights:
    """The weights of one mean: the crop of ``border``, times the valid pixels of the union of
    the operands' masks (M-BORDER, M-MASK, M-MEAN)."""
    crop = crops[(level, border)]
    if masks is None:
        weights = np.array(crop, dtype=np.float64)
    else:
        invalid = np.logical_or.reduce([_invalid(masks, h, level) for h in histories])
        weights = np.where(invalid, 0.0, crop)
    denominator = float(weights.sum())
    if denominator == 0.0:
        raise ValueError(
            f"no pixel is left for the mean of {label}, at level {level}: the missing values "
            f"and the border crop of {border} pixel(s) remove every pixel of the "
            f"{crop.shape} grid, so the mean would divide by zero (maths.md M-MEAN, OP-3). Use "
            "a smaller J, data with fewer missing values, or pbc=True if the data are periodic."
        )
    weights.flags.writeable = False
    return MeanWeights(level=level, border=border, weights=weights, denominator=denominator)
