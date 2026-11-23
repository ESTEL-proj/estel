"""The ``STPlan``: the static description of a scattering transform (Part II, §12.6;
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
* **Scales and levels** (maths.md M-NOT; Part III, §19.12, point 5): a scale ``j`` indexes a
  wavelet, a level ``d`` a resolution. The plan holds the level at which each scale is
  computed, ``scale_levels`` (from the wavelet plan), and the levels the loop goes through,
  ``levels``; in 1.0.0 scale ``j`` is computed at level d = j (STL's ``j_to_dg``), so the
  levels are 0 … J-1. Every mean is taken at the level of the scale of the last convolution
  of its operands, read from ``scale_levels``.
* **The static part of the wavelet operator** (``wavelets``, a ``WaveletPlan``; Part I, §2.3):
  the shape of each level, how the low-pass filter is realised and at which level, the masks
  and reweighting maps of the data and of each downsampling step (M-DOWN-MASK, M-MASK), and
  the rules of borders and masks. Everything that depends on the wavelets and on the geometry
  comes from it: this module holds no geometry formula.
* **The weights and denominator of every mean** (``MeanWeights``; M-MEAN, M-BORDER): 1 on
  the pixels that enter the mean, 0 elsewhere, and their sum; and the means that have no
  valid pixel, whose coefficients are NaN (OP-3).

* **The windows of the second layer** (``second_layer``, one :class:`SecondLayerWindows` per
  scale j3 of the second convolution; maths_step2.md M-GRAM): the scales j2 whose second-layer
  maps ψ_{j3} ⋆ U_{j2} are computed, and the window of each of these maps and of the band
  map, the weights of a mean of that map alone. The loop sets each map to zero outside its own
  window, takes one Gram product, and divides each mean by its denominator, which is the number
  of pixels of the two windows' intersection: the weights of S3 and S4 above are the products
  of two windows, and STL's windows are reproduced exactly. Weights are stored once per
  distinct level, border and mask: the means that have the same three share one array.

* **Windows have a kind** (``MeanWeights.kind``; maths_step2.md M-WIN; Part III, §19.12,
  point 1; task T2.13): ``full``, every pixel of the level (periodic data without missing
  values); ``slice``, a rectangle (non-periodic data without missing values); ``weights``, a
  0/1 map (data with missing values). For the means of the field and of the first layer, the
  loop multiplies by the weights only for the third: a full window is a plain sum, a slice a
  sum over a slice. In the second layer a full window sets nothing to zero, and the other two
  multiply each map by its 0/1 window: summing the nested rectangles of a level on slices,
  one product per shell between two of them, was measured slower with the gradient than this
  product (issue #91), and the common window of M-WIN would change values.

Each mean of the loop is a (co)variance of operands described by their convolution history,
the scales of the convolutions applied, which in ESTEL is plan data rather than data state
(Part I, §2.3):

* ``mean`` and ``var``: the field, history ``()``, at level 0 (STL ``mean``, ``cov``);
* S1 and S2 at ``j``: ``(j,)``, at the level of scale ``j`` (``mean``, ``square_mean``,
  ``cov``);
* S3 at ``(j2, j3)``: ``(j2, j3)`` and ``(j3,)``, at the level of scale ``j3``;
* S4 at ``(j1, j2, j3)``: ``(j1, j3)`` and ``(j2, j3)``, at the level of scale ``j3``.

The plan lists these means, and the wavelet operator gives their weights
(``WaveletOperator.mean_weights``). The border of a mean is the larger of its operands' borders
(M-BORDER, "fully flexible"); the geometry turns it into weights, reducing a border too large
for the grid as STL does, with one warning per plan (OP-13). The invalid pixels of a mean are
the union of its operands' masks (M-MASK, "Covariances"). STL takes the mask of the operand
with more convolutions instead, assuming that it contains the other one; where it does not
(masked 1D data), STL's S3 is NaN, and the union gives a value (OP-35). Without missing values
every pixel is valid.

**Channels and options.** The plan also fixes what STL decides at each call: the number of
channels and the channel pairs of ``cross_matrix`` (:class:`ChannelPairs`; M-CHAN), the
normalisation (``norm``, ``ref_batch_mean``; M-NORM) and the standardisation of the data
(M-MOM). They are part of its identity, so that ``estel.functional.scattering`` needs nothing but
the data, the plan, the filters and an optional reference.

**Public and opaque.** ``STPlan`` is public as a name only (Part II, §12.6): users get it
from ``STOperator.plan`` and pass it to ``estel.functional.scattering``, and it has no public
field. Its fields are those of the private base class ``_PlanFields``, read by ESTEL only.

A mean with no pixel left is where STL divides by zero, and its coefficient is NaN (M-MEAN,
OP-3). The plan knows these means statically: it leaves them out of ``first_layer``, ``s3``
and ``s4``, lists them in ``empty_scales``, ``empty_s3_pairs`` and ``empty_s4_triples``, and
warns once, naming them. The
computation does not compute them, the full grid holds NaN there, like the structural NaN, with
no division 0/0 and so no NaN in gradients, and ``to_vector`` leaves them out (tasks T2.4, T2.5).
Building the plan raises an error only when a whole family (mean and var, S1 and S2, S3, S4) has
no computable coefficient.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from ..data._base import Data
from ..wavelets._operator import (
    KERNEL_SIZE,
    History,
    MeanWeights,
    WaveletOperator,
    WaveletPlan,
    build_wavelet_plan,
)
from ..wavelets._tensor_prod import TensorProdWaveletPlan, build_tensor_prod_plan

NORMS = ("none", "self", "ref")
"""The values of ``norm`` (maths.md M-NORM)."""

ChannelPair = tuple[int, int]


@dataclass(frozen=True)
class ChannelPairs:
    """The channel pairs of each family, from ``cross_matrix`` (STL's ``compute_cross_matrix``;
    maths.md M-CHAN).

    Built by :func:`channel_pairs`; static and hashable.
    """

    n_channels: int
    """Nc."""
    auto: tuple[int, ...]
    """The channels with auto statistics: the True diagonal entries."""
    cross: tuple[ChannelPair, ...]
    """The pairs ``(c1, c2)``, c1 < c2, of the cross S1 and S2: stored at ``[c1, c2]`` only,
    ``[c2, c1]`` stays NaN (OP-14)."""
    ordered: tuple[ChannelPair, ...]
    """The pairs of S3 and S4, in STL's order: ``(c, c)`` for each True diagonal entry, and
    ``(c1, c2)`` then ``(c2, c1)`` for each True entry above the diagonal (OP-14)."""


def channel_pairs(matrix: NDArray[np.bool_]) -> ChannelPairs:
    """The :class:`ChannelPairs` of a boolean ``(Nc, Nc)`` matrix; its entries below the
    diagonal are ignored, as in STL."""
    n = int(matrix.shape[0])
    auto = tuple(c for c in range(n) if matrix[c, c])
    cross = tuple((c1, c2) for c1 in range(n) for c2 in range(c1 + 1, n) if matrix[c1, c2])
    ordered = tuple(
        pair
        for c1 in range(n)
        for c2 in range(c1, n)
        if matrix[c1, c2]
        for pair in (((c1, c2),) if c1 == c2 else ((c1, c2), (c2, c1)))
    )
    return ChannelPairs(n_channels=n, auto=auto, cross=cross, ordered=ordered)


@dataclass(frozen=True, eq=False)
class SecondLayerWindows:
    """The second-layer maps of one scale j3 of the second convolution and their windows, for the
    Gram form (maths_step2.md M-GRAM; Part III, §19.3); they are computed at the level of j3.

    The window of a map is the weights of a mean of that map alone: the crop of its own border
    (M-BORDER) times its own valid pixels (M-MASK). The window of a pair of maps is the
    intersection of their windows, the larger of their borders and the union of their masks,
    which is the weights of their mean in the plan (``s3``, ``s4``).
    """

    arms: tuple[int, ...]
    """The scales j2 ≤ j3 whose maps ψ_{j3} ⋆ U_{j2} enter a mean with a valid pixel, in
    increasing order: the rows of the Gram matrix, with the channels and the orientations."""
    windows: tuple[NDArray[np.float64], ...]
    """The distinct windows of these maps, each 1 on the pixels of the window and 0 elsewhere,
    read-only, of the spatial shape of the level of j3; empty when every window is the whole grid
    (periodic data without missing values), and nothing is set to zero."""
    arm_windows: tuple[int, ...]
    """For each scale of :attr:`arms`, the position of its window in :attr:`windows`; empty
    when :attr:`windows` is."""
    band_window: int | None
    """The position in :attr:`windows` of the window of the band map W_{j3} x, the other
    operand of S3; ``None`` when :attr:`windows` is empty or j3 has no S3."""


@dataclass(frozen=True, eq=False)
class _PlanFields:
    """The fields of :class:`STPlan`, internal to ESTEL (the plan is public and
    opaque)."""

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
    n_channels: int
    """Nc, fixed when the plan is built."""
    cross_matrix: NDArray[np.bool_]
    """``cross_matrix``, read-only, ``(Nc, Nc)``, upper triangle (M-CHAN)."""
    pairs: ChannelPairs
    """The channel pairs of each family (M-CHAN)."""
    norm: str
    """The normalisation, one of :data:`NORMS` (M-NORM)."""
    ref_batch_mean: bool
    """Whether the ``norm="self"`` reference is averaged over the batch (M-NORM)."""
    standardize: bool
    """Whether each map is standardised before the transform (M-MOM)."""
    wavelets: WaveletPlan
    """The static part of the wavelet operator (Part I, §2.3): the levels and their shapes, the
    low-pass filter and how it is realised, the masks and reweighting maps, and the rules of
    borders and masks."""
    scale_levels: tuple[int, ...]
    """For each scale ``j`` of :attr:`scales`, the resolution level at which it is computed
    (``wavelets.scale_levels``): level d = j in 1.0.0. The loop and the wavelet operator read
    it here; nothing derives a level from a scale otherwise."""
    levels: tuple[int, ...]
    """The resolution levels of the loop, in execution order: 0 … the largest of
    :attr:`scale_levels` (0 … J-1 in 1.0.0)."""
    scales: tuple[int, ...]
    """The index set of S1 and S2: the scales 0 … J-1."""
    s3_pairs: tuple[tuple[int, int], ...]
    """The index set of S3: ``(j2, j3)``, j2 ≤ j3, in STL's loop order (M-S3)."""
    s4_triples: tuple[tuple[int, int, int], ...]
    """The index set of S4: ``(j1, j2, j3)``, j1 ≤ j2 ≤ j3, in STL's loop order (M-S4)."""
    moments: MeanWeights
    """The weights of ``mean`` and ``var``, at level 0 (M-MOM)."""
    first_layer: Mapping[int, MeanWeights]
    """For each scale ``j`` of :attr:`scales` not in :attr:`empty_scales`: the weights of S1
    and S2 at ``j`` (M-S1, M-S2), at the level of scale ``j``."""
    s3: Mapping[tuple[int, int], MeanWeights]
    """For each pair of :attr:`s3_pairs` not in :attr:`empty_s3_pairs`: the weights of S3, at
    the level of scale ``j3`` (M-S3)."""
    s4: Mapping[tuple[int, int, int], MeanWeights]
    """For each triple of :attr:`s4_triples` not in :attr:`empty_s4_triples`: the weights of
    S4, at the level of scale ``j3`` (M-S4)."""
    second_layer: Mapping[int, SecondLayerWindows]
    """For each scale j3 with an S3 or an S4 mean: its second-layer maps and their windows
    (M-GRAM)."""
    empty_scales: tuple[int, ...]
    """The scales whose S1 and S2 have no valid pixel: NaN coefficients, not computed (M-MEAN,
    OP-3). Empty without missing values, as for the next two."""
    empty_s3_pairs: tuple[tuple[int, int], ...]
    """The pairs of :attr:`s3_pairs` whose S3 has no valid pixel (OP-3)."""
    empty_s4_triples: tuple[tuple[int, int, int], ...]
    """The triples of :attr:`s4_triples` whose S4 has no valid pixel (OP-3)."""
    _key: tuple[Any, ...] = field(repr=False)


class STPlan(_PlanFields):
    """The static description of a scattering transform: an opaque object (Part II, §12.6).

    Get it from ``STOperator.plan`` and pass it, with ``STOperator.filters``, to
    ``estel.functional.scattering``. It is immutable and hashable, so that JAX can take it as a
    static argument; it has no public field. Two plans are equal, and hash equal, when they
    were built from the same parameters and the same static metadata of the data (geometry,
    spatial shape, ``pbc``, mask, number of channels), so that they hold the same values.
    """

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, STPlan):
            return NotImplemented
        return self._key == other._key


def build_plan(
    data: Data,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    L: int | None = None,  # noqa: N803
    *,
    kernel_size: int = KERNEL_SIZE,
    cross_matrix: Any = None,
    norm: str = "none",
    ref_batch_mean: bool = False,
    standardize: bool = False,
) -> STPlan:
    """The ``STPlan`` of a scattering transform with ``J`` scales and ``L``
    orientations on data with the geometry, spatial shape, ``pbc``, mask and number of channels
    of ``data``; by default J and L are those of the geometry (M-DEF, M-DEF-1D) and the kernel
    size that of the wavelets.

    ``cross_matrix`` is STL's ``compute_cross_matrix``, a boolean ``(Nc, Nc)`` array, or
    ``None`` for every pair c1 ≤ c2, the upper triangle (M-CHAN, M-DEF); ``norm``,
    ``ref_batch_mean`` and ``standardize`` are the operator's options (M-NORM, M-MOM). The
    values of ``data.array`` are not read. Raises a ``ValueError`` when the sizes are not
    divisible by 2^(J-1) (maths.md M-DATA-2D, OP-8), and when a whole family of statistics has
    no mean with a valid pixel (M-MEAN, OP-3). Warns once when some means have no valid pixel,
    listing them (their coefficients are NaN, OP-3), and once when a border is too large for the
    grid and is reduced (M-BORDER, OP-13).
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"build_plan takes a Data object, such as estel.Data2D, got {type(data).__name__}."
        )
    wavelets = build_wavelet_plan(data, J, L, kernel_size=kernel_size)
    matrix = _cross_matrix(cross_matrix, data.n_channels)
    _check_options(norm, ref_batch_mean, standardize)
    levels = wavelets.levels
    scale_levels = wavelets.scale_levels
    scales = tuple(range(wavelets.J))
    s3_pairs = tuple((j2, j3) for j3 in scales for j2 in range(j3 + 1))
    s4_triples = tuple(
        (j1, j2, j3) for j3 in scales for j2 in range(j3 + 1) for j1 in range(j2 + 1)
    )

    # The means of STL's loop: a label for error messages, the operands' histories (their
    # scales), the level of the mean (that of the scale of the last convolution).
    means: dict[Any, tuple[str, tuple[History, ...], int]] = {
        "moments": ("mean and var", ((),), 0),
        **{("S1", j): (f"S1 and S2 (j={j})", ((j,),), scale_levels[j]) for j in scales},
        **{
            ("S3", j2, j3): (f"S3 (j2={j2}, j3={j3})", ((j2, j3), (j3,)), scale_levels[j3])
            for j2, j3 in s3_pairs
        },
        **{
            ("S4", j1, j2, j3): (
                f"S4 (j1={j1}, j2={j2}, j3={j3})",
                tuple(dict.fromkeys(((j1, j3), (j2, j3)))),
                scale_levels[j3],
            )
            for j1, j2, j3 in s4_triples
        },
    }
    candidates = WaveletOperator.from_parts(data, wavelets).mean_weights(
        {key: (histories, level) for key, (_, histories, level) in means.items()}
    )
    weights = {key: w for key, w in candidates.items() if w is not None}
    _check_empty(means, weights)
    s3 = {(j2, j3): weights[("S3", j2, j3)] for j2, j3 in s3_pairs if ("S3", j2, j3) in weights}
    s4 = {t: weights[("S4", *t)] for t in s4_triples if ("S4", *t) in weights}

    return STPlan(
        geometry=wavelets.geometry,
        J=wavelets.J,
        L=wavelets.L,
        kernel_size=wavelets.kernel_size,
        ndim=wavelets.ndim,
        spatial_shape=wavelets.spatial_shape,
        pbc=wavelets.pbc,
        mask=wavelets.mask,
        nan_threshold=wavelets.nan_threshold,
        n_channels=data.n_channels,
        cross_matrix=matrix,
        pairs=channel_pairs(matrix),
        norm=norm,
        ref_batch_mean=bool(ref_batch_mean),
        standardize=bool(standardize),
        wavelets=wavelets,
        scale_levels=scale_levels,
        levels=levels,
        scales=scales,
        s3_pairs=s3_pairs,
        s4_triples=s4_triples,
        moments=weights["moments"],
        first_layer=MappingProxyType(
            {j: weights[("S1", j)] for j in scales if ("S1", j) in weights}
        ),
        s3=MappingProxyType(s3),
        s4=MappingProxyType(s4),
        second_layer=MappingProxyType(_second_layer(scales, weights, s3, s4)),
        empty_scales=tuple(j for j in scales if ("S1", j) not in weights),
        empty_s3_pairs=tuple(p for p in s3_pairs if ("S3", *p) not in weights),
        empty_s4_triples=tuple(t for t in s4_triples if ("S4", *t) not in weights),
        _key=(
            wavelets,
            data.n_channels,
            matrix.tobytes(),
            norm,
            bool(ref_batch_mean),
            bool(standardize),
        ),
    )


# ---------------------------------------------------------------------- tensor products

LOWPASS_TP = ("ST", None)
"""The values of ``lowpass`` of a tensor product in ESTEL 1.0.0 (maths.md M-LP): ``"ST"``, the
low-pass filter of each block where it exists, or ``None``. ESTEL 1.0.0 has no tensor-product
spectrum, so STL's ``"PS"`` and ``("ST", "PS")`` are not taken."""


@dataclass(frozen=True, eq=False)
class TensorProdPlan:
    """The static description of the statistics of a tensor product (maths.md M-NOT-TP, M-LP,
    M-IDX-TP, M-CHAN; Part II, §12.6; task T3.6): what the compact layout, the normalisation and
    the files need, without the loop, which task T3.2 adds.

    Built by :func:`build_tensor_prod_statistics_plan` from the plan of the wavelet operator of
    the tensor product (``TensorProdWaveletPlan``, task T3.1); NumPy only, immutable and
    hashable. Its index sets are those of each block (M-IDX-TP): the index set of the tensor
    product is their Cartesian product, which the layout enumerates chunk by chunk, never as a
    table the size of the full grid.
    """

    geometry: type[Data]
    """``DataTensorProd``."""
    wavelets: TensorProdWaveletPlan
    """The plans of the blocks, in block order (task T3.1)."""
    block_geometries: tuple[type[Data], ...]
    """The geometry of each block, ``Data2D`` or ``Data1D``."""
    pbc: tuple[bool, ...]
    """Whether each block is periodic."""
    J: tuple[int, ...]
    """J_a, the number of wavelet scales of each block."""
    L: tuple[int, ...]
    """L_a, the number of orientations of each block."""
    kernel_size: int
    """K, the size of the wavelet kernels, the same in every block (M-W1-TP)."""
    spatial_shape: tuple[int, ...]
    """The spatial shape of the data at full resolution, the axes of every block in order."""
    lowpass: str | None
    """``"ST"`` or ``None`` (:data:`LOWPASS_TP`)."""
    lowpass_blocks: tuple[bool, ...]
    """Whether each block has a low-pass filter (M-LP): ``lowpass="ST"`` and every size of the
    block divisible by 2^J_a."""
    n_channels: int
    """Nc, fixed when the plan is built."""
    cross_matrix: NDArray[np.bool_]
    """``cross_matrix``, read-only, ``(Nc, Nc)``, upper triangle (M-CHAN)."""
    pairs: ChannelPairs
    """The channel pairs of each family (M-CHAN)."""
    norm: str
    """The normalisation, one of :data:`NORMS` (M-NORM, M-NORM-TP)."""
    ref_batch_mean: bool
    """Whether the ``norm="self"`` reference is averaged over the batch (M-NORM)."""
    standardize: bool
    """Whether each map is standardised before the transform (M-MOM, M-STD)."""
    first_layer: tuple[tuple[int, ...], ...]
    """For each block, its scales of S1 and S2: 0 … J'_a - 1, with J'_a = J_a + 1 when the
    block has a low-pass filter, whose scale is J_a (M-IDX-TP, M-LP)."""
    s3: tuple[tuple[tuple[int, int], ...], ...]
    """For each block, its pairs ``(j2, j3)`` of S3, j2 ≤ j3 < J_a (M-IDX-TP)."""
    s4: tuple[tuple[tuple[int, int, int], ...], ...]
    """For each block, its triples ``(j1, j2, j3)`` of S4, j1 ≤ j2 ≤ j3 < J_a (M-IDX-TP)."""
    _key: tuple[Any, ...] = field(repr=False)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorProdPlan):
            return NotImplemented
        return self._key == other._key

    @property
    def n_blocks(self) -> int:
        return len(self.J)

    @property
    def n_scales(self) -> tuple[int, ...]:
        """J'_a, the number of scales of S1 and S2 in each block, the low-pass scale included."""
        return tuple(len(scales) for scales in self.first_layer)

    def orientations_at(self, scale: tuple[int, ...]) -> tuple[int, ...]:
        """The orientation shape of the chunks of S1 and S2 at the multi-index ``scale``
        (:func:`orientations_at`)."""
        return orientations_at(scale, self.J, self.L)


def orientations_at(
    scale: tuple[int, ...],
    J: tuple[int, ...],  # noqa: N803 (maths.md notation)
    L: tuple[int, ...],  # noqa: N803
) -> tuple[int, ...]:
    """The orientation shape of a tensor product at the multi-index ``scale``, for the blocks
    of ``J`` wavelet scales and ``L`` orientations: L_a, or 1 along a block taken at its
    low-pass scale J_a, which has orientation 0 only (maths.md M-LP). The chunks of S1 and S2
    and the rows of the normalisation scales sigma have this shape."""
    return tuple(1 if j == n else size for j, n, size in zip(scale, J, L, strict=True))


def build_tensor_prod_statistics_plan(
    data: Data,
    J: Any = None,  # noqa: N803 (maths.md notation)
    L: Any = None,  # noqa: N803
    *,
    lowpass: str | None = "ST",
    kernel_size: int = KERNEL_SIZE,
    cross_matrix: Any = None,
    norm: str = "none",
    ref_batch_mean: bool = False,
    standardize: bool = False,
) -> TensorProdPlan:
    """The :class:`TensorProdPlan` of a ``DataTensorProd``, with one J and one L per block
    (M-DATA-TP, M-DEF-TP: a list, an entry ``None`` giving the default of the block). The values
    of ``data`` are not read. ``lowpass`` is ``"ST"`` (STL's default) or ``None`` (M-LP);
    ``cross_matrix``, ``norm``, ``ref_batch_mean`` and ``standardize`` are those of
    :func:`build_plan`. Warns once, as STL does, for each block whose sizes are not divisible
    by 2^J_a: it has no low-pass filter (M-LP, OP-34)."""
    if not isinstance(data, Data) or data._geometry != "tensor_prod":
        raise TypeError(
            f"the statistics of a tensor product take a DataTensorProd, got "
            f"{type(data).__name__}; for 1D and 2D data, use build_plan."
        )
    if isinstance(lowpass, bool) or lowpass not in LOWPASS_TP:
        raise ValueError(
            f'lowpass={lowpass!r}: pass "ST" (the low-pass filter of each block, the default) '
            'or None (maths.md M-LP); ESTEL 1.0.0 has no tensor-product spectrum, so "PS" is '
            "not available."
        )
    wavelets = build_tensor_prod_plan(cast(Any, data), J, L, kernel_size=kernel_size)
    matrix = _cross_matrix(cross_matrix, data.n_channels)
    _check_options(norm, ref_batch_mean, standardize)
    flags = []
    for a, block in enumerate(wavelets.blocks):
        has = lowpass == "ST" and all(n % 2**block.J == 0 for n in block.spatial_shape)
        if lowpass == "ST" and not has:
            warnings.warn(
                f"block {a} ({block.geometry.__name__} of shape {block.spatial_shape}) has no "
                f"low-pass filter: its sizes are not all divisible by 2^J_{a} = {2**block.J}, "
                "which the low-pass filter needs, one downsampling more than its wavelets "
                "(maths.md M-LP, OP-34). Its S1 and S2 have the wavelet scales only; use a size "
                f"divisible by {2**block.J}, or J_{a} = {block.J - 1}, to keep it.",
                UserWarning,
                stacklevel=2,
            )
        flags.append(has)
    scales = tuple(range(block.J) for block in wavelets.blocks)
    return TensorProdPlan(
        geometry=type(data),
        wavelets=wavelets,
        block_geometries=tuple(b.geometry for b in wavelets.blocks),
        pbc=tuple(b.pbc for b in wavelets.blocks),
        J=wavelets.J,
        L=wavelets.L,
        kernel_size=wavelets.blocks[0].kernel_size,
        spatial_shape=wavelets.spatial_shape,
        lowpass=lowpass,
        lowpass_blocks=tuple(flags),
        n_channels=data.n_channels,
        cross_matrix=matrix,
        pairs=channel_pairs(matrix),
        norm=norm,
        ref_batch_mean=bool(ref_batch_mean),
        standardize=bool(standardize),
        first_layer=tuple(
            tuple(range(block.J + int(flag)))
            for block, flag in zip(wavelets.blocks, flags, strict=True)
        ),
        s3=tuple(tuple((j2, j3) for j3 in js for j2 in range(j3 + 1)) for js in scales),
        s4=tuple(
            tuple((j1, j2, j3) for j3 in js for j2 in range(j3 + 1) for j1 in range(j2 + 1))
            for js in scales
        ),
        _key=(
            wavelets,
            lowpass,
            data.n_channels,
            matrix.tobytes(),
            norm,
            bool(ref_batch_mean),
            bool(standardize),
        ),
    )


# ---------------------------------------------------------------------- the sums of the loop


class LoopSum(NamedTuple):
    """One weighted sum of the scattering loop, as the loop takes it (``_loop.py``): what the
    gather to the compact buffer reads (Part III, §19.2, §19.9; ``statistics._layout``)."""

    family: str
    """``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``."""
    key: tuple[int, ...]
    """The grid coordinates before the axes the sum carries: ``()`` for mean and var, whose
    sum carries the channels; ``(c1, c2, j)`` for S1 and S2, ``(c1, c2, j2, j3)`` for S3 and
    ``(c1, c2, j1, j2, j3)`` for S4, whose sums carry the orientations."""
    is_complex: bool
    """Whether the loop computes it complex: the cross S1 and S2, S3 and S4."""


def loop_sums(plan: STPlan) -> tuple[LoopSum, ...]:
    """The sums of the scattering loop, in the order the loop takes them (``_loop.py``): mean
    and var, then for each level and each scale j3 computed there, S1 (auto channels, then
    cross pairs) and S2 alike, then for each scale j2 of the second layer, S3 at (j2, j3) and
    S4 at (j1, j2, j3) for j1 ≤ j2, every channel pair of S3 and S4 inside (M-CHAN). The means
    the plan lists as empty (OP-3) are not taken. The loop concatenates the divided sums in
    this order, and one static gather turns them into the compact buffer."""
    pairs = plan.pairs
    channels_12 = (*((c, c) for c in pairs.auto), *pairs.cross)
    sums = [LoopSum("mean", (), False), LoopSum("var", (), False)]
    for level in plan.levels:
        for j3 in (j for j in plan.scales if plan.scale_levels[j] == level):
            if j3 in plan.first_layer:
                for family in ("S1", "S2"):
                    sums.extend(LoopSum(family, (c1, c2, j3), c1 != c2) for c1, c2 in channels_12)
            arms = plan.second_layer[j3].arms if j3 in plan.second_layer else ()
            for j2 in arms:
                if (j2, j3) in plan.s3:
                    sums.extend(LoopSum("S3", (c1, c2, j2, j3), True) for c1, c2 in pairs.ordered)
                for j1 in range(j2 + 1):
                    if (j1, j2, j3) in plan.s4:
                        sums.extend(
                            LoopSum("S4", (c1, c2, j1, j2, j3), True) for c1, c2 in pairs.ordered
                        )
    return tuple(sums)


# ---------------------------------------------------------------------- helpers


def _second_layer(
    scales: tuple[int, ...],
    weights: Mapping[Any, MeanWeights],
    s3: Mapping[tuple[int, int], MeanWeights],
    s4: Mapping[tuple[int, int, int], MeanWeights],
) -> dict[int, SecondLayerWindows]:
    """The second-layer maps of each scale j3 that has an S3 or an S4 mean, and their windows
    (:class:`SecondLayerWindows`)."""
    found = {j3: _second_layer_windows(j3, weights, s3, s4) for j3 in scales}
    return {j3: maps for j3, maps in found.items() if maps is not None}


def _second_layer_windows(
    j3: int,
    weights: Mapping[Any, MeanWeights],
    s3: Mapping[tuple[int, int], MeanWeights],
    s4: Mapping[tuple[int, int, int], MeanWeights],
) -> SecondLayerWindows | None:
    """The second-layer maps of scale ``j3`` and their windows, or ``None`` when ``j3`` has no
    S3 and no S4 mean.

    The window of the map ψ_{j3} ⋆ U_{j2} is the weights of S4 at (j2, j2, j3), a mean of that
    map alone, and the window of the band map W_{j3} x those of S1 at j3: a map that enters a
    mean with a valid pixel has a valid pixel itself, so both exist.
    """
    arms = tuple(
        j2
        for j2 in range(j3 + 1)
        if (j2, j3) in s3 or any(j2 in key[:2] and key[2] == j3 for key in s4)
    )
    if not arms:
        return None
    of_arms = [weights[("S4", j2, j2, j3)] for j2 in arms]
    of_band = weights[("S1", j3)] if any(key[1] == j3 for key in s3) else None
    used = [*of_arms, *([] if of_band is None else [of_band])]
    if all(w.denominator == w.weights.size for w in used):
        return SecondLayerWindows(arms=arms, windows=(), arm_windows=(), band_window=None)
    # The weights of one level, border and mask are one object (``mean_weights``).
    distinct = list({id(w.weights): w.weights for w in used}.values())
    position = {id(window): i for i, window in enumerate(distinct)}
    return SecondLayerWindows(
        arms=arms,
        windows=tuple(distinct),
        arm_windows=tuple(position[id(w.weights)] for w in of_arms),
        band_window=None if of_band is None else position[id(of_band.weights)],
    )


def _cross_matrix(matrix: Any, n_channels: int) -> NDArray[np.bool_]:
    """``cross_matrix`` as a read-only NumPy boolean array, upper triangle; ``None`` is
    every pair c1 ≤ c2. Every channel must appear in a pair (maths.md M-CHAN, M-DEF, OP-30)."""
    if matrix is None:
        upper = np.triu(np.ones((n_channels, n_channels), dtype=bool))
        upper.flags.writeable = False
        return upper
    found = np.asarray(matrix)
    if found.dtype != np.bool_ or found.ndim != 2 or found.shape[0] != found.shape[1]:
        raise TypeError(
            f"cross_matrix: pass a square boolean array (Nc, Nc); got dtype "
            f"{found.dtype} and shape {found.shape}."
        )
    if found.shape[0] != n_channels:
        raise ValueError(
            f"cross_matrix is {found.shape}, but the data have {n_channels} channel(s): "
            f"pass a ({n_channels}, {n_channels}) matrix, or None for every pair."
        )
    upper = np.triu(found)
    for c in range(n_channels):
        if not (upper[c, :].any() or upper[:, c].any()):
            raise ValueError(
                f"cross_matrix: channel {c} has no auto statistics and appears in no "
                "cross pair (maths.md M-CHAN, OP-30). Set its diagonal entry, or an entry of one "
                "of its pairs above the diagonal, or remove the channel."
            )
    upper.flags.writeable = False
    return upper


def _check_options(norm: object, ref_batch_mean: object, standardize: object) -> None:
    if norm not in NORMS:
        raise ValueError(
            f"norm={norm!r}: pass one of {', '.join(repr(n) for n in NORMS)} (maths.md M-NORM); "
            '"none" computes the statistics without normalisation.'
        )
    for name, value in (("ref_batch_mean", ref_batch_mean), ("standardize", standardize)):
        if not isinstance(value, bool | np.bool_):
            raise TypeError(f"{name}={value!r}: pass True or False.")


_FAMILIES = {"moments": "mean and var", "S1": "S1 and S2", "S3": "S3", "S4": "S4"}


def _family(key: Any) -> str:
    return "moments" if key == "moments" else str(key[0])


def _check_empty(
    means: Mapping[Any, tuple[str, tuple[History, ...], int]], weights: Mapping[Any, MeanWeights]
) -> None:
    """Raise if a whole family has no mean with a valid pixel; otherwise warn once, listing the
    means that have none, whose coefficients are NaN (M-MEAN, OP-3)."""
    empty = [key for key in means if key not in weights]
    if not empty:
        return
    advice = (
        "Use a smaller J, data with fewer missing values, or pbc=True if the data are periodic "
        "(the mask grows by the wavelet footprint at each layer, and the border crop with depth)."
    )
    for family, name in _FAMILIES.items():
        members = [key for key in means if _family(key) == family]
        if members and all(key not in weights for key in members):
            raise ValueError(
                f"no {name} coefficient can be computed: the missing values and the border crops "
                f"leave no pixel for any of them (maths.md M-MEAN, OP-3). {advice}"
            )
    listed = "; ".join(means[key][0] for key in empty)
    warnings.warn(
        f"{len(empty)} mean(s) have no pixel left after the missing values and the border crop, "
        f"so their coefficients are NaN and are not computed (maths.md M-MEAN, OP-3): "
        f"{listed}. {advice}",
        UserWarning,
        stacklevel=3,
    )
