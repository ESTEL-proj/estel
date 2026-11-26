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
from an operator's ``plan`` and pass it to ``estel.functional.scattering``, and it has no public
field. It is the base class of the plan of each geometry: :class:`PlanarPlan`, whose fields are
those of the private base class ``_PlanFields``, and :class:`TensorProdPlan` (task T3.2), both
read by ESTEL only.

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
from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend._protocol import Backend
from ..data._base import Data
from ..spectra._bins import SpectrumBins, log_gaussian_bins
from ..spectra._estimators import choose_estimator, static_tables
from ..statistics._grid import auto_scales, orientations_at, tensor_prod_auto_scales
from ..statistics._layout import Layout, base_layout, tensor_prod_layout
from ..statistics._normalization import (
    _NormReference,
    check_reference,
    check_tensor_prod_reference,
    self_reference,
    spectrum_reference_of,
    tensor_prod_blocks,
)
from ..wavelets._operator import (
    KERNEL_SIZE,
    History,
    MeanWeights,
    WaveletOperator,
    WaveletPlan,
    build_wavelet_plan_with_tables,
)
from ..wavelets._tensor_prod import TensorProdWaveletPlan, build_tensor_prod_wavelet_plan
from ._filters import (
    PlanarFilters,
    STFilters,
    TensorProdFilters,
    build_filters,
    build_tensor_prod_filters,
)
from ._loop import ScatteringValues, scattering_values
from ._loop_tp import tensor_prod_values
from ._tensor_prod import TensorProdLoop, build_tensor_prod_loop

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


@dataclass(frozen=True)
class SpectrumPlan:
    """The static description of the 2D power spectrum (maths.md §1.9, "Range", "Number of
    bins", "Choice of estimator"; tasks T4.4, T4.1): how many bins, J, J_min, the estimator
    chosen at construction, and the spatial shape whose bins, windows and borders it reads
    (:meth:`bins`, ``spectra._bins.log_gaussian_bins``, built once in NumPy). Hashable. In the
    plan of ``STOperator`` its channel pairs are those of the plan (``PlanarPlan.pairs``): the
    pairs c1 ≤ c2 of S1 and S2, one matrix for the scattering statistics and the spectrum
    (M-CHAN; decision on #148); σᴾˢ exists for the channels of ``pairs.auto`` (M-NORM, OP-30)."""

    n_bins: int
    """n_b, the number of bins: given, or max(4, round(3 (J - 1 - J_min)))."""
    J: int
    """J, the J of the operator: the J given to it, or ⌊log₂ N_max⌋ - 1, equal to the default J
    of the scattering statistics (M-DEF)."""
    Jmin: int
    """J_min, with 0 ≤ J_min ≤ J - 2 (OP-93)."""
    method: str
    """The estimator, one of ``spectra._estimators.ESTIMATORS``, chosen at construction from
    ``pbc`` and the missing pixels of the example, or given (maths.md §1.9)."""
    spatial_shape: tuple[int, ...]
    """(N₁, N₂), even."""
    n_valid: int
    """The number of valid pixels of the data, N₁ N₂ without missing pixels: the denominator of
    the mean that fills the missing pixels of each map (``nan_mask``, ``pbc_nan_mask``)."""

    def content(self) -> dict[str, Any]:
        """The parameters of the spectrum for ``estel.io``."""
        return {"n_bins": self.n_bins, "J": self.J, "Jmin": self.Jmin, "method": self.method}

    def bins(self) -> SpectrumBins:
        """The bins, windows, normalisations and borders of the spectrum (maths.md §1.9)."""
        n1, n2 = self.spatial_shape
        return log_gaussian_bins((int(n1), int(n2)), self.J, self.Jmin, self.n_bins)


class SpectrumAdvice(NamedTuple):
    """What an operator tells its user to do when the power spectrum cannot be built (golden
    rule 8): each operator class carries its own (``_spectrum_advice``) and passes it to
    :func:`build_spectrum_plan`, so that the shared code compares no operator name."""

    not_2d: str
    """After data that are not 2D (:func:`spectrum_2d_only`: LF23, LF24)."""
    odd_size: str
    """After an odd size (OP-20)."""
    jmin: str
    """After J_min outside 0 … J - 2 (OP-93)."""


SCATTERING_SPECTRUM_ADVICE = SpectrumAdvice(
    not_2d="pass compute_PS=False.",
    odd_size="crop the data to even sizes, or pass compute_PS=False.",
    jmin="Pass compute_PS=False, a smaller Jmin, or use a larger image (a larger J).",
)
"""The advice of ``STOperator``, which can leave the spectrum out (``compute_PS=False``)."""


def spectrum_2d_only(received: str, advice: str) -> str:
    """The one message that refuses the power spectrum of data that are not 2D, the same for
    every geometry (Part II, §12.6; the amendment to decision 2 on #167): ``received`` is the
    name of the class of the data, for display only, and ``advice`` what to do next (golden
    rule 8)."""
    return (
        f"the data are a {received}, which has no power spectrum in ESTEL 1.0.0: it computes "
        "power spectra of 2D data only; those of 1D data and of tensor products come later "
        f"(LF23), and those of HEALPix data too (LF24). {advice}"
    )


def build_spectrum_plan(
    spatial_shape: tuple[int, ...],
    geometry: type[Data],
    J: int | None,  # noqa: N803 (maths.md notation)
    n_bins: Any,
    Jmin: Any,  # noqa: N803 (STL's name)
    method: Any = "auto",
    pbc: bool = True,
    mask: NDArray[np.bool_] | None = None,
    advice: SpectrumAdvice | None = None,
) -> SpectrumPlan:
    """The :class:`SpectrumPlan` of 2D data of ``spatial_shape`` (maths.md §1.9), ``geometry``
    the Data subclass of the data, whose declared tag (``Data._geometry``) must be 2D: the J of the
    spectrum is ``J``, the J given to the operator, or ⌊log₂ N_max⌋ - 1 with N_max = ⌊min(N₁, N₂)/2⌋
    when it is ``None``; the estimator is ``method``, or with ``"auto"`` the choice of §1.9 from
    ``pbc`` and the missing pixels of ``mask``. Raises a clear error on data that are not 2D (LF23,
    LF24), on an odd size (OP-20), when J_min is not between 0 and J - 2 (OP-93), on a number of
    bins that is not a positive integer (a ``TypeError`` when ``Jmin`` or ``n_bins`` is not an
    integer, a ``ValueError`` when it is out of range), on an unknown ``method``, on ``fft`` or
    ``pbc_mask`` with missing pixels, and on a bin of ``pbc_nan_mask`` without an observed pixel
    inside its border. ``advice`` is the advice of the operator that asks, which its messages
    end with (``STOperator._spectrum_advice``); by default that of ``STOperator``, whose plan
    :func:`build_plan` builds."""
    hint = SCATTERING_SPECTRUM_ADVICE if advice is None else advice
    if geometry._geometry != "2d":
        raise ValueError(spectrum_2d_only(geometry.__name__, hint.not_2d))
    if any(n % 2 for n in spatial_shape):
        raise ValueError(
            f"the power spectrum requires even sizes, and the data are {tuple(spatial_shape)} "
            f"(maths.md §1.9, OP-20): {hint.odd_size}"
        )
    n_max = min(spatial_shape) // 2
    j_ps = int(np.floor(np.log2(n_max))) - 1 if J is None else int(J)
    if isinstance(Jmin, bool) or not isinstance(Jmin, int | np.integer):
        raise TypeError(f"Jmin={Jmin!r}: pass an integer, 0 by default (maths.md §1.9).")
    if not 0 <= int(Jmin) <= j_ps - 2:
        raise ValueError(
            f"Jmin={Jmin} with J = {j_ps}: the power spectrum needs 0 ≤ Jmin ≤ J - 2 (maths.md "
            f"§1.9, OP-93). {hint.jmin}"
        )
    bins_advice = (
        "pass a positive integer, or None for max(4, round(3 (J - 1 - Jmin))) bins, about three "
        "per octave (maths.md §1.9)."
    )
    if n_bins is None:
        found = max(4, round(3 * (j_ps - 1 - int(Jmin))))
    elif isinstance(n_bins, bool) or not isinstance(n_bins, int | np.integer):
        raise TypeError(f"n_bins={n_bins!r}: {bins_advice}")
    elif n_bins < 1:
        raise ValueError(f"n_bins={n_bins!r}: {bins_advice}")
    else:
        found = int(n_bins)
    estimator = choose_estimator(method, bool(pbc), mask is not None and bool(mask.any()))
    plan = SpectrumPlan(
        n_bins=found,
        J=j_ps,
        Jmin=int(Jmin),
        method=estimator,
        spatial_shape=tuple(int(n) for n in spatial_shape),
        n_valid=int(np.prod(spatial_shape)) if mask is None else int((~mask).sum()),
    )
    if estimator == "pbc_nan_mask":
        bins = plan.bins()
        static_tables(  # raises on a bin without an observed pixel inside its border
            estimator,
            bins.mode_windows(),
            bins.mode_weights,
            bins.norm,
            bins.hard_masks(),
            None if mask is None else ~mask,
        )
    return plan


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


class _SharedFields:
    """The fields every plan has, read by the code shared by the operators (the data checks,
    the normalisation, the files): declared here, set by each subclass of :class:`STPlan`.
    Private, so that the public, opaque ``STPlan`` has no public field."""

    __slots__ = ()

    geometry: type[Data]
    """The geometry of the data, the Data subclass the plan was built for."""
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


class STPlan(_SharedFields):
    """The static description of a scattering transform: an opaque object (Part II, §12.6).

    Get it from ``STOperator.plan`` or ``STOperatorTensorProd.plan`` and pass it, with the
    operator's filters, to ``estel.functional.scattering``. It is immutable and hashable, so
    that JAX can take it as a static argument; it has no public field. The plan of each
    geometry is a subclass, internal to ESTEL: :class:`PlanarPlan` for single-domain data (1D,
    2D and HEALPix), :class:`TensorProdPlan` for tensor products (task T3.2). Two plans are
    equal, and hash equal, when they are of one class and were built from the same parameters
    and the same static metadata of the data (geometry, spatial shape, ``pbc``, mask, number of
    channels; on HEALPix, ``depth`` and ``cell_ids``;
    for a tensor product, its blocks), so that they hold the same values.

    **Hooks** (Part I, §2.3; Part II, §12.6; issue #144). What depends on the geometry goes
    through a small, fixed set of internal hooks, declared here and implemented by each
    subclass, so that the operator, ``estel.functional.scattering`` and ``STStatistics`` make
    one call where they would branch on the class of the plan: :attr:`_operator_name`, a
    constant of the class; :meth:`_check_data`, :meth:`_build_filters` and :meth:`_values`,
    for the computation; :meth:`_layout`,
    :meth:`_check_reference` and :meth:`_self_reference`, for the normalisation;
    :meth:`_content`, for the files. Each implementation calls the function of its geometry.
    """

    __slots__ = ()
    _key: tuple[Any, ...]

    _operator_name: ClassVar[str]
    """The name of the operator of this geometry, for the messages: ``"STOperator"`` or
    ``"STOperatorTensorProd"``."""

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, STPlan):
            return NotImplemented
        return type(self) is type(other) and self._key == other._key

    def _check_data(self, data: Data, filters: STFilters) -> None:
        """Raise unless ``filters`` are those of a plan of this geometry and ``data`` the data
        this plan was built for: its geometry, spatial shape and ``pbc``, and its mask or its
        blocks. The number of channels, the backend and the dtype are checked by the caller
        (``functional._scattering.check_data``)."""
        raise NotImplementedError

    def _build_filters(
        self, backend_name: str, dtype: FloatDType, device: Any, tables: Any = None
    ) -> STFilters:
        """The filters of this plan on the backend ``backend_name``, cast once to ``dtype`` on
        ``device`` (Part I, §3.1, rules 4 and 7). ``tables``: those the plan was built with
        (:func:`build_plan_with_tables`), read instead of built again, or ``None``."""
        raise NotImplementedError

    def _values(self, data: Data, filters: STFilters) -> ScatteringValues:
        """The scattering loop of this geometry: the statistics of ``data`` without
        normalisation, in the buffer of :meth:`_layout`. ``data`` and ``filters`` were checked
        by :meth:`_check_data`. Pure."""
        raise NotImplementedError

    def _layout(self) -> Layout:
        """The layout of the raw buffer of the loop (``statistics._layout``)."""
        raise NotImplementedError

    def _check_reference(self, reference: _NormReference, n_batch: int, dtype: Any) -> None:
        """Raise unless ``reference`` can normalise the statistics of this plan, for a batch of
        ``n_batch`` in ``dtype`` (maths.md M-NORM, M-NORM-TP)."""
        raise NotImplementedError

    def _self_reference(self, backend: Backend, buffer: Any, layout: Layout) -> _NormReference:
        """The reference of ``norm="self"``, from the raw ``buffer`` of the loop in ``layout``
        (M-NORM, M-NORM-TP): the scales of sigma, the orientations its diagonal averages, and the
        blocks of a tensor product, read from this plan."""
        raise NotImplementedError

    def _content(self) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        """The parameters of this plan for ``estel.io``, from which ``STStatistics.load``
        builds it again, and its arrays other than ``cross_matrix``, by name."""
        raise NotImplementedError


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
    meta: tuple[Any, ...]
    """The static metadata of the data beyond the mask and ``pbc``, ``Data._meta()`` of the
    example: ``()`` for 1D and 2D data, ``(depth, cell_ids)`` for ``DataHealpix``. The data
    given to the plan must have the same (:meth:`PlanarPlan._check_data`), and the files store it
    (``Data._meta_content``). The equality and the hash of plans rest on the wavelet plan
    (``wavelets._key``, in ``_key``), which holds the same metadata; ``meta`` is what
    ``_check_data`` and the files read."""
    _key: tuple[Any, ...] = field(repr=False)
    spectrum: SpectrumPlan | None = None
    """The power spectrum of the statistics (maths.md §1.9), or ``None`` when they do not hold
    it (``compute_PS=False``, or data that are not 2D). The loop computes it with the estimators
    of ``spectra._estimators`` on the field after the standardisation, and
    writes it after the values of the loop, the last floats of the buffer (task T4.1)."""


class PlanarPlan(_PlanFields, STPlan):
    """The :class:`STPlan` of single-domain data, 1D, 2D and HEALPix (``STOperator.plan``),
    built by :func:`build_plan`; its fields are those of ``_PlanFields``."""

    _operator_name: ClassVar[str] = "STOperator"

    def _check_data(self, data: Data, filters: STFilters) -> None:
        if not isinstance(filters, PlanarFilters):
            raise TypeError(_OTHER_OPERATORS)
        if data.mask is None or self.mask is None:
            same_mask = data.mask is None and self.mask is None
        else:
            same_mask = bool(np.array_equal(data.mask, self.mask))
        if not (
            isinstance(data, self.geometry)
            and tuple(data.shape[2:]) == self.spatial_shape
            and data.pbc == self.pbc
            and same_mask
        ):
            raise ValueError(
                f"the data ({type(data).__name__}, spatial shape {tuple(data.shape[2:])}, "
                f"pbc={data.pbc}) differ from those the plan was built for "
                f"({self.geometry.__name__}, spatial shape {self.spatial_shape}, pbc={self.pbc}, "
                "and the same mask of missing values). The plan is static: build a new "
                "STOperator for these data."
            )
        if data._meta() != self.meta:
            content, arrays = type(data)._meta_content(data._meta())
            raise ValueError(
                f"the data ({type(data).__name__}, spatial shape {tuple(data.shape[2:])}) lie on "
                f"other pixels than those the plan was built for: their "
                f"{', '.join([*content, *arrays])} differ. The plan is static: build a new "
                "STOperator for these data."
            )

    def _build_filters(
        self, backend_name: str, dtype: FloatDType, device: Any, tables: Any = None
    ) -> PlanarFilters:
        return build_filters(self, backend_name, dtype, device, tables)

    def _values(self, data: Data, filters: STFilters) -> ScatteringValues:
        return scattering_values(data, self, cast(PlanarFilters, filters))  # by _check_data

    def _layout(self) -> Layout:
        return base_layout(self)

    def _check_reference(self, reference: _NormReference, n_batch: int, dtype: Any) -> None:
        check_reference(reference, self, n_batch, dtype)

    def _self_reference(self, backend: Backend, buffer: Any, layout: Layout) -> _NormReference:
        return self_reference(
            backend,
            buffer,
            layout,
            auto_scales(self),
            self.L,
            self.ref_batch_mean,
            spectrum=None
            if self.spectrum is None
            else spectrum_reference_of(self.spectrum, self.pairs),
        )

    def _content(self) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        content = {
            "geometry": self.geometry._geometry,
            "J": self.J,
            "L": self.L,
            "kernel_size": self.kernel_size,
            "spatial_shape": list(self.spatial_shape),
            "pbc": self.pbc,
            "nan_threshold": self.nan_threshold,
            "n_channels": self.n_channels,
            "norm": self.norm,
            "ref_batch_mean": self.ref_batch_mean,
            "standardize": self.standardize,
            "spectrum": None if self.spectrum is None else self.spectrum.content(),
        }
        meta_content, meta_arrays = self.geometry._meta_content(self.meta)
        arrays = {} if self.mask is None else {"mask": np.asarray(self.mask)}
        return {**content, "meta": meta_content}, {**arrays, **meta_arrays}


_OTHER_OPERATORS = (
    "the plan and the filters come from different operators: pass op.plan and op.filters of "
    "one operator."
)


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
    compute_PS: bool | None = False,  # noqa: N803 (STL's name)
    n_bins: int | None = None,
    Jmin: int = 0,  # noqa: N803 (STL's name)
    spectrum_method: str = "auto",
    spectrum_advice: SpectrumAdvice | None = None,
) -> PlanarPlan:
    """The plan of :func:`build_plan_with_tables`, without the tables of its wavelets."""
    return build_plan_with_tables(
        data,
        J,
        L,
        kernel_size=kernel_size,
        cross_matrix=cross_matrix,
        norm=norm,
        ref_batch_mean=ref_batch_mean,
        standardize=standardize,
        compute_PS=compute_PS,
        n_bins=n_bins,
        Jmin=Jmin,
        spectrum_method=spectrum_method,
        spectrum_advice=spectrum_advice,
    )[0]


def build_plan_with_tables(
    data: Data,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    L: int | None = None,  # noqa: N803
    *,
    kernel_size: int = KERNEL_SIZE,
    cross_matrix: Any = None,
    norm: str = "none",
    ref_batch_mean: bool = False,
    standardize: bool = False,
    compute_PS: bool | None = False,  # noqa: N803 (STL's name)
    n_bins: int | None = None,
    Jmin: int = 0,  # noqa: N803 (STL's name)
    spectrum_method: str = "auto",
    spectrum_advice: SpectrumAdvice | None = None,
) -> tuple[PlanarPlan, Any]:
    """The :class:`PlanarPlan` of a scattering transform with ``J`` scales and ``L``
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

    With ``compute_PS``, the statistics also hold the 2D power spectrum (``None``: on 2D data
    only, the default of ``STOperator``), with ``n_bins`` bins,
    J_min ``Jmin`` and the estimator ``spectrum_method`` (maths.md §1.9;
    :func:`build_spectrum_plan`); without it, ``n_bins``, ``Jmin`` and ``spectrum_method`` keep
    their defaults (decision on #148). ``spectrum_advice`` is the advice of the operator that
    builds the plan, for the errors of the spectrum (``STOperator._spectrum_advice`` by
    default).

    Beside the plan, the NumPy tables its wavelets were built with
    (:func:`build_wavelet_plan_with_tables`), for :meth:`PlanarPlan._build_filters` of the same
    operator (``tables=``; task T5.7): a value, not part of the plan.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"build_plan takes a Data object, such as estel.Data2D, got {type(data).__name__}."
        )
    wavelets, tables = build_wavelet_plan_with_tables(data, J, L, kernel_size=kernel_size)
    matrix = _cross_matrix(cross_matrix, data.n_channels)
    _check_options(norm, ref_batch_mean, standardize)
    pairs = channel_pairs(matrix)
    spectrum = _spectrum(wavelets, J, compute_PS, n_bins, Jmin, spectrum_method, spectrum_advice)
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

    plan = PlanarPlan(
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
        pairs=pairs,
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
        second_layer=MappingProxyType(
            {
                j3: maps
                for j3 in scales
                if (maps := _second_layer_windows(j3, weights, s3, s4)) is not None
            }
        ),
        empty_scales=tuple(j for j in scales if ("S1", j) not in weights),
        empty_s3_pairs=tuple(p for p in s3_pairs if ("S3", *p) not in weights),
        empty_s4_triples=tuple(t for t in s4_triples if ("S4", *t) not in weights),
        meta=data._meta(),
        _key=(
            wavelets,
            data.n_channels,
            matrix.tobytes(),
            norm,
            bool(ref_batch_mean),
            bool(standardize),
            spectrum,
        ),
        spectrum=spectrum,
    )
    return plan, tables


def resolve_compute_ps(compute_ps: object, geometry: type[Data]) -> bool:
    """Whether an operator on data of the class ``geometry`` computes the power spectrum, from
    its ``compute_PS``: ``None`` (the default) means the spectrum on 2D data and none on other
    data, where ESTEL 1.0.0 has none (Part II, §12.6); True on other data raises
    :func:`spectrum_2d_only`. The one rule of ``STOperator`` and of the syntheses (M-SYN-OPS)."""
    if compute_ps is not None and not isinstance(compute_ps, bool | np.bool_):
        raise TypeError(f"compute_PS={compute_ps!r}: pass True, False or None (the default).")
    two_d = geometry._geometry == "2d"
    if compute_ps and not two_d:
        raise ValueError(
            "compute_PS=True: "
            + spectrum_2d_only(geometry.__name__, "Pass compute_PS=False, or leave it to None.")
        )
    return two_d if compute_ps is None else bool(compute_ps)


def _spectrum(
    wavelets: WaveletPlan,
    J: int | None,  # noqa: N803 (maths.md notation)
    compute_ps: object,
    n_bins: object,
    jmin: object,
    method: object = "auto",
    advice: SpectrumAdvice | None = None,
) -> SpectrumPlan | None:
    """The :class:`SpectrumPlan` of :func:`build_plan`, or ``None`` without ``compute_PS``, whose
    options must then keep their defaults (decision on #148). ``compute_PS=None`` means the
    default: the spectrum on 2D data, none on other data, where ESTEL 1.0.0 has none (Part II,
    §12.6), and where ``compute_PS=True`` or a spectrum option raises :func:`spectrum_2d_only`."""
    two_d = wavelets.geometry._geometry == "2d"
    if not resolve_compute_ps(compute_ps, wavelets.geometry):
        changed = [
            f"{name}={value!r}"
            for name, value, default in (
                ("n_bins", n_bins, None),
                ("Jmin", jmin, 0),
                ("spectrum_method", method, "auto"),
            )
            if value is not default and value != default
        ]
        if changed and not two_d:
            raise ValueError(
                f"{', '.join(changed)}: "
                + spectrum_2d_only(
                    wavelets.geometry.__name__,
                    "These options act only on the power spectrum, with compute_PS=True; leave "
                    "them to their defaults.",
                )
            )
        if changed:
            raise ValueError(
                f"{', '.join(changed)}: these options act only on the power spectrum, with "
                "compute_PS=True; leave them to their defaults otherwise (maths.md §1.9)."
            )
        return None
    return build_spectrum_plan(
        wavelets.spatial_shape,
        wavelets.geometry,
        J,
        n_bins,
        jmin,
        method,
        wavelets.pbc,
        wavelets.mask,
        advice,
    )


# ---------------------------------------------------------------------- tensor products

LOWPASS_TP = ("ST", None)
"""The values of ``lowpass`` of a tensor product in ESTEL 1.0.0 (maths.md M-LP): ``"ST"``, the
low-pass filter of each block where it exists, or ``None``. ESTEL 1.0.0 has no tensor-product
spectrum, so STL's ``"PS"`` and ``("ST", "PS")`` are not taken."""


@dataclass(frozen=True, eq=False)
class TensorProdPlan(STPlan):
    """The static description of the statistics of a tensor product (maths.md M-NOT-TP, M-LP,
    M-IDX-TP, M-CHAN; Part II, §12.6; task T3.6): what the compact layout, the normalisation and
    the files need, and the static description of the loop of task T3.2 (:attr:`loop`).

    Built by :func:`build_tensor_prod_statistics_plan` from the plan of the wavelet operator of
    the tensor product (``TensorProdWaveletPlan``, task T3.1); NumPy only, immutable and
    hashable. Its index sets are those of each block (M-IDX-TP): the index set of the tensor
    product is their Cartesian product, which the layout enumerates chunk by chunk, never as a
    table the size of the full grid.
    """

    _operator_name: ClassVar[str] = "STOperatorTensorProd"

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
    loop: TensorProdLoop
    """The static description of the loop (``_tensor_prod``; task T3.2): per scale, the
    windows and denominators of S1 and S2; per wavelet scale j3, the arms, the crops of the
    blocks, and the rows and denominators of the chunks of S3 and S4. Determined by the other
    fields, so not part of the identity of the plan."""
    first_layer: tuple[tuple[int, ...], ...]
    """For each block, its scales of S1 and S2: 0 … J'_a - 1, with J'_a = J_a + 1 when the
    block has a low-pass filter, whose scale is J_a (M-IDX-TP, M-LP)."""
    s3: tuple[tuple[tuple[int, int], ...], ...]
    """For each block, its pairs ``(j2, j3)`` of S3, j2 ≤ j3 < J_a (M-IDX-TP)."""
    s4: tuple[tuple[tuple[int, int, int], ...], ...]
    """For each block, its triples ``(j1, j2, j3)`` of S4, j1 ≤ j2 ≤ j3 < J_a (M-IDX-TP)."""
    _key: tuple[Any, ...] = field(repr=False)

    @property
    def n_blocks(self) -> int:
        return len(self.J)

    @property
    def n_scales(self) -> tuple[int, ...]:
        """J'_a, the number of scales of S1 and S2 in each block, the low-pass scale included."""
        return tuple(len(scales) for scales in self.first_layer)

    def orientations_at(self, scale: tuple[int, ...]) -> tuple[int, ...]:
        """The orientation shape of the chunks of S1 and S2 at the multi-index ``scale``
        (``statistics._grid.orientations_at``)."""
        return orientations_at(scale, self.J, self.L)

    def _check_data(self, data: Data, filters: STFilters) -> None:
        if not isinstance(filters, TensorProdFilters):
            raise TypeError(_OTHER_OPERATORS)
        # A DataTensorProd (isinstance), whose blocks L2 reads without importing its class
        # (Part I, §2.4).
        if not (
            isinstance(data, self.geometry)
            and tuple(cast(Any, data).blocks) == self.block_geometries
            and tuple(data.shape[2:]) == self.spatial_shape
            and data.pbc == self.pbc
        ):
            names = ", ".join(g.__name__ for g in self.block_geometries)
            raise ValueError(
                f"the data ({type(data).__name__}, spatial shape {tuple(data.shape[2:])}, "
                f"pbc={data.pbc}) differ from those the plan was built for (DataTensorProd with "
                f"the blocks [{names}], spatial shape {self.spatial_shape}, "
                f"pbc={list(self.pbc)}). The plan is static: build a new STOperatorTensorProd "
                "for these data."
            )

    def _build_filters(
        self, backend_name: str, dtype: FloatDType, device: Any, tables: Any = None
    ) -> TensorProdFilters:
        # A tensor product builds no tables with its plan: ``tables`` is None.
        return build_tensor_prod_filters(self, backend_name, dtype, device)

    def _values(self, data: Data, filters: STFilters) -> ScatteringValues:
        return tensor_prod_values(data, self, cast(TensorProdFilters, filters))  # by _check_data

    def _layout(self) -> Layout:
        return tensor_prod_layout(self)

    def _check_reference(self, reference: _NormReference, n_batch: int, dtype: Any) -> None:
        check_tensor_prod_reference(reference, self, n_batch, dtype)

    def _self_reference(self, backend: Backend, buffer: Any, layout: Layout) -> _NormReference:
        return self_reference(
            backend,
            buffer,
            layout,
            tensor_prod_auto_scales(self),
            int(np.prod(self.L, dtype=np.int64)),
            self.ref_batch_mean,
            tensor_prod_blocks(self),
        )

    def _content(self) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        """Schema version 5."""
        content = {
            "geometry": self.geometry._geometry,
            "blocks": [g._geometry for g in self.block_geometries],
            "J": list(self.J),
            "L": list(self.L),
            "lowpass": self.lowpass,
            "lowpass_blocks": list(self.lowpass_blocks),
            "kernel_size": self.kernel_size,
            "spatial_shape": list(self.spatial_shape),
            "pbc": list(self.pbc),
            "n_channels": self.n_channels,
            "norm": self.norm,
            "ref_batch_mean": self.ref_batch_mean,
            "standardize": self.standardize,
        }
        return content, {}


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
    stacklevel: int = 1,
) -> TensorProdPlan:
    """The :class:`TensorProdPlan` of a ``DataTensorProd``, with one J and one L per block
    (M-DATA-TP, M-DEF-TP: a list, an entry ``None`` giving the default of the block). The values
    of ``data`` are not read. ``lowpass`` is ``"ST"`` (STL's default) or ``None`` (M-LP);
    ``cross_matrix``, ``norm``, ``ref_batch_mean`` and ``standardize`` are those of
    :func:`build_plan`. The static description of the loop, its index tables, crop widths and
    mean denominators, is built here too (:attr:`TensorProdPlan.loop`).

    Warns once, as STL does, for each block whose sizes are not divisible by 2^J_a: it has no
    low-pass filter (M-LP, OP-34); and once for the plan when a border is too large for a block
    and is reduced (M-BORDER-TP, OP-13). ``stacklevel`` places the warnings, as for
    ``warnings.warn``, from the caller of this function: 1 for that caller, 2 for its caller."""
    if not isinstance(data, Data) or data._geometry != "tensor_prod":
        raise TypeError(
            f"the statistics of a tensor product take a DataTensorProd, got "
            f"{type(data).__name__}; for 1D, 2D and HEALPix data, use build_plan."
        )
    if isinstance(lowpass, bool) or lowpass not in LOWPASS_TP:
        raise ValueError(
            f'lowpass={lowpass!r}: pass "ST" (the low-pass filter of each block, the default) '
            'or None (maths.md M-LP); ESTEL 1.0.0 has no tensor-product spectrum, so "PS" is '
            "not available."
        )
    _check_options(norm, ref_batch_mean, standardize)
    # The low-pass flags (M-LP, OP-34), with STL's warning, come with the plans of the blocks.
    wavelets = build_tensor_prod_wavelet_plan(
        cast(Any, data), J, L, kernel_size=kernel_size, lowpass=lowpass, stacklevel=stacklevel + 1
    )
    matrix = _cross_matrix(cross_matrix, data.n_channels)
    pairs = channel_pairs(matrix)
    # The crops of the blocks warn for each border too large for a block (OP-13): the plan
    # gives the first of these warnings, once, as the planar plan does (mean_weights).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loop = build_tensor_prod_loop(
            wavelets,
            tuple({(c, c) for c in pairs.auto} | set(pairs.cross)),
            pairs.ordered,
            data.n_channels,
        )
    if caught:
        warnings.warn(caught[0].message, stacklevel=stacklevel + 1)
    flags = list(wavelets.lowpass_blocks)
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
        pairs=pairs,
        norm=norm,
        ref_batch_mean=bool(ref_batch_mean),
        standardize=bool(standardize),
        loop=loop,
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


# ---------------------------------------------------------------------- helpers


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


def _mean_family(key: Any) -> str:
    return "moments" if key == "moments" else str(key[0])


def _check_empty(
    means: Mapping[Any, tuple[str, tuple[History, ...], int]], weights: Mapping[Any, MeanWeights]
) -> None:
    """Raise if a whole family has no mean with a valid pixel; otherwise warn once, listing the
    means that have none, whose coefficients are NaN (M-MEAN, OP-3). One advice for every
    geometry (the amendment to decision 2 on #167)."""
    empty = [key for key in means if key not in weights]
    if not empty:
        return
    advice = (
        "Use a smaller J or data with fewer missing values: the mask grows at each layer, and, on "
        "1D and 2D data that are not periodic, the border crop grows with depth (pbc=True if "
        "they are periodic)."
    )
    for family, name in _FAMILIES.items():
        members = [key for key in means if _mean_family(key) == family]
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
