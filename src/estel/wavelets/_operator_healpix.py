"""The static part and the arrays of the wavelet operator of HEALPix data (Part II, §12.5;
maths.md §4; task T5.5, issue #166).

:class:`WaveletPlanHealpix` is a :class:`~estel.wavelets._operator.WaveletPlanSingleDomain`,
beside the ``WaveletPlanCartesian`` of 1D and 2D data, and :class:`WaveletFiltersHealpix` is a
``WaveletFilters``, beside ``WaveletFiltersCartesian``: ``WaveletOperatorSingleDomain`` and the
``STPlan`` reach them through the same hooks as the Cartesian ones (``_build_filters``,
``_conv_tables``, ``_down_tables``, ``border``, ``bank``), with no branch on the geometry at the
call sites; ``build_wavelet_plan`` chooses the plan once, when it is built.

**What is static, and where** (issue #166, decisions 2 and 7). The plan holds only what is small
and static: the depth and the ``cell_ids`` of the data, the levels and their numbers of pixels,
and the masks and reweighting maps of M-MASK-HEALPIX (one family for the field and one per scale of
the first layer, as M-DOWN-MASK). The tables of T5.1 (``stencils_healpix``,
``decimation_healpix``) are built in NumPy float64 when the filters are built, one set per level,
cast at once to the operator's dtype (indices int32), and the float64 arrays are dropped: at
``nside`` 128 the stencils of one level take 900 MiB in float64 (#168). The kernel of M-WAV-HEALPIX
is folded into the stencils (:func:`_fold_kernel`; task T5.7): each row reads its distinct
pixels once, as real pairs. With missing values the
plan builds them first, for the support test and the decimated fractions of the masks, and
returns them beside the plan, as a value (:class:`PlanTablesHealpix`; task T5.7): the
constructor of the operator passes them to the builder of its filters, which reads them instead
of building them again, so the tables of a level are built once in an operator. The plan holds
none of them, and nothing is shared between operators.

**The rules of HEALPix** (maths.md M-MASK-HEALPIX, M-DEF-HEALPIX): no border (β = 0 on the full sky
and on a partial sky, OP-50); the support of a pixel is what its interpolation reads, above 10⁻¹²
for at least one gauge (OP-110); the reweighting 1 / (1 - f) with 1 - f not below 10⁻⁸;
τ = 0.33; J = max(1, depth - 1) by default, at most depth + 1 (the last level of ``nside`` 1),
L = 4 gauges, K = 5.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend import _registry
from ..data._base import _StaticMask
from ..data._healpix import ConvTablesHealpix, DataHealpix, DownTablesHealpix, _StaticCellIds
from ..data._masks import (
    SUPPORT_THRESHOLD_HEALPIX,
    BoolMap,
    MaskMaps,
    RealMap,
    build_mask_maps_with,
)
from . import _tables_healpix
from ._banks_healpix import GAUGES_HEALPIX, _check_kernel_size, kernel_weights_healpix
from ._checks import _check_n_orientations
from ._operator import (
    History,
    WaveletFilters,
    WaveletPlanSingleDomain,
    _read_only,
    _register_filters,
    downsample_maps_of,
)
from ._tables_healpix import (
    DecimationHealpix,
    StencilsHealpix,
    _check_depth,
    decimation_healpix,
    stencils_healpix,
)

WAVELETS_HEALPIX = ("healpix",)
"""The wavelets of HEALPix data in 1.0.0: the kernel of maths.md M-WAV-HEALPIX, STL's
``"HealpixWavelet"`` (and its ``"Morlet"``, which gives the same kernel)."""


@dataclass(frozen=True, eq=False)
class WaveletPlanHealpix(WaveletPlanSingleDomain):
    """The static part of the wavelet operator of HEALPix data: a
    :class:`WaveletPlanSingleDomain` whose spatial shape is ``(Npix,)``, one pixel axis, plus the
    depth and the pixels of the data.
    Built by :func:`build_wavelet_plan_healpix`."""

    depth: int = 0
    """log₂ ``nside`` of the data at full resolution."""
    cell_ids: _StaticCellIds | None = None
    """The NESTED pixels of a partial sky, or ``None`` on the full sky."""

    def bank(self) -> NDArray[np.complex128]:
        """The kernel of M-WAV-HEALPIX on the flattened stencil, ``(K², )``, complex128: one kernel
        for every gauge and every level, the orientations being the gauges of the stencils."""
        return kernel_weights_healpix(kernel_size=self.kernel_size)

    def border(self, history: History, level: int) -> int:
        """No border on HEALPix, on the full sky or a partial sky (M-MASK-HEALPIX, OP-50)."""
        return 0

    def level_cell_ids(self, level: int) -> NDArray[np.int64] | None:
        """The NESTED pixels of ``level``, ascending, or ``None`` on the full sky."""
        return None if self.cell_ids is None else self.cell_ids.at_level(level)

    def _build_filters(
        self, backend_name: str, dtype: FloatDType, device: Any, tables: Any = None
    ) -> WaveletFiltersHealpix:
        return build_wavelet_filters_healpix(self, backend_name, dtype, device, tables)


class PlanTablesHealpix(NamedTuple):
    """The NumPy tables, float64, that the masks of a plan were built from, by level (task T5.7,
    issue #168): returned beside the plan by :func:`build_wavelet_plan_with_tables_healpix`, a
    value that the constructor of the operator passes to the builder of its filters, so that the
    tables of a level are built once in an operator instead of twice. Read-only mappings; the
    plan does not hold them."""

    stencils: Mapping[int, StencilsHealpix]
    """The stencils of each scale level, as :func:`stencils_healpix` gives them."""
    decimation: Mapping[int, DecimationHealpix]
    """The decimation from each level the masks downsample, as :func:`decimation_healpix`."""


@dataclass(frozen=True, eq=False)
class WaveletFiltersHealpix(WaveletFilters):
    """The arrays of the wavelet operator of HEALPix data, on the operator's backend and dtype:
    the tables of every level, and the maps of each downsampling step. A pytree whose leaves are
    its arrays, in the order of its fields (``_operator._register_filters``)."""

    conv: tuple[ConvTablesHealpix, ...]
    """For each level of the plan, the tables of ``DataHealpix._conv``."""
    down: tuple[DownTablesHealpix, ...]
    """For each level but the last, the tables of the decimation to the next level."""
    field_down: dict[int, tuple[Any, Any]]
    """As ``WaveletFilters.field_down``: per level d but the last, the ``(valid, reweight)``
    maps of the field from d to d + 1, or no entry without missing values."""
    modulus_down: dict[tuple[int, int], tuple[Any, Any]]
    """As ``WaveletFilters.modulus_down``: per ``(d, j)``, the maps of first-layer data
    convolved at scale j from d to d + 1, or no entry without missing values."""

    def _conv_tables(self, level: int) -> ConvTablesHealpix:
        """The tables of ``_conv`` at ``level``."""
        return self.conv[level]

    def _down_tables(self, level: int) -> DownTablesHealpix:
        """The tables of the step from ``level`` to ``level + 1``."""
        return self.down[level]

    @property
    def _any_array(self) -> Any:
        """An array of the operator's dtype, which gives the filters' backend and dtype."""
        return self.conv[0].kernel


_register_filters(WaveletFiltersHealpix)


# ---------------------------------------------------------------------- the plan


def default_scales_healpix(depth: int) -> int:
    """J = max(1, ⌊log₂ nside⌋ - 1) = max(1, depth - 1) (maths.md M-DEF-HEALPIX)."""
    return max(1, depth - 1)


def build_wavelet_plan_healpix(
    data: DataHealpix,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    L: int | None = None,  # noqa: N803
    wavelet: str | None = None,
    *,
    kernel_size: int,
) -> WaveletPlanHealpix:
    """The plan of :func:`build_wavelet_plan_with_tables_healpix`, without its tables."""
    return build_wavelet_plan_with_tables_healpix(data, J, L, wavelet, kernel_size=kernel_size)[0]


def build_wavelet_plan_with_tables_healpix(
    data: DataHealpix,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    L: int | None = None,  # noqa: N803
    wavelet: str | None = None,
    *,
    kernel_size: int,
) -> tuple[WaveletPlanHealpix, PlanTablesHealpix | None]:
    """The :class:`WaveletPlanHealpix` of ``data`` with ``J`` scales and ``L`` gauges: J by
    default max(1, depth - 1), L = 4, K = 5 (maths.md M-DEF-HEALPIX); its values are not read. With
    missing values, also the tables its masks were built from (:class:`PlanTablesHealpix`),
    for the filters of the same operator; ``None`` without.

    Raises when J is not an integer in 1 … depth + 1 (each scale one level coarser, down to
    ``nside`` 1: the bound STL accepts), when L is not an integer ≥ 1, when K is not odd and
    ≥ 3, when the depth exceeds the int32 limit of the tables, and when ``wavelet`` is neither
    ``None`` nor ``"healpix"``.
    """
    if wavelet not in (None, *WAVELETS_HEALPIX):
        raise ValueError(
            f"wavelet={wavelet!r}: DataHealpix accepts wavelet=None (the default, its wavelet) "
            'or "healpix", the kernel of maths.md M-WAV-HEALPIX, its only wavelet in ESTEL 1.0.0; '
            '"bump_steerable" is for 1D and 2D data. Pass None or "healpix".'
        )
    depth = _check_depth(data.depth)  # the int32 limit of the tables
    n_scales = _check_scales_healpix(J, depth)
    n_gauges = _check_n_orientations(GAUGES_HEALPIX if L is None else L)
    size = _check_kernel_size(kernel_size)
    static_ids = data._cell_ids
    scale_levels = tuple(range(n_scales))
    levels = scale_levels
    level_shapes = tuple((data._n_pixels(d),) for d in levels)
    masks, tables = None, None
    if data.mask is not None:
        masks, tables = _masks_healpix(
            data.mask, depth, static_ids, scale_levels, n_gauges, size, data.pbc
        )
    static_mask = None if data.mask is None else _StaticMask(data.mask)
    plan = WaveletPlanHealpix(
        geometry=DataHealpix,
        J=n_scales,
        L=n_gauges,
        kernel_size=size,
        ndim=1,
        spatial_shape=level_shapes[0],
        pbc=data.pbc,
        mask=None if data.mask is None else _read_only(data.mask),
        nan_threshold=DataHealpix._nan_threshold,
        scale_levels=scale_levels,
        levels=levels,
        level_shapes=level_shapes,
        lowpass_realization="smoothing_cascade",
        lowpass_level=n_scales,
        masks=masks,
        _key=(DataHealpix, n_scales, n_gauges, size, depth, static_ids, static_mask, "healpix"),
        depth=depth,
        cell_ids=static_ids,
    )
    return plan, tables


def _check_scales_healpix(value: object, depth: int) -> int:
    """J as an int, by default max(1, depth - 1) (M-DEF-HEALPIX), in 1 … depth + 1."""
    if value is None:
        return default_scales_healpix(depth)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"J={value!r}: the number of scales is an integer ≥ 1, such as J=2, or None for its "
            f"default max(1, depth - 1) = {default_scales_healpix(depth)} (maths.md M-DEF-HEALPIX)."
        )
    n = int(value)
    if not 1 <= n <= depth + 1:
        raise ValueError(
            f"J={n}: a HEALPix map of depth {depth} (nside {1 << depth}) has the levels 0 … "
            f"{depth}, one per scale, the last of nside 1, so 1 ≤ J ≤ {depth + 1}. The default "
            f"is max(1, depth - 1) = {default_scales_healpix(depth)} (maths.md M-DEF-HEALPIX)."
        )
    return n


def _masks_healpix(
    mask: BoolMap,
    depth: int,
    static_ids: _StaticCellIds | None,
    scale_levels: tuple[int, ...],
    n_gauges: int,
    size: int,
    pbc: bool,
) -> tuple[MaskMaps, PlanTablesHealpix]:
    """The masks and reweighting maps of M-MASK-HEALPIX, from the tables of each level, each built
    once, and those tables, for the filters of the operator (module docstring)."""

    def ids(level: int) -> NDArray[np.int64] | None:
        return None if static_ids is None else static_ids.at_level(level)

    @functools.cache
    def built_stencils(level: int) -> StencilsHealpix:
        return stencils_healpix(ids(level), depth - level, L=n_gauges, kernel_size=size)

    @functools.cache
    def decimation(level: int) -> DecimationHealpix:
        return decimation_healpix(ids(level), depth - level)

    def stencils(level: int) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        found = built_stencils(level)
        shape = (n_gauges, found.index.shape[0] // n_gauges, -1)  # (L, Npix, K² · 4)
        return found.index.reshape(shape), found.weight.reshape(shape)

    def support(level: int, invalid: BoolMap) -> BoolMap:
        index, weight = stencils(level)
        bad = invalid.astype(np.float64)
        batch = _tables_healpix._STENCIL_BATCH

        def touched(rows: slice) -> BoolMap:
            # One gauge and one batch of pixels at a time bound the temporaries (task T5.7).
            found: BoolMap = np.logical_or.reduce(
                [
                    (bad[index[g, rows]] * weight[g, rows]).sum(axis=-1) > SUPPORT_THRESHOLD_HEALPIX
                    for g in range(n_gauges)
                ]
            )
            return found

        return np.concatenate(
            [touched(slice(begin, begin + batch)) for begin in range(0, index.shape[1], batch)]
        )

    def decimate(level: int, values: RealMap) -> RealMap:
        tables = decimation(level)
        gathered = values[tables.index[:, 0, 0]] * tables.weight[:, 0, 0]
        summed = np.bincount(tables.segment_ids, weights=gathered, minlength=tables.num_segments)
        return np.asarray(summed, dtype=np.float64)

    masks = build_mask_maps_with(
        mask, scale_levels, support, decimate, threshold=DataHealpix._nan_threshold, pbc=pbc
    )
    # The levels the masks read (every scale level, and the steps between them), each from the
    # cache: nothing is built here.
    tables = PlanTablesHealpix(
        stencils=MappingProxyType({d: built_stencils(d) for d in scale_levels}),
        decimation=MappingProxyType({d: decimation(d) for d in range(max(scale_levels))}),
    )
    return masks, tables


# ---------------------------------------------------------------------- the arrays


def build_wavelet_filters_healpix(
    plan: WaveletPlanHealpix,
    backend_name: str,
    dtype: FloatDType,
    device: Any,
    tables: PlanTablesHealpix | None = None,
) -> WaveletFiltersHealpix:
    """The arrays of ``plan`` on the backend ``backend_name``: the tables of T5.1 of each level,
    built in float64 and cast once to ``dtype`` on ``device``, indices int32 (Part I, §3.1,
    rules 4 and 7; OP-46); the float64 tables are not kept. ``tables``, the tables the masks of
    ``plan`` were built from (:class:`PlanTablesHealpix`), are read instead of built again;
    without them (``from_statistics``), every table is built here."""
    backend = _registry.backend_named(backend_name)
    handed = tables or PlanTablesHealpix(stencils={}, decimation={})
    kernel = plan.bank()
    # The kernel is folded into the tables (_fold_kernel): the product with it is the identity.
    identity = backend.asarray(np.eye(2), dtype, device)

    def conv(level: int) -> ConvTablesHealpix:
        found = handed.stencils.get(level) or stencils_healpix(
            plan.level_cell_ids(level),
            plan.depth - level,
            L=plan.L,
            kernel_size=plan.kernel_size,
        )
        index, weight = _fold_kernel(found, kernel, dtype)
        rows = (plan.L, plan.level_shapes[level][0])
        return ConvTablesHealpix(
            index=backend.asarray(index.reshape(*rows, *index.shape[1:]), "int32", device),
            weight=backend.asarray(weight.reshape(*rows, *weight.shape[1:]), dtype, device),
            kernel=identity,
        )

    def down(level: int) -> DownTablesHealpix:
        found = handed.decimation.get(level) or decimation_healpix(
            plan.level_cell_ids(level), plan.depth - level
        )
        return DownTablesHealpix(
            index=backend.asarray(found.index, "int32", device),
            weight=backend.asarray(found.weight, dtype, device),
            segment_ids=backend.asarray(found.segment_ids, "int32", device),
        )

    field_down, modulus_down = downsample_maps_of(plan, backend_name, dtype, device)
    return WaveletFiltersHealpix(
        conv=tuple(conv(d) for d in plan.levels),
        down=tuple(down(d) for d in plan.wavelet_levels[:-1]),
        field_down=field_down,
        modulus_down=modulus_down,
    )


_FOLD_BATCH = 1 << 14
"""Rows of the stencil tables folded at once by :func:`_fold_kernel`: bounds its temporaries."""


def _fold_kernel(
    tables: StencilsHealpix, kernel: NDArray[np.complex128], dtype: str = "float64"
) -> tuple[NDArray[np.int32], NDArray[np.floating[Any]]]:
    """The stencils of a level with the kernel folded in (task T5.7, trial 4): for each row
    (gauge, pixel), the K² · 4 entries merged into the distinct pixels they read, each with
    Σ W[p] · w[p, m] over its entries, as real pairs (Part I, §3.1, rule 6). Returns ``index``
    of shape ``(L · Npix, 1, M)`` and ``weight`` of shape ``(L · Npix, 2, M)``, M the largest
    number of distinct pixels of a row, the rows padded with weight 0: the real part on the
    first entry of the second axis of ``weight``, the imaginary part on the second, both
    reading the one row of ``index``, which ``gather_weighted`` gathers once (finding P5-01 of
    the final review). It then gives the convolution as real pairs at once; the values change
    by rounding only (M-CONV-HEALPIX).
    The sums are made in float64, by batches of :data:`_FOLD_BATCH` rows, each stored in
    ``dtype``, the operator's, then joined: the weights are cast once."""
    rows, q, m = tables.index.shape
    flat_index = tables.index.reshape(rows, q * m)
    flat_weight = tables.weight.reshape(rows, q * m)
    batches = [slice(begin, begin + _FOLD_BATCH) for begin in range(0, rows, _FOLD_BATCH)]

    def distinct(part: slice) -> NDArray[np.intp]:
        by_pixel = np.sort(flat_index[part], axis=1)
        counts: NDArray[np.intp] = (by_pixel[:, 1:] != by_pixel[:, :-1]).sum(axis=1) + 1
        return counts

    width = max(int(distinct(part).max()) for part in batches)
    folded = np.repeat(kernel, m)  # W[p] for each entry (p, m)

    def fold(part: slice) -> tuple[NDArray[np.int32], NDArray[np.floating[Any]]]:
        index, weight = flat_index[part], flat_weight[part]
        n = index.shape[0]
        order = np.argsort(index, axis=1, kind="stable")
        by_pixel = np.take_along_axis(index, order, axis=1)
        first = np.concatenate(
            [np.ones((n, 1), dtype=bool), by_pixel[:, 1:] != by_pixel[:, :-1]], axis=1
        )
        slots = (np.arange(n)[:, None] * width + np.cumsum(first, axis=1) - 1).ravel()
        values = (np.take_along_axis(weight, order, axis=1) * folded[order]).ravel()
        # The entries of a slot share their pixel: their sum over their count is that pixel,
        # exactly (int32 values in float64); an empty slot, padding, reads pixel 0.
        counts = np.maximum(np.bincount(slots, minlength=n * width), 1)
        pixels = np.bincount(slots, by_pixel.ravel().astype(np.float64), n * width) / counts
        pairs = np.stack(
            [np.bincount(slots, values.real, n * width), np.bincount(slots, values.imag, n * width)]
        ).astype(dtype)  # (2, n · width)
        found_index = pixels.astype(np.int32).reshape(n, 1, width)
        return found_index, pairs.reshape(2, n, width).transpose(1, 0, 2)

    parts = [fold(part) for part in batches]
    out_weight = np.concatenate([weight for _, weight in parts])
    out_index = np.concatenate([index for index, _ in parts])
    return out_index, out_weight
