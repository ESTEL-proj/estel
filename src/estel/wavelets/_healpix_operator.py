"""The static part and the arrays of the wavelet operator of HEALPix data (Part II, §12.5;
maths.md §4; task T5.5, issue #166).

:class:`HealpixWaveletPlan` is a :class:`~estel.wavelets._operator.WaveletPlan`, and
:class:`HealpixWaveletFilters` plays the part of ``WaveletFilters``: ``WaveletOperator`` and
the ``STPlan`` reach them through the same hooks as the planar ones (``_build_filters``,
``_conv_tables``, ``_down_tables``, ``border``, ``bank``), with no branch on the geometry at the
call sites; ``build_wavelet_plan`` chooses the plan once, when it is built.

**What is static, and where** (issue #166, decisions 2 and 7). The plan holds only what is small
and static: the depth and the ``cell_ids`` of the data, the levels and their numbers of pixels,
and the masks and reweighting maps of M-MASK-HPX (one family for the field and one per scale of
the first layer, as M-DOWN-MASK). The tables of T5.1 (``healpix_stencils``,
``healpix_decimation``) are built in NumPy float64 when the filters are built, one set per level,
cast at once to the operator's dtype (indices int32), and the float64 arrays are dropped: at
``nside`` 128 the stencils of one level take 900 MiB in float64 (#168). The kernel of M-WAV-HPX
is folded into the stencils (:func:`_fold_kernel`; task T5.7): each row reads its distinct
pixels once, as real pairs. With missing values the
plan builds them first, for the support test and the decimated fractions of the masks, and hands
them over to the filters of the operator, which take them instead of building them again
(:class:`_TablesHandOver`; task T5.7): the tables of a level are built once in an operator.
Nothing is shared between operators.

**The rules of HEALPix** (maths.md M-MASK-HPX, M-DEF-HPX): no border (β = 0 on the full sky and
on a partial sky, OP-50); the support of a pixel is what its interpolation reads, above 10⁻¹²
for at least one gauge (OP-110); the reweighting 1 / (1 - f) with 1 - f not below 10⁻⁸;
τ = 0.33; J = max(1, depth - 1) by default, at most depth + 1 (the last level of ``nside`` 1),
L = 4 gauges, K = 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend import _registry
from ..data._base import _StaticMask
from ..data._healpix import DataHealpix, HealpixConvTables, HealpixDownTables, _StaticCellIds
from ..data._masks import (
    HEALPIX_SUPPORT_THRESHOLD,
    BoolMap,
    MaskMaps,
    RealMap,
    build_mask_maps_with,
)
from . import _healpix_geometry
from ._banks import _check_n_orientations
from ._healpix_bank import HEALPIX_GAUGES, _check_kernel_size, healpix_kernel_weights
from ._healpix_geometry import (
    HealpixDecimation,
    HealpixStencils,
    _check_depth,
    healpix_decimation,
    healpix_stencils,
)
from ._operator import History, WaveletPlan, _read_only, downsample_maps_of

HEALPIX_WAVELETS = ("healpix",)
"""The wavelets of HEALPix data in 1.0.0: the kernel of maths.md M-WAV-HPX, STL's
``"HealpixWavelet"`` (and its ``"Morlet"``, which gives the same kernel)."""


@dataclass(frozen=True, eq=False)
class HealpixWaveletPlan(WaveletPlan):
    """The static part of the wavelet operator of HEALPix data: a :class:`WaveletPlan` whose
    spatial shape is ``(Npix,)``, one pixel axis, plus the depth and the pixels of the data.
    Built by :func:`build_healpix_wavelet_plan`."""

    depth: int = 0
    """log₂ ``nside`` of the data at full resolution."""
    cell_ids: _StaticCellIds | None = None
    """The NESTED pixels of a partial sky, or ``None`` on the full sky."""
    _tables: _TablesHandOver | None = field(default=None, repr=False)
    """With missing values, the tables the masks were built from, until the filters of the
    operator take them (:class:`_TablesHandOver`); not part of the plan's identity."""

    def bank(self) -> NDArray[np.complex128]:
        """The kernel of M-WAV-HPX on the flattened stencil, ``(K², )``, complex128: one kernel
        for every gauge and every level, the orientations being the gauges of the stencils."""
        return healpix_kernel_weights(kernel_size=self.kernel_size)

    def border(self, history: History, level: int) -> int:
        """No border on HEALPix, on the full sky or a partial sky (M-MASK-HPX, OP-50)."""
        return 0

    def level_cell_ids(self, level: int) -> NDArray[np.int64] | None:
        """The NESTED pixels of ``level``, ascending, or ``None`` on the full sky."""
        return None if self.cell_ids is None else self.cell_ids.at_level(level)

    def _build_filters(
        self, backend_name: str, dtype: FloatDType, device: Any
    ) -> HealpixWaveletFilters:
        return build_healpix_wavelet_filters(self, backend_name, dtype, device)


class _TablesHandOver:
    """The NumPy tables of each level that the masks of a plan were built from, held until the
    filters of the operator take them, once each: the tables of a level are built once in an
    operator instead of twice (task T5.7, issue #168). Emptied as the filters are built, so a
    plan keeps no tables once its operator has its filters; filters built again from the same
    plan (``from_statistics``) build their own."""

    __slots__ = ("decimation", "stencils")

    def __init__(self) -> None:
        self.stencils: dict[int, HealpixStencils] = {}
        self.decimation: dict[int, HealpixDecimation] = {}


class HealpixWaveletFilters(NamedTuple):
    """The arrays of the wavelet operator of HEALPix data, on the operator's backend and dtype:
    the tables of every level, and the maps of each downsampling step, as ``WaveletFilters``.
    A named tuple of arrays, so a pytree."""

    conv: tuple[HealpixConvTables, ...]
    """For each level of the plan, the tables of ``DataHealpix._conv``."""
    down: tuple[HealpixDownTables, ...]
    """For each level but the last, the tables of the decimation to the next level."""
    field_down: dict[int, tuple[Any, Any]]
    """As ``WaveletFilters.field_down``: per level d but the last, the ``(valid, reweight)``
    maps of the field from d to d + 1, or no entry without missing values."""
    modulus_down: dict[tuple[int, int], tuple[Any, Any]]
    """As ``WaveletFilters.modulus_down``: per ``(d, j)``, the maps of first-layer data
    convolved at scale j from d to d + 1, or no entry without missing values."""

    def _conv_tables(self, level: int) -> HealpixConvTables:
        """The tables of ``_conv`` at ``level``."""
        return self.conv[level]

    def _down_tables(self, level: int) -> HealpixDownTables:
        """The tables of the step from ``level`` to ``level + 1``."""
        return self.down[level]

    @property
    def _any_array(self) -> Any:
        """An array of the operator's dtype, which gives the filters' backend and dtype."""
        return self.conv[0].kernel


# ---------------------------------------------------------------------- the plan


def default_healpix_scales(depth: int) -> int:
    """J = max(1, ⌊log₂ nside⌋ - 1) = max(1, depth - 1) (maths.md M-DEF-HPX)."""
    return max(1, depth - 1)


def build_healpix_wavelet_plan(
    data: DataHealpix,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    L: int | None = None,  # noqa: N803
    wavelet: str | None = None,
    *,
    kernel_size: int,
) -> HealpixWaveletPlan:
    """The :class:`HealpixWaveletPlan` of ``data`` with ``J`` scales and ``L`` gauges: J by
    default max(1, depth - 1), L = 4, K = 5 (maths.md M-DEF-HPX); its values are not read.

    Raises when J is not an integer in 1 … depth + 1 (each scale one level coarser, down to
    ``nside`` 1: the bound STL accepts), when L is not an integer ≥ 1, when K is not odd and
    ≥ 3, when the depth exceeds the int32 limit of the tables, and when ``wavelet`` is neither
    ``None`` nor ``"healpix"``.
    """
    if wavelet not in (None, *HEALPIX_WAVELETS):
        raise ValueError(
            f"wavelet={wavelet!r}: DataHealpix accepts wavelet=None (the default, its wavelet) "
            'or "healpix", the kernel of maths.md M-WAV-HPX, its only wavelet in ESTEL 1.0.0; '
            '"bump_steerable" is for 1D and 2D data. Pass None or "healpix".'
        )
    depth = _check_depth(data.depth)  # the int32 limit of the tables
    n_scales = _check_healpix_scales(J, depth)
    n_gauges = _check_n_orientations(HEALPIX_GAUGES if L is None else L)
    size = _check_kernel_size(kernel_size)
    static_ids = data._cell_ids
    scale_levels = tuple(range(n_scales))
    levels = scale_levels
    level_shapes = tuple((data._n_pixels(d),) for d in levels)
    masks, tables = None, None
    if data.mask is not None:
        tables = _TablesHandOver()
        masks = _healpix_masks(
            data.mask, depth, static_ids, scale_levels, n_gauges, size, data.pbc, tables
        )
    static_mask = None if data.mask is None else _StaticMask(data.mask)
    return HealpixWaveletPlan(
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
        _tables=tables,
    )


def _check_healpix_scales(value: object, depth: int) -> int:
    """J as an int, by default max(1, depth - 1) (M-DEF-HPX), in 1 … depth + 1."""
    if value is None:
        return default_healpix_scales(depth)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"J={value!r}: the number of scales is an integer ≥ 1, such as J=2, or None for its "
            f"default max(1, depth - 1) = {default_healpix_scales(depth)} (maths.md M-DEF-HPX)."
        )
    n = int(value)
    if not 1 <= n <= depth + 1:
        raise ValueError(
            f"J={n}: a HEALPix map of depth {depth} (nside {1 << depth}) has the levels 0 … "
            f"{depth}, one per scale, the last of nside 1, so 1 ≤ J ≤ {depth + 1}. The default "
            f"is max(1, depth - 1) = {default_healpix_scales(depth)} (maths.md M-DEF-HPX)."
        )
    return n


def _healpix_masks(
    mask: BoolMap,
    depth: int,
    static_ids: _StaticCellIds | None,
    scale_levels: tuple[int, ...],
    n_gauges: int,
    size: int,
    pbc: bool,
    tables: _TablesHandOver,
) -> MaskMaps:
    """The masks and reweighting maps of M-MASK-HPX, from the tables of each level, built once
    and kept in ``tables`` for the filters (module docstring)."""

    def ids(level: int) -> NDArray[np.int64] | None:
        return None if static_ids is None else static_ids.at_level(level)

    def stencils(level: int) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        if level not in tables.stencils:
            tables.stencils[level] = healpix_stencils(
                ids(level), depth - level, L=n_gauges, kernel_size=size
            )
        found = tables.stencils[level]
        shape = (n_gauges, found.index.shape[0] // n_gauges, -1)  # (L, Npix, K² · 4)
        return found.index.reshape(shape), found.weight.reshape(shape)

    def decimation(level: int) -> HealpixDecimation:
        if level not in tables.decimation:
            tables.decimation[level] = healpix_decimation(ids(level), depth - level)
        return tables.decimation[level]

    def support(level: int, invalid: BoolMap) -> BoolMap:
        index, weight = stencils(level)
        bad = invalid.astype(np.float64)
        touched: BoolMap = np.empty(index.shape[1], dtype=bool)
        # One gauge and one batch of pixels at a time bound the temporaries (task T5.7).
        batch = _healpix_geometry._STENCIL_BATCH
        for begin in range(0, touched.size, batch):
            rows = slice(begin, begin + batch)
            touched[rows] = np.logical_or.reduce(
                [
                    (bad[index[g, rows]] * weight[g, rows]).sum(axis=-1) > HEALPIX_SUPPORT_THRESHOLD
                    for g in range(n_gauges)
                ]
            )
        return touched

    def decimate(level: int, values: RealMap) -> RealMap:
        tables = decimation(level)
        gathered = values[tables.index[:, 0, 0]] * tables.weight[:, 0, 0]
        summed = np.bincount(tables.segment_ids, weights=gathered, minlength=tables.num_segments)
        return np.asarray(summed, dtype=np.float64)

    return build_mask_maps_with(
        mask, scale_levels, support, decimate, threshold=DataHealpix._nan_threshold, pbc=pbc
    )


# ---------------------------------------------------------------------- the arrays


def build_healpix_wavelet_filters(
    plan: HealpixWaveletPlan, backend_name: str, dtype: FloatDType, device: Any
) -> HealpixWaveletFilters:
    """The arrays of ``plan`` on the backend ``backend_name``: the tables of T5.1 of each level,
    built in float64 and cast once to ``dtype`` on ``device``, indices int32 (Part I, §3.1,
    rules 4 and 7; OP-46); the float64 tables are not kept. The tables the masks of the plan
    were built from are taken from it, once (:class:`_TablesHandOver`)."""
    backend = _registry.backend_named(backend_name)
    handed = plan._tables or _TablesHandOver()
    kernel = plan.bank()
    # The kernel is folded into the tables (_fold_kernel): the product with it is the identity.
    identity = backend.asarray(np.eye(2), dtype, device)

    def conv(level: int) -> HealpixConvTables:
        tables = handed.stencils.pop(level, None) or healpix_stencils(
            plan.level_cell_ids(level),
            plan.depth - level,
            L=plan.L,
            kernel_size=plan.kernel_size,
        )
        index, weight = _fold_kernel(tables, kernel, dtype)
        shape = (plan.L, plan.level_shapes[level][0], *index.shape[1:])
        return HealpixConvTables(
            index=backend.asarray(index.reshape(shape), "int32", device),
            weight=backend.asarray(weight.reshape(shape), dtype, device),
            kernel=identity,
        )

    def down(level: int) -> HealpixDownTables:
        tables = handed.decimation.pop(level, None) or healpix_decimation(
            plan.level_cell_ids(level), plan.depth - level
        )
        return HealpixDownTables(
            index=backend.asarray(tables.index, "int32", device),
            weight=backend.asarray(tables.weight, dtype, device),
            segment_ids=backend.asarray(tables.segment_ids, "int32", device),
        )

    field_down, modulus_down = downsample_maps_of(plan, backend_name, dtype, device)
    filters = HealpixWaveletFilters(
        conv=tuple(conv(d) for d in plan.levels),
        down=tuple(down(d) for d in plan.wavelet_levels[:-1]),
        field_down=field_down,
        modulus_down=modulus_down,
    )
    handed.stencils.clear()  # what the filters did not take: the plan keeps no tables
    handed.decimation.clear()
    return filters


_FOLD_BATCH = 1 << 14
"""Rows of the stencil tables folded at once by :func:`_fold_kernel`: bounds its temporaries."""


def _fold_kernel(
    tables: HealpixStencils, kernel: NDArray[np.complex128], dtype: str = "float64"
) -> tuple[NDArray[np.int32], NDArray[np.floating[Any]]]:
    """The stencils of a level with the kernel folded in (task T5.7, trial 4): for each row
    (gauge, pixel), the K² · 4 entries merged into the distinct pixels they read, each with
    Σ W[p] · w[p, m] over its entries, as real pairs (Part I, §3.1, rule 6). Returns ``index``
    and ``weight`` of shape ``(L · Npix, 2, M)``, M the largest number of distinct pixels of a
    row, the rows padded with weight 0: the real part on the first entry of the second axis,
    the imaginary part on the second, the same pixels on both. ``gather_weighted`` then gives
    the convolution as real pairs at once; the values change by rounding only (M-CONV-HPX).
    The sums are made in float64, by batches of :data:`_FOLD_BATCH` rows, and stored in
    ``dtype``, the operator's: the weights are cast once."""
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
    out_index = np.zeros((rows, 2, width), dtype=np.int32)
    out_weight = np.zeros((rows, 2, width), dtype=dtype)
    for part in batches:
        index, weight = flat_index[part], flat_weight[part]
        n = index.shape[0]
        order = np.argsort(index, axis=1, kind="stable")
        by_pixel = np.take_along_axis(index, order, axis=1)
        first = np.ones(by_pixel.shape, dtype=bool)
        first[:, 1:] = by_pixel[:, 1:] != by_pixel[:, :-1]
        slots = (np.arange(n)[:, None] * width + np.cumsum(first, axis=1) - 1).ravel()
        values = (np.take_along_axis(weight, order, axis=1) * folded[order]).ravel()
        pixels = np.zeros(n * width, dtype=np.int32)
        pixels[slots] = by_pixel.ravel()
        out_index[part] = pixels.reshape(n, 1, width)
        out_weight[part, 0] = np.bincount(slots, values.real, n * width).reshape(n, width)
        out_weight[part, 1] = np.bincount(slots, values.imag, n * width).reshape(n, width)
    return out_index, out_weight
