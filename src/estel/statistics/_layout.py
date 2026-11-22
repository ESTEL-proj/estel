"""The compact coefficient layout (Part III, §19.2, §19.4 to §19.6, §19.9; Part IV, §22, "Storage
of the compact layout"; task T2.15).

The statistics are stored in one real buffer ``(Nb, n_floats)``: every coefficient STL computes,
and only those, each real entry as one float (mean, var, the auto S1 and S2, the diagonal of S4),
each complex entry as two consecutive floats, (re, im). STL's conjugate duplicates are kept
(Part III, §19.8, option a). Nothing is NaN: absent coefficients are not stored.

:class:`Layout` describes the buffer, statically: for each family its full-grid view
(``_grid.FamilyGrid``: named axes, coordinates, where coefficients exist), and its
:class:`Segment` s, one per family and level (the output resolution of the loop), whose named
index table gives the coordinates of every entry. The families follow one another in the order
mean, var, S1 … S4; within a family the segments follow the levels, and within a segment the
entries follow the order of the full grid. A layout is hashable: it is static data of the pytree
of ``STStatistics`` and a static argument of JAX.

Everything that moves values between two representations is a static map built here, in NumPy,
from layouts and the plan (Part III, §19.6, §19.9), applied by :func:`apply` with a few generic
primitives (``take``, ``gather_weighted``, ``concat``) of the L0 protocol, so that it is
differentiable to any order on every backend:

* :func:`loop_gather`: from the sums of the scattering loop (``scattering._plan.loop_sums``) to
  the buffer of :func:`base_layout`;
* :class:`FloatMap`: from one buffer to another, family by family, as a copy, a gather
  (``select``, the normalisation) or a sparse real matrix in padded CSR form (the reductions),
  built from a complex linear map on the coefficients by :func:`make_map`.

The full-grid views (``scatter_to_grid``) are built only for :meth:`STStatistics.full_grid`,
never inside the computation (review rule X5; Part III, §19.5).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend._protocol import Backend
from ..io._schema import SCHEMA_VERSION
from ..scattering._plan import STPlan, loop_sums
from ._arrays import constant, dtype_name
from ._grid import FamilyGrid, base_grids

Level = tuple[int, ...]
"""The level of a segment: ``(d,)`` for 1D and 2D data, a multi-index in tensor products (T3.2);
``()`` when the entries of a segment come from several levels (after ``scale_cosine``)."""

_FROZEN = "the arrays of a layout are static: build a new layout instead."


def _frozen(array: NDArray[Any], dtype: Any) -> NDArray[Any]:
    out = np.array(array, dtype=dtype)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class Segment:
    """The entries of one family at one level, in one contiguous range of the buffer (Part III,
    §19.4)."""

    family: str
    """``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``."""
    level: Level
    """The output level of its entries (:data:`Level`)."""
    index_names: tuple[str, ...]
    """The names of the columns of :attr:`index`: the axes of the family's full grid."""
    index: NDArray[np.int32]
    """``(n_entries, len(index_names))``: the position of each entry along each axis of the
    family's full grid, read-only."""
    is_complex: NDArray[np.bool_]
    """``(n_entries,)``: whether each entry takes two floats (re, im) or one, read-only."""
    offset: int
    """The first float of the segment in the buffer."""
    n_floats: int
    """The number of floats of the segment: one per entry, two per complex entry."""

    def __post_init__(self) -> None:
        flags = _frozen(self.is_complex, bool).reshape(-1)
        index = _frozen(self.index, np.int32)
        if index.size != flags.shape[0] * len(self.index_names):
            raise ValueError(f"{self.family}: {index.shape} index, {flags.shape[0]} flags.")
        index = index.reshape(flags.shape[0], len(self.index_names))
        if self.n_floats != index.shape[0] + int(flags.sum()):
            raise ValueError(f"{self.family}: n_floats does not count the entries.")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "is_complex", flags)

    @property
    def n_entries(self) -> int:
        return int(self.index.shape[0])

    def _key(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.level,
            self.index_names,
            self.index.shape,
            self.index.tobytes(),
            self.is_complex.tobytes(),
            self.offset,
            self.n_floats,
        )


class FamilyValues(NamedTuple):
    """The entries of one family, in any order, from which :func:`build_layout` makes its
    segments."""

    grid: FamilyGrid
    """The full-grid view: axes, coordinates and dtype; its ``present`` is recomputed."""
    positions: NDArray[np.int64]
    """``(n, len(grid.axes))``: the position of each entry along each axis."""
    is_complex: NDArray[np.bool_]
    """``(n,)``."""
    levels: tuple[Level, ...]
    """The distinct levels :attr:`level_ids` refers to."""
    level_ids: NDArray[np.int64]
    """``(n,)``: the level of each entry, as a position in :attr:`levels`."""


class FamilyEntries(NamedTuple):
    """The entries of one family of a layout, in the order of the buffer."""

    grid: FamilyGrid
    positions: NDArray[np.int64]
    """``(n, n_axes)``."""
    is_complex: NDArray[np.bool_]
    """``(n,)``."""
    first: NDArray[np.int64]
    """``(n,)``: the float of each entry's real part in the buffer (its imaginary part, if it
    has one, follows)."""
    levels: tuple[Level, ...]
    level_ids: NDArray[np.int64]
    start: int
    """The first float of the family."""
    stop: int
    """One past its last float."""
    by_position: EntryIndex
    """The entries by their flat position in the full grid, to find them (:meth:`find`)."""

    @property
    def n_entries(self) -> int:
        return int(self.first.shape[0])

    def find(self, flat: NDArray[np.int64]) -> NDArray[np.int64]:
        """The entry at each flat position (C order) of the full grid, -1 where none."""
        return self.by_position.find(flat)

    def values(self) -> FamilyValues:
        return FamilyValues(self.grid, self.positions, self.is_complex, self.levels, self.level_ids)


def flat_positions(positions: NDArray[np.int64], shape: tuple[int, ...]) -> NDArray[np.int64]:
    """The positions ``(n, len(shape))`` flattened in C order."""
    if not shape:
        return np.zeros(positions.shape[0], dtype=np.int64)
    if positions.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    flat: NDArray[np.int64] = np.ravel_multi_index(tuple(positions.T), shape).astype(np.int64)
    return flat


class EntryIndex(NamedTuple):
    """Distinct flat positions, sorted, to find them with ``searchsorted``: tables the size of
    the entries, never of the full grid (whose flattened size can be in the billions for the
    tensor products)."""

    keys: NDArray[np.int64]
    """The flat positions, sorted."""
    ids: NDArray[np.int64]
    """``ids[i]``: the position of ``keys[i]`` among the flat positions indexed."""

    def find(self, flat: NDArray[np.int64]) -> NDArray[np.int64]:
        """The position among the indexed values of each value of ``flat``, -1 where none."""
        query = np.asarray(flat, dtype=np.int64)
        if self.keys.shape[0] == 0:
            return np.full(query.shape, -1, dtype=np.int64)
        at = np.minimum(np.searchsorted(self.keys, query), self.keys.shape[0] - 1)
        found: NDArray[np.int64] = np.where(self.keys[at] == query, self.ids[at], -1)
        return found.astype(np.int64)


def entry_index(flat: NDArray[np.int64]) -> EntryIndex:
    """The :class:`EntryIndex` of ``flat``, whose values are distinct."""
    order = np.argsort(flat, kind="stable").astype(np.int64)
    return EntryIndex(np.asarray(flat, dtype=np.int64)[order], order)


def mask_of(flat: NDArray[np.int64], shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """The boolean array of ``shape``, True at the flat positions ``flat`` (C order). Built
    from runs, with temporaries the size of ``flat``: the only array of the full grid's size
    is the result, one byte per position."""
    size = int(np.prod(shape))
    keys = np.unique(np.asarray(flat, dtype=np.int64))
    before = np.diff(np.concatenate(([-1], keys))) - 1  # positions False before each key
    tail = size - (int(keys[-1]) + 1 if keys.shape[0] else 0)
    counts = np.concatenate((np.stack([before, np.ones_like(before)], axis=1).ravel(), [tail]))
    values = np.concatenate((np.tile(np.array([False, True]), keys.shape[0]), [False]))
    mask: NDArray[np.bool_] = np.repeat(values, counts).reshape(shape)
    return mask


@dataclass(frozen=True, eq=False)
class Layout:
    """The compact layout of a buffer of statistics (Part III, §19.4): hashable, so that it
    is static data of a pytree and a static argument of JAX."""

    grids: tuple[FamilyGrid, ...]
    """The full-grid view of each family, in the order of the buffer."""
    segments: tuple[Segment, ...]
    """The segments, in the order of the buffer."""
    n_floats: int
    """The length of the buffer."""
    schema_version: int = SCHEMA_VERSION
    """The version of the schema of ``estel.io`` that describes this layout."""

    def __post_init__(self) -> None:
        offset = 0
        names = [g.name for g in self.grids]
        order = [names.index(s.family) for s in self.segments]
        if order != sorted(order):
            raise ValueError("the segments of a layout follow its families, one after the other.")
        for segment in self.segments:
            if segment.offset != offset:
                raise ValueError(f"{segment.family}: segments must be contiguous from float 0.")
            offset += segment.n_floats
        if offset != self.n_floats:
            raise ValueError(f"the segments hold {offset} floats, not {self.n_floats}.")
        for grid in self.grids:
            positions = self.entries(grid.name).positions
            distinct = np.unique(flat_positions(positions, grid.shape)).shape[0]
            inside = not grid.axes or bool(np.all(grid.present[tuple(positions.T)]))
            if not (distinct == positions.shape[0] == np.count_nonzero(grid.present) and inside):
                raise ValueError(f"{grid.name}: the segments do not match the full-grid view.")

    @cached_property
    def _hash_key(self) -> tuple[Any, ...]:
        return (
            tuple(g._key() for g in self.grids),
            tuple(s._key() for s in self.segments),
            self.n_floats,
            self.schema_version,
        )

    def __hash__(self) -> int:
        return hash(self._hash_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return self is other or self._hash_key == other._hash_key

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.grids)

    def grid(self, name: str) -> FamilyGrid:
        return self.grids[self.names.index(name)]

    def full_grid_shape(self, name: str) -> tuple[int, ...]:
        """The shape of the full-grid view of ``name``, after the batch axis (STL's shape)."""
        return self.grid(name).shape

    def entries(self, name: str) -> FamilyEntries:
        """The entries of family ``name``, in the order of the buffer."""
        return self._entries[name]

    @cached_property
    def _entries(self) -> dict[str, FamilyEntries]:
        found: dict[str, FamilyEntries] = {}
        for grid in self.grids:
            segments = [s for s in self.segments if s.family == grid.name]
            levels = tuple(s.level for s in segments)
            positions = (
                np.concatenate([s.index for s in segments]).astype(np.int64)
                if segments
                else np.zeros((0, len(grid.axes)), dtype=np.int64)
            )
            flags = (
                np.concatenate([s.is_complex for s in segments])
                if segments
                else np.zeros(0, dtype=bool)
            )
            start = segments[0].offset if segments else _family_start(self, grid.name)
            sizes = 1 + flags.astype(np.int64)
            first = start + np.cumsum(sizes) - sizes
            level_ids = np.repeat(
                np.arange(len(segments), dtype=np.int64), [s.n_entries for s in segments]
            )
            found = {
                **found,
                grid.name: FamilyEntries(
                    grid=grid,
                    positions=positions,
                    is_complex=flags,
                    first=first.astype(np.int64),
                    levels=levels,
                    level_ids=level_ids,
                    start=int(start),
                    stop=int(start + sizes.sum()),
                    by_position=entry_index(flat_positions(positions, grid.shape)),
                ),
            }
        return found


def _family_start(layout: Layout, name: str) -> int:
    """The first float of a family without entries: the end of the families before it."""
    names = layout.names
    before = {n for n in names[: names.index(name)]}
    return sum(s.n_floats for s in layout.segments if s.family in before)


def build_layout(families: Sequence[FamilyValues]) -> tuple[Layout, tuple[NDArray[np.int64], ...]]:
    """The layout of ``families``, in this order, and for each family the order of its entries
    in the buffer (``order[e]``: the position in ``positions`` of its e-th entry). Within a
    family the entries are sorted by level, then in the order of the full grid."""
    segments: list[Segment] = []
    grids: list[FamilyGrid] = []
    orders: list[NDArray[np.int64]] = []
    offset = 0
    for family in families:
        grid = family.grid
        shape = tuple(len(c) for c in grid.coords)
        flat = flat_positions(family.positions, shape)
        if np.unique(flat).shape[0] != flat.shape[0]:
            raise ValueError(f"{grid.name}: two entries at one position of the full grid.")
        used = sorted({family.levels[int(i)] for i in np.unique(family.level_ids)})
        rank = np.array([used.index(lv) if lv in used else -1 for lv in family.levels])
        entry_rank = rank[family.level_ids] if flat.shape[0] else np.zeros(0, dtype=np.int64)
        order = np.lexsort((flat, entry_rank)).astype(np.int64)
        for r, level in enumerate(used):
            chosen = order[entry_rank[order] == r]
            flags = family.is_complex[chosen]
            n_floats = int(chosen.shape[0] + flags.sum())
            segments.append(
                Segment(
                    family=grid.name,
                    level=tuple(int(d) for d in level),
                    index_names=grid.axes,
                    index=family.positions[chosen],
                    is_complex=flags,
                    offset=offset,
                    n_floats=n_floats,
                )
            )
            offset += n_floats
        grids.append(grid.replace(present=mask_of(flat, shape)))
        orders.append(order)
    return Layout(grids=tuple(grids), segments=tuple(segments), n_floats=offset), tuple(orders)


# ---------------------------------------------------------------------- the base layout


@functools.lru_cache(maxsize=64)
def base_layout(plan: STPlan) -> Layout:
    """The layout of the statistics the scattering loop computes, from the plan alone
    (``_grid.base_grids``): real entries are mean, var, the auto S1 and S2, and the diagonal of
    S4 (c1 = c2, j1 = j2, l1 = l2); every entry is at the level of the scale of its last
    convolution (the level of j, or of j3)."""
    levels = tuple((int(d),) for d in plan.levels)
    level_of = {int(d): i for i, d in enumerate(plan.levels)}
    scale_level = np.array([level_of[int(plan.scale_levels[j])] for j in plan.scales])
    families: list[FamilyValues] = []
    for grid in base_grids(plan):
        positions = np.argwhere(grid.present).astype(np.int64)
        p = positions.T
        if grid.name in ("mean", "var"):
            flags = np.zeros(positions.shape[0], dtype=bool)
            level_ids = np.full(positions.shape[0], level_of[int(plan.levels[0])])
        elif grid.name in ("S1", "S2"):
            flags = p[0] != p[1]
            level_ids = scale_level[p[2]]
        elif grid.name == "S3":
            flags = np.ones(positions.shape[0], dtype=bool)
            level_ids = scale_level[p[3]]
        else:
            flags = ~((p[0] == p[1]) & (p[2] == p[3]) & (p[5] == p[6]))
            level_ids = scale_level[p[4]]
        families.append(FamilyValues(grid, positions, flags, levels, level_ids.astype(np.int64)))
    layout, _ = build_layout(families)
    return layout


def _carried_shape(plan: STPlan, family: str) -> tuple[int, ...]:
    """The axes a sum of the loop carries after the batch axis (``loop_sums``)."""
    if family in ("mean", "var"):
        return (plan.n_channels,)
    return (plan.L,) * {"S1": 1, "S2": 1, "S3": 2, "S4": 3}[family]


@functools.lru_cache(maxsize=64)
def loop_gather(plan: STPlan) -> tuple[NDArray[np.int64], int]:
    """The static gather from the sums of the scattering loop to the buffer of
    :func:`base_layout` (Part III, §19.3, §19.9), and the number of floats the loop produces.

    The loop divides its sums level by level and concatenates them, in the order of
    ``loop_sums``, as real floats: a complex sum as (re, im) pairs, its carried axes in C order.
    Entry ``k`` of the result is the float of the loop that goes to float ``k`` of the buffer.
    A real entry the loop computes complex (the diagonal of S4) keeps its real part."""
    layout = base_layout(plan)
    raw: dict[str, list[tuple[NDArray[np.int64], NDArray[np.int64], bool]]] = {
        name: [] for name in layout.names
    }
    offset = 0
    for item in loop_sums(plan):
        carried = _carried_shape(plan, item.family)
        grid = layout.grid(item.family)
        tail = np.argwhere(np.ones(carried, dtype=bool)).astype(np.int64)
        positions = np.concatenate(
            [
                np.broadcast_to(np.array(item.key, dtype=np.int64), (tail.shape[0], len(item.key))),
                tail,
            ],
            axis=1,
        )
        width = 2 if item.is_complex else 1
        raw[item.family].append(
            (
                flat_positions(positions, grid.shape),
                offset + width * np.arange(tail.shape[0], dtype=np.int64),
                item.is_complex,
            )
        )
        offset += width * tail.shape[0]
    pieces: list[NDArray[np.int64]] = []
    for name in layout.names:
        entries = layout.entries(name)
        grid = entries.grid
        found = raw[name]
        flat = np.concatenate([f for f, _, _ in found]) if found else np.zeros(0, np.int64)
        first = np.concatenate([r for _, r, _ in found]) if found else np.zeros(0, np.int64)
        complex_ = (
            np.concatenate([np.full(f.shape[0], c) for f, _, c in found])
            if found
            else np.zeros(0, bool)
        )
        where = entry_index(flat).find(flat_positions(entries.positions, grid.shape))
        if (where < 0).any() or (entries.is_complex & ~complex_[where]).any():
            raise ValueError(
                f"{name}: the loop does not compute every coefficient of the layout; "
                "scattering._plan.loop_sums and statistics._layout.base_layout disagree."
            )
        re = first[where]
        pieces.append(_interleave(re, entries.is_complex))
    return np.concatenate(pieces), offset


def _interleave(re: NDArray[np.int64], is_complex: NDArray[np.bool_]) -> NDArray[np.int64]:
    """The floats of entries whose real part is at ``re``: ``re`` for a real entry, ``re`` and
    ``re + 1`` for a complex one, in order."""
    counts = 1 + is_complex.astype(np.int64)
    within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    out: NDArray[np.int64] = np.repeat(re, counts) + within
    return out


# ---------------------------------------------------------------------- maps between buffers


class _Part(NamedTuple):
    """The floats of one family of the output of a :class:`FloatMap`: a slice of the input
    (``columns`` is ``None``), a gather (``columns`` ``(P,)``), or a sparse real matrix in
    padded CSR form (``columns`` and ``weights`` ``(P, M)``, ``gather_weighted``)."""

    start: int
    stop: int
    columns: NDArray[np.int64] | None
    weights: NDArray[np.float64] | None


@dataclass(frozen=True, eq=False)
class FloatMap:
    """A static linear map from a buffer of one layout to a buffer of :attr:`layout` (Part
    III, §19.6), one part per output family."""

    layout: Layout
    parts: tuple[_Part, ...]


class Copy(NamedTuple):
    """A family kept as it is."""

    name: str


class Gather(NamedTuple):
    """A family whose entries are entries of the input family of the same name: ``sources[i]``
    is the input entry of output entry ``i`` (``select``, the normalisation)."""

    values: FamilyValues
    sources: NDArray[np.int64]


class Linear(NamedTuple):
    """A family whose entries are complex linear combinations of the entries of the input family
    of the same name (the reductions): output position ``out[k]`` (flat, on ``grid``) receives
    ``weights[k]`` times input entry ``sources[k]``. The output entries are where ``grid.present``
    is True; contributions elsewhere are dropped. An output entry is real when every one of its
    contributions is a real weight times a real entry; its level is that of its contributions
    when they share one, ``()`` otherwise."""

    grid: FamilyGrid
    out: NDArray[np.int64]
    sources: NDArray[np.int64]
    weights: NDArray[np.complex128]


def make_map(layout: Layout, specs: Sequence[Copy | Gather | Linear]) -> FloatMap:
    """The :class:`FloatMap` from a buffer of ``layout`` that makes the families of ``specs``,
    in this order."""
    values: list[FamilyValues] = []
    linear: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]]] = []
    for spec in specs:
        if isinstance(spec, Copy):
            values.append(layout.entries(spec.name).values())
            linear.append(_EMPTY)
        elif isinstance(spec, Gather):
            values.append(spec.values)
            linear.append(_EMPTY)
        else:
            found, contributions = _linear_values(layout.entries(spec.grid.name), spec)
            values.append(found)
            linear.append(contributions)
    out_layout, orders = build_layout(values)
    parts: list[_Part] = []
    for spec, order, contributions, grid in zip(
        specs, orders, linear, out_layout.grids, strict=True
    ):
        source = layout.entries(grid.name)
        target = out_layout.entries(grid.name)
        if isinstance(spec, Copy):
            if not np.array_equal(order, np.arange(order.shape[0])):
                raise ValueError(f"{spec.name}: a copied family keeps its order.")
            parts.append(_Part(source.start, source.stop, None, None))
        elif isinstance(spec, Gather):
            parts.append(
                _Part(0, 0, _interleave(source.first[spec.sources[order]], target.is_complex), None)
            )
        else:
            parts.append(_sparse(source, target, order, *contributions))
    return FloatMap(layout=out_layout, parts=tuple(parts))


_EMPTY: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]] = (
    np.zeros(0, np.int64),
    np.zeros(0, np.int64),
    np.zeros(0, np.complex128),
)


def _linear_values(
    source: FamilyEntries, spec: Linear
) -> tuple[FamilyValues, tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]]]:
    """The output entries of ``spec`` (complex flags, levels) and its contributions, with the
    output entries numbered in the order of ``np.argwhere(spec.grid.present)``."""
    grid = spec.grid
    positions = np.argwhere(grid.present).astype(np.int64)
    out_id = entry_index(flat_positions(positions, grid.shape)).find(spec.out)
    keep = (out_id >= 0) & (spec.weights != 0)
    out_id, sources, weights = out_id[keep], spec.sources[keep], spec.weights[keep]
    n = positions.shape[0]
    complex_part = source.is_complex[sources] | (weights.imag != 0)
    flags = np.bincount(out_id, weights=complex_part.astype(float), minlength=n) > 0
    levels = (*source.levels, ())
    level_ids = np.full(n, len(levels) - 1, dtype=np.int64)
    if out_id.shape[0]:
        order = np.argsort(out_id, kind="stable")
        sorted_out = out_id[order]
        heads = np.flatnonzero(np.r_[True, sorted_out[1:] != sorted_out[:-1]])
        their = source.level_ids[sources[order]]
        low = np.minimum.reduceat(their, heads)
        high = np.maximum.reduceat(their, heads)
        same = entry_index(sorted_out[heads]).find(np.arange(n, dtype=np.int64))
        has = same >= 0
        chosen = np.where(low == high, low, len(levels) - 1)
        level_ids = np.where(has, chosen[np.maximum(same, 0)], level_ids)
    return (
        FamilyValues(grid, positions, flags, levels, level_ids),
        (out_id, sources, weights),
    )


def _sparse(
    source: FamilyEntries,
    target: FamilyEntries,
    order: NDArray[np.int64],
    out_id: NDArray[np.int64],
    sources: NDArray[np.int64],
    weights: NDArray[np.complex128],
) -> _Part:
    """The padded CSR matrix on floats of a complex linear map on entries: with w = a + ib, a
    real part receives a·re - b·im, an imaginary part b·re + a·im, the parts a real entry does
    not have left out. The padding of a row reads the row's own first column with weight 0, so
    that a non-finite value elsewhere in the buffer cannot reach it."""
    where = np.argsort(order)  # output entry (argwhere order) -> its position in the buffer
    row = target.first[where[out_id]] - target.start
    col = source.first[sources]
    out_complex = target.is_complex[where[out_id]]
    in_complex = source.is_complex[sources]
    a, b = weights.real, weights.imag
    groups = (
        (row, col, a, np.ones_like(in_complex)),
        (row, col + 1, -b, in_complex),
        (row + 1, col, b, out_complex),
        (row + 1, col + 1, a, out_complex & in_complex),
    )
    rows = np.concatenate([g[0][g[3] & (g[2] != 0)] for g in groups])
    cols = np.concatenate([g[1][g[3] & (g[2] != 0)] for g in groups])
    values = np.concatenate([g[2][g[3] & (g[2] != 0)] for g in groups])
    n_rows = target.stop - target.start
    counts = np.bincount(rows, minlength=n_rows)
    width = max(int(counts.max(initial=0)), 1)
    ranked = np.argsort(rows, kind="stable")
    rows, cols, values = rows[ranked], cols[ranked], values[ranked]
    heads = np.cumsum(counts) - counts
    slot = np.arange(rows.shape[0]) - np.repeat(heads, counts)
    flat = rows * width + slot
    placed = np.rint(np.bincount(flat, weights=cols.astype(float), minlength=n_rows * width))
    weight = np.bincount(flat, weights=values, minlength=n_rows * width)
    # A row without entries (none in practice) reads the first column of the part.
    fallback = int(cols[0]) if cols.shape[0] else 0
    own = np.where(counts > 0, np.append(cols, fallback)[heads], fallback)
    used = np.arange(width)[None, :] < counts[:, None]
    index = np.where(used, placed.astype(np.int64).reshape(n_rows, width), own[:, None])
    return _Part(0, 0, index.astype(np.int64), weight.astype(np.float64).reshape(n_rows, width))


def apply(backend: Backend, buffer: Any, transform: FloatMap) -> Any:
    """The buffer ``(Nb, n)`` mapped by ``transform``: ``(Nb, transform.layout.n_floats)``."""
    xp = backend.xp
    pieces: list[Any] = []
    for part in transform.parts:
        if part.columns is None:
            pieces.append(buffer[:, part.start : part.stop])
        elif part.weights is None:
            pieces.append(xp.take(buffer, constant(backend, part.columns, buffer, "int32"), axis=1))
        else:
            rows, width = part.columns.shape
            index = constant(backend, part.columns.reshape(rows, 1, width), buffer, "int32")
            weight = constant(
                backend, part.weights.reshape(rows, 1, width), buffer, dtype_name(buffer)
            )
            mapped = backend.gather_weighted(buffer, 1, index, weight)
            pieces.append(xp.reshape(mapped, (int(buffer.shape[0]), rows)))
    return pieces[0] if len(pieces) == 1 else xp.concat(pieces, axis=1)


# ---------------------------------------------------------------------- reading the buffer


def with_zero(backend: Backend, buffer: Any) -> Any:
    """The buffer with one more float, 0, at position ``n_floats``: the imaginary part read for
    a real entry."""
    xp = backend.xp
    return xp.concat([buffer, xp.zeros_like(buffer[:, :1])], axis=1)


def real_and_imaginary(layout: Layout, name: str) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """For the entries of ``name`` in the order of the buffer, the float of the real part and
    that of the imaginary part, ``layout.n_floats`` (the 0 of :func:`with_zero`) for a real
    entry."""
    entries = layout.entries(name)
    imaginary = np.where(entries.is_complex, entries.first + 1, layout.n_floats)
    return entries.first, imaginary


def full_grid(backend: Backend, buffer: Any, layout: Layout, name: str) -> Any:
    """The full-grid view of family ``name`` (Part III, §19.5): STL's shape, NaN where no
    coefficient exists, complex for S1 to S4 with imaginary part 0 for the real entries. Built
    with ``scatter_to_grid``, for :meth:`STStatistics.full_grid` only."""
    xp = backend.xp
    entries = layout.entries(name)
    grid = entries.grid
    re, im = real_and_imaginary(layout, name)
    n_batch = int(buffer.shape[0])
    extended = with_zero(backend, buffer)

    def scatter(floats: NDArray[np.int64]) -> Any:
        values = xp.take(extended, constant(backend, floats, buffer, "int32"), axis=1)
        if not grid.axes:
            if entries.n_entries:
                return xp.reshape(values, (n_batch,))
            return xp.full((n_batch,), float("nan"), dtype=buffer.dtype)
        if not entries.n_entries:
            return xp.full((n_batch, *grid.shape), float("nan"), dtype=buffer.dtype)
        index = [constant(backend, column, buffer, "int32") for column in entries.positions.T]
        return backend.scatter_to_grid(values, index, grid.shape)

    if not grid.is_complex:
        return scatter(re)
    return backend.real_pairs_to_complex(xp.stack([scatter(re), scatter(im)], axis=-1))


# ---------------------------------------------------------------------- storage


def payload(layout: Layout) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """The arrays and the JSON description of ``layout``, for ``estel.io``."""
    arrays: dict[str, NDArray[Any]] = {}
    for i, segment in enumerate(layout.segments):
        arrays = {
            **arrays,
            f"segment__{i}__index": np.asarray(segment.index),
            f"segment__{i}__is_complex": np.asarray(segment.is_complex),
        }
    content = {
        "families": [
            {
                "name": g.name,
                "axes": list(g.axes),
                "coords": [list(c) for c in g.coords],
                "is_complex": g.is_complex,
            }
            for g in layout.grids
        ],
        "segments": [
            {
                "family": s.family,
                "level": list(s.level),
                "index_names": list(s.index_names),
                "offset": s.offset,
                "n_floats": s.n_floats,
            }
            for s in layout.segments
        ],
        "n_floats": layout.n_floats,
    }
    return arrays, content


def from_payload(arrays: dict[str, NDArray[Any]], content: dict[str, Any]) -> Layout:
    """The layout written by :func:`payload`."""
    segments = tuple(
        Segment(
            family=s["family"],
            level=tuple(int(d) for d in s["level"]),
            index_names=tuple(s["index_names"]),
            index=arrays[f"segment__{i}__index"],
            is_complex=arrays[f"segment__{i}__is_complex"].astype(bool),
            offset=int(s["offset"]),
            n_floats=int(s["n_floats"]),
        )
        for i, s in enumerate(content["segments"])
    )
    grids: list[FamilyGrid] = []
    for family in content["families"]:
        coords = tuple(tuple(int(i) for i in c) for c in family["coords"])
        shape = tuple(len(c) for c in coords)
        found = [s.index.astype(np.int64) for s in segments if s.family == family["name"]]
        positions = np.concatenate(found) if found else np.zeros((0, len(shape)), np.int64)
        present = mask_of(flat_positions(positions, shape), shape)
        grids.append(
            FamilyGrid(
                name=family["name"],
                axes=tuple(family["axes"]),
                coords=coords,
                present=present,
                is_complex=bool(family["is_complex"]),
            )
        )
    return Layout(grids=tuple(grids), segments=segments, n_floats=int(content["n_floats"]))
