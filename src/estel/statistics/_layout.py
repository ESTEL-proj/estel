"""The compact coefficient layout (Part III, §19.2, §19.4 to §19.6, §19.9; Part IV, §22, "Storage
of the compact layout" and "The compact layout stored by chunks"; tasks T2.15, T3.7).

The statistics are stored in one real buffer ``(Nb, n_floats)``: every coefficient STL computes,
and only those, each real entry as one float (mean, var, the auto S1 and S2, the diagonal of S4),
each complex entry as two consecutive floats, (re, im). STL's conjugate duplicates are kept
(Part III, §19.8, option a). Nothing is NaN: absent coefficients are not stored.

:class:`Layout` describes the buffer, statically, by chunks (T3.7): for each family its full-grid
view (``_grid.FamilyGrid``: named axes, coordinates, and its chunks, one per channel pair and
scale combination, each holding the dense array of its orientations, with the kind of each
chunk saying which of its entries take one float), and its :class:`Segment` s, one per family
and level (the output resolution of the loop), each a range of the family's chunks. The families
follow one another in the order mean, var, S1 … S4; within a family the segments follow the
levels, and within a segment the entries follow the order of the full grid: the chunks in the
order of their axes, then their orientations. The layout keeps one small row per chunk, never a
table per coefficient; the coordinates of an entry are computed on demand (:meth:`Layout.entries`).
A layout is hashable: it is static data of the pytree of ``STStatistics`` and a static argument
of JAX.

Everything that moves values between two representations is a static map built here, in NumPy,
from layouts and the plan (Part III, §19.6, §19.9), applied with a few generic primitives
(``take``, ``gather_weighted``, ``segment_sum``, ``concat``) of the L0 protocol, so that it is
differentiable to any order on every backend. Each acts chunk by chunk, with tables that grow
with the number of chunks and of orientations, never with the number of coefficients:

* :func:`loop_gather`: from the sums of the scattering loop (``scattering._plan.loop_sums``),
  each of which is a chunk, to the buffer of :func:`base_layout`;
* :class:`FloatMap`: from one buffer to another, family by family: a copy, or the chunks of a
  family read as dense arrays ``(Nb, n_chunks, n_orientations, 2)`` (re, im; :func:`unpack`),
  gathered or combined across chunks, transformed on their orientations, and written back
  (:func:`pack`). ``select``, the normalisation and the reductions are such maps.

The full-grid views (``scatter_to_grid``) are built only for :meth:`STStatistics.full_grid`,
never inside the computation (review rule X5; Part III, §19.5).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
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


@dataclass(frozen=True, eq=False)
class Segment:
    """The chunks of one family at one level, in one contiguous range of the buffer (Part III,
    §19.4)."""

    family: str
    """``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``."""
    level: Level
    """The output level of its entries (:data:`Level`)."""
    index_names: tuple[str, ...]
    """The axes of the family's full grid: the columns of :attr:`chunks`, then the
    orientations."""
    chunks: NDArray[np.int32]
    """``(n_chunks, n_chunk_axes)``: the channel pair and scales of each chunk, a read-only view
    of the rows ``first_chunk`` … of the family's ``FamilyGrid.chunks``."""
    orientations: tuple[int, ...]
    """The dense shape of the orientations of every chunk."""
    offset: int
    """The first float of the segment in the buffer."""
    n_floats: int
    """The number of floats of the segment."""
    first_chunk: int
    """The position of its first chunk among the chunks of its family."""

    @property
    def n_chunks(self) -> int:
        return int(self.chunks.shape[0])

    def _key(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.level,
            self.index_names,
            self.first_chunk,
            self.n_chunks,
            self.orientations,
            self.offset,
            self.n_floats,
        )


class FamilyChunks(NamedTuple):
    """The chunks of one family, in any order, from which :func:`build_layout` makes its
    segments."""

    grid: FamilyGrid
    """The family: axes, coordinates, chunks, their kinds."""
    levels: tuple[Level, ...]
    """The distinct levels :attr:`level_ids` refers to."""
    level_ids: NDArray[np.int64]
    """``(n_chunks,)``: the level of each chunk, as a position in :attr:`levels`."""


class FamilyEntries(NamedTuple):
    """The entries of one family of a layout, in the order of the buffer, computed on demand
    from its chunks (:meth:`Layout.entries`): tables the size of the coefficients, for the
    tests and the plots, never held by a layout nor used by the maps."""

    grid: FamilyGrid
    positions: NDArray[np.int64]
    """``(n, n_axes)``."""
    is_complex: NDArray[np.bool_]
    """``(n,)``."""
    first: NDArray[np.int64]
    """``(n,)``: the float of each entry's real part in the buffer (its imaginary part, if it
    has one, follows)."""
    start: int
    """The first float of the family."""
    stop: int
    """One past its last float."""

    @property
    def n_entries(self) -> int:
        return int(self.first.shape[0])


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
    what is indexed (chunks), never of the full grid."""

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


def chunk_index(grid: FamilyGrid) -> EntryIndex:
    """The chunks of ``grid`` by their flat position along the chunk axes."""
    return entry_index(flat_positions(grid.chunks.astype(np.int64), grid.chunk_shape))


def orientation_positions(orientations: tuple[int, ...]) -> NDArray[np.int64]:
    """``(prod(orientations), len(orientations))``: every position of the orientations of a
    chunk, in C order."""
    n = int(np.prod(orientations, dtype=np.int64))
    if not orientations:
        return np.zeros((1, 0), dtype=np.int64)
    found: NDArray[np.int64] = np.stack(
        np.unravel_index(np.arange(n), orientations), axis=1
    ).astype(np.int64)
    return found


def within_chunk(real: NDArray[np.bool_]) -> NDArray[np.int64]:
    """``(n_kinds, n_orientations, 2)``: for each kind of chunk, the float of the real and of
    the imaginary part of each entry within the chunk, -1 for the imaginary part of a real
    entry."""
    counts = 1 + (~real).astype(np.int64)
    first = np.cumsum(counts, axis=1) - counts
    found: NDArray[np.int64] = np.stack([first, np.where(real, -1, first + 1)], axis=-1)
    return found.astype(np.int64)


@dataclass(frozen=True, eq=False)
class Layout:
    """The compact layout of a buffer of statistics (Part III, §19.4): hashable, so that it
    is static data of a pytree and a static argument of JAX."""

    grids: tuple[FamilyGrid, ...]
    """The full-grid view of each family, with its chunks in the order of the buffer."""
    segments: tuple[Segment, ...]
    """The segments, in the order of the buffer."""
    n_floats: int
    """The length of the buffer."""
    schema_version: int = SCHEMA_VERSION
    """The version of the schema of ``estel.io`` that describes this layout."""

    def __post_init__(self) -> None:
        names = [g.name for g in self.grids]
        order = [names.index(s.family) for s in self.segments]
        if order != sorted(order):
            raise ValueError("the segments of a layout follow its families, one after the other.")
        offset = 0
        for grid in self.grids:
            floats = grid.floats_per_kind[grid.kinds]
            chunk = 0
            for segment in (s for s in self.segments if s.family == grid.name):
                if segment.offset != offset or segment.first_chunk != chunk:
                    raise ValueError(f"{grid.name}: segments must be contiguous from float 0.")
                if segment.orientations != grid.orientations:
                    raise ValueError(f"{grid.name}: the orientations of a segment differ.")
                stop = chunk + segment.n_chunks
                if segment.n_floats != int(floats[chunk:stop].sum()):
                    raise ValueError(f"{grid.name}: n_floats does not count the chunks.")
                offset += segment.n_floats
                chunk = stop
            flat = flat_positions(grid.chunks.astype(np.int64), grid.chunk_shape)
            inside = grid.n_chunks == 0 or bool(
                (grid.chunks >= 0).all() and (grid.chunks < np.array(grid.chunk_shape)).all()
            )
            if chunk != grid.n_chunks or not inside or np.unique(flat).shape[0] != flat.shape[0]:
                raise ValueError(f"{grid.name}: the segments do not match the chunks.")
        if offset != self.n_floats:
            raise ValueError(f"the segments hold {offset} floats, not {self.n_floats}.")

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

    def span(self, name: str) -> tuple[int, int]:
        """The first float of family ``name`` and one past its last."""
        return self._spans[name]

    def chunk_starts(self, name: str) -> NDArray[np.int64]:
        """``(n_chunks,)``: the first float of each chunk of ``name`` in the buffer."""
        return self._starts[name]

    def levels(self, name: str) -> tuple[tuple[Level, ...], NDArray[np.int64]]:
        """The levels of the segments of ``name``, and the level of each of its chunks as a
        position among them."""
        segments = [s for s in self.segments if s.family == name]
        ids = np.repeat(np.arange(len(segments), dtype=np.int64), [s.n_chunks for s in segments])
        return tuple(s.level for s in segments), ids

    def chunked(self, name: str) -> FamilyChunks:
        """Family ``name`` as :func:`build_layout` takes it."""
        levels, ids = self.levels(name)
        return FamilyChunks(self.grid(name), levels, ids)

    @cached_property
    def _spans(self) -> dict[str, tuple[int, int]]:
        found: dict[str, tuple[int, int]] = {}
        start = 0
        for grid in self.grids:
            stop = start + int(grid.floats_per_kind[grid.kinds].sum())
            found = {**found, grid.name: (start, stop)}
            start = stop
        return found

    @cached_property
    def _starts(self) -> dict[str, NDArray[np.int64]]:
        found: dict[str, NDArray[np.int64]] = {}
        for grid in self.grids:
            sizes = grid.floats_per_kind[grid.kinds]
            starts = self._spans[grid.name][0] + np.cumsum(sizes) - sizes
            starts.flags.writeable = False
            found = {**found, grid.name: starts.astype(np.int64)}
        return found

    def entries(self, name: str) -> FamilyEntries:
        """The entries of family ``name``, in the order of the buffer, computed on demand from
        its chunks: for the tests and the plots, not for the maps."""
        grid = self.grid(name)
        n_o = grid.n_orientations
        tail = orientation_positions(grid.orientations)
        positions = np.concatenate(
            [
                np.repeat(grid.chunks.astype(np.int64), n_o, axis=0),
                np.tile(tail, (grid.n_chunks, 1)),
            ],
            axis=1,
        )
        is_complex = ~grid.real[grid.kinds].reshape(-1)
        sizes = 1 + is_complex.astype(np.int64)
        start, stop = self.span(name)
        first = start + np.cumsum(sizes) - sizes
        return FamilyEntries(grid, positions, is_complex, first.astype(np.int64), start, stop)


def _canonical_kinds(
    kinds: NDArray[np.int32], real: NDArray[np.bool_]
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """The kinds renumbered in the order of their first chunk, equal rows of ``real`` merged
    and unused rows dropped, so that equal families have equal tables."""
    if kinds.shape[0] == 0:
        return kinds.astype(np.int32), np.zeros((0, real.shape[1]), dtype=bool)
    rows, row_of_kind = np.unique(real, axis=0, return_inverse=True)
    merged = row_of_kind.reshape(-1)[kinds]
    used, first = np.unique(merged, return_index=True)
    rank = np.argsort(np.argsort(first))
    renumbered = rank[np.searchsorted(used, merged)]
    return renumbered.astype(np.int32), rows[used[np.argsort(first)]]


def build_layout(
    families: Sequence[FamilyChunks],
) -> tuple[Layout, tuple[NDArray[np.int64], ...]]:
    """The layout of ``families``, in this order, and for each family the order of its chunks
    in the buffer (``order[i]``: the position in ``grid.chunks`` of its i-th chunk). Within a
    family the chunks are sorted by level, then in the order of the full grid."""
    segments: list[Segment] = []
    grids: list[FamilyGrid] = []
    orders: list[NDArray[np.int64]] = []
    offset = 0
    for family in families:
        grid = family.grid
        flat = flat_positions(grid.chunks.astype(np.int64), grid.chunk_shape)
        if np.unique(flat).shape[0] != flat.shape[0]:
            raise ValueError(f"{grid.name}: two chunks at one position of the full grid.")
        used = sorted({family.levels[int(i)] for i in np.unique(family.level_ids)})
        rank = np.array([used.index(lv) if lv in used else -1 for lv in family.levels])
        chunk_rank = rank[family.level_ids] if flat.shape[0] else np.zeros(0, dtype=np.int64)
        order = np.lexsort((flat, chunk_rank)).astype(np.int64)
        kinds, real = _canonical_kinds(grid.kinds[order], grid.real)
        built = grid.replace(chunks=grid.chunks[order], kinds=kinds, real=real)
        sizes = built.floats_per_kind[built.kinds]
        ranks = chunk_rank[order]
        for r, level in enumerate(used):
            a, b = (int(i) for i in np.searchsorted(ranks, [r, r + 1]))
            n_floats = int(sizes[a:b].sum())
            segments.append(
                Segment(
                    family=grid.name,
                    level=tuple(int(d) for d in level),
                    index_names=grid.axes,
                    chunks=built.chunks[a:b],
                    orientations=built.orientations,
                    offset=offset,
                    n_floats=n_floats,
                    first_chunk=a,
                )
            )
            offset += n_floats
        grids.append(built)
        orders.append(order)
    return Layout(grids=tuple(grids), segments=tuple(segments), n_floats=offset), tuple(orders)


# ---------------------------------------------------------------------- the base layout


@functools.lru_cache(maxsize=64)
def base_layout(plan: STPlan) -> Layout:
    """The layout of the statistics the scattering loop computes, from the plan alone
    (``_grid.base_grids``): every chunk is at the level of the scale of its last convolution
    (the level of j, or of j3)."""
    levels = tuple((int(d),) for d in plan.levels)
    level_of = {int(d): i for i, d in enumerate(plan.levels)}
    scale_level = np.array([level_of[int(plan.scale_levels[j])] for j in plan.scales])
    last_scale = {"S1": 2, "S2": 2, "S3": 3, "S4": 4}
    families: list[FamilyChunks] = []
    for grid in base_grids(plan):
        if grid.name in last_scale:
            level_ids = scale_level[grid.chunks[:, last_scale[grid.name]]]
        else:
            level_ids = np.full(grid.n_chunks, level_of[int(plan.levels[0])])
        families.append(FamilyChunks(grid, levels, np.asarray(level_ids, dtype=np.int64)))
    layout, _ = build_layout(families)
    return layout


class LoopGather(NamedTuple):
    """The static gather from the sums of the scattering loop to the buffer of
    :func:`base_layout` (Part III, §19.3, §19.9), chunk by chunk: each sum of ``loop_sums`` is
    a chunk (or, for mean and var, the chunks of every channel)."""

    pieces: NDArray[np.int64]
    """``(n_pieces, 4)``: in the order of the buffer, the sum each piece reads, its first float
    in the sum, one past its last, and the row of :attr:`takes` it goes through, -1 for
    none."""
    takes: tuple[NDArray[np.int64], ...]
    """For each kind of chunk that the loop computes complex and the buffer stores with real
    entries (the diagonal of S4), the floats of the sum it keeps, the real part only of the
    real entries."""
    n_sums: int
    """The number of sums the loop produces."""


def _carried_shape(plan: STPlan, family: str) -> tuple[int, ...]:
    """The axes a sum of the loop carries after the batch axis (``loop_sums``)."""
    if family in ("mean", "var"):
        return (plan.n_channels,)
    return (plan.L,) * {"S1": 1, "S2": 1, "S3": 2, "S4": 3}[family]


@functools.lru_cache(maxsize=64)
def loop_gather(plan: STPlan) -> LoopGather:
    """The :class:`LoopGather` of ``plan``.

    The loop divides its sums level by level, each as real floats: a complex sum as (re, im)
    pairs, its carried axes in C order. The buffer is their concatenation in the order of its
    chunks; a real entry the loop computes complex (the diagonal of S4) keeps its real part."""
    layout = base_layout(plan)
    sums = loop_sums(plan)
    found: dict[tuple[str, tuple[int, ...]], tuple[int, int, bool]] = {
        key: where
        for i, item in enumerate(sums)
        for key, where in (
            [((item.family, (c,)), (i, c, False)) for c in range(plan.n_channels)]
            if item.family in ("mean", "var")
            else [((item.family, tuple(item.key)), (i, 0, item.is_complex))]
        )
    }
    pieces: list[tuple[int, int, int, int]] = []
    takes: list[NDArray[np.int64]] = []
    take_of: dict[bytes, int] = {}
    for grid in layout.grids:
        n_o = int(np.prod(_carried_shape(plan, grid.name))) if grid.is_complex else 1
        floats = grid.floats_per_kind
        for row, kind in zip(grid.chunks.tolist(), grid.kinds.tolist(), strict=True):
            where = found.get((grid.name, tuple(row)))
            real = grid.real[kind]
            if where is None or (not where[2] and not real.all()):
                raise ValueError(
                    f"{grid.name}: the loop does not compute every coefficient of the layout; "
                    "scattering._plan.loop_sums and statistics._layout.base_layout disagree."
                )
            source, start, is_complex = where
            if is_complex and real.any():
                key = real.tobytes()
                if key not in take_of:
                    pairs = np.stack(
                        [2 * np.arange(n_o), np.where(real, -1, 2 * np.arange(n_o) + 1)]
                    )
                    flat = pairs.T.reshape(-1)
                    take_of = {**take_of, key: len(takes)}
                    takes.append(flat[flat >= 0].astype(np.int64))
                pieces.append((source, start, start + 2 * n_o, take_of[key]))
            else:
                pieces.append((source, start, start + int(floats[kind]), -1))
    # The chunks of mean and var, consecutive floats of one sum, are read as one piece.
    merged: list[tuple[int, int, int, int]] = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if (
            last is not None
            and piece[3] < 0
            and last[3] < 0
            and last[0] == piece[0]
            and last[2] == piece[1]
        ):
            merged = [*merged[:-1], (last[0], last[1], piece[2], -1)]
        else:
            merged.append(piece)
    return LoopGather(
        pieces=np.array(merged, dtype=np.int64).reshape(-1, 4),
        takes=tuple(takes),
        n_sums=len(sums),
    )


def gather_loop(backend: Backend, sums: Sequence[Any], gather: LoopGather, takes: Any) -> Any:
    """The buffer ``(Nb, n_floats)`` from the divided sums of the loop, each a real ``(Nb, m)``
    array in the order of ``loop_sums``: their chunks concatenated in the order of the buffer.
    ``takes`` holds :attr:`LoopGather.takes` as integer arrays of the backend."""
    xp = backend.xp
    pieces: list[Any] = []
    for source, start, stop, take in gather.pieces.tolist():
        value = sums[source]
        if take >= 0:
            pieces.append(xp.take(value, takes[take], axis=1))
        elif start == 0 and stop == int(value.shape[1]):
            pieces.append(value)
        else:
            pieces.append(value[:, start:stop])
    return xp.concat(pieces, axis=1)


# ---------------------------------------------------------------------- dense chunks


def _single(grid: FamilyGrid) -> NDArray[np.bool_] | None:
    """The row of real entries of the chunks of ``grid`` when they are all of one kind, else
    ``None``."""
    if grid.real.shape[0] == 1:
        row: NDArray[np.bool_] = grid.real[0]
        return row
    return None


class Unpack(NamedTuple):
    """How some chunks of one family of a buffer are read as a dense array
    ``(Nb, n, n_orientations, parts)`` (:func:`unpack`)."""

    starts: NDArray[np.int64]
    """``(n,)``: the first float of each chunk read."""
    kinds: NDArray[np.int64]
    """``(n,)``: its kind."""
    within: NDArray[np.int64]
    """``(n_kinds, n_orientations, parts)``: :func:`within_chunk`, the parts kept."""
    view: tuple[int, int] | None
    """When the chunks are consecutive in the buffer and of one kind whose entries are all
    complex (``parts`` 2) or all real: the floats they span, read as a view."""


def unpack_spec(layout: Layout, name: str, chunks: NDArray[np.int64], parts: int = 2) -> Unpack:
    """The :class:`Unpack` of the chunks ``chunks`` (positions among the chunks of ``name``,
    in the order wanted) of ``layout``; ``parts`` is 2 for (re, im), 1 for the real part."""
    grid = layout.grid(name)
    within = within_chunk(grid.real)[:, :, :parts]
    starts = layout.chunk_starts(name)[chunks]
    kinds = grid.kinds[chunks].astype(np.int64)
    view = None
    row = _single(grid)
    whole = np.array_equal(chunks, np.arange(grid.n_chunks))
    if row is not None and whole and (row.all() or (not row.any() and parts == 2)):
        view = layout.span(name)
    return Unpack(starts, kinds, within, view)


def unpack(backend: Backend, buffer: Any, spec: Unpack) -> Any:
    """The chunks of ``spec`` read from ``buffer`` ``(Nb, n_floats)`` as a dense array
    ``(Nb, n, n_orientations, parts)``, with 0 for the imaginary part of a real entry: a view,
    or one ``take`` whose index is built from the tables of the chunks when it applies."""
    xp = backend.xp
    n_batch = int(buffer.shape[0])
    n, n_o, parts = int(spec.starts.shape[0]), int(spec.within.shape[1]), int(spec.within.shape[2])
    if spec.view is not None:
        start, stop = spec.view
        if stop - start == n * n_o * parts:
            return xp.reshape(buffer[:, start:stop], (n_batch, n, n_o, parts))
        real = xp.reshape(buffer[:, start:stop], (n_batch, n, n_o, 1))
        return xp.concat([real, xp.zeros_like(real)], axis=3) if parts == 2 else real
    if n == 0:
        return xp.zeros((n_batch, 0, n_o, parts), dtype=buffer.dtype)
    index = _floats(spec.within, spec.kinds, spec.starts, int(buffer.shape[1]))
    source = with_zero(backend, buffer) if (spec.within[spec.kinds] < 0).any() else buffer
    values = xp.take(source, constant(backend, index, buffer, "int32"), axis=1)
    return xp.reshape(values, (n_batch, n, n_o, parts))


def _floats(
    within: NDArray[np.int64], kinds: NDArray[np.int64], starts: NDArray[np.int64], missing: int
) -> NDArray[np.int64]:
    """The float of each entry and part of chunks of kinds ``kinds`` starting at ``starts``,
    ``missing`` for the imaginary part of a real entry: built when a map is applied, from the
    tables of the chunks, and not kept."""
    table = within[kinds]
    found: NDArray[np.int64] = np.where(table >= 0, table + starts[:, None, None], missing)
    return found.reshape(-1)


class Pack(NamedTuple):
    """How a dense array ``(Nb, n, n_orientations, 2)`` is written as the floats of a family
    (:func:`pack`)."""

    kinds: NDArray[np.int64]
    """``(n,)``: the kind of each chunk."""
    within: NDArray[np.int64]
    """``(n_kinds, n_orientations, 2)``: :func:`within_chunk`."""
    starts: NDArray[np.int64]
    """``(n,)``: the first float of each chunk, from the first float of the family."""
    n_floats: int
    """The floats of the family."""


def pack_spec(layout: Layout, name: str) -> Pack:
    """The :class:`Pack` of family ``name`` of ``layout``, its chunks in the order of the
    buffer."""
    grid = layout.grid(name)
    start, stop = layout.span(name)
    return Pack(
        kinds=grid.kinds.astype(np.int64),
        within=within_chunk(grid.real),
        starts=layout.chunk_starts(name) - start,
        n_floats=stop - start,
    )


def pack(backend: Backend, dense: Any, spec: Pack) -> Any:
    """The floats ``(Nb, n_floats)`` of a family from its dense array ``(Nb, n, n_o, 2)``,
    without the imaginary part of its real entries: a reshape when the chunks are of one kind
    whose entries are all complex or all real, a ``take`` on the orientations when they are of
    one kind, a ``segment_sum`` whose ids are built from the tables of the chunks when it
    applies otherwise (each float receives one value)."""
    xp = backend.xp
    n_batch = int(dense.shape[0])
    n, n_o = int(dense.shape[1]), int(dense.shape[2])
    if n == 0 or spec.n_floats == 0:
        return xp.zeros((n_batch, 0), dtype=dense.dtype)
    kinds = np.unique(spec.kinds)
    if kinds.shape[0] == 1:
        within = spec.within[int(kinds[0])]
        if (within[:, 1] >= 0).all():
            return xp.reshape(dense, (n_batch, n * n_o * 2))
        if (within[:, 1] < 0).all():
            return xp.reshape(dense[..., 0], (n_batch, n * n_o))
        keep = np.flatnonzero(within.reshape(-1) >= 0)
        flat = xp.reshape(dense, (n_batch, n, n_o * 2))
        taken = xp.take(flat, constant(backend, keep, dense, "int32"), axis=2)
        return xp.reshape(taken, (n_batch, n * int(keep.shape[0])))
    ids = _floats(spec.within, spec.kinds, spec.starts, spec.n_floats)
    summed = backend.segment_sum(
        xp.reshape(dense, (n_batch, n * n_o * 2)),
        1,
        constant(backend, ids, dense, "int32"),
        spec.n_floats + 1,
    )
    return summed[:, : spec.n_floats]


# ---------------------------------------------------------------------- maps between buffers


class Take(NamedTuple):
    """Positions of one orientation axis, in order, possibly repeated (``select``)."""

    axis: int
    """The axis, among the orientation axes as they are when the step applies."""
    positions: NDArray[np.int64]
    """The positions kept."""
    drop: bool
    """Whether the axis is dropped (one position, given as an integer)."""


class OnOrientations(NamedTuple):
    """A complex linear map on a group of consecutive orientation axes of every chunk (the
    reductions ``iso`` and ``angular_fourier``), in padded CSR form: output position ``p`` of
    the group receives ``Σ_m (real[p, m] + i imaginary[p, m]) x[columns[p, m]]``, the group
    flattened in C order. Rows with fewer entries are padded with weight 0 on their first
    column."""

    axis: int
    """The first axis of the group, among the orientation axes."""
    n_axes: int
    """The number of axes of the group."""
    shape: tuple[int, ...]
    """The shape of the axes that replace the group (``()`` to remove it)."""
    columns: NDArray[np.int64]
    """``(P, M)``."""
    real: NDArray[np.float64]
    """``(P, M)``."""
    imaginary: NDArray[np.float64] | None
    """``(P, M)``, or ``None`` when every weight is real."""


Step = Take | OnOrientations


class _Part(NamedTuple):
    """How one family of the output of a :class:`FloatMap` is made: a slice of the input
    (``copy``), or chunks of an input family unpacked, combined across chunks (``mix``),
    transformed on their orientations (``steps``), and packed."""

    name: str
    copy: tuple[int, int] | None
    unpack: Unpack | None
    mix: tuple[NDArray[np.int64], NDArray[np.float64]] | None
    """``(index, weights)``, each ``(n_out, M)``: output chunk ``i`` is
    ``Σ_m weights[i, m] · unpacked[index[i, m]]``."""
    steps: tuple[Step, ...]
    orientations: tuple[int, ...]
    """The orientation shape of the chunks read, before the steps."""
    pack: Pack | None

    @property
    def columns(self) -> NDArray[np.int64] | None:
        """The columns of the map on the orientations, ``None`` if there is none."""
        on = [s for s in self.steps if isinstance(s, OnOrientations)]
        return on[0].columns if on else None

    @property
    def weights(self) -> NDArray[np.float64] | None:
        """The real parts of its weights, ``None`` if there is none."""
        on = [s for s in self.steps if isinstance(s, OnOrientations)]
        return on[0].real if on else None


@dataclass(frozen=True, eq=False)
class FloatMap:
    """A static linear map from a buffer of one layout to a buffer of :attr:`layout` (Part
    III, §19.6), one part per output family."""

    layout: Layout
    parts: tuple[_Part, ...]


class Copy(NamedTuple):
    """A family kept as it is."""

    name: str


class Chunked(NamedTuple):
    """A family made from the chunks of the input family of the same name: output chunk ``i``
    (row ``i`` of ``chunks.grid``) is ``Σ_m weights[i, m] · input chunk sources[i, m]`` (a
    plain gather when ``weights`` is ``None``), then ``steps`` act on its orientations. The
    builder gives the kinds of the output chunks and their levels in ``chunks``."""

    chunks: FamilyChunks
    sources: NDArray[np.int64]
    """``(n, M)``."""
    weights: NDArray[np.float64] | None
    """``(n, M)``, or ``None`` with ``M`` = 1."""
    steps: tuple[Step, ...] = ()


def make_map(layout: Layout, specs: Sequence[Copy | Chunked]) -> FloatMap:
    """The :class:`FloatMap` from a buffer of ``layout`` that makes the families of ``specs``,
    in this order."""
    families = [
        layout.chunked(spec.name) if isinstance(spec, Copy) else spec.chunks for spec in specs
    ]
    out_layout, orders = build_layout(families)
    parts: list[_Part] = []
    for spec, order in zip(specs, orders, strict=True):
        if isinstance(spec, Copy):
            if not np.array_equal(order, np.arange(order.shape[0])):
                raise ValueError(f"{spec.name}: a copied family keeps its order.")
            parts.append(_Part(spec.name, layout.span(spec.name), None, None, (), (), None))
            continue
        name = spec.chunks.grid.name
        sources = spec.sources[order]
        if spec.weights is None:
            read = unpack_spec(layout, name, sources[:, 0])
            mix = None
        else:
            read = unpack_spec(layout, name, np.arange(layout.grid(name).n_chunks))
            mix = (sources, spec.weights[order])
        orientations = layout.grid(name).orientations
        parts.append(
            _Part(name, None, read, mix, spec.steps, orientations, pack_spec(out_layout, name))
        )
    return FloatMap(layout=out_layout, parts=tuple(parts))


def _on_orientations(
    backend: Backend, dense: Any, step: OnOrientations, shape: tuple[int, ...]
) -> Any:
    """``step`` applied to ``dense`` ``(Nb, n, *shape, 2)``."""
    xp = backend.xp
    n_batch, n = int(dense.shape[0]), int(dense.shape[1])
    before = int(np.prod(shape[: step.axis], dtype=np.int64))
    group = int(np.prod(shape[step.axis : step.axis + step.n_axes], dtype=np.int64))
    after = int(np.prod(shape[step.axis + step.n_axes :], dtype=np.int64))
    x = xp.reshape(dense, (n_batch, n, before, group, after, 2))
    rows, width = step.columns.shape
    index = constant(backend, step.columns.reshape(rows, 1, width), dense, "int32")

    def weighted(w: NDArray[np.float64]) -> Any:
        weight = constant(backend, w.reshape(rows, 1, width), dense, dtype_name(dense))
        return xp.reshape(
            backend.gather_weighted(x, 3, index, weight), (n_batch, n, before, rows, after, 2)
        )

    a = weighted(step.real)
    if step.imaginary is not None:
        b = weighted(step.imaginary)
        a = xp.stack([a[..., 0] - b[..., 1], b[..., 0] + a[..., 1]], axis=-1)
    new = (*shape[: step.axis], *step.shape, *shape[step.axis + step.n_axes :])
    return xp.reshape(a, (n_batch, n, *new, 2))


def apply(
    backend: Backend,
    buffer: Any,
    transform: FloatMap,
    scale: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """The buffer ``(Nb, n)`` mapped by ``transform``: ``(Nb, transform.layout.n_floats)``.
    ``scale`` may give, for a family made from chunks, a function applied to its dense array
    ``(Nb, n, n_orientations, 2)`` before it is packed (the divisors of the normalisation)."""
    xp = backend.xp
    scale = {} if scale is None else scale
    pieces: list[Any] = []
    for part in transform.parts:
        if part.copy is not None:
            pieces.append(buffer[:, part.copy[0] : part.copy[1]])
            continue
        assert part.unpack is not None
        assert part.pack is not None
        dense = unpack(backend, buffer, part.unpack)
        grid = transform.layout.grid(part.name)
        if part.mix is not None:
            index, weights = part.mix
            rows, width = index.shape
            n_batch, n_o = int(dense.shape[0]), int(dense.shape[2])
            dense = xp.reshape(
                backend.gather_weighted(
                    dense,
                    1,
                    constant(backend, index.reshape(rows, 1, width), buffer, "int32"),
                    constant(backend, weights.reshape(rows, 1, width), buffer, dtype_name(buffer)),
                ),
                (n_batch, rows, n_o, 2),
            )
        if part.steps:
            dense = _apply_steps(backend, dense, part.steps, part.orientations)
        if part.name in scale:
            dense = scale[part.name](dense)
        n_batch, n = int(dense.shape[0]), int(dense.shape[1])
        pieces.append(
            pack(backend, xp.reshape(dense, (n_batch, n, grid.n_orientations, 2)), part.pack)
        )
    return pieces[0] if len(pieces) == 1 else xp.concat(pieces, axis=1)


def _apply_steps(
    backend: Backend, dense: Any, steps: Sequence[Step], shape: tuple[int, ...]
) -> Any:
    """``steps`` applied in order to ``dense`` ``(Nb, n, n_o, 2)``, whose orientations have
    the shape ``shape``."""
    xp = backend.xp
    n_batch, n = int(dense.shape[0]), int(dense.shape[1])
    x = xp.reshape(dense, (n_batch, n, *shape, 2))
    for step in steps:
        if isinstance(step, Take):
            x = xp.take(x, constant(backend, step.positions, dense, "int32"), axis=2 + step.axis)
            if step.drop:
                shape = (*shape[: step.axis], *shape[step.axis + 1 :])
                x = xp.reshape(x, (n_batch, n, *shape, 2))
            else:
                shape = (*shape[: step.axis], int(step.positions.shape[0]), *shape[step.axis + 1 :])
        else:
            x = _on_orientations(backend, x, step, shape)
            shape = (*shape[: step.axis], *step.shape, *shape[step.axis + step.n_axes :])
    return x


# ---------------------------------------------------------------------- reading the buffer


def with_zero(backend: Backend, buffer: Any) -> Any:
    """The buffer with one more float, 0, at position ``n_floats``: the imaginary part read for
    a real entry."""
    xp = backend.xp
    return xp.concat([buffer, xp.zeros_like(buffer[:, :1])], axis=1)


def grid_order(grid: FamilyGrid) -> NDArray[np.int64]:
    """The chunks of ``grid`` in the order of its full grid (C order of the chunk axes)."""
    order: NDArray[np.int64] = np.argsort(
        flat_positions(grid.chunks.astype(np.int64), grid.chunk_shape), kind="stable"
    ).astype(np.int64)
    return order


def full_grid(backend: Backend, buffer: Any, layout: Layout, name: str) -> Any:
    """The full-grid view of family ``name`` (Part III, §19.5): STL's shape, NaN where no
    coefficient exists, complex for S1 to S4 with imaginary part 0 for the real entries. The
    chunks are placed in the grid of the chunk axes with ``scatter_to_grid``, their
    orientations carried along; for :meth:`STStatistics.full_grid` only."""
    xp = backend.xp
    grid = layout.grid(name)
    n_batch = int(buffer.shape[0])
    parts = 2 if grid.is_complex else 1
    if grid.n_chunks == 0:
        nan = xp.full((n_batch, *grid.shape, parts), float("nan"), dtype=buffer.dtype)
        return backend.real_pairs_to_complex(nan) if grid.is_complex else nan[..., 0]
    dense = unpack(backend, buffer, unpack_spec(layout, name, np.arange(grid.n_chunks), parts))
    k, n_axes = grid.n_chunk_axes, len(grid.orientations)
    if k == 0:
        placed = xp.reshape(dense, (n_batch, *grid.orientations, parts))
    else:
        # The real and imaginary parts are scattered apart, so that both are NaN outside.
        carried = xp.reshape(
            xp.permute_dims(dense, (0, 2, 3, 1)),
            (n_batch, *grid.orientations, parts, grid.n_chunks),
        )
        index = [constant(backend, column, buffer, "int32") for column in grid.chunks.T]
        scattered = backend.scatter_to_grid(carried, index, grid.chunk_shape)
        order = (0, *range(2 + n_axes, 2 + n_axes + k), *range(1, 1 + n_axes), 1 + n_axes)
        placed = xp.permute_dims(scattered, order)
    return backend.real_pairs_to_complex(placed) if grid.is_complex else placed[..., 0]


# ---------------------------------------------------------------------- storage


def payload(layout: Layout) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """The arrays and the JSON description of ``layout``, for ``estel.io``: its chunks."""
    arrays: dict[str, NDArray[Any]] = {}
    for i, grid in enumerate(layout.grids):
        arrays = {
            **arrays,
            f"family__{i}__chunks": np.asarray(grid.chunks),
            f"family__{i}__kinds": np.asarray(grid.kinds),
            f"family__{i}__real": np.asarray(grid.real),
        }
    content = {
        "families": [
            {
                "name": g.name,
                "axes": list(g.axes),
                "coords": [list(c) for c in g.coords],
                "is_complex": g.is_complex,
                "n_chunk_axes": g.n_chunk_axes,
            }
            for g in layout.grids
        ],
        "segments": [
            {
                "family": s.family,
                "level": list(s.level),
                "index_names": list(s.index_names),
                "first_chunk": s.first_chunk,
                "n_chunks": s.n_chunks,
                "orientations": list(s.orientations),
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
    grids = tuple(
        FamilyGrid(
            name=family["name"],
            axes=tuple(family["axes"]),
            coords=tuple(tuple(int(i) for i in c) for c in family["coords"]),
            is_complex=bool(family["is_complex"]),
            n_chunk_axes=int(family["n_chunk_axes"]),
            chunks=arrays[f"family__{i}__chunks"],
            kinds=arrays[f"family__{i}__kinds"],
            real=arrays[f"family__{i}__real"].astype(bool),
        )
        for i, family in enumerate(content["families"])
    )
    by_name = {g.name: g for g in grids}
    segments = tuple(
        Segment(
            family=s["family"],
            level=tuple(int(d) for d in s["level"]),
            index_names=tuple(s["index_names"]),
            chunks=by_name[s["family"]].chunks[
                int(s["first_chunk"]) : int(s["first_chunk"]) + int(s["n_chunks"])
            ],
            orientations=tuple(int(n) for n in s["orientations"]),
            offset=int(s["offset"]),
            n_floats=int(s["n_floats"]),
            first_chunk=int(s["first_chunk"]),
        )
        for s in content["segments"]
    )
    return Layout(grids=grids, segments=segments, n_floats=int(content["n_floats"]))
