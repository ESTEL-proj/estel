"""The reductions of the statistics: ``iso``, ``angular_fourier`` and ``scale_cosine``
(maths.md M-RED-ISO, M-RED-ANG, M-RED-SCALE, OP-5, OP-17; STL ``to_iso``, ``to_angular_ft``,
``to_scale_ft``; Part III, §19.6; tasks T2.5, T2.15, T3.7), which ``STStatistics.reduce``
applies in this order.

In ESTEL "reduction" means these three operations only. Each is linear in the coefficients, and
on the compact buffer it is a static linear map (Part III, §19.6) that acts chunk by chunk
(``_layout.make_map``): ``iso`` and ``angular_fourier`` are small complex matrices on the dense
orientations of every chunk, ``angular_fourier`` one orientation axis at a time; ``scale_cosine``
combines chunks, with real weights, at the same orientations. Their tables grow with the number
of chunks and of orientations, never with the number of coefficients. Each function takes a
layout and returns the :class:`_layout.FloatMap` to the reduced layout (``angular_fourier``: one
map per orientation axis, applied in order); nothing is modified. Every map reads at most L
(``iso``, ``angular_fourier``) or J (``scale_cosine``) entries per output entry.

Where no coefficient exists the result is NaN in STL, and not stored here; the layout says
which, statically: ``iso`` and ``angular_fourier`` keep the chunks of their inputs, since a
coefficient exists for every orientation or for none; ``scale_cosine`` reads missing chunks as
0 and keeps the output chunks at the positions where STL puts NaN back, as STL does. An output
entry takes one float when every entry it reads does, with a real weight.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

from ._grid import FamilyGrid
from ._layout import (
    Chunked,
    Copy,
    FamilyChunks,
    FloatMap,
    Layout,
    OnOrientations,
    entry_index,
    make_map,
)

ORIENTATION_AXES = {
    "S1": ("l",),
    "S2": ("l",),
    "S3": ("l2", "l3"),
    "S4": ("l1", "l2", "l3"),
}
"""The orientation axes of each family before any reduction: the last axes of its grid."""

ISO_AXES = {"S3": ("dl",), "S4": ("dl2", "dl3")}
"""The orientation axes of S3 and S4 after ``iso``: differences modulo L (M-RED-ISO)."""

SCALE_AXES = {"S3": (("j2", "j3"), ("k", "a")), "S4": (("j1", "j2", "j3"), ("k", "a", "b"))}
"""The scale axes of S3 and S4 before ``scale_cosine``, and after it (M-RED-SCALE)."""


def _orientation_positions(grid: FamilyGrid, names: tuple[str, ...]) -> int:
    """The position of the first of the orientation axes ``names`` among the orientation axes
    of the grid, which they must end."""
    positions = tuple(grid.axis(n) for n in names)
    last = len(grid.axes)
    if positions != tuple(range(last - len(names), last)) or positions[0] < grid.n_chunk_axes:
        raise ValueError(f"{grid.name}: the orientation axes {names} must be the last axes.")
    return positions[0] - grid.n_chunk_axes


def _padded(
    matrix: NDArray[np.complex128],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64] | None]:
    """The nonzero entries of ``matrix`` ``(P, Q)`` in padded CSR form: columns, real parts and
    imaginary parts (``None`` if all are 0), each ``(P, M)``; a row's padding repeats its first
    column with weight 0."""
    nonzero = matrix != 0
    counts = nonzero.sum(axis=1)
    width = max(int(counts.max(initial=0)), 1)
    order = np.argsort(~nonzero, axis=1, kind="stable")[:, :width]
    kept = np.take_along_axis(nonzero, order, axis=1)
    columns = np.where(kept, order, order[:, :1]).astype(np.int64)
    weights = np.where(kept, np.take_along_axis(matrix, order, axis=1), 0)
    imaginary = weights.imag.astype(np.float64)
    return columns, weights.real.astype(np.float64), imaginary if imaginary.any() else None


def _real_after(
    grid: FamilyGrid, axis: int, n_axes: int, matrix: NDArray[np.complex128]
) -> NDArray[np.bool_]:
    """The real entries of each kind of chunk of ``grid`` after ``matrix`` acts on its
    orientation axes ``axis`` … ``axis + n_axes``: an output entry is complex when one of the
    entries it reads is, or its weight is."""
    shape = grid.orientations
    before = int(np.prod(shape[:axis], dtype=np.int64))
    group = int(np.prod(shape[axis : axis + n_axes], dtype=np.int64))
    after = int(np.prod(shape[axis + n_axes :], dtype=np.int64))
    complex_in = (~grid.real).reshape(-1, before, group, after).astype(np.int64)
    nonzero = matrix != 0
    reads = np.einsum("kbga,pg->kbpa", complex_in, nonzero.astype(np.int64)) > 0
    weighted = (nonzero & (matrix.imag != 0)).any(axis=1)[None, None, :, None]
    real: NDArray[np.bool_] = ~(reads | weighted).reshape(grid.real.shape[0], -1)
    return real


def _on_orientations(
    layout: Layout,
    grid: FamilyGrid,
    axes: tuple[str, ...],
    coords: tuple[tuple[int, ...], ...],
    axis: int,
    n_axes: int,
    shape: tuple[int, ...],
    matrix: NDArray[np.complex128],
) -> Chunked:
    """The family ``grid`` of ``layout`` with ``matrix`` ``(P, Q)`` acting on its orientation
    axes ``axis`` … ``axis + n_axes`` (``Q`` positions in C order), replaced by axes of shape
    ``shape`` (``P`` positions): ``y[.., p, ..] = Σ_q matrix[p, q] x[.., q, ..]`` in every
    chunk. ``axes`` and ``coords`` are those of the result."""
    levels, level_ids = layout.levels(grid.name)
    built = grid.replace(axes=axes, coords=coords, real=_real_after(grid, axis, n_axes, matrix))
    columns, real, imaginary = _padded(matrix)
    step = OnOrientations(axis, n_axes, shape, columns, real, imaginary)
    sources = np.arange(grid.n_chunks, dtype=np.int64)[:, None]
    return Chunked(FamilyChunks(built, levels, level_ids), sources, None, (step,))


# ---------------------------------------------------------------------- iso


def _iso_table(n_orientations: int, n_axes: int) -> NDArray[np.int64]:
    """For S3 (``n_axes = 2``): ``index[l1, d]`` = the flat position of ``(l1, l1 + d mod L)``;
    for S4 (``n_axes = 3``): ``index[l1, d2, d3]`` of ``(l1, l1 + d2, l1 + d3 mod L)``."""
    L = n_orientations  # noqa: N806
    l1 = np.arange(L)
    d = np.arange(L)
    if n_axes == 2:
        table: NDArray[np.int64] = l1[:, None] * L + (l1[:, None] + d[None, :]) % L
        return table
    table = (
        l1[:, None, None] * L * L
        + ((l1[:, None, None] + d[None, :, None]) % L) * L
        + (l1[:, None, None] + d[None, None, :]) % L
    )
    return table


@functools.lru_cache(maxsize=64)
def iso(layout: Layout, n_orientations: int) -> FloatMap:
    """Isotropy (M-RED-ISO; STL ``to_iso``): S1 and S2 averaged over l; S3 and S4 averaged over
    the first orientation at fixed differences modulo L, ``(1/L) Σ_{l1} S3[.., l1, l1 + d]`` and
    ``(1/L) Σ_{l1} S4[.., l1, l1 + d2, l1 + d3]``. The orientations are treated as cyclic over
    [0, π), as in STL (OP-5)."""
    L = n_orientations  # noqa: N806
    specs: list[Copy | Chunked] = []
    for grid in layout.grids:
        if grid.name not in ORIENTATION_AXES:
            specs.append(Copy(grid.name))
            continue
        names = ORIENTATION_AXES[grid.name]
        axis = _orientation_positions(grid, names)
        k = len(names)
        first = grid.n_chunk_axes + axis
        if grid.name in ("S1", "S2"):
            matrix = np.full((1, L), 1.0 / L, dtype=np.complex128)
            new_axes: tuple[str, ...] = ()
        else:
            table = _iso_table(L, k)  # (l1, *d) -> input position
            rows = np.broadcast_to(np.arange(L ** (k - 1)).reshape(table.shape[1:]), table.shape)
            matrix = (
                np.bincount((rows * L**k + table).ravel(), minlength=L ** (k - 1) * L**k)
                .reshape(L ** (k - 1), L**k)
                .astype(np.complex128)
                / L
            )
            new_axes = ISO_AXES[grid.name]
        shape = (L,) * len(new_axes)
        axes = grid.axes[:first] + new_axes
        coords = grid.coords[:first] + tuple(tuple(range(n)) for n in shape)
        specs.append(_on_orientations(layout, grid, axes, coords, axis, k, shape, matrix))
    return make_map(layout, specs)


# ---------------------------------------------------------------------- angular_fourier


def _dft(n_orientations: int, harmonics: int) -> NDArray[np.complex128]:
    """The first ``harmonics`` rows of the orthonormal DFT of size L (``norm="ortho"``), its
    entries that are 0, ±1 or ±i exactly so."""
    L = n_orientations  # noqa: N806
    turns = (np.arange(harmonics)[:, None] * np.arange(L)[None, :]) % L
    angle = -2.0 * np.pi * turns / L
    re, im = np.cos(angle), np.sin(angle)
    re = np.where(np.abs(re) < 1e-12, 0.0, re)
    im = np.where(np.abs(im) < 1e-12, 0.0, im)
    dft: NDArray[np.complex128] = (re + 1j * im) / np.sqrt(L)
    return dft


@functools.lru_cache(maxsize=64)
def angular_fourier(layout: Layout, harmonics: int, after_iso: bool) -> tuple[FloatMap, ...]:
    """The angular Fourier transform (M-RED-ANG; STL ``to_angular_ft``): an orthonormal DFT over
    the orientation axes of S3 and S4 (``norm="ortho"``), keeping the first ``harmonics``
    harmonics along each. After ``iso`` it acts on the orientation differences. Each axis name
    gets the prefix ``h``.

    The DFT is separable, so it is applied one orientation axis at a time (Part III, §19.6):
    one map per axis, to apply in order, each reading L entries per output entry. A single map
    over all the axes would be their Kronecker product, whose size grows as L to the power of
    twice the number of axes."""
    axes: dict[str, tuple[str, ...]] = {}
    for grid in layout.grids:
        if grid.name in ("S3", "S4"):
            names = ISO_AXES[grid.name] if after_iso else ORIENTATION_AXES[grid.name]
            _orientation_positions(grid, names)
            axes = {**axes, grid.name: names}
    steps = max((len(n) for n in axes.values()), default=0)
    maps: list[FloatMap] = []
    for step in range(steps):
        specs: list[Copy | Chunked] = []
        for grid in layout.grids:
            if grid.name not in axes or step >= len(axes[grid.name]):
                specs.append(Copy(grid.name))
                continue
            name = axes[grid.name][step]
            at = grid.axis(name)
            renamed = (*grid.axes[:at], f"h{name}", *grid.axes[at + 1 :])
            coords = (*grid.coords[:at], tuple(range(harmonics)), *grid.coords[at + 1 :])
            dft = _dft(grid.shape[at], harmonics)
            axis = at - grid.n_chunk_axes
            specs.append(
                _on_orientations(layout, grid, renamed, coords, axis, 1, (harmonics,), dft)
            )
        transform = make_map(layout, specs)
        maps.append(transform)
        layout = transform.layout
    return tuple(maps)


# ---------------------------------------------------------------------- scale_cosine


def cosine_basis(n: int) -> NDArray[np.float64]:
    """The orthonormal DCT-II basis ``F[k, j] = alpha_k cos(kπ (j + 1/2) / n)``,
    ``alpha_0 = √(1/n)``, ``alpha_k = √(2/n)`` (M-RED-SCALE; STL ``cosinus1``)."""
    j = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    alpha = np.concatenate(([np.sqrt(1.0 / n)], np.full(n - 1, np.sqrt(2.0 / n))))
    basis: NDArray[np.float64] = np.outer(alpha, np.ones(n)) * np.cos(
        np.pi * np.outer(k, j + 0.5) / n
    )
    return basis


def _scale_table(
    n_scales: int, dj: int, n_axes: int
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """The reparametrisation of M-RED-SCALE: for S3 (``n_axes = 2``) ``(j1, a)`` reads
    ``(j1, j1 + a)``; for S4 (``n_axes = 3``) ``(j1, a, b)`` reads ``(j1, j1 + a, j1 + b)``,
    with ``0 ≤ a ≤ b < min(dj, J - j1)`` (STL ``to_scale_ft``). Returns the flat positions in
    the ``J^n_axes`` scale grid (0 where nothing is read) and where something is read."""
    J = n_scales  # noqa: N806
    j1 = np.arange(J)[:, None, None]
    a = np.arange(dj)[None, :, None]
    b = np.arange(dj)[None, None, :]
    limit = np.minimum(dj, J - j1)
    if n_axes == 2:
        valid = (a < limit)[:, :, 0]
        index = np.where(valid, (j1 * J + j1 + a)[:, :, 0], 0)
    else:
        valid = (a <= b) & (b < limit)
        index = np.where(valid, j1 * J * J + (j1 + a) * J + (j1 + b), 0)
    return index.astype(np.int64), valid


def _combined_real(
    grid: FamilyGrid, kinds: NDArray[np.int64], used: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """For output chunks made of the input chunks of kinds ``kinds`` ``(n, M)`` where
    ``used``, with real weights: the kind of each, and the real entries of each kind (those
    real in every chunk read)."""
    marked = np.sort(np.where(used, kinds, -1), axis=1)
    combos, of_chunk = np.unique(marked, axis=0, return_inverse=True)
    rows = np.concatenate([grid.real, np.ones((1, grid.real.shape[1]), dtype=bool)])
    real = np.all(rows[combos], axis=1)  # -1 reads the last row: all real
    return of_chunk.reshape(-1).astype(np.int64), real


def _key(
    rows: NDArray[np.int64], scales: NDArray[np.int64], at: int, shape: tuple[int, ...]
) -> NDArray[np.int64]:
    """The flat position in ``shape`` of chunks whose other chunk axes are ``rows`` ``(..., m)``
    and whose scales, flattened, are ``scales`` ``(...)``, inserted at ``at``."""
    columns = [
        *(rows[..., i] for i in range(at)),
        np.asarray(scales),
        *(rows[..., i] for i in range(at, rows.shape[-1])),
    ]
    flat: NDArray[np.int64] = np.ravel_multi_index(tuple(columns), shape).astype(np.int64)
    return flat


@functools.lru_cache(maxsize=64)
def scale_cosine(layout: Layout, n_scales: int, harmonics: int, dj: int) -> FloatMap:
    """The scale cosine transform (M-RED-SCALE; STL ``to_scale_ft``), after ``iso``: the scales
    of S3 and S4 reparametrised as ``(j1, a)`` and ``(j1, a, b)``, an orthonormal DCT-II along
    j1 at fixed differences, the first ``harmonics`` harmonics kept; the axis j1 becomes ``k``.
    Missing inputs count as 0; an output exists where its input at j1 = k does, as STL puts
    NaN back at the same positions. Each output chunk combines at most J input chunks, at the
    same orientations."""
    J = n_scales  # noqa: N806
    basis = cosine_basis(J)
    specs: list[Copy | Chunked] = []
    for grid in layout.grids:
        if grid.name not in SCALE_AXES:
            specs.append(Copy(grid.name))
            continue
        before, after = SCALE_AXES[grid.name]
        n = len(before)
        first = grid.axis(before[0])
        if tuple(grid.axis(name) for name in before) != tuple(range(first, first + n)):
            raise ValueError(f"{grid.name}: the scale axes {before} must be consecutive.")
        if first + n > grid.n_chunk_axes:
            raise ValueError(f"{grid.name}: the scale axes {before} must be chunk axes.")
        table, valid = _scale_table(J, dj, n)  # (j1, *differences)
        n_differences = dj ** (n - 1)
        table = table.reshape(J, n_differences)
        valid = valid.reshape(J, n_differences)
        chunks = grid.chunks.astype(np.int64)
        lead_shape, rest_shape = grid.chunk_shape[:first], grid.chunk_shape[first + n :]
        key_shape = (*lead_shape, J**n, *rest_shape)
        scales = np.ravel_multi_index(tuple(chunks[:, first : first + n].T), (J,) * n)
        outer = np.concatenate([chunks[:, :first], chunks[:, first + n :]], axis=1)

        index = entry_index(_key(outer, scales, first, key_shape))
        groups = np.unique(outer, axis=0)
        g, k, d = (
            a.reshape(-1)
            for a in np.meshgrid(
                np.arange(groups.shape[0]),
                np.arange(harmonics),
                np.arange(n_differences),
                indexing="ij",
            )
        )
        exists = valid[k, d] & (index.find(_key(groups[g], table[k, d], first, key_shape)) >= 0)
        g, k, d = g[exists], k[exists], d[exists]
        # Every j1 at the same differences: the input chunks read, and their weights.
        j1 = np.arange(J)[None, :]
        rows = np.repeat(groups[g][:, None, :], J, axis=1)
        sources = index.find(_key(rows, table[j1, d[:, None]], first, key_shape))
        weights = basis[k[:, None], j1]
        used = valid[j1, d[:, None]] & (sources >= 0) & (weights != 0)
        width = max(int(used.sum(axis=1).max(initial=0)), 1)
        order = np.argsort(~used, axis=1, kind="stable")[:, :width]
        used = np.take_along_axis(used, order, axis=1)
        sources = np.take_along_axis(sources, order, axis=1)
        own = np.where(used[:, :1], sources[:, :1], 0)  # the padding reads the row's first
        sources = np.where(used, sources, own).astype(np.int64)
        weights = np.where(used, np.take_along_axis(weights, order, axis=1), 0.0)

        kinds, real = _combined_real(grid, grid.kinds[sources].astype(np.int64), used)
        levels, level_ids = layout.levels(grid.name)
        theirs = np.where(used, level_ids[sources], -1)
        low = np.where(used, theirs, np.iinfo(np.int64).max).min(axis=1)
        high = theirs.max(axis=1)
        mixed = len(levels)  # the level () of chunks from several levels
        out_levels = np.where(low == high, low, mixed).astype(np.int64)

        differences = np.stack(np.unravel_index(d, (dj,) * (n - 1)), axis=1)
        out_chunks = np.concatenate(
            [groups[g][:, :first], k[:, None], differences, groups[g][:, first:]], axis=1
        )
        sizes = (harmonics, *(dj,) * (n - 1))
        out_grid = grid.replace(
            axes=grid.axes[:first] + after + grid.axes[first + n :],
            coords=grid.coords[:first]
            + tuple(tuple(range(s)) for s in sizes)
            + grid.coords[first + n :],
            chunks=out_chunks,
            kinds=kinds,
            real=real,
        )
        family = FamilyChunks(out_grid, (*levels, ()), out_levels)
        specs.append(Chunked(family, sources, weights.astype(np.float64)))
    return make_map(layout, specs)
