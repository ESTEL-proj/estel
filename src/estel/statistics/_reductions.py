"""The reductions of the statistics: ``iso``, ``angular_fourier`` and ``scale_cosine``
(maths.md M-RED-ISO, M-RED-ANG, M-RED-SCALE, OP-5, OP-17; STL ``to_iso``, ``to_angular_ft``,
``to_scale_ft``; Part III, §19.6; tasks T2.5, T2.15), which ``STStatistics.reduce`` applies in
this order.

In ESTEL "reduction" means these three operations only. Each is linear in the coefficients, and
on the compact buffer it is a static linear map (Part III, §19.6): a complex matrix acting on a
group of consecutive axes of a family (the orientations, or the scales), built here in NumPy
from the layout, turned by ``_layout.make_map`` into a sparse real matrix on the floats of the
buffer, and applied with ``gather_weighted``; it is differentiable on every backend. Each
function takes a layout and returns the :class:`_layout.FloatMap` to the reduced layout;
nothing is modified.

Where no coefficient exists the result is NaN in STL, and not stored here; the layout says
which, statically: ``iso`` and ``angular_fourier`` keep the positions of their inputs, since a
coefficient exists for every orientation or for none; ``scale_cosine`` reads missing inputs as
0 and keeps the outputs at the positions where STL puts NaN back, as STL does.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._grid import FamilyGrid
from ._layout import Copy, FamilyEntries, FloatMap, Layout, Linear, flat_positions, make_map

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
    """The position of the first of the orientation axes ``names``, which must be the last
    axes of the grid."""
    positions = tuple(grid.axis(n) for n in names)
    last = len(grid.axes)
    if positions != tuple(range(last - len(names), last)):
        raise ValueError(f"{grid.name}: the orientation axes {names} must be the last axes.")
    return positions[0]


def _on_axes(
    entries: FamilyEntries,
    out_grid: FamilyGrid,
    first: int,
    n_in: int,
    matrix: NDArray[np.complex128],
) -> Linear:
    """The linear map of ``matrix`` ``(P, Q)`` on the ``n_in`` axes of the family from
    ``first``, whose ``Q`` positions it reads in C order; the output grid ``out_grid`` has, at
    ``first``, axes of ``P`` positions in all, the other axes unchanged:
    ``y[.., p, ..] = Σ_q matrix[p, q] x[.., q, ..]``."""
    shape = entries.grid.shape
    lead, inner, rest = shape[:first], shape[first : first + n_in], shape[first + n_in :]
    n_rest = int(np.prod(rest))
    n_out = int(matrix.shape[0])
    p = entries.positions
    lead_flat = flat_positions(p[:, :first], lead)
    inner_flat = flat_positions(p[:, first : first + n_in], inner)
    rest_flat = flat_positions(p[:, first + n_in :], rest)
    rows, cols = np.nonzero(matrix)
    by_col = np.argsort(cols, kind="stable")
    rows, cols, weights = rows[by_col], cols[by_col], matrix[rows[by_col], cols[by_col]]
    counts = np.bincount(cols, minlength=int(matrix.shape[1]))
    starts = np.cumsum(counts) - counts
    per_entry = counts[inner_flat]
    source = np.repeat(np.arange(entries.n_entries, dtype=np.int64), per_entry)
    within = np.arange(int(per_entry.sum())) - np.repeat(
        np.cumsum(per_entry) - per_entry, per_entry
    )
    chosen = starts[inner_flat][source] + within
    out = (lead_flat[source] * n_out + rows[chosen]) * n_rest + rest_flat[source]
    return Linear(out_grid, out.astype(np.int64), source, weights[chosen].astype(np.complex128))


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
    specs: list[Copy | Linear] = []
    for grid in layout.grids:
        if grid.name not in ORIENTATION_AXES:
            specs.append(Copy(grid.name))
            continue
        names = ORIENTATION_AXES[grid.name]
        first = _orientation_positions(grid, names)
        k = len(names)
        if grid.name in ("S1", "S2"):
            out_grid = grid.replace(
                axes=grid.axes[:first], coords=grid.coords[:first], present=grid.present[..., 0]
            )
            matrix = np.full((1, L), 1.0 / L, dtype=np.complex128)
        else:
            table = _iso_table(L, k)  # (l1, *d) -> input position
            rows = np.broadcast_to(np.arange(L ** (k - 1)).reshape(table.shape[1:]), table.shape)
            matrix = (
                np.bincount((rows * L**k + table).ravel(), minlength=L ** (k - 1) * L**k)
                .reshape(L ** (k - 1), L**k)
                .astype(np.complex128)
                / L
            )
            base = grid.present.reshape(*grid.shape[:first], -1)[..., 0]
            present = np.broadcast_to(
                base[(Ellipsis, *(None,) * (k - 1))], (*base.shape, *(L,) * (k - 1))
            )
            out_grid = grid.replace(
                axes=grid.axes[:first] + ISO_AXES[grid.name],
                coords=grid.coords[:first] + (tuple(range(L)),) * (k - 1),
                present=np.array(present),
            )
        specs.append(_on_axes(layout.entries(grid.name), out_grid, first, k, matrix))
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
def angular_fourier(layout: Layout, harmonics: int, after_iso: bool) -> FloatMap:
    """The angular Fourier transform (M-RED-ANG; STL ``to_angular_ft``): an orthonormal DFT over
    the orientation axes of S3 and S4 (``norm="ortho"``), keeping the first ``harmonics``
    harmonics along each. After ``iso`` it acts on the orientation differences. Each axis name
    gets the prefix ``h``."""
    specs: list[Copy | Linear] = []
    for grid in layout.grids:
        if grid.name not in ("S3", "S4"):
            specs.append(Copy(grid.name))
            continue
        names = ISO_AXES[grid.name] if after_iso else ORIENTATION_AXES[grid.name]
        first = _orientation_positions(grid, names)
        k = len(names)
        L = grid.shape[first]  # noqa: N806
        dft = _dft(L, harmonics)
        matrix: NDArray[Any] = dft
        for _ in range(k - 1):
            matrix = np.kron(matrix, dft)
        keep: tuple[Any, ...] = (Ellipsis, *(slice(0, harmonics),) * k)
        out_grid = grid.replace(
            axes=grid.axes[:first] + tuple(f"h{n}" for n in names),
            coords=grid.coords[:first] + (tuple(range(harmonics)),) * k,
            present=np.array(grid.present[keep]),
        )
        specs.append(_on_axes(layout.entries(grid.name), out_grid, first, k, matrix))
    return make_map(layout, specs)


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


@functools.lru_cache(maxsize=64)
def scale_cosine(layout: Layout, n_scales: int, harmonics: int, dj: int) -> FloatMap:
    """The scale cosine transform (M-RED-SCALE; STL ``to_scale_ft``), after ``iso``: the scales
    of S3 and S4 reparametrised as ``(j1, a)`` and ``(j1, a, b)``, an orthonormal DCT-II along
    j1 at fixed differences, the first ``harmonics`` harmonics kept; the axis j1 becomes ``k``.
    Missing inputs count as 0; an output exists where its input at j1 = k does, as STL puts
    NaN back at the same positions."""
    J = n_scales  # noqa: N806
    basis = cosine_basis(J)
    specs: list[Copy | Linear] = []
    for grid in layout.grids:
        if grid.name not in SCALE_AXES:
            specs.append(Copy(grid.name))
            continue
        before, after = SCALE_AXES[grid.name]
        n = len(before)
        first = grid.axis(before[0])
        if tuple(grid.axis(name) for name in before) != tuple(range(first, first + n)):
            raise ValueError(f"{grid.name}: the scale axes {before} must be consecutive.")
        table, valid = _scale_table(J, dj, n)  # (j1, *differences)
        n_differences = dj ** (n - 1)
        # matrix[(k, differences), table[j1, differences]] = basis[k, j1] where valid.
        k_index, j1_index, d_index = np.meshgrid(
            np.arange(harmonics), np.arange(J), np.arange(n_differences), indexing="ij"
        )
        flat_valid = valid.reshape(J, n_differences)[j1_index, d_index]
        rows = (k_index * n_differences + d_index)[flat_valid]
        cols = table.reshape(J, n_differences)[j1_index, d_index][flat_valid]
        weights = basis[k_index, j1_index][flat_valid]
        size = harmonics * n_differences * J**n
        flat = rows * J**n + cols
        matrix = (
            np.bincount(flat, weights=weights, minlength=size)
            .reshape(harmonics * n_differences, J**n)
            .astype(np.complex128)
        )

        grid_lead = grid.present.shape[:first]
        grid_rest = grid.present.shape[first + n :]
        present_flat = grid.present.reshape(*grid_lead, J**n, *grid_rest)
        present = np.take(present_flat, table.reshape(-1), axis=first).reshape(
            *grid_lead, *table.shape, *grid_rest
        )
        present = present & valid.reshape((1,) * first + table.shape + (1,) * len(grid_rest))
        kept_present = np.array(present[(*(slice(None),) * first, slice(0, harmonics))])
        sizes = (harmonics, *(dj,) * (n - 1))
        out_grid = grid.replace(
            axes=grid.axes[:first] + after + grid.axes[first + n :],
            coords=grid.coords[:first]
            + tuple(tuple(range(s)) for s in sizes)
            + grid.coords[first + n :],
            present=kept_present,
        )
        specs.append(_on_axes(layout.entries(grid.name), out_grid, first, n, matrix))
    return make_map(layout, specs)
