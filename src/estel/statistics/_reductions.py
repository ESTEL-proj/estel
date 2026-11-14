"""The reductions of the statistics: ``iso``, ``angular_ft`` and ``scale_ft`` (maths.md
M-RED-ISO, M-RED-ANG, M-RED-SCALE, OP-5, OP-17; STL ``to_iso``, ``to_angular_ft``,
``to_scale_ft``; task T2.5).

In ESTEL "reduction" means these three operations only. Each takes the full grids and their
static layout (``_grid.FamilyGrid``) and returns new ones; nothing is modified. They are linear
in the coefficients, written with static index tables and the L0 protocol, so that they are
differentiable on every backend. Where no coefficient exists the result is NaN, as in STL, and
the layout says so statically: ``iso`` and ``angular_ft`` keep the positions of their inputs,
since a coefficient exists for every orientation or for none; ``scale_ft`` sets missing inputs to
0 for the transform and puts NaN back afterwards, at the same positions, as STL does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend._protocol import Backend
from ._arrays import constant, nan_absent, zero_absent
from ._grid import FamilyGrid

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
"""The scale axes of S3 and S4 before ``scale_ft``, and after it (M-RED-SCALE)."""


Values = dict[str, Any]
Grids = tuple[FamilyGrid, ...]


def _orientation_positions(grid: FamilyGrid, names: tuple[str, ...]) -> tuple[int, ...]:
    """The positions of the orientation axes ``names`` in the arrays (batch axis included);
    they are the last axes."""
    positions = tuple(grid.axis(n) + 1 for n in names)
    last = len(grid.axes)
    if positions != tuple(range(last - len(names) + 1, last + 1)):
        raise ValueError(f"{grid.name}: the orientation axes {names} must be the last axes.")
    return positions


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


def iso(
    backend: Backend, values: Values, grids: Grids, n_orientations: int
) -> tuple[Values, Grids]:
    """Isotropy (M-RED-ISO; STL ``to_iso``): S1 and S2 averaged over l; S3 and S4 averaged over
    the first orientation at fixed differences modulo L, ``(1/L) Σ_{l1} S3[.., l1, l1 + d]`` and
    ``(1/L) Σ_{l1} S4[.., l1, l1 + d2, l1 + d3]``. The orientations are treated as cyclic over
    [0, π), as in STL (OP-5)."""
    xp = backend.xp
    L = n_orientations  # noqa: N806
    out: Values = dict(values)
    new: list[FamilyGrid] = []
    for grid in grids:
        if grid.name not in ORIENTATION_AXES:
            new.append(grid)
            continue
        names = ORIENTATION_AXES[grid.name]
        _orientation_positions(grid, names)
        x = values[grid.name]
        k = len(names)
        kept = len(grid.axes) - k
        if grid.name in ("S1", "S2"):
            reduced = xp.mean(x, axis=-1)
            new_axes, new_coords = grid.axes[:kept], grid.coords[:kept]
            present = grid.present[..., 0]
        else:
            lead = tuple(x.shape[: 1 + kept])
            table = _iso_table(L, k)
            flat = xp.reshape(x, (*lead, L**k))
            index = constant(backend, table.reshape(-1), x, "int32")
            gathered = xp.reshape(xp.take(flat, index, axis=-1), (*lead, *table.shape))
            reduced = xp.sum(gathered, axis=-k) / L
            new_axes = grid.axes[:kept] + ISO_AXES[grid.name]
            new_coords = grid.coords[:kept] + (tuple(range(L)),) * (k - 1)
            first: tuple[Any, ...] = (Ellipsis, *(0,) * k)
            base = grid.present[first]
            expand: tuple[Any, ...] = (Ellipsis, *(None,) * (k - 1))
            present = np.broadcast_to(base[expand], (*base.shape, *(L,) * (k - 1)))
        out = {**out, grid.name: reduced}
        new.append(grid.replace(axes=new_axes, coords=new_coords, present=np.array(present)))
    return out, tuple(new)


# ---------------------------------------------------------------------- angular_ft


def angular_ft(
    backend: Backend, values: Values, grids: Grids, harmonics: int, after_iso: bool
) -> tuple[Values, Grids]:
    """The angular Fourier transform (M-RED-ANG; STL ``to_angular_ft``): an orthonormal DFT over
    the orientation axes of S3 and S4 (``norm="ortho"``), keeping the first ``harmonics``
    harmonics along each. After ``iso`` it acts on the orientation differences. Each axis name
    gets the prefix ``h``."""
    xp = backend.xp
    out: Values = dict(values)
    new: list[FamilyGrid] = []
    for grid in grids:
        if grid.name not in ("S3", "S4"):
            new.append(grid)
            continue
        names = ISO_AXES[grid.name] if after_iso else ORIENTATION_AXES[grid.name]
        positions = _orientation_positions(grid, names)
        x = values[grid.name]
        transformed = xp.fft.fftn(x, axes=positions, norm="ortho")
        keep: tuple[Any, ...] = (Ellipsis, *(slice(0, harmonics),) * len(names))
        kept = len(grid.axes) - len(names)
        out = {**out, grid.name: transformed[keep]}
        new.append(
            grid.replace(
                axes=grid.axes[:kept] + tuple(f"h{n}" for n in names),
                coords=grid.coords[:kept] + (tuple(range(harmonics)),) * len(names),
                present=np.array(grid.present[keep]),
            )
        )
    return out, tuple(new)


# ---------------------------------------------------------------------- scale_ft


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


def scale_ft(
    backend: Backend,
    values: Values,
    grids: Grids,
    n_scales: int,
    harmonics: int,
    dj: int,
) -> tuple[Values, Grids]:
    """The scale cosine transform (M-RED-SCALE; STL ``to_scale_ft``), after ``iso``: the scales
    of S3 and S4 reparametrised as ``(j1, a)`` and ``(j1, a, b)``, an orthonormal DCT-II along
    j1 at fixed differences, the first ``harmonics`` harmonics kept. Missing entries are 0 for
    the transform and NaN afterwards, at the same positions; the axis j1 becomes ``k``."""
    xp = backend.xp
    J = n_scales  # noqa: N806
    out: Values = dict(values)
    new: list[FamilyGrid] = []
    for grid in grids:
        if grid.name not in SCALE_AXES:
            new.append(grid)
            continue
        before, after = SCALE_AXES[grid.name]
        n = len(before)
        first = grid.axis(before[0])
        if tuple(grid.axis(name) for name in before) != tuple(range(first, first + n)):
            raise ValueError(f"{grid.name}: the scale axes {before} must be consecutive.")
        x = values[grid.name]
        position = first + 1  # batch axis first
        table, valid = _scale_table(J, dj, n)
        lead, rest = tuple(x.shape[:position]), tuple(x.shape[position + n :])
        flat = xp.reshape(x, (*lead, J**n, *rest))
        index = constant(backend, table.reshape(-1), x, "int32")
        reparam = xp.reshape(xp.take(flat, index, axis=position), (*lead, *table.shape, *rest))

        grid_lead = grid.present.shape[:first]
        grid_rest = grid.present.shape[first + n :]
        present_flat = grid.present.reshape(*grid_lead, J**n, *grid_rest)
        present = np.take(present_flat, table.reshape(-1), axis=first).reshape(
            *grid_lead, *table.shape, *grid_rest
        )
        present = present & valid.reshape((1,) * first + table.shape + (1,) * len(grid_rest))

        safe = zero_absent(backend, present, reparam)
        basis = constant(backend, cosine_basis(J), x, _real_dtype(x))
        pairs = backend.complex_to_real_pairs(safe)
        transformed = backend.real_pairs_to_complex(
            xp.stack(
                [
                    _dct(backend, pairs[..., 0], basis, position),
                    _dct(backend, pairs[..., 1], basis, position),
                ],
                axis=-1,
            )
        )
        keep = (*(slice(None),) * position, slice(0, harmonics))
        kept_present = np.array(present[(*(slice(None),) * first, slice(0, harmonics))])
        out = {**out, grid.name: nan_absent(backend, kept_present, transformed[keep])}
        sizes = (harmonics, *(dj,) * (n - 1))
        new.append(
            grid.replace(
                axes=grid.axes[:first] + after + grid.axes[first + n :],
                coords=grid.coords[:first]
                + tuple(tuple(range(s)) for s in sizes)
                + grid.coords[first + n :],
                present=kept_present,
            )
        )
    return out, tuple(new)


def _dct(backend: Backend, x: Any, basis: Any, axis: int) -> Any:
    """``y[.., k, ..] = Σ_j basis[k, j] x[.., j, ..]`` along ``axis`` of the real array ``x``."""
    xp = backend.xp
    order = (*range(axis), *range(axis + 1, len(x.shape)), axis)
    inverse = tuple(int(i) for i in np.argsort(order))
    moved = xp.permute_dims(x, order)
    return xp.permute_dims(xp.matmul(moved, xp.permute_dims(basis, (1, 0))), inverse)


def _real_dtype(x: Any) -> Any:
    name = str(x.dtype).split(".")[-1]
    return {"complex64": "float32", "complex128": "float64"}.get(name, name)
