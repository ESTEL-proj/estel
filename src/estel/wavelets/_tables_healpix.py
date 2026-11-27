"""The HEALPix tables: stencils and decimation (Part II, §12.5; task T5.1, issue #26).

Where each block comes from (the five blocks decided on 24 November 2026, Part IV, §22):

1. the NESTED pixel numbering and centres (:func:`nest2xyf`, :func:`xyf2nest`, :func:`xyf2ang`)
   and the RING to NESTED conversion (:func:`ring2nest`) are written in ``_pixels_healpix.py``
   from the published HEALPix formulas only (Górski et al. 2005, ApJ 622, 759);
2. the bilinear interpolation of 4 pixels (``_pixels_healpix.interpolation_ring``) is written
   from its definition in ``docs/maths.md``, M-CONV-HEALPIX, step 3; :func:`interp_weights_nested`
   applies it to directions, through degrees as healpix-analyse ``aac5e26`` does;
3. the rotations of the stencil, the gauge ``"cosmo"`` only (:func:`rotation_matrices`);
4. the stencil and its binding on a partial sky (:func:`local_kernel_grid`,
   :func:`_bind_support`), with the rule decided by the maintainer on 24 November 2026 (issue
   #26), which differs from STL at the ill-conditioned points of the edge (maths.md OP-110);
5. the decimation (:func:`centres_lonlat`, :func:`_haversine`, :func:`decimation_healpix`),
   whose search of the candidate fine pixels is ESTEL's own (:func:`_cone_pairs`), exact and in
   NumPy, where STL uses healpix-geo.

Blocks 3 to 5 are ported from ``TorchGeometry`` of STL `HealpixUpdate` ``90eb1a9``
(``STL_main/TorchGeometry.py``: ``rotation_matrices``, ``local_kernel_grid``, ``bind_support``,
``centres_lonlat``, ``_haversine``, ``smooth_decimation_csr``), which reproduces in vectorised
code the tables of healpix-analyse ``aac5e26`` (``HealPixConv``, ``HealPixDown``) on the
sphere. Blocks 1 and 2, which ``TorchGeometry`` transcribes from HEALPix C++ (GPL-2), were
rewritten from the published sources above, not ported (Part I, §1.5; Part IV, §21, F23).

The stencils are built by batches of output pixels (:data:`_STENCIL_BATCH`; task T5.7), as
the candidate search of the decimation is (:data:`_CONE_CHUNK`). Not ported: the cast of the
kernel through float32 (maths.md OP-46), ``TorchConv`` and everything torch. Only the sphere:
ESTEL has no ellipsoid (OP-45).

The tables are built once, in NumPy, when the plan is built (tag N of Part II, §12); they are
static, never traced and never differentiated. Every weight is float64, cast by the plan to the
dtype of the data (OP-46); every index is int32, which holds the 12 · 4^13 pixels of ``nside``
8192, and a deeper ``depth`` raises an error. The code sets no other limit on ``nside``.
``nside`` 128 is the documented target of 1.0.0 on one GPU: with L = 4 and K = 5 its stencil
tables hold 196 608 · 4 · 25 · 4 = 78.6 M entries, 315 MB of int32 indices and 629 MB of float64
weights (315 MB once the plan casts them to float32); its decimation tables, 1.4 M entries.

Table layout (for T5.5, which applies the tables, and T6.3, which checks the signature of
``gather_weighted`` against them). Both fit the L0 primitives of Part II, §12.3, applied along
the pixel axis of a map ``x`` of shape ``(..., Npix)`` whose pixels are ``cell_ids``, ascending:

* **Convolution** (M-CONV-HEALPIX), :func:`stencils_healpix`: ``index`` and ``weight`` of shape
  ``(L · Npix, K², 4)``. Row ``r = g · Npix + k`` is gauge ``g`` at pixel ``cell_ids[k]``;
  column ``p = (v + c) K + (u + c)`` the stencil point (u, v); the last axis the 4 neighbours of
  the bilinear interpolation, ``index`` their positions in ``cell_ids`` and ``weight`` their
  renormalised weights (0 for an absent neighbour, whose index is then 0). Then::

      z = gather_weighted(x, -1, index, weight)       # (..., L · Npix, K²): the interpolated
      y = z @ W                                       # stencil values; W the kernel (K², )
      y = reshape(y, (..., L, Npix))                  # of kernel_weights_healpix

  the product with the complex kernel being two real products, with its real and imaginary
  parts (Part I, §3.1, rule 6).
* **Decimation** (M-DOWN-HEALPIX), :func:`decimation_healpix`: a CSR matrix of ``nnz`` entries,
  sorted by coarse pixel then by fine pixel. ``index`` and ``weight`` of shape ``(nnz, 1, 1)``,
  the positions in ``cell_ids`` of the fine pixels and their weights; ``segment_ids`` of shape
  ``(nnz, )``, the position of the coarse pixel in ``cell_ids_out``. Then::

      z = gather_weighted(x, -1, index, weight)       # (..., nnz, 1)
      y = segment_sum(z[..., 0], -1, segment_ids, len(cell_ids_out))   # (..., Npix_coarse)

``tests/geometry/test_geometry_healpix.py`` checks each pixel function against healpy (in the
tests only, Part I, §1.5) and the rotations against M-CONV-HEALPIX, step 2;
``tests/parity/test_tables_healpix.py`` checks the tables, as operators (OP-92), against STL's
fixtures (T5.3).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from estel.data._healpix import check_cell_ids, check_depth
from estel.data._masks import SUPPORT_THRESHOLD_HEALPIX
from estel.wavelets._banks_healpix import GAUGES_HEALPIX, KERNEL_SIZE_HEALPIX, _check_kernel_size
from estel.wavelets._checks import _check_n_orientations
from estel.wavelets._pixels_healpix import (
    interpolation_ring,
    nest2xyf,
    ring2nest,
    xyf2ang,
    xyf2nest,
)

__all__ = [
    "MAX_DEPTH",
    "DecimationHealpix",
    "StencilsHealpix",
    "centres_lonlat",
    "decimation_healpix",
    "interp_weights_nested",
    "local_kernel_grid",
    "nest2xyf",
    "ring2nest",
    "rotation_matrices",
    "stencils_healpix",
    "xyf2ang",
    "xyf2nest",
]

MAX_DEPTH = 13
"""The deepest level the tables index: 12 · 4^13 pixels (``nside`` 8192) fit int32 indices."""

_CONE_CHUNK = 1 << 14
"""Coarse pixels searched at once by :func:`_cone_pairs`: bounds its temporary arrays."""

_STENCIL_BATCH = 1 << 10
"""Output pixels whose stencils :func:`stencils_healpix` builds at once (task T5.7, issue #168):
the NumPy temporaries of the interpolation, about 30 arrays of L · K² · 4 entries per pixel,
are bounded by a batch (0.11 GB with L = 4 and K = 5) instead of growing with the map (22 GB
at ``nside`` 128). From 256 to 4096 pixels a batch, the build takes the same time. The tables
do not depend on it."""


# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------


def _check_depth(depth: object, minimum: int = 0) -> int:
    """``depth`` as an int in [minimum, MAX_DEPTH]: the check of the data
    (``estel.data._healpix.check_depth``), then the int32 limit of the tables."""
    value = check_depth(depth, minimum=minimum)
    if value > MAX_DEPTH:
        raise ValueError(
            f"depth={value} (nside {1 << value}): the HEALPix tables index pixels in int32, "
            f"which holds at most depth {MAX_DEPTH} (nside {1 << MAX_DEPTH}, 12 · 4^{MAX_DEPTH} "
            "pixels). Use a coarser map."
        )
    return value


def _pixels(cell_ids: ArrayLike | None, depth: int) -> NDArray[np.int64]:
    """The NESTED pixels as int64: all of them for ``None``, else checked as the data check
    them (``estel.data._healpix.check_cell_ids``; maths.md OP-90)."""
    if cell_ids is None:
        return np.arange(12 * 4**depth, dtype=np.int64)
    return check_cell_ids(cell_ids, depth)


def interp_weights_nested(
    vecs: NDArray[np.float64], depth: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The 4 NESTED pixels and bilinear weights of directions ``vecs`` ``(M, 3)`` (sphere).

    As healpix-analyse's ``_get_interp_weights``: direction -> (θ, φ) -> degrees -> radians
    again, so that the rounding is the library's. Returns two ``(M, 4)`` arrays; the 4 weights
    of a direction sum to 1.
    """
    norm = np.maximum(np.linalg.norm(vecs, axis=-1), 1e-12)
    theta = np.arccos(np.clip(vecs[:, 2] / norm, -1.0, 1.0))
    phi = np.mod(np.arctan2(vecs[:, 1] / norm, vecs[:, 0] / norm), 2.0 * math.pi)
    lon_deg = np.rad2deg(phi)
    lat_deg = 90.0 - np.rad2deg(theta)
    theta2 = np.deg2rad(90.0 - lat_deg)
    phi2 = np.mod(np.deg2rad(lon_deg), 2.0 * math.pi)
    pix_ring, weights = interpolation_ring(theta2, phi2, depth)
    return ring2nest(pix_ring, depth), weights


# ---------------------------------------------------------------------------------------------
# 3. Rotations, 4. the stencil and its binding
# ---------------------------------------------------------------------------------------------


def rotation_matrices(
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    L: int,  # noqa: N803
) -> NDArray[np.float64]:
    """The rotations of the gauge ``"cosmo"`` (maths.md M-CONV-HEALPIX, step 2), ``(N, L, 3, 3)``.

    R = R_gauge(alpha_g) · R_z(φ) · R_y(θ) for pixel (θ, φ) and gauge g, R_gauge a rotation about
    the pixel's direction by alpha_g = -φ + π g / L in the northern hemisphere (θ ≤ π/2, the
    equator included) and alpha_g = φ - π g / L in the southern one (healpix-analyse's
    ``_build_rotation_matrices``).
    """
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    zero = np.zeros_like(theta)
    r_base = np.stack(
        [
            np.stack([cp * ct, -sp, cp * st], axis=-1),
            np.stack([sp * ct, cp, sp * st], axis=-1),
            np.stack([-st, zero, ct], axis=-1),
        ],
        axis=-2,
    )  # (N, 3, 3)
    n = r_base[:, :, 2]
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    is_south = theta > math.pi / 2
    alpha_base = np.where(is_south, phi, -phi)
    sign_g = np.where(is_south, -1.0, 1.0)
    shifts = np.arange(L, dtype=np.float64) * (math.pi / L)
    alpha = alpha_base[:, None] + sign_g[:, None] * shifts[None, :]  # (N, L)
    ca, sa = np.cos(alpha)[..., None, None], np.sin(alpha)[..., None, None]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    skew = np.stack(
        [
            np.stack([zero, -nz, ny], axis=-1),
            np.stack([nz, zero, -nx], axis=-1),
            np.stack([-ny, nx, zero], axis=-1),
        ],
        axis=-2,
    )[:, None]  # (N, 1, 3, 3)
    outer = (n[:, :, None] * n[:, None, :])[:, None]
    r_gauge = np.eye(3) * ca + skew * sa + outer * (1.0 - ca)  # (N, L, 3, 3)
    result: NDArray[np.float64] = np.matmul(r_gauge, r_base[:, None])
    return result


def local_kernel_grid(kernel_size: int, depth: int) -> NDArray[np.float64]:
    """The K x K stencil around the North Pole, unit vectors ``(K², 3)`` (M-CONV-HEALPIX, step 1).

    Point (u, v) at colatitude √(u² + v²) alpha and azimuth atan2(v, u), with
    alpha = √(4π / (12 nside²)), nside = 2^depth; entry p = (v + c) K + (u + c).
    """
    nside = 1 << depth
    grid = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(grid, grid)
    alpha_pix = np.sqrt(4 * np.pi / (12 * nside**2))
    dtheta = np.sqrt(xx**2 + yy**2).ravel() * alpha_pix
    dphi = np.arctan2(yy, xx).ravel()
    return np.stack(
        [np.sin(dtheta) * np.cos(dphi), np.sin(dtheta) * np.sin(dphi), np.cos(dtheta)], axis=-1
    ).astype(np.float64)


def _stencil_interpolation(
    ids: NDArray[np.int64],
    depth: int,
    L: int,  # noqa: N803
    kernel_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The full-sky interpolation of the rotated stencils: NESTED pixels and weights,
    ``(L, N, K², 4)``, before the binding on ``ids``."""
    theta, phi = xyf2ang(*nest2xyf(ids, depth), depth)
    # healpix-analyse reads the pixel centres from healpix-geo, in degrees.
    theta = np.deg2rad(90.0 - np.rad2deg(math.pi / 2 - theta))
    phi = np.deg2rad(np.rad2deg(phi))
    rotations = rotation_matrices(theta, phi, L)  # (N, L, 3, 3)
    stencil = local_kernel_grid(kernel_size, depth)  # (P, 3)
    rotated = np.einsum("kgij,pj->gkpi", rotations, stencil)  # (L, N, P, 3)
    index, weight = interp_weights_nested(rotated.reshape(-1, 3), depth)
    shape = (*rotated.shape[:-1], 4)
    return index.reshape(shape), weight.reshape(shape)


def _bind_support(
    index: NDArray[np.int64],
    weight: NDArray[np.float64],
    ids: NDArray[np.int64],
    kernel_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """Bind the neighbours ``(L, N, P, 4)`` on the pixels ``ids`` (ascending): the partial-sky
    rule of maths.md M-CONV-HEALPIX (OP-50), as decided on 24 November 2026 (issue #26).

    A neighbour counts as absent when it is outside ``ids`` or when its interpolation weight is
    below ``SUPPORT_THRESHOLD_HEALPIX`` (1e-12, the threshold of the support test of
    M-MASK-HEALPIX, defined in ``estel.data._masks``). The weights of the remaining neighbours of a
    stencil point are renormalised to sum to 1 exactly, with no floor on their sum. A stencil
    point with no remaining neighbour takes the neighbours of the stencil centre of its pixel
    and gauge, which reads the pixel itself.

    **This differs from STL** (healpix-analyse's ``_bind_support_batched`` and
    ``TorchGeometry.bind_support``) at the ill-conditioned points of the edge (maths.md
    OP-110): there STL keeps neighbours of weight 0 or of rounding
    size, divides by ``max(sum, 1e-12)``, and so returns values set by the rounding of the
    interpolation weights. Everywhere else, and on the full sky, the result is STL's to rounding.

    Returns the positions in ``ids`` (0 where absent), the weights, and the stencil points that
    had no remaining neighbour.
    """
    n_ids = ids.size

    def lookup(idx: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        pos = np.minimum(np.searchsorted(ids, idx), n_ids - 1)
        found: NDArray[np.bool_] = np.asarray(ids[pos] == idx)
        return pos, found

    _, found = lookup(index)
    present = found & (weight >= SUPPORT_THRESHOLD_HEALPIX)
    empty: NDArray[np.bool_] = ~np.asarray(present.any(axis=-1))  # (L, N, P)
    centre = (kernel_size // 2) * (kernel_size + 1)
    index = np.where(empty[..., None], index[:, :, centre : centre + 1], index)
    weight = np.where(empty[..., None], weight[:, :, centre : centre + 1], weight)
    pos, found = lookup(index)
    present = found & (weight >= SUPPORT_THRESHOLD_HEALPIX)
    w = np.where(present, weight, 0.0)
    total = w.sum(axis=-1, keepdims=True)
    if not np.all(total > 0):
        # The stencil centre is the pixel centre, read with weight 1 by the pixel itself.
        raise RuntimeError(
            "stencils_healpix: a stencil centre reads no pixel of cell_ids; this is a bug of "
            "ESTEL, please report it with cell_ids and depth."
        )
    positions: NDArray[np.int64] = np.where(present, pos, 0)
    return positions, w / total, empty


@dataclass(frozen=True, eq=False)
class StencilsHealpix:
    """The convolution tables of one level, for ``gather_weighted`` (module docstring).

    ``index`` int32 and ``weight`` float64, shape ``(L · Npix, K², 4)``, row ``g · Npix + k``.
    """

    index: NDArray[np.int32]
    weight: NDArray[np.float64]
    n_gauges: int
    kernel_size: int


def stencils_healpix(
    cell_ids: ArrayLike | None,
    depth: int,
    *,
    L: int = GAUGES_HEALPIX,  # noqa: N803
    kernel_size: int = KERNEL_SIZE_HEALPIX,
) -> StencilsHealpix:
    """The interpolation tables of the rotated stencils at one level (maths.md M-CONV-HEALPIX).

    ``cell_ids``: the NESTED pixels of the map, strictly increasing, or ``None`` for the full
    sky; ``depth``: log2(nside) of the level, 0 to :data:`MAX_DEPTH`; ``L``: the number of
    gauges of ``"cosmo"``; ``kernel_size``: K, odd. The kernel is not in the tables: it is the
    same at every level and for every gauge (M-WAV-HEALPIX, :func:`kernel_weights_healpix`). On a
    partial sky, the rule of M-CONV-HEALPIX (OP-50) as decided on issue #26, which differs from STL
    at the ill-conditioned points of the edge (OP-110; :func:`_bind_support`). Layout: the
    module docstring.
    """
    depth = _check_depth(depth)
    ids = _pixels(cell_ids, depth)
    n_gauges = _check_n_orientations(L)
    size = _check_kernel_size(kernel_size)
    # Built by batches of output pixels (:data:`_STENCIL_BATCH`): each pixel's stencil depends
    # on that pixel and on ``ids`` only, so the tables equal those built at once, bit for bit.
    # Each batch is cast to int32 before it is kept; the weights are joined first, then the
    # positions, so that each list is dropped once joined.
    indices, weights = [], []
    for begin in range(0, ids.size, _STENCIL_BATCH):
        found, found_weight = _stencil_interpolation(
            ids[begin : begin + _STENCIL_BATCH], depth, n_gauges, size
        )
        positions, bound, _ = _bind_support(found, found_weight, ids, size)
        indices.append(positions.astype(np.int32))
        weights.append(bound)
    weight = np.concatenate(weights, axis=1)
    del weights
    index = np.concatenate(indices, axis=1)
    del indices
    rows = (n_gauges * ids.size, size * size, 4)
    return StencilsHealpix(
        index=index.reshape(rows),
        weight=weight.reshape(rows),
        n_gauges=n_gauges,
        kernel_size=size,
    )


# ---------------------------------------------------------------------------------------------
# 5. The decimation (port of HealPixDown("smooth"), sphere)
# ---------------------------------------------------------------------------------------------


def _haversine(
    lat1: NDArray[np.float64],
    lon1: NDArray[np.float64],
    lat2: NDArray[np.float64],
    lon2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The angular distance, in radians, between two points given in radians."""
    dlat = 0.5 * (lat2 - lat1)
    dlon = 0.5 * (lon2 - lon1)
    a = np.clip(np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2, 0.0, 1.0)
    distance: NDArray[np.float64] = 2.0 * np.arcsin(np.sqrt(a))
    return distance


def centres_lonlat(
    ids: NDArray[np.int64], depth: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The longitude and latitude, in radians, of NESTED pixel centres at ``depth``, rounded
    through degrees as healpix-geo's output that healpix-analyse reads."""
    theta, phi = xyf2ang(*nest2xyf(ids, depth), depth)
    return np.deg2rad(np.rad2deg(phi)), np.deg2rad(90.0 - np.rad2deg(theta))


def _unit_vectors(lon: NDArray[np.float64], lat: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _cone_pairs(
    lon_c: NDArray[np.float64],
    lat_c: NDArray[np.float64],
    lon_f: NDArray[np.float64],
    lat_f: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Every pair (coarse c, fine f) whose angular distance gamma is at most ``radius``, and no
    other, with gamma: ESTEL's own search of the candidates, exact (STL uses healpix-geo's
    ``cone_coverage``, then drops the candidates beyond ``radius``).

    A grid of cubic cells of side ``radius`` over [-1, 1]³ holds the fine centres. Two points
    at angular distance gamma ≤ radius are at a chord 2 sin(gamma / 2) < radius, so in the same
    cell or in two adjacent ones: each coarse centre reads the 27 cells around its own, and
    keeps the fine centres with gamma ≤ radius, gamma as healpix-analyse computes it
    (:func:`_haversine`). The
    cost is linear in the number of pixels, up to the sort of the cells. Returns the pairs
    sorted by coarse then by fine position, and their gamma.
    """
    cell = radius
    n_cell = math.floor(2.0 / cell) + 1

    def cells(points: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.clip(np.floor((points + 1.0) / cell).astype(np.int64), 0, n_cell - 1)

    def keys(c: NDArray[np.int64]) -> NDArray[np.int64]:
        return (c[..., 0] * n_cell + c[..., 1]) * n_cell + c[..., 2]

    fine_keys = keys(cells(_unit_vectors(lon_f, lat_f)))
    order = np.argsort(fine_keys, kind="stable")
    sorted_keys = fine_keys[order]
    coarse_cells = cells(_unit_vectors(lon_c, lat_c))
    offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)

    rows_list, cols_list, gamma_list = [], [], []
    for begin in range(0, lon_c.size, _CONE_CHUNK):
        chunk = np.arange(begin, min(begin + _CONE_CHUNK, lon_c.size))
        neighbours = coarse_cells[chunk][:, None, :] + offsets[None]  # (Q, 27, 3)
        inside = np.all((neighbours >= 0) & (neighbours < n_cell), axis=-1)
        neighbour_keys = keys(neighbours)
        start = np.searchsorted(sorted_keys, neighbour_keys, side="left")
        count = np.where(
            inside, np.searchsorted(sorted_keys, neighbour_keys, side="right") - start, 0
        )
        count, start = count.ravel(), start.ravel()
        first = np.cumsum(count) - count
        rank = np.arange(int(count.sum())) - np.repeat(first, count)
        rows = np.repeat(np.repeat(chunk, offsets.shape[0]), count)
        cols = order[np.repeat(start, count) + rank]
        gamma = _haversine(lat_c[rows], lon_c[rows], lat_f[cols], lon_f[cols])
        keep = ~(gamma > radius)
        rows_list.append(rows[keep])
        cols_list.append(cols[keep])
        gamma_list.append(gamma[keep])
    rows, cols, gamma = (np.concatenate(a) for a in (rows_list, cols_list, gamma_list))
    sort = np.lexsort((cols, rows))
    return rows[sort], cols[sort], gamma[sort]


@dataclass(frozen=True, eq=False)
class DecimationHealpix:
    """The decimation tables of one step, for ``gather_weighted`` then ``segment_sum``
    (module docstring).

    ``index`` int32 and ``weight`` float64, shape ``(nnz, 1, 1)``; ``segment_ids`` int32,
    ``(nnz, )``, non-decreasing; ``cell_ids_out`` int64, the coarse NESTED pixels, ascending.
    """

    index: NDArray[np.int32]
    weight: NDArray[np.float64]
    segment_ids: NDArray[np.int32]
    cell_ids_out: NDArray[np.int64]

    @property
    def num_segments(self) -> int:
        """The number of coarse pixels, the ``num_segments`` of ``segment_sum``."""
        return int(self.cell_ids_out.size)


def decimation_healpix(cell_ids: ArrayLike | None, depth: int) -> DecimationHealpix:
    """The tables of one decimation step, from ``depth`` to ``depth - 1`` (maths.md M-DOWN-HEALPIX).

    ``cell_ids``: the NESTED fine pixels, strictly increasing, or ``None`` for the full sky;
    ``depth``: log2(nside) of the fine level, 1 to :data:`MAX_DEPTH`. The coarse pixels are the
    parents of the fine ones, ascending. The weight of fine pixel n for coarse pixel n₀ is
    exp(-gamma² / 2 sigma²), gamma the angular distance of their centres, over every available
    fine pixel with gamma ≤ rho, rho = 3 fine pixel sizes, sigma = rho / 2, normalised to sum to 1
    for each coarse pixel. On a partial sky, only the fine pixels of ``cell_ids`` enter (OP-50).
    Layout: the module docstring.
    """
    depth = _check_depth(depth, minimum=1)
    fine = _pixels(cell_ids, depth)
    nside = 1 << depth
    coarse = np.unique(fine // 4)
    lon_c, lat_c = centres_lonlat(coarse, depth - 1)
    lon_f, lat_f = centres_lonlat(fine, depth)
    pix_deg = math.degrees(math.sqrt(4.0 * math.pi / (12 * nside * nside)))
    radius = math.radians(3.0 * pix_deg)
    sigma = radius / 2.0
    rows, cols, gamma = _cone_pairs(lon_c, lat_c, lon_f, lat_f, radius)
    w = np.exp(-0.5 * (gamma / sigma) ** 2)
    row_sum = np.bincount(rows, weights=w, minlength=coarse.size)
    if not np.all(row_sum > 0):
        # healpix-analyse's fallbacks (uniform weights, or a zero row) cannot be reached: every
        # coarse pixel has a fine child within rho of its centre.
        raise RuntimeError(
            "decimation_healpix: a coarse pixel has no fine pixel within the radius; this is a "
            "bug of ESTEL, please report it with cell_ids and depth."
        )
    w = w / row_sum[rows]
    return DecimationHealpix(
        index=cols.astype(np.int32).reshape(-1, 1, 1),
        weight=w.reshape(-1, 1, 1),
        segment_ids=rows.astype(np.int32),
        cell_ids_out=coarse,
    )
