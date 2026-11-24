"""The HEALPix geometry tables: stencils and decimation (Part II, §12.5; task T5.1, issue #26).

LICENCE WARNING (F23, issue #173). The bilinear interpolation of this module
(``_get_interpol_ring`` and its helpers ``_ring_above``, ``_ring_info``, ``_ring_pair``)
transcribes ``get_interpol`` of HEALPix C++ (``healpix_base.cc``, GPL-2), through STL's
``TorchGeometry``. It must not be published as is. It is settled before the release on PyPI
(T8.2, #32): by the written confirmation of the licences, or by a rewrite from the published
algorithm of ``docs/maths.md`` (M-CONV-HPX, step 3, written by T5.4). Part IV, §21, F23; Part I,
§1.5.

Ported from ``TorchGeometry`` of STL `HealpixUpdate` ``90eb1a9`` (``STL_main/TorchGeometry.py``),
which reproduces in vectorised code the geometry of healpix-analyse ``aac5e26`` (``HealPixConv``,
``HealPixDown``) on the sphere. The five blocks decided on 24 November 2026 (Part IV, §22, "What
is ported from `TorchGeometry`"):

1. the NESTED pixel centres: :func:`nest2xyf`, :func:`xyf2ang` (HEALPix ``pix2loc``);
2. the bilinear interpolation of the 4 nearest pixels: :func:`ring2nest`,
   :func:`_get_interpol_ring`, :func:`interp_weights_nested`, ported as is (F23, above);
3. the rotations of the stencil, the gauge ``"cosmo"`` only: :func:`rotation_matrices`;
4. the stencil and its binding on a partial sky: :func:`local_kernel_grid`,
   :func:`_bind_support`;
5. the decimation: :func:`centres_lonlat`, :func:`_haversine`, and :func:`healpix_decimation`,
   the port of ``smooth_decimation_csr``, whose search of the candidate fine pixels is ESTEL's
   own (:func:`_cone_pairs`), exact and in NumPy, where STL uses healpix-geo.

Not ported: the pixel batches (T5.7), the cast of the kernel through float32 (maths.md OP-46),
``TorchConv`` and everything torch. Only the sphere: ESTEL has no ellipsoid (OP-45).

The tables are built once, in NumPy, when the plan is built (tag N of Part II, §12); they are
static, never traced and never differentiated. Every weight is float64, cast by the plan to the
dtype of the data (OP-46); every index is int32, which holds the 12 · 4^13 pixels of ``nside``
8192, and a deeper ``depth`` raises an error. The code sets no other limit on ``nside``;
``nside`` 256 is the documented target of 1.0.0 on one GPU (tables of about 3.4 GB in float32).

Table layout (for T5.5, which applies the tables, and T6.3, which checks the signature of
``gather_weighted`` against them). Both fit the L0 primitives of Part II, §12.3, applied along
the pixel axis of a map ``x`` of shape ``(..., Npix)`` whose pixels are ``cell_ids``, ascending:

* **Convolution** (M-CONV-HPX), :func:`healpix_stencils`: ``index`` and ``weight`` of shape
  ``(L · Npix, K², 4)``. Row ``r = g · Npix + k`` is gauge ``g`` at pixel ``cell_ids[k]``;
  column ``p = (v + c) K + (u + c)`` the stencil point (u, v); the last axis the 4 neighbours of
  the bilinear interpolation, ``index`` their positions in ``cell_ids`` and ``weight`` their
  renormalised weights (0 for a neighbour outside ``cell_ids``, whose index is then 0). Then::

      z = gather_weighted(x, -1, index, weight)       # (..., L · Npix, K²): the interpolated
      y = z @ W                                       # stencil values; W the kernel (K², )
      y = reshape(y, (..., L, Npix))                  # of healpix_kernel_weights

  the product with the complex kernel being two real products, with its real and imaginary
  parts (Part I, §3.1, rule 6).
* **Decimation** (M-DOWN-HPX), :func:`healpix_decimation`: a CSR matrix of ``nnz`` entries,
  sorted by coarse pixel then by fine pixel. ``index`` and ``weight`` of shape ``(nnz, 1, 1)``,
  the positions in ``cell_ids`` of the fine pixels and their weights; ``segment_ids`` of shape
  ``(nnz, )``, the position of the coarse pixel in ``cell_ids_out``. Then::

      z = gather_weighted(x, -1, index, weight)       # (..., nnz, 1)
      y = segment_sum(z[..., 0], -1, segment_ids, len(cell_ids_out))   # (..., Npix_coarse)

``tests/geometry/test_healpix_geometry.py`` checks each geometry function against healpy (in the
tests only, Part I, §1.5) and the rotations against M-CONV-HPX, step 2;
``tests/parity/test_healpix_tables.py`` checks the tables, as operators (OP-92), against STL's
fixtures (T5.3).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from estel.wavelets._banks import KERNEL_SIZE, _check_n_orientations
from estel.wavelets._healpix_bank import _check_kernel_size

__all__ = [
    "MAX_DEPTH",
    "HealpixDecimation",
    "HealpixStencils",
    "centres_lonlat",
    "healpix_decimation",
    "healpix_stencils",
    "interp_weights_nested",
    "local_kernel_grid",
    "nest2xyf",
    "ring2nest",
    "rotation_matrices",
    "xyf2ang",
    "xyf2nest",
]

MAX_DEPTH = 13
"""The deepest level the tables index: 12 · 4^13 pixels (``nside`` 8192) fit int32 indices."""

_JRLL = np.array((2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4), dtype=np.int64)
_JPLL = np.array((1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7), dtype=np.int64)

_CONE_CHUNK = 1 << 14
"""Coarse pixels searched at once by :func:`_cone_pairs`: bounds its temporary arrays."""


# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------


def _check_depth(depth: object, minimum: int = 0) -> int:
    """``depth`` as an int in [minimum, MAX_DEPTH]."""
    if isinstance(depth, bool) or not isinstance(depth, Integral):
        raise TypeError(
            f"depth={depth!r}: the depth is an integer, log2(nside), such as depth=4 for nside 16."
        )
    value = int(depth)
    if value < minimum:
        raise ValueError(
            f"depth={value}: the depth must be at least {minimum} here"
            + (" (a decimation goes from depth to depth - 1)." if minimum else ".")
        )
    if value > MAX_DEPTH:
        raise ValueError(
            f"depth={value} (nside {1 << value}): the HEALPix tables index pixels in int32, "
            f"which holds at most depth {MAX_DEPTH} (nside {1 << MAX_DEPTH}, 12 · 4^{MAX_DEPTH} "
            "pixels). Use a coarser map."
        )
    return value


def _check_cell_ids(cell_ids: ArrayLike | None, depth: int) -> NDArray[np.int64]:
    """The NESTED pixels as int64: all of them for ``None``, else checked (maths.md OP-90)."""
    n_pix = 12 * 4**depth
    if cell_ids is None:
        return np.arange(n_pix, dtype=np.int64)
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(
            f"cell_ids has shape {ids.shape}: give a non-empty 1D array of NESTED pixel "
            "indices, or None for the full sky."
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(
            f"cell_ids has dtype {ids.dtype}: give integer NESTED pixel indices, for example "
            "cell_ids.astype(np.int64)."
        )
    ids = ids.astype(np.int64)
    if ids.min() < 0 or ids.max() >= n_pix:
        raise ValueError(
            f"cell_ids holds pixels outside [0, {n_pix}), the {n_pix} pixels of depth {depth} "
            f"(nside {1 << depth}): check depth, or the NESTED indices."
        )
    if np.any(np.diff(ids) <= 0):
        raise ValueError(
            "cell_ids must be strictly increasing (maths.md OP-90): sort the pixels and the "
            "data together, order = np.argsort(cell_ids); cell_ids, x = cell_ids[order], "
            "x[..., order], and remove repeated pixels."
        )
    return ids


# ---------------------------------------------------------------------------------------------
# 1. NESTED <-> (x, y, face) <-> angles, RING -> NESTED
# ---------------------------------------------------------------------------------------------


def _compress_bits(v: NDArray[np.int64]) -> NDArray[np.int64]:
    """Keep the even bits of ``v``, packed: the inverse of :func:`_spread_bits`."""
    v = v & 0x5555555555555555
    v = (v | (v >> 1)) & 0x3333333333333333
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFF
    v = (v | (v >> 16)) & 0x00000000FFFFFFFF
    return v


def _spread_bits(v: NDArray[np.int64]) -> NDArray[np.int64]:
    """Interleave zeros between the bits of ``v`` (v < 2³²)."""
    v = v & 0x00000000FFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def nest2xyf(
    pix: NDArray[np.int64], depth: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """The (x, y, face) coordinates of NESTED pixels (HEALPix ``nest2xyf``)."""
    pix = np.asarray(pix, dtype=np.int64)
    face = pix >> (2 * depth)
    local = pix & ((1 << (2 * depth)) - 1)
    return _compress_bits(local), _compress_bits(local >> 1), face


def xyf2nest(
    ix: NDArray[np.int64], iy: NDArray[np.int64], face: NDArray[np.int64], depth: int
) -> NDArray[np.int64]:
    """The NESTED pixels of (x, y, face) coordinates (HEALPix ``xyf2nest``)."""
    return (face << (2 * depth)) + _spread_bits(ix) + (_spread_bits(iy) << 1)


def xyf2ang(
    ix: NDArray[np.int64], iy: NDArray[np.int64], face: NDArray[np.int64], nside: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The colatitude and longitude, in radians, of pixel centres (HEALPix ``pix2loc``)."""
    jrll, jpll = _JRLL[face], _JPLL[face]
    jr = jrll * nside - ix - iy - 1
    fact2 = 4.0 / (12.0 * nside * nside)
    fact1 = 2.0 / (3.0 * nside)
    north = jr < nside
    south = jr > 3 * nside
    nr = np.where(north, jr, np.where(south, 4 * nside - jr, nside))
    kshift = np.where(north | south, 0, (jr - nside) & 1)
    nrf = nr.astype(np.float64)
    tmp = nrf * nrf * fact2
    z_cap = 1.0 - tmp
    sth_cap = np.sqrt(tmp * (2.0 - tmp))
    z_eq = (2.0 * nside - jr.astype(np.float64)) * fact1
    theta_cap = np.arctan2(sth_cap, z_cap)
    theta = np.where(
        north, theta_cap, np.where(south, math.pi - theta_cap, np.arccos(np.clip(z_eq, -1, 1)))
    )
    jp = (jpll * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * nr, jp - 4 * nr, jp)
    jp = np.where(jp < 1, jp + 4 * nr, jp)
    phi = (jp.astype(np.float64) - (kshift.astype(np.float64) + 1.0) * 0.5) * (0.5 * math.pi / nrf)
    return theta, phi


def _isqrt(v: NDArray[np.int64]) -> NDArray[np.int64]:
    """The integer square root of non-negative int64 values."""
    r = np.sqrt(np.maximum(v, 0).astype(np.float64)).astype(np.int64)
    r = np.where(r * r > v, r - 1, r)
    return np.where((r + 1) * (r + 1) <= v, r + 1, r)


def ring2nest(pix: NDArray[np.int64], depth: int) -> NDArray[np.int64]:
    """RING -> NESTED pixel numbers (HEALPix ``ring2xyf``, then ``xyf2nest``)."""
    pix = np.asarray(pix, dtype=np.int64)
    nside = 1 << depth
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    nl2 = 2 * nside
    north = pix < ncap
    south = pix >= npix - ncap
    equator = ~(north | south)

    # North polar cap.
    ir_n = (1 + _isqrt(1 + 2 * pix)) >> 1
    ip_n = pix + 1 - 2 * ir_n * (ir_n - 1)
    face_n = (ip_n - 1) // np.maximum(ir_n, 1)

    # Equatorial belt.
    ip_e = pix - ncap
    tmp = ip_e >> (depth + 2)
    ir_e = tmp + nside
    ip_e2 = ip_e - tmp * 4 * nside + 1
    ks_e = (ir_e + nside) & 1
    ire = ir_e - nside + 1
    irm = nl2 + 2 - ire
    ifm = (ip_e2 - ire // 2 + nside - 1) >> depth
    ifp = (ip_e2 - irm // 2 + nside - 1) >> depth
    face_e = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))

    # South polar cap.
    ip_s = npix - pix
    ir_s = (1 + _isqrt(2 * ip_s - 1)) >> 1
    ip_s2 = 4 * ir_s + 1 - (ip_s - 2 * ir_s * (ir_s - 1))
    face_s = 8 + (ip_s2 - 1) // np.maximum(ir_s, 1)

    iring = np.where(north, ir_n, np.where(equator, ir_e, 2 * nl2 - ir_s))
    iphi = np.where(north, ip_n, np.where(equator, ip_e2, ip_s2))
    kshift = np.where(equator, ks_e, 0)
    nr = np.where(north, ir_n, np.where(equator, nside, ir_s))
    face = np.clip(np.where(north, face_n, np.where(equator, face_e, face_s)), 0, 11)

    irt = iring - _JRLL[face] * nside + 1
    ipt = 2 * iphi - _JPLL[face] * nr - kshift - 1
    ipt = np.where(ipt >= nl2, ipt - 8 * nside, ipt)
    ix = (ipt - irt) >> 1
    iy = (-ipt - irt) >> 1
    return xyf2nest(ix, iy, face, depth)


# ---------------------------------------------------------------------------------------------
# 2. Bilinear interpolation (RING scheme), port of T_Healpix_Base::get_interpol.
#    LICENCE WARNING (F23): the four functions below transcribe GPL-2 code; see the module
#    docstring.
# ---------------------------------------------------------------------------------------------


def _ring_above(z: NDArray[np.float64], nside: int) -> NDArray[np.int64]:
    """The ring just north of ``z = cos θ`` (0 above the first ring)."""
    az = np.abs(z)
    out_eq = np.trunc(nside * (2.0 - 1.5 * z)).astype(np.int64)
    iring = np.trunc(nside * np.sqrt(3.0 * (1.0 - az))).astype(np.int64)
    out_pol = np.where(z > 0, iring, 4 * nside - iring - 1)
    return np.where(az <= 2.0 / 3.0, out_eq, out_pol)


def _ring_info(
    ring: NDArray[np.int64], nside: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """First pixel, number of pixels, colatitude and shift of rings 1 … 4 nside - 1."""
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    fact1 = 2.0 / (3.0 * nside)
    fact2 = 1.0 / (3.0 * nside * nside)
    northring = np.where(ring > 2 * nside, 4 * nside - ring, ring)
    cap = northring < nside
    nrf = northring.astype(np.float64)
    tmp = nrf * nrf * fact2
    theta_cap = np.arctan2(np.sqrt(tmp * (2.0 - tmp)), 1.0 - tmp)
    theta_eq = np.arccos(np.clip((2 * nside - northring).astype(np.float64) * fact1, -1, 1))
    theta = np.where(cap, theta_cap, theta_eq)
    ringpix = np.where(cap, 4 * northring, 4 * nside)
    shifted = np.where(cap, True, ((northring - nside) & 1) == 0)
    startpix = np.where(
        cap, 2 * northring * (northring - 1), ncap + (northring - nside) * 4 * nside
    )
    south = northring != ring
    theta = np.where(south, math.pi - theta, theta)
    startpix = np.where(south, npix - startpix - ringpix, startpix)
    return startpix, ringpix, theta, shifted


def _ring_pair(
    phi: NDArray[np.float64], ring: NDArray[np.int64], nside: int
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """The two pixels of ``ring`` around longitude ``phi``, their weights and the ring's θ."""
    sp, nr, theta_ring, shifted = _ring_info(ring, nside)
    dphi = (2.0 * math.pi) / nr.astype(np.float64)
    half = np.where(shifted, 0.5, 0.0)
    tmp = phi / dphi - half
    i1 = np.where(tmp < 0, np.trunc(tmp).astype(np.int64) - 1, np.trunc(tmp).astype(np.int64))
    w1 = (phi - (i1.astype(np.float64) + half) * dphi) / dphi
    i2 = i1 + 1
    i1 = np.where(i1 < 0, i1 + nr, i1)
    i2 = np.where(i2 >= nr, i2 - nr, i2)
    return sp + i1, sp + i2, 1.0 - w1, w1, theta_ring


# LICENCE WARNING (F23, issue #173): transcribes GPL-2 code; see the module docstring.
def _get_interpol_ring(
    theta: NDArray[np.float64], phi: NDArray[np.float64], nside: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The 4 RING pixels and bilinear weights of directions (θ, φ), shapes ``(M, 4)``."""
    npix = 12 * nside * nside
    z = np.cos(theta)
    ir1 = _ring_above(z, nside)
    ir2 = ir1 + 1
    m1 = ir1 > 0
    p0, p1, w0, w1, theta1 = _ring_pair(phi, np.maximum(ir1, 1), nside)
    m2 = ir2 < 4 * nside
    p2, p3, w2, w3, theta2 = _ring_pair(phi, np.minimum(ir2, 4 * nside - 1), nside)
    pix = [np.where(m1, p0, 0), np.where(m1, p1, 0), np.where(m2, p2, 0), np.where(m2, p3, 0)]
    wgt = [
        np.where(m1, w0, 0.0),
        np.where(m1, w1, 0.0),
        np.where(m2, w2, 0.0),
        np.where(m2, w3, 0.0),
    ]

    north = ir1 == 0
    south = (ir2 == 4 * nside) & ~north
    # The three branches are computed everywhere and selected: silence the branches not taken
    # (0 / 0 where the two rings coincide near a pole).
    with np.errstate(divide="ignore", invalid="ignore"):
        wn = theta / theta2
        ws = (theta - theta1) / (math.pi - theta1)
        wm = (theta - theta1) / (theta2 - theta1)
        facn = (1.0 - wn) * 0.25
        facs = ws * 0.25
        weights = [
            np.where(north, facn, np.where(south, wgt[0] * (1.0 - ws) + facs, wgt[0] * (1.0 - wm))),
            np.where(north, facn, np.where(south, wgt[1] * (1.0 - ws) + facs, wgt[1] * (1.0 - wm))),
            np.where(north, wgt[2] * wn + facn, np.where(south, facs, wgt[2] * wm)),
            np.where(north, wgt[3] * wn + facn, np.where(south, facs, wgt[3] * wm)),
        ]
    pixels = [
        np.where(north, (pix[2] + 2) & 3, pix[0]),
        np.where(north, (pix[3] + 2) & 3, pix[1]),
        np.where(south, ((pix[0] + 2) & 3) + npix - 4, pix[2]),
        np.where(south, ((pix[1] + 2) & 3) + npix - 4, pix[3]),
    ]
    return np.stack(pixels, axis=-1), np.stack(weights, axis=-1)


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
    pix_ring, weights = _get_interpol_ring(theta2, phi2, 1 << depth)
    return ring2nest(pix_ring, depth), weights


# ---------------------------------------------------------------------------------------------
# 3. Rotations, 4. the stencil and its binding
# ---------------------------------------------------------------------------------------------


def rotation_matrices(
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    L: int,  # noqa: N803
) -> NDArray[np.float64]:
    """The rotations of the gauge ``"cosmo"`` (maths.md M-CONV-HPX, step 2), ``(N, L, 3, 3)``.

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


def local_kernel_grid(kernel_size: int, nside: int) -> NDArray[np.float64]:
    """The K x K stencil around the North Pole, unit vectors ``(K², 3)`` (M-CONV-HPX, step 1).

    Point (u, v) at colatitude √(u² + v²) alpha and azimuth atan2(v, u), with
    alpha = √(4π / (12 nside²)); entry p = (v + c) K + (u + c).
    """
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
    nside = 1 << depth
    theta, phi = xyf2ang(*nest2xyf(ids, depth), nside)
    # healpix-analyse reads the pixel centres from healpix-geo, in degrees.
    theta = np.deg2rad(90.0 - np.rad2deg(math.pi / 2 - theta))
    phi = np.deg2rad(np.rad2deg(phi))
    rotations = rotation_matrices(theta, phi, L)  # (N, L, 3, 3)
    stencil = local_kernel_grid(kernel_size, nside)  # (P, 3)
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
    """Bind the neighbours ``(L, N, P, 4)`` on the pixels ``ids`` (ascending), as
    healpix-analyse's ``_bind_support_batched`` (maths.md M-CONV-HPX, partial sky; OP-50).

    A neighbour outside ``ids`` is dropped, and the weights of the others renormalised to sum
    to 1. A stencil point with no neighbour in ``ids`` takes the neighbours of the stencil
    centre of its pixel and gauge. A point whose present neighbours all have weight 0 gives
    weight 1 to its first neighbour if present. Returns the positions in ``ids`` (0 where
    absent), the weights, and the stencil points that had no neighbour in ``ids``.
    """
    n_ids = ids.size

    def lookup(idx: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        pos = np.minimum(np.searchsorted(ids, idx), n_ids - 1)
        found: NDArray[np.bool_] = np.asarray(ids[pos] == idx)
        return pos, found

    _, present = lookup(index)
    empty: NDArray[np.bool_] = ~np.asarray(present.any(axis=-1))  # (L, N, P)
    centre = (kernel_size // 2) * (kernel_size + 1)
    index = np.where(empty[..., None], index[:, :, centre : centre + 1], index)
    weight = np.where(empty[..., None], weight[:, :, centre : centre + 1], weight)
    pos, present = lookup(index)
    w = weight * present
    first = np.where(w.sum(axis=-1) == 0, present[..., 0].astype(w.dtype), w[..., 0])
    w = np.concatenate([first[..., None], w[..., 1:]], axis=-1)
    w_norm = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-12)
    positions: NDArray[np.int64] = np.where(present, pos, 0)
    return positions, w_norm, empty


@dataclass(frozen=True, eq=False)
class HealpixStencils:
    """The convolution tables of one level, for ``gather_weighted`` (module docstring).

    ``index`` int32 and ``weight`` float64, shape ``(L · Npix, K², 4)``, row ``g · Npix + k``.
    """

    index: NDArray[np.int32]
    weight: NDArray[np.float64]
    n_gauges: int
    kernel_size: int


def healpix_stencils(
    cell_ids: ArrayLike | None,
    depth: int,
    L: int = 4,  # noqa: N803
    kernel_size: int = KERNEL_SIZE,
) -> HealpixStencils:
    """The interpolation tables of the rotated stencils at one level (maths.md M-CONV-HPX).

    ``cell_ids``: the NESTED pixels of the map, strictly increasing, or ``None`` for the full
    sky; ``depth``: log2(nside) of the level, 0 to :data:`MAX_DEPTH`; ``L``: the number of
    gauges of ``"cosmo"``; ``kernel_size``: K, odd. The kernel is not in the tables: it is the
    same at every level and for every gauge (M-WAV-HPX, :func:`healpix_kernel_weights`). On a
    partial sky, the rule of M-CONV-HPX (OP-50). Layout: the module docstring.
    """
    depth = _check_depth(depth)
    ids = _check_cell_ids(cell_ids, depth)
    n_gauges = _check_n_orientations(L)
    size = _check_kernel_size(kernel_size)
    index, weight = _stencil_interpolation(ids, depth, n_gauges, size)
    pos, w_norm, _ = _bind_support(index, weight, ids, size)
    shape = (n_gauges * ids.size, size * size, 4)
    return HealpixStencils(
        index=pos.reshape(shape).astype(np.int32),
        weight=w_norm.reshape(shape),
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
    ids: NDArray[np.int64], nside: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The longitude and latitude, in radians, of NESTED pixel centres, rounded through
    degrees as healpix-geo's output that healpix-analyse reads."""
    depth = int(nside).bit_length() - 1
    theta, phi = xyf2ang(*nest2xyf(ids, depth), nside)
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
class HealpixDecimation:
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


def healpix_decimation(cell_ids: ArrayLike | None, depth: int) -> HealpixDecimation:
    """The tables of one decimation step, from ``depth`` to ``depth - 1`` (maths.md M-DOWN-HPX).

    ``cell_ids``: the NESTED fine pixels, strictly increasing, or ``None`` for the full sky;
    ``depth``: log2(nside) of the fine level, 1 to :data:`MAX_DEPTH`. The coarse pixels are the
    parents of the fine ones, ascending. The weight of fine pixel n for coarse pixel n₀ is
    exp(-gamma² / 2 sigma²), gamma the angular distance of their centres, over every available
    fine pixel with gamma ≤ rho, rho = 3 fine pixel sizes, sigma = rho / 2, normalised to sum to 1
    for each coarse pixel. On a partial sky, only the fine pixels of ``cell_ids`` enter (OP-50).
    Layout: the module docstring.
    """
    depth = _check_depth(depth, minimum=1)
    fine = _check_cell_ids(cell_ids, depth)
    nside = 1 << depth
    coarse = np.unique(fine // 4)
    lon_c, lat_c = centres_lonlat(coarse, nside // 2)
    lon_f, lat_f = centres_lonlat(fine, nside)
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
            "healpix_decimation: a coarse pixel has no fine pixel within the radius; this is a "
            "bug of ESTEL, please report it with cell_ids and depth."
        )
    w = w / row_sum[rows]
    return HealpixDecimation(
        index=cols.astype(np.int32).reshape(-1, 1, 1),
        weight=w.reshape(-1, 1, 1),
        segment_ids=rows.astype(np.int32),
        cell_ids_out=coarse,
    )
