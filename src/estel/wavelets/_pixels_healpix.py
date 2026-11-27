"""The HEALPix pixel functions the geometry tables need: centres, orderings and interpolation.

Written from published sources only, without reading the code of HEALPix C++, of healpy or of
any port of them:

* the geometry of the grid, Górski et al. 2005, ApJ 622, 759 (arXiv:astro-ph/0409513), §4:
  the rings of the north polar cap, equations (2) to (5), with z = 1 - i²/(3 nside²) and
  φ = π/(2i) (j - 1/2); those of the equatorial zone, equations (6) to (9), with
  z = 4/3 - 2i/(3 nside) and φ = π/(2 nside) (j - s/2), s = (i - nside + 1) mod 2; the south by
  the mirror symmetry about the equator; the RING ordering of §5.2, ring after ring from north
  to south. :func:`xyf2ang` and :func:`ring2nest` follow these equations; the placement of the
  12 base faces and the coordinates within a face are those of the paper's figures (face
  centres at φ = π/4 + fπ/2 in the polar rows, fπ/2 in the equatorial row; within a face, x
  runs to the north-east and y to the north-west from the southern corner);
* the NESTED ordering of §5.2, the interleaved bits of x and y within a face:
  :func:`nest2xyf` and :func:`xyf2nest`, with the usual bit-spreading masks;
* the bilinear interpolation on the rings, step 3 of M-CONV-HEALPIX in ESTEL's ``docs/maths.md``:
  :func:`interpolation_ring`, exactly as written there.

Every function is vectorised NumPy: integers int64, floats float64, ``depth`` = log2(nside)
from 0 to 13.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["interpolation_ring", "nest2xyf", "ring2nest", "xyf2ang", "xyf2nest"]

_SQRT6 = math.sqrt(6.0)


# ---------------------------------------------------------------------------------------------
# Bits of the NESTED index
# ---------------------------------------------------------------------------------------------


def _spread_bits(v: NDArray[np.int64]) -> NDArray[np.int64]:
    """Bit k of ``v`` (k < 16) moved to bit 2k."""
    v = v & 0x0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    return (v | (v << 1)) & 0x55555555


def _compact_bits(v: NDArray[np.int64]) -> NDArray[np.int64]:
    """Bit 2k of ``v`` (k < 16) moved to bit k; the odd bits dropped."""
    v = v & 0x55555555
    v = (v | (v >> 1)) & 0x33333333
    v = (v | (v >> 2)) & 0x0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF
    return (v | (v >> 8)) & 0x0000FFFF


def nest2xyf(
    pix: ArrayLike, depth: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """The base face (0 to 11) and the indices x and y (0 to nside - 1) of NESTED pixels.

    Within a face the NESTED index interleaves the bits of x (even bits) and y (odd bits);
    ``pix = face · nside² + index``. Returns ``(ix, iy, face)``, each of the shape of ``pix``.
    """
    p = np.asarray(pix, dtype=np.int64)
    face = p >> (2 * depth)
    index = p & ((1 << (2 * depth)) - 1)
    return _compact_bits(index), _compact_bits(index >> 1), face


def xyf2nest(ix: ArrayLike, iy: ArrayLike, face: ArrayLike, depth: int) -> NDArray[np.int64]:
    """The NESTED index of the pixel (x, y) of a base face: the inverse of :func:`nest2xyf`."""
    x = np.asarray(ix, dtype=np.int64)
    y = np.asarray(iy, dtype=np.int64)
    f = np.asarray(face, dtype=np.int64)
    return (f << (2 * depth)) | _spread_bits(x) | (_spread_bits(y) << 1)


# ---------------------------------------------------------------------------------------------
# Rings
# ---------------------------------------------------------------------------------------------


def _ring_colatitude(ring: NDArray[np.int64], nside: int) -> NDArray[np.float64]:
    """θ_r of rings r = 1 … 4 nside - 1 (Górski et al. 2005, equations 4 and 8, and mirror).

    In the caps, 1 - cos θ = r²/(3 nside²) = 2 sin²(θ/2), so θ = 2 arcsin(r / (√6 nside)),
    accurate near the poles; in the belt, θ = arccos((4 nside - 2r) / (3 nside)).
    """
    r = ring.astype(np.float64)
    north = 2.0 * np.arcsin(np.clip(r / (_SQRT6 * nside), 0.0, 1.0))
    south = math.pi - 2.0 * np.arcsin(np.clip((4 * nside - r) / (_SQRT6 * nside), 0.0, 1.0))
    belt = np.arccos(np.clip((4 * nside - 2.0 * r) / (3.0 * nside), -1.0, 1.0))
    return np.where(ring < nside, north, np.where(ring > 3 * nside, south, belt))


def _ring_info(
    ring: NDArray[np.int64], nside: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """For rings r = 1 … 4 nside - 1: the RING index of their first pixel, their number of
    pixels N_r, and the longitude φ_r of their first pixel (M-CONV-HEALPIX, step 3)."""
    n_pix = 12 * nside * nside
    south_ring = 4 * nside - ring
    count = 4 * np.minimum(np.minimum(ring, south_ring), nside)
    start = np.where(
        ring < nside,
        2 * ring * (ring - 1),
        np.where(
            ring > 3 * nside,
            n_pix - 2 * south_ring * (south_ring + 1),
            2 * nside * (nside - 1) + (ring - nside) * 4 * nside,
        ),
    )
    shifted = (ring < nside) | (ring > 3 * nside) | ((ring - nside) % 2 == 0)
    phi0 = np.where(shifted, math.pi / count, 0.0)
    return start, count, phi0


def _cap_ring(q: NDArray[np.int64]) -> NDArray[np.int64]:
    """The ring i >= 1 of the cap with 2 i (i - 1) <= q < 2 i (i + 1) (equation 2, exact)."""
    i = np.floor((1.0 + np.sqrt(1.0 + 2.0 * q.astype(np.float64))) / 2.0).astype(np.int64)
    i = np.where(2 * i * (i - 1) > q, i - 1, i)
    return np.where(2 * i * (i + 1) <= q, i + 1, i)


def xyf2ang(
    ix: ArrayLike, iy: ArrayLike, face: ArrayLike, depth: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The colatitude θ in [0, π] and longitude φ in [0, 2π) (radians) of pixel centres.

    Pixel (x, y) of face f lies on ring i = j_f nside - x - y - 1, with j_f = 2, 3, 4 for the
    northern, equatorial and southern faces, at φ = (p_f + (x - y) / m) π/4, with p_f the
    longitude of the face centre in units of π/4 and m = min(i, nside, 4 nside - i) the number
    of pixels of the ring in one face (Górski et al. 2005, equations 4, 5, 8 and 9).
    """
    nside = 1 << depth
    x = np.asarray(ix, dtype=np.int64)
    y = np.asarray(iy, dtype=np.int64)
    f = np.asarray(face, dtype=np.int64)
    row = f // 4
    ring = (row + 2) * nside - x - y - 1
    centre = 2 * (f % 4) + (row != 1)
    m = np.minimum(np.minimum(ring, 4 * nside - ring), nside)
    numerator = np.mod(centre * m + x - y, 8 * m)
    phi = numerator * (math.pi / 4.0) / m
    return _ring_colatitude(ring, nside), phi.astype(np.float64)


def ring2nest(pix: ArrayLike, depth: int) -> NDArray[np.int64]:
    """The NESTED index of RING pixels.

    The ring i and the position j in the ring follow Górski et al. 2005, equations 2, 3, 6 and
    7, and their mirror in the south. In a cap, the face is the quarter of the ring; in the belt,
    the diagonal coordinates X = (u - i + 3 nside - 1)/2 and Y = (-u - i + 3 nside - 1)/2, with
    u = φ / (π / (4 nside)), tile the plane in squares of nside x nside pixels, one per face.
    """
    nside = 1 << depth
    n_pix = 12 * nside * nside
    p = np.asarray(pix, dtype=np.int64)
    n_cap = 2 * nside * (nside - 1)

    # North cap: ring i < nside, 4 i pixels, i of them in each of the faces 0 to 3.
    i_n = np.maximum(_cap_ring(np.minimum(p, max(n_cap - 1, 0))), 1)
    j_n = p - 2 * i_n * (i_n - 1)
    m_n = j_n % i_n
    face_n = j_n // i_n
    ix_n = nside - i_n + m_n
    iy_n = nside - 1 - m_n

    # South cap: ring 4 nside - i, mirror; faces 8 to 11, x increasing eastwards from 0.
    q = n_pix - 1 - p
    i_s = np.maximum(_cap_ring(np.clip(q, 0, max(n_cap - 1, 0))), 1)
    j_s = 4 * i_s - 1 - (q - 2 * i_s * (i_s - 1))
    face_s = 8 + j_s // i_s
    ix_s = j_s % i_s
    iy_s = i_s - 1 - ix_s

    # Belt: rings nside to 3 nside, 4 nside pixels each.
    belt = p - n_cap
    i_b = nside + belt // (4 * nside)
    j_b = belt % (4 * nside)
    u = 2 * j_b + ((i_b - nside) % 2 == 0)
    big_x = (u - i_b + 3 * nside - 1) // 2
    big_y = (-u - i_b + 3 * nside - 1) // 2
    fx = big_x // nside
    fy = big_y // nside
    diff = fx - fy
    total = fx + fy
    face_b = np.where(
        total == 1, (diff - 1) // 2, np.where(total == -1, 8 + (diff - 1) // 2, 4 + (diff // 2) % 4)
    )
    ix_b = big_x - fx * nside
    iy_b = big_y - fy * nside

    north = p < n_cap
    south = p >= n_pix - n_cap
    ix = np.where(north, ix_n, np.where(south, ix_s, ix_b))
    iy = np.where(north, iy_n, np.where(south, iy_s, iy_b))
    face = np.where(north, face_n, np.where(south, face_s, face_b))
    return xyf2nest(ix, iy, face, depth)


# ---------------------------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------------------------


def _ring_pair(
    ring: NDArray[np.int64], phi: NDArray[np.float64], nside: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """The two RING pixels of ring r whose longitudes enclose φ, and the fraction t - ⌊t⌋, with
    t = (φ - φ_r) N_r / 2π (M-CONV-HEALPIX, step 3, on a ring)."""
    start, count, phi0 = _ring_info(ring, nside)
    t = (phi - phi0) * count / (2.0 * math.pi)
    k = np.floor(t)
    frac = t - k
    k0 = np.mod(k.astype(np.int64), count)
    k1 = np.mod(k0 + 1, count)
    return start + k0, start + k1, frac


def interpolation_ring(
    theta: ArrayLike, phi: ArrayLike, depth: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The 4 RING pixels and bilinear weights of points (θ, φ), φ in [0, 2π) (M-CONV-HEALPIX, step
    3).

    Between rings r and r + 1 (θ_r <= θ < θ_{r + 1}): the two pixels of each ring that enclose
    φ, linear in longitude on each ring and linear in colatitude between them. Above ring 1 or
    below ring 4 nside - 1: between the pole, the mean of the 4 pixels of the last ring, and
    that ring. Returns two ``(M, 4)`` arrays, RING indices (int64) and weights (float64) that
    sum to 1; ``theta`` and ``phi`` are 1D of length M.
    """
    nside = 1 << depth
    n_rings = 4 * nside - 1
    th = np.asarray(theta, dtype=np.float64)
    ph = np.asarray(phi, dtype=np.float64)

    rings = np.arange(1, n_rings + 1, dtype=np.int64)
    colatitudes = _ring_colatitude(rings, nside)
    above = np.searchsorted(colatitudes, th, side="right").astype(np.int64)  # θ_r <= θ, 0 … 4n-1
    north = above == 0
    south = above == n_rings
    r_top = np.clip(above, 1, n_rings - 1)
    r_bottom = r_top + 1
    theta_top = colatitudes[r_top - 1]
    theta_bottom = colatitudes[r_bottom - 1]

    a0, a1, fa = _ring_pair(r_top, ph, nside)
    b0, b1, fb = _ring_pair(r_bottom, ph, nside)
    w = np.clip((th - theta_top) / (theta_bottom - theta_top), 0.0, 1.0)
    between = (
        np.stack([a0, a1, b0, b1], axis=-1),
        np.stack([(1 - w) * (1 - fa), (1 - w) * fa, w * (1 - fb), w * fb], axis=-1),
    )

    # Near a pole: ring 1 (north) or ring 4 nside - 1 (south), its 4 pixels.
    cap_ring = np.where(south, n_rings, 1)
    c0, c1, fc = _ring_pair(cap_ring, ph, nside)
    first = np.where(south, 12 * nside * nside - 4, 0)
    c2 = first + np.mod(c1 - first + 1, 4)
    c3 = first + np.mod(c1 - first + 2, 4)
    theta_cap = colatitudes[cap_ring - 1]
    w_pole = np.where(
        south,
        (math.pi - th) / (math.pi - theta_cap),
        th / theta_cap,
    )
    w_pole = np.clip(w_pole, 0.0, 1.0)
    quarter = (1 - w_pole) / 4
    pole = (
        np.stack([c0, c1, c2, c3], axis=-1),
        np.stack([w_pole * (1 - fc) + quarter, w_pole * fc + quarter, quarter, quarter], axis=-1),
    )

    at_pole = (north | south)[:, None]
    pix = np.where(at_pole, pole[0], between[0]).astype(np.int64)
    weight = np.where(at_pole, pole[1], between[1]).astype(np.float64)
    return pix, weight
