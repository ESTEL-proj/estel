"""The oracle for HEALPix data: section 4 of ``docs/maths.md``.

A second implementation of the mathematics, written to be read against ``docs/maths.md`` rather
than to be fast (Part I, §4.2): float64, NumPy and the standard library only, nothing from the
rest of ``estel``. The statistics of HEALPix data are those of section 1 on this geometry
(§4, "Scattering loop"), so this module calls the planar oracle (``planar.py``) for the means,
the moments, the standardisation, the normalisation, the reductions and the flattening, with one
pixel axis (``ndim=1``) and no border.

**Its own geometry** (Part IV, §22, row of 24 November 2026, "The HEALPix geometry of the
oracle"; F23). The NESTED pixel centres, the ring structure, the bilinear interpolation, the
rotations of the gauge ``"cosmo"``, the stencil with its partial-sky rule and the decimation are
written here from ``docs/maths.md`` §4 and the published HEALPix geometry (Górski et al. 2005,
ApJ 622, 759): the 12 base faces, their nested quadtree, and the rings of constant latitude.
They are naive: every pixel centre is computed, and the decimation compares every pair of
pixels. healpy is a test-only cross-check (Part I, §1.5), never imported here.

**Conventions.**

- Arrays are ``(Nb, Nc, Npix)``: one pixel axis, last; every function acts on it whatever the
  leading axes are. Pixels are NESTED indices; a level is described by its ``nside`` and its
  ``cell_ids``, the NESTED indices of its pixels in ascending order (the full sky when every
  pixel is there).
- A mask is a boolean array of shape ``(Npix,)``, True where a pixel is invalid, or ``None``.
  The data hold NaN at their missing pixels (M-DATA-HPX); the scattering sets them to 0 before
  the first convolution, and every invalid pixel of a later level holds 0 too: the masks are
  weights, and an interpolation weight of rounding size, below the support threshold of
  M-MASK-HPX, reads a 0 rather than a NaN (OP-102).
- Angles are colatitude θ ∈ [0, π] and longitude φ ∈ [0, 2π), on the unit sphere: the oracle
  computes on the sphere only (OP-45).
- The statistics are STL's full grids, as in ``planar.py``.

Every public function lists the ``M-`` IDs it implements; ``tools/check_maths_ids.py`` checks
the mapping in both directions. The angular spectrum (LF24) and the syntheses (section 5) have
no oracle here.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import planar

Array = NDArray[Any]
Mask = NDArray[np.bool_] | None
Statistics = dict[str, Any]

KERNEL_SIZE = 5
"""K, the size of the stencil (M-CONV-HPX, M-DEF-HPX)."""
N_GAUGES = 4
"""L, the default number of gauges (M-DEF-HPX)."""
NAN_THRESHOLD = 0.33
"""τ, above which a decimated pixel is invalid (M-DOWN-HPX, M-DEF-HPX)."""
SUPPORT_THRESHOLD = 1e-12
"""Above it, a pixel's support touches an invalid pixel (M-MASK-HPX); below it, an
interpolation weight counts as absent on a partial sky (M-CONV-HPX, OP-110)."""
REWEIGHT_FLOOR = 1e-8
"""The floor of 1 - f in the reweighting maps (M-MASK-HPX)."""
DECIMATION_RADIUS = 3.0
"""rho, the radius of the decimation, in fine pixel sizes (M-DOWN-HPX)."""
DECIMATION_WIDTH = 0.5
"""sigma / rho, the width of the Gaussian of the decimation (M-DOWN-HPX)."""
GAUGES = ("cosmo",)
"""The gauges of the oracle: ``"cosmo"``, the default and the only one ESTEL transfers."""

# The base faces (Górski et al. 2005, §4): faces 0-3 around the North Pole, 4-7 on the equator,
# 8-11 around the South Pole. FACE_RING[f] * nside is the ring of the face's southern vertex, in
# ring units; FACE_LONGITUDE[f] * π/4 is the longitude of its centre.
FACE_RING = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
FACE_LONGITUDE = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def _check_nside(nside: int) -> int:
    n = int(nside)
    if n < 1 or n & (n - 1):
        raise ValueError(f"nside={nside!r}: pass a power of 2 (M-DATA-HPX).")
    return n


# ---------------------------------------------------------------------- pixel centres and rings


@functools.cache
def _nested_geometry(nside: int) -> tuple[Array, Array, Array]:
    """``(θ, φ, ring)`` of every NESTED pixel of ``nside``, read-only (``spec_pixel_centres``)."""
    pixels = np.arange(12 * nside * nside, dtype=np.int64)
    face, within = pixels // (nside * nside), pixels % (nside * nside)
    # The NESTED index within a face interleaves the bits of (x, y): x on the even bits.
    x = np.zeros_like(within)
    y = np.zeros_like(within)
    for bit in range(nside.bit_length() - 1):
        x |= ((within >> (2 * bit)) & 1) << bit
        y |= ((within >> (2 * bit + 1)) & 1) << bit
    # Ring i, from 1 at the North Pole to 4 nside - 1 at the South Pole: from the southern vertex
    # of the face, each step in x or in y moves the centre one ring north.
    ring = FACE_RING[face] * nside - x - y - 1
    north, south = ring < nside, ring > 3 * nside
    # Polar caps: ring i holds 4 i pixels (4 (4 nside - i) in the south), z = ±(1 - i²/(3 nside²)).
    # Equatorial belt: 4 nside pixels per ring, z linear in i.
    n_ring = np.where(north, ring, np.where(south, 4 * nside - ring, nside))
    z = np.where(
        north,
        1.0 - n_ring**2 / (3.0 * nside**2),
        np.where(south, n_ring**2 / (3.0 * nside**2) - 1.0, (2 * nside - ring) * 2.0 / (3 * nside)),
    )
    # The rings of the caps, and every other ring of the belt, are shifted by half a pixel.
    unshifted = np.where(north | south, 0, (ring - nside) % 2)
    # Position along the ring, from 1: the face's centre is at FACE_LONGITUDE π/4.
    position = (FACE_LONGITUDE[face] * n_ring + x - y + 1 + unshifted) // 2
    position = np.where(position > 4 * nside, position - 4 * nside, position)
    position = np.where(position < 1, position + 4 * nside, position)
    phi = (position - (unshifted + 1) * 0.5) * (math.pi / 2.0 / n_ring)
    theta = np.arccos(z)
    for a in (theta, phi, ring):
        a.flags.writeable = False
    return theta, phi, ring


def spec_pixel_centres(nside: int, cell_ids: Any = None) -> tuple[Array, Array]:
    """The colatitude θ and longitude φ of the centres of the NESTED pixels ``cell_ids`` (every
    pixel by default) of ``nside``. Implements M-DATA-HPX (the NESTED ordering) and the pixel
    centres of M-CONV-HPX, on the sphere (OP-45).

    The published HEALPix geometry (Górski et al. 2005): pixel p lies on the base face
    f = ⌊p / nside²⌋, at the coordinates (x, y) whose bits interleave to p mod nside² (x on the
    even bits). Its ring is i = r_f nside - x - y - 1, with r_f = 2, 3, 4 for the northern,
    equatorial and southern faces. In the polar caps (i < nside, and symmetrically
    i > 3 nside), z = cos θ = ±(1 - i'²/(3 nside²)) with i' the ring counted from the nearer
    pole, and the ring holds 4 i' pixels at φ = (π / 2i')(k + 1/2); in the equatorial belt,
    z = (2 nside - i) · 2/(3 nside), and the ring holds 4 nside pixels at φ = (π/2 nside)(k + s/2),
    with s = 1 on the rings i ≡ nside mod 2 and 0 on the others. The position k along the ring
    is ((c_f i' + x - y + 1 + s') / 2) - 1, modulo the ring, with c_f π/4 the longitude of the
    face's centre and s' = 1 - s in the belt, 0 in the caps.
    """
    theta, phi, _ = _nested_geometry(_check_nside(nside))
    if cell_ids is None:
        return np.array(theta), np.array(phi)
    ids = np.asarray(cell_ids, dtype=np.int64)
    return theta[ids], phi[ids]


@functools.cache
def _ring_table(nside: int) -> tuple[Array, Array, Array, Array]:
    """The rings of ``nside``, from north to south: their colatitude ``(R,)``, the longitude of
    their first pixel ``(R,)``, their number of pixels ``(R,)``, and their pixels in increasing
    longitude, ``(R, 4 nside)``, padded with -1."""
    theta, phi, ring = _nested_geometry(nside)
    n_rings = 4 * nside - 1
    ring_theta = np.zeros(n_rings)
    ring_phi0 = np.zeros(n_rings)
    counts = np.zeros(n_rings, dtype=np.int64)
    pixels = np.full((n_rings, 4 * nside), -1, dtype=np.int64)
    for r in range(n_rings):
        members = np.flatnonzero(ring == r + 1)
        members = members[np.argsort(phi[members], kind="stable")]
        ring_theta[r] = theta[members[0]]
        ring_phi0[r] = phi[members[0]]
        counts[r] = members.size
        pixels[r, : members.size] = members
    for a in (ring_theta, ring_phi0, counts, pixels):
        a.flags.writeable = False
    return ring_theta, ring_phi0, counts, pixels


def spec_rings(nside: int) -> tuple[Array, Array]:
    """The rings of ``nside``, from north to south: their colatitudes ``(4 nside - 1,)`` and, for
    each, its NESTED pixels in increasing longitude, ``(4 nside - 1, 4 nside)`` padded with -1.
    Implements the ring geometry of M-CONV-HPX, step 3.

    The order of the pixels along the rings, ring after ring, is HEALPix's RING ordering.
    """
    ring_theta, _, _, pixels = _ring_table(_check_nside(nside))
    return np.array(ring_theta), np.array(pixels)


# ---------------------------------------------------------------------- interpolation


def spec_interp_weights(nside: int, theta: Any, phi: Any) -> tuple[Array, Array]:
    """The bilinear interpolation of the 4 nearest pixels at the points ``(θ, φ)``: their NESTED
    indices and their weights, ``(4, *shape)`` each, the weights summing to 1. Implements
    M-CONV-HPX, step 3.

    The rings are those of ``spec_rings``, ring r at colatitude θ_r holding N_r pixels at the
    longitudes φ_r + 2πk/N_r. On a ring, the point reads the two pixels whose longitudes enclose
    φ, k = ⌊t⌋ and k + 1 modulo N_r with t = (φ - φ_r) N_r / 2π, with the weights 1 - (t - k)
    and t - k: linear in longitude.

    - **Between two rings** θ_r ≤ θ < θ_{r+1}: the two pixels of each ring, the weights of ring
      r multiplied by 1 - w and those of ring r + 1 by w, w = (θ - θ_r) / (θ_{r+1} - θ_r):
      linear in colatitude.
    - **North of the first ring** (θ < θ_1): the North Pole takes the mean of the 4 pixels of
      the first ring, and the point is interpolated between the pole and the first ring,
      w = θ / θ_1: the two pixels enclosing φ on the first ring get w (1 - (t - k)) + (1 - w)/4
      and w (t - k) + (1 - w)/4, the two others (1 - w)/4 each.
    - **South of the last ring**, the same with the South Pole, w = (π - θ) / (π - θ_last).

    The edges of the base faces need no special case: the rings run round the whole sphere.
    """
    ring_theta, ring_phi0, counts, pixels = _ring_table(_check_nside(nside))
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.mod(np.asarray(phi, dtype=np.float64), 2.0 * math.pi)
    n_rings = ring_theta.size
    # The number of rings at or north of the point: the ring above is that one minus 1.
    above = np.searchsorted(ring_theta, theta, side="right")
    north, south = above == 0, above == n_rings
    upper = np.clip(above - 1, 0, n_rings - 1)
    lower = np.clip(above, 0, n_rings - 1)

    def on_ring(r: Array, shift: int = 0) -> tuple[Array, Array, Array]:
        """The two pixels of ring r enclosing φ (moved ``shift`` positions along the ring), and
        the weight of the second."""
        n = counts[r]
        t = (phi - ring_phi0[r]) * n / (2.0 * math.pi)
        k = np.floor(t)
        first = (k.astype(np.int64) + shift) % n
        return pixels[r, first], pixels[r, (first + 1) % n], t - k

    a0, a1, fa = on_ring(upper)
    b0, b1, fb = on_ring(lower)
    with np.errstate(divide="ignore", invalid="ignore"):  # w is replaced at the poles
        w = (theta - ring_theta[upper]) / (ring_theta[lower] - ring_theta[upper])
        between = np.stack([(1 - w) * (1 - fa), (1 - w) * fa, w * (1 - fb), w * fb])

    # Around a pole: the cap ring, its two pixels enclosing φ and the two opposite ones.
    cap = np.where(north, 0, n_rings - 1)
    c0, c1, fc = on_ring(cap)
    c2, c3, _ = on_ring(cap, shift=2)
    w_pole = np.where(north, theta / ring_theta[0], (math.pi - theta) / (math.pi - ring_theta[-1]))
    rest = (1 - w_pole) / 4
    polar = np.stack([w_pole * (1 - fc) + rest, w_pole * fc + rest, rest, rest])

    at_pole = north | south
    index = np.where(at_pole, np.stack([c0, c1, c2, c3]), np.stack([a0, a1, b0, b1]))
    weight = np.where(at_pole, polar, between)
    return index, weight


# ---------------------------------------------------------------------- stencil and rotations


def spec_stencil(nside: int, K: int = KERNEL_SIZE) -> tuple[Array, Array]:  # noqa: N803
    """The K x K stencil around the North Pole: the colatitude and the azimuth of its points,
    ``(K²,)`` each, in the flattened order p = (v + K//2) K + (u + K//2). Implements
    M-CONV-HPX, step 1.

    The point of offsets (u, v) ∈ {-K//2, …, K//2}² is at colatitude √(u² + v²) alpha and azimuth
    atan2(v, u), with alpha = √(4π / (12 nside²)) the mean pixel size at ``nside``.
    """
    alpha = math.sqrt(4.0 * math.pi / (12.0 * _check_nside(nside) ** 2))
    offsets = np.arange(K) - K // 2
    v_grid, u_grid = np.meshgrid(offsets, offsets, indexing="ij")  # v along the first axis
    u = u_grid.reshape(-1).astype(np.float64)
    v = v_grid.reshape(-1).astype(np.float64)
    return np.hypot(u, v) * alpha, np.arctan2(v, u)


def spec_gauge_angles(theta: Any, phi: Any, L: int, gauge: str = "cosmo") -> Array:  # noqa: N803
    """The angle alpha_g of each gauge g = 0 … L-1 at the pixels ``(θ, φ)``, ``(L, *shape)``.
    Implements the gauges of M-CONV-HPX, step 2 (OP-47).

    alpha_g = alpha_base + s π g / L. Gauge ``"cosmo"``: alpha_base = -φ and s = +1 in the northern
    hemisphere, θ ≤ π/2 (the equator included); alpha_base = +φ and s = -1 in the southern one.
    """
    if gauge not in GAUGES:
        raise ValueError(f"gauge={gauge!r}: the oracle has the gauge 'cosmo' only (M-CONV-HPX).")
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    northern = theta <= math.pi / 2
    base = np.where(northern, -phi, phi)
    sign = np.where(northern, 1.0, -1.0)
    return np.stack([base + sign * math.pi * g / L for g in range(L)])


def spec_rotation(theta: Any, phi: Any, alpha: Any) -> Array:
    """The rotation R = R_gauge(alpha) · R_z(φ) · R_y(θ) that carries the stencil to the pixel
    ``(θ, φ)`` in the gauge of angle alpha, ``(*shape, 3, 3)``. Implements M-CONV-HPX, step 2.

    R_y(θ) and R_z(φ) are the rotations about the y and z axes; R_z R_y carries the North Pole
    to the pixel's direction n, the x axis (u) onto e_θ and the y axis (v) onto e_φ.
    R_gauge(alpha) = cos alpha I + sin alpha [n]ₓ + (1 - cos alpha) n nᵀ turns by alpha about n,
    counter-clockwise seen from outside the sphere, so that the u axis becomes
    cos alpha e_θ + sin alpha e_φ.
    """
    theta, phi, alpha = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (theta, phi, alpha))
    )
    zero, one = np.zeros(theta.shape), np.ones(theta.shape)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    r_y = np.stack(
        [
            np.stack([ct, zero, st], -1),
            np.stack([zero, one, zero], -1),
            np.stack([-st, zero, ct], -1),
        ],
        -2,
    )
    r_z = np.stack(
        [
            np.stack([cp, -sp, zero], -1),
            np.stack([sp, cp, zero], -1),
            np.stack([zero, zero, one], -1),
        ],
        -2,
    )
    n = np.stack([st * cp, st * sp, ct], -1)
    cross = np.stack(
        [
            np.stack([zero, -n[..., 2], n[..., 1]], -1),
            np.stack([n[..., 2], zero, -n[..., 0]], -1),
            np.stack([-n[..., 1], n[..., 0], zero], -1),
        ],
        -2,
    )
    ca, sa = np.cos(alpha)[..., None, None], np.sin(alpha)[..., None, None]
    r_gauge = ca * np.eye(3) + sa * cross + (1 - ca) * n[..., :, None] * n[..., None, :]
    out: Array = r_gauge @ r_z @ r_y
    return out


# ---------------------------------------------------------------------- convolution


def _as_cell_ids(nside: int, cell_ids: Any) -> Array:
    if cell_ids is None:
        return np.arange(12 * nside * nside, dtype=np.int64)
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("cell_ids must be a non-empty 1D array of NESTED indices (M-DATA-HPX).")
    if np.issubdtype(ids.dtype, np.floating) and not np.array_equal(ids, np.round(ids)):
        raise ValueError("cell_ids must be integers (M-DATA-HPX).")
    out = ids.astype(np.int64)
    if np.any(np.diff(out) <= 0):
        raise ValueError(
            "cell_ids must be strictly increasing (M-DATA-HPX, OP-90): sort the pixels, and the "
            "data with them."
        )
    if out[0] < 0 or out[-1] >= 12 * nside * nside:
        raise ValueError(f"cell_ids must lie in [0, 12 nside²) = [0, {12 * nside * nside}).")
    return out


def spec_stencil_points(
    nside: int,
    cell_ids: Any = None,
    L: int = N_GAUGES,  # noqa: N803
    K: int = KERNEL_SIZE,  # noqa: N803
) -> tuple[Array, Array]:
    """The colatitude and the longitude of the stencil points carried to each pixel of
    ``cell_ids`` (the full sky by default) in each gauge, ``(L, Npix, K²)`` each. Implements
    M-CONV-HPX, steps 1 and 2.

    The stencil (``spec_stencil``) is carried to each pixel by ``spec_rotation``, with the angles
    of ``spec_gauge_angles``.
    """
    nside = _check_nside(nside)
    ids = _as_cell_ids(nside, cell_ids)
    theta, phi = spec_pixel_centres(nside, ids)
    colatitude, azimuth = spec_stencil(nside, K)
    points = np.stack(
        [
            np.sin(colatitude) * np.cos(azimuth),
            np.sin(colatitude) * np.sin(azimuth),
            np.cos(colatitude),
        ],
        -1,
    )  # [p, xyz]
    rotation = spec_rotation(theta[None], phi[None], spec_gauge_angles(theta, phi, L))  # [g, k]
    rotated = np.einsum("gkab,pb->gkpa", rotation, points)
    theta_p = np.arctan2(np.hypot(rotated[..., 0], rotated[..., 1]), rotated[..., 2])
    phi_p = np.arctan2(rotated[..., 1], rotated[..., 0])
    return theta_p, phi_p


def spec_conv_tables(
    nside: int,
    cell_ids: Any = None,
    L: int = N_GAUGES,  # noqa: N803
    K: int = KERNEL_SIZE,  # noqa: N803
) -> tuple[Array, Array]:
    """The interpolation tables of the convolution on the pixels ``cell_ids`` (the full sky by
    default) of ``nside``: for each gauge g, pixel k, stencil point p and neighbour i, the
    position in ``cell_ids`` of the pixel read and its weight, ``(L, Npix, K², 4)`` each.
    Implements M-CONV-HPX, steps 1 to 3, with its partial-sky rule (OP-50, OP-110).

    Each stencil point of ``spec_stencil_points`` is interpolated by ``spec_interp_weights``.
    **Partial sky:** a neighbour that is not in ``cell_ids``, or whose weight is below 10⁻¹², is
    dropped, and the weights of the remaining ones are divided by their sum, with no floor; a
    stencil point with no remaining neighbour takes the neighbours and the weights of the stencil
    centre, which reads the pixel itself. The threshold is applied on the full sky too, where it
    changes nothing beyond rounding.
    """
    nside = _check_nside(nside)
    ids = _as_cell_ids(nside, cell_ids)
    neighbours, weights = spec_interp_weights(nside, *spec_stencil_points(nside, ids, L, K))
    neighbours, weights = np.moveaxis(neighbours, 0, -1), np.moveaxis(weights, 0, -1)  # [g,k,p,i]

    position = np.clip(np.searchsorted(ids, neighbours), 0, ids.size - 1)
    kept_mask = (ids[position] == neighbours) & (weights >= SUPPORT_THRESHOLD)
    kept = np.where(kept_mask, weights, 0.0)
    empty = ~kept_mask.any(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        kept = np.where(empty, 0.0, kept / kept.sum(axis=-1, keepdims=True))
    centre = (K * K) // 2  # the point (0, 0)
    if np.any(empty[..., centre, :]):
        raise ValueError("a stencil centre reads no pixel of cell_ids (M-CONV-HPX).")
    index = np.where(empty, position[..., centre : centre + 1, :], position)
    weight = np.where(empty, kept[..., centre : centre + 1, :], kept)
    return index, weight


def spec_wavelet_hpx(K: int = KERNEL_SIZE) -> Array:  # noqa: N803
    """The HEALPix wavelet on the flattened stencil, ``(K²,)``, complex. Implements M-WAV-HPX.

    ψ[u, v] = exp(-(u² + v²)/2) exp(iπu/2) on the offsets of the stencil, u along its second
    axis; then the mean is subtracted, and the kernel is divided by Σ|ψ|, taken after the
    subtraction (OP-48).
    """
    offsets = np.arange(K) - K // 2
    v, u = np.meshgrid(offsets, offsets, indexing="ij")
    psi = np.exp(-(u**2 + v**2) / 2.0) * np.exp(1j * math.pi * u / 2.0)
    psi = psi - psi.mean()
    out: Array = (psi / np.abs(psi).sum()).reshape(-1)
    return out


def spec_conv_hpx(x: Array, tables: tuple[Array, Array], kernel: Array) -> Array:
    """The convolution of ``x``, ``(..., Npix)``, real: ``(..., L, Npix)``, complex. Implements
    M-CONV-HPX, step 4.

    y[g, k] = Σ_p W[p] Σ_i w[g, k, p, i] x[n[g, k, p, i]], with (n, w) the tables of
    ``spec_conv_tables`` and W the kernel; no bias. The stencil turns with the gauge, the kernel
    does not.
    """
    index, weight = tables
    interpolated = (np.asarray(x)[..., index] * weight).sum(axis=-1)  # [..., g, k, p]
    out: Array = interpolated @ kernel.astype(np.complex128)
    return out


def spec_support_hpx(invalid: Mask, tables: tuple[Array, Array]) -> Mask:
    """The invalid pixels of data convolved on this geometry, from those of its input.
    Implements M-MASK-HPX (the support of a layer).

    A pixel is invalid if the convolution of the invalid 0/1 map with a kernel of ones over the
    stencil is above 10⁻¹² for at least one gauge.
    """
    if invalid is None:
        return None
    index, weight = tables
    touched = (invalid.astype(np.float64)[index] * weight).sum(axis=(-1, -2))  # [g, k]
    out: NDArray[np.bool_] = (touched > SUPPORT_THRESHOLD).any(axis=0)
    return out


# ---------------------------------------------------------------------- decimation


def spec_decimation_hpx(nside: int, cell_ids: Any = None) -> tuple[Array, Array]:
    """One step of the decimation, from the pixels ``cell_ids`` of ``nside`` (the full sky by
    default) to their parents at ``nside / 2``: the matrix ``(Ncoarse, Nfine)`` and the coarse
    ``cell_ids``. Implements M-DOWN-HPX.

    The coarse pixels are the NESTED parents ⌊n / 4⌋ of the fine ones, in ascending order. Row
    n₀ holds e^(-gamma²/2sigma²) for every fine pixel of ``cell_ids`` whose centre is within rho
    of that of n₀, gamma the angular distance between the centres, divided by its sum:
    rho = 3 √(4π / (12 nside²)), three fine pixel sizes, and sigma = rho / 2. Every fine pixel is
    compared with every coarse pixel. The 4 children of n₀ are within rho, so each row has a
    positive sum: the fallbacks of healpix-analyse are never reached.
    """
    nside = _check_nside(nside)
    if nside < 2:
        raise ValueError("nside 1 has no coarser level (M-DOWN-HPX).")
    ids = _as_cell_ids(nside, cell_ids)
    coarse = np.unique(ids // 4)
    rho = DECIMATION_RADIUS * math.sqrt(4.0 * math.pi / (12.0 * nside * nside))
    sigma = DECIMATION_WIDTH * rho

    def unit(theta: Array, phi: Array) -> Array:
        return np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1
        )

    fine_points = unit(*spec_pixel_centres(nside, ids))
    matrix = np.zeros((coarse.size, ids.size))
    for row, centre in enumerate(unit(*spec_pixel_centres(nside // 2, coarse))):
        chord = np.linalg.norm(fine_points - centre, axis=-1)
        gamma = 2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0))
        weights = np.where(gamma <= rho, np.exp(-(gamma**2) / (2.0 * sigma**2)), 0.0)
        matrix[row] = weights / weights.sum()
    return matrix, coarse


def spec_reweighting_hpx(fraction: Array, invalid: NDArray[np.bool_]) -> Array:
    """The reweighting map of a decimated level: 1 / (1 - f), with 1 - f not below 10⁻⁸, and 0
    on the invalid pixels. Implements M-MASK-HPX (the reweighting maps)."""
    out: Array = np.where(invalid, 0.0, 1.0 / np.maximum(1.0 - fraction, REWEIGHT_FLOOR))
    return out


def spec_downsample_hpx(x: Array, matrix: Array, invalid: Mask = None) -> tuple[Array, Mask]:
    """One decimation step of ``x``, ``(..., Nfine)``, with the matrix of
    ``spec_decimation_hpx``: the result and its invalid pixels. Implements M-DOWN-HPX, and
    M-DOWN-MASK with this step (M-MASK-HPX).

    Without a mask, ``x @ matrixᵀ``; a complex array is decimated part by part, which this
    linear map does. With one, the invalid pixels are set to 0, the step is applied, and the
    result is multiplied by the reweighting map 1/(1 - f), with f the invalid 0/1 mask
    decimated by the same step; a coarse pixel is invalid if f > τ = 0.33, and holds 0.
    """
    if invalid is None:
        out: Array = np.asarray(x) @ matrix.T
        return out, None
    fraction = matrix @ invalid.astype(np.float64)
    invalid_out: NDArray[np.bool_] = fraction > NAN_THRESHOLD
    down = np.where(invalid, 0.0, x) @ matrix.T
    return down * spec_reweighting_hpx(fraction, invalid_out), invalid_out


# ---------------------------------------------------------------------- data and defaults


def spec_defaults_hpx(nside: int) -> dict[str, Any]:
    """The default parameters for HEALPix data of ``nside``. Implements M-DEF-HPX.

    The sphere (OP-45) and the tables in float64 (OP-46) are ESTEL's decisions, not STL's
    defaults; the angular spectrum is unavailable (LF24).
    """
    n = _check_nside(nside)
    return {
        "J": max(1, math.floor(math.log2(n)) - 1),
        "L": N_GAUGES,
        "K": KERNEL_SIZE,
        "wavelet": "HealpixWavelet",
        "gauge": "cosmo",
        "ellipsoid": "sphere",
        "nan_threshold": NAN_THRESHOLD,
        "border": 0,
        "decimation_radius": DECIMATION_RADIUS,
        "decimation_width": DECIMATION_WIDTH,
        "norm": "self",
        "ref_batch_mean": False,
        "standardize": True,
        "cross_matrix": "upper triangle",
    }


def spec_data_hpx(
    array: Any,
    nside: int | None = None,
    cell_ids: Any = None,
    mask: Any = None,
) -> tuple[Array, Mask, int, Array]:
    """The array promoted to ``(Nb, Nc, Npix)`` in float64, with NaN at its missing pixels, its
    mask, its ``nside`` and its ``cell_ids``. Implements M-DATA-HPX.

    A 1D array is one map, ``(1, 1, Npix)``; a 2D array ``(Nc, Npix)``. On a full sky
    (``cell_ids`` not given), ``nside`` is read from Npix = 12 nside²; with ``cell_ids``, the
    NESTED indices of the pixels, strictly increasing (OP-90), ``nside`` must be given (OP-91).
    The mask is one map ``(Npix,)``, shared by the batch and the channels; without it, it is
    read from the NaN of the array, which must then be a single map.
    """
    x = np.asarray(array)
    if np.iscomplexobj(x):
        raise ValueError("the data must be real (M-DATA-HPX); pass a real array.")
    if not 1 <= x.ndim <= 3:
        raise ValueError(
            f"the array has {x.ndim} axes; pass (Npix,), (Nc, Npix) or (Nb, Nc, Npix)."
        )
    x = np.array(x, dtype=np.float64).reshape((1,) * (3 - x.ndim) + x.shape)
    npix = x.shape[-1]
    if cell_ids is None:
        n = math.isqrt(npix // 12)
        if 12 * n * n != npix or n & (n - 1):
            raise ValueError(
                f"{npix} pixels are not a full sky 12 nside² with nside a power of 2 "
                "(M-DATA-HPX); pass nside and cell_ids for a partial sky."
            )
        if nside is not None and nside != n:
            raise ValueError(f"nside={nside} but {npix} pixels are a full sky of nside {n}.")
        nside = n
    elif nside is None:
        raise ValueError("cell_ids needs nside (M-DATA-HPX, OP-91); pass nside.")
    nside = _check_nside(nside)
    ids = _as_cell_ids(nside, cell_ids)
    if ids.size != npix:
        raise ValueError(f"{ids.size} cell_ids for {npix} pixels: pass one index per pixel.")
    if mask is None:
        nan = np.isnan(x)
        if not nan.any():
            return x, None, nside, ids
        if x.shape[0] * x.shape[1] != 1:
            raise ValueError(
                "the mask is read from the NaN of a single map only (M-DATA-HPX); pass the mask "
                "explicitly."
            )
        invalid = nan.reshape(npix)
    else:
        invalid = np.asarray(mask, dtype=bool)
        if invalid.shape != (npix,):
            raise ValueError(
                f"the mask has shape {invalid.shape}; pass one map ({npix},), shared by every "
                "batch element and channel (M-DATA-HPX)."
            )
    return np.where(invalid, np.nan, x), invalid, nside, ids


# ---------------------------------------------------------------------- levels and masks


@functools.cache
def _levels(nside: int, ids: bytes, J: int, L: int, K: int) -> tuple[dict[str, Any], ...]:  # noqa: N803
    cell_ids = np.frombuffer(ids, dtype=np.int64)
    levels = []
    for d in range(J):
        level: dict[str, Any] = {
            "nside": nside,
            "cell_ids": cell_ids,
            "conv": spec_conv_tables(nside, cell_ids, L, K),
        }
        if d < J - 1:
            level["down"], cell_ids = spec_decimation_hpx(nside, cell_ids)
            nside //= 2
        levels.append(level)
    for level in levels:
        for value in (level["cell_ids"], *level["conv"], level.get("down")):
            if value is not None:
                value.flags.writeable = False
    return tuple(levels)


def spec_levels_hpx(
    nside: int,
    cell_ids: Any,
    J: int,  # noqa: N803
    L: int = N_GAUGES,  # noqa: N803
    K: int = KERNEL_SIZE,  # noqa: N803
) -> tuple[dict[str, Any], ...]:
    """The geometry of the levels d = 0 … J-1: for each, its ``nside`` (``nside / 2^d``), its
    ``cell_ids``, its convolution tables (``"conv"``) and, but for the last, its decimation
    matrix to the next level (``"down"``). Implements the levels of M-DATA-HPX and M-DOWN-HPX
    ("several levels are reached by repeating the step"). Read-only, built once per geometry.
    """
    nside = _check_nside(nside)
    if J < 1 or nside < 2 ** (J - 1):
        raise ValueError(f"J={J}: nside {nside} has at most {nside.bit_length()} levels.")
    ids = _as_cell_ids(nside, cell_ids)
    return _levels(nside, ids.tobytes(), J, L, K)


def spec_masks_hpx(levels: tuple[dict[str, Any], ...], invalid: Mask) -> dict[str, Any]:
    """The static masks and reweighting maps of the scattering, on the pixels of their level.
    Implements M-MASK-HPX, with M-MASK.

    - ``"field"[d]``: layer 0 at level d, decimated with τ; ``"reweight_field"[d]``, d ≥ 1;
    - ``"layer1"[j]``: the support of layer 0 at level j;
    - ``"down1"[d, j]``: layer 1 of scale j decimated to level d > j, and
      ``"reweight_down1"[d, j]``;
    - ``"layer2"[j, j3]``: the support of layer 1 of scale j at level j3 ≥ j.

    Without missing data every entry is ``None``.
    """
    n_levels = len(levels)
    out: dict[str, Any] = {
        "field": [invalid],
        "reweight_field": [None],
        "layer1": [],
        "down1": {},
        "reweight_down1": {},
        "layer2": {},
    }

    def step(mask: Mask, d: int) -> tuple[Mask, Any]:
        if mask is None:
            return None, None
        fraction = levels[d]["down"] @ mask.astype(np.float64)
        down = fraction > NAN_THRESHOLD
        return down, spec_reweighting_hpx(fraction, down)

    for d in range(1, n_levels):
        mask, reweight = step(out["field"][-1], d - 1)
        out["field"].append(mask)
        out["reweight_field"].append(reweight)
    for j in range(n_levels):
        out["layer1"].append(spec_support_hpx(out["field"][j], levels[j]["conv"]))
    for j in range(n_levels):
        mask = out["layer1"][j]
        for j3 in range(j, n_levels):
            if j3 > j:
                mask, out["reweight_down1"][j3, j] = step(mask, j3 - 1)
                out["down1"][j3, j] = mask
            out["layer2"][j, j3] = spec_support_hpx(mask, levels[j3]["conv"])
    return out


# ---------------------------------------------------------------------- scattering statistics


def spec_scattering_hpx(
    x: Array,
    nside: int,
    cell_ids: Any,
    J: int,  # noqa: N803
    L: int = N_GAUGES,  # noqa: N803
    invalid: Mask = None,
    cross_matrix: Any = None,
    K: int = KERNEL_SIZE,  # noqa: N803
) -> Statistics:
    """The scattering statistics S1 to S4 of ``x``, ``(Nb, Nc, Npix)`` on the pixels
    ``cell_ids`` of ``nside``, with its ``invalid`` pixels, in STL's full grids. Implements
    M-MASK-HPX (masks, covariances, no border), with the loop of section 1 (§4, "Scattering
    loop"): M-W1, M-S1 to M-S4 and M-CHAN, on the convolution of M-CONV-HPX with the wavelet of
    M-WAV-HPX and the decimation of M-DOWN-HPX.

    As in ``planar.spec_scattering``, with one pixel axis and no border: every mean is taken at
    the level of its arguments, over the pixels left by the union of their two masks. The
    invalid pixels of the field are set to 0 first, and those of every decimated level hold 0
    (``spec_downsample_hpx``), so that no NaN enters a convolution (OP-102).
    """
    nb, nc = x.shape[:2]
    pairs = planar._channel_pairs(cross_matrix, nc)
    levels = spec_levels_hpx(nside, cell_ids, J, L, K)
    masks = spec_masks_hpx(levels, invalid)
    kernel = spec_wavelet_hpx(K)

    # Layer 0: the field at each level (M-DOWN-HPX, M-DOWN-MASK).
    field = [x if invalid is None else np.where(invalid, 0.0, x)]
    for d in range(1, J):
        field.append(
            spec_downsample_hpx(field[-1], levels[d - 1]["down"], masks["field"][d - 1])[0]
        )

    # Layer 1: W_j x and U_j x at level j, [b, c, g, k] (M-W1).
    w = [spec_conv_hpx(field[j], levels[j]["conv"], kernel) for j in range(J)]
    u = [np.abs(w_j) for w_j in w]
    mask1 = masks["layer1"]

    # Layer 2: ψ_{l3} ⋆ D_{j→j3} U_j x at level j3, [b, c, l, l3, k], and its mask.
    v: dict[tuple[int, int], Array] = {}
    for j in range(J):
        modulus, modulus_invalid = u[j], mask1[j]
        for j3 in range(j, J):
            if j3 > j:
                modulus = spec_downsample_hpx(modulus, levels[j3 - 1]["down"], modulus_invalid)[0]
                modulus_invalid = masks["down1"][j3, j]
            v[j, j3] = spec_conv_hpx(modulus, levels[j3]["conv"], kernel)
    mask2 = masks["layer2"]

    def mean(a: Array, m: Mask) -> Array:
        return planar.spec_valid_mean(a, 1, m)

    s1 = np.full((nb, nc, nc, J, L), np.nan, dtype=np.complex128)
    s2 = np.full((nb, nc, nc, J, L), np.nan, dtype=np.complex128)
    s3 = np.full((nb, nc, nc, J, J, L, L), np.nan, dtype=np.complex128)
    s4 = np.full((nb, nc, nc, J, J, J, L, L, L), np.nan, dtype=np.complex128)

    for c1 in range(nc):
        for c2 in range(c1, nc):
            if not pairs[c1, c2]:
                continue
            orders = [(c1, c2)] if c1 == c2 else [(c1, c2), (c2, c1)]
            for j in range(J):
                for ell in range(L):
                    z1, z2 = w[j][:, c1, ell], w[j][:, c2, ell]
                    if c1 == c2:
                        first, second = np.abs(z1), np.abs(z1) ** 2
                    else:
                        r1 = z1 / np.sqrt(np.abs(z1) + planar.S1_EPSILON)
                        r2 = z2 / np.sqrt(np.abs(z2) + planar.S1_EPSILON)
                        first, second = r1 * np.conj(r2), z1 * np.conj(z2)
                    s1[:, c1, c2, j, ell] = mean(first, mask1[j])
                    s2[:, c1, c2, j, ell] = mean(second, mask1[j])
            for a, b in orders:
                for j3 in range(J):
                    for j2 in range(j3 + 1):
                        union = planar._union(mask2[j2, j3], mask1[j3])
                        for l2 in range(L):
                            for l3 in range(L):
                                product = v[j2, j3][:, a, l2, l3] * np.conj(w[j3][:, b, l3])
                                s3[:, a, b, j2, j3, l2, l3] = mean(product, union)
                        for j1 in range(j2 + 1):
                            union = planar._union(mask2[j1, j3], mask2[j2, j3])
                            for l1 in range(L):
                                for l2 in range(L):
                                    for l3 in range(L):
                                        product = v[j1, j3][:, a, l1, l3] * np.conj(
                                            v[j2, j3][:, b, l2, l3]
                                        )
                                        s4[:, a, b, j1, j2, j3, l1, l2, l3] = mean(product, union)
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}


# ---------------------------------------------------------------------- apply


def spec_apply_hpx(
    array: Any,
    nside: int | None = None,
    cell_ids: Any = None,
    mask: Any = None,
    J: int | None = None,  # noqa: N803
    L: int | None = None,  # noqa: N803
    *,
    norm: str = "self",
    ref: Statistics | None = None,
    ref_batch_mean: bool = False,
    standardize: bool = True,
    cross_matrix: Any = None,
    iso: bool = False,
    angular_fourier: bool = False,
    angular_harmonics: int | None = None,
    scale_cosine: bool = False,
    scale_harmonics: int | None = None,
    dj: int | None = None,
    flatten: bool = False,
) -> Statistics:
    """The statistics of HEALPix data: the whole of ``apply``, in the order of section 1.
    Implements M-DATA-HPX and M-DEF-HPX, with M-ORDER.

    1. promotion to ``(Nb, Nc, Npix)``, ``nside`` and ``cell_ids`` (M-DATA-HPX);
    2. standardisation, if asked (M-STD);
    3. ``mean`` and ``var``, the plain means over the valid pixels (M-MOM);
    4. the scattering statistics S1 to S4 (``spec_scattering_hpx``);
    5. normalisation (M-NORM): ``"self"``, ``"ref"`` with ``ref``, the ``"reference"`` of other
       statistics, or ``"none"``;
    6. the reductions asked for, in the order ``iso``, ``angular_fourier``, ``scale_cosine``;
    7. the flattening, if asked (M-FLAT), in its real form, under ``"vector"``.

    J and L default to M-DEF-HPX; ``standardize`` is True by default (M-DEF-HPX, OP-120). There
    is no spectrum (LF24).
    """
    x, invalid, nside, ids = spec_data_hpx(array, nside, cell_ids, mask)  # 1
    defaults = spec_defaults_hpx(nside)
    n_scales = defaults["J"] if J is None else J
    n_gauges = defaults["L"] if L is None else L
    if norm not in ("self", "ref", "none"):
        raise ValueError(f'norm={norm!r}: pass "self", "ref" or "none" (M-NORM).')
    if norm == "self" and ref is not None:
        raise ValueError('norm="self" takes no reference (M-NORM).')
    if norm == "ref" and ref is None:
        raise ValueError('norm="ref" needs a reference (M-NORM).')
    if scale_cosine and not iso:
        raise ValueError("scale_cosine is defined after iso only (M-RED-SCALE).")

    out: Statistics = {}
    if standardize:  # 2
        x, out["mean_pre_std"], out["std_pre_std"] = planar.spec_standardize(x, 1, invalid)
    out["mean"], out["var"] = planar.spec_moments(x, 1, invalid)  # 3
    out.update(  # 4
        spec_scattering_hpx(x, nside, ids, n_scales, n_gauges, invalid, cross_matrix)
    )
    out["reference"] = None
    if norm != "none":  # 5
        reference = planar.spec_reference(out, ref_batch_mean) if norm == "self" else ref
        assert reference is not None
        out = {**planar.spec_normalize(out, reference), "reference": reference}
    if iso:  # 6
        out = planar.spec_iso(out)
    if angular_fourier:
        out = planar.spec_angular_fourier(out, angular_harmonics, after_iso=iso)
    if scale_cosine:
        out = planar.spec_scale_cosine(out, scale_harmonics, dj)
    if flatten:  # 7
        out["vector"] = planar.spec_flatten(out, flatten_complex=True)
    return out
