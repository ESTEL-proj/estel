"""The oracle for Cartesian data, 1D and 2D grids: sections 1 (2D) and 2 (1D) of
``docs/maths.md``.

A second implementation of the mathematics, written to be read against ``docs/maths.md`` rather
than to be fast (Part I, §4.2): float64, NumPy and the standard library only, nothing from the
rest of ``estel``. The loops over scales, orientations, channels and kernel taps are explicit, as
the formulas are written; only the sums over pixels are NumPy operations. What every geometry
shares (the means, the moments, the standardisation, the normalisation, the reductions and the
flattening) is in ``common``.

**Conventions.**

- ``ndim`` is the number of spatial axes: 2 (section 1) or 1 (section 2). Arrays are
  ``(Nb, Nc, *spatial)``; the spatial axes are always the last ``ndim`` axes, and every function
  acts on them whatever the leading axes are.
- Masks and statistics follow the conventions of ``common``.

Every public function lists the ``M-`` IDs it implements; ``tools/check_maths_ids.py`` checks
the mapping in both directions. The power spectrum (M-PS-2D) is in ``spectra``; ``spec_apply``
adds it to the statistics when asked (``compute_PS=True``), as the family ``PS``, ``(Nb, Nc, Nc,
n_b)``, with its reference in the normalisation and its place, last, in the vectors.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import spectra
from .common import (
    S1_EPSILON,
    Array,
    Mask,
    Statistics,
    _channel_pairs,
    _union,
    spec_angular_fourier,
    spec_flatten,
    spec_iso,
    spec_moments,
    spec_normalize,
    spec_reference,
    spec_scale_cosine,
    spec_standardize,
    spec_valid_mean,
)

FOURIER_GRID = 256
"""N, the size of the grid on which the wavelets are designed (M-WAV-2D, M-WAV-1D)."""
KERNEL_SIZE_CARTESIAN = 5
"""K, the size of the wavelet kernels (M-DEF, M-DEF-1D)."""
WIDTH_FACTOR = 2.5
"""c, the width factor of the 2D radial bump (M-WAV-2D)."""
NAN_THRESHOLD = {2: 0.33, 1: 0.5}
"""τ, per number of spatial axes (M-DOWN-MASK, M-SMOOTH-1D)."""


def _check_ndim(ndim: int) -> None:
    if ndim not in (1, 2):
        raise ValueError(f"ndim={ndim!r}: planar data have 1 or 2 spatial axes; pass 1 or 2.")


# ---------------------------------------------------------------------- data and defaults


def spec_defaults(spatial_shape: tuple[int, ...]) -> dict[str, Any]:
    """The default parameters for data of ``spatial_shape``. Implements M-DEF and M-DEF-1D.

    ``compute_PS`` is left out: ``spec_apply`` computes the spectrum only when it is asked for
    (``compute_PS=True``), so that without it the statistics are those of sections 1 and 2 alone.
    """
    ndim = len(spatial_shape)
    _check_ndim(ndim)
    return {
        "J": math.floor(math.log2(min(spatial_shape))) - 2,
        "L": 4 if ndim == 2 else 1,
        "K": KERNEL_SIZE_CARTESIAN,
        "nan_threshold": NAN_THRESHOLD[ndim],
        "norm": "self",
        "ref_batch_mean": False,
        "standardize": True,
        "cross_matrix": "upper triangle",
    }


def spec_data(
    array: Any,
    ndim: int,
    J: int,  # noqa: N803
    mask: Any = None,
    pbc: Any = None,
) -> tuple[Array, Mask]:
    """The array promoted to ``(Nb, Nc, *spatial)`` in float64, with NaN at its missing pixels,
    and its mask. Implements M-DATA-2D and M-DATA-1D.

    An array with ``ndim`` axes is one map, ``(1, 1, *spatial)``; with one more axis, its first
    axis is the channels, ``(1, Nc, *spatial)`` (in 1D too, OP-29). The data are real, the
    sizes divisible by 2^(J-1), ``pbc`` a boolean (OP-4), and the mask one map of the spatial
    shape, shared by every batch element and channel (OP-26). Without a mask, it is read from
    the NaN of the array, which must then be a single map.
    """
    _check_ndim(ndim)
    x = np.asarray(array)
    if np.iscomplexobj(x):
        raise ValueError("the data must be real (M-DATA-2D); pass a real array.")
    if not ndim <= x.ndim <= ndim + 2:
        raise ValueError(
            f"the array has {x.ndim} axes; pass ({ndim} spatial axes), (Nc, ...) or (Nb, Nc, ...)."
        )
    x = np.array(x, dtype=np.float64).reshape((1,) * (ndim + 2 - x.ndim) + x.shape)
    spatial = x.shape[2:]
    if any(n % 2 ** (J - 1) for n in spatial):
        raise ValueError(
            f"the sizes {spatial} are not divisible by 2^(J-1) = {2 ** (J - 1)} (M-DATA-2D, "
            "OP-8); use a smaller J or other sizes."
        )
    if not isinstance(pbc, bool | np.bool_):
        raise ValueError(f"pbc={pbc!r}: pass True or False (M-DATA-2D, OP-4).")
    if mask is None:
        nan = np.isnan(x)
        if not nan.any():
            return x, None
        if x.shape[0] * x.shape[1] != 1:
            raise ValueError(
                "the mask is read from the NaN of a single map only (M-DATA-2D, OP-26); pass "
                "the mask explicitly."
            )
        invalid = nan.reshape(spatial)
    else:
        invalid = np.asarray(mask, dtype=bool)
        if invalid.shape != spatial:
            raise ValueError(
                f"the mask has shape {invalid.shape}; pass one map of the spatial shape "
                f"{spatial}, shared by every batch element and channel (M-DATA-2D, OP-26)."
            )
    return np.where(invalid, np.nan, x), invalid


# ---------------------------------------------------------------------- filtering and wavelets


def _pad(a: Array, ndim: int, before: int, after: int, pbc: bool) -> Array:
    """``a`` padded on its spatial axes: circular if ``pbc``, replicate otherwise (M-CONV-2D)."""
    widths = [(0, 0)] * (a.ndim - ndim) + [(before, after)] * ndim
    return np.pad(a, widths, mode="wrap" if pbc else "edge")


def spec_correlate_cartesian(a: Array, kernel: Array, pbc: bool) -> Array:
    """The filtering of ``a`` by ``kernel`` along its last axes, one per axis of the kernel.
    Implements M-CONV-2D, and the filtering of M-WAV-1D.

    A cross-correlation, the kernel not flipped: ``(w ⋆ a)[n] = Σ_m w[m] ã[n + m - p]``, with
    p = ⌊K/2⌋ and ã the array padded by p pixels on each side, circular if ``pbc`` and
    replicate otherwise. The output has the size of the input. A complex kernel on a real array
    gives the two real correlations of its real and imaginary parts.
    """
    ndim = kernel.ndim
    spatial = a.shape[a.ndim - ndim :]
    p = kernel.shape[0] // 2
    padded = _pad(a, ndim, p, p, pbc)
    out = np.zeros(a.shape, dtype=np.result_type(a.dtype, kernel.dtype))
    for m in np.ndindex(kernel.shape):
        window: tuple[Any, ...] = (..., *(slice(m[i], m[i] + spatial[i]) for i in range(ndim)))
        out = out + kernel[m] * padded[window]
    return out


def _bump(r: Array, c: float) -> Array:
    """``exp(-c r² / (1 - c r²))`` where ``r < c^(-1/2)``, 0 elsewhere (M-WAV-2D, M-WAV-1D)."""
    inside = c * r**2 < 1.0
    safe = np.where(inside, c * r**2, 0.0)
    return np.where(inside, np.exp(-safe / (1.0 - safe)), 0.0)


def _inverse_dft(spectrum: Array, omega: Array, K: int) -> Array:  # noqa: N803
    """The central ``K`` pixels (per axis) of the inverse discrete Fourier transform of
    ``spectrum``, given at the integer frequencies ``omega`` along each of its axes, the pixel 0
    at the centre: ``ψ[n] = N^(-d) Σ_ω ψ̂(ω) exp(2πi ω·n / N)`` for -⌊K/2⌋ ≤ n_i ≤ ⌊K/2⌋."""
    n_grid = omega.size
    pixels = np.arange(K) - K // 2
    phases = np.exp(2j * np.pi * np.outer(pixels, omega) / n_grid)  # [n, ω]
    out: Array = spectrum.astype(np.complex128)
    for _ in range(spectrum.ndim):
        # Sum over the first frequency axis; the new pixel axis goes last.
        out = np.tensordot(out, phases, axes=([0], [1])) / n_grid
    return out


def spec_wavelets_2d(L: int, K: int = KERNEL_SIZE_CARTESIAN) -> Array:  # noqa: N803
    """The ``L`` bump-steerable kernels, ``(L, K, K)``, complex. Implements M-WAV-2D.

    In Fourier space, on the N x N grid of integer frequencies ω_i in [-N/2, N/2), N = 256:
    θ_l = πl/L + π/2, q = ω R(θ_l), the radial bump B(r) with r = | ‖q‖ - ξ₀ | / ξ₀,
    ξ₀ = N/(2√2), c = 2.5, the angular part cos^(L-1) φ for |φ| < π/2, the normalisation by
    the maximum and by c_L / c_4. In pixel space: the inverse transform, the central K x K
    pixels, the mean subtracted, no renormalisation (OP-10).
    """
    n_grid = FOURIER_GRID
    omega = np.arange(-n_grid // 2, n_grid // 2, dtype=np.float64)
    w1, w2 = np.meshgrid(omega, omega, indexing="ij")
    xi0 = n_grid / (2.0 * math.sqrt(2.0))

    def steerable_constant(n: int) -> float:
        return float(
            2 ** (n - 1) * math.factorial(n - 1) / math.sqrt(n * math.factorial(2 * (n - 1)))
        )

    kernels = []
    for ell in range(L):
        theta = math.pi * ell / L + math.pi / 2
        # q = ω R(θ), a row vector times R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]].
        q1 = w1 * math.cos(theta) + w2 * math.sin(theta)
        q2 = -w1 * math.sin(theta) + w2 * math.cos(theta)
        radial = _bump(np.abs(np.hypot(q1, q2) - xi0) / xi0, WIDTH_FACTOR)
        phi = np.arctan2(q2, q1)
        angular = np.where(np.abs(phi) < math.pi / 2, np.cos(phi) ** (L - 1), 0.0)
        psi_hat = radial * angular
        psi_hat = psi_hat / np.abs(psi_hat).max() * steerable_constant(L) / steerable_constant(4)
        psi = _inverse_dft(psi_hat, omega, K)
        kernels.append(psi - psi.mean())
    return np.stack(kernels)


def spec_wavelets_1d(L: int, K: int = KERNEL_SIZE_CARTESIAN) -> Array:  # noqa: N803
    """The 1D bump-steerable kernels, ``(L, K)``, complex, L = 1 or 2. Implements M-WAV-1D.

    In Fourier space, on N = 256 integer frequencies ω in [-N/2, N/2), with ξ₀ = N/2:
    ``exp(-r² / (1 - r²))`` for r = |ω - ξ₀| / ξ₀ < 1, without width factor, angular part or
    normalisation (OP-28). In pixel space: the inverse transform, the central K taps, the mean
    subtracted, no renormalisation. With L = 2 the second kernel is the conjugate of the first.
    """
    if L not in (1, 2):
        raise ValueError(f"L={L!r}: 1D data have L = 1 or L = 2 (M-WAV-1D).")
    n_grid = FOURIER_GRID
    omega = np.arange(-n_grid // 2, n_grid // 2, dtype=np.float64)
    xi0 = n_grid / 2.0
    psi = _inverse_dft(_bump(np.abs(omega - xi0) / xi0, 1.0), omega, K)
    psi = psi - psi.mean()
    return np.stack([psi, np.conj(psi)][:L])


@functools.cache
def _wavelets_cartesian(ndim: int, L: int, K: int) -> Array:  # noqa: N803
    """The wavelet bank of M-WAV-2D or M-WAV-1D, built once per ``(ndim, L, K)`` and read-only,
    so that the cached array cannot be altered."""
    bank = spec_wavelets_2d(L, K) if ndim == 2 else spec_wavelets_1d(L, K)
    bank.flags.writeable = False
    return bank


# ---------------------------------------------------------------------- downsampling


def spec_smoothing_kernel_2d() -> Array:
    """The 6 x 6 smoothing kernel, as STL writes it. Implements M-SMOOTH-2D.

    The table of maths.md: its rows 4 to 6 mirror its rows 3 to 1. It sums to 1.00000004
    (OP-59).
    """
    rows = [
        [0.00238073, -0.00275097, -0.02402609, -0.02402609, -0.00275097, 0.00238073],
        [-0.00275097, 0.0031788, 0.0277626, 0.0277626, 0.0031788, -0.00275097],
        [-0.02402609, 0.0277626, 0.2424694, 0.2424694, 0.0277626, -0.02402609],
    ]
    return np.array(rows + rows[::-1], dtype=np.float64)


def spec_smoothing_kernel_1d() -> Array:
    """The 6-tap smoothing kernel: the third row of the 2D table divided by its sum.
    Implements M-SMOOTH-1D.

    The sum is the correctly rounded one (``math.fsum``), as in STL: its last bit decides the
    pixels whose missing fraction equals τ exactly (OP-12).
    """
    row = spec_smoothing_kernel_2d()[2]
    out: Array = row / math.fsum(row)
    return out


def _smoothing_kernel(ndim: int) -> Array:
    return spec_smoothing_kernel_2d() if ndim == 2 else spec_smoothing_kernel_1d()


def spec_downsample_cartesian(x: Array, ndim: int, pbc: bool) -> Array:
    """One downsampling step of ``x`` along its last ``ndim`` axes, without mask. Implements
    M-DOWN (in 1D, the downsampling of M-SMOOTH-1D).

    ``x^(j+1)[n] = Σ_m s[m] x̃^(j)[2n + m]``, with s the smoothing kernel and x̃ the array padded
    by 2 pixels before and 3 after on each axis (OP-2), circular if ``pbc``, replicate otherwise.
    """
    _check_ndim(ndim)
    s = _smoothing_kernel(ndim)
    spatial = x.shape[x.ndim - ndim :]
    if any(n % 2 for n in spatial):
        raise ValueError(f"cannot downsample the odd sizes {spatial} (M-DATA-2D, OP-8).")
    padded = _pad(x, ndim, 2, 3, pbc)
    out = np.zeros((*x.shape[: x.ndim - ndim], *(n // 2 for n in spatial)), dtype=x.dtype)
    for m in np.ndindex(s.shape):
        window: tuple[Any, ...] = (
            ...,
            *(slice(m[i], m[i] + spatial[i], 2) for i in range(ndim)),
        )
        out = out + s[m] * padded[window]
    return out


def spec_downsample_masked(
    x: Array, invalid: NDArray[np.bool_], ndim: int, pbc: bool
) -> tuple[Array, NDArray[np.bool_]]:
    """One downsampling step of ``x`` whose pixels ``invalid`` are missing: the downsampled
    array and its invalid pixels. Implements M-DOWN-MASK.

    The invalid input pixels are set to 0, the step of M-DOWN is applied, and the result is
    divided by 1 - f, where f is the invalid mask, as 0/1, downsampled by the same step. An
    output pixel is invalid if f > τ (0.33 in 2D, 0.5 in 1D; OP-12), and holds NaN.
    """
    fraction = spec_downsample_cartesian(invalid.astype(np.float64), ndim, pbc)
    invalid_out = fraction > NAN_THRESHOLD[ndim]
    down = spec_downsample_cartesian(np.where(invalid, 0.0, x), ndim, pbc)
    with np.errstate(divide="ignore", invalid="ignore"):
        reweighted = down / (1.0 - fraction)
    return np.where(invalid_out, np.nan, reweighted), invalid_out


def _down(x: Array, invalid: Mask, ndim: int, pbc: bool) -> tuple[Array, Mask]:
    """One step, with M-DOWN-MASK when there is a mask and M-DOWN otherwise."""
    if invalid is None:
        return spec_downsample_cartesian(x, ndim, pbc), None
    return spec_downsample_masked(x, invalid, ndim, pbc)


# ---------------------------------------------------------------------- masks and borders


def spec_mask_conv(invalid: Mask, ndim: int, pbc: bool, K: int = KERNEL_SIZE_CARTESIAN) -> Mask:  # noqa: N803
    """The invalid pixels of data filtered by a wavelet, from those of its input. Implements
    M-MASK (the rule of layers 1 and 2).

    ``(E ⋆ m) > 0``, with E the K-wide support of the wavelets, all ones: an output pixel is
    invalid if the support touches an invalid input pixel. The mask is padded like the data.
    """
    if invalid is None:
        return None
    support = np.ones((K,) * ndim)
    touched: NDArray[np.bool_] = (
        spec_correlate_cartesian(invalid.astype(np.float64), support, pbc) > 0
    )
    return touched


def spec_border_cartesian(
    history: tuple[int, ...],
    level: int,
    pbc: bool,
    K: int = KERNEL_SIZE_CARTESIAN,  # noqa: N803
) -> int:
    """The border width β, in pixels at ``level``, of data filtered at the scales ``history``
    (none, one or two). Implements M-BORDER (the fully flexible method).

    0 without filtering or with ``pbc``; ⌈2^(j-d) ⌊K/2⌋⌉ after one filtering at scale j, the
    data at level d; after two, at j2 then j3: β₁ = ⌈2^(j2-j3) ⌊K/2⌋⌉, then
    ⌈2^(j3-d) (β₁ + ⌊K/2⌋)⌉.
    """
    half = K // 2
    if pbc or not history:
        return 0
    if len(history) == 1:
        return math.ceil(2.0 ** (history[0] - level) * half)
    j2, j3 = history
    first = math.ceil(2.0 ** (j2 - j3) * half)
    return math.ceil(2.0 ** (j3 - level) * (first + half))


def spec_first_layer(
    x_level: Array,
    ndim: int,
    L: int,  # noqa: N803
    pbc: bool,
    K: int = KERNEL_SIZE_CARTESIAN,  # noqa: N803
) -> tuple[Array, Array]:
    """The first layer at one scale, from the field at that level, ``(Nb, Nc, *spatial)``:
    ``W = ψ_l ⋆ x^(j)`` and ``U = |W|``, ``(Nb, Nc, L, *spatial)`` each. Implements M-W1.
    """
    psi = _wavelets_cartesian(ndim, L, K)
    w = np.stack([spec_correlate_cartesian(x_level, psi[ell], pbc) for ell in range(L)], axis=2)
    return w, np.abs(w)


# ---------------------------------------------------------------------- scattering statistics


def spec_scattering(
    x: Array,
    ndim: int,
    J: int,  # noqa: N803
    L: int,  # noqa: N803
    pbc: bool,
    invalid: Mask = None,
    cross_matrix: Any = None,
    K: int = KERNEL_SIZE_CARTESIAN,  # noqa: N803
) -> Statistics:
    """The scattering statistics S1 to S4 of ``x``, ``(Nb, Nc, *spatial)`` with NaN at its
    ``invalid`` pixels, in STL's full grids. Implements M-S1, M-S2, M-S3, M-S4 and M-CHAN, with
    the masks of M-MASK, the borders of M-BORDER and the first layer of M-W1.

    Every mean is taken at the level of its arguments, over the pixels left by the larger of
    their two borders and by the union of their two masks. S1 and S2 are filled for the pairs
    c1 ≤ c2 of ``cross_matrix`` only; S3 and S4 for both orders of each pair. Everything is
    stored complex, and NaN where it is not computed.
    """
    _check_ndim(ndim)
    nb, nc = x.shape[:2]
    pairs = _channel_pairs(cross_matrix, nc)
    psi = _wavelets_cartesian(ndim, L, K)

    # Layer 0: the field and its mask at each level (M-DOWN, M-DOWN-MASK).
    field, field_invalid = [x], [invalid]
    for _ in range(1, J):
        down, down_invalid = _down(field[-1], field_invalid[-1], ndim, pbc)
        field.append(down)
        field_invalid.append(down_invalid)

    # Layer 1: W_j x and U_j x at level j, and their mask (M-W1, M-MASK).
    w, u, mask1 = [], [], []
    for j in range(J):
        w_j, u_j = spec_first_layer(field[j], ndim, L, pbc, K)
        w.append(w_j)
        u.append(u_j)
        mask1.append(spec_mask_conv(field_invalid[j], ndim, pbc, K))

    # Layer 2: ψ_{l3} ⋆ D_{j→j3} U_j x at level j3, [b, c, l, l3, *spatial], and its mask.
    v: dict[tuple[int, int], Array] = {}
    mask2: dict[tuple[int, int], Mask] = {}
    for j in range(J):
        modulus, modulus_invalid = u[j], mask1[j]
        for j3 in range(j, J):
            if j3 > j:
                modulus, modulus_invalid = _down(modulus, modulus_invalid, ndim, pbc)
            v[j, j3] = np.stack(
                [spec_correlate_cartesian(modulus, psi[l3], pbc) for l3 in range(L)], axis=3
            )
            mask2[j, j3] = spec_mask_conv(modulus_invalid, ndim, pbc, K)

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
                border = spec_border_cartesian((j,), j, pbc, K)
                for ell in range(L):
                    z1, z2 = w[j][:, c1, ell], w[j][:, c2, ell]
                    if c1 == c2:
                        first = np.abs(z1)
                        second = np.abs(z1) ** 2
                    else:
                        with np.errstate(invalid="ignore"):  # NaN at the invalid pixels
                            r1 = z1 / np.sqrt(np.abs(z1) + S1_EPSILON)
                            r2 = z2 / np.sqrt(np.abs(z2) + S1_EPSILON)
                        first = r1 * np.conj(r2)
                        second = z1 * np.conj(z2)
                    s1[:, c1, c2, j, ell] = spec_valid_mean(first, ndim, mask1[j], border)
                    s2[:, c1, c2, j, ell] = spec_valid_mean(second, ndim, mask1[j], border)

            for a, b in orders:
                for j3 in range(J):
                    for j2 in range(j3 + 1):
                        border = max(
                            spec_border_cartesian((j2, j3), j3, pbc, K),
                            spec_border_cartesian((j3,), j3, pbc, K),
                        )
                        union = _union(mask2[j2, j3], mask1[j3])
                        for l2 in range(L):
                            for l3 in range(L):
                                product = v[j2, j3][:, a, l2, l3] * np.conj(w[j3][:, b, l3])
                                s3[:, a, b, j2, j3, l2, l3] = spec_valid_mean(
                                    product, ndim, union, border
                                )
                        for j1 in range(j2 + 1):
                            border = max(
                                spec_border_cartesian((j1, j3), j3, pbc, K),
                                spec_border_cartesian((j2, j3), j3, pbc, K),
                            )
                            union = _union(mask2[j1, j3], mask2[j2, j3])
                            for l1 in range(L):
                                for l2 in range(L):
                                    for l3 in range(L):
                                        product = v[j1, j3][:, a, l1, l3] * np.conj(
                                            v[j2, j3][:, b, l2, l3]
                                        )
                                        s4[:, a, b, j1, j2, j3, l1, l2, l3] = spec_valid_mean(
                                            product, ndim, union, border
                                        )
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}


# ---------------------------------------------------------------------- apply


def spec_apply(
    array: Any,
    ndim: int,
    pbc: bool,
    mask: Any = None,
    J: int | None = None,  # noqa: N803
    L: int | None = None,  # noqa: N803
    *,
    norm: str | None = None,
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
    compute_PS: bool = False,  # noqa: N803
    n_bins: int | None = None,
    Jmin: int = 0,  # noqa: N803
    spectrum_method: str = "auto",
) -> Statistics:
    """The statistics of ``array``: the whole of ``apply``, in its order. Implements M-ORDER.

    1. promotion to ``(Nb, Nc, …)`` and channel pairs (M-DATA-2D, M-DATA-1D, M-CHAN);
    2. standardisation, if asked (M-STD);
    3. ``mean`` and ``var`` (M-MOM);
    4. the scattering statistics S1 to S4 (M-S1 to M-S4);
    5. the power spectrum, if asked (``compute_PS``, 2D only; M-PS-2D): ``spectra.spec_spectra_2d``
       on the field at full resolution, after the standardisation, with the J of the operator,
       the pairs of ``cross_matrix``, ``Jmin``, ``n_bins`` and ``spectrum_method`` (the
       automatic choice by default, from ``pbc`` and the NaN of the field), under ``"PS"``;
    6. normalisation (M-NORM): ``norm="self"`` by the statistics themselves, ``norm="ref"`` by
       ``ref``, the ``"reference"`` of other statistics, ``norm="none"`` not at all;
    7. the reductions asked for, in the order ``iso``, ``angular_fourier``, ``scale_cosine``
       (M-RED-ISO, M-RED-ANG, M-RED-SCALE);
    8. the flattening, if asked (M-FLAT), in its real form, under ``"vector"``.

    Returns the families, the ``"reference"`` used (``None`` with ``norm="none"``) and, with
    ``standardize``, ``"mean_pre_std"`` and ``"std_pre_std"``. J and L default to M-DEF and
    M-DEF-1D; ``standardize`` is True by default, as in M-DEF (OP-120; STL: False).

    ``compute_PS`` is False by default, so that without it every value is that of the
    statistics alone; ``n_bins``, ``Jmin`` and ``spectrum_method`` other than their defaults
    need it, and it needs J ≥ 2 + Jmin (decisions of the maintainer on #148).
    """
    spatial = np.shape(array)[-ndim:]
    defaults = spec_defaults(tuple(int(n) for n in spatial))
    n_scales = defaults["J"] if J is None else J
    n_orient = defaults["L"] if L is None else L
    if norm not in ("self", "ref", "none"):
        raise ValueError(f'norm={norm!r}: pass "self", "ref" or "none" (M-NORM).')
    if norm == "self" and ref is not None:
        raise ValueError('norm="self" takes no reference (M-NORM).')
    if norm == "ref" and ref is None:
        raise ValueError('norm="ref" needs a reference (M-NORM).')
    if scale_cosine and not iso:
        raise ValueError("scale_cosine is defined after iso only (M-RED-SCALE).")
    if compute_PS and ndim != 2:
        raise ValueError("the power spectrum is defined on 2D data only (M-PS-2D); pass ndim=2.")
    if not compute_PS and (n_bins is not None or Jmin != 0 or spectrum_method != "auto"):
        raise ValueError(
            "n_bins, Jmin and spectrum_method act only with compute_PS=True (M-PS-2D); pass "
            "compute_PS=True or leave them at their defaults."
        )
    if compute_PS and n_scales < 2 + Jmin:
        raise ValueError(
            f"J={n_scales} < 2 + Jmin = {2 + Jmin}: the spectrum needs J >= 2 + Jmin (M-PS-2D, "
            "OP-93); pass compute_PS=False or use a larger image."
        )

    x, invalid = spec_data(array, ndim, n_scales, mask, pbc)  # 1
    out: Statistics = {}
    if standardize:  # 2
        x, out["mean_pre_std"], out["std_pre_std"] = spec_standardize(x, ndim, invalid)
    out["mean"], out["var"] = spec_moments(x, ndim, invalid)  # 3
    out.update(spec_scattering(x, ndim, n_scales, n_orient, pbc, invalid, cross_matrix))  # 4
    if compute_PS:  # 5
        out["PS"] = spectra.spec_spectra_2d(
            x, pbc, n_scales, Jmin, n_bins, spectrum_method, cross_matrix
        )["PS"]
    out["reference"] = None
    if norm != "none":  # 6
        reference = spec_reference(out, ref_batch_mean) if norm == "self" else ref
        assert reference is not None
        out = {**spec_normalize(out, reference), "reference": reference}
    if iso:  # 7
        out = spec_iso(out)
    if angular_fourier:
        out = spec_angular_fourier(out, angular_harmonics, after_iso=iso)
    if scale_cosine:
        out = spec_scale_cosine(out, scale_harmonics, dj)
    if flatten:  # 8
        out["vector"] = spec_flatten(out, flatten_complex=True)
    return out
