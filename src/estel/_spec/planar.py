"""The oracle for planar data: sections 1 (planar 2D) and 2 (1D) of ``docs/maths.md``.

A second implementation of the mathematics, written to be read against ``docs/maths.md`` rather
than to be fast (Part I, §4.2): float64, NumPy and the standard library only, nothing from the
rest of ``estel``. The loops over scales, orientations, channels and kernel taps are explicit, as
the formulas are written; only the sums over pixels are NumPy operations.

**Conventions.**

- ``ndim`` is the number of spatial axes: 2 (section 1) or 1 (section 2). Arrays are
  ``(Nb, Nc, *spatial)``; the spatial axes are always the last ``ndim`` axes, and every function
  acts on them whatever the leading axes are.
- A mask is a boolean array of the spatial shape, True where a pixel is invalid (M-NOT), or
  ``None`` when no value is missing. Invalid pixels hold NaN (STL's ``replace_nan_value``,
  M-DOWN-MASK): no mean ever reads them, so a NaN in a result is a mean with no valid pixel
  (M-MEAN) or a coefficient that is not computed (M-CHAN).
- The statistics are STL's full grids (M-CHAN, §2.4): ``mean`` and ``var`` are ``(Nb, Nc)``,
  real; S1 and S2 ``(Nb, Nc, Nc, J, L)``, S3 ``(Nb, Nc, Nc, J, J, L, L)`` indexed
  ``[c1, c2, j2, j3, l2, l3]`` and S4 ``(Nb, Nc, Nc, J, J, J, L, L, L)`` indexed
  ``[c1, c2, j1, j2, j3, l1, l2, l3]``, complex, NaN where no coefficient exists.

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

Array = NDArray[Any]
Mask = NDArray[np.bool_] | None
Statistics = dict[str, Any]

FOURIER_GRID = 256
"""N, the size of the grid on which the wavelets are designed (M-WAV-2D, M-WAV-1D)."""
KERNEL_SIZE = 5
"""K, the size of the wavelet kernels (M-DEF, M-DEF-1D)."""
WIDTH_FACTOR = 2.5
"""c, the width factor of the 2D radial bump (M-WAV-2D)."""
NAN_THRESHOLD = {2: 0.33, 1: 0.5}
"""τ, per number of spatial axes (M-DOWN-MASK, M-SMOOTH-1D)."""
S1_EPSILON = 1e-8
"""ε of the cross-channel S1 (M-S1)."""
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
"""The families, in the order of the flattening (M-FLAT)."""
SCATTERING = ("S1", "S2", "S3", "S4")
N_ORIENTATION_AXES = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}
N_SCALE_AXES = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}


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
        "K": KERNEL_SIZE,
        "nan_threshold": NAN_THRESHOLD[ndim],
        "norm": "self",
        "ref_batch_mean": False,
        "standardize": False,
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


def spec_correlate(a: Array, kernel: Array, pbc: bool) -> Array:
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


def spec_wavelets_2d(L: int, K: int = KERNEL_SIZE) -> Array:  # noqa: N803
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


def spec_wavelets_1d(L: int, K: int = KERNEL_SIZE) -> Array:  # noqa: N803
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
def _wavelets(ndim: int, L: int, K: int) -> Array:  # noqa: N803
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


def spec_downsample(x: Array, ndim: int, pbc: bool) -> Array:
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
    fraction = spec_downsample(invalid.astype(np.float64), ndim, pbc)
    invalid_out = fraction > NAN_THRESHOLD[ndim]
    down = spec_downsample(np.where(invalid, 0.0, x), ndim, pbc)
    with np.errstate(divide="ignore", invalid="ignore"):
        reweighted = down / (1.0 - fraction)
    return np.where(invalid_out, np.nan, reweighted), invalid_out


def _down(x: Array, invalid: Mask, ndim: int, pbc: bool) -> tuple[Array, Mask]:
    """One step, with M-DOWN-MASK when there is a mask and M-DOWN otherwise."""
    if invalid is None:
        return spec_downsample(x, ndim, pbc), None
    return spec_downsample_masked(x, invalid, ndim, pbc)


# ---------------------------------------------------------------------- masks, borders, means


def spec_mask_conv(invalid: Mask, ndim: int, pbc: bool, K: int = KERNEL_SIZE) -> Mask:  # noqa: N803
    """The invalid pixels of data filtered by a wavelet, from those of its input. Implements
    M-MASK (the rule of layers 1 and 2).

    ``(E ⋆ m) > 0``, with E the K-wide support of the wavelets, all ones: an output pixel is
    invalid if the support touches an invalid input pixel. The mask is padded like the data.
    """
    if invalid is None:
        return None
    support = np.ones((K,) * ndim)
    touched: NDArray[np.bool_] = spec_correlate(invalid.astype(np.float64), support, pbc) > 0
    return touched


def _union(a: Mask, b: Mask) -> Mask:
    """The union of the masks of the two arguments of a covariance (M-MASK, OP-35)."""
    if a is None or b is None:
        return None if a is None and b is None else (a if b is None else b)
    return a | b


def spec_border(
    history: tuple[int, ...],
    level: int,
    pbc: bool,
    K: int = KERNEL_SIZE,  # noqa: N803
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


def spec_valid_mean(a: Array, ndim: int, invalid: Mask = None, border: int = 0) -> Array:
    """The mean of ``a`` over its valid pixels, after the border crop: ``⟨a⟩_m``, over the last
    ``ndim`` axes. Implements M-NOT (the valid-pixel mean, M1) and M-MEAN, with the crop of
    M-BORDER.

    ``border`` pixels are removed at both ends of each axis; a border too large for the array
    (2β ≥ min N) is reduced to ⌊(min N - 1)/2⌋ (OP-13). The sum over the valid pixels is
    divided by their number; with no valid pixel the result is NaN (OP-3).
    """
    spatial = a.shape[a.ndim - ndim :]
    if border and 2 * border >= min(spatial):
        border = (min(spatial) - 1) // 2
    crop: tuple[Any, ...] = (..., *(slice(border, n - border) for n in spatial))
    kept = a[crop]
    valid = np.ones(kept.shape[kept.ndim - ndim :], dtype=bool)
    if invalid is not None:
        valid = ~invalid[crop]
    count = int(valid.sum())
    if count == 0:
        return np.full(kept.shape[: kept.ndim - ndim], np.nan, dtype=kept.dtype)
    axes = tuple(range(kept.ndim - ndim, kept.ndim))
    total: Array = np.where(valid, kept, 0.0).sum(axis=axes)
    return total / count


def spec_moments(x: Array, ndim: int, invalid: Mask = None) -> tuple[Array, Array]:
    """``mean`` and ``var`` of each map, ``(Nb, Nc)`` each. Implements M-MOM.

    ``mean = ⟨x⟩`` and ``var = ⟨x²⟩``: the second moment, not centred (OP-11).
    """
    return spec_valid_mean(x, ndim, invalid), spec_valid_mean(x * x, ndim, invalid)


def spec_standardize(x: Array, ndim: int, invalid: Mask = None) -> tuple[Array, Array, Array]:
    """The standardised field, with the mean μ and the standard deviation sigma removed from each
    map, ``(Nb, Nc)`` each. Implements M-STD.

    ``μ = ⟨x⟩``, ``sigma² = ⟨(x - μ)²⟩`` over the valid pixels, divided by their number, and
    ``x' = (x - μ) / sigma``.
    """
    expand = (..., *(None,) * ndim)
    mu = spec_valid_mean(x, ndim, invalid)
    centred = x - mu[expand]
    sigma = np.sqrt(spec_valid_mean(centred * centred, ndim, invalid))
    return centred / sigma[expand], mu, sigma


def spec_first_layer(
    x_level: Array,
    ndim: int,
    L: int,  # noqa: N803
    pbc: bool,
    K: int = KERNEL_SIZE,  # noqa: N803
) -> tuple[Array, Array]:
    """The first layer at one scale, from the field at that level, ``(Nb, Nc, *spatial)``:
    ``W = ψ_l ⋆ x^(j)`` and ``U = |W|``, ``(Nb, Nc, L, *spatial)`` each. Implements M-W1.
    """
    psi = _wavelets(ndim, L, K)
    w = np.stack([spec_correlate(x_level, psi[ell], pbc) for ell in range(L)], axis=2)
    return w, np.abs(w)


# ---------------------------------------------------------------------- scattering statistics


def _channel_pairs(cross_matrix: Any, n_channels: int) -> NDArray[np.bool_]:
    """The boolean ``(Nc, Nc)`` matrix of the pairs c1 ≤ c2 to compute (M-CHAN): by default the
    upper triangle; the entries below the diagonal are ignored."""
    if cross_matrix is None:
        return np.triu(np.ones((n_channels, n_channels), dtype=bool))
    matrix = np.triu(np.asarray(cross_matrix, dtype=bool))
    if matrix.shape != (n_channels, n_channels):
        raise ValueError(f"cross_matrix must have shape ({n_channels}, {n_channels}) (M-CHAN).")
    in_no_pair = [c for c in range(n_channels) if not (matrix[c].any() or matrix[:, c].any())]
    if in_no_pair:
        raise ValueError(f"channel(s) {in_no_pair} are in no pair of cross_matrix (M-CHAN).")
    return matrix


def spec_scattering(
    x: Array,
    ndim: int,
    J: int,  # noqa: N803
    L: int,  # noqa: N803
    pbc: bool,
    invalid: Mask = None,
    cross_matrix: Any = None,
    K: int = KERNEL_SIZE,  # noqa: N803
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
    psi = _wavelets(ndim, L, K)

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
            v[j, j3] = np.stack([spec_correlate(modulus, psi[l3], pbc) for l3 in range(L)], axis=3)
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
                border = spec_border((j,), j, pbc, K)
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
                            spec_border((j2, j3), j3, pbc, K), spec_border((j3,), j3, pbc, K)
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
                                spec_border((j1, j3), j3, pbc, K),
                                spec_border((j2, j3), j3, pbc, K),
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


# ---------------------------------------------------------------------- normalisation


def spec_reference(stats: Statistics, ref_batch_mean: bool = False) -> Statistics:
    """The normalisation reference of statistics computed without normalisation: ``var``,
    ``(Nr, Nc)``, and ``sigma``, ``(Nr, Nc, J, L)``, with ``sigma_c(j, l) = √S2[c, c, j, l]``.
    Implements M-NORM (the reference of ``norm="self"``).

    With ``ref_batch_mean`` the variance and the auto S2 are first averaged over the batch, and
    Nr = 1; otherwise Nr = Nb. A channel without auto statistics has sigma = NaN (OP-30).

    With the spectrum (a family ``PS`` in ``stats``), the reference also holds ``sigma_PS``,
    ``(Nr, Nc, n_b)``, with ``sigma_PS_c(b) = √PS[c, c, b]``, of the batch mean of the auto
    spectra with ``ref_batch_mean`` (M-PS-2D, "In the scattering operator"); a channel without
    auto spectrum has sigma_PS = NaN (OP-30).
    """
    var, s2 = stats["var"], stats["S2"]
    nc = s2.shape[1]
    diagonal = np.stack([s2[:, c, c] for c in range(nc)], axis=1)
    if ref_batch_mean:
        var = var.mean(axis=0, keepdims=True)
        diagonal = diagonal.mean(axis=0, keepdims=True)
    reference = {"var": var, "sigma": np.sqrt(diagonal)}
    if "PS" in stats:
        ps = stats["PS"]
        auto = np.stack([ps[:, c, c] for c in range(nc)], axis=1)
        if ref_batch_mean:
            auto = auto.mean(axis=0, keepdims=True)
        reference["sigma_PS"] = np.sqrt(auto)
    return reference


def spec_normalize(stats: Statistics, reference: Statistics) -> Statistics:
    """The statistics divided by the reference, as in the table of M-NORM. Implements M-NORM.

    ``mean`` is not normalised; ``var`` is divided by ``var_ref``; S1 by
    √(sigma_{c1}(j, l) sigma_{c2}(j, l)); S2 by sigma_{c1}(j, l) sigma_{c2}(j, l); S3 by
    sigma_{c1}(j2, l2) sigma_{c2}(j3, l3); S4 by sigma_{c1}(j1, l1) sigma_{c2}(j2, l2); PS, when
    the statistics hold it, by sigma_PS_{c1}(b) sigma_PS_{c2}(b) (M-PS-2D). A reference of one
    batch element applies to every batch element.
    """
    normalized: Statistics = {}
    if "PS" in stats:
        if "sigma_PS" not in reference:
            raise ValueError(
                "the statistics hold the spectrum but the reference has no sigma_PS (M-NORM, "
                "M-PS-2D); pass the reference of statistics computed with compute_PS=True."
            )
        sigma_ps = reference["sigma_PS"]
        ps = np.array(stats["PS"])
        with np.errstate(invalid="ignore"):  # NaN where no pair or no sigma_PS exists
            for c1 in range(ps.shape[1]):
                for c2 in range(ps.shape[2]):
                    for b in range(ps.shape[3]):
                        divisor = sigma_ps[:, c1, b] * sigma_ps[:, c2, b]
                        ps[:, c1, c2, b] = ps[:, c1, c2, b] / divisor
        normalized["PS"] = ps
    sigma = reference["sigma"]
    nc, n_scales, n_orient = sigma.shape[1:]
    s1, s2, s3, s4 = (np.array(stats[k]) for k in SCATTERING)
    with np.errstate(invalid="ignore"):  # NaN where no coefficient or no sigma exists
        for c1 in range(nc):
            for c2 in range(nc):
                for j in range(n_scales):
                    for ell in range(n_orient):
                        divisor = sigma[:, c1, j, ell] * sigma[:, c2, j, ell]
                        s1[:, c1, c2, j, ell] = s1[:, c1, c2, j, ell] / np.sqrt(divisor)
                        s2[:, c1, c2, j, ell] = s2[:, c1, c2, j, ell] / divisor
                for j2, j3, l2, l3 in np.ndindex(n_scales, n_scales, n_orient, n_orient):
                    divisor = sigma[:, c1, j2, l2] * sigma[:, c2, j3, l3]
                    s3[:, c1, c2, j2, j3, l2, l3] = s3[:, c1, c2, j2, j3, l2, l3] / divisor
                for j1, j2, l1, l2 in np.ndindex(n_scales, n_scales, n_orient, n_orient):
                    divisor = sigma[:, c1, j1, l1] * sigma[:, c2, j2, l2]
                    # Every j3 and l3 at once: [b, j3, l3].
                    s4[:, c1, c2, j1, j2, :, l1, l2, :] = (
                        s4[:, c1, c2, j1, j2, :, l1, l2, :] / divisor[:, None, None]
                    )
    return {
        **stats,
        "var": stats["var"] / reference["var"],
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
        **normalized,
    }


# ---------------------------------------------------------------------- reductions


def spec_iso(stats: Statistics) -> Statistics:
    """The isotropic statistics. Implements M-RED-ISO.

    S1 and S2 are averaged over l. S3 becomes ``[.., δ]`` with
    ``(1/L) Σ_{l2} S3[.., l2, l2 + δ]``, and S4 ``[.., δ2, δ3]`` with
    ``(1/L) Σ_{l1} S4[.., l1, l1 + δ2, l1 + δ3]``, the orientations taken modulo L (OP-5).
    """
    s1, s2, s3, s4 = (stats[k] for k in SCATTERING)
    n_orient = s1.shape[-1]
    iso3 = np.zeros(s3.shape[:-1], dtype=s3.dtype)
    iso4 = np.zeros(s4.shape[:-1], dtype=s4.dtype)
    for delta in range(n_orient):
        for l2 in range(n_orient):
            iso3[..., delta] += s3[..., l2, (l2 + delta) % n_orient] / n_orient
    for d2 in range(n_orient):
        for d3 in range(n_orient):
            for l1 in range(n_orient):
                iso4[..., d2, d3] += (
                    s4[..., l1, (l1 + d2) % n_orient, (l1 + d3) % n_orient] / n_orient
                )
    return {**stats, "S1": s1.mean(axis=-1), "S2": s2.mean(axis=-1), "S3": iso3, "S4": iso4}


def spec_angular_fourier(
    stats: Statistics, harmonics: int | None = None, after_iso: bool = False
) -> Statistics:
    """The angular Fourier transform of S3 and S4. Implements M-RED-ANG.

    An orthonormal discrete Fourier transform, ``L^(-1/2) Σ_l x[l] exp(-2πi h l / L)``, along
    each orientation axis of S3 and S4 (their last 2 and 3 axes; 1 and 2 after ``iso``), keeping
    the harmonics h < ``harmonics``: all L by default (OP-17).
    """
    out = dict(stats)
    for family in ("S3", "S4"):
        x = stats[family]
        n_orient = x.shape[-1]
        kept = n_orient if harmonics is None else harmonics
        h, ell = np.arange(kept), np.arange(n_orient)
        basis = np.exp(-2j * np.pi * np.outer(h, ell) / n_orient) / math.sqrt(n_orient)  # [h, l]
        n_axes = N_ORIENTATION_AXES[family] - (1 if after_iso else 0)
        for axis in range(-n_axes, 0):
            # The sum over l along this axis; tensordot puts the new axis h last: move it back.
            x = np.moveaxis(np.tensordot(x, basis, axes=([axis], [1])), -1, axis)
        out[family] = x
    return out


def spec_scale_cosine(
    stats: Statistics, harmonics: int | None = None, dj: int | None = None
) -> Statistics:
    """The scale cosine transform of S3 and S4, after ``iso``. Implements M-RED-SCALE.

    The scales are reparametrised: S3 as ``(j1, a)`` reading the scales ``(j1, j1 + a)``, S4 as
    ``(j1, a, b)`` reading ``(j1, j1 + a, j1 + b)``, with 0 ≤ a ≤ b < ``dj`` (J by default). The
    missing entries are set to 0, the orthonormal DCT-II
    ``F[k, j] = alpha_k cos(kπ (j + 1/2) / J)`` is applied along j1, the missing entries are set
    back to NaN at their positions, and the harmonics k < ``harmonics`` (J by default) are kept.
    """
    out = dict(stats)
    for family in ("S3", "S4"):
        x = stats[family]  # (Nb, Nc, Nc, J, .., J, *orientations)
        n_scales = x.shape[3]
        width = n_scales if dj is None else dj
        kept = n_scales if harmonics is None else harmonics
        n_axes = N_SCALE_AXES[family]
        rest = x.shape[3 + n_axes :]
        reparam = np.full(
            (*x.shape[:3], n_scales, *(width,) * (n_axes - 1), *rest), np.nan, x.dtype
        )
        for j1 in range(n_scales):
            for b in range(min(width, n_scales - j1)):
                if family == "S3":
                    reparam[:, :, :, j1, b] = x[:, :, :, j1, j1 + b]
                else:
                    for a in range(b + 1):
                        reparam[:, :, :, j1, a, b] = x[:, :, :, j1, j1 + a, j1 + b]
        missing = np.isnan(reparam)
        filled = np.where(missing, 0.0, reparam)
        transformed = np.zeros_like(filled)
        for k in range(n_scales):
            alpha = math.sqrt((1.0 if k == 0 else 2.0) / n_scales)
            for j in range(n_scales):
                weight = alpha * math.cos(k * math.pi * (j + 0.5) / n_scales)
                transformed[:, :, :, k] += weight * filled[:, :, :, j]
        out[family] = np.where(missing, np.nan, transformed)[:, :, :, :kept]
    return out


# ---------------------------------------------------------------------- flattening


def spec_flatten(
    stats: Statistics,
    flatten_complex: bool = False,
    keep_batch_dim: bool = False,
    mean_along_batch: bool = False,
    mask_st: Any = None,
) -> Array:
    """The statistics as one vector. Implements M-FLAT.

    The families in the order ``mean``, ``var``, S1, S2, S3, S4, then PS when the statistics hold
    it (M-PS-2D), each in the row-major order of its array, without its NaN entries. With
    ``flatten_complex=False`` the vector is complex; with ``flatten_complex=True`` it is real:
    each complex coefficient gives its real and imaginary parts, except those of S1, S2 and PS,
    which give their real part only (OP-37), and
    ``mean`` and ``var`` give one value each. The batch axis is flattened with the rest, unless
    ``keep_batch_dim``, which gives ``(Nb, n)``; ``mean_along_batch`` first averages every
    family over the batch. ``mask_st`` is a boolean array over the vector: True keeps a
    coefficient.
    """
    rows = []
    for family in FAMILIES + (("PS",) if "PS" in stats else ()):
        x = np.asarray(stats[family])
        if mean_along_batch:
            x = x.mean(axis=0, keepdims=True)
        x = x.reshape(x.shape[0], -1)
        present = ~np.isnan(x)
        if keep_batch_dim:
            if not (present == present[0]).all():
                raise ValueError("keep_batch_dim: the NaN differ between batch elements (M-FLAT).")
            kept = x[:, present[0]]
        else:
            kept = x[present][None, :]
        if not flatten_complex:
            rows.append(kept.astype(np.complex128))
        elif family in ("mean", "var", "S1", "S2", "PS"):
            rows.append(kept.real.astype(np.float64))
        else:
            parts = np.stack([kept.real, kept.imag], axis=-1)
            rows.append(parts.reshape(kept.shape[0], -1))
    vector = np.concatenate(rows, axis=1)
    if mask_st is not None:
        keep = np.asarray(mask_st, dtype=bool)
        if keep.shape != (vector.shape[1],):
            raise ValueError(
                f"mask_st has shape {keep.shape}; its length must be the vector's, "
                f"{vector.shape[1]} (M-FLAT)."
            )
        vector = vector[:, keep]
    out: Array = vector if keep_batch_dim else vector.reshape(-1)
    return out


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
    standardize: bool = False,
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
    5. the power spectrum, if asked (``compute_PS``, 2D only; M-PS-2D): ``spectra.spec_spectra``
       on the field at full resolution, after the standardisation, with J_PS = J, the pairs of
       ``cross_matrix``, ``Jmin``, ``n_bins`` and ``spectrum_method`` (the automatic choice by
       default, from ``pbc`` and the NaN of the field), under ``"PS"``;
    6. normalisation (M-NORM): ``norm="self"`` by the statistics themselves, ``norm="ref"`` by
       ``ref``, the ``"reference"`` of other statistics, ``norm="none"`` not at all;
    7. the reductions asked for, in the order ``iso``, ``angular_fourier``, ``scale_cosine``
       (M-RED-ISO, M-RED-ANG, M-RED-SCALE);
    8. the flattening, if asked (M-FLAT), in its real form, under ``"vector"``.

    Returns the families, the ``"reference"`` used (``None`` with ``norm="none"``) and, with
    ``standardize``, ``"mean_pre_std"`` and ``"std_pre_std"``. J and L default to M-DEF and
    M-DEF-1D.

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
        out["PS"] = spectra.spec_spectra(
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
