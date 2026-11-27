"""The oracle of the 2D power spectrum: section 1.9 of ``docs/maths.md`` (M-PS-2D).

A second implementation of the mathematics, written to be read against ``docs/maths.md`` rather
than to be fast (Part I, §4.2): float64, NumPy and the standard library only, nothing from the
rest of ``estel``, nothing shared with the library or with STL. The loops over bins, radial
frequencies, radii and channel pairs are explicit, as the formulas are written; only the sums
over pixels and Fourier modes are NumPy operations.

**Conventions.**

- Every frequency is in mode units (§1.9): a Fourier mode is the vector of integers
  **k** = (k₁, k₂), its radial frequency the integer k = round(|**k**|), and the radial
  frequencies run from 0 to N_max = ⌊min(N₁, N₂)/2⌋. STL's frequencies in cycles per pixel are
  k / (2 N_max).
- The transforms use the **full complex plane** (``np.fft.fft2``, ``np.fft.ifft2``, orthonormal),
  not the half plane of the real FFT that STL and the library use: for a real field the full
  plane holds every mode of the half plane with its weight w_**k** (2, or 1 on the zero and Nyquist
  columns), so the sums are those of §1.9.
- Arrays are ``(Nb, Nc, N1, N2)``, real, NaN at the missing pixels; the spectrum is
  ``(Nb, Nc, Nc, n_b)``, complex with a null imaginary part, NaN where no pair is computed
  (c₁ > c₂, or a pair that ``cross_matrix`` leaves out; M-CHAN).
- Sizes are even (OP-20), ``pbc`` a boolean (OP-4).

The spectrum within the scattering statistics (its reference, its normalisation and its place in
the flattened vectors) is in ``planar`` (``spec_apply``, ``spec_reference``, ``spec_normalize``,
``spec_flatten``), which calls ``spec_spectra_2d``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

METHODS_2D = ("fft", "pbc_mask", "nan_mask", "pbc_nan_mask")
"""The four estimators of §1.9, in the order of the fixtures of T4.3."""
ENERGY_FRACTION_2D = 0.60
"""The fraction of the energy of g_b within the border β_b (§1.9; OP-21: not an option)."""


# ---------------------------------------------------------------------- the Bessel function J₀


def spec_bessel_j0(x: Any) -> Array:
    """The Bessel function of the first kind of order 0, element by element, in float64.
    Implements the Bessel function J₀ that the border of M-PS-2D needs.

    From Bessel's integral ``J₀(x) = (1/2π) ∫₀^{2π} cos(x sin θ) dθ``, by the trapezoidal rule on
    M equally spaced points, which is exact for a periodic integrand up to its aliases:
    ``(1/M) Σ_m cos(x sin(2πm/M)) = J₀(x) + 2 Σ_{q≥1} J_{qM}(x)`` for even M. With
    M ≥ |x| + 20 |x|^(1/3) + 32, the first alias, 2 J_M(x), is far below 1e-15 (J_nu(x) decays
    faster than exponentially once nu - x exceeds a few x^(1/3)). The rounding of each term,
    that of θ_m = 2πm/M included, averages out over the M terms. Measured against references
    summed with 40 significant digits: an absolute error of at most 8.1e-15 on a grid of 20 001
    points over [0, 256 π], and of 7.0e-15 on every argument π rho k / N_max of the borders at
    N = 512. The 1e-12 that §1.9 requires on [0, π N_max] holds with a margin of about 100.
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    largest = float(x.max()) if x.size else 0.0
    n_points = 2 * math.ceil((largest + 20.0 * largest ** (1.0 / 3.0) + 32.0) / 2.0)
    total = np.zeros_like(x)
    for m in range(n_points):
        total = total + np.cos(x * math.sin(2.0 * math.pi * m / n_points))
    out: Array = total / n_points
    return out


# ---------------------------------------------------------------------- frequencies and bins


def _check_shape_2d(spatial_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(spatial_shape) != 2:
        raise ValueError(
            f"the spatial shape {spatial_shape} is not 2D: the power spectrum is defined on 2D "
            "data only (M-PS-2D); pass (N1, N2)."
        )
    n1, n2 = (int(n) for n in spatial_shape)
    if n1 % 2 or n2 % 2:
        raise ValueError(
            f"the sizes ({n1}, {n2}) are not even: the power spectrum requires even sizes "
            "(M-PS-2D, OP-20); crop or pad the data to even sizes."
        )
    return n1, n2


def _radial_index_2d(n1: int, n2: int) -> NDArray[np.int64]:
    """``(N1, N2)``: the radial frequency k = round(|**k**|) of each mode of the full plane, in the
    layout of ``np.fft.fft2`` (k₁ = 0, 1, …, N₁/2 - 1, -N₁/2, …, -1, and likewise k₂). No tie
    occurs: a sum of two squares of integers is never the square of a half-integer."""
    k1 = np.rint(np.fft.fftfreq(n1) * n1)
    k2 = np.rint(np.fft.fftfreq(n2) * n2)
    out: NDArray[np.int64] = np.rint(np.hypot(k1[:, None], k2[None, :])).astype(np.int64)
    return out


def spec_spectrum_bins_2d(
    spatial_shape: tuple[int, ...],
    J: int | None = None,  # noqa: N803 (maths.md notation)
    Jmin: int = 0,  # noqa: N803
    n_bins: int | None = None,
) -> dict[str, Any]:
    """The bins of the spectrum of data of ``spatial_shape`` = (N1, N2). Implements M-PS-2D
    ("Frequencies and bins", and the border β_b of ``pbc_mask``).

    ``J`` is the J of the operator, by default ⌊log₂ N_max⌋ - 1; ``Jmin`` is J_min, with
    0 ≤ J_min ≤ J - 2; ``n_bins`` is n_b, by default max(4, round(3 (J - 1 - J_min))). In mode
    units: k_min = N_max / (√2 2^(J - 1)), k_max = N_max / (√2 2^J_min); the edges e_0 … e_{n_b}
    evenly spaced in ln k from ln k_min to ln k_max; ln c_b = (e_b + e_{b+1}) / 2 and
    Δ = e_{b+1} - e_b; W_b(k) = exp(-(ln k - ln c_b)² / (2Δ²)) for k ≥ 1, W_b(0) = 0;
    N_k the number of modes of the full plane with round(|**k**|) = k; Z_b = Σ_k N_k W_b(k).

    Returns ``J``, ``Jmin``, ``n_bins``, ``N_max``, ``k_min``, ``k_max``, ``radial_k`` (the
    integers 0 … N_max), ``n_modes`` (N_k), ``centers`` (c_b), ``windows`` (W_b(k),
    ``(n_b, N_max + 1)``), ``norm`` (Z_b) and ``borders`` (β_b in pixels, ``spec_borders_2d``).
    """
    n1, n2 = _check_shape_2d(spatial_shape)
    n_max = min(n1, n2) // 2
    j_ps = math.floor(math.log2(n_max)) - 1 if J is None else int(J)
    if not 0 <= Jmin <= j_ps - 2:
        raise ValueError(
            f"J={j_ps}, J_min={Jmin}: the spectrum requires 0 <= J_min <= J - 2 "
            "(M-PS-2D, OP-93); pass a smaller Jmin, a larger J, or use a larger image."
        )
    k_min = n_max / (math.sqrt(2.0) * 2.0 ** (j_ps - 1))
    k_max = n_max / (math.sqrt(2.0) * 2.0**Jmin)
    # log₂(k_max / k_min) = J - 1 - J_min exactly (§1.9, "Number of bins").
    n_b = max(4, round(3 * (j_ps - 1 - Jmin))) if n_bins is None else int(n_bins)
    if n_b < 1:
        raise ValueError(f"n_bins={n_bins!r}: pass a positive number of bins (M-PS-2D).")

    edges = np.linspace(math.log(k_min), math.log(k_max), n_b + 1)
    radial_k = np.arange(n_max + 1)
    index = _radial_index_2d(n1, n2)
    n_modes = np.array([np.count_nonzero(index == k) for k in radial_k], dtype=np.float64)

    centers = np.zeros(n_b)
    windows = np.zeros((n_b, n_max + 1))
    norm = np.zeros(n_b)
    for b in range(n_b):
        log_center = (edges[b] + edges[b + 1]) / 2.0
        delta = edges[b + 1] - edges[b]
        centers[b] = math.exp(log_center)
        for k in range(1, n_max + 1):
            windows[b, k] = math.exp(-((math.log(k) - log_center) ** 2) / (2.0 * delta**2))
        norm[b] = sum(n_modes[k] * windows[b, k] for k in range(n_max + 1))

    return {
        "J": j_ps,
        "Jmin": Jmin,
        "n_bins": n_b,
        "N_max": n_max,
        "k_min": k_min,
        "k_max": k_max,
        "radial_k": radial_k,
        "n_modes": n_modes,
        "centers": centers,
        "windows": windows,
        "norm": norm,
        "borders": spec_borders_2d(windows),
    }


def spec_borders_2d(windows: Array) -> NDArray[np.int64]:
    """β_b, the border in pixels of each bin, from its window W_b(k), ``(n_b, N_max + 1)``.
    Implements M-PS-2D (the border of ``pbc_mask``).

    g_b(rho) = π / (2 N_max²) Σ_{k=0}^{N_max} W_b(k) k J₀(π rho k / N_max) at the integer
    radii rho = 0 … N_max; Q_b(rho) = ∫₀^rho 2π rho' g_b(rho')² drho' by the cumulative
    trapezoidal rule on those radii, Q_b(0) = 0; β_b is the smallest rho with
    Q_b(rho) ≥ 0.60 Q_b(N_max), and 0 when Q_b(N_max) = 0. The window is W_b, not the √W_b of
    the band maps (OP-96).
    """
    n_bins, n_k = windows.shape
    n_max = n_k - 1
    radii = np.arange(n_max + 1, dtype=np.float64)
    ks = np.arange(n_max + 1, dtype=np.float64)
    bessel = spec_bessel_j0(math.pi * radii[:, None] * ks[None, :] / n_max)  # [rho, k]
    borders = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        g = np.zeros(n_max + 1)
        for rho in range(n_max + 1):
            g[rho] = (
                math.pi
                / (2.0 * n_max**2)
                * sum(windows[b, k] * k * bessel[rho, k] for k in range(n_max + 1))
            )
        energy = 2.0 * math.pi * radii * g**2
        cumulative = np.zeros(n_max + 1)
        for rho in range(1, n_max + 1):
            cumulative[rho] = cumulative[rho - 1] + 0.5 * (energy[rho - 1] + energy[rho])
        if cumulative[n_max] == 0.0:
            continue
        borders[b] = next(
            rho
            for rho in range(n_max + 1)
            if cumulative[rho] >= ENERGY_FRACTION_2D * cumulative[n_max]
        )
    return borders


# ---------------------------------------------------------------------- choice of estimator


def spec_spectrum_method_2d(has_nan: bool, pbc: bool, method: str = "auto") -> str:
    """The estimator of the spectrum. Implements M-PS-2D ("Choice of estimator").

    With ``method="auto"``: ``pbc_nan_mask`` for data with NaN and ``pbc=False``, ``nan_mask``
    for data with NaN otherwise, ``pbc_mask`` for data without NaN and ``pbc=False``, ``fft``
    otherwise. Any other ``method`` is one of the four estimators, and is returned as given.
    """
    if not isinstance(pbc, bool | np.bool_):
        raise ValueError(f"pbc={pbc!r}: pass True or False (M-DATA-2D, OP-4).")
    if method == "auto":
        if has_nan:
            return "nan_mask" if pbc else "pbc_nan_mask"
        return "fft" if pbc else "pbc_mask"
    if method not in METHODS_2D:
        raise ValueError(f'method={method!r}: pass "auto" or one of {METHODS_2D} (M-PS-2D).')
    return method


# ---------------------------------------------------------------------- estimators


def _promote_2d(array: Any) -> Array:
    x = np.asarray(array)
    if np.iscomplexobj(x):
        raise ValueError("the data must be real (M-DATA-2D); pass a real array.")
    if not 2 <= x.ndim <= 4:
        raise ValueError(
            f"the array has {x.ndim} axes; pass (N1, N2), (Nc, N1, N2) or (Nb, Nc, N1, N2)."
        )
    out: Array = np.array(x, dtype=np.float64).reshape((1,) * (4 - x.ndim) + x.shape)
    return out


def _pairs(cross_matrix: Any, n_channels: int) -> NDArray[np.bool_]:
    """The boolean ``(Nc, Nc)`` matrix of the pairs c1 ≤ c2 (M-CHAN, OP-94): by default the upper
    triangle, every pair c1 ≤ c2; the entries below the diagonal are ignored."""
    if cross_matrix is None:
        return np.triu(np.ones((n_channels, n_channels), dtype=bool))
    matrix = np.triu(np.asarray(cross_matrix, dtype=bool))
    if matrix.shape != (n_channels, n_channels):
        raise ValueError(f"cross_matrix must have shape ({n_channels}, {n_channels}) (M-CHAN).")
    return matrix


def _hard_mask_2d(n1: int, n2: int, border: int) -> Array:
    """M_b, the hard border mask: 1 where β ≤ n₁ < N₁ - β and β ≤ n₂ < N₂ - β, 0 elsewhere."""
    rows = (np.arange(n1) >= border) & (np.arange(n1) < n1 - border)
    cols = (np.arange(n2) >= border) & (np.arange(n2) < n2 - border)
    out: Array = (rows[:, None] & cols[None, :]).astype(np.float64)
    return out


def spec_spectra_2d(
    array: Any,
    pbc: bool,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    Jmin: int = 0,  # noqa: N803
    n_bins: int | None = None,
    method: str = "auto",
    cross_matrix: Any = None,
) -> dict[str, Any]:
    """The power spectrum of ``array``, ``(Nb, Nc, Nc, n_b)``. Implements M-PS-2D.

    ``array`` is ``(N1, N2)``, ``(Nc, N1, N2)`` or ``(Nb, Nc, N1, N2)``, real, NaN at its missing
    pixels; ``J``, ``Jmin`` and ``n_bins`` set the bins (``spec_spectrum_bins_2d``), ``method`` the
    estimator (``spec_spectrum_method_2d``), ``cross_matrix`` the pairs c1 ≤ c2 (every pair by
    default, OP-94). With x̂ = FFT2(x) / √(N1 N2) on the full plane and N_pix = N1 N2:

    - ``fft``: P(k) = (1/N_k) Σ_{round(|**k**|)=k} Re(x̂_{c1}(**k**) conj(x̂_{c2}(**k**))), then
      PS_b = Σ_k N_k W_b(k) P(k) / Z_b;
    - the band maps f_b = IFFT2(x̂ √W_b(round(|**k**|))), √W_b = 0 beyond N_max, orthonormal; their
      real part (the imaginary part is rounding);
    - ``pbc_mask``: PS_b = N_pix / Σ_n M_b[n] · (1/Z_b) Σ_n M_b[n] f_{b,c1}[n] f_{b,c2}[n], M_b
      the hard mask of border β_b;
    - ``nan_mask``: each map has its missing pixels filled with the mean of its valid pixels,
      then is band-filtered; O is the pixels valid in every batch element and channel, and
      PS_b = N_pix / N_obs · (1/Z_b) Σ_{n∈O} f_{b,c1}[n] f_{b,c2}[n];
    - ``pbc_nan_mask``: as ``pbc_mask``, with M_b O in place of M_b, after the same filling.

    Data with missing pixels given to ``fft`` or ``pbc_mask``, no observed pixel for ``nan_mask``
    or ``pbc_nan_mask``, and a bin whose M_b O is null everywhere for ``pbc_nan_mask`` raise a
    ``ValueError``.

    Only the real part of the cross spectra is computed (OP-19). Returns ``PS``, ``method`` (the
    estimator used) and ``bins`` (the result of ``spec_spectrum_bins_2d``).
    """
    x = _promote_2d(array)
    nb, nc, n1, n2 = x.shape
    bins = spec_spectrum_bins_2d((n1, n2), J, Jmin, n_bins)
    missing = np.isnan(x)
    chosen = spec_spectrum_method_2d(bool(missing.any()), pbc, method)
    if chosen in ("fft", "pbc_mask") and missing.any():
        raise ValueError(
            f"the data have missing pixels, which the estimator {chosen!r} does not accept "
            '(M-PS-2D); use "nan_mask" or "pbc_nan_mask", or "auto".'
        )
    pairs = _pairs(cross_matrix, nc)
    n_max, n_b = bins["N_max"], bins["n_bins"]
    windows, norm, n_modes = bins["windows"], bins["norm"], bins["n_modes"]
    index = _radial_index_2d(n1, n2)
    n_pix = n1 * n2

    # The observed pixels O, valid in every batch element and channel; then each map with its
    # missing pixels filled with the mean of its valid pixels (nan_mask and pbc_nan_mask). With
    # some observed pixel, every map has a valid pixel.
    observed: Array = np.asarray(~missing.any(axis=(0, 1)), dtype=np.float64)
    if chosen in ("nan_mask", "pbc_nan_mask") and not observed.any():
        raise ValueError(
            "no pixel is valid in every batch element and channel: the estimator "
            f"{chosen!r} has nothing to average (M-PS-2D); give data with common valid pixels."
        )
    filled = np.array(x)
    for batch in range(nb):
        for c in range(nc):
            valid = ~missing[batch, c]
            if (~valid).any():
                filled[batch, c] = np.where(valid, x[batch, c], x[batch, c][valid].mean())

    x_hat = np.fft.fft2(filled, norm="ortho")  # (Nb, Nc, N1, N2), complex
    ps = np.full((nb, nc, nc, n_b), np.nan, dtype=np.complex128)

    if chosen == "fft":
        for c1 in range(nc):
            for c2 in range(c1, nc):
                if not pairs[c1, c2]:
                    continue
                product = (x_hat[:, c1] * np.conj(x_hat[:, c2])).real  # (Nb, N1, N2)
                radial = np.zeros((nb, n_max + 1))
                for k in range(n_max + 1):
                    radial[:, k] = product[:, index == k].sum(axis=-1) / n_modes[k]
                for b in range(n_b):
                    total = sum(n_modes[k] * windows[b, k] * radial[:, k] for k in range(n_max + 1))
                    ps[:, c1, c2, b] = total / norm[b]
        return {"PS": ps, "method": chosen, "bins": bins}

    for b in range(n_b):
        root = np.zeros((n1, n2))
        for k in range(n_max + 1):
            root = np.where(index == k, math.sqrt(windows[b, k]), root)
        band = np.fft.ifft2(x_hat * root, norm="ortho").real  # f_b, (Nb, Nc, N1, N2)
        weight: Array
        if chosen == "pbc_mask":
            weight = _hard_mask_2d(n1, n2, int(bins["borders"][b]))
        elif chosen == "nan_mask":
            weight = observed
        else:  # pbc_nan_mask
            weight = _hard_mask_2d(n1, n2, int(bins["borders"][b])) * observed
        if chosen == "pbc_nan_mask" and not weight.any():
            raise ValueError(
                f"bin {b}: no observed pixel lies inside its border of {bins['borders'][b]} "
                f"pixels, so the estimator {chosen!r} has nothing to average (M-PS-2D); use "
                "larger data or fewer missing pixels."
            )
        for c1 in range(nc):
            for c2 in range(c1, nc):
                if not pairs[c1, c2]:
                    continue
                total = (weight * band[:, c1] * band[:, c2]).sum(axis=(-2, -1))
                ps[:, c1, c2, b] = n_pix / weight.sum() * total / norm[b]
    return {"PS": ps, "method": chosen, "bins": bins}
