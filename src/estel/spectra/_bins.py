"""The bins of the 2D power spectrum, built once in NumPy with the plan (maths.md §1.9,
"Frequencies and bins" and the border of ``pbc_mask``; Part II, §12.8; task T4.1).

:func:`log_gaussian_bins_2d` gives, for data of an even spatial shape (N₁, N₂) and the range J,
J_min of the spectrum (J the J of the operator), everything static the estimators need: in mode
units, the radial frequency round(|(k₁, k₂)|) of every Fourier mode (k₁, k₂) of the half plane of
the real FFT and its weight w(k₁, k₂) (2, or 1 on the zero and Nyquist columns), the number N_k of
modes of each radial frequency k = 0 … N_max, the log-Gaussian windows W_b(k), their normalisations
Z_b = Σ_k N_k W_b(k), the centres c_b, and the border β_b of each bin in pixels (the Hankel
transform of W_b, 60 % of its energy within rho ≤ N_max; OP-21, OP-96). Every array is float64 and
read-only, whatever the dtype of the data: the operator casts what it needs once, when it builds its
filters.

**The Bessel function J₀** (:func:`bessel_j0`). NumPy has none and SciPy is not a dependency
of ESTEL (decision on #148): ESTEL computes it here, in float64, to an absolute error of at most
1e-12 on [0, π N_max] (maths.md §1.9), with nothing shared with the oracle (decision on #149).
Three regions: the power series for |x| ≤ 8, Miller's backward recurrence for 8 < |x| < 25,
and Hankel's asymptotic expansion for |x| ≥ 25. Its absolute error is at most about 1.4e-14,
near x = 8, where the cancellation of the series is largest, and below 3e-16 beyond x = 8
(measured against the exact series).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ENERGY_FRACTION_2D = 0.60
"""The fraction of the energy of the kernel of a bin within its border (maths.md §1.9; OP-21:
fixed, not an option)."""

_SERIES_LIMIT = 8.0
"""Below this |x|, J₀ is its power series: the largest term is about 114 at x = 8, so the
cancellation costs up to about 1.5e-14."""
_ASYMPTOTIC_LIMIT = 25.0
"""From this |x| on, J₀ is Hankel's asymptotic expansion, whose 48 terms decrease below 1e-17 at
x = 25 (they would grow again from about the 2x-th)."""
_MILLER_START = 80
"""The order at which Miller's recurrence starts for 8 < |x| < 25: J_80(x) is at most 1.2e-32
there (at x = 25), so the truncation costs nothing measurable."""
_SERIES_TERMS = 40
_ASYMPTOTIC_TERMS = 24


def bessel_j0(x: Any) -> NDArray[np.float64]:
    """The Bessel function of the first kind of order 0, element by element, in float64
    (maths.md §1.9: absolute error at most 1e-12 on [0, π N_max]).

    * |x| ≤ 8: the series Σ_m (-x²/4)^m / (m!)², 40 terms.
    * 8 < |x| < 25: Miller's backward recurrence J_{n-1} = (2n/x) J_n - J_{n+1} from n = 80,
      normalised by 1 = J_0 + 2 Σ_{k≥1} J_{2k}.
    * |x| ≥ 25: J₀(x) = √(2/(πx)) (P(x) cos χ - Q(x) sin χ), χ = x - π/4, with Hankel's
      series P and Q, 24 terms each.
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(x)
    small = x <= _SERIES_LIMIT
    large = x >= _ASYMPTOTIC_LIMIT
    middle = ~(small | large)
    out = np.where(small, _series(np.where(small, x, 0.0)), out)
    out = np.where(middle, _miller(np.where(middle, x, 10.0)), out)
    out = np.where(large, _asymptotic(np.where(large, x, 30.0)), out)
    result: NDArray[np.float64] = out
    return result


def _series(x: NDArray[np.float64]) -> NDArray[np.float64]:
    q = -(x * x) / 4.0
    term = np.ones_like(x)
    total = np.ones_like(x)
    for m in range(1, _SERIES_TERMS):
        term = term * q / float(m * m)
        total = total + term
    return total


def _miller(x: NDArray[np.float64]) -> NDArray[np.float64]:
    following = np.zeros_like(x)  # J_{n+1}, unnormalised
    current = np.full_like(x, 1e-30)  # J_n
    even = np.zeros_like(x)  # Σ J_{2k}, k ≥ 1
    for n in range(_MILLER_START, 0, -1):
        previous = (2.0 * n / x) * current - following  # J_{n-1}
        following, current = current, previous
        if (n - 1) % 2 == 0 and n - 1 > 0:
            even = even + current
    result: NDArray[np.float64] = current / (current + 2.0 * even)
    return result


def _asymptotic(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # With a_k = Π_{i=1}^{k} (2i - 1)² / (k! 8^k), Hankel's coefficients of order 0 are (-1)^k a_k:
    # P = Σ (-1)^m a_{2m} / x^{2m} and Q = -Σ (-1)^m a_{2m+1} / x^{2m+1} (Q ≈ -1/(8x)).
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    a = 1.0
    power = np.ones_like(x)
    for k in range(2 * _ASYMPTOTIC_TERMS):
        if k > 0:
            a = a * (2 * k - 1) ** 2 / (k * 8.0)
            power = power / x
        sign = -1.0 if (k // 2) % 2 else 1.0
        if k % 2 == 0:
            p = p + sign * a * power
        else:
            q = q - sign * a * power
    cos, sin = np.cos(x), np.sin(x)
    root = math.sqrt(0.5)
    cos_chi = root * (cos + sin)  # cos(x - π/4)
    sin_chi = root * (sin - cos)  # sin(x - π/4)
    result: NDArray[np.float64] = np.sqrt(2.0 / (math.pi * x)) * (p * cos_chi - q * sin_chi)
    return result


# ---------------------------------------------------------------------- the bins


@dataclass(frozen=True, eq=False)
class SpectrumBins2D:
    """The static tables of the spectrum of one spatial shape and range (maths.md §1.9), in mode
    units, float64, read-only. Built by :func:`log_gaussian_bins_2d`, cached."""

    spatial_shape: tuple[int, int]
    """(N₁, N₂), even."""
    J: int
    """J, the J of the operator."""
    Jmin: int
    """J_min."""
    n_bins: int
    """n_b."""
    n_max: int
    """N_max = ⌊min(N₁, N₂)/2⌋."""
    k_min: float
    k_max: float
    radial: NDArray[np.int64]
    """``(N₁, N₂/2 + 1)``: round(|(k₁, k₂)|) of each mode of the half plane of ``rfft2``, -1 beyond
    N_max."""
    mode_weights: NDArray[np.float64]
    """``(N₁, N₂/2 + 1)``: w(k₁, k₂), 2, or 1 on the zero and Nyquist columns, 0 beyond N_max."""
    n_modes: NDArray[np.float64]
    """``(N_max + 1,)``: N_k, the number of modes of the full plane with round(|(k₁, k₂)|) = k."""
    centers: NDArray[np.float64]
    """``(n_b,)``: c_b, in mode units."""
    windows: NDArray[np.float64]
    """``(n_b, N_max + 1)``: W_b(k), W_b(0) = 0."""
    norm: NDArray[np.float64]
    """``(n_b,)``: Z_b = Σ_k N_k W_b(k)."""
    borders: NDArray[np.int64]
    """``(n_b,)``: β_b, the border of each bin in pixels (``flexible_crop``)."""

    @property
    def centers_cycles(self) -> NDArray[np.float64]:
        """c_b / (2 N_max): the centres in cycles per pixel, as STL's ``bin_centers``."""
        found: NDArray[np.float64] = self.centers / (2.0 * self.n_max)
        return found

    def mode_windows(self) -> NDArray[np.float64]:
        """``(n_b, N₁, N₂/2 + 1)``: W_b(round(|(k₁, k₂)|)) at each mode of the half plane, 0 beyond
        N_max."""
        inside = self.radial >= 0
        index = np.where(inside, self.radial, 0)
        found: NDArray[np.float64] = np.where(inside[None], self.windows[:, index], 0.0)
        return found

    def hard_masks(self) -> NDArray[np.float64]:
        """``(n_b, N₁, N₂)``: the hard border mask M_b of each bin, 1 where β_b ≤ n₁ < N₁ - β_b
        and β_b ≤ n₂ < N₂ - β_b, 0 elsewhere (maths.md §1.9; the smooth mask is not
        transferred)."""
        n1, n2 = self.spatial_shape
        rows, cols = np.arange(n1), np.arange(n2)
        masks = [
            ((rows >= b) & (rows < n1 - b))[:, None] & ((cols >= b) & (cols < n2 - b))[None, :]
            for b in self.borders.tolist()
        ]
        return np.stack(masks).astype(np.float64)


def _read_only(array: NDArray[Any]) -> NDArray[Any]:
    array.flags.writeable = False
    return array


@functools.lru_cache(maxsize=32)
def log_gaussian_bins_2d(
    spatial_shape: tuple[int, int],
    J: int,  # noqa: N803 (maths.md notation)
    Jmin: int,  # noqa: N803
    n_bins: int,
) -> SpectrumBins2D:
    """The :class:`SpectrumBins2D` of data of the even ``spatial_shape`` with the range ``J``
    (the J of the operator) and ``Jmin`` (J_min) and ``n_bins`` bins (maths.md §1.9; STL
    ``_build_log_gaussian_bin_masks``, ``estimate_pbc_border_crop``). The arguments were checked
    by the plan (``spectra._estimators.build_spectrum_plan_2d``)."""
    n1, n2 = (int(n) for n in spatial_shape)
    n_max = min(n1, n2) // 2
    k_min = n_max / (math.sqrt(2.0) * 2.0 ** (J - 1))
    k_max = n_max / (math.sqrt(2.0) * 2.0**Jmin)

    # The half plane of rfft2: k₁ in the order of the FFT, k₂ = 0 … N₂/2.
    k1 = np.concatenate([np.arange(0, (n1 + 1) // 2), np.arange(-(n1 // 2), 0)]).astype(float)
    k2 = np.arange(n2 // 2 + 1, dtype=float)
    rounded = np.rint(np.sqrt(k1[:, None] ** 2 + k2[None, :] ** 2)).astype(np.int64)
    inside = rounded <= n_max
    radial = np.where(inside, rounded, -1)
    columns = np.full(k2.shape[0], 2.0)
    columns[0] = 1.0
    if n2 % 2 == 0:
        columns[-1] = 1.0
    weights = np.where(inside, columns[None, :], 0.0)
    n_modes = np.bincount(radial[inside], weights=weights[inside], minlength=n_max + 1).astype(
        np.float64
    )

    edges = np.linspace(math.log(k_min), math.log(k_max), n_bins + 1)
    log_centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    ks = np.arange(n_max + 1, dtype=np.float64)
    log_k = np.log(np.where(ks > 0, ks, 1.0))
    windows = np.exp(-((log_k[None, :] - log_centers[:, None]) ** 2) / (2.0 * widths[:, None] ** 2))
    windows[:, 0] = 0.0
    norm = windows @ n_modes

    return SpectrumBins2D(
        spatial_shape=(n1, n2),
        J=int(J),
        Jmin=int(Jmin),
        n_bins=int(n_bins),
        n_max=n_max,
        k_min=k_min,
        k_max=k_max,
        radial=_read_only(radial),
        mode_weights=_read_only(weights),
        n_modes=_read_only(n_modes),
        centers=_read_only(np.exp(log_centers)),
        windows=_read_only(windows),
        norm=_read_only(norm),
        borders=_read_only(_borders_2d(windows, n_max)),
    )


def _borders_2d(windows: NDArray[np.float64], n_max: int) -> NDArray[np.int64]:
    """β_b (maths.md §1.9): g_b(rho) = π / (2 N_max²) Σ_k W_b(k) k J₀(π rho k / N_max) at the
    integer radii rho = 0 … N_max, Q_b the cumulative trapezoidal rule of 2π rho g_b(rho)², and
    β_b the smallest rho with Q_b(rho) ≥ 0.60 Q_b(N_max); 0 when Q_b(N_max) = 0. From W_b, not
    √W_b (OP-96)."""
    grid = np.arange(n_max + 1, dtype=np.float64)
    kernel = bessel_j0(math.pi * np.outer(grid, grid) / n_max)  # [rho, k]
    g = (math.pi / (2.0 * n_max**2)) * (windows * grid[None, :]) @ kernel.T  # (n_b, rho)
    energy = 2.0 * math.pi * grid[None, :] * g**2
    steps = 0.5 * (energy[:, 1:] + energy[:, :-1])
    cumulative = np.concatenate([np.zeros((windows.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1)
    total = cumulative[:, -1:]
    reached = cumulative >= ENERGY_FRACTION_2D * total
    found = np.where(total[:, 0] > 0.0, np.argmax(reached, axis=1), 0)
    return found.astype(np.int64)
