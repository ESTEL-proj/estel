"""Planar wavelet banks and the anti-aliasing kernel H (Part II, §12.5; task T2.2).

Built once, in NumPy float64, when an operator is built; never traced, never differentiated (tag
N of Part II, §12). The operator casts them to its dtype and backend.

* :func:`bump_steerable_2d`: the 2D bump-steerable kernels of maths.md M-WAV-2D, ported from
  STL `master` ``8eaf769`` (``WaveletOperator2D_FFT_torch.bump_steerable_bank`` and
  ``bump_steerable_2d``, then
  ``WaveletOperator2Dkernel_torch._build_wavelet_kernel_from_ifft_crop``).
* :func:`bump_steerable_1d`: the 1D bump-steerable kernel of maths.md M-WAV-1D, ported from STL
  `2D+1D` ``8f51143`` (``WaveletOperator1Dkernel_torch.bump_steerable_1d``,
  ``bump_steerable_bank`` and ``_build_wavelet_kernel_from_ifft_crop``), with its conjugate for
  L = 2.
* :func:`smoothing_kernel_H`: the anti-aliasing kernel of maths.md M-H and M-H-1D
  (``_get_smooth_kernel`` of both classes).

STL's kernels are reproduced as they are, problems included: the 2D width factor tuned for L = 4
only (OP-9), the crop without renormalisation (OP-10), the 1D bump centred on the Nyquist
frequency (OP-28). They do not satisfy the Littlewood-Paley condition exactly, and 1.0.0 does
not test it; a Littlewood-Paley requirement, tested in CI with the low-pass filter included,
comes with the wavelet optimisation (LF22, LF6). STL builds the Fourier-space bank for every
scale but uses only its first: coarser scales come from downsampling the data (M-WAV-2D). Only
the first scale is built here. STL builds its kernels at torch's default dtype; ESTEL builds them
in float64 (OP-31). ``tests/parity/test_kernels.py`` checks the equality with STL.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

FOURIER_GRID = 256
"""N, the side of the Fourier grid the kernels are built on, independent of the data size
(M-WAV-2D, M-WAV-1D)."""

KERNEL_SIZE = 5
"""K, STL's default kernel size: 5 x 5 pixels in 2D, 5 taps in 1D (M-DEF, M-DEF-1D)."""

WIDTH_FACTOR_2D = 2.5
"""c, the width factor of the 2D radial bump (M-WAV-2D, OP-9)."""

_BUMP_EPS = 1e-12
"""STL clamps the denominator of the bump at this value (``clamp_min(eps)``)."""

# STL's literal 6 x 6 anti-aliasing kernel, "cropped raised cosine, bandwidth .71, beta .35"
# (M-H). As written in STL it sums to 1.00000004, not exactly 1; it is reproduced as it is.
_H_2D = np.array(
    [
        [0.00238073, -0.00275097, -0.02402609, -0.02402609, -0.00275097, 0.00238073],
        [-0.00275097, 0.0031788, 0.0277626, 0.0277626, 0.0031788, -0.00275097],
        [-0.02402609, 0.0277626, 0.2424694, 0.2424694, 0.0277626, -0.02402609],
        [-0.02402609, 0.0277626, 0.2424694, 0.2424694, 0.0277626, -0.02402609],
        [-0.00275097, 0.0031788, 0.0277626, 0.0277626, 0.0031788, -0.00275097],
        [0.00238073, -0.00275097, -0.02402609, -0.02402609, -0.00275097, 0.00238073],
    ],
    dtype=np.float64,
)


def _check_n_orientations(value: object) -> int:
    """L as an int: an integer ≥ 1, as STL accepts (Part II, §12.5)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"L={value!r}: the number of orientations is an integer ≥ 1, such as L=4.")
    n = int(value)
    if n < 1:
        raise ValueError(f"L={n}: the number of orientations is at least 1; STL's default is 4.")
    return n


def _check_size(size: object) -> int:
    """K as an int: odd, so that the kernel has a centre, and at most the Fourier grid."""
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError(f"size={size!r}: the kernel size is an odd integer, such as 5.")
    n = int(size)
    if n < 1 or n % 2 == 0 or n > FOURIER_GRID:
        raise ValueError(
            f"size={n}: the kernel size must be odd, so that the kernel has a centre, and "
            f"between 1 and {FOURIER_GRID} (the Fourier grid); STL's default is "
            f"{KERNEL_SIZE}."
        )
    return n


def _bump(r: NDArray[np.float64], width_factor: float) -> NDArray[np.float64]:
    """STL's radial bump exp(-c r² / (1 - c r²)) on r < c^(-1/2), 0 elsewhere."""
    r2 = width_factor * r**2
    support = (r >= 0.0) & (r < width_factor**-0.5)
    denominator = np.maximum(1.0 - r2, _BUMP_EPS)
    return np.where(support, np.exp(-r2 / denominator), 0.0)


def _steerable_constant(n: int) -> float:
    """c_L = 2^(L-1) (L-1)! / √(L (2(L-1))!) of M-WAV-2D, for L = ``n``."""
    return float(2 ** (n - 1) * math.factorial(n - 1) / math.sqrt(n * math.factorial(2 * (n - 1))))


def _centred_frequencies(n: int) -> NDArray[np.float64]:
    """The integer frequencies of an n-point grid, zero in the centre: -n/2 … n/2 - 1."""
    return np.fft.fftshift(np.fft.fftfreq(n) * n)


def _crop_centre(kernel: NDArray[np.complex128], size: int, ndim: int) -> NDArray[np.complex128]:
    """Inverse FFT over the last ``ndim`` axes, centred, cropped to ``size`` around the centre,
    with its mean subtracted (steps 1 to 3 of the pixel-space part of M-WAV-2D)."""
    axes = tuple(range(-ndim, 0))
    n, w = kernel.shape[-1], size // 2
    spatial = np.fft.fftshift(np.fft.ifftn(kernel, axes=axes), axes=axes)
    window = (Ellipsis, *([slice(n // 2 - w, n // 2 + w + 1)] * ndim))
    cropped: NDArray[np.complex128] = spatial[window]  # type: ignore[index]
    result: NDArray[np.complex128] = cropped - cropped.mean(axis=axes, keepdims=True)
    return result


def _fourier_bump_steerable_2d(n_orientations: int, n: int = FOURIER_GRID) -> NDArray[np.float64]:
    """psi_l for l = 0 … L-1 at the first scale, shape (L, n, n), zero frequency at (0, 0)
    (M-WAV-2D; STL's ``bump_steerable_bank`` at j = 0)."""
    omega = _centred_frequencies(n)
    grid = np.stack(np.meshgrid(omega, omega, indexing="ij"), axis=-1)
    xi0 = n / (2.0 * math.sqrt(2.0))
    scale = _steerable_constant(n_orientations) / _steerable_constant(4)
    filters = []
    for ell in range(n_orientations):
        theta = math.pi * ell / n_orientations + math.pi / 2
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        q = grid @ rotation  # row vector times R(theta_l)
        radial = _bump(np.abs(np.hypot(q[..., 0], q[..., 1]) - xi0) / xi0, WIDTH_FACTOR_2D)
        phi = np.arctan2(q[..., 1], q[..., 0])
        angular = np.where(np.abs(phi) < math.pi / 2, np.cos(phi) ** (n_orientations - 1), 0.0)
        weights = radial * angular
        weights = weights / np.abs(weights).max()
        filters.append(np.fft.fftshift(weights * scale))
    return np.stack(filters)


def bump_steerable_2d(L: int, size: int = KERNEL_SIZE) -> NDArray[np.complex128]:  # noqa: N803
    """The L oriented 2D bump-steerable kernels of maths.md M-WAV-2D, shape ``(L, size, size)``.

    Built in Fourier space on a 256 x 256 grid, brought to pixel space by an inverse FFT,
    centred, cropped to ``size`` x ``size`` and made zero-mean, without renormalisation
    (OP-10). Kernel l responds to frequencies along theta_l = pi l/L + pi/2 from axis 1
    (OP-22). The filtering is a correlation with these kernels (M-CONV-2D).

    ``L`` is any integer ≥ 1, as in STL; the width factor 2.5 was tuned for L = 4 (OP-9).
    ``size`` is odd. The result is complex128, a new array at each call.
    """
    n_orientations = _check_n_orientations(L)
    size = _check_size(size)
    return _crop_centre(
        _fourier_bump_steerable_2d(n_orientations).astype(np.complex128), size, ndim=2
    )


def _fourier_bump_steerable_1d(xi0: float, n: int = FOURIER_GRID) -> NDArray[np.float64]:
    """psi at the first scale, shape (n,), zero frequency at 0 (M-WAV-1D; STL's
    ``bump_steerable_bank`` at j = 0): the bump with c = 1, no angular part and no
    normalisation."""
    omega = _centred_frequencies(n)
    bump: NDArray[np.float64] = np.fft.fftshift(_bump(np.abs(omega - xi0) / xi0, 1.0))
    return bump


def bump_steerable_1d(
    xi0: float | None = None, size: int = KERNEL_SIZE, conjugate: bool = False
) -> NDArray[np.complex128]:
    """The 1D bump-steerable kernel of maths.md M-WAV-1D, shape ``(L, size)``.

    Built like the 2D kernels, on a grid of 256 points: inverse FFT, centred, cropped to
    ``size`` taps, made zero-mean, without renormalisation (OP-10). ``xi0`` is the centre of
    the bump in frequency units of that grid; ``None`` gives STL's ξ₀ = N/2 = 128, the Nyquist
    frequency (OP-28). With ``conjugate=False`` the result holds the analytic kernel, L = 1;
    with ``conjugate=True`` it also holds its conjugate, psi_1 = conj(psi_0), L = 2, which covers
    the negative frequencies (the 1D blocks of tensor products).

    The result is complex128, a new array at each call.
    """
    size = _check_size(size)
    if xi0 is None:
        xi0 = FOURIER_GRID / 2.0
    if isinstance(xi0, bool) or not isinstance(xi0, int | float) or not 0.0 < xi0 < math.inf:
        raise ValueError(
            f"xi0={xi0!r}: the centre of the bump is a positive frequency on the grid of "
            f"{FOURIER_GRID} points; leave it to None for STL's value, {FOURIER_GRID // 2}."
        )
    if not isinstance(conjugate, bool | np.bool_):
        raise TypeError(f"conjugate={conjugate!r}: pass True (L = 2) or False (L = 1).")
    kernel = _crop_centre(
        _fourier_bump_steerable_1d(float(xi0)).astype(np.complex128)[None], size, ndim=1
    )
    if conjugate:
        kernel = np.concatenate([kernel, np.conj(kernel)])
    return kernel


def smoothing_kernel_H(ndim: int) -> NDArray[np.float64]:  # noqa: N802
    """The anti-aliasing kernel H of the planar downsampling, float64, a new array at each call.

    * ``ndim=2``: STL's 6 x 6 kernel of maths.md M-H, as written in STL (it sums to
      1.00000004);
    * ``ndim=1``: its third row divided by its sum, as STL's 1D class builds it (M-H-1D),
      bit for bit; the 2D kernel is its outer product with itself up to the rounding of
      STL's literals.

    STL's ``sigma_smooth`` has no effect and is dropped (OP-25).
    """
    if isinstance(ndim, bool) or ndim not in (1, 2):
        raise ValueError(
            f"ndim={ndim!r}: the planar anti-aliasing kernel exists for 1D and 2D data; pass "
            "ndim=1 or ndim=2."
        )
    if ndim == 2:
        return _H_2D.copy()
    # Divided by the correctly rounded sum (0.49241182), as torch's sum gives in STL; NumPy's
    # sequential sum is one unit in the last place lower, and that bit decides the masked
    # downsampling where a fraction of invalid input equals τ = 0.5 exactly (M-DOWN-MASK).
    row = _H_2D[2]
    h: NDArray[np.float64] = row / math.fsum(row)
    return h
