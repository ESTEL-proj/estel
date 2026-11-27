"""The HEALPix wavelet kernel (Part II, §12.5; task T5.1, issue #26).

Built once, in NumPy float64, when an operator is built; never traced, never differentiated (tag
N of Part II, §12). The plan casts it to the dtype of the data (maths.md OP-46, decided on 24
November 2026: no cast through float32, unlike healpix-analyse's ``HealPixConv.set_kernel``).

:func:`kernel_weights_healpix` is the kernel of maths.md M-WAV-HPX, ported from STL
`HealpixUpdate` ``90eb1a9`` (``WaveletOperatorHealpixKernel_torch._wavelet_kernel``): one complex
kernel on the K x K stencil of M-CONV-HPX, the same at every level and for every gauge; the
orientations are the gauges of the stencil (``_tables_healpix.stencils_healpix``), not
rotations of the kernel. ``tests/parity/test_healpix_tables.py`` checks it against STL's.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from estel.wavelets._checks import _check_zero_sum

__all__ = ["GAUGES_HEALPIX", "KERNEL_SIZE_HEALPIX", "kernel_weights_healpix"]

KERNEL_SIZE_HEALPIX = 5
"""K, the stencil size of the HEALPix convolution, 5 x 5 points (maths.md M-DEF-HPX)."""

GAUGES_HEALPIX = 4
"""L, the number of gauges of the HEALPix convolution (maths.md M-DEF-HPX)."""


def _check_kernel_size(kernel_size: object) -> int:
    """K as an int: odd, so that the stencil has a centre, and at least 3."""
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral):
        raise TypeError(
            f"kernel_size={kernel_size!r}: the stencil size is an odd integer ≥ 3, such as "
            f"{KERNEL_SIZE_HEALPIX}."
        )
    size = int(kernel_size)
    if size < 3 or size % 2 == 0:
        raise ValueError(
            f"kernel_size={size}: the stencil size must be odd, so that the stencil has a centre "
            f"pixel, and at least 3 (a 1 x 1 kernel is zero once its mean is subtracted); the "
            f"default is {KERNEL_SIZE_HEALPIX} (maths.md M-DEF-HPX)."
        )
    return size


def kernel_weights_healpix(*, kernel_size: int = KERNEL_SIZE_HEALPIX) -> NDArray[np.complex128]:
    """The complex wavelet kernel of maths.md M-WAV-HPX, flattened on the stencil.

    ψ[u, v] = exp(-(u² + v²) / 2) · exp(i π u / 2) on the offsets u, v ∈ {-c, …, c},
    c = (K - 1) / 2, then the mean is subtracted and the kernel divided by Σ |ψ|, taken after
    the subtraction.

    Shapes:
        ``(K², )``, entry p = (v + c) K + (u + c): v runs along the first stencil axis and u
        along the second, the flattening of the stencil of M-CONV-HPX, step 1, and of the
        tables of ``stencils_healpix``.
    Dtypes:
        complex128, whatever the dtype of the data: the plan casts it (OP-46).
    Values:
        Sums to zero (checked, as the planar kernels, Part III, §20.11) and the sum of its
        moduli is 1. ψ[-u, v] = conj ψ[u, v] and ψ[u, -v] = ψ[u, v].
    Edge cases:
        ``kernel_size`` is an odd integer ≥ 3, else ``TypeError`` or ``ValueError``.
    """
    size = _check_kernel_size(kernel_size)
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    v, u = np.meshgrid(coords, coords, indexing="ij")
    envelope = np.exp(-(u**2 + v**2) / 2.0)
    phase = 0.5 * np.pi * u
    kernel = np.cos(phase) * envelope + 1j * (np.sin(phase) * envelope)
    centred = kernel - kernel.mean()
    result: NDArray[np.complex128] = (centred / np.abs(centred).sum()).reshape(-1)
    _check_zero_sum(result[None], 1, "kernel_weights_healpix")
    return result
