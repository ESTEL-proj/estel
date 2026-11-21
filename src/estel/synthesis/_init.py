"""The start of a synthesis: the Nyquist pre-filter, the standardisation of the target and the
Gaussian start (maths.md M-SYN-NYQ, M-SYN-TARGET, M-SYN-INIT; Part II, §12.9; task T6.1).

Pure functions on backend arrays of shape ``(Nb, Nc, N1, N2)``, periodic and without missing
values (LF47). The mask of the pre-filter is static, built in NumPy from the spatial shape.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend._protocol import Backend, DType
from ..statistics._arrays import constant

SPATIAL_AXES = (-2, -1)


def nyquist_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    """The modes kept by the pre-filter (M-SYN-NYQ): k₁² + k₂² ≤ (min(N₁, N₂)/2)², with the
    integer frequencies of the discrete Fourier transform, -Nᵢ/2 included for even Nᵢ."""
    n1, n2 = shape
    k1 = np.fft.fftfreq(n1, d=1.0 / n1)
    k2 = np.fft.fftfreq(n2, d=1.0 / n2)
    radius = min(n1, n2) / 2
    kept: NDArray[np.bool_] = k1[:, None] ** 2 + k2[None, :] ** 2 <= radius**2
    return kept


def nyquist_filter(backend: Backend, x: Any) -> Any:
    """``x`` with its modes above the Nyquist circle set to 0, on the last two axes (M-SYN-NYQ;
    STL's ``apply_nyquist_filter``): the real part of the inverse transform of the masked
    transform."""
    xp = backend.xp
    shape = (int(x.shape[-2]), int(x.shape[-1]))
    spectrum = xp.fft.fftn(x, axes=SPATIAL_AXES)
    kept = constant(backend, nyquist_mask(shape), spectrum)
    filtered = xp.where(kept, spectrum, xp.zeros_like(spectrum))
    return xp.real(xp.fft.ifftn(filtered, axes=SPATIAL_AXES))


class Standardized(NamedTuple):
    """What :func:`standardize` returns."""

    x: Any
    """``(x - mean) / std``, ``(Nb, Nc, N1, N2)``."""
    mean: Any
    """μ, ``(1, Nc)`` with ``mean_field``, else ``(Nb, Nc)``."""
    std: Any
    """sigma, of the shape of ``mean``."""


def standardize(backend: Backend, x: Any, mean_field: bool) -> Standardized:
    """The standardisation of the target (M-SYN-TARGET, M-STD; STL's ``standardize``).

    Without missing values the mean over valid pixels is the mean over the grid. With
    ``mean_field``, μ is the batch mean of the means and sigma² the batch mean of the second
    moments of the fields centred by that μ; otherwise μ and sigma are those of each map.
    """
    xp = backend.xp
    mean = xp.mean(x, axis=SPATIAL_AXES)  # (Nb, Nc)
    if mean_field:
        mean = xp.mean(mean, axis=0, keepdims=True)
    centered = x - mean[..., None, None]
    var = xp.mean(centered * centered, axis=SPATIAL_AXES)
    if mean_field:
        var = xp.mean(var, axis=0, keepdims=True)
    std = xp.sqrt(var)
    return Standardized(x=centered / std[..., None, None], mean=mean, std=std)


def unstandardize(u: Any, mean: Any, std: Any) -> Any:
    """x = u sigma + μ (M-SYN-OUT, M-STD), with μ and sigma of shape ``(1, Nc)``, ``(Nb, Nc)``
    or ``(Nc,)``."""
    return u * std[..., None, None] + mean[..., None, None]


def gaussian_start(backend: Backend, seed: int, shape: tuple[int, ...], dtype: DType) -> Any:
    """A Gaussian white noise of unit variance, of ``shape`` (M-SYN-INIT), drawn with the key of
    ``seed`` (Part I, §3.1, rule 5; OP-55): the same seed gives the same start on the same
    backend and device."""
    return backend.random_normal(backend.make_key(seed), shape, dtype)
