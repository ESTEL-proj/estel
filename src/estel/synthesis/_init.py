"""The start of a synthesis: the standardisation of the target and the Gaussian start (maths.md
M-SYN-TARGET, M-SYN-INIT; Part II, §12.9; task T6.1).

Pure functions on backend arrays of shape ``(Nb, Nc, *spatial)``, ``(Nb, Nc, N1, N2)`` on 2D data
and ``(Nb, Nc, Npix)`` on HEALPix data, without missing values (LF47). The pre-filter of
M-SYN-NYQ is the geometry hook ``Data._nyquist_filter`` (Part II, §12.4).
"""

from __future__ import annotations

from typing import Any

from ..backend._protocol import Backend, DType
from ..data._moments import Standardized


def _spatial_axes(x: Any) -> tuple[int, ...]:
    """The spatial axes of ``x``, ``(Nb, Nc, *spatial)``: every axis after the channel axis."""
    return tuple(range(2, len(x.shape)))


def _per_map(value: Any, n_spatial: int) -> Any:
    """``value``, of shape ``(..., Nc)``, with ``n_spatial`` axes of length 1 after it, to
    broadcast over the maps."""
    return value[(..., *(None,) * n_spatial)]


def standardize_target(backend: Backend, x: Any, mean_field: bool) -> Standardized:
    """The standardisation of the target (M-SYN-TARGET, M-STD; STL's ``standardize``), ``x``
    being ``(Nb, Nc, *spatial)``: μ and sigma are ``(1, Nc)`` with ``mean_field``, else
    ``(Nb, Nc)``.

    Without missing values the mean over valid pixels is the mean over the grid, or over the
    pixels of the sphere. With ``mean_field``, μ is the batch mean of the means and sigma² the
    batch mean of the second moments of the fields centred by that μ; otherwise μ and sigma are
    those of each map.
    """
    xp = backend.xp
    axes = _spatial_axes(x)
    mean = xp.mean(x, axis=axes)  # (Nb, Nc)
    if mean_field:
        mean = xp.mean(mean, axis=0, keepdims=True)
    centered = x - _per_map(mean, len(axes))
    var = xp.mean(centered * centered, axis=axes)
    if mean_field:
        var = xp.mean(var, axis=0, keepdims=True)
    std = xp.sqrt(var)
    return Standardized(x=centered / _per_map(std, len(axes)), mean=mean, std=std)


def unstandardize(u: Any, mean: Any, std: Any) -> Any:
    """x = u sigma + μ (M-SYN-OUT, M-STD), ``u`` being ``(Nb, Nc, *spatial)`` and μ and sigma of
    shape ``(1, Nc)``, ``(Nb, Nc)`` or ``(Nc,)``."""
    n_spatial = len(u.shape) - 2
    return u * _per_map(std, n_spatial) + _per_map(mean, n_spatial)


def gaussian_start(backend: Backend, seed: int, shape: tuple[int, ...], dtype: DType) -> Any:
    """A Gaussian white noise of unit variance, of ``shape`` (M-SYN-INIT), drawn with the key of
    ``seed`` (Part I, §3.1, rule 5; OP-55): the same seed gives the same start on the same
    backend and device."""
    return backend.random_normal(backend.make_key(seed), shape, dtype)
