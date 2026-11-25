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
    axes = tuple(range(2, len(x.shape)))  # every axis after the channel axis
    expand = (Ellipsis, *([None] * len(axes)))
    mean = xp.mean(x, axis=axes)  # (Nb, Nc)
    if mean_field:
        mean = xp.mean(mean, axis=0, keepdims=True)
    centered = x - mean[expand]
    var = xp.mean(centered * centered, axis=axes)
    if mean_field:
        var = xp.mean(var, axis=0, keepdims=True)
    std = xp.sqrt(var)
    return Standardized(x=centered / std[expand], mean=mean, std=std)


def unstandardize(u: Any, mean: Any, std: Any) -> Any:
    """x = u sigma + μ (M-SYN-OUT, M-STD), ``u`` being ``(Nb, Nc, *spatial)`` and μ and sigma of
    shape ``(1, Nc)``, ``(Nb, Nc)`` or ``(Nc,)``."""
    expand = (Ellipsis, *([None] * (len(u.shape) - 2)))
    return u * std[expand] + mean[expand]


def gaussian_start(backend: Backend, seed: int, shape: tuple[int, ...], dtype: DType) -> Any:
    """A Gaussian white noise of unit variance, of ``shape`` (M-SYN-INIT), drawn with the key of
    ``seed`` (Part I, §3.1, rule 5; OP-55): the same seed gives the same start on the same
    backend and device."""
    return backend.random_normal(backend.make_key(seed), shape, dtype)
