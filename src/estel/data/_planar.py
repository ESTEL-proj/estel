"""Planar data: ``Data1D`` and ``Data2D`` (Part II, §12.4; maths.md M-DATA-1D, M-DATA-2D).

The two classes share everything but the number of spatial axes and the threshold τ of the
masked downsampling (maths.md M-DOWN-MASK, OP-12): STL's 1D class inherits from its 2D class
and differs only there (STL `2D+1D`, `STL_1D_Kernel_Torch`). Their geometry hooks ``_conv``,
``_downsample`` and ``_crop_weights`` are the same functions, written for any number of planar
axes.
"""

from __future__ import annotations

import warnings
from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import PadMode
from ._base import DataClass, DownsampleWeights


def _check_array(data: DataClass, x: Any, hook: str) -> None:
    """``x`` is an array of ``data``'s backend with the axes ``(Nb, Nc, *spatial)``."""
    _registry.check_backend(data.backend, x, owner=f"{type(data).__name__}.{hook}")
    if len(x.shape) != 2 + data.ndim:
        raise ValueError(
            f"{type(data).__name__}.{hook}: the array has shape {tuple(x.shape)}; pass "
            f"(Nb, Nc, *spatial), with {data.ndim} spatial axis(es)."
        )


def _is_complex(xp: Any, x: Any) -> bool:
    return bool(xp.isdtype(x.dtype, "complex floating"))


def _padding_mode(data: DataClass) -> PadMode:
    """Circular if ``pbc``, replicate otherwise (maths.md M-CONV-2D, M-DOWN)."""
    return "circular" if data.pbc else "replicate"


def _planar_conv(self: DataClass, x: Any, kernels: Any, level: int) -> Any:
    """Planar ``_conv``: pad by ⌊K/2⌋ on each side of every spatial axis, then correlate with
    each kernel (maths.md M-CONV-2D, M-W1, OP-1). ``level`` does not matter: the planar
    kernels are the same at every level (M-WAV-2D, M-WAV-1D).

    A complex kernel on a real array is two real correlations; on a complex array, four.
    """
    _check_array(self, x, "_conv")
    backend = _registry.backend_named(self.backend)
    xp = backend.xp
    if not _is_complex(xp, kernels) or len(kernels.shape) != self.ndim + 1:
        raise TypeError(
            f"{type(self).__name__}._conv: pass the complex wavelet bank of the geometry, "
            f"shape (L, {', '.join(['K'] * self.ndim)}); got dtype {kernels.dtype} and shape "
            f"{tuple(kernels.shape)}."
        )
    axes = self._spatial_axes()
    widths = [(int(k) // 2, int(k) // 2) for k in kernels.shape[1:]]
    padded = backend.pad(x, widths, axes, _padding_mode(self))
    k_re, k_im = xp.real(kernels), xp.imag(kernels)

    def correlate(a: Any, k: Any) -> Any:
        return backend.conv_axes(a, k, axes)

    if _is_complex(xp, padded):
        x_re, x_im = xp.real(padded), xp.imag(padded)
        real = correlate(x_re, k_re) - correlate(x_im, k_im)
        imag = correlate(x_re, k_im) + correlate(x_im, k_re)
    else:
        real, imag = correlate(padded, k_re), correlate(padded, k_im)
    return backend.real_pairs_to_complex(xp.stack([real, imag], axis=-1))


def _planar_downsample(self: DataClass, x: Any, weights: DownsampleWeights, level: int) -> Any:
    """Planar ``_downsample``: one step of maths.md M-DOWN, normalised as in M-DOWN-MASK when
    the data have missing values.

    Invalid input pixels are set to 0 (selected out, so that a NaN there is never read and
    gives no NaN gradient, OP-27); the array is padded by ``(K-1)//2`` before and ``K//2``
    after each spatial axis (2 and 3 for the smoothing kernel, OP-2), circular if ``pbc``,
    replicate otherwise; correlated with the smoothing kernel at stride 2; then multiplied by
    the reweighting map. Complex arrays are downsampled part by part. ``level`` does not
    matter: the smoothing kernel is the same at every level.
    """
    _check_array(self, x, "_downsample")
    backend = _registry.backend_named(self.backend)
    xp = backend.xp
    kernel, valid, reweight = weights
    if len(kernel.shape) != self.ndim:
        raise ValueError(
            f"{type(self).__name__}._downsample: the kernel has shape {tuple(kernel.shape)}; "
            f"pass the smoothing kernel of the geometry, smoothing_kernel({self.ndim})."
        )
    if (valid is None) != (reweight is None):
        raise ValueError(
            f"{type(self).__name__}._downsample: pass both the valid input pixels and the "
            "reweighting map (data with missing values), or neither."
        )
    spatial = tuple(int(n) for n in x.shape[2:])
    if any(n % 2 for n in spatial):
        raise ValueError(
            f"{type(self).__name__}._downsample: cannot downsample spatial shape {spatial}: "
            "every axis must be even at each level, so the sizes must be divisible by 2^(J-1) "
            "(maths.md M-DATA-2D, OP-8)."
        )
    if valid is not None:
        x = xp.where(valid, x, xp.zeros_like(x))
    axes = self._spatial_axes()
    widths = [((int(k) - 1) // 2, int(k) // 2) for k in kernel.shape]
    padded = backend.pad(x, widths, axes, _padding_mode(self))

    def step(a: Any) -> Any:
        return backend.conv_axes(a, kernel, axes, stride=2)

    if _is_complex(xp, padded):
        pair = xp.stack([step(xp.real(padded)), step(xp.imag(padded))], axis=-1)
        out = backend.real_pairs_to_complex(pair)
    else:
        out = step(padded)
    return out if reweight is None else out * reweight


def _planar_crop_weights(self: DataClass, level: int, border: int) -> NDArray[np.float64]:
    """Planar ``_crop_weights``: 0 on ``border`` pixels at both ends of every spatial axis of
    the grid at ``level``, 1 elsewhere (maths.md M-BORDER; STL ``_crop``).

    As in STL, a border that leaves no pixel on the shortest axis (2·border ≥ its length) is
    reduced to ⌊(min N - 1)/2⌋, with a warning (OP-13). A border of 0 is never reduced.
    """
    if isinstance(level, bool) or not isinstance(level, Integral) or level < 0:
        raise ValueError(f"{type(self).__name__}._crop_weights: level={level!r}; pass a level ≥ 0.")
    if isinstance(border, bool) or not isinstance(border, Integral) or border < 0:
        raise ValueError(
            f"{type(self).__name__}._crop_weights: border={border!r}; pass a width ≥ 0, in "
            "pixels at that level (maths.md M-BORDER)."
        )
    full = self.shape[2:]
    if any(n % 2 ** int(level) for n in full):
        raise ValueError(
            f"{type(self).__name__}._crop_weights: the spatial shape {full} cannot be "
            f"downsampled to level {level}: the sizes must be divisible by 2^level (maths.md "
            "M-DATA-2D, OP-8). Use a smaller level."
        )
    shape = tuple(n // 2 ** int(level) for n in full)
    width = int(border)
    if width and min(shape) <= 2 * width:
        reduced = (min(shape) - 1) // 2
        warnings.warn(
            f"{type(self).__name__}: at level {level} the spatial shape {shape} is too small "
            f"for a border crop of {width} pixel(s); using {reduced} instead, as STL does "
            "(maths.md M-BORDER, OP-13). Use a smaller J, larger data or pbc=True to avoid it.",
            UserWarning,
            stacklevel=2,
        )
        width = reduced
    inner = np.ones(tuple(n - 2 * width for n in shape), dtype=np.float64)
    return np.pad(inner, width)


class Data2D(DataClass):
    """2D planar data, shape ``(Nb, Nc, N1, N2)``.

    ``Data2D(array, mask=None, *, pbc)``: see :class:`estel.DataClass` for the parameters.
    A 2D array is promoted to ``(1, 1, N1, N2)``, a 3D array to ``(1, Nc, N1, N2)``
    (maths.md M-DATA-2D).
    """

    ndim: ClassVar[int] = 2
    _nan_threshold: ClassVar[float] = 0.33

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)

    _conv = _planar_conv
    _downsample = _planar_downsample
    _crop_weights = _planar_crop_weights


class Data1D(DataClass):
    """1D planar data, shape ``(Nb, Nc, N)``.

    ``Data1D(array, mask=None, *, pbc)``: see :class:`estel.DataClass` for the parameters.
    A 1D array is promoted to ``(1, 1, N)``; a 2D array to ``(1, Nc, N)``, its first axis read
    as channels, not as a batch, as in STL (maths.md M-DATA-1D, OP-29).
    """

    ndim: ClassVar[int] = 1
    _nan_threshold: ClassVar[float] = 0.5

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)

    _conv = _planar_conv
    _downsample = _planar_downsample
    _crop_weights = _planar_crop_weights
