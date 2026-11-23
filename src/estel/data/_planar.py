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
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import PadMode
from ._base import Data, DownsampleWeights
from ._moments import _is_complex


def _check_array(data: Data, x: Any, hook: str) -> None:
    """``x`` is an array of ``data``'s backend with the axes ``(Nb, Nc, *spatial)``."""
    _registry.check_backend(data.backend, x, owner=f"{type(data).__name__}.{hook}")
    if len(x.shape) != 2 + data.ndim:
        raise ValueError(
            f"{type(data).__name__}.{hook}: the array has shape {tuple(x.shape)}; pass "
            f"(Nb, Nc, *spatial), with {data.ndim} spatial axis(es)."
        )


def _padding_mode(pbc: bool) -> PadMode:
    """Circular if ``pbc``, replicate otherwise (maths.md M-CONV-2D, M-DOWN)."""
    return "circular" if pbc else "replicate"


def _planar_conv(self: Data, x: Any, tables: Any, level: int) -> Any:
    """Planar ``_conv``: pad by ⌊K/2⌋ on each side of every spatial axis, then correlate with
    each kernel (maths.md M-CONV-2D, M-W1, OP-1). The ``tables`` are the kernel bank, and
    ``level`` does not matter: the planar kernels are the same at every level (M-WAV-2D,
    M-WAV-1D). The filtering itself is :func:`_planar_conv_along`, on the spatial axes.
    """
    _check_array(self, x, "_conv")
    return type(self)._conv_along(
        x,
        tables,
        axes=self._spatial_axes(),
        pbc=cast(bool, self.pbc),  # planar data: one boolean
        backend=self.backend,
        owner=f"{type(self).__name__}._conv",
    )


def _planar_conv_along(
    cls: type[Data],
    x: Any,
    tables: Any,
    *,
    axes: tuple[int, ...],
    pbc: bool,
    backend: str,
    owner: str,
) -> Any:
    """The planar filtering along ``axes`` of ``x``, contiguous and increasing, the
    ``cls.ndim`` axes of the geometry or of one block of a tensor product (Part II, §12.4):
    padded and correlated along these axes only, every other axis carried through, none moved
    (maths.md M-CONV-2D; M-NOT-TP, the block convolution ⋆_a).

    A complex kernel on a real array is one real correlation, with the real and imaginary
    kernels stacked; on a complex array, four (Part I, §3.1, rule 6; T3.5).
    The orientation axis, which the correlation inserts before ``axes[0]``, is read right after
    the batch axis, ``(Nb, L, Nc, ...)``, when the complex result is assembled: no further copy.
    """
    array_backend = _registry.backend_named(backend)
    xp = array_backend.xp
    kernels = tables
    if not _is_complex(xp, kernels) or len(kernels.shape) != len(axes) + 1:
        raise TypeError(
            f"{owner}: pass the complex wavelet bank of the geometry, shape "
            f"(L, {', '.join(['K'] * len(axes))}); got dtype {kernels.dtype} and shape "
            f"{tuple(kernels.shape)}."
        )
    widths = [(int(k) // 2, int(k) // 2) for k in kernels.shape[1:]]
    padded = array_backend.pad(x, widths, axes, _padding_mode(pbc))
    first = axes[0]
    n_l = int(kernels.shape[0])
    if _is_complex(xp, padded):
        # Four real calls: one call on the real and imaginary parts stacked raises the peak
        # memory on PyTorch by 23 % at 64 x 64 x 32 (T3.5, #128).
        k_re, k_im = xp.real(kernels), xp.imag(kernels)
        x_re, x_im = xp.real(padded), xp.imag(padded)

        def correlate(a: Any, k: Any) -> Any:
            return array_backend.conv_axes(a, k, axes)

        real = correlate(x_re, k_re) - correlate(x_im, k_im)
        imag = correlate(x_re, k_im) + correlate(x_im, k_re)
        # (Nb, Nc, ..., L, *block, ...) -> (Nb, L, Nc, ..., *block, ...)
        order = (0, first, *(i for i in range(1, len(real.shape)) if i != first))
        real, imag = xp.permute_dims(real, order), xp.permute_dims(imag, order)
        return array_backend.real_pairs_to_complex(xp.stack([real, imag], axis=-1))
    # A real array: one real call with the real and imaginary kernels stacked, (2L, K...).
    both_k = xp.concat([xp.real(kernels), xp.imag(kernels)], axis=0)
    y = array_backend.conv_axes(padded, both_k, axes)
    # (Nb, Nc, ..., 2L, *block, ...) -> (Nb, Nc, ..., 2, L, *block, ...) -> (Nb, L, Nc, ...,
    # *block, ..., 2): the real and imaginary parts become the pairs, with no copy before them.
    shape = tuple(int(n) for n in y.shape)
    y = xp.reshape(y, (*shape[:first], 2, n_l, *shape[first + 1 :]))
    rest = (i for i in range(1, len(y.shape)) if i not in (first, first + 1))
    return array_backend.real_pairs_to_complex(xp.permute_dims(y, (0, first + 1, *rest, first)))


def _planar_downsample(self: Data, x: Any, weights: DownsampleWeights, level: int) -> Any:
    """Planar ``_downsample``: one step of maths.md M-DOWN, normalised as in M-DOWN-MASK when
    the data have missing values; :func:`_planar_downsample_along`, on the spatial axes. ``level``
    does not matter: the smoothing kernel is the same at every level.
    """
    _check_array(self, x, "_downsample")
    return type(self)._downsample_along(
        x,
        weights,
        axes=self._spatial_axes(),
        pbc=cast(bool, self.pbc),  # planar data: one boolean
        backend=self.backend,
        owner=f"{type(self).__name__}._downsample",
    )


def _planar_downsample_along(
    cls: type[Data],
    x: Any,
    weights: DownsampleWeights,
    *,
    axes: tuple[int, ...],
    pbc: bool,
    backend: str,
    owner: str,
) -> Any:
    """One planar downsampling step along ``axes`` of ``x``, contiguous and increasing, the
    axes of the geometry or of one block of a tensor product (maths.md M-DOWN, M-DOWN-MASK;
    M-NOT-TP): every other axis is carried through, none moved.

    Invalid input pixels are set to 0 (selected out, so that a NaN there is never read and
    gives no NaN gradient, OP-27); the array is padded by ``(K-1)//2`` before and ``K//2``
    after each of ``axes`` (2 and 3 for the smoothing kernel, OP-2), circular if ``pbc``,
    replicate otherwise; correlated with the smoothing kernel at stride 2; then multiplied by
    the reweighting map. The masks are maps of the trailing spatial axes, so ``valid`` and
    ``reweight`` need ``axes`` to be the last axes. Complex arrays are downsampled part by part.
    """
    array_backend = _registry.backend_named(backend)
    xp = array_backend.xp
    kernel, valid, reweight = weights
    if len(kernel.shape) != len(axes):
        raise ValueError(
            f"{owner}: the kernel has shape {tuple(kernel.shape)}; pass the smoothing kernel of "
            f"the geometry, smoothing_kernel({len(axes)})."
        )
    if (valid is None) != (reweight is None):
        raise ValueError(
            f"{owner}: pass both the valid input pixels and the reweighting map (data with "
            "missing values), or neither."
        )
    sizes = tuple(int(x.shape[a]) for a in axes)
    if any(n % 2 for n in sizes):
        raise ValueError(
            f"{owner}: cannot downsample spatial shape {sizes}: every axis must be even at each "
            "level, so the sizes must be divisible by 2^(J-1) (maths.md M-DATA-2D, OP-8)."
        )
    if valid is not None:
        if axes[-1] != len(x.shape) - 1:
            raise ValueError(
                f"{owner}: the maps of missing values cover the trailing spatial axes; data "
                "with missing values are not supported in tensor products (LF8)."
            )
        x = xp.where(valid, x, xp.zeros_like(x))
    widths = [((int(k) - 1) // 2, int(k) // 2) for k in kernel.shape]
    padded = array_backend.pad(x, widths, axes, _padding_mode(pbc))

    def step(a: Any) -> Any:
        return array_backend.conv_axes(a, kernel, axes, stride=2)

    if _is_complex(xp, padded):
        pair = xp.stack([step(xp.real(padded)), step(xp.imag(padded))], axis=-1)
        out = array_backend.real_pairs_to_complex(pair)
    else:
        out = step(padded)
    return out if reweight is None else out * reweight


def _planar_crop_weights(self: Data, level: int, border: int) -> NDArray[np.float64]:
    """Planar ``_crop_weights``: :func:`_planar_crop_weights_along` on the spatial shape."""
    return type(self)._crop_weights_along(
        self.shape[2:], level, border, owner=f"{type(self).__name__}._crop_weights"
    )


def _planar_crop_weights_along(
    cls: type[Data], shape: tuple[int, ...], level: int, border: int, *, owner: str
) -> NDArray[np.float64]:
    """0 on ``border`` pixels at both ends of every axis of the grid of full-resolution
    ``shape`` taken at ``level``, 1 elsewhere (maths.md M-BORDER; M-BORDER-TP for one block of
    a tensor product; STL ``_crop``).

    As in STL, a border that leaves no pixel on the shortest axis (2·border ≥ its length) is
    reduced to ⌊(min N - 1)/2⌋, with a warning (OP-13). A border of 0 is never reduced.
    """
    if isinstance(level, bool) or not isinstance(level, Integral) or level < 0:
        raise ValueError(f"{owner}: level={level!r}; pass a level ≥ 0.")
    if isinstance(border, bool) or not isinstance(border, Integral) or border < 0:
        raise ValueError(
            f"{owner}: border={border!r}; pass a width ≥ 0, in pixels at that level (maths.md "
            "M-BORDER)."
        )
    full = tuple(int(n) for n in shape)
    if any(n % 2 ** int(level) for n in full):
        raise ValueError(
            f"{owner}: the spatial shape {full} cannot be downsampled to level {level}: the "
            "sizes must be divisible by 2^level (maths.md M-DATA-2D, OP-8). Use a smaller level."
        )
    grid = tuple(n // 2 ** int(level) for n in full)
    width = int(border)
    if width and min(grid) <= 2 * width:
        reduced = (min(grid) - 1) // 2
        warnings.warn(
            f"{owner.split('.')[0]}: at level {level} the spatial shape {grid} is too small for "
            f"a border crop of {width} pixel(s); using {reduced} instead (maths.md M-BORDER, "
            "OP-13). Use a smaller J, larger data or pbc=True to avoid it.",
            UserWarning,
            stacklevel=3,
        )
        width = reduced
    inner = np.ones(tuple(n - 2 * width for n in grid), dtype=np.float64)
    return np.pad(inner, width)


class Data2D(Data):
    """2D planar data, shape ``(Nb, Nc, N1, N2)``.

    ``Data2D(array, mask=None, *, pbc)``: see :class:`estel.Data` for the parameters.
    A 2D array is promoted to ``(1, 1, N1, N2)``, a 3D array to ``(1, Nc, N1, N2)``
    (maths.md M-DATA-2D).
    """

    ndim: ClassVar[int] = 2
    _nan_threshold: ClassVar[float] = 0.33
    _geometry: ClassVar[str] = "2d"

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)

    _conv = _planar_conv
    _downsample = _planar_downsample
    _crop_weights = _planar_crop_weights
    _conv_along = classmethod(_planar_conv_along)
    _downsample_along = classmethod(_planar_downsample_along)
    _crop_weights_along = classmethod(_planar_crop_weights_along)


class Data1D(Data):
    """1D planar data, shape ``(Nb, Nc, N)``.

    ``Data1D(array, mask=None, *, pbc)``: see :class:`estel.Data` for the parameters.
    A 1D array is promoted to ``(1, 1, N)``; a 2D array to ``(1, Nc, N)``, its first axis read
    as channels, not as a batch (maths.md M-DATA-1D, OP-29).
    """

    ndim: ClassVar[int] = 1
    _nan_threshold: ClassVar[float] = 0.5
    _geometry: ClassVar[str] = "1d"

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)

    _conv = _planar_conv
    _downsample = _planar_downsample
    _crop_weights = _planar_crop_weights
    _conv_along = classmethod(_planar_conv_along)
    _downsample_along = classmethod(_planar_downsample_along)
    _crop_weights_along = classmethod(_planar_crop_weights_along)
