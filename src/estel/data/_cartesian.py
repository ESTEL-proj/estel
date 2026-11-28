"""Cartesian data: ``DataCartesian``, ``Data1D`` and ``Data2D`` (Part I, §2.3.1; Part II, §12.4;
maths.md M-DATA-1D, M-DATA-2D).

The two classes share everything but the number of spatial axes and the threshold τ of the
masked downsampling (maths.md M-DOWN-MASK, OP-12): STL's 1D class inherits from its 2D class
and differs only there (STL `2D+1D`, `STL_1D_Kernel_Torch`). What they share is the internal
class ``DataCartesian``: their geometry hooks ``_conv``, ``_downsample`` and ``_crop_weights``,
written for any number of Cartesian axes, the same three along the axes of one block of a
tensor product, and the capability of being such a block. ``Data2D`` alone has the pre-filter of
syntheses, ``nyquist_filter`` (maths.md M-SYN-NYQ).
"""

from __future__ import annotations

import warnings
from numbers import Integral
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import Backend, PadMode
from ._base import Data, DownsampleWeights
from ._moments import _is_complex


class DataCartesian(Data):
    """The data of a regular Cartesian grid, the base of ``Data1D`` and ``Data2D`` (Part I,
    §2.3.1): internal, never built itself. It holds what the two share, the construction from an
    array of the grid, the geometry hooks of filtering, downsampling and border crop, and their
    forms along the axes of one block of a tensor product (``DataTensorProd``, maths.md
    M-NOT-TENSORPROD); a subclass sets ``ndim`` and the threshold of M-DOWN-MASK."""

    ndim: ClassVar[int]
    """The number of spatial axes, a class constant that each subclass sets."""

    _block_tensor_prod: ClassVar[bool] = False
    """Whether the class may be a block of a tensor product (maths.md M-DATA-TENSORPROD), a
    capability each class declares in its own body: ``DataTensorProd`` reads it there, never
    inherited, so that ``Data1D`` and ``Data2D`` are the blocks of ESTEL 1.0.0, and neither
    ``DataCartesian`` nor a subclass of them. HEALPix blocks come later (LF9)."""

    # ------------------------------------------------------------------ construction

    @classmethod
    def _accepted_shapes(cls) -> str:
        """The shapes the constructor accepts, for error messages."""
        ndim = cls.ndim  # a class constant of DataCartesian
        spatial = ", ".join(f"N{i + 1}" for i in range(ndim)) if ndim > 1 else "N"
        return f"({spatial}), (Nc, {spatial}) or (Nb, Nc, {spatial})"

    def _meta(self) -> tuple[Any, ...]:
        """The static metadata of the geometry beyond the mask and ``pbc``, hashable: what
        :meth:`_from_parts` needs, with them, to rebuild an object of this class. A geometry
        with more metadata overrides it and :meth:`_set_meta`; Cartesian data have none."""
        return ()

    @classmethod
    def _meta_content(cls, meta: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        """The metadata ``meta`` of :meth:`_meta` for a file (``estel.io``): JSON parameters,
        and arrays by name, which :meth:`_example_from_content` reads back. 1D and 2D data have
        none."""
        return {}, {}

    @classmethod
    def _example_from_content(
        cls,
        n_channels: int,
        spatial_shape: tuple[int, ...],
        pbc: bool,
        content: dict[str, Any],
        arrays: dict[str, NDArray[Any]],
    ) -> Data:
        """An example of the data a file describes (``STStatistics.load``), zeros of one map of
        ``n_channels`` channels, built by the constructor, which checks it: the spatial shape,
        ``pbc``, the metadata of :meth:`_meta_content`, its JSON ``content``, and the plan's
        ``arrays`` by name, the mask (``"mask"``, when values are missing) and the arrays of
        :meth:`_meta_content`."""
        mask = arrays["mask"].astype(bool) if "mask" in arrays else None
        return cls(np.zeros((1, n_channels, *spatial_shape)), mask=mask, pbc=pbc)

    def _set_meta(self, meta: tuple[Any, ...]) -> None:
        """Set the metadata of :meth:`_meta` on a new object (from :meth:`_set`)."""
        if meta:
            raise TypeError(
                f"{type(self).__name__} has no static metadata beyond its mask and pbc; got "
                f"{meta!r}. Pass meta=() or the _meta() of an object of the same class."
            )

    def _rebuild_advice(self) -> str:
        """The constructor call that builds an object like this one from a NumPy ``array``,
        for the messages."""
        return f"{type(self).__name__}(array, mask=data.mask, pbc=data.pbc)"

    # ------------------------------------------------------------------ geometry hooks

    def _spatial_axes(self) -> tuple[int, ...]:
        """The axes the geometry acts on: the last ``ndim`` axes of the array."""
        return tuple(range(2, 2 + self.ndim))

    def _check_array(self, x: Any, hook: str) -> None:
        """``x`` is an array of this object's backend with the axes ``(Nb, Nc, *spatial)``."""
        _registry.check_backend(self.backend, x, owner=f"{type(self).__name__}.{hook}")
        if len(x.shape) != 2 + self.ndim:
            raise ValueError(
                f"{type(self).__name__}.{hook}: the array has shape {tuple(x.shape)}; pass "
                f"(Nb, Nc, *spatial), with {self.ndim} spatial axis(es)."
            )

    @staticmethod
    def _padding_mode(pbc: bool) -> PadMode:
        """Circular if ``pbc``, replicate otherwise (maths.md M-CONV, M-DOWN)."""
        return "circular" if pbc else "replicate"

    def _conv(self, x: Any, tables: Any, level: int) -> Any:
        """Cartesian ``_conv``: pad by ⌊K/2⌋ on each side of every spatial axis, then correlate with
        each kernel (maths.md M-CONV, M-W1, OP-1). The ``tables`` are the kernel bank, and
        ``level`` does not matter: the Cartesian kernels are the same at every level (M-WAV-2D,
        M-WAV-1D). The filtering itself is :meth:`_conv_along`, on the spatial axes.
        """
        self._check_array(x, "_conv")
        return type(self)._conv_along(
            x,
            tables,
            axes=self._spatial_axes(),
            pbc=cast(bool, self.pbc),  # Cartesian data: one boolean
            backend=self.backend,
            owner=f"{type(self).__name__}._conv",
        )

    @classmethod
    def _conv_along(
        cls,
        x: Any,
        tables: Any,
        *,
        axes: tuple[int, ...],
        pbc: bool,
        backend: str,
        owner: str,
    ) -> Any:
        """The Cartesian filtering along ``axes`` of ``x``, contiguous and increasing, the
        ``cls.ndim`` axes of the geometry or of one block of a tensor product (Part II, §12.4):
        padded and correlated along these axes only, every other axis carried through, none moved
        (maths.md M-CONV; M-NOT-TENSORPROD, the block convolution ⋆_a).

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
        padded = array_backend.pad(x, widths, axes, cls._padding_mode(pbc))
        first = axes[0]
        n_l = int(kernels.shape[0])
        if _is_complex(xp, padded):
            # Four real calls on both backends, by the maintainer's decision (T3.5, #128): one call
            # on the real and imaginary parts stacked raises the peak memory on PyTorch by 23 % at
            # 64 x 64 x 32, though it gains on JAX. One call for the complex intermediate maps is
            # deferred to step 2a (LF62).
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

    def _downsample(self, x: Any, weights: DownsampleWeights, level: int) -> Any:
        """Cartesian ``_downsample``: one step of maths.md M-DOWN, normalised as in M-DOWN-MASK when
        the data have missing values; :meth:`_downsample_along`, on the spatial axes. ``level``
        does not matter: the smoothing kernel is the same at every level.
        """
        self._check_array(x, "_downsample")
        return type(self)._downsample_along(
            x,
            weights,
            axes=self._spatial_axes(),
            pbc=cast(bool, self.pbc),  # Cartesian data: one boolean
            backend=self.backend,
            owner=f"{type(self).__name__}._downsample",
        )

    @classmethod
    def _downsample_along(
        cls,
        x: Any,
        weights: DownsampleWeights,
        *,
        axes: tuple[int, ...],
        pbc: bool,
        backend: str,
        owner: str,
    ) -> Any:
        """One Cartesian downsampling step along ``axes`` of ``x``, contiguous and increasing, the
        axes of the geometry or of one block of a tensor product (maths.md M-DOWN, M-DOWN-MASK;
        M-NOT-TENSORPROD): every other axis is carried through, none moved.

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
                f"{owner}: the kernel has shape {tuple(kernel.shape)}; pass the smoothing kernel "
                f"of the geometry, smoothing_kernel_cartesian({len(axes)})."
            )
        if (valid is None) != (reweight is None):
            raise ValueError(
                f"{owner}: pass both the valid input pixels and the reweighting map (data with "
                "missing values), or neither."
            )
        sizes = tuple(int(x.shape[a]) for a in axes)
        if any(n % 2 for n in sizes):
            raise ValueError(
                f"{owner}: cannot downsample spatial shape {sizes}: every axis must be even at "
                "each level, so the sizes must be divisible by 2^(J-1) (maths.md M-DATA-2D, OP-8)."
            )
        if valid is not None:
            if axes[-1] != len(x.shape) - 1:
                raise ValueError(
                    f"{owner}: the maps of missing values cover the trailing spatial axes; data "
                    "with missing values are not supported in tensor products (LF8)."
                )
            x = xp.where(valid, x, xp.zeros_like(x))
        widths = [((int(k) - 1) // 2, int(k) // 2) for k in kernel.shape]
        padded = array_backend.pad(x, widths, axes, cls._padding_mode(pbc))

        def step(a: Any) -> Any:
            return array_backend.conv_axes(a, kernel, axes, stride=2)

        if _is_complex(xp, padded):
            pair = xp.stack([step(xp.real(padded)), step(xp.imag(padded))], axis=-1)
            out = array_backend.real_pairs_to_complex(pair)
        else:
            out = step(padded)
        return out if reweight is None else out * reweight

    def _crop_weights(self, level: int, border: int) -> NDArray[np.float64]:
        """Cartesian ``_crop_weights``: :meth:`_crop_weights_along` on the spatial shape."""
        return type(self)._crop_weights_along(
            self.shape[2:], level, border, owner=f"{type(self).__name__}._crop_weights"
        )

    @classmethod
    def _crop_weights_along(
        cls, shape: tuple[int, ...], level: int, border: int, *, owner: str
    ) -> NDArray[np.float64]:
        """0 on ``border`` pixels at both ends of every axis of the grid of full-resolution
        ``shape`` taken at ``level``, 1 elsewhere (maths.md M-BORDER; M-BORDER-TENSORPROD for one
        block of a tensor product; STL ``_crop``).

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
                "sizes must be divisible by 2^level (maths.md M-DATA-2D, OP-8). Use a smaller "
                "level."
            )
        grid = tuple(n // 2 ** int(level) for n in full)
        width = int(border)
        if width and min(grid) <= 2 * width:
            reduced = (min(grid) - 1) // 2
            warnings.warn(
                f"{owner.split('.')[0]}: at level {level} the spatial shape {grid} is too small "
                f"for a border crop of {width} pixel(s); using {reduced} instead (maths.md "
                "M-BORDER, OP-13). Use a smaller J, larger data or pbc=True to avoid it.",
                UserWarning,
                stacklevel=3,
            )
            width = reduced
        inner = np.ones(tuple(n - 2 * width for n in grid), dtype=np.float64)
        return np.pad(inner, width)

    def __repr__(self) -> str:
        n_missing = 0 if self._mask is None else int(self._mask.sum())
        return (
            f"{type(self).__name__}(shape={self.shape}, pbc={self._pbc}, "
            f"missing={n_missing}, backend={self._backend!r})"
        )


# ---------------------------------------------------------------------- the pre-filter of syntheses


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
    transform. The mask is static, built in NumPy from the spatial shape."""
    xp = backend.xp
    axes = (-2, -1)
    shape = (int(x.shape[-2]), int(x.shape[-1]))
    spectrum = xp.fft.fftn(x, axes=axes)
    kept = backend.asarray(nyquist_mask(shape), "bool", _registry.device_of(spectrum))
    filtered = xp.where(kept, spectrum, xp.zeros_like(spectrum))
    return xp.real(xp.fft.ifftn(filtered, axes=axes))


class Data2D(DataCartesian):
    """2D data, shape ``(Nb, Nc, N1, N2)``.

    ``Data2D(array, mask=None, *, pbc)``.

    Parameters
    ----------
    array : array_like
        As for :class:`estel.Data`. A 2D array is promoted to ``(1, 1, N1, N2)``, a 3D array to
        ``(1, Nc, N1, N2)`` (maths.md M-DATA-2D).
    mask : array_like or None
        As for :class:`estel.Data`.
    pbc : bool
        As for :class:`estel.Data`.
    """

    ndim: ClassVar[int] = 2
    """The number of spatial axes of the array, 2."""
    _nan_threshold: ClassVar[float] = 0.33
    _geometry: ClassVar[str] = "2d"
    _block_tensor_prod: ClassVar[bool] = True

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)

    def _nyquist_filter(self, x: Any) -> Any:
        """The pre-filter of M-SYN-NYQ: :func:`nyquist_filter` on the two spatial axes."""
        return nyquist_filter(_registry.backend_named(self._backend), x)


class Data1D(DataCartesian):
    """1D data, shape ``(Nb, Nc, N)``.

    ``Data1D(array, mask=None, *, pbc)``.

    Parameters
    ----------
    array : array_like
        As for :class:`estel.Data`. A 1D array is promoted to ``(1, 1, N)``; a 2D array to
        ``(1, Nc, N)``, its first axis read as channels, not as a batch (maths.md M-DATA-1D,
        OP-29).
    mask : array_like or None
        As for :class:`estel.Data`.
    pbc : bool
        As for :class:`estel.Data`.
    """

    ndim: ClassVar[int] = 1
    """The number of spatial axes of the array, 1."""
    _nan_threshold: ClassVar[float] = 0.5
    _geometry: ClassVar[str] = "1d"
    _block_tensor_prod: ClassVar[bool] = True

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)
