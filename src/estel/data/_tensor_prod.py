"""Tensor products of planar geometries: ``DataTensorProd`` (Part II, §12.4; maths.md M-NOT-TP,
M-DATA-TP).

A tensor-product field is one array whose spatial axes are those of its blocks, in block order:
``(Nb, Nc, *N⁽⁰⁾, …, *N⁽ᴮ⁻¹⁾)``, for example ``[Data2D, Data1D]`` for space and time. The
object **composes** the geometries of its blocks: a static description per block
(:class:`BlockSpecTensorProd`: its class, its axes, its ``pbc``), never a class created at run
time nor an assignment to ``__class__`` (Part I, §2.4).

Its geometry hooks ``_conv``, ``_downsample`` and ``_crop_weights`` act on one block at a time,
named by ``block=a``: they dispatch to the geometry of that block, along that block's own axes
of the array, without moving any axis (no ``swapaxes``, unlike STL's ``_as_data_class_of``).
Like every hook of filtering and downsampling, only the wavelet operator calls them
(``estel.wavelets._operator_tensor_prod``; Part I, §2.3, §2.4).

Missing values are not supported in tensor products in 1.0.0 (maths.md OP-41; LF8): data that
hold a NaN, or are given a mask, raise at construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ._base import Data, DownsampleWeights, _promote_spatial, _to_backend
from ._cartesian import DataCartesian


class BlockSpecTensorProd(NamedTuple):
    """The static description of one block of a tensor product: hashable, compared by value,
    so that it is pytree metadata (Part I, §2.3)."""

    geometry: type[DataCartesian]
    """The geometry of the block, ``Data2D`` or ``Data1D``."""
    axes: tuple[int, ...]
    """The axes of the block in the array ``(Nb, Nc, *spatial)``, contiguous and increasing."""
    pbc: bool
    """Whether the block is periodic."""


class DataTensorProd(Data):
    """A tensor product of planar geometries, one array (maths.md M-DATA-TP).

    Parameters
    ----------
    array:
        The data, real-valued, as for :class:`estel.Data`: the spatial axes of the blocks, in
        block order, with no axis between them. An array with the spatial axes only is
        promoted to ``(1, 1, *spatial)``, one with a single leading axis to
        ``(1, Nc, *spatial)``; the result is ``(Nb, Nc, *spatial)``.
    blocks:
        The geometries of the blocks, in axis order: a list of ``estel.Data2D`` and
        ``estel.Data1D``, in any order, for example ``[Data2D, Data1D]`` for space and time.
        A single block is accepted (maths.md OP-82).
    mask:
        Not supported: missing values in tensor products come after 1.0.0 (LF8). Given, it
        raises. Its place is that of ``Data2D(array, mask=None, *, pbc)``, so that LF8 can
        fill it without changing the signature.
    pbc:
        Keyword only, as for ``Data2D``: one boolean per block, written explicitly, ``True``
        (periodic) or ``False`` (maths.md OP-4).
    """

    _geometry: ClassVar[str] = "tensor_prod"

    __slots__ = ("_blocks",)

    _blocks: tuple[BlockSpecTensorProd, ...]

    def __init__(
        self,
        array: Any,
        blocks: Sequence[type[Data]],
        mask: Any = None,
        *,
        pbc: Sequence[bool],
    ) -> None:
        cls = type(self)
        geometries = _check_blocks(blocks)
        flags = _check_pbc(pbc, len(geometries))
        if mask is not None:
            raise NotImplementedError(
                "DataTensorProd: missing values are not supported in tensor products in ESTEL "
                "1.0.0; they come later (LF8; maths.md OP-41). Pass mask=None, and fill or crop "
                "the missing values first."
            )
        ndim = sum(g.ndim for g in geometries)
        backend_name = _registry.get_backend()
        values = _to_backend(array, backend_name, cls)
        values = _promote_spatial(values, ndim, cls.__name__, _accepted_shapes(geometries))
        backend = _registry.current_backend()
        nan = np.asarray(backend.to_numpy(backend.xp.isnan(values)), dtype=bool)
        if nan.any():
            raise NotImplementedError(
                f"DataTensorProd: the array holds {int(nan.sum())} NaN. Missing values are not "
                "supported in tensor products in ESTEL 1.0.0; they come later (LF8; maths.md "
                "OP-41). Replace the NaN with values, or crop the data to a part without them."
            )
        specs = []
        first = 2
        for geometry, flag in zip(geometries, flags, strict=True):
            specs.append(
                BlockSpecTensorProd(geometry, tuple(range(first, first + geometry.ndim)), flag)
            )
            first += geometry.ndim
        self._set_blocks(values, tuple(specs), backend_name)

    # ------------------------------------------------------------------ construction

    def _set_blocks(
        self, array: Any, blocks: tuple[BlockSpecTensorProd, ...], backend: str
    ) -> None:
        # The pbc of the mother class is one boolean; a tensor product has one per block, held
        # by its blocks, and no mask (LF8).
        object.__setattr__(self, "_array", array)
        object.__setattr__(self, "_mask", None)
        object.__setattr__(self, "_pbc", tuple(b.pbc for b in blocks))
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_blocks", blocks)

    @classmethod
    def _from_blocks(
        cls, array: Any, blocks: tuple[BlockSpecTensorProd, ...], backend: str
    ) -> Data:
        """A new object from parts already validated: no conversion, no check of values, so
        that it works inside traced code."""
        new = cls.__new__(cls)
        new._set_blocks(array, blocks, backend)
        return new

    def _like(self, array: Any) -> Data:
        return self._from_blocks(array, self._blocks, self._backend)

    def _rebuild_advice(self) -> str:
        return "DataTensorProd(array, data.blocks, pbc=data.pbc)"

    # ------------------------------------------------------------------ accessors

    @property
    def ndim(self) -> int:
        """The number of spatial axes, those of every block."""
        return sum(len(b.axes) for b in self._blocks)

    @property
    def blocks(self) -> tuple[type[Data], ...]:
        """The geometries of the blocks, in axis order: ``Data2D`` and ``Data1D``."""
        return tuple(b.geometry for b in self._blocks)

    @property
    def pbc(self) -> tuple[bool, ...]:
        """Whether each block is periodic, one boolean per block."""
        return tuple(b.pbc for b in self._blocks)

    @property
    def _block_specs(self) -> tuple[BlockSpecTensorProd, ...]:
        """The static description of each block: its geometry, its axes and its ``pbc``."""
        return self._blocks

    # ------------------------------------------------------------------ pytree

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        """The array is the only leaf; the blocks and the backend are static (Part I, §2.3)."""
        return (self._array,), (self._blocks, self._backend)

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> Data:
        blocks, backend = aux
        return cls._from_blocks(children[0], blocks, backend)

    # ------------------------------------------------------------------ geometry hooks

    # Only the wavelet operator of a tensor product (``estel.wavelets._operator_tensor_prod``) calls
    # these hooks (Part I, §2.3, §2.4). Each acts on one block, named by ``block``, at the level
    # ``level`` of that block, and dispatches to the geometry of the block, along its axes.

    def _conv(self, x: Any, tables: Any, level: int, *, block: int | None = None) -> Any:
        """The block convolution ⋆_a of maths.md M-NOT-TP, a = ``block``: the filtering of the
        geometry of block a (M-CONV-2D, or its 1D form), with the padding of that block, along
        the axes of that block only; ``tables`` are the wavelet bank of block a, and ``level``
        its level, which the planar kernels do not depend on.

        ``x`` is ``(Nb, Nc, *spatial)``, real or complex, at any level of the other blocks; the
        result is complex, ``(Nb, L_a, Nc, *spatial)``, the orientation of this convolution
        right after the batch axis, as for planar data.
        """
        spec = self._spec(block, "_conv")
        self._check_array(x, "_conv")
        return spec.geometry._conv_along(
            x,
            tables,
            axes=spec.axes,
            pbc=spec.pbc,
            backend=self.backend,
            owner=f"DataTensorProd._conv (block {block})",
        )

    def _downsample(
        self, x: Any, weights: DownsampleWeights, level: int, *, block: int | None = None
    ) -> Any:
        """One downsampling step along block a = ``block``, from ``level`` to ``level + 1`` of
        that block (maths.md M-DOWN along the axes of block a, M-NOT-TP): the other axes are
        carried through. ``weights`` holds the smoothing kernel of block a; tensor products
        have no missing values, so no maps (LF8)."""
        spec = self._spec(block, "_downsample")
        self._check_array(x, "_downsample")
        if weights.valid is not None or weights.reweight is not None:
            raise NotImplementedError(
                "DataTensorProd._downsample: tensor products have no missing values in ESTEL "
                "1.0.0 (LF8; maths.md OP-41), so a step has no maps: pass "
                "DownsampleWeights(kernel) only."
            )
        return spec.geometry._downsample_along(
            x,
            weights,
            axes=spec.axes,
            pbc=spec.pbc,
            backend=self.backend,
            owner=f"DataTensorProd._downsample (block {block})",
        )

    def _crop_weights(
        self, level: int, border: int, *, block: int | None = None
    ) -> NDArray[np.float64]:
        """The border crop of block a = ``block`` at its level ``level`` (maths.md
        M-BORDER-TP): a NumPy float64 array of the shape of block a at that level, 0 on
        ``border`` pixels at both ends of each of its axes, reduced as in M-BORDER when too
        large for the block (OP-13). The crop of the field is the outer product of those of its
        blocks."""
        spec = self._spec(block, "_crop_weights")
        return spec.geometry._crop_weights_along(
            tuple(self.shape[a] for a in spec.axes),
            level,
            border,
            owner=f"DataTensorProd._crop_weights (block {block})",
        )

    def _spec(self, block: object, hook: str) -> BlockSpecTensorProd:
        n = len(self._blocks)
        if block is None:
            raise TypeError(
                f"DataTensorProd.{hook} acts on one block at a time: pass block=a, with "
                f"0 ≤ a < {n} (maths.md M-NOT-TP)."
            )
        if isinstance(block, bool) or not isinstance(block, int | np.integer):
            raise TypeError(f"DataTensorProd.{hook}(block={block!r}): pass an int, 0 ≤ a < {n}.")
        if not 0 <= int(block) < n:
            raise ValueError(
                f"DataTensorProd.{hook}(block={block}): this tensor product has {n} block(s), "
                f"0 … {n - 1}."
            )
        return self._blocks[int(block)]

    def _check_array(self, x: Any, hook: str) -> None:
        _registry.check_backend(self.backend, x, owner=f"DataTensorProd.{hook}")
        if len(x.shape) != 2 + self.ndim:
            raise ValueError(
                f"DataTensorProd.{hook}: the array has shape {tuple(x.shape)}; pass "
                f"(Nb, Nc, *spatial), with the {self.ndim} spatial axes of the blocks."
            )

    def __repr__(self) -> str:
        names = ", ".join(g.__name__ for g in self.blocks)
        return (
            f"DataTensorProd(shape={self.shape}, blocks=[{names}], pbc={list(self.pbc)}, "
            f"backend={self._backend!r})"
        )


# ---------------------------------------------------------------------- validation


def _check_blocks(blocks: object) -> tuple[type[DataCartesian], ...]:
    """The geometries of the blocks: a non-empty list of ``Data2D`` and ``Data1D``, the classes
    that declare the capability of a block, ``_block_tensor_prod``, in their own body."""
    example = "for example blocks=[estel.Data2D, estel.Data1D] for space and time"
    if isinstance(blocks, str) or not isinstance(blocks, Sequence):
        raise TypeError(
            f"DataTensorProd(blocks={blocks!r}): pass the geometries of the blocks as a list, "
            f"{example} (maths.md M-DATA-TP)."
        )
    if not blocks:
        raise ValueError(f"DataTensorProd(blocks=[]): pass at least one block, {example}.")
    for index, block in enumerate(blocks):
        # The capability is read in the body of the class itself, never inherited: Data2D and
        # Data1D, and neither DataCartesian nor a subclass of them (Part I, §2.3.1). Any class
        # of data may declare it (LF9: DataHealpix).
        accepted = (
            isinstance(block, type)
            and issubclass(block, Data)
            and block.__dict__.get("_block_tensor_prod", False) is True
        )
        if not accepted:
            raise TypeError(
                f"DataTensorProd: block {index} is {block!r}; each block is the class "
                "estel.Data2D or estel.Data1D, not an instance (maths.md M-DATA-TP). HEALPix "
                f"blocks come later (LF9). Pass the classes, {example}."
            )
    return tuple(blocks)


def _check_pbc(pbc: object, n_blocks: int) -> tuple[bool, ...]:
    """One explicit boolean per block (maths.md OP-4)."""
    if isinstance(pbc, bool | np.bool_) or not isinstance(pbc, Sequence) or isinstance(pbc, str):
        raise TypeError(
            f"DataTensorProd(pbc={pbc!r}): pass one boolean per block, written explicitly, as "
            f"a list of {n_blocks}: for example pbc=[True, False] (maths.md OP-4)."
        )
    if len(pbc) != n_blocks:
        raise ValueError(
            f"DataTensorProd(pbc={list(pbc)!r}): {len(pbc)} value(s) for {n_blocks} block(s); "
            "pass one boolean per block (maths.md OP-4)."
        )
    for index, flag in enumerate(pbc):
        if not isinstance(flag, bool | np.bool_):
            raise TypeError(
                f"DataTensorProd: pbc[{index}] is {flag!r}; each block's pbc is True (periodic) "
                "or False, written explicitly (maths.md OP-4)."
            )
    return tuple(bool(flag) for flag in pbc)


def _accepted_shapes(geometries: tuple[type[DataCartesian], ...]) -> str:
    """The shapes the constructor accepts, for error messages."""
    names: list[str] = []
    for index, geometry in enumerate(geometries):
        names.extend(
            f"N{index}" if geometry.ndim == 1 else f"N{index}_{i + 1}" for i in range(geometry.ndim)
        )
    spatial = ", ".join(names)
    return f"({spatial}), (Nc, {spatial}) or (Nb, Nc, {spatial})"
