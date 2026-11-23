"""The wavelet operator of a tensor product, composed from those of its blocks (Part I, §2.3;
Part II, §12.5; maths.md M-NOT-TP, M-W1-TP, M-DEF-TP; task T3.1).

Internal, like :class:`~estel.wavelets._operator.WaveletOperator`. A tensor product of B blocks
(``DataTensorProd``) has one J and one L per block, and its levels are multi-indices
d = (d₀, …, d_{B-1}): the field downsampled d_a times along block a (M-NOT-TP). Each block has
the static part and the arrays of a planar wavelet operator, built for its geometry, its sizes
and its ``pbc`` (:class:`WaveletPlan`, :class:`WaveletFilters`); the operator of the tensor
product holds them in block order and applies them along the axes of each block, through the
hooks of ``DataTensorProd``, which dispatch to the geometry of the block:

* :meth:`TensorProdWaveletOperator.apply`, the block convolution ψ⁽ᵃ⁾ ⋆_a of M-W1-TP, at the
  level of the scale j_a in block a and at any level of the other blocks;
* :meth:`TensorProdWaveletOperator.downsample`, one step along block a;
* :meth:`TensorProdWaveletOperator.crop_weights`, the border crop of M-BORDER-TP, the outer
  product of the crops of the blocks.

* :meth:`TensorProdWaveletOperator.convolve`, the same convolution with its orientation right
  after the batch axis, as ``WaveletOperator.apply`` places it, for the scattering loop;
* :meth:`TensorProdWaveletOperator.lowpass`, the last step of the low-pass filter φ⁽ᵃ⁾ of block
  a (M-LP), realised as the plan of the block says (``lowpass_realization``,
  ``lowpass_level``).

The wavelet W_j of M-W1-TP is the product of the block convolutions, applied block after
block in block order, block 0 first, as STL does; the scattering loop of task T3.2 composes
them (``estel.scattering._loop_tp``). A block has a low-pass filter when ``lowpass="ST"`` and
its sizes are divisible by 2^J_a (M-LP, OP-34): its plan then goes down to level J_a, one
downsampling more than its wavelets need. Tensor products have no missing values in 1.0.0
(LF8), so the blocks have no masks.

**Defaults** (M-DEF-TP): J_a = ⌊log₂ min N⁽ᵃ⁾⌋ - 2, the rule of planar data, shared with
``_operator.py``, and L_a = 4 for a 2D block, 2 for a 1D block. **Arguments** (OP-83,
OP-8): J and L are lists with one entry per block, an entry ``None`` giving the default of the
block; they are validated when the operator is built, with the sizes of each block divisible
by 2^(J_a - 1), and the caller's lists are never modified.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType, default_device, default_dtype
from ..backend import _registry
from ..data._base import Data, DownsampleWeights
from ..data._planar import Data1D
from ..data._tensor_prod import DataTensorProd
from ._banks import KERNEL_SIZE, _check_size
from ._operator import (
    WaveletFilters,
    WaveletPlan,
    _check_scales,
    _check_wavelet,
    _is_index,
    _wavelet_plan,
    build_wavelet_filters,
)

Level = tuple[int, ...]
"""A multi-index level d = (d₀, …, d_{B-1}): the field downsampled d_a times along block a
(maths.md M-NOT-TP)."""


# ---------------------------------------------------------------------- defaults


def default_block_orientations(geometry: type[Data]) -> int:
    """L_a = 4 for a 2D block, 2 for a 1D block, not 1 as for 1D data alone (maths.md
    M-DATA-TP, M-DEF-TP, M-W1-TP)."""
    return 2 if geometry is Data1D else 4


# ---------------------------------------------------------------------- the static part


@dataclass(frozen=True, eq=False)
class TensorProdWaveletPlan:
    """The static part of the wavelet operator of a tensor product: the plans of its blocks,
    in block order, and the axes of each block. NumPy only, immutable and hashable; two plans
    are equal when they were built from the same parameters and the same static metadata of
    the data."""

    blocks: tuple[WaveletPlan, ...]
    """The plan of each block, built for its geometry, its sizes and its ``pbc``."""
    axes: tuple[tuple[int, ...], ...]
    """The axes of each block in the array ``(Nb, Nc, *spatial)``."""
    _key: tuple[Any, ...] = field(repr=False)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorProdWaveletPlan):
            return NotImplemented
        return self._key == other._key

    @property
    def J(self) -> tuple[int, ...]:  # noqa: N802 (maths.md notation)
        """The number of scales of each block."""
        return tuple(b.J for b in self.blocks)

    @property
    def L(self) -> tuple[int, ...]:  # noqa: N802
        """The number of orientations of each block."""
        return tuple(b.L for b in self.blocks)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """The spatial shape of the data at full resolution."""
        return tuple(n for b in self.blocks for n in b.spatial_shape)

    @property
    def lowpass_blocks(self) -> tuple[bool, ...]:
        """Whether each block has a low-pass filter, its plan going down to its level
        (maths.md M-LP)."""
        return tuple(b.has_lowpass for b in self.blocks)

    def block_crop(self, block: int, level: int, border: int) -> NDArray[np.float64]:
        """The border crop of block a = ``block`` alone at its level ``level`` (maths.md
        M-BORDER-TP): a NumPy float64 array of the shape of that block at that level, 0 on
        ``border`` pixels at both ends of each of its axes, a border too large for the block
        reduced as in M-BORDER, with a warning (OP-13). The crop of a map is the outer product
        of those of its blocks; the scattering loop multiplies a map by them block by block.
        NumPy only: called when a plan is built."""
        plan = self.blocks[block]
        return plan.geometry._crop_weights_along(
            plan.spatial_shape, level, border, owner=f"DataTensorProd (block {block})"
        )

    def level_shape(self, level: Level) -> tuple[int, ...]:
        """The spatial shape at the multi-index ``level``: each block at its own level."""
        self.check_level(level)
        return tuple(n for b, d in zip(self.blocks, level, strict=True) for n in b.level_shapes[d])

    def check_level(self, level: object) -> None:
        """Raise unless ``level`` is a multi-index level of this operator: one level per block,
        each one of the levels of its block, 0 … J_a - 1."""
        n = len(self.blocks)
        if not isinstance(level, tuple) or len(level) != n or not all(_is_index(d) for d in level):
            raise TypeError(
                f"level={level!r}: a level of a tensor product of {n} block(s) is a tuple of "
                f"{n} ints, one per block (maths.md M-NOT-TP)."
            )
        for a, (plan, d) in enumerate(zip(self.blocks, level, strict=True)):
            if d not in plan.levels:
                raise ValueError(
                    f"level={level}: block {a} has the levels {plan.levels[0]} … "
                    f"{plan.levels[-1]} (J_{a} = {plan.J})."
                )


def build_tensor_prod_plan(
    data: DataTensorProd,
    J: Sequence[int | None] | None = None,  # noqa: N803 (maths.md notation)
    L: Sequence[int | None] | None = None,  # noqa: N803
    wavelet: str = "bump_steerable",
    *,
    kernel_size: int = KERNEL_SIZE,
    lowpass: str | None = None,
    stacklevel: int = 1,
) -> TensorProdWaveletPlan:
    """The :class:`TensorProdWaveletPlan` of ``data``, with the J and L of each block; the values
    of ``data`` are not read. The defaults are those of M-DEF-TP; the arguments are validated
    here, with the sizes of each block divisible by 2^(J_a - 1) (OP-8, OP-83).

    With ``lowpass="ST"``, each block whose sizes are divisible by 2^J_a has a low-pass filter,
    and its plan goes down to level J_a (maths.md M-LP); each other block warns once, as STL
    does, and has none (OP-34). With ``None`` (the default here), no block has one.
    ``stacklevel`` places that warning, as for ``warnings.warn``, from the caller of this
    function: 1 for that caller, 2 for its caller."""
    if not isinstance(data, DataTensorProd):
        raise TypeError(
            f"TensorProdWaveletOperator takes a DataTensorProd, got {type(data).__name__}; for "
            "1D and 2D data, use WaveletOperator."
        )
    _check_wavelet(wavelet)
    specs = data._block_specs
    shapes = tuple(tuple(data.shape[a] for a in spec.axes) for spec in specs)
    n_blocks = len(specs)
    scales = _per_block(J, "J", n_blocks, ", ".join("2" for _ in specs))
    orientations = _per_block(
        L, "L", n_blocks, ", ".join(str(default_block_orientations(s.geometry)) for s in specs)
    )
    size = _check_size(kernel_size)
    plans = []
    for a, (spec, shape, j_value, l_value) in enumerate(
        zip(specs, shapes, scales, orientations, strict=True)
    ):
        what = f"block {a} ({spec.geometry.__name__} of shape {shape})"
        n_scales = _check_scales(j_value, shape, f"J_{a}", f"J[{a}]", what, tensor_prod=True)
        n_orientations = _check_block_orientations(l_value, spec.geometry, what, a)
        has_lowpass = lowpass == "ST" and all(n % 2**n_scales == 0 for n in shape)
        if lowpass == "ST" and not has_lowpass:
            warnings.warn(
                f"block {a} ({spec.geometry.__name__} of shape {shape}) has no low-pass filter: "
                f"its sizes are not all divisible by 2^J_{a} = {2**n_scales}, which the low-pass "
                "filter needs, one downsampling more than its wavelets (maths.md M-LP, OP-34). "
                f"Its S1 and S2 have the wavelet scales only; use a size divisible by "
                f"{2**n_scales}, or J_{a} = {n_scales - 1}, to keep it.",
                UserWarning,
                stacklevel=stacklevel + 1,
            )
        plans.append(
            _wavelet_plan(
                spec.geometry,
                shape,
                spec.pbc,
                None,
                n_scales,
                n_orientations,
                wavelet,
                size,
                lowpass=has_lowpass,
            )
        )
    axes = tuple(spec.axes for spec in specs)
    return TensorProdWaveletPlan(blocks=tuple(plans), axes=axes, _key=(tuple(plans), axes))


def _per_block(value: object, name: str, n_blocks: int, example: str) -> tuple[object, ...]:
    """``J`` or ``L`` as a tuple of ``n_blocks`` entries, one per block, ``None`` for a
    default; a copy, so that the caller's list is never modified (OP-83). ``example`` is a
    valid list, for the messages."""
    if value is None:
        return (None,) * n_blocks
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"{name}={value!r}: a tensor product takes one {name} per block, as a list of "
            f"{n_blocks}, for example {name}=[{example}]; None, or an entry None, gives the "
            "default of the block (maths.md M-DATA-TP, OP-83)."
        )
    if len(value) != n_blocks:
        raise ValueError(
            f"{name}={list(value)!r}: {len(value)} entry(ies) for {n_blocks} block(s); pass one "
            f"per block, for example {name}=[{example}], None for the default of a block "
            "(maths.md M-DATA-TP, OP-83)."
        )
    return tuple(value)


def _check_block_orientations(value: object, geometry: type[Data], what: str, a: int) -> int:
    """L_a as an int: the default of the block if ``value`` is None, at least 1, and 1 or 2 for
    a 1D block (maths.md M-DATA-TP, M-WAV-1D)."""
    if value is None:
        return default_block_orientations(geometry)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"L[{a}]={value!r}: the number of orientations of {what} is an integer ≥ 1, or None "
            "for its default (maths.md M-DATA-TP)."
        )
    n = int(value)
    if geometry is Data1D and n not in (1, 2):
        raise ValueError(
            f"L[{a}]={n}: {what} has L = 1 or L = 2 (the default), the analytic wavelet and its "
            "conjugate (maths.md M-DATA-TP, M-WAV-1D)."
        )
    if n < 1:
        raise ValueError(f"L[{a}]={n}: the number of orientations of {what} is at least 1.")
    return n


# ---------------------------------------------------------------------- the arrays


def build_tensor_prod_filters(
    plan: TensorProdWaveletPlan, backend_name: str, dtype: FloatDType, device: Any
) -> tuple[WaveletFilters, ...]:
    """The arrays of each block of ``plan``: its wavelet bank and its smoothing kernel, on the
    backend ``backend_name``, cast once to ``dtype`` on ``device``."""
    return tuple(build_wavelet_filters(b, backend_name, dtype, device) for b in plan.blocks)


# ---------------------------------------------------------------------- the operator


class TensorProdWaveletOperator:
    """The wavelets of a tensor product, block by block: internal (Part II, §12.5).

    ``TensorProdWaveletOperator(data, J, L)`` builds the static part and, on the backend of
    ``data`` in the default dtype, the arrays.

    Parameters
    ----------
    data:
        A ``DataTensorProd``: its blocks, sizes, ``pbc`` and backend fix the operator; its
        values are not read.
    J:
        One number of scales per block, as a list; an entry ``None``, or ``J=None``, gives the
        default of the block, ⌊log₂ min N⁽ᵃ⁾⌋ - 2 (M-DEF-TP).
    L:
        One number of orientations per block, as a list; by default 4 for a 2D block and 2 for
        a 1D block.
    wavelet:
        ``"bump_steerable"``, the only wavelets of 1.0.0.
    lowpass:
        ``"ST"``: the low-pass filter of each block whose sizes allow it (M-LP); ``None``, the
        default here: none.
    """

    __slots__ = ("_data", "_filters", "_plan")

    _data: DataTensorProd
    _plan: TensorProdWaveletPlan
    _filters: tuple[WaveletFilters, ...] | None

    def __init__(
        self,
        data: DataTensorProd,
        J: Sequence[int | None] | None = None,  # noqa: N803 (maths.md notation)
        L: Sequence[int | None] | None = None,  # noqa: N803
        wavelet: str = "bump_steerable",
        *,
        kernel_size: int = KERNEL_SIZE,
        lowpass: str | None = None,
    ) -> None:
        plan = build_tensor_prod_plan(
            data, J, L, wavelet, kernel_size=kernel_size, lowpass=lowpass, stacklevel=2
        )
        filters = build_tensor_prod_filters(plan, data.backend, default_dtype(), default_device())
        self._bind(data, plan, filters)

    def _bind(
        self,
        data: DataTensorProd,
        plan: TensorProdWaveletPlan,
        filters: tuple[WaveletFilters, ...] | None,
    ) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_filters", filters)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TensorProdWaveletOperator is immutable: build a new one instead.")

    @classmethod
    def from_parts(
        cls,
        data: DataTensorProd,
        plan: TensorProdWaveletPlan,
        filters: tuple[WaveletFilters, ...] | None = None,
    ) -> TensorProdWaveletOperator:
        """The operator of a static part and its arrays, bound to ``data``, which must have
        the static metadata ``plan`` was built from: no validation and no conversion, so that
        it works inside traced code. Without ``filters``, the static rules only."""
        new = cls.__new__(cls)
        new._bind(data, plan, filters)
        return new

    @property
    def plan(self) -> TensorProdWaveletPlan:
        """The static part."""
        return self._plan

    @property
    def filters(self) -> tuple[WaveletFilters, ...] | None:
        """The arrays of each block, or ``None`` for an operator bound to its static part
        only."""
        return self._filters

    # ------------------------------------------------------------------ static rules

    def crop_weights(self, level: Level, border: tuple[int, ...]) -> NDArray[np.float64]:
        """The border crop of maths.md M-BORDER-TP at the multi-index ``level``, as weights: a
        NumPy float64 array of the spatial shape at that level, the outer product of the crops
        of the blocks, block a cropped by ``border[a]`` pixels at both ends of each of its axes.

        A border too large for a block is reduced as in M-BORDER, block by block, with one
        warning for all the blocks (OP-13). NumPy only: called once, when a plan is built.
        """
        plan = self._plan
        plan.check_level(level)
        n = len(plan.blocks)
        if (
            not isinstance(border, tuple)
            or len(border) != n
            or not all(_is_index(b) for b in border)
        ):
            raise TypeError(
                f"crop_weights(border={border!r}): pass one border width per block, a tuple of "
                f"{n} ints ≥ 0, in pixels at the level of that block (maths.md M-BORDER-TP)."
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crops = [
                self._data._crop_weights(d, b, block=a)
                for a, (d, b) in enumerate(zip(level, border, strict=True))
            ]
        if caught:
            warnings.warn(caught[0].message, stacklevel=2)
        weights: NDArray[np.float64] = reduce(np.multiply.outer, crops)
        return weights

    # ------------------------------------------------------------------ pure functions

    def apply(self, x: Any, j: int, *, block: int, level: Level) -> Any:
        """ψ⁽ᵃ⁾_{j} ⋆_a x, the block convolution of maths.md M-W1-TP along block a = ``block``
        at its scale ``j``: complex, L_a orientations.

        ``x`` is ``(Nb, Nc, *O, *spatial)`` at the multi-index ``level``, whose entry for block
        a is the level of scale j in that block (d_a = j_a in 1.0.0) and whose other entries
        are any levels of their blocks, real or complex, with any axes ``O`` in between (the
        orientations of earlier blocks or of a first layer). The result is
        ``(Nb, Nc, *O, L_a, *spatial)``: the orientation of this convolution after those of
        the earlier blocks, so that the blocks applied in block order give the layout of
        M-W1-TP, ``(Nb, Nc, L₀, …, L_{B-1}, *spatial)``. Calls the hook ``_conv`` of the
        block, with the bank of the block; the hook places the orientation right after the
        batch axis, as for planar data, and this method moves it in front of the spatial axes.
        Pure: nothing is modified, nothing depends on array values.
        """
        lead, spatial = self._split(x, level, "apply")
        out = self.convolve(x, j, block=block, level=level)  # (Nb, L_a, Nc, *O, *spatial)
        xp = _registry.backend_named(self._data.backend).xp
        # (Nb, L_a, Nc·O, *spatial) -> (Nb, Nc·O, L_a, *spatial) -> (Nb, Nc, *O, L_a, *spatial)
        n_spatial = len(spatial)
        out = xp.reshape(out, (lead[0], int(out.shape[1]), -1, *spatial))
        out = xp.permute_dims(out, (0, 2, 1, *range(3, 3 + n_spatial)))
        return xp.reshape(out, (*lead, self._plan.blocks[block].L, *spatial))

    def convolve(self, x: Any, j: int, *, block: int, level: Level) -> Any:
        """ψ⁽ᵃ⁾_{j} ⋆_a x as :meth:`apply`, with the orientation of this convolution right after
        the batch axis, as ``WaveletOperator.apply`` places it: ``x`` is ``(Nb, *O, *spatial)``
        at the multi-index ``level``, with any axes ``O``, and the result is
        ``(Nb, L_a, *O, *spatial)``, without a transposition. The scattering loop of tensor
        products (task T3.2) chains these block after block, so that the orientations of the
        blocks come in reverse block order, ``(Nb, L_{B-1}, …, L₀, *O, *spatial)``; the loop
        orders the means it takes. Pure."""
        plan, filters = self._plan, self._arrays("apply")
        a = self._block(block, "apply")
        block_plan = plan.blocks[a]
        if not _is_index(j) or not 0 <= j < block_plan.J:
            raise ValueError(
                f"apply(j={j!r}, block={a}): block {a} has the scales 0 … {block_plan.J - 1} "
                f"(J_{a} = {block_plan.J})."
            )
        lead, spatial = self._split(x, level, "apply", minimum=1)
        expected = block_plan.scale_levels[j]
        if level[a] != expected:
            raise ValueError(
                f"apply(j={j}, block={a}, level={level}): scale {j} of block {a} is computed at "
                f"level {expected} of that block, so pass a map whose level has d_{a} = "
                f"{expected}; bring it there with downsample(..., block={a}) first (maths.md "
                "M-W1-TP)."
            )
        xp = _registry.backend_named(self._data.backend).xp
        flat = xp.reshape(x, (lead[0], -1, *spatial))
        out = self._data._conv(flat, filters[a].kernels, level[a], block=a)
        return xp.reshape(out, (lead[0], block_plan.L, *lead[1:], *spatial))

    def lowpass(self, x: Any, level: Level, *, block: int) -> Any:
        """The last step of the low-pass filter φ⁽ᵃ⁾ of block a = ``block`` (maths.md M-LP), from
        the multi-index ``level``, whose entry for block a is the level before
        ``lowpass_level`` of that block, to the level with d_a = ``lowpass_level``.

        Realised as the plan of the block says (``lowpass_realization``): in 1.0.0,
        ``"smoothing_cascade"``, one smoothing step with its downsampling, so that φ⁽ᵃ⁾ is the
        cascade of the smoothing kernels of J_a downsamplings, as in STL; computing it without
        the final downsampling (LF6, M-LP2) changes the plan, not the loop. ``x`` is
        ``(Nb, *O, *spatial)``, real or complex; the result has the spatial shape of the new
        level. Pure."""
        plan, filters = self._plan, self._arrays("lowpass")
        a = self._block(block, "lowpass")
        block_plan = plan.blocks[a]
        if not block_plan.has_lowpass:
            raise ValueError(
                f"lowpass(block={a}): block {a} has no low-pass filter: lowpass=None, or its "
                f"sizes are not divisible by 2^J_{a} = {2**block_plan.J} (maths.md M-LP, OP-34)."
            )
        before = block_plan.lowpass_level - 1
        if level[a] != before:
            raise ValueError(
                f"lowpass(level={level}, block={a}): the last step of the low-pass filter of "
                f"block {a} starts from its level {before}, the level of its last wavelet scale "
                f"(maths.md M-LP); bring the map there with downsample(..., block={a}) first."
            )
        if block_plan.lowpass_realization != "smoothing_cascade":  # pragma: no cover (LF6)
            raise NotImplementedError(
                f"lowpass: the realisation {block_plan.lowpass_realization!r} comes with LF6."
            )
        lead, spatial = self._split(x, level, "lowpass", minimum=1)
        xp = _registry.backend_named(self._data.backend).xp
        flat = xp.reshape(x, (lead[0], -1, *spatial))
        weights = DownsampleWeights(filters[a].smoothing)
        out = self._data._downsample(flat, weights, level[a], block=a)
        return xp.reshape(out, (*lead, *tuple(out.shape[2:])))

    def downsample(self, x: Any, level: Level, *, block: int) -> Any:
        """One downsampling step of ``x`` along block a = ``block``, from the multi-index
        ``level`` to the level with d_a + 1 (maths.md M-DOWN along the axes of block a,
        M-NOT-TP); the other blocks keep their level.

        ``x`` is ``(Nb, Nc, *O, *spatial)`` at ``level``, real or complex; the result has the
        spatial shape of the new level. Calls the hook ``_downsample`` of the block, with the
        smoothing kernel of the block. Pure.
        """
        plan, filters = self._plan, self._arrays("downsample")
        a = self._block(block, "downsample")
        lead, spatial = self._split(x, level, "downsample", minimum=1)
        block_plan = plan.blocks[a]
        steps = block_plan.wavelet_levels[:-1]
        if level[a] not in steps:
            last = f"{steps[0]} … {steps[-1]}" if steps else "none"
            raise ValueError(
                f"downsample(level={level}, block={a}): block {a} is downsampled from its levels "
                f"{last} (J_{a} = {block_plan.J}); the step to the level of its low-pass filter "
                "is lowpass(..., block=a) (maths.md M-LP)."
            )
        xp = _registry.backend_named(self._data.backend).xp
        flat = xp.reshape(x, (lead[0], -1, *spatial))
        weights = DownsampleWeights(filters[a].smoothing)
        out = self._data._downsample(flat, weights, level[a], block=a)
        return xp.reshape(out, (*lead, *tuple(out.shape[2:])))

    def _arrays(self, function: str) -> tuple[WaveletFilters, ...]:
        if self._filters is None:
            raise ValueError(
                f"TensorProdWaveletOperator.{function}: this operator holds its static part "
                "only. Bind the arrays too: TensorProdWaveletOperator.from_parts(data, plan, "
                "filters), with filters = build_tensor_prod_filters(plan, backend, dtype, device)."
            )
        return self._filters

    def _block(self, block: object, function: str) -> int:
        n = len(self._plan.blocks)
        if isinstance(block, bool) or not isinstance(block, Integral) or not 0 <= int(block) < n:
            raise ValueError(
                f"TensorProdWaveletOperator.{function}(block={block!r}): pass a block index, "
                f"0 ≤ block < {n}."
            )
        return int(block)

    def _split(
        self, x: Any, level: Level, function: str, minimum: int = 2
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The leading and the spatial shape of ``x``, a map ``(Nb, Nc, *O, *spatial)`` at the
        multi-index ``level`` (``(Nb, *O, *spatial)`` with ``minimum=1``); shapes are static, so
        the check works in traced code."""
        expected = self._plan.level_shape(level)
        ndim = len(expected)
        shape = tuple(int(n) for n in x.shape)
        if len(shape) < minimum + ndim or shape[-ndim:] != expected:
            raise ValueError(
                f"TensorProdWaveletOperator.{function}: the array has shape {shape}, but a map "
                f"at level {level} is (Nb, Nc, ..., {', '.join(str(n) for n in expected)}). "
                "Bring it to that level first, one downsample step per block and level."
            )
        return shape[:-ndim], shape[-ndim:]

    def __repr__(self) -> str:
        plan = self._plan
        names = ", ".join(b.geometry.__name__ for b in plan.blocks)
        return (
            f"TensorProdWaveletOperator(J={list(plan.J)}, L={list(plan.L)}, blocks=[{names}], "
            f"spatial_shape={plan.spatial_shape}, pbc={list(self._data.pbc)})"
        )
