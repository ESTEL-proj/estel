"""The static description of the scattering loop of a tensor product (Part II, §12.6; maths.md
M-W1-TP, M-LP, M-IDX-TP, M-S-TP, M-BORDER-TP; maths_step2.md M-GRAM; task T3.2).

Built once, in NumPy, from a :class:`~estel.scattering._plan.TensorProdPlan` alone
(:func:`tensor_prod_loop`, cached per plan), never traced. It holds what the loop of
``_loop_tp.py`` reads besides the plan of each block:

* **the first layer**, one :class:`FirstLayerScale` per scale j of S1 and S2, in lexicographic
  order (the last block fastest), the order in which the loop reaches them: the orientation
  shape of the scale (1 along a block taken at its low-pass scale, M-LP), its window as a
  rectangle (the border of each block, M-BORDER-TP, 0 along a block at its low-pass scale,
  OP-81), its denominator, and whether it is the centred scale with the low-pass filter in
  every block (OP-42);
* **the second layer**, one :class:`SecondLayerScale` per wavelet scale j3: the arms j2 ≤ j3
  in the order the loop stacks them, the crop of each block for each arm and for the band map,
  and, for the Gram products of the level, which of their rows are the chunks of the segments
  of S3 and S4 at j3 in the compact layout of T3.6 (``tensor_prod_layout``), with their
  denominators, each the product of per-block counts of pixels (M-BORDER-TP);
* **the gather**: the order of the channel pairs of the S1 and S2 segments, and how the S4
  chunks are packed (``statistics._layout.pack_spec``).

Every table grows with the number of chunks of a segment and of orientations, never with the
number of coefficients, and nothing has the size of the full grid (decision 3 of T3.7; issue
#23, "Decisions of the maintainer for T3.2", point 4). The crops of the blocks are the only
pixel arrays, one per block, arm and level, of the shape of that block.

**Why the windows are per block.** The window of a map of a tensor product is the outer
product of the windows of its blocks (M-BORDER-TP), and that of a pair of maps is the product,
block by block, of the larger border of the two. With every map set to zero outside its own
window, block after block, the Gram product A A^H sums each pair over the intersection of the
two windows, and the number of its pixels is the product of the per-block counts: STL's
window of each pair, exactly (maths_step2.md M-GRAM, "Windows").
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..statistics._layout import Pack, pack_spec, tensor_prod_layout
from ..wavelets._operator import Box, window_box
from ._plan import TensorProdPlan

Scale = tuple[int, ...]
"""A multi-index scale j = (j₀, …, j_{B-1}) (maths.md M-NOT-TP)."""


class FirstLayerScale(NamedTuple):
    """One scale j of S1 and S2 (maths.md M-W1-TP, M-LP, M-BORDER-TP)."""

    scale: Scale
    """j; j_a = J_a along a block taken at its low-pass scale."""
    level: Scale
    """The level of W_j x, read from the plans of the blocks: the level of scale j_a, or
    ``lowpass_level`` along a block at its low-pass scale."""
    orientations: tuple[int, ...]
    """L'_a per block, in block order: L_a, or 1 along a block at its low-pass scale."""
    window: Box | None
    """The rectangle of the means of S1 and S2, the bounds on each spatial axis; ``None`` for
    every pixel."""
    denominator: float
    """Its number of pixels, the product of the per-block counts."""
    centred: bool
    """Whether j has the low-pass filter in every block: the field X there is centred,
    X - ⟨X⟩, over all its pixels (maths.md M-LP, OP-42)."""
    pairs: tuple[tuple[int, int], ...]
    """The channel pairs of the chunks of the segment of j, in the order of the buffer."""


@dataclass(frozen=True, eq=False)
class SecondLayerScale:
    """The second layer of one wavelet scale j3, at its level (maths.md M-S-TP; M-GRAM)."""

    j3: Scale
    level: Scale
    """The level of j3, read from the plans of the blocks."""
    arms: tuple[Scale, ...]
    """The scales j2 ≤ j3 (componentwise), in the order the loop stacks them: sorted on the
    last block first (``_loop_tp``). The rows of the Gram matrix, with the orientations and
    channels of each arm."""
    arm_crops: tuple[NDArray[np.float64] | None, ...]
    """Per block, the crop of each arm along that block, ``(n_arms, *shape of the block at
    j3_a)``, read-only; ``None`` for a block where every crop is the whole block (periodic)."""
    band_crops: tuple[NDArray[np.float64] | None, ...]
    """Per block, the crop of the band map W_{j3} x along that block, or ``None``."""
    s3_rows: NDArray[np.int64]
    """The chunks of the segment of S3 at j3, in the order of the buffer, as rows of the
    S3 sums ``(arm, c1, c2)`` flattened: ``(arm · Nc + c1) · Nc + c2``."""
    s3_denominators: NDArray[np.float64]
    """The number of pixels of each of these chunks."""
    s4_rows: NDArray[np.int64]
    """The chunks of the segment of S4 at j3, as rows of ``(arm1, c1, arm2, c2)`` flattened."""
    s4_denominators: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class TensorProdLoop:
    """The static description of the scattering loop of a tensor product (this module)."""

    first_layer: tuple[FirstLayerScale, ...]
    """Every scale of S1 and S2, in lexicographic order."""
    second_layer: tuple[SecondLayerScale, ...]
    """Every wavelet scale j3, in lexicographic order."""
    moments_denominator: float
    """The number of pixels of the field, the denominator of mean and var (no border)."""
    s4_pack: Pack
    """How the S4 chunks, in the order of the buffer, are packed into its floats."""
    reduced_border: str | None
    """The warning of a border too large for a block (OP-13), given once by the operator."""


def lexicographic(sizes: tuple[int, ...]) -> list[Scale]:
    """Every multi-index of ``sizes``, in lexicographic order, the last block fastest."""
    return [tuple(int(j) for j in index) for index in itertools.product(*(range(n) for n in sizes))]


def arm_order(j3: Scale) -> tuple[Scale, ...]:
    """The scales j2 ≤ j3 in the order the loop stacks them (``_loop_tp``): at block a, the
    arms already filtered along the blocks before a are concatenated over j2_a, so the last
    block varies slowest."""
    return tuple(sorted(lexicographic(tuple(j + 1 for j in j3)), key=lambda j: j[::-1]))


@functools.lru_cache(maxsize=32)
def tensor_prod_loop(plan: TensorProdPlan) -> TensorProdLoop:
    """The :class:`TensorProdLoop` of ``plan``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loop = _build(plan)
    message = str(caught[0].message) if caught else None
    return TensorProdLoop(
        first_layer=loop.first_layer,
        second_layer=loop.second_layer,
        moments_denominator=loop.moments_denominator,
        s4_pack=loop.s4_pack,
        reduced_border=message,
    )


def _build(plan: TensorProdPlan) -> TensorProdLoop:
    blocks = plan.wavelets.blocks
    n_blocks, nc = plan.n_blocks, plan.n_channels
    layout = tensor_prod_layout(plan)

    @functools.cache
    def crop(a: int, level: int, border: int) -> NDArray[np.float64]:
        found = np.asarray(plan.wavelets.block_crop(a, level, border), dtype=np.float64)
        found.flags.writeable = False
        return found

    @functools.cache
    def count(a: int, level: int, border: int) -> float:
        return float(crop(a, level, border).sum())

    def border(a: int, history: tuple[int, ...], level: int) -> int:
        return blocks[a].border(history, level)

    def segments(name: str) -> dict[Scale, NDArray[np.int32]]:
        return {s.level: s.chunks for s in layout.segments if s.family == name}

    # ---- first layer
    rows_12 = segments("S1")
    first = []
    for scale in lexicographic(plan.n_scales):
        lowpass = tuple(j == n for j, n in zip(scale, plan.J, strict=True))
        level = tuple(
            b.lowpass_level if lp else b.scale_levels[j]
            for b, j, lp in zip(blocks, scale, lowpass, strict=True)
        )
        widths = [0 if lowpass[a] else border(a, (scale[a],), level[a]) for a in range(n_blocks)]
        crops = [crop(a, level[a], widths[a]) for a in range(n_blocks)]
        boxes = [window_box(c) for c in crops]
        box: Box = tuple(bound for b in boxes for bound in (b or ()))
        full = all(w == 0 for w in widths)
        first.append(
            FirstLayerScale(
                scale=scale,
                level=level,
                orientations=plan.orientations_at(scale),
                window=None if full else box,
                denominator=math.prod(count(a, level[a], widths[a]) for a in range(n_blocks)),
                centred=all(lowpass),
                pairs=tuple((int(r[0]), int(r[1])) for r in rows_12[scale]),
            )
        )

    # ---- second layer
    rows_3, rows_4 = segments("S3"), segments("S4")
    second = []
    for j3 in lexicographic(plan.J):
        level = tuple(b.scale_levels[j] for b, j in zip(blocks, j3, strict=True))
        arms = arm_order(j3)
        position = {j2: i for i, j2 in enumerate(arms)}
        n_arms = len(arms)

        def arm_border(a: int, j2: int, level: Scale = level, j3: Scale = j3) -> int:
            return border(a, (j2, j3[a]), level[a])

        def band_border(a: int, level: Scale = level, j3: Scale = j3) -> int:
            return border(a, (j3[a],), level[a])

        arm_crops: list[NDArray[np.float64] | None] = []
        band_crops: list[NDArray[np.float64] | None] = []
        for a in range(n_blocks):
            widths = [arm_border(a, j2[a]) for j2 in arms]
            if any(widths):
                stacked = np.stack([crop(a, level[a], w) for w in widths])
                stacked.flags.writeable = False
                arm_crops.append(stacked)
            else:
                arm_crops.append(None)
            band_crops.append(crop(a, level[a], band_border(a)) if band_border(a) else None)

        s3_rows, s3_den = [], []
        for row in rows_3[j3]:
            c1, c2 = int(row[0]), int(row[1])
            j2 = tuple(int(j) for j in row[2 : 2 + n_blocks])
            s3_rows.append((position[j2] * nc + c1) * nc + c2)
            s3_den.append(
                math.prod(
                    count(a, level[a], max(arm_border(a, j2[a]), band_border(a)))
                    for a in range(n_blocks)
                )
            )
        s4_rows, s4_den = [], []
        for row in rows_4[j3]:
            c1, c2 = int(row[0]), int(row[1])
            j1 = tuple(int(j) for j in row[2 : 2 + n_blocks])
            j2 = tuple(int(j) for j in row[2 + n_blocks : 2 + 2 * n_blocks])
            s4_rows.append(((position[j1] * nc + c1) * n_arms + position[j2]) * nc + c2)
            s4_den.append(
                math.prod(
                    count(a, level[a], max(arm_border(a, j1[a]), arm_border(a, j2[a])))
                    for a in range(n_blocks)
                )
            )
        second.append(
            SecondLayerScale(
                j3=j3,
                level=level,
                arms=arms,
                arm_crops=tuple(arm_crops),
                band_crops=tuple(band_crops),
                s3_rows=_read_only(np.array(s3_rows, dtype=np.int64)),
                s3_denominators=_read_only(np.array(s3_den, dtype=np.float64)),
                s4_rows=_read_only(np.array(s4_rows, dtype=np.int64)),
                s4_denominators=_read_only(np.array(s4_den, dtype=np.float64)),
            )
        )
    return TensorProdLoop(
        first_layer=tuple(first),
        second_layer=tuple(second),
        moments_denominator=float(math.prod(plan.spatial_shape)),
        s4_pack=pack_spec(layout, "S4"),
        reduced_border=None,
    )


_A = TypeVar("_A", bound=np.ndarray)


def _read_only(array: _A) -> _A:
    array.flags.writeable = False
    return array
