"""The scattering loop of a tensor product: STL's order, with a cache of prefixes, and the Gram
form of the second layer (Part II, §12.6; maths.md M-W1-TENSORPROD, M-LP, M-S-TENSORPROD,
M-BORDER-TENSORPROD; maths_step2.md M-GRAM; task T3.2).

:func:`values_tensor_prod` is the counterpart of ``_loop_single_domain.values_single_domain``
for a ``DataTensorProd``: the data, the plan and the filters in, the statistics without
normalisation out, in the compact buffer of ``statistics._layout.layout_tensor_prod`` (task
T3.6). It shares the code of the single-domain loop for what does not depend on the geometry
(``_loop``: the means of the field, the sums of S1 and S2 of a scale, the Gram products of a
level, the sums then the division with the hook of distributed runs, ``_loop._global_sum``,
read from its module at each call so that a replaced hook reaches this loop too); it differs
by the hooks of Part II, §12.6: the index sets and levels are multi-indices, and the first and
the second layer apply the block convolutions one block after the other.

**The first layer** follows STL (M-W1-TENSORPROD, "How STL reaches it"): the field is downsampled
along block 0 to the level of j₀ and filtered along block 0; the complex result is downsampled along
block 1 to the level of j₁ and filtered along block 1; and so on. The result of the blocks
0 … a is kept, and serves every scale that starts with (j₀, …, j_a): the cache of prefixes is
the recursion of :func:`_first_layer`, depth first, in lexicographic order. Along a block taken
at its low-pass scale, the map is brought to the level of the low-pass filter
(``WaveletOperatorTensorProd.lowpass``, M-LP) and nothing is filtered; at the scale with the
low-pass filter in every block, the real field X is centred, X - ⟨X⟩ (OP-42).

**The second layer** follows STL too (M-S-TENSORPROD, "How STL reaches the second layer"): for each
j3, the moduli U_{j2}, j2 ≤ j3, are brought along block 0 to the level of j3₀ and filtered
along block 0, the complex results are downsampled along block 1 and filtered along block 1,
and so on; the result of the blocks 0 … a serves every j3 that starts with (j3₀, …, j3_a). At
block a, the arms that are at one level in every later block are concatenated, so that one
convolution filters them all; at the last block they form, for each band orientation l3, the
rows of the Gram matrix A of M-GRAM, and the sums of every S4 of j3 are A A^H, those of every
S3 the product of A with the band map, after each map is set to zero outside its window, block
by block (``_tensor_prod``). The rows outside M-IDX-TENSORPROD (j1 not ≤ j2) are computed and not
read.

**Orientations.** Each block convolution puts its orientation right after the batch axis
(``WaveletOperatorTensorProd.convolve``), so that the maps of one band orientation are one
matrix without a copy; the orientations of the blocks then come in reverse block order. The
means, small arrays, are put in the C order of the blocks before they are gathered.

**Into the layout.** The step from the Gram products to the chunks of the buffer is shared
with the loop of single-domain data (``_loop.gram_chunk_sums``): the sums of S3 and S4 of j3
are read from the products, those of the chunks of their segment (``SecondLayerScale.s3_rows``,
``s4_rows``), divided by their per-chunk denominators, and the S4 chunks are packed with their
real diagonal (``_loop.compact_buffer``, ``statistics._layout.pack``); every table grows with
the chunks, never with the coefficients.
Nothing is written in place, and nothing depends on array values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import Data
from ..data._moments import modulus
from ..statistics._layout import gather_tensor_prod
from ..wavelets._operator_tensor_prod import WaveletOperatorTensorProd
from . import _loop
from ._filters import STFiltersTensorProd
from ._loop import (
    GramRows,
    ScatteringValues,
    _divide_level,
    compact_buffer,
    field_moments,
    first_layer_means,
    gram_chunk_sums,
    gram_sums,
)
from ._tensor_prod import LoopTensorProd, ScaleTensorProd

if TYPE_CHECKING:
    from ._plan import STPlanTensorProd

# ---------------------------------------------------------------------- the loop


def values_tensor_prod(
    data: Data, plan: STPlanTensorProd, filters: STFiltersTensorProd
) -> ScatteringValues:
    """The statistics of ``data``, a ``DataTensorProd``, without normalisation (maths.md
    M-MOM, M-S-TENSORPROD), STL's ``ST_Operator_TensorProd`` with ``norm="vanilla"`` and no
    spectrum, in the compact buffer of ``layout_tensor_prod(plan)``. Pure."""
    backend = _registry.backend_named(data.backend)
    xp = backend.xp
    loop = plan.loop
    wavelets = WaveletOperatorTensorProd.from_parts(data, plan.wavelets, filters.wavelets)
    n_spatial = len(plan.spatial_shape)
    axes = tuple(range(-n_spatial, 0))

    # Tensor products have no missing values and no border on the field: every pixel.
    x, sums, mean_pre_std, std_pre_std = field_moments(
        data.array, None, loop.moments_denominator, axes, plan.standardize
    )
    moments = _divide_level(backend, sums, _loop._global_sum)

    # The first layer, every scale of S1 and S2 in lexicographic order: W_j x, (Nb, L'_{B-1},
    # …, L'_0, Nc, *spatial at level j), the orientations in reverse block order.
    maps = _first_layer(backend, wavelets, plan, x)
    scales = {s.scale: s for s in loop.first_layer}
    s12: dict[str, list[Any]] = {"S1": [], "S2": []}
    band: dict[ScaleTensorProd, Any] = {}
    moduli: dict[ScaleTensorProd, Any] = {}
    for scale, w in maps:
        info = scales[scale]
        if info.centred:  # X - ⟨X⟩ over every pixel of X (M-LP, OP-42), complex like the others
            # A mean of the loop: its sum goes through the hook of distributed runs too.
            mean = _loop._global_sum(xp.sum(w, axis=axes)) / info.denominator
            centred = w - xp.reshape(mean, (*tuple(mean.shape), *(1,) * len(axes)))
            w = xp.astype(centred, _complex_of(backend, w))
        u = modulus(w)
        window = None if info.window is None else tuple(slice(a, b) for a, b in info.window)
        floats = first_layer_means(
            backend, w, u, window, info.denominator, plan.pairs, info.pairs, info.orientations, axes
        )
        s12 = {name: [*s12[name], floats[name]] for name in s12}
        if not any(j == n for j, n in zip(scale, plan.J, strict=True)):
            band = {**band, scale: w}
            moduli = {**moduli, scale: u}

    # The second layer, every wavelet scale j3 in lexicographic order: its sums, one global
    # sum (the hook of distributed runs), then the division.
    s34: dict[str, list[Any]] = {"S3": [], "S4": []}
    for j3, cross, gram in _second_layer_grams(backend, wavelets, plan, moduli, band, filters):
        second = loop.second(j3)
        rows = GramRows(
            second.s3_rows,
            filters.s3_denominators[j3],
            filters.s4_rows[j3],
            filters.s4_denominators[j3],
        )
        n_arms = len(second.arms)
        sums = gram_chunk_sums(backend, cross, gram, rows, plan.L, n_arms, plan.n_channels, False)
        for item, value in zip(sums, _divide_level(backend, sums, _loop._global_sum), strict=True):
            s34 = {**s34, item.family: [*s34[item.family], value]}

    buffer = compact_buffer(
        backend,
        {"moments": moments, **s12, **s34},
        math.prod(plan.L),
        gather_tensor_prod(plan),
        filters.s4_pack,
    )
    return ScatteringValues(buffer=buffer, mean_pre_std=mean_pre_std, std_pre_std=std_pre_std)


def _complex_of(backend: Backend, x: Any) -> Any:
    xp = backend.xp
    return xp.complex128 if x.dtype == xp.float64 else xp.complex64


# ---------------------------------------------------------------------- first layer


def _first_layer(
    backend: Backend, wavelets: WaveletOperatorTensorProd, plan: STPlanTensorProd, x: Any
) -> list[tuple[ScaleTensorProd, Any]]:
    """W_j x for every scale j of S1 and S2, in lexicographic order, with STL's cache of
    prefixes (M-W1-TENSORPROD): the map filtered along the blocks 0 … a - 1 is downsampled along
    block a level by level, filtered at each wavelet scale of block a, and, when the block has a
    low-pass filter, brought to its level (M-LP)."""
    blocks = plan.wavelets.blocks
    n_blocks = len(blocks)
    xp = backend.xp

    def stage(
        a: int, prefix: ScaleTensorProd, level: ScaleTensorProd, y: Any
    ) -> list[tuple[ScaleTensorProd, Any]]:
        if a == n_blocks:
            return [(prefix, y)]
        block = blocks[a]
        out: list[tuple[ScaleTensorProd, Any]] = []
        chain, current = y, level
        for j in range(block.J):
            chain, current = _bring(wavelets, chain, current, a, block.scale_levels[j])
            filtered = wavelets.convolve(chain, j, block=a, level=current)
            out.extend(stage(a + 1, (*prefix, j), current, filtered))
        if block.has_lowpass:
            target = block.lowpass_level - 1
            chain, current = _bring(wavelets, chain, current, a, target)
            low = wavelets.lowpass(chain, current, block=a)
            current = _with(current, a, block.lowpass_level)
            low = xp.reshape(low, (int(low.shape[0]), 1, *tuple(low.shape[1:])))
            out.extend(stage(a + 1, (*prefix, block.J), current, low))
        return out

    return stage(0, (), (0,) * n_blocks, x)


# ---------------------------------------------------------------------- second layer


def _second_layer_grams(
    backend: Backend,
    wavelets: WaveletOperatorTensorProd,
    plan: STPlanTensorProd,
    moduli: dict[ScaleTensorProd, Any],
    band: dict[ScaleTensorProd, Any],
    filters: STFiltersTensorProd,
) -> list[tuple[ScaleTensorProd, Any, Any]]:
    """For each wavelet scale j3, in lexicographic order, the Gram products of its level
    (M-GRAM): the products of the second-layer maps with the band map W_{j3} x,
    ``(Nb, L3, P, Nc)``, and their Gram matrix, ``(Nb, L3, P, P)``, the rows P being the arms
    j2 ≤ j3 in the order of ``SecondLayerScale.arms``, each with its orientations (in reverse
    block order) and channels; the band orientations l3 in reverse block order too.

    STL's order and cache of prefixes (M-S-TENSORPROD): at block a, with the scales j3₀ … j3_{a-1}
    fixed, the arms are grouped by their levels in the later blocks; for each j3_a, the group
    already brought to the previous scale is downsampled one more step along block a, the arms
    with j2_a = j3_a are appended, and the group is filtered along block a at once."""
    xp = backend.xp
    blocks = plan.wavelets.blocks
    n_blocks = len(blocks)
    loop = plan.loop
    n_rows = int(_prod(plan.L)) * plan.n_channels

    def as_arm(u: Any) -> Any:
        # (Nb, L_{B-1}, …, L_0, Nc, *spatial) -> (Nb, 1, L·Nc, *spatial): one arm.
        spatial = tuple(int(n) for n in u.shape[1 + n_blocks + 1 :])
        return xp.reshape(u, (int(u.shape[0]), 1, n_rows, *spatial))

    def level_of(prefix: ScaleTensorProd, a: int, j: int, tail: ScaleTensorProd) -> ScaleTensorProd:
        head = tuple(blocks[b].scale_levels[prefix[b]] for b in range(a))
        rest = tuple(blocks[a + 1 + i].scale_levels[t] for i, t in enumerate(tail))
        return (*head, blocks[a].scale_levels[j], *rest)

    def stage(
        a: int, prefix: ScaleTensorProd, groups: dict[ScaleTensorProd, Any]
    ) -> list[tuple[ScaleTensorProd, Any, Any]]:
        # ``groups``: for each tail (j2_a, …, j2_{B-1}), the arms with that tail and heads
        # (j2_0 … j2_{a-1}) ≤ prefix, filtered along the blocks 0 … a - 1, stacked on the axis
        # after the a band orientations: (Nb, L3_{a-1}, …, L3_0, n_arms, L·Nc, *spatial).
        if a == n_blocks:
            (stack,) = groups.values()
            return [_leaf(backend, plan, loop, filters, prefix, stack, band[prefix])]
        block = blocks[a]
        tails = sorted({tail[1:] for tail in groups})
        out: list[tuple[ScaleTensorProd, Any, Any]] = []
        chains: dict[ScaleTensorProd, Any] = {}
        for j in range(block.J):
            filtered: dict[ScaleTensorProd, Any] = {}
            for tail in tails:
                level = level_of(prefix, a, j, tail)
                previous = chains.get(tail)
                if previous is not None:
                    previous, _ = _bring(
                        wavelets,
                        previous,
                        level_of(prefix, a, j - 1, tail),
                        a,
                        block.scale_levels[j],
                    )
                new = groups[(j, *tail)]
                stacked = new if previous is None else xp.concat([previous, new], axis=1 + a)
                chains = {**chains, tail: stacked}
                filtered = {**filtered, tail: wavelets.convolve(stacked, j, block=a, level=level)}
            out.extend(stage(a + 1, (*prefix, j), filtered))
        return out

    return stage(0, (), {j2: as_arm(u) for j2, u in moduli.items()})


def _leaf(
    backend: Backend,
    plan: STPlanTensorProd,
    loop: LoopTensorProd,
    filters: STFiltersTensorProd,
    j3: ScaleTensorProd,
    stack: Any,
    band: Any,
) -> tuple[ScaleTensorProd, Any, Any]:
    """The Gram products of the level of j3: each map set to zero outside its window, block by
    block, then :func:`~estel.scattering._loop.gram_sums`."""
    xp = backend.xp
    n_blocks = plan.n_blocks
    info = loop.second(j3)
    n_spatial = len(plan.spatial_shape)
    spatial = tuple(int(n) for n in stack.shape[len(stack.shape) - n_spatial :])
    n_batch, n_arms = int(stack.shape[0]), len(info.arms)
    # (Nb, L3_{B-1}, …, L3_0, n_arms, L·Nc, *spatial) -> (Nb, L3, n_arms, L·Nc, *spatial)
    v = xp.reshape(stack, (n_batch, -1, n_arms, int(stack.shape[1 + n_blocks + 1]), *spatial))
    w = xp.reshape(band, (n_batch, -1, plan.n_channels, *spatial))
    first = 0
    for a, axes_a in enumerate(plan.wavelets.axes):
        n_axes = len(axes_a)
        before, after = first, n_spatial - first - n_axes
        crops, band_crop = filters.arm_crops[j3][a], filters.band_crops[j3][a]
        if crops is not None:
            shape = (n_arms, 1, *(1,) * before, *tuple(crops.shape[1:]), *(1,) * after)
            v = v * xp.reshape(crops, shape)
        if band_crop is not None:
            w = w * xp.reshape(band_crop, (*(1,) * before, *tuple(band_crop.shape), *(1,) * after))
        first += n_axes
    cross, gram = gram_sums(v, w, info.s3_rows.shape[0] > 0, info.s4_rows.shape[0] > 0)
    return j3, cross, gram


# ---------------------------------------------------------------------- helpers


def _bring(
    wavelets: WaveletOperatorTensorProd, x: Any, level: ScaleTensorProd, a: int, target: int
) -> tuple[Any, ScaleTensorProd]:
    """``x`` at ``level``, downsampled along block a until d_a = ``target``."""
    while level[a] < target:
        x = wavelets.downsample(x, level, block=a)
        level = _with(level, a, level[a] + 1)
    return x, level


def _with(level: ScaleTensorProd, a: int, d: int) -> ScaleTensorProd:
    return (*level[:a], d, *level[a + 1 :])


def _prod(values: Any) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out
