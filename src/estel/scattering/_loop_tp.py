"""The scattering loop of a tensor product: STL's order, with a cache of prefixes, and the Gram
form of the second layer (Part II, §12.6; maths.md M-W1-TP, M-LP, M-S-TP, M-BORDER-TP;
maths_step2.md M-GRAM; task T3.2).

:func:`tensor_prod_values` is the counterpart of ``_loop.scattering_values`` for a
``DataTensorProd``: the data, the plan and the filters in, the statistics without normalisation
out, in the compact buffer of ``statistics._layout.tensor_prod_layout`` (task T3.6). It shares
the code of the planar loop for what does not depend on the geometry (``_loop``: the means of
the field, the sums of S1 and S2 of a scale, the Gram products of a level, the sums then the
division with the hook of distributed runs); it differs by the hooks of Part II, §12.6: the
index sets and levels are multi-indices, and the first and the second layer apply the block
convolutions one block after the other.

**The first layer** follows STL (M-W1-TP, "How STL reaches it"): the field is downsampled along
block 0 to the level of j₀ and filtered along block 0; the complex result is downsampled along
block 1 to the level of j₁ and filtered along block 1; and so on. The result of the blocks
0 … a is kept, and serves every scale that starts with (j₀, …, j_a): the cache of prefixes is
the recursion of :func:`_first_layer`, depth first, in lexicographic order. Along a block taken
at its low-pass scale, the map is brought to the level of the low-pass filter
(``WaveletOperatorTensorProd.lowpass``, M-LP) and nothing is filtered; at the scale with the
low-pass filter in every block, the real field X is centred, X - ⟨X⟩ (OP-42).

**The second layer** follows STL too (M-S-TP, "How STL reaches the second layer"): for each
j3, the moduli U_{j2}, j2 ≤ j3, are brought along block 0 to the level of j3₀ and filtered
along block 0, the complex results are downsampled along block 1 and filtered along block 1,
and so on; the result of the blocks 0 … a serves every j3 that starts with (j3₀, …, j3_a). At
block a, the arms that are at one level in every later block are concatenated, so that one
convolution filters them all; at the last block they form, for each band orientation l3, the
rows of the Gram matrix A of M-GRAM, and the sums of every S4 of j3 are A A^H, those of every
S3 the product of A with the band map, after each map is set to zero outside its window, block
by block (``_tensor_prod``). The rows outside M-IDX-TP (j1 not ≤ j2) are computed and not read.

**Orientations.** Each block convolution puts its orientation right after the batch axis
(``WaveletOperatorTensorProd.convolve``), so that the maps of one band orientation are one
matrix without a copy; the orientations of the blocks then come in reverse block order. The
means, small arrays, are put in the C order of the blocks before they are gathered.

**Into the layout.** The sums of S3 and S4 of j3 are read from the Gram products by one
``take`` each, of the chunks of their segment (``SecondLayerScale.s3_rows``), divided by their
per-chunk denominators, and the S4 chunks are packed with their real diagonal
(``statistics._layout.pack``); every table grows with the chunks, never with the coefficients.
Nothing is written in place, and nothing depends on array values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import Data
from ..data._moments import modulus
from ..statistics._layout import pack, tensor_prod_gather
from ..wavelets._tensor_prod import WaveletOperatorTensorProd
from ._filters import TensorProdFilters
from ._loop import (
    ScatteringValues,
    _divide_level,
    _global_sum,
    field_moments,
    first_layer_sums,
    gram_sums,
)
from ._tensor_prod import FirstLayerScale, Scale, SecondLayerScale, TensorProdLoop

if TYPE_CHECKING:
    from ._plan import TensorProdPlan

# ---------------------------------------------------------------------- the loop


def tensor_prod_values(
    data: Data, plan: TensorProdPlan, filters: TensorProdFilters
) -> ScatteringValues:
    """The statistics of ``data``, a ``DataTensorProd``, without normalisation (maths.md
    M-MOM, M-S-TP), STL's ``ST_Operator_TensorProd`` with ``norm="vanilla"`` and no spectrum, in
    the compact buffer of ``tensor_prod_layout(plan)``. Pure."""
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
    moments = _divide_level(backend, sums, _global_sum)

    # The first layer, every scale of S1 and S2 in lexicographic order: W_j x, (Nb, L'_{B-1},
    # …, L'_0, Nc, *spatial at level j), the orientations in reverse block order.
    maps = _first_layer(backend, wavelets, plan, x)
    scales = {s.scale: s for s in loop.first_layer}
    s12: dict[str, list[Any]] = {"S1": [], "S2": []}
    band: dict[Scale, Any] = {}
    moduli: dict[Scale, Any] = {}
    for scale, w in maps:
        info = scales[scale]
        if info.centred:  # X - ⟨X⟩ over every pixel of X (M-LP, OP-42), complex like the others
            # A mean of the loop: its sum goes through the hook of distributed runs too.
            mean = _global_sum(xp.sum(w, axis=axes)) / info.denominator
            centred = w - xp.reshape(mean, (*tuple(mean.shape), *(1,) * len(axes)))
            w = xp.astype(centred, _complex_of(backend, w))
        u = modulus(w)
        floats = _first_layer_means(backend, plan, info, w, u, axes)
        s12 = {name: [*s12[name], floats[name]] for name in s12}
        if not any(j == n for j, n in zip(scale, plan.J, strict=True)):
            band = {**band, scale: w}
            moduli = {**moduli, scale: u}

    # The second layer, every wavelet scale j3 in lexicographic order.
    s3: list[Any] = []
    s4: list[Any] = []
    for j3, cross, gram in _second_layer_grams(backend, wavelets, plan, moduli, band, filters):
        second = loop.second(j3)
        three, four = _second_layer_means(backend, plan, second, filters, cross, gram)
        s3, s4 = [*s3, three], [*s4, four]

    n_batch = int(x.shape[0])
    s4_dense = xp.concat(s4, axis=1)
    s4_floats = pack(backend, s4_dense, tensor_prod_gather(plan), filters.s4_pack)
    buffer = xp.concat(
        [
            *moments,
            *s12["S1"],
            *s12["S2"],
            *(xp.reshape(item, (n_batch, -1)) for item in s3),
            s4_floats,
        ],
        axis=1,
    )
    return ScatteringValues(buffer=buffer, mean_pre_std=mean_pre_std, std_pre_std=std_pre_std)


def _complex_of(backend: Backend, x: Any) -> Any:
    xp = backend.xp
    return xp.complex128 if x.dtype == xp.float64 else xp.complex64


# ---------------------------------------------------------------------- first layer


def _first_layer(
    backend: Backend, wavelets: WaveletOperatorTensorProd, plan: TensorProdPlan, x: Any
) -> list[tuple[Scale, Any]]:
    """W_j x for every scale j of S1 and S2, in lexicographic order, with STL's cache of
    prefixes (M-W1-TP): the map filtered along the blocks 0 … a - 1 is downsampled along block
    a level by level, filtered at each wavelet scale of block a, and, when the block has a
    low-pass filter, brought to its level (M-LP)."""
    blocks = plan.wavelets.blocks
    n_blocks = len(blocks)
    xp = backend.xp

    def stage(a: int, prefix: Scale, level: Scale, y: Any) -> list[tuple[Scale, Any]]:
        if a == n_blocks:
            return [(prefix, y)]
        block = blocks[a]
        out: list[tuple[Scale, Any]] = []
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


def _first_layer_means(
    backend: Backend,
    plan: TensorProdPlan,
    info: FirstLayerScale,
    w: Any,
    u: Any,
    axes: tuple[int, ...],
) -> dict[str, Any]:
    """The divided S1 and S2 means of one scale, as the floats of their segments: the chunks
    of the channel pairs in the order of the buffer, the orientations of each chunk in the C
    order of the blocks, an auto chunk real, a cross chunk complex as (re, im)."""
    xp = backend.xp
    n_batch, n_blocks = int(w.shape[0]), plan.n_blocks
    reverse = info.orientations[::-1]
    n_o = int(_prod(reverse))
    spatial = tuple(int(n) for n in w.shape[len(w.shape) - len(axes) :])
    flat_w = xp.reshape(w, (n_batch, n_o, plan.n_channels, *spatial))
    flat_u = xp.reshape(u, (n_batch, n_o, plan.n_channels, *spatial))
    window = None if info.window is None else tuple(slice(a, b) for a, b in info.window)
    pending = first_layer_sums(flat_w, flat_u, window, info.denominator, plan.pairs, axes)
    divided = _divide_level(backend, pending, _global_sum)
    pairs = plan.pairs
    order = [
        *(("S1", (c, c)) for c in pairs.auto),
        *(("S1", p) for p in pairs.cross),
        *(("S2", (c, c)) for c in pairs.auto),
        *(("S2", p) for p in pairs.cross),
    ]
    found = dict(zip(order, divided, strict=True))
    # From the reverse block order of the loop to the C order of the blocks.
    to_c = tuple(range(n_blocks - 1, -1, -1))

    def ordered(value: Any, is_complex: bool) -> Any:
        parts = (2,) if is_complex else ()
        grid = xp.reshape(value, (n_batch, *reverse, *parts))
        tail = tuple(1 + n_blocks + i for i in range(len(parts)))
        moved = xp.permute_dims(grid, (0, *(1 + a for a in to_c), *tail))
        return xp.reshape(moved, (n_batch, -1))

    return {
        name: xp.concat(
            [ordered(found[(name, pair)], pair[0] != pair[1]) for pair in info.pairs], axis=1
        )
        for name in ("S1", "S2")
    }


# ---------------------------------------------------------------------- second layer


def _second_layer_grams(
    backend: Backend,
    wavelets: WaveletOperatorTensorProd,
    plan: TensorProdPlan,
    moduli: dict[Scale, Any],
    band: dict[Scale, Any],
    filters: TensorProdFilters,
) -> list[tuple[Scale, Any, Any]]:
    """For each wavelet scale j3, in lexicographic order, the Gram products of its level
    (M-GRAM): the products of the second-layer maps with the band map W_{j3} x,
    ``(Nb, L3, P, Nc)``, and their Gram matrix, ``(Nb, L3, P, P)``, the rows P being the arms
    j2 ≤ j3 in the order of ``SecondLayerScale.arms``, each with its orientations (in reverse
    block order) and channels; the band orientations l3 in reverse block order too.

    STL's order and cache of prefixes (M-S-TP): at block a, with the scales j3₀ … j3_{a-1}
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

    def level_of(prefix: Scale, a: int, j: int, tail: Scale) -> Scale:
        head = tuple(blocks[b].scale_levels[prefix[b]] for b in range(a))
        rest = tuple(blocks[a + 1 + i].scale_levels[t] for i, t in enumerate(tail))
        return (*head, blocks[a].scale_levels[j], *rest)

    def stage(a: int, prefix: Scale, groups: dict[Scale, Any]) -> list[tuple[Scale, Any, Any]]:
        # ``groups``: for each tail (j2_a, …, j2_{B-1}), the arms with that tail and heads
        # (j2_0 … j2_{a-1}) ≤ prefix, filtered along the blocks 0 … a - 1, stacked on the axis
        # after the a band orientations: (Nb, L3_{a-1}, …, L3_0, n_arms, L·Nc, *spatial).
        if a == n_blocks:
            (stack,) = groups.values()
            return [_leaf(backend, plan, loop, filters, prefix, stack, band[prefix])]
        block = blocks[a]
        tails = sorted({tail[1:] for tail in groups})
        out: list[tuple[Scale, Any, Any]] = []
        chains: dict[Scale, Any] = {}
        for j in range(block.J):
            filtered: dict[Scale, Any] = {}
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
    plan: TensorProdPlan,
    loop: TensorProdLoop,
    filters: TensorProdFilters,
    j3: Scale,
    stack: Any,
    band: Any,
) -> tuple[Scale, Any, Any]:
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


def _second_layer_means(
    backend: Backend,
    plan: TensorProdPlan,
    info: SecondLayerScale,
    filters: TensorProdFilters,
    cross: Any,
    gram: Any,
) -> tuple[Any, Any]:
    """The divided S3 and S4 means of the segments of j3, from its Gram products: S3 as
    ``(Nb, n3, L², 2)`` and S4 as ``(Nb, n4, L³, 2)``, (re, im), the chunks in the order of the
    buffer and the orientations of each chunk in the C order of the blocks (l1, l2, l3, each
    with one axis per block)."""
    xp = backend.xp
    n_blocks, nc = plan.n_blocks, plan.n_channels
    n_batch = int(cross.shape[0] if cross is not None else gram.shape[0])
    n_arms = len(info.arms)
    lrev = plan.L[::-1]
    n_o = int(_prod(plan.L))
    b = n_blocks

    def c_order(first: int) -> tuple[int, ...]:
        # The axes first … first + B - 1 hold one orientation per block, in reverse order.
        return tuple(first + b - 1 - i for i in range(b))

    sums: list[Any] = []
    shapes: list[tuple[int, ...]] = []
    if cross is not None:
        # [Nb, l3 (B, reversed), arm, l2 (B, reversed), c1, c2] -> [Nb, arm, c1, c2, l2, l3]
        grid = xp.reshape(cross, (n_batch, *lrev, n_arms, *lrev, nc, nc))
        moved = xp.permute_dims(
            grid, (0, b + 1, 2 * b + 2, 2 * b + 3, *c_order(b + 2), *c_order(1))
        )
        rows = xp.reshape(moved, (n_batch, n_arms * nc * nc, n_o * n_o))
        taken = xp.take(rows, filters.s3_rows[info.j3], axis=1)
        sums.append(backend.complex_to_real_pairs(taken))
        shapes.append((int(info.s3_rows.shape[0]), n_o * n_o, 2))
    if gram is not None:
        # [Nb, l3, arm1, l1, c1, arm2, l2, c2] -> [Nb, arm1, c1, arm2, c2, l1, l2, l3]
        grid = xp.reshape(gram, (n_batch, *lrev, n_arms, *lrev, nc, n_arms, *lrev, nc))
        moved = xp.permute_dims(
            grid,
            (
                0,
                b + 1,
                2 * b + 2,
                2 * b + 3,
                3 * b + 4,
                *c_order(b + 2),
                *c_order(2 * b + 4),
                *c_order(1),
            ),
        )
        rows = xp.reshape(moved, (n_batch, (n_arms * nc) ** 2, n_o**3))
        taken = xp.take(rows, filters.s4_rows[info.j3], axis=1)
        sums.append(backend.complex_to_real_pairs(taken))
        shapes.append((int(info.s4_rows.shape[0]), n_o**3, 2))
    # Sums, then their global sum (the hook of distributed runs), then the division.
    flat = [xp.reshape(s, (n_batch, -1)) for s in sums]
    total = _global_sum(xp.concat(flat, axis=1))
    out: list[Any] = []
    offset = 0
    denominators = [filters.s3_denominators[info.j3], filters.s4_denominators[info.j3]]
    if cross is None:
        denominators = denominators[1:]
    for shape, denominator in zip(shapes, denominators, strict=False):
        size = int(_prod(shape))
        part = xp.reshape(total[:, offset : offset + size], (n_batch, *shape))
        out.append(part / xp.reshape(denominator, (1, -1, 1, 1)))
        offset += size
    empty3 = xp.zeros((n_batch, 0, n_o * n_o, 2), dtype=total.dtype)
    if cross is None:
        return empty3, out[0]
    if gram is None:
        return out[0], xp.zeros((n_batch, 0, n_o**3, 2), dtype=total.dtype)
    return out[0], out[1]


# ---------------------------------------------------------------------- helpers


def _bring(
    wavelets: WaveletOperatorTensorProd, x: Any, level: Scale, a: int, target: int
) -> tuple[Any, Scale]:
    """``x`` at ``level``, downsampled along block a until d_a = ``target``."""
    while level[a] < target:
        x = wavelets.downsample(x, level, block=a)
        level = _with(level, a, level[a] + 1)
    return x, level


def _with(level: Scale, a: int, d: int) -> Scale:
    return (*level[:a], d, *level[a + 1 :])


def _prod(values: Any) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out
