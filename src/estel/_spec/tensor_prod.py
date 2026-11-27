"""The oracle for tensor products: section 3 of ``docs/maths.md``.

A second implementation of the mathematics, written to be read against ``docs/maths.md`` rather
than to be fast (Part I, §4.2): float64, NumPy and the standard library only, nothing from the
rest of ``estel``. The entries of sections 1 and 2 apply block by block (§3, "Source"), so this
module calls the Cartesian oracle (``cartesian.py``) along the axes of one block at a time: it
moves them to the end, applies the Cartesian function, and moves them back. The moments, the
standardisation, the reference and the flattening are those of ``common.py``.

**Conventions.**

- ``blocks`` lists the number of spatial axes of each block, in block order: 2 for a 2D block
  (section 1), 1 for a 1D block (section 2); ``(2, 1)`` is space and time. Arrays are
  ``(Nb, Nc, *N⁽⁰⁾, …, *N⁽ᴮ⁻¹⁾)``: the spatial axes of the blocks, in block order, last
  (M-DATA-TP). Every function acts on the last ``sum(blocks)`` axes, whatever the leading axes
  are.
- A scale, an orientation and a level are tuples with one entry per block (M-NOT-TP). The
  low-pass filter of block a is its scale J_a (M-LP).
- The field is brought to the level of a scale in every block first, then the block
  convolutions are applied in block order (M-W1-TP, M-S-TP); STL's interleaved order is not
  followed (decisions of the maintainer on issue #126).
- The statistics are STL's full grids (M-S-TP, "Shapes"): ``mean`` and ``var`` ``(Nb, Nc)``,
  real; S1 and S2 ``(Nb, Nc, Nc, *J', *L)``, S3 ``(Nb, Nc, Nc, *J, *J, *L, *L)`` indexed
  ``[c1, c2, *j2, *j3, *l2, *l3]`` and S4 ``(Nb, Nc, Nc, *J, *J, *J, *L, *L, *L)`` indexed
  ``[c1, c2, *j1, *j2, *j3, *l1, *l2, *l3]``, one axis per block in each group, complex, NaN
  where no coefficient exists.

Every public function lists the ``M-`` IDs it implements; ``tools/check_maths_ids.py`` checks
the mapping in both directions. Masks (LF8), the reductions (OP-44) and the spectrum are not
part of tensor products in ESTEL 1.0.0, and are not here.
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import cartesian, common

Array = NDArray[Any]
Statistics = dict[str, Any]
Scale = tuple[int, ...]

ORIENTATIONS = {2: 4, 1: 2}
"""The default L of a block, per number of spatial axes (M-DATA-TP, M-DEF-TP)."""
LOWPASS = ("ST", None)
"""The values of ``lowpass`` for tensor products in ESTEL 1.0.0 (M-LP)."""


def _check_blocks(blocks: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(n) for n in blocks)
    if not out or any(n not in (1, 2) for n in out):
        raise ValueError(
            f"blocks={blocks!r}: pass one entry per block, 2 for a 2D block and 1 for a 1D "
            "block, for example (2, 1) (M-DATA-TP)."
        )
    return out


def _block_sizes(spatial: Sequence[int], blocks: Sequence[int]) -> list[tuple[int, ...]]:
    """The sizes of each block, from the spatial shape."""
    sizes, start = [], 0
    for n_axes in blocks:
        sizes.append(tuple(int(n) for n in spatial[start : start + n_axes]))
        start += n_axes
    return sizes


# ---------------------------------------------------------------------- notation


def spec_along_block_tensor_prod(
    y: Array, blocks: Sequence[int], a: int, function: Callable[[Array], Array]
) -> Array:
    """``function`` applied along the axes of block ``a`` of ``y`` only. Implements M-NOT-TP
    (the operations of one block act on its axes only).

    The axes of block a are moved to the end, ``function`` acts on the last ``blocks[a]`` axes
    (it may change their sizes), and they are moved back to their place.
    """
    n_spatial = sum(blocks)
    start = y.ndim - n_spatial + sum(blocks[:a])
    axes = list(range(start, start + blocks[a]))
    last = list(range(-blocks[a], 0))
    out: Array = np.moveaxis(function(np.moveaxis(y, axes, last)), last, axes)
    return out


def spec_level_tensor_prod(
    y: Array, blocks: Sequence[int], steps: Sequence[int], pbc: Sequence[bool]
) -> Array:
    """``y`` downsampled ``steps[a]`` times along block a, for every block. Implements M-NOT-TP
    (x⁽ᵈ⁾, by M-DOWN along the axes of each block; M-SMOOTH-2D, M-SMOOTH-1D).

    The downsamplings of different blocks act on different axes and commute; they are applied
    block after block, in block order.
    """
    for a, n_axes in enumerate(blocks):
        for _ in range(steps[a]):
            down = functools.partial(cartesian.spec_downsample_cartesian, ndim=n_axes, pbc=pbc[a])
            y = spec_along_block_tensor_prod(y, blocks, a, down)
    return y


# ---------------------------------------------------------------------- data and defaults


def spec_defaults_tensor_prod(
    blocks: Sequence[int], spatial_shape: Sequence[int]
) -> dict[str, Any]:
    """The default parameters of a tensor product with ``blocks`` and ``spatial_shape``.
    Implements M-DEF-TP.

    J_a = ⌊log₂ min N⁽ᵃ⁾⌋ - 2 and L_a = 4 for a 2D block, 2 for a 1D block; K = 5 in every
    block; ``lowpass="ST"``; the other parameters as in M-DEF.
    """
    blocks = _check_blocks(blocks)
    sizes = _block_sizes(spatial_shape, blocks)
    return {
        "J": [math.floor(math.log2(min(block))) - 2 for block in sizes],
        "L": [ORIENTATIONS[n_axes] for n_axes in blocks],
        "K": cartesian.KERNEL_SIZE_CARTESIAN,
        "lowpass": "ST",
        "norm": "self",
        "ref_batch_mean": False,
        "standardize": True,
        "cross_matrix": "upper triangle",
    }


def spec_data_tensor_prod(
    array: Any,
    blocks: Sequence[int],
    pbc: Any,
    J: Any = None,  # noqa: N803
    L: Any = None,  # noqa: N803
) -> tuple[Array, list[int], list[int], list[bool]]:
    """The array promoted to ``(Nb, Nc, *spatial)`` in float64, and the J, L and ``pbc`` of each
    block. Implements M-DATA-TP.

    An array with the spatial axes only is one map, ``(1, 1, …)``; with one more axis, its
    first axis is the channels. The data are real and hold no NaN (OP-41, OP-80); ``pbc`` is one
    boolean per block (OP-4). J and L are lists with one entry per block, an entry ``None`` (or
    no list) giving the default of the block (M-DEF-TP); they are copied, never completed in
    place (OP-83). Each size of block a must be divisible by 2^(J_a - 1) (OP-8). A 1D block
    takes L = 1 or 2 (M-WAV-1D).
    """
    blocks = _check_blocks(blocks)
    n_spatial = sum(blocks)
    x = np.asarray(array)
    if np.iscomplexobj(x):
        raise ValueError("the data must be real (M-DATA-TP); pass a real array.")
    if not n_spatial <= x.ndim <= n_spatial + 2:
        raise ValueError(
            f"the array has {x.ndim} axes; pass ({n_spatial} spatial axes), (Nc, ...) or "
            "(Nb, Nc, ...) (M-DATA-TP)."
        )
    x = np.array(x, dtype=np.float64).reshape((1,) * (n_spatial + 2 - x.ndim) + x.shape)
    if np.isnan(x).any():
        raise ValueError(
            "the data hold NaN: tensor products take no missing data (M-DATA-TP, OP-41, OP-80); "
            "pass data without NaN."
        )
    sizes = _block_sizes(x.shape[2:], blocks)
    defaults = spec_defaults_tensor_prod(blocks, x.shape[2:])
    parameters = {}
    for name, given in (("J", J), ("L", L)):
        if given is None:
            given = [None] * len(blocks)
        # A Python or NumPy integer, or a string, is not a list (OP-83).
        if isinstance(given, str) or np.ndim(given) != 1 or len(given) != len(blocks):
            raise ValueError(
                f"{name}={given!r}: pass a list with one entry per block ({len(blocks)}), an "
                "entry None for the default of the block (M-DATA-TP, OP-83)."
            )
        parameters[name] = [
            default if value is None else int(value)
            for value, default in zip(given, defaults[name], strict=True)
        ]
    n_scales, n_orient = parameters["J"], parameters["L"]
    if isinstance(pbc, bool | np.bool_) or len(pbc) != len(blocks):
        raise ValueError(
            f"pbc={pbc!r}: pass one boolean per block ({len(blocks)}) (M-DATA-TP, OP-4)."
        )
    periodic = list(pbc)
    if not all(isinstance(p, bool | np.bool_) for p in periodic):
        raise ValueError(f"pbc={pbc!r}: pass True or False for each block (M-DATA-TP, OP-4).")
    for a, (n_axes, block) in enumerate(zip(blocks, sizes, strict=True)):
        if n_scales[a] < 1:
            raise ValueError(
                f"block {a} has J = {n_scales[a]}: pass J >= 1 for each block (block sizes "
                f"{block}; M-DATA-TP, OP-83)."
            )
        if any(n % 2 ** (n_scales[a] - 1) for n in block):
            raise ValueError(
                f"the sizes {block} of block {a} are not divisible by 2^(J-1) = "
                f"{2 ** (n_scales[a] - 1)} (M-DATA-TP, OP-8); use a smaller J or other sizes."
            )
        if n_orient[a] < 1 or (n_axes == 1 and n_orient[a] not in (1, 2)):
            raise ValueError(
                f"block {a} has L = {n_orient[a]}: a 1D block takes L = 1 or 2, a 2D block "
                "L >= 1 (M-DATA-TP, M-WAV-1D)."
            )
    return x, n_scales, n_orient, [bool(p) for p in periodic]


# ---------------------------------------------------------------------- low-pass filter


def spec_lowpass_tensor_prod(
    blocks: Sequence[int],
    spatial_shape: Sequence[int],
    J: Sequence[int],  # noqa: N803
    lowpass: str | None = "ST",
) -> list[bool]:
    """Whether each block has a low-pass filter. Implements M-LP (existence; OP-34).

    With ``lowpass="ST"``, block a has one only if each of its sizes is divisible by 2^J_a; with
    ``lowpass=None`` no block has one. Its scale is J_a, realised by the cascade of the
    smoothing kernels: the field at level J_a along the block, with no further computation
    (``spec_first_layer_tensor_prod``).
    """
    if lowpass not in LOWPASS:
        raise ValueError(f'lowpass={lowpass!r}: pass "ST" or None (M-LP).')
    sizes = _block_sizes(spatial_shape, _check_blocks(blocks))
    return [
        lowpass == "ST" and all(n % 2**j_a == 0 for n in block)
        for j_a, block in zip(J, sizes, strict=True)
    ]


# ---------------------------------------------------------------------- index sets


def spec_precedes_tensor_prod(j: Scale, j_prime: Scale) -> bool:
    """j ≤ j': j_a ≤ j'_a in every block. Implements M-IDX-TP (the partial order)."""
    return all(a <= b for a, b in zip(j, j_prime, strict=True))


def spec_index_sets_tensor_prod(J: Sequence[int], lowpass: Sequence[bool]) -> dict[str, list[Any]]:  # noqa: N803
    """The index sets of the statistics, in lexicographic order (the last block fastest).
    Implements M-IDX-TP.

    ``"all"``: J_all, the scales of S1 and S2, 0 ≤ j_a ≤ J'_a - 1 with J'_a = J_a + 1 when block a
    has a low-pass filter; ``"wavelet"``: J_w, 0 ≤ j_a ≤ J_a - 1; ``"S3"``: the pairs
    (j₂, j₃) of J_w with j₂ ≤ j₃; ``"S4"``: the triples (j₁, j₂, j₃) of J_w with j₁ ≤ j₂ ≤ j₃.
    """
    extended = [j_a + int(lp) for j_a, lp in zip(J, lowpass, strict=True)]
    every = list(itertools.product(*(range(n) for n in extended)))
    wavelet = list(itertools.product(*(range(n) for n in J)))
    pairs = [(j2, j3) for j3 in wavelet for j2 in wavelet if spec_precedes_tensor_prod(j2, j3)]
    triples = [
        (j1, j2, j3)
        for j3 in wavelet
        for j2 in wavelet
        if spec_precedes_tensor_prod(j2, j3)
        for j1 in wavelet
        if spec_precedes_tensor_prod(j1, j2)
    ]
    return {"all": every, "wavelet": wavelet, "S3": pairs, "S4": triples}


# ---------------------------------------------------------------------- first layer


def spec_wavelet_transform_tensor_prod(
    y: Array,
    blocks: Sequence[int],
    L: Sequence[int],  # noqa: N803
    pbc: Sequence[bool],
    wavelet: Sequence[bool],
) -> Array:
    """The block convolutions of a tensor wavelet, on ``y`` already at its level, in block
    order. Implements M-W1-TP (without its downsampling, as M-S-TP uses it).

    For each block a with ``wavelet[a]``, ``ψ⁽ᵃ⁾_l ⋆_a`` for l = 0 … L_a - 1, a new orientation
    axis of size L_a; for a block taken at its low-pass scale, nothing is computed and the
    orientation axis has size 1 (orientation 0, M-LP). The orientation axes are placed in block
    order in front of the spatial axes: ``(…, L₀, …, L_{B-1}, *spatial)``. The intermediate
    results are complex: no modulus is taken.
    """
    n_spatial = sum(blocks)
    for a, n_axes in enumerate(blocks):
        position = y.ndim - n_spatial
        if not wavelet[a]:
            y = np.expand_dims(y, position)
            continue
        psi = cartesian._wavelets_cartesian(n_axes, L[a], cartesian.KERNEL_SIZE_CARTESIAN)
        y = np.stack(
            [
                spec_along_block_tensor_prod(
                    y,
                    blocks,
                    a,
                    functools.partial(
                        cartesian.spec_correlate_cartesian, kernel=psi[ell], pbc=pbc[a]
                    ),
                )
                for ell in range(L[a])
            ],
            axis=position,
        )
    return y


def spec_first_layer_tensor_prod(
    x: Array,
    blocks: Sequence[int],
    j: Scale,
    J: Sequence[int],  # noqa: N803
    L: Sequence[int],  # noqa: N803
    pbc: Sequence[bool],
) -> Array:
    """W_{j,l} x for every orientation l of scale ``j``, ``(Nb, Nc, *L', *spatial at level j)``.
    Implements M-W1-TP and M-LP (realisation, the centred scale).

    The field is brought to level d = j in every block (j_a = J_a for the low-pass filter of
    block a, M-LP), then the block wavelets are applied in block order
    (``spec_wavelet_transform_tensor_prod``). L'_a = L_a along a block at a wavelet scale, 1 along a
    block at its low-pass scale. At the scale with the low-pass filter in every block, the
    result is the field X at level (J₀, …, J_{B-1}) centred, X - ⟨X⟩, the mean over all its
    pixels (OP-42). The modulus U_{j,l} = |W_{j,l}| is left to the caller.
    """
    wavelet = [j_a < J_a for j_a, J_a in zip(j, J, strict=True)]
    level = spec_level_tensor_prod(x, blocks, j, pbc)
    if not any(wavelet):
        axes = tuple(range(level.ndim - sum(blocks), level.ndim))
        level = level - level.mean(axis=axes, keepdims=True)
    return spec_wavelet_transform_tensor_prod(level, blocks, L, pbc, wavelet)


# ---------------------------------------------------------------------- borders and means


def spec_border_tensor_prod(
    histories: Sequence[tuple[int, ...]], levels: Sequence[int], pbc: Sequence[bool]
) -> list[int]:
    """The border width of each block, from its own convolution history and level. Implements
    M-BORDER-TP.

    M-BORDER (or its 1D form) block by block: 0 if the block is periodic, or without a
    convolution in it; a block taken at its low-pass scale has an empty history, since the low-
    pass filter is not a convolution in this count (OP-81).
    """
    return [
        cartesian.spec_border_cartesian(history, level, periodic)
        for history, level, periodic in zip(histories, levels, pbc, strict=True)
    ]


def spec_crop_tensor_prod(y: Array, blocks: Sequence[int], borders: Sequence[int]) -> Array:
    """``y`` without the border of each block, along the axes of that block. Implements
    M-BORDER-TP (the crop, and the reduction of a border too large, OP-13).

    A border with 2β_a ≥ the smallest size of block a is reduced to ⌊(min N⁽ᵃ⁾ - 1) / 2⌋, as in
    M-BORDER, block by block.
    """
    n_spatial = sum(blocks)
    index: list[Any] = [slice(None)] * (y.ndim - n_spatial)
    sizes = _block_sizes(y.shape[y.ndim - n_spatial :], blocks)
    for block, border in zip(sizes, borders, strict=True):
        if border and 2 * border >= min(block):
            border = (min(block) - 1) // 2
        index += [slice(border, n - border) for n in block]
    out: Array = y[tuple(index)]
    return out


def spec_mean_tensor_prod(y: Array, blocks: Sequence[int], borders: Sequence[int]) -> Array:
    """The mean of ``y`` over the spatial axes of every block, after the crop of each block.
    Implements M-BORDER-TP and M-MEAN (without mask, LF8)."""
    kept = spec_crop_tensor_prod(y, blocks, borders)
    n_spatial = sum(blocks)
    out: Array = kept.mean(axis=tuple(range(kept.ndim - n_spatial, kept.ndim)))
    return out


# ---------------------------------------------------------------------- statistics


def _at(*index: Any) -> tuple[Any, ...]:
    """An index of the batch axis whole and of the given entries: ``[:, *index]``."""
    return (slice(None), *index)


def _pixels(y: Array, n_spatial: int) -> Array:
    """``y`` with its spatial axes flattened into one, for the sums over pixels."""
    return y.reshape(*y.shape[: y.ndim - n_spatial], -1)


def spec_scattering_tensor_prod(
    x: Array,
    blocks: Sequence[int],
    J: Sequence[int],  # noqa: N803
    L: Sequence[int],  # noqa: N803
    pbc: Sequence[bool],
    lowpass: Sequence[bool],
    cross_matrix: Any = None,
) -> Statistics:
    """The scattering statistics S1 to S4 of ``x``, ``(Nb, Nc, *spatial)``, in STL's full
    grids. Implements M-S-TP, with M-IDX-TP, M-LP, M-BORDER-TP and M-CHAN.

    S1 and S2 (M-S1, M-S2) at every scale of J_all, on W_{j,l} of ``spec_first_layer_tensor_prod``:
    the centred field at the scale with the low-pass filter in every block, orientation 0 only along
    a block at its low-pass scale. S3 and S4 (M-S3, M-S4) on the wavelet scales only (OP-43):
    U_{j₂,l₂} at level j₂ is downsampled block by block to the level of j₃ (D_{j₂→j₃}), then the
    block convolutions of scale j₃ are applied (``spec_wavelet_transform_tensor_prod``). Every mean
    is taken at the level of its arguments, over the pixels left by the larger of their two borders,
    block by block. S1 and S2 are filled for the pairs c1 ≤ c2 of ``cross_matrix`` only; S3 and S4
    for both orders of each pair. Everything is stored complex, and NaN where it is not computed.
    """
    blocks = _check_blocks(blocks)
    n_blocks, n_spatial = len(blocks), sum(blocks)
    nb, nc = x.shape[:2]
    pairs = common._channel_pairs(cross_matrix, nc)
    sets = spec_index_sets_tensor_prod(J, lowpass)
    extended = [j_a + int(lp) for j_a, lp in zip(J, lowpass, strict=True)]
    n_orient = math.prod(L)

    # Layer 1: W_j x at level j for every j of J_all, (Nb, Nc, *L', *spatial), and its borders.
    w = {j: spec_first_layer_tensor_prod(x, blocks, j, J, L, pbc) for j in sets["all"]}
    border1 = {
        j: spec_border_tensor_prod(
            [(j_a,) if j_a < J_a else () for j_a, J_a in zip(j, J, strict=True)], j, pbc
        )
        for j in sets["all"]
    }

    # Layer 2: W_{j3} D_{j2→j3} U_{j2} x at level j3, (Nb, Nc, *L(l2), *L(l3), *spatial).
    every_block = [True] * n_blocks
    v: dict[tuple[Scale, Scale], Array] = {}
    for j2, j3 in sets["S3"]:
        steps = [b - a for a, b in zip(j2, j3, strict=True)]
        modulus = spec_level_tensor_prod(np.abs(w[j2]), blocks, steps, pbc)
        v[j2, j3] = spec_wavelet_transform_tensor_prod(modulus, blocks, L, pbc, every_block)

    def border2(j2: Scale, j3: Scale) -> list[int]:
        return spec_border_tensor_prod(list(zip(j2, j3, strict=True)), j3, pbc)

    s1 = np.full((nb, nc, nc, *extended, *L), np.nan, dtype=np.complex128)
    s2 = np.full((nb, nc, nc, *extended, *L), np.nan, dtype=np.complex128)
    s3 = np.full((nb, nc, nc, *J, *J, *L, *L), np.nan, dtype=np.complex128)
    s4 = np.full((nb, nc, nc, *J, *J, *J, *L, *L, *L), np.nan, dtype=np.complex128)

    for c1 in range(nc):
        for c2 in range(c1, nc):
            if not pairs[c1, c2]:
                continue
            orders = [(c1, c2)] if c1 == c2 else [(c1, c2), (c2, c1)]

            for j in sets["all"]:
                orientations = w[j].shape[2 : 2 + n_blocks]
                for ell in itertools.product(*(range(n) for n in orientations)):
                    z1, z2 = w[j][_at(c1, *ell)], w[j][_at(c2, *ell)]
                    if c1 == c2:
                        first = np.abs(z1)
                        second = np.abs(z1) ** 2
                    else:
                        r1 = z1 / np.sqrt(np.abs(z1) + common.S1_EPSILON)
                        r2 = z2 / np.sqrt(np.abs(z2) + common.S1_EPSILON)
                        first = r1 * np.conj(r2)
                        second = z1 * np.conj(z2)
                    s1[_at(c1, c2, *j, *ell)] = spec_mean_tensor_prod(first, blocks, border1[j])
                    s2[_at(c1, c2, *j, *ell)] = spec_mean_tensor_prod(second, blocks, border1[j])

            # S3 and S4: the sums over pixels, every orientation at once, after the crop.
            # Orientations are flattened in row-major order (block 0 slowest), then unflattened.
            for a, b in orders:
                for j2, j3 in sets["S3"]:
                    border = np.maximum(border2(j2, j3), border1[j3]).tolist()
                    second_layer = _pixels(
                        spec_crop_tensor_prod(v[j2, j3][:, a], blocks, border), n_spatial
                    )
                    first_layer = _pixels(
                        spec_crop_tensor_prod(w[j3][:, b], blocks, border), n_spatial
                    )
                    second_layer = second_layer.reshape(nb, n_orient, n_orient, -1)
                    first_layer = first_layer.reshape(nb, n_orient, -1)
                    total = np.einsum("bikp,bkp->bik", second_layer, np.conj(first_layer))
                    s3[_at(a, b, *j2, *j3)] = (total / second_layer.shape[-1]).reshape(nb, *L, *L)
                for j1, j2, j3 in sets["S4"]:
                    border = np.maximum(border2(j1, j3), border2(j2, j3)).tolist()
                    left = _pixels(
                        spec_crop_tensor_prod(v[j1, j3][:, a], blocks, border), n_spatial
                    )
                    right = _pixels(
                        spec_crop_tensor_prod(v[j2, j3][:, b], blocks, border), n_spatial
                    )
                    left = left.reshape(nb, n_orient, n_orient, -1)
                    right = right.reshape(nb, n_orient, n_orient, -1)
                    total = np.einsum("bikp,bjkp->bijk", left, np.conj(right))
                    s4[_at(a, b, *j1, *j2, *j3)] = (total / left.shape[-1]).reshape(nb, *L, *L, *L)
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}


# ---------------------------------------------------------------------- normalisation


def spec_normalize_tensor_prod(
    stats: Statistics,
    reference: Statistics,
    J: Sequence[int],  # noqa: N803
) -> Statistics:
    """The statistics divided by the reference, as M-NORM with multi-indices. Implements
    M-NORM-TP.

    The reference is that of M-NORM (``common.spec_reference``): ``sigma``, ``(Nr, Nc, *J',
    *L)``, the square root of the channel diagonal of S2, low-pass scales included. ``mean`` is
    not normalised; ``var`` is divided by ``var_ref``; S1 and S2 at every scale of J_all, by
    √(sigma_{c1} sigma_{c2}) and sigma_{c1} sigma_{c2} at the same scale and orientation; S3 by
    sigma_{c1}(j₂, l₂) sigma_{c2}(j₃, l₃) and S4 by sigma_{c1}(j₁, l₁) sigma_{c2}(j₂, l₂), with
    the wavelet scales of the reference only. A reference of one batch element applies to every
    batch element.
    """
    sigma = reference["sigma"]
    n_blocks = len(J)
    nr, nc = sigma.shape[:2]
    orientations = sigma.shape[2 + n_blocks :]
    ones = (1,) * n_blocks
    wavelet = list(itertools.product(*(range(n) for n in J)))
    s1, s2, s3, s4 = (np.array(stats[k]) for k in common.SCATTERING)
    with np.errstate(invalid="ignore"):  # NaN where no coefficient or no sigma exists
        for c1 in range(nc):
            for c2 in range(nc):
                divisor = sigma[:, c1] * sigma[:, c2]  # (Nr, *J', *L)
                s1[:, c1, c2] = s1[:, c1, c2] / np.sqrt(divisor)
                s2[:, c1, c2] = s2[:, c1, c2] / divisor
                for j2, j3 in itertools.product(wavelet, wavelet):
                    # [b, *l2, *l3]
                    outer = sigma[_at(c1, *j2)].reshape(nr, *orientations, *ones) * sigma[
                        _at(c2, *j3)
                    ].reshape(nr, *ones, *orientations)
                    index = _at(c1, c2, *j2, *j3)
                    s3[index] = s3[index] / outer
                for j1, j2 in itertools.product(wavelet, wavelet):
                    # Every j3 and l3 at once: [b, *j3, *l1, *l2, *l3].
                    outer = sigma[_at(c1, *j1)].reshape(nr, *ones, *orientations, *ones, *ones)
                    outer = outer * sigma[_at(c2, *j2)].reshape(
                        nr, *ones, *ones, *orientations, *ones
                    )
                    index = _at(c1, c2, *j1, *j2)
                    s4[index] = s4[index] / outer
    return {
        **stats,
        "var": stats["var"] / reference["var"],
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
    }


# ---------------------------------------------------------------------- apply


def spec_apply_tensor_prod(
    array: Any,
    blocks: Sequence[int],
    pbc: Any,
    J: Any = None,  # noqa: N803
    L: Any = None,  # noqa: N803
    *,
    lowpass: str | None = "ST",
    norm: str | None = None,
    ref: Statistics | None = None,
    ref_batch_mean: bool = False,
    standardize: bool = True,
    cross_matrix: Any = None,
    flatten: bool = False,
) -> Statistics:
    """The statistics of a tensor-product field: the whole of ``apply``, in its order (M-ORDER).
    Implements M-DATA-TP, M-DEF-TP, M-S-TP, M-NORM-TP, and M-MOM, M-STD and M-FLAT over the
    spatial axes of every block together.

    1. promotion to ``(Nb, Nc, …)``, J and L per block, and channel pairs (M-DATA-TP, M-DEF-TP,
       M-CHAN); the low-pass filters (M-LP);
    2. standardisation, if asked, over the spatial axes of every block (M-STD);
    3. ``mean`` and ``var`` over the same axes, without border (M-MOM);
    4. the scattering statistics S1 to S4 (M-S-TP);
    5. normalisation (M-NORM-TP): ``norm="self"`` by the statistics themselves, ``norm="ref"``
       by ``ref``, the ``"reference"`` of other statistics, ``norm="none"`` not at all;
    6. the flattening, if asked (M-FLAT), in its real form, under ``"vector"``.

    Returns the families, the ``"reference"`` used (``None`` with ``norm="none"``), the
    ``"J"``, ``"L"`` and ``"lowpass"`` flags of the blocks and, with ``standardize``,
    ``"mean_pre_std"`` and ``"std_pre_std"``. ``norm`` must be given (M-NORM); ``standardize`` is
    True by default (M-DEF-TP, OP-120).
    """
    if norm not in ("self", "ref", "none"):
        raise ValueError(f'norm={norm!r}: pass "self", "ref" or "none" (M-NORM).')
    if norm == "self" and ref is not None:
        raise ValueError('norm="self" takes no reference (M-NORM).')
    if norm == "ref" and ref is None:
        raise ValueError('norm="ref" needs a reference (M-NORM).')

    x, n_scales, n_orient, periodic = spec_data_tensor_prod(array, blocks, pbc, J, L)  # 1
    flags = spec_lowpass_tensor_prod(blocks, x.shape[2:], n_scales, lowpass)
    out: Statistics = {"J": n_scales, "L": n_orient, "lowpass": flags}
    # Every pixel of every block, as one axis, for the means of M-MOM and M-STD.
    flat = x.reshape(*x.shape[:2], -1)
    if standardize:  # 2
        flat, out["mean_pre_std"], out["std_pre_std"] = common.spec_standardize(flat, 1)
        x = flat.reshape(x.shape)
    out["mean"], out["var"] = common.spec_moments(flat, 1)  # 3
    out.update(  # 4
        spec_scattering_tensor_prod(x, blocks, n_scales, n_orient, periodic, flags, cross_matrix)
    )
    out["reference"] = None
    if norm != "none":  # 5
        reference = common.spec_reference(out, ref_batch_mean) if norm == "self" else ref
        assert reference is not None
        out = {**spec_normalize_tensor_prod(out, reference, n_scales), "reference": reference}
    if flatten:  # 6
        out["vector"] = common.spec_flatten(out, flatten_complex=True)
    return out
