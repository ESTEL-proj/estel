"""What the scattering loops share: the hook of distributed runs, the means of the field, the
sums of S1 and S2 of a scale, the Gram products of a level, the step from these products to the
chunks of the buffer, the division of the sums of a level, and the assembly of the buffer (Part
I, §3.1.1; Part II, §9.4, §12.6; Part III, §19.3, §19.9; maths.md §1.6 to 1.7; tasks T2.4,
T2.5).

The loop of single-domain data is ``_loop_single_domain.values_single_domain``, the loop of
tensor products ``_loop_tensor_prod.values_tensor_prod``; both return
:class:`ScatteringValues`, the statistics without normalisation in the compact buffer (Part
III, §19.2), to which ``estel.functional.scattering`` adds the normalisation.

**Sums, then division.** Every mean is a weighted sum over the pixels divided by a denominator
of the plan (M-MEAN). A loop takes the sums of a level (single-domain data) or of a scale
(tensor products), stacks them, passes them to :func:`_global_sum`, then divides
(:func:`_divide_level`); :func:`_global_sum` says when each loop calls it, and on what. In 1.0.0
it is the identity; in step 2 it adds the partial sums of the partitions of the data across MPI
ranks (Part III, §18.3), between the sum and the division, so that the statistics are global.
Both loops read it from this module at each call (``_loop._global_sum``), never as a name
imported from it, so that a replaced hook reaches both.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend._protocol import Backend
from ..data._moments import _backend as _backend_of
from ..data._moments import cov_sum_valid, divide, gram_sum, standardize, sum_valid
from ..statistics._arrays import constant
from ..statistics._layout import Pack, pack

if TYPE_CHECKING:
    from ._plan import ChannelPairs

S1_EPSILON = 1e-8
"""ε of the cross S1 terms, ``z / (|z| + ε)^(1/2)`` (maths.md M-S1, OP-16)."""


def _global_sum(partial_sums: Any) -> Any:
    """The sum, over the partitions of the data, of weighted pixel sums of the means: the hook
    of distributed runs (Part II, §9.4; Part III, §18.3).

    ``partial_sums`` holds weighted sums over the pixels, before their division by the
    denominators of the plan, the batch axis first. The loop of single-domain data calls it once
    per level that has means, on every sum of that level stacked as one real ``(Nb, n)`` array.
    The loop of tensor products calls it once for the mean and var of the field, once per scale
    of the first layer and once per wavelet scale of the second, each time on the sums of that
    scale stacked as one real ``(Nb, n)`` array, and once on the sum of the field at the scale
    with the low-pass filter in every block, before it is centred (M-LP, OP-42): a real
    ``(Nb, 1, …, 1, Nc)`` array, unstacked, one axis per block. With ``standardize``, both loops
    call it once more for each of the two moments, on a real ``(Nb, Nc)`` array. It must
    therefore add element by element, whatever the shape, as an Allreduce does.

    In 1.0.0 the data are never partitioned, and this is the identity. In step 2, each MPI rank
    holds a partition of the data (``DataHealpix(partition=…)``), and this is an MPI Allreduce
    with a sum, so that every rank divides the global sums by the global denominators and
    holds the global statistics.
    """
    return partial_sums


class ScatteringValues(NamedTuple):
    """The statistics without normalisation, in the compact buffer of
    ``statistics._layout.layout_single_domain(plan)`` (Part III, §19.2, §19.4; maths.md M-MOM,
    M-S1 to M-S4, M-CHAN): every coefficient STL computes, and only those, real entries as one
    float, complex entries as (re, im). ``estel.functional.scattering`` stores them in
    ``STStatistics``."""

    buffer: Any
    """``(Nb, n_floats)``, real."""
    mean_pre_std: Any = None
    """With ``standardize``, the mean removed from each map, ``(Nb, Nc)``; otherwise ``None``."""
    std_pre_std: Any = None
    """With ``standardize``, the standard deviation each map was divided by, ``(Nb, Nc)``;
    otherwise ``None``."""


class _Sum(NamedTuple):
    """A weighted sum of the loop, waiting for its level's :func:`_global_sum`."""

    family: str
    value: Any
    denominator: Any
    """A float, or the denominators of the chunks of ``value``, ``(Nb, n, …)``: an array
    ``(n,)`` in the operator's dtype (:func:`gram_chunk_sums`)."""


class FieldMoments(NamedTuple):
    """The field after its standardisation, and the sums of its mean and var
    (:func:`field_moments`)."""

    x: Any
    sums: list[_Sum]
    mean_pre_std: Any
    std_pre_std: Any


def field_moments(
    x: Any, weights: Any, denominator: float, axes: tuple[int, ...], standardized: bool
) -> FieldMoments:
    """The field standardised when ``standardized`` (STL's ``apply(standardize=True)``, M-MOM,
    M-STD), with the mean and the standard deviation removed, and the sums of its ``mean`` and
    ``var`` over the window ``weights`` (``window_single_domain``), waiting for their division
    (M-MOM). Shared by the loops of single-domain data and of tensor products; every sum goes
    through :func:`_global_sum`."""
    mean_pre_std = std_pre_std = None
    if standardized:
        x, mean_pre_std, std_pre_std = standardize(
            x, weights, denominator, axes, global_sum=_global_sum
        )
    sums = [
        _Sum("mean", sum_valid(x, weights, axes), denominator),
        _Sum("var", cov_sum_valid(x, x, weights, axes), denominator),
    ]
    return FieldMoments(x, sums, mean_pre_std, std_pre_std)


def first_layer_sums(
    w: Any, u: Any, weights: Any, denominator: float, pairs: ChannelPairs, axes: tuple[int, ...]
) -> list[_Sum]:
    """The sums of the S1 and S2 means of one scale (M-S1, M-S2, M-CHAN, OP-16), in the order
    :func:`first_layer_floats` reads them: S1 of the auto channels, channel by channel, then of
    the cross pairs, pair by pair; then S2 alike. ``w`` is the first-layer map W_j x and ``u`` its
    modulus,
    ``(Nb, L, Nc, *spatial)``, ``weights`` the window of the means (``window_single_domain``).
    Shared by the loops of single-domain data and of tensor products, whose orientations are
    flattened into L."""
    auto_s1 = sum_valid(u, weights, axes)  # (Nb, L, Nc)
    auto_s2 = cov_sum_valid(u, u, weights, axes)
    sums = [_Sum("S1", auto_s1[:, :, c], denominator) for c in pairs.auto]
    if pairs.cross:
        rooted = divide(w, u, S1_EPSILON, 0.5)  # z / (|z| + eps)^(1/2), OP-16
        sums.extend(
            _Sum(
                "S1", cov_sum_valid(rooted[:, :, c1], rooted[:, :, c2], weights, axes), denominator
            )
            for c1, c2 in pairs.cross
        )
    sums.extend(_Sum("S2", auto_s2[:, :, c], denominator) for c in pairs.auto)
    sums.extend(
        _Sum("S2", cov_sum_valid(w[:, :, c1], w[:, :, c2], weights, axes), denominator)
        for c1, c2 in pairs.cross
    )
    return sums


def gram_sums(v: Any, band: Any, with_s3: bool, with_s4: bool) -> tuple[Any, Any]:
    """The sums of every S3 and S4 mean of one level, as two Gram products (maths_step2.md
    M-GRAM): ``v`` holds the second-layer maps, ``(Nb, L3, n_arms, r, *spatial)``, the band
    orientation leading, then the arms, each with ``r`` rows (its orientations and channels),
    and ``band`` the band map W_{j3} x, ``(Nb, L3, Nc, *spatial)``, both already zero outside
    their windows. Returns the products with the band map, ``(Nb, L3, P, Nc)``, and the Gram
    matrix, ``(Nb, L3, P, P)``, P = n_arms · r, each ``None`` when not asked. Shared by the
    loops of single-domain data and of tensor products."""
    xp = _backend_of(v).xp
    n_batch, n_band = int(v.shape[0]), int(v.shape[1])
    rows = xp.reshape(v, (n_batch, n_band, int(v.shape[2]) * int(v.shape[3]), -1))
    cross = gram = None
    if with_s3:
        cross = gram_sum(rows, xp.reshape(band, (n_batch, n_band, int(band.shape[2]), -1)))
    if with_s4:
        gram = gram_sum(rows)
    return cross, gram


def _divide_level(backend: Backend, pending: Sequence[_Sum], total: Any) -> list[Any]:
    """The means of one level: its weighted sums, stacked as one real ``(Nb, n)`` array, passed
    once to ``total`` (:func:`_global_sum`), split back, then divided by their denominators,
    in their order, each as a real ``(Nb, m)`` array (complex values as (re, im) pairs, the
    axes in C order; a sum of chunks divided chunk by chunk). The stacking is exact: it only
    moves values."""
    if not pending:
        return []
    xp = backend.xp
    flat: list[Any] = []
    for item in pending:
        is_complex = bool(xp.isdtype(item.value.dtype, "complex floating"))
        real = backend.complex_to_real_pairs(item.value) if is_complex else item.value
        flat.append(xp.reshape(real, (real.shape[0], -1)))
    summed = total(xp.concat(flat, axis=1))
    out: list[Any] = []
    offset = 0
    for item, part in zip(pending, flat, strict=True):
        n_batch, size = int(part.shape[0]), int(part.shape[1])
        found = summed[:, offset : offset + size]
        if isinstance(item.denominator, (int, float)):
            out.append(found / item.denominator)
        else:
            n = int(item.value.shape[1])
            per_chunk = xp.reshape(found, (n_batch, n, -1)) / xp.reshape(
                item.denominator, (1, -1, 1)
            )
            out.append(xp.reshape(per_chunk, (n_batch, size)))
        offset += size
    return out


def first_layer_means(
    backend: Backend,
    w: Any,
    u: Any,
    weights: Any,
    denominator: float,
    channels: ChannelPairs,
    pairs: tuple[tuple[int, int], ...],
    orientations: tuple[int, ...],
    axes: tuple[int, ...],
) -> dict[str, Any]:
    """The divided S1 and S2 means of one scale of a tensor product, as the floats of their
    segments (:func:`first_layer_sums`, one :func:`_global_sum`, then
    :func:`first_layer_floats`). ``w`` and ``u`` are W_j x and its modulus, ``(Nb,
    *orientations reversed, Nc, *spatial)``, one orientation axis per block in reverse block
    order (``orientations`` in C order); ``weights`` the window of the means."""
    n_o = math.prod(orientations)
    spatial = tuple(int(n) for n in w.shape[len(w.shape) - len(axes) :])
    shape = (int(w.shape[0]), n_o, channels.n_channels, *spatial)
    xp = backend.xp
    pending = first_layer_sums(
        xp.reshape(w, shape), xp.reshape(u, shape), weights, denominator, channels, axes
    )
    divided = _divide_level(backend, pending, _global_sum)
    return first_layer_floats(backend, divided, channels, pairs, orientations)


def first_layer_floats(
    backend: Backend,
    divided: Sequence[Any],
    channels: ChannelPairs,
    pairs: tuple[tuple[int, int], ...],
    orientations: tuple[int, ...],
) -> dict[str, Any]:
    """The floats of the S1 and S2 segments of one scale from its divided means, in the order
    of :func:`first_layer_sums`: the chunks of the channel pairs ``pairs`` (in the order of the
    buffer), the orientations of each chunk in C order, an auto chunk real, a cross chunk
    complex as (re, im). The orientations of the loop are in reverse block order (one axis for
    single-domain data), ``orientations`` in C order. Shared by the two loops."""
    xp = backend.xp
    n_blocks = len(orientations)
    reverse = orientations[::-1]
    order = [
        *(("S1", (c, c)) for c in channels.auto),
        *(("S1", p) for p in channels.cross),
        *(("S2", (c, c)) for c in channels.auto),
        *(("S2", p) for p in channels.cross),
    ]
    found = dict(zip(order, divided, strict=True))
    # From the reverse block order of the loop to the C order of the blocks.
    to_c = tuple(range(n_blocks - 1, -1, -1))

    def ordered(value: Any, is_complex: bool) -> Any:
        n_batch = int(value.shape[0])
        parts = (2,) if is_complex else ()
        grid = xp.reshape(value, (n_batch, *reverse, *parts))
        tail = tuple(1 + n_blocks + i for i in range(len(parts)))
        moved = xp.permute_dims(grid, (0, *(1 + a for a in to_c), *tail))
        return xp.reshape(moved, (n_batch, -1))

    return {
        name: xp.concat(
            [ordered(found[(name, pair)], pair[0] != pair[1]) for pair in pairs], axis=1
        )
        for name in ("S1", "S2")
    }


class GramRows(NamedTuple):
    """The tables of :func:`gram_chunk_sums` for one scale j3, built in NumPy with the plan:
    for S3 and S4, the rows of the Gram products that are the chunks of the segments of j3, in
    the order of the buffer, and their denominators. They grow with the chunks, never with the
    coefficients (T3.7)."""

    s3_rows: NDArray[np.int64]
    """The rows of S3, ``(arm · Nc + c1) · Nc + c2``, static: NumPy."""
    s3_denominators: Any
    """Their denominators, in the operator's dtype."""
    s4_rows: Any
    """The rows of S4, ``((arm1 · Nc + c1) · n_arms + arm2) · Nc + c2``, int32."""
    s4_denominators: Any
    """Their denominators, in the operator's dtype."""


def gram_chunk_sums(
    backend: Backend,
    cross: Any,
    gram: Any,
    rows: GramRows,
    orientations: tuple[int, ...],
    n_arms: int,
    n_channels: int,
    s3_by_arms: bool,
) -> list[_Sum]:
    """The step from the Gram products of one scale j3 (:func:`gram_sums`) to the chunks of
    its S3 and S4 segments, shared by the two loops (Part III, §19.3): the sums of S3, complex
    ``(Nb, n3, n_o²)``, when ``cross`` is given, then those of S4, ``(Nb, n4, n_o³)``, when
    ``gram`` is, each with its per-chunk denominators, waiting for :func:`_divide_level`. The
    chunks are in the order of the buffer, the orientations of each chunk in C order (l1, l2,
    l3, each with one axis per block of a tensor product).

    The rows P of the products are the arms, each with its orientations (``orientations`` in C
    order, held in reverse block order) and channels; the band orientations l3 lead, in reverse
    block order too. S4 is read where it is, never transposed whole: one ``take`` of the entries
    of its chunks, whose positions are built inside the computation from the rows of the chunks
    and the offsets of their orientations; only the small result is put in the order of the
    buffer. S3, a smaller product, is read by one slice per arm, every channel pair at once
    (``s3_by_arms``, single-domain data), or transposed whole, then one ``take`` of its rows
    (tensor products): the same values either way, and each the read with which JAX compiles
    the gradient of its loop as before this step was shared, bit for bit (XLA contracts
    products and sums differently after another read)."""
    xp = backend.xp
    nc = n_channels
    n_batch = int(cross.shape[0] if cross is not None else gram.shape[0])
    n_o = math.prod(orientations)
    sums: list[_Sum] = []
    if cross is not None and rows.s3_rows.shape[0] > 0:
        # The arms and the channel pairs of the chunks: rows = (pair, arm), pairs outer.
        table = np.asarray(rows.s3_rows, dtype=np.int64)
        arms = tuple(int(a) for a in dict.fromkeys((table // (nc * nc)).tolist()))
        pairs = np.unique(table % (nc * nc))
        if not np.array_equal(table, (pairs[:, None] + np.array(arms) * nc * nc).reshape(-1)):
            raise RuntimeError(
                "the S3 chunks of a scale of the scattering loop are not every arm for every "
                "channel pair, an internal inconsistency of ESTEL: report it as a bug, with the "
                "shape of the data, J, L and cross_matrix."
            )
        if s3_by_arms:
            # [Nb, l3, arm, l2, c1, c2]: one arm at a time, each moved to [Nb, c1, c2, l2, l3]
            # before the arms are stacked, [Nb, c1, c2, arm, l2, l3], then the pairs kept.
            grid = xp.reshape(cross, (n_batch, n_o, n_arms, n_o, nc * nc))
            moved = xp.stack([xp.permute_dims(grid[:, :, a], (0, 3, 2, 1)) for a in arms], axis=2)
            chunks = _c_order(xp, moved, orientations, 2)
            if pairs.shape[0] < nc * nc:
                chunks = xp.take(chunks, constant(backend, pairs, chunks, "int32"), axis=1)
        else:
            # [Nb, l3, arm, l2, c1, c2] -> [Nb, (arm, c1, c2), l2, l3], then the rows kept.
            grid = xp.reshape(cross, (n_batch, n_o, n_arms, n_o, nc * nc))
            moved = _c_order(xp, xp.permute_dims(grid, (0, 2, 4, 3, 1)), orientations, 2)
            flat = xp.reshape(moved, (n_batch, n_arms * nc * nc, n_o * n_o))
            chunks = xp.take(flat, constant(backend, table, flat, "int32"), axis=1)
        value = xp.reshape(chunks, (n_batch, -1, n_o * n_o))
        sums.append(_Sum("S3", value, rows.s3_denominators))
    if gram is not None and int(rows.s4_rows.shape[0]) > 0:
        # [Nb, l3, (arm1, l1, c1, arm2, l2, c2)]: the entries of each chunk at its row's corner
        # plus the offsets of (l1, l2), each l in reverse block order.
        size2 = n_o * nc
        size1 = nc * n_arms * size2
        r = rows.s4_rows
        c2, rest = r % nc, r // nc
        arm2, rest = rest % n_arms, rest // n_arms
        c1, arm1 = rest % nc, rest // nc
        corner = arm1 * (n_o * size1) + c1 * (n_arms * size2) + arm2 * size2 + c2
        reverse = _reverse_positions(orientations)
        offsets = (reverse[:, None] * size1 + reverse[None, :] * nc).reshape(-1)
        positions = xp.reshape(
            xp.reshape(corner, (-1, 1)) + constant(backend, offsets, corner, "int32"), (-1,)
        )
        flat = xp.reshape(gram, (n_batch, n_o, -1))
        taken = xp.reshape(xp.take(flat, positions, axis=2), (n_batch, n_o, -1, n_o * n_o))
        moved = _c_order(xp, xp.permute_dims(taken, (0, 2, 3, 1)), orientations, 1)
        sums.append(_Sum("S4", xp.reshape(moved, (n_batch, -1, n_o**3)), rows.s4_denominators))
    return sums


def _reverse_positions(orientations: tuple[int, ...]) -> NDArray[np.int64]:
    """For each orientation of a map in C order (one index per block), its position among the
    orientations held in reverse block order."""
    strides = np.cumprod((1, *orientations[:-1]))
    grid = np.indices(orientations).reshape(len(orientations), -1)
    found: NDArray[np.int64] = (strides[:, None] * grid).sum(axis=0).astype(np.int64)
    return found


def _c_order(xp: Any, value: Any, orientations: tuple[int, ...], n_groups: int) -> Any:
    """``value``, ``(Nb, n, …, n_o)`` with its last ``n_groups`` axes of n_o orientations each
    held in reverse block order, with each of them in C order: nothing to do for one block."""
    if len(orientations) == 1:
        return value
    b = len(orientations)
    head = tuple(int(n) for n in value.shape[: len(value.shape) - n_groups])
    grid = xp.reshape(value, (*head, *(orientations[::-1] * n_groups)))
    first = len(head)
    order = tuple(first + g * b + b - 1 - i for g in range(n_groups) for i in range(b))
    moved = xp.permute_dims(grid, (*range(first), *order))
    return xp.reshape(moved, (*head, *(math.prod(orientations),) * n_groups))


def compact_buffer(
    backend: Backend,
    families: Mapping[str, Sequence[Any]],
    n_orientations: int,
    s4_pack: Pack,
    s4_tables: Any,
) -> Any:
    """The compact buffer ``(Nb, n_floats)`` of a loop from its divided means, by family in
    the order of the buffer: ``families`` holds, for mean and var, S1, S2 and S3, real
    ``(Nb, m)`` arrays whose concatenation is the floats of the family, and for S4 the divided
    sums of :func:`gram_chunk_sums`, scale by scale, their chunks packed with their real
    diagonal (``statistics._layout.pack``, by ``s4_pack`` and its tables ``s4_tables``;
    ``n_orientations`` is n_o, the orientations of one map). Shared by the two loops."""
    xp = backend.xp
    parts = [*families["moments"], *families["S1"], *families["S2"], *families["S3"]]
    if families["S4"]:
        dense = xp.concat(
            [
                xp.reshape(value, (int(value.shape[0]), -1, n_orientations**3, 2))
                for value in families["S4"]
            ],
            axis=1,
        )
        parts.append(pack(backend, dense, s4_pack, s4_tables))
    return xp.concat(parts, axis=1)
