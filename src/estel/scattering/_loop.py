"""The scattering loop: STL's loop structure, in functional style (Part I, §3.1.1; Part II,
§12.6; maths.md §1.6 to 1.7; tasks T2.4, T2.5).

:func:`scattering_values` is the pure function at the core of ``estel.functional.scattering``:
it takes the data, the static plan and the filters, and returns the statistics without
normalisation, in the compact buffer (:class:`ScatteringValues`; Part III, §19.2). The
normalisation and the ``STStatistics`` object are added by ``estel.functional.scattering``.

It follows STL's loop over the levels (``ST_Operator.apply`` at `master` ``8eaf769``,
``_compute_ST_loop`` at `2D+1D` ``8f51143``): for each level d, and for each scale j3 computed
at d, read from the plan (``plan.scale_levels``; in 1.0.0 the scale j3 = d alone, maths.md
M-NOT), the first layer W_{j3} x and U_{j3} = |W_{j3} x| at level d, the S1 and S2 means; the
second layer ψ_{l3} ⋆ U_{j2} at level d for every j2 ≤ j3 at once, and the S3 and S4 means of
j3; then the means of the level are divided, and the field and every U_{j2} go down one
level.

**The second layer is a Gram product** (maths_step2.md M-GRAM; Part III, §19.3; task T2.12).
Where STL loops over j2 and j1 and takes one mean of a product array per pair of scales, the
second-layer maps of a level are computed by one convolution, with the band orientation l3 as
their leading axis, ``(Nb, L3, j2, L2, Nc, *spatial)``: for one l3, the maps of every scale,
channel and orientation are the rows of a matrix A, with no copy. Each map is set to zero
outside its own window (its border crop and its valid pixels), and the sums of every S4 of the
level are the matrix A A^H, those of every S3 the product of A with the windowed band map
W_{j3} x: one matrix product each, for all the band orientations (:func:`gram_sum`, by blocks
of pixels added in double precision when the data are in single precision). The sum of a pair
then runs over the intersection of its two windows, the larger of the two borders and the
union of the two masks, which is STL's window for that pair, and it is divided by the number
of pixels of that intersection, the denominator of the plan. The product also holds the pairs
STL does not compute (j1 > j2, about half of it): they are not read. Every value is STL's up
to the order of the sums. The convolutions and the downsamplings are those of the wavelet operator
(``WaveletOperator.apply``, ``downsample``; Part I, §2.3): the loop sees levels, maps and
weights, and does not depend on the geometry. Nothing is written in place (Part I, §3.1,
rule 1): every step returns a new array, the
divided sums of each level are collected in a list, in the order of ``_plan.loop_sums``, and one
static gather, built in NumPy with the plan (``statistics._layout.loop_gather``), concatenates
them, chunk by chunk, into the compact buffer at the end (Part III, §19.3, §19.9): no full-grid
tensor is written in the computation (review rule X5). The loops run over scales, levels and
channel pairs, as STL's, never over pixels or orientations.

**Sums, then division.** Every mean is a weighted sum over the pixels divided by a denominator
of the plan (M-MEAN). The loop takes the sums of a level, stacks them, passes them once to
:func:`_global_sum`, then divides. In 1.0.0 :func:`_global_sum` is the identity; in step 2 it
adds the partial sums of the partitions of the data across MPI ranks (Part III, §18.3), between
the sum and the division, so that the statistics are global.

Missing values are weights (Part I, §3.1.1; maths.md OP-27): the field is set to 0 at its
missing pixels before anything else, so that every intermediate array is finite, and every mean
reads only the pixels its weights keep (M-MEAN, M-MASK, M-BORDER). The means the plan lists as
empty (OP-3) are not computed: they are absent from the buffer, and NaN in the full-grid view,
like the structural NaN (M-CHAN).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import Data
from ..data._moments import _backend as _backend_of
from ..data._moments import cov_sum_valid, divide, gram_sum, modulus, standardize, sum_valid
from ..statistics._layout import gather_loop, loop_gather
from ..wavelets._operator import WaveletOperator
from ._filters import PlanarFilters, window
from ._plan import ChannelPairs, PlanarPlan

S1_EPSILON = 1e-8
"""ε of the cross S1 terms, ``z / (|z| + ε)^(1/2)`` (maths.md M-S1, OP-16)."""


def _global_sum(partial_sums: Any) -> Any:
    """The sum, over the partitions of the data, of the weighted pixel sums of a level's means:
    the hook of distributed runs (Part II, §9.4; Part III, §18.3).

    ``partial_sums`` is ``(Nb, n)``, real: every weighted sum the loop takes at one level,
    stacked, before the division by the denominators of the plan. Called once per level that
    has means, and once for each of the two moments of ``standardize``.

    In 1.0.0 the data are never partitioned, and this is the identity. In step 2, each MPI rank
    holds a partition of the data (``DataHealpix(partition=…)``), and this is an MPI Allreduce
    with a sum, so that every rank divides the global sums by the global denominators and
    holds the global statistics.
    """
    return partial_sums


class ScatteringValues(NamedTuple):
    """The statistics without normalisation, in the compact buffer of
    ``statistics._layout.base_layout(plan)`` (Part III, §19.2, §19.4; maths.md M-MOM, M-S1 to
    M-S4, M-CHAN): every coefficient STL computes, and only those, real entries as one float,
    complex entries as (re, im). ``estel.functional.scattering`` stores them in
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
    denominator: float


def first_layer_sums(
    w: Any, u: Any, weights: Any, denominator: float, pairs: ChannelPairs, axes: tuple[int, ...]
) -> list[_Sum]:
    """The sums of the S1 and S2 means of one scale (M-S1, M-S2, M-CHAN, OP-16), in the order of
    ``loop_sums``: S1 of the auto channels, every channel at once, then of the cross pairs, pair
    by pair; then S2 alike. ``w`` is the first-layer map W_j x and ``u`` its modulus,
    ``(Nb, L, Nc, *spatial)``, ``weights`` the window of the means (``window``). Shared by the
    loops of planar data and of tensor products, whose orientations are flattened into L."""
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
    loops of planar data and of tensor products."""
    xp = _backend_of(v).xp
    n_batch, n_band = int(v.shape[0]), int(v.shape[1])
    rows = xp.reshape(v, (n_batch, n_band, int(v.shape[2]) * int(v.shape[3]), -1))
    cross = gram = None
    if with_s3:
        cross = gram_sum(rows, xp.reshape(band, (n_batch, n_band, int(band.shape[2]), -1)))
    if with_s4:
        gram = gram_sum(rows)
    return cross, gram


# ---------------------------------------------------------------------- the loop


def scattering_values(data: Data, plan: PlanarPlan, filters: PlanarFilters) -> ScatteringValues:
    """The statistics of ``data`` without normalisation (maths.md M-MOM, M-W1, M-S1 to M-S4),
    STL's ``ST_Operator`` with ``norm="vanilla"`` and no spectrum.

    ``data`` has the static metadata ``plan`` was built from (geometry, spatial shape, ``pbc``,
    mask, number of channels); ``filters`` are those of the plan, on the data's backend and
    dtype. With ``plan.standardize``, each map is first standardised (STL's
    ``apply(standardize=True)``, M-MOM), and the mean and standard deviation removed are
    returned with the statistics. Pure: nothing is modified, nothing depends on array values.
    """
    backend = _registry.backend_named(data.backend)
    xp = backend.xp
    axes = tuple(range(-data.ndim, 0))
    pairs = plan.pairs
    wavelets = WaveletOperator.from_parts(data, plan.wavelets, filters.wavelets)

    def total(x: Any) -> Any:
        return _global_sum(x)

    # The field, finite everywhere: missing pixels are 0, never read (OP-27).
    x = data.array
    if filters.valid is not None:
        x = xp.where(filters.valid, x, xp.zeros_like(x))
    moments = plan.moments
    weights = window(moments, filters.moments)
    mean_pre_std = std_pre_std = None
    if plan.standardize:
        x, mean_pre_std, std_pre_std = standardize(
            x, weights, moments.denominator, axes, global_sum=total
        )
    pending: list[_Sum] = [
        _Sum("mean", sum_valid(x, weights, axes), moments.denominator),
        _Sum("var", cov_sum_valid(x, x, weights, axes), moments.denominator),
    ]
    divided: list[Any] = []  # the divided sums of each level, as real (Nb, n) arrays

    moduli: dict[int, Any] = {}  # U_{j2} x of every scale computed so far, at the current level
    for level in plan.levels:
        # The scales computed at this level, from the plan: in 1.0.0, the scale j3 = level.
        for j3 in (j for j in plan.scales if plan.scale_levels[j] == level):
            w = wavelets.apply(x, j3)  # W_{j3} x: (Nb, L, Nc, *spatial)
            u = modulus(w)  # U_{j3} x
            moduli = {**moduli, j3: u}

            if j3 in plan.first_layer:
                weights = window(plan.first_layer[j3], filters.first_layer[j3])
                denominator = plan.first_layer[j3].denominator
                pending.extend(first_layer_sums(w, u, weights, denominator, pairs, axes))

            if j3 in plan.second_layer:
                pending.extend(_second_layer(backend, wavelets, plan, filters, j3, moduli, w))

        divided.extend(_divide_level(backend, pending, total))
        pending = []

        if level < plan.levels[-1]:
            x = wavelets.downsample(x, level)
            moduli = {j2: wavelets.downsample(u2, level, (j2,)) for j2, u2 in moduli.items()}

    # One static gather from the sums, in the order of loop_sums, to the compact buffer
    # (Part III, §19.3, §19.9), chunk by chunk: no full-grid tensor inside the computation
    # (review rule X5).
    gather = loop_gather(plan)
    if len(divided) != gather.n_sums:
        raise RuntimeError(
            f"the loop produced {len(divided)} sums where the plan expects {gather.n_sums}: "
            "scattering._plan.loop_sums no longer follows the loop; make them agree."
        )
    return ScatteringValues(
        buffer=gather_loop(backend, divided, gather, filters.gather),
        mean_pre_std=mean_pre_std,
        std_pre_std=std_pre_std,
    )


def _second_layer(
    backend: Backend,
    wavelets: WaveletOperator,
    plan: PlanarPlan,
    filters: PlanarFilters,
    j3: int,
    moduli: dict[int, Any],
    w: Any,
) -> list[_Sum]:
    """The sums of the S3 and S4 means of scale ``j3``, at the level of ``j3``, in the order of
    STL's loop (for j2, S3 then, for j1, S4; the channel pairs inside), as two Gram products
    (M-GRAM).

    ``moduli`` holds U_{j2} at the level of j3, ``(Nb, L, Nc, *spatial)``, for every j2 ≤ j3, and
    ``w`` is the band map W_{j3} x, ``(Nb, L, Nc, *spatial)``: every map has the orientation
    of its last convolution right after the batch axis.
    """
    xp = backend.xp
    maps = plan.second_layer[j3]
    pairs, n_channels, L = plan.pairs, plan.n_channels, plan.L  # noqa: N806
    # The moduli of the scales j2 ≤ j3, stacked (real, 1/(2L) of the second layer), then one
    # convolution: ψ_{l3} ⋆ U_{j2} x, (Nb, L3, j2, L2, Nc, *spatial), l3 leading.
    arms = xp.stack([moduli[j2] for j2 in maps.arms], axis=1)
    v = wavelets.apply(arms, j3)
    band = w  # (Nb, L3, Nc, *spatial)
    windows = filters.second_layer[j3]
    with_s3 = any((j2, j3) in plan.s3 for j2 in maps.arms)
    with_s4 = any(key[2] == j3 for key in plan.s4)
    n_batch, n_arms = int(v.shape[0]), len(maps.arms)
    row = {j2: position for position, j2 in enumerate(maps.arms)}
    grid = (n_batch, L, n_arms, L, n_channels)
    if windows is not None:
        # Each map is zero outside its own window: a pair is then summed over the
        # intersection of its two windows, STL's window for that pair.
        of_arms = xp.stack([windows[i] for i in maps.arm_windows], axis=0)
        v = v * of_arms[(slice(None), None, None, *([slice(None)] * plan.ndim))]
        if with_s3 and maps.band_window is not None:
            band = band * windows[maps.band_window]
    cross, gram = gram_sums(
        xp.reshape(v, (n_batch, L, n_arms, L * n_channels, *tuple(v.shape[5:]))),
        band,
        with_s3,
        with_s4,
    )
    if cross is not None:
        cross = xp.reshape(cross, (*grid, n_channels))  # [Nb, l3, j2, l2, c1, c2]
    if gram is not None:
        gram = xp.reshape(gram, (*grid, *grid[2:]))  # [Nb, l3, j1, l1, c1, j2, l2, c2]
    sums: list[_Sum] = []
    for j2 in maps.arms:
        if cross is not None and (j2, j3) in plan.s3:
            denominator = plan.s3[(j2, j3)].denominator
            sums.extend(
                # (Nb, L3, L2) -> (Nb, L2, L3)
                _Sum("S3", xp.permute_dims(cross[:, :, row[j2], :, c1, c2], (0, 2, 1)), denominator)
                for c1, c2 in pairs.ordered
            )
        for j1 in range(j2 + 1):
            if gram is not None and (j1, j2, j3) in plan.s4:
                denominator = plan.s4[(j1, j2, j3)].denominator
                sums.extend(
                    # (Nb, L3, L1, L2) -> (Nb, L1, L2, L3)
                    _Sum(
                        "S4",
                        xp.permute_dims(gram[:, :, row[j1], :, c1, row[j2], :, c2], (0, 2, 3, 1)),
                        denominator,
                    )
                    for c1, c2 in pairs.ordered
                )
    return sums


def _divide_level(backend: Backend, pending: Sequence[_Sum], total: Any) -> list[Any]:
    """The means of one level: its weighted sums, stacked as one real ``(Nb, n)`` array, passed
    once to ``total`` (:func:`_global_sum`), split back, then divided by their denominators,
    in their order, each as a real ``(Nb, m)`` array (complex values as (re, im) pairs, the
    order of ``loop_sums``). The stacking is exact: it only moves values."""
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
        size = int(part.shape[1])
        out.append(summed[:, offset : offset + size] / item.denominator)
        offset += size
    return out
