"""What the scattering loops share: the hook of distributed runs, the means of the field, the
sums of S1 and S2 of a scale, the Gram products of a level, and the division of the sums of a
level (Part I, §3.1.1; Part II, §9.4, §12.6; maths.md §1.6 to 1.7; tasks T2.4, T2.5).

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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from ..backend._protocol import Backend
from ..data._moments import _backend as _backend_of
from ..data._moments import cov_sum_valid, divide, gram_sum, standardize, sum_valid

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
    ``statistics._layout.layout_single_domain(plan)`` (Part III, §19.2, §19.4; maths.md M-MOM, M-S1 to
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
    (M-MOM). Shared by the loops of planar data and of tensor products; every sum goes through
    :func:`_global_sum`."""
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
    """The sums of the S1 and S2 means of one scale (M-S1, M-S2, M-CHAN, OP-16), in the order of
    ``loop_sums``: S1 of the auto channels, every channel at once, then of the cross pairs, pair
    by pair; then S2 alike. ``w`` is the first-layer map W_j x and ``u`` its modulus,
    ``(Nb, L, Nc, *spatial)``, ``weights`` the window of the means (``window_single_domain``).
    Shared by the loops of planar data and of tensor products, whose orientations are flattened into
    L."""
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
