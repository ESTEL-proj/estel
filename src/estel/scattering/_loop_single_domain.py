"""The scattering loop of single-domain data (1D, 2D and HEALPix): STL's loop structure, in
functional style (Part I, §3.1.1; Part II, §12.6; maths.md §1.6 to 1.7; tasks T2.4, T2.5).

:func:`values_single_domain` is the pure function at the core of
``estel.functional.scattering`` for single-domain data: it takes the data, the static plan and
the filters, and returns the statistics without normalisation, in the compact buffer
(``_loop.ScatteringValues``; Part III, §19.2). The normalisation and the ``STStatistics``
object are added by ``estel.functional.scattering``. What it shares with the loop of tensor
products (``_loop_tensor_prod``) is in ``_loop``.

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
(``WaveletOperatorSingleDomain.apply``, ``downsample``; Part I, §2.3): the loop sees levels, maps
and weights, and does not depend on the geometry. Nothing is written in place (Part I, §3.1,
rule 1): every step returns a new array. The step from the Gram products to the chunks of the
buffer is the one of tensor products (``_loop.gram_chunk_sums``): the S3 and S4 chunks of j3
are read from the products by one slice per arm and one ``take``, with the rows of its chunks,
built in NumPy with the plan (``statistics._layout.gather_single_domain``); the S1 and S2 chunks
of a scale are put in the order of the buffer (``_loop.first_layer_floats``); and the families
are concatenated at the end, S4 packed with its real diagonal (``_loop.compact_buffer``; Part
III, §19.3, §19.9): no full-grid tensor is written in the computation (review rule X5). The
loops run over scales, levels and channel pairs, as STL's, never over pixels or orientations.

**Sums, then division.** Every mean is a weighted sum over the pixels divided by a denominator
of the plan (M-MEAN). The loop takes the sums of a level, stacks them, passes them once to the
hook ``_loop._global_sum``, read from its module at each call, then divides (``_loop``).

Missing values are weights (Part I, §3.1.1; maths.md OP-27): the field is set to 0 at its
missing pixels before anything else, so that every intermediate array is finite, and every mean
reads only the pixels its weights keep (M-MEAN, M-MASK, M-BORDER). The means the plan lists as
empty (OP-3) are not computed: they are absent from the buffer, and NaN in the full-grid view,
like the structural NaN (M-CHAN).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import Data
from ..data._moments import modulus
from ..spectra._estimators import spectrum_floats_2d
from ..statistics._layout import GatherSingleDomain, gather_single_domain, spectrum_pairs
from ..wavelets._operator import WaveletOperatorSingleDomain
from . import _loop
from ._filters import STFiltersSingleDomain, window_single_domain
from ._loop import (
    GramRows,
    ScatteringValues,
    _divide_level,
    _Sum,
    compact_buffer,
    field_moments,
    first_layer_floats,
    first_layer_sums,
    gram_chunk_sums,
    gram_sums,
)

if TYPE_CHECKING:
    from ._plan import STPlanSingleDomain


def values_single_domain(
    data: Data, plan: STPlanSingleDomain, filters: STFiltersSingleDomain
) -> ScatteringValues:
    """The statistics of ``data`` without normalisation (maths.md M-MOM, M-W1, M-S1 to M-S4, and
    M-PS-2D when the plan holds the power spectrum), STL's ``ST_Operator`` with
    ``norm="vanilla"``.

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
    wavelets = WaveletOperatorSingleDomain.from_parts(data, plan.wavelets, filters.wavelets)

    def total(x: Any) -> Any:
        return _loop._global_sum(x)

    # The field, finite everywhere: missing pixels are 0, never read (OP-27).
    x = data.array
    if filters.valid is not None:
        x = xp.where(filters.valid, x, xp.zeros_like(x))
    moments = plan.moments
    x, pending, mean_pre_std, std_pre_std = field_moments(
        x,
        window_single_domain(moments, filters.moments),
        moments.denominator,
        axes,
        plan.standardize,
    )
    field = x  # at full resolution, after the standardisation: the field of the spectrum
    gather = gather_single_domain(plan)
    families: dict[str, list[Any]] = {"moments": [], "S1": [], "S2": [], "S3": [], "S4": []}
    pieces: list[tuple[str, int]] = [("moments", len(pending))]  # what pending holds, in order

    moduli: dict[int, Any] = {}  # U_{j2} x of every scale computed so far, at the current level
    for level in plan.levels:
        # The scales computed at this level, from the plan: in 1.0.0, the scale j3 = level.
        for j3 in (j for j in plan.scales if plan.scale_levels[j] == level):
            w = wavelets.apply(x, j3)  # W_{j3} x: (Nb, L, Nc, *spatial)
            u = modulus(w)  # U_{j3} x
            moduli = {**moduli, j3: u}

            if j3 in plan.first_layer:
                weights = window_single_domain(plan.first_layer[j3], filters.first_layer[j3])
                denominator = plan.first_layer[j3].denominator
                found = first_layer_sums(w, u, weights, denominator, pairs, axes)
                pending, pieces = [*pending, *found], [*pieces, ("first", len(found))]

            if j3 in plan.second_layer:
                found = _second_layer(backend, wavelets, plan, filters, j3, moduli, w)
                pending, pieces = [*pending, *found], [*pieces, ("second", len(found))]

        families = _collect(backend, plan, gather, families, pieces, pending, total)
        pending, pieces = [], []

        if level < plan.levels[-1]:
            x = wavelets.downsample(x, level)
            moduli = {j2: wavelets.downsample(u2, level, (j2,)) for j2, u2 in moduli.items()}

    # The families concatenated in the order of the buffer (Part III, §19.3, §19.9), the S4
    # chunks packed with their real diagonal: no full-grid tensor inside the computation (review
    # rule X5).
    buffer = compact_buffer(backend, families, plan.L, gather.s4, filters.s4_pack)
    if plan.spectrum is not None and filters.spectrum is not None:
        # The power spectrum, the last floats of the buffer (maths.md §1.9, M-FLAT; task T4.1),
        # with the estimators of spectra._estimators, on the field after the standardisation.
        floats = spectrum_floats_2d(
            backend, field, plan.spectrum, filters.spectrum, spectrum_pairs(plan._layout())
        )
        buffer = xp.concat([buffer, floats], axis=1)
    return ScatteringValues(
        buffer=buffer,
        mean_pre_std=mean_pre_std,
        std_pre_std=std_pre_std,
    )


def _collect(
    backend: Backend,
    plan: STPlanSingleDomain,
    gather: GatherSingleDomain,
    families: dict[str, list[Any]],
    pieces: list[tuple[str, int]],
    pending: list[_Sum],
    total: Any,
) -> dict[str, list[Any]]:
    """``families`` with the means of one level added: its sums ``pending`` divided once
    (``_loop._divide_level``, one global sum), then read piece by piece (``pieces``: mean and
    var, the S1 and S2 of a scale, the S3 and S4 of a scale j3), each into its family."""
    divided = _divide_level(backend, pending, total)
    offset = 0
    for kind, n in pieces:
        part, items = divided[offset : offset + n], pending[offset : offset + n]
        offset += n
        if kind == "moments":
            families = {**families, "moments": [*families["moments"], *part]}
        elif kind == "first":
            floats = first_layer_floats(backend, part, plan.pairs, gather.first_pairs, (plan.L,))
            families = {
                name: [*v, floats[name]] if name in floats else v for name, v in families.items()
            }
        else:
            for item, value in zip(items, part, strict=True):
                families = {**families, item.family: [*families[item.family], value]}
    return families


def _second_layer(
    backend: Backend,
    wavelets: WaveletOperatorSingleDomain,
    plan: STPlanSingleDomain,
    filters: STFiltersSingleDomain,
    j3: int,
    moduli: dict[int, Any],
    w: Any,
) -> list[_Sum]:
    """The sums of the S3 and S4 means of scale ``j3``, at the level of ``j3``, from two Gram
    products (M-GRAM), the chunks of the segments of j3 in the order of the buffer
    (``_loop.gram_chunk_sums``).

    ``moduli`` holds U_{j2} at the level of j3, ``(Nb, L, Nc, *spatial)``, for every j2 ≤ j3, and
    ``w`` is the band map W_{j3} x, ``(Nb, L, Nc, *spatial)``: every map has the orientation
    of its last convolution right after the batch axis.
    """
    xp = backend.xp
    maps = plan.second_layer[j3]
    n_channels, L = plan.n_channels, plan.L  # noqa: N806
    # The moduli of the scales j2 ≤ j3, stacked (real, 1/(2L) of the second layer), then one
    # convolution: ψ_{l3} ⋆ U_{j2} x, (Nb, L3, j2, L2, Nc, *spatial), l3 leading.
    arms = xp.stack([moduli[j2] for j2 in maps.arms], axis=1)
    v = wavelets.apply(arms, j3)
    band = w  # (Nb, L3, Nc, *spatial)
    windows = filters.second_layer[j3]
    with_s3 = any((j2, j3) in plan.s3 for j2 in maps.arms)
    with_s4 = any(key[2] == j3 for key in plan.s4)
    n_batch, n_arms = int(v.shape[0]), len(maps.arms)
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
    rows = GramRows(
        gather_single_domain(plan).s3_rows[j3],
        filters.s3_denominators[j3],
        filters.s4_rows[j3],
        filters.s4_denominators[j3],
    )
    return gram_chunk_sums(backend, cross, gram, rows, (L,), n_arms, n_channels, True)
