"""The scattering loop: STL's loop structure, in functional style (Part I, §3.1.1; Part II,
§12.6; maths.md §1.6 to 1.7; tasks T2.4, T2.5).

:func:`scattering_values` is the pure function at the core of ``estel.functional.scattering``:
it takes the data, the static plan and the filters, and returns the statistics without
normalisation, as STL's full-grid arrays (:class:`ScatteringValues`). The normalisation and the
``STStatistics`` object are added by ``estel.functional.scattering``.

It follows STL's loop (``ST_Operator.apply`` at `master` ``8eaf769``, ``_compute_ST_loop`` at
`2D+1D` ``8f51143``): for each scale j3, the first layer W_{j3} x and U_{j3} = |W_{j3} x| at
level j3, the S1 and S2 means; for each j2 ≤ j3, the second layer ψ_{l3} ⋆ U_{j2} at level j3
and the S3 means; for each j1 ≤ j2, the S4 means; then the field and every U_{j2} go down one
level. The convolutions and the downsamplings are those of the wavelet operator
(``WaveletOperator.apply``, ``downsample``; Part I, §2.3): the loop sees levels, maps and
weights, and does not depend on the geometry. Nothing is written in place (Part I, §3.1,
rule 1): every step returns a new array, the
means are collected in lists, and each family is assembled once, at the end, with
``scatter_to_grid`` (§3.1.1). The loops run over scales, levels and channel pairs, as STL's,
never over pixels or orientations.

**Sums, then division.** Every mean is a weighted sum over the pixels divided by a denominator
of the plan (M-MEAN). The loop takes the sums of a level, stacks them, passes them once to
:func:`_global_sum`, then divides. In 1.0.0 :func:`_global_sum` is the identity; in step 2 it
adds the partial sums of the partitions of the data across MPI ranks (Part III, §18.3), between
the sum and the division, so that the statistics are global.

Missing values are weights (Part I, §3.1.1; maths.md OP-27): the field is set to 0 at its
missing pixels before anything else, so that every intermediate array is finite, and every mean
reads only the pixels its weights keep (M-MEAN, M-MASK, M-BORDER). The means the plan lists as
empty (OP-3) are not computed: they are absent from the index of ``scatter_to_grid``, and NaN in
the full grid, like the structural NaN (M-CHAN).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import Data
from ..data._moments import cov_sum_valid, divide, modulus, standardize, sum_valid
from ..wavelets._operator import WaveletOperator
from ._filters import STFilters
from ._plan import STPlan

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
    """The statistics without normalisation in STL's full-grid shapes and dtypes (maths.md
    M-MOM, M-S1 to M-S4, M-CHAN), NaN where no coefficient exists (structural, or an empty
    mean, OP-3). ``estel.functional.scattering`` stores them in ``STStatistics``."""

    mean: Any
    """``(Nb, Nc)``, real."""
    var: Any
    """``(Nb, Nc)``, real: the non-centred second moment (OP-11)."""
    S1: Any
    """``(Nb, Nc, Nc, J, L)``, complex."""
    S2: Any
    """``(Nb, Nc, Nc, J, L)``, complex."""
    S3: Any
    """``(Nb, Nc, Nc, J, J, L, L)``, complex: ``[.., j2, j3, l2, l3]``."""
    S4: Any
    """``(Nb, Nc, Nc, J, J, J, L, L, L)``, complex: ``[.., j1, j2, j3, l1, l2, l3]``."""
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


# ---------------------------------------------------------------------- the loop


def scattering_values(data: Data, plan: STPlan, filters: STFilters) -> ScatteringValues:
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
    mean_pre_std = std_pre_std = None
    if plan.standardize:
        x, mean_pre_std, std_pre_std = standardize(
            x, filters.moments, moments.denominator, axes, global_sum=total
        )
    pending: list[_Sum] = [
        _Sum("mean", sum_valid(x, filters.moments, axes), moments.denominator),
        _Sum("var", cov_sum_valid(x, x, filters.moments, axes), moments.denominator),
    ]
    means: dict[str, list[Any]] = {"mean": [], "var": [], "S1": [], "S2": [], "S3": [], "S4": []}

    moduli: dict[int, Any] = {}
    for j3 in plan.levels:
        w = wavelets.apply(x, j3)  # W_{j3} x: (Nb, Nc, L, *spatial)
        u = modulus(w)  # U_{j3} x
        moduli = {**moduli, j3: u}

        if j3 in plan.first_layer:
            weights, denominator = filters.first_layer[j3], plan.first_layer[j3].denominator
            # Auto terms, every channel at once; cross terms, pair by pair (M-S1, M-S2).
            auto_s1 = sum_valid(u, weights, axes)  # (Nb, Nc, L)
            auto_s2 = sum_valid(u * u, weights, axes)
            pending.extend(_Sum("S1", auto_s1[:, c], denominator) for c in pairs.auto)
            if pairs.cross:
                rooted = divide(w, u, S1_EPSILON, 0.5)  # z / (|z| + eps)^(1/2), OP-16
                pending.extend(
                    _Sum(
                        "S1",
                        cov_sum_valid(rooted[:, c1], rooted[:, c2], weights, axes),
                        denominator,
                    )
                    for c1, c2 in pairs.cross
                )
            pending.extend(_Sum("S2", auto_s2[:, c], denominator) for c in pairs.auto)
            pending.extend(
                _Sum("S2", cov_sum_valid(w[:, c1], w[:, c2], weights, axes), denominator)
                for c1, c2 in pairs.cross
            )

        second: dict[int, Any] = {}
        for j2 in range(j3 + 1):
            # ψ_{l3} ⋆ U_{j2} x at level j3: (Nb, Nc, L2, L3, *spatial)
            v = wavelets.apply(moduli[j2], j3)
            second = {**second, j2: v}
            if (j2, j3) in plan.s3:
                weights, denominator = filters.s3[(j2, j3)], plan.s3[(j2, j3)].denominator
                pending.extend(
                    _Sum(
                        "S3", cov_sum_valid(v[:, c1], w[:, c2][:, None], weights, axes), denominator
                    )
                    for c1, c2 in pairs.ordered
                )
            for j1 in range(j2 + 1):
                if (j1, j2, j3) in plan.s4:
                    weights = filters.s4[(j1, j2, j3)]
                    denominator = plan.s4[(j1, j2, j3)].denominator
                    pending.extend(
                        _Sum(
                            "S4",
                            cov_sum_valid(
                                second[j1][:, c1][:, :, None], v[:, c2][:, None], weights, axes
                            ),
                            denominator,
                        )
                        for c1, c2 in pairs.ordered
                    )

        for family, value in _divide_level(backend, pending, total):
            means[family].append(value)
        pending = []

        if j3 < plan.J - 1:
            x = wavelets.downsample(x, j3)
            moduli = {j2: wavelets.downsample(u2, j3, j2) for j2, u2 in moduli.items()}

    nc, J = pairs.n_channels, plan.J  # noqa: N806
    s1 = [_as_complex(backend, v) for v in means["S1"]]
    s2 = [_as_complex(backend, v) for v in means["S2"]]
    return ScatteringValues(
        mean=means["mean"][0],
        var=means["var"][0],
        S1=_assemble(backend, s1, filters.s1_index, (nc, nc, J), n_orientation_axes=1),
        S2=_assemble(backend, s2, filters.s1_index, (nc, nc, J), n_orientation_axes=1),
        S3=_assemble(backend, means["S3"], filters.s3_index, (nc, nc, J, J), n_orientation_axes=2),
        S4=_assemble(
            backend, means["S4"], filters.s4_index, (nc, nc, J, J, J), n_orientation_axes=3
        ),
        mean_pre_std=mean_pre_std,
        std_pre_std=std_pre_std,
    )


def _divide_level(backend: Backend, pending: Sequence[_Sum], total: Any) -> list[tuple[str, Any]]:
    """The means of one level: its weighted sums, stacked as one real ``(Nb, n)`` array, passed
    once to ``total`` (:func:`_global_sum`), split back, then divided by their denominators,
    in their order. The stacking is exact: it only moves values."""
    if not pending:
        return []
    xp = backend.xp
    flat: list[Any] = []
    for item in pending:
        is_complex = bool(xp.isdtype(item.value.dtype, "complex floating"))
        real = backend.complex_to_real_pairs(item.value) if is_complex else item.value
        flat.append(xp.reshape(real, (real.shape[0], -1)))
    summed = total(xp.concat(flat, axis=1))
    out: list[tuple[str, Any]] = []
    offset = 0
    for item, part in zip(pending, flat, strict=True):
        size = int(part.shape[1])
        piece = summed[:, offset : offset + size]
        offset += size
        is_complex = bool(xp.isdtype(item.value.dtype, "complex floating"))
        shape = tuple(item.value.shape)
        if is_complex:
            value = backend.real_pairs_to_complex(xp.reshape(piece, (*shape, 2)))
        else:
            value = xp.reshape(piece, shape)
        out.append((item.family, value / item.denominator))
    return out


def _as_complex(backend: Backend, x: Any) -> Any:
    """A real array as complex, STL's storage of real coefficients (M-CHAN); complex arrays are
    returned unchanged."""
    xp = backend.xp
    if xp.isdtype(x.dtype, "complex floating"):
        return x
    return backend.real_pairs_to_complex(xp.stack([x, xp.zeros_like(x)], axis=-1))


def _assemble(
    backend: Backend,
    values: Sequence[Any],
    index: tuple[Any, ...],
    grid: tuple[int, ...],
    n_orientation_axes: int,
) -> Any:
    """The full grid of one family, assembled once (Part I, §3.1.1): the values, each
    ``(Nb, *orientations)``, at the static positions ``index`` of ``grid`` (channels and
    scales), NaN elsewhere; the axes in STL's order, ``(Nb, *grid, *orientations)``."""
    xp = backend.xp
    stacked = xp.stack(list(values), axis=-1)  # (Nb, *orientations, n)
    full = backend.scatter_to_grid(stacked, index, grid)  # (Nb, *orientations, *grid)
    o, g = n_orientation_axes, len(grid)
    order = (0, *range(1 + o, 1 + o + g), *range(1, 1 + o))
    return xp.permute_dims(full, order)
