"""The scattering loop: STL's loop structure, in functional style (Part I, §3.1.1; Part II,
§12.6; maths.md §1.6 to 1.7; task T2.4).

:func:`scattering` is the pure function behind ``STOperator.apply``: it takes the data, the
static plan, the static channel pairs and the backend arrays of the operator, and returns the
statistics as STL's full-grid arrays (:class:`ScatteringValues`). It becomes the public
``estel.functional.scattering`` in T2.5, with ``STStatistics``.

It follows STL's loop (``ST_Operator.apply`` at `master` ``8eaf769``, ``_compute_ST_loop`` at
`2D+1D` ``8f51143``): for each scale j3, the first layer W_{j3} x and U_{j3} = |W_{j3} x| at
level j3, the S1 and S2 means; for each j2 ≤ j3, the second layer ψ_{l3} ⋆ U_{j2} at level j3
and the S3 means; for each j1 ≤ j2, the S4 means; then the field and every U_{j2} go down one
level. Nothing is written in place (Part I, §3.1, rule 1): every step returns a new array, the
means are collected in lists, and each family is assembled once, at the end, with
``scatter_to_grid`` (§3.1.1). The loops run over scales, levels and channel pairs, as STL's,
never over pixels or orientations.

Missing values are weights (Part I, §3.1.1; maths.md OP-27): the field is set to 0 at its
missing pixels before anything else, so that every intermediate array is finite, and every mean
reads only the pixels its weights keep (M-MEAN, M-MASK, M-BORDER). The means the plan lists as
empty (OP-3) are not computed: they are absent from the index of ``scatter_to_grid``, and NaN in
the full grid, like the structural NaN (M-CHAN).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import DataClass, DownsampleWeights
from ..data._moments import cov_valid, divide, mean_valid, modulus, standardize
from ._plan import ScatteringPlan

S1_EPSILON = 1e-8
"""ε of the cross S1 terms, ``z / (|z| + ε)^(1/2)`` (maths.md M-S1, OP-16)."""

ChannelPair = tuple[int, int]


@dataclass(frozen=True)
class ChannelPairs:
    """The channel pairs of each family, from STL's ``compute_cross_matrix`` (maths.md M-CHAN).

    Built by :func:`channel_pairs`; static and hashable.
    """

    n_channels: int
    """Nc."""
    auto: tuple[int, ...]
    """The channels with auto statistics: the True diagonal entries."""
    cross: tuple[ChannelPair, ...]
    """The pairs ``(c1, c2)``, c1 < c2, of the cross S1 and S2: stored at ``[c1, c2]`` only,
    ``[c2, c1]`` stays NaN (OP-14)."""
    ordered: tuple[ChannelPair, ...]
    """The pairs of S3 and S4, in STL's order: ``(c, c)`` for each True diagonal entry, and
    ``(c1, c2)`` then ``(c2, c1)`` for each True entry above the diagonal (OP-14)."""


def channel_pairs(matrix: NDArray[np.bool_]) -> ChannelPairs:
    """The :class:`ChannelPairs` of a boolean ``(Nc, Nc)`` matrix; its entries below the
    diagonal are ignored, as in STL."""
    n = int(matrix.shape[0])
    auto = tuple(c for c in range(n) if matrix[c, c])
    cross = tuple((c1, c2) for c1 in range(n) for c2 in range(c1 + 1, n) if matrix[c1, c2])
    ordered = tuple(
        pair
        for c1 in range(n)
        for c2 in range(c1, n)
        if matrix[c1, c2]
        for pair in (((c1, c2),) if c1 == c2 else ((c1, c2), (c2, c1)))
    )
    return ChannelPairs(n_channels=n, auto=auto, cross=cross, ordered=ordered)


# ---------------------------------------------------------------------- static coefficient keys


def first_layer_keys(plan: ScatteringPlan, pairs: ChannelPairs) -> list[tuple[int, int, int]]:
    """The ``(c1, c2, j)`` of S1 (and of S2) the loop computes, in its order: for each scale
    with valid pixels, the auto channels, then the cross pairs."""
    return [
        (c1, c2, j)
        for j in plan.first_layer
        for c1, c2 in (*((c, c) for c in pairs.auto), *pairs.cross)
    ]


def s3_keys(plan: ScatteringPlan, pairs: ChannelPairs) -> list[tuple[int, int, int, int]]:
    """The ``(c1, c2, j2, j3)`` of S3 the loop computes, in its order."""
    return [(c1, c2, j2, j3) for j2, j3 in plan.s3 for c1, c2 in pairs.ordered]


def s4_keys(plan: ScatteringPlan, pairs: ChannelPairs) -> list[tuple[int, int, int, int, int]]:
    """The ``(c1, c2, j1, j2, j3)`` of S4 the loop computes, in its order."""
    return [(c1, c2, j1, j2, j3) for j1, j2, j3 in plan.s4 for c1, c2 in pairs.ordered]


def coefficient_index(
    keys: Sequence[tuple[int, ...]], n_axes: int
) -> tuple[NDArray[np.int64], ...]:
    """The index of ``scatter_to_grid`` for ``keys``: one coordinate array per grid axis."""
    table = np.asarray(keys, dtype=np.int64).reshape(len(keys), n_axes)
    return tuple(np.ascontiguousarray(table[:, axis]) for axis in range(n_axes))


# ---------------------------------------------------------------------- arrays and result


class ScatteringArrays(NamedTuple):
    """The backend arrays of an operator, cast once to its dtype, passed to :func:`scattering`
    as an argument rather than baked in (Part I, §3.1, rule 4). Built by the operator from the
    plan."""

    kernels: Any
    """The complex wavelet bank, ``(L, *kernel)`` (M-WAV-2D, M-WAV-1D)."""
    smoothing: Any
    """The real smoothing kernel (M-SMOOTH-2D, M-SMOOTH-1D)."""
    valid: Any
    """The valid pixels of the full-resolution field (bool), or ``None`` without missing
    values."""
    moments: Any
    """The weights of ``mean`` and ``var`` (``plan.moments``)."""
    first_layer: Mapping[int, Any]
    """The weights of S1 and S2, per scale of ``plan.first_layer``."""
    s3: Mapping[tuple[int, int], Any]
    """The weights of S3, per pair of ``plan.s3``."""
    s4: Mapping[tuple[int, int, int], Any]
    """The weights of S4, per triple of ``plan.s4``."""
    field_down: Mapping[int, tuple[Any, Any]]
    """Per level d < J-1, the ``(valid, reweight)`` maps of the field from d to d+1, or no
    entry without missing values (``plan.downsample_maps(d)``)."""
    modulus_down: Mapping[tuple[int, int], tuple[Any, Any]]
    """Per ``(d, j)``, j ≤ d < J-1, the maps of U_j from d to d+1, or no entry without missing
    values (``plan.downsample_maps(d, j)``)."""
    s1_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j)`` of the S1 and S2 values (:func:`first_layer_keys`)."""
    s3_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j2, j3)`` of the S3 values (:func:`s3_keys`)."""
    s4_index: tuple[Any, ...]
    """The grid positions ``(c1, c2, j1, j2, j3)`` of the S4 values (:func:`s4_keys`)."""


class ScatteringValues(NamedTuple):
    """The statistics in STL's full-grid shapes and dtypes (maths.md M-MOM, M-S1 to M-S4,
    M-CHAN), NaN where no coefficient exists (structural, or an empty mean, OP-3).

    In T2.5 they are stored in ``STStatistics``, behind accessors that do not depend on this
    storage (Part II, §12.7).
    """

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


# ---------------------------------------------------------------------- the loop


def scattering(
    data: DataClass,
    plan: ScatteringPlan,
    pairs: ChannelPairs,
    arrays: ScatteringArrays,
    *,
    standardize_data: bool = False,
) -> ScatteringValues:
    """The statistics of ``data`` (maths.md M-MOM, M-W1, M-S1 to M-S4), STL's ``ST_Operator``
    with ``norm="vanilla"`` and no spectrum.

    ``data`` has the static metadata ``plan`` was built from (geometry, spatial shape, ``pbc``,
    mask) and ``pairs.n_channels`` channels; ``arrays`` are the operator's arrays, on the data's
    backend and dtype. With ``standardize_data``, each map is first standardised (STL's
    ``apply(standardize=True)``, M-MOM), and the mean and standard deviation removed are
    returned with the statistics. Pure: nothing is modified, nothing depends on array values.
    """
    backend = _registry.backend_named(data.backend)
    xp = backend.xp
    axes = tuple(range(-data.ndim, 0))
    nb = data.n_batch
    L = plan.L  # noqa: N806

    def mean(x: Any, weights: Any, denominator: float) -> Any:
        return mean_valid(x, weights, denominator, axes)

    def cov(x: Any, y: Any, weights: Any, denominator: float) -> Any:
        return cov_valid(x, y, weights, denominator, axes)

    def conv(x: Any, level: int) -> Any:
        """ψ ⋆ x for ``x`` of shape ``(Nb, Nc, *O, *spatial)``: ``(Nb, Nc, *O, L, *spatial)``."""
        lead, spatial = tuple(x.shape[: -data.ndim]), tuple(x.shape[-data.ndim :])
        flat = xp.reshape(x, (nb, -1, *spatial))
        out = data._conv(flat, arrays.kernels, level)
        return xp.reshape(out, (*lead, L, *spatial))

    def down(x: Any, maps: tuple[Any, Any] | None, level: int) -> Any:
        """One downsampling step of ``x``, ``(Nb, Nc, *O, *spatial)`` (M-DOWN, M-DOWN-MASK)."""
        lead, spatial = tuple(x.shape[: -data.ndim]), tuple(x.shape[-data.ndim :])
        flat = xp.reshape(x, (nb, -1, *spatial))
        valid, reweight = (None, None) if maps is None else maps
        out = data._downsample(flat, DownsampleWeights(arrays.smoothing, valid, reweight), level)
        return xp.reshape(out, (*lead, *tuple(out.shape[2:])))

    # The field, finite everywhere: missing pixels are 0, never read (OP-27).
    x = data.array
    if arrays.valid is not None:
        x = xp.where(arrays.valid, x, xp.zeros_like(x))
    moments = plan.moments
    mean_pre_std = std_pre_std = None
    if standardize_data:
        x, mean_pre_std, std_pre_std = standardize(x, arrays.moments, moments.denominator, axes)
    field_mean = mean(x, arrays.moments, moments.denominator)
    field_var = cov(x, x, arrays.moments, moments.denominator)

    s1: list[Any] = []
    s2: list[Any] = []
    s3: list[Any] = []
    s4: list[Any] = []
    moduli: dict[int, Any] = {}
    for j3 in plan.levels:
        w = conv(x, j3)  # W_{j3} x: (Nb, Nc, L, *spatial)
        u = modulus(w)  # U_{j3} x
        moduli = {**moduli, j3: u}

        if j3 in plan.first_layer:
            weights, denominator = arrays.first_layer[j3], plan.first_layer[j3].denominator
            # Auto terms, every channel at once; cross terms, pair by pair (M-S1, M-S2).
            auto_s1 = mean(u, weights, denominator)  # (Nb, Nc, L)
            auto_s2 = mean(u * u, weights, denominator)
            s1.extend(_as_complex(backend, auto_s1[:, c]) for c in pairs.auto)
            if pairs.cross:
                rooted = divide(w, u, S1_EPSILON, 0.5)  # z / (|z| + eps)^(1/2), OP-16
                s1.extend(
                    cov(rooted[:, c1], rooted[:, c2], weights, denominator)
                    for c1, c2 in pairs.cross
                )
            s2.extend(_as_complex(backend, auto_s2[:, c]) for c in pairs.auto)
            s2.extend(cov(w[:, c1], w[:, c2], weights, denominator) for c1, c2 in pairs.cross)

        second: dict[int, Any] = {}
        for j2 in range(j3 + 1):
            v = conv(moduli[j2], j3)  # ψ_{l3} ⋆ U_{j2} x at level j3: (Nb, Nc, L2, L3, *spatial)
            second = {**second, j2: v}
            if (j2, j3) in plan.s3:
                weights, denominator = arrays.s3[(j2, j3)], plan.s3[(j2, j3)].denominator
                s3.extend(
                    cov(v[:, c1], w[:, c2][:, None], weights, denominator)
                    for c1, c2 in pairs.ordered
                )
            for j1 in range(j2 + 1):
                if (j1, j2, j3) in plan.s4:
                    weights = arrays.s4[(j1, j2, j3)]
                    denominator = plan.s4[(j1, j2, j3)].denominator
                    s4.extend(
                        cov(
                            second[j1][:, c1][:, :, None],
                            v[:, c2][:, None],
                            weights,
                            denominator,
                        )
                        for c1, c2 in pairs.ordered
                    )

        if j3 < plan.J - 1:
            x = down(x, arrays.field_down.get(j3), j3)
            moduli = {
                j2: down(u2, arrays.modulus_down.get((j3, j2)), j3) for j2, u2 in moduli.items()
            }

    nc, J = pairs.n_channels, plan.J  # noqa: N806
    return ScatteringValues(
        mean=field_mean,
        var=field_var,
        S1=_assemble(backend, s1, arrays.s1_index, (nc, nc, J), n_orientation_axes=1),
        S2=_assemble(backend, s2, arrays.s1_index, (nc, nc, J), n_orientation_axes=1),
        S3=_assemble(backend, s3, arrays.s3_index, (nc, nc, J, J), n_orientation_axes=2),
        S4=_assemble(backend, s4, arrays.s4_index, (nc, nc, J, J, J), n_orientation_axes=3),
        mean_pre_std=mean_pre_std,
        std_pre_std=std_pre_std,
    )


def _as_complex(backend: Backend, x: Any) -> Any:
    """A real array as complex, STL's storage of real coefficients (M-CHAN)."""
    xp = backend.xp
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
