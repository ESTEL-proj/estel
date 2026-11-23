"""The normalisation of the statistics (maths.md M-NORM, OP-18, OP-30; Part II, §12.6
``_normalize``; Part III, §19.6; tasks T2.5, T2.15).

``norm="self"`` normalises the statistics by themselves: the reference is computed from
them (:func:`self_reference`), is part of the computation, and gradients flow through it.
``norm="ref"`` normalises them by the reference of other statistics, a constant: gradients
do not flow into it (LF21, closed). With ``ref_batch_mean``, the ``norm="self"`` reference is
first averaged over the batch.

The reference (:class:`_NormReference`, internal) holds STL's ``var_ref`` and the normalisation
scales sigma_c(j, l) = √S2[c, c, j, l], where they exist: the channels with auto statistics, at
the scales with valid pixels. It stores the scales compactly, and builds STL's
``S2_ref_sqrt_chan_diag`` on demand (:attr:`_NormReference.s2_sqrt`). The divisors of M-NORM are
products of sigma; where a sigma does not exist the coefficient is NaN in STL (OP-30, OP-3), and
is not stored. On the compact buffer, for a fixed reference, the normalisation is a static linear
map (Part III, §19.6) that acts chunk by chunk (task T3.7): a static gather of the chunks that
keep a divisor, divided by their divisors, each the product of two rows of sigma over the
orientations of the chunk, with static tables of one row per chunk built in NumPy.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import Backend
from ._arrays import backend_of, constant
from ._layout import (
    Chunked,
    Copy,
    FamilyChunks,
    FloatMap,
    Layout,
    MapTables,
    apply,
    chunk_index,
    make_map,
    map_tables,
)

if TYPE_CHECKING:
    from ..scattering._plan import STPlan


@dataclass(frozen=True, eq=False)
class _NormReference:
    """The normalisation reference of M-NORM: STL's ``var_ref`` and the scales of
    ``S2_ref_sqrt_chan_diag``.

    Internal (issue #17): the operator stores it with ``norm="self"``, and ``norm="ref"`` reads it
    from the operator or from the statistics passed to ``apply``. From step 2 it also holds the
    first moments of the centred covariances (Part III, §20.11).
    """

    var: Any
    """``var_ref``, ``(Nr, Nc)``, real; Nr = 1 with ``ref_batch_mean``, else the batch size."""
    sigma: Any
    """``(Nr, n)``, real: sigma_c(j, l) for every ``(c, j)`` of :attr:`present`, in order, and
    every orientation l inside."""
    present: NDArray[np.bool_]
    """``(Nc, J)``, static: where sigma exists (``_grid.auto_scales``)."""
    n_orientations: int
    """L."""
    ref_batch_mean: bool
    """Whether the reference was averaged over the batch."""

    @property
    def n_batch(self) -> int:
        return int(self.var.shape[0])

    @property
    def s2_sqrt(self) -> Any:
        """STL's ``S2_ref_sqrt_chan_diag``: ``(Nr, Nc, J, L)``, complex, NaN where sigma does
        not exist. A view, built on demand; the normalisation reads :attr:`sigma`."""
        backend = backend_of(self.sigma)
        xp = backend.xp
        nc, n_scales = (int(n) for n in self.present.shape)
        shape = (nc, n_scales, self.n_orientations)
        positions = np.argwhere(np.broadcast_to(self.present[:, :, None], shape))
        if positions.shape[0] == 0:
            nan = xp.full((int(self.sigma.shape[0]), *shape), float("nan"), dtype=self.sigma.dtype)
            return backend.real_pairs_to_complex(xp.stack([nan, nan], axis=-1))
        index = [constant(backend, column, self.sigma, "int32") for column in positions.T]
        parts = [
            backend.scatter_to_grid(x, index, shape)
            for x in (self.sigma, xp.zeros_like(self.sigma))
        ]
        return backend.real_pairs_to_complex(xp.stack(parts, axis=-1))

    def static(self) -> tuple[Any, ...]:
        """The static part, hashable: for the pytree of ``STStatistics`` and for ``save``."""
        return (
            self.present.shape,
            self.present.tobytes(),
            self.ref_batch_mean,
            self.n_orientations,
        )

    @classmethod
    def from_static(cls, static: tuple[Any, ...], var: Any, sigma: Any) -> _NormReference:
        shape, data, ref_batch_mean, n_orientations = static
        present = np.frombuffer(data, dtype=bool).reshape(shape).copy()
        present.flags.writeable = False
        return cls(
            var=var,
            sigma=sigma,
            present=present,
            n_orientations=n_orientations,
            ref_batch_mean=ref_batch_mean,
        )


def _chunks_at(layout: Layout, name: str, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """The chunks of family ``name`` of ``layout`` at the positions ``rows`` ``(n, n_chunk_axes)``
    of its chunk axes, -1 where none."""
    grid = layout.grid(name)
    flat = np.ravel_multi_index(tuple(rows.T), grid.chunk_shape) if rows.shape[0] else rows[:, 0]
    return chunk_index(grid).find(np.asarray(flat, dtype=np.int64))


@functools.lru_cache(maxsize=64)
def _reference_tables(
    layout: Layout, shape: tuple[int, ...], data: bytes, n_orientations: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The floats of var and of the auto S2 at every sigma of the ``present`` given by
    ``shape`` and ``data``, in a buffer of ``layout``, a layout of the loop
    (``_layout.base_layout``): one real chunk of S2, of L floats, per sigma row."""
    present = np.frombuffer(data, dtype=bool).reshape(shape)
    channels = np.arange(int(shape[0]), dtype=np.int64)[:, None]
    var = _chunks_at(layout, "var", channels)
    c, j = np.nonzero(present)
    s2 = _chunks_at(layout, "S2", np.stack([c, c, j], axis=1).astype(np.int64))
    grid = layout.grid("S2")
    if (var < 0).any() or (s2 < 0).any() or not grid.real[grid.kinds[s2]].all():
        raise ValueError("the reference reads a real auto S2 at every sigma it holds.")
    starts = layout.chunk_starts("S2")[s2]
    s2_floats = (starts[:, None] + np.arange(n_orientations)).reshape(-1)
    return layout.chunk_starts("var")[var], s2_floats


def self_reference(
    backend: Backend,
    buffer: Any,
    layout: Layout,
    present: NDArray[np.bool_],
    n_orientations: int,
    ref_batch_mean: bool,
) -> _NormReference:
    """The reference of ``norm="self"``, from the statistics being normalised (M-NORM; STL
    ``to_norm(norm_type="self")``, ``_get_sqrt_chan_diag``): ``buffer`` in ``layout``, the
    layout of the loop. ``present`` is where sigma exists."""
    xp = backend.xp
    var_floats, s2_floats = _reference_tables(
        layout, present.shape, present.tobytes(), n_orientations
    )
    var = xp.take(buffer, constant(backend, var_floats, buffer, "int32"), axis=1)
    diagonal = xp.take(buffer, constant(backend, s2_floats, buffer, "int32"), axis=1)
    if ref_batch_mean:
        var = xp.mean(var, axis=0, keepdims=True)
        diagonal = xp.mean(diagonal, axis=0, keepdims=True)
    return _NormReference(
        var=var,
        sigma=xp.sqrt(diagonal),
        present=present,
        n_orientations=n_orientations,
        ref_batch_mean=ref_batch_mean,
    )


def as_constant(backend: Backend, reference: _NormReference) -> _NormReference:
    """The reference of ``norm="ref"``: a constant, through which no gradient flows (LF21)."""
    return _NormReference(
        var=backend.stop_gradient(reference.var),
        sigma=backend.stop_gradient(reference.sigma),
        present=reference.present,
        n_orientations=reference.n_orientations,
        ref_batch_mean=reference.ref_batch_mean,
    )


_SIGMA_AXES = {
    "S1": ((0, 2), (1, 2)),
    "S2": ((0, 2), (1, 2)),
    "S3": ((0, 2), (1, 3)),
    "S4": ((0, 2), (1, 3)),
}
"""For each normalised family, the (channel, scale) chunk axes of the two sigma of its divisor
(M-NORM): sigma_{c1}(j, l) sigma_{c2}(j, l) for S1 and S2, sigma_{c1}(j2, l2) sigma_{c2}(j3, l3)
for S3, sigma_{c1}(j1, l1) sigma_{c2}(j2, l2) for S4."""

_SIGMA_ORIENTATIONS = {"S1": (0, 0), "S2": (0, 0), "S3": (0, 1), "S4": (0, 1)}
"""The orientation axis of each of the two sigma: l, l; l2, l3; l1, l2."""


class _NormTables(NamedTuple):
    """The static part of :func:`normalize`, for one layout and one ``present`` of the
    reference: tables of one row per chunk."""

    transform: FloatMap
    """The gather of the chunks that keep a divisor."""
    rows: dict[str, tuple[NDArray[np.int64], NDArray[np.int64]]]
    """For each normalised family, the row of sigma of the first and of the second factor of
    the divisor of each chunk of the result, in the order of its buffer; for var, the channel
    of each chunk, twice (its divisor is ``var_ref``)."""


@functools.lru_cache(maxsize=64)
def _norm_tables(
    layout: Layout, shape: tuple[int, ...], data: bytes, n_orientations: int
) -> _NormTables:
    """:class:`_NormTables` of ``layout``, a layout of the loop, whose chunk positions are the
    channels and scales themselves, and of the ``present`` given by ``shape`` and ``data``."""
    present = np.frombuffer(data, dtype=bool).reshape(shape)
    rank = np.where(present, np.cumsum(present.ravel()).reshape(present.shape) - 1, -1)
    specs: list[Copy | Chunked] = []
    factors: dict[str, tuple[NDArray[np.int64], NDArray[np.int64]]] = {}
    for grid in layout.grids:
        levels, level_ids = layout.levels(grid.name)
        chunks = grid.chunks.astype(np.int64)
        if grid.name == "var":
            every = np.arange(grid.n_chunks, dtype=np.int64)
            specs.append(Chunked(layout.chunked("var"), every[:, None], None))
            factors = {**factors, "var": (chunks[:, 0], chunks[:, 0])}
            continue
        if grid.name not in _SIGMA_AXES:
            specs.append(Copy(grid.name))
            continue
        first, second = _SIGMA_AXES[grid.name]
        a = rank[chunks[:, first[0]], chunks[:, first[1]]]
        b = rank[chunks[:, second[0]], chunks[:, second[1]]]
        kept = np.flatnonzero((a >= 0) & (b >= 0))
        family = FamilyChunks(
            grid.replace(chunks=chunks[kept], kinds=grid.kinds[kept]), levels, level_ids[kept]
        )
        specs.append(Chunked(family, kept[:, None], None))
        factors = {**factors, grid.name: (a[kept], b[kept])}
    transform = make_map(layout, specs)
    for name, (a, _) in factors.items():
        n = transform.layout.grid(name).n_chunks
        if n != a.shape[0]:
            raise ValueError(f"{name}: the normalisation keeps {n} chunks, not {a.shape[0]}.")
    return _NormTables(transform=transform, rows=factors)


def normalize(
    backend: Backend,
    buffer: Any,
    layout: Layout,
    reference: _NormReference,
) -> tuple[Any, Layout]:
    """The statistics divided as in the table of M-NORM (STL ``to_norm``,
    ``_normalize_scatcov``), and their new layout: the coefficients whose divisor does not exist
    are not stored (NaN in STL's full grid).

    ``mean`` is not normalised; ``var`` is divided by ``var_ref``; S1 by
    √(sigma_{c1} sigma_{c2}) and S2 by sigma_{c1} sigma_{c2} at (j, l); S3 by
    sigma_{c1}(j2, l2) sigma_{c2}(j3, l3); S4 by sigma_{c1}(j1, l1) sigma_{c2}(j2, l2). The
    reference broadcasts over the batch (Nr = 1) or matches it. ``layout`` is a layout of the
    loop (``_layout.base_layout``).
    """
    present = reference.present
    key = (layout, present.shape, present.tobytes(), reference.n_orientations)
    divide = _compiled(backend.name, *key)
    transform = _norm_tables(*key).transform
    tables = map_tables(backend, transform, buffer)
    return divide(buffer, reference.var, reference.sigma, tables), transform.layout


@functools.lru_cache(maxsize=64)
def _compiled(
    backend_name: str, layout: Layout, shape: tuple[int, ...], data: bytes, n_orientations: int
) -> Any:
    """The division of :func:`normalize` for one layout and one ``present``, a function of the
    buffer, ``var_ref``, sigma and the tables of the chunks (``_layout.map_tables``), compiled
    once where the backend compiles (``Backend.jit``; the identity on PyTorch)."""
    backend = _registry.backend_named(backend_name)
    tables = _norm_tables(layout, shape, data, n_orientations)
    return backend.jit(functools.partial(_divide, backend, tables, layout, n_orientations))


def _divide(
    backend: Backend,
    tables: _NormTables,
    layout: Layout,
    n_orientations: int,
    buffer: Any,
    var: Any,
    sigma: Any,
    chunks: MapTables,
) -> Any:
    """The buffer of the kept chunks divided by their divisors (:func:`normalize`)."""
    xp = backend.xp
    L = n_orientations  # noqa: N806
    n_ref = int(var.shape[0])
    sigma = xp.reshape(sigma, (n_ref, -1, L))

    def divided(name: str) -> Any:
        a, b = tables.rows[name]
        n = int(a.shape[0])
        if name == "var":
            divisor = xp.take(var, constant(backend, a, var, "int32"), axis=1)
            return lambda dense: dense / xp.reshape(divisor, (n_ref, n, 1, 1))
        first = xp.take(sigma, constant(backend, a, var, "int32"), axis=1)  # (Nr, n, L)
        second = xp.take(sigma, constant(backend, b, var, "int32"), axis=1)
        shape = layout.grid(name).orientations
        axes = _SIGMA_ORIENTATIONS[name]

        def along(factor: Any, axis: int) -> Any:
            return xp.reshape(
                factor, (n_ref, n, *(L if i == axis else 1 for i in range(len(shape))))
            )

        product = along(first, axes[0]) * along(second, axes[1])
        if name == "S1":
            product = xp.sqrt(product)
        divisor = xp.reshape(xp.broadcast_to(product, (n_ref, n, *shape)), (n_ref, n, -1, 1))
        return lambda dense: dense / divisor

    scale = {name: divided(name) for name in tables.rows}
    return apply(backend, buffer, tables.transform, scale, chunks)


def check_reference(reference: _NormReference, plan: STPlan, n_batch: int, dtype: Any) -> None:
    """Raise unless ``reference`` can normalise the statistics of ``plan``, before they are
    computed (maths.md M-NORM).

    The reference must have the J, L and number of channels of the plan, and a normalisation
    scale sigma for every channel the plan normalises: each channel with auto statistics in the
    plan. A channel the plan meets only in cross pairs has no scale under ``norm="self"`` either,
    and its coefficients stay NaN when the reference has none (OP-30). The size, the mask and
    the periodicity of the data the reference comes from are free: the syntheses normalise by
    the reference of a target of another size. A scale the reference lacks because its mask
    left it empty gives NaN, as in STL (OP-3). The batch is 1 or that of the data, as STL
    broadcasts; the dtype is that of the operator.
    """
    found_nc, found_j = (int(n) for n in reference.present.shape)
    found_l = reference.n_orientations
    mismatches = [
        f"{name} = {found} (the operator has {expected})"
        for name, found, expected in (
            ("J", found_j, plan.J),
            ("L", found_l, plan.L),
            ("the number of channels", found_nc, plan.n_channels),
        )
        if found != expected
    ]
    if mismatches:
        raise ValueError(
            f"the reference has {', '.join(mismatches)}: normalise with statistics computed by an "
            "operator with the same J, L and channels; their size, mask and pbc may differ "
            "(maths.md M-NORM)."
        )
    with_scale: NDArray[np.bool_] = np.any(reference.present, axis=1)
    missing = [c for c in plan.pairs.auto if not with_scale[c]]
    if missing:
        raise ValueError(
            f"the reference has no normalisation scale for channel(s) {missing}, which this "
            "operator normalises (they have auto statistics in cross_matrix): compute "
            "the reference with their diagonal entries set, so that it holds their auto "
            "statistics (maths.md M-NORM, M-CHAN, OP-30)."
        )
    if reference.n_batch not in (1, n_batch):
        raise ValueError(
            f"the reference was computed on {reference.n_batch} batch elements without "
            f"ref_batch_mean, and these data have {n_batch}: pass data with {reference.n_batch} "
            "batch element(s), or compute the reference with ref_batch_mean=True (maths.md "
            "M-NORM)."
        )
    if reference.var.dtype != dtype:
        raise TypeError(
            f"the reference is {reference.var.dtype}, but the operator computes in {dtype}: "
            "compute both with the same default dtype (estel.set_default_dtype)."
        )
