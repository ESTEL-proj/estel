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

With the power spectrum (maths.md §1.9; task T4.4), the reference also holds STL's
``PS_ref_sqrt_chan_diag``, σᴾˢ_c(b) = √PS_ref[c, c, b], compactly, for the channels with an auto
spectrum (:attr:`_NormReference.sigma_ps`), and PS[c1, c2, b] is divided by σᴾˢ_{c1}(b) σᴾˢ_{c2}(b).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import Backend
from ._arrays import backend_object_of, constant
from ._grid import orientations_at_tensor_prod
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
    pad_nan,
)

if TYPE_CHECKING:
    from ..scattering._plan import STPlanSingleDomain, STPlanTensorProd


NORMS = ("self", "ref", "none")
"""The values of ``norm``, chosen at each call of ``apply`` (maths.md M-NORM; task T2.19)."""


class SpectrumParameters(NamedTuple):
    """The parameters of a power spectrum that statistics normalised by a reference must share
    with it (maths.md §1.9, M-NORM; task T4.4), named as ``estel.io`` writes them."""

    n_bins: int
    """n_b, the number of bins."""
    J: int
    """J, the J of the operator."""
    Jmin: int
    """J_min."""


class SpectrumReference(NamedTuple):
    """What the reference of statistics with the power spectrum holds of it, statically
    (maths.md M-NORM, §1.9; task T4.4): the channels with σᴾˢ, and the parameters of the
    spectrum it was computed with."""

    present: NDArray[np.bool_]
    """``(Nc,)``: the channels with an auto spectrum, where σᴾˢ exists (OP-30)."""
    parameters: SpectrumParameters
    """n_b, J and J_min."""


def spectrum_reference_of(spectrum: Any, pairs: Any) -> SpectrumReference:
    """The :class:`SpectrumReference` of the spectrum of a plan
    (``spectra._estimators.SpectrumEstimatorPlan2D``) whose channel pairs are ``pairs``
    (``scattering._plan.ChannelPairs``): σᴾˢ exists for the channels with an auto spectrum, the True
    diagonal entries of ``cross_matrix``."""
    present = np.isin(np.arange(pairs.n_channels), np.asarray(pairs.auto, dtype=np.int64))
    present.flags.writeable = False
    return SpectrumReference(
        present, SpectrumParameters(int(spectrum.n_bins), int(spectrum.J), int(spectrum.Jmin))
    )


class BlocksTensorProd(NamedTuple):
    """What a normalisation reference of a tensor product must share with the statistics it
    normalises (maths.md M-NORM-TP; task T3.6): the blocks, J, L and the low-pass flags; the
    sizes and the periodicity of the blocks are free, as in M-NORM."""

    geometries: tuple[str, ...]
    """The geometry of each block, by class name (``"Data2D"``, ``"Data1D"``)."""
    J: tuple[int, ...]
    L: tuple[int, ...]
    lowpass: tuple[bool, ...]
    """Whether each block has a low-pass filter (M-LP)."""

    def row_shape(self, scale: tuple[int, ...]) -> tuple[int, ...]:
        """The orientations of sigma at the multi-index ``scale``: L_a, or 1 along a block at
        its low-pass scale (M-LP)."""
        return orientations_at_tensor_prod(scale, self.J, self.L)


def blocks_tensor_prod(plan: STPlanTensorProd) -> BlocksTensorProd:
    """The :class:`BlocksTensorProd` of ``plan``."""
    return BlocksTensorProd(
        geometries=tuple(g.__name__ for g in plan.block_geometries),
        J=plan.J,
        L=plan.L,
        lowpass=plan.lowpass_blocks,
    )


@dataclass(frozen=True, eq=False)
class _NormReference:
    """The normalisation reference of M-NORM: STL's ``var_ref`` and the scales of
    ``S2_ref_sqrt_chan_diag``.

    Internal (issue #17): the operator stores it at each call of ``apply`` with ``norm="self"``,
    and ``norm="ref"`` reads it from the operator or from the statistics passed to ``apply``.
    From step 2 it also holds the first moments of the centred covariances (Part III, §20.11).
    """

    var: Any
    """``var_ref``, ``(Nr, Nc)``, real; Nr = 1 with ``ref_batch_mean``, else the batch size."""
    sigma: Any
    """``(Nr, n)``, real: sigma_c(j, l) for every ``(c, j)`` of :attr:`present`, in order, and
    every orientation l inside (in a tensor product, those of :meth:`BlocksTensorProd.row_shape`,
    in C order)."""
    present: NDArray[np.bool_]
    """``(Nc, J)``, static: where sigma exists (``_grid.auto_scales_single_domain``);
    ``(Nc, J'_0, …)`` in a tensor product (``_grid.auto_scales_tensor_prod``)."""
    n_orientations: int
    """L; in a tensor product, the product of the L_a."""
    ref_batch_mean: bool
    """Whether the reference was averaged over the batch."""
    blocks: BlocksTensorProd | None = None
    """The blocks of a tensor product, or ``None`` for planar statistics."""
    sigma_ps: Any = None
    """``(Nr, n)``, real: σᴾˢ_c(b) = √PS_ref[c, c, b] for every channel ``c`` of
    ``spectrum.present``, in order, and every bin b inside; ``None`` when the statistics hold no
    power spectrum."""
    spectrum: SpectrumReference | None = None
    """The channels and the parameters of :attr:`sigma_ps`, static; ``None`` with it."""

    def rows(self) -> tuple[NDArray[np.int64], tuple[tuple[int, ...], ...], NDArray[np.int64]]:
        """The ``(c, j)`` of each row of :attr:`sigma` ``(n_rows, 1 + n_scale_axes)``, the
        orientation shape of each row, and the first float of each row: one entry per row,
        never per coefficient."""
        return _sigma_rows(
            self.present.shape, self.present.tobytes(), self.blocks, self.n_orientations
        )

    @property
    def n_batch(self) -> int:
        return int(self.var.shape[0])

    @property
    def s2_sqrt(self) -> Any:
        """STL's ``S2_ref_sqrt_chan_diag``: ``(Nr, Nc, J, L)``, complex, NaN where sigma does
        not exist. A view, built on demand; the normalisation reads :attr:`sigma`."""
        backend = backend_object_of(self.sigma)
        xp = backend.xp
        if self.blocks is not None:
            return self._s2_sqrt_tensor_prod(backend)
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

    @property
    def ps_sqrt(self) -> Any:
        """STL's ``PS_ref_sqrt_chan_diag``: ``(Nr, Nc, n_b)``, complex, NaN where σᴾˢ does not
        exist (a channel without auto spectrum, OP-30); ``None`` without the spectrum. A view,
        built on demand; the normalisation reads :attr:`sigma_ps`."""
        if self.spectrum is None:
            return None
        backend = backend_object_of(self.sigma_ps)
        xp = backend.xp
        n_ref = int(self.sigma_ps.shape[0])
        n_bins = self.spectrum.parameters.n_bins
        channels = np.flatnonzero(self.spectrum.present)
        shape = (int(self.spectrum.present.shape[0]),)
        values = xp.reshape(self.sigma_ps, (n_ref, int(channels.shape[0]), n_bins))
        carried = xp.permute_dims(values, (0, 2, 1))  # (Nr, n_b, n)
        index = [constant(backend, channels, self.sigma_ps, "int32")]
        parts = [
            backend.scatter_to_grid(x, index, shape) for x in (carried, xp.zeros_like(carried))
        ]
        placed = xp.permute_dims(xp.stack(parts, axis=-1), (0, 2, 1, 3))  # (Nr, Nc, n_b, 2)
        return backend.real_pairs_to_complex(placed)

    def _s2_sqrt_tensor_prod(self, backend: Backend) -> Any:
        """:attr:`s2_sqrt` of a tensor product, ``(Nr, Nc, *J', *L)``: the rows of sigma, one
        orientation shape at a time, padded with NaN to the orientations of the blocks and
        placed in the grid of the channels and the scales."""
        xp = backend.xp
        if self.blocks is None:
            raise ValueError(
                "this reference comes from single-domain statistics, which have no blocks of a "
                "tensor product: use the reference of statistics computed by "
                "estel.STOperatorTensorProd on estel.DataTensorProd data."
            )
        full = self.blocks.L
        positions, shapes, starts = self.rows()
        n_ref = int(self.sigma.shape[0])
        grid_shape = tuple(int(n) for n in self.present.shape)
        if positions.shape[0] == 0:
            nan = xp.full((n_ref, *grid_shape, *full), float("nan"), dtype=self.sigma.dtype)
            return backend.real_pairs_to_complex(xp.stack([nan, nan], axis=-1))
        pieces, order = [], []
        for shape in dict.fromkeys(shapes):
            rows = np.array([i for i, s in enumerate(shapes) if s == shape], dtype=np.int64)
            size = int(np.prod(shape, dtype=np.int64))
            index = (starts[rows][:, None] + np.arange(size)).reshape(-1)
            values = xp.take(self.sigma, constant(backend, index, self.sigma, "int32"), axis=1)
            values = xp.reshape(values, (n_ref, int(rows.shape[0]), *shape))
            parts = xp.stack([values, xp.zeros_like(values)], axis=-1)
            padded = pad_nan(backend, parts, full)
            pieces.append(xp.reshape(padded, (n_ref, int(rows.shape[0]), -1)))
            order.append(rows)
        found = xp.permute_dims(xp.concat(pieces, axis=1), (0, 2, 1))
        located = positions[np.concatenate(order)]
        coordinates = [constant(backend, column, self.sigma, "int32") for column in located.T]
        placed = backend.scatter_to_grid(found, coordinates, grid_shape)  # (Nr, n_o * 2, *grid)
        moved = xp.permute_dims(
            xp.reshape(placed, (n_ref, *full, 2, *grid_shape)),
            (
                0,
                *range(2 + len(full), 2 + len(full) + len(grid_shape)),
                *range(1, 1 + len(full)),
                1 + len(full),
            ),
        )
        return backend.real_pairs_to_complex(moved)

    def static(self) -> tuple[Any, ...]:
        """The static part, hashable: for the pytree of ``STStatistics`` and for ``save``."""
        return (
            self.present.shape,
            self.present.tobytes(),
            self.ref_batch_mean,
            self.n_orientations,
            self.blocks,
            None
            if self.spectrum is None
            else (self.spectrum.present.tobytes(), self.spectrum.parameters),
        )

    @classmethod
    def from_static(
        cls, static: tuple[Any, ...], var: Any, sigma: Any, sigma_ps: Any = None
    ) -> _NormReference:
        shape, data, ref_batch_mean, n_orientations, blocks, spectrum = static
        present = np.frombuffer(data, dtype=bool).reshape(shape).copy()
        present.flags.writeable = False
        found = None
        if spectrum is not None:
            channels = np.frombuffer(spectrum[0], dtype=bool).copy()
            channels.flags.writeable = False
            found = SpectrumReference(channels, spectrum[1])
        return cls(
            var=var,
            sigma=sigma,
            present=present,
            n_orientations=n_orientations,
            ref_batch_mean=ref_batch_mean,
            blocks=blocks,
            sigma_ps=sigma_ps,
            spectrum=found,
        )


def _ranks(present: NDArray[np.bool_]) -> NDArray[np.int64]:
    """The rank of each True of ``present`` among them, -1 at each False."""
    found: NDArray[np.int64] = np.where(
        present, np.cumsum(present.ravel()).reshape(present.shape) - 1, -1
    ).astype(np.int64)
    return found


@functools.lru_cache(maxsize=64)
def _sigma_rows(
    shape: tuple[int, ...],
    data: bytes,
    blocks: BlocksTensorProd | None,
    n_orientations: int,
) -> tuple[NDArray[np.int64], tuple[tuple[int, ...], ...], NDArray[np.int64]]:
    """:meth:`_NormReference.rows` of the ``present`` given by ``shape`` and ``data``."""
    present = np.frombuffer(data, dtype=bool).reshape(shape)
    positions = np.argwhere(present).astype(np.int64)
    shapes: tuple[tuple[int, ...], ...]
    if blocks is None:
        shapes = tuple((n_orientations,) for _ in range(positions.shape[0]))
    else:
        shapes = tuple(blocks.row_shape(tuple(int(j) for j in row[1:])) for row in positions)
    sizes = np.array([int(np.prod(s, dtype=np.int64)) for s in shapes], dtype=np.int64)
    starts = (np.cumsum(sizes) - sizes).astype(np.int64)
    for array in (positions, starts):
        array.flags.writeable = False
    return positions, shapes, starts


def _chunks_at(layout: Layout, name: str, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """The chunks of family ``name`` of ``layout`` at the positions ``rows`` ``(n, n_chunk_axes)``
    of its chunk axes, -1 where none."""
    grid = layout.grid(name)
    flat = np.ravel_multi_index(tuple(rows.T), grid.chunk_shape) if rows.shape[0] else rows[:, 0]
    return chunk_index(grid).find(np.asarray(flat, dtype=np.int64))


@functools.lru_cache(maxsize=64)
def _reference_tables(
    layout: Layout,
    shape: tuple[int, ...],
    data: bytes,
    n_orientations: int,
    blocks: BlocksTensorProd | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The floats of var and of the auto S2 at every sigma of the ``present`` given by
    ``shape`` and ``data``, in a buffer of ``layout``, a layout of the loop
    (``_layout.layout_single_domain``, ``_layout.layout_tensor_prod``): one real chunk of S2 per
    sigma row, of L floats (of the orientations of its row in a tensor product)."""
    positions, shapes, _ = _sigma_rows(shape, data, blocks, n_orientations)
    channels = np.arange(int(shape[0]), dtype=np.int64)[:, None]
    var = _chunks_at(layout, "var", channels)
    c = positions[:, :1]
    s2 = _chunks_at(layout, "S2", np.concatenate([c, c, positions[:, 1:]], axis=1))
    grid = layout.grid("S2")
    real = grid.real | ~grid.exists
    if (var < 0).any() or (s2 < 0).any() or not real[grid.kinds[s2]].all():
        raise ValueError("the reference reads a real auto S2 at every sigma it holds.")
    starts = layout.chunk_starts("S2")[s2]
    if blocks is None:
        s2_floats = (starts[:, None] + np.arange(n_orientations)).reshape(-1)
    else:
        sizes = [int(np.prod(s, dtype=np.int64)) for s in shapes]
        if [int(grid.floats_per_kind[k]) for k in grid.kinds[s2]] != sizes:
            raise ValueError("the reference reads a whole auto S2 chunk at every sigma it holds.")
        s2_floats = np.concatenate(
            [np.zeros(0, dtype=np.int64)]
            + [start + np.arange(size) for start, size in zip(starts.tolist(), sizes, strict=True)]
        )
    return layout.chunk_starts("var")[var], s2_floats.astype(np.int64)


def self_reference(
    backend: Backend,
    buffer: Any,
    layout: Layout,
    present: NDArray[np.bool_],
    n_orientations: int,
    ref_batch_mean: bool,
    blocks: BlocksTensorProd | None = None,
    spectrum: SpectrumReference | None = None,
) -> _NormReference:
    """The reference of ``norm="self"``, from the statistics being normalised (M-NORM,
    M-NORM-TP; STL ``to_norm(norm_type="self")``, ``_get_sqrt_chan_diag``): ``buffer`` in
    ``layout``, the layout of the loop. ``present`` is where sigma exists; ``blocks`` those of
    a tensor product, or ``None``; ``spectrum`` the channels and parameters of the power
    spectrum, whose σᴾˢ is read from the auto spectra of the buffer, or ``None`` without it."""
    xp = backend.xp
    var_floats, s2_floats = _reference_tables(
        layout, present.shape, present.tobytes(), n_orientations, blocks
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
        blocks=blocks,
        sigma_ps=None
        if spectrum is None
        else _sigma_ps(backend, buffer, layout, spectrum, ref_batch_mean),
        spectrum=spectrum,
    )


def _sigma_ps(
    backend: Backend,
    buffer: Any,
    layout: Layout,
    spectrum: SpectrumReference,
    ref_batch_mean: bool,
) -> Any:
    """σᴾˢ of ``norm="self"``, ``(Nr, n)``: the square root of the auto spectra of the channels of
    ``spectrum.present`` in ``buffer``, averaged over the batch first with ``ref_batch_mean``
    (maths.md M-NORM, §1.9). One helper for the references of both operators."""
    xp = backend.xp
    floats = _spectrum_tables(layout, spectrum.present.tobytes())
    auto_ps = xp.take(buffer, constant(backend, floats, buffer, "int32"), axis=1)
    if ref_batch_mean:
        auto_ps = xp.mean(auto_ps, axis=0, keepdims=True)
    return xp.sqrt(auto_ps)


@functools.lru_cache(maxsize=64)
def _spectrum_tables(layout: Layout, data: bytes) -> NDArray[np.int64]:
    """The floats of the auto spectrum PS[c, c, :] of every channel of the ``present`` given by
    ``data``, in a buffer of ``layout``, a layout of the loop: one real chunk of n_b floats per
    channel (maths.md §1.9, M-NORM)."""
    present = np.frombuffer(data, dtype=bool)
    channels = np.flatnonzero(present).astype(np.int64)[:, None]
    chunks = _chunks_at(layout, "PS", np.concatenate([channels, channels], axis=1))
    grid = layout.grid("PS")
    if (chunks < 0).any() or not grid.real[grid.kinds[chunks]].all():
        raise ValueError("the reference reads a real auto spectrum at every channel it holds.")
    n_bins = grid.n_orientations
    starts = layout.chunk_starts("PS")[chunks]
    found: NDArray[np.int64] = (starts[:, None] + np.arange(n_bins)).reshape(-1).astype(np.int64)
    return found


def as_constant(backend: Backend, reference: _NormReference) -> _NormReference:
    """The reference of ``norm="ref"``: a constant, through which no gradient flows (LF21)."""
    return _NormReference(
        var=backend.stop_gradient(reference.var),
        sigma=backend.stop_gradient(reference.sigma),
        present=reference.present,
        n_orientations=reference.n_orientations,
        ref_batch_mean=reference.ref_batch_mean,
        blocks=reference.blocks,
        sigma_ps=None if reference.sigma_ps is None else backend.stop_gradient(reference.sigma_ps),
        spectrum=reference.spectrum,
    )


def _sigma_columns(name: str, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """For a normalised family, the chunk column of the channel and the first chunk column of
    the ``k`` scale axes of each of the two sigma of its divisor (M-NORM, M-NORM-TP):
    sigma_{c1}(j, l) sigma_{c2}(j, l) for S1 and S2, sigma_{c1}(j2, l2) sigma_{c2}(j3, l3) for
    S3, sigma_{c1}(j1, l1) sigma_{c2}(j2, l2) for S4; ``k`` is 1, or the number of blocks."""
    second = 2 if name in ("S1", "S2") else 2 + k
    return (0, 2), (1, second)


_SIGMA_ORIENTATIONS = {"S1": (0, 0), "S2": (0, 0), "S3": (0, 1), "S4": (0, 1)}
"""The orientation group of each of the two sigma: l, l; l2, l3; l1, l2 (one axis per block
in a tensor product)."""


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
    layout: Layout,
    shape: tuple[int, ...],
    data: bytes,
    n_orientations: int,
    blocks: BlocksTensorProd | None = None,
    spectrum: bytes | None = None,
) -> _NormTables:
    """:class:`_NormTables` of ``layout``, a layout of the loop, whose chunk positions are the
    channels and scales themselves, and of the ``present`` given by ``shape`` and ``data``;
    ``spectrum`` is the ``present`` of σᴾˢ (``SpectrumReference.present``), by its bytes, when
    ``layout`` holds the power spectrum."""
    present = np.frombuffer(data, dtype=bool).reshape(shape)
    k = len(shape) - 1
    rank = _ranks(present)
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
        if grid.name == "PS":
            if spectrum is None:
                raise ValueError("the power spectrum is normalised by a reference that has it.")
            ps_rank = _ranks(np.frombuffer(spectrum, dtype=bool))
            a, b = ps_rank[chunks[:, 0]], ps_rank[chunks[:, 1]]
            kept = np.flatnonzero((a >= 0) & (b >= 0))
            family = FamilyChunks(
                grid.replace(chunks=chunks[kept], kinds=grid.kinds[kept]), levels, level_ids[kept]
            )
            specs.append(Chunked(family, kept[:, None], None))
            factors = {**factors, "PS": (a[kept], b[kept])}
            continue
        if grid.name not in _SIGMA_ORIENTATIONS:
            specs.append(Copy(grid.name))
            continue

        def row(columns: tuple[int, int], found: NDArray[np.int64] = chunks) -> NDArray[np.int64]:
            c, first = columns
            return rank[(found[:, c], *found[:, first : first + k].T)]  # type: ignore[no-any-return]

        first, second = _sigma_columns(grid.name, k)
        a, b = row(first), row(second)
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
    sigma_{c1}(j2, l2) sigma_{c2}(j3, l3); S4 by sigma_{c1}(j1, l1) sigma_{c2}(j2, l2); PS by
    σᴾˢ_{c1}(b) σᴾˢ_{c2}(b) (maths.md §1.9). The reference broadcasts over the batch (Nr = 1)
    or matches it. ``layout`` is a layout of the loop (``_layout.layout_single_domain``,
    ``_layout.layout_tensor_prod``). In a tensor product the scales and orientations are
    multi-indices, and S3 and S4 read the wavelet scales of the reference only (M-NORM-TP).
    """
    present = reference.present
    if reference.blocks is not None:
        key_tensor_prod = (layout, present.shape, present.tobytes(), reference.n_orientations)
        tables_tensor_prod = _norm_tables(*key_tensor_prod, reference.blocks)
        divide_tensor_prod = _compiled_tensor_prod(backend.name, *key_tensor_prod, reference.blocks)
        chunks = map_tables(backend, tables_tensor_prod.transform, buffer)
        found = _factor_tables_tensor_prod(backend, tables_tensor_prod, reference, buffer)
        return (
            divide_tensor_prod(buffer, reference.var, reference.sigma, chunks, found),
            tables_tensor_prod.transform.layout,
        )
    spectrum = None
    if "PS" in layout.names and reference.spectrum is not None:
        spectrum = reference.spectrum.present.tobytes()
    key = (layout, present.shape, present.tobytes(), reference.n_orientations, spectrum)
    divide = _compiled_single_domain(backend.name, *key)
    transform = _norm_tables(*key[:4], None, spectrum).transform
    tables = map_tables(backend, transform, buffer)
    sigma_ps = reference.sigma_ps if spectrum is not None else None
    return divide(buffer, reference.var, reference.sigma, tables, sigma_ps), transform.layout


@functools.lru_cache(maxsize=64)
def _compiled_single_domain(
    backend_name: str,
    layout: Layout,
    shape: tuple[int, ...],
    data: bytes,
    n_orientations: int,
    spectrum: bytes | None = None,
) -> Any:
    """The division of :func:`normalize` for one layout and one ``present`` (and that of σᴾˢ),
    a function of the buffer, ``var_ref``, sigma, the tables of the chunks
    (``_layout.map_tables``) and σᴾˢ, compiled once where the backend compiles
    (``Backend.jit``; the identity on PyTorch)."""
    backend = _registry.backend_named(backend_name)
    tables = _norm_tables(layout, shape, data, n_orientations, None, spectrum)
    return backend.jit(
        functools.partial(_divide_single_domain, backend, tables, layout, n_orientations)
    )


def _divide_single_domain(
    backend: Backend,
    tables: _NormTables,
    layout: Layout,
    n_orientations: int,
    buffer: Any,
    var: Any,
    sigma: Any,
    chunks: MapTables,
    sigma_ps: Any = None,
) -> Any:
    """The buffer of the kept chunks divided by their divisors (:func:`normalize`); ``sigma_ps`` is
    σᴾˢ, ``(Nr, n)``, when the layout holds the power spectrum."""
    xp = backend.xp
    L = n_orientations  # noqa: N806
    n_ref = int(var.shape[0])
    sigma = xp.reshape(sigma, (n_ref, -1, L))

    def divided(name: str) -> Any:
        a, b = tables.rows[name]
        n = int(a.shape[0])
        if name == "var":
            divisor = xp.take(var, constant(backend, a, var, "int32"), axis=1)
            return lambda dense, *_: dense / xp.reshape(divisor, (n_ref, n, 1, 1))
        if name == "PS":
            n_bins = layout.grid("PS").n_orientations
            rows = xp.reshape(sigma_ps, (n_ref, -1, n_bins))
            first = xp.take(rows, constant(backend, a, var, "int32"), axis=1)  # (Nr, n, n_b)
            second = xp.take(rows, constant(backend, b, var, "int32"), axis=1)
            spectrum = xp.reshape(first * second, (n_ref, n, n_bins, 1))
            return lambda dense, *_: dense / spectrum
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
        return lambda dense, *_: dense / divisor

    scale = {name: divided(name) for name in tables.rows}
    return apply(backend, buffer, tables.transform, scale, chunks)


# ---------------------------------------------------------------------- tensor products


def _groups_tensor_prod(
    tables: _NormTables, name: str
) -> tuple[tuple[NDArray[np.int64] | None, tuple[int, ...]], ...]:
    """The groups of chunks the map of the normalisation passes to the divisor of ``name``, in
    the order it passes them: the positions of their chunks (``None`` for all) and their
    orientation shape."""
    part = next(p for p in tables.transform.parts if p.name == name)
    groups = getattr(part, "groups", None)
    if groups is None:
        return ((None, tables.transform.layout.grid(name).orientations),)
    return tuple((g.rows, g.orientations) for g in groups)


def _factor_tables_tensor_prod(
    backend: Backend, tables: _NormTables, reference: _NormReference, like: Any
) -> dict[str, Any]:
    """The first float in sigma of the two factors of the divisor of every chunk, group by
    group, as integer arrays of the backend (for var, the channel of every chunk): tables of
    one entry per chunk, arguments of the compiled division, never constants."""
    _, _, starts = reference.rows()
    found: dict[str, Any] = {}
    for name, (a, b) in tables.rows.items():
        if name == "var":
            found = {**found, name: (constant(backend, a, like, "int32"),)}
            continue
        found = {
            **found,
            name: tuple(
                (
                    constant(backend, starts[a if rows is None else a[rows]], like, "int32"),
                    constant(backend, starts[b if rows is None else b[rows]], like, "int32"),
                )
                for rows, _ in _groups_tensor_prod(tables, name)
            ),
        }
    return found


@functools.lru_cache(maxsize=64)
def _compiled_tensor_prod(
    backend_name: str,
    layout: Layout,
    shape: tuple[int, ...],
    data: bytes,
    n_orientations: int,
    blocks: BlocksTensorProd,
) -> Any:
    """The division of :func:`normalize` on a tensor product, a function of the buffer,
    ``var_ref``, sigma, the tables of the chunks and those of the factors, compiled once where
    the backend compiles (``Backend.jit``; the identity on PyTorch)."""
    backend = _registry.backend_named(backend_name)
    tables = _norm_tables(layout, shape, data, n_orientations, blocks)
    return backend.jit(functools.partial(_divide_tensor_prod, backend, tables, blocks))


def _divide_tensor_prod(
    backend: Backend,
    tables: _NormTables,
    blocks: BlocksTensorProd,
    buffer: Any,
    var: Any,
    sigma: Any,
    chunks: MapTables,
    factors: dict[str, Any],
) -> Any:
    """The buffer of the kept chunks of a tensor product divided by their divisors
    (:func:`normalize`, M-NORM-TP), one orientation shape at a time: each factor is a row of
    sigma, read from its first float, its orientations along those of its filter."""
    xp = backend.xp
    n_ref = int(var.shape[0])
    n_blocks = len(blocks.L)

    def rows_of(starts: Any, shape: tuple[int, ...]) -> Any:
        size = int(np.prod(shape, dtype=np.int64))
        offsets = constant(backend, np.arange(size), starts, "int32")
        index = xp.reshape(xp.reshape(starts, (-1, 1)) + xp.reshape(offsets, (1, -1)), (-1,))
        n = int(starts.shape[0])
        return xp.reshape(xp.take(sigma, index, axis=1), (n_ref, n, *shape))

    def divided(name: str) -> Any:
        if name == "var":
            (channels,) = factors[name]
            n = int(channels.shape[0])
            divisor = xp.reshape(xp.take(var, channels, axis=1), (n_ref, n, 1, 1))
            return lambda dense, *_: dense / divisor
        groups = _groups_tensor_prod(tables, name)
        axes = _SIGMA_ORIENTATIONS[name]
        # The orientation groups of the family, one axis per block each: l; l2, l3; l1, l2, l3.
        n_groups = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}[name]
        divisors = []
        for (_, shape), (a, b) in zip(groups, factors[name], strict=True):
            row_shape = shape[:n_blocks]
            n = int(a.shape[0])

            def along(
                factor: Any, group: int, n: int = n, ones: tuple[int, ...] = (1,) * n_blocks
            ) -> Any:
                return xp.reshape(
                    factor,
                    (
                        n_ref,
                        n,
                        *(
                            s
                            for g in range(n_groups)
                            for s in (factor.shape[2:] if g == group else ones)
                        ),
                    ),
                )

            first = rows_of(a, row_shape)
            second = rows_of(b, shape[n_blocks * axes[1] : n_blocks * (axes[1] + 1)])
            product = along(first, axes[0]) * along(second, axes[1])
            if name == "S1":
                product = xp.sqrt(product)
            divisors.append(
                xp.reshape(xp.broadcast_to(product, (n_ref, n, *shape)), (n_ref, n, -1, 1))
            )
        return lambda dense, group: dense / divisors[group]

    scale = {name: divided(name) for name in tables.rows}
    return apply(backend, buffer, tables.transform, scale, chunks)


def check_reference_single_domain(
    reference: _NormReference, plan: STPlanSingleDomain, n_batch: int, dtype: Any
) -> None:
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
    if reference.blocks is not None:
        raise ValueError(
            "the reference comes from a tensor product, and these statistics are single-domain: "
            "normalise with single-domain statistics of the same J, L and channels (maths.md "
            "M-NORM)."
        )
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
    if plan.spectrum is not None:
        _check_spectrum(reference, plan)
    _check_batch_and_dtype(reference, n_batch, dtype)


def _check_spectrum(reference: _NormReference, plan: Any) -> None:
    """Raise unless ``reference`` holds σᴾˢ for the power spectrum of ``plan`` (maths.md M-NORM,
    §1.9), the plan of ``STOperator``: the same n_b, J and J_min, and σᴾˢ for every channel
    with an auto spectrum."""
    spectrum = plan.spectrum
    assert spectrum is not None
    expected = SpectrumParameters(spectrum.n_bins, spectrum.J, spectrum.Jmin)
    if reference.spectrum is None:
        raise ValueError(
            "the reference holds no power spectrum, and these statistics hold one: normalise "
            "with statistics computed with the power spectrum (compute_PS=True), with the same "
            f"n_bins, J and Jmin, {expected._asdict()} (maths.md M-NORM, §1.9)."
        )
    found = reference.spectrum.parameters
    mismatches = [
        f"{name} = {got} (the operator has {want})"
        for name, got, want in zip(("n_bins", "J", "Jmin"), found, expected, strict=True)
        if got != want
    ]
    if mismatches:
        raise ValueError(
            f"the power spectrum of the reference has {', '.join(mismatches)}: normalise with "
            "statistics computed by an operator with the same n_bins, J and Jmin of the "
            "spectrum; their size, mask and pbc may differ (maths.md M-NORM, §1.9)."
        )
    missing = [c for c in plan.pairs.auto if not reference.spectrum.present[c]]
    if missing:
        raise ValueError(
            f"the reference has no spectrum scale for channel(s) {missing}, which have an auto "
            "spectrum in cross_matrix: compute the reference with their diagonal entries set "
            "(maths.md M-NORM, M-CHAN, OP-30)."
        )


def check_reference_tensor_prod(
    reference: _NormReference, plan: STPlanTensorProd, n_batch: int, dtype: Any
) -> None:
    """Raise unless ``reference`` can normalise the statistics of the tensor product ``plan``
    (maths.md M-NORM-TP, M-NORM; Part IV, R-Y03): it must come from a tensor product with the
    same blocks, J, L, low-pass flags and number of channels, with a normalisation scale for
    every channel the plan normalises; the sizes and the periodicity of the blocks are free, as
    in M-NORM. The batch and the dtype follow :func:`check_reference_single_domain`."""
    expected = blocks_tensor_prod(plan)
    found = reference.blocks
    if found is None:
        raise ValueError(
            "the reference comes from single-domain statistics, and these are a tensor product: "
            "normalise with statistics of a tensor product with the same blocks, J, L and "
            "low-pass flags (maths.md M-NORM-TP)."
        )
    n_channels = int(reference.present.shape[0])
    mismatches = [
        f"{name} = {got} (the operator has {want})"
        for name, got, want in (
            ("the blocks", list(found.geometries), list(expected.geometries)),
            ("J", list(found.J), list(expected.J)),
            ("L", list(found.L), list(expected.L)),
            ("the low-pass flags", list(found.lowpass), list(expected.lowpass)),
            ("the number of channels", n_channels, plan.n_channels),
        )
        if got != want
    ]
    if mismatches:
        raise ValueError(
            f"the reference has {', '.join(mismatches)}: normalise with statistics of a tensor "
            "product with the same blocks, J, L, low-pass flags and channels; the sizes and the "
            "periodicity of its blocks may differ (maths.md M-NORM-TP, M-LP)."
        )
    with_scale = np.asarray(reference.present.reshape(n_channels, -1).any(axis=1), dtype=bool)
    missing = [c for c in plan.pairs.auto if not with_scale[c]]
    if missing:
        raise ValueError(
            f"the reference has no normalisation scale for channel(s) {missing}, which this "
            "operator normalises (they have auto statistics in cross_matrix): compute "
            "the reference with their diagonal entries set, so that it holds their auto "
            "statistics (maths.md M-NORM, M-CHAN, OP-30)."
        )
    _check_batch_and_dtype(reference, n_batch, dtype)


def _check_batch_and_dtype(reference: _NormReference, n_batch: int, dtype: Any) -> None:
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
