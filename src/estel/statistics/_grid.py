"""The full-grid view of the statistics: named axes and the positions where a coefficient
exists (Part II, §12.7; Part III, §19.5; maths.md M-CHAN, M-MEAN, M-FLAT; tasks T2.5, T2.15).

The statistics are stored in the compact layout (``_layout.py``); their full-grid views are
STL's full-grid tensors, NaN where no coefficient exists: the structural NaN (j1 ≤ j2 ≤ j3,
the channel pairs not computed, M-CHAN), the means with no valid pixel (OP-3), the cross
statistics of a channel without auto statistics after normalisation (OP-30), and the harmonics
of ``scale_cosine`` built from missing scales (M-RED-SCALE). Every one of them is known when
the operator is built, from the plan alone, and depends on the channels and the scales only:
:class:`FamilyGrid` records them by chunks, one row per channel pair and scale combination where
coefficients exist, each holding the dense array of its orientations (Part III, §19.4; task
T3.7). ``present``, where the layout has an entry, is built from them on demand: nothing absent
is stored, so no NaN reaches a reduction or a gradient (Part I, §3.1, rule 3).

The axes have names, used by ``STStatistics.select`` (Part II, §12.7):

* ``mean``, ``var``: ``(c,)``;
* ``S1``, ``S2``: ``(c1, c2, j, l)``;
* ``S3``: ``(c1, c2, j2, j3, l2, l3)`` (maths.md M-S3, OP-15);
* ``S4``: ``(c1, c2, j1, j2, j3, l1, l2, l3)``;
* ``PS``: ``(c1, c2, bin)``, the 2D power spectrum (maths.md §1.9), when the plan has it: one
  chunk per channel pair, holding the dense array of its bins (the axis ``bin`` plays the role
  of the orientations of the other families), every entry real (task T4.4).

``iso`` replaces the orientations of S1 and S2 by nothing, those of S3 by ``dl`` (l3 - l2 mod
L) and those of S4 by ``dl2``, ``dl3`` (l2 - l1, l3 - l1 mod L); ``angular_fourier`` prefixes
each orientation axis it transforms with ``h`` (``hl2``, ``hdl``, …); ``scale_cosine`` replaces
the scales of S3 by ``(k, a)`` and those of S4 by ``(k, a, b)`` (M-RED-ISO, M-RED-ANG,
M-RED-SCALE). The batch axis comes first in every array and is not named.

In a tensor product (:func:`grids_tensor_prod`; task T3.6) each scale and orientation axis is
one axis per block, ``j[0]``, ``j[1]`` … (maths.md M-S-TP); ``select`` takes the channels only,
naming the blocks being LF44.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..scattering._plan import STPlanSingleDomain, STPlanTensorProd

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4", "PS")
"""The families of ESTEL 1.0.0, in STL's order (``to_flatten``, M-FLAT): PS last, held only when
the plan has the power spectrum (maths.md §1.9; task T4.4)."""

REAL_PART_ONLY = ("S1", "S2", "PS")
"""The families whose imaginary part ``to_vector`` drops, as STL's ``to_flatten`` (M-FLAT,
OP-37)."""


def orientations_at_tensor_prod(
    scale: tuple[int, ...],
    J: tuple[int, ...],  # noqa: N803 (maths.md notation)
    L: tuple[int, ...],  # noqa: N803
) -> tuple[int, ...]:
    """The orientation shape of a tensor product at the multi-index ``scale``, for the blocks
    of ``J`` wavelet scales and ``L`` orientations: L_a, or 1 along a block taken at its
    low-pass scale J_a, which has orientation 0 only (maths.md M-LP). The chunks of S1 and S2
    and the rows of the normalisation scales sigma have this shape."""
    return tuple(1 if j == n else size for j, n, size in zip(scale, J, L, strict=True))


def _digest(array: NDArray[Any]) -> tuple[Any, ...]:
    """A short key of a static array, for hashing and equality: its shape, dtype and a digest
    of its bytes, so that a key holds no copy of the array."""
    data = np.ascontiguousarray(array)
    return (data.shape, str(data.dtype), hashlib.blake2b(data.tobytes(), digest_size=16).digest())


def _read_only_as(array: Any, dtype: Any) -> NDArray[Any]:
    found = np.asarray(array, dtype=dtype)
    if found.flags.writeable or found.base is not None:
        found = np.array(found, dtype=dtype)
        found.flags.writeable = False
    return found


@dataclass(frozen=True, eq=False)
class FamilyGrid:
    """The static layout of one family: its named axes, the original index of each position
    along them, and its chunks (Part III, §19.4; task T3.7). Hashable, so that it can be static
    data of a pytree.

    The leading :attr:`n_chunk_axes` axes are the chunk axes: the channels and the scales (and
    the axes that replace the scales after ``scale_cosine``); the others are the orientation
    axes. A coefficient exists for one position of the chunk axes at every orientation or at
    none (OP-3 and M-CHAN depend on the channels and the scales only), so the family is stored
    by chunks, one per position of the chunk axes where its coefficients exist, each holding
    the dense array of its orientations: one row per chunk, never a table per coefficient. In
    tensor products (T3.6) each scale becomes one chunk axis per block, and the orientations
    one axis per block and orientation index; a chunk of S1 or S2 whose scale takes a block at
    its low-pass scale holds orientation 0 only along that block (maths.md M-LP), a smaller
    dense array, the corner of the orientations at index 0: its kind says so (:attr:`shapes`).
    """

    name: str
    """``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``, ``"PS"``."""
    axes: tuple[str, ...]
    """The names of the axes after the batch axis: the chunk axes, then the orientation axes."""
    coords: tuple[tuple[int, ...], ...]
    """For each axis, the original index of each of its positions (changed by ``select``)."""
    is_complex: bool
    """Whether the family is stored complex (S1 to S4, PS) or real (mean, var)."""
    n_chunk_axes: int
    """The number of chunk axes, the leading axes."""
    chunks: NDArray[np.int32]
    """``(n_chunks, n_chunk_axes)``: the position of each chunk along the chunk axes, distinct;
    in a layout, in the order of the buffer. Read-only."""
    kinds: NDArray[np.int32]
    """``(n_chunks,)``: the kind of each chunk, a row of :attr:`real`. Read-only."""
    real: NDArray[np.bool_]
    """``(n_kinds, n_orientations)``: for each kind of chunk, which of its entries (the
    orientations flattened in C order) take one float, the others taking two (re, im): the auto
    terms, the diagonal of S4, and what the reductions make of them; every entry in a real
    family. Read-only. In a family with :attr:`shapes`, the entries are those of the full
    orientations, False outside the orientations of the kind."""
    shapes: NDArray[np.int32] | None = None
    """``(n_kinds, n_orientation_axes)``: the dense shape of the orientations of the chunks of
    each kind, a corner of :attr:`orientations` (the positions from 0 along each axis), or
    ``None`` when every kind holds every orientation (planar statistics, and the families of
    tensor products without a low-pass scale). Read-only."""

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.coords):
            raise ValueError(f"{self.name}: {len(self.axes)} axes, {len(self.coords)} coords.")
        if not 0 <= self.n_chunk_axes <= len(self.axes):
            raise ValueError(f"{self.name}: {self.n_chunk_axes} chunk axes of {len(self.axes)}.")
        kinds = _read_only_as(self.kinds, np.int32).reshape(-1)
        chunks = _read_only_as(self.chunks, np.int32).reshape(kinds.shape[0], self.n_chunk_axes)
        real = _read_only_as(self.real, bool).reshape(-1, self.n_orientations)
        if kinds.shape[0] != chunks.shape[0]:
            raise ValueError(f"{self.name}: {chunks.shape[0]} chunks, {kinds.shape[0]} kinds.")
        if kinds.shape[0] and (kinds.min() < 0 or kinds.max() >= real.shape[0]):
            raise ValueError(f"{self.name}: a chunk has a kind without its row of real entries.")
        if not self.is_complex and not real.all():
            raise ValueError(f"{self.name}: every entry of a real family takes one float.")
        shapes = None
        if self.shapes is not None:
            full = np.array(self.orientations, dtype=np.int32)
            shapes = _read_only_as(self.shapes, np.int32).reshape(real.shape[0], full.shape[0])
            if ((shapes < 1) | (shapes > full)).any():
                raise ValueError(f"{self.name}: a kind has orientations outside the family's.")
            if (shapes == full).all():
                shapes = None
            else:
                outside = real & ~_corners(shapes, self.orientations)
                if outside.any():
                    real = _read_only_as(real & ~outside, bool)
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "shapes", shapes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(c) for c in self.coords)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """The shape of the chunk axes."""
        return self.shape[: self.n_chunk_axes]

    @property
    def orientations(self) -> tuple[int, ...]:
        """The dense shape of the orientations of every chunk."""
        return self.shape[self.n_chunk_axes :]

    @property
    def n_orientations(self) -> int:
        return int(np.prod(self.orientations, dtype=np.int64))

    @property
    def n_chunks(self) -> int:
        return int(self.chunks.shape[0])

    @property
    def floats_per_kind(self) -> NDArray[np.int64]:
        """``(n_kinds,)``: the floats of a chunk of each kind."""
        if self.shapes is None:
            counts: NDArray[np.int64] = self.n_orientations + np.count_nonzero(~self.real, axis=1)
            return counts.astype(np.int64)
        exists = self.exists
        found: NDArray[np.int64] = np.asarray(
            np.count_nonzero(exists, axis=1) + np.count_nonzero(exists & ~self.real, axis=1),
            dtype=np.int64,
        )
        return found

    @property
    def exists(self) -> NDArray[np.bool_]:
        """``(n_kinds, n_orientations)``: which of the orientations of the family the chunks of
        each kind hold (the corner of their :attr:`shapes`)."""
        if self.shapes is None:
            return np.ones(self.real.shape, dtype=bool)
        return _corners(self.shapes, self.orientations)

    def kind_shape(self, kind: int) -> tuple[int, ...]:
        """The dense shape of the orientations of the chunks of ``kind``."""
        if self.shapes is None:
            return self.orientations
        return tuple(int(n) for n in self.shapes[kind])

    def real_in(self, shape: tuple[int, ...]) -> NDArray[np.bool_]:
        """``(n_kinds, prod(shape))``: :attr:`real` at the orientations of the corner ``shape``,
        in its C order: the real entries of a chunk of that shape."""
        if shape == self.orientations:
            return self.real
        found: NDArray[np.bool_] = self.real[:, corner(shape, self.orientations)]
        return found

    def shape_groups(
        self, rows: NDArray[np.int64]
    ) -> list[tuple[tuple[int, ...], NDArray[np.int64]]]:
        """The positions of ``rows`` (positions among the chunks) grouped by the orientation
        shape of their chunks, each group in the order of ``rows``, the groups in the order of
        their first chunk: what the maps of a family with several shapes act on, one group at
        a time (task T3.6)."""
        rows = np.asarray(rows, dtype=np.int64)
        if self.shapes is None:
            return [(self.orientations, np.arange(rows.shape[0], dtype=np.int64))]
        kinds = self.kinds[rows]
        rows_shapes = self.shapes[kinds] if rows.shape[0] else self.shapes[:0]
        if rows_shapes.shape[0] == 0:
            return []
        distinct, first, inverse = np.unique(
            rows_shapes, axis=0, return_index=True, return_inverse=True
        )
        inverse = inverse.reshape(-1)
        return [
            (tuple(int(n) for n in distinct[g]), np.flatnonzero(inverse == g).astype(np.int64))
            for g in np.argsort(first, kind="stable")
        ]

    @property
    def present(self) -> NDArray[np.bool_]:
        """True where a coefficient exists, over the axes after the batch axis: built on demand
        from the chunks, for the full-grid views and the tests of planar statistics; read-only.
        A family with several orientation shapes (tensor products) has no such mask: nothing
        the size of its full grid is built (task T3.6)."""
        if self.shapes is not None:
            raise ValueError(
                f"{self.name}: the chunks of this family have several orientation shapes (a "
                "tensor product with low-pass scales); it has no full-grid mask, read its chunks."
            )
        size = int(np.prod(self.chunk_shape, dtype=np.int64))
        flat = (
            np.ravel_multi_index(tuple(self.chunks.T.astype(np.int64)), self.chunk_shape)
            if self.n_chunk_axes
            else np.zeros(self.n_chunks, dtype=np.int64)
        )
        mask = (np.bincount(np.asarray(flat, dtype=np.int64), minlength=size) > 0).reshape(
            self.chunk_shape
        )
        out = np.array(
            np.broadcast_to(
                mask.reshape(self.chunk_shape + (1,) * len(self.orientations)), self.shape
            )
        )
        out.flags.writeable = False
        return out

    def _key(self) -> tuple[Any, ...]:
        return (
            self.name,
            self.axes,
            self.coords,
            self.is_complex,
            self.n_chunk_axes,
            _digest(self.chunks),
            _digest(self.kinds),
            _digest(self.real),
            None if self.shapes is None else _digest(self.shapes),
        )

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FamilyGrid):
            return NotImplemented
        return self is other or self._key() == other._key()

    def replace(self, **changes: Any) -> FamilyGrid:
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return FamilyGrid(**{**fields, **changes})

    def axis(self, name: str) -> int:
        """The position of the axis ``name`` among :attr:`axes`."""
        return self.axes.index(name)

    def is_complete(self, names: Sequence[str], size: int) -> bool:
        """Whether every axis of ``names`` holds all of its ``size`` original positions, in
        order: what a reduction along them needs."""
        return all(self.coords[self.axis(n)] == tuple(range(size)) for n in names)


def corner(shape: tuple[int, ...], orientations: tuple[int, ...]) -> NDArray[np.int64]:
    """The positions, flattened in the C order of ``orientations``, of the corner ``shape``
    (the positions from 0 along each axis), in its own C order: increasing."""
    if not orientations:
        return np.zeros(1, dtype=np.int64)
    grid = np.indices(shape).reshape(len(shape), -1)
    found: NDArray[np.int64] = np.asarray(
        np.ravel_multi_index(tuple(grid), orientations), dtype=np.int64
    )
    return found


def _corners(shapes: NDArray[np.int32], orientations: tuple[int, ...]) -> NDArray[np.bool_]:
    """``(n_kinds, prod(orientations))``: the corner of each row of ``shapes``."""
    size = int(np.prod(orientations, dtype=np.int64))
    rows = [
        np.isin(np.arange(size), corner(tuple(int(n) for n in row), orientations)) for row in shapes
    ]
    found: NDArray[np.bool_] = (
        np.stack(rows).reshape(-1, size) if rows else np.zeros((0, size), dtype=bool)
    )
    return found


def _family(
    name: str,
    axes: tuple[str, ...],
    chunk_present: NDArray[np.bool_],
    orientations: tuple[int, ...],
    is_complex: bool,
    kind_of: Any,
    real: NDArray[np.bool_],
) -> FamilyGrid:
    """The family ``name`` with a chunk at every True of ``chunk_present``, whose kind
    ``kind_of(chunks)`` gives (a row of ``real``)."""
    chunks = np.argwhere(chunk_present).astype(np.int32)
    shape = (*chunk_present.shape, *orientations)
    return FamilyGrid(
        name=name,
        axes=axes,
        coords=tuple(tuple(range(n)) for n in shape),
        is_complex=is_complex,
        n_chunk_axes=chunk_present.ndim,
        chunks=chunks,
        kinds=np.asarray(kind_of(chunks), dtype=np.int32).reshape(-1),
        real=real,
    )


def grids_single_domain(plan: STPlanSingleDomain) -> tuple[FamilyGrid, ...]:
    """The layout of the statistics without normalisation (maths.md M-MOM, M-S1 to M-S4,
    M-CHAN, OP-3): what the scattering loop computes, from the plan alone, by chunks. Real
    entries are mean, var, the auto S1 and S2, and the diagonal of S4 (c1 = c2, j1 = j2,
    l1 = l2): two kinds of chunks in S1, S2 and S4, auto (or diagonal) and cross."""
    nc, J, L = plan.n_channels, plan.J, plan.L  # noqa: N806
    pairs = plan.pairs
    computed_12 = {(c, c) for c in pairs.auto} | set(pairs.cross)
    first = np.array(
        [[(c1, c2) in computed_12 for c2 in range(nc)] for c1 in range(nc)], dtype=bool
    ).reshape(nc, nc)
    ordered = np.array(
        [[(c1, c2) in pairs.ordered for c2 in range(nc)] for c1 in range(nc)], dtype=bool
    ).reshape(nc, nc)
    scales = np.array([j in plan.first_layer for j in range(J)], dtype=bool)
    s3 = np.array([[(j2, j3) in plan.s3 for j3 in range(J)] for j2 in range(J)], dtype=bool)
    s4 = np.array(
        [[[(j1, j2, j3) in plan.s4 for j3 in range(J)] for j2 in range(J)] for j1 in range(J)],
        dtype=bool,
    ).reshape(J, J, J)
    l1, l2 = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    diagonal = np.broadcast_to((l1 == l2)[:, :, None], (L, L, L)).reshape(-1)
    s12_kinds = np.stack([np.ones(L, dtype=bool), np.zeros(L, dtype=bool)])

    def cross(chunks: NDArray[np.int32]) -> NDArray[np.bool_]:
        found: NDArray[np.bool_] = chunks[:, 0] != chunks[:, 1]
        return found

    def off_diagonal(chunks: NDArray[np.int32]) -> NDArray[np.bool_]:
        found: NDArray[np.bool_] = ~(
            (chunks[:, 0] == chunks[:, 1]) & (chunks[:, 2] == chunks[:, 3])
        )
        return found

    def single(chunks: NDArray[np.int32]) -> NDArray[np.int64]:
        return np.zeros(chunks.shape[0], dtype=np.int64)

    channels = np.ones(nc, dtype=bool)
    s12 = first[:, :, None] & scales
    # The spectrum has the channel pairs of S1 and S2: one matrix for both (decision on #148).
    spectrum = () if plan.spectrum is None else (spectrum_grid(plan.spectrum.n_bins, first),)
    return (
        _family("mean", ("c",), channels, (), False, single, np.ones((1, 1), dtype=bool)),
        _family("var", ("c",), channels, (), False, single, np.ones((1, 1), dtype=bool)),
        _family("S1", ("c1", "c2", "j", "l"), s12, (L,), True, cross, s12_kinds),
        _family("S2", ("c1", "c2", "j", "l"), s12, (L,), True, cross, s12_kinds),
        _family(
            "S3",
            ("c1", "c2", "j2", "j3", "l2", "l3"),
            ordered[:, :, None, None] & s3,
            (L, L),
            True,
            single,
            np.zeros((1, L * L), dtype=bool),
        ),
        _family(
            "S4",
            ("c1", "c2", "j1", "j2", "j3", "l1", "l2", "l3"),
            ordered[:, :, None, None, None] & s4,
            (L, L, L),
            True,
            off_diagonal,
            np.stack([diagonal, np.zeros(L**3, dtype=bool)]),
        ),
        *spectrum,
    )


PS_AXES = ("c1", "c2", "bin")
"""The axes of the power spectrum (task T4.4; decision of the maintainer on #151: the axis of
the bins is named ``bin``, to be thought again with the notation, W5, #71)."""


def spectrum_grid(n_bins: int, pairs: NDArray[np.bool_]) -> FamilyGrid:
    """The family ``"PS"`` of the 2D power spectrum with ``n_bins`` bins (maths.md §1.9, "Channel
    pairs and layout"; M-CHAN): one chunk per True of ``pairs`` ``(Nc, Nc)``, the pairs c1 ≤ c2
    of ``cross_matrix``, holding the dense array of its bins; one kind, every entry real (one
    float), as the auto S1 and S2, the full grid complex with an imaginary part 0."""
    return _family(
        "PS",
        PS_AXES,
        pairs,
        (n_bins,),
        True,
        lambda chunks: np.zeros(chunks.shape[0], dtype=np.int64),
        np.ones((1, n_bins), dtype=bool),
    )


def auto_scales_single_domain(plan: STPlanSingleDomain) -> NDArray[np.bool_]:
    """``(Nc, J)``: where the auto S2 exists, so where the normalisation scale sigma_c(j, l) of
    M-NORM is defined: the channels with auto statistics, at the scales with valid pixels."""
    present = np.array(
        [
            [c in plan.pairs.auto and j in plan.first_layer for j in range(plan.J)]
            for c in range(plan.n_channels)
        ],
        dtype=bool,
    ).reshape(plan.n_channels, plan.J)
    present.flags.writeable = False
    return present


# ---------------------------------------------------------------------- tensor products


def block_axes_tensor_prod(prefix: str, n_blocks: int) -> tuple[str, ...]:
    """The names of the axes of one multi-index of a tensor product, one per block:
    ``j[0]``, ``j[1]`` … for ``prefix`` ``"j"``. Internal: naming the blocks is LF44, and
    ``select`` does not take these axes."""
    return tuple(f"{prefix}[{a}]" for a in range(n_blocks))


def _product_rows_tensor_prod(
    pairs: Sequence[tuple[int, int]], per_block: Sequence[Sequence[tuple[int, ...]]]
) -> NDArray[np.int32]:
    """``(n, 2 + k · B)``: the channel pairs crossed with the Cartesian product of the index
    sets of the blocks (M-IDX-TP), each a tuple of k scales; the scales of a chunk are written
    filter by filter, each with one axis per block. One row per chunk: never a table the size
    of the full grid."""
    n_blocks = len(per_block)
    k = len(per_block[0][0]) if per_block and per_block[0] else 0
    scales = [
        [combo[a][i] for i in range(k) for a in range(n_blocks)]
        for combo in itertools.product(*per_block)
    ]
    rows = [[c1, c2, *row] for c1, c2 in pairs for row in scales]
    return np.array(rows, dtype=np.int32).reshape(len(rows), 2 + k * n_blocks)


def _from_rows_tensor_prod(
    name: str,
    axes: tuple[str, ...],
    chunk_shape: tuple[int, ...],
    orientations: tuple[int, ...],
    chunks: NDArray[np.int32],
    kinds: NDArray[np.int64],
    real: NDArray[np.bool_],
    shapes: NDArray[np.int32] | None = None,
) -> FamilyGrid:
    shape = (*chunk_shape, *orientations)
    return FamilyGrid(
        name=name,
        axes=axes,
        coords=tuple(tuple(range(n)) for n in shape),
        is_complex=True,
        n_chunk_axes=len(chunk_shape),
        chunks=chunks,
        kinds=kinds.astype(np.int32),
        real=real,
        shapes=shapes,
    )


def grids_tensor_prod(plan: STPlanTensorProd) -> tuple[FamilyGrid, ...]:
    """The layout of the statistics of a tensor product without normalisation (maths.md
    M-S-TP, M-IDX-TP, M-LP, M-CHAN), from the plan alone, by chunks (Part III, §19.4; task
    T3.6): the chunks are the channel pairs crossed with the Cartesian product of the index
    sets of the blocks, never read from an array the size of the full grid.

    The axes are those of STL's full grid (M-S-TP), with one scale and one orientation axis per
    block: ``(c1, c2, j[0] … , l[0] …)`` for S1 and S2, ``(c1, c2, j2[0] …, j3[0] …, l2[0] …,
    l3[0] …)`` for S3, and likewise ``j1``, ``j2``, ``j3``, ``l1``, ``l2``, ``l3`` for S4. A chunk
    of S1 or S2 at a scale with the low-pass filter of a block holds orientation 0 only along
    that block (M-LP). Real entries are mean, var, the auto S1 and S2, and the diagonal of S4
    (c1 = c2, j1 = j2, l1 = l2 in every block)."""
    nc, n_blocks = plan.n_channels, plan.n_blocks
    J, L = plan.J, plan.L  # noqa: N806
    n_o = int(np.prod(L, dtype=np.int64))
    pairs = plan.pairs
    pairs_12 = sorted({(c, c) for c in pairs.auto} | set(pairs.cross))
    channels = np.arange(nc, dtype=np.int32).reshape(nc, 1)
    single = np.zeros(nc, dtype=np.int64)

    def moments(name: str) -> FamilyGrid:
        return FamilyGrid(
            name=name,
            axes=("c",),
            coords=(tuple(range(nc)),),
            is_complex=False,
            n_chunk_axes=1,
            chunks=channels,
            kinds=single.astype(np.int32),
            real=np.ones((1, 1), dtype=bool),
        )

    # S1 and S2: two kinds (auto, cross) per orientation shape.
    rows_12 = _product_rows_tensor_prod(pairs_12, [[(j,) for j in js] for js in plan.first_layer])
    scale_12 = rows_12[:, 2:]
    chunk_shapes = np.array(
        [plan.orientations_at(tuple(int(j) for j in row)) for row in scale_12], dtype=np.int32
    ).reshape(-1, n_blocks)
    cross = (rows_12[:, 0] != rows_12[:, 1]).astype(np.int32)
    keys = np.concatenate([cross[:, None], chunk_shapes], axis=1)
    distinct, kinds_12 = np.unique(keys, axis=0, return_inverse=True)
    shapes_12 = distinct[:, 1:].astype(np.int32)
    real_12 = _corners(shapes_12, L) & (distinct[:, :1] == 0)
    axes_12 = (
        "c1",
        "c2",
        *block_axes_tensor_prod("j", n_blocks),
        *block_axes_tensor_prod("l", n_blocks),
    )
    shape_12 = (nc, nc, *plan.n_scales)

    # S3: one kind, every entry complex.
    rows_3 = _product_rows_tensor_prod(pairs.ordered, plan.s3)
    axes_3 = (
        "c1",
        "c2",
        *(name for p in ("j2", "j3") for name in block_axes_tensor_prod(p, n_blocks)),
        *(name for p in ("l2", "l3") for name in block_axes_tensor_prod(p, n_blocks)),
    )

    # S4: the diagonal kind (c1 = c2, j1 = j2), real where l1 = l2 in every block, and the rest.
    rows_4 = _product_rows_tensor_prod(pairs.ordered, plan.s4)
    j1, j2 = rows_4[:, 2 : 2 + n_blocks], rows_4[:, 2 + n_blocks : 2 + 2 * n_blocks]
    off_diagonal = ~((rows_4[:, 0] == rows_4[:, 1]) & (j1 == j2).all(axis=1))
    l1, l2 = np.meshgrid(np.arange(n_o), np.arange(n_o), indexing="ij")
    diagonal = np.broadcast_to((l1 == l2)[:, :, None], (n_o, n_o, n_o)).reshape(-1)
    axes_4 = (
        "c1",
        "c2",
        *(name for p in ("j1", "j2", "j3") for name in block_axes_tensor_prod(p, n_blocks)),
        *(name for p in ("l1", "l2", "l3") for name in block_axes_tensor_prod(p, n_blocks)),
    )
    return (
        moments("mean"),
        moments("var"),
        *(
            _from_rows_tensor_prod(
                name,
                axes_12,
                shape_12,
                L,
                rows_12,
                kinds_12.reshape(-1),
                real_12,
                shapes_12,
            )
            for name in ("S1", "S2")
        ),
        _from_rows_tensor_prod(
            "S3",
            axes_3,
            (nc, nc, *J, *J),
            (*L, *L),
            rows_3,
            np.zeros(rows_3.shape[0], dtype=np.int64),
            np.zeros((1, n_o * n_o), dtype=bool),
        ),
        _from_rows_tensor_prod(
            "S4",
            axes_4,
            (nc, nc, *J, *J, *J),
            (*L, *L, *L),
            rows_4,
            off_diagonal.astype(np.int64),
            np.stack([diagonal, np.zeros(n_o**3, dtype=bool)]),
        ),
    )


def auto_scales_tensor_prod(plan: STPlanTensorProd) -> NDArray[np.bool_]:
    """``(Nc, J'_0, …, J'_{B-1})``: where the normalisation scale sigma_c(j, l) of M-NORM-TP is
    defined: the channels with auto statistics, at every scale of S1 and S2, the low-pass
    scales included. Its size is that of the channels and the scales, not of the full grid."""
    auto = np.isin(np.arange(plan.n_channels), plan.pairs.auto)
    present = np.array(
        np.broadcast_to(auto.reshape(-1, *(1,) * plan.n_blocks), (plan.n_channels, *plan.n_scales))
    )
    present.flags.writeable = False
    return present
