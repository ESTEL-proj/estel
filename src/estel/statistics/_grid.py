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
* ``S4``: ``(c1, c2, j1, j2, j3, l1, l2, l3)``.

``iso`` replaces the orientations of S1 and S2 by nothing, those of S3 by ``dl`` (l3 - l2 mod
L) and those of S4 by ``dl2``, ``dl3`` (l2 - l1, l3 - l1 mod L); ``angular_fourier`` prefixes
each orientation axis it transforms with ``h`` (``hl2``, ``hdl``, …); ``scale_cosine`` replaces
the scales of S3 by ``(k, a)`` and those of S4 by ``(k, a, b)`` (M-RED-ISO, M-RED-ANG,
M-RED-SCALE). The batch axis comes first in every array and is not named.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..scattering._plan import STPlan

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
"""The families of ESTEL 1.0.0, in STL's order (``to_flatten``); PS joins them in T4.1."""

REAL_PART_ONLY = ("S1", "S2", "PS")
"""The families whose imaginary part ``to_vector`` drops, as STL's ``to_flatten`` (M-FLAT,
OP-37)."""


def _digest(array: NDArray[Any]) -> tuple[Any, ...]:
    """A short key of a static array, for hashing and equality: its shape, dtype and a digest
    of its bytes, so that a key holds no copy of the array."""
    data = np.ascontiguousarray(array)
    return (data.shape, str(data.dtype), hashlib.blake2b(data.tobytes(), digest_size=16).digest())


def _read_only(array: Any, dtype: Any) -> NDArray[Any]:
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
    one axis per block and orientation index.
    """

    name: str
    """``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``."""
    axes: tuple[str, ...]
    """The names of the axes after the batch axis: the chunk axes, then the orientation axes."""
    coords: tuple[tuple[int, ...], ...]
    """For each axis, the original index of each of its positions (changed by ``select``)."""
    is_complex: bool
    """Whether the family is stored complex (S1 to S4) or real (mean, var)."""
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
    family. Read-only."""

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.coords):
            raise ValueError(f"{self.name}: {len(self.axes)} axes, {len(self.coords)} coords.")
        if not 0 <= self.n_chunk_axes <= len(self.axes):
            raise ValueError(f"{self.name}: {self.n_chunk_axes} chunk axes of {len(self.axes)}.")
        kinds = _read_only(self.kinds, np.int32).reshape(-1)
        chunks = _read_only(self.chunks, np.int32).reshape(kinds.shape[0], self.n_chunk_axes)
        real = _read_only(self.real, bool).reshape(-1, self.n_orientations)
        if kinds.shape[0] != chunks.shape[0]:
            raise ValueError(f"{self.name}: {chunks.shape[0]} chunks, {kinds.shape[0]} kinds.")
        if kinds.shape[0] and (kinds.min() < 0 or kinds.max() >= real.shape[0]):
            raise ValueError(f"{self.name}: a chunk has a kind without its row of real entries.")
        if not self.is_complex and not real.all():
            raise ValueError(f"{self.name}: every entry of a real family takes one float.")
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "real", real)

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
        counts: NDArray[np.int64] = self.n_orientations + np.count_nonzero(~self.real, axis=1)
        return counts.astype(np.int64)

    @property
    def present(self) -> NDArray[np.bool_]:
        """True where a coefficient exists, over the axes after the batch axis: built on demand
        from the chunks, for the full-grid views and the tests; read-only."""
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


def base_grids(plan: STPlan) -> tuple[FamilyGrid, ...]:
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
    )


def auto_scales(plan: STPlan) -> NDArray[np.bool_]:
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
