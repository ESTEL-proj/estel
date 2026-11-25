"""HEALPix data: ``DataHealpix`` (Part II, §12.4; maths.md M-DATA-HPX; task T5.5, issue #166).

One pixel axis, NESTED ordering, on the full sky or on the partial sky of ``cell_ids``. The
resolution is ``depth`` = log₂ ``nside`` (rule A4): inferred from the number of pixels on a full
sky, required with ``cell_ids`` (maths.md OP-91); ``cell_ids`` strictly increasing (OP-90).
``pbc`` is not an argument: it is True on a full sky and False with ``cell_ids`` (M-DATA-HPX).
The mask of missing values is read or checked as for the planar classes (OP-26).

The geometry hooks apply the static tables of each level that the wavelet operator gives them
(``estel.wavelets._healpix_operator``), built in NumPy with the plan from the tables of
``estel.wavelets._healpix_geometry`` (T5.1), whose layout they follow:

* ``_conv``: the interpolated stencils with ``gather_weighted``, then the kernel of M-WAV-HPX as
  real pairs, one real product (M-CONV-HPX; Part I, §3.1, rule 6);
* ``_downsample``: the decimation with ``gather_weighted`` then ``segment_sum`` (M-DOWN-HPX),
  normalised as M-DOWN-MASK with a mask;
* ``_crop_weights``: no border (M-MASK-HPX), every pixel of the level.

Each application is one function of a range of output pixels (:func:`_stencil_rows`,
:func:`_decimation_rows`), called here once on all of them, so that the pixel batches of T5.7
can split it without changing the hooks or the plan.

``data`` does not import ``wavelets``: the tables come in as :class:`HealpixConvTables` and
:class:`HealpixDownTables`, named tuples of backend arrays, built by the wavelet operator.
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ._base import (
    Data,
    DownsampleWeights,
    _promote,
    _to_backend,
    _validate_mask,
)
from ._moments import _is_complex

_PARTITION_ADVICE = (
    "partition is reserved for the MPI partition of step 2 (Part II, §9.4; LF34): ESTEL 1.0.0 "
    "computes on the whole map, the full sky or the partial sky of cell_ids. Pass partition=None "
    "(the default)."
)


# ---------------------------------------------------------------------- tables of the hooks


class HealpixConvTables(NamedTuple):
    """The static tables of ``DataHealpix._conv`` at one level, backend arrays built by the
    wavelet operator from ``healpix_stencils`` and ``healpix_kernel_weights`` (Part II, §12.5).
    """

    index: Any
    """The positions, in the pixels of the level, of the 4 neighbours of each stencil point,
    int32, ``(L, Npix, K², 4)``: gauge, pixel, stencil point, neighbour."""
    weight: Any
    """Their interpolation weights, real, of the operator's dtype, the shape of ``index``."""
    kernel: Any
    """The kernel of M-WAV-HPX as real pairs, ``(K², 2)``: its real and imaginary parts."""


class HealpixDownTables(NamedTuple):
    """The static tables of one decimation step, ``DataHealpix._downsample`` from a level to the
    next: a CSR matrix sorted by coarse pixel, from ``healpix_decimation``."""

    index: Any
    """The positions of the fine pixels in the pixels of the level, int32, ``(nnz, 1, 1)``."""
    weight: Any
    """Their weights, real, of the operator's dtype, ``(nnz, 1, 1)``."""
    segment_ids: Any
    """The position of the coarse pixel of each entry, int32, ``(nnz, )``, non-decreasing."""


# ---------------------------------------------------------------------- pytree metadata


class _StaticCellIds:
    """``cell_ids`` as pytree metadata: hashable, by a digest of its bytes, and equal to another
    ``_StaticCellIds`` exactly when the pixels are equal (issue #166, decision 7).

    JAX hashes the static part of a pytree; a Python tuple of up to 196 608 integers would be
    slow to hash and compare, so the pixels are a read-only int64 array behind this wrapper. It
    also holds the number of pixels of every level, from 0 to ``depth``: the parents of the
    pixels of the level above, as the decimation gives them (M-DOWN-HPX).
    """

    __slots__ = ("_digest", "cell_ids", "level_sizes")

    def __init__(self, cell_ids: NDArray[np.int64], depth: int) -> None:
        ids = np.array(cell_ids, dtype=np.int64, copy=True)
        ids.flags.writeable = False
        self.cell_ids = ids
        self._digest = hashlib.blake2b(ids.tobytes(), digest_size=16).digest()
        self.level_sizes = tuple(int(np.unique(ids >> (2 * d)).size) for d in range(depth + 1))

    def at_level(self, level: int) -> NDArray[np.int64]:
        """The NESTED pixels of ``level``, ascending: the ancestors of :attr:`cell_ids`."""
        return np.unique(self.cell_ids >> (2 * level))

    def __hash__(self) -> int:
        return hash(self._digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _StaticCellIds):
            return NotImplemented
        return self.cell_ids is other.cell_ids or (
            self._digest == other._digest and bool(np.array_equal(self.cell_ids, other.cell_ids))
        )

    def __repr__(self) -> str:
        return f"_StaticCellIds(n={self.cell_ids.size})"


# ---------------------------------------------------------------------- the class


class DataHealpix(Data):
    """HEALPix data, shape ``(Nb, Nc, Npix)``, NESTED ordering (maths.md M-DATA-HPX).

    ``DataHealpix(array, mask=None, *, depth=None, cell_ids=None, partition=None)``.

    Parameters
    ----------
    array:
        The data, real: as for :class:`estel.Data`. ``(Npix,)`` is promoted to ``(1, 1, Npix)``
        and ``(Nc, Npix)`` to ``(1, Nc, Npix)``. The pixels are in the NESTED ordering.
    mask:
        Optional boolean array ``(Npix,)``, True where a value is missing, shared by every batch
        element and channel. If it is not given, it is read from the NaN of ``array``, which must
        then hold a single map (maths.md M-DATA-HPX, OP-26).
    depth:
        log₂ ``nside`` of the map. On a full sky it is read from Npix = 12 · 4^depth, and may be
        given to check it; with ``cell_ids`` it is required (OP-91).
    cell_ids:
        The NESTED indices of the pixels of a partial sky, integers, strictly increasing (OP-90),
        one per pixel; ``None`` (the default) for the full sky. ``cell_ids`` that cover the whole
        sphere, 12 · 4^depth of them, raise: omit them for a full sky (rule A6). With
        ``cell_ids``, ``pbc`` is False.
    partition:
        Reserved for step 2 (LF34): ``None`` only.

    ``pbc`` is a read-only property, True on a full sky and False with ``cell_ids``.

    ``ndim`` is 1: it counts the spatial axes of the array, and the pixels lie on one axis,
    though the sphere is a 2D surface. The geometry is tested by its tag, ``_geometry``
    (``"healpix"``), never by ``ndim``.
    """

    ndim: ClassVar[int] = 1
    _nan_threshold: ClassVar[float] = 0.33
    _geometry: ClassVar[str] = "healpix"

    __slots__ = ("_cell_ids", "_depth")

    _depth: int
    _cell_ids: _StaticCellIds | None

    def __init__(
        self,
        array: Any,
        mask: Any = None,
        *,
        depth: int | None = None,
        cell_ids: Any = None,
        partition: Any = None,
    ) -> None:
        cls = type(self)
        if partition is not None:
            raise ValueError(f"DataHealpix(partition={partition!r}): {_PARTITION_ADVICE}")
        backend_name = _registry.get_backend()
        values = _promote(_to_backend(array, backend_name, cls), cls)
        n_pix = int(values.shape[-1])
        if cell_ids is None:
            found_depth = _full_sky_depth(n_pix, depth)
            static_ids = None
        else:
            found_depth, ids = _partial_sky(cell_ids, depth, n_pix)
            static_ids = _StaticCellIds(ids, found_depth)
        backend = _registry.current_backend()
        nan = np.asarray(backend.to_numpy(backend.xp.isnan(values)), dtype=bool)
        full_mask = _validate_mask(mask, nan, values.shape, backend, cls)
        self._set(values, full_mask, static_ids is None, backend_name, (found_depth, static_ids))

    @classmethod
    def _accepted_shapes(cls) -> str:
        return "(Npix), (Nc, Npix) or (Nb, Nc, Npix)"

    # ------------------------------------------------------------------ construction

    def _meta(self) -> tuple[Any, ...]:
        """``(depth, cell_ids)``: the static metadata beyond the mask, ``cell_ids`` as its
        hashable wrapper (``None`` on the full sky)."""
        return (self._depth, self._cell_ids)

    def _set_meta(self, meta: tuple[Any, ...]) -> None:
        if len(meta) != 2:
            raise TypeError(
                f"DataHealpix needs its depth and cell_ids to be rebuilt, got meta={meta!r}: "
                "pass the _meta() of a DataHealpix, data._meta(), to _from_parts."
            )
        depth, cell_ids = meta
        object.__setattr__(self, "_depth", depth)
        object.__setattr__(self, "_cell_ids", cell_ids)
        # pbc follows cell_ids, whatever was passed: True on the full sky only (M-DATA-HPX).
        object.__setattr__(self, "_pbc", cell_ids is None)

    def _rebuild_advice(self) -> str:
        return "DataHealpix(array, mask=data.mask, depth=data.depth, cell_ids=data.cell_ids)"

    @classmethod
    def _meta_content(cls, meta: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        """``depth`` as a JSON parameter, and ``cell_ids`` as an int64 array on a partial sky."""
        depth, cell_ids = meta
        arrays = {} if cell_ids is None else {"cell_ids": np.asarray(cell_ids.cell_ids)}
        return {"depth": int(depth)}, arrays

    @classmethod
    def _example_from_content(
        cls,
        n_channels: int,
        spatial_shape: tuple[int, ...],
        mask: NDArray[np.bool_] | None,
        pbc: bool,
        content: dict[str, Any],
        arrays: dict[str, NDArray[Any]],
    ) -> Data:
        """The example of the file built by the constructor, from its ``depth`` and its
        ``cell_ids``, which it checks; ``pbc`` must follow them (M-DATA-HPX)."""
        example = cls(
            np.zeros((1, n_channels, *spatial_shape)),
            mask=mask,
            depth=content["depth"],
            cell_ids=arrays.get("cell_ids"),
        )
        if example.pbc != pbc:
            raise ValueError(
                f"the file describes HEALPix data with pbc={pbc}, but its cell_ids give "
                f"pbc={example.pbc} (True on the full sky only, M-DATA-HPX): the file is "
                "inconsistent; recompute the statistics and save them again."
            )
        return example

    # ------------------------------------------------------------------ accessors

    @property
    def pbc(self) -> bool:
        """True on a full sky, False on the partial sky of :attr:`cell_ids` (M-DATA-HPX)."""
        return cast(bool, self._pbc)

    @property
    def depth(self) -> int:
        """log₂ ``nside`` of the map at full resolution."""
        return self._depth

    @property
    def cell_ids(self) -> NDArray[np.int64] | None:
        """The NESTED pixels of a partial sky, a read-only int64 array, ascending; ``None`` on
        the full sky."""
        return None if self._cell_ids is None else self._cell_ids.cell_ids

    def _n_pixels(self, level: int) -> int:
        """The number of pixels at ``level``: 12 · 4^(depth - level) on the full sky, the
        ancestors of :attr:`cell_ids` on a partial sky."""
        valid = isinstance(level, Integral) and not isinstance(level, bool)
        if not valid or not 0 <= level <= self._depth:
            raise ValueError(
                f"DataHealpix: level={level!r}; the levels of a map of depth {self._depth} are "
                f"0 … {self._depth}."
            )
        if self._cell_ids is None:
            return int(12 * 4 ** (self._depth - int(level)))
        return self._cell_ids.level_sizes[int(level)]

    # ------------------------------------------------------------------ geometry hooks

    def _conv(self, x: Any, tables: Any, level: int) -> Any:
        """The convolution of maths.md M-CONV-HPX at ``level``, with the
        :class:`HealpixConvTables` of that level: ``(Nb, Nc, Npix)`` real or complex to
        ``(Nb, L, Nc, Npix)`` complex. A complex map is convolved part by part, two real
        applications; a real one, one (Part I, §3.1, rule 6)."""
        owner = "DataHealpix._conv"
        self._check_array(x, level, "_conv")
        if not isinstance(tables, HealpixConvTables):
            raise TypeError(
                f"{owner}: pass the HealpixConvTables of the level, built by the wavelet "
                f"operator; got {type(tables).__name__}."
            )
        backend = _registry.backend_named(self._backend)
        xp = backend.xp
        everything = slice(0, int(x.shape[-1]))
        if _is_complex(xp, x):
            re = _stencil_rows(backend, xp.real(x), tables, everything)
            im = _stencil_rows(backend, xp.imag(x), tables, everything)
            pairs = xp.stack(
                [re[..., 0] - im[..., 1], re[..., 1] + im[..., 0]], axis=-1
            )  # (Nb, Nc, L, Npix, 2)
        else:
            pairs = _stencil_rows(backend, x, tables, everything)
        # (Nb, Nc, L, Npix, 2) -> (Nb, L, Nc, Npix, 2): the orientation after the batch axis.
        return backend.real_pairs_to_complex(xp.permute_dims(pairs, (0, 2, 1, 3, 4)))

    def _downsample(self, x: Any, weights: DownsampleWeights, level: int) -> Any:
        """One step of maths.md M-DOWN-HPX, from ``level`` to ``level + 1``, with the
        :class:`HealpixDownTables` of the step in ``weights.kernel``; normalised as M-DOWN-MASK
        with missing values: invalid input pixels set to 0 (never read, OP-27), then the
        reweighting 1 / (1 - f) of the output (M-MASK-HPX). A complex map is decimated part by
        part."""
        owner = "DataHealpix._downsample"
        self._check_array(x, level, "_downsample")
        tables, valid, reweight = weights
        if not isinstance(tables, HealpixDownTables):
            raise TypeError(
                f"{owner}: pass the HealpixDownTables of the step in DownsampleWeights.kernel, "
                f"built by the wavelet operator; got {type(tables).__name__}."
            )
        if (valid is None) != (reweight is None):
            raise ValueError(
                f"{owner}: pass both the valid input pixels and the reweighting map (data with "
                "missing values), or neither."
            )
        backend = _registry.backend_named(self._backend)
        xp = backend.xp
        if valid is not None:
            x = xp.where(valid, x, xp.zeros_like(x))
        n_out = self._n_pixels(level + 1)
        entries = slice(0, int(tables.segment_ids.shape[0]))

        def step(a: Any) -> Any:
            return _decimation_rows(backend, a, tables, entries, 0, n_out)

        if _is_complex(xp, x):
            out = backend.real_pairs_to_complex(
                xp.stack([step(xp.real(x)), step(xp.imag(x))], axis=-1)
            )
        else:
            out = step(x)
        return out if reweight is None else out * reweight

    def _crop_weights(self, level: int, border: int) -> NDArray[np.float64]:
        """No border on HEALPix (maths.md M-MASK-HPX, OP-50): 1 on every pixel of ``level``,
        whatever ``border``."""
        return np.ones(self._n_pixels(level), dtype=np.float64)

    # The hooks HEALPix does not have in ESTEL 1.0.0 (issue #166, decision 7).

    def _lowpass(self, x: Any, plan: Any, block: int) -> Any:
        """Not in 1.0.0: the low-pass filter serves the blocks of tensor products only
        (maths.md M-LP), and HEALPix is not a block (M-DATA-TP)."""
        raise NotImplementedError(
            "DataHealpix has no low-pass filter in ESTEL 1.0.0: it serves the blocks of tensor "
            "products only (maths.md M-LP), and the blocks are Data1D and Data2D (M-DATA-TP)."
        )

    def _nyquist_filter(self, x: Any) -> Any:
        """Not in 1.0.0: HEALPix fields are synthesised without a pre-filter (maths.md
        M-SYN-NYQ, OP-77)."""
        raise NotImplementedError(
            "DataHealpix has no Nyquist pre-filter in ESTEL 1.0.0: HEALPix fields are "
            "synthesised without it (maths.md M-SYN-NYQ, OP-77)."
        )

    def _spectrum_geometry(self) -> Any:
        """Not in 1.0.0: the angular spectrum of HEALPix data comes later (LF24)."""
        raise NotImplementedError(
            "DataHealpix has no spectrum geometry in ESTEL 1.0.0: the angular spectrum of "
            "HEALPix data comes later (LF24); compute the statistics with compute_PS=False."
        )

    def _check_array(self, x: Any, level: int, hook: str) -> None:
        """``x`` is an array of this backend, ``(Nb, Nc, Npix)`` at ``level``."""
        _registry.check_backend(self._backend, x, owner=f"DataHealpix.{hook}")
        expected = self._n_pixels(level)
        if len(x.shape) != 3 or int(x.shape[-1]) != expected:
            raise ValueError(
                f"DataHealpix.{hook}: the array has shape {tuple(x.shape)}; pass (Nb, Nc, Npix) "
                f"with the {expected} pixels of level {level}."
            )

    def __repr__(self) -> str:
        n_missing = 0 if self._mask is None else int(self._mask.sum())
        sky = "full sky" if self._cell_ids is None else f"{self._cell_ids.cell_ids.size} cells"
        return (
            f"DataHealpix(shape={self.shape}, depth={self._depth}, {sky}, "
            f"missing={n_missing}, backend={self._backend!r})"
        )


# ---------------------------------------------------------------------- application of tables


def _stencil_rows(backend: Any, x: Any, tables: HealpixConvTables, pixels: slice) -> Any:
    """The convolution of the real map ``x``, ``(..., Npix)``, at the output pixels ``pixels``
    of the level, as real pairs ``(..., L, n, 2)``, n the number of these pixels (M-CONV-HPX,
    step 4): the stencil values interpolated with ``gather_weighted``, then one real product
    with the kernel's real and imaginary parts. One function of a range of output pixels, so
    that pixel batches (T5.7) can call it on parts."""
    xp = backend.xp
    index, weight = tables.index[:, pixels], tables.weight[:, pixels]
    n_l, n, q, m = (int(s) for s in index.shape)
    z = backend.gather_weighted(
        x, -1, xp.reshape(index, (n_l * n, q, m)), xp.reshape(weight, (n_l * n, q, m))
    )  # (..., L · n, K²)
    y = xp.matmul(z, tables.kernel)  # (..., L · n, 2)
    return xp.reshape(y, (*tuple(int(s) for s in y.shape[:-2]), n_l, n, 2))


def _decimation_rows(
    backend: Any,
    x: Any,
    tables: HealpixDownTables,
    entries: slice,
    first_segment: int,
    num_segments: int,
) -> Any:
    """The decimation of the real map ``x``, ``(..., Npix)``, at ``num_segments`` coarse
    pixels from ``first_segment`` on, whose entries of the CSR tables are ``entries``
    (M-DOWN-HPX): ``gather_weighted`` then ``segment_sum``. One function of a range of output
    pixels, so that pixel batches (T5.7) can call it on parts."""
    z = backend.gather_weighted(x, -1, tables.index[entries], tables.weight[entries])[..., 0]
    segments = tables.segment_ids[entries]
    if first_segment:
        segments = segments - first_segment
    return backend.segment_sum(z, -1, segments, num_segments)


# ---------------------------------------------------------------------- validation


def check_depth(depth: object, *, minimum: int = 0, owner: str = "") -> int:
    """``depth``, log₂ ``nside``, as an int ≥ ``minimum``: the one check of a depth, for the
    data and for the tables of ``estel.wavelets._healpix_geometry``. ``owner`` prefixes the
    messages (``"DataHealpix: "``)."""
    if isinstance(depth, bool) or not isinstance(depth, Integral):
        raise TypeError(
            f"{owner}depth={depth!r}: the depth is an integer, log2(nside), such as depth=4 for "
            "nside 16."
        )
    value = int(depth)
    if value < minimum:
        raise ValueError(
            f"{owner}depth={value}: the depth must be at least {minimum} here"
            + (" (a decimation goes from depth to depth - 1)." if minimum else " (nside 1).")
        )
    return value


def check_cell_ids(cell_ids: Any, depth: int, *, owner: str = "") -> NDArray[np.int64]:
    """``cell_ids`` as int64: a non-empty 1D array of integers within the 12 · 4^depth pixels of
    ``depth``, strictly increasing (maths.md OP-90). The one check of NESTED pixel indices, for
    the data and for the tables of ``estel.wavelets._healpix_geometry``. ``owner`` prefixes the
    messages."""
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError(
            f"{owner}cell_ids has shape {ids.shape}: give a non-empty 1D array of NESTED pixel "
            "indices, or None for the full sky."
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(
            f"{owner}cell_ids has dtype {ids.dtype}: give integer NESTED pixel indices, for "
            "example cell_ids.astype(np.int64)."
        )
    ids = ids.astype(np.int64)
    total = 12 * 4**depth
    if ids.min() < 0 or ids.max() >= total:
        raise ValueError(
            f"{owner}cell_ids holds pixels outside [0, {total}), the pixels of depth {depth} "
            f"(nside {1 << depth}): check depth and the NESTED indices."
        )
    if np.any(np.diff(ids) <= 0):
        raise ValueError(
            f"{owner}cell_ids must be strictly increasing (maths.md OP-90). Sort the pixels and "
            "the data together: order = np.argsort(cell_ids); cell_ids, array = "
            "cell_ids[order], array[..., order]; and remove repeated pixels."
        )
    return cast(NDArray[np.int64], ids)


def _partial_sky(cell_ids: Any, depth: object, n_pix: int) -> tuple[int, NDArray[np.int64]]:
    """The depth and the pixels of a partial sky of ``n_pix`` pixels: ``depth`` required
    (OP-91), ``cell_ids`` checked (OP-90), one per pixel, and not the whole sphere, which is a
    full sky given without ``cell_ids`` (rule A6). The whole sphere is tested last, so that
    a wrong depth or an unsorted sphere get their own message."""
    if depth is None:
        raise ValueError(
            "DataHealpix(cell_ids=...) needs depth, log2(nside) of the map: on a partial sky it "
            "cannot be read from the number of pixels (maths.md OP-91). Pass depth=4 for "
            "nside 16, for example."
        )
    owner = "DataHealpix: "
    found = check_depth(depth, owner=owner)
    ids = check_cell_ids(cell_ids, found, owner=owner)
    if ids.size != n_pix:
        raise ValueError(
            f"{owner}cell_ids has {ids.size} entries, but the array has {n_pix} pixels; pass "
            f"one NESTED index per pixel, shape ({n_pix},)."
        )
    total = 12 * 4**found
    if ids.size == total:
        raise ValueError(
            f"{owner}cell_ids has {total} entries, the whole sphere at depth {found} "
            f"(nside {1 << found}). A full sky is given without cell_ids: omit cell_ids (and "
            "depth, which is read from the number of pixels); cell_ids are for a partial sky."
        )
    return found, ids


def _full_sky_depth(n_pix: int, depth: object) -> int:
    """The depth of a full sky of ``n_pix`` pixels, 12 · 4^depth, checked against ``depth``
    when it is given (maths.md M-DATA-HPX, OP-91)."""
    quarter = n_pix // 12
    found = (quarter.bit_length() - 1) // 2
    if n_pix % 12 or quarter != 4**found:
        raise ValueError(
            f"DataHealpix: {n_pix} pixels are not a full sky, 12 · 4^depth pixels (12, 48, 192, "
            "768, …). For a partial sky, pass cell_ids, the NESTED indices of the pixels, and "
            "depth (maths.md M-DATA-HPX, OP-91)."
        )
    if depth is not None and check_depth(depth, owner="DataHealpix: ") != found:
        raise ValueError(
            f"DataHealpix(depth={depth}): the {n_pix} pixels are a full sky of depth {found} "
            f"(nside {1 << found}). Pass depth={found} or None, or cell_ids for a partial sky."
        )
    return found
