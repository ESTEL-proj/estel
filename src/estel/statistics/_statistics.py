"""``STStatistics``: the scattering statistics and their accessors (Part II, §12.7; maths.md
§1.8; Part III, §19; tasks T2.5, T2.15).

An ``STStatistics`` holds the statistics in the compact layout (Part III, §19.2, §19.4;
``compact.py``): one real buffer ``(Nb, n_floats)``, every coefficient STL computes and only
those, real entries as one float and complex entries as (re, im), with its static
:class:`compact.Layout` (for each family its named axes, its coordinates, and the position of
every entry). The normalisation reference (``_normalization._NormReference``) and, with
``standardize``, the mean and standard deviation removed from the data are kept with it. It is
immutable: every method returns a new object.

The accessors do not depend on this storage (Part II, §12.7; ``docs/agents/storage.md``):
``full_grid`` builds STL's full-grid view on demand, ``select`` restricts the statistics (a
static gather on the buffer), ``to_vector`` flattens them for losses, and ``reduce`` applies the
reductions ``iso``, ``angular_fourier`` and ``scale_cosine``, static linear maps on the buffer
(Part III, §19.6).

It is a pytree on every backend (Part I, §3.1, rule 9; T2.9): the buffer is its leaf, with the
arrays of the reference and of the standardisation; the layout, the plan and the reductions
applied are static.
"""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import geometry_class
from ..io import _schema
from ..scattering._plan import STPlan, build_plan
from . import _layout as compact
from ._arrays import constant
from ._grid import REAL_PART_ONLY, FamilyGrid
from ._layout import FamilyValues, Gather, Layout
from ._normalization import _NormReference
from ._reductions import ORIENTATION_AXES, SCALE_AXES
from ._reductions import angular_fourier as reduce_angular_fourier
from ._reductions import iso as reduce_iso
from ._reductions import scale_cosine as reduce_scale_cosine


class _Reductions(NamedTuple):
    """The reductions applied by ``reduce``, in the order of M-NORM ("Order of operations")."""

    iso: bool = False
    """Whether ``iso`` was applied."""
    angular_fourier: int | None = None
    """The harmonics kept by ``angular_fourier``, or ``None``."""
    scale_cosine: tuple[int, int] | None = None
    """``(harmonics, dj)`` of ``scale_cosine``, or ``None``."""


_NO_REDUCTION = _Reductions()


class STStatistics:
    """Scattering statistics: what ``STOperator.apply`` and ``estel.functional.scattering``
    return (Part II, §12.7).

    The families are ``"mean"``, ``"var"``, ``"S1"``, ``"S2"``, ``"S3"`` and ``"S4"`` (maths.md
    M-MOM, M-S1 to M-S4), each with the batch as its first axis. Read them with
    :meth:`full_grid` and :meth:`select`, flatten them with :meth:`to_vector`, reduce them with
    :meth:`reduce`, and store them with :meth:`save` and :meth:`load`. Every method returns a
    new object; nothing is modified.

    The named axes, after the batch axis, are ``(c,)`` for mean and var, ``(c1, c2, j, l)`` for
    S1 and S2, ``(c1, c2, j2, j3, l2, l3)`` for S3 and ``(c1, c2, j1, j2, j3, l1, l2, l3)`` for
    S4. ``iso`` turns the orientations of S3 into ``dl`` and those of S4 into ``dl2, dl3``;
    ``angular_fourier`` prefixes the orientation axes it transforms with ``h``; ``scale_cosine``
    turns the scales of S3 into ``(k, a)`` and those of S4 into ``(k, a, b)``.
    """

    __slots__ = (
        "_backend",
        "_buffer",
        "_layout",
        "_plan",
        "_reductions",
        "_reference",
        "_standardization",
    )

    _backend: str
    _buffer: Any
    _layout: Layout
    _plan: STPlan
    _reductions: _Reductions
    _reference: _NormReference | None
    _standardization: tuple[Any, Any] | None

    def __init__(self) -> None:
        raise TypeError(
            "STStatistics are computed, not built: use STOperator(data, ...).apply(data), "
            "estel.functional.scattering(data, plan, filters), or STStatistics.load(path)."
        )

    @classmethod
    def _create(
        cls,
        buffer: Any,
        layout: Layout,
        plan: STPlan,
        backend: str,
        *,
        reductions: _Reductions = _NO_REDUCTION,
        reference: _NormReference | None = None,
        standardization: tuple[Any, Any] | None = None,
    ) -> STStatistics:
        """A new object from parts already validated: no conversion, so that it works inside
        traced code."""
        new = cls.__new__(cls)
        for name, value in (
            ("_buffer", buffer),
            ("_layout", layout),
            ("_plan", plan),
            ("_backend", backend),
            ("_reductions", reductions),
            ("_reference", reference),
            ("_standardization", standardization),
        ):
            object.__setattr__(new, name, value)
        return new

    def _replace(self, buffer: Any, layout: Layout, **changes: Any) -> STStatistics:
        parts: dict[str, Any] = {
            "reductions": self._reductions,
            "reference": self._reference,
            "standardization": self._standardization,
        }
        return STStatistics._create(
            buffer, layout, self._plan, self._backend, **{**parts, **changes}
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "STStatistics are immutable: every method returns new statistics (select, reduce)."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("STStatistics are immutable: their attributes stay set.")

    @property
    def _grids(self) -> tuple[FamilyGrid, ...]:
        """The full-grid view of each family (named axes, coordinates, where coefficients
        exist), in order: static, from the layout."""
        return self._layout.grids

    # ------------------------------------------------------------------ reading

    def full_grid(self, family: str) -> Any:
        """The tensor of ``family`` (``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``) in its full
        grid shape, NaN where no coefficient exists (Part II, §12.7).

        Without reduction or selection the shapes are: ``(Nb, Nc)`` for mean and var,
        ``(Nb, Nc, Nc, J, L)`` for S1 and S2, ``(Nb, Nc, Nc, J, J, L, L)`` for S3 and
        ``(Nb, Nc, Nc, J, J, J, L, L, L)`` for S4; mean and var are real, S1 to S4 complex, the
        coefficients that are real by construction with an imaginary part 0. The view is built
        on demand from the compact storage (Part III, §19.5).
        """
        grid = self._grid(family)
        return compact.full_grid(self._backend_object(), self._buffer, self._layout, grid.name)

    def select(self, family: str | Sequence[str] | None = None, **index: Any) -> STStatistics:
        """The statistics restricted to ``family`` and along the named axes of ``index``
        (Part II, §12.7).

        ``family`` is a family name (``"S3"``) or a sequence of names; ``None`` keeps every
        family. Each keyword of ``index`` is an axis name with an integer, which drops the axis,
        or a slice or a sequence of integers, which keeps it; the integers are positions along
        the axis as it is now. ``channel=(a, b)`` is shorthand for ``c1=a, c2=b``. A family that
        does not have every named axis is left out of the result: ``select(j3=2)`` returns S3
        and S4 restricted to j3 = 2, without mean, var, S1 or S2. An axis that no selected
        family has raises an error listing the valid names.
        """
        names = self._family_names(family)
        index = _expand_channel(index)
        chosen = [g for g in self._grids if g.name in names]
        valid = sorted({a for g in chosen for a in g.axes})
        for axis in index:
            if axis not in valid:
                raise ValueError(
                    f"select({axis}=...): no selected family ({', '.join(names)}) has an axis "
                    f"{axis!r}; valid axis names: {', '.join(valid)} (channel=(a, b) stands for "
                    "c1, c2)."
                )
        kept = [g for g in chosen if all(a in g.axes for a in index)]
        if not kept:
            raise ValueError(
                f"select: no selected family has every axis of {sorted(index)}: "
                + "; ".join(f"{g.name} has {', '.join(g.axes)}" for g in chosen)
                + "."
            )
        specs = [_select_family(self._layout.entries(g.name), index) for g in kept]
        transform = compact.make_map(self._layout, specs)
        return self._replace(*_mapped(self._backend_object(), self._buffer, transform))

    def to_vector(self, mask_st: Any = None) -> Any:
        """The statistics as one flat real vector, for losses (Part II, §12.7; maths.md
        M-FLAT).

        Families in the order mean, var, S1, S2, S3, S4; within each, every batch element in
        turn, then the coefficients in the order of the full grid, without the positions where
        no coefficient exists. Complex coefficients give their real and imaginary parts, one
        after the other, except S1 and S2, of which only the real part is kept (OP-37: the
        imaginary part of the cross-channel terms is dropped). Mean and var give one real value
        each.

        ``mask_st`` is an optional boolean array of the length of this vector:
        True keeps a coefficient. It drops coefficients found unreliable by a boolean test; in
        a synthesis it is computed once on the target and reused at every iteration.
        """
        backend = self._backend_object()
        xp = backend.xp
        extended = compact.with_zero(backend, self._buffer)
        parts = [
            xp.reshape(
                xp.take(extended, constant(backend, floats, extended, "int32"), axis=1), (-1,)
            )
            for floats in _vector_floats(self._layout)
        ]
        vector = xp.concat(parts, axis=0)
        if mask_st is None:
            return vector
        keep = _mask_positions(mask_st, int(vector.shape[0]))
        return xp.take(vector, constant(backend, keep, vector, "int32"), axis=0)

    def _to_complex_vector(
        self,
        *,
        keep_batch_dim: bool = False,
        mean_along_batch: bool = False,
        weights: dict[str, float] | None = None,
    ) -> Any:
        """The complex form of M-FLAT (STL's ``to_flatten(flatten_complex=False)``), the vector
        of the synthesis loss (M-SYN-LOSS; Part IV, §22, "The vector of the synthesis loss").

        Internal: :meth:`to_vector` stays the real form. Families in the order mean, var, S1 to
        S4; within each, the coefficients in the order of the full grid, without the positions
        where no coefficient exists; mean and var cast to complex, every complex coefficient
        kept whole. ``mean_along_batch`` first averages every family over the batch (one
        row); ``keep_batch_dim`` returns ``(Nb, n)`` (``(1, n)`` with ``mean_along_batch``)
        instead of one flat vector, in which each family holds every batch element in turn.
        ``weights`` multiplies
        the families it names (STL's ``reweight``: ``{"S3": 3.5, "S4": 12.25}``) before the
        flattening.
        """
        backend = self._backend_object()
        xp = backend.xp
        weights = {} if weights is None else weights
        unknown = sorted(set(weights) - {g.name for g in self._grids})
        if unknown:
            raise ValueError(
                f"weights names {unknown}, which these statistics do not hold; name families "
                f"among {[g.name for g in self._grids]}."
            )
        extended = compact.with_zero(backend, self._buffer)
        parts: list[Any] = []
        for grid, (re, im) in zip(self._grids, _complex_floats(self._layout), strict=True):
            pairs = [
                xp.take(extended, constant(backend, floats, extended, "int32"), axis=1)
                for floats in (re, im)
            ]
            x = backend.real_pairs_to_complex(xp.stack(pairs, axis=-1))
            if grid.name in weights:
                x = x * weights[grid.name]
            if mean_along_batch:
                x = xp.mean(x, axis=0, keepdims=True)
            parts.append(x if keep_batch_dim else xp.reshape(x, (-1,)))
        return xp.concat(parts, axis=1 if keep_batch_dim else 0)

    # ------------------------------------------------------------------ reductions

    def reduce(
        self,
        *,
        iso: bool = False,
        angular_fourier: bool = False,
        angular_harmonics: int | None = None,
        scale_cosine: bool = False,
        scale_harmonics: int | None = None,
        dj: int | None = None,
    ) -> STStatistics:
        """The statistics reduced by the selected reductions, applied in the order of maths.md
        M-NORM: ``iso``, then ``angular_fourier``, then ``scale_cosine``. Returns new statistics.

        * ``iso`` (M-RED-ISO): S1 and S2 averaged over the orientations, S3 and S4 averaged
          over the first orientation at fixed orientation differences modulo L.
        * ``angular_fourier`` (M-RED-ANG): an orthonormal DFT over the
          orientation axes of S3 and S4 (over the orientation differences after ``iso``),
          keeping the first ``angular_harmonics`` harmonics, between 1 and L; all L by default
          (OP-17). It is allowed without ``iso``.
        * ``scale_cosine`` (M-RED-SCALE), which needs ``iso``:
          the scales of S3 and S4 reparametrised as ``(j1, a, b)`` with ``j2 = j1 + a``,
          ``j3 = j1 + b`` and ``0 ≤ a ≤ b < dj``, an orthonormal DCT-II along j1, the first
          ``scale_harmonics`` harmonics kept. ``scale_harmonics`` and ``dj`` lie between 1 and
          J, and are J by default.

        Reduce once, with every reduction wanted: reduced statistics are not reduced again.
        """
        state = self._reductions
        if state != _NO_REDUCTION:
            raise ValueError(
                f"reduce(): these statistics were already reduced ({_applied(state)}); call "
                "reduce once, on statistics that were not reduced, with every reduction wanted, "
                "for example reduce(iso=True, angular_fourier=True, scale_cosine=True)."
            )
        for name, flag in (
            ("iso", iso),
            ("angular_fourier", angular_fourier),
            ("scale_cosine", scale_cosine),
        ):
            if not isinstance(flag, bool | np.bool_):
                raise TypeError(f"reduce({name}={flag!r}): pass True or False.")
        for name, value, transform, selected in (
            ("angular_harmonics", angular_harmonics, "angular_fourier", angular_fourier),
            ("scale_harmonics", scale_harmonics, "scale_cosine", scale_cosine),
            ("dj", dj, "scale_cosine", scale_cosine),
        ):
            if value is not None and not selected:
                raise ValueError(
                    f"reduce({name}={value!r}) without {transform}=True: pass {transform}=True "
                    f"as well, or leave {name} out."
                )
        if not (iso or angular_fourier or scale_cosine):
            raise ValueError(
                "reduce(): no reduction selected; pass iso=True, angular_fourier=True or "
                "scale_cosine=True (with iso=True), or use the statistics as they are."
            )
        if scale_cosine and not iso:
            raise ValueError(
                "reduce(scale_cosine=True) without iso=True: the scale cosine transform is "
                "defined on isotropic statistics (maths.md M-RED-SCALE); pass iso=True as well."
            )
        L, J = self._plan.L, self._plan.J  # noqa: N806
        n_angular = _check_count("angular_harmonics", angular_harmonics, L, "L")
        n_scale = _check_count("scale_harmonics", scale_harmonics, J, "J")
        width = _check_count("dj", dj, J, "J")
        if iso:
            self._check_complete(ORIENTATION_AXES, L, "iso", "orientation")
        elif angular_fourier:
            self._check_complete(
                {k: v for k, v in ORIENTATION_AXES.items() if k in ("S3", "S4")},
                L,
                "angular_fourier",
                "orientation",
            )
        if scale_cosine:
            scales = {k: v[0] for k, v in SCALE_AXES.items()}
            self._check_complete(scales, J, "scale_cosine", "scale")

        backend = self._backend_object()
        buffer, layout = self._buffer, self._layout
        if iso:
            buffer, layout = _mapped(backend, buffer, reduce_iso(layout, L))
        if angular_fourier:
            buffer, layout = _mapped(
                backend, buffer, reduce_angular_fourier(layout, n_angular, iso)
            )
        if scale_cosine:
            buffer, layout = _mapped(
                backend, buffer, reduce_scale_cosine(layout, J, n_scale, width)
            )
        reductions = _Reductions(
            iso=iso,
            angular_fourier=n_angular if angular_fourier else None,
            scale_cosine=(n_scale, width) if scale_cosine else None,
        )
        return self._replace(buffer, layout, reductions=reductions)

    # ------------------------------------------------------------------ storage

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the statistics to ``path``, an ``.npz`` archive in the versioned format of
        ``estel.io``: the arrays, and a JSON document with the schema version, the reference
        counters, the plan's parameters, the layout and the reductions applied. Read it back
        with :meth:`STStatistics.load`."""
        arrays, content = self._payload()
        _schema.write(path, _schema.STATISTICS_FORMAT, arrays, content)

    @staticmethod
    def load(path: str | os.PathLike[str]) -> STStatistics:
        """Read statistics written by :meth:`save`, on the current backend, in the dtype they
        were saved in. The plan is rebuilt from the saved parameters."""
        arrays, content = _schema.read(path, _schema.STATISTICS_FORMAT)
        return _from_payload(arrays, content)

    # ------------------------------------------------------------------ internals

    def _backend_object(self) -> Backend:
        return _registry.backend_named(self._backend)

    def _family_names(self, family: str | Sequence[str] | None) -> tuple[str, ...]:
        present = tuple(g.name for g in self._grids)
        if family is None:
            return present
        wanted = (family,) if isinstance(family, str) else tuple(family)
        for name in wanted:
            if name not in present:
                raise ValueError(
                    f"family {name!r}: these statistics hold {', '.join(present)}; pass one of "
                    "those, or a sequence of them."
                )
        return wanted

    def _grid(self, family: str) -> FamilyGrid:
        for grid in self._grids:
            if grid.name == family:
                return grid
        present = ", ".join(g.name for g in self._grids)
        raise ValueError(f"family {family!r}: these statistics hold {present}; pass one of those.")

    def _check_complete(
        self, axes: dict[str, tuple[str, ...]], size: int, method: str, kind: str
    ) -> None:
        for grid in self._grids:
            if grid.name in axes and not grid.is_complete(axes[grid.name], size):
                raise ValueError(
                    f"reduce({method}=True): {grid.name} lacks some of its {kind} axes "
                    f"{', '.join(axes[grid.name])} after select; apply the reductions before "
                    "selecting along those axes."
                )

    def _payload(self) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        backend = self._backend_object()
        plan = self._plan
        layout_arrays, layout_content = compact.payload(self._layout)
        arrays: dict[str, NDArray[Any]] = {
            "buffer": np.asarray(backend.to_numpy(self._buffer)),
            **{f"layout__{k}": v for k, v in layout_arrays.items()},
            "plan__cross_matrix": np.asarray(plan.cross_matrix),
        }
        if plan.mask is not None:
            arrays = {**arrays, "plan__mask": np.asarray(plan.mask)}
        if self._reference is not None:
            ref = self._reference
            arrays = {
                **arrays,
                "reference__var": np.asarray(backend.to_numpy(ref.var)),
                "reference__sigma": np.asarray(backend.to_numpy(ref.sigma)),
                "reference__present": np.asarray(ref.present),
            }
        if self._standardization is not None:
            mean, std = self._standardization
            arrays = {
                **arrays,
                "standardization__mean": np.asarray(backend.to_numpy(mean)),
                "standardization__std": np.asarray(backend.to_numpy(std)),
            }
        state = self._reductions
        content = {
            "plan": {
                "geometry": plan.geometry._geometry,
                "J": plan.J,
                "L": plan.L,
                "kernel_size": plan.kernel_size,
                "spatial_shape": list(plan.spatial_shape),
                "pbc": plan.pbc,
                "nan_threshold": plan.nan_threshold,
                "n_channels": plan.n_channels,
                "norm": plan.norm,
                "ref_batch_mean": plan.ref_batch_mean,
                "standardize": plan.standardize,
            },
            "layout": layout_content,
            "reductions": {
                "iso": state.iso,
                "angular_fourier": state.angular_fourier,
                "scale_cosine": None if state.scale_cosine is None else list(state.scale_cosine),
            },
            "reference": None
            if self._reference is None
            else {
                "ref_batch_mean": self._reference.ref_batch_mean,
                "n_orientations": self._reference.n_orientations,
            },
        }
        return arrays, content

    def _tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        reference = self._reference
        children = (
            self._buffer,
            None if reference is None else (reference.var, reference.sigma),
            self._standardization,
        )
        aux = (
            self._layout,
            self._plan,
            self._backend,
            self._reductions,
            None if reference is None else reference.static(),
        )
        return children, aux

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> STStatistics:
        layout, plan, backend, reductions, reference_static = aux
        buffer, reference_arrays, standardization = children
        reference = (
            None
            if reference_static is None
            else _NormReference.from_static(reference_static, *reference_arrays)
        )
        return cls._create(
            buffer,
            layout,
            plan,
            backend,
            reductions=reductions,
            reference=reference,
            standardization=None if standardization is None else tuple(standardization),
        )

    def __repr__(self) -> str:
        n_batch = int(self._buffer.shape[0])
        shapes = ", ".join(f"{g.name}{(n_batch, *g.shape)}" for g in self._grids)
        state = self._reductions
        return (
            f"STStatistics({shapes}; norm={self._plan.norm!r}"
            + (f", reductions={_applied(state)}" if state != _NO_REDUCTION else "")
            + f", backend={self._backend!r})"
        )


_registry.register_pytree(STStatistics, STStatistics._tree_flatten, STStatistics._tree_unflatten)


# ---------------------------------------------------------------------- static tables


def _mapped(backend: Backend, buffer: Any, transform: compact.FloatMap) -> tuple[Any, Layout]:
    """The buffer mapped by ``transform``, and its new layout."""
    return compact.apply(backend, buffer, transform), transform.layout


def _grid_order(layout: Layout, name: str) -> NDArray[np.int64]:
    """The entries of family ``name`` in the order of its full grid (C order)."""
    entries = layout.entries(name)
    order: NDArray[np.int64] = np.argsort(
        compact.flat_positions(entries.positions, entries.grid.shape), kind="stable"
    ).astype(np.int64)
    return order


@functools.lru_cache(maxsize=64)
def _vector_floats(layout: Layout) -> tuple[NDArray[np.int64], ...]:
    """For each family, the floats of :meth:`STStatistics.to_vector` in the buffer extended by
    ``compact.with_zero``: the coefficients in the order of the full grid, (re, im) for a
    complex family (the 0 for the imaginary part of a real entry), re alone for mean, var and
    the families of ``REAL_PART_ONLY`` (M-FLAT, OP-37)."""
    found: list[NDArray[np.int64]] = []
    for grid in layout.grids:
        order = _grid_order(layout, grid.name)
        re, im = (floats[order] for floats in compact.real_and_imaginary(layout, grid.name))
        if grid.is_complex and grid.name not in REAL_PART_ONLY:
            found.append(np.stack([re, im], axis=1).reshape(-1))
        else:
            found.append(re)
    return tuple(found)


@functools.lru_cache(maxsize=64)
def _complex_floats(
    layout: Layout,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    """For each family, the floats of the real and of the imaginary parts of its coefficients
    in the order of the full grid, in the buffer extended by ``compact.with_zero``."""
    return tuple(
        tuple(  # type: ignore[misc]
            floats[_grid_order(layout, grid.name)]
            for floats in compact.real_and_imaginary(layout, grid.name)
        )
        for grid in layout.grids
    )


# ---------------------------------------------------------------------- helpers


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _applied(state: _Reductions) -> str:
    """The reductions of ``state``, joined by ``+``, for messages and ``repr``."""
    applied = (
        ("iso", state.iso),
        ("angular_fourier", state.angular_fourier is not None),
        ("scale_cosine", state.scale_cosine is not None),
    )
    return "+".join(name for name, on in applied if on)


def _check_count(name: str, value: object, largest: int, symbol: str) -> int:
    if value is None:
        return largest
    if not _is_int(value) or not 1 <= int(cast(int, value)) <= largest:
        raise ValueError(
            f"reduce({name}={value!r}): pass an integer between 1 and {symbol} = {largest}, or "
            f"None for {symbol}."
        )
    return int(cast(int, value))


def _expand_channel(index: dict[str, Any]) -> dict[str, Any]:
    if "channel" not in index:
        return index
    pair = index["channel"]
    if "c1" in index or "c2" in index:
        raise ValueError("select: pass either channel=(a, b) or c1 and c2, not both.")
    if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
        raise ValueError(f"select(channel={pair!r}): pass a pair of channels, channel=(a, b).")
    rest = {k: v for k, v in index.items() if k != "channel"}
    return {**rest, "c1": pair[0], "c2": pair[1]}


def _positions(axis: str, value: Any, size: int) -> int | NDArray[np.int64]:
    """An integer position, or the positions of a slice or a sequence, along an axis of
    ``size``."""
    if _is_int(value):
        position = int(value)
        if not 0 <= position < size:
            raise IndexError(f"select({axis}={position}): the axis has {size} position(s).")
        return position
    if isinstance(value, slice):
        return np.arange(size)[value]
    found = np.asarray(value)
    if found.ndim != 1 or not (found.size == 0 or np.issubdtype(found.dtype, np.integer)):
        raise TypeError(
            f"select({axis}={value!r}): pass an integer, a slice or a sequence of integers."
        )
    if found.size and (found.min() < 0 or found.max() >= size):
        raise IndexError(f"select({axis}={value!r}): the axis has {size} position(s).")
    return found.astype(np.int64)


def _select_family(entries: compact.FamilyEntries, index: dict[str, Any]) -> Gather:
    """The selection of one family as a static gather (Part III, §19.6): which entry of the
    family each entry of the result is, with the new axes and coordinates."""
    grid = entries.grid
    chosen = {axis: _positions(axis, index[axis], grid.shape[grid.axis(axis)]) for axis in index}
    source = entries.lookup.reshape(grid.shape)  # the entry at each position, -1 where none
    coords = list(grid.coords)
    for axis, where in chosen.items():
        if isinstance(where, int):
            continue
        position = grid.axis(axis)
        source = np.take(source, where, axis=position)
        coords = [
            *coords[:position],
            tuple(grid.coords[position][int(i)] for i in where),
            *coords[position + 1 :],
        ]
    drops = sorted(
        ((grid.axis(a), w) for a, w in chosen.items() if isinstance(w, int)), reverse=True
    )
    axes = list(grid.axes)
    for position, where in drops:
        source = np.take(source, where, axis=position)
        coords = [*coords[:position], *coords[position + 1 :]]
        axes = [*axes[:position], *axes[position + 1 :]]
    present = np.asarray(source) >= 0
    positions = np.argwhere(present).astype(np.int64)
    sources = np.asarray(source)[present].astype(np.int64)
    new_grid = grid.replace(axes=tuple(axes), coords=tuple(coords), present=present)
    values = FamilyValues(
        new_grid, positions, entries.is_complex[sources], entries.levels, entries.level_ids[sources]
    )
    return Gather(values, sources)


def _mask_positions(mask_st: Any, length: int) -> NDArray[np.int64]:
    name = _registry.backend_of(mask_st)
    found = (
        np.asarray(_registry.backend_named(name).to_numpy(mask_st))
        if name is not None
        else np.asarray(mask_st)
    )
    if found.dtype != np.bool_ or found.ndim != 1:
        raise TypeError(
            f"to_vector(mask_st=...): pass a 1D boolean array, True to keep a coefficient; got "
            f"dtype {found.dtype} and shape {found.shape}."
        )
    if found.shape[0] != length:
        raise ValueError(
            f"to_vector(mask_st=...): mask_st has {found.shape[0]} entries, but the vector has "
            f"{length}: compute mask_st on statistics of the same operator, after the same "
            "reductions and selection."
        )
    return np.flatnonzero(found)


def _from_payload(arrays: dict[str, NDArray[Any]], content: dict[str, Any]) -> STStatistics:
    spec = content["plan"]
    cls = geometry_class(spec["geometry"])
    mask = arrays.get("plan__mask")
    example = cls(
        np.zeros((1, spec["n_channels"], *spec["spatial_shape"])),
        mask=None if mask is None else mask.astype(bool),
        pbc=bool(spec["pbc"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3 and OP-13 were shown when computed
        plan = build_plan(
            example,
            spec["J"],
            spec["L"],
            kernel_size=spec["kernel_size"],
            cross_matrix=arrays["plan__cross_matrix"].astype(bool),
            norm=spec["norm"],
            ref_batch_mean=spec["ref_batch_mean"],
            standardize=spec["standardize"],
        )
    if plan.nan_threshold != spec["nan_threshold"]:
        raise ValueError(
            f"the file was computed with a NaN threshold {spec['nan_threshold']}, but "
            f"{cls.__name__} now uses {plan.nan_threshold}: recompute the statistics."
        )
    backend_name = _registry.get_backend()
    backend = _registry.backend_named(backend_name)

    def to_backend(array: NDArray[Any]) -> Any:
        return backend.asarray(array, str(array.dtype), None)  # type: ignore[arg-type]

    prefix = "layout__"
    layout = compact.from_payload(
        {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)},
        content["layout"],
    )
    buffer = to_backend(arrays["buffer"])
    reference = None
    if content["reference"] is not None:
        present = arrays["reference__present"].astype(bool)
        present.flags.writeable = False
        reference = _NormReference(
            var=to_backend(arrays["reference__var"]),
            sigma=to_backend(arrays["reference__sigma"]),
            present=present,
            n_orientations=int(content["reference"]["n_orientations"]),
            ref_batch_mean=bool(content["reference"]["ref_batch_mean"]),
        )
    standardization = None
    if "standardization__mean" in arrays:
        standardization = (
            to_backend(arrays["standardization__mean"]),
            to_backend(arrays["standardization__std"]),
        )
    state = content["reductions"]
    reductions = _Reductions(
        iso=bool(state["iso"]),
        angular_fourier=state["angular_fourier"],
        scale_cosine=None if state["scale_cosine"] is None else tuple(state["scale_cosine"]),
    )
    return STStatistics._create(
        buffer,
        layout,
        plan,
        backend_name,
        reductions=reductions,
        reference=reference,
        standardization=standardization,
    )
