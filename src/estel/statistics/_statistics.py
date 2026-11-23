"""``STStatistics``: the scattering statistics and their accessors (Part II, §12.7; maths.md
§1.8; Part III, §19; tasks T2.5, T2.15).

An ``STStatistics`` holds the statistics in the compact layout (Part III, §19.2, §19.4;
``_layout.py``): one real buffer ``(Nb, n_floats)``, every coefficient STL computes and only
those, real entries as one float and complex entries as (re, im), with its static
:class:`_layout.Layout` (for each family its named axes, its coordinates, and the position of
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
from ..scattering._plan import (
    STPlan,
    TensorProdPlan,
    build_plan,
    build_tensor_prod_statistics_plan,
)
from . import _layout as compact
from ._arrays import constant, dtype_name
from ._grid import REAL_PART_ONLY, FamilyGrid, tensor_prod_auto_scales
from ._layout import Chunked, Copy, FamilyChunks, Layout, Pack, Take, Unpack
from ._normalization import (
    TensorProdBlocks,
    _NormReference,
    as_constant,
    check_tensor_prod_reference,
    normalize,
    self_reference,
    tensor_prod_blocks,
)
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

_FULL_GRID_LIMIT = 2 * 1024**3
"""The largest full-grid view of the statistics of a tensor product, in bytes, batch included,
in its dtype (Part II, §12.7; Part IV, §22, "No role for the full grid in tensor products"):
beyond it ``full_grid`` raises. Internal and temporary: removed after 1.0.0 (LF65)."""

_TENSOR_PROD_SELECT = ("c", "c1", "c2")
"""The axes ``select`` takes on the statistics of a tensor product, with ``family`` and the
shorthand ``channel``: the channels only (Part II, §12.7; naming the blocks is LF44)."""


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
    _plan: STPlan | TensorProdPlan
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
        plan: STPlan | TensorProdPlan,
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

        On the statistics of a tensor product, each scale and orientation axis is one axis per
        block (maths.md M-S-TP), and a view over 2 GiB, batch included, is refused: read the
        statistics with ``to_vector``, or ``select`` a family and channels first.
        """
        grid = self._grid(family)
        if isinstance(self._plan, TensorProdPlan):
            n_batch = int(self._buffer.shape[0])
            itemsize = int(np.dtype(dtype_name(self._buffer)).itemsize)
            size = compact.full_grid_bytes(self._layout, grid.name, n_batch, itemsize)
            if size > _FULL_GRID_LIMIT:
                raise ValueError(
                    f"full_grid({family!r}): the full-grid view of this tensor product would take "
                    f"{size / 1024**3:.2f} GiB ({n_batch} batch element(s), shape "
                    f"{grid.shape}), more than the {_FULL_GRID_LIMIT / 1024**3:.0f} GiB that "
                    "full_grid builds for a tensor product. Use to_vector(), which holds the "
                    "coefficients alone, or select(family=..., channel=(a, b)) to reduce the "
                    "view first (Part II, §12.7; LF65)."
                )
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
        if isinstance(self._plan, TensorProdPlan):
            others = sorted(a for a in index if a not in _TENSOR_PROD_SELECT)
            if others:
                raise ValueError(
                    f"select({', '.join(f'{a}=...' for a in others)}): on the statistics of a "
                    "tensor product, select takes family and the channels only (c1, c2, "
                    "channel=(a, b), and c for mean and var); naming the scale and orientation "
                    "axes of the blocks comes later (LF44). Use full_grid on a small case, or "
                    "to_vector."
                )
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
        specs = [_select_family(self._layout, g.name, index) for g in kept]
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
        tables = _vector_tables(backend, _vector_specs(self._layout), self._buffer)
        vector = _vector_function(backend.name, self._layout)(self._buffer, tables)
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
        n_batch = int(self._buffer.shape[0])
        parts: list[Any] = []
        specs = _complex_specs(self._layout)
        tables = _vector_tables(backend, specs, self._buffer)
        for grid, spec, found in zip(self._grids, specs, tables, strict=True):
            floats = _family_floats(backend, self._buffer, spec, found)
            x = backend.real_pairs_to_complex(xp.reshape(floats, (n_batch, -1, 2)))
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
        The reductions are not defined on the statistics of a tensor product: they raise.
        """
        if isinstance(self._plan, TensorProdPlan):
            raise ValueError(
                "reduce(): the reductions iso, angular_fourier and scale_cosine are written for "
                "one scale and one orientation axis per layer, and are not defined on the "
                "statistics of a tensor product in ESTEL 1.0.0 (maths.md M-NORM-TP, OP-44); "
                "their definition on products is LF26. Use the statistics as they are, with "
                "to_vector or select."
            )
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
        plan = self._plan
        assert isinstance(plan, STPlan)
        L, J = plan.L, plan.J  # noqa: N806
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
            buffer, layout = _reduced(backend, buffer, reduce_iso(layout, L))
        if angular_fourier:
            for per_axis in reduce_angular_fourier(layout, n_angular, iso):
                buffer, layout = _reduced(backend, buffer, per_axis)
        if scale_cosine:
            buffer, layout = _reduced(
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
        if isinstance(plan, TensorProdPlan):
            plan_content = _tensor_prod_content(plan)
        else:
            plan_content = {
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
            }
        arrays: dict[str, NDArray[Any]] = {
            "buffer": np.asarray(backend.to_numpy(self._buffer)),
            **{f"layout__{k}": v for k, v in layout_arrays.items()},
            "plan__cross_matrix": np.asarray(plan.cross_matrix),
        }
        if isinstance(plan, STPlan) and plan.mask is not None:
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
            "plan": plan_content,
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
                **(
                    {}
                    if self._reference.blocks is None
                    else {"blocks": _blocks_content(self._reference.blocks)}
                ),
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


def _reduced(backend: Backend, buffer: Any, transform: compact.FloatMap) -> tuple[Any, Layout]:
    """:func:`_mapped` for a map of the reductions, which are cached: compiled once per map
    where the backend compiles (``Backend.jit``; the identity on PyTorch), so that its many
    small operations on the chunks are not compiled one by one, as JAX does outside a
    compiled function. The tables of its chunks are arguments, not constants: the index the
    size of the coefficients is built inside the computation."""
    tables = compact.map_tables(backend, transform, buffer)
    return _compiled(backend.name, transform)(buffer, tables), transform.layout


@functools.lru_cache(maxsize=64)
def _compiled(backend_name: str, transform: compact.FloatMap) -> Any:
    backend = _registry.backend_named(backend_name)

    def mapped(buffer: Any, tables: compact.MapTables) -> Any:
        return compact.apply(backend, buffer, transform, tables=tables)

    return backend.jit(mapped)


VectorSpec = Unpack | tuple[tuple[Unpack, Pack], ...]
"""How :meth:`STStatistics.to_vector` reads one family: its chunks in the order of its full
grid, read as one dense array (:class:`_layout.Unpack`), or, for a family whose chunks have
several orientation shapes (tensor products), one group of chunks of one shape at a time,
each written at its place in the vector of the family (a :class:`_layout.Pack` whose floats
are the vector's)."""


def _vector_spec(layout: Layout, grid: FamilyGrid, parts: int) -> VectorSpec:
    """The :data:`VectorSpec` of ``grid`` in ``layout``, ``parts`` floats per entry."""
    order = compact.grid_order(grid)
    if grid.shapes is None:
        return compact.unpack_spec(layout, grid.name, order, parts)
    sizes = np.array(
        [int(np.prod(grid.kind_shape(int(k)), dtype=np.int64)) for k in grid.kinds[order]],
        dtype=np.int64,
    )
    starts = (np.cumsum(sizes) - sizes) * parts
    n_floats = int(sizes.sum()) * parts
    groups = []
    for shape, rows in grid.shape_groups(order):
        n_o = int(np.prod(shape, dtype=np.int64))
        write = Pack(
            kinds=np.zeros(rows.shape[0], dtype=np.int64),
            within=np.arange(n_o * parts, dtype=np.int64).reshape(1, n_o, parts),
            starts=starts[rows],
            n_floats=n_floats,
            whole=False,
        )
        groups.append((compact.unpack_spec(layout, grid.name, order[rows], parts), write))
    return tuple(groups)


@functools.lru_cache(maxsize=64)
def _vector_specs(layout: Layout) -> tuple[VectorSpec, ...]:
    """For each family, how :meth:`STStatistics.to_vector` reads it: its chunks in the order of
    its full grid, (re, im) for a complex family (0 for the imaginary part of a real entry), re
    alone for mean, var and the families of ``REAL_PART_ONLY`` (M-FLAT, OP-37). Tables of one
    row per chunk."""
    return tuple(
        _vector_spec(layout, grid, 2 if grid.is_complex and grid.name not in REAL_PART_ONLY else 1)
        for grid in layout.grids
    )


def _vector_tables(backend: Backend, specs: Sequence[VectorSpec], like: Any) -> tuple[Any, ...]:
    """The tables of the chunks of ``specs`` (``_layout.ChunkTables``), arguments of the
    compiled flattening."""

    def tables(spec: Unpack | Pack) -> compact.ChunkTables:
        return compact.chunk_tables(backend, spec.within, spec.kinds, spec.starts, like)

    return tuple(
        tables(spec)
        if isinstance(spec, Unpack)
        else tuple((tables(read), tables(write)) for read, write in spec)
        for spec in specs
    )


def _family_floats(backend: Backend, buffer: Any, spec: VectorSpec, tables: Any) -> Any:
    """The floats ``(Nb, m)`` of one family in the order of its vector (:data:`VectorSpec`):
    one gather, or, one orientation shape at a time, a gather written at its places in the
    vector, the groups summed (each float receives one value)."""
    xp = backend.xp
    if isinstance(spec, Unpack):
        dense = compact.unpack(backend, buffer, spec, tables)
        return xp.reshape(dense, (int(buffer.shape[0]), -1))
    total = None
    for (read, write), (read_tables, write_tables) in zip(spec, tables, strict=True):
        dense = compact.unpack(backend, buffer, read, read_tables)
        found = compact.pack(backend, dense, write, write_tables)
        total = found if total is None else total + found
    if total is None:
        return xp.zeros((int(buffer.shape[0]), 0), dtype=buffer.dtype)
    return total


@functools.lru_cache(maxsize=64)
def _vector_function(backend_name: str, layout: Layout) -> Any:
    """The flattening of :meth:`STStatistics.to_vector` for ``layout``, a function of the
    buffer and of the tables of the chunks (``_layout.ChunkTables``, one per family), compiled
    once per layout where the backend compiles (``Backend.jit``; the identity on PyTorch)."""
    backend = _registry.backend_named(backend_name)
    specs = _vector_specs(layout)

    def vector(buffer: Any, tables: tuple[Any, ...]) -> Any:
        xp = backend.xp
        return xp.concat(
            [
                xp.reshape(_family_floats(backend, buffer, spec, found), (-1,))
                for spec, found in zip(specs, tables, strict=True)
            ],
            axis=0,
        )

    return backend.jit(vector)


@functools.lru_cache(maxsize=64)
def _complex_specs(layout: Layout) -> tuple[VectorSpec, ...]:
    """For each family, its chunks in the order of its full grid, read as (re, im)."""
    return tuple(_vector_spec(layout, grid, 2) for grid in layout.grids)


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


def _select_family(layout: Layout, name: str, index: dict[str, Any]) -> Copy | Chunked:
    """The selection of one family as a static map (Part III, §19.6): which chunk of the family
    each chunk of the result is, then the positions kept along its orientation axes, with the
    new axes and coordinates. Computed on the chunks, with tables their size, never the size of
    the coefficients or of the full grid."""
    if not index:
        return Copy(name)
    grid = layout.grid(name)
    k = grid.n_chunk_axes
    chosen = {axis: _positions(axis, index[axis], grid.shape[grid.axis(axis)]) for axis in index}
    positions = grid.chunks.astype(np.int64)
    sources = np.arange(grid.n_chunks, dtype=np.int64)
    coords = list(grid.coords)
    orientation_axes = list(range(k, len(grid.axes)))  # those not dropped yet, in order
    steps: list[Take] = []
    for axis, where in chosen.items():
        column = grid.axis(axis)
        if column >= k:
            at = orientation_axes.index(column)
            if isinstance(where, int):
                steps.append(Take(at, np.array([where], dtype=np.int64), True))
                orientation_axes = [a for a in orientation_axes if a != column]
            else:
                steps.append(Take(at, where, False))
                coords = [
                    *coords[:column],
                    tuple(grid.coords[column][int(i)] for i in where),
                    *coords[column + 1 :],
                ]
            continue
        if isinstance(where, int):
            kept = positions[:, column] == where
            positions, sources = positions[kept], sources[kept]
            continue
        # Each chunk at position p along the axis goes to every k with where[k] == p.
        order = np.argsort(where, kind="stable")
        ordered = where[order]
        low = np.searchsorted(ordered, positions[:, column], side="left")
        counts = np.searchsorted(ordered, positions[:, column], side="right") - low
        taken = np.repeat(np.arange(positions.shape[0]), counts)
        within = np.arange(taken.shape[0]) - np.repeat(np.cumsum(counts) - counts, counts)
        new_column = order[np.repeat(low, counts) + within].astype(np.int64)
        positions = np.concatenate(
            [positions[taken, :column], new_column[:, None], positions[taken, column + 1 :]],
            axis=1,
        )
        sources = sources[taken]
        coords = [
            *coords[:column],
            tuple(grid.coords[column][int(i)] for i in where),
            *coords[column + 1 :],
        ]
    dropped = {grid.axis(a) for a, w in chosen.items() if isinstance(w, int)}
    keep = [i for i in range(len(grid.axes)) if i not in dropped]
    chunk_axes = [i for i in range(k) if i not in dropped]
    real = grid.real.reshape(-1, *grid.orientations)
    for step in steps:
        real = np.take(real, step.positions, axis=1 + step.axis)
        if step.drop:
            real = real.reshape(*real.shape[: 1 + step.axis], *real.shape[2 + step.axis :])
    new_grid = grid.replace(
        axes=tuple(grid.axes[i] for i in keep),
        coords=tuple(coords[i] for i in keep),
        n_chunk_axes=len(chunk_axes),
        chunks=positions[:, chunk_axes],
        kinds=grid.kinds[sources],
        real=real.reshape(real.shape[0], -1),
    )
    levels, level_ids = layout.levels(name)
    family = FamilyChunks(new_grid, levels, level_ids[sources])
    return Chunked(family, sources[:, None], None, tuple(steps))


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


def _tensor_prod_content(plan: TensorProdPlan) -> dict[str, Any]:
    """The parameters of the plan of a tensor product, for ``estel.io`` (schema version 5)."""
    return {
        "geometry": plan.geometry._geometry,
        "blocks": [g._geometry for g in plan.block_geometries],
        "J": list(plan.J),
        "L": list(plan.L),
        "lowpass": plan.lowpass,
        "lowpass_blocks": list(plan.lowpass_blocks),
        "kernel_size": plan.kernel_size,
        "spatial_shape": list(plan.spatial_shape),
        "pbc": list(plan.pbc),
        "n_channels": plan.n_channels,
        "norm": plan.norm,
        "ref_batch_mean": plan.ref_batch_mean,
        "standardize": plan.standardize,
    }


def _blocks_content(blocks: TensorProdBlocks) -> dict[str, Any]:
    return {
        "geometries": list(blocks.geometries),
        "J": list(blocks.J),
        "L": list(blocks.L),
        "lowpass": list(blocks.lowpass),
    }


def _tensor_prod_plan(spec: dict[str, Any], cross_matrix: NDArray[np.bool_]) -> TensorProdPlan:
    """The plan of a tensor product rebuilt from the parameters of a file: the data example
    takes the signature of ``DataTensorProd`` (blocks, no mask, one ``pbc`` per block)."""
    cls = geometry_class(spec["geometry"])
    blocks = [geometry_class(name) for name in spec["blocks"]]
    example = cast(Any, cls)(
        np.zeros((1, spec["n_channels"], *spec["spatial_shape"])),
        blocks,
        pbc=[bool(flag) for flag in spec["pbc"]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-34 was shown when computed
        plan = build_tensor_prod_statistics_plan(
            example,
            list(spec["J"]),
            list(spec["L"]),
            lowpass=spec["lowpass"],
            kernel_size=spec["kernel_size"],
            cross_matrix=cross_matrix,
            norm=spec["norm"],
            ref_batch_mean=spec["ref_batch_mean"],
            standardize=spec["standardize"],
        )
    if list(plan.lowpass_blocks) != list(spec["lowpass_blocks"]):
        raise ValueError(
            f"the file has the low-pass flags {spec['lowpass_blocks']}, but its blocks now give "
            f"{list(plan.lowpass_blocks)} (maths.md M-LP): recompute the statistics."
        )
    return plan


def _from_payload(arrays: dict[str, NDArray[Any]], content: dict[str, Any]) -> STStatistics:
    spec = content["plan"]
    cls = geometry_class(spec["geometry"])
    plan: STPlan | TensorProdPlan
    if "blocks" in spec:
        plan = _tensor_prod_plan(spec, arrays["plan__cross_matrix"].astype(bool))
    else:
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
        blocks = content["reference"].get("blocks")
        reference = _NormReference(
            var=to_backend(arrays["reference__var"]),
            sigma=to_backend(arrays["reference__sigma"]),
            present=present,
            n_orientations=int(content["reference"]["n_orientations"]),
            ref_batch_mean=bool(content["reference"]["ref_batch_mean"]),
            blocks=None
            if blocks is None
            else TensorProdBlocks(
                geometries=tuple(blocks["geometries"]),
                J=tuple(int(n) for n in blocks["J"]),
                L=tuple(int(n) for n in blocks["L"]),
                lowpass=tuple(bool(f) for f in blocks["lowpass"]),
            ),
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


# ---------------------------------------------------------------------- tensor products


def tensor_prod_statistics(
    buffer: Any,
    plan: TensorProdPlan,
    reference: _NormReference | None = None,
    standardization: tuple[Any, Any] | None = None,
) -> STStatistics:
    """The statistics of a tensor product from its raw buffer, ``(Nb, n_floats)`` in the layout
    of ``_layout.tensor_prod_layout(plan)``, normalised as the plan says (maths.md M-NORM-TP;
    task T3.6): what the loop of the operator (task T3.2) returns through. ``reference`` is the
    reference of ``norm="ref"``, from statistics of a tensor product computed with
    ``norm="self"``, checked against the plan; with ``norm="self"`` the reference is computed
    from the buffer, and gradients flow through it."""
    backend_name = _registry.backend_of(buffer)
    if backend_name is None:
        raise TypeError("tensor_prod_statistics takes a buffer of a backend, not a NumPy array.")
    backend = _registry.backend_named(backend_name)
    layout = compact.tensor_prod_layout(plan)
    if tuple(buffer.shape[1:]) != (layout.n_floats,):
        raise ValueError(
            f"the buffer has shape {tuple(buffer.shape)}, and the layout of this tensor product "
            f"holds {layout.n_floats} floats per batch element."
        )
    blocks = tensor_prod_blocks(plan)
    n_orientations = int(np.prod(plan.L, dtype=np.int64))
    if plan.norm == "self":
        if reference is not None:
            raise ValueError('norm="self" takes no reference (maths.md M-NORM).')
        reference = self_reference(
            backend,
            buffer,
            layout,
            tensor_prod_auto_scales(plan),
            n_orientations,
            plan.ref_batch_mean,
            blocks,
        )
    elif plan.norm == "ref":
        if reference is None:
            raise ValueError(
                'norm="ref" needs a reference: pass statistics of a tensor product computed '
                'with norm="self" (maths.md M-NORM).'
            )
        check_tensor_prod_reference(reference, plan, int(buffer.shape[0]), buffer.dtype)
        reference = as_constant(backend, reference)
    elif reference is not None:
        raise ValueError('norm="none" takes no reference (maths.md M-NORM).')
    if reference is not None:
        buffer, layout = normalize(backend, buffer, layout, reference)
    return STStatistics._create(
        buffer,
        layout,
        plan,
        backend_name,
        reference=reference,
        standardization=standardization,
    )
