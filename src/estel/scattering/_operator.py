"""``STOperator``: the scattering operator of one planar domain (Part II, §12.6; tasks T2.4,
T2.5).

An operator is a frozen plan plus a pure function (Part I, §2.3): ``STOperator(data, ...)``
builds the static plan (``build_plan``) and the filters (``build_filters``), cast once to the
default dtype, and ``apply(data, ref=None)`` calls ``estel.functional.scattering``. Every option
is set once, at construction (review rule A6): the number of channels and
``cross_matrix`` too, read from the example data (issue #17). The operator records its
backend, dtype and the static metadata of the data it was built for (geometry, spatial shape,
``pbc``, mask, number of channels), and refuses other data.

**The reference.** As in STL (maths.md M-NORM), an operator built with ``norm="self"``
keeps the reference of the last statistics it computed, and replaces it, with a
``UserWarning``, at each call. The reference is internal (``_NormReference``). An operator built
with ``norm="ref"`` normalises by the reference of the ``STStatistics`` passed to
``apply(data, ref=...)``, computed with ``norm="self"``, or else by the one it holds, from
:meth:`STOperator.from_statistics`. With ``norm="self"`` the reference is part of the computation
and gradients flow through it; with ``norm="ref"`` it is a constant (LF21, closed).

**Engines.** ``engine=`` is reserved (Part II, §9.4): the operator checks the name and the
capabilities of the engine in the engine registry, which in 1.0.0 holds only ``"python"``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from .._config import FloatDType, default_device, default_dtype
from ..backend import _registry
from ..data._base import Data
from ..engines import _registry as _engines
from ..functional._scattering import check_data, compute, reference_of
from ..statistics._arrays import dtype_name
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics
from ._filters import STFilters, TensorProdFilters, build_filters, build_tensor_prod_filters
from ._plan import STPlan, TensorProdPlan, build_plan, build_tensor_prod_statistics_plan
from ._tensor_prod import tensor_prod_loop

_WIDE = ("float64", "complex128")


class STOperator:
    """The scattering operator of 1D and 2D planar data: the scattering statistics mean, var
    and S1 to S4 (maths.md §1.6 to 1.8, §2.3 to 2.4).

    Parameters
    ----------
    data:
        An example of the data, ``estel.Data1D`` or ``estel.Data2D``: its geometry, spatial
        shape, ``pbc``, mask and number of channels fix the operator; its values are not read.
    J:
        The number of scales; by default ⌊log₂ min N⌋ - 2 (M-DEF, M-DEF-1D). The spatial sizes
        must be divisible by 2^(J-1) (OP-8).
    L:
        The number of orientations; by default 4 in 2D and 1 in 1D. In 1D, L is 1 or 2 (the
        second filter is the conjugate of the first, M-WAV-1D).
    norm:
        ``"self"`` (the default): normalise the statistics by themselves, and keep the
        reference; ``"ref"``: normalise by the reference of other statistics; ``"none"``: no
        normalisation (M-NORM).
    ref_batch_mean:
        With ``norm="self"``, average the reference over the batch first; False by default
        (M-NORM).
    standardize:
        Remove the mean and divide by the standard deviation of each map before the transform
        (M-MOM); both values are kept with the statistics.
    cross_matrix:
        A boolean ``(Nc, Nc)`` array: the diagonal entry of a channel asks for its auto
        statistics, the entry ``[c1, c2]``, c1 < c2, for the cross statistics of the pair; the
        entries below the diagonal are ignored. By default every pair, the upper
        triangle (M-CHAN, M-DEF). Every channel must appear in at least one pair (OP-30).
    compute_PS:
        Whether to compute the power spectrum: True by default for 2D data, unavailable for 1D
        data in 1.0.0 (LF23). The 2D spectra come in T4.1: pass ``compute_PS=False``.
    engine:
        The engine that runs the computation (Part II, §9.4): ``"python"``, the only engine of
        1.0.0.
    """

    _plan: STPlan
    _filters: STFilters
    _backend: str
    _dtype: FloatDType
    _device: Any
    _engine: str
    _reference: _NormReference | None

    def __init__(
        self,
        data: Data,
        J: int | None = None,  # noqa: N803 (maths.md notation)
        L: int | None = None,  # noqa: N803
        *,
        norm: str = "self",
        ref_batch_mean: bool = False,
        standardize: bool = False,
        cross_matrix: Any = None,
        compute_PS: bool | None = None,  # noqa: N803 (STL's name)
        engine: str = "python",
    ) -> None:
        if not isinstance(data, Data):
            raise TypeError(
                f"STOperator takes a Data object, such as estel.Data2D, got {type(data).__name__}."
            )
        _check_compute_ps(compute_PS, data.ndim)
        plan = build_plan(
            data,
            J,
            L,
            cross_matrix=_as_numpy(cross_matrix),
            norm=norm,
            ref_batch_mean=ref_batch_mean,
            standardize=standardize,
        )
        plan.wavelets.bank()  # validates L for the geometry (M-WAV-1D)
        if norm != "none":
            _warn_channels_without_auto(plan)
        self._setup(plan, data.backend, default_dtype(), default_device(), engine, None)

    @classmethod
    def from_statistics(cls, stats: STStatistics) -> STOperator:
        """The operator that computed ``stats``: the same plan (geometry, mask, J, L, channels,
        normalisation, standardisation), on the current backend and default device, in the
        dtype of ``stats``, holding the reference ``stats`` were normalised with."""
        if not isinstance(stats, STStatistics):
            raise TypeError(
                f"STOperator.from_statistics takes STStatistics, got {type(stats).__name__}."
            )
        plan = stats._plan
        if not isinstance(plan, STPlan):
            raise ValueError(
                "STOperator.from_statistics: these are the statistics of a tensor product, which "
                "STOperator does not compute: use STOperatorTensorProd.from_statistics(stats)."
            )
        operator = cls.__new__(cls)
        operator._setup(
            plan, stats._backend, _dtype_of(stats), default_device(), "python", stats._reference
        )
        return operator

    def _setup(
        self,
        plan: STPlan,
        backend: str,
        dtype: FloatDType,
        device: Any,
        engine: str,
        reference: _NormReference | None,
    ) -> None:
        _setup(self, plan, backend, dtype, device, engine, reference, build_filters)

    # ------------------------------------------------------------------ accessors

    @property
    def plan(self) -> STPlan:
        """The static plan of the transform, for ``estel.functional.scattering``."""
        return self._plan

    @property
    def filters(self) -> STFilters:
        """The filters and the arrays derived to apply them, for
        ``estel.functional.scattering``."""
        return self._filters

    @property
    def J(self) -> int:  # noqa: N802 (maths.md notation)
        """The number of scales."""
        return self._plan.J

    @property
    def L(self) -> int:  # noqa: N802
        """The number of orientations."""
        return self._plan.L

    @property
    def norm(self) -> str:
        """The normalisation of the statistics: ``"self"``, ``"ref"`` or
        ``"none"``."""
        return self._plan.norm

    @property
    def standardize(self) -> bool:
        """Whether each map is standardised before the transform."""
        return self._plan.standardize

    # ------------------------------------------------------------------ apply

    def apply(self, data: Data, ref: STStatistics | None = None) -> STStatistics:
        """The statistics of ``data`` (maths.md §1.6 to 1.8), as ``STStatistics``.

        ``data`` must have the geometry, spatial shape, ``pbc``, mask and number of channels of
        the example the operator was built from, its backend, and its dtype; the batch size may
        differ. ``ref`` is for ``norm="ref"`` only: statistics computed with
        ``norm="self"``, whose reference normalises these; without it, the operator uses
        the reference it holds (from :meth:`from_statistics`). With ``norm="self"`` the
        operator keeps the reference of the result, replacing the one it held, with a
        ``UserWarning``.
        """
        return _apply(self, data, ref, "STOperator")

    def __repr__(self) -> str:
        plan = self._plan
        return (
            f"STOperator(J={plan.J}, L={plan.L}, norm={plan.norm!r}, "
            f"ref_batch_mean={plan.ref_batch_mean}, standardize={plan.standardize}, "
            f"spatial_shape={plan.spatial_shape}, n_channels={plan.n_channels}, "
            f"pbc={plan.pbc}, backend={self._backend!r}, dtype={self._dtype!r})"
        )


class STOperatorTensorProd:
    """The scattering operator of a tensor product of planar geometries: the scattering
    statistics mean, var and S1 to S4 with multi-index scales (maths.md §3; Part II, §12.6;
    task T3.2), STL's ``ST_Operator_TensorProd``.

    It shares the code path of :class:`STOperator`, the plan, the filters, the normalisation
    and the gather into the compact buffer, and differs by the index sets, the levels and the
    two layers of its loop (Part II, §12.6). Its statistics are stored in the compact layout of
    tensor products; their full grid is a view built on demand, for small cases only.

    Parameters
    ----------
    data:
        An example of the data, an ``estel.DataTensorProd``: its blocks, their sizes and
        ``pbc``, and its number of channels fix the operator; its values are not read.
    J:
        One number of scales per block, as a list, for example ``[4, 3]``; an entry ``None``,
        or ``J=None``, gives the default of the block, ⌊log₂ min N⁽ᵃ⁾⌋ - 2 (M-DEF-TP). The
        sizes of block a must be divisible by 2^(J_a - 1) (OP-8).
    L:
        One number of orientations per block, as a list; by default 4 for a 2D block and 2
        for a 1D block, which takes 1 or 2 (M-DATA-TP).
    lowpass:
        ``"ST"`` (the default): each block whose sizes are divisible by 2^J_a has a low-pass
        filter, its scale J_a in S1 and S2, and the scale with the low-pass filter in every
        block is centred (M-LP); a block whose sizes are not warns and has none (OP-34).
        ``None``: no low-pass filter. ESTEL 1.0.0 has no tensor-product spectrum, so STL's
        ``"PS"`` is not taken.
    norm, ref_batch_mean, standardize, cross_matrix:
        As for :class:`STOperator` (M-NORM, M-NORM-TP, M-MOM, M-CHAN).
    compute_PS:
        ``None`` (the default) or ``False``: tensor products have no power spectrum in ESTEL
        1.0.0 (LF23).
    engine:
        The engine that runs the computation (Part II, §9.4): ``"python"``, the only engine of
        1.0.0.
    """

    _plan: TensorProdPlan
    _filters: TensorProdFilters
    _backend: str
    _dtype: FloatDType
    _device: Any
    _engine: str
    _reference: _NormReference | None

    def __init__(
        self,
        data: Data,
        J: Sequence[int | None] | None = None,  # noqa: N803 (maths.md notation)
        L: Sequence[int | None] | None = None,  # noqa: N803
        *,
        lowpass: str | None = "ST",
        norm: str = "self",
        ref_batch_mean: bool = False,
        standardize: bool = False,
        cross_matrix: Any = None,
        compute_PS: bool | None = None,  # noqa: N803 (STL's name)
        engine: str = "python",
    ) -> None:
        if not isinstance(data, Data) or data._geometry != "tensor_prod":
            raise TypeError(
                f"STOperatorTensorProd takes an estel.DataTensorProd, got {type(data).__name__}; "
                "for 1D and 2D data, use STOperator."
            )
        if compute_PS is not None and not isinstance(compute_PS, bool | np.bool_):
            raise TypeError(f"compute_PS={compute_PS!r}: pass None (the default) or False.")
        if compute_PS:
            raise ValueError(
                "compute_PS=True: tensor products have no power spectrum in ESTEL 1.0.0; it "
                "comes later (LF23). Pass compute_PS=False, or leave it to None."
            )
        plan = build_tensor_prod_statistics_plan(
            data,
            J,
            L,
            lowpass=lowpass,
            cross_matrix=_as_numpy(cross_matrix),
            norm=norm,
            ref_batch_mean=ref_batch_mean,
            standardize=standardize,
        )
        for block in plan.wavelets.blocks:
            block.bank()  # validates L for the geometry of the block (M-WAV-1D)
        if norm != "none":
            _warn_channels_without_auto(plan)
        reduced = tensor_prod_loop(plan).reduced_border
        if reduced is not None:
            warnings.warn(reduced, UserWarning, stacklevel=2)  # OP-13, once for the operator
        self._setup(plan, data.backend, default_dtype(), default_device(), engine, None)

    @classmethod
    def from_statistics(cls, stats: STStatistics) -> STOperatorTensorProd:
        """The operator that computed ``stats``, statistics of a tensor product: the same
        plan (blocks, sizes, J, L, low-pass filters, channels, normalisation,
        standardisation), on the current backend and default device, in the dtype of
        ``stats``, holding the reference ``stats`` were normalised with."""
        if not isinstance(stats, STStatistics):
            raise TypeError(
                "STOperatorTensorProd.from_statistics takes STStatistics, got "
                f"{type(stats).__name__}."
            )
        plan = stats._plan
        if not isinstance(plan, TensorProdPlan):
            raise ValueError(
                "STOperatorTensorProd.from_statistics: these are the statistics of planar data: "
                "use STOperator.from_statistics(stats)."
            )
        operator = cls.__new__(cls)
        operator._setup(
            plan, stats._backend, _dtype_of(stats), default_device(), "python", stats._reference
        )
        return operator

    def _setup(
        self,
        plan: TensorProdPlan,
        backend: str,
        dtype: FloatDType,
        device: Any,
        engine: str,
        reference: _NormReference | None,
    ) -> None:
        _setup(self, plan, backend, dtype, device, engine, reference, build_tensor_prod_filters)

    # ------------------------------------------------------------------ accessors

    @property
    def plan(self) -> TensorProdPlan:
        """The static plan of the transform, for ``estel.functional.scattering``."""
        return self._plan

    @property
    def filters(self) -> TensorProdFilters:
        """The filters and the arrays derived to apply them, for
        ``estel.functional.scattering``."""
        return self._filters

    @property
    def J(self) -> tuple[int, ...]:  # noqa: N802 (maths.md notation)
        """The number of scales of each block."""
        return self.plan.J

    @property
    def L(self) -> tuple[int, ...]:  # noqa: N802
        """The number of orientations of each block."""
        return self.plan.L

    @property
    def lowpass(self) -> str | None:
        """``"ST"`` or ``None`` (maths.md M-LP)."""
        return self.plan.lowpass

    @property
    def norm(self) -> str:
        """The normalisation of the statistics: ``"self"``, ``"ref"`` or ``"none"``."""
        return self.plan.norm

    @property
    def standardize(self) -> bool:
        """Whether each map is standardised before the transform."""
        return self.plan.standardize

    # ------------------------------------------------------------------ apply

    def apply(self, data: Data, ref: STStatistics | None = None) -> STStatistics:
        """The statistics of ``data`` (maths.md §3), as ``STStatistics``, as
        :meth:`STOperator.apply`: ``data`` must have the blocks, sizes, ``pbc`` and number of
        channels of the example the operator was built from, its backend and its dtype; the
        batch size may differ. ``ref`` is for ``norm="ref"`` only, statistics of a tensor
        product with the same blocks, J, L and low-pass filters computed with
        ``norm="self"``."""
        return _apply(self, data, ref, "STOperatorTensorProd")

    def __repr__(self) -> str:
        plan = self.plan
        names = ", ".join(g.__name__ for g in plan.block_geometries)
        return (
            f"STOperatorTensorProd(J={list(plan.J)}, L={list(plan.L)}, "
            f"lowpass={plan.lowpass!r}, norm={plan.norm!r}, "
            f"ref_batch_mean={plan.ref_batch_mean}, standardize={plan.standardize}, "
            f"blocks=[{names}], spatial_shape={plan.spatial_shape}, "
            f"n_channels={plan.n_channels}, pbc={list(plan.pbc)}, backend={self._backend!r}, "
            f"dtype={self._dtype!r})"
        )


# ---------------------------------------------------------------------- shared by both operators


def _setup(
    operator: Any,
    plan: STPlan | TensorProdPlan,
    backend: str,
    dtype: FloatDType,
    device: Any,
    engine: str,
    reference: _NormReference | None,
    filters_of: Any,
) -> None:
    """The state of an operator: its plan, its filters built once on ``backend`` in
    ``dtype``, its engine checked in the registry (Part II, §9.4), and its reference."""
    capabilities = _engines.resolve(engine)
    geometry = plan.geometry._geometry
    capabilities.check(geometry=geometry, dtype=dtype)
    operator._plan = plan
    operator._backend = backend
    operator._dtype = dtype
    operator._device = device
    operator._engine = engine
    operator._filters = filters_of(plan, backend, dtype, device)
    operator._reference = reference


def _apply(operator: Any, data: Data, ref: STStatistics | None, name: str) -> STStatistics:
    """``apply`` of both operators: the reference of ``norm="ref"`` (given, or held), the pure
    function, and, with ``norm="self"``, the reference kept, replacing the one held with a
    ``UserWarning`` (maths.md M-NORM)."""
    plan = operator._plan
    check_data(data, plan, operator._filters)
    reference = reference_of(ref, plan) if ref is not None else None
    if reference is None and plan.norm == "ref":
        reference = operator._reference
        if reference is None:
            raise ValueError(
                f'{name}.apply: norm="ref" needs a reference, and this operator '
                'holds none. Pass ref=, statistics computed with norm="self", or build '
                f"the operator with {name}.from_statistics (maths.md M-NORM)."
            )
    stats = compute(data, plan, operator._filters, reference)
    if plan.norm == "self":
        if operator._reference is not None:
            warnings.warn(
                f'{name}.apply: norm="self" replaces the reference this operator '
                "held by that of the new statistics (maths.md M-NORM).",
                UserWarning,
                stacklevel=3,
            )
        operator._reference = stats._reference
    return stats


def _dtype_of(stats: STStatistics) -> FloatDType:
    return "float64" if dtype_name(stats._buffer) in _WIDE else "float32"


# ---------------------------------------------------------------------- validation


def _as_numpy(matrix: Any) -> Any:
    """``cross_matrix`` as NumPy when it is an array of a backend."""
    name = _registry.backend_of(matrix)
    return matrix if name is None else _registry.backend_named(name).to_numpy(matrix)


def _warn_channels_without_auto(plan: STPlan | TensorProdPlan) -> None:
    """STL's warning of ``norm="self"`` (maths.md M-CHAN, OP-30), given once, at construction."""
    missing = [c for c in range(plan.n_channels) if c not in plan.pairs.auto]
    if missing:
        warnings.warn(
            f"cross_matrix: channel(s) {missing} have no auto statistics, so no "
            "normalisation scale: their cross statistics are NaN after normalisation "
            "(maths.md M-CHAN, OP-30). Set their diagonal entries to normalise them.",
            UserWarning,
            stacklevel=3,
        )


def _check_compute_ps(compute_ps: object, ndim: int) -> None:
    if compute_ps is not None and not isinstance(compute_ps, bool | np.bool_):
        raise TypeError(f"compute_PS={compute_ps!r}: pass True, False or None (the default).")
    if ndim != 2:
        if compute_ps:
            raise ValueError(
                f"compute_PS=True: ESTEL 1.0.0 computes power spectra of 2D data only; for "
                f"{ndim}D data they come later (LF23). Pass compute_PS=False, or leave it to None."
            )
        return
    if compute_ps is None or compute_ps:
        raise NotImplementedError(
            "compute_PS: the 2D power spectra come in task T4.1, and compute_PS is True by "
            "default for 2D data. Pass compute_PS=False."
        )
