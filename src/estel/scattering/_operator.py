"""``STOperator``: the scattering operator of one domain, 1D, 2D or HEALPix (Part II, §12.6;
tasks T2.4, T2.5, T5.6).

An operator is a frozen plan plus a pure function (Part I, §2.3): ``STOperator(data, ...)``
builds the static plan (``build_plan``) and the filters (``build_filters``), cast once to the
default dtype, and ``apply(data, ref=None, *, norm="self", ref_batch_mean=False)`` calls
``estel.functional.scattering``. Every option is set in one place only (review rule A6): the
normalisation, ``norm`` and ``ref_batch_mean``, at each call of ``apply``, so that one operator
serves every normalisation, as in STL (task T2.19); the rest at construction, the number of
channels and ``cross_matrix`` too, read from the example data (issue #17). The operator records
its backend, dtype and the static metadata of the data it was built for (geometry, spatial shape,
``pbc``, mask, number of channels), and refuses other data.

**The reference.** As in STL (maths.md M-NORM), each call of ``apply`` with ``norm="self"``
stores the reference of the statistics it computes in the operator, replacing the one it held
with a ``UserWarning``. The reference is internal (``_NormReference``). A call with
``norm="ref"`` normalises by the reference of the ``STStatistics`` passed to
``apply(data, ref=...)``, computed with ``norm="self"``, or else by the one the operator holds,
stored by an earlier call or by :meth:`STOperator.from_statistics`. With ``norm="self"`` the
reference is part of the computation and gradients flow through it; with ``norm="ref"`` it is a
constant (LF21, closed; OP-18). Operators share no state: each holds its own reference.

**Engines.** ``engine=`` is reserved (Part II, §9.4): the operator checks the name and the
capabilities of the engine in the engine registry, which in 1.0.0 holds only ``"python"``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

import numpy as np

from .._config import FloatDType, default_device, default_dtype
from ..backend import _registry
from ..data._base import Data
from ..engines import _registry as _engines
from ..functional._scattering import apply_operator
from ..statistics._arrays import dtype_name
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics
from ._filters import STFilters, STFiltersSingleDomain, STFiltersTensorProd
from ._plan import (
    STPlan,
    STPlanSingleDomain,
    STPlanTensorProd,
    build_plan_with_tables,
    build_statistics_plan_tensor_prod,
    spectrum_2d_only,
)

_WIDE = ("float64", "complex128")


class STOperator:
    """The scattering operator of single-domain data, 1D, 2D and HEALPix: the scattering
    statistics mean, var and S1 to S4 (maths.md §1.6 to 1.8, §2.3 to 2.4, §4).

    Parameters
    ----------
    data:
        An example of the data, ``estel.Data1D``, ``estel.Data2D`` or ``estel.DataHealpix``: its
        geometry, spatial shape, ``pbc``, mask and number of channels fix the operator, and on
        HEALPix its ``depth`` and ``cell_ids``; its values are not read.
    J:
        The number of scales; by default ⌊log₂ min N⌋ - 2 (M-DEF, M-DEF-1D). The spatial sizes
        must be divisible by 2^(J-1) (OP-8). On HEALPix, max(1, depth - 1), with
        1 ≤ J ≤ depth + 1 (M-DEF-HPX).
    L:
        The number of orientations; by default 4 in 2D and 1 in 1D. In 1D, L is 1 or 2 (the
        second filter is the conjugate of the first, M-WAV-1D). On HEALPix, the number of gauges,
        4 by default (M-DEF-HPX).
    standardize:
        Remove the mean and divide by the standard deviation of each map before the transform
        (M-STD), True by default (OP-120; STL: False, whose values ``standardize=False``
        gives); both values are kept with the statistics. A map of zero variance then has NaN
        statistics.
    cross_matrix:
        A boolean ``(Nc, Nc)`` array: the diagonal entry of a channel asks for its auto
        statistics, the entry ``[c1, c2]``, c1 < c2, for the cross statistics of the pair; the
        entries below the diagonal are ignored. By default every pair, the upper
        triangle (M-CHAN, M-DEF). Every channel must appear in at least one pair (OP-30).
    compute_PS:
        Whether the statistics hold the 2D power spectrum, the family ``"PS"`` (maths.md §1.9):
        True by default for 2D data, unavailable for 1D data (LF23) and HEALPix data (LF24) in
        1.0.0, where ``None`` (the default) leaves it out and True raises. It is computed
        on the field after the standardisation, with the channel pairs of ``cross_matrix``,
        over the band of the operator's J, given or by default: the centres of its scales 0 to
        J - 1 (maths.md §1.9, "Range"); it needs even sizes (OP-20) and J ≥ 2 + Jmin (OP-93).
    n_bins:
        The number of bins of the spectrum: by default max(4, round(3 (J - 1 - Jmin))).
    Jmin:
        J_min of the spectrum, 0 by default, with 0 ≤ Jmin ≤ J - 2.
    spectrum_method:
        The estimator of the spectrum: ``"auto"`` (the default) chooses it from ``pbc`` and the
        missing pixels of ``data``; or ``"fft"``, ``"pbc_mask"``, ``"nan_mask"`` or
        ``"pbc_nan_mask"`` (maths.md §1.9). ``n_bins``,
        ``Jmin`` and ``spectrum_method`` act only with the spectrum: another value than their
        default without it raises, naming LF23 or LF24 on data without a spectrum.
    engine:
        The engine that runs the computation (Part II, §9.4): ``"python"``, the only engine of
        1.0.0.
    """

    _plan: STPlanSingleDomain | STPlanTensorProd
    _filters: STFiltersSingleDomain | STFiltersTensorProd
    _backend: str
    _dtype: FloatDType
    _device: Any
    _engine: str
    _reference: _NormReference | None
    _plan_type: ClassVar[type[STPlanSingleDomain] | type[STPlanTensorProd]] = STPlanSingleDomain
    """The plan of the operator's geometry, which :meth:`from_statistics` takes."""
    _other_statistics: ClassVar[str] = (
        "these are the statistics of a tensor product, which STOperator does not compute: use "
        "STOperatorTensorProd.from_statistics(stats)."
    )

    def __init__(
        self,
        data: Data,
        J: int | None = None,  # noqa: N803 (maths.md notation)
        L: int | None = None,  # noqa: N803
        *,
        standardize: bool = True,
        cross_matrix: Any = None,
        compute_PS: bool | None = None,  # noqa: N803 (STL's name)
        n_bins: int | None = None,
        Jmin: int = 0,  # noqa: N803 (STL's name)
        spectrum_method: str = "auto",
        engine: str = "python",
    ) -> None:
        if not isinstance(data, Data):
            raise TypeError(
                f"STOperator takes a Data object, such as estel.Data2D, got {type(data).__name__}."
            )
        plan, tables = build_plan_with_tables(
            data,
            J,
            L,
            cross_matrix=_as_numpy(cross_matrix),
            standardize=standardize,
            compute_PS=compute_PS,  # None: the default of the geometry, resolved by the plan
            n_bins=n_bins,
            Jmin=Jmin,
            spectrum_method=spectrum_method,
        )
        plan.wavelets.bank()  # validates L for the geometry (M-WAV-1D)
        _warn_channels_without_auto(plan)
        self._setup(plan, data.backend, default_dtype(), default_device(), engine, None, tables)

    @classmethod
    def from_statistics(cls, stats: STStatistics) -> STOperator:
        """The operator that computed ``stats``: the same plan (geometry, mask or blocks, J, L,
        channels, standardisation), on the current backend and default device, in the dtype of
        ``stats``, holding the reference ``stats`` were normalised with: their own with
        ``norm="self"``, which ``apply(data, norm="ref")`` then uses (M-NORM). Called on
        ``STOperatorTensorProd``, it takes the statistics of a tensor product."""
        name = cls.__name__
        if not isinstance(stats, STStatistics):
            raise TypeError(
                f"{name}.from_statistics takes STStatistics, got {type(stats).__name__}."
            )
        plan = stats._plan
        if not isinstance(plan, cls._plan_type):
            raise ValueError(f"{name}.from_statistics: {cls._other_statistics}")
        operator = cls.__new__(cls)
        wide = dtype_name(stats._buffer) in _WIDE
        dtype: FloatDType = "float64" if wide else "float32"
        operator._setup(plan, stats._backend, dtype, default_device(), "python", stats._reference)
        return operator

    def _setup(
        self,
        plan: STPlanSingleDomain | STPlanTensorProd,
        backend: str,
        dtype: FloatDType,
        device: Any,
        engine: str,
        reference: _NormReference | None,
        tables: Any = None,
    ) -> None:
        """The state of an operator: its plan, its filters built once on ``backend`` in
        ``dtype`` by the plan (``STPlan._build_filters``), its engine checked in the registry
        (Part II, §9.4), and its reference. ``tables``: the tables the plan was built with
        (``build_plan_with_tables``), passed to the filters once and not kept (task T5.7)."""
        capabilities = _engines.resolve(engine)
        capabilities.check(geometry=plan.geometry._geometry, dtype=dtype)
        self._plan = plan
        self._backend = backend
        self._dtype = dtype
        self._device = device
        self._engine = engine
        self._filters = plan._build_filters(backend, dtype, device, tables)
        self._reference = reference

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
    def J(self) -> int | tuple[int, ...]:  # noqa: N802 (maths.md notation)
        """The number of scales: an int for a single geometry, one per block for a tensor
        product."""
        return self._plan.J

    @property
    def L(self) -> int | tuple[int, ...]:  # noqa: N802
        """The number of orientations: an int for a single geometry, one per block for a
        tensor product."""
        return self._plan.L

    @property
    def standardize(self) -> bool:
        """Whether each map is standardised before the transform."""
        return self._plan.standardize

    # ------------------------------------------------------------------ apply

    def apply(
        self,
        data: Data,
        ref: STStatistics | None = None,
        *,
        norm: str = "self",
        ref_batch_mean: bool = False,
    ) -> STStatistics:
        """The statistics of ``data`` (maths.md §1.6 to 1.8, §3), as ``STStatistics``, normalised
        as chosen at this call (M-NORM).

        ``data`` must have the geometry, spatial shape, ``pbc``, mask and number of channels of
        the example the operator was built from (on HEALPix, its ``depth`` and ``cell_ids``; for
        a tensor product, its blocks, their sizes and ``pbc``), its backend, and its dtype; the
        batch size may differ.

        ``norm="self"`` (the default) normalises the statistics by themselves; the operator
        stores their reference, replacing the one it held with a ``UserWarning``, and gradients
        flow through it. ``norm="ref"`` normalises them by the reference of ``ref``, statistics
        of the same geometry computed with ``norm="self"``, or else by the one the operator
        holds, from an earlier ``norm="self"`` call or :meth:`from_statistics`; no gradient flows
        through it. ``norm="none"`` does not normalise. ``ref`` is taken with ``norm="ref"``
        only. ``ref_batch_mean=True`` averages the ``norm="self"`` reference over the batch, and
        is taken with ``norm="self"`` only. A channel of ``cross_matrix`` without auto
        statistics has no normalisation scale: its normalised cross statistics are NaN, which
        the constructor warns of (OP-30).
        """
        name = type(self).__name__
        stats, self._reference = apply_operator(
            name,
            data,
            self._plan,
            self._filters,
            ref,
            self._reference,
            norm,
            ref_batch_mean,
        )
        return stats

    def __repr__(self) -> str:
        plan = _plan_of(self, STPlanSingleDomain)
        return (
            f"STOperator(J={plan.J}, L={plan.L}, compute_PS={plan.spectrum is not None}, "
            f"standardize={plan.standardize}, "
            f"spatial_shape={plan.spatial_shape}, n_channels={plan.n_channels}, "
            f"pbc={plan.pbc}, backend={self._backend!r}, dtype={self._dtype!r})"
        )


class STOperatorTensorProd(STOperator):
    """The scattering operator of a tensor product of planar geometries: the scattering
    statistics mean, var and S1 to S4 with multi-index scales (maths.md §3; Part II, §12.6;
    task T3.2), STL's ``ST_Operator_TensorProd``.

    A subclass of :class:`STOperator` (Part I, §2.3; Part II, §12.6): it redefines its
    construction (J and L per block, ``lowpass``), ``lowpass``, ``__repr__`` and the plan that
    :meth:`from_statistics` takes, and inherits the rest, ``apply`` (and with it the
    normalisation chosen at each call), ``from_statistics``, ``plan``, ``filters``, ``J``, ``L``
    and ``standardize``. Its plan and filters,
    subclasses of ``STPlan`` and ``STFilters``, carry the hooks of its loop: the index sets,
    the levels and the two layers (``_loop_tp``). Its statistics are stored in the compact layout of
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
    standardize, cross_matrix:
        As for :class:`STOperator` (M-STD, M-CHAN): ``standardize`` True by default (OP-120).
    compute_PS:
        ``None`` (the default) or ``False``: tensor products have no power spectrum in ESTEL
        1.0.0 (LF23).
    engine:
        The engine that runs the computation (Part II, §9.4): ``"python"``, the only engine of
        1.0.0.
    """

    _plan_type = STPlanTensorProd
    _other_statistics = (
        "these are the statistics of single-domain data: use STOperator.from_statistics(stats)."
    )

    def __init__(
        self,
        data: Data,
        J: Sequence[int | None] | None = None,  # noqa: N803 (maths.md notation)
        L: Sequence[int | None] | None = None,  # noqa: N803
        *,
        lowpass: str | None = "ST",
        standardize: bool = True,
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
                "compute_PS=True: "
                + spectrum_2d_only(
                    type(data).__name__, "Pass compute_PS=False, or leave it to None."
                )
            )
        plan = build_statistics_plan_tensor_prod(
            data,
            J,
            L,
            lowpass=lowpass,
            cross_matrix=_as_numpy(cross_matrix),
            standardize=standardize,
            stacklevel=2,  # OP-34 and OP-13 at the line that builds the operator
        )
        for block in plan.wavelets.blocks:
            block.bank()  # validates L for the geometry of the block (M-WAV-1D)
        _warn_channels_without_auto(plan)
        self._setup(plan, data.backend, default_dtype(), default_device(), engine, None)

    @property
    def lowpass(self) -> str | None:
        """``"ST"`` or ``None`` (maths.md M-LP)."""
        return _plan_of(self, STPlanTensorProd).lowpass

    def __repr__(self) -> str:
        plan = _plan_of(self, STPlanTensorProd)
        names = ", ".join(g.__name__ for g in plan.block_geometries)
        return (
            f"STOperatorTensorProd(J={list(plan.J)}, L={list(plan.L)}, "
            f"lowpass={plan.lowpass!r}, standardize={plan.standardize}, "
            f"blocks=[{names}], spatial_shape={plan.spatial_shape}, "
            f"n_channels={plan.n_channels}, pbc={list(plan.pbc)}, backend={self._backend!r}, "
            f"dtype={self._dtype!r})"
        )


_P = TypeVar("_P", STPlanSingleDomain, STPlanTensorProd)


def _plan_of(operator: STOperator, kind: type[_P]) -> _P:
    """The plan of ``operator``, which must be a ``kind``, the plan of its geometry; a check
    that holds under ``python -O`` (rule B4), unlike an ``assert``."""
    plan = operator._plan
    if not isinstance(plan, kind):
        name = type(operator).__name__
        raise TypeError(
            f"{name} holds a {type(plan).__name__}, not the {kind.__name__} of its geometry: "
            f"build it with {name}(data, ...) or {name}.from_statistics(stats), which set its "
            "plan, and do not replace the plan of an operator."
        )
    return plan


# ---------------------------------------------------------------------- validation


def _as_numpy(matrix: Any) -> Any:
    """``cross_matrix`` as NumPy when it is an array of a backend."""
    name = _registry.backend_of(matrix)
    return matrix if name is None else _registry.backend_named(name).to_numpy(matrix)


def _warn_channels_without_auto(plan: STPlanSingleDomain | STPlanTensorProd) -> None:
    """STL's warning of the channels without auto statistics (maths.md M-CHAN, OP-30), given
    once, at construction, which knows the channel pairs; the normalisation is chosen later, at
    each call, so the warning says what happens if the statistics are normalised. It covers
    every path, ``apply``, ``estel.functional.scattering`` and the syntheses."""
    missing = [c for c in range(plan.n_channels) if c not in plan.pairs.auto]
    if missing:
        warnings.warn(
            f"cross_matrix: channel(s) {missing} have no auto statistics, so no "
            "normalisation scale: if the statistics are normalised (norm='self' or 'ref'), "
            "their cross coefficients are NaN (maths.md M-CHAN, OP-30). Set their diagonal "
            "entries to normalise them.",
            UserWarning,
            stacklevel=3,
        )
