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

import math
import warnings
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
from ._filters import STFilters, build_filters, wavelet_kernels
from ._plan import STPlan, build_plan

_WIDE = ("float64", "complex128")


class STOperator:
    """The scattering operator of 1D and 2D planar data: STL's ``ST_Operator`` with vanilla
    ScatCov statistics (maths.md §1.6 to 1.8, §2.3 to 2.4).

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
        ``"self"`` (the default; STL's ``store_ref``): normalise the statistics by themselves,
        and keep the reference; ``"ref"`` (STL's ``load_ref``): normalise by the reference of
        other statistics; ``"none"`` (STL's ``vanilla``): no normalisation (M-NORM).
    ref_batch_mean:
        With ``norm="self"``, average the reference over the batch first; False by default, as
        in STL, where it is ``norm_batch_mean`` (M-NORM).
    standardize:
        Remove the mean and divide by the standard deviation of each map before the transform,
        as STL's ``apply(standardize=True)`` (M-MOM); both values are kept with the statistics.
    cross_matrix:
        A boolean ``(Nc, Nc)`` array: the diagonal entry of a channel asks for its auto
        statistics, the entry ``[c1, c2]``, c1 < c2, for the cross statistics of the pair; the
        entries below the diagonal are ignored, as in STL. By default every pair, the upper
        triangle (M-CHAN, M-DEF). Every channel must appear in at least one pair (OP-30).
    compute_PS:
        Whether to compute the power spectrum: True by default for 2D data, unavailable for 1D
        data in 1.0.0 (LF23). The 2D spectra come in T4.1: pass ``compute_PS=False``.
    engine:
        The engine that runs the computation (Part II, §9.4): ``"python"``, the only engine of
        1.0.0.
    """

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
        n_scales = _default_scales(data) if J is None else J
        n_orientations = _default_orientations(data) if L is None else L
        plan = build_plan(
            data,
            n_scales,
            n_orientations,
            cross_matrix=_as_numpy(cross_matrix),
            norm=norm,
            ref_batch_mean=ref_batch_mean,
            standardize=standardize,
        )
        wavelet_kernels(plan.ndim, plan.L)  # validates L for the geometry (M-WAV-1D)
        if norm != "none":
            _warn_channels_without_auto(plan)
        self._setup(plan, data.backend, default_dtype(), default_device(), engine, None)

    @classmethod
    def from_statistics(cls, stats: STStatistics) -> STOperator:
        """The operator that computed ``stats``: the same plan (geometry, mask, J, L, channels,
        normalisation, standardisation), on the current backend and default device, in the
        dtype of ``stats``, holding the reference ``stats`` were normalised with (STL
        ``from_ST_Statistics``)."""
        if not isinstance(stats, STStatistics):
            raise TypeError(
                f"STOperator.from_statistics takes STStatistics, got {type(stats).__name__}."
            )
        operator = cls.__new__(cls)
        wide = dtype_name(next(iter(stats._values.values()))) in _WIDE
        dtype: FloatDType = "float64" if wide else "float32"
        operator._setup(
            stats._plan, stats._backend, dtype, default_device(), "python", stats._reference
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
        capabilities = _engines.resolve(engine)
        capabilities.check(geometry=plan.geometry._geometry, dtype=dtype)
        self._plan = plan
        self._backend = backend
        self._dtype = dtype
        self._device = device
        self._engine = engine
        self._filters = build_filters(plan, backend, dtype, device)
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
        check_data(data, self._plan, self._filters)
        reference = reference_of(ref, self._plan) if ref is not None else None
        if reference is None and self._plan.norm == "ref":
            reference = self._reference
            if reference is None:
                raise ValueError(
                    'STOperator.apply: norm="ref" needs a reference, and this operator '
                    'holds none. Pass ref=, statistics computed with norm="self", or build '
                    "the operator with STOperator.from_statistics (maths.md M-NORM)."
                )
        stats = compute(data, self._plan, self._filters, reference)
        if self._plan.norm == "self":
            if self._reference is not None:
                warnings.warn(
                    'STOperator.apply: norm="self" replaces the reference this operator '
                    "held by that of the new statistics (maths.md M-NORM).",
                    UserWarning,
                    stacklevel=2,
                )
            self._reference = stats._reference
        return stats

    def __repr__(self) -> str:
        plan = self._plan
        return (
            f"STOperator(J={plan.J}, L={plan.L}, norm={plan.norm!r}, "
            f"ref_batch_mean={plan.ref_batch_mean}, standardize={plan.standardize}, "
            f"spatial_shape={plan.spatial_shape}, n_channels={plan.n_channels}, "
            f"pbc={plan.pbc}, backend={self._backend!r}, dtype={self._dtype!r})"
        )


# ---------------------------------------------------------------------- validation


def _as_numpy(matrix: Any) -> Any:
    """``cross_matrix`` as NumPy when it is an array of a backend."""
    name = _registry.backend_of(matrix)
    return matrix if name is None else _registry.backend_named(name).to_numpy(matrix)


def _warn_channels_without_auto(plan: STPlan) -> None:
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


def _default_scales(data: Data) -> int:
    """J = ⌊log₂ min N⌋ - 2 (maths.md M-DEF, M-DEF-1D), at least 1."""
    smallest = min(data.shape[2:])
    n = math.floor(math.log2(smallest)) - 2
    if n < 1:
        raise ValueError(
            f"the default J = ⌊log2 {smallest}⌋ - 2 = {n} is below 1: the data are too small "
            "for the default (maths.md M-DEF). Pass J explicitly, or use larger data."
        )
    return n


def _default_orientations(data: Data) -> int:
    """L = 4 in 2D, 1 in 1D (maths.md M-DEF, M-DEF-1D)."""
    return 4 if data.ndim == 2 else 1
