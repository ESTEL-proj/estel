"""``STOperator``: the scattering operator of one planar domain (Part II, §12.6; task T2.4).

Internal in T2.4: ``STOperator``, ``estel.functional.scattering`` and ``STStatistics`` become
public together in T2.5 (Part II, §15.2), when ``apply`` returns an ``STStatistics`` and the
normalisations ``store_ref`` and ``load_ref`` exist. Until then the operator computes the
statistics without normalisation (``norm="vanilla"``) and returns them as STL's full-grid arrays
(:class:`~estel.scattering._loop.ScatteringValues`).

Every option is set once, at construction (review rule A6; Part I, §2.3): the operator builds
the static plan (``build_plan``), the channel pairs and the backend arrays, cast to the default
dtype; ``apply`` takes only the data, and an optional reference that ``norm="vanilla"`` does not
accept. The operator records its backend, dtype and the static metadata of the data it was built
for (geometry, spatial shape, ``pbc``, mask), and refuses other data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType, default_device, default_dtype
from ..backend import _registry
from ..backend._protocol import DType
from ..data._base import DataClass, _StaticMask
from ..wavelets._banks import bump_steerable_1d, bump_steerable_2d, smoothing_kernel
from ._loop import (
    ChannelPairs,
    ScatteringArrays,
    ScatteringValues,
    channel_pairs,
    coefficient_index,
    first_layer_keys,
    s3_keys,
    s4_keys,
    scattering,
)
from ._plan import ScatteringPlan, build_plan

NORMS = ("vanilla", "store_ref", "load_ref")
_COMPLEX: dict[FloatDType, DType] = {"float32": "complex64", "float64": "complex128"}


class STOperator:
    """The scattering operator of 1D and 2D planar data: STL's ``ST_Operator`` with vanilla
    ScatCov statistics (maths.md §1.6 to 1.7, §2.3 to 2.4).

    Parameters
    ----------
    data:
        An example of the data, ``estel.Data1D`` or ``estel.Data2D``: its geometry, spatial
        shape, ``pbc`` and mask fix the operator; its values are not read.
    J:
        The number of scales; by default ⌊log₂ min N⌋ - 2 (M-DEF, M-DEF-1D). The spatial sizes
        must be divisible by 2^(J-1) (OP-8).
    L:
        The number of orientations; by default 4 in 2D and 1 in 1D. In 1D, L is 1 or 2 (the
        second filter is the conjugate of the first, M-WAV-1D).
    norm:
        ``"vanilla"`` in T2.4. ``"store_ref"`` (the default of 1.0.0) and ``"load_ref"`` come
        with ``STStatistics`` in T2.5 (M-NORM).
    standardize:
        Remove the mean and divide by the standard deviation of each map before the transform,
        as STL's ``apply(standardize=True)`` (M-MOM); both values are kept with the statistics.
    compute_cross_matrix:
        A boolean ``(Nc, Nc)`` array: the diagonal entry of a channel asks for its auto
        statistics, the entry ``[c1, c2]``, c1 < c2, for the cross statistics of the pair; the
        entries below the diagonal are ignored, as in STL. By default every pair, the upper
        triangle (M-CHAN, M-DEF). Every channel must appear in at least one pair (OP-30).
    compute_PS:
        Whether to compute the power spectrum: True by default for 2D data, unavailable for 1D
        data in 1.0.0 (LF23). The 2D spectra come in T4.1: pass ``compute_PS=False``.
    """

    def __init__(
        self,
        data: DataClass,
        J: int | None = None,  # noqa: N803 (maths.md notation)
        L: int | None = None,  # noqa: N803
        *,
        norm: str = "store_ref",
        standardize: bool = False,
        compute_cross_matrix: Any = None,
        compute_PS: bool | None = None,  # noqa: N803 (STL's name)
    ) -> None:
        if not isinstance(data, DataClass):
            raise TypeError(
                f"STOperator takes a DataClass, such as estel.Data2D, got {type(data).__name__}."
            )
        _check_norm(norm)
        if not isinstance(standardize, bool | np.bool_):
            raise TypeError(f"standardize={standardize!r}: pass True or False.")
        _check_compute_ps(compute_PS, data.ndim)
        n_scales = _default_scales(data) if J is None else J
        n_orientations = _default_orientations(data) if L is None else L
        self._plan: ScatteringPlan = build_plan(data, n_scales, n_orientations)
        self._kernels = _kernels(data.ndim, self._plan.L)
        self._matrix = _cross_matrix(compute_cross_matrix, data.n_channels)
        self._norm = norm
        self._standardize = bool(standardize)
        self._backend = data.backend
        self._dtype = default_dtype()
        self._device = default_device()
        self._key = (type(data), data.shape[2:], data.pbc, _static_mask(data))
        self._arrays: dict[int, tuple[ChannelPairs, ScatteringArrays]] = {}
        if self._matrix is not None:
            self._arrays_for(data.n_channels)

    # ------------------------------------------------------------------ accessors

    @property
    def plan(self) -> ScatteringPlan:
        """The static plan of the transform."""
        return self._plan

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
        """The normalisation of the statistics."""
        return self._norm

    @property
    def standardize(self) -> bool:
        """Whether each map is standardised before the transform."""
        return self._standardize

    # ------------------------------------------------------------------ apply

    def apply(self, data: DataClass, ref: Any = None) -> ScatteringValues:
        """The statistics of ``data``, without normalisation, in STL's full-grid shapes.

        ``data`` must have the geometry, spatial shape, ``pbc`` and mask of the example the
        operator was built from, its backend, and its dtype; the batch size may differ, and so
        may the number of channels unless ``compute_cross_matrix`` was given. ``ref`` is the
        normalisation reference of ``norm="load_ref"`` (T2.5): ``norm="vanilla"`` takes none.
        """
        if ref is not None:
            raise ValueError(
                'apply(data, ref): norm="vanilla" takes no reference; pass ref only to an '
                'operator built with norm="load_ref" (maths.md M-NORM).'
            )
        self._check_data(data)
        pairs, arrays = self._arrays_for(data.n_channels)
        return scattering(data, self._plan, pairs, arrays, standardize_data=self._standardize)

    # ------------------------------------------------------------------ internals

    def _check_data(self, data: DataClass) -> None:
        if not isinstance(data, DataClass):
            raise TypeError(
                f"STOperator.apply takes a DataClass, such as estel.Data2D, got "
                f"{type(data).__name__}."
            )
        _registry.check_backend(self._backend, data.array, owner="this STOperator")
        key = (type(data), data.shape[2:], data.pbc, _static_mask(data))
        if key != self._key:
            kind, spatial, pbc, _ = self._key
            raise ValueError(
                f"STOperator.apply: the data ({type(data).__name__}, spatial shape "
                f"{data.shape[2:]}, pbc={data.pbc}) differ from those the operator was built for "
                f"({kind.__name__}, spatial shape {spatial}, pbc={pbc}, and the same mask of "
                "missing values). The plan is static: build a new STOperator for these data."
            )
        xp = _registry.backend_named(self._backend).xp
        if data.array.dtype != getattr(xp, self._dtype):
            raise TypeError(
                f"STOperator.apply: the data are {data.array.dtype}, but the operator computes "
                f"in {self._dtype}, the default dtype when it was built. Build the data and the "
                "operator with the same default dtype (estel.set_default_dtype)."
            )

    def _arrays_for(self, n_channels: int) -> tuple[ChannelPairs, ScatteringArrays]:
        """The channel pairs and backend arrays for data with ``n_channels`` channels, built
        once per number of channels."""
        if self._matrix is not None and n_channels != self._matrix.shape[0]:
            raise ValueError(
                f"STOperator.apply: the data have {n_channels} channel(s), but "
                f"compute_cross_matrix is {self._matrix.shape}. Pass data with "
                f"{self._matrix.shape[0]} channel(s), or build a new STOperator."
            )
        if n_channels not in self._arrays:
            matrix = (
                self._matrix
                if self._matrix is not None
                else np.triu(np.ones((n_channels, n_channels), dtype=bool))
            )
            pairs = channel_pairs(matrix)
            self._arrays = {**self._arrays, n_channels: (pairs, self._build_arrays(pairs))}
        return self._arrays[n_channels]

    def _build_arrays(self, pairs: ChannelPairs) -> ScatteringArrays:
        """The backend arrays of the plan, cast once to the operator's dtype (Part I, §3.1,
        rules 4 and 7)."""
        backend = _registry.backend_named(self._backend)
        plan, dtype, device = self._plan, self._dtype, self._device

        def real(a: NDArray[Any]) -> Any:
            return backend.asarray(np.asarray(a, dtype=np.float64), dtype, device)

        def boolean(a: NDArray[Any]) -> Any:
            return backend.asarray(np.asarray(a, dtype=bool), "bool", device)

        def index(keys: Sequence[tuple[int, ...]], n_axes: int) -> tuple[Any, ...]:
            return tuple(
                backend.asarray(c.astype(np.int32), "int32", device)
                for c in coefficient_index(keys, n_axes)
            )

        def maps(level: int, convolved_at: int | None = None) -> tuple[Any, Any] | None:
            found = plan.downsample_maps(level, convolved_at)
            return None if found is None else (boolean(found[0]), real(found[1]))

        down_levels = range(plan.J - 1)
        field_down = {d: maps(d) for d in down_levels}
        modulus_down = {(d, j): maps(d, j) for d in down_levels for j in range(d + 1)}
        return ScatteringArrays(
            kernels=backend.asarray(self._kernels, _COMPLEX[dtype], device),
            smoothing=real(smoothing_kernel(plan.ndim)),
            valid=None if plan.masks is None else boolean(~plan.masks.full_res),
            moments=real(plan.moments.weights),
            first_layer={j: real(w.weights) for j, w in plan.first_layer.items()},
            s3={key: real(w.weights) for key, w in plan.s3.items()},
            s4={key: real(w.weights) for key, w in plan.s4.items()},
            field_down={d: m for d, m in field_down.items() if m is not None},
            modulus_down={k: m for k, m in modulus_down.items() if m is not None},
            s1_index=index(first_layer_keys(plan, pairs), 3),
            s3_index=index(s3_keys(plan, pairs), 4),
            s4_index=index(s4_keys(plan, pairs), 5),
        )

    def __repr__(self) -> str:
        return (
            f"STOperator(J={self.J}, L={self.L}, norm={self._norm!r}, "
            f"standardize={self._standardize}, spatial_shape={self._plan.spatial_shape}, "
            f"pbc={self._plan.pbc}, backend={self._backend!r}, dtype={self._dtype!r})"
        )


# ---------------------------------------------------------------------- validation


def _check_norm(norm: object) -> None:
    if norm not in NORMS:
        raise ValueError(
            f"norm={norm!r}: pass one of {', '.join(repr(n) for n in NORMS)} (maths.md M-NORM); "
            '"vanilla" computes the statistics without normalisation.'
        )
    if norm != "vanilla":
        raise NotImplementedError(
            f'norm={norm!r} comes with STStatistics (task T2.5); pass norm="vanilla" for the '
            "statistics without normalisation."
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


def _default_scales(data: DataClass) -> int:
    """J = ⌊log₂ min N⌋ - 2 (maths.md M-DEF, M-DEF-1D), at least 1."""
    smallest = min(data.shape[2:])
    n = math.floor(math.log2(smallest)) - 2
    if n < 1:
        raise ValueError(
            f"the default J = ⌊log2 {smallest}⌋ - 2 = {n} is below 1: the data are too small "
            "for the default (maths.md M-DEF). Pass J explicitly, or use larger data."
        )
    return n


def _default_orientations(data: DataClass) -> int:
    """L = 4 in 2D, 1 in 1D (maths.md M-DEF, M-DEF-1D)."""
    return 4 if data.ndim == 2 else 1


def _kernels(ndim: int, n_orientations: int) -> NDArray[np.complex128]:
    """The wavelet bank of the geometry (maths.md M-WAV-2D, M-WAV-1D)."""
    if ndim == 2:
        return bump_steerable_2d(n_orientations)
    if ndim == 1:
        if n_orientations not in (1, 2):
            raise ValueError(
                f"L={n_orientations}: 1D data have L = 1 (the default) or L = 2, the analytic "
                "wavelet and its conjugate (maths.md M-WAV-1D)."
            )
        return bump_steerable_1d(conjugate=n_orientations == 2)
    raise ValueError(f"STOperator handles 1D and 2D planar data; got {ndim} spatial axes.")


def _cross_matrix(matrix: Any, n_channels: int) -> NDArray[np.bool_] | None:
    """``compute_cross_matrix`` as a read-only NumPy boolean array, or ``None`` for the default;
    every channel must appear in a pair (maths.md M-CHAN, OP-30)."""
    if matrix is None:
        return None
    if _registry.backend_of(matrix) is not None:
        matrix = _registry.backend_named(_registry.backend_of(matrix) or "").to_numpy(matrix)
    found = np.asarray(matrix)
    if found.dtype != np.bool_ or found.ndim != 2 or found.shape[0] != found.shape[1]:
        raise TypeError(
            f"compute_cross_matrix: pass a square boolean array (Nc, Nc); got dtype "
            f"{found.dtype} and shape {found.shape}."
        )
    if found.shape[0] != n_channels:
        raise ValueError(
            f"compute_cross_matrix is {found.shape}, but the data have {n_channels} channel(s)."
        )
    upper = np.triu(found)
    for c in range(n_channels):
        if not (upper[c, :].any() or upper[:, c].any()):
            raise ValueError(
                f"compute_cross_matrix: channel {c} has no auto statistics and appears in no "
                "cross pair (maths.md M-CHAN, OP-30). Set its diagonal entry, or an entry of one "
                "of its pairs above the diagonal, or remove the channel."
            )
    upper.flags.writeable = False
    return upper


def _static_mask(data: DataClass) -> _StaticMask | None:
    return None if data.mask is None else _StaticMask(data.mask)
