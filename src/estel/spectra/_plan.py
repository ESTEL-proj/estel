"""The plan and the filters of ``SpectrumOperator`` (Part II, §12.8; maths.md §1.9; task T4.1).

:class:`SpectrumOperatorPlan` is the ``STPlan`` of an operator that computes the power spectrum
alone: the static metadata of the data (geometry, spatial shape, ``pbc``, mask, number of
channels), the channel pairs of ``cross_matrix`` (M-CHAN; every pair c₁ ≤ c₂ by default, OP-94),
the normalisation (``norm``, ``ref_batch_mean``; M-NORM), and the description of the spectrum,
the ``SpectrumPlan`` that the plan of ``STOperator`` holds too (``scattering._plan``): one
description, one set of filters (``spectra._estimators.SpectrumFilters``) and one set of
estimators for both operators. Its hooks give ``STStatistics`` the layout of the family
``"PS"`` alone, the reference of ``norm="self"`` (σᴾˢ only) and its parameters for ``estel.io``.
Built once, in NumPy, when the operator is built; immutable and hashable, so that JAX takes it
as a static argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend import _registry
from ..backend._protocol import Backend
from ..data._base import Data
from ..scattering._filters import STFilters
from ..scattering._loop import ScatteringValues
from ..scattering._plan import (
    ChannelPairs,
    SpectrumAdvice,
    SpectrumPlan,
    STPlan,
    _check_options,
    _cross_matrix,
    build_spectrum_plan,
    channel_pairs,
)
from ..statistics._layout import Layout, spectrum_layout, spectrum_pairs
from ..statistics._normalization import (
    _NormReference,
    check_spectrum_reference,
    spectrum_reference_of,
    spectrum_self_reference,
)
from ._estimators import SpectrumFilters, build_spectrum_filters, spectrum_floats

SPECTRUM_OPERATOR_ADVICE = SpectrumAdvice(
    not_2d="pass estel.Data2D data to SpectrumOperator.",
    odd_size="crop the data to even sizes.",
    jmin="Pass a smaller Jmin or a larger J, or use a larger image.",
)
"""The advice of ``SpectrumOperator``, which has no ``compute_PS`` to switch off."""


class _SpectrumFilterFields(NamedTuple):
    spectrum: SpectrumFilters
    """The static arrays of the estimator (``spectra._estimators``)."""


class SpectrumOperatorFilters(_SpectrumFilterFields, STFilters):
    """The :class:`STFilters` of ``SpectrumOperator`` (``SpectrumOperator.filters``): the arrays of
    its estimator, on its backend and dtype. A named tuple, so a pytree whose leaves are the
    arrays."""

    __slots__ = ()

    @property
    def _any_array(self) -> Any:
        return self.spectrum.modes


@dataclass(frozen=True, eq=False)
class _SpectrumPlanFields:
    geometry: type[Data]
    """The geometry of the data, ``Data2D`` in 1.0.0."""
    spatial_shape: tuple[int, ...]
    """(N₁, N₂), even."""
    pbc: bool
    mask: NDArray[np.bool_] | None
    """The missing values of the data (True = missing), read-only, or ``None``."""
    n_channels: int
    cross_matrix: NDArray[np.bool_]
    """``(Nc, Nc)``, read-only, upper triangle (M-CHAN)."""
    pairs: ChannelPairs
    """The channel pairs: the spectrum of every pair c1 ≤ c2 of ``cross_matrix``."""
    norm: str
    ref_batch_mean: bool
    spectrum: SpectrumPlan
    """The description of the spectrum: n_b, J_PS, J_min, the estimator, the spatial shape."""
    standardize: bool
    """False: ``SpectrumOperator`` does not standardise its data."""
    _key: tuple[Any, ...] = field(repr=False)


class SpectrumOperatorPlan(_SpectrumPlanFields, STPlan):
    """The :class:`STPlan` of ``SpectrumOperator`` (``SpectrumOperator.plan``), built by
    :func:`build_spectrum_operator_plan`; public and opaque as ``STPlan``."""

    _operator_name: ClassVar[str] = "SpectrumOperator"

    def _check_data(self, data: Data, filters: STFilters) -> None:
        if not isinstance(filters, SpectrumOperatorFilters):
            raise TypeError(
                "the plan and the filters come from different operators: pass op.plan and "
                "op.filters of one SpectrumOperator."
            )
        if data.mask is None or self.mask is None:
            same_mask = data.mask is None and self.mask is None
        else:
            same_mask = bool(np.array_equal(data.mask, self.mask))
        if not (
            isinstance(data, self.geometry)
            and tuple(data.shape[2:]) == self.spatial_shape
            and data.pbc == self.pbc
            and same_mask
        ):
            raise ValueError(
                f"the data ({type(data).__name__}, spatial shape {tuple(data.shape[2:])}, "
                f"pbc={data.pbc}) differ from those the plan was built for "
                f"({self.geometry.__name__}, spatial shape {self.spatial_shape}, pbc={self.pbc}, "
                "and the same mask of missing values). The plan is static: build a new "
                "SpectrumOperator for these data."
            )

    def _build_filters(
        self, backend_name: str, dtype: FloatDType, device: Any
    ) -> SpectrumOperatorFilters:
        backend = _registry.backend_named(backend_name)
        return SpectrumOperatorFilters(
            spectrum=build_spectrum_filters(self.spectrum, self.mask, backend, dtype, device)
        )

    def _values(self, data: Data, filters: STFilters) -> ScatteringValues:
        backend = _registry.backend_named(data.backend)
        found = cast(SpectrumOperatorFilters, filters)  # by _check_data
        floats = spectrum_floats(
            backend, data.array, self.spectrum, found.spectrum, spectrum_pairs(self._layout())
        )
        return ScatteringValues(buffer=floats)

    def _layout(self) -> Layout:
        matrix = np.zeros((self.n_channels, self.n_channels), dtype=bool)
        for c in self.pairs.auto:
            matrix[c, c] = True
        for c1, c2 in self.pairs.cross:
            matrix[c1, c2] = True
        return spectrum_layout(self.spectrum.n_bins, matrix.tobytes(), self.n_channels)

    def _check_reference(self, reference: _NormReference, n_batch: int, dtype: Any) -> None:
        check_spectrum_reference(reference, self, n_batch, dtype)

    def _self_reference(self, backend: Backend, buffer: Any, layout: Layout) -> _NormReference:
        return spectrum_self_reference(
            backend,
            buffer,
            layout,
            self.n_channels,
            self.ref_batch_mean,
            spectrum_reference_of(self.spectrum, self.pairs),
        )

    def _content(self) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        content = {
            "operator": self._operator_name,
            "geometry": self.geometry._geometry,
            "spatial_shape": list(self.spatial_shape),
            "pbc": self.pbc,
            "n_channels": self.n_channels,
            "norm": self.norm,
            "ref_batch_mean": self.ref_batch_mean,
            "spectrum": self.spectrum.content(),
        }
        return content, ({} if self.mask is None else {"mask": np.asarray(self.mask)})


def build_spectrum_operator_plan(
    data: Data,
    J: int | None = None,  # noqa: N803 (maths.md notation)
    *,
    Jmin: Any = 0,  # noqa: N803
    n_bins: Any = None,
    method: Any = "auto",
    cross_matrix: Any = None,
    norm: str = "self",
    ref_batch_mean: bool = False,
    advice: SpectrumAdvice = SPECTRUM_OPERATOR_ADVICE,
) -> SpectrumOperatorPlan:
    """The :class:`SpectrumOperatorPlan` of the power spectrum of data with the geometry, spatial
    shape, ``pbc``, mask and number of channels of ``data`` (maths.md §1.9; Part II, §12.8): the
    checks of ``scattering._plan.build_spectrum_plan`` (2D data of even sizes, J_min, ``n_bins``,
    the estimator), those of ``cross_matrix`` (M-CHAN) and of ``norm`` and ``ref_batch_mean``
    (M-NORM), whose errors end with ``advice``, that of ``SpectrumOperator``. The values of
    ``data.array`` are not read."""
    if not isinstance(data, Data):
        raise TypeError(
            f"SpectrumOperator takes a Data object, such as estel.Data2D, got "
            f"{type(data).__name__}."
        )
    if data._geometry == "tensor_prod":
        raise ValueError(
            "SpectrumOperator: tensor products have no power spectrum in ESTEL 1.0.0; it comes "
            "later (LF23). Pass estel.Data2D data."
        )
    if J is not None and (isinstance(J, bool) or not isinstance(J, int | np.integer)):
        raise TypeError(f"J={J!r}: pass an integer, or None for ⌊log₂ N_max⌋ - 1 (maths.md §1.9).")
    _check_options(norm, ref_batch_mean, False)
    matrix = _cross_matrix(cross_matrix, data.n_channels)
    spatial = tuple(int(n) for n in data.shape[2:])
    spectrum = build_spectrum_plan(
        spatial,
        data.ndim,
        J,
        n_bins,
        Jmin,
        method,
        bool(data.pbc),
        data.mask,
        advice,
    )
    return SpectrumOperatorPlan(
        geometry=type(data),
        spatial_shape=spatial,
        pbc=bool(data.pbc),
        mask=data.mask,
        n_channels=data.n_channels,
        cross_matrix=matrix,
        pairs=channel_pairs(matrix),
        norm=norm,
        ref_batch_mean=bool(ref_batch_mean),
        spectrum=spectrum,
        standardize=False,
        _key=(
            type(data),
            spatial,
            bool(data.pbc),
            None if data.mask is None else (data.mask.shape, np.packbits(data.mask).tobytes()),
            data.n_channels,
            matrix.tobytes(),
            norm,
            bool(ref_batch_mean),
            spectrum,
        ),
    )
