"""``SpectrumOperator``: the 2D power spectrum as an operator (Part II, §12.8; maths.md §1.9;
task T4.1).

Built as ``STOperator`` (Part I, §2.3): ``SpectrumOperator(data, ...)`` builds the static plan
(``spectra._plan``: the bins, the windows, the borders and the estimator, in NumPy) and the
filters, cast once to the default dtype, and ``apply(data, ref=None)`` calls
``estel.functional.spectrum``. Every option is set once, at construction. Its statistics hold the
family ``"PS"`` alone, with the channel pairs of ``cross_matrix`` (M-CHAN), normalised as ``norm``
says (M-NORM), by the normalisation of ``STStatistics``, the same as for ``STOperator``. STL's
spectrum operator normalises nothing: its values are those of ``norm="none"``.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from .._config import FloatDType, default_device, default_dtype
from ..data._base import Data
from ..functional._scattering import reference_of
from ..functional._spectrum import compute_spectrum
from ..scattering._filters import STFilters
from ..scattering._operator import _as_numpy
from ..scattering._plan import STPlan
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics
from ._estimators import METHODS
from ._plan import SpectrumOperatorFilters, SpectrumOperatorPlan, build_spectrum_operator_plan


class SpectrumOperator:
    """The 2D power spectrum: binned auto and cross spectra of every channel pair (maths.md
    §1.9), as ``STStatistics`` holding the family ``"PS"`` alone.

    Parameters
    ----------
    data:
        An example of the data, ``estel.Data2D`` of even sizes (OP-20): its spatial shape,
        ``pbc``, mask and number of channels fix the operator; its values are not read.
    J:
        J_PS, the range of the bins: by default ⌊log₂ N_max⌋ - 1, N_max = ⌊min(N₁, N₂)/2⌋, the
        default J of ``STOperator``.
    Jmin:
        J_min, 0 by default, with 0 ≤ Jmin ≤ J - 2 (OP-93).
    n_bins:
        The number of bins: by default max(4, round(3 (J - 1 - Jmin))), about three per octave.
    method:
        The estimator, one of :attr:`METHODS`: ``"auto"`` (the default) chooses it from ``pbc``
        and the missing pixels of ``data``: ``fft`` (periodic), ``pbc_mask`` (non-periodic),
        ``nan_mask`` (periodic, missing pixels), ``pbc_nan_mask`` (non-periodic, missing
        pixels). ``fft`` and ``pbc_mask`` do not accept missing pixels.
    cross_matrix:
        A boolean ``(Nc, Nc)`` array: the pairs c1 ≤ c2 whose spectrum is computed, the
        diagonal for the auto spectra; the entries below the diagonal are ignored. By default
        every pair c1 ≤ c2, as on ``STOperator`` (M-CHAN, OP-94); STL's operator alone computes
        the auto spectra by default: pass ``numpy.eye(Nc, dtype=bool)`` for that.
    norm:
        ``"self"`` (the default): divide PS[c1, c2, b] by σᴾˢ_{c1}(b) σᴾˢ_{c2}(b), σᴾˢ_c(b) the
        square root of its own auto spectrum, and keep the reference; ``"ref"``: by the
        reference of other statistics, of a ``SpectrumOperator`` or of an ``STOperator`` whose
        spectrum has the same ``n_bins``, J and ``Jmin``; ``"none"``: the spectrum as STL's
        operator gives it (M-NORM).
    ref_batch_mean:
        With ``norm="self"``, average the reference over the batch first; False by default.
    """

    METHODS: ClassVar[tuple[str, ...]] = METHODS
    """The values of ``method``: ``"auto"``, ``"fft"``, ``"pbc_mask"``, ``"nan_mask"`` and
    ``"pbc_nan_mask"`` (maths.md §1.9)."""

    _plan: SpectrumOperatorPlan
    _filters: SpectrumOperatorFilters
    _backend: str
    _dtype: FloatDType
    _device: Any
    _reference: _NormReference | None

    def __init__(
        self,
        data: Data,
        J: int | None = None,  # noqa: N803 (maths.md notation)
        *,
        Jmin: int = 0,  # noqa: N803 (STL's name)
        n_bins: int | None = None,
        method: str = "auto",
        cross_matrix: Any = None,
        norm: str = "self",
        ref_batch_mean: bool = False,
    ) -> None:
        plan = build_spectrum_operator_plan(
            data,
            J,
            Jmin=Jmin,
            n_bins=n_bins,
            method=method,
            cross_matrix=_as_numpy(cross_matrix),
            norm=norm,
            ref_batch_mean=ref_batch_mean,
        )
        if norm != "none":
            missing = [c for c in range(plan.n_channels) if c not in plan.pairs.auto]
            if missing:
                warnings.warn(
                    f"cross_matrix: channel(s) {missing} have no auto spectrum, so no "
                    "normalisation scale: their cross spectra are NaN after normalisation "
                    "(maths.md M-CHAN, OP-30). Set their diagonal entries to normalise them.",
                    UserWarning,
                    stacklevel=2,
                )
        self._plan = plan
        self._backend = data.backend
        self._dtype = default_dtype()
        self._device = default_device()
        self._filters = plan._build_filters(self._backend, self._dtype, self._device)
        self._reference = None

    @property
    def plan(self) -> STPlan:
        """The static plan of the spectrum, for ``estel.functional.spectrum``."""
        return self._plan

    @property
    def filters(self) -> STFilters:
        """The arrays of the estimator, for ``estel.functional.spectrum``."""
        return self._filters

    def apply(self, data: Data, ref: STStatistics | None = None) -> STStatistics:
        """The power spectrum of ``data`` (maths.md §1.9), as ``STStatistics`` holding the family
        ``"PS"`` alone, ``(Nb, Nc, Nc, n_bins)`` in its full grid.

        ``data`` must have the spatial shape, ``pbc``, mask and number of channels of the example
        the operator was built from, its backend and its dtype; the batch size may differ.
        ``ref`` is for ``norm="ref"`` only: statistics computed with ``norm="self"`` by a
        ``SpectrumOperator``, or by an ``STOperator`` whose spectrum has the same ``n_bins``, J
        and ``Jmin``; without it, the operator uses the reference it holds. With
        ``norm="self"`` the operator keeps the reference of the result, replacing the one it
        held, with a ``UserWarning``.
        """
        plan = self._plan
        reference = reference_of(ref, plan) if ref is not None else None  # type: ignore[arg-type]
        if reference is None and plan.norm == "ref":
            reference = self._reference
            if reference is None:
                raise ValueError(
                    'SpectrumOperator.apply: norm="ref" needs a reference, and this operator '
                    'holds none. Pass ref=, statistics computed with norm="self" (maths.md '
                    "M-NORM)."
                )
        stats = compute_spectrum(data, plan, self._filters, reference)
        if plan.norm == "self":
            if self._reference is not None:
                warnings.warn(
                    'SpectrumOperator.apply: norm="self" replaces the reference this operator '
                    "held by that of the new statistics (maths.md M-NORM).",
                    UserWarning,
                    stacklevel=2,
                )
            self._reference = stats._reference
        return stats

    def __repr__(self) -> str:
        plan = self._plan
        spectrum = plan.spectrum
        return (
            f"SpectrumOperator(J={spectrum.J}, Jmin={spectrum.Jmin}, n_bins={spectrum.n_bins}, "
            f"method={spectrum.method!r}, norm={plan.norm!r}, "
            f"ref_batch_mean={plan.ref_batch_mean}, spatial_shape={plan.spatial_shape}, "
            f"n_channels={plan.n_channels}, pbc={plan.pbc}, backend={self._backend!r}, "
            f"dtype={self._dtype!r})"
        )
