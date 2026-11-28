"""The bricks of the 2D power spectrum: its plan, its filters, the four estimators of STL v1.6.0
and the automatic choice (maths.md §1.9, "Estimators", "Choice of estimator"; Part II, §12.8;
tasks T4.1, T7.9).

The plan (:class:`SpectrumEstimatorPlan2D`, :func:`build_spectrum_plan_2d`) is the static
description of the spectrum that the plan of ``STOperator`` holds; it moved here from
``scattering/_plan.py`` with T7.9, so that the bricks of the spectrum live together.

Written on the L0 primitives (the ``fft`` extension of ``xp``, on the half plane of the real FFT
as in STL), functional and out of place, in float32 and float64, on every backend, and
differentiable to the second order: nothing depends on the values of the arrays, every table
is static, built in NumPy with the plan (``_bins.py``) and passed in the filters
(:class:`SpectrumEstimatorFilters2D`).

With x̂ = ``rfft2(x, norm="ortho")`` and the pairs (c₁, c₂), c₁ ≤ c₂, of the chunks of ``"PS"``:

* :func:`estimate_fft_2d`: PS_b = Σ_m w(m) W_b(round(|m|)) Re(x̂_{c₁}(m) conj(x̂_{c₂}(m))) / Z_b,
  over the modes m = (k₁, k₂) of the half plane, which is Σ_k N_k W_b(k) P(k) / Z_b with the
  radial average P(k) of §1.9, the sum over the modes of each radial frequency taken once;
* the band maps f_b = ``irfft2(x̂ √W_b)`` and the weighted sums of their products:
  :func:`estimate_pbc_mask_2d` (the hard border mask M_b), :func:`estimate_nan_mask_2d` (the
  observed pixels O, after the missing pixels of each map are filled with the mean of its valid
  pixels) and :func:`estimate_pbc_nan_mask_2d` (M_b O, after the same filling); each weight carries
  its prefactor N_pix / (Σ_n weight · Z_b).

Only the real part of the cross spectra is computed (OP-19). The missing pixels are static,
those of the mask of the data, one map shared by every batch element and channel, read when
the operator is built, as for ``STOperator`` (OP-98).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend._protocol import Backend
from ..data._base import Data
from ..statistics._arrays import constant
from ._bins import SpectrumBins2D, log_gaussian_bins_2d

ESTIMATORS = ("fft", "pbc_mask", "nan_mask", "pbc_nan_mask")
"""The estimators of maths.md §1.9."""
METHODS = ("auto", *ESTIMATORS)
"""The values of ``spectrum_method`` of ``STOperator`` (maths.md §1.9)."""


def choose_estimator(method: object, pbc: bool, has_missing: bool) -> str:
    """The estimator of ``method`` for data with the periodicity ``pbc`` and, or not, missing
    pixels (maths.md §1.9, "Choice of estimator"). ``"auto"``: ``pbc_nan_mask`` with missing
    pixels and ``pbc=False``, ``nan_mask`` with missing pixels otherwise, ``pbc_mask`` without
    and ``pbc=False``, ``fft`` otherwise. A forced ``fft`` or ``pbc_mask`` on data with missing
    pixels raises, at construction, where STL raises at ``apply`` (OP-98)."""
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(
            f"spectrum method {method!r}: pass one of {', '.join(repr(m) for m in METHODS)} "
            "(maths.md §1.9)."
        )
    if method == "auto":
        if has_missing:
            return "nan_mask" if pbc else "pbc_nan_mask"
        return "fft" if pbc else "pbc_mask"
    if method in ("fft", "pbc_mask") and has_missing:
        raise ValueError(
            f"the spectrum estimator {method!r} does not accept missing pixels, and these data "
            'have some (maths.md §1.9): pass "nan_mask" or "pbc_nan_mask", or "auto".'
        )
    return method


# ---------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class SpectrumEstimatorPlan2D:
    """The static description of the 2D power spectrum (maths.md §1.9, "Range", "Number of
    bins", "Choice of estimator"; tasks T4.4, T4.1): how many bins, J, J_min, the estimator
    chosen at construction, and the spatial shape whose bins, windows and borders it reads
    (:meth:`bins`, ``spectra._bins.log_gaussian_bins_2d``, built once in NumPy). Hashable. In the
    plan of ``STOperator`` its channel pairs are those of the plan
    (``STPlanSingleDomain.pairs``): the pairs c1 ≤ c2 of S1 and S2, one matrix for the
    scattering statistics and the spectrum (M-CHAN; decision on #148); σᴾˢ exists for the
    channels of ``pairs.auto`` (M-NORM, OP-30)."""

    n_bins: int
    """n_b, the number of bins: given, or max(4, round(3 (J - 1 - J_min)))."""
    J: int
    """J, the J of the operator: the J given to it, or ⌊log₂ N_max⌋ - 1, equal to the default J
    of the scattering statistics (M-DEF)."""
    Jmin: int
    """J_min, with 0 ≤ J_min ≤ J - 2 (OP-93)."""
    method: str
    """The estimator, one of :data:`ESTIMATORS`, chosen at construction from ``pbc`` and the
    missing pixels of the example, or given (maths.md §1.9)."""
    spatial_shape: tuple[int, ...]
    """(N₁, N₂), even."""
    n_valid: int
    """The number of valid pixels of the data, N₁ N₂ without missing pixels: the denominator of
    the mean that fills the missing pixels of each map (``nan_mask``, ``pbc_nan_mask``)."""

    def content(self) -> dict[str, Any]:
        """The parameters of the spectrum for ``estel.io``."""
        return {"n_bins": self.n_bins, "J": self.J, "Jmin": self.Jmin, "method": self.method}

    def bins(self) -> SpectrumBins2D:
        """The bins, windows, normalisations and borders of the spectrum (maths.md §1.9)."""
        n1, n2 = self.spatial_shape
        return log_gaussian_bins_2d((int(n1), int(n2)), self.J, self.Jmin, self.n_bins)


def spectrum_only_2d(received: str, advice: str) -> str:
    """The one message that refuses the power spectrum of data that are not 2D, the same for
    every geometry (Part II, §12.6; the amendment to decision 2 on #167): ``received`` is the
    name of the class of the data, for display only, and ``advice`` what to do next (golden
    rule 8)."""
    return (
        f"the data are a {received}, which has no power spectrum in ESTEL 1.0.0: it computes "
        "power spectra of 2D data only; those of 1D data and of tensor products come later "
        f"(LF23), and those of HEALPix data too (LF24). {advice}"
    )


def build_spectrum_plan_2d(
    spatial_shape: tuple[int, ...],
    geometry: type[Data],
    J: int | None,  # noqa: N803 (maths.md notation)
    n_bins: Any,
    Jmin: Any,  # noqa: N803 (STL's name)
    method: Any = "auto",
    pbc: bool = True,
    mask: NDArray[np.bool_] | None = None,
) -> SpectrumEstimatorPlan2D:
    """The :class:`SpectrumEstimatorPlan2D` of 2D data of ``spatial_shape`` (maths.md §1.9),
    ``geometry`` the Data subclass of the data, whose declared tag (``Data._geometry``) must be 2D:
    the J of the spectrum is ``J``, the J given to the operator, or ⌊log₂ N_max⌋ - 1 with N_max =
    ⌊min(N₁, N₂)/2⌋ when it is ``None``; the estimator is ``method``, or with ``"auto"`` the choice
    of §1.9 from ``pbc`` and the missing pixels of ``mask``. Raises a clear error on data that are
    not 2D (LF23, LF24), on an odd size (OP-20), when J_min is not between 0 and J - 2 (OP-93), on a
    number of bins that is not a positive integer (a ``TypeError`` when ``Jmin`` or ``n_bins`` is
    not an integer, a ``ValueError`` when it is out of range), on an unknown ``method``, on ``fft``
    or ``pbc_mask`` with missing pixels, and on a bin of ``pbc_nan_mask`` without an observed pixel
    inside its border. The errors that the data or the options cause say how to leave the spectrum
    out (``compute_PS=False``, golden rule 8)."""
    if geometry._geometry != "2d":
        raise ValueError(spectrum_only_2d(geometry.__name__, "pass compute_PS=False."))
    if any(n % 2 for n in spatial_shape):
        raise ValueError(
            f"the power spectrum requires even sizes, and the data are {tuple(spatial_shape)} "
            "(maths.md §1.9, OP-20): crop the data to even sizes, or pass compute_PS=False."
        )
    n_max = min(spatial_shape) // 2
    j_ps = int(np.floor(np.log2(n_max))) - 1 if J is None else int(J)
    if isinstance(Jmin, bool) or not isinstance(Jmin, int | np.integer):
        raise TypeError(f"Jmin={Jmin!r}: pass an integer, 0 by default (maths.md §1.9).")
    if not 0 <= int(Jmin) <= j_ps - 2:
        raise ValueError(
            f"Jmin={Jmin} with J = {j_ps}: the power spectrum needs 0 ≤ Jmin ≤ J - 2 (maths.md "
            "§1.9, OP-93). Pass compute_PS=False, a smaller Jmin, or use a larger image (a larger "
            "J)."
        )
    bins_advice = (
        "pass a positive integer, or None for max(4, round(3 (J - 1 - Jmin))) bins, about three "
        "per octave (maths.md §1.9)."
    )
    if n_bins is None:
        found = max(4, round(3 * (j_ps - 1 - int(Jmin))))
    elif isinstance(n_bins, bool) or not isinstance(n_bins, int | np.integer):
        raise TypeError(f"n_bins={n_bins!r}: {bins_advice}")
    elif n_bins < 1:
        raise ValueError(f"n_bins={n_bins!r}: {bins_advice}")
    else:
        found = int(n_bins)
    estimator = choose_estimator(method, bool(pbc), mask is not None and bool(mask.any()))
    plan = SpectrumEstimatorPlan2D(
        n_bins=found,
        J=j_ps,
        Jmin=int(Jmin),
        method=estimator,
        spatial_shape=tuple(int(n) for n in spatial_shape),
        n_valid=int(np.prod(spatial_shape)) if mask is None else int((~mask).sum()),
    )
    if estimator == "pbc_nan_mask":
        bins = plan.bins()
        static_tables_2d(  # raises on a bin without an observed pixel inside its border
            estimator,
            bins.mode_windows(),
            bins.mode_weights,
            bins.norm,
            bins.hard_masks(),
            None if mask is None else ~mask,
        )
    return plan


# ---------------------------------------------------------------------- the filters


class SpectrumEstimatorFilters2D(NamedTuple):
    """The static arrays of the spectrum on the backend, in the operator's dtype (maths.md
    §1.9): built once by :func:`build_spectrum_filters_2d`. A named tuple, so a pytree whose
    leaves are the arrays."""

    modes: Any
    """``fft``: ``(n_b, N₁ · (N₂/2 + 1))``, w(m) W_b(round(|m|)) / Z_b at each mode m = (k₁, k₂).
    Otherwise: ``(n_b, N₁, N₂/2 + 1)``, √W_b(round(|m|)), 0 beyond N_max."""
    pixels: Any
    """The band estimators: ``(n_b, N₁, N₂)``, the weight of each pixel with its prefactor,
    M_b N_pix / (Σ_n M_b · Z_b), or with O in place of (or times) M_b; ``None`` for ``fft``."""
    valid: Any
    """``nan_mask`` and ``pbc_nan_mask`` on data with missing pixels: ``(N₁, N₂)``, the valid
    pixels (bool); ``None`` otherwise. The denominator of the filling mean, their number, is in
    the plan (``SpectrumEstimatorPlan2D.n_valid``)."""


def static_tables_2d(
    estimator: str,
    modes: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    norm: NDArray[np.float64],
    hard_masks: NDArray[np.float64],
    valid: NDArray[np.bool_] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """The two float tables of :class:`SpectrumEstimatorFilters2D` for ``estimator``, in NumPy
    float64: ``modes`` is W_b at each mode of the half plane (``SpectrumBins2D.mode_windows``),
    ``mode_weights`` w(k₁, k₂), ``norm`` Z_b, ``hard_masks`` the M_b ``(n_b, N₁, N₂)``, ``valid``
    the valid pixels (``None`` without missing pixels). Raises when a bin has no observed pixel
    inside its border (``pbc_nan_mask``; maths.md §1.9), at construction, where STL raises at
    ``apply``."""
    n_bins = int(modes.shape[0])
    if estimator == "fft":
        found = mode_weights[None] * modes / norm[:, None, None]
        return found.reshape(n_bins, -1), None
    weights = hard_masks if estimator in ("pbc_mask", "pbc_nan_mask") else np.ones_like(hard_masks)
    if estimator in ("nan_mask", "pbc_nan_mask") and valid is not None:
        weights = weights * valid.astype(np.float64)[None]
    totals = weights.sum(axis=(1, 2))
    empty = np.flatnonzero(totals == 0.0)
    if empty.size:
        raise ValueError(
            f"spectrum estimator {estimator!r}: bin(s) {empty.tolist()} have no observed pixel "
            "inside their border, so there is nothing to average (maths.md §1.9); use larger "
            'data, fewer missing pixels, or the estimator "nan_mask".'
        )
    n_pix = float(weights.shape[1] * weights.shape[2])
    pixels = weights * (n_pix / (totals * norm))[:, None, None]
    return np.sqrt(modes), pixels


def build_spectrum_filters_2d(
    spectrum: Any,
    mask: NDArray[np.bool_] | None,
    backend: Backend,
    dtype: FloatDType,
    device: Any,
) -> SpectrumEstimatorFilters2D:
    """The :class:`SpectrumEstimatorFilters2D` of ``spectrum`` (:class:`SpectrumEstimatorPlan2D`)
    for data with the missing pixels ``mask`` (True = missing, or ``None``), on ``backend`` in
    ``dtype`` on ``device``: cast once (Part I, §3.1, rules 4 and 7)."""
    bins = spectrum.bins()
    estimator = spectrum.method
    valid = None if mask is None else ~np.asarray(mask, dtype=bool)
    modes, pixels = static_tables_2d(
        estimator,
        bins.mode_windows(),
        bins.mode_weights,
        bins.norm,
        bins.hard_masks() if estimator != "fft" else np.zeros((0, 1, 1)),
        valid,
    )

    def real(a: NDArray[Any] | None) -> Any:
        return None if a is None else backend.asarray(np.asarray(a, np.float64), dtype, device)

    filling = estimator in ("nan_mask", "pbc_nan_mask") and valid is not None
    return SpectrumEstimatorFilters2D(
        modes=real(modes),
        pixels=real(pixels),
        valid=backend.asarray(valid, "bool", device) if filling else None,
    )


# ---------------------------------------------------------------------- the estimators


def _pair_index(backend: Backend, pairs: tuple[tuple[int, int], ...], like: Any) -> tuple[Any, Any]:
    first = np.array([c1 for c1, _ in pairs], dtype=np.int64)
    second = np.array([c2 for _, c2 in pairs], dtype=np.int64)
    return constant(backend, first, like, "int32"), constant(backend, second, like, "int32")


def estimate_fft_2d(
    backend: Backend,
    x: Any,
    filters: SpectrumEstimatorFilters2D,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``fft`` (periodic, no missing pixel): ``(Nb, n_pairs, n_b)``, the spectrum of each pair
    of ``pairs`` of the field ``x`` ``(Nb, Nc, N₁, N₂)``. The estimators share one signature:
    ``n_valid``, the denominator of the filling mean, is read by those that fill."""
    xp = backend.xp
    spectrum = xp.fft.rfft2(x, norm="ortho")  # (Nb, Nc, N1, K2)
    first, second = _pair_index(backend, pairs, filters.modes)
    a = xp.take(spectrum, first, axis=1)
    b = xp.take(spectrum, second, axis=1)
    product = xp.real(a) * xp.real(b) + xp.imag(a) * xp.imag(b)  # Re(a conj(b))
    n_batch, n_pairs = int(product.shape[0]), int(product.shape[1])
    rows = xp.reshape(product, (n_batch, n_pairs, -1))
    return xp.matmul(rows, xp.permute_dims(filters.modes, (1, 0)))


def _bands_2d(
    backend: Backend, x: Any, filters: SpectrumEstimatorFilters2D, shape: tuple[int, int]
) -> Any:
    """The band maps f_b of every map: ``(Nb, n_b, Nc, N₁, N₂)``, the bin before the channel, so
    that the Gram product of :func:`_band_sums_2d` needs no copy."""
    xp = backend.xp
    spectrum = xp.fft.rfft2(x, norm="ortho")
    return xp.fft.irfft2(spectrum[:, None] * filters.modes[:, None], s=shape, norm="ortho")


def _filled_2d(backend: Backend, x: Any, filters: SpectrumEstimatorFilters2D, n_valid: int) -> Any:
    """Each map with its missing pixels set to the mean of its ``n_valid`` valid pixels (maths.md
    §1.9, ``nan_mask``), with ``where``: the values at the missing pixels are never read."""
    xp = backend.xp
    valid = filters.valid
    if valid is None:  # no missing pixel: nothing to fill
        return x
    kept = xp.where(valid, x, xp.zeros_like(x))
    mean = xp.sum(kept, axis=(-2, -1), keepdims=True) / n_valid
    return xp.where(valid, kept, xp.broadcast_to(mean, x.shape))


def _band_sums_2d(
    backend: Backend,
    x: Any,
    filters: SpectrumEstimatorFilters2D,
    pairs: tuple[tuple[int, int], ...],
) -> Any:
    """Σ_n P_b(n) f_{b,c₁}(n) f_{b,c₂}(n) for each pair of ``pairs``: ``(Nb, n_pairs, n_b)``. In
    the Gram form, as the second layer (Part III, §19.3): per batch element and bin, one matrix
    product of the weighted band maps of every channel with the band maps, ``(Nc, Nc)``, then
    the entries of ``pairs``; no copy of the band maps per pair."""
    xp = backend.xp
    n1, n2 = int(x.shape[-2]), int(x.shape[-1])
    n_batch, n_channels = int(x.shape[0]), int(x.shape[1])
    n_bins = int(filters.pixels.shape[0])
    bands = _bands_2d(backend, x, filters, (n1, n2))  # (Nb, n_b, Nc, N1, N2)
    weighted = xp.reshape(bands * filters.pixels[:, None], (n_batch, n_bins, n_channels, -1))
    rows = xp.reshape(bands, (n_batch, n_bins, n_channels, -1))
    gram = xp.matmul(weighted, xp.permute_dims(rows, (0, 1, 3, 2)))  # (Nb, n_b, Nc, Nc)
    flat = np.array([c1 * n_channels + c2 for c1, c2 in pairs], dtype=np.int64)
    index = constant(backend, flat, filters.pixels, "int32")
    found = xp.take(xp.reshape(gram, (n_batch, n_bins, -1)), index, axis=2)
    return xp.permute_dims(found, (0, 2, 1))  # (Nb, n_pairs, n_b)


def estimate_pbc_mask_2d(
    backend: Backend,
    x: Any,
    filters: SpectrumEstimatorFilters2D,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``pbc_mask`` (non-periodic, no missing pixel): ``(Nb, n_pairs, n_b)``."""
    return _band_sums_2d(backend, x, filters, pairs)


def estimate_nan_mask_2d(
    backend: Backend,
    x: Any,
    filters: SpectrumEstimatorFilters2D,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``nan_mask`` (periodic, missing pixels): ``(Nb, n_pairs, n_b)``."""
    return _band_sums_2d(backend, _filled_2d(backend, x, filters, n_valid), filters, pairs)


def estimate_pbc_nan_mask_2d(
    backend: Backend,
    x: Any,
    filters: SpectrumEstimatorFilters2D,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``pbc_nan_mask`` (non-periodic, missing pixels): ``(Nb, n_pairs, n_b)``."""
    return _band_sums_2d(backend, _filled_2d(backend, x, filters, n_valid), filters, pairs)


_ESTIMATE_2D = {
    "fft": estimate_fft_2d,
    "pbc_mask": estimate_pbc_mask_2d,
    "nan_mask": estimate_nan_mask_2d,
    "pbc_nan_mask": estimate_pbc_nan_mask_2d,
}


def spectrum_floats_2d(
    backend: Backend,
    x: Any,
    spectrum: Any,
    filters: SpectrumEstimatorFilters2D,
    pairs: tuple[tuple[int, int], ...],
) -> Any:
    """The floats of the family ``"PS"`` in a buffer, ``(Nb, n_pairs · n_b)``: the spectrum of
    each pair of ``pairs``, in this order (that of the chunks of ``"PS"`` in the layout), each
    followed by its n_b bins, one float each. ``x`` is the field ``(Nb, Nc, N₁, N₂)``, its missing
    pixels never read; ``spectrum`` is the plan of the spectrum (:class:`SpectrumEstimatorPlan2D`):
    its estimator and the denominator of the filling mean."""
    found = _ESTIMATE_2D[spectrum.method](backend, x, filters, pairs, spectrum.n_valid)
    return backend.xp.reshape(found, (int(found.shape[0]), -1))
