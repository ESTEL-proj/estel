"""The four estimators of the 2D power spectrum of STL v1.6.0 and the automatic choice
(maths.md §1.9, "Estimators", "Choice of estimator"; Part II, §12.8; task T4.1).

Written on the L0 primitives (the ``fft`` extension of ``xp``, on the half plane of the real FFT
as in STL), functional and out of place, in float32 and float64, on every backend, and
differentiable to the second order: nothing depends on the values of the arrays, every table
is static, built in NumPy with the plan (``_bins.py``) and passed in the filters
(:class:`SpectrumFilters`).

With x̂ = ``rfft2(x, norm="ortho")`` and the pairs (c₁, c₂), c₁ ≤ c₂, of the chunks of ``"PS"``:

* :func:`estimate_fft`: PS_b = Σ_m w(m) W_b(round(|m|)) Re(x̂_{c₁}(m) conj(x̂_{c₂}(m))) / Z_b,
  over the modes m = (k₁, k₂) of the half plane, which is Σ_k N_k W_b(k) P(k) / Z_b with the
  radial average P(k) of §1.9, the sum over the modes of each radial frequency taken once;
* the band maps f_b = ``irfft2(x̂ √W_b)`` and the weighted sums of their products:
  :func:`estimate_pbc_mask` (the hard border mask M_b), :func:`estimate_nan_mask` (the observed
  pixels O, after the missing pixels of each map are filled with the mean of its valid pixels)
  and :func:`estimate_pbc_nan_mask` (M_b O, after the same filling); each weight carries its
  prefactor N_pix / (Σ_n weight · Z_b).

Only the real part of the cross spectra is computed (OP-19). The missing pixels are static,
those of the mask of the data, one map shared by every batch element and channel, read when
the operator is built, as for ``STOperator`` (OP-98).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import FloatDType
from ..backend._protocol import Backend
from ..statistics._arrays import constant

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


class SpectrumFilters(NamedTuple):
    """The static arrays of the spectrum on the backend, in the operator's dtype (maths.md
    §1.9): built once by ``spectra._plan.build_spectrum_filters``. A named tuple, so a pytree
    whose leaves are the arrays."""

    modes: Any
    """``fft``: ``(n_b, N₁ · (N₂/2 + 1))``, w(m) W_b(round(|m|)) / Z_b at each mode m = (k₁, k₂).
    Otherwise: ``(n_b, N₁, N₂/2 + 1)``, √W_b(round(|m|)), 0 beyond N_max."""
    pixels: Any
    """The band estimators: ``(n_b, N₁, N₂)``, the weight of each pixel with its prefactor,
    M_b N_pix / (Σ_n M_b · Z_b), or with O in place of (or times) M_b; ``None`` for ``fft``."""
    valid: Any
    """``nan_mask`` and ``pbc_nan_mask`` on data with missing pixels: ``(N₁, N₂)``, the valid
    pixels (bool); ``None`` otherwise. The denominator of the filling mean, their number, is in
    the plan (``SpectrumPlan.n_valid``)."""


def static_tables(
    estimator: str,
    modes: NDArray[np.float64],
    mode_weights: NDArray[np.float64],
    norm: NDArray[np.float64],
    hard_masks: NDArray[np.float64],
    valid: NDArray[np.bool_] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """The two float tables of :class:`SpectrumFilters` for ``estimator``, in NumPy float64:
    ``modes`` is W_b at each mode of the half plane (``SpectrumBins.mode_windows``),
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


def build_spectrum_filters(
    spectrum: Any,
    mask: NDArray[np.bool_] | None,
    backend: Backend,
    dtype: FloatDType,
    device: Any,
) -> SpectrumFilters:
    """The :class:`SpectrumFilters` of ``spectrum`` (``scattering._plan.SpectrumPlan``) for data
    with the missing pixels ``mask`` (True = missing, or ``None``), on ``backend`` in ``dtype``
    on ``device``: cast once (Part I, §3.1, rules 4 and 7)."""
    bins = spectrum.bins()
    estimator = spectrum.method
    valid = None if mask is None else ~np.asarray(mask, dtype=bool)
    modes, pixels = static_tables(
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
    return SpectrumFilters(
        modes=real(modes),
        pixels=real(pixels),
        valid=backend.asarray(valid, "bool", device) if filling else None,
    )


# ---------------------------------------------------------------------- the estimators


def _pair_index(backend: Backend, pairs: tuple[tuple[int, int], ...], like: Any) -> tuple[Any, Any]:
    first = np.array([c1 for c1, _ in pairs], dtype=np.int64)
    second = np.array([c2 for _, c2 in pairs], dtype=np.int64)
    return constant(backend, first, like, "int32"), constant(backend, second, like, "int32")


def estimate_fft(
    backend: Backend,
    x: Any,
    filters: SpectrumFilters,
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


def _bands(backend: Backend, x: Any, filters: SpectrumFilters, shape: tuple[int, int]) -> Any:
    """The band maps f_b of every map: ``(Nb, Nc, n_b, N₁, N₂)``."""
    xp = backend.xp
    spectrum = xp.fft.rfft2(x, norm="ortho")
    return xp.fft.irfft2(spectrum[:, :, None] * filters.modes, s=shape, norm="ortho")


def _filled(backend: Backend, x: Any, filters: SpectrumFilters, n_valid: int) -> Any:
    """Each map with its missing pixels set to the mean of its ``n_valid`` valid pixels (maths.md
    §1.9, ``nan_mask``), with ``where``: the values at the missing pixels are never read."""
    xp = backend.xp
    valid = filters.valid
    if valid is None:  # no missing pixel: nothing to fill
        return x
    kept = xp.where(valid, x, xp.zeros_like(x))
    mean = xp.sum(kept, axis=(-2, -1), keepdims=True) / n_valid
    return xp.where(valid, kept, xp.broadcast_to(mean, x.shape))


def _band_sums(
    backend: Backend, x: Any, filters: SpectrumFilters, pairs: tuple[tuple[int, int], ...]
) -> Any:
    xp = backend.xp
    shape = (int(x.shape[-2]), int(x.shape[-1]))
    bands = _bands(backend, x, filters, shape)
    first, second = _pair_index(backend, pairs, filters.pixels)
    products = xp.take(bands, first, axis=1) * xp.take(bands, second, axis=1)
    return xp.sum(products * filters.pixels, axis=(-2, -1))  # (Nb, n_pairs, n_b)


def estimate_pbc_mask(
    backend: Backend,
    x: Any,
    filters: SpectrumFilters,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``pbc_mask`` (non-periodic, no missing pixel): ``(Nb, n_pairs, n_b)``."""
    return _band_sums(backend, x, filters, pairs)


def estimate_nan_mask(
    backend: Backend,
    x: Any,
    filters: SpectrumFilters,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``nan_mask`` (periodic, missing pixels): ``(Nb, n_pairs, n_b)``."""
    return _band_sums(backend, _filled(backend, x, filters, n_valid), filters, pairs)


def estimate_pbc_nan_mask(
    backend: Backend,
    x: Any,
    filters: SpectrumFilters,
    pairs: tuple[tuple[int, int], ...],
    n_valid: int,
) -> Any:
    """``pbc_nan_mask`` (non-periodic, missing pixels): ``(Nb, n_pairs, n_b)``."""
    return _band_sums(backend, _filled(backend, x, filters, n_valid), filters, pairs)


_ESTIMATE = {
    "fft": estimate_fft,
    "pbc_mask": estimate_pbc_mask,
    "nan_mask": estimate_nan_mask,
    "pbc_nan_mask": estimate_pbc_nan_mask,
}


def spectrum_floats(
    backend: Backend,
    x: Any,
    spectrum: Any,
    filters: SpectrumFilters,
    pairs: tuple[tuple[int, int], ...],
) -> Any:
    """The floats of the family ``"PS"`` in a buffer, ``(Nb, n_pairs · n_b)``: the spectrum of
    each pair of ``pairs``, in this order (that of the chunks of ``"PS"`` in the layout), each
    followed by its n_b bins, one float each. ``x`` is the field ``(Nb, Nc, N₁, N₂)``, its missing
    pixels never read; ``spectrum`` is the plan of the spectrum (``scattering._plan.SpectrumPlan``):
    its estimator and the denominator of the filling mean."""
    found = _ESTIMATE[spectrum.method](backend, x, filters, pairs, spectrum.n_valid)
    return backend.xp.reshape(found, (int(found.shape[0]), -1))
