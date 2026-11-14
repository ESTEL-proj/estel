"""Moments over valid pixels, the safe modulus and STL's regularised division (Part II, §12.4;
maths.md M-NOT, M-MEAN, M-MOM, M-S1; task T2.4).

The internal functions of the mother DataClass that the scattering loop calls: one
implementation for every geometry, written on the L0 protocol, functional and differentiable to
second order (Part I, §3.2.3):

* :func:`modulus`: |z| with the double ``where`` of ``docs/agents/differentiation.md``, so that
  exact zeros (zero-filled missing pixels, constant regions) give no NaN in the gradients;
* :func:`divide`: STL's ``divide(x, y, epsilon, pow)``, ``x / (y + ε)^pow``, used by the cross
  S1 terms with ``y = |x|``, ``ε = 1e-8`` and ``pow = 1/2`` (M-S1, OP-16);
* :func:`mean_valid` and :func:`cov_valid`: the mean of M-NOT, ``sum(w · a) / sum(w)`` over the
  spatial axes, with the weights and the denominator of the plan (``MeanWeights``): missing
  pixels and border crops are weights, never boolean indexing (Part I, §3.1.1). ``cov_valid``
  is the raw second moment ``⟨x · conj(y)⟩``, as STL's ``cov`` (M-MOM, OP-11, OP-36);
* :func:`standardize`: STL's ``standardize`` with ``mean_field=False`` (M-MOM).

The weights are 0 wherever a pixel does not enter the mean, so the values there must be finite:
the loop fills the missing pixels of the field with 0 before anything else (OP-27).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from ..backend import _registry
from ..backend._protocol import Backend


def _backend(x: Any) -> Backend:
    """The backend of the array ``x``."""
    name = _registry.backend_of(x)
    if name is None:
        raise TypeError(
            f"expected an array of a backend of ESTEL, got {type(x).__name__}: convert NumPy "
            "inputs at the boundary, with a DataClass."
        )
    return _registry.backend_named(name)


def _is_complex(xp: Any, x: Any) -> bool:
    return bool(xp.isdtype(x.dtype, "complex floating"))


def modulus(x: Any) -> Any:
    """|x|, real, with the safe double ``where`` (Part I, §3.2.3, rule 3).

    ``m = |x|² > 0; |x| = where(m, sqrt(where(m, |x|², 1)), 0)``: the values are those of
    ``abs`` up to rounding, and the first and second derivatives are finite at ``x = 0``
    (zero there), where those of ``abs`` are not. Works on real and complex arrays; the result
    has the real dtype matching ``x``.
    """
    xp = _backend(x).xp
    if _is_complex(xp, x):
        re, im = xp.real(x), xp.imag(x)
        square = re * re + im * im
    else:
        square = x * x
    positive = square > 0
    root = xp.sqrt(xp.where(positive, square, xp.ones_like(square)))
    return xp.where(positive, root, xp.zeros_like(square))


def divide(x: Any, y: Any, eps: float = 1e-8, pow: float = 1.0) -> Any:
    """``x / (y + eps)^pow``, STL's regularised division (``divide(epsilon, pow)``; M-S1, OP-16).

    ``y`` is real and non-negative (a modulus), so ``y + eps > 0`` and the power is smooth: the
    result is differentiable to any order. ``x`` may be complex.
    """
    xp = _backend(x).xp
    if _is_complex(xp, y):
        raise TypeError("divide: the denominator is a modulus, real and non-negative.")
    return x / (y + eps) ** pow


def mean_valid(x: Any, weights: Any, denominator: float, axes: Sequence[int]) -> Any:
    """The mean of ``x`` over the valid pixels: ``sum(weights · x, axes) / denominator``
    (maths.md M-NOT, M-MEAN).

    ``weights`` is the real array of the plan (1 where a pixel enters the mean, 0 on the border
    crop and on invalid pixels; ``MeanWeights.weights``), broadcast against the trailing
    ``axes`` of ``x``; ``denominator`` is its sum, a Python float computed once in the plan
    (``MeanWeights.denominator``). ``x`` must be finite where the weights are 0.
    """
    xp = _backend(x).xp
    return xp.sum(weights * x, axis=tuple(axes)) / denominator


def cov_valid(x: Any, y: Any, weights: Any, denominator: float, axes: Sequence[int]) -> Any:
    """The raw second moment ``⟨x · conj(y)⟩`` over the valid pixels (maths.md M-NOT, M-MOM).

    STL's ``cov``, which subtracts no mean (OP-11, OP-36): M2(x, y), not the covariance. The
    centred covariance of step 2 adds first moments (LF33, Part III, §20.11). ``x`` and ``y``
    broadcast together; for a real ``y``, ``conj(y) = y``.
    """
    xp = _backend(x).xp
    product = x * xp.conj(y) if _is_complex(xp, y) else x * y
    return mean_valid(product, weights, denominator, axes)


class Standardized(NamedTuple):
    """What :func:`standardize` returns."""

    x: Any
    """The standardised data, ``(x - mean) / std``."""
    mean: Any
    """The mean removed, ``(Nb, Nc)``."""
    std: Any
    """The standard deviation divided by, ``(Nb, Nc)``."""


def standardize(x: Any, weights: Any, denominator: float, axes: Sequence[int]) -> Standardized:
    """Remove the mean of each map and divide by its standard deviation (maths.md M-MOM; STL
    ``standardize`` with ``mean_field=False``).

    ``x`` is ``(Nb, Nc, *spatial)``, real, with the full-resolution weights of the plan
    (``plan.moments``). μ = ⟨x⟩, then sigma² = ⟨(x - μ)²⟩, both over the valid pixels, with
    population normalisation; the result is (x - μ) / sigma. μ and sigma are ``(Nb, Nc)``.
    """
    xp = _backend(x).xp
    expand = (Ellipsis, *([None] * len(axes)))
    mean = mean_valid(x, weights, denominator, axes)
    centred = x - mean[expand]
    std = xp.sqrt(cov_valid(centred, centred, weights, denominator, axes))
    return Standardized(x=centred / std[expand], mean=mean, std=std)
