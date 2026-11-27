"""Moments over valid pixels, the safe modulus and STL's regularised division (Part II, §12.4;
maths.md M-NOT, M-MEAN, M-MOM, M-S1; task T2.4).

The internal functions of the mother class Data that the scattering loop calls: one
implementation for every geometry, written on the L0 protocol, functional and differentiable to
second order (Part I, §3.2.3):

* :func:`modulus`: |z| with the double ``where`` of ``docs/agents/differentiation.md``, so that
  exact zeros (zero-filled missing pixels, constant regions) give no NaN in the gradients;
* :func:`divide`: STL's ``divide(x, y, epsilon, pow)``, ``x / (y + ε)^pow``, used by the cross
  S1 terms with ``y = |x|``, ``ε = 1e-8`` and ``pow = 1/2`` (M-S1, OP-16);
* :func:`sum_valid` and :func:`cov_sum_valid`: the weighted sums, before the division, which the
  loop adds across the partitions of the data (``_global_sum``) and then divides;
* :func:`gram_sum`: the sums over the pixels of every product of two maps of a set, as one
  matrix product (maths_step2.md M-GRAM): the second-layer sums of a level;
* :func:`mean_valid` and :func:`cov_valid`: the mean of M-NOT, ``sum(w · a) / sum(w)`` over the
  spatial axes, with the weights and the denominator of the plan (``MeanWeights``): missing
  pixels and border crops are weights, never boolean indexing (Part I, §3.1.1). ``cov_valid``
  is the raw second moment ``⟨x · conj(y)⟩``, as STL's ``cov`` (M-MOM, OP-11, OP-36);
* :func:`standardize`: STL's ``standardize`` with ``mean_field=False`` (M-MOM).

The weights are 0 wherever a pixel does not enter the mean, so the values there must be finite:
the loop fills the missing pixels of the field with 0 before anything else (OP-27).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from ..backend import _registry
from ..backend._protocol import Backend


def _backend(x: Any) -> Backend:
    """The backend of the array ``x``."""
    name = _registry.backend_of(x)
    if name is None:
        raise TypeError(
            f"expected an array of a backend of ESTEL, got {type(x).__name__}: convert NumPy "
            "inputs at the boundary, with a Data object."
        )
    return _registry.backend_named(name)


def _is_complex(xp: Any, x: Any) -> bool:
    return bool(xp.isdtype(x.dtype, "complex floating"))


def modulus(x: Any) -> Any:
    """|x|, real, with the safe double ``where`` (Part I, §3.2.3, rule 3).

    ``m = |x|² > 0; |x| = where(m, sqrt(where(m, |x|², 1)), 0 · |x|²)``: the values are those
    of ``abs`` up to rounding, and the first and second derivatives are finite at ``x = 0``
    (zero there), where those of ``abs`` are not. The branch ``0 · |x|²`` is 0, with zero first
    and second derivatives, wherever ``x`` is finite, and NaN where ``x`` is NaN, as ``abs``:
    a NaN field (a map of zero variance, standardised) gives NaN statistics, not 0. Works on
    real and complex arrays; the result has the real dtype matching ``x``.
    """
    xp = _backend(x).xp
    if _is_complex(xp, x):
        re, im = xp.real(x), xp.imag(x)
        square = re * re + im * im
    else:
        square = x * x
    positive = square > 0
    root = xp.sqrt(xp.where(positive, square, xp.ones_like(square)))
    return xp.where(positive, root, 0 * square)


def divide(x: Any, y: Any, eps: float = 1e-8, pow: float = 1.0) -> Any:
    """``x / (y + eps)^pow``, STL's regularised division (``divide(epsilon, pow)``; M-S1, OP-16).

    ``y`` is real and non-negative (a modulus), so ``y + eps > 0`` and the power is smooth: the
    result is differentiable to any order. ``x`` may be complex.
    """
    xp = _backend(x).xp
    if _is_complex(xp, y):
        raise TypeError("divide: the denominator is a modulus, real and non-negative.")
    return x / (y + eps) ** pow


def crop(x: Any, window: Any) -> Any:
    """``x`` on the pixels of a window that is a rectangle: ``window`` is a tuple of slices,
    one per trailing (spatial) axis of ``x``. A view, no weights (maths_step2.md M-WIN)."""
    return x[(Ellipsis, *window)]


def sum_valid(x: Any, weights: Any, axes: Sequence[int]) -> Any:
    """The sum of ``x`` over the pixels of a window, along ``axes``, the spatial axes: the
    numerator of :func:`mean_valid`, before the division by the denominator.

    ``weights`` is the window, in one of its three kinds (maths_step2.md M-WIN; Part III,
    §19.12, point 1): ``None``, every pixel, a plain sum; a tuple of slices, one per spatial
    axis, a sum over that slice; or the 0/1 array of the plan, ``sum(weights · x)``, for data
    with missing values. The three give the same value up to the order of the sum.

    The scattering loop sums first, adds the partial sums of every partition of the data (the
    identity in 1.0.0; ``estel.scattering._loop._global_sum``), then divides.
    """
    xp = _backend(x).xp
    if weights is None:
        return xp.sum(x, axis=tuple(axes))
    if isinstance(weights, tuple):
        return xp.sum(crop(x, weights), axis=tuple(axes))
    return xp.sum(weights * x, axis=tuple(axes))


def cov_sum_valid(x: Any, y: Any, weights: Any, axes: Sequence[int]) -> Any:
    """The sum of ``x · conj(y)`` over the pixels of a window (see :func:`sum_valid`): the
    numerator of :func:`cov_valid`. ``x`` and ``y`` have the same spatial shape."""
    xp = _backend(x).xp
    if isinstance(weights, tuple):
        x, y, weights = crop(x, weights), crop(y, weights), None
    product = x * xp.conj(y) if _is_complex(xp, y) else x * y
    return sum_valid(product, weights, axes)


GRAM_BLOCK = 16384
"""The number of pixels of a block of :func:`gram_sum` in single precision."""


def gram_sum(a: Any, b: Any = None) -> Any:
    """The sums over the pixels of the products of the maps of ``a`` with the conjugates of
    those of ``b``, as a matrix product: ``G[..., p, q] = Σ_n a[..., p, n] · conj(b[..., q, n])``,
    that is ``a b^H`` (maths_step2.md M-GRAM). Without ``b``, the Gram matrix ``a a^H``.

    ``a`` is ``(..., P, n)`` and ``b`` ``(..., Q, n)``, complex, with their pixels flattened on
    the last axis; the result is ``(..., P, Q)``, in the dtype of ``a``. The maps are already
    zero outside their windows: no weight enters the product, and the sum of a pair runs over
    the intersection of its two windows.

    **Single precision.** One product over all the pixels would accumulate in the precision of
    the data. In single precision the pixels are cut in blocks of :data:`GRAM_BLOCK`, each block
    is one product, and the products are added in double precision, then cast back. The blocks
    are cut with ``unstack``, one operation whose derivative fills the gradient once, where a
    slice per block would fill it once per block. In double precision, and on a backend without
    double precision (JAX in its 32-bit mode), it is one product.
    """
    xp = _backend(a).xp
    other = a if b is None else b

    def product(x: Any, y: Any) -> Any:
        return xp.matmul(x, xp.conj(xp.matrix_transpose(y)))

    wide = xp.result_type(xp.complex128)  # complex64 on JAX in its 32-bit mode
    n = int(a.shape[-1])
    if a.dtype == wide or n <= GRAM_BLOCK:
        return product(a, other)
    count = n // GRAM_BLOCK
    head = count * GRAM_BLOCK

    def blocks(x: Any) -> list[Any]:
        whole = x if head == n else x[..., :head]
        cut = xp.reshape(whole, (*tuple(x.shape[:-1]), count, GRAM_BLOCK))
        tail = [] if head == n else [x[..., head:]]
        return [*xp.unstack(cut, axis=-2), *tail]

    of_a = blocks(a)
    of_b = of_a if b is None else blocks(b)
    parts = [xp.astype(product(x, y), wide) for x, y in zip(of_a, of_b, strict=True)]
    return xp.astype(xp.sum(xp.stack(parts, axis=0), axis=0), a.dtype)


def mean_valid(x: Any, weights: Any, denominator: float, axes: Sequence[int]) -> Any:
    """The mean of ``x`` over the valid pixels: ``sum(weights · x, axes) / denominator``
    (maths.md M-NOT, M-MEAN).

    ``weights`` is the window of the mean (:func:`sum_valid`): the real array of the plan (1
    where a pixel enters the mean, 0 on the border crop and on invalid pixels;
    ``MeanWeights.weights``), broadcast against the trailing ``axes`` of ``x``, or, for a
    window without missing values, a tuple of slices or ``None``; ``denominator`` is the number
    of its pixels, a Python float computed once in the plan
    (``MeanWeights.denominator``). ``x`` must be finite where the weights are 0.
    """
    return sum_valid(x, weights, axes) / denominator


def cov_valid(x: Any, y: Any, weights: Any, denominator: float, axes: Sequence[int]) -> Any:
    """The raw second moment ``⟨x · conj(y)⟩`` over the valid pixels (maths.md M-NOT, M-MOM).

    STL's ``cov``, which subtracts no mean (OP-11, OP-36): M2(x, y), not the covariance. The
    centred covariance of step 2 adds first moments (LF33, Part III, §20.11). ``x`` and ``y``
    broadcast together; for a real ``y``, ``conj(y) = y``.
    """
    return cov_sum_valid(x, y, weights, axes) / denominator


def _identity(x: Any) -> Any:
    return x


class Standardized(NamedTuple):
    """What :func:`standardize` returns, and the standardisation of the target of a synthesis
    (``synthesis._init.standardize_target``)."""

    x: Any
    """The standardised data, ``(x - mean) / std``."""
    mean: Any
    """The mean removed, ``(Nb, Nc)``; ``(1, Nc)`` for a batch mean (a synthesis with
    ``mean_field``)."""
    std: Any
    """The standard deviation divided by, of the shape of ``mean``."""


def standardize(
    x: Any,
    weights: Any,
    denominator: float,
    axes: Sequence[int],
    global_sum: Callable[[Any], Any] | None = None,
) -> Standardized:
    """Remove the mean of each map and divide by its standard deviation (maths.md M-MOM; STL
    ``standardize`` with ``mean_field=False``).

    ``x`` is ``(Nb, Nc, *spatial)``, real, with the full-resolution weights of the plan
    (``plan.moments``). μ = ⟨x⟩, then sigma² = ⟨(x - μ)²⟩, both over the valid pixels, with
    population normalisation; the result is (x - μ) / sigma. μ and sigma are ``(Nb, Nc)``.
    ``global_sum``, applied to each weighted sum before its division, adds the partial sums of
    the partitions of the data (``estel.scattering._loop._global_sum``); none by default.
    """
    xp = _backend(x).xp
    total = _identity if global_sum is None else global_sum
    expand = (Ellipsis, *([None] * len(axes)))
    mean = total(sum_valid(x, weights, axes)) / denominator
    centered = x - mean[expand]
    std = xp.sqrt(total(cov_sum_valid(centered, centered, weights, axes)) / denominator)
    return Standardized(x=centered / std[expand], mean=mean, std=std)
