"""The JVP and the Hessian-vector product of the scattering statistics through the L0
primitives, and the derivative through the normalisation reference (task T2.7, issue #19;
Part I, §3.2.2; maths.md M-NORM).

These tests run on both backends, through ``Backend.jvp``, ``vjp``, ``hvp`` and
``value_and_grad``, which downstream code uses; ``test_torch.py`` and ``test_jax.py`` run each
backend's own checker on the same configurations (``tests/differentiation/_cases.py``). All of
it is in float64 and compared with central finite differences.

Tolerances. With a step h along a direction of unit root mean square, a central difference
has a truncation error of h² times a third derivative and a round-off error of about 1e-16 / h
of the values: about 1e-10 for h = 1e-6. The comparisons are at ``rtol = 1e-6`` with
``atol = 1e-8``, far above this error and far below any missing term.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data1D
from estel.backend._protocol import Backend
from tests.differentiation._cases import (
    CASES,
    Case,
    coefficients,
    data_of,
    direction_like,
    operator_of,
    scattering_function,
)

STEP = 1e-6
RTOL, ATOL = 1e-6, 1e-8
NORMALIZED = Case("1d-open-cross", Data1D, (1, 2, 16), pbc=False, options={"L": 2})


def _central_difference(
    backend: Backend, function: Callable[[Any], Any], x: Any, v: Any, step: float = STEP
) -> NDArray[np.float64]:
    """(f(x + h v) - f(x - h v)) / 2h, as NumPy."""
    plus = backend.to_numpy(function(x + step * v))
    minus = backend.to_numpy(function(x - step * v))
    difference: NDArray[np.float64] = (plus - minus) / (2 * step)
    return difference


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_jvp_and_hvp_match_finite_differences(case: Case, backend: Backend) -> None:
    """``Backend.jvp`` of every coefficient, and ``Backend.hvp`` of a random linear combination
    of them, against central differences of the function and of its gradient.

    Slow: a Hessian-vector product and four more gradients through the whole loop (second
    order), on every configuration (Part I, §5.4)."""
    data, function = scattering_function(backend, case)
    function = backend.jit(function)  # on JAX, four times faster than op by op
    x = data.array
    v = direction_like(backend, x, seed=1)
    _, tangent = backend.jvp(function, (x,), (v,))
    np.testing.assert_allclose(
        backend.to_numpy(tangent),
        _central_difference(backend, function, x, v),
        rtol=RTOL,
        atol=ATOL,
    )
    weights = direction_like(backend, function(x), seed=2)

    def scalar(array: Any) -> Any:
        return backend.xp.sum(weights * function(array))

    def gradient(array: Any) -> Any:
        return backend.value_and_grad(scalar)(array)[1]

    np.testing.assert_allclose(
        backend.to_numpy(backend.hvp(scalar, x, v)),
        _central_difference(backend, gradient, x, v),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.slow
def test_norm_self_differentiates_the_reference_and_norm_ref_does_not(backend: Backend) -> None:
    """M-NORM. With ``norm="self"`` the derivative includes the one through the reference: it
    is the finite difference of the statistics normalised by their own, moving, reference.
    With ``norm="ref"`` the reference is a constant, even when it was computed from the
    differentiated array: the derivative is the finite difference of the statistics normalised
    by a reference that stays where it was. Both in forward and in reverse mode.

    Slow: the fast tier keeps one differentiation check, the gradcheck of the smallest case
    (issue #19); this one takes six more passes through the loop of two operators."""
    data = data_of(NORMALIZED)
    storing = operator_of(NORMALIZED, data, norm="self")
    loading = operator_of(NORMALIZED, data, norm="ref")
    x = data.array
    v = direction_like(backend, x, seed=1)

    def own(array: Any) -> Any:
        return estel.functional.scattering(data.with_array(array), storing.plan, storing.filters)

    def loaded(array: Any, reference: Any) -> Any:
        stats = estel.functional.scattering(
            data.with_array(array), loading.plan, loading.filters, reference
        )
        return coefficients(backend, stats)

    def self_normalized(array: Any) -> Any:
        return coefficients(backend, own(array))

    def by_its_own_reference(array: Any) -> Any:
        return loaded(array, own(array))

    fixed = own(x)

    def by_a_fixed_reference(array: Any) -> Any:
        return loaded(array, fixed)

    moving = _central_difference(backend, self_normalized, x, v)
    still = _central_difference(backend, by_a_fixed_reference, x, v)
    # The two functions have the same values at x, and derivatives that differ by the term
    # through the reference: the comparisons below tell them apart.
    assert np.abs(moving - still).max() > 1e-3 * np.abs(still).max()
    cotangent = direction_like(backend, self_normalized(x), seed=2)
    for function, expected in ((self_normalized, moving), (by_its_own_reference, still)):
        value, tangent = backend.jvp(function, (x,), (v,))
        np.testing.assert_allclose(backend.to_numpy(tangent), expected, rtol=RTOL, atol=ATOL)
        (pulled,) = backend.vjp(function, x)[1](cotangent)
        np.testing.assert_allclose(
            float(backend.to_numpy(backend.xp.sum(pulled * v))),
            float(np.sum(backend.to_numpy(cotangent) * expected)),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            backend.to_numpy(value), backend.to_numpy(self_normalized(x)), rtol=1e-12, atol=1e-12
        )
