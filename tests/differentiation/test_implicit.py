"""Implicit differentiation, end to end, on a small argmin problem built on the scattering
statistics (task T2.7, issue #19; Part I, §3.2.1 and the last row of §3.2.2).

Written with the L0 primitives, and collected only on PyTorch: one backend is enough for this
test (issue #19), and each backend's second order is checked on its own in the other files.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data1D
from estel.backend._protocol import Backend
from tests.differentiation._cases import (
    Case,
    coefficients,
    data_of,
    direction_like,
    operator_of,
)

IMPLICIT = Case("1d-pbc-self", Data1D, (1, 1, 16), pbc=True, options={"norm": "self"})


@pytest.mark.slow
def test_implicit_differentiation_end_to_end(backend: Backend) -> None:
    """Part I, §3.2.1, on a small problem. For φ = (a, log μ),

        L(θ, φ) = ½ |Φ(θ) - Φ(d + a u)|² + ½ μ |θ - d|²,

    with Φ the self-normalised scattering statistics of a field of 16 pixels, d and u fixed
    fields: the field θ whose statistics match those of a target that depends on an amplitude
    a, under a quadratic prior of weight μ. θ*(φ) = argmin L is found by Newton's method to
    the precision of float64, and dθ*/dφ = -[∂²L/∂θ²]⁻¹ ∂²L/∂θ∂φ, from Hessian-vector products
    (``Backend.hvp``) and the mixed derivatives (``Backend.jvp`` of the gradient), is compared
    with central differences of θ* itself, solved again at φ ± h.

    Slow: a few Newton solves, each iteration with one Hessian-vector product per pixel
    through the whole loop (second order)."""
    xp = backend.xp
    data = data_of(IMPLICIT)
    operator = operator_of(IMPLICIT, data)
    d = data.array
    u = direction_like(backend, d, seed=1)

    def statistics(array: Any) -> Any:
        stats = estel.functional.scattering(data.with_array(array), operator.plan, operator.filters)
        return coefficients(backend, stats)

    def loss(theta: Any, phi: Any) -> Any:
        misfit = statistics(theta) - statistics(d + phi[0] * u)
        return 0.5 * xp.sum(misfit**2) + 0.5 * xp.exp(phi[1]) * xp.sum((theta - d) ** 2)

    def gradient(theta: Any, phi: Any) -> Any:
        return backend.value_and_grad(loss)(theta, phi)[1]

    def hessian(theta: Any, phi: Any) -> NDArray[np.float64]:
        columns = []
        for e in np.eye(math.prod(theta.shape)).reshape(-1, *theta.shape):
            direction = backend.asarray(e, "float64")
            column = backend.hvp(lambda t: loss(t, phi), theta, direction)
            columns.append(backend.to_numpy(column).ravel())
        return np.stack(columns, axis=1)

    def argmin(phi: Any, start: Any) -> Any:
        theta = start
        for _ in range(30):
            g = backend.to_numpy(gradient(theta, phi)).ravel()
            if np.abs(g).max() < 1e-13:
                return theta
            step = np.linalg.solve(hessian(theta, phi), g).reshape(theta.shape)
            theta = theta - backend.asarray(step, "float64")
        raise AssertionError(f"Newton's method did not converge: |gradient| = {np.abs(g).max()}")

    phi0 = np.array([0.5, np.log(0.02)])
    phi = backend.asarray(phi0, "float64")
    solution = argmin(phi, d)
    curvature = hessian(solution, phi)
    assert np.linalg.eigvalsh(curvature).min() > 0  # a strict minimum
    # The statistics matter: the solution is far from the prior's centre.
    assert float(np.abs(backend.to_numpy(solution - d)).max()) > 1e-2
    mixed = []
    for e in np.eye(2):
        direction = backend.asarray(e, "float64")
        _, column = backend.jvp(lambda p: gradient(solution, p), (phi,), (direction,))
        mixed.append(backend.to_numpy(column).ravel())
    implicit = -np.linalg.solve(curvature, np.stack(mixed, axis=1))
    step = 1e-4
    for k in range(2):
        shifted = [
            backend.to_numpy(
                argmin(backend.asarray(phi0 + sign * step * np.eye(2)[k], "float64"), solution)
            ).ravel()
            for sign in (1, -1)
        ]
        difference = (shifted[0] - shifted[1]) / (2 * step)
        assert np.abs(difference).max() > 1e-3  # θ* does move with each parameter
        np.testing.assert_allclose(implicit[:, k], difference, rtol=1e-6, atol=1e-8)
