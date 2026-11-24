"""``torch.autograd.gradcheck`` and ``gradgradcheck`` of the scattering statistics (task T2.7,
issue #19; Part I, §3.2.2).

``gradcheck`` builds the whole Jacobian of ``estel.functional.scattering`` (every coefficient
that exists, ``tests/differentiation/_cases.py``) by reverse mode, and by forward mode with
``check_forward_ad``, and compares both with central finite differences in float64;
``gradgradcheck`` does the same for the derivative of the VJP, the double backward of
Part I, §3.2.2. Collected only on PyTorch; ``test_jax.py`` is the JAX counterpart.

Tolerances. The step of the finite differences is 1e-6, so their error is about 1e-10 of the
coefficients (round-off) and less in truncation; the comparison is at ``rtol = 1e-6`` and
``atol = 1e-8``, a thousand times tighter than PyTorch's defaults.
"""

from __future__ import annotations

import pytest
import torch

from estel import Data2D
from estel.backend._protocol import Backend
from tests.differentiation._cases import (
    CASES,
    SMALLEST,
    SPECTRUM_CASES,
    SPECTRUM_SMALLEST,
    Case,
    Function,
    SpectrumCase,
    scattering_function,
    spectrum_function,
)

EPS, ATOL, RTOL = 1e-6, 1e-8, 1e-6


def _point(case: Case, backend: Backend) -> tuple[torch.Tensor, Function]:
    data, function = scattering_function(backend, case)
    return data.array.detach().clone().requires_grad_(True), function


def test_gradcheck_of_the_smallest_case(backend: Backend) -> None:
    """The VJP of the whole loop against finite differences: the one differentiation check of
    the fast tier (8 pixels, J = 2, 24 coefficients)."""
    x, function = _point(SMALLEST, backend)
    assert torch.autograd.gradcheck(function, (x,), eps=EPS, atol=ATOL, rtol=RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_gradcheck_and_gradgradcheck(case: Case, backend: Backend) -> None:
    """First order in both modes, then the double backward, on every configuration. In 2D the
    double backward is checked in ``fast_mode``, along random directions rather than entry by
    entry of its Jacobian, which would take a minute.

    Slow: the full Jacobian of every coefficient, and second-order gradients through the whole
    loop (Part I, §5.4)."""
    x, function = _point(case, backend)
    assert torch.autograd.gradcheck(
        function, (x,), check_forward_ad=True, eps=EPS, atol=ATOL, rtol=RTOL
    )
    fast_mode = case.cls is Data2D
    assert torch.autograd.gradgradcheck(
        function, (x,), fast_mode=fast_mode, eps=EPS, atol=ATOL, rtol=RTOL
    )


def _spectrum_point(case: SpectrumCase, backend: Backend) -> tuple[torch.Tensor, Function]:
    data, function = spectrum_function(backend, case)
    return data.array.detach().clone().requires_grad_(True), function


def test_gradcheck_and_gradgradcheck_of_the_spectrum(backend: Backend) -> None:
    """The 2D power spectrum (maths.md §1.9; task T4.1) through ``SpectrumOperator``, its
    reference included (``norm="self"``): first order in both modes and the double backward,
    along random directions, on one 16 x 16 map: the spectrum check of the fast tier."""
    x, function = _spectrum_point(SPECTRUM_SMALLEST, backend)
    assert torch.autograd.gradcheck(
        function, (x,), check_forward_ad=True, fast_mode=True, eps=EPS, atol=ATOL, rtol=RTOL
    )
    assert torch.autograd.gradgradcheck(
        function, (x,), fast_mode=True, eps=EPS, atol=ATOL, rtol=RTOL
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=lambda case: case.name)
def test_gradcheck_and_gradgradcheck_of_every_spectrum_case(
    case: SpectrumCase, backend: Backend
) -> None:
    """First order in both modes, entry by entry, then the double backward along random
    directions, through ``SpectrumOperator`` (each estimator, the three normalisations, the
    filling of missing pixels) and through ``STOperator`` with the spectrum.

    Slow: the full Jacobian of every coefficient of 16 x 16 maps, and second order through the
    whole loop (Part I, §5.4)."""
    x, function = _spectrum_point(case, backend)
    assert torch.autograd.gradcheck(
        function, (x,), check_forward_ad=True, eps=EPS, atol=ATOL, rtol=RTOL
    )
    assert torch.autograd.gradgradcheck(
        function, (x,), fast_mode=True, eps=EPS, atol=ATOL, rtol=RTOL
    )
