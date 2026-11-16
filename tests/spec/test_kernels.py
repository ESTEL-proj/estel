"""Spec parity of the kernels: ESTEL's wavelet and smoothing kernels against the oracle
``estel._spec`` (Part I, §5.1; task T2.6, issue #18; maths.md M-WAV-2D, M-WAV-1D, M-SMOOTH-2D,
M-SMOOTH-1D).

The oracle builds the wavelets by a direct inverse Fourier sum over the 256-point design grid,
ESTEL by an FFT; both are compared with ``assert_parity``, the float64 tolerances of Part II,
§14.3. The smoothing kernels are literals, and must be equal bit for bit (OP-12).
"""

from __future__ import annotations

import numpy as np
import pytest

from estel._spec import planar as spec
from estel.wavelets import _banks
from tests.parity._compare import assert_parity


@pytest.mark.maths("M-WAV-2D")
@pytest.mark.parametrize("L", [1, 3, 4, 6])
def test_wavelets_2d_equal_the_oracle(L: int) -> None:  # noqa: N803 (maths.md notation)
    assert_parity(_banks.bump_steerable_2d(L), spec.spec_wavelets_2d(L))


@pytest.mark.maths("M-WAV-1D")
@pytest.mark.parametrize("L", [1, 2])
def test_wavelets_1d_equal_the_oracle(L: int) -> None:  # noqa: N803
    assert_parity(_banks.bump_steerable_1d(conjugate=L == 2), spec.spec_wavelets_1d(L))


@pytest.mark.maths("M-SMOOTH-2D", "M-SMOOTH-1D")
def test_smoothing_kernels_equal_the_oracle() -> None:
    assert np.array_equal(_banks.smoothing_kernel(2), spec.spec_smoothing_kernel_2d())
    assert np.array_equal(_banks.smoothing_kernel(1), spec.spec_smoothing_kernel_1d())
