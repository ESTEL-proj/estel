"""Unit tests of the planar wavelet banks and of H (maths.md M-WAV-2D, M-WAV-1D, M-H, M-H-1D;
task T2.2).

Properties that hold by construction, and the validation of the parameters. Equality with STL's
kernels is in ``tests/parity/test_kernels.py``. The banks are NumPy float64 whatever the run's
dtype and backend (tag N of Part II, §12). STL's kernels do not satisfy the Littlewood-Paley
condition exactly, and 1.0.0 does not test it (Part II, §12.5; LF22).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from estel.wavelets._banks import (
    FOURIER_GRID,
    KERNEL_SIZE,
    bump_steerable_1d,
    bump_steerable_2d,
    smoothing_kernel_H,
)

pytestmark = pytest.mark.maths("M-WAV-2D", "M-WAV-1D", "M-H", "M-H-1D")


@pytest.mark.parametrize("n_orientations", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("size", [3, KERNEL_SIZE, 7])
def test_bump_steerable_2d_shape_dtype_and_zero_mean(n_orientations: int, size: int) -> None:
    kernels = bump_steerable_2d(n_orientations, size)
    assert kernels.shape == (n_orientations, size, size)
    assert kernels.dtype == np.complex128
    np.testing.assert_allclose(kernels.sum(axis=(1, 2)), 0.0, atol=1e-15)
    assert np.all(np.abs(kernels).max(axis=(1, 2)) > 0.0)


def test_bump_steerable_2d_orientation_0_oscillates_along_axis_2() -> None:
    """θ_0 = π/2 from axis 1: orientation 0 responds to oscillations along axis 2 (OP-22)."""
    response = np.abs(np.fft.fft2(bump_steerable_2d(4)[0], s=(64, 64)))
    k1, k2 = np.unravel_index(np.argmax(response), response.shape)
    k1, k2 = (k - 64 if k >= 32 else k for k in (k1, k2))
    assert abs(k2) > abs(k1)


def test_bump_steerable_1d_shape_and_conjugate() -> None:
    analytic = bump_steerable_1d()
    both = bump_steerable_1d(conjugate=True)
    assert analytic.shape == (1, KERNEL_SIZE)
    assert both.shape == (2, KERNEL_SIZE)
    assert both.dtype == np.complex128
    np.testing.assert_array_equal(both[0], analytic[0])
    np.testing.assert_array_equal(both[1], np.conj(analytic[0]))
    np.testing.assert_allclose(analytic.sum(), 0.0, atol=1e-15)


def test_bump_steerable_1d_default_xi0_is_nyquist() -> None:
    """ξ₀ = N/2 by default (M-WAV-1D, OP-28)."""
    np.testing.assert_array_equal(bump_steerable_1d(), bump_steerable_1d(FOURIER_GRID / 2))
    assert not np.allclose(bump_steerable_1d(), bump_steerable_1d(FOURIER_GRID / 4))


def test_banks_are_new_arrays() -> None:
    first = bump_steerable_2d(4)
    first[...] = 0.0
    assert np.abs(bump_steerable_2d(4)).max() > 0.0
    h = smoothing_kernel_H(2)
    h[...] = 0.0
    assert smoothing_kernel_H(2).sum() > 0.5


@pytest.mark.parametrize("value", [0, -1])
def test_bump_steerable_2d_rejects_l_below_one(value: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        bump_steerable_2d(value)


@pytest.mark.parametrize("value", [2.0, True, "4", None])
def test_bump_steerable_2d_rejects_non_integer_l(value: Any) -> None:
    with pytest.raises(TypeError, match="integer"):
        bump_steerable_2d(value)


@pytest.mark.parametrize("size", [0, 4, -3, FOURIER_GRID + 1])
def test_banks_reject_bad_sizes(size: int) -> None:
    with pytest.raises(ValueError, match="odd"):
        bump_steerable_2d(4, size)
    with pytest.raises(ValueError, match="odd"):
        bump_steerable_1d(size=size)


@pytest.mark.parametrize("xi0", [0.0, -1.0, float("nan"), float("inf"), "128"])
def test_bump_steerable_1d_rejects_bad_xi0(xi0: Any) -> None:
    with pytest.raises(ValueError, match="positive frequency"):
        bump_steerable_1d(xi0)


def test_bump_steerable_1d_rejects_non_boolean_conjugate() -> None:
    with pytest.raises(TypeError, match="True"):
        bump_steerable_1d(conjugate=2)  # type: ignore[arg-type]


def test_smoothing_kernel_h_2d_is_stls_table() -> None:
    """M-H: 6 x 6, symmetric, the product of the 1D kernel up to the rounding of STL's
    literals, summing to 1 up to that rounding (1.00000004)."""
    h2 = smoothing_kernel_H(2)
    h1 = smoothing_kernel_H(1)
    assert h2.shape == (6, 6)
    assert h2.dtype == np.float64
    np.testing.assert_array_equal(h2, h2.T)
    np.testing.assert_array_equal(h2, h2[::-1, ::-1])
    np.testing.assert_allclose(h2, np.outer(h1, h1), atol=1e-8)
    assert h2.sum() == pytest.approx(1.0, abs=1e-7)
    assert h2[2, 2] == 0.2424694


def test_smoothing_kernel_h_1d_is_the_third_row_normalised() -> None:
    """M-H-1D: the third row of the 2D table divided by its sum (0.49241182 at the centre)."""
    h1 = smoothing_kernel_H(1)
    row = smoothing_kernel_H(2)[2]
    np.testing.assert_array_equal(h1, row / 0.49241182)  # the correctly rounded sum
    assert h1.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(
        h1, [-0.04879268, 0.05638086, 0.49241182, 0.49241182, 0.05638086, -0.04879268], atol=5e-9
    )


@pytest.mark.parametrize("ndim", [0, 3, True])
def test_smoothing_kernel_h_rejects_other_dimensions(ndim: Any) -> None:
    with pytest.raises(ValueError, match="ndim=1 or ndim=2"):
        smoothing_kernel_H(ndim)
