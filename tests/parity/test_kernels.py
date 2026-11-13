"""Parity of the wavelet kernels and of the smoothing kernel with STL (task T2.2, issue #14).

* the 2D bump-steerable kernels with ``bump_steerable_kernel_2d`` (STL `master`, L = 4, K = 5);
* the 1D kernels with ``bump_steerable_kernel_1d`` (STL `2D+1D`, K = 5, L = 1 and L = 2);
* the smoothing kernel with the ``smooth_kernel`` STL used for ``masks_2d`` (2D) and
  ``masks_1d`` (1D).

The kernels are built in NumPy float64 whatever the run's dtype and backend (tag N of Part II,
§12), so they are compared in float64, with the tolerances of Part II, §14.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from estel.wavelets._banks import (
    KERNEL_SIZE,
    bump_steerable_1d,
    bump_steerable_2d,
    smoothing_kernel,
)
from tests.parity._compare import assert_parity, load_fixture, read_manifest


@pytest.mark.maths("M-WAV-2D")
def test_bump_steerable_2d_equals_stl() -> None:
    parameters = read_manifest()["bump_steerable_kernel_2d.npz"]["parameters"]
    assert parameters["K"] == KERNEL_SIZE
    expected = load_fixture("bump_steerable_kernel_2d")["kernel"]  # (1, L, K, K)
    assert_parity(bump_steerable_2d(parameters["L"], parameters["K"])[None], expected)


@pytest.mark.maths("M-WAV-1D")
@pytest.mark.parametrize("n_orientations", [1, 2])
def test_bump_steerable_1d_equals_stl(n_orientations: int) -> None:
    parameters = read_manifest()["bump_steerable_kernel_1d.npz"]["parameters"]
    assert n_orientations in parameters["L"]
    assert parameters["K"] == KERNEL_SIZE
    expected = load_fixture("bump_steerable_kernel_1d")[f"kernel_L{n_orientations}"]  # (1, L, K)
    actual = bump_steerable_1d(size=parameters["K"], conjugate=n_orientations == 2)
    assert_parity(actual[None], expected)


@pytest.mark.maths("M-SMOOTH-2D", "M-SMOOTH-1D")
@pytest.mark.parametrize(("case", "ndim"), [("masks_2d", 2), ("masks_1d", 1)])
def test_smoothing_kernel_equals_stl(case: str, ndim: int) -> None:
    """Equal bit for bit, not only within the tolerances: the masked downsampling compares
    fractions of invalid input with τ, and in ``masks_1d`` one fraction equals τ = 0.5
    exactly, so the last bit of the smoothing kernel decides whether that pixel is valid
    (M-DOWN-MASK)."""
    expected = load_fixture(case)["smooth_kernel"]
    assert_parity(smoothing_kernel(ndim), expected)
    np.testing.assert_array_equal(smoothing_kernel(ndim), expected)
