"""The bins of the 2D power spectrum and the Bessel function J₀ of the library (task T4.1,
issue #24; maths.md §1.9; decision on #149).

The bins themselves are compared with STL's in ``tests/parity/test_spectrum.py``. Here: J₀,
which the library implements in NumPy without SciPy and without the oracle's code, against its
power series summed exactly in ``decimal``, to the absolute 1e-12 of §1.9, on [0, 256 π] (images up
to 512 pixels), at the boundaries of its three regions (8 and 25), the first zeros, the
neighbourhood of x = 5, and points spread over the range.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from estel.spectra._bins import bessel_j0, log_gaussian_bins

pytestmark = pytest.mark.maths("M-PS-2D")
TOLERANCE = 1e-12
"""The absolute error of J₀ that maths.md §1.9 requires."""
ZEROS = (
    2.404825557695773,
    5.520078110286311,
    8.653727912911013,
    11.79153443901428,
    14.93091770848779,
)


def _series(x: float) -> float:
    """Σ_m (-x²/4)^m / (m!)², with enough digits that the cancellation of its terms, up to about
    e^x, costs nothing."""
    with localcontext() as context:
        context.prec = int(x / math.log(10)) + 40
        quarter = -(Decimal(x) ** 2) / 4  # the exact value of the float x
        total, term, m = Decimal(0), Decimal(1), 0
        while m <= x or abs(term) > Decimal(10) ** -40:
            total += term
            m += 1
            term = term * quarter / (m * m)
        return float(total)


def test_bessel_j0_equals_its_series() -> None:
    rng = np.random.default_rng(24)
    x = np.concatenate(
        [
            [0.0, 1e-8, 0.5, 4.99, 5.0, 5.0001, 7.999, 8.0, 8.001, 24.999, 25.0, 25.001, 100.0],
            ZEROS,
            rng.uniform(0.0, 30.0, 16),
            rng.uniform(30.0, 256 * math.pi, 12),
            [256 * math.pi],
        ]
    )
    expected = np.array([_series(float(v)) for v in x])
    np.testing.assert_allclose(bessel_j0(x), expected, rtol=0, atol=TOLERANCE)
    np.testing.assert_array_equal(bessel_j0(-x), bessel_j0(x))  # J₀ is even


def test_bins_are_static_and_read_only() -> None:
    """The bins are built once per spatial shape and range, in float64, read-only: plan data."""
    bins = log_gaussian_bins((32, 32), 3, 0, 6)
    assert bins is log_gaussian_bins((32, 32), 3, 0, 6)
    for array in (bins.windows, bins.norm, bins.borders, bins.radial, bins.mode_weights):
        assert not array.flags.writeable
    assert bins.windows.dtype == np.float64
    # N_k counts the modes of the full plane: Σ_k N_k is the number of modes within N_max.
    k1, k2 = np.meshgrid(np.fft.fftfreq(32) * 32, np.fft.fftfreq(32) * 32, indexing="ij")
    assert bins.n_modes.sum() == np.count_nonzero(np.rint(np.hypot(k1, k2)) <= bins.n_max)
