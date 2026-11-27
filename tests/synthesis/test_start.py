"""The start of a synthesis and the Nyquist pre-filter (task T6.1, issue #28; maths.md
M-SYN-INIT, M-SYN-NYQ).

Section 5 of ``docs/maths.md`` has no oracle (Part IV, §22, "No oracle for the syntheses"):
these tests check the start against the definitions of M-SYN-INIT and M-SYN-NYQ directly. The
start is ``(n_batch, Nc, N1, N2)``, or ``running_shape``; a Gaussian white noise of unit variance
drawn from an explicit key, or the initial field given, as it is; then the pre-filter, which
sets to 0 every mode outside the disc k₁² + k₂² ≤ (min(N₁, N₂)/2)² and keeps the others. The
statistical thresholds are five standard deviations of the estimator, computed from the number
of pixels or of modes kept.

On the JAX backend, which has no L-BFGS before step 2 (LF1), the start is still drawn and
filtered: these tests run there too, except the public shapes, which need a synthesis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import estel
from estel import Data2D, _config
from estel.backend._protocol import Backend
from estel.data._cartesian import nyquist_filter, nyquist_mask
from estel.synthesis._api import _running_data, _start
from estel.synthesis._init import gaussian_start

pytestmark = pytest.mark.maths("M-SYN-INIT", "M-SYN-NYQ")
SIGMAS = 5.0


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")


def _running(shape: tuple[int, ...]) -> Any:
    """Running data of ``shape`` (n_batch, Nc, N1, N2), as a synthesis builds them."""
    data = Data2D(np.zeros(shape[1:]), pbc=True)
    return _running_data(data, shape, estel.backend._registry.current_backend())


def _definition(x: np.ndarray) -> np.ndarray:
    """M-SYN-NYQ written in NumPy, with the integer frequencies of the DFT."""
    n1, n2 = x.shape[-2:]
    k1 = np.fft.fftfreq(n1) * n1
    k2 = np.fft.fftfreq(n2) * n2
    kept = k1[:, None] ** 2 + k2[None, :] ** 2 <= (min(n1, n2) / 2) ** 2
    return np.fft.ifft2(np.where(kept, np.fft.fft2(x), 0.0)).real


def test_shape_follows_the_target_or_running_shape(backend: Backend) -> None:
    """(n_batch, Nc, N1, N2) of the target, or ``running_shape`` (M-SYN-INIT), seen in the
    fields a synthesis returns: one iteration, the batch and channel axes kept (OP-75)."""
    if "lbfgs" in getattr(backend, "not_provided", frozenset()):
        return  # no synthesis on this backend: tests/synthesis/test_api.py checks the error
    target = Data2D(np.random.default_rng(0).standard_normal((2, 16, 16)), pbc=True)
    options: dict[str, Any] = {"compute_PS": False, "max_iter": 1, "st_kwargs": {"J": 2}}
    same = estel.synthesize_from_maps(target, n_batch=3, **options)
    assert tuple(same.field.shape) == (3, 2, 16, 16)
    other = estel.synthesize_from_maps(target, n_batch=3, running_shape=(32, 16), **options)
    assert tuple(other.field.shape) == (3, 2, 32, 16)
    assert other.manifest["running_shape"] == [3, 2, 32, 16]


def test_same_key_same_start_other_key_other_start(backend: Backend) -> None:
    running = _running((2, 1, 16, 16))
    first = backend.to_numpy(_start(backend, "gaussian", 5, running, "float64"))
    again = backend.to_numpy(_start(backend, "gaussian", 5, running, "float64"))
    other = backend.to_numpy(_start(backend, "gaussian", 6, running, "float64"))
    assert first.shape == (2, 1, 16, 16)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_initial_field_is_used_as_it_is(backend: Backend) -> None:
    """The initial field given, broadcast to the running shape, unchanged (M-SYN-INIT): the
    pre-filter comes after, as for the Gaussian start."""
    running = _running((2, 1, 16, 16))
    init = np.random.default_rng(1).standard_normal((1, 16, 16))
    start = backend.to_numpy(_start(backend, init, 0, running, "float64"))
    np.testing.assert_array_equal(start, np.broadcast_to(init, (2, 1, 16, 16)))


def test_gaussian_start_is_a_unit_white_noise(backend: Backend) -> None:
    """Before the pre-filter: mean 0 and variance 1, within five standard deviations of their
    estimators over n pixels, 1/√n and √(2/n); and white: its periodogram is flat, the mean
    power of the modes in each half of the frequencies within five standard deviations of 1."""
    shape = (2, 1, 64, 64)
    noise = backend.to_numpy(gaussian_start(backend, 11, shape, "float64"))
    n = noise.size
    assert abs(noise.mean()) <= SIGMAS / math.sqrt(n)
    assert abs(noise.var() - 1.0) <= SIGMAS * math.sqrt(2.0 / n)
    power = np.abs(np.fft.fft2(noise)) ** 2 / (64 * 64)  # expectation 1 at every mode
    low = nyquist_mask((64, 64))
    for part in (low, ~low):
        count = int(part.sum()) * 2  # modes, over the two batch elements
        assert abs(power[..., part].mean() - 1.0) <= SIGMAS / math.sqrt(count / 2)


def test_prefilter_zeroes_the_modes_outside_the_disc(backend: Backend) -> None:
    """After the pre-filter: every mode outside the disc of M-SYN-NYQ is 0, those inside are
    unchanged, and the variance is the fraction of modes kept, within five standard deviations
    (a projection of a unit white noise onto m modes of n: mean m/n, standard deviation
    √(2m)/n). At 32 x 32 the filter keeps 795 modes of 1024 (M-SYN-INIT: 0.77 to 0.78)."""
    shape = (4, 1, 32, 32)
    noise = gaussian_start(backend, 12, shape, "float64")
    start = backend.to_numpy(nyquist_filter(backend, noise))
    kept = nyquist_mask((32, 32))
    assert int(kept.sum()) == 795
    before, after = np.fft.fft2(backend.to_numpy(noise)), np.fft.fft2(start)
    scale = np.abs(before).max()
    assert np.abs(after[..., ~kept]).max() <= 1e-12 * scale
    np.testing.assert_allclose(after[..., kept], before[..., kept], rtol=0, atol=1e-12 * scale)
    m, n = int(kept.sum()) * shape[0], start.size
    assert abs(np.mean(start**2) - m / n) <= SIGMAS * math.sqrt(2.0 * m) / n


@pytest.mark.parametrize("shape", [(16, 16), (32, 32), (16, 24), (24, 16), (15, 15), (15, 20)])
def test_prefilter_is_its_definition(backend: Backend, shape: tuple[int, int]) -> None:
    """M-SYN-NYQ against its definition written in NumPy, on square, rectangular and odd
    fields: modes on the circle kept, corners removed, a projection that leaves the mean
    unchanged; for a rectangle, the radius of the shorter side on both axes (OP-54)."""
    x = np.random.default_rng(sum(shape)).standard_normal((2, 3, *shape))
    once = backend.to_numpy(nyquist_filter(backend, backend.asarray(x, "float64")))
    np.testing.assert_allclose(once, _definition(x), rtol=0, atol=1e-13)
    twice = backend.to_numpy(nyquist_filter(backend, backend.asarray(once, "float64")))
    np.testing.assert_allclose(twice, once, rtol=0, atol=1e-13)
    np.testing.assert_allclose(once.mean(axis=(-2, -1)), x.mean(axis=(-2, -1)), atol=1e-14)
    kept = nyquist_mask(shape)
    counts = {(16, 16): 195, (32, 32): 795}
    if shape in counts:
        assert int(kept.sum()) == counts[shape]
        half = shape[0] // 2
        assert kept[half, 0]  # (N/2, 0), on the circle
        assert kept[0, half]  # (0, N/2), on the circle
        assert not kept[half, half]  # a corner
    if shape == (16, 24):
        assert int(np.abs(np.fft.fftfreq(24) * 24)[kept.any(axis=0)].max()) == 8
