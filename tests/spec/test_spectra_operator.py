"""Spec parity of the 2D power spectrum: ``SpectrumOperator`` and ``STOperator`` with the
spectrum against the oracle ``estel._spec.spectra`` (task T4.1, issue #24; maths.md §1.9,
M-NORM, M-FLAT; Part I, §4.2, §5.1).

Small seeded cases, 16 x 16 or 16 x 24 pixels, two channels, every pair c₁ ≤ c₂: each estimator
forced, and the automatic choice, with and without missing pixels, periodic or not, with a J_min
and a number of bins given; the spectrum within the scattering statistics, after the
standardisation, normalised by itself (M-NORM), last in the vector (M-FLAT); and the
configuration ``nan_per_map`` of ``spectrum_2d`` on the union of its missing pixels, the only form
ESTEL takes (OP-98; comment on #24). Compared with ``assert_parity`` in the run's dtype: float64,
and float32 to the float32 tolerance of Part II, §14.3.

A disagreement is a finding about ``docs/maths.md`` or about the oracle, since ESTEL matches
STL's fixtures: report the coefficients on an issue, change nothing to make a test pass.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data2D, SpectrumOperator, STOperator, _config
from estel._spec import planar as spec
from estel._spec import spectra as spec_spectra
from estel.backend import _registry
from estel.backend._protocol import DType
from tests.parity._compare import assert_parity, load_fixture

pytestmark = [pytest.mark.maths("M-PS-2D", "M-NORM", "M-FLAT"), pytest.mark.dtype]


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    monkeypatch.setattr(_config, "_dtype", dtype)


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.3
    return values


def _hole(shape: tuple[int, int]) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    mask[5:8, 3:9] = True
    mask[12, 2] = True
    return mask


CASES: dict[str, tuple[tuple[int, int], bool, bool, dict[str, Any]]] = {
    # name: (spatial shape, pbc, masked, options)
    "periodic-auto": ((16, 16), True, False, {}),
    "open-auto-rect": ((16, 24), False, False, {}),
    "periodic-mask-auto": ((16, 16), True, True, {}),
    "open-mask-auto": ((16, 16), False, True, {"n_bins": 5}),
    "open-fft": ((16, 16), False, False, {"method": "fft"}),
    "periodic-pbc-mask": ((16, 16), True, False, {"method": "pbc_mask", "n_bins": 3}),
    "open-nan-mask": ((16, 16), False, True, {"method": "nan_mask"}),
    "periodic-pbc-nan-mask-jmin": ((32, 32), True, True, {"method": "pbc_nan_mask", "Jmin": 1}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_spectrum_operator_equals_the_oracle(name: str, dtype: DType) -> None:
    """``SpectrumOperator`` with ``norm="none"`` equals ``spec_spectra``: the estimator chosen and
    the spectrum of every pair c₁ ≤ c₂, NaN below the diagonal."""
    shape, pbc, masked, options = CASES[name]
    field = _field((2, 2, *shape))
    mask = _hole(shape) if masked else None
    data = Data2D(field, mask=mask, pbc=pbc)
    operator = SpectrumOperator(data, norm="none", **options)
    oracle_field = field if mask is None else np.where(mask, np.nan, field)
    expected = spec_spectra.spec_spectra(oracle_field, pbc, **options)
    assert operator._plan.spectrum.method == expected["method"]
    assert_parity(_numpy(operator.apply(data).full_grid("PS")), expected["PS"], dtype)


def test_statistics_with_the_spectrum_equal_the_oracle(dtype: DType) -> None:
    """``STOperator`` with the spectrum, ``standardize`` and a mask: the spectrum of the field after
    the standardisation, normalised by itself, every family and the vector equal ``spec_apply``
    with ``compute_PS=True`` (maths.md §1.9, "In the scattering operator"; M-NORM; M-FLAT)."""
    field = _field((1, 2, 32, 32), seed=3)
    mask = _hole((32, 32))
    data = Data2D(field, mask=mask, pbc=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13: tested with the plan
        stats = STOperator(data, 2, 4, standardize=True).apply(data)
    oracle = spec.spec_apply(
        field, 2, False, mask, 2, 4, norm="self", standardize=True, compute_PS=True, flatten=True
    )
    for family in ("var", "S1", "S2", "S3", "S4", "PS"):
        assert_parity(_numpy(stats.full_grid(family)), oracle[family], dtype)
    assert_parity(_numpy(stats.to_vector()), oracle["vector"], dtype)


def test_maps_with_their_own_missing_pixels_on_their_union(dtype: DType) -> None:
    """The configuration ``nan_per_map`` of ``spectrum_2d``, whose missing pixels differ per map,
    given to ESTEL with the union of its missing pixels as its one mask (OP-98): ``nan_mask`` and
    ``pbc_nan_mask`` equal the oracle on that union. STL, which fills each map from its own
    valid pixels, differs by up to 29 % (comment on #24)."""
    field = load_fixture("spectrum_2d")["nan_per_map_field"]
    union = np.isnan(field).any(axis=(0, 1))
    data = Data2D(field, mask=union, pbc=True)
    for method in ("nan_mask", "pbc_nan_mask"):
        found = SpectrumOperator(data, method=method, norm="none").apply(data)
        expected = spec_spectra.spec_spectra(np.where(union, np.nan, field), True, method=method)
        assert_parity(_numpy(found.full_grid("PS")), expected["PS"], dtype)
