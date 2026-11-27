"""The oracle of the 2D power spectrum against STL's parity fixtures (task T4.2, issue #149;
maths.md §1.9, M-PS-2D, with M-NORM, M-FLAT and M-CHAN; Part I, §4.2; Part II, §12.11 and
§14.3).

The fixtures of T4.3 (``spectrum_2d``, ``statistics_ps_2d``, STL `master` at ``8eaf769``) hold
STL's spectra, bins and statistics with the spectrum. The oracle runs on the same fields, with
the parameters the manifest records, and every stored array is compared with ``assert_parity``:
same shape, NaN at the same positions, values within the float64 tolerances of Part II, §14.3
(for ``mean``, atol = 0.1 x rtol x sqrt(max(var))). The fixtures hold STL's frequencies in
cycles per pixel, the oracle's are in mode units: a frequency k is k / (2 N_max) there.

The oracle's Bessel function J₀ is checked against its power series summed exactly, with
Python's ``decimal``, not against ``torch.special.bessel_j0``, which is off by up to 3.8e-7 in
float64 (decision of the maintainer on #149); the borders that depend on it are compared with
STL's in ``spectrum_2d``.

Every test runs in the fast tier.

A disagreement is a finding about ``docs/maths.md`` or about the oracle: do not change the
oracle, a fixture or a tolerance to make a test pass; report the configuration and the
coefficients on the issue.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel._spec import planar
from estel._spec import spectra as spec
from tests.parity._compare import assert_parity, load_fixture, read_manifest

pytestmark = pytest.mark.maths("M-PS-2D", "M-CHAN", "M-NORM", "M-FLAT")
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4", "PS")
SPECTRUM_OPTIONS = ("J", "Jmin", "n_bins")
BESSEL_TOLERANCE = 1e-12
"""The accuracy that maths §1.9 requires of J₀: an absolute error of at most 1e-12."""
BESSEL_ZEROS = (
    2.404825557695773,
    5.520078110286311,
    8.653727912911013,
    11.791534439014281,
    14.930917708487787,
)
"""The first five zeros of J₀."""


@cache
def _arrays(case: str) -> dict[str, NDArray[Any]]:
    return load_fixture(case)


def _configs(case: str) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = read_manifest()[f"{case}.npz"]["parameters"]["configs"]
    return configs


def _cycles(frequency: Any, n_max: int) -> Any:
    """A frequency in mode units, in STL's cycles per pixel (maths §1.9)."""
    return np.asarray(frequency) / (2 * n_max)


# ---------------------------------------------------------------------- spectrum_2d


@pytest.mark.parametrize("name", list(_configs("spectrum_2d")))
def test_bins_equal_the_fixture(name: str) -> None:
    """J, J_min, n_b, k_min, k_max, N_max, the radial frequencies, N_k, the centres, the
    windows, Z_b and the borders β_b of each configuration of ``spectrum_2d``."""
    config, arrays = _configs("spectrum_2d")[name], _arrays("spectrum_2d")
    options = {key: config[key] for key in SPECTRUM_OPTIONS if key in config}
    bins = spec.spec_spectrum_bins(tuple(config["shape"]), **options)
    n_max = bins["N_max"]
    scalars = np.array(
        [
            bins["J"],
            bins["Jmin"],
            bins["n_bins"],
            _cycles(bins["k_min"], n_max),
            _cycles(bins["k_max"], n_max),
            n_max,
        ]
    )
    assert_parity(scalars, arrays[f"{name}_bin_scalars"])
    assert_parity(_cycles(bins["radial_k"], n_max), arrays[f"{name}_radial_k"])
    assert np.array_equal(bins["n_modes"], arrays[f"{name}_n_modes_radial"])
    assert_parity(_cycles(bins["centers"], n_max), arrays[f"{name}_bin_centers"])
    assert_parity(bins["windows"], arrays[f"{name}_bin_windows"])
    assert_parity(bins["norm"], arrays[f"{name}_bin_norm"])
    assert np.array_equal(bins["borders"], arrays[f"{name}_crop_borders"])


@pytest.mark.parametrize("name", list(_configs("spectrum_2d")))
def test_spectra_equal_the_fixture(name: str) -> None:
    """The automatic choice and its spectrum with the auto spectra only (STL's default pairs of
    its spectrum operator, OP-94), and each estimator that accepts the field with every pair
    c1 ≤ c2 (the oracle's default), for each configuration of ``spectrum_2d``."""
    config, arrays = _configs("spectrum_2d")[name], _arrays("spectrum_2d")
    options = {key: config[key] for key in SPECTRUM_OPTIONS if key in config}
    field = arrays[f"{name}_field"]
    auto = spec.spec_spectra(field, config["pbc"], **options, cross_matrix=np.eye(config["Nc"]))
    assert auto["method"] == config["auto"]
    assert auto["method"] == spec.METHODS[int(arrays[f"{name}_auto_method"][0])]
    assert_parity(auto["PS"], arrays[f"{name}_auto"])
    compared = 0
    for method in spec.METHODS:
        key = f"{name}_method_{method}"
        if key in arrays:
            out = spec.spec_spectra(field, config["pbc"], **options, method=method)
            assert_parity(out["PS"], arrays[key])
            compared += 1
    assert compared == (2 if np.isnan(field).any() else 4)


# ---------------------------------------------------------------------- statistics_ps_2d


@pytest.mark.parametrize("name", list(_configs("statistics_ps_2d")))
def test_statistics_with_the_spectrum_equal_the_fixture(name: str) -> None:
    """For each configuration of ``statistics_ps_2d``: the estimator, the spectrum without
    normalisation, every family of ``norm="self"`` (STL's ``store_ref``) and of ``norm="ref"``
    on the second field (``load_ref``), PS included, the reference (σᴾˢ included), and the four
    flat vectors, PS last (M-FLAT; its real part only in the real form, OP-37)."""
    config, arrays = _configs("statistics_ps_2d")[name], _arrays("statistics_ps_2d")
    J, J_PS, Jmin, n_bins = (int(v) for v in arrays[f"{name}_ps_bins"])  # noqa: N806
    assert J == J_PS
    mask = arrays[f"{name}_mask"].astype(bool) if f"{name}_mask" in arrays else None
    pbc, cross_matrix = config["pbc"], config.get("cross_matrix")
    options: dict[str, Any] = {
        "cross_matrix": cross_matrix,
        "compute_PS": True,
        "ref_batch_mean": config["norm_batch_mean"],
        "standardize": False,  # as STL computed the fixture (OP-120)
        **{key: config[key] for key in ("Jmin", "n_bins", "spectrum_method") if key in config},
    }
    target, other = arrays[f"{name}_field"], arrays[f"{name}_ref_field"]

    store = planar.spec_apply(target, 2, pbc, mask, config["J"], 4, norm="self", **options)
    reference = store["reference"]
    load = planar.spec_apply(
        other, 2, pbc, mask, config["J"], 4, norm="ref", ref=reference, **options
    )

    method = spec.spec_spectrum_method(mask is not None, pbc, config.get("spectrum_method", "auto"))
    assert method == spec.METHODS[int(arrays[f"{name}_ps_method"][0])]
    x, _ = planar.spec_data(target, 2, J, mask, pbc)
    raw = spec.spec_spectra(x, pbc, J, Jmin, config.get("n_bins"), method, cross_matrix)
    assert raw["bins"]["n_bins"] == n_bins
    assert_parity(raw["PS"], arrays[f"{name}_raw_PS"])

    for prefix, out, field in (("store", store, target), ("load", load, other)):
        data, invalid = planar.spec_data(field, 2, J, mask, pbc)
        _, var = planar.spec_moments(data, 2, invalid)
        for family in FAMILIES:
            scale = float(np.sqrt(np.nanmax(var))) if family == "mean" else None
            assert_parity(out[family], arrays[f"{name}_{prefix}_{family}"], scale=scale)
    assert_parity(reference["var"], arrays[f"{name}_var_ref"])
    assert_parity(reference["sigma"], arrays[f"{name}_s2_ref"])
    assert_parity(reference["sigma_PS"], arrays[f"{name}_ps_ref"])

    vectors = {
        "vector": planar.spec_flatten(store),
        "vector_real": planar.spec_flatten(store, flatten_complex=True),
        "vector_batch_mean": planar.spec_flatten(store, mean_along_batch=True),
        "load_vector": planar.spec_flatten(load),
    }
    for key, vector in vectors.items():
        assert_parity(vector, arrays[f"{name}_{key}"])


def test_statistics_without_the_spectrum_are_unchanged() -> None:
    """Without ``compute_PS``, ``spec_apply`` returns no PS, and the same statistics, reference
    and vector as with it, PS left out."""
    arrays = _arrays("statistics_ps_2d")
    field = arrays["periodic_J2_field"]
    without = planar.spec_apply(field, 2, True, None, 2, 4, norm="self", flatten=True)
    with_ps = planar.spec_apply(
        field, 2, True, None, 2, 4, norm="self", flatten=True, compute_PS=True
    )
    assert "PS" not in without
    assert "sigma_PS" not in without["reference"]
    for family in FAMILIES[:-1]:
        assert np.array_equal(without[family], with_ps[family], equal_nan=True)
    n_ps = int(np.isfinite(with_ps["PS"]).sum())
    assert np.array_equal(without["vector"], with_ps["vector"][:-n_ps])


@pytest.mark.parametrize("masked", [False, True])
def test_spectrum_is_computed_after_the_standardisation(masked: bool) -> None:
    """With ``standardize``, the spectrum of ``spec_apply`` is that of the field standardised by
    ``spec_standardize`` (maths §1.9, "Field"; decision 3 on #149): no fixture sets
    ``standardize``."""
    arrays = _arrays("statistics_ps_2d")
    name = "open_masked" if masked else "periodic"
    pbc = _configs("statistics_ps_2d")[name]["pbc"]
    mask = arrays[f"{name}_mask"].astype(bool) if masked else None
    field = arrays[f"{name}_field"]
    out = planar.spec_apply(
        field, 2, pbc, mask, None, 4, norm="none", standardize=True, compute_PS=True
    )
    x, invalid = planar.spec_data(field, 2, 3, mask, pbc)
    standardized, _, _ = planar.spec_standardize(x, 2, invalid)
    assert_parity(out["PS"], spec.spec_spectra(standardized, pbc)["PS"])


# ---------------------------------------------------------------------- the Bessel function J₀


def _exact_j0(x: float) -> float:
    """J₀(x) from its power series Σ_m (-x²/4)^m / (m!)², summed in decimal arithmetic with
    enough digits that the cancellation between its terms, of size up to e^x, costs nothing."""
    with localcontext() as context:
        context.prec = int(x / math.log(10)) + 40
        quarter = -(Decimal(x) ** 2) / 4
        total, term, m = Decimal(0), Decimal(1), 0
        while m < 2 or abs(term) > Decimal(10) ** -40:
            total += term
            m += 1
            term = term * quarter / (m * m)
        return float(total)


def test_bessel_j0_equals_its_series() -> None:
    """The oracle's J₀ against its power series summed exactly, to the 1e-12 of maths §1.9, on
    [0, 256 π] (images up to 512 pixels): the small arguments, the first zeros, the
    neighbourhood of x = 5, and points spread over the range."""
    rng = np.random.default_rng(149)
    x = np.concatenate(
        [
            [0.0, 1e-6, 0.5, 4.99, 5.0, 5.0001, 5.01, 14.0, 30.0],
            BESSEL_ZEROS,
            rng.uniform(0.0, 20.0, 12),
            rng.uniform(20.0, 256 * math.pi, 12),
            [256 * math.pi],
        ]
    )
    expected = np.array([_exact_j0(float(v)) for v in x])
    np.testing.assert_allclose(spec.spec_bessel_j0(x), expected, rtol=0, atol=BESSEL_TOLERANCE)


# ---------------------------------------------------------------------- errors


def _disjoint_maps() -> NDArray[np.float64]:
    """Two channels, each missing the half of the pixels that the other has, and one channel
    missing every pixel: no pixel is valid in every map."""
    field = np.ones((3, 16, 16))
    field[0, :, :8] = np.nan
    field[1, :, 8:] = np.nan
    field[2] = np.nan
    return field


def _observed_on_the_edge() -> NDArray[np.float64]:
    """One map observed on its first row only, inside no border (β_b ≥ 3 at 16 x 16)."""
    field = np.full((16, 16), np.nan)
    field[0] = 1.0
    return field


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda: spec.spec_spectra(np.zeros((2, 32, 31)), True), "not even", id="odd-size"
        ),
        pytest.param(
            lambda: spec.spec_spectrum_bins((32, 32), J=3, Jmin=2), "J_min", id="Jmin-too-large"
        ),
        pytest.param(
            lambda: spec.spec_spectra(np.where(np.eye(16) > 0, np.nan, 1.0), True, method="fft"),
            "missing pixels",
            id="nan-with-fft",
        ),
        pytest.param(
            lambda: spec.spec_spectra(np.zeros((16, 16)), None),  # type: ignore[arg-type]
            "pbc",
            id="pbc-none",
        ),
        pytest.param(
            lambda: planar.spec_apply(np.zeros((16, 16)), 2, True, norm="none", Jmin=1),
            "compute_PS=True",
            id="option-without-spectrum",
        ),
        pytest.param(
            lambda: planar.spec_apply(
                np.zeros((16, 16)), 2, True, J=1, norm="none", compute_PS=True
            ),
            "compute_PS=False",
            id="J-too-small",
        ),
        pytest.param(
            lambda: planar.spec_apply(np.zeros(64), 1, True, norm="none", compute_PS=True),
            "2D data only",
            id="spectrum-of-1d-data",
        ),
        pytest.param(
            lambda: spec.spec_spectra(_disjoint_maps(), True),
            "no pixel is valid in every batch element and channel",
            id="no-observed-pixel",
        ),
        pytest.param(
            lambda: spec.spec_spectra(_disjoint_maps(), False),
            "no pixel is valid in every batch element and channel",
            id="no-observed-pixel-open",
        ),
        pytest.param(
            lambda: spec.spec_spectra(_observed_on_the_edge(), False),
            "no observed pixel lies inside its border",
            id="null-bin-of-pbc-nan-mask",
        ),
        pytest.param(
            lambda: planar.spec_normalize(
                {"PS": np.ones((1, 1, 1, 6))}, {"sigma_PS": np.ones((1, 1, 4))}
            ),
            "the same bins",
            id="reference-with-other-bins",
        ),
    ],
)
def test_errors(call: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
