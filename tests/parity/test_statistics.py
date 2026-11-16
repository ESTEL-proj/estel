"""Parity of the normalised and reduced statistics with STL (task T2.5, issue #17; maths.md
M-NORM, M-RED-ISO, M-RED-ANG, M-RED-SCALE, M-FLAT; OP-3, OP-30, OP-37).

The fixtures ``statistics_2d`` (STL `master`) and ``statistics_1d`` (STL `2D+1D`) hold, for each
configuration of their case (``tools/stl_parity/cases/``): STL's statistics with
``norm="store_ref"`` on a target field and the reference its operator stored, the statistics of
a second field with ``norm="load_ref"`` and that reference, the reductions of the ``store_ref``
statistics (``iso``, ``angular_ft`` with and without ``iso``, ``iso`` then ``scale_ft``, and the
three in a row), and the flat vectors of ``to_flatten(flatten_complex=True)``. ESTEL computes the
same on the stored inputs with ``STOperator`` (``norm="self"`` and ``norm="ref"``) and one call
of ``STStatistics.reduce`` per chain of STL's reductions, and every array is compared with
``assert_parity``: same shape, NaN at the same positions, values within the tolerances of
Part II, §14.3, in the run's dtype (float64 is the parity check, float32 its secondary check).
``mean``, whose theoretical value may be zero, takes the standard deviation of the data as its
scale: atol = 0.1 x rtol x sqrt(max(var)), with the variance before normalisation (§14.3).

Slow: the configuration ``statistics_2d`` ``deep_open`` (64 x 64 pixels, J = 4) runs in the full
tier only (``MARKS``).
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data, Data1D, Data2D, STOperator, STStatistics, _config
from estel.backend._protocol import DType
from tests.parity._compare import assert_parity, load_fixture, read_manifest

CASES: dict[str, type[Data]] = {"statistics_2d": Data2D, "statistics_1d": Data1D}
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
REDUCED = {
    "iso": ("S1", "S2", "S3", "S4"),
    "angular": ("S3", "S4"),
    "iso_angular": ("S3", "S4"),
    "iso_scale": ("S3", "S4"),
    "iso_angular_scale": ("S3", "S4"),
}
pytestmark = pytest.mark.maths("M-NORM", "M-RED-ISO", "M-RED-ANG", "M-RED-SCALE", "M-FLAT")
# Tiers (Part I, §5.4): the 64 x 64, J = 4 configuration is large, so slow; one representative
# configuration per dimension also runs in float32. On JAX the fast tier runs one parity case, of
# the scattering statistics (tests/parity/test_scattering.py); these run in the full tier. The 2D
# periodic cases of the parity tests share one xdist worker (``--dist loadgroup``), where JAX
# compiles their operations once: they have the same shapes.
MARKS = {
    ("statistics_2d", "deep_open"): (pytest.mark.slow,),
    ("statistics_2d", "periodic"): (
        pytest.mark.dtype,
        pytest.mark.xdist_group("parity_2d_periodic"),
    ),
    ("statistics_1d", "periodic"): (pytest.mark.dtype,),
}


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _configs() -> list[Any]:
    return [
        pytest.param(case, name, marks=MARKS.get((case, name), ()))
        for case in sorted(CASES)
        for name in _parameters(case)["configs"]
    ]


class _Run:
    """ESTEL on one configuration: the operators, the statistics and their reductions."""

    def __init__(self, case: str, name: str) -> None:
        parameters = _parameters(case)
        self.config = config = parameters["configs"][name]
        self.fixture = fixture = load_fixture(case)
        cls = CASES[case]
        mask = fixture[f"{name}_mask"] > 0 if f"{name}_mask" in fixture else None
        target = cls(fixture[f"{name}_field"], mask=mask, pbc=config["pbc"])
        other = cls(fixture[f"{name}_ref_field"], mask=mask, pbc=config["pbc"])
        matrix = config.get("cross_matrix")
        options: dict[str, Any] = {
            "cross_matrix": None if matrix is None else np.array(matrix, dtype=bool),
            "compute_PS": False if cls.ndim == 2 else None,
        }
        J, L = config["J"], config.get("L", parameters["L"])  # noqa: N806
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30: tested elsewhere
            # STL's store_ref, load_ref and norm_batch_mean are ESTEL's norm="self", norm="ref"
            # and ref_batch_mean.
            self.operator = STOperator(
                target, J, L, norm="self", ref_batch_mean=config["norm_batch_mean"], **options
            )
            self.stored = self.operator.apply(target)
            loading = STOperator(target, J, L, norm="ref", **options)
            self.loaded = loading.apply(other, ref=self.stored)
        # STL's names map to ESTEL's: to_iso, to_angular_ft(harmonics_angle) and
        # to_scale_ft(harmonics_scale, dj) are one call of reduce, whatever the chain.
        angular: dict[str, Any] = {
            "angular_fourier": True,
            "angular_harmonics": config["harmonics_angle"],
        }
        scale: dict[str, Any] = {
            "scale_cosine": True,
            "scale_harmonics": config["harmonics_scale"],
            "dj": config["dj"],
        }
        stored = self.stored
        self.reduced: dict[str, STStatistics] = {
            "iso": stored.reduce(iso=True),
            "angular": stored.reduce(**angular),
            "iso_angular": stored.reduce(iso=True, **angular),
            "iso_scale": stored.reduce(iso=True, **scale),
            "iso_angular_scale": stored.reduce(iso=True, **angular, **scale),
        }

    def expected(self, name: str, key: str) -> NDArray[Any]:
        return self.fixture[f"{name}_{key}"]


@functools.cache
def _run(case: str, name: str, dtype: DType) -> _Run:
    """One run per configuration and dtype, shared by the three tests below, which check different
    outputs of the same computation; ``dtype`` is the run's, which ``_run_dtype`` sets."""
    return _Run(case, name)


@pytest.mark.parametrize(("case", "name"), _configs())
def test_normalized_statistics_equal_stl(case: str, name: str, dtype: DType) -> None:
    run = _run(case, name, dtype)
    reference = run.operator._reference
    assert reference is not None
    var_ref = run.expected(name, "var_ref")
    assert_parity(np.asarray(reference.var), var_ref, dtype)
    assert_parity(np.asarray(reference.s2_sqrt), run.expected(name, "s2_ref"), dtype)
    for kind, stats in (("store", run.stored), ("load", run.loaded)):
        # The variance before normalisation, for the scale of mean (§14.3).
        raw_var = run.expected(name, f"{kind}_var") * var_ref
        for family in FAMILIES:
            scale = float(np.sqrt(np.max(raw_var))) if family == "mean" else None
            expected = run.expected(name, f"{kind}_{family}")
            assert_parity(np.asarray(stats.full_grid(family)), expected, dtype, scale=scale)


@pytest.mark.parametrize(("case", "name"), _configs())
def test_reduced_statistics_equal_stl(case: str, name: str, dtype: DType) -> None:
    run = _run(case, name, dtype)
    for key, families in REDUCED.items():
        for family in families:
            expected = run.expected(name, f"{key}_{family}")
            assert_parity(np.asarray(run.reduced[key].full_grid(family)), expected, dtype)


@pytest.mark.parametrize(("case", "name"), _configs())
def test_vectors_equal_stl(case: str, name: str, dtype: DType) -> None:
    """``to_vector`` is STL's ``to_flatten(flatten_complex=True)``, OP-37 included; the
    coefficients it leaves out (structural, OP-3, OP-30, missing scales of ``scale_cosine``) are
    decided statically, and are exactly the NaN STL removes."""
    run = _run(case, name, dtype)
    assert_parity(np.asarray(run.stored.to_vector()), run.expected(name, "vector"), dtype)
    reduced = run.reduced["iso_angular_scale"].to_vector()
    assert_parity(np.asarray(reduced), run.expected(name, "vector_reduced"), dtype)


def test_fixtures_cover_the_cells_of_t25() -> None:
    """The configurations are useful only if they exercise what they are named for."""
    for case in CASES:
        parameters = _parameters(case)
        configs = parameters["configs"]
        fixture = load_fixture(case)
        assert {c["norm_batch_mean"] for c in configs.values()} == {False, True}, case
        assert any("mask" in c for c in configs.values()), case
        # Truncations: fewer harmonics than L and than J, and dj below J.
        assert any(c["harmonics_angle"] < c.get("L", parameters["L"]) for c in configs.values()), (
            case
        )
        assert any(c["harmonics_scale"] < c["J"] for c in configs.values()), case
        assert any(c["dj"] < c["J"] for c in configs.values()), case
        # OP-30: channel 1 has no auto statistics, so its cross S1 are NaN after normalisation.
        matrix = np.array(configs["cross_matrix"]["cross_matrix"])
        assert not matrix[1, 1], case
        assert matrix[1, 2], case
        assert np.isnan(fixture["cross_matrix_store_S1"][0, 1, 2]).all(), case
        # OP-3: an empty mean reaches scale_cosine: a harmonic is NaN at (k, a) with k + a < J,
        # where the reparametrisation reads a scale pair of the index set.
        n_scales = configs["deep_open"]["J"]
        s3 = fixture["deep_open_iso_scale_S3"][0, 0, 0, :, :, 0]
        inside = np.add.outer(np.arange(s3.shape[0]), np.arange(s3.shape[1])) < n_scales
        assert np.isnan(s3[inside]).any(), case
