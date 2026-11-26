"""Parity of ``STOperator`` on HEALPix data with STL ``HealpixUpdate`` (task T5.6, issue #167;
maths.md §4 "Scattering loop", M-DATA-HPX, M-MASK-HPX; section 1 from M-MEAN to M-FLAT).

The fixture ``statistics_hpx`` (T5.3, STL ``HealpixUpdate`` at ``90eb1a9``) holds, for each
configuration: STL's statistics without normalisation (``vanilla``), with ``norm="store_ref"``
on a target field (``store``) and the reference its operator stored (``var_ref``, ``s2_ref``),
the statistics of a second field with ``norm="load_ref"`` and that reference (``load``), the
five chains of reductions of the ``store`` statistics, and the two flat vectors of
``to_flatten(flatten_complex=True)``. ESTEL computes the same on the stored inputs with one
``STOperator``, built with STL's ``standardize=False`` and applied with ``norm="none"``,
``"self"`` and ``"ref"`` (task T2.19), and one call of
``STStatistics.reduce`` per chain, and every array is compared with ``assert_parity``: same
shape, NaN at the same positions, values within the tolerances of Part II, §14.3, in the run's
dtype. ``mean`` takes the standard deviation of the data as its scale (§14.3).

The configurations compared with STL are the full skies, ``full_8``, ``full_16_cross`` and
``nan_16``. ``nan_16`` runs on the fields with their NaN and its mask given, as STL builds its
operator (OP-102), with ``ref_batch_mean=True``. The partial skies ``disc_16`` and
``disc_nan_16`` are not compared with STL, whose values at the edge of a partial sky are set by
its rounding (OP-110; decision 3 of the maintainer on #26 and #165): they are compared with the
oracle (``tests/spec/test_healpix_scattering.py``).

Tiers (Part I, §5.4; decision 7 on #167): ``full_8`` in the fast tier, in float32 too
(``dtype``), the representative parity case of HEALPix. Slow: the two configurations of
``nside`` 16 (``MARKS``), whose three operators build the tables of 3072 pixels and run the
loop at J = 3, over the budget; the fixtures exist at no smaller size.

A disagreement beyond the tolerance is a finding: report the configuration and the coefficients
on the issue, and change no fixture, tolerance or reference to make a test pass.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import DataHealpix, STOperator, STStatistics, _config
from estel.backend._protocol import DType
from tests.parity._compare import assert_parity, load_fixture, read_manifest

CASE = "statistics_hpx"
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
REDUCED = {
    "iso": ("S1", "S2", "S3", "S4"),
    "angular": ("S3", "S4"),
    "iso_angular": ("S3", "S4"),
    "iso_scale": ("S3", "S4"),
    "iso_angular_scale": ("S3", "S4"),
}
COMPARED_WITH_STL = ("full_8", "full_16_cross", "nan_16")
"""The configurations of ``statistics_hpx`` compared with STL: every full sky (OP-110)."""
MARKS = {
    "full_8": (pytest.mark.dtype,),
    "full_16_cross": (pytest.mark.slow,),  # nside 16: three operators over the budget
    "nan_16": (pytest.mark.slow,),  # nside 16, with the masks of the plan
}
pytestmark = pytest.mark.maths(
    "M-DATA-HPX", "M-MASK-HPX", "M-MEAN", "M-MOM", "M-S1", "M-S2", "M-S3", "M-S4", "M-CHAN",
    "M-NORM", "M-RED-ISO", "M-RED-ANG", "M-RED-SCALE", "M-FLAT",
)  # fmt: skip


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


def _configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = read_manifest()[f"{CASE}.npz"]["parameters"]["configs"]
    return configs


def _params() -> list[Any]:
    return [pytest.param(name, marks=MARKS.get(name, ())) for name in COMPARED_WITH_STL]


class _Run:
    """ESTEL on one configuration: the three operators, their statistics and the reductions."""

    def __init__(self, name: str) -> None:
        self.config = config = _configs()[name]
        self.fixture = fixture = load_fixture(CASE)
        depth = int(config["nside"]).bit_length() - 1
        cell_ids = fixture[f"{name}_cell_ids"].astype(np.int64)
        # The fixture stores every pixel of a full sky; DataHealpix takes no cell_ids there.
        options: dict[str, Any] = (
            {"depth": depth}
            if cell_ids.size == 12 * 4**depth
            else {"depth": depth, "cell_ids": cell_ids}
        )
        mask = fixture[f"{name}_mask"] > 0 if f"{name}_mask" in fixture else None
        target = DataHealpix(fixture[f"{name}_field"], mask=mask, **options)
        other = DataHealpix(fixture[f"{name}_ref_field"], mask=mask, **options)
        J = config["J"]  # noqa: N806 (maths.md notation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # OP-3: tested with the plan
            # STL's vanilla, store_ref, load_ref and norm_batch_mean are ESTEL's norm="none",
            # norm="self", norm="ref" and ref_batch_mean: one operator applied with each, as in
            # STL, with its standardize=False (task T2.19).
            self.operator = STOperator(target, J, standardize=False)
            self.vanilla = self.operator.apply(target, norm="none")
            self.stored = self.operator.apply(
                target, norm="self", ref_batch_mean=config["norm_batch_mean"]
            )
            self.loaded = self.operator.apply(other, norm="ref")
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
def _run(name: str, dtype: DType) -> _Run:
    """One run per configuration and dtype, shared by the three tests below, which check
    different outputs of the same computation; ``dtype`` is the run's, which ``_run_dtype``
    sets."""
    return _Run(name)


@pytest.mark.parametrize("name", _params())
def test_statistics_equal_stl(name: str, dtype: DType) -> None:
    """The statistics without normalisation, with ``norm="self"`` and its reference, and those of
    a second field with ``norm="ref"``."""
    run = _run(name, dtype)
    reference = run.operator._reference
    assert reference is not None
    var_ref = run.expected(name, "var_ref")
    assert_parity(np.asarray(reference.var), var_ref, dtype)
    assert_parity(np.asarray(reference.s2_sqrt), run.expected(name, "s2_ref"), dtype)
    raw = run.expected(name, "vanilla_var")
    for kind, stats in (("vanilla", run.vanilla), ("store", run.stored), ("load", run.loaded)):
        # The variance before normalisation, for the scale of mean (§14.3).
        raw_var = raw if kind == "vanilla" else run.expected(name, f"{kind}_var") * var_ref
        for family in FAMILIES:
            scale = float(np.sqrt(np.max(raw_var))) if family == "mean" else None
            expected = run.expected(name, f"{kind}_{family}")
            assert_parity(np.asarray(stats.full_grid(family)), expected, dtype, scale=scale)


@pytest.mark.parametrize("name", _params())
def test_reduced_statistics_equal_stl(name: str, dtype: DType) -> None:
    """The five chains of STL's reductions, each one call of ``reduce``."""
    run = _run(name, dtype)
    for key, families in REDUCED.items():
        for family in families:
            expected = run.expected(name, f"{key}_{family}")
            assert_parity(np.asarray(run.reduced[key].full_grid(family)), expected, dtype)


@pytest.mark.parametrize("name", _params())
def test_vectors_equal_stl(name: str, dtype: DType) -> None:
    """``to_vector`` is STL's ``to_flatten(flatten_complex=True)``, of the ``norm="self"``
    statistics and of their three reductions in a row."""
    run = _run(name, dtype)
    assert_parity(np.asarray(run.stored.to_vector()), run.expected(name, "vector"), dtype)
    reduced = run.reduced["iso_angular_scale"].to_vector()
    assert_parity(np.asarray(reduced), run.expected(name, "vector_reduced"), dtype)


def test_the_configurations_exercise_what_they_are_named_for() -> None:
    """``nan_16`` has its missing pixels as NaN in both fields, its mask given, and a reference
    averaged over a batch; ``full_16_cross`` has two channels; the three are full skies."""
    configs, fixture = _configs(), load_fixture(CASE)
    for name in COMPARED_WITH_STL:
        nside = configs[name]["nside"]
        assert fixture[f"{name}_cell_ids"].size == 12 * nside**2, name
    mask = fixture["nan_16_mask"] > 0
    assert mask.any()
    for key in ("field", "ref_field"):
        assert np.array_equal(np.isnan(fixture[f"nan_16_{key}"]).any(axis=(0, 1)), mask), key
    assert configs["nan_16"]["norm_batch_mean"]
    assert configs["nan_16"]["Nb"] == 2
    assert configs["full_16_cross"]["Nc"] == 2
