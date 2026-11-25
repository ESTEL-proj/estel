"""Spec parity of ``STOperator`` on HEALPix data: ESTEL against the oracle ``spec_apply_hpx``
(task T5.6, issue #167; maths.md §4 "Scattering loop", M-DATA-HPX, M-MASK-HPX, M-DEF-HPX;
section 1 from M-MEAN to M-FLAT; Part I, §4.2).

ESTEL's ``STOperator`` and the oracle run on the same inputs, in float64, and every family is
compared with ``assert_parity``: same full-grid shape, NaN at the same positions, values within
the float64 tolerances of Part II, §14.3 (for ``mean``, the standard deviation of the data as
its scale). The helper's messages name STL as the expected side: here it is the oracle.

Two kinds of cases, under decision 7 of the maintainer on #167:

- **The partial skies of the fixtures,** ``disc_16`` and ``disc_nan_16`` of ``statistics_hpx``,
  in full, as the full skies are compared with STL (``tests/parity/test_healpix_statistics.py``):
  the statistics without normalisation, with ``norm="self"`` and its reference, those of the
  second field with ``norm="ref"``, the five chains of reductions and the two vectors. They are
  not compared with STL, whose values at the edge of a partial sky are set by its rounding, but
  with the oracle, which applies the threshold of OP-110 as ESTEL does (decision 3 on #26 and
  #165). ``disc_nan_16`` has its mask inferred from the NaN of its single map.
- **The options the fixtures do not cover,** at ``nside`` 4 and J = 2, on a masked full sky
  and a masked partial sky: ``standardize=True``, a ``cross_matrix`` without the auto statistics
  of a channel (OP-30), and ``norm="ref"`` with a reference taken with and without
  ``ref_batch_mean``.

Fast tier: the cases at ``nside`` 4 (decision 7 on #167). Full tier: the discs of ``nside`` 16.

A disagreement is a finding about ``docs/maths.md``, the oracle or ESTEL: report the M- ID and
the coefficients on the issue, change nothing to make a test pass.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import DataHealpix, STOperator, STStatistics, _config
from estel._spec import healpix as spec
from estel._spec import planar
from tests.parity._compare import assert_parity, load_fixture, read_manifest

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
pytestmark = pytest.mark.maths(
    "M-DATA-HPX", "M-MASK-HPX", "M-DEF-HPX", "M-MEAN", "M-MOM", "M-STD", "M-W1", "M-S1",
    "M-S2", "M-S3", "M-S4", "M-CHAN", "M-ORDER", "M-NORM", "M-RED-ISO", "M-RED-ANG",
    "M-RED-SCALE", "M-FLAT",
)  # fmt: skip

# Channel 1 has cross statistics but no auto statistics (OP-30).
CROSS_MATRIX = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
OPTIONS: dict[str, dict[str, Any]] = {
    "standardize": {"nc": 2, "standardize": True},
    "cross_matrix": {"nc": 3, "cross_matrix": CROSS_MATRIX},
    "ref": {"nc": 2},
    "ref_batch_mean": {"nc": 2, "ref_batch_mean": True},
}
"""The options of the cases at ``nside`` 4; ``ref`` and ``ref_batch_mean`` are ``norm="ref"``
with a reference taken without and with the batch mean."""
SKIES = {"full": None, "partial": np.arange(40, 150)}
"""The pixels at ``nside`` 4: the full sky, and a partial sky across the north cap and the
equatorial faces."""
CHAIN = {
    "iso": True,
    "angular_fourier": True,
    "angular_harmonics": 3,
    "scale_cosine": True,
    "scale_harmonics": 1,
    "dj": 2,
}
"""The three reductions in a row, truncated (M-ORDER)."""


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec parity is a float64 comparison (Part I, §5.1)."""
    monkeypatch.setattr(_config, "_dtype", "float64")


def _compare(actual: STStatistics, expected: dict[str, Any], scale: float) -> None:
    for family in FAMILIES:
        assert_parity(
            np.asarray(actual.full_grid(family)),
            expected[family],
            scale=scale if family == "mean" else None,
        )


def _reduced(normalized: dict[str, Any], chain: dict[str, Any]) -> dict[str, Any]:
    """The oracle's reductions of ``chain``, in the order of M-ORDER, and their vector."""
    out = normalized
    iso = bool(chain.get("iso"))
    if iso:
        out = planar.spec_iso(out)
    if chain.get("angular_fourier"):
        out = planar.spec_angular_fourier(out, chain.get("angular_harmonics"), after_iso=iso)
    if chain.get("scale_cosine"):
        out = planar.spec_scale_cosine(out, chain.get("scale_harmonics"), chain.get("dj"))
    return {**out, "vector": planar.spec_flatten(out, flatten_complex=True)}


def _check(
    target: NDArray[np.float64],
    other: NDArray[np.float64],
    depth: int,
    cell_ids: NDArray[np.int64] | None,
    mask: NDArray[np.bool_] | None,
    J: int,  # noqa: N803 (maths.md notation)
    chains: list[dict[str, Any]],
    *,
    cross_matrix: NDArray[np.bool_] | None = None,
    ref_batch_mean: bool = False,
    standardize: bool = False,
) -> None:
    """ESTEL against the oracle: without normalisation and its vector, ``norm="self"`` and its
    reference, ``norm="ref"`` on ``other``, and each chain of reductions with its vector. The
    oracle runs twice, on the target without normalisation and on ``other`` with
    ``norm="ref"``; the rest is derived from the first result, function by function. ``mask``
    ``None`` with NaN in a single map lets both infer it (M-DATA-HPX)."""
    options: dict[str, Any] = {"cross_matrix": cross_matrix, "standardize": standardize}
    sky: dict[str, Any] = {"depth": depth, "cell_ids": cell_ids}
    data, second = DataHealpix(target, mask=mask, **sky), DataHealpix(other, mask=mask, **sky)
    assert (data.mask is None) == (second.mask is None)
    assert data.mask is None or np.array_equal(data.mask, np.asarray(second.mask))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-30: tested with the plan
        raw = STOperator(data, J, norm="none", **options).apply(data)
        stored = STOperator(data, J, norm="self", ref_batch_mean=ref_batch_mean, **options).apply(
            data
        )
        loaded = STOperator(data, J, norm="ref", **options).apply(second, ref=stored)

    def oracle(array: NDArray[np.float64], **more: Any) -> dict[str, Any]:
        return spec.spec_apply_hpx(array, 1 << depth, cell_ids, mask, J, 4, **options, **more)

    expected = oracle(target, norm="none", flatten=True)
    scale = float(np.sqrt(np.nanmax(expected["var"])))
    _compare(raw, expected, scale)
    assert_parity(np.asarray(raw.to_vector()), expected["vector"])
    if standardize:
        assert raw._standardization is not None
        for key, removed in zip(("mean_pre_std", "std_pre_std"), raw._standardization, strict=True):
            assert_parity(np.asarray(removed), expected[key])
    reference = planar.spec_reference(expected, ref_batch_mean)
    assert stored._reference is not None
    assert_parity(np.asarray(stored._reference.var), reference["var"])
    assert_parity(np.asarray(stored._reference.s2_sqrt), reference["sigma"])
    normalized = planar.spec_normalize(expected, reference)
    _compare(stored, normalized, scale)
    _compare(loaded, oracle(other, norm="ref", ref=reference), scale)
    for chain in chains:
        reduced = stored.reduce(**chain)
        expected_reduced = _reduced(normalized, chain)
        _compare(reduced, expected_reduced, scale)
        assert_parity(np.asarray(reduced.to_vector()), expected_reduced["vector"])


def _fields(nc: int, npix: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two seeded batches of two maps, with a non-zero mean and correlated channels."""
    rng = np.random.default_rng(167)
    common = rng.normal(size=(2, 2, 1, npix))
    fields = rng.normal(size=(2, 2, nc, npix)) + 0.5 * common + 0.3
    return fields[0], fields[1]


@pytest.mark.parametrize("option", sorted(OPTIONS))
@pytest.mark.parametrize("sky", sorted(SKIES))
def test_options_equal_the_oracle(sky: str, option: str) -> None:
    """The options the fixtures do not cover, at ``nside`` 4, J = 2, two maps, three missing
    pixels given as a mask, on the full sky and on a partial sky."""
    settings = dict(OPTIONS[option])
    nc = settings.pop("nc")
    cell_ids = SKIES[sky]
    npix = 192 if cell_ids is None else cell_ids.size
    mask = np.zeros(npix, dtype=bool)
    mask[[5, 6, 70]] = True
    target, other = _fields(nc, npix)
    _check(target, other, 2, cell_ids, mask, 2, [CHAIN], **settings)


# ---------------------------------------------------------------------- the partial skies of T5.3


@functools.cache
def _statistics() -> dict[str, NDArray[Any]]:
    return load_fixture("statistics_hpx")


def _configuration(name: str) -> dict[str, Any]:
    config: dict[str, Any] = read_manifest()["statistics_hpx.npz"]["parameters"]["configs"][name]
    return config


@pytest.mark.slow
@pytest.mark.parametrize("name", ["disc_16", "disc_nan_16"])
def test_partial_skies_equal_the_oracle(name: str) -> None:
    """The partial skies of ``statistics_hpx`` in full: the statistics of the three
    normalisations, the reference, the five chains of STL's reductions and the two vectors
    (decision 3 on #26 and #165). ``disc_nan_16`` holds NaN in a single map, and its mask is
    inferred by both from them.

    Slow: ``nside`` 16, which decision 7 on #167 puts in the full tier, with the oracle's naive
    tables (decision 4 on #165); each case takes 0.1 to 0.3 s of CPU locally."""
    fixture, config = _statistics(), _configuration(name)
    nside = config["nside"]
    cell_ids = fixture[f"{name}_cell_ids"].astype(np.int64)
    assert cell_ids.size < 12 * nside**2
    mask = None if config.get("mask_from_nan") else fixture.get(f"{name}_mask")
    angular = {"angular_fourier": True, "angular_harmonics": config["harmonics_angle"]}
    scale = {
        "scale_cosine": True,
        "scale_harmonics": config["harmonics_scale"],
        "dj": config["dj"],
    }
    chains = [
        {"iso": True},
        angular,
        {"iso": True, **angular},
        {"iso": True, **scale},
        {"iso": True, **angular, **scale},
    ]
    if name == "disc_nan_16":
        assert np.isnan(fixture[f"{name}_field"]).any()
    _check(
        fixture[f"{name}_field"],
        fixture[f"{name}_ref_field"],
        nside.bit_length() - 1,
        cell_ids,
        None if mask is None else mask > 0,
        config["J"],
        chains,
        ref_batch_mean=config["norm_batch_mean"],
    )
