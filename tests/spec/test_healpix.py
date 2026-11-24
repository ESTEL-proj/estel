"""The oracle of HEALPix (task T5.4, issue #165; maths.md section 4; Part I, §1.5 and §4.2; Part II,
§12.11 and §14.3).

Two kinds of checks, under the decisions of the maintainer on #165:

- **The oracle's own geometry against healpy.** The NESTED pixel centres, the ring ordering and
  the bilinear interpolation of ``estel._spec.healpix`` are compared with healpy's documented
  functions (``pix2ang``, ``ring2nest``, ``get_interp_weights``) on dense samples. healpy (GPL-2)
  is test-only (Part I, §1.5): it is imported inside the tests, never by the package. The
  interpolation weights are compared as linear functionals (the weight of each pixel, summed
  over the 4 slots), since a zero weight may sit on another pixel (OP-92).
- **The oracle against the fixtures of T5.3** (``operators_hpx``, ``statistics_hpx``, STL
  ``HealpixUpdate`` at ``90eb1a9``), with ``assert_parity``: same shapes, NaN at the same
  positions, the float64 tolerances of Part II, §14.3.

Fast tier: the geometry against healpy, and the operators at ``nside`` 4 and 8. Full tier: the
operators at ``nside`` 16, the masks and the statistics, as the oracle is naive by design.

**The edge of a partial sky** (OP-110, decisions of the maintainer on #165). At a stencil point
that falls on a pixel longitude or on a ring, one interpolation weight is null in exact
arithmetic and of rounding size in practice; on a partial sky STL's result there is set by its
rounding, while the oracle drops every neighbour of weight below 10⁻¹² (M-CONV-HPX). So:

- the convolutions of ``disc_8`` and ``disc_16`` are compared with STL on every (gauge, pixel)
  row that is not ill-conditioned: a row is ill-conditioned when one of its stencil points has
  no neighbour in ``cell_ids`` with a weight of at least 10⁻¹², while one of its 4 neighbours,
  in ``cell_ids`` or not, has a weight below 10⁻¹²;
- the statistics of ``disc_16`` and ``disc_nan_16`` are not compared with STL, since a mean mixes
  every pixel: T5.6 compares them between ESTEL and the oracle. Their masks, their decimations
  and the pixels of their levels are compared with STL.

A disagreement is a finding about ``docs/maths.md`` or about the oracle: do not change the oracle,
a fixture or a tolerance to make a test pass; report the M- ID, the configuration and the
coefficients on the issue.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel._spec import healpix as spec
from estel._spec import planar
from tests.parity._compare import assert_parity, load_fixture, read_manifest

pytestmark = pytest.mark.maths(
    "M-DATA-HPX", "M-CONV-HPX", "M-WAV-HPX", "M-DOWN-HPX", "M-MASK-HPX", "M-DEF-HPX"
)
GEOMETRY_RTOL = 1e-12
"""The agreement asked of the oracle's geometry with healpy (decision 3 on #165)."""
ILL_CONDITIONED = {"full_8": 0, "disc_8": 16, "full_16": 0, "disc_16": 14}
"""The number of ill-conditioned (gauge, pixel) rows of each domain of ``operators_hpx``,
left out of the comparison with STL (OP-110)."""


def _healpy() -> Any:
    """healpy, imported by the tests only: GPL-2, never imported by the package (Part I, §1.5)."""
    import healpy  # type: ignore[import-untyped]

    return healpy


def _functional(index: NDArray[Any], weight: NDArray[Any], npix: int) -> dict[int, float]:
    """The interpolation as linear functionals: the weight of each (point, pixel) pair, summed
    over the 4 slots, keyed by point x npix + pixel; pairs of null weight are left out."""
    keys = (np.arange(index.shape[1]) * npix + index).reshape(-1)
    unique, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=weight.reshape(-1))
    return {int(k): float(w) for k, w in zip(unique, sums, strict=True) if w != 0.0}


def _functional_difference(actual: dict[int, float], expected: dict[int, float]) -> float:
    return max(abs(actual.get(k, 0.0) - expected.get(k, 0.0)) for k in actual.keys() | expected)


def _samples(nside: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dense samples of the sphere: uniform points, points near and at the poles, the pixel
    centres, points on the ring colatitudes, and longitudes outside [0, 2π)."""
    rng = np.random.default_rng(165 + nside)
    n = 2000
    cap = rng.uniform(0.0, 0.3, 300)
    rings, _ = spec.spec_rings(nside)
    theta_c, phi_c = spec.spec_pixel_centres(nside)
    theta = np.concatenate(
        [
            np.arccos(rng.uniform(-1.0, 1.0, n)),
            cap,
            math.pi - cap,
            [0.0, 0.0, math.pi, math.pi],
            theta_c,
            np.repeat(rings, 8),
        ]
    )
    phi = np.concatenate(
        [
            rng.uniform(0.0, 2.0 * math.pi, n),
            rng.uniform(0.0, 2.0 * math.pi, 600),
            [0.0, 1.0, 0.0, 4.0],
            phi_c,
            rng.uniform(-2.0 * math.pi, 4.0 * math.pi, 8 * rings.size),
        ]
    )
    return theta, phi


# ---------------------------------------------------------------------- geometry against healpy


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16, 32])
def test_pixel_centres_equal_healpy(nside: int) -> None:
    """M-DATA-HPX: the NESTED pixel centres, against ``healpy.pix2ang(nest=True)``."""
    healpy = _healpy()

    theta, phi = spec.spec_pixel_centres(nside)
    expected_theta, expected_phi = healpy.pix2ang(nside, np.arange(12 * nside**2), nest=True)
    np.testing.assert_allclose(theta, expected_theta, rtol=0, atol=GEOMETRY_RTOL)
    np.testing.assert_allclose(phi, expected_phi, rtol=0, atol=GEOMETRY_RTOL)
    some = np.array([0, 12 * nside**2 - 1, 5 * nside**2 + 1])
    assert np.array_equal(spec.spec_pixel_centres(nside, some)[0], theta[some])


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16])
def test_rings_are_the_ring_ordering(nside: int) -> None:
    """M-CONV-HPX, step 3: the rings, their pixels by increasing longitude, are HEALPix's RING
    ordering, against ``healpy.ring2nest``."""
    healpy = _healpy()

    theta, pixels = spec.spec_rings(nside)
    assert theta.shape == (4 * nside - 1,)
    assert np.all(np.diff(theta) > 0)
    ring_order = pixels[pixels >= 0]
    assert np.array_equal(ring_order, healpy.ring2nest(nside, np.arange(12 * nside**2)))


@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16, 32])
def test_interpolation_equals_healpy(nside: int) -> None:
    """M-CONV-HPX, step 3: the bilinear interpolation, against
    ``healpy.get_interp_weights(nest=True)``, as linear functionals, on dense samples with the
    poles, the pixel centres and the ring colatitudes; the weights sum to 1."""
    healpy = _healpy()

    theta, phi = _samples(nside)
    index, weight = spec.spec_interp_weights(nside, theta, phi)
    assert index.shape == weight.shape == (4, theta.size)
    np.testing.assert_allclose(weight.sum(axis=0), 1.0, rtol=0, atol=1e-14)
    assert np.all(weight >= -1e-15)
    expected_index, expected_weight = healpy.get_interp_weights(nside, theta, phi, nest=True)
    npix = 12 * nside**2
    actual = _functional(index, weight, npix)
    expected = _functional(expected_index, expected_weight, npix)
    assert _functional_difference(actual, expected) <= GEOMETRY_RTOL


def test_rotation_carries_the_stencil_axes() -> None:
    """M-CONV-HPX, step 2: R carries the North Pole to the pixel, the u axis onto
    cos alpha e_θ + sin alpha e_φ and the v axis onto -sin alpha e_θ + cos alpha e_φ; R is a
    rotation."""
    rng = np.random.default_rng(2)
    theta = np.arccos(rng.uniform(-1, 1, 50))
    phi, alpha = rng.uniform(0, 2 * math.pi, (2, 50))
    rotation = spec.spec_rotation(theta, phi, alpha)
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    e_theta = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], -1
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros(50)], -1)
    c, s = np.cos(alpha)[:, None], np.sin(alpha)[:, None]
    np.testing.assert_allclose(rotation[..., 2], n, atol=1e-15)
    np.testing.assert_allclose(rotation[..., 0], c * e_theta + s * e_phi, atol=1e-15)
    np.testing.assert_allclose(rotation[..., 1], -s * e_theta + c * e_phi, atol=1e-15)
    np.testing.assert_allclose(
        rotation @ np.swapaxes(rotation, -1, -2), np.eye(3)[None].repeat(50, 0), atol=1e-15
    )
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-14)


def test_stencil_and_gauge_cosmo() -> None:
    """M-CONV-HPX, steps 1 and 2: the stencil's points, and the gauge ``"cosmo"``: regular at
    the poles, where the u axis of gauge 0 tends to +x in the north and -x in the south."""
    colatitude, azimuth = spec.spec_stencil(4)
    alpha = math.sqrt(4 * math.pi / (12 * 16))
    p = (0 + 2) * 5 + (1 + 2)  # (u, v) = (1, 0)
    assert colatitude[p] == pytest.approx(alpha)
    assert azimuth[p] == 0.0
    p = (2 + 2) * 5 + (0 + 2)  # (u, v) = (0, 2)
    assert colatitude[p] == pytest.approx(2 * alpha)
    assert azimuth[p] == pytest.approx(math.pi / 2)
    assert colatitude[12] == 0.0

    phi = np.linspace(0, 2 * math.pi, 9)
    for theta, sign in ((1e-6, 1.0), (math.pi - 1e-6, -1.0)):
        angles = spec.spec_gauge_angles(np.full(9, theta), phi, 4)
        assert angles.shape == (4, 9)
        u_axis = spec.spec_rotation(theta, phi, angles[0])[..., 0]
        np.testing.assert_allclose(u_axis, np.tile([sign, 0.0, 0.0], (9, 1)), atol=1e-5)
    # The equator follows the northern rule.
    assert spec.spec_gauge_angles(math.pi / 2, 1.0, 2).tolist() == [-1.0, -1.0 + math.pi / 2]


def test_mirror_through_the_equator_conjugates_the_convolution() -> None:
    """M-CONV-HPX, "The gauge cosmo": for a map mirrored through the equator, the convolution at
    the mirrored pixel is the complex conjugate of that of the map, off the equatorial ring."""
    nside = 4
    theta, phi = spec.spec_pixel_centres(nside)
    key = {(round(t, 9), round(p, 9)): k for k, (t, p) in enumerate(zip(theta, phi, strict=True))}
    mirror = np.array(
        [key[round(math.pi - t, 9), round(p, 9)] for t, p in zip(theta, phi, strict=True)]
    )
    x = np.random.default_rng(3).normal(size=theta.size)
    tables = spec.spec_conv_tables(nside)
    kernel = spec.spec_wavelet_hpx()
    y, y_mirrored = (
        spec.spec_conv_hpx(x, tables, kernel),
        spec.spec_conv_hpx(x[mirror], tables, kernel),
    )
    off = np.abs(theta - math.pi / 2) > 1e-9
    np.testing.assert_allclose(y_mirrored[:, off], np.conj(y[:, mirror][:, off]), atol=1e-13)
    assert np.abs(y_mirrored[:, ~off] - np.conj(y[:, mirror][:, ~off])).max() > 1e-3


# ---------------------------------------------------------------------- operators against STL


@cache
def _operators() -> dict[str, NDArray[Any]]:
    return load_fixture("operators_hpx")


def test_kernel_equals_the_fixture() -> None:
    """M-WAV-HPX: the kernel, its sum, the sum of its moduli and its two symmetries."""
    kernel = spec.spec_wavelet_hpx()
    assert_parity(kernel, _operators()["kernel"].reshape(-1))
    assert abs(kernel.sum()) < 1e-15
    assert np.abs(kernel).sum() == pytest.approx(1.0, abs=1e-15)
    grid = kernel.reshape(5, 5)  # [v, u]
    np.testing.assert_allclose(grid[:, ::-1], np.conj(grid), atol=1e-18)
    np.testing.assert_allclose(grid[::-1, :], grid, atol=1e-18)


def test_dense_operators_equal_the_fixture() -> None:
    """M-CONV-HPX, M-DOWN-HPX: the convolution at ``nside`` 4 as a dense ``(L, Npix, Npix)``
    operator, and the decimation from ``nside`` 4 to 2, full sky."""
    fixture = _operators()
    tables = spec.spec_conv_tables(4)
    kernel = spec.spec_wavelet_hpx()
    dense = np.stack([spec.spec_conv_hpx(e, tables, kernel) for e in np.eye(192)], axis=-1)
    assert_parity(dense, fixture["conv_4"])
    matrix, coarse = spec.spec_decimation_hpx(4)
    assert_parity(matrix, fixture["down_4"])
    assert np.array_equal(coarse, fixture["down_4_cell_ids"].astype(np.int64))
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-15)


def _ill_conditioned(nside: int, cell_ids: NDArray[np.int64]) -> NDArray[np.bool_]:
    """The ill-conditioned (gauge, pixel) rows, ``(L, Npix)`` (OP-110; decision 2 on #165): one
    of the row's stencil points has no neighbour in ``cell_ids`` with a weight of at least
    10⁻¹², while one of its 4 neighbours, in ``cell_ids`` or not, has a weight below 10⁻¹²."""
    neighbours, weights = spec.spec_interp_weights(
        nside, *spec.spec_stencil_points(nside, cell_ids)
    )
    position = np.clip(np.searchsorted(cell_ids, neighbours), 0, cell_ids.size - 1)
    inside = cell_ids[position] == neighbours
    small = weights < spec.SUPPORT_THRESHOLD
    no_neighbour = ~(inside & ~small).any(axis=0)  # [g, k, p]
    ill: NDArray[np.bool_] = (no_neighbour & small.any(axis=0)).any(axis=-1)
    return ill


def _domain(name: str) -> None:
    fixture = _operators()
    nside = int(name.rsplit("_", 1)[1])
    cell_ids = fixture[f"{name}_cell_ids"].astype(np.int64)
    maps = fixture[f"{name}_maps"]
    tables = spec.spec_conv_tables(nside, cell_ids)
    actual = spec.spec_conv_hpx(maps, tables, spec.spec_wavelet_hpx())
    kept = ~_ill_conditioned(nside, cell_ids)
    assert int((~kept).sum()) == ILL_CONDITIONED[name]
    assert_parity(actual[:, kept], fixture[f"{name}_conv"][:, kept])


def _decimation(name: str) -> None:
    fixture = _operators()
    nside = int(name.rsplit("_", 1)[1])
    cell_ids = fixture[f"{name}_cell_ids"].astype(np.int64)
    matrix, coarse = spec.spec_decimation_hpx(nside, cell_ids)
    assert np.array_equal(coarse, fixture[f"{name}_down_cell_ids"].astype(np.int64))
    down, _ = spec.spec_downsample_hpx(fixture[f"{name}_maps"], matrix)
    assert_parity(down, fixture[f"{name}_down"])


@pytest.mark.parametrize(
    "name",
    [
        "full_8",
        "disc_8",
        pytest.param("full_16", marks=pytest.mark.slow),
        pytest.param("disc_16", marks=pytest.mark.slow),
    ],
)
def test_convolution_equals_the_fixture(name: str) -> None:
    """M-CONV-HPX: the convolution of three maps, full sky and on a disc crossing the equator
    (partial-sky rule, OP-50), on every (gauge, pixel) row that is not ill-conditioned (OP-110);
    the full sky has none.

    Slow: at ``nside`` 16, the tables of 3072 pixels, 4 gauges and 25 stencil points.
    """
    _domain(name)


@pytest.mark.parametrize(
    "name",
    [
        "full_8",
        "disc_8",
        pytest.param("full_16", marks=pytest.mark.slow),
        pytest.param("disc_16", marks=pytest.mark.slow),
    ],
)
def test_decimation_equals_the_fixture(name: str) -> None:
    """M-DOWN-HPX: the decimation of three maps, full sky and on a disc (the coarse pixels are the
    sorted parents, the weights renormalised over the available fine pixels).

    Slow: at ``nside`` 16, the oracle compares every pair of fine and coarse pixels.
    """
    _decimation(name)


# ---------------------------------------------------------------------- masks and statistics


@cache
def _statistics() -> dict[str, NDArray[Any]]:
    return load_fixture("statistics_hpx")


def _configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = read_manifest()["statistics_hpx.npz"]["parameters"][
        "configs"
    ]
    return configs


def _cell_ids(name: str) -> tuple[int, NDArray[np.int64] | None, NDArray[np.int64]]:
    nside = _configs()[name]["nside"]
    cell_ids = _statistics()[f"{name}_cell_ids"].astype(np.int64)
    return nside, None if cell_ids.size == 12 * nside**2 else cell_ids, cell_ids


@pytest.mark.slow
@pytest.mark.parametrize("name", ["nan_16", "disc_nan_16"])
def test_masks_equal_the_fixture(name: str) -> None:
    """M-MASK-HPX: the static masks and reweighting maps of the two masked configurations, on the
    pixels of their level, and the pixels of each level.

    Slow: ``nside`` 16 (decision 4 on #165).
    """
    fixture, config = _statistics(), _configs()[name]
    nside, _, cell_ids = _cell_ids(name)
    levels = spec.spec_levels_hpx(nside, cell_ids, config["J"])
    masks = spec.spec_masks_hpx(levels, fixture[f"{name}_mask"].astype(bool))

    def same(mask: Any, key: str) -> None:
        assert np.array_equal(np.asarray(mask, dtype=np.float64), fixture[f"{name}_{key}"]), key

    for d in range(config["J"]):
        same(masks["field"][d], f"smooth_{d}")
        same(masks["layer1"][d], f"layer1_{d}")
        if d:
            assert np.array_equal(levels[d]["cell_ids"], fixture[f"{name}_cell_ids_{d}"])
            assert_parity(masks["reweight_field"][d], fixture[f"{name}_reweight_smooth_{d}"])
    for (d, j), mask in masks["down1"].items():
        same(mask, f"wav_{d}_{j}")
        assert_parity(masks["reweight_down1"][d, j], fixture[f"{name}_reweight_wav_{d}_{j}"])
    for (j2, j3), mask in masks["layer2"].items():
        same(mask, f"layer2_{j3}_{j2}")


@pytest.mark.slow
@pytest.mark.maths(
    "M-MEAN", "M-MOM", "M-W1", "M-S1", "M-S2", "M-S3", "M-S4", "M-CHAN", "M-ORDER", "M-NORM",
    "M-RED-ISO", "M-RED-ANG", "M-RED-SCALE", "M-FLAT",
)  # fmt: skip
@pytest.mark.parametrize(
    "name",
    [
        "full_8",
        "full_16_cross",
        "nan_16",
    ],
)
def test_statistics_equal_the_fixture(name: str) -> None:
    """Section 1 on this geometry: the statistics without normalisation (``vanilla``), with
    ``norm="self"`` (``store``) and those of a second field with ``norm="ref"`` (``load``), the
    reference, the reductions and the vectors of every full-sky configuration of
    ``statistics_hpx``, the NaN at the same positions; the partial skies ``disc_16`` and
    ``disc_nan_16`` are left to T5.6 (OP-110, see the module docstring). The masked configuration
    runs on the fields with their NaN and the mask given, as STL builds its operator.

    Slow: the oracle loops over the coefficients of J = 3 at ``nside`` 16 (decision 4 on #165).
    """
    fixture, config = _statistics(), _configs()[name]

    def expected(key: str) -> NDArray[Any]:
        return fixture[f"{name}_{key}"]

    nside, cell_ids, _ = _cell_ids(name)
    mask = None if config.get("mask_from_nan") else fixture.get(f"{name}_mask")
    options: dict[str, Any] = {
        "nside": nside,
        "cell_ids": cell_ids,
        "mask": None if mask is None else mask.astype(bool),
        "J": config["J"],
        "L": 4,
    }
    vanilla = spec.spec_apply_hpx(expected("field"), norm="none", **options)
    reference = planar.spec_reference(vanilla, config["norm_batch_mean"])
    stored = planar.spec_normalize(vanilla, reference)
    loaded = spec.spec_apply_hpx(expected("ref_field"), norm="ref", ref=reference, **options)
    assert_parity(reference["var"], expected("var_ref"))
    assert_parity(reference["sigma"], expected("s2_ref"))
    for kind, stats in (("vanilla", vanilla), ("store", stored), ("load", loaded)):
        raw_var = (
            vanilla["var"] if kind == "vanilla" else expected(f"{kind}_var") * expected("var_ref")
        )
        for family in planar.FAMILIES:
            scale = float(np.sqrt(np.max(raw_var))) if family == "mean" else None
            assert_parity(stats[family], expected(f"{kind}_{family}"), scale=scale)

    def reduce(iso: bool, angular: bool, scale: bool) -> dict[str, Any]:
        out = planar.spec_iso(stored) if iso else stored
        if angular:
            out = planar.spec_angular_fourier(out, config["harmonics_angle"], after_iso=iso)
        if scale:
            out = planar.spec_scale_cosine(out, config["harmonics_scale"], config["dj"])
        return out

    reductions = {
        "iso": (True, False, False),
        "angular": (False, True, False),
        "iso_angular": (True, True, False),
        "iso_scale": (True, False, True),
        "iso_angular_scale": (True, True, True),
    }
    for key, chain in reductions.items():
        reduced = reduce(*chain)
        for family in planar.SCATTERING if key == "iso" else ("S3", "S4"):
            assert_parity(reduced[family], expected(f"{key}_{family}"))
    assert_parity(planar.spec_flatten(stored, flatten_complex=True), expected("vector"))
    reduced = reduce(True, True, True)
    assert_parity(planar.spec_flatten(reduced, flatten_complex=True), expected("vector_reduced"))


# ---------------------------------------------------------------------- data and defaults


def test_defaults() -> None:
    """M-DEF-HPX: J = 1, 1, 2, 3, 4 for ``nside`` = 2, 4, 8, 16, 32, and the other defaults."""
    assert [spec.spec_defaults_hpx(n)["J"] for n in (2, 4, 8, 16, 32)] == [1, 1, 2, 3, 4]
    defaults = spec.spec_defaults_hpx(16)
    assert (defaults["L"], defaults["K"], defaults["gauge"], defaults["border"]) == (
        4,
        5,
        "cosmo",
        0,
    )
    assert defaults["nan_threshold"] == 0.33
    assert (defaults["decimation_radius"], defaults["decimation_width"]) == (3.0, 0.5)


def test_data_are_promoted_and_checked() -> None:
    """M-DATA-HPX: the promotion to ``(Nb, Nc, Npix)``, ``nside`` read from a full sky and
    required with ``cell_ids`` (OP-91), ``cell_ids`` strictly increasing (OP-90), and the mask
    read from the NaN of a single map only."""
    rng = np.random.default_rng(4)
    for shape in ((48,), (3, 48), (2, 3, 48)):
        x, mask, nside, cell_ids = spec.spec_data_hpx(rng.normal(size=shape))
        assert x.shape == (1,) * (3 - len(shape)) + shape
        assert mask is None
        assert nside == 2
        assert np.array_equal(cell_ids, np.arange(48))
    with pytest.raises(ValueError, match="full sky"):
        spec.spec_data_hpx(np.zeros(47))
    with pytest.raises(ValueError, match="needs nside"):
        spec.spec_data_hpx(np.zeros(3), cell_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="strictly increasing"):
        spec.spec_data_hpx(np.zeros(3), nside=4, cell_ids=[1, 3, 2])
    disc = np.arange(48)
    _, _, nside, _ = spec.spec_data_hpx(np.zeros(48), nside=4, cell_ids=disc)
    assert nside == 4
    holes = rng.normal(size=192)
    holes[[3, 70]] = np.nan
    x, mask, _, _ = spec.spec_data_hpx(holes)
    assert mask is not None
    assert np.flatnonzero(mask).tolist() == [3, 70]
    assert np.isnan(x[0, 0, [3, 70]]).all()
    with pytest.raises(ValueError, match="single map"):
        spec.spec_data_hpx(np.stack([holes, holes]))


def test_support_of_a_partial_sky_mask() -> None:
    """M-MASK-HPX: a pixel is invalid after a convolution if its stencil reads an invalid pixel
    with a weight above 10⁻¹² in some gauge; without a mask, nothing is."""
    tables = spec.spec_conv_tables(4)
    assert spec.spec_support_hpx(None, tables) is None
    invalid = np.zeros(192, dtype=bool)
    invalid[100] = True
    support = spec.spec_support_hpx(invalid, tables)
    assert support is not None
    assert support[100]
    index, weight = tables
    reads = ((index == 100) & (weight > spec.SUPPORT_THRESHOLD)).any(axis=(0, 2, 3))
    assert np.array_equal(support, reads)
