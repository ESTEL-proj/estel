"""Unit tests of the HEALPix geometry tables (task T5.1, issue #26; maths.md M-CONV-HPX,
M-WAV-HPX, M-DOWN-HPX).

Each geometry function of ``estel.wavelets._tables_healpix`` against healpy (``pix2ang``,
``ring2nest``, ``get_interp_weights``), which is imported here only: it is GPL-2 and never
imported by ESTEL (Part I, §1.5). The rotations against the formula of M-CONV-HPX, step 2; the
kernel against M-WAV-HPX; the search of the decimation's candidates against a brute-force search;
the layout of the tables and the validation of the arguments. Equality with STL's tables, as
operators (OP-92), is in ``tests/parity/test_healpix_tables.py``.
"""

from __future__ import annotations

import math
from typing import Any

import healpy as hp  # type: ignore[import-untyped]
import numpy as np
import pytest

from estel import DataHealpix
from estel.wavelets import _operator_healpix, _tables_healpix
from estel.wavelets._banks_healpix import kernel_weights_healpix
from estel.wavelets._operator_healpix import build_wavelet_plan_healpix
from estel.wavelets._tables_healpix import (
    MAX_DEPTH,
    _bind_support,
    _cone_pairs,
    _haversine,
    _stencil_interpolation,
    centres_lonlat,
    decimation_healpix,
    interp_weights_nested,
    local_kernel_grid,
    nest2xyf,
    ring2nest,
    rotation_matrices,
    stencils_healpix,
    xyf2ang,
    xyf2nest,
)


def _pixels(depth: int) -> np.ndarray:
    """Every pixel up to depth 4, else 4096 seeded pixels and the first and last ones."""
    n_pix = 12 * 4**depth
    if depth <= 4:
        return np.arange(n_pix, dtype=np.int64)
    rng = np.random.default_rng(depth)
    return np.concatenate([[0, 1, n_pix // 2, n_pix - 1], rng.integers(0, n_pix, 4096)])


def _disc(depth: int, colatitude: float, longitude: float, radius: float) -> np.ndarray:
    """The NESTED pixels whose centre is within ``radius`` (radians) of a point."""
    nside = 1 << depth
    theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True)
    centre = np.array([math.pi / 2 - colatitude, longitude])
    distance = _haversine(centre[:1], centre[1:], math.pi / 2 - theta, phi)
    return np.flatnonzero(distance <= radius).astype(np.int64)


# ------------------------------------------------------------------------- pixel centres, RING


@pytest.mark.maths("M-DATA-HPX")
@pytest.mark.parametrize("depth", [0, 1, 4, MAX_DEPTH])
def test_pixel_centres_equal_healpy(depth: int) -> None:
    pix = _pixels(depth)
    theta, phi = xyf2ang(*nest2xyf(pix, depth), depth)
    expected_theta, expected_phi = hp.pix2ang(1 << depth, pix, nest=True)
    np.testing.assert_allclose(theta, expected_theta, rtol=0, atol=1e-14)
    np.testing.assert_allclose(phi, expected_phi, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(xyf2nest(*nest2xyf(pix, depth), depth), pix)


@pytest.mark.maths("M-DATA-HPX")
@pytest.mark.parametrize("depth", [0, 1, 4, MAX_DEPTH])
def test_ring2nest_equals_healpy(depth: int) -> None:
    pix = _pixels(depth)
    np.testing.assert_array_equal(ring2nest(pix, depth), hp.ring2nest(1 << depth, pix))


# -------------------------------------------------------------------------------- interpolation


@pytest.mark.maths("M-CONV-HPX")
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_interpolation_equals_healpy_as_operator(depth: int) -> None:
    """The 4 neighbours and weights, compared as a linear functional of the map (OP-92): a zero
    weight may fill another slot. Directions: seeded on the sphere, the poles, and pixel
    centres (a rotated point can fall on one)."""
    nside = 1 << depth
    rng = np.random.default_rng(3)
    z, phi = rng.uniform(-1, 1, 1000), rng.uniform(0, 2 * np.pi, 1000)
    theta = np.arccos(z)
    centres = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True)
    theta = np.concatenate([theta, [0.0, np.pi], centres[0]])
    phi = np.concatenate([phi, [0.3, 1.2], centres[1]])
    vecs = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    index, weight = interp_weights_nested(vecs, depth)
    assert index.shape == weight.shape == (theta.size, 4)
    np.testing.assert_allclose(weight.sum(axis=1), 1.0, rtol=0, atol=1e-14)

    expected_index, expected_weight = hp.get_interp_weights(nside, theta, phi, nest=True)
    rows = np.arange(theta.size)[:, None]
    actual = np.zeros((theta.size, 12 * nside**2))
    np.add.at(actual, (rows, index), weight)
    expected = np.zeros_like(actual)
    np.add.at(expected, (rows, expected_index.T), expected_weight.T)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------- rotations, stencil


@pytest.mark.maths("M-CONV-HPX")
def test_rotations_equal_the_formula_of_the_gauge_cosmo() -> None:
    """Step 2 of M-CONV-HPX: R carries the pole to the pixel, and the u and v axes of the stencil
    onto cos alpha_g e_θ + sin alpha_g e_φ and -sin alpha_g e_θ + cos alpha_g e_φ, with
    alpha_g = -φ + π g / L in the north (the equator included) and φ - π g / L in the south."""
    n_gauges = 4
    theta, phi = hp.pix2ang(8, np.arange(768), nest=True)
    assert np.any(theta == np.pi / 2)  # the equatorial ring follows the northern rule
    rotations = rotation_matrices(theta, phi, n_gauges)
    assert rotations.shape == (768, n_gauges, 3, 3)

    e_theta = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], -1
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], -1)
    e_r = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    south = theta > np.pi / 2
    gauges = np.arange(n_gauges) * np.pi / n_gauges
    alpha = np.where(south[:, None], phi[:, None] - gauges, -phi[:, None] + gauges)[..., None]
    u_axis = np.cos(alpha) * e_theta[:, None] + np.sin(alpha) * e_phi[:, None]
    v_axis = -np.sin(alpha) * e_theta[:, None] + np.cos(alpha) * e_phi[:, None]
    np.testing.assert_allclose(rotations[..., 0], u_axis, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rotations[..., 1], v_axis, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        rotations[..., 2], np.broadcast_to(e_r[:, None], u_axis.shape), atol=1e-14
    )


@pytest.mark.maths("M-CONV-HPX")
def test_stencil_points_and_their_order() -> None:
    """Point (u, v), entry p = (v + 2) K + (u + 2), at colatitude √(u² + v²) alpha and azimuth
    atan2(v, u)."""
    nside, size = 16, 5
    grid = local_kernel_grid(size, 4)
    alpha = math.sqrt(4 * math.pi / (12 * nside**2))
    v, u = np.divmod(np.arange(size * size), size)
    u, v = u - 2, v - 2
    np.testing.assert_allclose(
        np.arccos(np.clip(grid[:, 2], -1, 1)), np.hypot(u, v) * alpha, atol=1e-7
    )
    off_centre = (u != 0) | (v != 0)
    np.testing.assert_allclose(
        np.arctan2(grid[off_centre, 1], grid[off_centre, 0]),
        np.arctan2(v, u)[off_centre],
        atol=1e-14,
    )


# ------------------------------------------------------------------------------------- kernel


@pytest.mark.maths("M-WAV-HPX")
@pytest.mark.parametrize("size", [3, 5, 7])
def test_kernel_equals_the_formula(size: int) -> None:
    kernel = kernel_weights_healpix(kernel_size=size)
    assert kernel.shape == (size * size,)
    assert kernel.dtype == np.complex128
    c = (size - 1) // 2
    v, u = np.divmod(np.arange(size * size), size)
    u, v = u - c, v - c
    psi = np.exp(-(u**2 + v**2) / 2) * np.exp(1j * np.pi * u / 2)
    psi = psi - psi.mean()
    np.testing.assert_allclose(kernel, psi / np.abs(psi).sum(), rtol=0, atol=1e-17)
    assert abs(kernel.sum()) < 1e-15
    assert np.abs(kernel).sum() == pytest.approx(1.0, abs=1e-15)
    grid = kernel.reshape(size, size)
    np.testing.assert_allclose(grid[:, ::-1], np.conj(grid), atol=1e-17)  # ψ[-u, v] = conj ψ[u, v]
    np.testing.assert_allclose(grid[::-1, :], grid, atol=1e-17)  # ψ[u, -v] = ψ[u, v]


@pytest.mark.parametrize(
    ("size", "error"), [(4, ValueError), (1, ValueError), (True, TypeError), (5.0, TypeError)]
)
def test_kernel_size_is_checked(size: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="kernel_size"):
        kernel_weights_healpix(kernel_size=size)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- decimation


@pytest.mark.maths("M-DOWN-HPX")
@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("sky", ["full", "disc"])
@pytest.mark.parametrize("chunk", [7, _tables_healpix._CONE_CHUNK])
def test_candidate_search_equals_brute_force(
    depth: int, sky: str, chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every fine pixel within rho of a coarse centre, and no other (decision 2 on #26), in one
    chunk of coarse pixels or in many (7 coarse pixels a chunk)."""
    monkeypatch.setattr(_tables_healpix, "_CONE_CHUNK", chunk)
    nside = 1 << depth
    fine = np.arange(12 * nside**2) if sky == "full" else _disc(depth, 1.4, 0.5, 0.6)
    coarse = np.unique(fine // 4)
    lon_c, lat_c = centres_lonlat(coarse, depth - 1)
    lon_f, lat_f = centres_lonlat(fine, depth)
    radius = math.radians(3.0 * math.degrees(math.sqrt(4.0 * math.pi / (12 * nside**2))))
    rows, cols, gamma = _cone_pairs(lon_c, lat_c, lon_f, lat_f, radius)
    all_gamma = _haversine(lat_c[:, None], lon_c[:, None], lat_f[None], lon_f[None])
    expected_rows, expected_cols = np.nonzero(all_gamma <= radius)
    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_array_equal(cols, expected_cols)
    np.testing.assert_array_equal(gamma, all_gamma[rows, cols])


@pytest.mark.maths("M-DOWN-HPX")
def test_decimation_of_the_full_sky() -> None:
    """26 to 32 fine pixels per coarse pixel, weights of sum 1, the 4 children carrying 0.28 to
    0.30 of the weight (M-DOWN-HPX)."""
    tables = decimation_healpix(None, 4)
    assert tables.index.shape == tables.weight.shape == (tables.segment_ids.size, 1, 1)
    assert tables.index.dtype == tables.segment_ids.dtype == np.int32
    assert tables.weight.dtype == np.float64
    np.testing.assert_array_equal(tables.cell_ids_out, np.arange(768))
    assert np.all(np.diff(tables.segment_ids) >= 0)
    counts = np.bincount(tables.segment_ids, minlength=tables.num_segments)
    assert counts.min() == 26
    assert counts.max() == 32
    sums = np.bincount(tables.segment_ids, weights=tables.weight[:, 0, 0])
    np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-15)
    is_child = tables.index[:, 0, 0] // 4 == tables.segment_ids
    children = np.bincount(tables.segment_ids, weights=tables.weight[:, 0, 0] * is_child)
    assert children.min() >= 0.28
    assert children.max() <= 0.30


# ----------------------------------------------------------------------- layout and validation


@pytest.mark.maths("M-CONV-HPX")
def test_stencil_layout_and_dtypes() -> None:
    depth, n_gauges, size = 2, 3, 5
    tables = stencils_healpix(None, depth, L=n_gauges, kernel_size=size)
    n_pix = 12 * 4**depth
    assert tables.index.shape == tables.weight.shape == (n_gauges * n_pix, size * size, 4)
    assert tables.index.dtype == np.int32
    assert tables.weight.dtype == np.float64
    assert (tables.n_gauges, tables.kernel_size) == (n_gauges, size)
    assert tables.index.min() >= 0
    assert tables.index.max() < n_pix
    np.testing.assert_allclose(tables.weight.sum(axis=-1), 1.0, rtol=0, atol=1e-14)
    # Row g · Npix + k, column the stencil centre: the pixel k itself, for every gauge.
    centre = tables.weight[:, 12] * (
        tables.index[:, 12] == np.tile(np.arange(n_pix), n_gauges)[:, None]
    )
    np.testing.assert_allclose(centre.sum(axis=-1), 1.0, atol=1e-12)


@pytest.mark.maths("M-CONV-HPX")
@pytest.mark.parametrize("sky", ["full", "disc"])
def test_stencils_built_by_batches_equal_one_batch(
    sky: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stencils built by batches of output pixels (task T5.7) equal, bit for bit, those
    built at once: 7 pixels a batch at ``nside`` 4, on the full sky and on a disc."""
    depth = 2
    ids = None if sky == "full" else _disc(depth, 1.4, 0.5, 0.9)
    whole = stencils_healpix(ids, depth)
    monkeypatch.setattr(_tables_healpix, "_STENCIL_BATCH", 7)
    batched = stencils_healpix(ids, depth)
    np.testing.assert_array_equal(batched.index, whole.index)
    np.testing.assert_array_equal(batched.weight, whole.weight)
    assert batched.index.dtype == np.int32
    assert batched.weight.dtype == np.float64


@pytest.mark.maths("M-MASK-HPX")
def test_masks_built_by_batches_equal_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The support test of the masks, by batches of pixels (task T5.7), gives the masks and
    the reweighting maps of one batch, bit for bit: 7 pixels a batch at ``nside`` 4."""
    mask = np.zeros(192, dtype=bool)
    mask[[5, 6, 70, 71, 140]] = True
    data = DataHealpix(np.ones(192), mask=mask)
    whole = build_wavelet_plan_healpix(data, 2, kernel_size=5).masks
    monkeypatch.setattr(_tables_healpix, "_STENCIL_BATCH", 7)
    batched = build_wavelet_plan_healpix(data, 2, kernel_size=5).masks
    assert whole is not None
    assert batched is not None
    for name in ("smooth_invalid", "smooth_reweight", "wav_invalid", "wav_reweight", "layer1",
                 "layer2"):  # fmt: skip
        expected, actual = getattr(whole, name), getattr(batched, name)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert any(m.any() for m in whole.layer2.values())


@pytest.mark.maths("M-CONV-HPX")
@pytest.mark.parametrize("sky", ["full", "disc"])
def test_kernel_folded_into_the_stencils(sky: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The kernel folded into the stencils (task T5.7): each row reads its distinct pixels
    once, with Σ W[p] w[p, m], as real pairs; the convolution equals the stencils then the
    kernel to rounding, and a fold by batches (7 rows) equals one batch, bit for bit."""
    depth = 2
    ids = None if sky == "full" else _disc(depth, 1.4, 0.5, 0.9)
    tables = stencils_healpix(ids, depth)
    kernel = kernel_weights_healpix(kernel_size=5)
    index, weight = _operator_healpix._fold_kernel(tables, kernel)
    rows, width = tables.index.shape[0], index.shape[-1]
    assert index.shape == weight.shape == (rows, 2, width)
    assert index.dtype == np.int32
    np.testing.assert_array_equal(index[:, 0], index[:, 1])
    assert width <= 45 < tables.index.shape[1] * tables.index.shape[2]
    x = np.random.default_rng(0).standard_normal(int(tables.index.max()) + 1)
    expected = (x[tables.index] * tables.weight).sum(axis=-1) @ kernel
    folded = (x[index] * weight).sum(axis=-1)
    np.testing.assert_allclose(folded[:, 0] + 1j * folded[:, 1], expected, rtol=0, atol=1e-14)
    monkeypatch.setattr(_operator_healpix, "_FOLD_BATCH", 7)
    batched = _operator_healpix._fold_kernel(tables, kernel)
    np.testing.assert_array_equal(batched[0], index)
    np.testing.assert_array_equal(batched[1], weight)


@pytest.mark.maths("M-CONV-HPX")
def test_partial_sky_rule() -> None:
    """The rule decided on issue #26: a neighbour of weight below 1e-12 counts as
    absent, the remaining weights sum to 1 exactly, and a stencil point without a remaining
    neighbour reads the stencil centre: the pixel itself, with weight 1."""
    depth = 4
    ids = _disc(depth, 1.4, 0.5, 0.3)
    index, weight = _stencil_interpolation(ids, depth, 4, 5)
    positions, bound, empty = _bind_support(index, weight, ids, 5)
    assert empty.any()
    np.testing.assert_allclose(bound.sum(axis=-1), 1.0, rtol=0, atol=1e-15)
    assert np.all((bound == 0) | (bound >= 1e-12))
    np.testing.assert_array_equal(positions[bound == 0], 0)
    absent = ~np.isin(index, ids) | (weight < 1e-12)
    np.testing.assert_array_equal(empty, absent.all(axis=-1))
    pixel = np.broadcast_to(np.arange(ids.size)[None, :, None], empty.shape)
    reads_itself = (bound * (positions == pixel[..., None])).sum(axis=-1)
    np.testing.assert_allclose(reads_itself[empty], 1.0, atol=1e-12)


def test_deepest_level_fits_int32() -> None:
    ids = np.array([0, 1, 2, 3, 12 * 4**MAX_DEPTH - 1], dtype=np.int64)
    stencils = stencils_healpix(ids, MAX_DEPTH, L=1, kernel_size=3)
    decimation = decimation_healpix(ids, MAX_DEPTH)
    assert stencils.index.dtype == decimation.index.dtype == np.int32
    np.testing.assert_array_equal(decimation.cell_ids_out, [0, (12 * 4**MAX_DEPTH - 1) // 4])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"cell_ids": [3, 1, 2], "depth": 1}, ValueError, "strictly increasing"),
        ({"cell_ids": [1, 1, 2], "depth": 1}, ValueError, "strictly increasing"),
        ({"cell_ids": [1.0, 2.0], "depth": 1}, TypeError, "integer"),
        ({"cell_ids": [0, 48], "depth": 1}, ValueError, "outside"),
        ({"cell_ids": [], "depth": 1}, ValueError, "non-empty"),
        ({"cell_ids": None, "depth": MAX_DEPTH + 1}, ValueError, "int32"),
        ({"cell_ids": None, "depth": 1.0}, TypeError, "depth"),
        ({"cell_ids": None, "depth": 1, "L": 0}, ValueError, "L=0"),
        ({"cell_ids": None, "depth": 1, "kernel_size": 4}, ValueError, "kernel_size"),
    ],
)
def test_stencil_arguments_are_checked(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        stencils_healpix(**kwargs)


def test_decimation_needs_a_finer_level() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        decimation_healpix(None, 0)
