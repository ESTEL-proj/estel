"""Parity of the HEALPix geometry tables with STL, as operators (task T5.1, issue #26; maths.md
M-CONV-HPX, M-WAV-HPX, M-DOWN-HPX, OP-92).

The fixture ``operators_hpx`` (T5.3) holds STL's convolution and decimation through STL's own
entry points, on the sphere, without geometry cache and with the kernel in float64 (OP-45,
OP-46): the dense matrices at ``nside`` 4, and the responses to 3 random maps at ``nside`` 8 and
16, on the full sky and on a disc that crosses the equator. The tables are applied here as T5.5
will apply them (``gather_weighted``, then ``segment_sum`` for the decimation; the layout is in
the docstring of ``_tables_healpix``), written in NumPy, and compared with the tolerances of
Part II, §14.3. The tables equal healpix-analyse's as operators, not entry by entry (OP-92).

On the discs, STL's partial-sky rule (M-CONV-HPX, OP-50) is set by interpolation weights at the
level of rounding at a few stencil points of the edge; ESTEL applies the well-posed rule decided
by the maintainer (a neighbour of weight below 1e-12 counts as absent; maths.md
OP-110), which differs from STL there. The discs are compared with the fixtures
on every row that is not ill-conditioned, with the definition of that decision (point 2 of
https://github.com/ESTEL-proj/estel/issues/26#issuecomment-7134295984).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel.wavelets._banks_healpix import kernel_weights_healpix
from estel.wavelets._tables_healpix import (
    DecimationHealpix,
    StencilsHealpix,
    _stencil_interpolation,
    decimation_healpix,
    stencils_healpix,
)
from tests.parity._compare import assert_parity, load_fixture, read_manifest

CASE = "operators_hpx"
DOMAINS = ["full_8", "disc_8", "full_16", "disc_16"]
_ILL_CONDITIONED = 1e-12
"""A weight below this is at the level of rounding (decision on issue #26, point 1)."""
EXCLUDED_ROWS = {"disc_8": 16, "disc_16": 14}
"""The ill-conditioned (gauge, pixel) rows of each disc, of 200 and 836."""


@functools.cache
def _fixture() -> dict[str, NDArray[Any]]:
    return load_fixture(CASE)


def _grid(name: str) -> tuple[NDArray[np.int64] | None, int, NDArray[np.int64]]:
    """``cell_ids`` (``None`` on the full sky, as T5.5 passes it), the depth, and the pixels."""
    ids = _fixture()[f"{name}_cell_ids"].astype(np.int64)
    depth = int(name.split("_")[1]).bit_length() - 1
    return (None if name.startswith("full") else ids), depth, ids


@functools.cache
def _tables(name: str) -> tuple[StencilsHealpix, DecimationHealpix]:
    cell_ids, depth, _ = _grid(name)
    parameters = read_manifest()[f"{CASE}.npz"]["parameters"]
    stencils = stencils_healpix(cell_ids, depth, L=parameters["L"], kernel_size=parameters["K"])
    return stencils, decimation_healpix(cell_ids, depth)


def _convolve(tables: StencilsHealpix, maps: NDArray[np.float64]) -> NDArray[np.complex128]:
    """``gather_weighted`` along the pixels, then the kernel: ``(B, Npix)`` -> ``(B, L, Npix)``."""
    gathered = (maps[:, tables.index] * tables.weight).sum(axis=-1)  # (B, L · Npix, K²)
    kernel = kernel_weights_healpix(kernel_size=tables.kernel_size)
    out: NDArray[np.complex128] = gathered @ kernel.real + 1j * (gathered @ kernel.imag)
    return out.reshape(maps.shape[0], tables.n_gauges, -1)


def _decimate(tables: DecimationHealpix, maps: NDArray[np.float64]) -> NDArray[np.float64]:
    """``gather_weighted`` then ``segment_sum``: ``(B, Npix)`` -> ``(B, Npix_coarse)``."""
    gathered = maps[:, tables.index[:, 0, 0]] * tables.weight[:, 0, 0]
    rows = np.arange(maps.shape[0])[:, None]
    out = np.zeros((maps.shape[0], tables.num_segments))
    np.add.at(out, (rows, tables.segment_ids[None]), gathered)
    return out


def _well_conditioned_rows(name: str) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """The (gauge, pixel) rows whose binding does not depend on weights at the level of rounding,
    ``(L, Npix)``, and the stencil points with no neighbour in ``cell_ids``, ``(L, Npix, K²)``.

    A row is ill-conditioned when one of its stencil points has no neighbour in ``cell_ids``
    with a weight ≥ 1e-12, while one of its 4 neighbours, in ``cell_ids`` or not, has a weight
    < 1e-12 (decision on issue #26, point 2): STL's value there is set by rounding,
    whichever slot the rounding gives the near-zero weight (OP-92).
    """
    _, depth, ids = _grid(name)
    parameters = read_manifest()[f"{CASE}.npz"]["parameters"]
    index, weight = _stencil_interpolation(ids, depth, parameters["L"], parameters["K"])
    present = np.isin(index, ids)
    significant = (present & (weight >= _ILL_CONDITIONED)).any(-1)
    ill = ~significant & (weight < _ILL_CONDITIONED).any(-1)
    rows: NDArray[np.bool_] = ~np.asarray(ill.any(-1))
    empty: NDArray[np.bool_] = ~np.asarray(present.any(-1))
    return rows, empty


@pytest.mark.maths("M-WAV-HPX")
def test_kernel_equals_stl() -> None:
    expected = _fixture()["kernel"]  # (1, 1, K²)
    assert_parity(
        kernel_weights_healpix(kernel_size=read_manifest()[f"{CASE}.npz"]["parameters"]["K"])[
            None, None
        ],
        expected,
    )


@pytest.mark.maths("M-CONV-HPX", "M-WAV-HPX")
def test_dense_convolution_equals_stl() -> None:
    """``conv_4[g, k, n]``: the response at pixel k and gauge g to the unit map at pixel n."""
    stencils = stencils_healpix(None, 2)
    n_pix = 192
    kernel = kernel_weights_healpix(kernel_size=stencils.kernel_size)
    values = kernel[None, :, None] * stencils.weight  # (L · Npix, K², 4)
    rows = np.broadcast_to(np.arange(stencils.index.shape[0])[:, None, None], stencils.index.shape)
    dense = np.zeros((stencils.index.shape[0], n_pix), dtype=np.complex128)
    np.add.at(dense, (rows, stencils.index), values)
    assert_parity(dense.reshape(stencils.n_gauges, n_pix, n_pix), _fixture()["conv_4"])


@pytest.mark.maths("M-DOWN-HPX")
def test_dense_decimation_equals_stl() -> None:
    tables = decimation_healpix(None, 2)
    dense = np.zeros((tables.num_segments, 192))
    np.add.at(dense, (tables.segment_ids, tables.index[:, 0, 0]), tables.weight[:, 0, 0])
    np.testing.assert_array_equal(tables.cell_ids_out, _fixture()["down_4_cell_ids"])
    assert_parity(dense, _fixture()["down_4"])


@pytest.mark.maths("M-DOWN-HPX")
@pytest.mark.parametrize("name", DOMAINS)
def test_decimation_responses_equal_stl(name: str) -> None:
    _, decimation = _tables(name)
    np.testing.assert_array_equal(decimation.cell_ids_out, _fixture()[f"{name}_down_cell_ids"])
    assert_parity(_decimate(decimation, _fixture()[f"{name}_maps"]), _fixture()[f"{name}_down"])


@pytest.mark.maths("M-CONV-HPX")
@pytest.mark.parametrize("name", ["full_8", "full_16"])
def test_convolution_responses_equal_stl_on_the_full_sky(name: str) -> None:
    stencils, _ = _tables(name)
    assert_parity(_convolve(stencils, _fixture()[f"{name}_maps"]), _fixture()[f"{name}_conv"])


@pytest.mark.maths("M-CONV-HPX")
@pytest.mark.parametrize("name", ["disc_8", "disc_16"])
def test_convolution_responses_equal_stl_on_a_disc(name: str) -> None:
    """The partial-sky rule (OP-50): renormalised weights at the truncated pixels, and the
    stencil points without a neighbour, which read the pixel itself. Compared on every row that
    is not ill-conditioned (module docstring; decision on issue #26)."""
    stencils, _ = _tables(name)
    rows, empty = _well_conditioned_rows(name)
    assert empty.sum() > 500  # the rule of the empty point is exercised
    assert rows.size - rows.sum() == EXCLUDED_ROWS[name]
    expected = _fixture()[f"{name}_conv"]
    actual = _convolve(stencils, _fixture()[f"{name}_maps"])
    assert_parity(actual[:, rows], expected[:, rows], scale=float(np.abs(expected).max()))
