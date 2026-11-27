"""The finite positions of STL's tensor-product statistics (task T3.4, issue #127; maths.md
M-IDX-TENSORPROD, M-LP, M-CHAN, §3.9).

The fixtures ``scattering_tp`` and ``statistics_tp`` (STL `2D+1D`) store S1 to S4, raw or
normalised, and the reference ``S2_ref_sqrt_chan_diag`` as packed grids: their finite entries
and the mask of their finite positions (``full_grid``). For each of them, these tests rebuild
the mask from the rules of maths.md alone, with the parameters the manifest records:

* scales (M-IDX-TENSORPROD): S1 and S2 at every scale of the index set, the low-pass scale J_a of
  block a included when the block has one (M-LP: ``lowpass="ST"`` and every size of the block
  divisible by 2^J_a); S3 at j₂ ≤ j₃ and S4 at j₁ ≤ j₂ ≤ j₃, componentwise, over the wavelet scales;
* orientations (M-LP): a block taken at its low-pass scale has orientation 0 only;
* channel pairs (M-CHAN): S1 and S2 at c₁ ≤ c₂ with ``cross_matrix[c₁, c₂]``, S3 and S4 at both
  orders of such a pair, the reference at the channels with auto statistics;

and compare it with the fixture's, entry by entry, then check the counts of §3.9 per map and
channel pair. They do not compare ESTEL with STL: T3.2 does, with ``full_grid``.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.parity._compare import full_grid, load_fixture, read_manifest

pytestmark = pytest.mark.maths("M-IDX-TENSORPROD", "M-LP", "M-CHAN")
GRIDS = ("S1", "S2", "S3", "S4")
# M-DATA-TENSORPROD: the default L of a block, and its number of spatial axes.
ORIENTATIONS = {"2D": 4, "1D": 2}


@cache
def _arrays(case: str) -> dict[str, NDArray[Any]]:
    return load_fixture(case)


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _geometry(case: dict[str, Any], lowpass: str | None) -> dict[str, Any]:
    """J, L and the low-pass flags λ of each block (M-DATA-TENSORPROD, M-DEF-TENSORPROD, M-LP)."""
    sizes = case["sizes"]
    j = case["J"] or [int(math.log2(min(block))) - 2 for block in sizes]
    lowpass_blocks = [
        lowpass == "ST" and all(n % 2**j_a == 0 for n in block)
        for j_a, block in zip(j, sizes, strict=True)
    ]
    return {
        "J": list(j),
        "L": [ORIENTATIONS[name] for name in case["blocks"]],
        "lowpass": lowpass_blocks,
    }


def _grids() -> list[tuple[str, str, dict[str, Any], int, list[list[bool]]]]:
    """(fixture, key, geometry, Nb, cross matrix) for each packed grid of the two fixtures."""
    out = []
    scattering = _parameters("scattering_tp")
    for name, config in scattering["configs"].items():
        case = scattering["cases"][config["case"]]
        nc = config["Nc"]
        matrix = config.get("cross_matrix") or np.triu(np.ones((nc, nc), dtype=bool)).tolist()
        for lowpass in config["lowpass"]:
            mode = "st" if lowpass == "ST" else "none"
            for family in GRIDS if lowpass == "ST" else ("S1", "S2"):
                key = f"{name}_{mode}_{family}"
                out.append(("scattering_tp", key, _geometry(case, lowpass), config["Nb"], matrix))
    statistics = _parameters("statistics_tp")
    geometry = _geometry(statistics, statistics["lowpass"])
    nc = statistics["Nc"]
    matrix = np.triu(np.ones((nc, nc), dtype=bool)).tolist()
    for run in ("store", "load", "store_batch_mean"):
        families = GRIDS + (("s2_ref",) if run.startswith("store") else ())
        for family in families:
            # M-NORM-TENSORPROD: with the batch mean, the reference has one map.
            nb = 1 if (run, family) == ("store_batch_mean", "s2_ref") else statistics["Nb"]
            out.append(("statistics_tp", f"{run}_{family}", geometry, nb, matrix))
    return out


GRID_CASES = _grids()


def _scale_mask(geometry: dict[str, Any], family: str) -> NDArray[np.bool_]:
    """The finite positions over the scale and orientation axes of ``family`` (M-IDX-TENSORPROD,
    M-LP)."""
    j, n_orient, lowpass = geometry["J"], geometry["L"], geometry["lowpass"]
    blocks = len(j)
    if family in ("S1", "S2", "s2_ref"):
        scales = [j_a + int(lp) for j_a, lp in zip(j, lowpass, strict=True)]
        index = np.indices(scales + n_orient)
        finite = np.ones(index.shape[1:], dtype=bool)
        for a in range(blocks):
            scale, orientation = index[a], index[blocks + a]
            finite &= (scale < j[a]) | (orientation == 0)
        return finite
    groups = 2 if family == "S3" else 3
    index = np.indices(j * groups)
    ordered = np.ones(index.shape[1:], dtype=bool)
    for g in range(groups - 1):
        for a in range(blocks):
            ordered &= index[g * blocks + a] <= index[(g + 1) * blocks + a]
    orientations = (1,) * (groups * blocks)
    return np.broadcast_to(
        ordered.reshape(ordered.shape + orientations), ordered.shape + tuple(n_orient) * groups
    )


def _expected(geometry: dict[str, Any], family: str, nb: int, matrix: Any) -> NDArray[np.bool_]:
    """The finite positions of ``family`` over its whole grid, channel axes included (M-CHAN)."""
    cross = np.array(matrix, dtype=bool)
    if family == "s2_ref":
        channels = np.broadcast_to(np.diag(cross), (nb, len(cross)))
    elif family in ("S1", "S2"):
        channels = np.broadcast_to(np.triu(cross), (nb, *cross.shape))
    else:
        channels = np.broadcast_to(np.triu(cross) | np.triu(cross).T, (nb, *cross.shape))
    scales = _scale_mask(geometry, family)
    return channels.reshape(channels.shape + (1,) * scales.ndim) & scales


def _count(geometry: dict[str, Any], family: str) -> int:
    """§3.9: the coefficients STL computes per map and channel pair."""
    j: list[int] = geometry["J"]
    n_orient: list[int] = geometry["L"]
    lowpass: list[bool] = geometry["lowpass"]
    product = math.prod(n_orient)
    if family in ("S1", "S2", "s2_ref"):
        blocks = zip(j, n_orient, lowpass, strict=True)
        return math.prod(j_a * l_a + int(lp) for j_a, l_a, lp in blocks)
    if family == "S3":
        return product**2 * math.prod(j_a * (j_a + 1) // 2 for j_a in j)
    return product**3 * math.prod(j_a * (j_a + 1) * (j_a + 2) // 6 for j_a in j)


@pytest.mark.parametrize(
    ("fixture", "key", "geometry", "nb", "matrix"),
    GRID_CASES,
    ids=[f"{fixture}-{key}" for fixture, key, *_ in GRID_CASES],
)
def test_finite_positions_follow_the_index_sets(
    fixture: str, key: str, geometry: dict[str, Any], nb: int, matrix: Any
) -> None:
    family = key.rsplit("_", 1)[-1] if not key.endswith("s2_ref") else "s2_ref"
    grid = full_grid(_arrays(fixture), key)
    expected = _expected(geometry, family, nb, matrix)
    assert grid.shape == expected.shape, (
        f"{fixture}:{key} has shape {grid.shape}, not the full grid {expected.shape} of maths.md "
        "M-S-TENSORPROD for this geometry."
    )
    finite = np.isfinite(grid)
    if not np.array_equal(finite, expected):
        positions = np.argwhere(finite != expected)[:5]
        raise AssertionError(
            f"{fixture}:{key}: STL's finite positions differ from maths.md M-IDX-TENSORPROD and "
            f"M-LP at {int((finite != expected).sum())} entries, first {positions.tolist()}. "
            "Report it in an issue: do not regenerate the fixture."
        )
    pairs = _pairs(expected, family)
    assert int(finite.sum()) == pairs * _count(geometry, family), (
        f"{fixture}:{key}: {int(finite.sum())} finite entries for {pairs} maps and channel "
        f"pairs, not {_count(geometry, family)} per map and pair (maths.md §3.9)."
    )


def _pairs(expected: NDArray[np.bool_], family: str) -> int:
    """The number of (map, channel pair) — or (map, channel) for the reference — with finite
    entries."""
    leading = 2 if family == "s2_ref" else 3
    return int(expected.reshape((*expected.shape[:leading], -1)).any(axis=-1).sum())
