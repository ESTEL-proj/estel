"""Invariance of the statistics of tensor products under a permutation of the blocks (Part I,
§5.1; task T3.2, issue #23; maths.md M-W1-TP, "Order of the blocks", M-S-TP).

A tensor product ``[2D, 1D]`` of a field and the tensor product ``[1D, 2D]`` of the same field
with its axes permuted are one transform: every statistic of the second is that of the first
with its scale and orientation axes permuted block by block, within each group (j, or j2 and
j3, or j1, j2 and j3; and the orientations alike). STL's order of the blocks only changes the
rounding (maths.md M-W1-TP, "Checked": 1.0e-15), so the two sides are compared in float64 with
the tolerances of Part II, §14.3. The fields are open in the 2D block and periodic in the 1D
block, so that the borders of the blocks are permuted too (M-BORDER-TP), with the low-pass
filter of every block and the centred scale (M-LP); and with ``norm="self"``, so that the
normalisation is permuted with them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataTensorProd, STOperatorTensorProd
from estel.backend import _registry
from tests.parity._compare import assert_parity

pytestmark = [
    pytest.mark.maths("M-W1-TP", "M-S-TP", "M-BORDER-TP", "M-LP"),
    pytest.mark.filterwarnings("ignore:.*too small for a border crop:UserWarning"),  # OP-13
]
# The number of index groups (of scales, then of orientations) of each family.
GROUPS = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _swap_blocks(grid: NDArray[Any], groups: int) -> NDArray[Any]:
    """A full grid of ``[2D, 1D]`` with the axes of its two blocks exchanged in every group of
    scales and of orientations: the full grid of ``[1D, 2D]``."""
    axes = [0, 1, 2]
    for group in range(2 * groups):
        first = 3 + 2 * group
        axes += [first + 1, first]
    return np.transpose(grid, axes)


@pytest.mark.parametrize("norm", ["none", "self"])
def test_permuting_the_blocks_permutes_the_statistics(norm: str) -> None:
    rng = np.random.default_rng(0)
    field = rng.standard_normal((2, 2, 8, 8, 8)) + 0.3
    space_time = DataTensorProd(field, [Data2D, Data1D], pbc=[False, True])
    time_space = DataTensorProd(np.moveaxis(field, -1, 2), [Data1D, Data2D], pbc=[True, False])
    first = STOperatorTensorProd(space_time, [2, 2], norm=norm).apply(space_time)
    second = STOperatorTensorProd(time_space, [2, 2], norm=norm).apply(time_space)
    for family in ("mean", "var"):
        assert_parity(_numpy(second.full_grid(family)), _numpy(first.full_grid(family)))
    for family, groups in GROUPS.items():
        expected = _swap_blocks(_numpy(first.full_grid(family)), groups)
        assert_parity(_numpy(second.full_grid(family)), expected)
