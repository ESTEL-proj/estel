"""``STOperatorTensorProd`` against the oracle of tensor products (task T3.2, issue #23; Part I,
§4.2; maths.md section 3).

The oracle ``spec_apply_tensor_prod`` (T3.3) transcribes maths.md §3 apart from STL's loop: it
brings the field to the level of a scale in every block first, then applies the block convolutions,
and sums each mean over its own window. ESTEL interleaves the downsamplings and the block
convolutions as STL does and sums the second layer as Gram products: the difference is rounding,
compared with the float64 tolerances of Part II, §14.3.

The configurations are not those of the parity fixtures: three small fields that cover what the
fixtures leave to their full tier or do not hold, each in under a second: a ``[2D, 1D]`` field
open in its 2D block, two channels and two maps with ``norm="self"``; a ``[1D, 2D, 1D]`` field,
three blocks with J = 1 in each, standardised, with a ``cross_matrix``; and ``norm="ref"`` with
``ref_batch_mean``, by the reference of another field.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataTensorProd, STOperatorTensorProd, STStatistics, _config
from estel._spec import tensor_prod as spec
from estel.backend import _registry
from tests.parity._compare import assert_parity

pytestmark = pytest.mark.maths(
    "M-W1-TENSORPROD",
    "M-LP",
    "M-IDX-TENSORPROD",
    "M-S-TENSORPROD",
    "M-BORDER-TENSORPROD",
    "M-NORM-TENSORPROD",
    "M-MOM",
    "M-STD",
)
GEOMETRIES = {2: Data2D, 1: Data1D}
GRIDS = ("S1", "S2", "S3", "S4")


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _field(shape: tuple[int, ...], seed: int) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.3
    return values


def _compare(stats: STStatistics, expected: dict[str, Any]) -> None:
    scale = float(np.sqrt(np.max(np.abs(expected["var"]))))
    assert_parity(_numpy(stats.full_grid("mean")), expected["mean"], scale=scale)
    assert_parity(_numpy(stats.full_grid("var")), expected["var"])
    for family in GRIDS:
        assert_parity(_numpy(stats.full_grid(family)), expected[family])


def _run(
    field: NDArray[np.float64],
    blocks: tuple[int, ...],
    pbc: list[bool],
    J: list[int],  # noqa: N803 (maths.md notation)
    **options: Any,
) -> tuple[STOperatorTensorProd, DataTensorProd]:
    data = DataTensorProd(field, [GEOMETRIES[n] for n in blocks], pbc=pbc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13
        return STOperatorTensorProd(data, J, **options), data


def test_open_block_two_channels_norm_self() -> None:
    """[2D, 1D], 8 x 8 x 8, the 2D block open, J = (2, 2), two maps of two channels, every
    block with its low-pass filter, ``norm="self"``: the statistics and their reference."""
    blocks, pbc, J = (2, 1), [False, True], [2, 2]  # noqa: N806
    field = _field((2, 2, 8, 8, 8), 0)
    operator, data = _run(field, blocks, pbc, J)
    stats = operator.apply(data, norm="self")
    expected = spec.spec_apply_tensor_prod(field, blocks, pbc, J, norm="self")
    _compare(stats, expected)
    assert stats._reference is not None
    assert_parity(_numpy(stats._reference.var), expected["reference"]["var"])
    assert_parity(_numpy(stats._reference.s2_sqrt), expected["reference"]["sigma"])


def test_three_blocks_standardized_cross_matrix() -> None:
    """[1D, 2D, 1D], 8 x 8 x 8 x 4, J = 1 in every block, open in the first and last blocks,
    standardised, with a ``cross_matrix`` where channel 1 has cross statistics only (OP-30),
    ``norm="none"``: every family and the moments removed."""
    blocks, pbc, J = (1, 2, 1), [False, True, False], [1, 1, 1]  # noqa: N806
    matrix = np.array([[True, True], [False, False]])
    field = _field((1, 2, 8, 8, 8, 4), 1)
    operator, data = _run(field, blocks, pbc, J, standardize=True, cross_matrix=matrix)
    stats = operator.apply(data, norm="none")
    expected = spec.spec_apply_tensor_prod(
        field, blocks, pbc, J, norm="none", standardize=True, cross_matrix=matrix
    )
    _compare(stats, expected)
    assert stats._standardization is not None
    for key, removed in zip(("mean_pre_std", "std_pre_std"), stats._standardization, strict=True):
        assert_parity(_numpy(removed), expected[key])


def test_norm_ref_with_batch_mean() -> None:
    """[2D, 1D], 8 x 8 x 4, open in both blocks, J = (2, 1), the reference of another field
    averaged over its two maps (``ref_batch_mean``), then ``norm="ref"`` on a field of two
    maps."""
    blocks, pbc, J = (2, 1), [False, False], [2, 1]  # noqa: N806
    target, other = _field((2, 1, 8, 8, 4), 2), _field((2, 1, 8, 8, 4), 3)
    operator, data = _run(target, blocks, pbc, J)
    operator.apply(data, norm="self", ref_batch_mean=True)
    stats = operator.apply(data.with_array(_backend_array(other)), norm="ref")
    stored = spec.spec_apply_tensor_prod(target, blocks, pbc, J, norm="self", ref_batch_mean=True)
    expected = spec.spec_apply_tensor_prod(
        other, blocks, pbc, J, norm="ref", ref=stored["reference"]
    )
    _compare(stats, expected)


def _backend_array(values: NDArray[np.float64]) -> Any:
    return _registry.current_backend().asarray(values, "float64", None)
