"""Spec parity of the complex form of the flattening, the vector of the synthesis loss: ESTEL
against the oracle ``spec_flatten`` (task T6.1, issue #28, decision 5 and note of 16 November
2026; maths.md M-FLAT, M-SYN-LOSS; OP-39).

``STStatistics._to_complex_vector`` is STL's ``to_flatten(flatten_complex=False)`` with its
options ``keep_batch_dim`` and ``mean_along_batch``, which ``to_vector`` does not have; it is
compared with ``spec_flatten`` as T2.6 compared ``to_vector`` and ``mask_st``
(``tests/spec/test_statistics.py``), on two batch elements and two channels, so that the
cross-channel S1 and S2, whose imaginary parts the real form drops, are in the vector, and so
that averaging over the batch differs from keeping it. The tolerances are those of Part II,
§14.3, in float64.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from estel import Data2D, STOperator, _config
from estel._spec import cartesian as spec
from estel._spec import common
from tests.parity._compare import assert_parity

pytestmark = pytest.mark.maths("M-FLAT", "M-SYN-LOSS")


@pytest.fixture(scope="module")
def statistics() -> tuple[Any, dict[str, Any]]:
    """ESTEL's and the oracle's statistics without normalisation of one small periodic case:
    16 x 16, two batch elements, two channels, J = 2, L = 4."""
    field = np.random.default_rng(7).standard_normal((2, 2, 16, 16))
    field[:, 1] = 0.6 * field[:, 0] + 0.8 * field[:, 1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_config, "_dtype", "float64")
        data = Data2D(field, pbc=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            stats = STOperator(data, 2, 4, compute_PS=False).apply(data, norm="none")
    return stats, spec.spec_apply(field, 2, True, None, 2, 4, norm="none")


@pytest.mark.parametrize("mean_along_batch", [False, True], ids=["batch", "batch_mean"])
@pytest.mark.parametrize("keep_batch_dim", [False, True], ids=["flat", "rows"])
def test_complex_form_equals_the_oracle(
    statistics: tuple[Any, dict[str, Any]], keep_batch_dim: bool, mean_along_batch: bool
) -> None:
    stats, expected = statistics
    options = {"keep_batch_dim": keep_batch_dim, "mean_along_batch": mean_along_batch}
    actual = np.asarray(stats._to_complex_vector(**options))
    assert actual.dtype == np.complex128
    assert_parity(actual, common.spec_flatten(expected, flatten_complex=False, **options))


def test_weights_scale_their_families(statistics: tuple[Any, dict[str, Any]]) -> None:
    """STL's ``reweight`` (M-SYN-LOSS, OP-51): S3 and S4 multiplied by their weights before the
    flattening, the other families unchanged."""
    stats, expected = statistics
    weights = {"S3": 3.5, "S4": 3.5**2}
    weighted = {k: v * weights.get(k, 1.0) for k, v in expected.items() if k in common.FAMILIES}
    actual = np.asarray(stats._to_complex_vector(mean_along_batch=True, weights=weights))
    assert_parity(
        actual, common.spec_flatten(weighted, flatten_complex=False, mean_along_batch=True)
    )
    with pytest.raises(ValueError, match="PS"):
        stats._to_complex_vector(weights={"PS": 2.0})
