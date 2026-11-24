"""The reserved ``partition`` argument of ``DataHealpix`` (Part II, §9.4; LF34; task T5.5, issue
#166).

In 1.0.0 only ``partition=None`` is accepted: the whole map, the full sky or the partial sky of
``cell_ids``. Any other value raises an error that names LF34, so that the interface cannot rot
before step 2 gives it the MPI partition (Part III, §18.3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from estel import DataHealpix

pytestmark = pytest.mark.maths("M-DATA-HPX")


def test_partition_none_is_the_whole_map() -> None:
    values = np.arange(48.0)
    data = DataHealpix(values, partition=None)
    assert data.shape == (1, 1, 48)
    partial = DataHealpix(values[:5], depth=1, cell_ids=np.arange(5), partition=None)
    assert partial.shape == (1, 1, 5)


@pytest.mark.parametrize("partition", ["rings", 4, (0, 1), np.arange(3)])
def test_any_other_partition_raises_and_names_lf34(partition: Any) -> None:
    with pytest.raises(ValueError, match=r"partition=None.*|LF34") as raised:
        DataHealpix(np.arange(48.0), partition=partition)
    assert "LF34" in str(raised.value)
