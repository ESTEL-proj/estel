"""The split of a full-tier run into parts (``ESTEL_TEST_SHARD``, ``tests/conftest.py``; task
T2.17): the parts are disjoint and together hold every test."""

from __future__ import annotations

import pytest

from tests.conftest import in_shard

IDS = [
    f"tests/{folder}/test_{name}.py::test_case[{case}-{dtype}]"
    for folder in ("spec", "statistics", "differentiation")
    for name in ("scattering", "plan")
    for case in range(20)
    for dtype in ("float32", "float64")
]


@pytest.mark.parametrize("count", [2, 3])
def test_every_test_belongs_to_exactly_one_part(count: int) -> None:
    parts = range(1, count + 1)
    for nodeid in IDS:
        assert [in_shard(nodeid, index, count) for index in parts].count(True) == 1, nodeid
    assert all(any(in_shard(nodeid, index, count) for nodeid in IDS) for index in parts)
