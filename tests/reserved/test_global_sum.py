"""The reserved MPI hook ``_global_sum`` (Part II, §9.4; Part III, §18.3; task T2.5, issue #17).

In 1.0.0 it is the identity. It sits between the weighted pixel sums of the means and their
division, and the loop of single-domain data calls it once per level that has means, on the
stacked sums of that level, plus once for each of the two moments of ``standardize``; the loop
of tensor products, once for the moments of the field, once per scale of the first layer and
once per wavelet scale of the second, and once for the centring of the low-pass scale. Both
loops read it from ``estel.scattering._loop`` at each call. These tests exercise the trivial
implementation through a counting wrapper, on both loops, so that the interface cannot rot
before step 2 makes it an MPI Allreduce.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from estel import Data, Data1D, Data2D, DataTensorProd, STOperator, STOperatorTensorProd, _config
from estel.functional import _scattering
from estel.scattering import _loop
from estel.scattering._plan import STPlanTensorProd


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    monkeypatch.setattr(_config, "_dtype", dtype)


@pytest.fixture
def replace_global_sum(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Any], None]]:
    """Replace the hook. On a backend that compiles the loop (JAX; task T2.13), the hook is
    read when the loop is compiled: the compiled loops are dropped when the hook changes, and
    again after the test, so that no other test runs a loop compiled with a replaced hook."""

    def replace(hook: Any) -> None:
        monkeypatch.setattr(_loop, "_global_sum", hook)
        _scattering._loop.cache_clear()

    yield replace
    _scattering._loop.cache_clear()


def _operator(data: Data, J: int, **kwargs: Any) -> STOperator:  # noqa: N803
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        options = {"standardize": False, "compute_PS": False if data.ndim == 2 else None}
        return STOperator(data, J, **{**options, **kwargs})


def _vector(data: Data, operator: STOperator) -> Any:
    return np.asarray(operator.apply(data, norm="none").to_vector())


@pytest.mark.parametrize(("standardize", "extra"), [(False, 0), (True, 2)])
def test_global_sum_is_called_once_per_level(
    replace_global_sum: Callable[[Any], None], standardize: bool, extra: int
) -> None:
    data = Data2D(np.random.default_rng(0).standard_normal((2, 2, 32, 32)), pbc=False)
    operator = _operator(data, 3, standardize=standardize)
    baseline = _vector(data, operator)
    calls: list[tuple[int, ...]] = []

    def counting(partial_sums: Any) -> Any:
        calls.append(tuple(int(n) for n in partial_sums.shape))
        return partial_sums

    replace_global_sum(counting)
    np.testing.assert_array_equal(_vector(data, operator), baseline)
    n_scales = operator.J
    assert isinstance(n_scales, int)  # planar data
    assert len(calls) == n_scales + extra
    assert all(len(shape) == 2 and shape[0] == 2 for shape in calls)  # (Nb, n), stacked


@pytest.mark.parametrize(("standardize", "extra"), [(False, 0), (True, 2)])
def test_global_sum_is_called_by_the_loop_of_tensor_prod_data(
    replace_global_sum: Callable[[Any], None], standardize: bool, extra: int
) -> None:
    """The loop of tensor products reaches a replaced hook too: once for the moments of the
    field, once per scale of the first layer, once for the centring of the scale with the
    low-pass filter in every block, and once per wavelet scale of the second layer."""
    field = np.random.default_rng(2).standard_normal((2, 1, 8, 8, 8))
    data = DataTensorProd(field, [Data2D, Data1D], pbc=[True, True])
    operator = STOperatorTensorProd(data, J=(1, 1), L=(4, 2), standardize=standardize)
    baseline = _vector(data, operator)
    calls: list[tuple[int, ...]] = []

    def counting(partial_sums: Any) -> Any:
        calls.append(tuple(int(n) for n in partial_sums.shape))
        return partial_sums

    replace_global_sum(counting)
    np.testing.assert_array_equal(_vector(data, operator), baseline)
    plan = operator.plan
    assert isinstance(plan, STPlanTensorProd)
    loop = plan.loop
    centred = sum(scale.centred for scale in loop.first_layer)
    assert centred == 1
    assert len(calls) == 1 + len(loop.first_layer) + centred + len(loop.second_layer) + extra
    assert all(shape[0] == 2 for shape in calls)


def test_global_sum_sits_between_the_sum_and_the_division(
    replace_global_sum: Callable[[Any], None],
) -> None:
    """Doubling every partial sum, as two identical partitions would, doubles every mean."""
    data = Data1D(np.random.default_rng(1).standard_normal((1, 2, 64)) + 0.5, pbc=True)
    operator = _operator(data, 3)
    baseline = operator.apply(data, norm="none")
    replace_global_sum(lambda partial_sums: 2 * partial_sums)
    doubled = operator.apply(data, norm="none")
    for family in ("mean", "var", "S1", "S2", "S3", "S4"):
        np.testing.assert_allclose(
            np.asarray(doubled.full_grid(family)),
            2 * np.asarray(baseline.full_grid(family)),
            rtol=1e-5,
            err_msg=family,
        )


def test_global_sum_is_the_identity() -> None:
    x = np.arange(6.0).reshape(2, 3)
    assert _loop._global_sum(x) is x
