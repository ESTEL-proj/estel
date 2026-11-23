"""Parity of ``STOperatorTensorProd`` with STL on tensor products (task T3.2, issue #23; Part II,
§14.3, §14.4; maths.md M-DATA-TP, M-W1-TP, M-LP, M-IDX-TP, M-S-TP, M-BORDER-TP, M-NORM-TP,
M-FLAT, M-SYN-LOSS; OP-30, OP-34, OP-42, OP-43, OP-81, OP-82).

The fixtures of T3.4 (STL `2D+1D` at ``8f51143``):

* ``scattering_tp``: the vanilla statistics of every configuration, ``lowpass="ST"`` and
  ``None``; ESTEL's operator runs with ``norm="none"`` on the stored field, with the parameters
  the manifest records;
* ``statistics_tp``: the normalised statistics of three runs (``norm="self"``, ``norm="ref"`` by
  the first one's reference, ``norm="self"`` with ``ref_batch_mean``), their references and the
  two flat vectors;
* ``gradient_tp``: the synthesis loss of M-SYN-LOSS on a running field against a target and its
  gradient with respect to the field (Part II, §14.3, "Gradients").

Every comparison is in float64, with the tolerances of Part II, §14.3: same full-grid shapes,
NaN at the same positions, values within rtol 1e-10 and atol 0.1 x rtol x max|STL| (for
``mean``, its scale is the standard deviation of the data). STL interleaves the downsamplings
and the block convolutions as ESTEL does (M-W1-TP); the second layer is summed as a Gram
product (M-GRAM): the difference is rounding.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataTensorProd, STOperatorTensorProd, STStatistics, _config
from estel.backend import _registry
from estel.functional._loss import loss
from estel.synthesis._loss import flatten
from tests.parity._compare import assert_parity, full_grid, load_fixture, read_manifest

pytestmark = pytest.mark.maths(
    "M-DATA-TP", "M-W1-TP", "M-LP", "M-IDX-TP", "M-S-TP", "M-BORDER-TP", "M-NORM-TP", "M-FLAT"
)
GEOMETRIES = {"2D": Data2D, "1D": Data1D}
FAMILIES = ("var", "S1", "S2", "S3", "S4")
MODES = {"ST": "st", None: "none"}
SLOW = ("D",)  # three blocks: 1.5 s of CPU on one thread


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parity runs in float64 (Part II, §14.3)."""
    monkeypatch.setattr(_config, "_dtype", "float64")


@cache
def _arrays(case: str) -> dict[str, NDArray[Any]]:
    return load_fixture(case)


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _data(field: NDArray[np.float64], blocks: list[str], pbc: list[bool]) -> DataTensorProd:
    return DataTensorProd(field, [GEOMETRIES[b] for b in blocks], pbc=pbc)


def _operator(data: DataTensorProd, J: Any, **options: Any) -> STOperatorTensorProd:  # noqa: N803
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13, OP-34: tested in tests/scattering
        return STOperatorTensorProd(data, J, **options)


def _scattering_runs() -> list[Any]:
    return [
        pytest.param(
            name,
            lowpass,
            id=f"{name}-{MODES[lowpass]}",
            marks=(pytest.mark.slow,) if name in SLOW else (),
        )
        for name, config in _parameters("scattering_tp")["configs"].items()
        for lowpass in config["lowpass"]
    ]


@pytest.mark.parametrize(("name", "lowpass"), _scattering_runs())
def test_statistics_equal_stl(name: str, lowpass: str | None) -> None:
    """``scattering_tp``: mean, var and S1 to S4 of each configuration and low-pass setting,
    and the moments removed by ``standardize``, equal STL's (S3 and S4 of ``lowpass=None`` are
    those of ``"ST"``, maths.md M-LP).

    Slow: case D, three blocks (2D + 1D + 1D), takes 1.5 s of CPU on one thread; it runs in the
    full tier only."""
    parameters = _parameters("scattering_tp")
    config = parameters["configs"][name]
    case = parameters["cases"][config["case"]]
    arrays = _arrays("scattering_tp")
    matrix = config.get("cross_matrix")
    data = _data(arrays[f"{name}_field"], case["blocks"], case["pbc"])
    stats = _operator(
        data,
        case["J"],
        lowpass=lowpass,
        norm="none",
        standardize=config.get("standardize", False),
        cross_matrix=None if matrix is None else np.array(matrix, dtype=bool),
    ).apply(data)
    prefix = f"{name}_{MODES[lowpass]}"
    scale = float(np.sqrt(np.max(arrays[f"{prefix}_var"])))
    assert_parity(_numpy(stats.full_grid("mean")), arrays[f"{prefix}_mean"], scale=scale)
    assert_parity(_numpy(stats.full_grid("var")), arrays[f"{prefix}_var"])
    for family in ("S1", "S2"):
        assert_parity(_numpy(stats.full_grid(family)), full_grid(arrays, f"{prefix}_{family}"))
    for family in ("S3", "S4"):
        assert_parity(_numpy(stats.full_grid(family)), full_grid(arrays, f"{name}_st_{family}"))
    if config.get("standardize", False):
        assert stats._standardization is not None
        for key, removed in zip(
            ("mean_pre_std", "std_pre_std"), stats._standardization, strict=True
        ):
            assert_parity(_numpy(removed), arrays[f"{prefix}_{key}"])
    else:
        assert stats._standardization is None


def test_normalised_statistics_equal_stl() -> None:
    """``statistics_tp``: the three runs, the references the operator keeps (``var_ref`` and
    ``S2_ref_sqrt_chan_diag``) and the real and complex flat vectors equal STL's (maths.md
    M-NORM-TP, M-FLAT). The second run uses the reference the first operator holds, as STL's
    ``load_ref``."""
    parameters = _parameters("statistics_tp")
    arrays = _arrays("statistics_tp")
    blocks, pbc, J = parameters["blocks"], parameters["pbc"], parameters["J"]  # noqa: N806
    field, other = (_data(arrays[k], blocks, pbc) for k in ("field", "other_field"))
    storing = _operator(field, J, norm="self")
    runs: dict[str, STStatistics] = {"store": storing.apply(field)}
    loading = STOperatorTensorProd.from_statistics(runs["store"])
    assert loading.norm == "self"
    runs["load"] = _operator(field, J, norm="ref").apply(other, ref=runs["store"])
    runs["store_batch_mean"] = _operator(field, J, norm="self", ref_batch_mean=True).apply(field)
    for run, stats in runs.items():
        var_ref = arrays[
            "store_batch_mean_var_ref" if run == "store_batch_mean" else "store_var_ref"
        ]
        scale = float(np.sqrt(np.max(arrays[f"{run}_var"] * var_ref)))
        assert_parity(_numpy(stats.full_grid("mean")), arrays[f"{run}_mean"], scale=scale)
        for family in FAMILIES:
            expected = (
                arrays[f"{run}_var"] if family == "var" else full_grid(arrays, f"{run}_{family}")
            )
            assert_parity(_numpy(stats.full_grid(family)), expected)
        if run != "load":
            assert stats._reference is not None
            assert_parity(_numpy(stats._reference.var), arrays[f"{run}_var_ref"])
            assert_parity(_numpy(stats._reference.s2_sqrt), full_grid(arrays, f"{run}_s2_ref"))
    assert_parity(_numpy(runs["store"].to_vector()), arrays["vector_real"])
    assert_parity(_numpy(runs["store"]._to_complex_vector()), arrays["vector_complex"])


def test_gradient_of_the_loss_equals_stl() -> None:
    """``gradient_tp``: the synthesis loss of M-SYN-LOSS (the target's statistics with
    ``norm="self"`` and ``ref_batch_mean``, the running field's with the target's reference,
    both weighted, averaged over the batch and flattened in the complex form of M-FLAT) and its
    gradient with respect to the running field equal STL's (Part II, §14.3, "Gradients")."""
    parameters = _parameters("gradient_tp")
    arrays = _arrays("gradient_tp")
    blocks, pbc, J = parameters["blocks"], parameters["pbc"], parameters["J"]  # noqa: N806
    backend = _registry.current_backend()
    target = _data(arrays["target"], blocks, pbc)
    target_stats = _operator(target, J, norm="self", ref_batch_mean=True).apply(target)
    target_vector = flatten(target_stats, mean_field=True, keep_batch_dim=False)
    running = _operator(target, J, norm="ref")
    field = _data(arrays["field"], blocks, pbc)

    def objective(u: Any) -> Any:
        stats = running.apply(field.with_array(u), ref=target_stats)
        return loss(flatten(stats, mean_field=True, keep_batch_dim=False), target_vector)

    value, gradient = backend.value_and_grad(objective)(field.array)
    assert_parity(np.reshape(_numpy(value), (1,)), arrays["loss"])
    assert_parity(_numpy(gradient), arrays["gradient"])
