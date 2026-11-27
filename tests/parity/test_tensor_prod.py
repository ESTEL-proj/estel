"""Parity of ``STOperatorTensorProd`` with STL on tensor products (task T3.2, issue #23; Part II,
§14.3, §14.4; maths.md M-DATA-TENSORPROD, M-W1-TENSORPROD, M-LP, M-IDX-TENSORPROD, M-S-TENSORPROD,
M-BORDER-TENSORPROD, M-NORM-TENSORPROD, M-FLAT, M-SYN-LOSS; OP-30, OP-34, OP-42, OP-43, OP-81,
OP-82).

The fixtures of T3.4 (STL `2D+1D` at ``8f51143``):

* ``scattering_tensor_prod``: the vanilla statistics of every configuration, ``lowpass="ST"`` and
  ``None``; ESTEL's operator runs with ``norm="none"`` on the stored field, with the parameters
  the manifest records;
* ``statistics_tensor_prod``: the normalised statistics of three runs (``norm="self"``,
``norm="ref"`` by
  the first one's reference, ``norm="self"`` with ``ref_batch_mean``), their references and the
  two flat vectors; ESTEL's operators are built with STL's ``standardize=False``;
* ``gradient_tensor_prod``: the synthesis loss of M-SYN-LOSS on a running field against a target and
its
  gradient with respect to the field (Part II, §14.3, "Gradients").

Every comparison is in float64, with the tolerances of Part II, §14.3: same full-grid shapes,
NaN at the same positions, values within rtol 1e-10 and atol 0.1 x rtol x max|STL| (for
``mean``, its scale is the standard deviation of the data). STL interleaves the downsamplings
and the block convolutions as ESTEL does (M-W1-TENSORPROD); the second layer is summed as a Gram
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
    "M-DATA-TENSORPROD",
    "M-W1-TENSORPROD",
    "M-LP",
    "M-IDX-TENSORPROD",
    "M-S-TENSORPROD",
    "M-BORDER-TENSORPROD",
    "M-NORM-TENSORPROD",
    "M-FLAT",
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
        # STL's standardize=False (OP-120), unless the case standardises.
        return STOperatorTensorProd(data, J, **{"standardize": False, **options})


def _scattering_runs() -> list[Any]:
    return [
        pytest.param(
            name,
            lowpass,
            id=f"{name}-{MODES[lowpass]}",
            marks=(pytest.mark.slow,) if name in SLOW else (),
        )
        for name, config in _parameters("scattering_tensor_prod")["configs"].items()
        for lowpass in config["lowpass"]
    ]


@pytest.mark.parametrize(("name", "lowpass"), _scattering_runs())
def test_statistics_equal_stl(name: str, lowpass: str | None) -> None:
    """``scattering_tensor_prod``: mean, var and S1 to S4 of each configuration and low-pass
    setting, and the moments removed by ``standardize``, equal STL's (S3 and S4 of ``lowpass=None``
    are those of ``"ST"``, maths.md M-LP).

    Slow: case D, three blocks (2D + 1D + 1D), takes 1.5 s of CPU on one thread; it runs in the
    full tier only."""
    parameters = _parameters("scattering_tensor_prod")
    config = parameters["configs"][name]
    case = parameters["cases"][config["case"]]
    arrays = _arrays("scattering_tensor_prod")
    matrix = config.get("cross_matrix")
    data = _data(arrays[f"{name}_field"], case["blocks"], case["pbc"])
    stats = _operator(
        data,
        case["J"],
        lowpass=lowpass,
        standardize=config.get("standardize", False),
        cross_matrix=None if matrix is None else np.array(matrix, dtype=bool),
    ).apply(data, norm="none")
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
    """``statistics_tensor_prod``: the three runs, the references of the two ``norm="self"`` runs
    (``var_ref`` and ``S2_ref_sqrt_chan_diag``) and the real and complex flat vectors equal
    STL's (maths.md M-NORM-TENSORPROD, M-FLAT). As in STL, the first two runs are one operator,
    applied with ``norm="self"`` then ``norm="ref"`` (STL's ``store_ref`` then ``load_ref``), whose
    stored reference normalises the other field; the third is a new operator (task T2.19)."""
    parameters = _parameters("statistics_tensor_prod")
    arrays = _arrays("statistics_tensor_prod")
    blocks, pbc, J = parameters["blocks"], parameters["pbc"], parameters["J"]  # noqa: N806
    field, other = (_data(arrays[k], blocks, pbc) for k in ("field", "other_field"))
    operator = _operator(field, J)
    runs: dict[str, STStatistics] = {"store": operator.apply(field, norm="self")}
    runs["load"] = operator.apply(other, norm="ref")
    runs["store_batch_mean"] = _operator(field, J).apply(field, ref_batch_mean=True)
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
    """``gradient_tensor_prod``: the synthesis loss of M-SYN-LOSS (the target's statistics with
    ``norm="self"`` and ``ref_batch_mean``, the running field's with the target's reference,
    both weighted, averaged over the batch and flattened in the complex form of M-FLAT) and its
    gradient with respect to the running field equal STL's (Part II, §14.3, "Gradients")."""
    parameters = _parameters("gradient_tensor_prod")
    arrays = _arrays("gradient_tensor_prod")
    blocks, pbc, J = parameters["blocks"], parameters["pbc"], parameters["J"]  # noqa: N806
    backend = _registry.current_backend()
    target = _data(arrays["target"], blocks, pbc)
    # One operator, as STL's synthesis model: norm="self" on the target, then norm="ref" on the
    # running field, by the reference it stored (task T2.19).
    operator = _operator(target, J)
    target_stats = operator.apply(target, norm="self", ref_batch_mean=True)
    target_vector = flatten(target_stats, mean_field=True, keep_batch_dim=False)
    field = _data(arrays["field"], blocks, pbc)

    def objective(u: Any) -> Any:
        stats = operator.apply(field.with_array(u), norm="ref")
        return loss(flatten(stats, mean_field=True, keep_batch_dim=False), target_vector)

    value, gradient = backend.value_and_grad(objective)(field.array)
    assert_parity(np.reshape(_numpy(value), (1,)), arrays["loss"])
    assert_parity(_numpy(gradient), arrays["gradient"])
