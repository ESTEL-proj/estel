"""The oracle of tensor products against STL's parity fixtures (task T3.3, issue #126; maths.md
section 3; Part I, §4.2 and §5.1; Part II, §12.11 and §14.3).

The fixtures of T3.4 (``scattering_tp``, ``statistics_tp``, STL `2D+1D` at ``8f51143``) hold
STL's statistics of tensor-product fields. The oracle ``spec_apply_tensor_prod`` runs on the same
fields, with the parameters the manifest records, and every stored array is compared with
``assert_parity``: same full-grid shape, NaN at the same positions, values within the float64
tolerances of Part II, §14.3 (for ``mean``, atol = 0.1 x rtol x sqrt(max(var))). The oracle
brings the field to the level of a scale in every block first, STL interleaves the
downsamplings and the block convolutions (M-W1-TP): the difference is rounding.

Every test runs in the fast tier, except case D of ``scattering_tp`` (three blocks), which runs
in the full tier.

A disagreement is a finding about ``docs/maths.md`` or about the oracle: do not change the
oracle, a fixture or a tolerance to make a test pass; report the M- ID, the configuration and
the coefficients on the issue.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel._spec import cartesian, common
from estel._spec import tensor_prod as spec
from tests.parity._compare import assert_parity, full_grid, load_fixture, read_manifest

pytestmark = pytest.mark.maths(
    "M-NOT-TP",
    "M-DATA-TP",
    "M-W1-TP",
    "M-LP",
    "M-IDX-TP",
    "M-S-TP",
    "M-BORDER-TP",
    "M-NORM-TP",
    "M-DEF-TP",
    "M-MOM",
    "M-STD",
    "M-CHAN",
    "M-FLAT",
)
GRIDS = ("S1", "S2", "S3", "S4")
# M-DATA-TP: the number of spatial axes of a block.
AXES = {"2D": 2, "1D": 1}
MODES = {"ST": "st", None: "none"}
SLOW = ("D",)


@cache
def _arrays(case: str) -> dict[str, NDArray[Any]]:
    return load_fixture(case)


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _scattering_runs() -> list[Any]:
    out = []
    for name, config in _parameters("scattering_tp")["configs"].items():
        for lowpass in config["lowpass"]:
            marks = (pytest.mark.slow,) if name in SLOW else ()
            out.append(pytest.param(name, lowpass, id=f"{name}-{MODES[lowpass]}", marks=marks))
    return out


@pytest.mark.parametrize(("name", "lowpass"), _scattering_runs())
def test_scattering_equals_the_fixture(name: str, lowpass: str | None) -> None:
    """``mean``, ``var``, S1 to S4 as stored (S1 and S2 only with ``lowpass=None``), and the
    moments of the standardisation, of each configuration of ``scattering_tp``, without
    normalisation.

    Slow: case D, three blocks, about 0.8 s of CPU locally and over 1 s on CI.
    """
    parameters = _parameters("scattering_tp")
    config = parameters["configs"][name]
    case = parameters["cases"][config["case"]]
    arrays = _arrays("scattering_tp")
    prefix = f"{name}_{MODES[lowpass]}"
    out = spec.spec_apply_tensor_prod(
        arrays[f"{name}_field"],
        [AXES[block] for block in case["blocks"]],
        case["pbc"],
        case["J"],
        lowpass=lowpass,
        norm="none",
        standardize=config.get("standardize", False),
        cross_matrix=config.get("cross_matrix"),
    )
    scale = float(np.sqrt(np.max(arrays[f"{prefix}_var"])))
    assert_parity(out["mean"], arrays[f"{prefix}_mean"], scale=scale)
    assert_parity(out["var"], arrays[f"{prefix}_var"])
    for family in GRIDS if lowpass == "ST" else ("S1", "S2"):
        assert_parity(out[family], full_grid(arrays, f"{prefix}_{family}"))
    if config.get("standardize", False):
        for key in ("mean_pre_std", "std_pre_std"):
            assert_parity(out[key], arrays[f"{prefix}_{key}"])


def test_normalised_statistics_equal_the_fixture() -> None:
    """The three runs of ``statistics_tp`` (``norm="self"``, ``norm="ref"`` with its reference,
    ``norm="self"`` with ``ref_batch_mean``), their references and the two flat vectors: the
    real one from ``spec_apply_tensor_prod(..., flatten=True)``, the complex one from
    ``spec_flatten``."""
    parameters = _parameters("statistics_tp")
    arrays = _arrays("statistics_tp")
    options = {
        "blocks": [AXES[block] for block in parameters["blocks"]],
        "pbc": parameters["pbc"],
        "J": parameters["J"],
        "lowpass": parameters["lowpass"],
        "standardize": False,  # as STL computed the fixture (OP-120)
    }
    store = spec.spec_apply_tensor_prod(arrays["field"], norm="self", flatten=True, **options)
    runs = {
        "store": store,
        "load": spec.spec_apply_tensor_prod(
            arrays["other_field"], norm="ref", ref=store["reference"], **options
        ),
        "store_batch_mean": spec.spec_apply_tensor_prod(
            arrays["field"], norm="self", ref_batch_mean=True, **options
        ),
    }
    for run, out in runs.items():
        # The raw var of the run, from the expected side: its normalised var times var_ref.
        var_ref = arrays[
            "store_batch_mean_var_ref" if run == "store_batch_mean" else "store_var_ref"
        ]
        scale = float(np.sqrt(np.max(arrays[f"{run}_var"] * var_ref)))
        assert_parity(out["mean"], arrays[f"{run}_mean"], scale=scale)
        assert_parity(out["var"], arrays[f"{run}_var"])
        for family in GRIDS:
            assert_parity(out[family], full_grid(arrays, f"{run}_{family}"))
        if run != "load":
            assert_parity(out["reference"]["var"], arrays[f"{run}_var_ref"])
            assert_parity(out["reference"]["sigma"], full_grid(arrays, f"{run}_s2_ref"))
    assert_parity(common.spec_flatten(store, flatten_complex=False), arrays["vector_complex"])
    assert_parity(store["vector"], arrays["vector_real"])


@pytest.mark.parametrize(
    ("ndim", "shape", "pbc"),
    [pytest.param(2, (16, 16), True, id="2d"), pytest.param(1, (16,), False, id="1d")],
)
def test_one_block_equals_the_cartesian_oracle(
    ndim: int, shape: tuple[int, ...], pbc: bool
) -> None:
    """With ``lowpass=None``, a tensor product of one block is the planar geometry (OP-82):
    same families, same full grids, the 1D block with its tensor default L = 2."""
    field = np.random.default_rng(12).standard_normal((2, *shape))
    out = spec.spec_apply_tensor_prod(field, [ndim], [pbc], [2], lowpass=None, norm="self")
    assert out["L"] == [4 if ndim == 2 else 2]
    expected = cartesian.spec_apply(field, ndim, pbc, None, 2, out["L"][0], norm="self")
    for family in ("mean", "var", *GRIDS):
        assert_parity(out[family], expected[family])


def _with_nan() -> NDArray[np.float64]:
    field = np.random.default_rng(13).standard_normal((8, 8, 8))
    field[0, 0, 0] = np.nan
    return field


@pytest.mark.parametrize(
    ("array", "J", "L", "match"),
    [
        pytest.param(_with_nan(), None, None, "NaN", id="nan-OP-80"),
        pytest.param(np.zeros((8, 8, 6)), [1, 3], None, "not divisible", id="size-OP-8"),
        pytest.param(np.zeros((8, 8, 8)), 1, None, "one entry per block", id="int-J"),
        pytest.param(np.zeros((8, 8, 8)), np.int64(1), None, "one entry per block", id="numpy-J"),
        pytest.param(np.zeros((8, 8, 8)), None, np.int64(2), "one entry per block", id="numpy-L"),
    ],
)
def test_invalid_data_raise(array: NDArray[Any], J: Any, L: Any, match: str) -> None:  # noqa: N803
    """A NaN in the data (OP-41, OP-80), a size not divisible by 2^(J_a - 1) (OP-8), and a J or
    L that is not a list, a NumPy integer included (OP-83), raise a ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        spec.spec_data_tensor_prod(array, [2, 1], [True, True], J, L)


def test_caller_lists_and_defaults() -> None:
    """The caller's lists J and L are left unchanged (OP-83), and their ``None`` entries take the
    default of the block (M-DEF-TP)."""
    field = np.random.default_rng(13).standard_normal((8, 8, 8))
    scales, orientations = [None, 1], [None, None]
    x, n_scales, n_orient, _ = spec.spec_data_tensor_prod(
        field, [2, 1], [True, False], scales, orientations
    )
    assert scales == [None, 1]
    assert orientations == [None, None]
    assert (x.shape, n_scales, n_orient) == ((1, 1, 8, 8, 8), [1, 1], [4, 2])
