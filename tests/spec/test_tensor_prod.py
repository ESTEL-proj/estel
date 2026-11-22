"""The oracle of tensor products against STL's parity fixtures (task T3.3, issue #126; maths.md
section 3; Part I, §4.2 and §5.1; Part II, §12.11 and §14.3).

The fixtures of T3.4 (``scattering_tp``, ``statistics_tp``, STL `2D+1D` at ``8f51143``) hold
STL's statistics of tensor-product fields. The oracle ``spec_tensor_prod`` runs on the same
fields, with the parameters the manifest records, and every stored array is compared with
``assert_parity``: same full-grid shape, NaN at the same positions, values within the float64
tolerances of Part II, §14.3 (for ``mean``, atol = 0.1 x rtol x sqrt(max(var))). The oracle
brings the field to the level of a scale in every block first, STL interleaves the
downsamplings and the block convolutions (M-W1-TP): the difference is rounding.

The fast tier runs case C (a 16 x 16 periodic 2D block and an 8-sample open 1D block, the
default J and L), with ``lowpass="ST"`` and ``None``, the one-block tests and the errors. The
other configurations and the normalised runs are in the full tier.

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

from estel._spec import planar
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
FAST = ("C",)


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
            marks = () if name in FAST else (pytest.mark.slow,)
            out.append(pytest.param(name, lowpass, id=f"{name}-{MODES[lowpass]}", marks=marks))
    return out


@pytest.mark.parametrize(("name", "lowpass"), _scattering_runs())
def test_scattering_equals_the_fixture(name: str, lowpass: str | None) -> None:
    """``mean``, ``var``, S1 to S4 as stored (S1 and S2 only with ``lowpass=None``), and the
    moments of the standardisation, of each configuration of ``scattering_tp``, without
    normalisation.

    Slow: every configuration but C, by the decisions on issue #126 (the fast tier runs one
    configuration); together about 2.7 s of CPU, case D alone 0.8 s.
    """
    parameters = _parameters("scattering_tp")
    config = parameters["configs"][name]
    case = parameters["cases"][config["case"]]
    arrays = _arrays("scattering_tp")
    prefix = f"{name}_{MODES[lowpass]}"
    out = spec.spec_tensor_prod(
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


@pytest.mark.slow
def test_normalised_statistics_equal_the_fixture() -> None:
    """The three runs of ``statistics_tp`` (``norm="self"``, ``norm="ref"`` with its reference,
    ``norm="self"`` with ``ref_batch_mean``), their references and the two flat vectors.

    Slow: by the decisions on issue #126, the normalised runs are in the full tier (three runs
    of two maps of two channels, about 0.1 s of CPU).
    """
    parameters = _parameters("statistics_tp")
    arrays = _arrays("statistics_tp")
    options = {
        "blocks": [AXES[block] for block in parameters["blocks"]],
        "pbc": parameters["pbc"],
        "J": parameters["J"],
        "lowpass": parameters["lowpass"],
    }
    store = spec.spec_tensor_prod(arrays["field"], norm="self", **options)
    runs = {
        "store": store,
        "load": spec.spec_tensor_prod(
            arrays["other_field"], norm="ref", ref=store["reference"], **options
        ),
        "store_batch_mean": spec.spec_tensor_prod(
            arrays["field"], norm="self", ref_batch_mean=True, **options
        ),
    }
    for run, out in runs.items():
        scale = float(np.sqrt(np.max(out["var"] * out["reference"]["var"])))
        assert_parity(out["mean"], arrays[f"{run}_mean"], scale=scale)
        assert_parity(out["var"], arrays[f"{run}_var"])
        for family in GRIDS:
            assert_parity(out[family], full_grid(arrays, f"{run}_{family}"))
        if run != "load":
            assert_parity(out["reference"]["var"], arrays[f"{run}_var_ref"])
            assert_parity(out["reference"]["sigma"], full_grid(arrays, f"{run}_s2_ref"))
    for form, flatten_complex in (("complex", False), ("real", True)):
        vector = planar.spec_flatten(store, flatten_complex=flatten_complex)
        assert_parity(vector, arrays[f"vector_{form}"])


@pytest.mark.parametrize(
    ("ndim", "shape", "pbc"),
    [pytest.param(2, (16, 16), True, id="2d"), pytest.param(1, (16,), False, id="1d")],
)
def test_one_block_equals_the_planar_oracle(ndim: int, shape: tuple[int, ...], pbc: bool) -> None:
    """With ``lowpass=None``, a tensor product of one block is the planar geometry (OP-82):
    same families, same full grids, the 1D block with its tensor default L = 2."""
    field = np.random.default_rng(12).standard_normal((2, *shape))
    out = spec.spec_tensor_prod(field, [ndim], [pbc], [2], lowpass=None, norm="self")
    assert out["L"] == [4 if ndim == 2 else 2]
    expected = planar.spec_apply(field, ndim, pbc, None, 2, out["L"][0], norm="self")
    for family in ("mean", "var", *GRIDS):
        assert_parity(out[family], expected[family])


def test_errors_and_arguments() -> None:
    """A NaN in the data (OP-41, OP-80) and a size not divisible by 2^(J_a - 1) (OP-8) raise a
    ``ValueError``; the caller's lists J and L are left unchanged (OP-83), and their ``None``
    entries take the default of the block (M-DEF-TP)."""
    field = np.random.default_rng(13).standard_normal((8, 8, 8))
    with_nan = field.copy()
    with_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        spec.spec_data_tp(with_nan, [2, 1], [True, True])
    with pytest.raises(ValueError, match="not divisible"):
        spec.spec_data_tp(np.zeros((8, 8, 6)), [2, 1], [True, True], [1, 3])
    scales, orientations = [None, 1], [None, None]
    x, n_scales, n_orient, _ = spec.spec_data_tp(field, [2, 1], [True, False], scales, orientations)
    assert scales == [None, 1]
    assert orientations == [None, None]
    assert (x.shape, n_scales, n_orient) == ((1, 1, 8, 8, 8), [1, 1], [4, 2])
