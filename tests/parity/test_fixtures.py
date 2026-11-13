"""The STL parity fixtures and the comparison helpers (Part II, §14.2, §14.3; task T0.8).

These tests check the integrity of the fixtures (against MANIFEST.json: checksum, STL commit,
image ID, torch's default dtype; their dtypes and shapes) and the comparison helpers, on
synthetic arrays. They do not compare ESTEL with STL: each parity test does that for its own
fixture, with the helpers tested here; ESTEL's 1D and 2D kernels are compared with
`bump_steerable_kernel_1d` and `bump_steerable_kernel_2d` in `test_kernels.py` (task T2.2).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.parity._compare import (
    FIXTURE_DTYPES,
    FIXTURES,
    MANIFEST,
    PARITY_RTOL,
    STL_COMMITS,
    assert_parity,
    load_fixture,
    parity_atol,
    read_manifest,
    sha256,
)

REPO = Path(__file__).resolve().parents[2]
ENTRIES = sorted(read_manifest().items())
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def test_manifest_lists_exactly_the_fixture_files() -> None:
    files = {path.name for path in FIXTURES.iterdir() if path.name != MANIFEST}
    assert files == {name for name, _ in ENTRIES}, (
        f"{FIXTURES} and {MANIFEST} disagree: every fixture file has one manifest entry. Only "
        "make parity-fixtures writes them; restore the directory from main."
    )


@pytest.mark.parametrize(("name", "entry"), ENTRIES, ids=[name for name, _ in ENTRIES])
def test_fixture_matches_its_manifest_entry(name: str, entry: dict[str, Any]) -> None:
    assert sha256(FIXTURES / name) == entry["sha256"]
    assert entry["stl_commit"] == STL_COMMITS[entry["stl_branch"]], (
        f"{name} was produced at STL {entry['stl_branch']} {entry['stl_commit']}, not at the "
        "commit of Part II, §9.1: rebuild the image (make parity-image) and regenerate it."
    )
    assert IMAGE_ID.fullmatch(entry["image_id"])
    assert entry.get("torch_default_dtype") == "float64", (
        f"{name} was produced with torch's default dtype {entry.get('torch_default_dtype')}: STL "
        "runs with float64 as default (make_fixtures.py). Regenerate it with make parity-fixtures."
    )
    assert entry["case"] == f"tools/stl_parity/cases/{Path(name).stem}.py"
    assert (REPO / entry["case"]).is_file(), f"{name}: its case script {entry['case']} is gone"
    arrays = load_fixture(Path(name).stem)
    assert sorted(arrays) == sorted(entry["arrays"])
    for key, array in arrays.items():
        assert str(array.dtype) in FIXTURE_DTYPES, f"{name}:{key} is {array.dtype}"
        assert str(array.dtype) == entry["arrays"][key]["dtype"]
        assert list(array.shape) == entry["arrays"][key]["shape"]


def test_bump_steerable_kernel_2d_fixture() -> None:
    """The example case: STL master's 2D bump-steerable kernels, L=4 and K=5 (STL's defaults)."""
    entry = read_manifest()["bump_steerable_kernel_2d.npz"]
    assert entry["stl_branch"] == "master"
    assert entry["parameters"] == {"J": 3, "L": 4, "K": 5, "N": 32, "seed": 0}
    kernel = load_fixture("bump_steerable_kernel_2d")["kernel"]
    assert kernel.dtype == np.complex128
    assert kernel.shape == (1, 4, 5, 5)
    assert np.isfinite(kernel).all()


def test_bump_steerable_kernel_1d_fixture() -> None:
    """STL 2D+1D's 1D bump-steerable kernels, K=5, for L=1 (the default) and L=2 (the 1D
    blocks of tensor products), as STL holds them: (1, L, K)."""
    entry = read_manifest()["bump_steerable_kernel_1d.npz"]
    assert entry["stl_branch"] == "2D+1D"
    assert entry["parameters"] == {"J": 5, "L": [1, 2], "K": 5, "N": 128, "seed": 0}
    arrays = load_fixture("bump_steerable_kernel_1d")
    for n_orientations in (1, 2):
        kernel = arrays[f"kernel_L{n_orientations}"]
        assert kernel.dtype == np.complex128
        assert kernel.shape == (1, n_orientations, 5)
        assert np.isfinite(kernel).all()


# ---------------------------------------------------------------- load_fixture


def _copy_fixtures(tmp_path: Path) -> Path:
    directory = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, directory)
    return directory


def test_load_fixture_refuses_a_changed_file(tmp_path: Path) -> None:
    directory = _copy_fixtures(tmp_path)
    path = directory / "bump_steerable_kernel_2d.npz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="checksum"):
        load_fixture("bump_steerable_kernel_2d", directory)


def test_load_fixture_refuses_a_case_missing_from_the_manifest(tmp_path: Path) -> None:
    directory = _copy_fixtures(tmp_path)
    (directory / MANIFEST).write_text(json.dumps({}))
    with pytest.raises(KeyError, match="ask a maintainer"):
        load_fixture("bump_steerable_kernel_2d", directory)


def test_read_manifest_explains_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make parity-fixtures"):
        read_manifest(tmp_path)


# ---------------------------------------------------------------- assert_parity


def _reference(dtype: type = np.float64) -> NDArray[Any]:
    values = np.random.default_rng(0).standard_normal((3, 4, 5)) + 0.5
    if dtype is np.complex128:
        values = values + 1j * np.random.default_rng(1).standard_normal((3, 4, 5))
    return values.astype(dtype)


@pytest.mark.parametrize("kind", [np.float64, np.complex128])
@pytest.mark.parametrize("dtype", sorted(PARITY_RTOL))
def test_assert_parity_accepts_differences_within_rtol(kind: type, dtype: str) -> None:
    expected = _reference(kind)
    assert_parity(expected.copy(), expected, dtype)
    assert_parity(expected * (1 + 0.5 * PARITY_RTOL[dtype]), expected, dtype)


@pytest.mark.parametrize("kind", [np.float64, np.complex128])
@pytest.mark.parametrize("dtype", sorted(PARITY_RTOL))
def test_assert_parity_rejects_differences_beyond_rtol(kind: type, dtype: str) -> None:
    expected = _reference(kind)
    actual = expected.copy()
    allowed = parity_atol(expected, dtype) + PARITY_RTOL[dtype] * abs(expected[1, 2, 3])
    actual[1, 2, 3] += 2 * allowed
    with pytest.raises(AssertionError, match=r"1 of 60 coefficients .* Worst at \(1, 2, 3\)"):
        assert_parity(actual, expected, dtype)


def test_assert_parity_float32_check_is_looser_than_float64() -> None:
    expected = _reference()
    actual = expected * (1 + 1e-7)
    assert_parity(actual, expected, "float32")
    with pytest.raises(AssertionError, match="rtol 1e-10"):
        assert_parity(actual, expected, "float64")


def test_assert_parity_accepts_identical_nan_positions() -> None:
    expected = _reference()
    expected[0, 1, :] = np.nan
    assert_parity(expected.copy(), expected)


@pytest.mark.parametrize("where", ["estel", "stl"])
def test_assert_parity_rejects_different_nan_positions(where: str) -> None:
    expected = _reference()
    actual = expected.copy()
    (actual if where == "estel" else expected)[2, 0, 4] = np.nan
    with pytest.raises(AssertionError, match=r"NaN positions differ .* \[\(2, 0, 4\)\]"):
        assert_parity(actual, expected)


@pytest.mark.parametrize(("dtype", "atol"), [("float64", 1e-11), ("float32", 1e-6)])
def test_parity_atol_is_a_tenth_of_rtol_times_the_largest_finite_modulus(
    dtype: str, atol: float
) -> None:
    assert parity_atol(np.array([0.5, -1.0, 0.25]), dtype) == pytest.approx(atol)
    assert parity_atol(np.array([0.5, -4.0, 2.0]), dtype) == pytest.approx(4 * atol)
    assert parity_atol(np.array([3 + 4j, 1.0]), dtype) == pytest.approx(5 * atol)
    assert parity_atol(np.array([np.nan, np.inf, -np.inf, 2.0]), dtype) == pytest.approx(2 * atol)
    assert parity_atol(np.array([np.nan, np.inf]), dtype) == 0.0
    assert parity_atol(np.zeros(3), dtype) == 0.0


@pytest.mark.parametrize(("dtype", "atol"), [("float64", 1e-11), ("float32", 1e-6)])
def test_parity_atol_takes_a_given_scale_instead_of_the_largest_value(
    dtype: str, atol: float
) -> None:
    """§14.3: a quantity that may be zero in theory, such as ``mean``, takes its natural scale
    (sqrt(max(var)) for ``mean``) instead of its own largest value."""
    assert parity_atol(np.array([1e-17, -1e-17]), dtype, scale=3.0) == pytest.approx(3 * atol)
    assert parity_atol(np.array([100.0]), dtype, scale=2.0) == pytest.approx(2 * atol)
    assert parity_atol(np.array([1.0]), dtype, scale=0.0) == 0.0
    for scale in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="scale"):
            parity_atol(np.ones(2), dtype, scale=scale)


def test_assert_parity_uses_the_given_scale() -> None:
    expected = np.array([1e-17, -2e-17])  # a mean zero in theory: its own max is rounding
    actual = np.array([5e-12, -3e-12])
    with pytest.raises(AssertionError, match="beyond atol"):
        assert_parity(actual, expected)
    assert_parity(actual, expected, scale=1.0)  # within 1e-11 x 1
    with pytest.raises(AssertionError, match=r"atol 1e-12 .* Worst at \(0,\)"):
        assert_parity(actual, expected, scale=0.1)


def test_parity_atol_refuses_an_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="float16"):
        parity_atol(np.ones(2), "float16")


@pytest.mark.parametrize("dtype", sorted(PARITY_RTOL))
def test_assert_parity_accepts_rounding_on_values_zero_in_theory(dtype: str) -> None:
    assert_parity(np.array([1.0, 1e-17]), np.array([1.0, 0.0]), dtype)
    assert_parity(np.array([1.0 + 1e-17j, -1e-17j]), np.array([1.0 + 0j, 0j]), dtype)


def test_assert_parity_float32_atol_is_looser_than_float64() -> None:
    expected = np.array([1.0, 0.0])
    actual = np.array([1.0, 5e-7])  # within 1e-6 (float32), beyond 1e-11 (float64)
    assert_parity(actual, expected, "float32")
    with pytest.raises(AssertionError, match=r"atol 1e-11 .* Worst at \(1,\)"):
        assert_parity(actual, expected, "float64")


@pytest.mark.parametrize("dtype", sorted(PARITY_RTOL))
def test_assert_parity_atol_scales_with_the_largest_value(dtype: str) -> None:
    expected = np.array([100.0, 0.0])
    atol = parity_atol(expected, dtype)
    assert atol == pytest.approx(10 * PARITY_RTOL[dtype])  # 0.1 x rtol x 100
    assert_parity(np.array([100.0, 0.9 * atol]), expected, dtype)
    message = rf"atol {re.escape(f'{atol:.3g}')} .* Worst at \(1,\)"
    with pytest.raises(AssertionError, match=message):
        assert_parity(np.array([100.0, 2 * atol]), expected, dtype)


def test_assert_parity_needs_exact_zeros_when_everything_is_zero() -> None:
    expected = np.zeros(4)
    assert_parity(np.zeros(4), expected)
    with pytest.raises(AssertionError, match="allowed 0"):
        assert_parity(np.array([0.0, 1e-300, 0.0, 0.0]), expected)


def test_assert_parity_atol_ignores_infinities() -> None:
    expected = np.array([1.0, np.inf, 0.0])
    with pytest.raises(AssertionError, match=r"atol 1e-11 .* Worst at \(2,\)"):
        assert_parity(np.array([1.0, np.inf, 1e-3]), expected)


def test_assert_parity_rejects_infinities_of_opposite_sign() -> None:
    expected = np.array([1.0, np.inf])
    assert_parity(expected.copy(), expected)
    with pytest.raises(AssertionError, match="1 of 2"):
        assert_parity(np.array([1.0, -np.inf]), expected)


def test_assert_parity_rejects_another_shape() -> None:
    with pytest.raises(AssertionError, match="full-grid shapes"):
        assert_parity(np.zeros((2, 3)), np.zeros((3, 2)))


def test_assert_parity_refuses_an_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="float16"):
        assert_parity(np.zeros(2), np.zeros(2), "float16")
