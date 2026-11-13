"""STL parity fixtures and the comparison helpers of the parity tests (Part II, §14.2, §14.3).

STL's results enter the tests as fixtures: `.npz` files of `tests/parity/fixtures/`, produced in
the parity container by `make parity-fixtures` and recorded in `MANIFEST.json`. A parity test
loads one with `load_fixture`, which checks it against the manifest, and compares ESTEL with it
with `assert_parity`:

    expected = load_fixture("bump_steerable_kernel_2d")["kernel"]
    assert_parity(actual, expected, dtype)

`tests/parity/fixtures/` is protected: only a maintainer regenerates it. A failing comparison is
reported in an issue, with the coefficients it names; never regenerate the fixture or loosen the
tolerance to make it pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MANIFEST = "MANIFEST.json"
# Part II, §14.3: both libraries run in float64 for parity; float32 is a secondary check.
PARITY_RTOL = {"float64": 1e-10, "float32": 1e-5}
# The STL-Dev commit of each branch (Part II, §9.1), as the parity container pins them.
STL_COMMITS = {
    "master": "8eaf7691d55f3c3441cde9b95f1f80eb17aa8014",
    "2D+1D": "8f51143c5f8ebc259328830f246894d7031acae5",
    "HealpixUpdate": "90eb1a954e71abb7af621b10e70b0456e8a86b4e",
}
FIXTURE_DTYPES = ("float64", "complex128")
_REPORT = (
    "Do not regenerate the fixture or loosen the tolerance: report the coefficients above in an "
    "issue (Part II, §14.3)."
)


def read_manifest(directory: Path = FIXTURES) -> dict[str, dict[str, Any]]:
    """The entries of ``directory/MANIFEST.json``, keyed by fixture file name."""
    path = directory / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: a maintainer produces the fixtures with make "
            "parity-fixtures (tools/stl_parity/README.md)."
        )
    manifest: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(case: str, directory: Path = FIXTURES) -> dict[str, NDArray[Any]]:
    """The arrays of fixture ``case``, after checking its file against the manifest."""
    name = f"{case}.npz"
    entry = read_manifest(directory).get(name)
    if entry is None:
        raise KeyError(
            f"no fixture {name} in {directory / MANIFEST}: write the case in "
            "tools/stl_parity/cases/ and ask a maintainer for its fixture in your PR (§14.2)."
        )
    path = directory / name
    if sha256(path) != entry["sha256"]:
        raise ValueError(
            f"{path} does not match the checksum of {MANIFEST}: the file was changed outside "
            "make parity-fixtures. Restore it from main; only a maintainer regenerates fixtures."
        )
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def assert_parity(actual: Any, expected: NDArray[Any], dtype: str = "float64") -> None:
    """Assert that ``actual`` equals the STL fixture ``expected`` within §14.3's tolerances.

    Same shape; NaN at exactly the same positions; elsewhere ``|actual - expected| <= rtol *
    |expected|`` element by element (complex moduli for complex arrays), with ``rtol`` the §14.3
    value for the ``dtype`` ESTEL computed in: 1e-10 in float64, 1e-5 in float32.
    """
    if dtype not in PARITY_RTOL:
        raise ValueError(f"dtype={dtype!r}: pass one of {sorted(PARITY_RTOL)}.")
    rtol = PARITY_RTOL[dtype]
    actual = np.asarray(actual)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape {actual.shape} differs from STL's {expected.shape}: in step 1, ESTEL stores "
            "results in STL's full-grid shapes (Part II, §12.7)."
        )
    nan_actual, nan_expected = np.isnan(actual), np.isnan(expected)
    if not np.array_equal(nan_actual, nan_expected):
        positions = np.argwhere(nan_actual != nan_expected)
        raise AssertionError(
            f"NaN positions differ from STL's at {len(positions)} indices, first "
            f"{[tuple(int(i) for i in p) for p in positions[:5]]}: they must match exactly "
            f"(§14.3). {_REPORT}"
        )
    valid = ~nan_expected
    got, want = actual[valid], expected[valid]
    with np.errstate(invalid="ignore"):
        error = np.abs(got.astype(np.complex128) - want)
    magnitude = np.abs(want)
    # An infinite value passes only when it is equal: its error is not finite.
    failing = ~((got == want) | (np.isfinite(error) & (error <= rtol * magnitude)))
    if failing.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(magnitude > 0, error / magnitude, np.inf)[failing]
        worst = int(np.argmax(relative))
        index = tuple(int(i) for i in np.argwhere(valid)[failing][worst])
        raise AssertionError(
            f"{int(failing.sum())} of {int(valid.sum())} coefficients differ from STL beyond "
            f"rtol {rtol:g} ({dtype}). Worst at {index}: ESTEL {actual[index]!r}, STL "
            f"{expected[index]!r}, relative difference {relative[worst]:.3g}. {_REPORT}"
        )
