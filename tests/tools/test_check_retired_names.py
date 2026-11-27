"""tools/check_retired_names.py: retired public names stay out, and the rule of names of Part I,
§2.4.1 holds, through the table of renamed names (task T7.2, #193).

Each tree below is a small git repository, since the tool reads the files git tracks.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from check_retired_names import check, run

from .conftest import REPO, run_tool, write

# ------------------------------------------------------------------ retired public names (R-A12)

PART4 = """# Part IV
## 21. Review
Nothing retired here.
## 22. Decision log
| R-A02 | `norm` | `norm_batch_mean` becomes `ref_batch_mean` |
## 23. Points to settle
"""

MATHS = """# maths
### 1.7 Scattering statistics — vanilla ScatCov
- [1.7 Scattering statistics — vanilla ScatCov](#17-scattering-statistics--vanilla-scatcov)
- **Source:** `Base_DataClass`, `ST_Operator.vanilla`.
`norm="self"` (STL's `store_ref`, `load_ref` and `vanilla`); STL's `to_angular_ft`.
"""

PUBLIC = '''
class Data:
    """Data of every geometry."""

    def copy(self):
        """A copy."""


def _private():
    """STL's DataClass and store_ref may be named here."""
'''

PART2 = """# Part II
| T4.1 | P4 | `SpectrumOperator`, built as `STOperator` |
| T4.6 | P4 | The spectrum within `STOperator` only; `SpectrumOperator` removed |
"""

GOOD = {
    "docs/transfer/part4.md": PART4,
    "docs/transfer/part2.md": PART2,
    "docs/maths.md": MATHS,
    "AGENTS.md": '# AGENTS\nUse `norm="self"`.\n',
    "src/estel/__init__.py": "from ._data import Data\n",
    "src/estel/_data.py": PUBLIC,
}


@pytest.mark.slow
def test_repository_passes() -> None:
    """Slow: the tool reads every file of the repository (about 2 s); `make check-tools` runs it
    on every pull request anyway."""
    result = run_tool("check_retired_names.py", "--root", str(REPO))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_allowed_uses_pass(tmp_path: Path) -> None:
    write(tmp_path, GOOD)
    assert check(tmp_path) == []


@pytest.mark.parametrize(
    ("change", "where", "found"),
    [
        pytest.param(
            {"docs/maths.md": MATHS + 'With `norm="vanilla"`.\n'},
            "docs/maths.md:6",
            "vanilla",
            id="maths-body",
        ),
        pytest.param(
            {"docs/transfer/part2.md": "Use angular\\_ft.\n"},
            "docs/transfer/part2.md:1",
            "angular\\_ft",
            id="escaped-underscores",
        ),
        pytest.param(
            {"docs/transfer/part4.md": PART4 + "Uses ScatteringPlan.\n"},
            "docs/transfer/part4.md:7",
            "ScatteringPlan",
            id="part4-after-section-22",
        ),
        pytest.param(
            {"AGENTS.md": "STL's `store_ref` is not attributed outside docs/.\n"},
            "AGENTS.md:1",
            "store_ref",
            id="agents-attribution",
        ),
        pytest.param(
            {"src/estel/_data.py": PUBLIC.replace("A copy.", "As STL's `load_ref`.")},
            "src/estel/_data.py:5 (docstring of estel.Data.copy)",
            "load_ref",
            id="public-docstring",
        ),
        pytest.param(
            {"docs/transfer/part2.md": PART2 + "| T5.1 | P5 | Uses `SpectrumOperator` |\n"},
            "docs/transfer/part2.md:4",
            "SpectrumOperator",
            id="spectrum-operator-in-another-row",
        ),
    ],
)
def test_retired_name_fails(tmp_path: Path, change: dict[str, str], where: str, found: str) -> None:
    write(tmp_path, {**GOOD, **change})
    hits = check(tmp_path)
    assert [(h.where, h.found) for h in hits] == [(where, found)]
    assert "Use `" in hits[0].message()


# ------------------------------------------------------------------ the rule of names (§2.4.1)

BASE = {
    "src/estel/__init__.py": "",
    "src/estel/data/__init__.py": "",
    "src/estel/data/_healpix.py": "def stencils_healpix():\n    return 0\n",
    "src/estel/scattering/__init__.py": "",
    "src/estel/scattering/_loop.py": "from estel.data._healpix import stencils_healpix\n",
    "docs/guide.md": "Cartesian grids, single-domain data; the benchmark case `planar-periodic`.\n",
    "changes/1.patch.md": "History: the planar data of `_loop_tp`.\n",
    "pyproject.toml": (
        '[project]\ndependencies = ["numpy>=2"]\n'
        '[tool.importlinter]\nroot_packages = ["estel"]\n'
        '[[tool.importlinter.contracts]]\nid = "c"\ntype = "forbidden"\n'
        'source_modules = ["estel.scattering"]\nforbidden_modules = ["numpy", "estel_rust"]\n'
    ),
}


def repository(root: Path, files: Mapping[str, str | None]) -> None:
    """Write ``files`` (None: left out) under ``root`` and track them in a new git repository."""
    write(root, {rel: text for rel, text in files.items() if text is not None})
    for command in (["git", "init", "-q"], ["git", "add", "-A"]):
        subprocess.run(command, cwd=root, check=True, capture_output=True)


def names(root: Path, files: Mapping[str, str | None], **options: bool) -> tuple[int, str]:
    repository(root, {**BASE, **files})
    status, lines = run(root, **options)
    return status, "\n".join(lines)


def table(*rows: str) -> dict[str, str]:
    return {"tools/renamed_names.toml": "row = [\n" + ",\n".join(rows) + "\n]\n"}


def row(old: str, new: str, task: str = "T7.6", *, done: bool = False, kind: str = "rename") -> str:
    flag = "true" if done else "false"
    return f'{{ old = "{old}", new = "{new}", kind = "{kind}", task = "{task}", done = {flag} }}'


def test_names_that_follow_the_rule_pass(tmp_path: Path) -> None:
    status, output = names(tmp_path, {})
    assert status == 0, output


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {"src/estel/data/_cartesian.py": "def healpix_stencils():\n    pass\n"},
            "rule 1",
            id="rule1-order",
        ),
        pytest.param(
            {"src/estel/data/_cartesian.py": "class DataHEALPix:\n    pass\n"},
            "rule 2",
            id="rule2-form-class",
        ),
        pytest.param(
            {"src/estel/data/_cartesian.py": "def conv_1D():\n    pass\n"},
            "rule 2",
            id="rule2-form-function",
        ),
        pytest.param(
            {"src/estel/scattering/_values.py": "class X:\n    scales_tp = 1\n"},
            "rule 2",
            id="rule2-retired-field",
        ),
        pytest.param({"docs/guide.md": "Call `spec_hpx` here.\n"}, "rule 2", id="rule2-in-text"),
        pytest.param(
            {"tests/data/test_x.py": "def test_planar_data_are_refused():\n    pass\n"},
            "rule 10",
            id="rule10-test-name",
        ),
        pytest.param(
            {"src/estel/wavelets/_healpix_bank.py": "X = 1\n"}, "rule 1", id="rule1-module-path"
        ),
        pytest.param(
            {
                "src/estel/data/_healpix.py": "def stencils():\n    return 0\n",
                "src/estel/scattering/_loop.py": "from estel.data._healpix import stencils\n",
            },
            "rule 3",
            id="rule3-import",
        ),
        pytest.param(
            {
                "src/estel/data/_healpix.py": "def stencils():\n    return 0\n",
                "src/estel/scattering/_loop.py": "from estel.data import _healpix\n"
                "x = _healpix.stencils()\n",
            },
            "rule 3",
            id="rule3-through-the-module",
        ),
        pytest.param({"docs/guide.md": "See M-CONV-HPX.\n"}, "rule 9", id="rule9-retired"),
        pytest.param({"docs/guide.md": "See M-LP2.\n"}, "rule 9", id="rule9-digits"),
        pytest.param({"docs/guide.md": "See M-TENSORPROD.\n"}, "rule 9", id="rule9-geometry-alone"),
        pytest.param(
            {"docs/guide.md": "See M-MASK-TENSORPROD-BLOCK.\n"}, "rule 9", id="rule9-order"
        ),
        pytest.param({"docs/guide.md": "On planar data.\n"}, "rule 10", id="rule10-prose"),
        pytest.param({"docs/guide.md": "A single geometry.\n"}, "rule 10", id="rule10-phrase"),
        pytest.param(
            {"docs/guide.md": "Two plain geometries.\n"}, "rule 10", id="rule10-phrase-plural"
        ),
        pytest.param({"tests/x.py": 'CASES = ["planar-wide"]\n'}, "rule 10", id="rule10-label"),
        pytest.param(
            {
                "src/estel/wavelets/_pixels_healpix.py": "def nest2pix():\n    pass\n",
                "src/estel/scattering/_loop.py": (
                    "from estel.wavelets._pixels_healpix import nest2pix\n"
                ),
            },
            "rule 3",
            id="rule3-not-a-convention-name",
        ),
        pytest.param(
            {"tools/stl_parity/cases/c.py": "HPX_SIZE = 4\n"}, "rule 2", id="parity-case-constant"
        ),
        pytest.param(
            {"tools/prove_unchanged.py": "def case_2d_mask():\n    pass\n"},
            "rule 1",
            id="the-instruments-follow-the-rule",
        ),
    ],
)
def test_a_violation_with_no_row_fails(
    tmp_path: Path, files: dict[str, str], expected: str
) -> None:
    status, output = names(tmp_path, files)
    assert status == 1
    assert expected in output
    assert "needs a row" in output


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(
            {"tests/data/test_x.py": "def test_healpix_data_are_refused():\n    pass\n"},
            id="test-names-are-sentences",
        ),
        pytest.param(
            {
                "tests/data/test_healpix.py": "from estel.data._healpix import stencils\n",
                "src/estel/data/_healpix.py": "def stencils():\n    return 0\n",
            },
            id="rule3-inside-its-geometry",
        ),
        pytest.param({"docs/guide.md": "A planar grid (STL's word).\n"}, id="rule10-stl-quote"),
        pytest.param(
            {"docs/guide.md": "See M-S1, M-W1-TENSORPROD, M-DATA-HEALPIX.\n"}, id="rule9-good-ids"
        ),
        pytest.param(
            {"docs/guide.md": "| `M-…-TENSORPROD` (`docs/maths.md` §3) | Tensor products |\n"},
            id="rule9-a-range-of-ids",
        ),
        pytest.param({"tools/x.py": 'HEALPIX_ANALYSE_VERSION = "1"\n'}, id="rule7-package"),
        pytest.param(
            {
                "src/estel/wavelets/_pixels_healpix.py": "def nest2xyf():\n    pass\n",
                "src/estel/scattering/_loop.py": (
                    "from estel.wavelets._pixels_healpix import nest2xyf\n"
                ),
            },
            id="rule7-healpix-convention",
        ),
        pytest.param(
            {
                "src/estel/data/_healpix.py": "def stencils():\n    return 0\n",
                "tests/scattering/test_loop.py": "from estel.data._healpix import stencils\n",
                "bench/cases.py": "from estel.data import _healpix\nx = _healpix.stencils()\n",
            },
            id="rule3-package-only",
        ),
        pytest.param(
            {
                "tools/stl_parity/cases/c.py": (
                    "STL_ENV = 1\nPARAMETERS = {}\n\n\ndef run():\n    pass\n"
                )
            },
            id="rule7-parity-interface",
        ),
        pytest.param(
            {
                "tests/conftest.py": "import pytest\n\n\n@pytest.fixture\n"
                "def healpix_backend():\n    pass\n"
            },
            id="rule3-fixture",
        ),
    ],
)
def test_allowed_names_pass(tmp_path: Path, files: dict[str, str]) -> None:
    status, output = names(tmp_path, files)
    assert status == 0, output


def test_a_row_not_done_is_reported(tmp_path: Path) -> None:
    files = {"src/estel/data/_cartesian.py": "def healpix_stencils():\n    pass\n"}
    rows = table(row("estel.data._cartesian:healpix_stencils", "stencils_healpix", "T7.4"))
    status, output = names(tmp_path, {**files, **rows})
    assert status == 0, output
    assert "T7.4: 1 rows to rename, 1 occurrences reported" in output
    _, report = names(tmp_path, {**files, **rows}, report=True)
    assert "src/estel/data/_cartesian.py:1" in report


def test_a_row_done_fails_while_its_old_name_occurs(tmp_path: Path) -> None:
    files = {
        "src/estel/data/_cartesian.py": "def stencils_healpix():\n    pass\n",
        "docs/guide.md": "Call `healpix_stencils`.\n",
    }
    rows = table(row("estel.data._cartesian:healpix_stencils", "stencils_healpix", done=True))
    status, output = names(tmp_path, {**files, **rows})
    assert status == 1
    assert "docs/guide.md:1" in output
    assert "its row is done" in output
    status, output = names(
        tmp_path, {**files, "docs/guide.md": "Call `stencils_healpix`.\n", **rows}
    )
    assert status == 0, output


def test_a_name_comes_back_only_after_its_rename_row_is_done(tmp_path: Path) -> None:
    files = {"src/estel/wavelets/_operator.py": "class WaveletPlan:\n    pass\n"}
    rename = row(
        "estel.wavelets._operator:WaveletPlan", "WaveletPlanSingleDomain", "T7.5", done=True
    )
    back = row("estel.wavelets._operator:WaveletPlan", "root", "T7.5", kind="reintroduce")
    status, output = names(tmp_path, {**files, **table(rename, back)})
    assert status == 1  # the old name occurs: its reintroduction is not done
    done_back = back.replace("done = false", "done = true")
    status, output = names(tmp_path, {**files, **table(rename, done_back)})
    assert status == 0, output
    early = rename.replace("done = true", "done = false")
    status, output = names(tmp_path, {**files, **table(early, done_back)})
    assert status == 1
    assert "before its rename row is done" in output


def test_a_prose_row_follows_its_renamed_file(tmp_path: Path) -> None:
    files = {"src/estel/data/_cartesian.py": '"""Planar data."""\n'}
    rows = table(
        row("estel.data._planar", "estel.data._cartesian", "T7.4", done=True),
        row("src/estel/data/_planar.py#planar", "by hand", "T7.11"),
    )
    status, output = names(tmp_path, {**files, **rows})
    assert status == 0, output


def test_strict_fails_while_a_row_is_not_done(tmp_path: Path) -> None:
    rows = table(row("estel.data._healpix:x", "y"))
    assert names(tmp_path, rows)[0] == 0
    status, output = names(tmp_path, rows, strict=True)
    assert status == 1
    assert "not done" in output


@pytest.mark.parametrize(
    ("module", "fails"),
    [
        pytest.param("estel.scattering", False, id="exists"),
        pytest.param("estel_rust", False, id="reserved"),
        pytest.param("estel.statistics", True, id="lost"),
        pytest.param("polars", True, id="no-dependency"),
    ],
)
def test_the_contracts_lose_no_module(tmp_path: Path, module: str, fails: bool) -> None:
    pyproject = BASE["pyproject.toml"].replace('"estel_rust"', f'"{module}"')
    status, output = names(tmp_path, {"pyproject.toml": pyproject})
    assert (status == 1) == fails, output
    if fails:
        assert module in output


COLLISIONS = {
    # The oracle's own Scale, a _data of the benchmark, attributes _data and _buffer: other
    # definitions than those of the rows done below.
    "src/estel/_spec/tensor_prod.py": "Scale = tuple[int, ...]\n",
    "bench/_estel.py": "_data = 1\n",
    "src/estel/wavelets/_operator.py": (
        "class W:\n    def __init__(self):\n        self._data = 1\n"
    ),
    "src/estel/statistics/_statistics.py": "class STStatistics:\n    _buffer: int = 0\n",
    "src/estel/scattering/_tensor_prod.py": "class ScaleTensorProd:\n    pass\n",
    "tests/spectra/test_stoperator.py": "def _data_2d():\n    pass\n",
    "docs/guide.md": "A `Scale` of the oracle; the `_data` of `W`; `STStatistics._buffer`.\n",
}
COLLISION_ROWS = table(
    row("estel.scattering._tensor_prod:Scale", "ScaleTensorProd", "T7.7", done=True),
    row("tests.spectra.test_stoperator:_data", "_data_2d", "T7.9", done=True),
    row("tests.statistics.test_tensor_prod:_buffer", "_buffer_tensor_prod", "T7.8", done=True),
)


def test_a_row_done_ignores_other_definitions_of_its_name(tmp_path: Path) -> None:
    status, output = names(tmp_path, {**COLLISIONS, **COLLISION_ROWS})
    assert status == 0, output


@pytest.mark.parametrize(
    ("change", "where"),
    [
        pytest.param(
            {"src/estel/scattering/_tensor_prod.py": "class Scale:\n    pass\n"},
            "src/estel/scattering/_tensor_prod.py:1",
            id="defined-in-its-module",
        ),
        pytest.param(
            {
                "src/estel/scattering/_filters.py": (
                    "from estel.scattering._tensor_prod import Scale\n"
                )
            },
            "src/estel/scattering/_filters.py:1",
            id="imported-from-its-module",
        ),
        pytest.param(
            {
                "src/estel/scattering/_filters.py": (
                    "from . import _tensor_prod\nx = _tensor_prod.Scale\n"
                )
            },
            "src/estel/scattering/_filters.py:2",
            id="read-through-its-module",
        ),
        pytest.param(
            {"docs/other.md": "See _tensor_prod.Scale.\n"},
            "docs/other.md:1",
            id="qualified-in-prose",
        ),
        pytest.param(
            {"tests/spectra/test_stoperator.py": "def _data():\n    pass\n"},
            "tests/spectra/test_stoperator.py:1",
            id="a-test-helper",
        ),
    ],
)
def test_a_row_done_fails_on_its_own_old_name(
    tmp_path: Path, change: dict[str, str], where: str
) -> None:
    status, output = names(tmp_path, {**COLLISIONS, **change, **COLLISION_ROWS})
    assert status == 1
    assert f"{where}: " in output
    assert "its row is done" in output


@pytest.mark.parametrize(
    ("prose", "status"),
    [
        pytest.param(
            "STL's pyramid engine adds `build_plan` to `ST_Operator`.\n", 0, id="stl-bare"
        ),
        pytest.param("See `_plan.build_plan`.\n", 1, id="estel-qualified"),
    ],
)
def test_a_row_done_lets_stl_names_be_cited_bare(tmp_path: Path, prose: str, status: int) -> None:
    """STL's own ``build_plan`` (``STL_NAMES``) keeps its bare name in the text once ESTEL's
    ``build_plan`` is renamed; the qualified forms of ESTEL's name still fail."""
    files = {
        "src/estel/scattering/_plan.py": "def build_plan_single_domain():\n    pass\n",
        "docs/other.md": prose,
        **table(
            row("estel.scattering._plan:build_plan", "build_plan_single_domain", "T7.7", done=True)
        ),
    }
    found, output = names(tmp_path, files)
    assert found == status, output


MODULE_ROW = table(
    row("estel.wavelets._tensor_prod", "estel.wavelets._operator_tensor_prod", done=True)
)


@pytest.mark.parametrize(
    ("files", "fails"),
    [
        pytest.param({"docs/guide.md": "See `_tensor_prod.py`.\n"}, True, id="its-file-name-alone"),
        pytest.param(
            {
                "src/estel/data/_tensor_prod.py": "X = 1\n",
                "docs/guide.md": "See `_tensor_prod.py`.\n",
            },
            False,
            id="the-file-name-of-another-module",
        ),
        pytest.param(
            {
                "src/estel/data/_tensor_prod.py": "X = 1\n",
                "docs/guide.md": "See `wavelets/_tensor_prod.py`.\n",
            },
            True,
            id="its-qualified-path",
        ),
        pytest.param(
            {
                "src/estel/data/_tensor_prod.py": "X = 1\n",
                "src/estel/wavelets/_operator.py": '"""The blocks, in ``_tensor_prod.py``."""\n',
            },
            True,
            id="its-file-name-in-its-own-package",
        ),
    ],
)
def test_a_module_row_done_leaves_the_file_name_of_another_module(
    tmp_path: Path, files: dict[str, str], fails: bool
) -> None:
    """A renamed module's file name alone, `_tensor_prod.py`, may cite another module of that
    name, here `data/_tensor_prod.py`: it fails while no other module has that name, and always
    in the files of its own package, whose siblings cite it that way."""
    module = {"src/estel/wavelets/_operator_tensor_prod.py": "Y = 1\n"}
    status, output = names(tmp_path, {**module, **files, **MODULE_ROW})
    assert status == int(fails), output
    if fails:
        assert "docs/guide.md:1" in output or "src/estel/wavelets/_operator.py:1" in output


@pytest.mark.parametrize(
    ("prose", "fails"),
    [
        pytest.param("The benchmark case `planar-masked`.\n", False, id="bench-label"),
        pytest.param("A `planar` grid (STL's word).\n", False, id="stl-quote"),
        pytest.param("A `planar` grid.\n", True, id="neither"),
    ],
)
def test_a_row_done_honours_the_exemptions_of_the_rule(
    tmp_path: Path, prose: str, fails: bool
) -> None:
    """The bare name of a module row done, `planar`, follows the exemptions of the main check
    in prose: a benchmark label of step 1 (rule 7) and a quote of STL followed by "(STL's
    word)" (rule 10) may still say it."""
    files = {
        "src/estel/_spec/cartesian.py": "X = 1\n",
        "docs/guide.md": prose,
        **table(row("estel._spec.planar", "estel._spec.cartesian", "T7.3", done=True)),
    }
    status, output = names(tmp_path, files)
    assert status == int(fails), output
    assert ("docs/guide.md:1: " in output) == fails, output


def test_prose_names_that_are_no_words_are_matched_outside_backticks(tmp_path: Path) -> None:
    files = {
        "src/estel/wavelets/_operator.py": "class WaveletPlanSingleDomain:\n    pass\n",
        "docs/guide.md": "The WaveletPlan holds.\n",
    }
    rename = row(
        "estel.wavelets._operator:WaveletPlan", "WaveletPlanSingleDomain", "T7.5", done=True
    )
    status, output = names(tmp_path, {**files, **table(rename)})
    assert status == 1
    assert "docs/guide.md:1: `WaveletPlan`" in output


def test_a_reintroduced_name_is_defined_in_its_module_only(tmp_path: Path) -> None:
    rows = table(
        row("estel.wavelets._operator:WaveletPlan", "WaveletPlanSingleDomain", "T7.5", done=True),
        row("estel.wavelets._operator:WaveletPlan", "root", "T7.5", done=True, kind="reintroduce"),
    )
    files = {
        "src/estel/wavelets/_operator.py": "class WaveletPlan:\n    pass\n",
        "src/estel/scattering/_plan.py": "from estel.wavelets._operator import WaveletPlan\n",
    }
    assert names(tmp_path, {**files, **rows})[0] == 0
    elsewhere = {**files, "src/estel/data/_base.py": "class WaveletPlan:\n    pass\n"}
    status, output = names(tmp_path / "elsewhere", {**elsewhere, **rows})
    assert status == 1
    assert "src/estel/data/_base.py:1" in output
    assert "brings it back in estel.wavelets._operator only" in output
