"""tools/check_orientations.py fails on L above 4 in tests, benchmark and parity cases."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from check_orientations import check, check_python, check_toml

from .conftest import REPO, run_tool, write


def test_repository_passes() -> None:
    result = run_tool("check_orientations.py", "--root", str(REPO))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("op = STOperator(data, J=3, L=4)\n", id="keyword"),
        pytest.param('PARAMETERS = {"J": 5, "L": 4}\n', id="dictionary"),
        pytest.param('@pytest.mark.parametrize("L", [1, 2, 3, 4])\ndef f(L): ...\n', id="param"),
        pytest.param("op = STOperator(data_1d, L=2)\n", id="1d"),
        pytest.param("op = STOperator(tensor_prod, J=[3, 2], L=[4, 2])\n", id="per-block"),
        pytest.param("op = STOperator(data, L=n_orientations)\n", id="variable"),
        pytest.param('@pytest.mark.parametrize("n", [8, 16])\ndef f(n): ...\n', id="other-name"),
        pytest.param('x = {"N": 64, "K": 8}\nf(J=6)\n', id="other-keys"),
        pytest.param("op = STOperator(data, L=8)  # allow-large-L: #133\n", id="marker"),
        pytest.param("J, L = 5, 4\nL: int = 4\nL = [4, 2]\n", id="assignments"),
        pytest.param("L = n_orientations\nJ, L = sizes\nop.L = 8\n", id="assigned-variable"),
        pytest.param("J, L = 5, 8  # allow-large-L: #133\n", id="assignment-marker"),
    ],
)
def test_allowed_forms_pass(source: str) -> None:
    assert check_python(source, "tests/t.py") == []


@pytest.mark.parametrize(
    ("source", "line", "found"),
    [
        pytest.param("op = STOperator(data, L=8)\n", 1, "keyword argument L=8", id="keyword"),
        pytest.param('P = {"J": 5,\n     "L": 6}\n', 2, 'dictionary key "L": 6', id="dictionary"),
        pytest.param(
            '@pytest.mark.parametrize("L", [1, 3, 6])\ndef f(L): ...\n',
            1,
            "parametrize entry L = 6",
            id="param",
        ),
        pytest.param(
            '@pytest.mark.parametrize(("n", "L"), [(16, 4),\n    (32, 8)])\ndef f(n, L): ...\n',
            2,
            "parametrize entry L = 8",
            id="param-among-names",
        ),
        pytest.param(
            '@pytest.mark.parametrize("n, L", [pytest.param(16, 5, id="a")])\ndef f(n, L): ...\n',
            1,
            "parametrize entry L = 5",
            id="param-pytest-param",
        ),
        pytest.param(
            "op = STOperator(tensor_prod, J=[3, 2], L=[8, 2])\n",
            1,
            "keyword argument L=[8, 2]",
            id="per-block",
        ),
        pytest.param("L = 8\n", 1, "assignment L = 8", id="assignment"),
        pytest.param("L: int = 6\n", 1, "assignment L = 6", id="annotated"),
        pytest.param("x = 1\nJ, L = 5, 8\n", 2, "assignment L = 8", id="unpacking"),
        pytest.param("L = [4, 8]\n", 1, "assignment L = [4, 8]", id="assignment-per-block"),
        pytest.param("n = L = 5\n", 1, "assignment L = 5", id="chained"),
        pytest.param(
            "op = STOperator(data, L=8)  # allow-large-L\n",
            1,
            "keyword argument L=8",
            id="marker-without-issue",
        ),
    ],
)
def test_large_L_fails(source: str, line: int, found: str) -> None:  # noqa: N802
    hits = [h for h in check_python(source, "tests/t.py") if h.found != "allow-large-L"]
    assert [(h.where, h.found) for h in hits] == [(f"tests/t.py:{line}", found)]
    assert "allow-large-L: #<issue>" in hits[0].message()


def test_marker_without_issue_fails() -> None:
    hits = check_python("x = 1  # allow-large-L: 133\n", "tests/t.py")
    assert [(h.where, h.found) for h in hits] == [("tests/t.py:1", "allow-large-L")]
    assert "names the issue" in hits[0].message()


@pytest.mark.parametrize(
    ("source", "hits"),
    [
        pytest.param("[cases.a]\nJ = 7\nL = 4\n", [], id="default"),
        pytest.param("[cases.a]\nL = 8  # allow-large-L: #133\n", [], id="marker"),
        pytest.param('[cases.a]\nnote = "L = 8"\n', [], id="in-string"),
        pytest.param("[cases.a]\nL = 8\n", [("b.toml:2", "TOML field L = 8")], id="field"),
        pytest.param('a = { "L" = 6 }\n', [("b.toml:1", "TOML field L = 6")], id="inline"),
        pytest.param("L = [4, 8]\n", [("b.toml:1", "TOML field L = [4, 8]")], id="per-block"),
        pytest.param("L = 8  # allow-large-L\n", [("b.toml:1", "allow-large-L")], id="no-issue"),
    ],
)
def test_toml(source: str, hits: list[tuple[str, str]]) -> None:
    assert [(h.where, h.found) for h in check_toml(source, "b.toml")] == hits


def test_files_read(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "tests/a/test_x.py": "f(L=8)\n",
            "bench/cases.toml": "L = 8\n",
            "tools/stl_parity/cases/c.py": 'P = {"L": 8}\n',
            "tools/other.py": "f(L=8)\n",
            "src/estel/x.py": "f(L=8)\n",
        },
    )
    assert sorted(h.where for h in check(tmp_path)) == [
        "bench/cases.toml:1",
        "tests/a/test_x.py:1",
        "tools/stl_parity/cases/c.py:1",
    ]


def test_only_tracked_files_are_read(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "tests/test_x.py": "f(L=8)\n",
            "bench/results/.work/copy.py": "f(L=8)\n",  # a local copy, not tracked
        },
    )
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "tests/test_x.py"], check=True)
    assert [h.where for h in check(tmp_path)] == ["tests/test_x.py:1"]
