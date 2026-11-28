"""tools/check_docstrings.py checks the numpydoc docstrings of the public names (rule D1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from check_docstrings import check

from .conftest import REPO, run_tool, write

CLEAN = '''
def scale(x, factor=2.0, *, axes, **options):
    """Scale ``x`` (maths.md M-X).

    Parameters
    ----------
    x : array_like
        The data.
    factor : float
        The factor.
    axes, options
        The axes, a tuple of int; the options, passed on.

    Returns
    -------
    array_like
        The scaled data.
    """


class Box:
    """A box.

    Parameters
    ----------
    size : int
        Its size.
    """

    def __init__(self, size):
        self.size = size

    def grow(self, by):
        """Grow the box.

        Parameters
        ----------
        by : int
            How much.
        """
'''


def _codes(tmp_path: Path, source: str) -> list[tuple[str, str]]:
    """The (name, code) findings on a package whose public names are defined by ``source``."""
    write(
        tmp_path,
        {
            "src/estel/__init__.py": "from ._mod import Box, f\n",
            "src/estel/_mod.py": source,
        },
    )
    return [(f.name, f.code) for f in check(tmp_path)]


def _function(doc: str, signature: str = "x, factor") -> str:
    return f'class Box:\n    """A box."""\n\n\ndef f({signature}):\n    """{doc}"""\n'


def test_clean_case_passes(tmp_path: Path) -> None:
    write(tmp_path, {"src/estel/__init__.py": "from ._mod import scale, Box\n"})
    write(tmp_path, {"src/estel/_mod.py": CLEAN})
    assert check(tmp_path) == []


def test_internal_names_are_not_checked(tmp_path: Path) -> None:
    source = _function("Scale.\n\n    Parameters\n    ----------\n    x : int\n    factor : int\n")
    assert _codes(tmp_path, source + "\n\ndef _internal(y):\n    pass\n") == []


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        pytest.param("", [("estel.f", "GL08")], id="missing"),
        pytest.param(
            "Scale.\n\n    Parameters\n    ----------\n    x : int\n",
            [("estel.f", "PR01")],
            id="not-documented",
        ),
        pytest.param(
            "Scale.\n\n    Parameters\n    ----------\n    x : int\n    factor : int\n"
            "    y : int\n",
            [("estel.f", "PR02")],
            id="documented-not-in-signature",
        ),
        pytest.param(
            "Scale.\n\n    Parameters\n    ----------\n    x: int\n    factor : int\n",
            [("estel.f", "PR10")],
            id="colon-without-space",
        ),
        pytest.param(
            "Scale.\n\n    Parameters\n    ----------\n    x\n    factor : int\n",
            [("estel.f", "PR04")],
            id="no-type",
        ),
        pytest.param(
            "Parameters\n    ----------\n    x, factor : int\n",
            [("estel.f", "SS01")],
            id="no-summary",
        ),
        pytest.param(
            "Scale.\n\n    Parameters\n    ----------\n    x, factor : int\n\n"
            "    Notes\n    -----\n    N.\n\n    Returns\n    -------\n    int\n",
            [("estel.f", "GL07")],
            id="order",
        ),
        pytest.param(
            "Scale.\n\n    Parameters\n    ----------\n    x, factor : int\n\n"
            "    Remarks\n    -------\n    R.\n",
            [("estel.f", "GL06")],
            id="unknown-section",
        ),
    ],
)
def test_defects_are_found(tmp_path: Path, doc: str, expected: list[tuple[str, str]]) -> None:
    assert _codes(tmp_path, _function(doc)) == expected


def test_message_says_where_and_what_to_do(tmp_path: Path) -> None:
    write(tmp_path, {"src/estel/__init__.py": "from ._mod import f, Box\n"})
    write(tmp_path, {"src/estel/_mod.py": _function("Scale.\n")})
    result = run_tool("check_docstrings.py", "--root", str(tmp_path))
    assert result.returncode == 1
    assert "src/estel/_mod.py:5: PR01 estel.f: parameters not documented: x, factor." in (
        result.stderr
    )
    assert '"Parameters" section, as "name : type"' in result.stderr


def test_repository_passes() -> None:
    result = run_tool("check_docstrings.py", "--root", str(REPO))
    assert result.returncode == 0, result.stderr
