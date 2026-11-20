"""tools/register.py fails on an invalid card or a stale view; it writes the view of the cards."""

from __future__ import annotations

from pathlib import Path

import pytest

from register import load, main, view

from .conftest import REPO, run_tool, write


def card(ident: str, **fields: str) -> str:
    """A valid card of step 2a, with ``fields`` replacing lines of its front matter."""
    lines = {
        "id": f'"{ident}"',
        "title": '"Centred covariances"',
        "category": '"Mathematical choice"',
        "step": '"2a"',
        "phase": '"2a.1"',
        "status": '"decision"',
        "decide": '["centring method in float32"]',
        "depends": "[]",
        "changes_values": '"yes"',
        "issue": "0",
    }
    lines.update(fields)
    front = "\n".join(f"{key} = {value}" for key, value in lines.items() if value)
    return f"+++\n{front}\n+++\n\n## Feature\n\nThe text of the entry.\n"


GOOD = {
    "docs/register/LF6.md": card("LF6"),
    "docs/register/LF33.md": card(
        "LF33", status='"ready"', decide="[]", depends='["LF6"]', issue="12"
    ),
    "docs/register/W4.md": card(
        "W4", category='"Work that is not a feature"', step='"3"', phase='""', status='"done"'
    ),
    "docs/register/decisions.md": "| Date | ID | Point | Decision | Where |\n",
}


def tree(root: Path, change: dict[str, str] | None = None) -> Path:
    """The cards of ``GOOD`` with ``change``, and their view written by the tool."""
    write(root, GOOD)
    assert main(["--root", str(root), "--write"]) == 0
    write(root, change or {})
    return root


def test_repository_passes() -> None:
    result = run_tool("register.py", "--root", str(REPO), "--check")
    assert result.returncode == 0, result.stderr


def test_view_of_good_tree(tmp_path: Path) -> None:
    root = tree(tmp_path)
    assert main(["--root", str(root), "--check"]) == 0
    text = (root / "docs/register/README.md").read_text()
    assert "| [LF6](LF6.md) | Centred covariances | 2a.1 | centring method in float32 |" in text
    assert "| [LF33](LF33.md) | Centred covariances | 2a.1 | ready | LF6 | #12 |" in text
    assert "## Done\n\n1 entry.\n\n| ID | Title | Step | Phase | Issue |" in text
    assert view(load(root)[0]) == text  # deterministic


def test_terminal_view_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tree(tmp_path)
    capsys.readouterr()  # the message of --write
    assert main(["--root", str(root), "--step", "2a", "--human"]) == 0
    out = capsys.readouterr().out
    assert [line.split()[0] for line in out.splitlines()] == ["LF6", "to", "1"]
    assert out.endswith("1 card (step 2a, needs a human).\n")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"docs/register/README.md": "edited by hand\n"},
            "docs/register/README.md is out of date with the cards: run `make register`",
            id="stale-view",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", title='"Low-pass filters everywhere"')},
            "docs/register/README.md is out of date with the cards: run `make register`",
            id="card-changed",
        ),
        pytest.param(
            {"docs/register/LF6.md": "## Feature\n"},
            "docs/register/LF6.md has no front matter",
            id="no-front-matter",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", step="2a")},
            "docs/register/LF6.md: its front matter is not valid TOML",
            id="invalid-toml",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", issue="")},
            "docs/register/LF6.md lacks the fields issue",
            id="missing-field",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", issue='"#12"')},
            "docs/register/LF6.md: `issue` must be an integer",
            id="wrong-type",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", id='"LF7"')},
            "`id` is 'LF7', but the file is LF6.md",
            id="id-not-file-name",
        ),
        pytest.param(
            {"docs/register/notes.md": "Notes.\n"},
            "docs/register/notes.md is not a card",
            id="not-a-card",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", category='"Feature"')},
            "`category` is 'Feature', not one of Structure",
            id="category",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", step='"2"', phase='""')},
            "`step` is '2', not one of queue, 2a, 2b, 3",
            id="step",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", status='"blocked"')},
            "`status` is 'blocked', not one of decision",
            id="status",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", phase='"2a.6"')},
            "a card of step 2a has a `phase`, one of 2a.0",
            id="phase-unknown",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", step='"2b"')},
            "`phase` is '2a.1', but only cards of step 2a have a phase",
            id="phase-outside-2a",
        ),
        pytest.param(
            {
                "docs/register/LF6.md": card(
                    "LF6", title='"one two three four five six seven eight nine"'
                )
            },
            "the title has 9 words",
            id="title-too-long",
        ),
        pytest.param(
            {"docs/register/LF6.md": card("LF6", depends='["LF34"]')},
            "docs/register/LF6.md: `depends` names 'LF34', which is not another card",
            id="depends-unknown",
        ),
    ],
)
def test_problem_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, str], message: str
) -> None:
    root = tree(tmp_path, change)
    assert main(["--root", str(root), "--check"]) == 1
    errors = capsys.readouterr().err.strip().splitlines()
    assert len(errors) == 1, errors
    assert message in errors[0]
