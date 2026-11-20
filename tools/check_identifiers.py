"""Check that every identifier of the lists of the specification that is cited exists (issue #107).

The families checked, and the list that defines each, by a table row that starts with it:

* the open points ``OP-n``: the tables of ``docs/maths.md``;
* the register ``LFn`` and ``Wn``: the cards of ``docs/register/``, one file ``<ID>.md`` per
  entry (``tools/register.py`` checks that each card's ``id`` is its file name);
* the risks ``Fn``: the table of Part IV, §21;
* the tasks ``Tn.m``: the table of Part II, §15.2 (``docs/transfer/part2.md``);
* the remarks ``R-Xnn`` of the review of 16 November 2026, the old open points ``Gn`` and the
  old points to settle ``Qn`` (Part IV, §23 until 19 November 2026): no list; only the decision
  log (Part IV, §22).

The decision logs: that of the specification (Part IV, §22) and that of step 2
(``docs/register/decisions.md``), with the same columns.

The rule. A cited identifier is accepted if its list has a row (or a card) for it, or if the
column "ID" of a row of a decision log names it: that is where a closed, done or retired item
lives. So removing an item from its list without its row in a log fails. Moreover:

* an identifier is defined once: two rows with the same identifier in a list fail;
* a remark ``R-…`` with no row in the log is accepted only if an issue keeps it
  (``KEPT_IN_ISSUES``).

Citations are searched in every Markdown file tracked by git (``git ls-files``), except those
under ``changes/`` (history), and in the Python files of ``src/``, ``tests/``, ``tools/`` and
``bench/`` (comments and docstrings cite identifiers too). The decision logs themselves (the
block of Part IV, §22, from its heading to the next ``## `` heading, and the whole of
``docs/register/decisions.md``) are not searched: their rows tell the history and may name
retired identifiers.

Not checked, because the same letters mean other things: S1 to S18, C1 to C23, A1 to A12 and
R3 to R13 of the decision log (S1 to S4 are also the statistics; A, B, C, D, E and X with a
number are also the review rules of ``docs/agents/review_rules.md``); and the ``M-``
identifiers, which ``tools/check_maths_ids.py`` checks.

The tool reads text only (and runs ``git ls-files``), uses the standard library only, and never
imports estel.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IdList:
    """A table of the specification whose rows define identifiers, or a directory of cards."""

    name: str  # as the messages name it
    file: str  # a file, or the directory of the cards
    heading: str | None  # the heading of its section, or None for the whole file
    entry: str = "row"  # what defines an identifier: a "row" of the table, or a "card" <ID>.md


OPEN_POINTS = IdList("the open points of docs/maths.md", "docs/maths.md", None)
REGISTER = IdList("the register (docs/register/)", "docs/register", None, entry="card")
RISKS = IdList("the risks (Part IV, §21)", "docs/transfer/part4.md", "## 21. ")
TASKS = IdList("the tasks (Part II, §15.2)", "docs/transfer/part2.md", "### 15.2 ")
LISTS = (OPEN_POINTS, REGISTER, RISKS, TASKS)

DECISION_LOG = IdList("the decision log (Part IV, §22)", "docs/transfer/part4.md", "## 22. ")
STEP2_LOG = IdList("the decision log of step 2", "docs/register/decisions.md", None)
DECISION_LOGS = (DECISION_LOG, STEP2_LOG)
LOGS_NAME = "the decision logs (Part IV, §22; docs/register/decisions.md)"


@dataclass(frozen=True)
class Family:
    """Identifiers written alike, defined in one list (or only in the decision log)."""

    name: str
    pattern: re.Pattern[str]
    defined_in: IdList | None


# The look-behinds keep the letters inside other identifiers out: the W of the maths ID M-W1,
# the F of LF12, the letter after "R-" in a remark; and, for the tasks, a T inside a word or
# after a dot.
FAMILIES = (
    Family("open point", re.compile(r"\bOP-\d+\b"), OPEN_POINTS),
    Family("later feature", re.compile(r"\bLF\d+\b"), REGISTER),
    Family("deferred work", re.compile(r"(?<![\w-])W[1-9]\b"), REGISTER),
    Family("risk", re.compile(r"(?<![\w-])F\d+\b"), RISKS),
    Family("task", re.compile(r"(?<![\w.])T\d+\.\d+\b"), TASKS),
    Family("remark of the review of 16 November 2026", re.compile(r"\bR-[A-Z]\d+\b"), None),
    Family("old open point", re.compile(r"(?<![\w-])G\d+\b"), None),
    Family("old point to settle", re.compile(r"(?<![\w-])Q\d+\b"), None),
)


def _remarks(prefix: str, first: int, last: int) -> tuple[str, ...]:
    return tuple(f"R-{prefix}{n:02d}" for n in range(first, last + 1))


# The remarks of the review of 16 November 2026 that have no row in the decision log because an
# issue keeps them, word for word, with that issue. Only a maintainer changes these sets.
KEPT_IN_ISSUES: dict[str, tuple[str, ...]] = {
    "#71": _remarks("N", 1, 18),  # notation
    "#72": _remarks("V", 1, 10),  # vocabulary of the code
    "#73": ("R-S03", "R-S04", "R-S05", "R-S06", "R-S09"),  # structure of the specification
}
KEPT = {remark: issue for issue, remarks in KEPT_IN_ISSUES.items() for remark in remarks}

ROW_RE = re.compile(r"^\|\s*([^|]*?)\s*\|")
NOT_SEARCHED = ("changes/",)
PYTHON_DIRS = ("src", "tests", "tools", "bench")


class ListNotFoundError(Exception):
    """A list of the specification is not where the tool looks for it."""


class NotACheckoutError(Exception):
    """The root is not a git checkout, so its Markdown files cannot be listed."""


def _section(root: Path, idlist: IdList) -> Iterator[tuple[int, str]]:
    """The numbered lines of ``idlist``'s section (all of its file when it has no heading)."""
    path = root / idlist.file
    if not path.exists():
        raise ListNotFoundError(f"{idlist.file} is missing: it holds {idlist.name}.")
    lines = list(enumerate(path.read_text().splitlines(), 1))
    if idlist.heading is None:
        yield from lines
        return
    level = idlist.heading.split(" ", 1)[0]
    start = next((i for i, (_, line) in enumerate(lines) if line.startswith(idlist.heading)), None)
    if start is None:
        raise ListNotFoundError(
            f"{idlist.file} has no heading starting with {idlist.heading!r}, the section of "
            f"{idlist.name}: if the list moved, update its place in tools/check_identifiers.py."
        )
    for number, line in lines[start + 1 :]:
        heading = line.split(" ", 1)[0]
        if heading and set(heading) == {"#"} and len(heading) <= len(level):
            return
        yield number, line


@dataclass
class Definitions:
    rows: dict[str, tuple[str, int]]  # identifier -> (file, line) of its row in its list
    logged: set[str]  # the identifiers of the column "ID" of the decision logs
    errors: list[str]


def _cards(root: Path, idlist: IdList, families: list[Family]) -> dict[str, tuple[str, int]]:
    """The identifiers of a directory of cards: the names of its files ``<ID>.md``."""
    directory = root / idlist.file
    if not directory.is_dir():
        raise ListNotFoundError(f"{idlist.file}/ is missing: it holds {idlist.name}.")
    return {
        path.stem: (f"{idlist.file}/{path.name}", 1)
        for path in sorted(directory.glob("*.md"))
        if any(f.pattern.fullmatch(path.stem) for f in families)
    }


def definitions(root: Path) -> Definitions:
    rows: dict[str, tuple[str, int]] = {}
    errors = []
    for idlist in LISTS:
        families = [f for f in FAMILIES if f.defined_in is idlist]
        if idlist.entry == "card":
            rows.update(_cards(root, idlist, families))
            continue
        for number, line in _section(root, idlist):
            match = ROW_RE.match(line)
            if not match or not any(f.pattern.fullmatch(match.group(1)) for f in families):
                continue
            ident = match.group(1)
            if ident in rows:
                _, first = rows[ident]
                errors.append(
                    f"{idlist.file}:{number} defines {ident} again in {idlist.name}, already "
                    f"defined at line {first}: an identifier is defined once; give one of the two "
                    "rows a new identifier, or merge them."
                )
            else:
                rows[ident] = (idlist.file, number)
    return Definitions(rows, _logged(root), errors)


def _logged(root: Path) -> set[str]:
    logged: set[str] = set()
    for log in DECISION_LOGS:
        column = None
        for _, line in _section(root, log):
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if column is None:
                if "ID" not in cells:
                    raise ListNotFoundError(
                        f"{log.file}: the first row of {log.name} has no column "
                        '"ID": if the log changed form, update _logged in '
                        "tools/check_identifiers.py."
                    )
                column = cells.index("ID")
            elif len(cells) > column:
                logged.update(ident.strip() for ident in cells[column].split(","))
    return logged


def _cited_files(root: Path) -> list[Path]:
    """The Markdown files tracked by git, outside ``NOT_SEARCHED``, and the Python files."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        raise NotACheckoutError(
            f"`git ls-files` failed in {root}: {listed.stderr.strip()} Run the tool on a git "
            "checkout (--root), with git installed."
        )
    markdown = {
        root / rel for rel in listed.stdout.split("\0") if rel and not rel.startswith(NOT_SEARCHED)
    }
    python = {path for d in PYTHON_DIRS for path in root.glob(f"{d}/**/*.py")}
    return sorted(path for path in markdown | python if path.is_file())


def citations(root: Path, files: list[Path]) -> Iterator[tuple[str, Family, str]]:
    """Every identifier cited in ``files``: (file:line, family, identifier), the logs left out."""
    logs = {log.file: log.heading for log in DECISION_LOGS}
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in logs and logs[rel] is None:
            continue  # the whole file is a log
        heading = logs.get(rel)
        in_log = False
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if heading is not None and line.startswith("## "):
                in_log = line.startswith(heading)
            if in_log:
                continue
            for family in FAMILIES:
                for match in family.pattern.finditer(line):
                    yield f"{rel}:{number}", family, match.group()


def _problem(where: str, family: Family, ident: str) -> str:
    log = LOGS_NAME
    if family.defined_in is not None:
        entry = family.defined_in.entry
        return (
            f"{where} cites {ident}, which has no {entry} in {family.defined_in.name} and no row "
            f"in {log}: add its {entry}, or a row to a log if it is closed."
        )
    if ident.startswith("R-"):
        return (
            f"{where} cites {ident}, a remark of the review of 16 November 2026 with no row in "
            f"{log}, and not one of the remarks kept in issues "
            f"{', '.join(KEPT_IN_ISSUES)} (KEPT_IN_ISSUES, tools/check_identifiers.py): add its "
            "row to the log, or cite the issue that keeps it instead of the identifier."
        )
    return (
        f"{where} cites {ident}, an {family.name} with no row in {log}: cite the current "
        "identifier of its point instead, or add its row to the log."
    )


@dataclass
class Report:
    errors: list[str]
    files: int = 0
    citations: int = 0
    in_lists: int = 0
    in_log: int = 0
    in_issues: int = 0


def check(root: Path) -> Report:
    try:
        defs = definitions(root)
        files = _cited_files(root)
    except (ListNotFoundError, NotACheckoutError) as error:
        return Report([str(error)])
    report = Report(list(defs.errors), files=len(files))
    resolved: dict[str, str] = {}
    for where, family, ident in citations(root, files):
        report.citations += 1
        if family.defined_in is not None and ident in defs.rows:
            resolved[ident] = "list"
        elif ident in defs.logged:
            resolved[ident] = "log"
        elif ident in KEPT:
            resolved[ident] = "issue"
        else:
            report.errors.append(_problem(where, family, ident))
    report.in_lists = sum(kind == "list" for kind in resolved.values())
    report.in_log = sum(kind == "log" for kind in resolved.values())
    report.in_issues = sum(kind == "issue" for kind in resolved.values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    report = check(args.root)
    for error in report.errors:
        print(f"check_identifiers: {error}", file=sys.stderr)
    if not report.errors:
        identifiers = report.in_lists + report.in_log + report.in_issues
        print(
            f"check_identifiers: OK, {report.citations} citations of {identifiers} identifiers "
            f"in {report.files} files: {report.in_lists} defined in their lists, "
            f"{report.in_log} in the decision logs, {report.in_issues} kept in issues."
        )
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
