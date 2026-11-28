"""Check the docstrings of the public names (review rules D1 and D12; issue #190).

The public names are those of ``api/manifest.json``, wherever they are defined, read as
``tools/api_manifest.py`` reads them: with ``ast``, without importing estel. Internal names
are not checked (golden rule 12). A name re-exported in several places is checked once, at its
definition.

The checks keep the numpydoc form that PR #222 gave the public docstrings; they do not ask for
more (no "Examples", no "See Also", no "Returns"). Each carries the code numpydoc's own
validation (``numpydoc.validate``) gives the same defect, so its documentation applies:

* ``GL08``: the name has no docstring (D12);
* ``SS01``: the docstring has no summary line: its first line is empty or a section header;
* ``GL06``: a section header that numpydoc does not know (a typo, or a home-made section);
* ``GL07``: the sections are not in numpydoc's order (Parameters, ..., Returns, ..., Notes,
  References, Examples);
* ``PR01``: a parameter of the signature is not documented in "Parameters";
* ``PR02``: "Parameters" documents a name the signature lacks;
* ``PR04``: a parameter is documented without its type (``name`` alone instead of
  ``name : type``);
* ``PR10``: the colon of a parameter line has no space before it (``name: type``).

The parameters checked are those of the signature: of a function or method (``self`` and
``cls`` left out), and of a class's ``__init__`` (whose docstring, if any, counts as the
class's). Several parameters may share a line: numpydoc's grouped form ``a, b : type`` when
they have one type, and ``a, b`` alone when their types differ and the description gives them
(as PR #222 kept ``standardize, cross_matrix``); a single name alone is PR04. ``*args`` and
``**kwargs`` are documented with or without their stars.

Not checked, because numpydoc's parser cannot tell or ESTEL does not ask for it: the
punctuation and capitals of the summary (SS02 to SS06), the descriptions of the parameters
(PR07 to PR09), the "Returns", "Yields" and "See Also" sections (RT, YD, SA), the examples
(EX01) and the whitespace (GL01 to GL03). Review rule D12's ``M-`` IDs are content, left to
review.

The tool is quick (tenths of a second): it parses the source, never imports it, and does not
need numpydoc installed.

Usage::

    python tools/check_docstrings.py           # check the repository
    python tools/check_docstrings.py --root R  # check the repository at R
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from api_manifest import Definition, public_definitions

REPO = Path(__file__).resolve().parents[1]

# numpydoc's sections, in its order (numpydoc.validate.ALLOWED_SECTIONS).
SECTIONS = (
    "Parameters",
    "Attributes",
    "Methods",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
)

# A section header: a line, then a line of dashes as long as it (numpydoc's own test).
_UNDERLINE = re.compile(r"^-{3,}$")
# Several names on one line with no type: parameters of different types documented together.
_GROUPED = re.compile(r"^\**\w+(, \**\w+)+\s*$")


@dataclass(frozen=True)
class Finding:
    where: str  # path:line
    name: str  # the public name, fully qualified
    code: str
    message: str  # what is wrong and what to do

    def __str__(self) -> str:
        return f"{self.where}: {self.code} {self.name}: {self.message}"


def _sections(lines: list[str]) -> list[tuple[str, int]]:
    """The section headers of a docstring, each with the index of its first content line."""
    out = []
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if title and _UNDERLINE.match(underline) and len(underline) == len(title):
            out.append((title, i + 2))
    return out


def _parameter_lines(lines: list[str], start: int, stop: int) -> list[str]:
    """The entries of a "Parameters" section: its lines at the base indentation."""
    return [
        line
        for line in lines[start:stop]
        if line.strip() and not line[0].isspace() and not _UNDERLINE.match(line.strip())
    ]


def _signature_parameters(signature: str | None, kind: str) -> list[str] | None:
    """The parameter names of a signature as the manifest writes it, ``None`` if it has none."""
    if signature is None or kind not in {"function", "method", "class"}:
        return None
    try:
        tree = ast.parse(f"def _{signature}: pass")
    except SyntaxError:
        return None
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    args = function.args
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    if names and names[0] in {"self", "cls"} and kind in {"method", "class"}:
        names = names[1:]
    if args.vararg is not None:
        names.append(f"*{args.vararg.arg}")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg is not None:
        names.append(f"**{args.kwarg.arg}")
    return names


def check_definition(name: str, definition: Definition) -> list[Finding]:
    """The findings on one public name."""
    where = definition.where or "?"
    doc = definition.docstring
    if not doc or not doc.strip():
        return [
            Finding(
                where,
                name,
                "GL08",
                "no docstring. Write one: a summary line, then the numpydoc sections "
                "(Parameters, Returns, ...), with the M- IDs of docs/maths.md it implements "
                "(review rule D12).",
            )
        ]
    findings = []
    lines = doc.strip("\n").splitlines()
    sections = _sections(lines)
    if lines[0].strip() == "" or (sections and sections[0][1] == 2):
        findings.append(
            Finding(
                where,
                name,
                "SS01",
                "no summary line. Start the docstring with one sentence that says what "
                f"{name.rsplit('.', 1)[-1]} is or does.",
            )
        )
    known = [title for title, _ in sections if title in SECTIONS]
    for title, _ in sections:
        if title not in SECTIONS:
            findings.append(
                Finding(
                    where,
                    name,
                    "GL06",
                    f'unknown section "{title}". Use one of numpydoc\'s: {", ".join(SECTIONS)}; '
                    "or make it a paragraph of Notes.",
                )
            )
    order = [SECTIONS.index(t) for t in known]
    if order != sorted(order):
        expected = sorted(known, key=SECTIONS.index)
        findings.append(
            Finding(
                where,
                name,
                "GL07",
                f"sections in the order {', '.join(known)}. Reorder them as numpydoc does: "
                f"{', '.join(expected)}.",
            )
        )
    signature = _signature_parameters(definition.signature, definition.kind)
    if signature is None:
        return findings
    documented: list[str] = []
    for index, (title, start) in enumerate(sections):
        if title != "Parameters":
            continue
        stop = sections[index + 1][1] - 2 if index + 1 < len(sections) else len(lines)
        for line in _parameter_lines(lines, start, stop):
            names, sep, kind = line.partition(" : ")
            if not sep and _GROUPED.match(line):
                pass  # names of different types on one line, their types in the description
            elif not sep:
                if re.match(r"^[\w*, ]+:", line):
                    findings.append(
                        Finding(
                            where,
                            name,
                            "PR10",
                            f'"{line.strip()}": put a space before the colon, '
                            '"name : type" (numpydoc reads "name: type" as one name).',
                        )
                    )
                    names = line.split(":", 1)[0]
                else:
                    findings.append(
                        Finding(
                            where,
                            name,
                            "PR04",
                            f'parameter "{line.strip()}" has no type. Write it '
                            f'"{line.strip()} : <type>", as in the signature\'s annotation.',
                        )
                    )
            elif not kind.strip():
                findings.append(
                    Finding(
                        where,
                        name,
                        "PR04",
                        f'parameter "{names.strip()}" has no type after its colon. Write '
                        "its type, as in the signature's annotation.",
                    )
                )
            documented += [n.strip() for n in names.split(",") if n.strip()]
    wanted = {p.lstrip("*") for p in signature}
    given = {d.lstrip("*") for d in documented}
    missing = [p for p in signature if p.lstrip("*") not in given]
    unknown = [d for d in documented if d.lstrip("*") not in wanted]
    if missing:
        findings.append(
            Finding(
                where,
                name,
                "PR01",
                f"parameters not documented: {', '.join(missing)}. Add each to its "
                '"Parameters" section, as "name : type" and an indented description.',
            )
        )
    if unknown:
        findings.append(
            Finding(
                where,
                name,
                "PR02",
                f"documented parameters the signature lacks: {', '.join(unknown)}. Remove "
                "them, or correct their names to those of the signature"
                f" ({', '.join(signature) or 'none'}).",
            )
        )
    return findings


def check(root: Path) -> list[Finding]:
    """Every finding on the public names of the repository at ``root``, once per definition."""
    findings: list[Finding] = []
    seen: set[int] = set()
    for name, definition in public_definitions(root / "src").items():
        if definition.kind == "alias" or id(definition) in seen:
            continue
        seen.add(id(definition))
        findings += check_definition(name, definition)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    findings = check(args.root)
    for finding in findings:
        print(f"check_docstrings: {finding}", file=sys.stderr)
    if findings:
        print(
            f"check_docstrings: {len(findings)} defects in the docstrings of public names "
            "(api/manifest.json). Fix each as its line says; the codes are numpydoc's "
            "(tools/check_docstrings.py lists those checked).",
            file=sys.stderr,
        )
        return 1
    print("check_docstrings: OK, the docstrings of the public names follow numpydoc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
