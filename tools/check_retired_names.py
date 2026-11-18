"""Check that public names retired by a renaming task do not come back (Part IV, §22, R-A12).

A retired name (``RETIRED`` below) must not appear in

* the specification: ``docs/transfer/``, ``docs/maths.md``, ``docs/maths_step2.md``;
* ``AGENTS.md`` (its mirrors are generated from it, ``tools/check_agent_mirrors.py``);
* the docstrings of the public names (``api/manifest.json``, read from the source as
  ``tools/api_manifest.py`` does).

A name matches as a whole identifier, also in its Markdown form with escaped underscores
(``angular\\_ft``), and not inside a longer identifier: STL's own function names, such as
``to_angular_ft`` or ``Base_DataClass``, do not match. Allowed:

* Part IV, §22 (the decision log is history);
* in ``docs/maths.md``, the **Source** lines (STL's provenance);
* in ``docs/`` only, a name attributed to STL: the backticked names right after "STL's"
  (``(STL's `store_ref`, `load_ref` and `vanilla`)``), and the phrase "vanilla ScatCov"
  (see ``ATTRIBUTED_PHRASES``). Docstrings of public names do not mention STL at all.

The tool reads text and ASTs only; it never imports estel.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from api_manifest import public_definitions

REPO = Path(__file__).resolve().parents[1]

# The public names retired by a renaming task, each with the name that replaces it, in one
# place. A renaming task adds the names it retires here, with its news fragment
# (changes/76.major.md for the first ones: R-A02 to R-A11 and R-S08 of Part IV, §22).
RETIRED: dict[str, str] = {
    "vanilla": 'norm="none"',
    "store_ref": 'norm="self"',
    "load_ref": 'norm="ref"',
    "norm_batch_mean": "ref_batch_mean",
    "compute_cross_matrix": "cross_matrix",
    "ScatteringPlan": "STPlan",
    "ScatteringFilters": "STFilters",
    "DataClass": "Data",
    "angular_ft": "reduce(angular_fourier=True)",
    "scale_ft": "reduce(scale_cosine=True)",
}

# Phrases of docs/ that attribute a retired name to STL. "vanilla ScatCov" (STL's name for
# its statistics) is tolerated only until R-V07 (issue #72, vocabulary of the code) removes
# the phrase from the specification; remove this allowance then.
ATTRIBUTED_PHRASES = ("vanilla ScatCov",)

SPEC_FILES = ("docs/maths.md", "docs/maths_step2.md", "AGENTS.md")
SPEC_DIRS = ("docs/transfer",)
DECISION_LOG = ("docs/transfer/part4.md", re.compile(r"^## 22\.\s"))
SOURCE_LINE = re.compile(r"^\s*[-*]?\s*\*\*Source:?\*\*")


def _name_pattern(names: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(n).replace("_", r"\\?_") for n in names)
    return re.compile(rf"(?<![\w\\])({alternatives})(?![\w\\])")


RETIRED_RE = _name_pattern(sorted(RETIRED, key=len, reverse=True))
# "STL's `a`", "STL's `a` and `b`", "STL's `a`, `b` and `c`".
STL_ATTRIBUTION_RE = re.compile(
    r"STL's\s+`[^`]+`(?:(?:,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)`[^`]+`)*"
)
# Also in a heading's anchor, as "vanilla-scatcov".
PHRASES_RE = re.compile(
    "|".join(re.escape(p).replace(r"\ ", r"[\s-]") for p in ATTRIBUTED_PHRASES), re.IGNORECASE
)


@dataclass(frozen=True)
class Hit:
    where: str
    found: str  # as written, escaped underscores included

    def message(self) -> str:
        name = self.found.replace("\\", "")
        return (
            f"{self.where}: `{self.found}` is a retired public name. Use `{RETIRED[name]}` "
            f"instead. To cite STL's name in docs/, write it after \"STL's\" (STL's `{name}`); "
            "a docstring of a public name does not mention STL."
        )


def _allowed_spans(line: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in STL_ATTRIBUTION_RE.finditer(line)]
    return spans + [m.span() for m in PHRASES_RE.finditer(line)]


def check_markdown(path: Path, rel: str) -> list[Hit]:
    hits = []
    log_file, log_heading = DECISION_LOG
    in_log = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if rel == log_file and line.startswith("## "):
            in_log = bool(log_heading.match(line))
        if in_log or (rel == "docs/maths.md" and SOURCE_LINE.match(line)):
            continue
        spans = _allowed_spans(line) if rel.startswith("docs/") else []
        for match in RETIRED_RE.finditer(line):
            if not any(a <= match.start() and match.end() <= b for a, b in spans):
                hits.append(Hit(f"{rel}:{number}", match.group(1)))
    return hits


def check_docstrings(root: Path) -> list[Hit]:
    hits = []
    for qualname, definition in public_definitions(root / "src").items():
        for match in RETIRED_RE.finditer(definition.docstring or ""):
            hits.append(Hit(f"{definition.where} (docstring of {qualname})", match.group(1)))
    return hits


def check(root: Path) -> list[Hit]:
    files = [root / f for f in SPEC_FILES]
    for directory in SPEC_DIRS:
        files += sorted((root / directory).rglob("*.md"))
    hits = []
    for path in files:
        if path.exists():
            hits += check_markdown(path, path.relative_to(root).as_posix())
    return hits + check_docstrings(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    hits = check(args.root)
    for hit in hits:
        print(f"check_retired_names: {hit.message()}", file=sys.stderr)
    if not hits:
        print(f"check_retired_names: OK, none of the {len(RETIRED)} retired names is used.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
