"""Check the names of ESTEL: retired public names, and the rule of names (Part I, §2.4.1).

Two checks, one tool. It reads text and ASTs only, on the files git tracks; it never imports
estel.

**Retired public names** (Part IV, §22, R-A12; ``RETIRED`` below). A retired name must not
appear in the specification (``docs/transfer/``, ``docs/maths.md``, ``docs/maths_step2.md``),
in ``AGENTS.md`` (its mirrors are generated from it), or in the docstrings of the public names
(``api/manifest.json``, read as ``tools/api_manifest.py`` does). A name matches as a whole
identifier, also with escaped underscores (``angular\\_ft``). Allowed: Part IV, §22 (history);
the **Source** lines of ``docs/maths.md``; in ``docs/`` only, a name attributed to STL (the
backticked names right after "STL's") and the phrase "vanilla ScatCov"; for a name in
``RETIRED_ROWS``, the rows of Part II, §15.2 of the tasks that created and removed it.

**The rule of names** (Part I, §2.4.1; phase P7). The tool decides:

* rules 1 and 2 on every definition of ``src/estel/`` (the oracle included), ``tests/``,
  ``tools/`` and ``bench/`` (classes, functions, methods, module-level names, fields) and on
  every module path: the geometry word is the last word, or the last run, in the forms of rule
  2 (``Healpix``, ``healpix``, ``HEALPIX``; ``TensorProd``, ``tensor_prod``, ``TENSOR_PROD``; …).
  Test functions (sentences) are checked for their words, not their order;
* rule 2 in every identifier, file name and label of every text file: ``hpx``, ``tp`` and
  ``tensor_product`` are retired;
* rule 3: a name defined in a module of one geometry and used (imported, or read through its
  module) outside the modules of that geometry carries that geometry;
* rule 9: the maths IDs read ``M-<TOPIC>[-<VARIANT>]*[-<GEOMETRY>]*``;
* rule 10: "planar", "single geometry" and "plain geometry" are retired, in identifiers, paths
  and prose, except in a quote of STL followed by "(STL's word)";
* rule 7: the closed lists below; nothing else is exempt;
* rule 6, through the table ``tools/renamed_names.toml``: one row per name that breaks the rule,
  with the task of phase P7 that renames it. A violation with no row fails; a violation covered
  by a row not done is reported, one line per task (``--report`` lists them); a row done whose
  old name still occurs fails, except a ``reintroduce`` row done, which lets the old name come
  back with its new meaning, defined in the module the row names.

And the import contracts of ``pyproject.toml`` may lose no module: every module they name
exists, or is in ``RESERVED_MODULES``.

The meaning of a suffix, and rules 4, 5, 8 and 11, are left to review
(``docs/agents/review_rules.md``).

Usage::

    python tools/check_retired_names.py            # fail on a violation; one line per task
    python tools/check_retired_names.py --report   # also list every violation a row covers
    python tools/check_retired_names.py --strict   # also fail while a row is not done (T7.12)
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from api_manifest import public_definitions

REPO = Path(__file__).resolve().parents[1]
TABLE = "tools/renamed_names.toml"

# ================================================================== retired public names (R-A12)

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
    # Removed by T4.6 (#205); it must not come back (decisions of the maintainer on #193).
    "SpectrumOperator": "STOperator(data, J, compute_PS=True)",
}

# The rows of Part II, §15.2 that may name a retired name: those of the tasks that created and
# removed it, which record what was done with the names of the time.
RETIRED_ROWS: dict[str, tuple[str, ...]] = {"SpectrumOperator": ("T4.1", "T4.6")}

# Phrases of docs/ that attribute a retired name to STL. "vanilla ScatCov" (STL's name for
# its statistics) is tolerated only until R-V07 (issue #72, vocabulary of the code) removes
# the phrase from the specification; remove this allowance then.
ATTRIBUTED_PHRASES = ("vanilla ScatCov",)

SPEC_FILES = ("docs/maths.md", "docs/maths_step2.md", "AGENTS.md")
SPEC_DIRS = ("docs/transfer",)
DECISION_LOG = ("docs/transfer/part4.md", re.compile(r"^## 22\.\s"))
WORK_BREAKDOWN = "docs/transfer/part2.md"
SOURCE_LINE = re.compile(r"^\s*[-*]?\s*\*\*Source:?\*\*")
TASK_ROW = re.compile(r"^\|\s*(T\d+\.\d+)\s*\|")


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
        row = TASK_ROW.match(line) if rel == WORK_BREAKDOWN else None
        spans = _allowed_spans(line) if rel.startswith("docs/") else []
        for match in RETIRED_RE.finditer(line):
            name = match.group(1).replace("\\", "")
            if row and row.group(1) in RETIRED_ROWS.get(name, ()):
                continue
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


# ================================================================== the rule of names (§2.4.1)

# ------------------------------------------------------------------ rule 7: the closed lists

# The names of the published HEALPix convention (Górski et al. 2005, ApJ 622, 759), never a
# code source (Part I, §1.5, F23). They carry no geometry word; listed so that no rule ever
# asks to rename them.
CONVENTION_HEALPIX = frozenset({"nest2xyf", "xyf2nest", "xyf2ang", "ring2nest"})

# Package names (Part I, §1.5 and §1.6), as imported and as installed.
PACKAGES = frozenset({"healpix_analyse", "healpy", "healpix_geo", "healpix-geo", "optax"})

# Names fixed by another program, which never change (rule 3): pytest's module marker, the
# interface of the parity cases (tools/stl_parity/run_case.py reads them), and, computed, the
# fixtures of every conftest.py.
FIXED_BY_PROGRAM = frozenset({"pytestmark", "STL_ENV", "PARAMETERS", "run"})

# History (rule 7): what was done, with the names of the time. Part IV, §22 is the part of
# docs/transfer/part4.md under its heading; the others are whole files or directories.
HISTORY_FILES = frozenset({"docs/register/decisions.md"})
HISTORY_DIRS = ("changes/", "bench/results/")

# STL's own names (at the frozen commits of Part II, §9.1) that ESTEL's text cites, and the
# reserved public name `plot.nyquist_filter` (Part II, §12.10).
STL_NAMES = frozenset(
    {
        # STL-Dev `2D+1D` at 8f51143: the modules of tensor products (docs/maths.md, its
        # Source lines; imported by tools/stl_parity/cases/*_tp.py and smoke.py).
        "Tensor_Product_Kernel_Torch",
        "ST_Operator_Tensor_Product",
        # Reserved public name (Part II, §12.10).
        "nyquist_filter",
    }
)

# The four benchmark labels of step 1 (bench/cases.toml).
BENCH_LABELS = frozenset({"planar-periodic", "planar-masked", "planar-cross", "planar-open"})

# The instruments of phase P7, which name old names on purpose: this tool, its table and its
# tests, and the proof, whose cases are written with the names of `main` and resolved through
# the table.
INSTRUMENTS = frozenset(
    {
        "tools/check_retired_names.py",
        TABLE,
        "tests/tools/test_check_retired_names.py",
        "tools/prove_unchanged.py",
        "tests/tools/test_prove_unchanged.py",
    }
)

# The text that states the rule, and the rows of the tasks that apply it, cite the names they
# retire (as RETIRED_ROWS does for a retired public name): Part I, §2.4.1, and the rows T7.2 to
# T7.12 of Part II, §15.2.
RULE_SECTION = ("docs/transfer/part1.md", "#### 2.4.1 ")
P7_ROWS = ("docs/transfer/part2.md", re.compile(r"^\|\s*T7\.(?:[2-9]|1[0-2])\s*\|"))

# Modules named by an import contract that do not exist yet, each with its reason.
RESERVED_MODULES: dict[str, str] = {
    "estel_rust": "the Rust engine's wheel, from step 2 (Part I, §2.4; Part III, §18)",
    "optax": "the L-BFGS of the JAX backend, from step 2 (LF1; Part I, §1.6): forbidden "
    "outside estel.backend before it is a dependency (contract backend-purity)",
    "estel.diagnostics": "describe, estimate_memory, frequency_response, littlewood_paley: "
    "planned in Part II, §12.10, not written yet (contract wavelet-operator-only)",
}

# ------------------------------------------------------------------ geometry words

# The tree of geometries (Part I, §2.3.1): each node with its parent.
PARENT = {
    "single_domain": "",
    "tensor_prod": "",
    "cartesian": "single_domain",
    "healpix": "single_domain",
    "1d": "cartesian",
    "2d": "cartesian",
}
# The forms of rule 2, by style of identifier.
FORMS = {
    "single_domain": ("SingleDomain", "single_domain", "SINGLE_DOMAIN"),
    "cartesian": ("Cartesian", "cartesian", "CARTESIAN"),
    "1d": ("1D", "1d", "1D"),
    "2d": ("2D", "2d", "2D"),
    "healpix": ("Healpix", "healpix", "HEALPIX"),
    "tensor_prod": ("TensorProd", "tensor_prod", "TENSOR_PROD"),
}
# Retired words (rules 2 and 10), each with what replaces it.
RETIRED_WORDS = {
    "hpx": "healpix",
    "tp": "tensor_prod",
    "tensor_product": "tensor_prod",
    "planar": "cartesian or single_domain",
}
# Maths IDs (rule 9).
ID_GEOMETRIES = frozenset({"1D", "2D", "HEALPIX", "TENSORPROD"})
ID_RETIRED = {"HPX": "HEALPIX", "TP": "TENSORPROD", "PLANAR": "a geometry of rule 2"}
ID_QUANTITIES = frozenset({"S1", "S2", "S3", "S4", "W1"})

WORD_RE = re.compile(
    r"(?i:healpix[_-]analyse|healpix[_-]geo(?![a-z]))|(?i:tensor_?product)|(?i:tensor_?prod)|(?i:single_?domain)|(?i:healpix)"
    r"|[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+[A-Za-z]?"
)


def _word_geometry(word: str) -> str | None:
    """The node of a word, a retired word, or None (a package name is no geometry)."""
    key = word.lower().replace("_", "")
    if key.replace("-", "") in ("healpixanalyse", "healpixgeo"):
        return None
    for node in FORMS:
        if key == node.replace("_", ""):
            return node
    for retired in RETIRED_WORDS:
        if key == retired.replace("_", ""):
            return retired
    return None


def _style(name: str) -> int:
    """0 CapWords, 1 snake_case, 2 UPPER_CASE."""
    bare = name.strip("_")
    if bare and bare.upper() == bare and any(c.isalpha() for c in bare):
        return 2
    if bare[:1].isupper() and "_" not in bare:
        return 0
    return 1


@dataclass(frozen=True)
class Problem:
    rule: str
    expected: str


def name_problems(name: str, *, order: bool = True) -> list[Problem]:
    """What rules 1, 2 and 10 say of an identifier, a module path component or a label."""
    words = WORD_RE.findall(name)
    nodes = [_word_geometry(w) for w in words]
    problems = []
    style = _style(name)
    for word, node in zip(words, nodes, strict=True):
        if node in RETIRED_WORDS:
            rule = "rule 10" if node == "planar" else "rule 2"
            problems.append(Problem(rule, f"`{word}` is retired: {RETIRED_WORDS[node]}"))
        elif node is not None and word != FORMS[node][style] and "-" not in name:
            problems.append(Problem("rule 2", f"`{word}` is written `{FORMS[node][style]}` here"))
    if order:
        geometric = [n is not None for n in nodes]
        if True in geometric:
            first = geometric.index(True)
            if not all(geometric[first:]):
                problems.append(Problem("rule 1", "the geometry is the last word or the last run"))
    return problems


def proposed_name(name: str) -> str:
    """The name with its retired words replaced and its geometry run moved to the end."""
    words = WORD_RE.findall(name)
    style = _style(name)
    lead = name[: len(name) - len(name.lstrip("_"))]
    core, run = [], []
    for word in words:
        node = _word_geometry(word)
        if node == "planar":
            run.append({0: "Cartesian", 1: "cartesian", 2: "CARTESIAN"}[style])
        elif node in RETIRED_WORDS:
            run.append(FORMS[RETIRED_WORDS[node]][style])
        elif node is not None:
            run.append(FORMS[node][style])
        else:
            core.append(word)
    if style == 0:
        return lead + "".join(core + run)
    return lead + "_".join(core + run)


def covers(node: str, other: str) -> bool:
    """Whether ``other`` is ``node`` or below it in the tree."""
    while other:
        if other == node:
            return True
        other = PARENT[other]
    return False


def path_geometries(path: str) -> set[str]:
    """The geometries a module path names (retired words mapped to their node)."""
    found = set()
    for word in WORD_RE.findall(path):
        node = _word_geometry(word)
        if node in ("hpx",):
            found.add("healpix")
        elif node in ("tp", "tensor_product"):
            found.add("tensor_prod")
        elif node == "planar":
            found.add("planar")
        elif node is not None:
            found.add(node)
    return found


# ------------------------------------------------------------------ violations


@dataclass(frozen=True)
class Violation:
    rule: str
    key: str  # what a row of the table names: see Row
    where: str  # file:line
    detail: str
    names: tuple[str, ...] = ()  # the bare names a row may cover it by


@dataclass
class Row:
    """A row of the table.

    ``old`` is qualified: ``module`` (a dotted module path), ``module:Qual.name`` (a
    definition), ``path`` (a file that is not a module), ``path#word`` (a word or a label in a
    file, prose included), or a maths ID ``M-…``.
    """

    old: str
    new: str
    kind: str
    task: str
    done: bool
    note: str = ""

    @property
    def bare(self) -> str:
        return re.split(r"[:.#/]", self.old.removesuffix(".py"))[-1] if self.old else ""


KINDS = frozenset({"rename", "split", "move", "reintroduce"})
TASKS = frozenset(f"T7.{n}" for n in range(3, 13))


def load_table(root: Path) -> tuple[list[Row], list[str]]:
    path = root / TABLE
    if not path.exists():
        return [], []
    data = tomllib.loads(path.read_text())
    rows, errors = [], []
    for number, entry in enumerate(data.get("row", []), 1):
        try:
            row = Row(**entry)
        except TypeError as error:
            errors.append(f"{TABLE}: row {number}: {error}; fields: old, new, kind, task, done")
            continue
        if row.kind not in KINDS:
            errors.append(f"{TABLE}: row {number} ({row.old}): kind is one of {sorted(KINDS)}")
        if row.task not in TASKS:
            errors.append(f"{TABLE}: row {number} ({row.old}): task is one of T7.3 to T7.12")
        rows.append(row)
    olds = [(r.old, r.kind) for r in rows]
    for old, kind in sorted({o for o in olds if olds.count(o) > 1}):
        errors.append(f"{TABLE}: two {kind} rows for {old}: keep one")
    for row in rows:
        if row.kind == "reintroduce" and row.done:
            renamed = [r for r in rows if r.old == row.old and r.kind == "rename"]
            if not renamed or not all(r.done for r in renamed):
                errors.append(
                    f"{TABLE}: {row.old} is reintroduced before its rename row is done: "
                    "rename it everywhere first, in an earlier commit (rule 6)"
                )
    return rows, errors


# ------------------------------------------------------------------ the files


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    )
    files = [f for f in result.stdout.split("\0") if f and not f.startswith(".claude/")]
    return sorted(f for f in files if (root / f).is_file())


def is_history(rel: str) -> bool:
    return rel in HISTORY_FILES or rel.startswith(HISTORY_DIRS)


def module_name(rel: str) -> str | None:
    """The dotted module of a Python file of src/, tests/, tools/ or bench/."""
    if not rel.endswith(".py"):
        return None
    parts = rel.removesuffix(".py").split("/")
    if parts[0] == "src":
        parts = parts[1:]
    elif parts[0] not in ("tests", "tools", "bench"):
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


CHECKED_ROOTS = ("src/estel/", "tests/", "tools/", "bench/")
PACKAGE_ROOT = "src/estel/"


@dataclass
class Definition:
    module: str
    qualname: str
    line: int
    kind: str  # "class", "function", "method", "name", "field", "test", "fixture"


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _targets(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    names = []
    for target in targets:
        elements = target.elts if isinstance(target, ast.Tuple) else [target]
        names += [e.id for e in elements if isinstance(e, ast.Name)]
    return names


def definitions(module: str, tree: ast.Module, *, is_test: bool) -> list[Definition]:
    found: list[Definition] = []

    def visit_class(node: ast.ClassDef, prefix: str) -> None:
        qual = f"{prefix}{node.name}"
        found.append(Definition(module, qual, node.lineno, "class"))
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                visit_class(item, f"{qual}.")
            elif isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                kind = "test" if is_test and item.name.startswith("test") else "method"
                found.append(Definition(module, f"{qual}.{item.name}", item.lineno, kind))
                for sub in ast.walk(item):
                    if isinstance(sub, ast.Assign | ast.AnnAssign):
                        targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                        for target in targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                found.append(
                                    Definition(module, f"{qual}.{target.attr}", sub.lineno, "field")
                                )
            for name in _targets(item):
                found.append(Definition(module, f"{qual}.{name}", item.lineno, "field"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            visit_class(node, "")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if is_test and "fixture" in _decorators(node) and module.endswith("conftest"):
                kind = "fixture"
            elif is_test and node.name.startswith("test"):
                kind = "test"
            else:
                kind = "function"
            found.append(Definition(module, node.name, node.lineno, kind))
        for name in _targets(node):
            found.append(Definition(module, name, node.lineno, "name"))
    # One definition per name: a field assigned twice is defined once.
    unique: dict[str, Definition] = {}
    for definition in found:
        unique.setdefault(definition.qualname, definition)
    return list(unique.values())


def _is_exempt(name: str) -> bool:
    return (
        name in CONVENTION_HEALPIX
        or name in PACKAGES
        or name in STL_NAMES
        or (name.startswith("__") and name.endswith("__"))
    )


# ------------------------------------------------------------------ rules 1, 2 on definitions


def check_definitions(
    rel: str, module: str, tree: ast.Module, fixtures: set[str]
) -> Iterator[Violation]:
    is_test = rel.startswith("tests/")
    for definition in definitions(module, tree, is_test=is_test):
        name = definition.qualname.rsplit(".", 1)[-1]
        if name in FIXED_BY_PROGRAM or name in fixtures or definition.kind == "fixture":
            continue
        if _is_exempt(name):
            continue
        problems = name_problems(name, order=definition.kind != "test")
        for problem in problems:
            yield Violation(
                problem.rule,
                f"{module}:{definition.qualname}",
                f"{rel}:{definition.line}",
                f"the {definition.kind} `{definition.qualname}`: {problem.expected}"
                f" (proposed: `{proposed_name(name)}`)",
                (name,),
            )


# ------------------------------------------------------------------ rules 1, 2, 10 on paths


def check_paths(files: list[str]) -> Iterator[Violation]:
    directories: dict[str, str] = {}
    for rel in files:
        if is_history(rel):
            continue
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            directories.setdefault("/".join(parts[:depth]) + "/", rel)
        module = module_name(rel)
        stem = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
        is_module = module is not None and rel.startswith(CHECKED_ROOTS)
        if stem == "__init__" or _is_exempt(stem):
            continue
        for problem in name_problems(stem, order=is_module):
            key = module if is_module and module else rel
            yield Violation(
                problem.rule,
                key,
                f"{rel}:1",
                f"the {'module' if is_module else 'file'} `{rel}`: {problem.expected}"
                f" (proposed: `{proposed_name(stem)}`)",
                (stem, parts[-1]),
            )
    for directory, example in sorted(directories.items()):
        stem = directory.rstrip("/").rsplit("/", 1)[-1]
        if _is_exempt(stem) or stem.startswith("."):
            continue
        is_package = directory.startswith(CHECKED_ROOTS) and example.endswith(".py")
        for problem in name_problems(stem, order=is_package):
            yield Violation(
                problem.rule,
                directory,
                f"{directory}",
                f"the directory `{directory}`: {problem.expected}",
                (stem,),
            )


# ------------------------------------------------------------------ rules 2, 9, 10 in text

TOKEN_RE = re.compile(r"(?:\\?_|[A-Za-z0-9])(?:\\?_|[A-Za-z0-9]|[-./](?=\\?_|[A-Za-z0-9]))*")
MATHS_ID_RE = re.compile(r"^M-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
ANCHOR_RE = re.compile(r"(?<=\]\()#([a-z0-9-]+)")


def _anchor_words(anchor: str) -> str:
    """The words of a heading's anchor, its maths IDs left out (rule 9 checks them)."""
    return " ".join(p.replace("-", " ") for p in anchor.split("--") if not p.startswith("m-"))


PHRASE_RE = re.compile(r"(?i)\b(?:single|plain)[\s_-]+geometry\b")
STL_WORD = "(STL's word)"
RETIRED_TEXT_RE = re.compile(r"(?i)hpx|tp|tensor_?product|planar")


def _segments(rel: str, text: str) -> Iterator[tuple[int, str, bool]]:
    """(line, text, is_code): in Python, code tokens apart from comments and strings."""
    if rel.endswith(".py"):
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, SyntaxError):
            tokens = []
        if tokens:
            for token in tokens:
                if token.type == tokenize.NAME:
                    yield token.start[0], token.string, True
                elif token.type in (tokenize.STRING, tokenize.COMMENT) or (
                    sys.version_info >= (3, 12) and token.type == tokenize.FSTRING_MIDDLE
                ):
                    for offset, line in enumerate(token.string.split("\n")):
                        yield token.start[0] + offset, line, False
            return
    for number, line in enumerate(text.split("\n"), 1):
        yield number, line, False


def _maths_id_problems(identifier: str) -> list[str]:
    parts = identifier.split("-")[1:]
    problems = []
    geometric = [p in ID_GEOMETRIES or p in ID_RETIRED for p in parts]
    if parts and all(geometric):
        problems.append("a geometry alone is no topic: the ID names its topic first")
    for part in parts:
        if part in ID_RETIRED:
            problems.append(f"`-{part}` is written `-{ID_RETIRED[part]}`")
        elif (
            part not in ID_GEOMETRIES
            and part not in ID_QUANTITIES
            and any(c.isdigit() for c in part)
        ):
            problems.append(f"`{part}`: digits only for the named quantities S1 to S4 and W1")
    if True in geometric[1:]:
        first = geometric.index(True, 1)
        if not all(geometric[first:]):
            problems.append("the geometry is the last part")
    return problems


def _skipped_lines(rel: str, lines: list[str]) -> set[int]:
    """The lines that may name what the rule retires: history, and the text of the rule."""
    skip: set[int] = set()
    log_file, log_heading = DECISION_LOG
    if rel == log_file:
        inside = False
        for number, line in enumerate(lines, 1):
            if line.startswith("## "):
                inside = bool(log_heading.match(line))
            if inside:
                skip.add(number)
    if rel == RULE_SECTION[0]:
        inside = False
        for number, line in enumerate(lines, 1):
            if line.startswith("#"):
                inside = line.startswith(RULE_SECTION[1])
            if inside:
                skip.add(number)
    if rel == P7_ROWS[0]:
        skip |= {n for n, line in enumerate(lines, 1) if P7_ROWS[1].match(line)}
    return skip


def check_text(rel: str, text: str) -> Iterator[Violation]:
    skip = _skipped_lines(rel, text.split("\n"))
    for number, segment, is_code in _segments(rel, text):
        if number in skip:
            continue
        if not is_code:
            for match in PHRASE_RE.finditer(segment):
                if STL_WORD not in segment[match.end() : match.end() + 120]:
                    phrase = re.sub(r"[\s_-]+", " ", match.group(0).lower())
                    yield Violation(
                        "rule 10",
                        f"{rel}#{phrase}",
                        f"{rel}:{number}",
                        f'"{match.group(0)}" is retired: "Cartesian" or "single-domain"',
                    )
            for match in ANCHOR_RE.finditer(segment):
                # The heading checks its ID; its anchor, only for the retired parts, as the
                # end of the ID in an anchor is not marked.
                for piece in match.group(1).split("--"):
                    parts = piece.split("-")
                    at = [i for i, p in enumerate(parts) if p.upper() in ID_RETIRED]
                    if piece.startswith("m-") and at:
                        identifier = "-".join(parts[: at[-1] + 1])
                        yield from _check_maths_id(rel, number, identifier.upper(), piece)
            segment = ANCHOR_RE.sub(lambda m: _anchor_words(m.group(1)), segment)
        tokens = [segment] if is_code else [m.group(0) for m in TOKEN_RE.finditer(segment)]
        positions = [0] if is_code else [m.end() for m in TOKEN_RE.finditer(segment)]
        for token, end in zip(tokens, positions, strict=True):
            token = token.replace("\\", "")
            if MATHS_ID_RE.match(token):
                yield from _check_maths_id(rel, number, token, token)
                continue
            if not RETIRED_TEXT_RE.search(token):
                continue
            for component in re.split(r"[./]", token):
                if not component or _is_exempt(component) or component in BENCH_LABELS:
                    continue
                words = [_word_geometry(w) for w in WORD_RE.findall(component)]
                retired = [w for w in words if w in RETIRED_WORDS]
                if not retired:
                    continue
                if not is_code and STL_WORD in segment[end : end + 120]:
                    continue
                prose = not is_code and component.lower() == "planar" and token == component
                if prose:
                    yield Violation(
                        "rule 10",
                        f"{rel}#planar",
                        f"{rel}:{number}",
                        '"planar" is retired: "Cartesian" for 1D and 2D grids, '
                        '"single-domain" for what is not a tensor product',
                    )
                    continue
                rule = "rule 10" if retired == ["planar"] else "rule 2"
                yield Violation(
                    rule,
                    f"{rel}#{component}",
                    f"{rel}:{number}",
                    f"`{component}`: {', '.join(sorted({RETIRED_WORDS[w] for w in retired}))}"
                    f" (proposed: `{proposed_name(component)}`)",
                    (component,),
                )


def _check_maths_id(rel: str, number: int, identifier: str, written: str) -> Iterator[Violation]:
    for problem in _maths_id_problems(identifier):
        yield Violation(
            "rule 9", identifier, f"{rel}:{number}", f"the maths ID `{written}`: {problem}"
        )


# ------------------------------------------------------------------ the code: rule 3 and done rows


@dataclass
class ModuleInfo:
    rel: str
    tree: ast.Module
    defined: set[str] = field(default_factory=set)
    # name -> (module it comes from, its name there), for names imported into the module.
    imported: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Every definition (qualified name -> line), as check_definitions sees them.
    qualnames: dict[str, int] = field(default_factory=dict)
    # (module, name, line): a name imported from a module of the code, or read through it.
    uses: list[tuple[str, str, int]] = field(default_factory=list)
    # (module, line): a module imported, as a whole or by a name it might hold.
    imports: list[tuple[str, int]] = field(default_factory=list)


def _absolute(module: str, is_package: bool, level: int, target: str | None) -> str:
    if level == 0:
        return target or ""
    parts = module.split(".")
    base = parts if is_package else parts[:-1]
    base = base[: len(base) - (level - 1)]
    return ".".join([*base, target] if target else base)


def index_code(modules: dict[str, ModuleInfo]) -> None:
    """Fill each module's definitions, imports and uses (the names it reads from others)."""
    for module, info in modules.items():
        is_test = info.rel.startswith("tests/")
        info.qualnames = {
            d.qualname: d.line for d in definitions(module, info.tree, is_test=is_test)
        }
        for statement in info.tree.body:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                info.defined.add(statement.name)
            info.defined.update(_targets(statement))
        is_package = info.rel.endswith("__init__.py")
        aliases: dict[str, str] = {}  # a local name bound to a module
        for node in ast.walk(info.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append((alias.name, node.lineno))
                    if alias.asname:
                        aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom):
                source = _absolute(module, is_package, node.level, node.module)
                info.imports.append((source, node.lineno))
                for alias in node.names:
                    info.imported[alias.asname or alias.name] = (source, alias.name)
                    info.imports.append((f"{source}.{alias.name}", node.lineno))
                    if f"{source}.{alias.name}" in modules:
                        aliases[alias.asname or alias.name] = f"{source}.{alias.name}"
                    else:
                        info.uses.append((source, alias.name, node.lineno))
        for node in ast.walk(info.tree):
            if isinstance(node, ast.Attribute):
                chain = []
                value: ast.expr = node
                while isinstance(value, ast.Attribute):
                    chain.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    dotted = ".".join([value.id, *reversed(chain)])
                    head = aliases.get(value.id, value.id)
                    target, name = dotted.rsplit(".", 1)
                    target = head + target[len(value.id) :]
                    if target in modules and target != module:
                        info.uses.append((target, name, node.lineno))


def check_rule3(modules: dict[str, ModuleInfo]) -> Iterator[Violation]:
    """Rule 3, in the package only: in tests/, tools/ and bench/, rules 1, 2 and 10 apply."""

    def origin(module: str, name: str, depth: int = 0) -> str | None:
        info = modules.get(module)
        if info is None or depth > 10:
            return None
        if name in info.defined:
            return module
        if name in info.imported:
            return origin(*info.imported[name], depth + 1)
        return None

    for module, info in modules.items():
        if not info.rel.startswith(PACKAGE_ROOT):
            continue
        importer = path_geometries(info.rel)
        seen = set()
        for source, name, line in info.uses:
            home = origin(source, name)
            if home is None or home == module or (home, name) in seen:
                continue
            if not modules[home].rel.startswith(PACKAGE_ROOT):
                continue
            seen.add((home, name))
            geometries = path_geometries(modules[home].rel)
            if not geometries or "planar" in geometries or _is_exempt(name):
                continue
            words = [_word_geometry(w) for w in WORD_RE.findall(name)]
            for geometry in geometries:
                if any(covers(geometry, g) for g in importer if g != "planar"):
                    continue
                if (
                    "planar" in importer
                    and covers("single_domain", geometry)
                    and (geometry != "healpix")
                ):
                    continue
                if any(
                    w is not None and w not in RETIRED_WORDS and covers(geometry, w) for w in words
                ):
                    continue
                expected = FORMS[geometry][_style(name)]
                yield Violation(
                    "rule 3",
                    f"{home}:{name}",
                    f"{info.rel}:{line}",
                    f"`{name}`, defined in `{home}` (geometry {geometry}), is used in "
                    f"`{module}`: outside the modules of its geometry it ends with `{expected}`"
                    f" (proposed: `{proposed_name(name + '_' + geometry)}`)",
                    (name,),
                )


# ------------------------------------------------------------------ the import contracts


def contract_modules(root: Path) -> list[tuple[str, str]]:
    """(contract id, module) for every module named in an import contract."""
    if not (root / "pyproject.toml").exists():
        return []
    data = tomllib.loads((root / "pyproject.toml").read_text())
    named: list[tuple[str, str]] = []
    for contract in data.get("tool", {}).get("importlinter", {}).get("contracts", []):
        cid = contract.get("id", contract.get("name", "?"))
        containers = contract.get("containers", [])
        for key in ("source_modules", "forbidden_modules"):
            named += [(cid, m) for m in contract.get(key, [])]
        if "source_module" in contract:
            named.append((cid, contract["source_module"]))
        named += [(cid, m) for m in containers]
        for layer in contract.get("layers", []):
            for part in re.split(r"[:|]", layer):
                name = part.strip().strip("()").strip()
                named += [(cid, f"{c}.{name}") for c in containers] if containers else []
        for ignored in contract.get("ignore_imports", []):
            named += [(cid, m.strip()) for m in ignored.split("->")]
    return named


def check_contracts(root: Path) -> list[str]:
    if not (root / "pyproject.toml").exists():
        return []
    data = tomllib.loads((root / "pyproject.toml").read_text())
    project = data.get("project", {})
    requirements = list(project.get("dependencies", []))
    for group in [*project.get("optional-dependencies", {}).values()]:
        requirements += group
    for group in data.get("dependency-groups", {}).values():
        requirements += [r for r in group if isinstance(r, str)]
    external = {
        match.group(0).lower().replace("-", "_")
        for r in requirements
        if (match := re.match(r"[A-Za-z0-9_.-]+", r))
    }
    roots = set(data.get("tool", {}).get("importlinter", {}).get("root_packages", []))
    errors = []
    for cid, module in dict.fromkeys(contract_modules(root)):
        pattern = module.replace(".**", "").replace(".*", "")
        top = pattern.split(".")[0]
        if pattern in RESERVED_MODULES or top in RESERVED_MODULES:
            continue
        if top in roots:
            path = root / "src" / Path(*pattern.split("."))
            if path.with_suffix(".py").is_file() or (path / "__init__.py").is_file():
                continue
            errors.append(
                f"pyproject.toml, contract {cid}: the module {module} does not exist. A "
                "renamed module keeps its contract: write its new name in the contract; a "
                "module that does not exist yet goes in RESERVED_MODULES of "
                "tools/check_retired_names.py, with its reason."
            )
        elif top.lower() not in external and not (top == "jaxlib" and "jax" in external):
            errors.append(
                f"pyproject.toml, contract {cid}: {module} is no dependency of pyproject.toml; "
                "name it among the dependencies, or in RESERVED_MODULES with its reason."
            )
    return errors


# ------------------------------------------------------------------ the whole check


def read_texts(root: Path, files: list[str]) -> dict[str, str]:
    """The text files among ``files``."""
    texts = {}
    for rel in files:
        try:
            text = (root / rel).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "\0" not in text:
            texts[rel] = text
    return texts


def collect(
    root: Path, texts: dict[str, str] | None = None
) -> tuple[list[Violation], dict[str, ModuleInfo]]:
    """Every violation of the rule, and the index of the code (for the rows done)."""
    files = tracked_files(root)
    texts = read_texts(root, files) if texts is None else texts
    found: list[Violation] = list(check_paths(files))
    modules: dict[str, ModuleInfo] = {}
    fixtures: set[str] = set()
    for rel, text in texts.items():
        module = module_name(rel)
        if module and rel.startswith(CHECKED_ROOTS):
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            modules[module] = ModuleInfo(rel, tree)
            if rel.endswith("conftest.py"):
                fixtures.update(
                    n.name
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and "fixture" in _decorators(n)
                )
    index_code(modules)
    for module, info in modules.items():
        found += check_definitions(info.rel, module, info.tree, fixtures)
    for rel, text in texts.items():
        if is_history(rel) or rel in INSTRUMENTS:
            continue
        found += check_text(rel, text)
    found += check_rule3(modules)
    return found, modules


@dataclass
class Outcome:
    failures: list[str] = field(default_factory=list)
    covered: dict[str, list[tuple[Row, Violation]]] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)  # task -> rows not done


def _file_aliases(rows: list[Row]) -> dict[str, set[str]]:
    """Each new path of a renamed, split or moved file, with its old paths."""
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.kind == "reintroduce" or ":" in row.old or "#" in row.old:
            continue
        olds = {row.old}
        if "/" not in row.old and not row.old.startswith("M-"):
            olds.add(_module_path(row.old))
        for new in re.split(r"[,\s]+", row.new):
            if not new:
                continue
            news = {new, _module_path(new)} if "/" not in new else {new}
            for path in news:
                aliases[path] |= olds
    return aliases


def _module_path(module: str) -> str:
    parts = module.split(".")
    root = "src/" if parts[0] == "estel" else ""
    return root + "/".join(parts) + ".py"


def evaluate(violations: list[Violation], rows: list[Row], *, strict: bool = False) -> Outcome:
    outcome = Outcome()
    by_old: dict[str, list[Row]] = defaultdict(list)
    by_bare: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_old[row.old].append(row)
        if row.old and not row.old.startswith("M-") and "#" not in row.old:
            by_bare[row.bare].append(row)
    aliases = _file_aliases(rows)

    def candidates(violation: Violation) -> list[Row]:
        keys = {violation.key}
        if "#" in violation.key:
            path, word = violation.key.split("#", 1)
            keys |= {f"{old}#{word}" for old in aliases.get(path, ())}
        found = [r for k in keys for r in by_old.get(k, [])]
        # A definition, a module or a file is covered by its own row only, so that no new name
        # breaks the rule; a row also covers its name where the text uses it.
        if "#" in violation.key:
            for name in violation.names:
                found += by_bare.get(name, [])
        return found

    for violation in violations:
        rows_for = candidates(violation)
        reintroduced = [r for r in rows_for if r.kind == "reintroduce" and r.done]
        live = [r for r in rows_for if r.kind != "reintroduce" and not r.done]
        if reintroduced and not live:
            continue
        if live:
            row = live[0]
            outcome.covered.setdefault(row.task, []).append((row, violation))
            continue
        done = [r for r in rows_for if r.kind != "reintroduce" and r.done]
        if done:
            row = done[0]
            outcome.failures.append(
                f"{violation.where}: {violation.detail}. Its row is done ({row.task}: "
                f"{row.old} -> {row.new}): the old name occurs again. Rename it as the row says"
                + (
                    "; to give the old name a new meaning, add a reintroduce row (rule 6)."
                    if row.kind == "rename"
                    else "."
                )
            )
            continue
        outcome.failures.append(
            f"{violation.where}: {violation.rule} (Part I, §2.4.1): {violation.detail}. "
            "Follow the rule; a name of `main` that a task of phase P7 renames needs a row "
            f"in {TABLE} with that task."
        )
    for row in rows:
        if not row.done:
            outcome.pending[row.task] = outcome.pending.get(row.task, 0) + 1
    if strict:
        for row in rows:
            if not row.done:
                outcome.failures.append(
                    f"{TABLE}: the row {row.old} -> {row.new} ({row.task}) is not done: "
                    "the tool is strict once every task of phase P7 is merged (T7.12)."
                )
    return outcome


def check_done_rows(root: Path, rows: list[Row], files: list[str]) -> list[str]:
    """A row done: its old module or file is gone, and a reintroduced name is in its module."""
    errors = []
    tracked = set(files)
    for row in rows:
        if not row.done or ":" in row.old or "#" in row.old or row.old.startswith("M-"):
            continue
        path = row.old if "/" in row.old else _module_path(row.old)
        package = path.removesuffix(".py") + "/__init__.py"
        exists = path in tracked or package in tracked
        if row.kind == "reintroduce":
            continue
        if exists and not any(
            r.kind == "reintroduce" and r.done and r.old == row.old for r in rows
        ):
            errors.append(
                f"{path}: its row is done ({row.task}: {row.old} -> {row.new}), but the file "
                "still exists: rename it as the row says."
            )
    for row in rows:
        if row.kind == "reintroduce" and row.done and ":" in row.old:
            module, qualname = row.old.split(":", 1)
            rel = _module_path(module)
            if rel not in tracked:
                rel = rel.removesuffix(".py") + "/__init__.py"
            text = (root / rel).read_text() if rel in tracked else ""
            name = qualname.rsplit(".", 1)[-1]
            if not re.search(
                rf"^\s*(class|def)\s+{re.escape(name)}\b|^{re.escape(name)}\s*[:=]",
                text,
                re.MULTILINE,
            ):
                errors.append(
                    f"{TABLE}: {row.old} is reintroduced, but `{name}` is not defined in "
                    f"{module}: define it there, the module the row names (rule 6)."
                )
    return errors


BACKTICKS_RE = re.compile(r"`+([^`]+)`+")
MATHS_ID_SCAN_RE = re.compile(r"(?<![\w-])M-[A-Z0-9]+(?:-[A-Z0-9]+)*(?![\w-])")


def _is_distinctive(name: str) -> bool:
    """A name that cannot be an English word: an underscore, or an inner capital letter."""
    return "_" in name or any(c.isupper() for c in name.lstrip("_")[1:])


def _done_message(where: str, found: str, row: Row, what: str = "occurs") -> str:
    return (
        f"{where}: `{found}` {what}, but its row is done ({row.task}: {row.old} -> {row.new}). "
        "Rename it as the row says"
        + (
            "; to give the old name a new meaning, add a reintroduce row (rule 6)."
            if row.kind == "rename"
            else "."
        )
    )


def check_done_occurrences(
    rows: list[Row], texts: dict[str, str], modules: dict[str, ModuleInfo]
) -> list[str]:
    """A row done: its old name occurs nowhere outside history, matched by what it names.

    In the code, a definition is matched where it is defined (its module, or the modules that
    module became), imported from there, or read through that module; a module where it is
    imported. Elsewhere (prose, comments, strings): its qualified forms, and its bare name when
    no other definition has that name, in backticks, or anywhere if it cannot be an English
    word. A maths ID anywhere. A done reintroduce row lets the name back as defined in the
    module it names, and imported from there; a definition of it elsewhere still fails.
    """
    pending = {r.bare for r in rows if not r.done and r.kind != "reintroduce"}
    back = {r.old for r in rows if r.kind == "reintroduce" and r.done}
    renamed_modules: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if ":" not in row.old and "#" not in row.old and "/" not in row.old:
            renamed_modules[row.old] |= set(row.new.replace(",", " ").split())
    # Every bare name defined in the code (definitions and modules), with where.
    defined: dict[str, set[str]] = defaultdict(set)
    for module, info in modules.items():
        defined[module.rsplit(".", 1)[-1]].add(module)
        for qualname in info.qualnames:
            defined[qualname.rsplit(".", 1)[-1]].add(module)

    errors: list[str] = []
    ids: dict[str, Row] = {}
    anywhere: dict[str, Row] = {}  # prose forms matched anywhere
    quoted: dict[str, Row] = {}  # prose forms matched in backticks only
    for row in rows:
        if not row.done or row.kind == "reintroduce" or "#" in row.old:
            continue
        if row.old.startswith("M-"):
            ids[row.old] = row
            continue
        lifted = row.old in back
        if ":" in row.old:
            module, qualname = row.old.split(":", 1)
            homes = {module} | renamed_modules.get(module, set())
            name = qualname.rsplit(".", 1)[-1]
            errors += _done_in_code(row, homes, qualname, modules, lifted)
            forms = {f"{h.rsplit('.', 1)[-1]}.{qualname}" for h in homes}
            forms |= {f"{h}.{qualname}" for h in homes}
            if "." in qualname:
                forms.add(qualname)
        elif "/" in row.old:
            homes, name = {row.old}, row.bare
            forms = {row.old, row.old.rsplit("/", 1)[-1]}
        else:
            homes, name = {row.old}, row.bare
            parts = row.old.split(".")
            forms = {".".join(parts[i:]) for i in range(len(parts) - 1)}
            forms |= {"/".join(parts[i:]) + ".py" for i in range(len(parts))}
            if not lifted:
                for info in modules.values():
                    lines = sorted(
                        {
                            line
                            for imported, line in info.imports
                            if imported == row.old or imported.startswith(row.old + ".")
                        }
                    )
                    errors += [
                        _done_message(f"{info.rel}:{line}", row.old, row, "is imported")
                        for line in lines
                    ]
        if lifted:
            continue
        for form in forms:
            anywhere.setdefault(form, row)
        ambiguous = name in pending or any(m not in homes for m in defined.get(name, ()))
        if not ambiguous:
            (anywhere if _is_distinctive(name) else quoted).setdefault(name, row)
    errors += _done_in_prose(texts, ids, anywhere, quoted)
    errors += _reintroduced_elsewhere(rows, modules, renamed_modules)
    return errors


def _done_in_code(
    row: Row, homes: set[str], qualname: str, modules: dict[str, ModuleInfo], lifted: bool
) -> list[str]:
    """A definition row done: not defined in its modules, not imported from them, nor read
    through them (all allowed again once its reintroduce row is done)."""
    if lifted:
        return []
    errors = []
    name = qualname.rsplit(".", 1)[-1]
    for home in sorted(homes):
        info = modules.get(home)
        if info and qualname in info.qualnames:
            errors.append(
                _done_message(f"{info.rel}:{info.qualnames[qualname]}", name, row, "is defined")
            )
    if "." in qualname:
        return errors  # a method or a field: matched where it is defined, and in prose
    for info in modules.values():
        for source, used, line in info.uses:
            if used == name and source in homes:
                errors.append(_done_message(f"{info.rel}:{line}", f"{source}.{name}", row))
    return errors


def _reintroduced_elsewhere(
    rows: list[Row], modules: dict[str, ModuleInfo], renamed_modules: dict[str, set[str]]
) -> list[str]:
    """A name reintroduced (rule 6) is defined in the module its row names, and nowhere else."""
    errors = []
    for row in rows:
        if row.kind != "reintroduce" or not row.done or ":" not in row.old:
            continue
        module, qualname = row.old.split(":", 1)
        homes = {module} | renamed_modules.get(module, set())
        for other, info in modules.items():
            if other not in homes and qualname in info.qualnames:
                errors.append(
                    f"{info.rel}:{info.qualnames[qualname]}: `{qualname}` is defined here, but "
                    f"its reintroduce row ({row.task}) brings it back in {module} only (rule 6): "
                    "use that one, or another name."
                )
    return errors


def _alternation(names: dict[str, Row]) -> re.Pattern[str] | None:
    if not names:
        return None
    ordered = sorted(names, key=len, reverse=True)
    return re.compile(r"(?<![\w.\-/])(" + "|".join(re.escape(n) for n in ordered) + r")(?![\w])")


def _done_in_prose(
    texts: dict[str, str], ids: dict[str, Row], anywhere: dict[str, Row], quoted: dict[str, Row]
) -> list[str]:
    errors = []
    any_re, quoted_re = _alternation(anywhere), _alternation({**anywhere, **quoted})
    ids_re = _alternation(ids)
    for rel, text in texts.items():
        if is_history(rel) or rel in INSTRUMENTS:
            continue
        skip = _skipped_lines(rel, text.split("\n"))
        for number, segment, is_code in _segments(rel, text):
            if number in skip or is_code:
                continue
            segment = segment.replace("\\_", "_")
            found: dict[str, Row] = {}
            if ids_re:
                found |= {m.group(1): ids[m.group(1)] for m in ids_re.finditer(segment)}
            if any_re:
                found |= {m.group(1): anywhere[m.group(1)] for m in any_re.finditer(segment)}
            if quoted_re:
                for span in BACKTICKS_RE.findall(segment):
                    for m in quoted_re.finditer(span):
                        found[m.group(1)] = {**anywhere, **quoted}[m.group(1)]
            for form, row in found.items():
                errors.append(_done_message(f"{rel}:{number}", form, row))
    return errors


def run(root: Path, *, report: bool = False, strict: bool = False) -> tuple[int, list[str]]:
    lines: list[str] = []
    errors = [f"check_retired_names: {hit.message()}" for hit in check(root)]
    rows, table_errors = load_table(root)
    errors += table_errors
    errors += check_contracts(root)
    files = tracked_files(root)
    texts = read_texts(root, files)
    violations, modules = collect(root, texts)
    outcome = evaluate(violations, rows, strict=strict)
    errors += outcome.failures
    errors += check_done_rows(root, rows, files)
    reported = {line.split(": ", 1)[0] for line in outcome.failures}
    errors += [
        e
        for e in check_done_occurrences(rows, texts, modules)
        if e.split(": ", 1)[0] not in reported
    ]
    for task in sorted(outcome.pending, key=lambda t: int(t.split(".")[1])):
        covered = outcome.covered.get(task, [])
        lines.append(
            f"check_retired_names: {task}: {outcome.pending[task]} rows to rename, "
            f"{len(covered)} occurrences reported"
        )
        if report:
            for row, violation in covered:
                lines.append(f"  {violation.where}: {violation.detail} [{row.old} -> {row.new}]")
    return (1 if errors else 0), errors + lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    parser.add_argument("--report", action="store_true", help="list every reported violation")
    parser.add_argument("--strict", action="store_true", help="fail while a row is not done")
    args = parser.parse_args(argv)
    status, lines = run(args.root, report=args.report, strict=args.strict)
    errors = [line for line in lines if not line.startswith(("check_retired_names: T7", "  "))]
    for line in errors:
        print(
            line if line.startswith("check_retired_names") else f"check_retired_names: {line}",
            file=sys.stderr,
        )
    for line in lines[len(errors) :]:
        print(line)
    if status == 0:
        print(
            f"check_retired_names: OK, none of the {len(RETIRED)} retired names is used, and "
            "every name follows Part I, §2.4.1 or has its row in the table."
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
