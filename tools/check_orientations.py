"""Check that tests, benchmark cases and parity cases use L at most 4 (golden rule 13; issue #133).

The cost of the statistics grows fast with the number of orientations L (S4 has L³ of them). In
2D, including the 2D blocks of tensor products, tests, benchmark cases and parity cases use
L ≤ 4, the default. A larger L enters the repository only where a human approved it on an
issue, marked on its line with ``# allow-large-L: #<issue>``.

The tool reads, without importing or running them, the Python files of ``tests/``, ``bench/``
and ``tools/stl_parity/cases/``, and ``bench/cases.toml``: only the files git tracks (``git
ls-files``), so local copies such as ``bench/results/.work/`` are left out; outside a git
repository, every file of these directories. It fails on any of these whose value is an integer
literal above 4:

* a keyword argument ``L=`` (``STOperator(data, L=8)``);
* an assignment to a name ``L``: ``L = 8``, ``L: int = 8``, or the value at L's position in a
  tuple unpacking (``J, L = 5, 8``);
* a dictionary key ``"L"`` (``{"J": 2, "L": 8}``) or a TOML field ``L`` (``L = 8``);
* an entry of a ``pytest.mark.parametrize`` over a name ``L``, alone or among other names.

A list of integers, L per block of a tensor product (``L=[4, 2]``), is checked integer by
integer: a 1D block cannot exceed 2, so the limit of the 2D blocks applies to all. The tool does
not know the dimension, so a 1D L above 4 fails too; the library refuses it anyway. An L computed
from a variable is not detected: the written rule covers it.

A line that carries ``# allow-large-L: #<issue>`` passes. The marker without an issue number
fails, wherever it is.

The tool reads text and ASTs only; it never imports estel.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAX_L = 4
NAME = "L"

PYTHON_DIRS = ("tests", "bench", "tools/stl_parity/cases")
TOML_FILES = ("bench/cases.toml",)

MARKER_WORD = "allow-large-L"
MARKER_RE = re.compile(r"#\s*allow-large-L:\s*#\d+\b")

# A TOML field L (bare or quoted key, also inside an inline table) whose value is an integer or
# a one-line list.
TOML_FIELD_RE = re.compile(r"""(?:^|[{,])\s*(?:L|"L"|'L')\s*=\s*(\[[^\]]*\]|[+]?\d[\d_]*)""")
TOML_INT_RE = re.compile(r"[+]?\d[\d_]*")

ADVICE = (
    f"In 2D, tests, benchmark cases and parity cases use L ≤ {MAX_L}, the default (golden rule "
    f"13): use L ≤ {MAX_L}. If a human approved more on an issue, end the line with "
    "`# allow-large-L: #<issue>`."
)
MARKER_ADVICE = (
    "the marker `allow-large-L` names the issue where a human approved a larger L: write "
    "`# allow-large-L: #<issue>`, or remove it."
)


@dataclass(frozen=True)
class Hit:
    where: str  # path:line
    found: str  # what was found, as a short description

    def message(self) -> str:
        if self.found == MARKER_WORD:
            return f"{self.where}: {MARKER_ADVICE}"
        return f"{self.where}: {self.found} is above {MAX_L}. {ADVICE}"


def _large_ints(node: ast.expr) -> list[ast.Constant]:
    """The integer literals above MAX_L of ``node``: an integer, or a list or tuple of them."""
    items = node.elts if isinstance(node, ast.List | ast.Tuple) else [node]
    return [
        item
        for item in items
        if isinstance(item, ast.Constant)
        and isinstance(item.value, int)
        and not isinstance(item.value, bool)
        and item.value > MAX_L
    ]


def _parametrize_names(node: ast.expr) -> list[str] | None:
    """The names of a parametrize's first argument, ``"a, L"`` or ``("a", "L")``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [name.strip() for name in node.value.split(",")]
    if isinstance(node, ast.List | ast.Tuple):
        names = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if len(names) == len(node.elts) and all(isinstance(n, str) for n in names):
            return [str(n) for n in names]
    return None


def _is_parametrize(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "parametrize"


def _call_argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _parametrize_values(call: ast.Call) -> list[ast.expr]:
    """The values given to the name L by each entry of a parametrize."""
    names_node = _call_argument(call, 0, "argnames")
    entries = _call_argument(call, 1, "argvalues")
    names = _parametrize_names(names_node) if names_node is not None else None
    if names is None or NAME not in names or not isinstance(entries, ast.List | ast.Tuple):
        return []
    index, values = names.index(NAME), []
    for entry in entries.elts:
        fields: list[ast.expr]
        if isinstance(entry, ast.Call):
            # pytest.param(value, ..., id=...): its positional arguments are the entry.
            func = entry.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "param":
                continue
            fields = list(entry.args)
        elif len(names) == 1:
            fields = [entry]  # also a list, L per block: [4, 2]
        elif isinstance(entry, ast.List | ast.Tuple):
            fields = list(entry.elts)
        else:
            continue
        if len(fields) == len(names):
            values.append(fields[index])
    return values


def _assigned_values(target: ast.expr, value: ast.expr) -> list[ast.expr]:
    """The values an assignment ``target = value`` gives to the name L."""
    if isinstance(target, ast.Name):
        return [value] if target.id == NAME else []
    if (
        isinstance(target, ast.Tuple | ast.List)
        and isinstance(value, ast.Tuple | ast.List)
        and len(target.elts) == len(value.elts)
        and not any(isinstance(e, ast.Starred) for e in target.elts + value.elts)
    ):
        return [
            v for t, e in zip(target.elts, value.elts, strict=True) for v in _assigned_values(t, e)
        ]
    return []


def _python_candidates(tree: ast.AST) -> list[tuple[str, ast.expr]]:
    """Each value given to L in ``tree``, with the form that gives it."""
    found: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                found += [("assignment L = {}", v) for v in _assigned_values(target, node.value)]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            found += [("assignment L = {}", v) for v in _assigned_values(node.target, node.value)]
        elif isinstance(node, ast.keyword) and node.arg == NAME:
            found.append(("keyword argument L={}", node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == NAME:
                    found.append(('dictionary key "L": {}', value))
        elif isinstance(node, ast.Call) and _is_parametrize(node):
            found += [("parametrize entry L = {}", v) for v in _parametrize_values(node)]
    return found


def _comments(source: str) -> dict[int, str]:
    """The comment of each line of a Python source, by line number."""
    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string
    return comments


def _marker_hits(rel: str, comments: dict[int, str]) -> tuple[set[int], list[Hit]]:
    """The lines whose marker is valid, and a hit for each marker without an issue number."""
    valid, hits = set(), []
    for number, comment in sorted(comments.items()):
        if MARKER_WORD not in comment:
            continue
        if MARKER_RE.search(comment):
            valid.add(number)
        else:
            hits.append(Hit(f"{rel}:{number}", MARKER_WORD))
    return valid, hits


def check_python(source: str, rel: str) -> list[Hit]:
    # Only the comments holding the marker matter (_marker_hits), and tokenizing a file costs
    # as much as parsing it: a source without the marker word has none to read.
    comments = _comments(source) if MARKER_WORD in source else {}
    valid, hits = _marker_hits(rel, comments)
    seen: set[tuple[int, str]] = set()
    for form, value in _python_candidates(ast.parse(source, rel)):
        for literal in _large_ints(value):
            found = form.format(ast.unparse(value))
            if literal.lineno in valid or (literal.lineno, found) in seen:
                continue
            seen.add((literal.lineno, found))
            hits.append(Hit(f"{rel}:{literal.lineno}", found))
    return sorted(hits, key=lambda h: int(h.where.rsplit(":", 1)[1]))


def _split_toml_comment(line: str) -> tuple[str, str]:
    """A TOML line split into its code and its comment, outside quoted strings."""
    quote = ""
    for i, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "#":
            return line[:i], line[i:]
    return line, ""


def check_toml(source: str, rel: str) -> list[Hit]:
    hits = []
    for number, line in enumerate(source.splitlines(), 1):
        code, comment = _split_toml_comment(line)
        if MARKER_WORD in comment and not MARKER_RE.search(comment):
            hits.append(Hit(f"{rel}:{number}", MARKER_WORD))
            continue
        if MARKER_RE.search(comment):
            continue
        for match in TOML_FIELD_RE.finditer(code):
            value = match.group(1)
            if any(int(v.replace("_", "")) > MAX_L for v in TOML_INT_RE.findall(value)):
                hits.append(Hit(f"{rel}:{number}", f"TOML field L = {value}"))
    return hits


def _tracked(root: Path) -> list[str] | None:
    """The files git tracks under ``root``, if ``root`` is the top of a git repository."""
    git = ["git", "-C", str(root)]
    top = subprocess.run([*git, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        return None
    listed = subprocess.run(
        [*git, "ls-files", "-z", "--", *PYTHON_DIRS, *TOML_FILES],
        capture_output=True,
        text=True,
        check=True,
    )
    return [rel for rel in listed.stdout.split("\0") if rel]


def files(root: Path) -> list[Path]:
    """The files read: those git tracks, or every one outside a git repository."""
    tracked = _tracked(root)
    if tracked is None:
        paths: list[Path] = []
        for directory in PYTHON_DIRS:
            paths += sorted((root / directory).rglob("*.py"))
        return paths + [root / f for f in TOML_FILES if (root / f).exists()]
    wanted = [
        root / rel
        for rel in tracked
        if (rel.endswith(".py") and rel.startswith(tuple(d + "/" for d in PYTHON_DIRS)))
        or rel in TOML_FILES
    ]
    return sorted(path for path in wanted if path.is_file())


def check(root: Path) -> list[Hit]:
    hits = []
    for path in files(root):
        rel = path.relative_to(root).as_posix()
        source = path.read_text()
        hits += check_toml(source, rel) if path.suffix == ".toml" else check_python(source, rel)
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    hits = check(args.root)
    for hit in hits:
        print(f"check_orientations: {hit.message()}", file=sys.stderr)
    if not hits:
        n_files = len(files(args.root))
        print(f"check_orientations: OK, L ≤ {MAX_L} or approved in the {n_files} files read.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
