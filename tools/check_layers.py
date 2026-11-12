"""AST checks of the code rules of Part I, §2.4 that import-linter cannot express.

Everywhere in ``src/estel/``:

* **No class-name comparisons.** ``type(x).__name__`` or ``x.__class__.__name__`` compared
  with anything (``==``, ``!=``, ``in``, ``is``, ...). Use ``isinstance`` or a capability the
  object declares.
* **No assignment to ``__class__``**, by assignment or through ``setattr``. Compose instead.

In L1 and L2 (``data``, ``wavelets``, ``scattering``, ``statistics``, ``synthesis``,
``functional``), no in-place array mutation (§3.1):

* subscript assignment ``x[...] = ...`` (also inside tuple targets and ``for`` targets), and
  augmented subscript assignment ``x[...] += ...``;
* an ``out=`` keyword argument;
* a call to a method whose name ends in ``_`` (``.add_()``, ``.copy_()``), dunders excepted;
* an ``inplace=`` keyword argument other than ``inplace=False``.

The checks are syntactic, so they cannot tell an array from a list or a dict: in L1 and L2,
build containers with literals and comprehensions as well.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = Path("src/estel")
FUNCTIONAL_LAYERS = ("data", "wavelets", "scattering", "statistics", "synthesis", "functional")

CLASS_NAME = (
    "compares a class name as a string (`{expr}`). This breaks on subclasses. Use "
    "`isinstance(x, Cls)`, or better a capability the object declares (Part I, §2.4)."
)
CLASS_ASSIGN = (
    "assigns to `__class__`, so neither readers nor tools know what the object is. Compose "
    "instead: build an object of the right class that holds the parts (Part I, §2.4)."
)
SUBSCRIPT_ASSIGN = (
    "writes into an array in place (`{expr}[...] = ...`), which JAX forbids and which breaks "
    "second-order differentiation. Compute the values, collect them in a list and `stack` "
    "them, or place them with the out-of-place primitive `scatter_to_grid` (Part I, §3.1.1). "
    "For a dict or a list, build it with a literal or a comprehension."
)
SUBSCRIPT_AUGASSIGN = (
    "updates an array in place (`{expr}[...] {op}= ...`). Compute a new array instead "
    "(`y = x {op} ...`), or collect the values in a list and `stack` them (Part I, §3.1.1)."
)
OUT_KEYWORD = (
    "passes `out=` to `{expr}`, which writes into an existing array. Drop `out=` and use the "
    "returned value (Part I, §3.1, rule 1)."
)
INPLACE_METHOD = (
    "calls `.{name}()`, an in-place method. Call the out-of-place L0 primitive and use the "
    "value it returns (Part I, §3.1, rule 1)."
)
INPLACE_KEYWORD = (
    "passes `inplace={value}` to `{expr}`. Remove it and use the returned value: data objects "
    "are never modified, operations return new ones (`with_array`) (Part I, §3.1.1)."
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col + 1}: {self.message}"


def _is_class_name(node: ast.expr) -> bool:
    """``type(x).__name__`` or ``x.__class__.__name__``."""
    if not (isinstance(node, ast.Attribute) and node.attr == "__name__"):
        return False
    owner = node.value
    if isinstance(owner, ast.Attribute) and owner.attr == "__class__":
        return True
    return (
        isinstance(owner, ast.Call)
        and isinstance(owner.func, ast.Name)
        and owner.func.id == "type"
        and len(owner.args) == 1
        and not owner.keywords
    )


def _is_setattr_class(node: ast.Call) -> bool:
    """``setattr(x, "__class__", ...)`` or ``object.__setattr__(x, "__class__", ...)``."""
    func = node.func
    is_setattr = (isinstance(func, ast.Name) and func.id == "setattr") or (
        isinstance(func, ast.Attribute) and func.attr == "__setattr__"
    )
    if not is_setattr:
        return False
    return any(isinstance(arg, ast.Constant) and arg.value == "__class__" for arg in node.args)


def _flatten_targets(target: ast.expr) -> list[ast.expr]:
    if isinstance(target, ast.Tuple | ast.List):
        return [leaf for elt in target.elts for leaf in _flatten_targets(elt)]
    if isinstance(target, ast.Starred):
        return _flatten_targets(target.value)
    return [target]


class _Checker(ast.NodeVisitor):
    def __init__(self, path: Path, functional: bool) -> None:
        self.path = path
        self.functional = functional
        self.violations: list[Violation] = []

    def _report(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        self.violations.append(Violation(self.path, line, col, message))

    # ---------------------------------------------------------------- everywhere

    def visit_Compare(self, node: ast.Compare) -> None:
        for operand in (node.left, *node.comparators):
            if _is_class_name(operand):
                self._report(node, CLASS_NAME.format(expr=ast.unparse(operand)))
        self.generic_visit(node)

    def _check_targets(self, targets: list[ast.expr]) -> None:
        for target in (leaf for t in targets for leaf in _flatten_targets(t)):
            if isinstance(target, ast.Attribute) and target.attr == "__class__":
                self._report(target, CLASS_ASSIGN)
            elif self.functional and isinstance(target, ast.Subscript):
                self._report(target, SUBSCRIPT_ASSIGN.format(expr=ast.unparse(target.value)))

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_targets([node.target])
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target = node.target
        if isinstance(target, ast.Attribute) and target.attr == "__class__":
            self._report(target, CLASS_ASSIGN)
        elif self.functional and isinstance(target, ast.Subscript):
            op = _OPERATORS.get(type(node.op), "?")
            self._report(target, SUBSCRIPT_AUGASSIGN.format(expr=ast.unparse(target.value), op=op))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_targets([node.target])
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._check_targets([node.target])
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._check_targets([i.optional_vars for i in node.items if i.optional_vars is not None])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_setattr_class(node):
            self._report(node, CLASS_ASSIGN)
        if self.functional:
            callee = ast.unparse(node.func)
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr.endswith("_")
                and not func.attr.startswith("__")
            ):
                self._report(node, INPLACE_METHOD.format(name=func.attr))
            for keyword in node.keywords:
                if keyword.arg == "out":
                    self._report(keyword, OUT_KEYWORD.format(expr=callee))
                elif keyword.arg == "inplace" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                ):
                    value = ast.unparse(keyword.value)
                    self._report(keyword, INPLACE_KEYWORD.format(value=value, expr=callee))
        self.generic_visit(node)


_OPERATORS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.MatMult: "@",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}


def is_functional_layer(rel: Path) -> bool:
    """Whether ``rel`` (relative to ``src/estel``) belongs to L1 or L2."""
    first = rel.parts[0]
    return first in FUNCTIONAL_LAYERS or (len(rel.parts) == 1 and rel.stem in FUNCTIONAL_LAYERS)


def check_source(source: str, path: Path, functional: bool) -> list[Violation]:
    """Violations in one module; ``functional`` enables the L1/L2 in-place rules."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        message = f"cannot be parsed ({error.msg}). Fix the syntax error first."
        return [Violation(path, error.lineno or 0, (error.offset or 1) - 1, message)]
    checker = _Checker(path, functional)
    checker.visit(tree)
    return sorted(checker.violations, key=lambda v: (v.line, v.col))


def check(root: Path) -> tuple[list[Violation], int]:
    """All violations under ``root/src/estel``, and the number of files checked."""
    package = root / PACKAGE
    files = sorted(package.rglob("*.py"))
    violations: list[Violation] = []
    for path in files:
        rel = path.relative_to(package)
        violations += check_source(
            path.read_text(encoding="utf-8"), path.relative_to(root), is_functional_layer(rel)
        )
    return violations, len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    root: Path = args.root

    if not (root / PACKAGE).is_dir():
        print(
            f"check_layers: {PACKAGE} not found under {root}. Pass the repository root "
            "with --root.",
            file=sys.stderr,
        )
        return 1
    violations, count = check(root)
    for violation in violations:
        print(f"check_layers: {violation}", file=sys.stderr)
    if violations:
        print(
            f"check_layers: {len(violations)} violation(s) of Part I, §2.4. The checks are "
            "syntactic and have no escape hatch: rewrite the code as each message says.",
            file=sys.stderr,
        )
        return 1
    print(f"check_layers: OK ({count} file{'' if count == 1 else 's'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
