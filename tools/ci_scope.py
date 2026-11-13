"""Whether a change needs the test suites in tier 0 (task T0.7; Part I, §6.1).

A change whose files are all in the no-test scope runs only the check tools: lint, types, layer
checks, manifest, mirrors, maths IDs, workflows. The no-test scope is

- every file under ``docs/``, ``.github/`` or ``changes/``;
- ``.gitignore``, ``LICENSE``, ``NOTICE`` and ``README.md`` at the root;
- every file named ``AGENTS.md`` or ``CLAUDE.md``, at any depth.

Any other file (``Makefile``, ``pyproject.toml``, ``uv.lock``, ``.python-version``,
``api/manifest.json``, the code, the tests...), a diff that cannot be computed, or no ``--base``
at all (a push to the default branch) runs the whole suite. The required jobs run in every
case, so each still reports a result.

A no-test file that a test reads is a test input, and changing it runs the whole suite
(``tests/backend/test_protocol.py`` reads ``docs/maths.md``). The test inputs are found in the
tests themselves, so a new test that reads such a file needs no change here: every no-test path
written in a ``tests/**/*.py`` file, as one string (``"docs/maths.md"``) or as string parts
joined with ``/`` onto the repository root (``REPO / "src" / "estel" / "AGENTS.md"``). The
repository root is a name bound to an expression of ``__file__`` (or a name bound to one of
those), or ``REPO``, the name ``tests/tools/conftest.py`` exports. Tests also write fake files
with the same names in temporary repositories (an ``AGENTS.md`` for the mirror check, a
``README.md`` for the version check); those are not test inputs: a dictionary key (the files
given to ``write``) and a path joined onto any other base (``tmp_path / "AGENTS.md"``) name a
fake file. Whether a test input exists does not matter, so deleting one also runs the tests.

Prints ``tests=true`` or ``tests=false``, the format of ``$GITHUB_OUTPUT``. Standard library
only: CI runs it with the runner's Python, before the environment is installed.

    python tools/ci_scope.py --base origin/main
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

NO_TEST_PREFIXES = ("docs/", ".github/", "changes/")
NO_TEST_ROOT_FILES = frozenset({".gitignore", "LICENSE", "NOTICE", "README.md"})
NO_TEST_NAMES = frozenset({"AGENTS.md", "CLAUDE.md"})
NO_TEST_SCOPE = (
    f"{', '.join(NO_TEST_PREFIXES)}, {', '.join(sorted(NO_TEST_ROOT_FILES))} at the root, "
    f"{' and '.join(sorted(NO_TEST_NAMES))} at any depth"
)
TESTS = Path("tests")
# The repository root as tests/tools/conftest.py exports it.
REPOSITORY_NAMES = frozenset({"REPO"})


def is_no_test_path(path: str) -> bool:
    """True when changing ``path`` alone needs only the check tools (if no test reads it)."""
    return (
        path.startswith(NO_TEST_PREFIXES)
        or path in NO_TEST_ROOT_FILES
        or path.rsplit("/", 1)[-1] in NO_TEST_NAMES
    )


def _path_parts(node: ast.expr) -> list[str] | None:
    """The string parts of ``"a" / "b"`` or ``X / "a" / "b"``, or None for another expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        right = _path_parts(node.right)
        if right is None:
            return None
        left = _path_parts(node.left)
        return right if left is None else left + right
    return None


def _base(node: ast.expr) -> ast.expr:
    """The leftmost operand of ``X / "a" / "b"``."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        node = node.left
    return node


def _is_repository(node: ast.AST, names: set[str]) -> bool:
    """True when ``node`` uses ``__file__`` or a name bound to the repository root."""
    return any(
        isinstance(sub, ast.Name) and (sub.id == "__file__" or sub.id in names)
        for sub in ast.walk(node)
    )


def _repository_names(tree: ast.Module) -> set[str]:
    """``REPO`` and the names a module binds to the repository root (``Path(__file__)...``)."""
    bindings: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        bindings += [(target.id, value) for target in targets if isinstance(target, ast.Name)]
    names = set(REPOSITORY_NAMES)
    grew = True
    while grew:
        grew = False
        for name, value in bindings:
            if name not in names and _is_repository(value, names):
                names.add(name)
                grew = True
    return names


class _Paths(ast.NodeVisitor):
    """The paths a test module names, without the fake files of temporary repositories."""

    def __init__(self, repository: set[str]) -> None:
        self.repository = repository
        self.found: set[str] = set()

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.found.add(node.value.strip("/"))

    def visit_Dict(self, node: ast.Dict) -> None:
        # The keys are files a test writes (``write(tmp_path, {"AGENTS.md": ...})``).
        for value in node.values:
            self.visit(value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        parts = _path_parts(node)
        if parts is None:
            self.generic_visit(node)
            return
        base = _base(node)
        if isinstance(base, ast.Constant) or _is_repository(base, self.repository):
            self.found.add("/".join(part.strip("/") for part in parts))
        if not isinstance(base, ast.Constant):
            self.visit(base)


def inputs_read_by_tests(root: Path) -> set[str]:
    """The no-test paths that ``tests/**/*.py`` under ``root`` reads."""
    found: set[str] = set()
    for source in sorted((root / TESTS).rglob("*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        paths = _Paths(_repository_names(tree))
        paths.visit(tree)
        found |= paths.found
    return {path for path in found if is_no_test_path(path)}


def needs_tests(paths: Iterable[str], inputs: Iterable[str] = ()) -> bool:
    """True unless every path is in the no-test scope and none is a test input.

    An empty diff runs the tests.
    """
    paths = list(paths)
    inputs = set(inputs)
    return (
        not paths
        or not all(is_no_test_path(path) for path in paths)
        or any(path in inputs for path in paths)
    )


def changed_paths(root: Path, base: str, head: str) -> list[str]:
    """The files changed between the merge base of ``base`` and ``head``, and ``head``.

    Renames count as a deletion and an addition, so a file moved out of ``src/`` is a code
    change even when its new path is under ``docs/``.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}...{head}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ci_scope: `git diff {base}...{head}` failed: {result.stderr.strip()}. "
            "Fetch the base branch first (actions/checkout with fetch-depth: 0)."
        )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="base of the change; without it, the tests run")
    parser.add_argument("--head", default="HEAD", help="head of the change (default: HEAD)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    if args.base is None:
        tests = True
        reason = "no base to compare with: the whole suite runs"
    else:
        try:
            paths = changed_paths(args.root, args.base, args.head)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        inputs = inputs_read_by_tests(args.root)
        tests = needs_tests(paths, inputs)
        read = sorted(inputs.intersection(paths))
        if not paths:
            reason = "no changed file: the whole suite runs"
        elif read:
            reason = f"tests read {', '.join(read)}: the whole suite runs"
        elif tests:
            reason = f"{len(paths)} changed file(s), some outside the no-test scope: all tests run"
        else:
            reason = (
                f"every changed file is in the no-test scope ({NO_TEST_SCOPE}): check tools only"
            )
    print(f"ci_scope: {reason}", file=sys.stderr)
    print(f"tests={'true' if tests else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
