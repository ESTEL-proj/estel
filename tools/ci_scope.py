"""Whether a change needs the test suites in tier 0 (task T0.7; Part I, §6.1).

A change whose files are all under ``docs/``, ``.github/`` or ``changes/`` runs only the check
tools: lint, types, layer checks, manifest, mirrors, maths IDs, workflows. Any other file, a
diff that cannot be computed, or no ``--base`` at all (a push to the default branch) runs the
whole suite. The required jobs run in every case, so each still reports a result.

A file under ``docs/`` that a test reads is a test input, and changing it runs the whole suite
(``tests/backend/test_protocol.py`` reads ``docs/maths.md``). The test inputs are found in the
tests themselves, so a new test that reads a document needs no change here: every ``docs/...``
path written in a ``tests/**/*.py`` file, as one string (``"docs/maths.md"``) or as string
parts joined with ``/`` (``REPO / "docs" / "maths.md"``). Paths of fake files that tests write
in a temporary repository count too: they name no file of the repository, so no change matches
them. Whether the file exists does not matter, so deleting a test input also runs the tests.

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
TESTS = Path("tests")


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


def docs_read_by_tests(root: Path) -> set[str]:
    """The paths under ``docs/`` that ``tests/**/*.py`` under ``root`` names."""
    found: set[str] = set()
    for source in sorted((root / TESTS).rglob("*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            parts = _path_parts(node) if isinstance(node, ast.expr) else None
            if not parts:
                continue
            path = "/".join(part.strip("/") for part in parts)
            if path.startswith("docs/"):
                found.add(path)
    return found


def needs_tests(paths: Iterable[str], inputs: Iterable[str] = ()) -> bool:
    """True unless every path is under one of ``NO_TEST_PREFIXES`` and none is a test input.

    An empty diff runs the tests.
    """
    paths = list(paths)
    inputs = set(inputs)
    return (
        not paths
        or not all(path.startswith(NO_TEST_PREFIXES) for path in paths)
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
        inputs = docs_read_by_tests(args.root)
        tests = needs_tests(paths, inputs)
        prefixes = ", ".join(NO_TEST_PREFIXES)
        read = sorted(inputs.intersection(paths))
        if not paths:
            reason = "no changed file: the whole suite runs"
        elif read:
            reason = f"tests read {', '.join(read)}: the whole suite runs"
        elif tests:
            reason = f"{len(paths)} changed file(s), some outside {prefixes}: the whole suite runs"
        else:
            reason = f"every changed file is under {prefixes}: check tools only"
    print(f"ci_scope: {reason}", file=sys.stderr)
    print(f"tests={'true' if tests else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
