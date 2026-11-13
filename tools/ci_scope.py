"""Whether a change needs the test suites in tier 0 (task T0.7; Part I, §6.1).

A change whose files are all under ``docs/``, ``.github/`` or ``changes/`` runs only the check
tools: lint, types, layer checks, manifest, mirrors, maths IDs, workflows. Any other file, a
diff that cannot be computed, or no ``--base`` at all (a push to the default branch) runs the
whole suite. The required jobs run in every case, so each still reports a result.

Prints ``tests=true`` or ``tests=false``, the format of ``$GITHUB_OUTPUT``. Standard library
only: CI runs it with the runner's Python, before the environment is installed.

    python tools/ci_scope.py --base origin/main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

NO_TEST_PREFIXES = ("docs/", ".github/", "changes/")


def needs_tests(paths: Iterable[str]) -> bool:
    """True unless every path is under one of ``NO_TEST_PREFIXES``; an empty diff runs them."""
    paths = list(paths)
    return not paths or not all(path.startswith(NO_TEST_PREFIXES) for path in paths)


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
        tests = needs_tests(paths)
        prefixes = ", ".join(NO_TEST_PREFIXES)
        if not paths:
            reason = "no changed file: the whole suite runs"
        elif tests:
            reason = f"{len(paths)} changed file(s), some outside {prefixes}: the whole suite runs"
        else:
            reason = f"every changed file is under {prefixes}: check tools only"
    print(f"ci_scope: {reason}", file=sys.stderr)
    print(f"tests={'true' if tests else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
