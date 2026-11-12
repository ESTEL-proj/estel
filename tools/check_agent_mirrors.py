"""Check that every coding agent reads the same rules (Part I, §8.1).

* Every ``CLAUDE.md`` starts with the line ``@AGENTS.md`` and has an ``AGENTS.md`` beside it;
  every ``AGENTS.md`` has a ``CLAUDE.md`` beside it (the root and the nested directories).
* ``.github/copilot-instructions.md`` exists and is exactly the copy generated from the root
  ``AGENTS.md``.

``--fix`` regenerates the Copilot mirror from ``AGENTS.md``, then runs the checks.
``AGENTS.md`` and ``.github/`` are protected: commit the result only in a PR meant for
CODEOWNER review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COPILOT = Path(".github/copilot-instructions.md")
IMPORT_LINE = "@AGENTS.md"
SKIP_DIRS = {".git", ".venv", "node_modules", "target", "build", "dist"}
BANNER = (
    "<!-- GENERATED from AGENTS.md by `python tools/check_agent_mirrors.py --fix`. "
    "Do not edit: change AGENTS.md (protected) and regenerate. -->\n\n"
)


def copilot_mirror(agents_text: str) -> str:
    """The content ``.github/copilot-instructions.md`` must have."""
    return BANNER + agents_text


def _context_files(root: Path, name: str) -> list[Path]:
    found = []
    for path in root.rglob(name):
        rel = path.relative_to(root)
        if not any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            found.append(path)
    return sorted(found)


def check(root: Path) -> list[str]:
    errors = []
    agents = root / "AGENTS.md"
    if not agents.exists():
        return ["AGENTS.md is missing at the repository root. Restore it from main."]

    for claude in _context_files(root, "CLAUDE.md"):
        rel = claude.relative_to(root)
        first = (claude.read_text().splitlines() or [""])[0].strip()
        if first != IMPORT_LINE:
            errors.append(
                f"{rel} must start with the line `{IMPORT_LINE}` (found {first!r}). Put the "
                "shared rules in the AGENTS.md beside it and keep only Claude-specific notes "
                "after the import."
            )
        if not (claude.parent / "AGENTS.md").exists():
            errors.append(
                f"{rel} has no AGENTS.md beside it. Add one with the local rules; "
                "CLAUDE.md only imports it."
            )
    for nested in _context_files(root, "AGENTS.md"):
        if not (nested.parent / "CLAUDE.md").exists():
            rel = nested.relative_to(root)
            errors.append(
                f"{rel} has no CLAUDE.md beside it. Add one whose first line is `{IMPORT_LINE}`."
            )

    mirror = root / COPILOT
    if not mirror.exists():
        errors.append(
            f"{COPILOT} is missing. Generate it with `python tools/check_agent_mirrors.py --fix`."
        )
    elif mirror.read_text() != copilot_mirror(agents.read_text()):
        errors.append(
            f"{COPILOT} differs from AGENTS.md. Do not edit it by hand: change AGENTS.md and "
            "run `python tools/check_agent_mirrors.py --fix`."
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    parser.add_argument(
        "--fix",
        action="store_true",
        help=f"regenerate {COPILOT} from AGENTS.md, then check",
    )
    args = parser.parse_args(argv)
    root: Path = args.root

    agents = root / "AGENTS.md"
    if args.fix and agents.exists():
        target = root / COPILOT
        expected = copilot_mirror(agents.read_text())
        if not target.exists() or target.read_text() != expected:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(expected)
            print(f"check_agent_mirrors: wrote {COPILOT}.")

    errors = check(root)
    for error in errors:
        print(f"check_agent_mirrors: {error}", file=sys.stderr)
    if not errors:
        print("check_agent_mirrors: OK.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
