"""Check that every coding agent reads the same rules (Part I, §8.1).

* Every ``CLAUDE.md`` starts with the line ``@AGENTS.md`` and has an ``AGENTS.md`` beside it;
  every ``AGENTS.md`` has a ``CLAUDE.md`` beside it (the root and the nested directories).
* ``.github/copilot-instructions.md`` is exactly the copy generated from the root
  ``AGENTS.md`` by ``--write``.

``--allow-missing-mirror`` accepts an absent Copilot mirror (a mirror that exists must still
match). ``make check`` passes it until task T0.3 creates the mirror; removing it then is part
of that task. ``--write`` regenerates the mirror; ``.github/`` is protected, so commit the
result only in a PR meant for CODEOWNER review.
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
    "<!-- GENERATED from AGENTS.md by `python tools/check_agent_mirrors.py --write`. "
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


def check(root: Path, allow_missing_mirror: bool) -> list[str]:
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
        if not allow_missing_mirror:
            errors.append(
                f"{COPILOT} is missing. Generate it with "
                "`python tools/check_agent_mirrors.py --write`."
            )
    elif mirror.read_text() != copilot_mirror(agents.read_text()):
        errors.append(
            f"{COPILOT} differs from AGENTS.md. Do not edit it by hand: change AGENTS.md and "
            "run `python tools/check_agent_mirrors.py --write`."
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    parser.add_argument("--write", action="store_true", help="regenerate the Copilot mirror")
    parser.add_argument(
        "--allow-missing-mirror",
        action="store_true",
        help=f"accept an absent {COPILOT} (until task T0.3)",
    )
    args = parser.parse_args(argv)
    root: Path = args.root

    if args.write:
        target = root / COPILOT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(copilot_mirror((root / "AGENTS.md").read_text()))
        print(f"check_agent_mirrors: wrote {COPILOT}.")
        return 0

    errors = check(root, args.allow_missing_mirror)
    for error in errors:
        print(f"check_agent_mirrors: {error}", file=sys.stderr)
    if not errors:
        note = ""
        if args.allow_missing_mirror and not (root / COPILOT).exists():
            note = f" ({COPILOT} absent, accepted with --allow-missing-mirror)"
        print(f"check_agent_mirrors: OK{note}.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
