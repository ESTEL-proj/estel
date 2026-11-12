"""Compute the version bump a pull request requires, and compare it with the declared one.

Rules (docs/policy/VERSIONING.md; Part I, §7.2):

* ``api/manifest.json``: an entry removed or changed -> major; an entry added -> minor;
* the geometry registry changed -> major (paths in ``GEOMETRY_REGISTRY``; none yet);
* the ``bench`` label -> minor;
* anything else -> patch;
* a changed ``tests/reference/`` needs a reference-counter bump with CODEOWNER approval, shown
  by the ``major`` label (Part I, §5.3). Before 1.0.0 nothing is gated by reference values,
  so it is only reported.

The declared bump is read from the news fragments the branch adds,
``changes/<PR>.<major|minor|patch>.md`` (towncrier). The tool fails if there is none, if one
is misnamed, or if the declared bump is below the required one.

Usage (CI passes the PR number and its labels)::

    python tools/version_impact.py --base origin/main --pr 12 --labels bench,major
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from api_manifest import MANIFEST, compare, load_manifest_text

REPO = Path(__file__).resolve().parents[1]
LEVELS = ("patch", "minor", "major")
FRAGMENT_RE = re.compile(r"^changes/(?P<pr>\d+)\.(?P<level>major|minor|patch)\.md$")
REFERENCE_DIR = "tests/reference/"
# Files that define the geometry registry. The registry does not exist yet (phase P2);
# add its path here in the PR that creates it.
GEOMETRY_REGISTRY: tuple[str, ...] = ()


@dataclass
class Impact:
    required: str = "patch"
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def require(self, level: str, reason: str) -> None:
        if LEVELS.index(level) > LEVELS.index(self.required):
            self.required = level
        self.reasons.append(f"{level}: {reason}")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"version_impact: `git {' '.join(args)}` failed: {result.stderr.strip()}. "
            "Fetch the base branch (CI checks out with fetch-depth: 0) and pass it with --base."
        )
    return result.stdout


def show(root: Path, rev: str, path: str) -> str:
    """The content of ``path`` at ``rev``, or an empty string if it does not exist there."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def package_major(pyproject_text: str) -> int:
    version = tomllib.loads(pyproject_text).get("project", {}).get("version", "0.0.0")
    return int(str(version).split(".")[0])


def assess(root: Path, base: str, head: str, pr: int | None, labels: set[str]) -> Impact:
    impact = Impact()
    merge_base = git(root, "merge-base", base, head).strip()
    changes = [
        line.split("\t")
        for line in git(root, "diff", "--name-status", "--no-renames", merge_base, head)
        .strip()
        .splitlines()
        if line
    ]
    changed = {path for _, path in changes}
    added = {path for status, path in changes if status == "A"}

    # Public API.
    manifest = str(MANIFEST)
    if manifest in changed:
        old = load_manifest_text(show(root, merge_base, manifest), f"{manifest} at {base}")
        new = load_manifest_text(show(root, head, manifest), f"{manifest} at {head}")
        diff = compare(old, new)
        for name in diff.removed:
            impact.require("major", f"{name} removed from {manifest}")
        for name in diff.changed:
            impact.require("major", f"{name} changed in {manifest}")
        for name in diff.added:
            impact.require("minor", f"{name} added to {manifest}")

    # Geometry registry.
    for path in sorted(changed & set(GEOMETRY_REGISTRY)):
        impact.require("major", f"geometry registry {path} changed")

    # Benchmark-driven minors.
    if "bench" in labels:
        impact.require("minor", "the `bench` label is set")

    # Reference values.
    reference = sorted(p for p in changed if p.startswith(REFERENCE_DIR))
    if reference:
        shown = ", ".join(reference[:5]) + (" ..." if len(reference) > 5 else "")
        if package_major(show(root, head, "pyproject.toml")) == 0:
            impact.notes.append(
                f"{REFERENCE_DIR} changed ({shown}). Before 1.0.0 reference values gate "
                "nothing; the path is protected, so a CODEOWNER still reviews it."
            )
        elif "major" not in labels:
            impact.errors.append(
                f"{REFERENCE_DIR} changed ({shown}). This needs a bump of "
                "STATISTICS_REFERENCE or SYNTHESIS_REFERENCE and CODEOWNER approval, shown by "
                "the `major` label. If you did not mean to change reference values, revert "
                "these files and open an issue (AGENTS.md, golden rule 1)."
            )

    # Declared bump, from the news fragments.
    declared: str | None = None
    for path in sorted(added):
        if not path.startswith("changes/") or Path(path).name.startswith("."):
            continue
        match = FRAGMENT_RE.match(path)
        if match is None:
            impact.errors.append(
                f"{path} is not a valid news fragment. Name it "
                "changes/<PR number>.<major|minor|patch>.md."
            )
            continue
        if pr is not None and int(match["pr"]) != pr:
            impact.errors.append(
                f"{path} names PR {match['pr']}, but this is PR {pr}. "
                f"Rename it to changes/{pr}.{match['level']}.md."
            )
        level = match["level"]
        if declared is None or LEVELS.index(level) > LEVELS.index(declared):
            declared = level

    if declared is None:
        impact.errors.append(
            f"No news fragment. Add changes/{pr if pr is not None else '<PR>'}."
            f"{impact.required}.md describing the change in one or two sentences."
        )
    elif LEVELS.index(declared) < LEVELS.index(impact.required):
        impact.errors.append(
            f"The fragment declares a {declared} change, but the change requires "
            f"{impact.required}. Rename the fragment to <PR>.{impact.required}.md, or undo the "
            "change that requires it."
        )
    return impact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="base branch or commit, e.g. origin/main")
    parser.add_argument("--head", default="HEAD", help="head commit (default: HEAD)")
    parser.add_argument("--pr", type=int, default=None, help="pull-request number")
    parser.add_argument("--labels", default="", help="comma-separated PR labels")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    impact = assess(args.root, args.base, args.head, args.pr, labels)

    print(f"version_impact: required bump: {impact.required}")
    for reason in impact.reasons:
        print(f"  - {reason}")
    for note in impact.notes:
        print(f"version_impact: note: {note}")
    for error in impact.errors:
        print(f"version_impact: {error}", file=sys.stderr)
    return 1 if impact.errors else 0


if __name__ == "__main__":
    sys.exit(main())
