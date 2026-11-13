"""Compute the version bump a pull request requires, and compare it with the declared one.

Rules (docs/policy/VERSIONING.md; Part I, §7.2):

* ``api/manifest.json``: an entry removed or changed -> major; an entry added -> minor;
* the geometry registry changed -> major (paths in ``GEOMETRY_REGISTRY``; none yet);
* the ``bench`` label -> minor;
* anything else -> patch.

A PR whose changed files are all under ``.github/`` does not change the package: it needs no
news fragment and is reported as "no package change". ``.github/`` is protected, so such PRs,
Dependabot's included, still need CODEOWNER approval.

Reference counters (Part I, §5.3 and §7.2). ``STATISTICS_REFERENCE`` and
``SYNTHESIS_REFERENCE`` are integer literals in ``src/estel/_reference.py``; the tool reads them
on the base branch and on the PR, without importing estel. A PR that changes the references of
a counter must increase that counter by exactly one, and a PR that changes a counter must change
its references (``tests/reference/`` for ``STATISTICS_REFERENCE``; ``SYNTHESIS_REFERENCE`` has
no references yet). Before 1.0.0 a mismatch is only reported; from 1.0.0 it fails. No label is
involved: the bump is in the diff, and CODEOWNERS approves it. Counters absent on the base
branch count as 0.

The declared bump is read from the news fragments the branch adds,
``changes/<PR>.<major|minor|patch>.md`` (towncrier). The tool fails if there is none, if one
is misnamed, or if the declared bump is below the required one.

Usage (CI passes the PR number and its labels)::

    python tools/version_impact.py --base origin/main --pr 12 --labels bench
"""

from __future__ import annotations

import argparse
import ast
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
COUNTERS_FILE = "src/estel/_reference.py"
# Each reference counter and the directory of the references it versions. The synthesis
# references do not exist yet (after 1.0.0); set their directory in the PR that creates them.
COUNTERS: dict[str, str | None] = {
    "STATISTICS_REFERENCE": "tests/reference/",
    "SYNTHESIS_REFERENCE": None,
}
# Files that define the geometry registry. The registry does not exist yet (phase P2);
# add its path here in the PR that creates it.
GEOMETRY_REGISTRY: tuple[str, ...] = ()
# A PR that changes only files under this prefix does not change the package (Part I, §7.2).
NON_PACKAGE_PREFIX = ".github/"


@dataclass
class Impact:
    required: str = "patch"
    package_change: bool = True
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def require(self, level: str, reason: str) -> None:
        if LEVELS.index(level) > LEVELS.index(self.required):
            self.required = level
        self.reasons.append(f"{level}: {reason}")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
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


def read_counters(text: str, origin: str) -> tuple[dict[str, int], list[str]]:
    """The integer counters assigned at the top level of ``text``, and the problems found."""
    restore = f"Restore {COUNTERS_FILE} from main; each counter is a plain integer literal."
    try:
        tree = ast.parse(text, filename=origin)
    except SyntaxError as exc:
        return {}, [f"{origin} does not parse ({exc.msg}, line {exc.lineno}). {restore}"]
    values: dict[str, int] = {}
    problems: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in COUNTERS:
                continue
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)
            ):
                values[target.id] = value.value
            else:
                problems.append(
                    f"{origin}: {target.id} = {ast.unparse(value)} is not an integer literal. "
                    + restore
                )
    for name in COUNTERS:
        if name not in values and not any(name in p for p in problems):
            problems.append(f"{origin}: {name} is missing. {restore}")
    return values, problems


def check_counters(
    root: Path, merge_base: str, head: str, changed: set[str], impact: Impact
) -> None:
    """Apply the reference-counter rule of Part I, §7.2."""
    base_text = show(root, merge_base, COUNTERS_FILE)
    head_text = show(root, head, COUNTERS_FILE)
    if base_text:
        old, problems = read_counters(base_text, f"{COUNTERS_FILE} on the base branch")
        if problems:  # the base branch is what it is; compare with what can be read
            impact.notes.extend(problems)
    else:
        old = {}  # the counters did not exist yet: they were 0
    if not head_text:
        impact.errors.append(f"{COUNTERS_FILE} is missing. Restore it from main.")
        return
    new, problems = read_counters(head_text, COUNTERS_FILE)
    impact.errors.extend(problems)

    gated = package_major(show(root, head, "pyproject.toml")) >= 1
    for name, directory in COUNTERS.items():
        if name not in new:
            continue
        before, after = old.get(name, 0), new[name]
        references = sorted(p for p in changed if p.startswith(directory)) if directory else []
        where = directory or f"the references of {name}"
        mismatch: str | None = None
        if references and after != before + 1:
            shown = ", ".join(references[:5]) + (" ..." if len(references) > 5 else "")
            mismatch = (
                f"{where} changed ({shown}) but {name} went from {before} to {after}; it must "
                f"increase by exactly one, to {before + 1}, in {COUNTERS_FILE}, with CODEOWNER "
                "approval. If you did not mean to change reference values, revert these files "
                "and open an issue (AGENTS.md, golden rule 1)."
            )
        elif not references and after != before:
            if directory is None:
                mismatch = (
                    f"{name} changed from {before} to {after}, but {name} has no references "
                    f"yet. Restore it to {before} in {COUNTERS_FILE}; it is bumped together "
                    "with the synthesis references, once they exist."
                )
            else:
                mismatch = (
                    f"{name} changed from {before} to {after}, but {directory} did not change. "
                    f"A bump of {name} comes with the regenerated references, in the same PR: "
                    f"restore {name} to {before} in {COUNTERS_FILE}, or add the references."
                )
        if mismatch is None:
            if after != before:
                impact.notes.append(
                    f"{name} bumped from {before} to {after} with {where}; this needs "
                    "CODEOWNER approval."
                )
        elif gated:
            impact.errors.append(mismatch)
        else:
            impact.notes.append(f"{mismatch} (Before 1.0.0 this is only reported.)")


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

    # CI-only PRs (Part I, §7.2): nothing in the package changes, so no bump and no fragment.
    if changed and all(path.startswith(NON_PACKAGE_PREFIX) for path in changed):
        impact.package_change = False
        impact.notes.append(
            f"only {NON_PACKAGE_PREFIX} changed: no news fragment is needed. "
            f"{NON_PACKAGE_PREFIX} is protected, so this PR needs CODEOWNER approval."
        )
        if "bench" in labels:
            impact.notes.append(
                "the `bench` label is ignored: a PR that changes only "
                f"{NON_PACKAGE_PREFIX} cannot change performance. Remove the label."
            )
        return impact

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

    # Reference counters.
    check_counters(root, merge_base, head, changed, impact)

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

    if impact.package_change:
        print(f"version_impact: required bump: {impact.required}")
    else:
        print("version_impact: required bump: none (no package change)")
    for reason in impact.reasons:
        print(f"  - {reason}")
    for note in impact.notes:
        print(f"version_impact: note: {note}")
    for error in impact.errors:
        print(f"version_impact: {error}", file=sys.stderr)
    return 1 if impact.errors else 0


if __name__ == "__main__":
    sys.exit(main())
