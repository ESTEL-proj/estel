"""Check the GitHub Actions workflows against review rule E5 (docs/agents/review_rules.md).

For every ``.github/workflows/*.yml`` (and ``*.yaml``):

* **Pinned actions.** Every ``uses:`` (step or reusable workflow) names a full 40-character
  commit SHA followed by a version comment: ``uses: actions/checkout@<sha> # v4.2.2``. Local
  actions (``./path``) are exempt; ``docker://`` images must be pinned by ``@sha256:`` digest.
* **Checkout credentials.** Every ``actions/checkout`` step sets ``persist-credentials: false``
  in its ``with:``, so later steps cannot reuse the token stored in ``.git/config``.
* **Least privilege.** The workflow has a top-level ``permissions:`` block; jobs add only the
  extra permissions they need.

The workflows are read line by line, without a YAML library: they use the block style, one
key per line, which is all this check needs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = Path(".github/workflows")
SHA_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^docker://[^@\s]+@sha256:[0-9a-f]{64}$")
KEY_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+)\s*:(?:\s+(?P<value>.*))?$")
BLOCK_RE = re.compile(r"^[|>][-+0-9]*$")
FALSE_RE = re.compile(r"""^(?:false|'false'|"false")$""")
FIX_PIN = (
    "Resolve the tag to its commit with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` "
    "(for an annotated tag, dereference it with `gh api repos/<owner>/<repo>/git/tags/<sha>`), "
    "then write `uses: <action>@<40-character sha> # <version>`."
)


@dataclass(frozen=True)
class Line:
    """One meaningful line of a workflow: its number, key column and ``key: value``."""

    number: int
    column: int  # column of the key, after any list dash
    item: bool  # the line starts a list item (``- key: value``)
    key: str | None
    value: str
    comment: str


def _split_comment(text: str) -> tuple[str, str]:
    """``(content, comment)``: a ``#`` starts a comment outside quotes, after a space."""
    quote = ""
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip(), text[i + 1 :].strip()
    return text.rstrip(), ""


def parse(text: str) -> list[Line]:
    """The non-blank, non-comment lines of ``text``, skipping block scalars (``run: |``)."""
    lines = []
    block = -1  # inside a block scalar: lines indented deeper than this column are its text
    for number, raw in enumerate(text.splitlines(), start=1):
        indent = len(raw) - len(raw.lstrip())
        if block >= 0:
            if not raw.strip() or indent > block:
                continue
            block = -1
        content, comment = _split_comment(raw)
        if not content.strip():
            continue
        body = content.strip()
        column = indent
        item = body == "-" or body.startswith("- ")
        if item:
            rest = body[1:].lstrip()
            column += len(body) - len(rest)
            body = rest
        match = KEY_RE.match(body)
        key = match["key"] if match else None
        value = (match["value"] or "").strip() if match else body
        if key is not None and BLOCK_RE.match(value):
            block = column
        lines.append(Line(number, column, item, key, _unquote(value), comment))
    return lines


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _mapping(lines: list[Line], index: int) -> list[Line]:
    """The lines of the mapping that holds ``lines[index]`` (a step, for a ``uses:``)."""
    column = lines[index].column
    start = index  # walk back to the step's dash, which may sit on another key: `- name: ...`
    while (
        not (lines[start].item and lines[start].column == column)
        and start > 0
        and lines[start - 1].column >= column
    ):
        start -= 1
    end = index + 1
    while (
        end < len(lines)
        and lines[end].column >= column
        and not (lines[end].item and lines[end].column == column)
    ):
        end += 1
    return lines[start:end]


def _persists_credentials(step: list[Line]) -> bool:
    """Whether a checkout step leaves ``persist-credentials`` at its default (true)."""
    column = step[0].column
    for i, line in enumerate(step):
        if line.key != "with" or line.column != column:
            continue
        if line.value:  # flow mapping: `with: {persist-credentials: false}`
            return not re.search(r"persist-credentials\s*:\s*['\"]?false\b", line.value)
        for inner in step[i + 1 :]:
            if inner.column <= column:
                break
            if inner.key == "persist-credentials" and FALSE_RE.match(inner.value):
                return False
    return True


def check_workflow(path: Path, root: Path) -> list[str]:
    rel = path.relative_to(root)
    lines = parse(path.read_text())
    errors = []
    if not any(line.key == "permissions" and line.column == 0 for line in lines):
        errors.append(
            f"{rel} has no top-level `permissions:` block. Add `permissions:` with "
            "`contents: read` at the top of the workflow, and give each job only the extra "
            "permissions it needs in its own `permissions:`."
        )
    for i, line in enumerate(lines):
        if line.key != "uses":
            continue
        where = f"{rel}:{line.number}"
        ref = line.value
        if ref.startswith("./"):
            continue
        if ref.startswith("docker://"):
            if not DIGEST_RE.match(ref):
                errors.append(
                    f"{where}: `uses: {ref}` is not pinned to an image digest. "
                    "Write `uses: docker://<image>@sha256:<digest> # <tag>`."
                )
            continue
        if not SHA_RE.match(ref):
            errors.append(f"{where}: `uses: {ref}` is not pinned to a full commit SHA. {FIX_PIN}")
        elif not line.comment:
            errors.append(
                f"{where}: `uses: {ref}` has no version comment. Add the version the SHA "
                "resolves to after it, for example `# v4.2.2`, so Dependabot can update both."
            )
        action = ref.split("@", 1)[0].lower()
        if action == "actions/checkout" and _persists_credentials(_mapping(lines, i)):
            errors.append(
                f"{where}: this actions/checkout step keeps the token in .git/config. Add "
                "`persist-credentials: false` under its `with:`; a later step that needs to "
                "push should receive a token explicitly instead."
            )
    return errors


def workflows(root: Path) -> list[Path]:
    folder = root / WORKFLOWS
    return sorted([*folder.glob("*.yml"), *folder.glob("*.yaml")])


def check(root: Path) -> list[str]:
    errors = []
    for path in workflows(root):
        errors += check_workflow(path, root)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)
    errors = check(args.root)
    for error in errors:
        print(f"check_workflows: {error}", file=sys.stderr)
    if not errors:
        print(f"check_workflows: OK, {len(workflows(args.root))} workflows.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
