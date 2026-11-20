"""The register of step 2: one card per entry in ``docs/register/``, and the view made from them.

Each entry of the register of later features (Part III, §16) is a card, ``docs/register/<ID>.md``:
a front matter in TOML between two lines ``+++``, then the text of the entry in Markdown. This
tool reads the front matter only, never the text, and writes the view ``docs/register/README.md``,
whose rules (the paragraphs of ``RULES``) are written here.

    python tools/register.py --write    # regenerate docs/register/README.md (make register)
    python tools/register.py --check    # fail if a card is invalid or the view is out of date
    python tools/register.py --step 2a --human    # a view in the terminal, filtered

A card is valid when its front matter parses and has exactly the fields of ``FIELDS``, with their
types; its ``id`` is the name of its file; its ``category``, ``step`` and ``status`` take one of
their values; its ``phase`` is set for step 2a only, to one of ``PHASES``; its title has at most
``MAX_TITLE_WORDS`` words; and each of its ``depends`` names another card. ``README.md`` and
``decisions.md`` (the decision log of step 2) are the only other files of the directory.

The view is deterministic: the same cards give the same bytes. Standard library only (``tomllib``),
and the tool never imports estel.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

REPO = Path(__file__).resolve().parents[1]
REGISTER = Path("docs/register")
VIEW = REGISTER / "README.md"
NOT_CARDS = frozenset({"README.md", "decisions.md"})
FENCE = "+++"
FIX_VIEW = (
    "run `make register` (python tools/register.py --write) and commit the result; "
    "never edit the view by hand."
)

ID_RE = re.compile(r"(LF|W)([1-9]\d*)")
CATEGORIES = (
    "Structure",
    "New feature",
    "Optimisation",
    "Mathematical choice",
    "Work that is not a feature",
)
STEPS = ("queue", "2a", "2b", "3")
PHASES = {
    "2a.0": "ready: the decisions, the mathematics in `docs/maths_step2.md`, the benchmark",
    "2a.1": "means and borders",
    "2a.2": "the coefficient set and the reductions",
    "2a.3": "spectra and masks",
    "2a.4": "syntheses",
    "2a.5": "the end of step 2a: the benchmark baseline and the optimisation session",
    "2a.X": "the additive lane: the entries that do not hold up the end of step 2a, at any time",
}
STATUSES = ("decision", "study", "ready", "in-progress", "done")
HUMAN = ("decision",)
TREATABLE = ("study", "ready", "in-progress")
MAX_TITLE_WORDS = 8

# The fields of a front matter, in the order of a card, and their types.
FIELDS: dict[str, type] = {
    "id": str,
    "title": str,
    "category": str,
    "step": str,
    "phase": str,
    "status": str,
    "decide": list,
    "depends": list,
    "changes_values": str,
    "issue": int,
}
KINDS = {str: "a string", list: "a list of strings", int: "an integer"}


@dataclass(frozen=True)
class Card:
    """The front matter of one card."""

    ident: str
    title: str
    category: str
    step: str
    phase: str
    status: str
    decide: tuple[str, ...]
    depends: tuple[str, ...]
    changes_values: str
    issue: int


def id_key(ident: str) -> tuple[str, int]:
    """LF before W, then by number: LF2 before LF10."""
    match = ID_RE.fullmatch(ident)
    return (match.group(1), int(match.group(2))) if match else (ident, 0)


def _has_type(value: object, kind: type) -> bool:
    if kind is list:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, kind)


def _front_matter(path: Path, rel: str) -> tuple[dict[str, object] | None, str | None]:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != FENCE or FENCE not in lines[1:]:
        return None, (
            f"{rel} has no front matter: start the card with a line `{FENCE}`, its fields in "
            f"TOML and a line `{FENCE}` (the fields are listed in {VIEW})."
        )
    end = lines.index(FENCE, 1)
    try:
        return tomllib.loads("\n".join(lines[1:end])), None
    except tomllib.TOMLDecodeError as error:
        return None, f"{rel}: its front matter is not valid TOML ({error}); fix it."


def _card(rel: str, stem: str, meta: dict[str, object]) -> tuple[Card | None, list[str]]:
    """The card of a front matter, or the problems that keep it from being one."""
    missing = [key for key in FIELDS if key not in meta]
    unknown = sorted(set(meta) - set(FIELDS))
    errors = []
    if missing:
        errors.append(f"{rel} lacks the fields {', '.join(missing)}: add them ({VIEW}).")
    if unknown:
        errors.append(
            f"{rel} has unknown fields {', '.join(unknown)}: remove them, or add them to FIELDS "
            "in tools/register.py."
        )
    for key, kind in FIELDS.items():
        if key in meta and not _has_type(meta[key], kind):
            errors.append(f"{rel}: `{key}` must be {KINDS[kind]}, not {meta[key]!r}.")
    if errors:
        return None, errors

    card = Card(
        ident=str(meta["id"]),
        title=str(meta["title"]),
        category=str(meta["category"]),
        step=str(meta["step"]),
        phase=str(meta["phase"]),
        status=str(meta["status"]),
        decide=tuple(cast("list[str]", meta["decide"])),
        depends=tuple(cast("list[str]", meta["depends"])),
        changes_values=str(meta["changes_values"]),
        issue=cast(int, meta["issue"]),
    )
    if card.ident != stem:
        errors.append(
            f"{rel}: `id` is {card.ident!r}, but the file is {stem}.md: a card is named after "
            "its identifier; rename the file or correct `id`."
        )
    for key, value, allowed in (
        ("category", card.category, CATEGORIES),
        ("step", card.step, STEPS),
        ("status", card.status, STATUSES),
    ):
        if value not in allowed:
            errors.append(
                f"{rel}: `{key}` is {value!r}, not one of {', '.join(allowed)}: use one of them."
            )
    if card.step == "2a" and card.phase not in PHASES:
        errors.append(
            f"{rel}: a card of step 2a has a `phase`, one of {', '.join(PHASES)}; it has "
            f"{card.phase!r}. Set the phase its task names, or ask a maintainer."
        )
    elif card.step != "2a" and card.phase:
        errors.append(
            f"{rel}: `phase` is {card.phase!r}, but only cards of step 2a have a phase: set it "
            'to "".'
        )
    words = len(card.title.split())
    if not 0 < words <= MAX_TITLE_WORDS:
        errors.append(
            f"{rel}: the title has {words} words: give it 1 to {MAX_TITLE_WORDS}, faithful to "
            "its section Feature."
        )
    if card.issue < 0:
        errors.append(f"{rel}: `issue` is {card.issue}: give its issue number, or 0 for none.")
    return card, errors


def load(root: Path) -> tuple[list[Card], list[str]]:
    """Every card of ``root``'s register, in identifier order, and the problems found."""
    directory = root / REGISTER
    if not directory.is_dir():
        return [], [f"{REGISTER.as_posix()}/ is missing: it holds the register of step 2."]
    cards: list[Card] = []
    errors: list[str] = []
    named: set[str] = set()  # the identifiers of the files named as cards, valid or not
    for path in sorted(directory.iterdir(), key=lambda p: (id_key(p.stem), p.name)):
        rel = (REGISTER / path.name).as_posix()
        if path.name in NOT_CARDS or path.name.startswith("."):
            continue
        if path.suffix != ".md" or not ID_RE.fullmatch(path.stem):
            errors.append(
                f"{rel} is not a card: a card is named <ID>.md, with an identifier LFn or Wn; "
                f"{' and '.join(sorted(NOT_CARDS))} are the only other files of the register. "
                "Rename or move it."
            )
            continue
        named.add(path.stem)
        meta, error = _front_matter(path, rel)
        if meta is None:
            errors.append(error or rel)
            continue
        card, problems = _card(rel, path.stem, meta)
        errors += problems
        if card is not None:
            cards.append(card)
    for card in cards:
        for other in card.depends:
            if other not in named or other == card.ident:
                errors.append(
                    f"{REGISTER.as_posix()}/{card.ident}.md: `depends` names {other!r}, which is "
                    "not another card: name an existing entry, or remove it."
                )
    return cards, errors


# ---------------------------------------------------------------- the view

HEADER = (
    "# Register of later features",
    "Generated by `tools/register.py`; do not edit; run `make register`. `make check` fails when "
    "a card is invalid or this view is out of date. In a terminal, `python tools/register.py` "
    "prints a view, filtered by `--step`, `--status` or `--human`.",
)

RULES = (
    "The register gathers everything still to implement after step 1: every development planned, "
    "and every feature, optimisation or piece of work deferred during the reviews or by an open "
    "point of the maths tab, so that none is forgotten. Until 20 November 2026 it was the table "
    "of Part III, §16. Each entry is now a card, `docs/register/<ID>.md`, and this view is made "
    "from the front matter of the cards. The decisions on step 2 are logged in "
    "[decisions.md](decisions.md).",
    "## How an entry lives",
    "A remark made while working on the mathematics opens an open point of the maths tab (OP). "
    "An open point is decided and applied at once, closed as not relevant, or sent here as a "
    "card. Any discussion may add open points or cards. What an entry still has to decide is in "
    'its section "To decide", summed up by its field `decide`, and is settled by the time the '
    "entry is implemented. When a step opens (Part III, §19.13), each of its entries becomes one "
    "GitHub issue, given by its field `issue` and broken into tasks whose progress is followed "
    "in comments on the issue, and the step is broken into phases, as step 1 is (Part II, §15). "
    "When an entry is done, its issue is closed and its status becomes `done`: the card stays, "
    'in the section "Done" of this view.',
    'One card per entry, named after its identifier: LFn ("later feature") and Wn (deferred work '
    "that is not a feature); a new entry takes the next free number. An entry that is abandoned, "
    "or that becomes a task of step 1 (Part II, §15.2), loses its card and leaves a row in a "
    "decision log: that of step 1 (Part IV, §22) or that of step 2 "
    "([decisions.md](decisions.md)).",
    "## A card",
    f"A front matter in TOML between two lines `{FENCE}`, then the text of the entry: one "
    "section per column of the former table, copied verbatim on 20 November 2026 (Feature, Why, "
    'Builds on (reserved in 1.0.0), To decide, Details, Source), and the section "Step before '
    '20 November 2026", its column "Step" as it was. In that text, a section number without a '
    'part (§20.11) is one of Part III, and "the maths tab" is `docs/maths.md`. The section '
    '"Source" says where the entry comes from: a review point (Part IV, §22) or an open point '
    "of the maths tab (OP). The fields of the front matter:",
    "- **`id`**: the identifier, the name of the file.\n"
    f'- **`title`**: at most {MAX_TITLE_WORDS} words, faithful to the section "Feature".\n'
    "- **`category`**: *Structure*: backends and engines, nothing the user sees, everything the "
    "code is built on. *New feature*: what users will be able to compute or do that they cannot "
    "today. *Optimisation*: the same results, faster, larger, lighter or on more configurations. "
    "*Mathematical choice*: a decision that changes what existing statistics compute; its "
    "definition is in `docs/maths_step2.md` where one is given. *Work that is not a feature*: "
    "the deferred work (Wn).\n"
    "- **`step`**: `queue` (not placed yet), `2a`, `2b` or `3`, placed by the criterion of "
    "[decisions.md](decisions.md) (Part III, §19.13).\n"
    "- **`phase`**: for step 2a only, one of the phases below; empty otherwise.\n"
    "- **`status`**: one of the statuses below.\n"
    "- **`decide`**: what the entry still has to settle, by the time it is implemented, in "
    'short phrases; empty when its section "To decide" is "—".\n'
    '- **`depends`**: the entries it must wait for, named in its column "Step" ("with LF62"); '
    'an entry that completes it later ("completed by LF22") is not one.\n'
    "- **`changes_values`**: whether the reference values change, and the version impact where "
    "it is known.\n"
    "- **`issue`**: the number of its GitHub issue; 0 until it has one.",
    "## Statuses",
    "- **`decision`**: a human must decide before an agent can be tasked: the entry has "
    "something to decide; or it changes values and `docs/maths_step2.md` gives no `M-` "
    'definition for it; or its scope is a human choice ("a proposal, to examine", "only on a '
    'specific need", "if validated", a research question).\n'
    "- **`study`**: the entry is to be measured, tried or studied, and the result informs a "
    "decision; an agent can be tasked with the study.\n"
    "- **`ready`**: none of the above; an agent can be tasked.\n"
    "- **`in-progress`**: the work has started, under its issue.\n"
    "- **`done`**: done; its issue is closed.",
    'Below, each step is split into "Needs a human" (`decision`) and "Can be treated" '
    "(`study`, `ready`, `in-progress`). A maintainer reviews the statuses.",
    "## Phases of step 2a",
    "\n".join(f"- **{phase}**: {what}." for phase, what in PHASES.items()),
    "2a.0 → 2a.1 → 2a.2 → 2a.5 is the critical path; 2a.3 and 2a.4 run in parallel from 2a.0; "
    "2a.X runs at any time.",
    "## The queue",
    "Something found during step 1 that belongs to step 2 becomes a new card with "
    '`step = "queue"` and `status = "decision"` (placing it is a human decision), the next free '
    "LF number, and optionally a suggested step in its body. Never change the step, phase, "
    "status or content of an existing card unless your task names "
    "that card (AGENTS.md, golden rule 16).",
)

HUMAN_COLUMNS = ("ID", "Title", "Phase", "To decide", "Category")
TREATABLE_COLUMNS = ("ID", "Title", "Phase", "Status", "Depends", "Issue")
DONE_COLUMNS = ("ID", "Title", "Step", "Phase", "Issue")
SECTIONS = (("Queue", "queue"), ("Step 2a", "2a"), ("Step 2b", "2b"), ("Step 3", "3"))


def _order(card: Card) -> tuple[int, int, tuple[str, int]]:
    phases = list(PHASES)
    phase = phases.index(card.phase) if card.phase in PHASES else len(phases)
    step = STEPS.index(card.step) if card.step in STEPS else len(STEPS)
    return step, phase, id_key(card.ident)


def _cell(text: str) -> str:
    return text.replace("|", "\\|") if text else "—"


def _table(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "(none)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _link(card: Card) -> str:
    return f"[{card.ident}]({card.ident}.md)"


def _issue(card: Card) -> str:
    return f"#{card.issue}" if card.issue else ""


def _entries(count: int) -> str:
    return f"{count} {'entry' if count == 1 else 'entries'}"


def _count(cards: list[Card], human: int) -> str:
    need = "needs" if human == 1 else "need"
    return f"{_entries(len(cards))}: {human} {need} a human, {len(cards) - human} can be treated."


def view(cards: Iterable[Card]) -> str:
    """The content of ``docs/register/README.md``."""
    ordered = sorted(cards, key=_order)
    blocks = [*HEADER, *RULES]
    for title, step in SECTIONS:
        in_step = [c for c in ordered if c.step == step and c.status != "done"]
        human = [c for c in in_step if c.status in HUMAN]
        treatable = [c for c in in_step if c.status in TREATABLE]
        blocks += [f"## {title}", _count(in_step, len(human)), "### Needs a human"]
        blocks.append(
            _table(
                HUMAN_COLUMNS,
                [(_link(c), c.title, c.phase, "; ".join(c.decide), c.category) for c in human],
            )
        )
        blocks.append("### Can be treated")
        blocks.append(
            _table(
                TREATABLE_COLUMNS,
                [
                    (_link(c), c.title, c.phase, c.status, ", ".join(c.depends), _issue(c))
                    for c in treatable
                ],
            )
        )
    done = [c for c in ordered if c.status == "done"]
    blocks += ["## Done", f"{_entries(len(done))}."]
    blocks.append(
        _table(DONE_COLUMNS, [(_link(c), c.title, c.step, c.phase, _issue(c)) for c in done])
    )
    return "\n\n".join(blocks) + "\n"


def terminal(cards: Iterable[Card], step: str | None, status: str | None, human: bool) -> str:
    """A plain-text view of the cards that pass the filters."""
    chosen = [
        c
        for c in sorted(cards, key=_order)
        if (step is None or c.step == step)
        and (status is None or c.status == status)
        and (not human or c.status in HUMAN)
    ]
    lines = []
    for c in chosen:
        lines.append(f"{c.ident:<5} {c.step:<5} {c.phase or '-':<5} {c.status:<11} {c.title}")
        if c.decide:
            lines.append(f"{'':<29} to decide: {'; '.join(c.decide)}")
        if c.depends:
            lines.append(f"{'':<29} depends: {', '.join(c.depends)}")
    filters = [f"step {step}"] if step else []
    filters += [f"status {status}"] if status else []
    filters += ["needs a human"] if human else []
    noun = "card" if len(chosen) == 1 else "cards"
    lines.append(f"{len(chosen)} {noun}" + (f" ({', '.join(filters)})" if filters else "") + ".")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help=f"regenerate {VIEW}")
    action.add_argument(
        "--check", action="store_true", help="fail if a card is invalid or the view is stale"
    )
    parser.add_argument("--step", choices=STEPS, help="terminal view: one step only")
    parser.add_argument("--status", choices=STATUSES, help="terminal view: one status only")
    parser.add_argument(
        "--human", action="store_true", help="terminal view: only what needs a human (decision)"
    )
    args = parser.parse_args(argv)
    if (args.write or args.check) and (args.step or args.status or args.human):
        parser.error("--step, --status and --human filter the terminal view only.")
    root: Path = args.root

    cards, errors = load(root)
    if not args.write and not args.check:
        for error in errors:
            print(f"register: {error}", file=sys.stderr)
        print(terminal(cards, args.step, args.status, args.human))
        return 1 if errors else 0

    expected = view(cards)
    target = root / VIEW
    current = target.read_text() if target.exists() else None
    if args.write and not errors and current != expected:
        target.write_text(expected)
        current = expected
        print(f"register: wrote {VIEW}.")
    if not errors and current != expected:
        state = "is missing" if current is None else "is out of date with the cards"
        errors.append(f"{VIEW} {state}: {FIX_VIEW}")
    for error in errors:
        print(f"register: {error}", file=sys.stderr)
    if not errors:
        print(f"register: OK, {len(cards)} cards, {VIEW} up to date.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
