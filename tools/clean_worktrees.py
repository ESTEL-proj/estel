"""Remove the worktrees of merged pull requests, and their local branches (issue #109).

Each coding session works in a git worktree of its own (``../estel-<branch>``); once its pull
request is merged, the worktree and its local branch are left behind. ``make clean-worktrees``
removes them. For each worktree of ``git worktree list`` it removes the worktree
(``git worktree remove``, never forced) and deletes its local branch only when all of these hold:

- it is not the main checkout, and not the worktree the command runs from;
- it is on a branch (not a detached HEAD), not locked, and its directory exists;
- it has no uncommitted or untracked change (``git status --porcelain``);
- a pull request from its branch is merged (``gh pr list --state merged --head <branch>``), and
  the local branch holds no commit outside it: its tip is the head of that pull request, or an
  ancestor of it. A branch name reused after the merge, or a commit made after it, keeps the
  worktree.

Pull requests are squash-merged, so git does not see the branch as merged: the branch is deleted
with ``git branch -D``, once the checks above have shown that GitHub holds all of its commits.
Then ``git worktree prune`` drops the records of worktrees whose directory is gone. The command
prints one line per worktree, removed or kept and why. Without ``gh``, or with ``gh`` not logged
in, it says so and does nothing.

Standard library only: it runs without the project's environment.

    make clean-worktrees
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

BRANCH_PREFIX = "refs/heads/"


@dataclass(frozen=True)
class Worktree:
    """One entry of ``git worktree list --porcelain``."""

    path: Path
    head: str = ""
    branch: str | None = None
    locked: bool = False
    prunable: bool = False


@dataclass(frozen=True)
class MergedPR:
    number: int
    head: str


def parse_worktrees(porcelain: str) -> list[Worktree]:
    """The worktrees of ``git worktree list --porcelain``, the main checkout first."""
    worktrees: list[Worktree] = []
    for block in porcelain.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            fields[key] = value
        if "worktree" not in fields:
            continue
        branch = fields.get("branch")
        worktrees.append(
            Worktree(
                path=Path(fields["worktree"]),
                head=fields.get("HEAD", ""),
                branch=branch.removeprefix(BRANCH_PREFIX) if branch else None,
                locked="locked" in fields,
                prunable="prunable" in fields,
            )
        )
    return worktrees


class Repository(Protocol):
    """The git and gh calls of the command; the tests replace them with fakes."""

    def gh_problem(self) -> str | None:
        """Why gh cannot be used (missing, not logged in), or None."""

    def worktrees(self) -> list[Worktree]: ...

    def current(self) -> Path:
        """The top directory of the worktree the command runs from."""

    def exists(self, path: Path) -> bool: ...

    def has_changes(self, path: Path) -> bool:
        """True when the worktree at ``path`` has an uncommitted or untracked change."""

    def merged_prs(self, branch: str) -> list[MergedPR]:
        """The merged pull requests from ``branch``; raises RuntimeError when gh fails."""

    def is_ancestor(self, commit: str, of: str) -> bool: ...

    def remove_worktree(self, path: Path) -> str | None:
        """Remove the worktree without forcing; git's error message, or None."""

    def delete_branch(self, branch: str) -> str | None:
        """Delete the local branch; git's error message, or None."""

    def prune(self) -> None: ...


def _same(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve()


def _covering_pr(repo: Repository, worktree: Worktree, prs: list[MergedPR]) -> MergedPR | None:
    """The merged pull request that holds every commit of the worktree's branch, if any."""
    for pr in prs:
        if pr.head == worktree.head:
            return pr
    for pr in prs:
        if repo.is_ancestor(worktree.head, of=pr.head):
            return pr
    return None


def why_keep(repo: Repository, worktree: Worktree, *, is_main: bool, current: Path) -> str | None:
    """Why the worktree must stay, or None when it can be removed.

    The checks run from the cheapest to the dearest, so gh is asked only about clean worktrees.
    """
    if is_main:
        return "the main checkout"
    if _same(worktree.path, current):
        return "the worktree this command runs from"
    if worktree.branch is None:
        return "detached HEAD, no branch to match with a pull request"
    if worktree.locked:
        return "locked (git worktree unlock, if it should go)"
    if worktree.prunable or not repo.exists(worktree.path):
        return "its directory is missing; git worktree prune drops its record, the branch stays"
    if repo.has_changes(worktree.path):
        return "uncommitted or untracked changes"
    try:
        prs = repo.merged_prs(worktree.branch)
    except RuntimeError as error:
        return f"could not ask GitHub: {error}"
    if not prs:
        return "no merged pull request from this branch"
    if _covering_pr(repo, worktree, prs) is None:
        numbers = ", ".join(f"#{pr.number}" for pr in prs)
        return f"the branch has commits that are not in the merged pull request ({numbers})"
    return None


def clean(repo: Repository, out: Callable[[str], None] = print) -> int:
    """Remove the worktrees that can go; return the exit status."""
    problem = repo.gh_problem()
    if problem is not None:
        out(f"{problem} Nothing was removed.")
        return 1
    current = repo.current()
    for index, worktree in enumerate(repo.worktrees()):
        label = f"{worktree.path} [{worktree.branch or 'detached'}]"
        reason = why_keep(repo, worktree, is_main=index == 0, current=current)
        if reason is not None or worktree.branch is None:
            out(f"kept     {label}: {reason}")
            continue
        error = repo.remove_worktree(worktree.path)
        if error is not None:
            out(f"kept     {label}: git worktree remove refused: {error}")
            continue
        error = repo.delete_branch(worktree.branch)
        if error is not None:
            out(f"removed  {label}, but the branch stays: {error}")
        else:
            out(f"removed  {label}, and its branch")
    repo.prune()
    return 0


class GitRepository:
    """The real calls, run from ``cwd``."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def _run(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args), cwd=cwd or self.cwd, capture_output=True, text=True, check=False
        )

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        result = self._run("git", *args, cwd=cwd)
        if result.returncode != 0:
            raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    def gh_problem(self) -> str | None:
        if shutil.which("gh") is None:
            return (
                "gh, the GitHub CLI, is not installed: install it (https://cli.github.com), "
                "run gh auth login, then make clean-worktrees again."
            )
        if self._run("gh", "auth", "status").returncode != 0:
            return "gh is not logged in: run gh auth login, then make clean-worktrees again."
        return None

    def worktrees(self) -> list[Worktree]:
        return parse_worktrees(self._git("worktree", "list", "--porcelain"))

    def current(self) -> Path:
        return Path(self._git("rev-parse", "--show-toplevel").strip())

    def exists(self, path: Path) -> bool:
        return path.is_dir()

    def has_changes(self, path: Path) -> bool:
        status = self._git("status", "--porcelain", "--untracked-files=all", cwd=path)
        return status.strip() != ""

    def merged_prs(self, branch: str) -> list[MergedPR]:
        result = self._run(
            "gh", "pr", "list", "--state", "merged", "--head", branch, "--json", "number,headRefOid"
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"gh exited with {result.returncode}")
        return [MergedPR(pr["number"], pr["headRefOid"]) for pr in json.loads(result.stdout)]

    def is_ancestor(self, commit: str, of: str) -> bool:
        return self._run("git", "merge-base", "--is-ancestor", commit, of).returncode == 0

    def remove_worktree(self, path: Path) -> str | None:
        result = self._run("git", "worktree", "remove", str(path))
        if result.returncode == 0:
            return None
        return result.stderr.strip() or f"exit status {result.returncode}"

    def delete_branch(self, branch: str) -> str | None:
        result = self._run("git", "branch", "-D", branch)
        if result.returncode == 0:
            return None
        return result.stderr.strip() or f"exit status {result.returncode}"

    def prune(self) -> None:
        self._git("worktree", "prune")


def main() -> int:
    return clean(GitRepository(Path.cwd()))


if __name__ == "__main__":
    sys.exit(main())
