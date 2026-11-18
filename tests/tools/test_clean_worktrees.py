"""tools/clean_worktrees.py removes a worktree and its branch only when its pull request is merged
with every commit of the branch, it has no change, and it is neither the main checkout nor the
current worktree; without gh it does nothing (issue #109). The git and gh calls are fakes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from clean_worktrees import MergedPR, Worktree, clean, parse_worktrees

MAIN = Path("/repo/estel")
HERE = Path("/repo/estel-here")


@dataclass
class FakeRepository:
    """Worktrees, changes and pull requests given by the test; records what was called."""

    worktrees_: list[Worktree]
    prs: dict[str, list[MergedPR]] = field(default_factory=dict)
    changed: set[Path] = field(default_factory=set)
    missing: set[Path] = field(default_factory=set)
    ancestors: set[tuple[str, str]] = field(default_factory=set)
    problem: str | None = None
    gh_fails: bool = False
    remove_error: str | None = None
    calls: list[str] = field(default_factory=list)

    def gh_problem(self) -> str | None:
        return self.problem

    def worktrees(self) -> list[Worktree]:
        self.calls.append("list")
        return self.worktrees_

    def current(self) -> Path:
        return HERE

    def exists(self, path: Path) -> bool:
        return path not in self.missing

    def has_changes(self, path: Path) -> bool:
        return path in self.changed

    def merged_prs(self, branch: str) -> list[MergedPR]:
        self.calls.append(f"gh {branch}")
        if self.gh_fails:
            raise RuntimeError("network down")
        return self.prs.get(branch, [])

    def is_ancestor(self, commit: str, of: str) -> bool:
        return (commit, of) in self.ancestors

    def remove_worktree(self, path: Path) -> str | None:
        self.calls.append(f"remove {path}")
        return self.remove_error

    def delete_branch(self, branch: str) -> str | None:
        self.calls.append(f"delete {branch}")
        return None

    def prune(self) -> None:
        self.calls.append("prune")


def tree(name: str, head: str = "a1", *, locked: bool = False, prunable: bool = False) -> Worktree:
    path = Path(f"/repo/estel-{name}")
    return Worktree(path, head=head, branch=name, locked=locked, prunable=prunable)


def run(repo: FakeRepository) -> tuple[int, list[str]]:
    lines: list[str] = []
    status = clean(repo, lines.append)
    return status, lines


def test_parse_porcelain() -> None:
    porcelain = (
        "worktree /repo/estel\nHEAD aaa\ndetached\n\n"
        "worktree /repo/estel-x\nHEAD bbb\nbranch refs/heads/x\nlocked busy\n\n"
        "worktree /repo/estel-y\nHEAD ccc\nbranch refs/heads/feat/y\nprunable gitdir missing\n"
    )
    assert parse_worktrees(porcelain) == [
        Worktree(Path("/repo/estel"), head="aaa"),
        Worktree(Path("/repo/estel-x"), head="bbb", branch="x", locked=True),
        Worktree(Path("/repo/estel-y"), head="ccc", branch="feat/y", prunable=True),
    ]


def test_removes_only_merged_clean_worktrees() -> None:
    main = Worktree(MAIN, head="m0", branch="done")  # the main checkout, even on a merged branch
    repo = FakeRepository(
        [main, tree("done"), tree("later", head="b2"), tree("open"), tree("dirty")],
        prs={
            "done": [MergedPR(7, "a1")],
            "later": [MergedPR(8, "a1")],  # a commit was made on the branch after the merge
            "dirty": [MergedPR(9, "a1")],
        },
        changed={Path("/repo/estel-dirty")},
    )
    status, lines = run(repo)
    assert status == 0
    assert lines == [
        f"kept     {MAIN} [done]: the main checkout",
        "removed  /repo/estel-done [done], and its branch",
        "kept     /repo/estel-later [later]: the branch has commits that are not in the merged "
        "pull request (#8)",
        "kept     /repo/estel-open [open]: no merged pull request from this branch",
        "kept     /repo/estel-dirty [dirty]: uncommitted or untracked changes",
    ]
    removals = [c for c in repo.calls if c.startswith(("remove", "delete"))]
    assert removals == ["remove /repo/estel-done", "delete done"]
    assert "gh dirty" not in repo.calls  # GitHub is asked only about clean worktrees
    assert repo.calls[-1] == "prune"


def test_branch_behind_the_merged_head_goes() -> None:
    repo = FakeRepository(
        [Worktree(MAIN), tree("x", head="old")],
        prs={"x": [MergedPR(3, "new")]},
        ancestors={("old", "new")},
    )
    assert run(repo)[1][-1] == "removed  /repo/estel-x [x], and its branch"


@pytest.mark.parametrize(
    ("worktree", "reason"),
    [
        pytest.param(
            Worktree(HERE, branch="here"), "the worktree this command runs from", id="here"
        ),
        pytest.param(Worktree(Path("/repo/estel-d")), "detached HEAD", id="detached"),
        pytest.param(tree("x", locked=True), "locked", id="locked"),
        pytest.param(tree("x", prunable=True), "directory is missing", id="prunable"),
    ],
)
def test_keeps_without_asking_github(worktree: Worktree, reason: str) -> None:
    repo = FakeRepository([Worktree(MAIN), worktree], prs={"x": [MergedPR(1, "a1")]})
    status, lines = run(repo)
    assert status == 0
    assert lines[-1].startswith("kept")
    assert reason in lines[-1]
    assert repo.calls == ["list", "prune"]


def test_keeps_when_github_or_git_refuses() -> None:
    worktrees = [Worktree(MAIN), tree("x")]
    repo = FakeRepository(worktrees, gh_fails=True)
    assert run(repo)[1][-1].endswith("could not ask GitHub: network down")
    repo = FakeRepository(worktrees, prs={"x": [MergedPR(1, "a1")]}, remove_error="busy")
    assert run(repo)[1][-1].endswith("git worktree remove refused: busy")
    assert "delete x" not in repo.calls


def test_does_nothing_without_gh() -> None:
    repo = FakeRepository([Worktree(MAIN), tree("x")], problem="gh is not logged in.")
    status, lines = run(repo)
    assert status == 1
    assert lines == ["gh is not logged in. Nothing was removed."]
    assert repo.calls == []
