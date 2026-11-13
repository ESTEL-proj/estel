# Pull-request policy

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

## Every PR

- One topic, branched from `main`; follow-ups named explicitly.
- A news fragment `changes/<PR>.<major|minor|patch>.md`.
- Exception: a PR that changes only `.github/` needs no news fragment, since it does not change the package (Part I, §7.2). It still needs CODEOWNER approval.
- `make check` green locally; its summary in the description.
- No new `skip` or `xfail` without the `test-change` label; the number of tests may not drop.

## Required checks

Tier 0 (GitHub-hosted CPU, ≤ 15 min) on every push and PR. Tier 1 (cluster GPU) is set up after 1.0.0 and then becomes required too. Linear history; `main` protected.

## Merging (from 1.0.0)

| Required change | Merge |
|---|---|
| patch | auto-merge when the required checks pass |
| minor | auto-merge when CI passes; benchmark-driven minors after a human sets the `bench` label |
| major, or a bumped reference counter | CODEOWNER approval, then merge |

## During step 1 (before 1.0.0)

PRs merge when CI passes, except on protected paths, which always need CODEOWNER approval. The three CODEOWNERS review step 1 as a whole at its end, from the parity reports and test results of each phase.

## AI review

The adapted GRID4EARTH rubric (`docs/agents/review_rules.md`) runs on every PR, posts comments and never blocks. Every rule that can be checked mechanically is a CI check instead.
