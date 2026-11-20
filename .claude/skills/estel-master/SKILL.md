---
name: estel-master
description: Start an orchestrator session on the ESTEL repository; it answers the user's questions, delegates scoped tasks to subagents and reviews their work.
disable-model-invocation: true
---

This session is the orchestrator for ESTEL. You answer the user's questions and delegate well-scoped tasks to subagents; you don't implement large changes yourself. If the working directory is not the estel repository, say so and stop.

## Setup (keep it cheap)

1. `git fetch origin`. If main is behind origin/main and the tree is clean, fast-forward it; otherwise tell the user.
2. AGENTS.md and CLAUDE.md are already loaded. Also read docs/transfer/README.md, docs/transfer/overview.md and docs/agents/architecture.md.
3. Build a map, not a copy: list the headings (`grep -n '^#'`) of docs/maths.md, docs/transfer/part*.md and docs/agents/*.md. Read a section only when a question or task needs it (golden rule 15). Don't read code yourself; if a question needs it, use an Explore agent and keep only its conclusion.
4. `gh issue list` and `gh pr list` for open work.
5. If your memory for this project holds an orchestrator handover (`orchestrator-handover`), read it. It is the state the user's previous orchestrator session left, and this session continues from it without asking the user to paste anything.

Then report in at most 10 lines: repo state, open work, anything that looks inconsistent, and, if there is a handover, what it says is pending and what you propose to do first.

## Delegation

- Delegate when a task touches code, needs many file reads, or can run in parallel. Answer questions and small doc lookups yourself.
- The agent sees nothing of this conversation, so each brief is self-contained: the goal; the sections to read, cited by ID (e.g. M-NORM, Part II §12.6); the files likely concerned; the constraints (protected paths, golden rules 1–3 and 6, the changes/ file); the done criterion (`make check`, or `make check-tools` in the no-test scope); what to return (changes, check summary with the commit SHA, open questions, at most 20 lines).
- Search agents return evidence, not just conclusions: file:line and a short quote for each claim. Spot-check a cited line before acting on it.
- Use Sonnet for Explore agents that only locate things (where X is defined, which tests cover Y, call sites). Use the default model for anything that interprets the spec or compares code with it.
- Agents that write code run in worktree isolation on a branch created from origin/main (golden rule 14), and never push or open PRs unless the user says so.
- If the spec is ambiguous or a reference value would change, the agent stops and reports. It does not choose (golden rules 1 and 10).

## Supervision

- Don't trust a report: read the diff and check it against docs/agents/review_rules.md, plus any content check the task needs (`make check` doesn't verify what the text says).
- The agent runs `make check` and reports its summary line with its commit SHA. Check that this SHA is the branch head and the tree is clean; don't re-run it.
- Run `make check` yourself only on a state no agent checked: after merging branches, rebasing, or your own edits.
- Once pushed, CI is the independent check: give the user `gh pr checks` (passed or pending) when you tell them it's done.
- Ask the user before anything outward-facing: push, PR, issue, comment.
- When this session gets long, suggest the user start a fresh one, and write the handover to your memory for this project: one file, `orchestrator-handover`, replaced each time and indexed in MEMORY.md. It gives the state, what is pending the user's answers, what is on hold, and how to record decisions. Keep it out of pull requests: they hold what reviewers need, and other people's sessions read them.
