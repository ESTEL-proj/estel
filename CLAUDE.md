@AGENTS.md

## Notes for Claude Code

- Read `AGENTS.md` above first; it is the canonical context. This file only adds Claude-specific notes.
- Before editing a module, read its nested `AGENTS.md` if there is one (`src/estel/backend/`, `src/estel/engines/`, `src/estel/_spec/`, `tests/reference/`, `rust/`).
- Run `make check` before proposing a PR, and paste its summary in the PR description.
- If a reference test fails, do not regenerate references: report the failing coefficients and open an issue.
