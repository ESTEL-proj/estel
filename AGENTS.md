# AGENTS.md — ESTEL

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. Read docs/policy/VERSIONING.md. -->

**Within one value of `STATISTICS_REFERENCE`, statistics and test results are unchanged up to the tolerances of `tests/tolerances.toml`.**

ESTEL computes scattering-transform statistics and simple syntheses. Most changes are written by coding agents. This file is the context every agent reads first; Claude Code reads it through `CLAUDE.md`, GitHub Copilot through its generated mirror.

## Golden rules

1. Statistics never change silently. If your change alters any reference value, stop: it requires bumping `STATISTICS_REFERENCE` and a human decision. Open an issue instead.
2. Never edit protected paths (see below). Never weaken, skip or delete a test to make CI pass.
3. `torch` and `jax` are imported only in `src/estel/backend/`. L1 and L2 code is functional: no in-place updates, no Python control flow on array values.
4. Every mathematical change goes through `docs/maths.md` (with an `M-` ID), `estel._spec`, and a reference regeneration by a human.
5. Declare the version impact in `changes/<PR>.<major|minor|patch>.md`. CI computes the real one.
6. One topic per PR, branched from `main`. Name follow-ups explicitly.
7. Public API changes start in `api/manifest.json`. Additions are minors, breaking changes majors.
8. Error messages say what to do next: agents read them.
9. Before opening a PR, run `make check` (tier 0, locally).
10. When the specification is ambiguous, ask in an issue. Do not choose.
11. **Step discipline.** Before 1.0.0, never add a capability STL does not have, except synthesis checkpoints and the experimental JAX backend. After 1.0.0, a new capability enters as an opt-in feature described in the register of later features of `docs/reference.md`.
12. **Public or internal.** Only names listed in `api/manifest.json` are public. Anything else lives in a private module (`_name.py`) or starts with an underscore.

## Layer map

| Layer | Where | May import |
|---|---|---|
| Frontend (public API) | `src/estel/__init__.py` and public modules | everything below |
| Engine layer | `src/estel/engines/` | engines, never a backend directly |
| L2 skeleton | `src/estel/scattering/`, `statistics/`, `synthesis/`, `functional/` | L1, L0 protocol |
| L1 geometry | `src/estel/data/`, `wavelets/` | L0 protocol |
| L0 primitives | `src/estel/backend/` | `torch`, and `jax` (experimental from P1) |
| Oracle | `src/estel/_spec/` | NumPy only, nothing from `estel` |
| Rust engine (step 2) | `rust/` | — (local rules in `rust/AGENTS.md`) |

The static plan (`ScatteringPlan`) is built in NumPy when an operator is constructed. Masks and their reweighting maps are part of it: they are static.

## Numerical rules (summary; details in `docs/agents/jax_rules.md`)

- Build per-level results in lists, then `stack`; no preallocation filled by index.
- Shapes are static; index sets, crop widths, stencils and mean denominators live in the plan.
- No `.item()`, `float(x)`, `if x.isnan().any()` inside the computation.
- Every computation runs in float32 (default) and float64; no hard-coded float64 constant.
- Keep STL's loop structure in step 1; loops run over scales and levels, never over coefficients.
- Safe modulus (double `where`); masks are weights; no `custom_vjp` in the Python engine. Second order must work on PyTorch and JAX.

## Protected paths (CODEOWNER approval)

`tests/reference/**`, `tests/tolerances.toml`, `src/estel/_reference.py`, `docs/transfer/**`, `src/estel/_spec/**`, `docs/maths.md`, `docs/policy/**`, `api/manifest.json`, `AGENTS.md`, `CLAUDE.md` and their mirrors, `docs/agents/review_rules.md`, `.github/**`, `tools/make_reference.py`, `bench/thresholds.toml`; in step 2, the Rust binding and the capability schema.

`docs/transfer/` and `docs/maths.md` are the specification of ESTEL (`docs/transfer/README.md`). They are protected, and change only when a maintainer asks for it; never edit them on your own initiative.

## Commands

| Command | What |
|---|---|
| `make check` | tier 0 locally: lint, types, layer checks, manifest, mirrors, maths IDs, all CPU tests in float32 and float64 |
| `make check-tools` | tier 0 without the test suites: every check tool of `make check` |
| `make mirrors-fix` | regenerate `.github/copilot-instructions.md` after changing `AGENTS.md` |
| `make test-spec` | spec parity against the oracle |
| `make parity` | STL parity (step 1 only; needs the parity container) |
| `make bench` | benchmark cases (tier 2; humans trigger it in CI) |

When every changed file is in the no-test scope (`docs/`, `.github/`, `changes/`; `.gitignore`, `LICENSE`, `NOTICE` and `README.md` at the root; every `AGENTS.md` and `CLAUDE.md`; none read by a test), that is when `python tools/ci_scope.py --base origin/main` prints `tests=false`, run `make check-tools` instead of `make check` before opening the PR. CI does the same.

## More context

Before editing in `src/estel/backend/`, `src/estel/engines/`, `src/estel/_spec/`, `tests/reference/` or `rust/`, read the `AGENTS.md` of that directory: it holds the local rules.

`docs/agents/architecture.md` (and its "do not copy from STL" section), `jax_rules.md`, `differentiation.md`, `testing.md`, `review_rules.md`, `storage.md`; `docs/maths.md`; `docs/policy/`.
