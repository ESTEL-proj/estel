# Review rules for ESTEL

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

Adapted from the GRID4EARTH review rubric (`github.com/GRID4EARTH/review-rules`, `d38b775`). The AI review applies these rules to every PR, comments, and never blocks.

## Adopted as is

A1, A3, A6, A7, A8, A9, A10, B1, B3–B10, C1–C5, C7–C10, D1–D7, D9–D12, E1–E3, E5 — with the wording of the GRID4EARTH rubric.

## Adapted

- **A4** (terminology): healpix-geo terminology (`depth`, NESTED, `cell_ids`) for `DataHealpix` only.
- **A11** (one dtype contract): applied to Python: float32 by default, float64 on request, never a silent cast.

## Rust (`rust/` only)

A5, B2, C6, D8 and E4 apply in `rust/`; its local `AGENTS.md` repeats them. A2 is specific to healpix-geo and dropped.

## ESTEL-specific rules

- **X1 Reference values.** No change to `tests/reference/` or tolerances without a reference-counter bump and CODEOWNER approval.
- **X2 Backend purity.** No `torch` or `jax` import outside `src/estel/backend/`; no `estel_rust` import outside `src/estel/engines/`.
- **X3 Functional style.** No in-place updates in L1 and L2; no control flow on array values; static shapes.
- **X4 Second order.** The Python engine stays differentiable twice on PyTorch and JAX: safe modulus, masks as weights, no `custom_vjp`.
- **X5 Storage discipline.** No full-grid (NaN-filled) tensor inside the computation; full-grid views only for output.
- **X6 Mathematics, oracle, tests.** Every `M-` ID of `docs/maths.md` has a `_spec` function and a test, and vice versa.
- **X7 Version impact.** The declared bump matches the change.
- **X8 Actionable errors.** Every error message says what to do next.
- **X9 Step discipline.** Before 1.0.0, no capability STL lacks (synthesis checkpoints excepted).
- **X10 Public or internal.** New names are either in `api/manifest.json` or private (private module or leading underscore).
