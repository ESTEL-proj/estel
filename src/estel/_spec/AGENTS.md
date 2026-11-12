# AGENTS.md — `src/estel/_spec/` (the oracle)

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

Read the root `AGENTS.md` first; these rules add to it.

- **This is the oracle**: a naive, slow, **float64**, **NumPy-only** transcription of `docs/maths.md`. Reference values come from it, never from the library.
- **Protected.** Every file here needs CODEOWNER approval. Do not edit it to make a test pass; if ESTEL and the oracle disagree, report the coefficients in an issue.
- **Independent of the rest of `estel`.** It imports NumPy (and the standard library) only, and nothing from `estel`; the `oracle-independence` import contract enforces it. Do not share helpers with the library, in either direction.
- Every public function lists in its docstring the `M-` IDs of `docs/maths.md` it implements, and every ID has a function; `tools/check_maths_ids.py` checks both directions. Clarity beats speed: loops over coefficients are fine here, and only here.

Summary: `docs/agents/testing.md`. Source: Part I, §4.2 and §5 of `docs/transfer/`.
