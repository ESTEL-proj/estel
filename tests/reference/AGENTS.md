# AGENTS.md — `tests/reference/` (reference values)

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

Read the root `AGENTS.md` first; these rules add to it.

- **Protected.** The stored inputs and outputs (`*.npz`) and `MANIFEST.sha256` need CODEOWNER approval. `MANIFEST.sha256` is checked in CI: from 1.0.0, a change to a reference file without a `STATISTICS_REFERENCE` bump, with CODEOWNER approval, fails; before 1.0.0 the change is only reported.
- **Values are regenerated only by a human**, with `tools/make_reference.py`, from the `_spec` oracle in float64. Never write, regenerate or edit a reference file yourself, and never loosen `tests/tolerances.toml`.
- **If a reference test fails, stop.** Report the failing coefficients (family, indices, observed and expected values, tolerance) in an issue. A change that alters a reference value needs a `STATISTICS_REFERENCE` bump and a human decision.
- Comparisons use normalised coefficients and the tolerances of `tests/tolerances.toml`, per dtype, device class and family.

Summary: `docs/agents/testing.md`. Source: Part I, §5.1–§5.3 of `docs/transfer/`.
