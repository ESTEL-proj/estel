# AGENTS.md — `src/estel/backend/` (L0 primitives)

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

Read the root `AGENTS.md` first; these rules add to it.

- **The only code allowed to import `torch` or `jax`** (and `optax`). Nothing outside this directory may import them; the `backend-purity` import contract enforces it.
- **Every primitive follows `_protocol.py`.** It is the contract: shapes, dtypes and edge cases are in its docstrings, with value contracts citing `docs/maths.md`. A new or changed primitive starts there, then gets an implementation in every backend that exists.
- `_registry.py` holds `get_backend()`, `set_backend()` and the array → backend lookup; mixing backends raises `BackendMismatchError` with a message that says how to fix it.
- One subpackage per backend: `torch/` in step 1, `jax/` in step 2. No backend imports another.
- Primitives are written against the Array API (through `array-api-compat`) plus the extensions it lacks; they are pure and out of place, and run in float32 and float64.
- Every primitive passes the conformance suite in `tests/backend/` (values and gradients against NumPy float64). Adding a backend means passing that suite, nothing else.

Summary: `docs/agents/architecture.md`. Source: Part I, §2.2.1 and Part II, §12.3 of `docs/transfer/`.
