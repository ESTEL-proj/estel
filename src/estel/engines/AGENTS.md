# AGENTS.md — `src/estel/engines/` (engine layer)

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

Read the root `AGENTS.md` first; these rules add to it.

- **Only the Python engine exists in 1.0.0.** `engine=` accepts only `"python"`. Do not add another engine before step 2.
- **The engine protocol is reserved.** `_protocol.py`, `_registry.py` and the `engine=` argument exist in 1.0.0 with a trivial implementation and their tests; keep them minimal, and change them only through a major version or the step-2 work that uses them.
- An engine runs the whole computation: a forward pass returning the statistics plus an opaque residual, and a VJP taking the residual back. No hidden state.
- The engine layer never imports `estel.backend` directly: it goes through `estel.functional` or the L2 skeleton (`engines-not-backend` contract).
- `estel_rust` is imported only here, in step 2 (`rust-engine-purity` contract).
- Every engine declares its capabilities, including its differentiation order (`diff_order`); an unsupported request raises `EngineUnsupportedError` naming the option.

Summary: `docs/agents/architecture.md`. Source: Part I, §2.2.4 and §3.2.4 of `docs/transfer/`.
