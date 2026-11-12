# Architecture

Summary of [Part I, §2 — Architecture](../transfer/part1.md#2-architecture). The transfer document is authoritative; when this summary and it disagree, it wins.

## Layers ([§2.1](../transfer/part1.md#21-layers))

| Layer | Where | Role | May import |
| --- | --- | --- | --- |
| Frontend | `src/estel/__init__.py`, public modules | public API, frozen by `api/manifest.json` | everything below |
| Engine layer | `src/estel/engines/` | chooses the engine that runs the whole computation | engines, never a backend directly |
| L2 skeleton | `scattering/`, `statistics/`, `synthesis/`, `functional/` | scattering loop, statistics, reductions, synthesis driver; backend-neutral | L1, L0 protocol |
| L1 geometry | `data/`, `wavelets/` | geometry written once on L0; DataClasses provide a few hooks | L0 protocol |
| L0 primitives | `src/estel/backend/` | ~20 primitives, the only code importing a backend | `torch` (step 1), `jax` (step 2) |
| Oracle | `src/estel/_spec/` | naive float64 NumPy transcription of `docs/maths.md` | NumPy only |

Two independent choices: the **engine** (`engine=`: `"python"`, later `"rust"`) decides who runs the whole computation; the **backend** (`set_backend()`: `"torch"`, later `"jax"`) decides who runs the elementary operations of the Python engine.

## Multi-backend strategy ([§2.2](../transfer/part1.md#22-multi-backend-strategy))

- **L0** is written against the Array API (`array-api-compat`) plus the missing extensions. `backend/_protocol.py` is the contract; `tests/backend/` is the conformance suite every backend must pass.
- **Static plan.** `ScatteringPlan` is built once in NumPy at operator construction: index sets, kernels, crop widths per level, masks and reweighting maps, spectrum bins, stencils. Immutable and hashable; large arrays are passed as arguments, never baked in.
- **Backend selection.** A global default read when an operator is built; mixing backends raises `BackendMismatchError`; inputs are converted at the boundary, never silently between backends.
- **Engines.** One protocol (forward + VJP with an opaque residual). Only the Python engine exists in 1.0.0; the registry, protocol and `engine=` are reserved. The Rust code of step 2 is an engine, not a backend, because it replaces the whole computation and its hand-written gradient.

## Object model ([§2.3](../transfer/part1.md#23-object-model))

- A DataClass is an array plus static metadata, registered as a pytree (the array is the only leaf).
- Masks and their reweighting maps are static, computed at construction, part of the plan.
- The convolution history is plan data: the plan computes the crop width of every level once.
- An operator is a frozen plan plus a pure function: `op.apply(data, ref=None)` calls `estel.functional.scattering(x, plan, arrays, ref)`.
- Every option is set once, at construction; `apply` takes only the data and an optional reference.
- `estel.STOperator(data, ...)` and `estel.WaveletOperator(data, ...)` are the entry points; class names are PEP 8 CapWords (`STOperator`, `STOperatorTensorProd`, `STStatistics`).
- `STOperatorTensorProd` overrides a small, fixed set of hooks of the mother class, nothing else.

## Rules enforced in CI ([§2.4](../transfer/part1.md#24-code-rules-enforced-in-ci))

`make lint-imports` (import-linter contracts) and `make layers` (`tools/check_layers.py`) check: backend imports only under `estel/backend/`; `estel_rust` only under `estel/engines/`; L2 never imports a DataClass subclass; `_spec` imports NumPy only; no class-name comparisons or `__class__` assignment; no in-place array updates in L1/L2 (plan-building NumPy modules excepted). `tools/api_manifest.py --check` enforces public vs internal.

## Do not copy from STL

From [Part IV, §21, failure mode F14](../transfer/part4.md#21-critical-review-failure-modes). These STL patterns must not reach ESTEL, even when porting code line by line:

| STL pattern | Why it is forbidden | ESTEL instead |
| --- | --- | --- |
| Class-name comparisons (`type(x).__name__ == "STL_2D_Kernel_Torch"`) | break on any subclass and silently take the wrong branch | `isinstance`, or better a capability the object declares |
| Reassigning `__class__` (STL's tensor-product class does it at each instantiation) | no reader or tool (type checkers, JAX, `torch.compile`, serialisation) knows what the object is | composition: a tensor-product object holds one array and the geometry of each block |
| Module-level backend globals | hide the backend in global state, prevent mixing detection and a JAX port | the backend is recorded by each operator at construction (`set_backend()` sets only the default); L0 is reached through the protocol |
| Mutable `conv_history` on data objects | state that changes per call; the crop width depends on call order | the `ScatteringPlan` computes the crop width of every level once; data carry only immutable metadata |
| Complex storage of real quantities (mean, var, S1, auto-S2, diagonal of S4 stored complex) | wastes memory and blurs which coefficients are real | keep real quantities real; the compact layout stores them as one float ([storage.md](storage.md)) |

Also from STL, and also forbidden (Part I, §2.3, §3.1): options settable both in the constructor and in `apply`; preallocated NaN-filled tensors written in place; `inplace=True` downsampling; `.item()` or NaN detection inside the computation.
