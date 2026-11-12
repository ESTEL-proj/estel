# Part IV — Review and decisions

[Back to the overview](overview.md)

## Contents of this part

- [21. Critical review: failure modes](#21-critical-review-failure-modes)
- [22. Decision log](#22-decision-log)
  - [22.1 Scope and organisation](#221-scope-and-organisation) · [22.2 Architecture and backends](#222-architecture-and-backends) · [22.3 Process](#223-process) · [22.4 Review round of 12 November 2026](#224-review-round-of-12-november-2026)
- [23. Open questions](#23-open-questions)
- [24. Deferred work and points to discuss later](#24-deferred-work-and-points-to-discuss-later)
  - [24.1 Deferred work](#241-deferred-work) · [24.2 Points to discuss later](#242-points-to-discuss-later)

## 21. Critical review: failure modes

The plan is sound in its core: a frozen skeleton, versions tied to reference values, applications kept downstream, backends kept out of the classes, an agent-readable context, and now a two-step development that separates reproducing STL from extending it. The risks below are where it can still fail, ordered by severity, each with its mitigation.

| # | Failure mode | Why it happens here | Mitigation | Status |
| --- | --- | --- | --- | --- |
| F1 | **References that freeze bugs** | If reference values were produced by ESTEL itself, its bugs would become the truth | References from the independent `_spec` oracle, checked against STL in step 1 (Part II, §14.5), signed off by the three CODEOWNERS | largely mitigated by the two steps |
| F2 | **Agents make CI green the wrong way** | With no human review of each PR, the cheapest fix for a red test is to edit it: regenerate references, loosen tolerances, skip, special-case the input | Protected paths, a hash manifest, checks on test count and skips (Part I, §5.3), a hidden subset of reference inputs run by the maintainers before each release | open, monitored |
| F3 | **Quality drifts under advisory review** | Rules that are not mechanised become optional in practice: naming, "claims match what is implemented" (D4), "right for the reason it claims" (B10) | Mechanise what can be (Part I, §2.4); a human audit of a sample of merged PRs at each minor release; track the AI reviewer's findings to decide when it may block | open |
| F4 | **Benchmark gaming** | Minor versions are earned by benchmark gains, so an agent can specialise on the benchmark sizes | Public cases defined with the maintainers; thresholds chosen by humans (a factor 2 to begin with); correctness asserted in every run (Part I, §6.3) | mitigated |
| F5 | **The two scattering loops diverge** | In STL they are already two loops (\~190 and \~700 lines). Overriding the mother loop invites copy-paste | `STOperatorTensorProd` overrides only a fixed set of hooks (Part II, §12.6); a line budget on its file; the single-block invariance test | mitigated |
| F6 | **Backend neutrality on paper only** | Step 1 has no JAX backend, so PyTorch idioms can settle into L1 and L2 unnoticed, and the JAX port becomes a rewrite | AST checks (Part I, §2.4) and the JAX trace test from phase P1 (Part I, §3.1) | **the main risk of step 1** |
| F7 | **The memory wall** | The compact layout shrinks outputs, but the largest tensor products fail on intermediates (about 34 GB for the first layer of 2D+1D+1D) | `estimate_memory`; a budget error; the Rust engine bounds memory by block for HEALPix and 2D; contraction of the second-layer means (LF31); streaming later (LF19) | partly mitigated in step 2 |
| F8 | **Fragile second-order derivatives** | The modulus at zero, NaN gradients through `where`, the regularised division, in-place operations | The rules of Part I, §3.2.3 and their float64 tests | mitigated |
| F9 | **Major-version churn** | Many mathematical changes are queued (Part III, §20); if each changed default were its own major, versions would climb fast and downstream pins break | Opt-in first; `STATISTICS_REFERENCE` separates value changes from interface changes, and whether a value change is a major is decided case by case (Part I, §7.1) | mitigated |
| F10 | **Reproducibility across hardware and dependencies** | float32 GPU results vary with GPU generation and CUDA versions; synthesis trajectories move with PyTorch, optax and XLA | Tolerances per device class; deterministic mode in tests; a lock file (Part I, §1.6); synthesis references per backend, regenerated with human approval through `SYNTHESIS_REFERENCE` (Q6) | mitigated |
| F11 | **Agents misconfigure a wide API** | STL's `ST_Operator` takes about 25 arguments, some set in two places | Options set once (rule A6), validated once, with errors that name the fix; `describe()`, `estimate_memory()`, the manifest | mitigated |
| F12 | **The HEALPix dependency** | STL's HEALPix path relies on healpix-analyse: young, PyTorch-only, heavy dependencies | The parts STL uses are ported into ESTEL (Part II, §9.3); healpix-analyse is test-only, pinned at `aac5e26` | resolved |
| F13 | **Step 2 leaks into step 1** | The temptation to "just add" a small improvement while porting | The step rule (Part I, §1.3), golden rule 11 (with its one listed exception, synthesis checkpoints), review rule X9; every non-STL behaviour shows up as a parity failure | mitigated |
| F14 | **STL patterns get copied** | Class-name checks, `__class__` reassignment, module-level backend globals, mutable `conv_history`, complex storage of real quantities | AST checks; in step 1, full-grid tensors keep STL's shapes and dtypes for parity, and real storage of real quantities comes with the compact layout (LF14); a "do not copy" section in `docs/agents/architecture.md` | mitigated |
| F15 | **Downstream breaks on internals** | Implicit differentiation downstream calls the pure functions directly; anything that looks public will be used | `estel.functional` is in the manifest; the public/internal rule and its CI check (Part I, §2.4) | mitigated |
| F16 | **Name squatting** | `estel` is free on PyPI | Publish 1.0.0 on PyPI as soon as it is ready (Part II, §15, P8) | planned |
| F17 | **Engine and mathematics diverge** | Two implementations of the same statistics (Python, Rust) | One repository: a change of mathematics updates the oracle, the Python engine and the Rust engine in the same pull request; static plan mapping and engine parity tests (Part III, §18.2) | mitigated |
| F18 | **This document and the reference document drift** | After 1.0.0, two documents describe ESTEL | The reference document is written with 1.0.0, in `docs/reference.md`, including the register of later features and §24; this document is then archived | planned (P8) |
| F19 | **Reserved interfaces that constrain step 2 wrongly** | An interface designed before its use may not fit it | Keep reserved interfaces minimal and tested; remove or redesign them in a major rather than working around them | accepted |
| F20 | **The parity harness cannot be rerun** | STL's dependencies drift and its environment can no longer be installed | A pinned container image, archived with the parity report (Part II, §14.2) | mitigated |
| F21 | **A late, heavy review** | The CODEOWNERS review step 1 as a whole, at its end: problems found late are costly, and the review itself is large | Each phase produces its parity report and test results; the maintainers may look at any phase as soon as it ends; the work breakdown (Part II, §15.2) keeps each change small and traceable | open |
| F22 | **Deferred points are forgotten** | Many decisions defer work to later steps | Every deferred feature is in the register (Part III, §16), every other deferred item in §24, each with its source; both are carried over into the ESTEL reference document | mitigated |

## 22. Decision log

Decisions taken from 4 to 12 November 2026. Tables 22.1 to 22.3 record the decisions up to draft 0.2, updated where later decisions replaced them; 22.4 records the review round of 12 November 2026, point by point.

### 22.1 Scope and organisation

| Topic | Decision | Ref. |
| --- | --- | --- |
| Name | ESTEL — Estel is a Scattering Transform Exquisite Library | round 4 |
| Development | Two steps: transfer STL exactly (1.0.0), then extend. Whatever STL does not compute belongs to step 2. Interfaces for step 2 are reserved in 1.0.0 | round 4 |
| Documents | This transfer document; then the ESTEL reference document, without reference to STL, written with 1.0.0 in the repository | round 4, S10 |
| Repository | One new, public repository, holding the Python package and the Rust engine | Q9, round 4, C11 |
| Coefficients | Vanilla ScatCov only; Sihao variant and WPH out of scope | Q7 |
| Fourier-space wavelets | Obsolete in ESTEL | Q2, comments |
| Applications | Component separation and inverse problems stay downstream; they need true implicit differentiation | N2 |
| Utilities | Plotting and diagnostics stay in the core | N24 |
| Later features | Gathered in Part III, §16; other deferred items in §24; both carried over into the reference document | round 4, S10 |

### 22.2 Architecture and backends

| Topic | Decision | Ref. |
| --- | --- | --- |
| Classes | A simple mother class, overridden for tensor products; explicit 1D and 2D classes; one `STOperator`, one `STOperatorTensorProd`, one `STStatistics`, one generic `WaveletOperator` branching on `ndim`; class names in CapWords | Q3, Q4, Q17, N20, A9 |
| Backends | PyTorch in step 1; JAX in step 2; L0 and JAX-ready rules from the first commit | Q9, Q10, round 4 |
| Selection | Global default, fixed when an operator is built; float32 default; float64 path for tests | Q12, N11, N14 |
| Primitive layer | `array-api-compat` as the base of L0 | N12 |
| HEALPix | From `HealpixUpdate`, ported into ESTEL (no healpix-analyse at run time); on JAX after 1.0.0 | N1, Q3, C9 |
| Rust engine | STL-Dev-MPI as the Rust engine, in the ESTEL repository under `rust/` (wheels `estel-rust`, optional, and `estel-sht`); an engine, not a backend; order 1 now, second order by hand later; float64 by default | Q10–Q16, Q19–Q23, C11, C12, C16, C23 |
| HEALPix spectra | Not in 1.0.0; later, through the Rust anafast `estel-sht` | Q17, Q18, C22 |
| Compact layout | Kept in the design, implemented after 1.0.0; "dense view" renamed "full-grid view" | N21, comments, round 4 |
| Low-pass | Default for every geometry and every coefficient, including S3/S4 — in step 2 | Q5, N7, round 4 |

### 22.3 Process

| Topic | Decision | Ref. |
| --- | --- | --- |
| Versions | MAJOR.MINOR.PATCH, plus `STATISTICS_REFERENCE` and `SYNTHESIS_REFERENCE`; whether a change of statistics values is also a major is decided case by case | Q16, N22, C21 |
| Merging | Auto-merge when green; protected paths need CODEOWNERS; AI review advisory | N16, N17 |
| Before 1.0.0 | 1.Pull requests merge when CI passes; the three CODEOWNERS review step 1 as a whole at its end | Q8, C4 |
| Tolerances | Per dtype, rtol and atol; the normalisation references are reference values too | Q14, C20 |
| Synthesis | Built-in optimiser per backend; references per backend; synthesis tests and parity after 1.0.0 | N10, N13, Q6, S3 |
| Agents | Context files mirrored for every agent; a machine-readable API manifest | N18, N19 |
| Mathematics | `maths.md` as an executable specification; a disagreement fails CI; during the review, the maths tab is its reference version | N25, C7 |
| Deferred | Q1 (canonical set) and Q2 (gradients through the reference): after 1.0.0 | round 3 |
| Document | One tab per part; a short contents in the Overview tab, a developed one in each part | round 5 |

### 22.4 Review round of 12 November 2026

Every point of the review list (C: critical before the start, S: needed during step 1, A: API and naming), with the decision and where it is applied. "Traced" means the point is not implemented now and is recorded where shown, so that it is not forgotten.

| ID | Point | Decision | Where |
| --- | --- | --- | --- |
| C1 | Licence | Apache-2.0, as set out by Pablo in Part I, §1.5 | §23, R5 and R11 |
| C2 | Minimum versions | Python 3.11, NumPy 2.0, PyTorch 2.6, array-api-compat 1.11 (JAX 0.5 and optax 0.2.4 in step 2), compatible with the 2025 versions; confirmed | Part I, §1.6; §23, R13 |
| C3 | STL sources | Frozen at the commits of Part II, §9.1 | Part II, §9.1, §9.2 |
| C4 | Humans in the loop | CODEOWNERS: Erwan Allys (`Eralys`), Jean-Marc Delouis (`jmdelouis`) and Pablo Richard (`pablo-richard`), who review step 1 at its end | Part I, §7.1; §23, R12 |
| C5 | Public API | The current document holds; where it is silent, STL is the default, and doubts are raised with the maintainers | Part II, §12.1 |
| C6 | Public vs internal | Agreed: manifest, private modules, underscores, CI check | Part I, §2.4, §8.2; Part II, §11, §12 |
| C7 | `maths.md` | Started; stick to STL; improvements are for step 2 | maths tab; Part II, §15.2, T0.5 |
| C8 | Open points of STL's mathematics | G4 later (LF26); G5 defined in the maths tab | Part II, §13.2 |
| C9 | healpix-analyse | Port what is needed; no runtime dependency | Part II, §9.3; Part I, §1.6 |
| C10 | healpix-analyse version for parity | The latest pushed version, `aac5e26` (v2026.10.2) | Part II, §9.1, §14.2 |
| C11 | Rust engine repository | Everything in the same repository | Part II, §11; Part III, §18.2 |
| C12 | Naming | "rust", not "native" | everywhere |
| C13 | Work breakdown | Done | Part II, §15.2 |
| C14 | Agent context and policy files | Drafted from Part I | Part I, §8; draft files of 12 November 2026 |
| C15 | `apply` or `forward` | `apply` kept; Pablo is considering the question | §24.2 |
| C16 | Rust: engine or backend | An engine, kept as it is; splitting it into primitives would lose its gains | Part I, §2.2.4 |
| C17 | Masks | Static, as in STL | Part I, §2.3, §3.1 |
| C18 | Loops and memory | Step 1 keeps STL's loop; contraction in step 2; any reorganisation after a benchmark | Part I, §3.1; Part III, §19.3, LF31 |
| C19 | Translation invariance | Not tested in 1.0.0; step 2 | Part I, §5.1; Part III, §20.8, LF25 |
| C20 | Reference S2 scale | Compared too, with the S2 coefficients | Part I, §5.1, §5.2 |
| C21 | Versioning | Three numbers plus two reference counters; majors for value changes decided case by case | Part I, §7; §24.2 |
| C22 | Cross-spectrum conflict | 1.0.0: 2D spectra as in `master` only; 1D, tensor-product and HEALPix spectra later | Part II, §9; Part III, LF23, LF24 |
| C23 | Second order | For PyTorch and JAX now; for Rust later | Part I, §3.2.4; Part III, LF15 |
| S1 | JAX trace test | Yes, from P1 | Part I, §3.1 |
| S2 | Contracts of the L0 primitives | Those that affect values are in `maths.md` | Part II, §12.3 |
| S3 | Parity cases | Scattering statistics first; synthesis tests after 1.0.0 | Part I, §5.1; §24.1 |
| S4 | Reference inputs | No SST crop; STL's data are public | Part I, §5.1 |
| S5 | Parity precision | STL runs in float64 | Part II, §14.3 |
| S6 | Single-block tensor product | To check; the plain geometry is the reference | Part I, §5.1; Part II, §14.4 |
| S7 | Utilities | Later | Part III, LF11 |
| S8 | GPU CI | After 1.0.0 | Part I, §6.1; §24.1 |
| S9 | Benchmark thresholds | A factor 2 in speed or memory, to begin with | Part I, §6.3 |
| S10 | Reference document | Written with 1.0.0, in the repository, with the register of later features | Part II, §11, §15; F18 |
| S11 | PyPI | As soon as 1.0.0 is ready | Part II, §15, P8; F16 |
| S12 | More invariance tests | Added now | Part I, §5.1 |
| S13 | Cross-backend synthesis and gradients | After 1.0.0, traced | §24.1 |
| S14 | SST crop | Removed | Part I, §5.1 |
| S15 | Synthesis budget | To discuss later | §24.2 |
| S16 | Hidden benchmark pool | No: public cases defined together | Part I, §6.3 |
| S17 | `sum(w)` | Precomputed in the plan | Part I, §3.1.1; Part II, §12.4 |
| S18 | Memory of immutable objects | Explained; measured in P2 | Part I, §3.1.1 |
| A1 | `filters_fourier` | Moved to `estel.diagnostics.frequency_response` | Part II, §12.5, §12.10 |
| A2 | `littlewood_paley` | Moved to diagnostics; guaranteed by construction, tested in CI | Part II, §12.5, §12.10 |
| A3 | `_finalize` | Renamed `_normalize` | Part II, §12.6 |
| A4 | `to_vector` argument | `mask_st`, True = keep | Part II, §12.7 |
| A5 | Spectra interface | One `method=` for every geometry, HEALPix included | Part II, §12.8 |
| A6 | Document title | Kept | — |
| A7 | Part II cross-references | Now name their tab | Part II |
| A8 | Rust second order | Developed later (LF15); automatic routing to the Python engine to discuss | §24.2 |
| A9 | Class names | CapWords: `STOperator`, `STOperatorTensorProd`, `STStatistics` | Part I, §2.3; Part II, §12 |
| A10 | Tensor-product dispatch | Along each block's axes, without `swapaxes`; benchmarked in P3 | Part II, §12.4, §15 |
| A11 | Synthesis checkpoints | In step 1, although STL lacks them | Part II, §9.2, §12.9 |
| A12 | G3, partial order | Written explicitly with its consequences; relevance examined later | Part II, §13.2; Part III, §20.9, LF27 |
| — | float32 readiness (Erwan's comment on Part I) | Every computation runs in float32 as well as float64 | Part I, §3.1, rule 7 |
| OP-24 | Morlet wavelets | Excluded: bump-steerable only; a general optimisation of the wavelets later (LF22) | Part II, §10; maths tab |
| OP-3, OP-8 | Where STL fails | ESTEL raises a clear error; parity is compared only where STL works | Part II, §9.2; maths tab |
| OP-26, OP-27 | Masks: NaN filling in STL, weights in ESTEL | Parity cases with masks from phase P2 | Part II, §15.2, T2.4 |
| OP-5 | iso and angular\_ft on complex coefficients | Reproduced in 1.0.0; a problem to fix afterwards | Part III, LF32 |
| — | Philosophy of step 1 | Reproduce STL with its problems, then fix most of them as traced changes | Part II, §9.2 |
| — | JAX in step 1 | An experimental JAX backend from P1: every L0 primitive, the conformance suite and the whole test suite on both backends; not covered by the 1.0.0 guarantees; syntheses on JAX and stability in step 2 (LF1). Replaces the minimal trace-only JAX L0 | Part I, §2.2.5, §3.1; Part II, §15.2, T1.3 |

## 23. Open questions

The questions of draft 0.2 with their answers, and the questions raised on 12 November 2026 (R11 to R13), now answered. R1 and R2 are decided in step 2.

| # | Question | Status | Where |
| --- | --- | --- | --- |
| R1 | **Compact layout, implementation.** Should the conversion between structured arrays and the compact buffer be written once, in shared Python on a few generic primitives, or separately in each backend? Proposed: once, with fused kernels only if a benchmark asks for them | deferred: decided in step 2 | Part III, §19.9 |
| R2 | **Low-pass in S3/S4.** Confirm "all combinations" as the default (+38–112% coefficients); should the restricted variant (+0.3–3.5%) exist as an option? | deferred: decided in step 2 | Part III, §20.1 |
| R3 | **Utilities in step 1?** | answered: later (LF11) | Part II, §12.10 |
| R4 | **GPU CI on the cluster** | answered: after 1.0.0; details then | Part I, §6.2; §24.1 |
| R5 | **Licence** | answered: Apache-2.0 | Part I, §1.5 |
| R6 | **healpix-analyse in step 1** | answered: ported, no runtime dependency | Part II, §9.3 |
| R7 | **Parity precision** | answered: STL runs in float64 | Part II, §14.3 |
| R8 | **Single-block tensor products** | answered: to check in P3; the plain geometry is the reference | Part I, §5.1 |
| R9 | **The ESTEL reference document** | answered: written with 1.0.0, in `docs/reference.md`, with the register of later features | Part II, §11, §15 |
| R10 | **JAX trace test in step 1** | answered: yes | Part I, §3.1 |
| R11 | **Licence text.** Part I, §1.5, added on 12 November 2026 in another editing session, states Apache-2.0, with conditions on STL's authors and on the Rust code. The review answer C1 leaves the choice to the creation of the repository. Answered: §1.5, written by Pablo, is what we want. Status: answered. | answered: Apache-2.0 | Part I, §1.5 |
| R12 | **CODEOWNERS.** GitHub handles of Jean-Marc Delouis and Pablo Richard. Answered: `jmdelouis` and `pablo-richard` | answered | Part I, §7.1 |
| R13 | **Minimum versions.** Confirm Python 3.11, NumPy 2.0, PyTorch 2.6 and array-api-compat 1.11 (and JAX 0.5, optax 0.2.4 for step 2). Answered: confirmed | answered: confirmed | Part I, §1.6 |

## 24. Deferred work and points to discuss later

Everything decided as "later" that is not a feature — features are in the register, Part III, §16 — with its source. This section is carried over into the ESTEL reference document with 1.0.0.

### 24.1 Deferred work

| ID | Work | When | Source |
| --- | --- | --- | --- |
| W1 | Tier-1 GPU CI on the cluster: scheduler, container runtime, network rules of the gateway node (Part I, §6.2) | after 1.0.0 | S8, R4 |
| W2 | Synthesis tests: per-backend reference outputs, cross-backend acceptance criteria, synthesis parity with STL | after 1.0.0 | S3 |
| W3 | Cross-backend checks: syntheses matched over a few iterations when the optimisers can be standardised; gradients backend-independent up to numerical precision | after 1.0.0, with JAX (LF1) | S13 |
| W4 | Tuning guides for syntheses (`docs/agents/tuning/`) | with W2 | C14 |

### 24.2 Points to discuss later

| ID | Point | With | Source |
| --- | --- | --- | --- |
| D1 | Keep `apply`, or also offer PyTorch's `__call__` and `forward`, so that operators plug into ML libraries like any module | Pablo | C15 |
| D2 | Whether a change of statistics values is also a major version: decided case by case | the maintainers | C21 |
| D3 | Budget when a pull request legitimately changes synthesis results: run syntheses to convergence, possibly for a few hours | the maintainers | S15 |
| D4 | With `engine="auto"`, route second-order requests to the Python engine automatically while the Rust engine is at order 1 | the maintainers | A8 |
| D5 | The hidden subset of reference inputs (Part I, §5.1), now that the benchmark cases are public | the maintainers | S16 |
