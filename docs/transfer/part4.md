# Part IV — Review and decisions

[Back to the overview](overview.md)

## Contents of this part

- [21. Critical review: failure modes](#21-critical-review-failure-modes)
- [22. Decision log](#22-decision-log)
  - [22.1 Scope and organisation](#221-scope-and-organisation) · [22.2 Architecture and backends](#222-architecture-and-backends) · [22.3 Process](#223-process) · [22.4 Review round of 12 November 2026, and decisions of 13 November 2026](#224-review-round-of-12-november-2026-and-decisions-of-13-november-2026) · [22.5 Review of 16 November 2026](#225-review-of-16-november-2026) · [22.6 Questions of drafts 0.2 to 0.4, answered](#226-questions-of-drafts-02-to-04-answered)
- [23. Points to settle](#23-points-to-settle)
- [24. Deferred work and points to discuss later](#24-deferred-work-and-points-to-discuss-later)

## 21. Critical review: failure modes

The plan is sound in its core: a frozen skeleton, versions tied to reference values, applications kept downstream, backends kept out of the classes, an agent-readable context, and now a two-step development that separates reproducing STL from extending it. The risks below are where it can still fail, ordered by severity, each with its mitigation.

| # | Failure mode | Why it happens here | Mitigation | Status |
| --- | --- | --- | --- | --- |
| F1 | **References that freeze bugs** | If reference values were produced by ESTEL itself, its bugs would become the truth | References from the independent `_spec` oracle, checked against STL in step 1 (Part II, §14.5), signed off by the three CODEOWNERS | largely mitigated by the two steps |
| F2 | **Agents make CI green the wrong way** | With no human review of each PR, the cheapest fix for a red test is to edit it: regenerate references, loosen tolerances, skip, special-case the input | Protected paths, a hash manifest, checks on test count and skips (Part I, §5.3), a hidden subset of reference inputs run by the maintainers before each release | open, monitored |
| F3 | **Quality drifts under advisory review** | Rules that are not mechanised become optional in practice: naming, "claims match what is implemented" (D4), "right for the reason it claims" (B10) | Mechanise what can be (Part I, §2.4); a human audit of a sample of merged PRs at each minor release; track the AI reviewer's findings to decide when it may block | open |
| F4 | **Benchmark gaming** | Minor versions are earned by benchmark gains, so an agent can specialise on the benchmark sizes | Public cases defined with the maintainers; thresholds chosen by humans (a factor 2 to begin with); correctness asserted in every run (Part I, §6.3) | mitigated |
| F5 | **The two scattering loops diverge** | In STL they are already two loops (\~190 and \~700 lines). Overriding the mother loop invites copy-paste | `STOperatorTensorProd` overrides only a fixed set of hooks (Part II, §12.6); a line budget on its file; the single-block invariance test | mitigated |
| F6 | **Backend neutrality on paper only** | If step 1 ran only on PyTorch, PyTorch idioms could settle into L1 and L2 unnoticed, and the JAX port would become a rewrite | AST checks (Part I, §2.4) and the experimental JAX backend, which runs the whole test suite from phase P1 (Part I, §3.1) | largely mitigated |
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
| F22 | **Deferred points are forgotten** | Many decisions defer work to later steps | Every deferred feature is in the register (Part III, §16), every other deferred item in Part III, §16.7, each with its source; both are carried over into the ESTEL reference document | mitigated |

## 22. Decision log

Decisions taken from 4 to 16 November 2026. Tables 22.1 to 22.3 record the decisions up to draft 0.2, updated where later decisions replaced them; 22.4 records the review round of 12 November 2026, and decisions of 13 November 2026, point by point; 22.5 records the review of 16 November 2026, and 22.6 the questions of drafts 0.2 to 0.4, answered.

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
| Later features | Gathered in Part III, §16; the deferred work in Part III, §16.7; both carried over into the reference document | round 4, S10 |

### 22.2 Architecture and backends

| Topic | Decision | Ref. |
| --- | --- | --- |
| Classes | A simple mother class, overridden for tensor products; explicit 1D and 2D classes; one `STOperator`, one `STOperatorTensorProd`, one `STStatistics`, one generic `WaveletOperator` branching on `ndim`; class names in CapWords | Q3, Q4, Q17, N20, A9 |
| Backends | PyTorch in step 1, with JAX experimental from P1; JAX stable in step 2; L0 and JAX-ready rules from the first commit | Q9, Q10, round 4 |
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
| Tolerances | Per dtype, rtol and atol; the normalisation references are reference values too | C20 |
| Synthesis | Built-in optimiser per backend; references per backend; synthesis tests and parity after 1.0.0 | N10, N13, Q6, S3 |
| Agents | Context files mirrored for every agent; a machine-readable API manifest | N18, N19 |
| Mathematics | `maths.md` as an executable specification; a disagreement fails CI; during the review, the maths tab is its reference version | N25, C7 |
| Deferred | Q1 (canonical set) and Q2 (gradients through the reference): after 1.0.0 | round 3 |
| Document | One tab per part; a short contents in the Overview tab, a developed one in each part | round 5 |

### 22.4 Review round of 12 November 2026, and decisions of 13 November 2026

Every point of the review list (C: critical before the start, S: needed during step 1, A: API and naming), with the decision and where it is applied. "Traced" means the point is not implemented now and is recorded where shown, so that it is not forgotten.

| ID | Point | Decision | Where |
| --- | --- | --- | --- |
| C1 | Licence | Apache-2.0, as set out by Pablo in Part I, §1.5 | §23, R5 and R11 |
| C2 | Minimum versions | Python 3.11, NumPy 2.0, PyTorch 2.6, array-api-compat 1.11 (JAX 0.5 for the experimental backend; optax 0.2.4 in step 2), compatible with the 2025 versions; confirmed | Part I, §1.6; §23, R13 |
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
| C18 | Loops and memory | Step 1 keeps STL's loop; contraction in step 2; any reorganisation after a benchmark. Replaced by the row "Efficiency in step 1" of §22.5 | Part I, §3.1; Part III, §19.3, LF31 |
| C19 | Translation invariance | Not tested in 1.0.0; step 2 | Part I, §5.1; Part III, §20.8, LF25 |
| C20 | Reference S2 scale | Compared too, with the S2 coefficients | Part I, §5.1, §5.2 |
| C21 | Versioning | Three numbers plus two reference counters; majors for value changes decided case by case | Part I, §7; §24.2 |
| C22 | Cross-spectrum conflict | 1.0.0: 2D spectra as in `master` only; 1D, tensor-product and HEALPix spectra later | Part II, §9; Part III, LF23, LF24 |
| C23 | Second order | For PyTorch and JAX now; for Rust later | Part I, §3.2.4; Part III, LF15 |
| S1 | JAX trace test | Yes, from P1; superseded by the experimental JAX backend (row 'JAX in step 1') | Part I, §3.1 |
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
| OP-3, OP-8 | Where STL fails | ESTEL raises a clear error; parity is compared only where STL works; OP-3 revised (row 'OP-3 revised') | Part II, §9.2; maths tab |
| OP-26, OP-27 | Masks: NaN filling in STL, weights in ESTEL | Parity cases with masks from phase P2 | Part II, §15.2, T2.4 |
| OP-5 | iso and angular\_ft on complex coefficients | Reproduced in 1.0.0; a problem to fix afterwards | Part III, LF32 |
| — | Philosophy of step 1 | Reproduce STL with its problems, then fix most of them as traced changes | Part II, §9.2 |
| — | JAX in step 1 | An experimental JAX backend from P1: every L0 primitive, the conformance suite and the whole test suite on both backends; not covered by the 1.0.0 guarantees; syntheses on JAX and stability in step 2 (LF1). Replaces the minimal trace-only JAX L0 | Part I, §2.2.5, §3.1; Part II, §15.2, T1.3 |
| — | Littlewood–Paley | No pass/fail criterion in 1.0.0: STL's kernels are reproduced and do not satisfy it. It becomes a requirement of LF22, low-pass filter included (LF6) | Part II, §12.5; Part III, LF22, §20.10 |
| — | Smoothing kernel | "H" renamed "smoothing kernel" s (M-SMOOTH-2D, M-SMOOTH-1D), after STL's name. The low-pass filter φ\_J is a separate object, realised in 1.0.0 as the cascade of the smoothing kernels | maths tab; Part I, §4.3; glossary |
| — | Final downsampling of the low-pass filter | Parity with STL in 1.0.0 (the final downsampling is kept); without it in step 2 (LF6), as a traced change of reference values | Part III, LF6, §20.10 |
| — | Joint design of s and ψ | LF6, LF22 and LF25 are designed together, with the requirements of Part III §20.10; the three diagnostics of Part II §12.10 measure them | Part III, §20.10; Part II, §12.10 |
| — | `WaveletOperator` L | Any integer L ≥ 1, as in STL | Part II, §12.5 |
| OP-3 | OP-3 revised | A mean with no valid pixel gives a NaN coefficient, as in STL, decided statically by the plan; ESTEL warns once and raises only if a whole family is empty. The first decision (an error) would have refused many masked operators, since coarse levels often lose all their valid pixels | maths tab, M-MEAN, OP-3; Part II, §9.2 |
| — | Centred covariances | The scattering covariances are covariances by definition; STL's raw moments are a simplification. 1.0.0 reproduces STL. Step 2 centres by default, only where the plan cannot prove a zero mean, by correcting coefficients; one window per level; band-pass kernels sum to zero by requirement; `centre` and `norm` are independent | Part III, LF33, §20.11; maths tab, OP-36 |
| — | Mean of the data | Removed first and stored apart, explicitly; if it is not, the terms ending with a low-pass filter are centred explicitly | Part III, §20.11 |
| — | T2.5 public interface | norm_batch_mean False as in STL; reference stored in the operator, internal; LF21 closed; functional.scattering(x, plan, filters, ref=None) with ScatteringPlan and ScatteringFilters public and opaque; channels and compute_cross_matrix fixed at construction; MPI hook _global_sum, internal; "reduction" reserved for iso, angular_ft and scale_ft; one select method; to_vector as STL (OP-37) | Part II, §9.4, §11, §12.6, §12.7; Part III, LF21; maths tab, OP-37 |

### 22.5 Review of 16 November 2026

The review of the scattering chain (86 remarks, R-A00 to R-P17) was distributed: each remark went to the maths tab, to the register (Part III, §16.6), to the points to settle (§23), or here when it was decided. Its ID is kept as the source.

| ID | Point | Decision | Where |
| --- | --- | --- | --- |
| — | Lists of the specification | four lists, one purpose each: open points of the mathematics (OP, maths tab), everything later (register, Part III, §16), everything decided (this log), everything still to decide (§23). The G list of Part II, §13.2 is retired; §23 and §24.2 are merged into §23; §24.1 moves to Part III, §16.7 | Part II, §13.2; Part III, §16; §23 |
| R-A00 | Default channel pairs | `compute_cross_matrix=None` computes every pair c₁ ≤ c₂, as STL's code | maths tab, M-CHAN, OP-38 |
| R-A01 | Index numbering | STL's numbering, as a rule: indices numbered from the end. j₃, ℓ₃ is the wavelet applied last; j₂, ℓ₂ the arm of S3, also the second arm of S4; j₁, ℓ₁ the extra arm of S4. S3 is an S4 whose first arm is replaced by the field. S1 and S2 keep (j, ℓ) | maths tab, M-S3, M-S4 |
| R-A02 | `norm` | values `"none"`, `"self"`, `"ref"`; `norm_batch_mean` becomes `ref_batch_mean` | PR #76 |
| R-A03 | Spelling of identifiers | American (`normalize`, `standardize`, `center`); prose may stay British | PR #76 |
| R-A04 | Prefix | `STPlan`, `STFilters` | PR #76 |
| R-A05 | Channel pairs | `cross_matrix`, its diagonal selecting the auto terms; `compute_PS` kept | PR #76 |
| R-A06 | `mask_st` | kept, documented: True keeps a coefficient | — |
| R-A07, R-S08 | Reductions | one method `reduce(*, iso=False, angular_fourier=False, angular_harmonics=None, scale_cosine=False, scale_harmonics=None, dj=None)` instead of three: it applies the selected reductions in the order of M-NORM (`iso`, then `angular_fourier`, then `scale_cosine`) in one call; harmonics and `dj` default to all; `scale_cosine` needs `iso`, as in STL; reduced statistics are not reduced again | PR #76 |
| R-A08 | Mother data class | `Data` | PR #76 |
| R-A09 | `apply` | kept, for the symmetry with `WaveletOperator`; a direct call may be added later as a minor change. Closes D1 | — |
| R-A10 | `standardize` | stays a boolean: it removes the mean and divides by the standard deviation. Part III, §20.11, rule 7 is to correct accordingly | Q17 |
| R-A11 | `functional.scattering` | first argument named `data` | PR #76 |
| R-A12 | Messages | errors, warnings and the docstrings of public names speak ESTEL's vocabulary and do not mention STL; the link with STL is in maths.md, through the M- and OP- references they give | PR #76 |
| R-Y03 | A reference from another operator | accepted when J, L and the channels match, with a clear error otherwise | PR #68 |
| G1 | Coefficient count of plain 2D | follow STL: 1,474, since plain 2D has no low-pass term in STL (the deliverable's formula gives 1,476) | Part II, §13.2; LF6 |
| R-E01, R-N19 | Smoothing table, kernel rotations | recorded as OP-59 and OP-58 | maths tab |
| — | Wavelet operator | the only way to the geometry: one generic `WaveletOperator` provides, level by level, the convolution, the downsampling, the low-pass filter, the shapes of the levels and the rules of borders and masks; `STOperator`, its plan and its loop do not depend on the geometry; the code of each geometry stays in its `Data` hooks and table builders, called by the wavelet operator only; this replaces the public wrapper first planned for T2.10 | Part I, §2.1, §2.3; Part II, §12.4 to §12.6, T2.10, T3.1, T5.1 |
| — | Invariance under rotations | not tested in 1.0.0: STL's 2D kernels match a rotation by π/2 only to 3e-3 to 9e-3 (OP-58), so the test comes with the joint design of the filters, with the conjugation of the orientations that leave [0, π). The other invariances of Part I, §5.1 are tested in T2.8, where they hold to rounding with STL's definitions: not for the scaling of the cross S1 (OP-16), nor for the constant shift of 2D data with a mask (OP-59) | Part I, §5.1; Part III, §20.10 |
| — | Wavelet operator, internal | not public: ESTEL computes scattering statistics, and STL's `data.get_wavelet_op()` has no public counterpart. Its functions work on maps and levels, and `Data` carries no level; `apply` keeps its name without the symmetry of R-A09. Only the hooks of filtering and downsampling go through it; `_nyquist_filter` and `_spectrum_geometry` serve the syntheses and the spectra | Part I, §2.1, §2.3, §2.4; Part II, §10, §11, §12.4, §12.5, T2.10 |
| — | Efficiency in step 1 | Before 1.0.0, a reorganisation that makes the computation faster or lighter belongs to step 1 when every value stays within the parity tolerance with STL and the benchmark shows its gain: done early, it guides the developments that follow, and parity with STL checks that it changes nothing. Replaces C18. Order in P2: a benchmark of three 2D cases, on CPU, outside CI (T2.11); the Gram form with the band orientation as leading axis and the mean weights stored once (T2.12, LF31); small optimisations together, each measured (T2.13: windows without weights, R-S01; the Hermitian symmetry computed once; real arithmetic, R-P06; compilation on JAX; small kernels as constants, LF29); T2.14 closed without change (level-wise gradients, LF36, no gain; a first-order gradient on PyTorch, LF46, deferred); the compact layout (T2.15, LF14); then the planar syntheses (T6.0, T6.1), compared with STL over the first iterations, which validate the optimised engine end to end. Until they exist, each optimisation compares its gradient with that of `main` on the benchmark's cases. Measured on CPU first; a final optimisation session on native x86 Linux and on GPU confirms the conclusions and decides the choices that depend on the platform (T7.1). Tried after the implementation of their phase that follows STL: the recursion over blocks with a cache of prefixes (Q11), complex-to-complex convolutions (LF30), the HEALPix tables per pixel batch and the folded kernel (Q9). Fine tuning, done only on a specific need: pruned plans (LF37), sibling operators (LF45). After 1.0.0: partitioned execution (LF34), the Rust engine (LF3) | Part I, §3.1, §5.1, §6.3; Part II, §14, §15.1, §15.2; Part III, §16, §19.3, §19.12, §19.13; Q9, Q11 |
| — | Global optimisations only in 1.0.0 | An optimisation of 1.0.0 applies to every use. A choice that depends on the use, such as a gradient primitive for losses that are not differentiated again (LF46), comes after 1.0.0 | Part III, §16, LF46; Part II, §15.2, T2.14 |
| — | The compact layout at the start of P3 | T2.15 moves from the end of P2 to the start of P3: on planar data the statistics are about a thousandth of the memory of the computation, so its gain cannot be measured there, and its segments and multi-index levels are decided with the recursion over blocks (Q11) | Part II, §15.1, §15.2, T2.15; Part III, §19.14 |
| — | Periodic syntheses only in 1.0.0 | 1.0.0 synthesises periodic fields only. STL's non-periodic and masked syntheses are not reproduced: their choices are to be reconsidered (LF47). A target with missing values or `pbc=False` raises a clear error | Part II, §9.3, §12.9, §14; Part III, §16, LF47 |
| — | The vector of the synthesis loss (Q14, closed) | 1.0.0 reproduces STL: the loss uses the complex form of M-FLAT, which amounts to a complete real vector (real and imaginary parts of every coefficient); `to_vector` stays STL's `to_flatten(flatten_complex=True)`. Whether they become one vector is decided after 1.0.0 | maths tab, M-SYN-LOSS, M-FLAT, OP-37, OP-39 |
| — | Consistency of the estimators | A new family of tests: periodic, non-periodic and masked estimates of one stationary process (the same field read three ways, paired) agree within their statistical error, with a threshold corrected for the number of comparisons and a positive control; a systematic gap is measured and recorded, not corrected in step 1 (T2.16) | Part I, §5.1; Part II, §15.2, T2.16 |

### 22.6 Questions of drafts 0.2 to 0.4, answered

The questions of drafts 0.2 to 0.4 and of 12 November 2026, all answered. R1 and R2, still open, became Q1 and Q2 (§23).

| # | Question | Status | Where |
| --- | --- | --- | --- |
| R3 | **Utilities in step 1?** | answered: later (LF11) | Part II, §12.10 |
| R4 | **GPU CI on the cluster** | answered: after 1.0.0; details then | Part I, §6.2; §24.1 |
| R5 | **Licence** | answered: Apache-2.0 | Part I, §1.5 |
| R6 | **healpix-analyse in step 1** | answered: ported, no runtime dependency | Part II, §9.3 |
| R7 | **Parity precision** | answered: STL runs in float64 | Part II, §14.3 |
| R8 | **Single-block tensor products** | answered: to check in P3; the plain geometry is the reference | Part I, §5.1 |
| R9 | **The ESTEL reference document** | answered: written with 1.0.0, in `docs/reference.md`, with the register of later features | Part II, §11, §15 |
| R10 | **JAX trace test in step 1** | answered: yes; superseded by the experimental JAX backend (row 'JAX in step 1') | Part I, §3.1 |
| R11 | **Licence text.** Part I, §1.5, added on 12 November 2026 in another editing session, states Apache-2.0, with conditions on STL's authors and on the Rust code. The review answer C1 leaves the choice to the creation of the repository. Answered: §1.5, written by Pablo, is what we want. Status: answered. | answered: Apache-2.0 | Part I, §1.5 |
| R12 | **CODEOWNERS.** GitHub handles of Jean-Marc Delouis and Pablo Richard. Answered: `jmdelouis` and `pablo-richard` | answered | Part I, §7.1 |
| R13 | **Minimum versions.** Confirm Python 3.11, NumPy 2.0, PyTorch 2.6 and array-api-compat 1.11 (JAX 0.5 for the experimental backend; optax 0.2.4 in step 2). Answered: confirmed | answered: confirmed | Part I, §1.6 |

## 23. Points to settle

Everything still to decide, in one table. A point leaves it when it is decided (it then goes to the log, §22) or deferred (to the register, Part III, §16). The source column keeps the earlier IDs.

| ID | Point to settle | With | Before | Source |
| --- | --- | --- | --- | --- |
| Q1 | Compact layout: is the conversion between structured arrays and the compact buffer written once, in shared Python on a few generic primitives, or in each backend? | the maintainers | step 2 (LF14) | R1 |
| Q2 | Low-pass filter in S3 and S4: which combinations (the low-pass filter as the last filter only, +0.3 to +3.5 % coefficients, or also as a first filter, +38 to +112 %), and the half-pixel shift of the even smoothing kernel without the final downsampling | the maintainers | step 2 (LF6) | R2; R-P12 |
| Q3 | Whether a change of statistics values is also a major version: decided case by case | the maintainers | each case | D2 |
| Q4 | Budget when a pull request legitimately changes synthesis results: syntheses to convergence, possibly for hours | the maintainers | T6.1 | D3 |
| Q5 | With `engine="auto"`, route second-order requests to the Python engine while the Rust engine is at order 1 | the maintainers | step 2 (LF3) | D4 |
| Q6 | The hidden subset of reference inputs, now that the benchmark cases are public | the maintainers | P7 | D5 |
| Q7 | HEALPix on the sphere only in 1.0.0, the parity harness running STL with `ellipsoid="sphere"` (an ellipsoid later, LF43) | Jean-Marc, Pablo | T5.0 | R-H01; OP-45 |
| Q8 | float32 inside healpix-analyse: the harness disables the cache; reproduce the float32 kernel, or compare with a patched run | Jean-Marc | T5.0 | R-H02; OP-46 |
| Q9 | HEALPix design: the list of geometry functions to port with a reference test each; the memory of the gather tables (32-bit indices, tables per pixel batch, the largest `nside` of 1.0.0); the kernel folded into the interpolation as one sparse operator per gauge; the geometry cached on disk. The tables per pixel batch, the folded kernel and the cache on disk are tried after the implementation of P5 that follows STL (§22.5) | Jean-Marc | T5.0 | R-H03, R-H04, R-H07 |
| Q10 | HEALPix: confirm that `PatchScattering`, `PyramidScattering`, `LocalScattering`, `S3Multiscale` and `AdjointOps` are not transferred; the edge of a partial sky; the argument name, `depth` or `nside` | Jean-Marc | T5.1 | R-H05, R-H06; OP-50 |
| Q11 | Tensor products: the loop written as a recursion over blocks with a cache of prefixes, and on the Gram form from the start. The recursion with a cache of prefixes is tried after the implementation of P3 that follows STL's loop, on the Gram form that exists by then (T2.12; §22.5) | Pablo | T3.0 | R-T01, R-T06 |
| Q12 | Tensor products: the parity matrix of P3 (2D+1D in full, 2D+1D+1D at the smallest sizes that exercise every branch); block names for `select`; confirm the errors on masks and reductions and `lowpass="ST"` in 1.0.0 | Pablo | T3.0 | R-T03, R-T04, R-T05 |
| Q13 | Document in the reference document that uncentred statistics are biased on short non-periodic blocks, and schedule LF33 early in step 2 | the maintainers | 1.0.0 | R-T02; OP-36, OP-42 |
| Q15 | Syntheses: one named concept for `mean_field`; explicit arguments for J − 1, the weights 3.5 and 3.5², the seed | the maintainers | T6.1 | R-Y02, R-Y04 |
| Q16 | The default length of the L-BFGS history (STL: 50 pairs, 100 numbers per pixel), after a convergence test at 10 and 20 | the maintainers | T6.1 | R-P16 |
| Q17 | Part III, §20.11: rule 1 amended (the maps that end with a low-pass filter are centred themselves, for stability in single precision), and rule 7 corrected (standardisation: mean and standard deviation) | Erwan | before LF33 | R-E02, R-A10 |
| Q18 | The strided second layer (LF38) and the strict scale order (LF35): validation on syntheses and inference, then option or default | the maintainers | step 2, H1 | R-P02, R-P15 |
| Q19 | The scope of the Rust engine, after a benchmark of the reorganised Python engine against STL-Dev-MPI | Jean-Marc | step 2, H2 | R-P09 |
| Q20 | Partitioned execution (LF34) as the single way to scale out, absorbing LF4 and LF19 | Jean-Marc | step 2, before LF4 | R-P04 |
| Q21 | `iso`: return the real parts only of the reduced S3 and S4; the meaning of `iso` in 1D with L = 2 | the maintainers | step 2 (LF32) | R-P11; OP-5, OP-40 |
| Q22 | Group the changes of values of step 2 in one bump of `STATISTICS_REFERENCE` | the maintainers | step 2 | R-P13 |
| Q23 | The power spectrum: computed by the scattering operator (`compute_PS`), or by its own operator only | the maintainers | P4 | R-S07 |
| Q25 | The notation of the maths tab: the table of symbols and the collisions (issue #71) | Erwan | 1.0.0 | R-N01 to R-N18 |
| Q26 | The vocabulary of the code (issue #72) | Erwan | 1.0.0 | R-V01 to R-V10 |
| Q27 | The structure of `docs/maths.md` and of the specification (issue #73) | Erwan | 1.0.0 | R-S03 to R-S06, R-S09 |
| Q28 | L0 primitives with no user at the end of phase P6 are removed; the signature of `gather_weighted` is checked against Q9 | the maintainers | end of P6 | R-S02 |
| Q29 | Preparations in 1.0.0 that change no value: windows with a kind (full, slice, weights), index sets and evaluation positions from the plan, the band orientation as the leading axis of the second-layer maps, families as a list, `estimate_memory` with temporaries and the L-BFGS history, a benchmark with gradients (Part III, §19.12). The windows, the band orientation as the leading axis and the benchmark are now tasks of step 1 (Part II, §15.2, T2.11 to T2.13; §22.5) | the maintainers | 1.0.0 | R-S01, R-P01, R-P14 |
| Q30 | Precision on GPU: on NVIDIA GPUs of the Ampere generation and later, PyTorch computes convolutions in TF32 by default (about 3 significant digits instead of 7 for float32), without warning, while matrix products stay in float32. Disable TF32 during ESTEL's computations, so that float32 keeps its precision, or keep it for speed and check that the parity tolerance holds | the maintainers | the first measurement on GPU (T7.1) | review of 17 November 2026 |

## 24. Deferred work and points to discuss later

Moved on 16 November 2026: the deferred work W1 to W4 is in Part III, §16.7; the points to discuss D2 to D5 are Q3 to Q6 in §23; D1 is closed (§22.5, R-A09).
