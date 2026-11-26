# Part I — ESTEL general rules

[Back to the overview](overview.md)

These rules hold for every version of ESTEL, from the first commit of step 1 onwards. They will move, almost unchanged, into the ESTEL reference document.

## Contents of this part

- [1. Principles](#1-principles)
  - [1.1 Purpose and scope](#11-purpose-and-scope) · [1.2 A library maintained by agents](#12-a-library-maintained-by-agents) · [1.3 Two-step development](#13-two-step-development) · [1.4 Design for what comes next](#14-design-for-what-comes-next) · [1.5 Licensing](#15-licensing) · [1.6 Supported versions and dependencies](#16-supported-versions-and-dependencies)
- [2. Architecture](#2-architecture)
  - [2.1 Layers](#21-layers)
  - [2.2 Multi-backend strategy](#22-multi-backend-strategy): [2.2.1 The primitive layer (L0)](#221-the-primitive-layer-l0) · [2.2.2 Static plans](#222-static-plans) · [2.2.3 Backend selection](#223-backend-selection) · [2.2.4 Engines](#224-engines) · [2.2.5 Roll-out](#225-roll-out)
  - [2.3 Object model](#23-object-model): [2.3.1 Geometries and the hierarchy of classes](#231-geometries-and-the-hierarchy-of-classes)
  - [2.4 Code rules enforced in CI](#24-code-rules-enforced-in-ci): [2.4.1 Names](#241-names)
- [3. Numerical coding rules](#3-numerical-coding-rules)
  - [3.1 Functional, JAX-ready style](#31-functional-jax-ready-style): [3.1.1 The scattering loop without in-place updates](#311-the-scattering-loop-without-in-place-updates)
  - [3.2 Differentiation contract](#32-differentiation-contract): [3.2.1 Why second order](#321-why-second-order) · [3.2.2 Guarantees](#322-guarantees) · [3.2.3 Implementation rules](#323-implementation-rules) · [3.2.4 Engines and differentiation order](#324-engines-and-differentiation-order)
- [4. Mathematics](#4-mathematics)
  - [4.1 `maths.md` as an executable specification](#41-mathsmd-as-an-executable-specification) · [4.2 The oracle `_spec`](#42-the-oracle-_spec) · [4.3 Outline of `maths.md`](#43-outline-of-mathsmd)
- [5. Tests, reference values, tolerances](#5-tests-reference-values-tolerances)
  - [5.1 Test families](#51-test-families) · [5.2 Tolerances](#52-tolerances) · [5.3 Protection of reference values](#53-protection-of-reference-values)
- [6. CI, GPU access, benchmarks](#6-ci-gpu-access-benchmarks)
  - [6.1 Tiers](#61-tiers) · [6.2 Constrained GPU access from CI](#62-constrained-gpu-access-from-ci) · [6.3 Benchmarks](#63-benchmarks)
- [7. Versioning and merge policy](#7-versioning-and-merge-policy)
  - [7.1 Version numbers](#71-version-numbers) · [7.2 Computing the required bump](#72-computing-the-required-bump) · [7.3 Merging](#73-merging)
- [8. Context files for agents](#8-context-files-for-agents)
  - [8.1 Mirrored files](#81-mirrored-files) · [8.2 Golden rules](#82-golden-rules) · [8.3 Review rules](#83-review-rules) · [8.4 Tuning prompts](#84-tuning-prompts)

## 1. Principles

### 1.1 Purpose and scope

- **What ESTEL does.** It computes scattering-transform statistics and provides simple built-in syntheses. Each synthesis function links to a tuning guide (§8.4). Deeper customisation happens in the user's own code, through the public API, never by changing library defaults.
- **What ESTEL does not do.** Component separation, inverse problems and other applications live in separate repositories that depend on ESTEL (review rule A8: respect the library boundary). ESTEL gives them what they need — in particular statistics differentiable to second order (§3.2) — as a stable public API.
- **Real-space wavelets only.** ESTEL uses wavelet kernels in pixel space. The Fourier-space implementation of STL is not carried over; Fourier transforms remain only where a spectrum estimator needs them.

### 1.2 A library maintained by agents

- Changes are written mostly by AI coding agents, and merged without human review when every check passes (§7.3).
- The skeleton — public interface, class structure, scattering loop — is fixed in the repository. It changes only through a human-approved major version.
- Humans step in where judgement is needed: major versions, reference values, mathematics, policy files, benchmark thresholds.

### 1.3 Two-step development

- **The rule that separates the steps:** whatever STL does not compute today belongs to step 2, however small it is.
- **Step 1 ends with the 1.0.0 freeze:** parity with STL proven (Part II, §14), reference values generated and signed off by humans.
- **Step 2 is documented before it starts** (Part III), and the interfaces it needs are reserved in 1.0.0 (Part II, §9.4).
- **Step 2 has two parts, then step 3.** Step 2a corrects and stabilises the mathematics, with a benchmark properly developed; step 2b brings the advanced optimisations and the Rust engine, and ends with ESTEL 2.0.0. Each entry ships, as the new behaviour, in the next 1.x release once it is done, and each release drops the previous defaults at once (§7.1). Step 3 holds the long-term developments (Part III, §19.13).

### 1.4 Design for what comes next

- Every interface a planned feature will need is created in 1.0.0, minimal and tested, even when its only implementation is trivial — for example the `_global_sum` hook, which does nothing without MPI.
- Storage, backend and engine choices stay invisible to users: changing them never changes a public signature.
- A reserved interface that turns out unnecessary is removed in a major version, not left to decay.

### 1.5 Licensing

- **ESTEL is released under Apache-2.0.** It is permissive, like JAX, optax and healpix-analyse, and adds an explicit patent grant: each contributor licenses the patents their contribution needs, and anyone who sues claiming ESTEL infringes a patent loses that licence. The repository holds `LICENSE` and `NOTICE`; `pyproject.toml` declares `license = "Apache-2.0"` and `license-files = ["LICENSE", "NOTICE"]`.
- **Precondition: STL's licence.** ESTEL ports STL code (Part II, §12), so it is a derivative of STL and inherits its terms. Before the first public release, STL's licence is confirmed in writing; if the STL-Dev repositories carry none, every STL author, and their employer where it owns the code, agrees to Apache-2.0. The same holds for the STL-Dev-MPI Rust code reused in step 2. STL's copyright line goes into `NOTICE`. Funding contracts are checked for any required licence.
- **Dependencies stay permissive.** Runtime dependencies are MIT, BSD or Apache-2.0, as all of them are today. GPL code is never imported by the package nor ported into it: healpy (GPL-2) remains a test-only cross-check of the oracle, and the JAX backend does not use jax-healpy (GPL-3).
- **Rust wheels (step 2).** `estel-rust` wheels statically link their Rust crates, and so redistribute them: `cargo-deny` in CI rejects copyleft crates, `cargo-about` generates the third-party notices shipped in each wheel, and no GPL compiled library (FFTW, ducc0) is linked. Dual MIT OR Apache-2.0, the Rust convention, stays an option if the crates are published on crates.io.
- **Contributions** are accepted under Apache-2.0 with a DCO sign-off checked in CI, so that the origin of every change stays traceable.

### 1.6 Supported versions and dependencies

- **Minimum versions**, confirmed on 12 November 2026: Python 3.11, NumPy 2.0, PyTorch 2.6 and array-api-compat 1.11; JAX 0.5 for the experimental backend, as an optional extra (`estel[jax]`), and optax 0.2.4 in step 2. All were released in 2025 or earlier, so the 2025 versions of the scientific Python stack are supported.
- **Ranges and a lock file.** `pyproject.toml` declares minimum versions, with an upper bound only for dependencies that often break their API (for example `torch>=2.6,<3`). A lock file pins exact versions for CI and for generating reference values. Updating it is a PR like any other, and must leave reference values unchanged.
- **As few dependencies as possible, and none on young research code.** What ESTEL needs from small research packages is ported into ESTEL, with attribution, rather than imported. Such packages may only appear as pinned, test-only dependencies, used to prove parity. healpix-analyse is the first case (Part II, §9.3).

## 2. Architecture

### 2.1 Layers

&#91;embedded content: ESTEL layers · frontend to L0, with the static plan, the oracle and the engines\]

The engine layer hands each computation to one engine. Inside the Python engine, each layer calls only the one below it, and the static plan feeds L1 and L2. The oracle produces the reference values that the outputs of every engine are tested against.

- **Frontend.** The public API, frozen within a major version by `api/manifest.json`: data classes, operators, statistics, synthesis, utilities.
- **Engine layer.** Chooses which engine runs the whole computation, from `engine=` and the capabilities each engine declares (§2.2.4). Every engine returns the same `STStatistics` and must reproduce the same reference values.
- **Python engine** (`engine="python"`, the only one in step 1), made of three layers:
  - **L2, skeleton.** The scattering loop, statistics, reductions and synthesis driver, in plain Python, backend-neutral.
  - **L1, geometry.** Written once on top of L0. Each `Data` subclass provides a small set of geometry hooks; the mother class holds everything else. The generic `WaveletOperator` is the only caller of the hooks of filtering and downsampling (`_conv`, `_downsample`, `_crop_weights`, `_lowpass`): everything that depends on the wavelets and on the geometry of the pyramid reaches L2 through it (§2.3).
  - **L0, primitives.** About twenty functions in fourteen groups (Part II, §12.3), the only code that imports a backend: PyTorch, and JAX as an experimental backend from P1, chosen with `set_backend()`.
- **Rust engine** (`engine="rust"`, step 2). The Rust code of STL-Dev-MPI, moved into the ESTEL repository, runs the whole computation by itself, for the configurations it declares (Part III, §18). It has no L2, L1 or L0, and uses neither PyTorch nor JAX operations. It is an engine rather than a backend on purpose (§2.2.4).
- **Static plan.** Everything that depends only on shapes and parameters, built once in NumPy when an operator is constructed, and used by both engines. Masks of missing data and their reweighting maps are part of it (§2.3).
- **The oracle `_spec`.** A naive, slow, float64 NumPy transcription of `maths.md`, independent of the library. Reference values come from it (§4.2).

There are therefore two independent choices, at two levels:

| Choice | Set with | Options | Decides |
| --- | --- | --- | --- |
| Engine | `engine=` | `"python"`, `"rust"` | who runs the whole computation |
| Backend | `set_backend()` | `"torch"`, `"jax"` | who runs the elementary operations of the Python engine |

### 2.2 Multi-backend strategy

Running one library on several backends is the most critical design point of ESTEL. Three mechanisms make it possible, and the rules of §2.4 and §3.1 keep them intact.

#### 2.2.1 The primitive layer (L0)

The primitives are written against the Python Array API standard, through `array-api-compat`, plus the extensions the standard lacks: convolution along chosen axes, padding modes, weighted gathers, sums by segment, random keys, differentiation helpers, built-in optimizers. Their contracts are listed in Part II, §12.3. They live in `src/estel/backend/`:

| File | Content | Step |
| --- | --- | --- |
| `_protocol.py` | A `typing.Protocol` listing every primitive, with its shapes, dtypes and edge cases in the docstring. This file is the contract | 1 |
| `_registry.py` | `get_backend()`, `set_backend()`, and the lookup from an array to its backend, to detect mixing | 1 |
| `torch/` | PyTorch implementation, including the HEALPix operations of step 1 (Part II, §9.3) and the `torch.optim.LBFGS` wrapper | 1 |
| `jax/` | JAX implementation (experimental from P1); `optax.lbfgs` in step 2 | 1 |

`tests/backend/` holds a **conformance suite**. Each primitive is compared with a NumPy float64 implementation on random inputs, every padding mode, real and complex values, odd sizes, and gradients. Adding a backend means passing this suite; the rest of the library needs no change.

#### 2.2.2 Static plans

At construction, an operator builds an `STPlan` in NumPy: scale and orientation index sets, wavelet and low-pass kernels, crop widths per level, mask footprints, spectrum bins, HEALPix stencils. The plan is immutable and hashable. In PyTorch it simply avoids recomputation; in JAX it is a compile-time constant, which is what allows `jax.jit` to compile the whole computation. Large arrays of the plan are passed to the computation as arguments, never baked in as constants.

#### 2.2.3 Backend selection

- **A global default, read when an operator is built.** `estel.set_backend("jax")` sets the default; every operator records the backend it was built with.
- **Mixing raises an error.** A torch tensor passed to a JAX-built operator raises `BackendMismatchError`, whose message says how to fix it.
- **Inputs are converted at the boundary.** `Data` constructors accept NumPy and backend arrays and convert them to the current backend. They never convert silently from one backend to another.

#### 2.2.4 Engines

- **Two engines, one protocol.** Every engine implements the same protocol: a forward pass that returns the statistics, and a backward pass (VJP) for gradients. The Python engine (L2 + L1 + L0) is always available. The Rust engine (Part III, §18) arrives in step 2. In 1.0.0, `engine=` accepts only `"python"`; the argument, the registry and the protocol are reserved and tested.
- **Why the Rust code is an engine, not a backend.** A backend only provides elementary operations, whose compositions PyTorch or JAX differentiate automatically. Rewriting those operations in Rust would require a Rust automatic-differentiation system, and would lose what makes STL-Dev-MPI fast: running the whole algorithm by blocks, with fused operations, a hand-written gradient of the whole chain, and MPI. The Rust code therefore replaces the whole computation, behind the same Python frontend (Part IV, §22, review of 12 November 2026).
- **Selection.** `STOperator(..., engine="auto" | "python" | "rust")`. With `"auto"`, ESTEL uses, for each configuration, the engine designated as its default: the Python engine, unless a benchmark-gated minor version has promoted the Rust one (§7.1). With `"rust"`, an unsupported configuration raises `EngineUnsupportedError`, naming the unsupported option.
- **Capability declaration.** At construction, an engine reports what it computes: geometries, options, dtypes, devices, differentiation order. The Python engine covers everything ESTEL defines.
- **Purity at the boundary.** The forward pass returns the values and an opaque residual; the VJP takes the residual back. Nothing relies on state hidden inside an engine.
- **Same numbers.** Every engine passes the reference tests for every configuration it claims; the Rust engine also passes a parity test of its gradients against the Python engine. No engine produces reference values: they come from the oracle.
- **Differentiation order.** Declared by each engine (§3.2.4).

#### 2.2.5 Roll-out

| Stage | Engines (`engine=`) | Backends of the Python engine (`set_backend()`) | Precision |
| --- | --- | --- | --- |
| Step 1 (1.0.0) | Python engine only; engine layer and protocol reserved | PyTorch; JAX experimental: every L0 primitive, tested on every PR, selectable with a warning, not covered by the 1.0.0 guarantees; syntheses on JAX later | float32 default, float64 path |
| Step 2 | + Rust engine: the `estel-rust` wheel, built from `rust/` in the same repository; CPU and GPU, with MPI (Part III, §18) | JAX stable, with its synthesis optimiser (Part III, §17) | Rust engine: float64 default, float32 opt-in |
| Later | any engine that passes the parity tests | any backend that passes the conformance suite | — |

### 2.3 Object model

- **A `Data` object is data plus static metadata.** It holds the array and its geometry metadata (pbc, cell ids, block structure). It is registered as a pytree: the array is its only leaf, everything else is static. The same object therefore works in PyTorch and inside `jax.jit`.
- **Masks are static.** As in STL, a mask of missing data and its reweighting maps are computed once, when the operator is built, and stay fixed for every forward and backward pass. They belong to the static plan and reach the computation as constant arrays. What is forbidden is only to detect missing values, or to choose an estimator, during the computation (§3.1, rule 3).
- **The convolution history is plan data, not data state.** STL keeps a mutable `conv_history` list on each data object, to know which convolutions were applied and derive the border crop from it. In ESTEL the sequence of convolutions is fixed by the operator, so the `STPlan` computes the crop width of every level once. A data object carries only immutable metadata, which JAX treats as compile-time constants; the levels are tracked by the computation, not by the data.
- **An operator is a frozen plan plus a pure function.** `STOperator(...)` builds the plan and the backend arrays (kernels, masks). `op.apply(data, ref=None, *, norm="self", ref_batch_mean=False)` calls `estel.functional.scattering(data, plan, filters, ref, norm=..., ref_batch_mean=...)`, whose `norm` and `ref_batch_mean` are static, as the plan is. That pure function is what `jax.jit`, `jax.grad` and `torch.func` see, and, with L1 and L0, it makes up the Python engine. The method keeps STL's name, `apply`; a direct call `op(data)` may be added later as a minor change (Part IV, §22, R-A09).
- **One way to say it (review rule A6).** In STL, `ST_Operator.__init__` and `apply` accept the same options (`norm`, `iso`, `flatten`, `compute_PS`, …), with `None` meaning "inherit from the constructor": two ways to set each option. In ESTEL every option is set in one place only: the normalisation (`norm`, `ref_batch_mean`) at each call of `apply`, so that one operator serves every normalisation, as in STL (Part IV, §22, "One operator for every normalisation"); everything else at construction. STL's two ways to set an option, the constructor and `apply`, stay excluded. Setting the rest at construction is also what makes the plan static: a different configuration of the transform is a different operator.
- **Everything that depends on the geometry goes through the wavelet operator.** One generic `WaveletOperator` provides, level by level, the convolution by the wavelets, the downsampling, the low-pass filter, the shape of each level, and the rules of borders (M-BORDER) and of mask propagation (M-MASK). `STOperator`, its plan and its loop go through it for all of these: they see levels, maps and weights, and do not depend on the geometry. The code proper to each geometry stays in one place, the geometry hooks of its `Data` subclass and the NumPy functions that build its tables; only the wavelet operator calls its hooks of filtering and downsampling (`_conv`, `_downsample`, `_crop_weights`, `_lowpass`), which receive the static tables of the level. Like `STOperator`, the wavelet operator is a static part, held by the plan, plus arrays, held by the filters, applied by pure functions. It is internal: ESTEL computes scattering statistics, and has no public wavelet operator (Part IV, §22, "Wavelet operator, internal"). A tensor product composes the wavelet operators of its blocks, with multi-index levels (Part IV, §22, "Wavelet operator").
- **One entry point per object.** `estel.STOperator(data, ...)` replaces `data.get_ST_op()`; STL's `data.get_wavelet_op()` has no public counterpart (Part II, §10).
- **Class names follow PEP 8:** CapWords, no underscore. STL's `ST_Operator`, `ST_Operator_TensorProd` and `ST_Statistics` become `STOperator`, `STOperatorTensorProd` and `STStatistics`.
- **The mother class is the simple case.** `STOperator` handles a single domain (1D, 2D, HEALPix). `STOperatorTensorProd` overrides a small, fixed set of hooks for products of domains (Part II, §12.6), and nothing else.

#### 2.3.1 Geometries and the hierarchy of classes

ESTEL has one tree of geometries. Every family of classes follows it, and every name says which node of the tree it covers (§2.4.1). Phase P7 (Part II, §15.2, T7.2 to T7.12) brings the code to the hierarchy below; until a task of P7 is merged, the code it concerns keeps its former names.

```
(every geometry)                      bare
├── SingleDomain                      one field on one domain: not a tensor product
│   ├── Cartesian                     regular grids
│   │   ├── 1D
│   │   └── 2D
│   └── Healpix
└── TensorProd                        a product of SingleDomain blocks (Cartesian; Healpix with LF9)
```

- A geometry is a node of this tree; the leaves have the tags of `engines/_protocol.py` (`"1d"`, `"2d"`, `"healpix"`, `"tensor_prod"`), frozen in saved files.
- A geometry word names the geometry a computation acts on, a whole field or one block: `Cartesian`, `1D` and `2D` also cover the Cartesian blocks of a tensor product, and `TensorProd` names only what belongs to the product (multi-indices, joint scales, the loop over blocks, the composition).
- "1D" and "2D" always mean Cartesian: HEALPix data, though they lie on a surface, are never called 2D.
- An intermediate class of a family is created when it holds code shared by two leaves; inserting one renames no leaf.

**The hierarchy of classes after P7.** [P] public, (i) internal; a class marked abstract cannot be built. A *brick* is a class held by a plan or by filters: the wavelet plans and filters, the bricks of the spectrum.

```
Data [P]                               root; holds the single-domain behaviour
├── DataCartesian (i)                  the hooks shared by 1D and 2D data
│   ├── Data1D [P]
│   └── Data2D [P]
├── DataHealpix [P]
└── DataTensorProd [P]                 its blocks: BlockSpecTensorProd (i)

WaveletOperator (i, abstract)          the only way to the geometry (§2.3)
├── WaveletOperatorSingleDomain (i)
└── WaveletOperatorTensorProd (i)      composes the operators of its blocks

WaveletPlan (i, abstract)
├── WaveletPlanSingleDomain (i, abstract) the fields shared by every single-domain plan, the low-pass filter included
│   ├── WaveletPlanCartesian (i)
│   └── WaveletPlanHealpix (i)
└── WaveletPlanTensorProd (i)          holds the plans of its blocks

WaveletFilters (i, abstract)           a tensor product holds a tuple of them, one per block
├── WaveletFiltersCartesian (i)
└── WaveletFiltersHealpix (i)

STPlan [P]                             the plans of STOperator and STOperatorTensorProd
├── STPlanSingleDomain (i)             holds a WaveletPlanSingleDomain and the spectrum brick
└── STPlanTensorProd (i)
STFilters [P]
├── STFiltersSingleDomain (i)
└── STFiltersTensorProd (i)

Bricks of the spectrum (i): SpectrumEstimatorPlan2D, SpectrumEstimatorFilters2D, SpectrumBins2D
```

The operators are `STOperator` [P], single-domain, and `STOperatorTensorProd` [P], its subclass; `Data` and `STOperator` are bare and single-domain, the documented exceptions of the public names to §2.4.1. The power spectrum is computed by `STOperator` (`compute_PS`), with the bricks of the spectrum, from T4.6 (Part IV, §22, row of 26 November 2026, "The spectrum within `STOperator`").

**Planned classes.** Proposals of the register: the card may revise them, and its pull request turns them from planned to existing here.

| Card | Planned |
| --- | --- |
| LF10 | the v1.6.0 estimators in every geometry, generalised directly (Part III, §20.5): bricks of the spectrum written once for every Cartesian grid and every Cartesian block (`SpectrumEstimatorPlanCartesian`, replacing the `2D` bricks), reached through the hook `_spectrum_grids` of `Data`; `STPlanTensorProd` gains its spectrum brick; the spectral coverage (range and bins) rethought |
| LF24 | the HEALPix spectra: `SpectrumEstimatorPlanHealpix` |
| LF6 | the low-pass filter in every geometry: the low-pass fields of `WaveletPlanSingleDomain` read by every leaf; `DataHealpix._lowpass` implemented |
| LF9 | HEALPix blocks in tensor products: `DataHealpix` gains the block capability and the hooks along a block's axes; the blocks of `WaveletPlanTensorProd` become `WaveletPlanSingleDomain`; names ending with a run (`…_healpix_tensor_prod`, §2.4.1) |
| LF7, LF61 | an estimator for non-periodic data, spectra on odd sizes: values and options of the Cartesian estimators; no class |
| LF8 | masks in tensor products: mask builders ending with `_tensor_prod`; no class |
| LF42 | masks per channel: a channel axis in the masks of every geometry; no class |
| LF34 | partitioned execution, every geometry (`DataHealpix(partition=…)` reserved today); no class |
| LF22 | optimised wavelets: new banks, in the module of banks of their geometry |
| LF51 | a study of kernels defined mathematically; no class |
| LF43, LF67, LF72, LF73 | on HEALPix data: an ellipsoid for the pixel centres, the edge of a partial sky, the tables cached on disk, a cheaper gradient; functions ending with `_healpix`, no class |
| LF44 | named blocks of tensor products: fields of `BlockSpecTensorProd` |
| LF75 | a real name for the HEALPix wavelet |
| W5, W6 | the notation of `docs/maths.md` and the vocabulary of the code: stems of names renamed through the table of §2.4.1 |
| W7 | the structure of `docs/maths.md`, and with it which bare maths IDs are general |
| a 3D geometry (none planned) | `Data3D` under `DataCartesian`; its bank in the Cartesian module of banks |

### 2.4 Code rules enforced in CI

Each rule is enforced by a mechanical check — `import-linter` contracts and a small AST checker in `tools/` — not by review:

| Rule | Check |
| --- | --- |
| `torch`, `jax`, `optax` imported only under `estel/backend/` | import-linter "forbidden" contract |
| `estel_rust` imported only under `estel/engines/` (step 2) | import-linter "forbidden" contract |
| L2 never imports a `Data` subclass; it talks to the mother-class interface | import-linter "layers" contract |
| Outside `estel.wavelets` and `estel.data`, nothing reaches the wavelets or the geometry of the pyramid but through `WaveletOperator` (§2.3): no import of the banks of `estel.wavelets` or of the mask builders of `estel.data`, and no call to a hook of filtering or downsampling (`_conv`, `_downsample`, `_crop_weights`, `_lowpass`) | import-linter contract; AST check |
| `estel._spec` imports NumPy only, and nothing from `estel` | import-linter "independence" contract |
| No `type(x).__name__ ==` comparisons, no assignment to `__class__` | AST check |
| No in-place array mutation in L1/L2 (`x[...] =`, `out=`, `.add_`, `inplace=True`, `x += y`, `del x[...]`, `__setitem__`). The NumPy modules that build the static plan (tag N in Part II, §12) are exempt from this rule only, through an explicit list in the checker: they run once, at construction, and are never traced or differentiated | AST check |
| CI workflows: every action pinned to a full commit SHA with its version as a comment; every checkout with `persist-credentials: false`; an explicit top-level `permissions:` block (review rule E5). Dependabot proposes the updates of pinned actions | `tools/check_workflows.py` |
| **Public vs internal.** Public names are exactly those listed in `api/manifest.json` and exported from `estel/__init__.py`. Internal functions live in private modules, named with a leading underscore (`spectra/_estimators.py`), with plain names inside; internal methods of public classes start with an underscore | `tools/api_manifest.py --check` fails if a name is reachable without an underscore, outside a private module, and not in the manifest |
| **Names.** Every internal name follows the tree of geometries (§2.4.1) | the rules a tool can decide: `tools/check_retired_names.py`, with the table `tools/renamed_names.toml` (T7.2); the others in review |

**Why these rules, in plain terms:**

- **No class-name comparisons.** `type(x).__name__ == "STL_2D_Kernel_Torch"` compares the name of a class as a string. It breaks as soon as someone writes a subclass — a 1D class built on the 2D one, a tensor-product block — and the check silently rejects valid data or takes the wrong branch. The correct test is `isinstance`, or better, a capability the object declares. STL has three such checks; the one in the v1.6.0 cross-spectrum operator rejects 1D and tensor-product data.
- **No reassignment of `__class__`.** Changing an object's class after it is created means neither readers nor tools (type checkers, serialisation, JAX, `torch.compile`) can know what the object is. STL's tensor-product class does this at every instantiation. ESTEL composes instead: a tensor-product object holds one array and the geometry of each block, all on the same backend.
- **No in-place updates.** Writing into an existing array breaks JAX, which forbids it, and second-order differentiation in PyTorch. It also makes results depend on the order of calls.
- **No backend import outside L0.** This single rule is what keeps JAX and Rust engines possible. One stray `import torch` in the geometry code, and a JAX port becomes a rewrite.
- **Public or internal, visible in the code.** The frozen API is ESTEL's central promise, so users and agents must see from the code itself what they may rely on. A name that looks public will be used, and can then no longer change without a major version.

#### 2.4.1 Names

Every internal name follows the tree of geometries (§2.3.1). `tools/check_retired_names.py`, with the table of renamed names `tools/renamed_names.toml` (both from T7.2), checks rules 1 and 2 (the order and the forms of geometry words), rule 3 for the names imported outside the modules of their geometry, rule 6, the lists of rule 7, the form of rule 9 and the word of rule 10; the meaning of a suffix, and rules 4, 5, 8 and 11, are checked in review (`docs/agents/review_rules.md`).

1. **Order.** `<Kind><Role><Geometry>`, the kind being what the object is about (`ST`, `Wavelet`, `Spectrum`, `Data`) and the role what it is (`Plan`, `Filters`, `Operator`, `Bins`): a class `WaveletPlanHealpix`, a function `stencils_healpix`, a constant `KERNEL_SIZE_HEALPIX`, a module `_tables_healpix.py`. The geometry is the last word, or the last run (rule 4); the other words keep their order. This holds for the definitions of `src/estel/` (the oracle included), `tests/`, `tools/` and `bench/`, methods and fields included. Two exceptions with their own form: test function names are sentences, and benchmark labels are `<geometry>-<variant>-<size>`; the four labels `planar-*` of step 1 stay (rule 7).
2. **The suffix is the smallest node of the tree that covers everything the name does**; a bare name covers every geometry. Forms: `SingleDomain`, `single_domain`, `SINGLE_DOMAIN`; `Cartesian`, `cartesian`, `CARTESIAN`; `1D`, `1d`, `2D`, `2d`; `Healpix`, `healpix`, `HEALPIX`; `TensorProd`, `tensor_prod`, `TENSOR_PROD`; in maths IDs `HEALPIX`, `TENSORPROD`; in labels and anchors `tensor-prod`. Never abbreviated: `hpx`, `tp` and `tensor_product` are retired in every identifier, test names, file names and labels included.
3. **Modules.** A module is either of one geometry (its path names it) or bare. A name defined in a module of one geometry carries that geometry if and only if it is used outside the modules of that geometry; inside them the module says it. A name of a bare module that covers only part of the tree says which part. A method that overrides another keeps its name. In `tests/`, `tools/` and `bench/`, rules 1, 2 and 10 apply, and names fixed by another program (`pytestmark`, the fixtures of `conftest.py`, the interface of the parity cases) never change.
4. **Two geometries.** A final run, from the block to the product (`orientations_healpix_tensor_prod`); blocks of several geometries in the order of the data, joined by `_`.
5. **Roles are bases.** A type that generic code names in its annotations is a base of the hierarchy; a concrete class of one geometry is never the type of another.
6. **An old name never has two meanings at once.** It comes back with a new meaning only after it has disappeared entirely: renamed by script in one commit, its absence checked by the tool on that commit, then reintroduced by hand in a later commit; the table records both steps.
7. **Closed lists only.** No list of exempted project names. The tool holds commented lists: the names of the published HEALPix convention (Górski et al. 2005: `nest2xyf`, `xyf2nest`, `xyf2ang`, `ring2nest`), never a code source (§1.5, F23); package names (`healpix_analyse`, healpy, healpix-geo, `optax`); names fixed by another program; history (Part IV, §22, `docs/register/decisions.md`, `changes/`, `bench/results/`); STL's own names, and the reserved public `plot.nyquist_filter` (Part II, §12.10); the four benchmark labels `planar-*` of step 1.
8. **"geometry"** names a node of the tree, and, in the fixed phrases "L1 geometry" and "geometry hooks", the layer and the hooks a geometry provides; the tables of pixels are "tables" or "stencils", and the shapes of the levels of the pyramid are "levels".
9. **Maths IDs** read `M-<TOPIC>[-<VARIANT>]*[-<GEOMETRY>]*`, the geometry in the forms of rule 2 (`1D`, `2D`, `HEALPIX`, `TENSORPROD`), digits in a topic only for named quantities (S1 to S4, W1). Phase P7 changes `-HPX` into `-HEALPIX`, `-TP` into `-TENSORPROD` and `M-CONV-2D` into `M-CONV`; which bare IDs are general is decided with the structure of `docs/maths.md` (W7).
10. **"planar"** is retired: "Cartesian" for 1D and 2D grids, "single-domain" for what is not a tensor product; so are its synonyms "single geometry" and "plain geometry". In prose it survives only in a quote of STL or STL-Dev-MPI, followed by "(STL's word)".
11. **Messages** name public names only.

## 3. Numerical coding rules

### 3.1 Functional, JAX-ready style

These rules apply to L1 and L2 from the first commit of step 1. From phase P1, ESTEL also has an experimental JAX backend for L0, in `src/estel/backend/jax/`, which passes the same conformance suite as PyTorch, and CI runs the whole test suite on both backends, in parallel jobs: on every merge to `main`, nightly, and on PRs labelled `full-ci`; other PRs run the JAX smoke subset of §5.4. Any PyTorch idiom that settles into L1 or L2 therefore fails at once. JAX is opt-in and experimental in 1.0.0 (§2.2.5). The rules are copied into `docs/agents/jax_rules.md`.

1. **Functional code only.** Build per-level results in Python lists (§3.1.1), then `concat` or `stack` them. No preallocation filled by index, no `out=`, no `inplace=`. This applies to the code that computes the statistics; the NumPy code that builds the static plan, run once at construction, may write in place (§2.4).
2. **Static shapes.** Everything shape-related — index sets, crop widths, stencils, bins — lives in the `STPlan`, built in NumPy and passed as a static argument.
3. **No decisions on array values inside traced code.** No `if x.isnan().any()`, no `.item()`, no `float(x)`, no `print(x)`. NaN detection and estimator selection happen when the `Data` object or the operator is constructed; the masks and reweighting maps they produce are then static data of the plan (§2.3).
4. **Large constants are arguments, not closures.** Kernels, masks and stencils are passed as arrays to the computation (`filters` in `scattering(data, plan, filters)`), so they are not baked into the compiled program. Baking small kernels in as compiled constants was tried in step 1 and brought no gain (Part III, §19.14).
5. **Explicit randomness.** Every random function takes a key or a seed (`make_key(seed)`).
6. **Complex convolutions are real convolutions**, as STL's `_semicomplex_conv` already does: the arithmetic stays real and kernels are stored as real pairs; the rule does not fix the number of calls. A complex kernel on a real map is one real call, the real and imaginary parts of the kernel stacked; on the complex intermediate map of a tensor product (maths M-W1-TP), four real calls. Complex-to-complex convolutions were tried in step 1 and gained less (Part II, §15.2, T3.5; Part III, §19.14).
7. **float32 by default, float64 on request.** Every computation must run in float32, the default, as well as in float64: no hard-coded float64 constant, no silent change of dtype; kernels, masks and plan arrays are cast to the operator's dtype when it is built, and the tests run in both. `set_default_dtype("float64")` enables float64 on every backend; on JAX it switches on `jax_enable_x64`, and raises an error if JAX already ran in float32 in that process.
8. **Short Python loops, for JAX.** JAX unrolls Python loops when it traces the code, so their length drives compile time, which is part of the benchmark with a budget per case. Step 1 starts from the structure of STL's scattering loop, which limits memory and avoids recomputation, especially for tensor products, written in functional style. Before 1.0.0, a reorganisation that makes the computation faster or lighter belongs to step 1 if every value stays within the parity tolerance with STL and the benchmark shows its gain: done early, it guides the developments that follow, and parity with STL checks that it changes nothing (Part II, §15.2, T2.11 to T2.15; Part IV, §22, "Efficiency in step 1"). A reorganisation that changes values remains step 2 work (Part III, §19.13).
9. **Data and statistics are pytrees.** `Data` and `STStatistics` are registered as pytrees (arrays as leaves, metadata static), so that `jax.vmap` and `jax.jit` accept them.

#### 3.1.1 The scattering loop without in-place updates

STL fills its statistics by writing into tensors allocated in advance. JAX forbids this, because its arrays are immutable, and in PyTorch it breaks second-order differentiation. ESTEL writes the loop so that every step *returns* a value instead of modifying one: compute, collect, assemble.

```python
# STL: preallocate, then write in place
S4 = torch.full((Nb, Nc, Nc, J, J, J, L, L, L), nan)
for j3 in range(J):
    for j1 in range(j3 + 1):
        for j2 in range(j1, j3 + 1):
            S4[:, :, :, j1, j2, j3] = cov(U[j1, j3], U[j2, j3])
data.downsample(inplace=True)
```

```python
# ESTEL: compute, collect, assemble
values = [cov_valid(u[j1, j3], u[j2, j3], plan.weights[j3])
          for (j1, j2, j3) in plan.s4_triples]     # static list, built in NumPy
s4 = xp.stack(values, axis=-1)                     # one array, no write
x_coarse = geometry.downsample(x, level)           # a new object; x is untouched
```

- **Every step returns a value.** Results are collected in Python lists and stacked once. The loops run over the static index sets of the plan — scales and levels, a few dozen iterations — so JAX unrolls them when it traces the code.
- **Data objects are never modified.** Downsampling and filtering return new objects (`with_array`). This does not keep old arrays alive: once nothing refers to them, their memory is freed; under `jax.jit` the compiler reuses buffers itself; and when gradients are needed, autograd keeps the intermediate arrays in either style. Peak memory is measured against STL's on the benchmark cases in phase P2.
- **The statistics are gathered once, at the end, into the compact buffer.** One static gather, whose indices are built in NumPy with the plan, places the stacked values of the loop in the compact layout: every coefficient STL computes and only those, with no NaN (Part III, §19). No full-grid tensor is built in the computation. The full-grid view, with NaN where no coefficient exists, is built on demand for `full_grid` by one out-of-place primitive, `scatter_to_grid(values, index, shape, fill)` (Part II, §12.3; Part III, §19.5). Its inverse, `gather_from_grid(grid, index)`, reads the existing coefficients back.
- **Missing data are weights.** Means are computed as `sum(w * x) / sum(w)` with w equal to 0 or 1, never as `x[mask]` or `nanmean`, so that every shape stays fixed. The denominators `sum(w)` depend only on the static masks, so they are computed once, in the plan.

The cost is negligible. Under `jax.jit`, the compiler reuses memory itself; in PyTorch, stacking costs one copy at the end, small beside the convolutions. Since T2.12, the second-layer means are computed by contraction, in the Gram form, instead of building the full product first, which lowers the memory peak further (Part III, §19.3).

This style is required from step 1, and JAX runs it from P1: it is also what keeps second-order derivatives correct in PyTorch, and what makes results independent of the order of calls.

### 3.2 Differentiation contract

#### 3.2.1 Why second order

Most uses of ESTEL need only gradients. A synthesis adjusts a field until its statistics match a target, and needs the derivative of a loss with respect to the field: first order.

Downstream repositories go further. In component separation, for example, one solves an optimisation problem — find the maps that best explain the observations — and then wants to know how that solution moves when a parameter changes: a noise level, a foreground model, a prior. Solving again for every parameter value is too costly. The implicit function theorem gives the answer from the solution itself. For θ\*(φ) = argmin over θ of L(θ, φ):

```latex
\frac{d\theta^*}{d\phi} = -\left[\frac{\partial^2 L}{\partial\theta^2}\right]^{-1} \frac{\partial^2 L}{\partial\theta\,\partial\phi}
```

Computing this requires products of the Hessian — the matrix of second derivatives — with vectors, and mixed second derivatives. ESTEL therefore guarantees derivatives of order two of its statistics. This changes nothing in the formulas; it only forbids a few implementation shortcuts (§3.2.3).

#### 3.2.2 Guarantees

| Guarantee | PyTorch (step 1) | JAX (experimental from P1) | Test |
| --- | --- | --- | --- |
| VJP (reverse mode) of `scattering`, `loss` | autograd | `jax.vjp` | `gradcheck`, `check_grads(order=1)` |
| JVP (forward mode) | `torch.func.jvp` | `jax.jvp` | `check_grads(modes=("fwd",))` |
| Second order (Hessian-vector products, mixed derivatives) | double backward (`create_graph=True`), `torch.func` | `jax.jvp(jax.grad(...))` | `gradgradcheck`, `check_grads(order=2, modes=("fwd", "rev"))` |
| Implicit differentiation, end to end |  |  | a small argmin problem; implicit gradient vs finite differences |

All differentiation tests run in **float64** on small inputs, one reason the float64 path stays.

#### 3.2.3 Implementation rules

1. **No `custom_vjp` in the Python engine.** It removes forward mode in JAX. If a custom rule is ever needed there, use `custom_jvp`, which JAX transposes for reverse mode. Engines are the one sanctioned exception (§3.2.4).
2. **No in-place operations in PyTorch** inside L1 and L2: they break double backward. The §2.4 check covers this.
3. **Safe modulus.** |z| has an infinite second derivative at z = 0, and exact zeros do occur (zero-filled masked pixels, padding). The rule is `r = sqrt(where(m, |z|², 1)); |z| = where(m, r, 0)` with `m = |z|² > 0`, the *double-where* pattern, which keeps NaN out of the gradients of the branch not taken. Values are unchanged, so reference values are unaffected. The same care applies to STL's regularised division `z / sqrt(|z| + 1e-8)` used in the cross S1 terms.
4. **Masks are weights**, never boolean indexing or `nanmean`, for the same reason.
5. **The reference follows the normalisation** (maths tab, OP-18): with `norm="self"` the reference is part of the computation and gradients flow through it; with `norm="ref"` it is a constant, through which no gradient flows.

*Technical note on rule 3 — why a single `where` is not enough:* `where(cond, a, b)` evaluates both branches on the whole array and only then selects element by element, and automatic differentiation propagates through both, multiplying the derivative of the discarded branch by 0. A branch that is singular where it is discarded — `sqrt` or `abs` at 0 — therefore still injects 0 × NaN = NaN into the gradient. `where(m, abs(z), 0)` fixes the value but not the gradient; the inner `where` makes the input of `sqrt` safe everywhere, so that every branch has finite derivatives on the whole array.

#### 3.2.4 Engines and differentiation order

- **Every engine declares a differentiation order** (`diff_order`). The Python engine has order 2, on PyTorch and on JAX.
- **Order-1 engines serve synthesis and first-order losses.** The Rust engine, from STL-Dev-MPI, has hand-written VJPs that are not themselves differentiable, so it declares order 1. A request for a JVP, a Hessian-vector product or a double backward then raises an error pointing to `engine="python"`.
- **Wrapping.** At order 1, an engine is wrapped with `jax.custom_vjp` or a `torch.autograd.Function` marked `once_differentiable`. At order 2, it provides a JVP and a differentiable VJP, so both modes compose.
- **Roadmap.** Second order is guaranteed for the Python engine, on PyTorch and JAX. For the Rust engine it is written by hand later (LF15, Part III §18.2). Until then, implicit differentiation goes through the Python engine. Whether `engine="auto"` should route second-order requests to the Python engine automatically is decided with the Rust engine (Part III, §16, LF3).

## 4. Mathematics

### 4.1 `maths.md` as an executable specification

`docs/maths.md` is a protected file that defines every quantity ESTEL computes, in the style of the exact formulas of the STL-Dev-MPI `CLAUDE.md`. Every definition has:

- an ID (`M-…`);
- the formula;
- its conventions: normalisation, N or N−1, which pixels count, regularisations;
- the `_spec` function that implements it;
- the tests that use it.

`tools/check_maths_ids.py` enforces this mapping in both directions. A disagreement between the code and `maths.md` is a failing test, and CI blocks the change. Changing a mathematical choice requires a demonstration or a benchmark, and a major version.

In step 1, `maths.md` records the mathematics of STL **as it is**, including choices that look accidental; those are listed as gaps (Part II, §13), not fixed silently.

### 4.2 The oracle `_spec`

`estel._spec` is a second implementation of `maths.md`, written to be obviously correct rather than fast: direct sums, explicit loops, float64, NumPy only, no import from the rest of ESTEL. It is the source of the reference values (§5.1). Because it shares no code with the library, a bug in the library cannot hide in the references. Each function's docstring lists the `M-` IDs it implements.

### 4.3 Outline of `maths.md`

| ID range | Content | Step |
| --- | --- | --- |
| M-NOT | Notation: batch, channels, blocks, levels, slots, ⟨·⟩ over valid pixels | 1 |
| M-DATA | Fields, masks, valid-pixel weights, pbc per block | 1 |
| M-WAV-1D, M-WAV-2D | Bump-steerable kernels: construction in Fourier space, crop to 5 taps or 5×5, orientations; the analytic 1D wavelet and its conjugate | 1 |
| M-WAV-HPX | HEALPix gauge-oriented kernels, stencils, decimation. The definitions of the STL-Dev-MPI `CLAUDE.md` §3, which give the same numbers as the PyTorch code | 1 |
| M-SMOOTH-2D, M-SMOOTH-1D, M-LP | Smoothing kernel s, the filter of one downsampling step; low-pass filter φ\_J, the effective filter of scale J, realised as the cascade of the smoothing kernels | 1 |
| M-DOWN | Normalised downsampling, and the threshold below which an output pixel is masked | 1 |
| M-BORDER | Padding modes; crop widths as a function of the convolutions applied | 1 |
| M-MASK | Mask propagation through convolutions | 1 |
| M-MOM | mean, var | 1 |
| M-S1 … M-S4 | Vanilla ScatCov, including cross-channel terms and the regularised division of the cross S1 term | 1 |
| M-TP | Tensor products: multi-index scales, the componentwise partial order, product orientations, low-pass φ and box term per block, as in STL | 1 |
| M-NORM | norm="self", norm="ref", batch mean | 1 |
| M-RED | iso, angular\_fourier, scale\_cosine | 1 |
| M-PS | Spectra per geometry: 2D estimators of v1.6.0, in 1.0.0 the 2D estimators only; 1D, tensor-product and HEALPix spectra in step 2 (LF23, LF24) | 1 |
| M-SYN | Loss, weights, Gaussian start, Tukey taper, frozen pixels, Nyquist pre-filter | 1 |
| M-LP34 | Low-pass filters in every geometry and in S3/S4 | 2 |
| M-PS-POS | Positive spectrum estimator for non-periodic data | 2 |
| M-TP-MASK | NaN masks in tensor products | 2 |
| M-CANON | The canonical coefficient set, for the compact layout | 2 |

## 5. Tests, reference values, tolerances

### 5.1 Test families

| Family | What | Runs |
| --- | --- | --- |
| Backend conformance | Each L0 primitive against NumPy float64, values and gradients | every backend, every PR |
| Spec parity | ESTEL against `_spec`, float64, small inputs, every family, geometry, mask and pbc combination | every PR |
| **Reference values** | Fixed seeded inputs and their stored outputs (`tests/reference/*.npz`): the core of "unchanged statistics within one `STATISTICS_REFERENCE`". They include the normalisation references (`var_ref`, the S2 reference, the PS reference), computed on inputs of roughly unit variance, so that a bug in S2 cannot hide behind the normalisation | every PR, every backend and dtype, with the tolerances of §5.2 |
| Invariances | Time reversal swaps the 1D wavelet and its conjugate; S1 ∝ \|a\| and S2 ∝ a² for x → a·x; normalised coefficients unchanged for x → a·x with a > 0, and with a < 0 too except S3, whose sign changes, and mean, which is not normalised (M-NORM); coefficients without box and low-pass terms unchanged for x → x + constant; batch independence; channel-permutation equivariance; block-permutation equivariance of tensor products; an all-valid mask equals no mask; a single-block tensor product equals the plain geometry, which is the reference. Exact translation invariance is not tested in 1.0.0: STL's downsampling probably breaks it (LF25). Nor is invariance under rotations: STL's 2D kernels are not exact rotations of each other (maths tab, OP-58), so a rotation by π/2 matches only to about 1e-2; it is a requirement of the joint design of the filters, tested with it (Part III, §20.10). Two invariances hold only approximately with STL's definitions, and are tested where they are exact: the scaling of the cross S1, because of its ε (OP-16), and the constant shift of 2D data with a mask, because the 2D smoothing kernel sums to 1.00000004 (OP-59) | every PR |
| Consistency of the estimators | The statistics of one stationary process, estimated on a periodic field, on the same field read as non-periodic, and on the same field with a mask, agree within their statistical error (paired z-scores, with a threshold corrected for the number of comparisons); only the scales that keep enough valid pixels are compared. A configuration where a gap is known (a positive control) shows that the test can see one. A systematic gap is a property of STL's estimator: in step 1 it is measured and recorded as an open point, not corrected (Part II, §15.2, T2.16) | full tier |
| Differentiation | The §3.2.2 table, float64 | every PR, CPU |
| Synthesis | In step 1, only the synthesis API: it runs, writes checkpoints and resumes from them; and one comparison with STL: for periodic planar data, a synthesis started from the same field as STL, with the same settings, in float64, follows STL's loss over its first iterations, within a tolerance set by the task (Part II, §15.2, T6.1). Synthesis reference outputs, cross-backend criteria and parity come after 1.0.0 (Part III, §16) | every PR (API); full tier (the comparison with STL) |
| maths ↔ spec ↔ tests | Every `M-` ID has a `_spec` function and at least one test, and vice versa; the syntheses (section 5 of `maths.md`) have tests and no `_spec` function (Part IV, §22, row of 21 November 2026, "No oracle for the syntheses") | every PR |
| API | `api/manifest.json` up to date; public vs internal (§2.4); every error message names the fix | every PR |
| Reserved interfaces | Each interface reserved for step 2 (Part II, §9.4) is exercised through its trivial implementation | every PR |
| STL parity | ESTEL against STL on the same inputs, both in float64, for the scattering statistics and the 2D spectra (Part II, §14) | step 1 only; retired after 1.0.0 |
| Engine parity | Reference values and gradients of an engine against the Python engine, per claimed configuration (Part III, §18) | step 2, every PR touching engines |

**Reference inputs** are seeded synthetic fields, plus small versions of the public data used by STL's certification notebooks and of the advected turbulence case of the STOC deliverable.

**Reference values** are produced by `tools/make_reference.py` from `_spec` in float64, never from ESTEL itself. In step 1 they are also checked against STL, run in float64 (Part II, §14.5).

"Every PR" in this table reads with the tiers of §5.4: a PR runs the fast tier, and what it leaves out (the tests marked `slow`, the float32 runs of the tests not marked `dtype`, JAX outside `jax_smoke`) runs on every merge to `main`, nightly, and on a PR labelled `full-ci`.

### 5.2 Tolerances

`tests/tolerances.toml` (protected) has one entry per dtype, device class and family. Comparisons are made on **normalised** coefficients (divided by the reference S2 scale), so that `atol` means the same for every family; the reference S2 scale is itself a reference value, compared with its own tolerance (§5.1). Starting values, calibrated before the freeze:

| Comparison | rtol | atol (normalised) |
| --- | --- | --- |
| float64 CPU vs reference | 1e-10 | 1e-12 |
| float64 GPU vs reference | 1e-9 | 1e-11 |
| float32 CPU or GPU vs reference | 1e-4 | 1e-6 |
| PyTorch vs JAX, same dtype and device | 1e-12 (f64), 1e-5 (f32) | idem |
| Rust engine, float64 (step 2) | its own row; bit-identical between runs for a fixed thread count, block size and device | — |

Calibration, written in `TOLERANCES.md`: run every backend, device and dtype on every reference input; take the largest error observed; multiply it by 10; record the hardware and the date. Explicit tolerances in tests cite this table, which is how review rule C10 allows non-default tolerances.

### 5.3 Protection of reference values

- **CODEOWNERS** makes human review mandatory on the protected paths:
  - reference data and tolerances: `tests/reference/**`, `tests/tolerances.toml`;
  - the reference counters: `src/estel/_reference.py`;
  - the STL parity fixtures: `tests/parity/fixtures/**`;
  - the specification: `docs/transfer/**`, `docs/maths.md` and `docs/maths_step2.md`, changed only through pull requests that a maintainer requests and merges;
  - the register of later features and its decision log: `docs/register/**`;
  - the oracle and the mathematics: `src/estel/_spec/**`, `docs/maths.md`, `docs/maths_step2.md`;
  - policy and API: `docs/policy/**`, `api/manifest.json`;
  - context files: `AGENTS.md`, `CLAUDE.md`, their mirrors, `docs/agents/review_rules.md`;
  - infrastructure: `.github/**`, `tools/make_reference.py`, `bench/thresholds.toml`;
  - in step 2, the engine ABI header and capability schema.
- **Every protected file starts with a banner**, for example: `AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. Read docs/policy/VERSIONING.md.`
- **`tests/reference/MANIFEST.sha256`** is checked in CI. From 1.0.0, a PR that changes any reference file without bumping `STATISTICS_REFERENCE`, with CODEOWNER approval, fails; before 1.0.0 the change is only reported.
- **CI fails if the number of tests drops**, or if a new `skip` or `xfail` appears without the `test-change` label. These are the usual ways an agent "fixes" a red test.
- **Nightly mutation testing** (`mutmut`) on L2 measures how strong the tests are.

### 5.4 Test speed: the fast and the full tier

Tests are as short as possible. The fast tier must stay under 2 minutes of wall time on CI and under 1 minute for a local `make check`. A new test runs in under 1 second, on the smallest data that exercises the behaviour (for example 16 x 16 or 32 x 32 pixels, J = 2 or 3); a test that cannot is marked slow, with the reason in its docstring, and runs in the full tier only. Large configurations, second-order gradients, and a second dtype for tests of logic belong to the full tier. Before adding a test, check that no existing test already covers the behaviour. A pull request that makes the fast tier slower by more than 10 % says why in its description, with the before and after times.

The tests run in two tiers (`tests/conftest.py`, `pytest --tier fast|full`):

| Tier | When | What |
| --- | --- | --- |
| fast | every PR, and every run of the full tier, under the names of the checks a merge requires (CI); `make check` (locally) | the check tools; on PyTorch, every test not marked `slow` or `serial` in float64, and the tests marked `dtype` in float32 too (dtype propagation through every public function, a few representative parity cases, the precision tests); on the experimental JAX backend, the tests marked `jax_smoke` (the L0 conformance suite and one parity case, of the scattering statistics, which runs the whole loop), in float64. The tests of the check tools (`tests/tools/`) run only when the change touches `tools/`, `tests/tools/` or `pyproject.toml`; every PR runs the tools themselves on the repository. Each backend runs in one process, beside the check tools: the start-up of a process (importing PyTorch and its first gradient) costs more than the fast tier's tests. The exception is a PyTorch run that includes the tests of the check tools, which uses one worker per core: these tests start many subprocesses, and the workers cost about 4 s on such a PR |
| full | every push to `main`, nightly, by hand, a PR labelled `full-ci` (CI), in jobs of its own that a merge does not require, the JAX float64 run in two parts; `make check-full` (locally) | the check tools and every test, `slow` included, on both backends, in float32 and float64 |

Nothing leaves the full tier: a test marked `slow` or left out of the fast tier still runs on every merge to `main` and every night. What is enforced:

- **Per test.** In the fast tier, a test that takes more than 1 second (set-up and call) fails, with a message saying to make it faster or to mark it `slow`. The time is CPU time, the test's subprocesses included, so that the load of the other parallel workers does not count; a wall-time limit five times larger catches a test that waits. CI is the reference machine: a laptop with efficiency cores measures up to three times more under load, so outside CI a test fails at 2 seconds and the run lists those between 1 and 2. On the experimental JAX backend, whose eager mode compiles or loads one executable per operation the first time a process builds an operator (1 to 3 seconds for a parity case with JAX's compilation cache, 10 to 20 without it), the times of the smoke subset are reported, not enforced: the same tests meet the budget on PyTorch, and the JAX job has its own total budget. A test marked `slow` without a docstring sentence starting with `Slow:` that gives the reason fails at collection.
- **In total.** In CI, each job of the fast tier fails above 120 seconds of wall time, set-up included. `make check` warns above 60 seconds and fails above 120, since machines differ. Each pytest run ends with a summary line: its tier, its number of tests, their CPU time and the slowest one.

## 6. CI, GPU access, benchmarks

### 6.1 Tiers

| Tier | Trigger | Where | Budget | Content |
| --- | --- | --- | --- | --- |
| 0 | every push and PR; nightly | GitHub-hosted CPU | fast tier ≤ 2 min, full tier ≤ 15 min | ruff, type check, §2.4 checks (including public vs internal), manifest, mirrors, maths IDs, the tests on CPU, differentiation, also on JAX, in parallel jobs (§3.1). PRs run the fast tier of §5.4; `main`, the nightly run and PRs labelled `full-ci` run the full tier: all tests in float32 and float64, the whole suite on both backends. In step 2: CPU build of the Rust wheels |
| 1 | PR, after tier 0 passes. **Set up after 1.0.0** (Part III, §16) | HPC GPU, constrained (§6.2) | ≤ 20 min wall time, 1 GPU | `pytest -m gpu`: reference values on GPU, determinism (two runs bit-identical), GPU syntheses, benchmark smoke test. In step 2: GPU build of the Rust wheels |
| 2 | `bench` label set by humans; release candidates. **Set up in step 2a** (Part III, §16, LF62) | dedicated, pinned GPU node, outside CI | hours | full benchmark; results committed by a bot to `bench/results/` with a hardware fingerprint |

### 6.2 Constrained GPU access from CI

The repository is public, and the code in a PR is written by agents and will run on the cluster. It is treated as untrusted:

- **Ephemeral runners only.** A GitHub self-hosted runner registered with a just-in-time token, one job per runner, destroyed afterwards. It lives on a gateway or login node, under a dedicated unprivileged service account, never a person's account.
- **The runner submits, the cluster runs.** The runner only submits a batch job and streams its logs back. The job runs in a container built from the repository's lock file, with:
  - no outbound network from the compute node;
  - no access to shared data or project filesystems;
  - no secrets in the environment;
  - a read-only image, with only a scratch directory writable.
- **Hard limits:** wall time ≤ 20 min, one GPU, a daily job quota, and a `concurrency` group so that a new push cancels the previous run.
- **Only PRs from branches of the main repository**, never forks. The `pull_request` trigger, never `pull_request_target`. On a public repository this restriction is essential: GitHub advises against running fork PRs on self-hosted runners.
- **Results come back as a check status and artifacts only.** The runner holds no write token to the repository.
- **A pinned node type.** Smoke-benchmark numbers are recorded but never gate a merge, because a shared HPC node is too noisy. Benchmark decisions that affect versions come only from tier 2.

Tier 1 is set up after 1.0.0. The scheduler, the container runtime and the network rules of the gateway node are confirmed then (Part III, §16).

### 6.3 Benchmarks

- **In step 1, a light benchmark.** The 2D cases on CPU of `bench/cases.toml` (Part II, §15.2, T2.11), run outside CI, check that ESTEL's time and memory are commensurate with STL's, and judge the optimisations of step 1, their statistics and their gradients (T2.12). One 2D+1D case and one HEALPix case, measured on ESTEL only, judge the trials of phases P3 and P5 (Part II, §15.2, T3.5, T5.7); STL's times on tensor products and on HEALPix data are measured with the benchmark of step 2a (LF62).
- **In step 2a, the benchmark properly developed** (Part III, §16, LF62): public cases defined with the maintainers, one per geometry at a small and a medium size, the 2D+1D cases of the STOC deliverable at reduced sizes, JAX compile time, peak memory, synthesis time per iteration, the Rust engine against the Python engine; few cases, each clear, at a realistic size. `bench/thresholds.toml` (protected) then holds the "significant change" threshold of each metric, chosen by humans: to begin with, a factor 2 in speed or in peak memory.
- **Correctness.** Every benchmark run also asserts the correctness of its results, so that no gain can come from a wrong computation.
- **Comparisons.** An optimisation is judged by the comparison mode of the benchmark: the branch and `main` measured in the same session, on AC power, in alternating rounds of fresh processes, and the statistics and the gradient of each case compared with `main`'s. It is judged on the median ratio of each measure, against the noise floor of `main` against `main`: about ±2.5 % on the reference laptop, where a single round can be 5 to 20 % off. Each result enters the table of Part III, §19.14.

## 7. Versioning and merge policy

### 7.1 Version numbers

| Bump | When | Who decides |
| --- | --- | --- |
| **MAJOR** | a breaking interface change: a removed or renamed function or argument, a changed default, except in step 2, whose changes of defaults ship in 1.x releases until 2.0.0 (§1.3); a new geometry. Whether a change of statistics values also forces a major is decided case by case with the maintainers | humans (CODEOWNER approval) |
| **MINOR** | new features, including step-2 features and new engine configurations; a significant benchmark gain (§6.3), including making an engine the default for a configuration | automatic; humans set the `bench` label for benchmark-driven minors |
| **PATCH** | bug fixes that leave reference values unchanged; refactors; docs | automatic |

**Two reference counters.** Besides the package version, ESTEL publishes two integers, never reset:

- `estel.STATISTICS_REFERENCE`, bumped whenever any statistic value changes. The same number means the same statistics, within the tolerances of §5.2, whatever the package version.
- `estel.SYNTHESIS_REFERENCE`, the same for the reference synthesis trajectories, qualified by backend and pinned dependencies (§1.6).

Both are recorded in every saved result (its run manifest), shown in the release notes, and checked by CI, which computes when they must be bumped (§7.2). Bumping either needs CODEOWNER approval.

**Within one value of `STATISTICS_REFERENCE`, statistics and test results are unchanged up to the tolerances of §5.2.** This sentence opens `VERSIONING.md` and `AGENTS.md`.

How the two steps map onto versions:

- **Before 1.0.0** versions are 0.x and nothing is gated by reference values. Pull requests merge when CI passes, except on protected paths (§5.3). The three CODEOWNERS — Erwan Allys (GitHub `Eralys`), Jean-Marc Delouis (`jmdelouis`) and Pablo Richard (`pablo-richard`) — review step 1 as a whole at its end, before the freeze; the AI review is an advisory aid throughout.
- **1.0.0** is the end of step 1: parity with STL, references signed off by the three CODEOWNERS.
- **Step 2.** Every entry of step 2, in step 2a as in step 2b, lands on `main` as the new behaviour when it is done, and ships in the next release, a 1.x minor; step 2b ends with 2.0.0. Each release drops the previous defaults at once: no behaviour of STL is kept. A pull request that changes values regenerates the references and bumps `STATISTICS_REFERENCE`, with CODEOWNER approval as every change of reference, and removes the STL parity tests it breaks, a decision its approval records (AGENTS.md, golden rules 1 and 2). So users may see the statistics change in several 1.x releases; step 2a is done when its last entry has shipped (Part III, §19.13).

### 7.2 Computing the required bump

`tools/version_impact.py` runs in tier 0 and determines the **required** bump:

- reference tests outside tolerance, or a changed `tests/reference/` → `STATISTICS_REFERENCE` (or `SYNTHESIS_REFERENCE`) must be bumped, with CODEOWNER approval;
- `api/manifest.json`: an entry removed or changed → major, an entry added → minor;
- the geometry registry changed → major;
- the `bench` label → minor.
- a PR that changes only `.github/` needs no news fragment: it does not change the package. Such PRs, including Dependabot's, still need CODEOWNER approval, since `.github/` is protected.

**How CI sees a bump of a reference counter.** `STATISTICS_REFERENCE` and `SYNTHESIS_REFERENCE` are integer constants in `src/estel/_reference.py`, a protected file (§5.3), exported by `estel/__init__.py`. `tools/version_impact.py` reads them on the base branch and on the PR:

- a PR that changes `tests/reference/` must increase `STATISTICS_REFERENCE` by exactly one, and a PR that changes `STATISTICS_REFERENCE` must change `tests/reference/`; before 1.0.0 a mismatch is only reported, from 1.0.0 it fails;
- `SYNTHESIS_REFERENCE` follows the same rule for the synthesis references, once they exist (after 1.0.0, Part III, §16, W2). It stays 0 until then, and goes to 1 with the first synthesis references; at the 1.0.0 freeze, only `STATISTICS_REFERENCE` goes to 1;
- no label is involved: the bump is in the code, visible in the diff, and approved through CODEOWNERS.

Each PR adds a news fragment `changes/<PR>.<major|minor|patch>.md` (towncrier). CI fails if the declared bump is below the required one.

### 7.3 Merging

| Required change | Merge |
| --- | --- |
| patch | auto-merge when tier 0 passes (and tier 1, once it exists) |
| minor | auto-merge when CI passes; benchmark-driven minors after a human sets the `bench` label |
| major, or a bumped reference counter | CODEOWNER approval — the human in the loop — then merge |

The AI review (the GRID4EARTH rubric adapted to ESTEL, §8.3) runs on every PR, posts comments and **never blocks**. Every rule that can be checked mechanically must therefore be a CI check (§2.4, §5.3). Branch protection: `main` protected, required checks = tier 0 (and tier 1 once it exists), linear history, one topic per PR (rule E1).

## 8. Context files for agents

### 8.1 Mirrored files

Every coding agent reads the same rules:

| File | Read by | Content |
| --- | --- | --- |
| `AGENTS.md` (canonical) | Codex, Copilot, Cursor, Gemini CLI and others | ≤ 300 lines: the golden rules, the layer map, the commands to run, links to `docs/agents/*.md` |
| `CLAUDE.md` | Claude Code | first line `@AGENTS.md` — an import, which works whether or not the tool also reads `AGENTS.md` on its own — then Claude-specific notes |
| `.github/copilot-instructions.md` | GitHub Copilot | generated copy of `AGENTS.md` |
| nested `AGENTS.md` and `CLAUDE.md` in `src/estel/backend/`, `src/estel/engines/`, `src/estel/_spec/`, `tests/reference/` | the agent working in that directory | local rules, for example "this is the oracle: …" |

`tools/check_agent_mirrors.py` fails CI when a mirror differs from what it should be generated from.

### 8.2 Golden rules

The golden rules are the top of `AGENTS.md`, their only copy: every session reads them first, and the mirrored files (§8.1) follow them. They are not repeated here.

### 8.3 Review rules

`docs/agents/review_rules.md` adapts the GRID4EARTH review rubric. It states every rule in full, in ESTEL's own words, so that agents need nothing else; the GRID4EARTH repository carries no licence, so its text is credited, not copied. The rules kept:

- **Adopted as is:** A1, A3, A6, A7, A8 (which keeps component separation and inverse problems out of the core), A9, A10, B1, B3–B10, C1–C5, C7–C10, D1–D7, D9–D12, E1–E3, E5.
- **Adapted:** A4 (healpix-geo terminology, for `DataHealpix` only), A11 (one dtype contract, applied to Python).
- **For the Rust engine:** A5, B2, C6, D8 and E4 apply in the `rust/` directory of the repository (Part III, §18.2). A2 is specific to healpix-geo and dropped.
- **Added, ESTEL section:**
  - X1 reference values;
  - X2 backend purity;
  - X3 functional style;
  - X4 differentiability to second order;
  - X5 storage discipline: no full-grid tensor inside the computation;
  - X6 sync between mathematics, oracle and tests;
  - X7 version impact;
  - X8 errors an agent can act on;
  - X9 step discipline.

### 8.4 Tuning prompts

The "properly located prompts" of ESTEL's philosophy live in `docs/agents/tuning/` and are linked from the docstring of every function they concern ("Tuning: see …"). There is one file per theme: synthesis, scales and memory, masks and borders, power spectrum. Each file has three parts:

- **Knobs:** the parameters, their effect, and what they cost.
- **Symptoms to knobs:** for example, pixel-scale noise growing during a synthesis, a lost daily cycle, negative spectrum bins, a memory error.
- **The rule:** tune in your own code, through the public API. Library defaults change only through a major version.
