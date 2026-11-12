# Part I — ESTEL general rules

Back to Overview. These rules hold for every version of ESTEL, from the first commit of step 1 onwards. They will move, almost unchanged, into the ESTEL reference document.

## Contents of this part

- [1. Principles](#1-principles)
  - [1.1 Purpose and scope](#11-purpose-and-scope) · [1.2 A library maintained by agents](#12-a-library-maintained-by-agents) · [1.3 Two-step development](#13-two-step-development) · [1.4 Design for what comes next](#14-design-for-what-comes-next) · [1.5 Licensing](#15-licensing) · [1.6 Supported versions and dependencies](#16-supported-versions-and-dependencies)
- [2. Architecture](#2-architecture)
  - [2.1 Layers](#21-layers)
  - [2.2 Multi-backend strategy](#22-multi-backend-strategy): [2.2.1 The primitive layer](#221-the-primitive-layer-l0) · [2.2.2 Static plans](#222-static-plans) · [2.2.3 Backend selection](#223-backend-selection) · [2.2.4 Engines](#224-engines) · [2.2.5 Roll-out](#225-roll-out)
  - [2.3 Object model](#23-object-model)
  - [2.4 Code rules enforced in CI](#24-code-rules-enforced-in-ci)
- [3. Numerical coding rules](#3-numerical-coding-rules)
  - [3.1 Functional, JAX-ready style](#31-functional-jax-ready-style): [3.1.1 The scattering loop without in-place updates](#311-the-scattering-loop-without-in-place-updates)
  - [3.2 Differentiation contract](#32-differentiation-contract): [3.2.1 Why second order](#321-why-second-order) · [3.2.2 Guarantees](#322-guarantees) · [3.2.3 Implementation rules](#323-implementation-rules) · [3.2.4 Engines and differentiation order](#324-engines-and-differentiation-order)
- [4. Mathematics](#4-mathematics)
  - [4.1 maths.md as an executable specification](#41-mathsmd-as-an-executable-specification) · [4.2 The oracle](#42-the-oracle-_spec) · [4.3 Outline of maths.md](#43-outline-of-mathsmd)
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

### 1.4 Design for what comes next

- Every interface a planned feature will need is created in 1.0.0, minimal and tested, even when its only implementation is trivial — for example an `allreduce` hook that does nothing without MPI.
- Storage, backend and engine choices stay invisible to users: changing them never changes a public signature.
- A reserved interface that turns out unnecessary is removed in a major version, not left to decay.

### 1.5 Licensing

- **ESTEL is released under Apache-2.0.** It is permissive, like JAX, optax and healpix-analyse, and adds an explicit patent grant: each contributor licenses the patents their contribution needs, and anyone who sues claiming ESTEL infringes a patent loses that licence. The repository holds `LICENSE` and `NOTICE`; `pyproject.toml` declares `license = "Apache-2.0"` and `license-files = ["LICENSE", "NOTICE"]`.
- **Precondition: STL's licence.** ESTEL ports STL code (Part II, §12), so it is a derivative of STL and inherits its terms. Before the first public release, STL's licence is confirmed in writing; if the STL-Dev repositories carry none, every STL author, and their employer where it owns the code, agrees to Apache-2.0. The same holds for the STL-Dev-MPI Rust code reused in step 2. STL's copyright line goes into `NOTICE`. Funding contracts are checked for any required licence.
- **Dependencies stay permissive.** Runtime dependencies are MIT, BSD or Apache-2.0, as all of them are today. GPL code is never imported by the package nor ported into it: healpy (GPL-2) remains a test-only cross-check of the oracle, and the JAX backend does not use jax-healpy (GPL-3).
- **Rust wheels (step 2).** `estel-rust` wheels statically link their Rust crates, and so redistribute them: `cargo-deny` in CI rejects copyleft crates, `cargo-about` generates the third-party notices shipped in each wheel, and no GPL compiled library (FFTW, ducc0) is linked. Dual MIT OR Apache-2.0, the Rust convention, stays an option if the crates are published on crates.io.
- **Contributions** are accepted under Apache-2.0 with a DCO sign-off checked in CI, so that the origin of every change stays traceable.

### 1.6 Supported versions and dependencies

- **Minimum versions**, confirmed on 12 November 2026: Python 3.11, NumPy 2.0, PyTorch 2.6 and array-api-compat 1.11; in step 2, JAX 0.5 and optax 0.2.4. All were released in 2025 or earlier, so the 2025 versions of the scientific Python stack are supported.
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
  - **L1, geometry.** Written once on top of L0. Each DataClass provides a small set of geometry hooks; the mother class holds everything else.
  - **L0, primitives.** About twenty functions in fourteen groups (Part II, §12.3), the only code that imports a backend: PyTorch in step 1, JAX in step 2, chosen with `set_backend()`.
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
| `jax/` | JAX implementation, with `optax.lbfgs` | 2 |

`tests/backend/` holds a **conformance suite**. Each primitive is compared with a NumPy float64 implementation on random inputs, every padding mode, real and complex values, odd sizes, and gradients. Adding a backend means passing this suite; the rest of the library needs no change.

#### 2.2.2 Static plans

At construction, an operator builds a `ScatteringPlan` in NumPy: scale and orientation index sets, wavelet and low-pass kernels, crop widths per level, mask footprints, spectrum bins, HEALPix stencils. The plan is immutable and hashable. In PyTorch it simply avoids recomputation; in JAX it is a compile-time constant, which is what allows `jax.jit` to compile the whole computation. Large arrays of the plan are passed to the computation as arguments, never baked in as constants.

#### 2.2.3 Backend selection

- **A global default, read when an operator is built.** `estel.set_backend("jax")` sets the default; every operator records the backend it was built with.
- **Mixing raises an error.** A torch tensor passed to a JAX-built operator raises `BackendMismatchError`, whose message says how to fix it.
- **Inputs are converted at the boundary.** DataClass constructors accept NumPy and backend arrays and convert them to the current backend. They never convert silently from one backend to another.

#### 2.2.4 Engines

- **Two engines, one protocol.** Every engine implements the same protocol: a forward pass that returns the statistics, and a backward pass (VJP) for gradients. The Python engine (L2 + L1 + L0) is always available. The Rust engine (Part III, §18) arrives in step 2. In 1.0.0, `engine=` accepts only `"python"`; the argument, the registry and the protocol are reserved and tested.
- **Why the Rust code is an engine, not a backend.** A backend only provides elementary operations, whose compositions PyTorch or JAX differentiate automatically. Rewriting those operations in Rust would require a Rust automatic-differentiation system, and would lose what makes STL-Dev-MPI fast: running the whole algorithm by blocks, with fused operations, a hand-written gradient of the whole chain, and MPI. The Rust code therefore replaces the whole computation, behind the same Python frontend (Part IV, §22.4).
- **Selection.** `STOperator(..., engine="auto" | "python" | "rust")`. With `"auto"`, ESTEL uses, for each configuration, the engine designated as its default: the Python engine, unless a benchmark-gated minor version has promoted the Rust one (§7.1). With `"rust"`, an unsupported configuration raises `EngineUnsupportedError`, naming the unsupported option.
- **Capability declaration.** At construction, an engine reports what it computes: geometries, options, dtypes, devices, differentiation order. The Python engine covers everything ESTEL defines.
- **Purity at the boundary.** The forward pass returns the values and an opaque residual; the VJP takes the residual back. Nothing relies on state hidden inside an engine.
- **Same numbers.** Every engine passes the reference tests for every configuration it claims; the Rust engine also passes a parity test of its gradients against the Python engine. No engine produces reference values: they come from the oracle.
- **Differentiation order.** Declared by each engine (§3.2.4).

#### 2.2.5 Roll-out

| Stage | Engines (`engine=`) | Backends of the Python engine (`set_backend()`) | Precision |
| --- | --- | --- | --- |
| Step 1 (1.0.0) | Python engine only; engine layer and protocol reserved | PyTorch | float32 default, float64 path |
| Step 2 | + Rust engine: the `estel-rust` wheel, built from `rust/` in the same repository; CPU and GPU, with MPI (Part III, §18) | + JAX (Part III, §17) | Rust engine: float64 default, float32 opt-in |
| Later | any engine that passes the parity tests | any backend that passes the conformance suite | — |

### 2.3 Object model

- **A DataClass is data plus static metadata.** It holds the array and its geometry metadata (pbc, cell ids, block structure). It is registered as a pytree: the array is its only leaf, everything else is static. The same object therefore works in PyTorch and inside `jax.jit`.
- **Masks are static.** As in STL, a mask of missing data and its reweighting maps are computed once, when the operator is built, and stay fixed for every forward and backward pass. They belong to the static plan and reach the computation as constant arrays. What is forbidden is only to detect missing values, or to choose an estimator, during the computation (§3.1, rule 3).
- **The convolution history is plan data, not data state.** STL keeps a mutable `conv_history` list on each data object, to know which convolutions were applied and derive the border crop from it. In ESTEL the sequence of convolutions is fixed by the operator, so the `ScatteringPlan` computes the crop width of every level once. A data object carries only immutable metadata (a level index, tuples), which JAX treats as compile-time constants.
- **An operator is a frozen plan plus a pure function.** `STOperator(...)` builds the plan and the backend arrays (kernels, masks). `op.apply(data, ref=None)` calls `estel.functional.scattering(x, plan, arrays, ref)`. That pure function is what `jax.jit`, `jax.grad` and `torch.func` see, and, with L1 and L0, it makes up the Python engine. The method keeps STL's name, `apply`; whether to also offer PyTorch's `__call__` and `forward` is a point to discuss (Part IV, §24.2).
- **One way to say it (review rule A6).** In STL, `ST_Operator.__init__` and `apply` accept the same options (`norm`, `iso`, `flatten`, `compute_PS`, …), with `None` meaning "inherit from the constructor": two ways to set each option. In ESTEL every option is set once, at construction; `apply` takes only the data and an optional reference. This is also what makes the plan static. A different configuration is a different operator.
- **One entry point per object.** `estel.STOperator(data, ...)` replaces `data.get_ST_op()`, and `estel.WaveletOperator(data, ...)` replaces `data.get_wavelet_op()`.
- **Class names follow PEP 8:** CapWords, no underscore. STL's `ST_Operator`, `ST_Operator_TensorProd` and `ST_Statistics` become `STOperator`, `STOperatorTensorProd` and `STStatistics`.
- **The mother class is the simple case.** `STOperator` handles a single domain (1D, 2D, HEALPix). `STOperatorTensorProd` overrides a small, fixed set of hooks for products of domains (Part II, §12.6), and nothing else.

### 2.4 Code rules enforced in CI

Each rule is enforced by a mechanical check — `import-linter` contracts and a small AST checker in `tools/` — not by review:

| Rule | Check |
| --- | --- |
| `torch`, `jax`, `optax` imported only under `estel/backend/` | import-linter "forbidden" contract |
| `estel_rust` imported only under `estel/engines/` (step 2) | import-linter "forbidden" contract |
| L2 never imports a DataClass subclass; it talks to the mother-class interface | import-linter "layers" contract |
| `estel._spec` imports NumPy only, and nothing from `estel` | import-linter "independence" contract |
| No `type(x).__name__ ==` comparisons, no assignment to `__class__` | AST check |
| No in-place array mutation in L1/L2 (`x[...] =`, `out=`, `.add_`, `inplace=True`, `x += y`, `del x[...]`, `__setitem__`). The NumPy modules that build the static plan (tag N in Part II, §12) are exempt from this rule only, through an explicit list in the checker: they run once, at construction, and are never traced or differentiated | AST check |
| **Public vs internal.** Public names are exactly those listed in `api/manifest.json` and exported from `estel/__init__.py`. Internal functions live in private modules, named with a leading underscore (`spectra/_estimators.py`), with plain names inside; internal methods of public classes start with an underscore | `tools/api_manifest.py --check` fails if a name is reachable without an underscore, outside a private module, and not in the manifest |

**Why these rules, in plain terms:**

- **No class-name comparisons.** `type(x).__name__ == "STL_2D_Kernel_Torch"` compares the name of a class as a string. It breaks as soon as someone writes a subclass — a 1D class built on the 2D one, a tensor-product block — and the check silently rejects valid data or takes the wrong branch. The correct test is `isinstance`, or better, a capability the object declares. STL has three such checks; the one in the v1.6.0 cross-spectrum operator rejects 1D and tensor-product data.
- **No reassignment of `__class__`.** Changing an object's class after it is created means neither readers nor tools (type checkers, serialisation, JAX, `torch.compile`) can know what the object is. STL's tensor-product class does this at every instantiation. ESTEL composes instead: a tensor-product object holds one array and the geometry of each block, all on the same backend.
- **No in-place updates.** Writing into an existing array breaks JAX, which forbids it, and second-order differentiation in PyTorch. It also makes results depend on the order of calls.
- **No backend import outside L0.** This single rule is what keeps JAX and Rust engines possible. One stray `import torch` in the geometry code, and a JAX port becomes a rewrite.
- **Public or internal, visible in the code.** The frozen API is ESTEL's central promise, so users and agents must see from the code itself what they may rely on. A name that looks public will be used, and can then no longer change without a major version.

## 3. Numerical coding rules

### 3.1 Functional, JAX-ready style

These rules apply to L1 and L2 from the first commit of step 1, although the JAX backend only arrives in step 2. Rules that are not tested drift, so from phase P1 the CI also traces the L2 code on small inputs with a minimal, unreleased JAX implementation of L0. That test makes no numerical claim; it only proves the code is traceable. The rules are copied into `docs/agents/jax_rules.md`.

1. **Functional code only.** Build per-level results in Python lists (§3.1.1), then `concat` or `stack` them. No preallocation filled by index, no `out=`, no `inplace=`. This applies to the code that computes the statistics; the NumPy code that builds the static plan, run once at construction, may write in place (§2.4).
2. **Static shapes.** Everything shape-related — index sets, crop widths, stencils, bins — lives in the `ScatteringPlan`, built in NumPy and passed as a static argument.
3. **No decisions on array values inside traced code.** No `if x.isnan().any()`, no `.item()`, no `float(x)`, no `print(x)`. NaN detection and estimator selection happen when the DataClass or the operator is constructed; the masks and reweighting maps they produce are then static data of the plan (§2.3).
4. **Large constants are arguments, not closures.** Kernels, masks and stencils are passed as arrays to the computation (`arrays` in `scattering(x, plan, arrays)`), so they are not baked into the compiled program. Whether baking small kernels in as compiled constants would be faster is a step-2 question (LF29).
5. **Explicit randomness.** Every random function takes a key or a seed (`make_key(seed)`).
6. **Complex convolutions are two real convolutions**, as STL's `_semicomplex_conv` already does. Kernels are stored as real pairs. Whether complex-to-complex convolutions would be faster in tensor products is a step-2 question (LF30).
7. **float32 by default, float64 on request.** Every computation must run in float32, the default, as well as in float64: no hard-coded float64 constant, no silent change of dtype; kernels, masks and plan arrays are cast to the operator's dtype when it is built, and the tests run in both. `set_default_dtype("float64")` enables float64 on every backend; on JAX it switches on `jax_enable_x64`, and raises an error if JAX already ran in float32 in that process.
8. **Short Python loops, for JAX.** JAX unrolls Python loops when it traces the code, so their length drives compile time, which is part of the benchmark with a budget per case. Step 1 keeps the structure of STL's scattering loop, which limits memory and avoids recomputation, especially for tensor products, and only writes it in functional style. Any reorganisation of the loop belongs to step 2, and is adopted only after a realistic benchmark and a human validation (Part III, §19.3).
9. **Data and statistics are pytrees.** `DataClass` and `STStatistics` are registered as pytrees (arrays as leaves, metadata static), so that `jax.vmap` and `jax.jit` accept them.

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
- **The full-grid view is assembled once, at the end.** One out-of-place primitive, `scatter_to_grid(values, index, shape, fill)`, places the stacked values at the static positions of the plan, with NaN everywhere else (Part II, §12.3). Its inverse, `gather_from_grid(grid, index)`, reads the existing coefficients back.
- **Missing data are weights.** Means are computed as `sum(w * x) / sum(w)` with w equal to 0 or 1, never as `x[mask]` or `nanmean`, so that every shape stays fixed. The denominators `sum(w)` depend only on the static masks, so they are computed once, in the plan.

The cost is negligible. Under `jax.jit`, the compiler reuses memory itself; in PyTorch, stacking costs one copy at the end, small beside the convolutions. In step 2, computing the second-layer means by contraction instead of building the full product first lowers the memory peak further (Part III, §19.3, LF31).

This style is required from step 1, although JAX only arrives in step 2: it is also what keeps second-order derivatives correct in PyTorch, and what makes results independent of the order of calls.

### 3.2 Differentiation contract

#### 3.2.1 Why second order

Most uses of ESTEL need only gradients. A synthesis adjusts a field until its statistics match a target, and needs the derivative of a loss with respect to the field: first order.

Downstream repositories go further. In component separation, for example, one solves an optimisation problem — find the maps that best explain the observations — and then wants to know how that solution moves when a parameter changes: a noise level, a foreground model, a prior. Solving again for every parameter value is too costly. The implicit function theorem gives the answer from the solution itself. For θ\*(φ) = argmin over θ of L(θ, φ):

```latex
\frac{d\theta^*}{d\phi} = -\left[\frac{\partial^2 L}{\partial\theta^2}\right]^{-1} \frac{\partial^2 L}{\partial\theta\,\partial\phi}
```

Computing this requires products of the Hessian — the matrix of second derivatives — with vectors, and mixed second derivatives. ESTEL therefore guarantees derivatives of order two of its statistics. This changes nothing in the formulas; it only forbids a few implementation shortcuts (§3.2.3).

#### 3.2.2 Guarantees

| Guarantee | PyTorch (step 1) | JAX (step 2) | Test |
| --- | --- | --- | --- |
| VJP (reverse mode) of `scattering`, `cross_spectrum`, `loss` | autograd | `jax.vjp` | `gradcheck`, `check_grads(order=1)` |
| JVP (forward mode) | `torch.func.jvp` | `jax.jvp` | `check_grads(modes=("fwd",))` |
| Second order (Hessian-vector products, mixed derivatives) | double backward (`create_graph=True`), `torch.func` | `jax.jvp(jax.grad(...))` | `gradgradcheck`, `check_grads(order=2, modes=("fwd", "rev"))` |
| Implicit differentiation, end to end |  |  | a small argmin problem; implicit gradient vs finite differences |

All differentiation tests run in **float64** on small inputs, one reason the float64 path stays.

#### 3.2.3 Implementation rules

1. **No `custom_vjp` in the Python engine.** It removes forward mode in JAX. If a custom rule is ever needed there, use `custom_jvp`, which JAX transposes for reverse mode. Engines are the one sanctioned exception (§3.2.4).
2. **No in-place operations in PyTorch** inside L1 and L2: they break double backward. The §2.4 check covers this.
3. **Safe modulus.** |z| has an infinite second derivative at z = 0, and exact zeros do occur (zero-filled masked pixels, padding). The rule is `r = sqrt(where(m, |z|², 1)); |z| = where(m, r, 0)` with `m = |z|² > 0`, the *double-where* pattern, which keeps NaN out of the gradients of the branch not taken. Values are unchanged, so reference values are unaffected. The same care applies to STL's regularised division `z / sqrt(|z| + 1e-8)` used in the cross S1 terms.
4. **Masks are weights**, never boolean indexing or `nanmean`, for the same reason.
5. **The reference is an explicit argument.** In `apply(data, ref)`, gradients flow into `ref` unless the caller stops them. Whether the default normalisation should stop them is decided after 1.0.0 (Part III, §20.7).

*Technical note on rule 3 — why a single `where` is not enough:* `where(cond, a, b)` evaluates both branches on the whole array and only then selects element by element, and automatic differentiation propagates through both, multiplying the derivative of the discarded branch by 0. A branch that is singular where it is discarded — `sqrt` or `abs` at 0 — therefore still injects 0 × NaN = NaN into the gradient. `where(m, abs(z), 0)` fixes the value but not the gradient; the inner `where` makes the input of `sqrt` safe everywhere, so that every branch has finite derivatives on the whole array.

#### 3.2.4 Engines and differentiation order

- **Every engine declares a differentiation order** (`diff_order`). The Python engine has order 2, on PyTorch and on JAX.
- **Order-1 engines serve synthesis and first-order losses.** The Rust engine, from STL-Dev-MPI, has hand-written VJPs that are not themselves differentiable, so it declares order 1. A request for a JVP, a Hessian-vector product or a double backward then raises an error pointing to `engine="python"`.
- **Wrapping.** At order 1, an engine is wrapped with `jax.custom_vjp` or a `torch.autograd.Function` marked `once_differentiable`. At order 2, it provides a JVP and a differentiable VJP, so both modes compose.
- **Roadmap.** Second order is guaranteed for the Python engine, on PyTorch and JAX. For the Rust engine it is written by hand later (LF15, Part III §18.2). Until then, implicit differentiation goes through the Python engine. Whether `engine="auto"` should route second-order requests to the Python engine automatically is a point to discuss (Part IV, §24.2).

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
| M-WAV-HPX | HEALPix gauge-oriented kernels, stencils, decimation. The definitions of the STL-Dev-MPI `CLAUDE.md` §3, which give the same numbers as the PyTorch code (Q10) | 1 |
| M-H, M-LP | Anti-aliasing kernel H; low-pass φ\_J = H^(J) | 1 |
| M-DOWN | Normalised downsampling, and the threshold below which an output pixel is masked | 1 |
| M-BORDER | Padding modes; crop widths as a function of the convolutions applied | 1 |
| M-MASK | Mask propagation through convolutions | 1 |
| M-MOM | mean, var | 1 |
| M-S1 … M-S4 | Vanilla ScatCov, including cross-channel terms and the regularised division of the cross S1 term | 1 |
| M-TP | Tensor products: multi-index scales, the componentwise partial order, product orientations, low-pass φ and box term per block, as in STL | 1 |
| M-NORM | store\_ref, load\_ref, batch mean | 1 |
| M-RED | iso, angular\_ft, scale\_ft | 1 |
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
| Invariances | A rotation by π/2 permutes the 2D orientations; time reversal swaps the 1D wavelet and its conjugate; S1 ∝ \|a\| and S2 ∝ a² for x → a·x; normalised coefficients unchanged for x → a·x, a ≠ 0; coefficients without box and low-pass terms unchanged for x → x + constant; batch independence; channel-permutation equivariance; block-permutation equivariance of tensor products; an all-valid mask equals no mask; a single-block tensor product equals the plain geometry, which is the reference. Exact translation invariance is not tested in 1.0.0: STL's downsampling probably breaks it (LF25) | every PR |
| Differentiation | The §3.2.2 table, float64 | every PR, CPU |
| Synthesis | In step 1, only the synthesis API: it runs, writes checkpoints and resumes from them. Synthesis reference outputs, cross-backend criteria and parity come after 1.0.0 (Part IV, §24.1) | every PR (API only) |
| maths ↔ spec ↔ tests | Every `M-` ID has a `_spec` function and at least one test, and vice versa | every PR |
| API | `api/manifest.json` up to date; public vs internal (§2.4); every error message names the fix | every PR |
| Reserved interfaces | Each interface reserved for step 2 (Part II, §9.4) is exercised through its trivial implementation | every PR |
| STL parity | ESTEL against STL on the same inputs, both in float64, for the scattering statistics and the 2D spectra (Part II, §14) | step 1 only; retired after 1.0.0 |
| Engine parity | Reference values and gradients of an engine against the Python engine, per claimed configuration (Part III, §18) | step 2, every PR touching engines |

**Reference inputs** are seeded synthetic fields, plus small versions of the public data used by STL's certification notebooks and of the advected turbulence case of the STOC deliverable. A hidden subset of inputs is kept outside the public repository and run by the maintainers before each release, so that no change can be tuned to the visible cases.

**Reference values** are produced by `tools/make_reference.py` from `_spec` in float64, never from ESTEL itself. In step 1 they are also checked against STL, run in float64 (Part II, §14.5).

### 5.2 Tolerances

`tests/tolerances.toml` (protected) has one entry per dtype, device class and family. Comparisons are made on **normalised** coefficients (divided by the reference S2 scale), so that `atol` means the same for every family; the reference S2 scale is itself a reference value, compared with its own tolerance (§5.1). Starting values, calibrated before the freeze:

| Comparison | rtol | atol (normalised) |
| --- | --- | --- |
| float64 CPU vs reference | 1e-10 | 1e-12 |
| float64 GPU vs reference | 1e-9 | 1e-11 |
| float32 CPU or GPU vs reference | 1e-4 | 1e-6 |
| PyTorch vs JAX, same dtype and device (step 2) | 1e-12 (f64), 1e-5 (f32) | idem |
| Rust engine, float64 (step 2) | its own row; bit-identical between runs for a fixed thread count, block size and device | — |

Calibration, written in `TOLERANCES.md`: run every backend, device and dtype on every reference input; take the largest error observed; multiply it by 10; record the hardware and the date. Explicit tolerances in tests cite this table, which is how review rule C10 allows non-default tolerances.

### 5.3 Protection of reference values

- **CODEOWNERS** makes human review mandatory on the protected paths:
  - reference data and tolerances: `tests/reference/**`, `tests/tolerances.toml`;
  - the reference counters: `src/estel/_reference.py`;
  - the transfer specification: `docs/transfer/**` and, during the review, `docs/maths.md`, the specification itself, changed only through pull requests that a maintainer requests and merges;
  - the oracle and the mathematics: `src/estel/_spec/**`, `docs/maths.md`;
  - policy and API: `docs/policy/**`, `api/manifest.json`;
  - context files: `AGENTS.md`, `CLAUDE.md`, their mirrors, `docs/agents/review_rules.md`;
  - infrastructure: `.github/**`, `tools/make_reference.py`, `bench/thresholds.toml`;
  - in step 2, the engine ABI header and capability schema.
- **Every protected file starts with a banner**, for example: `AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. Read docs/policy/VERSIONING.md.`
- **`tests/reference/MANIFEST.sha256`** is checked in CI. From 1.0.0, a PR that changes any reference file without bumping `STATISTICS_REFERENCE`, with CODEOWNER approval, fails; before 1.0.0 the change is only reported.
- **CI fails if the number of tests drops**, or if a new `skip` or `xfail` appears without the `test-change` label. These are the usual ways an agent "fixes" a red test.
- **Nightly mutation testing** (`mutmut`) on L2 measures how strong the tests are.

## 6. CI, GPU access, benchmarks

### 6.1 Tiers

| Tier | Trigger | Where | Budget | Content |
| --- | --- | --- | --- | --- |
| 0 | every push and PR | GitHub-hosted CPU | ≤ 15 min | ruff, type check, §2.4 checks (including public vs internal), manifest, mirrors, maths IDs, all tests on CPU in float32 and float64, differentiation, the JAX trace test (§3.1). In step 2: JAX backend, and CPU build of the Rust wheels |
| 1 | PR, after tier 0 passes. **Set up after 1.0.0** (Part IV, §24.1) | HPC GPU, constrained (§6.2) | ≤ 20 min wall time, 1 GPU | `pytest -m gpu`: reference values on GPU, determinism (two runs bit-identical), GPU syntheses, benchmark smoke test. In step 2: GPU build of the Rust wheels |
| 2 | `bench` label set by humans; release candidates | dedicated, pinned GPU node, outside CI | hours | full benchmark; results committed by a bot to `bench/results/` with a hardware fingerprint |

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

Tier 1 is set up after 1.0.0. The scheduler, the container runtime and the network rules of the gateway node are confirmed then (Part IV, §24.1).

### 6.3 Benchmarks

- **Cases** (`bench/cases.toml`), public and defined together with the maintainers: one per geometry at a small and a medium size; the 2D+1D cases of the STOC deliverable at reduced sizes; JAX compile time (step 2); peak memory; synthesis time per iteration; in step 2, the Rust engine against the Python engine.
- **Thresholds.** `bench/thresholds.toml` (protected) holds the "significant change" threshold of each metric, chosen by humans. To begin with, a factor 2 in speed or in peak memory; it may be refined later.
- **Correctness.** Every benchmark run also asserts the correctness of its results, so that no gain can come from a wrong computation.

## 7. Versioning and merge policy

### 7.1 Version numbers

| Bump | When | Who decides |
| --- | --- | --- |
| **MAJOR** | a breaking interface change: a removed or renamed function or argument, a changed default; a new geometry. Whether a change of statistics values also forces a major is decided case by case with the maintainers (Part IV, §24.2) | humans (CODEOWNER approval) |
| **MINOR** | new features, including opt-in step-2 features and new engine configurations; a significant benchmark gain (§6.3), including making an engine the default for a configuration | automatic; humans set the `bench` label for benchmark-driven minors |
| **PATCH** | bug fixes that leave reference values unchanged; refactors; docs | automatic |

**Two reference counters.** Besides the package version, ESTEL publishes two integers, never reset:

- `estel.STATISTICS_REFERENCE`, bumped whenever any statistic value changes. The same number means the same statistics, within the tolerances of §5.2, whatever the package version.
- `estel.SYNTHESIS_REFERENCE`, the same for the reference synthesis trajectories, qualified by backend and pinned dependencies (§1.6).

Both are recorded in every saved result (its run manifest), shown in the release notes, and checked by CI, which computes when they must be bumped (§7.2). Bumping either needs CODEOWNER approval.

**Within one value of `STATISTICS_REFERENCE`, statistics and test results are unchanged up to the tolerances of §5.2.** This sentence opens `VERSIONING.md` and `AGENTS.md`.

How the two steps map onto versions:

- **Before 1.0.0** versions are 0.x and nothing is gated by reference values. Pull requests merge when CI passes, except on protected paths (§5.3). The three CODEOWNERS — Erwan Allys (GitHub `Eralys`), Jean-Marc Delouis (`jmdelouis`) and Pablo Richard (`pablo-richard`) — review step 1 as a whole at its end, before the freeze; the AI review is an advisory aid throughout.
- **1.0.0** is the end of step 1: parity with STL, references signed off by the three CODEOWNERS.
- **Step 2 features** land in 1.x as opt-in minors. Making one of them a default changes reference values: it bumps `STATISTICS_REFERENCE`, and whether it is also a major is decided with the maintainers. Such switches are batched rather than spread over many releases.

### 7.2 Computing the required bump

`tools/version_impact.py` runs in tier 0 and determines the **required** bump:

- reference tests outside tolerance, or a changed `tests/reference/` → `STATISTICS_REFERENCE` (or `SYNTHESIS_REFERENCE`) must be bumped, with CODEOWNER approval;
- `api/manifest.json`: an entry removed or changed → major, an entry added → minor;
- the geometry registry changed → major;
- the `bench` label → minor.

**How CI sees a bump of a reference counter.** `STATISTICS_REFERENCE` and `SYNTHESIS_REFERENCE` are integer constants in `src/estel/_reference.py`, a protected file (§5.3), exported by `estel/__init__.py`. `tools/version_impact.py` reads them on the base branch and on the PR:

- a PR that changes `tests/reference/` must increase `STATISTICS_REFERENCE` by exactly one, and a PR that changes `STATISTICS_REFERENCE` must change `tests/reference/`; before 1.0.0 a mismatch is only reported, from 1.0.0 it fails;
- `SYNTHESIS_REFERENCE` follows the same rule for the synthesis references, once they exist (after 1.0.0, Part IV, §24.1, W2). It stays 0 until then, and goes to 1 with the first synthesis references; at the 1.0.0 freeze, only `STATISTICS_REFERENCE` goes to 1;
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

The top of `AGENTS.md`:

1. Statistics never change silently. If your change alters any reference value, stop: it requires bumping `STATISTICS_REFERENCE` and a human decision. Open an issue instead.
2. Never edit protected paths (§5.3). Never weaken, skip or delete a test to make CI pass.
3. `torch` and `jax` are imported only in `src/estel/backend/`. L1 and L2 code is functional: no in-place updates, no Python control flow on array values (§3.1).
4. Every mathematical change goes through `docs/maths.md` (with an `M-` ID), `_spec`, and a reference regeneration by a human.
5. Declare the version impact in `changes/`. CI computes the real one.
6. One topic per PR, branched from `main`. Name follow-ups explicitly (rules E1, E2).
7. Public API changes start in `api/manifest.json`. Additions are minors, breaking changes majors.
8. Error messages say what to do next: agents read them.
9. Before opening a PR, run `make check` (tier 0, locally).
10. When the specification is ambiguous, ask in an issue. Do not choose.
11. **Step discipline.** Before 1.0.0, never add a capability STL does not have, except those this document lists explicitly. After 1.0.0, a new capability enters as an opt-in feature described in Part III of the transfer document — later, in the ESTEL reference document.
12. **Public or internal.** Only names listed in `api/manifest.json` are public. Anything else lives in a private module or starts with an underscore (§2.4).

### 8.3 Review rules

`docs/agents/review_rules.md` adapts the GRID4EARTH review rubric:

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
