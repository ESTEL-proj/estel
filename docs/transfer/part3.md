# Part III — Step 2: later developments

[Back to the overview](overview.md)

Nothing in this part is implemented before 1.0.0. It is documented now for two reasons: so that step 1 reserves the right interfaces (Part II, §9.4), and so that each development starts from an agreed design rather than from an agent's improvisation. Each item enters ESTEL as an opt-in feature (a minor version); making it a default bumps STATISTICS\_REFERENCE, and whether that is also a major version is decided with the maintainers (Part I, §7.1). The order of the items is not fixed.

## Contents of this part

- [16. Register of later features](#16-register-of-later-features)
  - [16.1 Library structure](#161-library-structure) · [16.2 New features for the user](#162-new-features-for-the-user) · [16.3 Optimisation of existing features](#163-optimisation-of-existing-features) · [16.4 Mathematical choices](#164-mathematical-choices) · [16.5 Entries added in the review of 12 November 2026](#165-entries-added-in-the-review-of-12-november-2026)
- [17. JAX backend](#17-jax-backend)
  - [17.1 Scope](#171-scope) · [17.2 What step 1 already guarantees](#172-what-step-1-already-guarantees) · [17.3 Points specific to JAX](#173-points-specific-to-jax) · [17.4 HEALPix on JAX](#174-healpix-on-jax) · [17.5 Exit criteria](#175-exit-criteria)
- [18. The Rust engine, from STL-Dev-MPI](#18-the-rust-engine-from-stl-dev-mpi)
  - [18.1 What STL-Dev-MPI is](#181-what-stl-dev-mpi-is) · [18.2 Integration rules](#182-integration-rules) · [18.3 MPI](#183-mpi) · [18.4 HEALPix spectra in Rust](#184-healpix-spectra-in-rust) · [18.5 Capability roadmap of the Rust engine](#185-capability-roadmap-of-the-rust-engine)
- [19. Compact coefficient layout](#19-compact-coefficient-layout)
  - [19.1 Where the predictable NaNs come from](#191-where-the-predictable-nans-come-from) · [19.2 Two representations](#192-two-representations) · [19.3 Level-wise Gram products](#193-level-wise-gram-products) · [19.4 `Layout` and `Segment`](#194-layout-and-segment) · [19.5 Full-grid views](#195-full-grid-views) · [19.6 Reductions as static linear maps](#196-reductions-as-static-linear-maps) · [19.7 What the compact layout does not solve](#197-what-the-compact-layout-does-not-solve) · [19.8 The canonical set](#198-the-canonical-set) · [19.9 Implementation choice still open](#199-implementation-choice-still-open)
- [20. New mathematics](#20-new-mathematics)
  - [20.1 Low-pass filters everywhere, including S3/S4](#201-low-pass-filters-everywhere-including-s3s4) · [20.2 Positive spectrum estimator for non-periodic data](#202-positive-spectrum-estimator-for-non-periodic-data) · [20.3 NaN masks in tensor products](#203-nan-masks-in-tensor-products) · [20.4 HEALPix blocks in tensor products](#204-healpix-blocks-in-tensor-products) · [20.5 Spectrum estimators in N dimensions](#205-spectrum-estimators-in-n-dimensions) · [20.6 Wavelet optimisation](#206-wavelet-optimisation) · [20.7 Canonical set and gradients through the reference](#207-canonical-set-and-gradients-through-the-reference) · [20.8 Exact translation invariance](#208-exact-translation-invariance) · [20.9 The partial order of tensor-product scales](#209-the-partial-order-of-tensor-product-scales) · [20.10 Joint design of the smoothing kernel and the wavelets](#2010-joint-design-of-the-smoothing-kernel-and-the-wavelets)

## 16. Register of later features

This register gathers every development planned after 1.0.0, and every feature or optimisation deferred during the reviews, so that none is forgotten. The ESTEL reference document carries it over into the repository (Part II, §15, P8). Deferred work that is not a feature, and points to discuss later, are traced in Part IV, §24.

The entries are numbered LF1 to LF32 ("later feature"). LF1 to LF22 are grouped in four categories (16.1 to 16.4); LF23 to LF32, added in the review of 12 November 2026, follow in 16.5 with their category. The "Source" column says where each entry comes from: a decision on the Rust-engine note (Q), a review point of 12 November 2026 (Part IV, §22.4), or an open point of the maths tab (OP).

### 16.1 Library structure

Backends and engines: nothing the user sees, everything the code is built on.

| ID | Feature | Why | Builds on (reserved in 1.0.0) | Changes reference values | Version impact | Details | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LF1 | JAX backend | `jit`, `vmap`, TPU and GPU, functional ecosystem | L0 protocol, functional rules | no | minor (JAX made stable; experimental since 1.0.0) | §17 | Q9, Q10 |
| LF2 | HEALPix on JAX | HEALPix on every backend | `gather_weighted`, `segment_sum`, the NumPy tables ported in step 1 | no | minor | §17.4 | C9 |
| LF3 | Rust engine (`estel-rust`, built from `rust/` in the ESTEL repository; CPU and GPU) | about 10× faster and 9× lighter in memory at nside 128 | `engine=` argument, engine protocol | no | minor; default for a configuration = benchmark-gated minor | §18 | Q14, Q20, C11, C12, C16 |
| LF4 | MPI in process | nside 1024–2048 and beyond | `partition`, `allreduce` hook | no | minor | §18.3 | Q16 |
| LF5 | Rust HEALPix anafast (`estel-sht`) | differentiable spherical-harmonic spectra on CPU and GPU | HEALPix spectrum names reserved | no (new capability) | minor | §18.4 | Q17, Q18 |

### 16.2 New features for the user

What users will be able to compute or do that they cannot today.

| ID | Feature | Why | Builds on (reserved in 1.0.0) | Changes reference values | Version impact | Details | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LF6 | Low-pass filters in every geometry and in S3/S4. At the last level the low-pass filter is applied without STL's final downsampling, which removes a size constraint (OP-34); this is a traced change of reference values. It is realised with the smoothing kernel; a dedicated low-pass kernel φ remains possible | the largest scales enter every statistic | index sets of the plan | yes (new default) | reference bump; major decided with the maintainers | §20.1, §20.10 | Q5, G9 |
| LF7 | Positive spectrum estimator for non-periodic data | today's estimator gives negative bins | estimator registry | yes, if it becomes the default | reference bump if default | §20.2 | G10, OP-21 |
| LF8 | NaN masks in tensor products | gaps in space–time data | masks as weights | no (new capability) | minor | §20.3 | — |
| LF9 | HEALPix blocks in tensor products | sphere × time, sphere × frequency | geometry hooks | no (new geometry) | major (new geometry) | §20.4 | — |
| LF10 | v1.6.0 spectrum estimators in every geometry | NaN-aware spectra for 1D and tensor products | estimator registry, LF23 | yes, for 1D and tensor-product spectra | reference bump if values change | §20.5 | C22 |
| LF11 | Improved plotting and diagnostics, `gaussianise`, surrogates | common needs of users and agents | public objects | no | minor | — | S7 (R3) |
| LF12 | Synthesis: coarse-to-fine, per-iteration filtering, ensembles | quality and cost of syntheses | `SynthesisResult` | no | minor | — | — |
| LF13 | xarray export | inspection | statistics accessors | no | minor | — | — |

### 16.3 Optimisation of existing features

The same results, faster, larger, lighter or on more configurations.

| ID | Feature | Why | Builds on (reserved in 1.0.0) | Changes reference values | Version impact | Details | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LF14 | Compact coefficient layout | output memory 4 to 26 times smaller | storage-independent accessors | no | minor | §19 | Q21 |
| LF15 | Second order in the Rust engine | implicit differentiation at Rust speed | `diff_order` | no | minor | §18.2 | Q13, C23 |
| LF16 | float32 in the Rust engine | speed for specific cases | capability declaration | no | minor | §18.2 | Q15 |
| LF17 | Tensor products in the Rust engine | 2D+1D at Rust speed | capability declaration | no | minor | §18.5 | Q22 |
| LF18 | Non-periodic boundaries in the Rust engine | regional data | capability declaration | no | minor | §18.5 | — |
| LF19 | Memory beyond one device for tensor products | the 2D+1D+1D sizes of the STOC deliverable | loop organised by levels | no | minor | §19.7 | — |

### 16.4 Mathematical choices

Decisions that change what existing statistics compute.

| ID | Feature | Why | Builds on (reserved in 1.0.0) | Changes reference values | Version impact | Details | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LF20 | Canonical coefficient set | no duplicate coefficients | compact layout | yes | reference bump; major decided with the maintainers | §19.8 | Q1, G2, OP-14 |
| LF21 | Gradient semantics through the reference | implicit differentiation with targets that depend on parameters | `ref` argument | yes, if the default changes | reference bump if default | §20.7 | Q2, G6, OP-18 |
| LF22 | Wavelet optimisation | better filters, satisfying a Littlewood–Paley condition, low-pass filter included (LF6), tested in CI | `WaveletOperator` | yes | reference bump; major decided with the maintainers | §20.6, §20.10 | — |

### 16.5 Entries added in the review of 12 November 2026

They continue the numbering; their category is given in the second column.

| ID | Category | Feature | Why | Builds on (reserved in 1.0.0) | Changes reference values | Details | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LF23 | 16.2 New feature | 1D and tensor-product spectra, as in STL `2D+1D`: binned spectra, joint and low-pass bins | spectra for every planar geometry | `method=` names reserved | no (new capability) | Part II, §9.2 | C22 |
| LF24 | 16.2 New feature | HEALPix spectra (`gaussian_rings`, `alm`, `anafast`), through the Rust anafast (LF5) | spectra on the sphere | `method=` names reserved | no (new capability) | §18.4 | C22, C9 |
| LF25 | 16.4 Mathematical choice | Exact translation invariance: remove the grid locking between the centring of the downsampling kernel and the stride, without breaking Littlewood–Paley | a property users expect, and a strong test | `smoothing_kernel`, downsampling | yes | §20.8, §20.10 | C19, OP-2 |
| LF26 | 16.4 Mathematical choice | Reductions on tensor products (`iso`, `angular_ft`, `scale_ft`) redefined: the product orientation mixes rotations with the direction of propagation | meaningful reduced statistics for space–time data | reductions | yes | Part II, §13.2, G4 | C8, OP-5 |
| LF27 | 16.4 Mathematical choice | Relevance of the partial order of tensor-product scales: should some incomparable pairs of scales be measured? | interactions between fine-in-space, coarse-in-time and coarse-in-space, fine-in-time components are never measured | `_scale_index_set` | yes, if the set of scales changes | §20.9 | A12 |
| LF28 | 16.4 Mathematical choice | A better regularisation of the cross S1 terms than `ε = 1e-8` | the constant is arbitrary and scale-dependent | M-S1 | yes | Part II, §13.2, G17 | G17, OP-16 |
| LF29 | 16.3 Optimisation | Small kernels baked in as compiled constants | possible speed-up under compilation | static plan | no | Part I, §3.1, rule 4 | review of 12 November 2026 (Pablo) |
| LF30 | 16.3 Optimisation | Complex-to-complex convolutions instead of pairs of real ones | possible speed-up in tensor products | `conv_axes` | no | Part I, §3.1, rule 6 | review of 12 November 2026 (Pablo) |
| LF31 | 16.3 Optimisation | Second-layer means computed by contraction over pixels, instead of building the full product array first | lower memory peak, same loop | `_second_layer` | no | §19.3 | C18 |
| LF32 | 16.4 Mathematical choice | Orientation reductions (iso, angular\_ft) made consistent with the conjugation of complex coefficients across π, in 2D | the cyclic average over \[0, π) mixes some S3 and S4 values with what should be their conjugates | reductions | yes | maths tab, M-RED-ISO | OP-5 |

## 17. JAX backend

### 17.1 Scope

The whole Python engine — L1, L2, synthesis, utilities — runs on JAX, with `jax.jit`, `jax.vmap` and differentiation to second order, on CPU, GPU and TPU. Users choose it with `estel.set_backend("jax")`; nothing else in their code changes.

### 17.2 What step 1 already guarantees

- The functional rules of Part I, §3.1, checked by AST rules and by running the whole test suite on the experimental JAX backend from phase P1.
- Static plans built in NumPy (Part I, §2.2.2).
- DataClasses and statistics registered as pytrees.
- No backend import outside L0 (Part I, §2.4).

The JAX backend is therefore the implementation of the L0 protocol in `estel/backend/jax/`, plus its optimizer, with no change elsewhere. Any change it would require in L1 or L2 reveals a breach of the step-1 rules and is fixed as a bug.

### 17.3 Points specific to JAX

- **float64.** `jax_enable_x64` is switched on by `set_default_dtype("float64")`, before the first JAX computation.
- **Compile time.** Loops over levels are unrolled when JAX traces the code. Compile time is measured for every benchmark case, with a budget; the tensor products with many levels are the cases to watch.
- **Optimizer.** `optax.lbfgs` with `optax.scale_by_zoom_linesearch`. Trajectories differ from PyTorch's, so synthesis references are kept per backend (Q6), with cross-backend statistical criteria.
- **Determinism.** `XLA_FLAGS=--xla_gpu_deterministic_ops=true` in tests.
- **Tolerances.** A PyTorch-vs-JAX row in `tests/tolerances.toml` (Part I, §5.2).

### 17.4 HEALPix on JAX

In step 1, HEALPix already runs on index and weight tables built in NumPy — stencils and decimation tables, a port of `TorchGeometry` — applied with `gather_weighted` and `segment_sum` (Part II, §9.3); healpix-analyse is only a test dependency. HEALPix on JAX therefore only needs the JAX implementation of these two primitives (LF2). HEALPix spectra come with the Rust anafast (§18.4, LF24).

### 17.5 Exit criteria

- The conformance suite passes on JAX, in float32 and float64.
- Every reference test passes on JAX within its tolerances.
- The differentiation tests pass in both modes, to second order.
- Compile times are within their benchmark budgets.

## 18. The Rust engine, from STL-Dev-MPI

This section condenses the note *ESTEL: proposal for integrating the Rust engines of STL-Dev-MPI*, whose rules are numbered RE1–RE14, the decisions taken on it (Q10–Q23), and the decisions of 12 November 2026 (Part IV, §22.4): the Rust code moves into the ESTEL repository, and it is called "rust", not "native".

### 18.1 What STL-Dev-MPI is

STL-Dev-MPI is the Rust, MPI and CUDA port of the block-wise pyramid engine of `HealpixUpdate`. It computes mean, var and S1–S4, and their exact gradient through hand-written VJPs, block by block, so that memory is bounded by a block or by one rank's share of the sky. It matches STL-Dev to 1e-10, and 1, 2 and 4 ranks agree to 1e-15. It has about 31,000 lines of Rust, under the MIT licence.

| Crate in STL-Dev-MPI | Content | In ESTEL |
| --- | --- | --- |
| `stl-core` | HEALPix geometry (port of `TorchGeometry`), convolution and adjoint, block-wise forward and reverse passes, planar periodic engine, masks, SHT spectra with VJP, L-BFGS with strong Wolfe line search. Pure Rust, float64 | yes: `rust/estel-core` |
| `stl-gpu` | CUDA versions of the same engines, with cuFFT and cuBLAS loaded at run time | yes: `rust/estel-gpu` |
| `stl-mpi` | Domain decomposition, halo exchange, distributed synthesis | yes: `rust/estel-mpi`, binaries first, then in process |
| `stl-bridge` | The C ABI (`stlp_forward`, `stlp_vjp`, `stlp_checkpoint`, `stlp_restore`, `stlp_lbfgs`, spectrum engines, device-pointer variants) | replaced by a PyO3 binding, `rust/estel-py` |
| `stl-cli`, ensemble driver, diffusion model, paper | Applications | no |

Reported by its README at nside 128, 4 members, one A100: STL-Dev autograd takes 1.229 s per evaluation and 19.8 GB; STL-Dev-MPI takes 0.106 s and 2.2 GB, or 0.885 s and 0.58 GB in its low-memory mode.

### 18.2 Integration rules

- **One repository (RE1, revised).** The Rust crates move into the ESTEL repository, under `rust/`, as one Cargo workspace (Part II, §11). maturin builds two wheels from `packages/`: `estel-rust`, optional (`pip install estel[rust]`), and `estel-sht`, required (§18.4). Both are released with `estel`, with the same version number, so they can never drift apart. A PyO3 binding (`rust/estel-py`) replaces the versioned C ABI of `stl-bridge`. ESTEL imports `estel_rust` only under `src/estel/engines/`, and installs and passes every test without it.
- **Engine, not backend.** The Rust code replaces the whole computation behind the Python frontend; it is not split into L0 primitives, which would lose its block-wise algorithm, its hand-written gradients and MPI (Part I, §2.2.4).
- **Capabilities (RE2, Q15, Q19).** The engine declares geometries, L, kernel sizes, gauges, dtypes, masks, cross channels, devices and differentiation order. It computes in float64 by default, even when ESTEL's default is float32, unless float32 is requested explicitly; `describe()` shows it, so this is never a silent cast.
- **Differentiation (RE3, Q13, Q21).** Today `diff_order=1`. Second order is written by hand in Rust later (LF15) — a JVP of the forward pass and a JVP of the VJP — and then passes the tests of Part I, §3.2.2 against the Python engine in float64.
- **Purity (RE4).** The forward pass returns the values and an opaque residual (a checkpoint for HEALPix, the pyramid for 2D); the VJP takes it back. Handles are cached per plan and device, guarded by a lock.
- **Plan mapping (RE5, Q12).** The engine's coefficient order, exported as JSON, is mapped once onto ESTEL's statistics by a static gather. The `has_fewer_convolutions` variant is not carried over, so the engine computes only coefficients that ESTEL defines.
- **Same numbers (RE6).** The engine passes the reference tests of every configuration it claims, at float64 tolerances, plus a VJP parity test against the Python engine (relative 1e-10). It never produces reference values; its own fixtures are an independent check.
- **Devices and memory (RE7).** Zero-copy float64 NumPy views on CPU; CUDA device pointers of PyTorch tensors on GPU; JAX through host memory until an XLA FFI binding exists. `estimate_memory()` asks the engine for its peak.
- **Determinism (RE8).** Bit-identical results for a fixed thread count, block size and device; its own row in the tolerances.
- **Optimizer (RE9).** The Rust L-BFGS is exposed through `lbfgs(..., impl="rust")`, tested against `torch.optim.LBFGS` on a fixed problem, and is the default synthesis optimizer when the engine is selected.
- **Version impact (RE11).** Adding the engine or a configuration is a minor; making it the default for a configuration is a minor gated by a tier-2 benchmark. The intended end state: the engine is the default wherever it covers a configuration and wins the benchmark, and the Python engine stays the readable implementation, the second-order path and the fallback. An engine change that moves values outside tolerance is a bug in the engine, never a reason to regenerate references.
- **CI (RE12).** Tier 0 builds the CPU wheel and runs the reference and VJP tests with `engine="rust"`; tier 1, once set up, runs the GPU build. Review rules A5, B2, C6, D8 and E4 apply in `rust/`, whose local `AGENTS.md` holds them. The binding and the capability schema are protected paths.
- **Growth (RE13).** Every new capability follows three steps: the mathematics first (`maths.md`, `_spec`, the Python engine); then the engine declares the capability and passes its parity tests; then it ships as a minor, and becomes a default once the benchmark shows the gain. A change of mathematics updates the oracle, the Python engine and the Rust engine in the same pull request; the Rust code follows the PyTorch version, never the reverse (Q11).

### 18.3 MPI

Distributed memory is one reason the engine was written in Rust. Each rank holds a contiguous NESTED range of the sky and the matching slice of the pyramid. Statistics are global — one `Allreduce` per level — and identical on every rank. Scaling of one evaluation (statistics and gradient), from STL-Dev-MPI `text/figures/scaling.txt`:

| Case | Ranks | Time per evaluation | Parallel efficiency |
| --- | --- | --- | --- |
| nside 1024, L = 2, Irene (EPYC 7H12) | 128 → 512 cores | 30.2 s → 8.9 s | 0.85 |
| nside 1024, L = 2, one EPYC 9655 node | 24 → 192 cores | 60.9 s → 10.2 s | 0.75 |
| nside 2048, L = 4, Tesla T4 GPUs | 2 → 3 GPUs | 474 s → 114 s | superlinear: the sky fits in device memory |

ESTEL integrates MPI in two stages (Q16):

1. **Out of process (1.x).** Distributed runs use the STL-Dev-MPI binaries under `mpirun` or Slurm. ESTEL reads and writes their plan, statistics and map formats through `estel.io`, so that a target can be prepared and a result checked in ESTEL.
2. **In process.** ESTEL runs under `mpirun` with mpi4py, an optional dependency. Each rank holds a `DataHealpix` slice (`partition=`, the STL-Dev-MPI partition: contiguous NESTED ranges, coarsest levels replicated). `scattering()` returns the global statistics on every rank, and the VJP the rank's slice of the gradient. Because 1.0.0 reserves `partition` and the `allreduce` hook, this stage needs no API change. A test checks that 1, 2 and 4 ranks agree within 1e-13.

### 18.4 HEALPix spectra in Rust

The engine's `SpectrumEngine` is a differentiable anafast: auto, cross and spin-s spectra (EE, BB, EB), with masks, forward and VJP, on CPU and GPU (spin 0 on the device). It matches `healpy.anafast(..., iter=0)` and `healpy.map2alm_spin`, which are its tests.

- **It becomes the HEALPix spectrum method of ESTEL (Q17, LF24)**, hidden behind `estel.spectra`: users call the ESTEL API, on PyTorch or JAX arrays, and never see Rust.
- **A separate, required wheel (Q18).** The anafast code is independent of the scattering engine, so it lives in its own crate, `rust/estel-sht`, and its own wheel, `estel-sht`, which ESTEL requires once HEALPix spectra exist. The scattering engine `estel-rust` stays optional. Binary wheels are built in tier 0 for Linux (x86\_64, aarch64), macOS (arm64) and Windows.
- **No HEALPix spectra in 1.0.0.** 1.0.0 has spectra for 2D data only (Part II, §9.3). HEALPix spectra arrive with LF24, compared with STL `HealpixUpdate`'s spectra (computed through healpix-analyse, pinned at `aac5e26`) on the reference inputs.

Measured on one map, nside 256, `lmax = 511` (STL-Dev-MPI `docs/spectrum.md` §5):

| Setting | healpy forward | Rust forward | Rust forward + VJP |
| --- | --- | --- | --- |
| Spin 0, 20 CPU threads | 0.026 s | 0.022 s | 0.038 s |
| Spin 2, 20 CPU threads | 0.043 s | 0.030 s | 0.077 s |
| Spin 0, 1 CPU thread | 0.036 s | 0.073 s | 0.143 s |
| Spin 0, 5 masks, Tesla T4 | — | 0.099 s | 0.191 s |

### 18.5 Capability roadmap of the Rust engine

| Capability | Rust engine today | Status |
| --- | --- | --- |
| HEALPix full sky, masks as weights, cross channels | yes | done |
| Planar 2D, `pbc=True` | yes | done |
| Planar 2D, `pbc=False` (border crop) | no | next, in development (LF18) |
| Low-pass filters in S3/S4 (§20.1) | no | to implement in Rust, in sync with PyTorch (Q11, LF6) |
| Tensor products (2D+1D, HEALPix blocks) | no | planned (Q22, LF17) |
| float32 | no | opt-in, to implement (Q15, LF16) |
| Second order | no | by hand, later (Q13, LF15) |
| MPI, distributed sky (HEALPix) | through the binaries | in process from ESTEL: next (LF4) |
| HEALPix anafast: auto, cross, spin s, masks, VJP | yes (GPU for spin 0) | exposed through `estel.spectra` (LF5, LF24); spin s on GPU later |

A generated capability matrix, `docs/engines.md`, built from the engine's declaration and the Python engine's registry, shows for each configuration which paths exist and which is the default.

## 19. Compact coefficient layout

Implemented after 1.0.0 is stable. In 1.0.0 the statistics keep STL's full-grid storage behind accessors that do not depend on it (Part II, §12.7), so switching to the compact layout changes no public signature and no value (LF14).

The principle: **compute on structured arrays, store in a flat compact buffer, and never mix the two.** The conversion between them is a static gather whose indices are built once, in NumPy, with the plan.

### 19.1 Where the predictable NaNs come from

STL allocates S4 as `(Nb, Nc, Nc, *J, *J, *J, *L, *L, *L)`, complex and filled with NaN. Four things make most of that space useless:

1. **Scale order.** Only `j1 ≤ j2 ≤ j3` is filled: about 1/6 of `J³` per block.
2. **Partial order on multi-indices.** In tensor products the order holds componentwise, so the fill ratio is the *product* over blocks.
3. **Low-pass entries have no orientation.** Their `L − 1` other orientation entries stay NaN.
4. **Everything is stored complex**, even the families that are real by construction: mean, var, S1, auto-S2, the diagonal of S4.

Per batch element and channel, in float32, for the sizes of Table 1 of the STOC deliverable:

| Data | Allocated entries | Useful coefficients | Full-grid storage | Compact storage | Ratio |
| --- | --- | --- | --- | --- | --- |
| 2D, J = 4, L = 4 | 4,394 | 1,476 | 35 kB | 9 kB | 3.9× |
| 2D+1D space–depth, J = (4, 2), L = (4, 2) | 266,482 | 43,052 | 2.1 MB | 0.3 MB | 7.6× |
| 2D+1D space–time, J = (4, 6), L = (4, 2) | 7,115,314 | 587,324 | 57 MB | 4.3 MB | 13× |
| 2D+1D+1D, J = (4, 6, 2), L = (4, 2, 2) | 453,578,018 | 18,513,572 | 3.6 GB | 138 MB | 26× |

These counts include the low-pass filters in S1/S2 only. Almost all of the saving comes from dropping the NaN positions; storing real entries as one float and removing conjugate duplicates add about 7% more.

### 19.2 Two representations

|  | Compute representation | Storage representation |
| --- | --- | --- |
| Used by | convolution, downsampling, mask propagation, Gram products | loss, normalisation, reductions, I/O, plots |
| Shape | per level: `(B, n_slots, *spatial_at_level)`, structured and static | one array `(B, n_floats)`, real dtype |
| Lifetime | inside `scattering()` only | `STStatistics` |
| NaN | never: masks are weight arrays | never: absent entries are not stored |

The storage buffer is **real**: complex entries are stored as consecutive (re, im) pairs, real entries as one float. One real buffer makes the loss, the normalisation and the reductions plain elementwise or sparse-linear operations on every backend, and keeps the optimizers working on real vectors.

### 19.3 Level-wise Gram products

**What a Gram matrix is.** Given a set of vectors v\_1 … v\_n, their Gram matrix collects all pairwise inner products: G\_st = ⟨v\_s, v\_t⟩. A covariance matrix is the Gram matrix of centred vectors. Here each "vector" is one filtered field at a given resolution level — one *slot*: a channel, a scale, an orientation — and the inner product is the mean over valid pixels of v\_s · conj(v\_t). Computing a whole Gram matrix at a level is a single matrix product, which replaces thousands of separate covariance calls.

The loop is organised by **level** (the output resolution), not by coefficient. At each level it stacks a few structured arrays and takes one batched Gram product per family:

| Family at level `j` | Stack, at the resolution of level `j` | Product |
| --- | --- | --- |
| S1 | `\|W_{j,l} x_c\|` | weighted mean over pixels |
| S2 | `V = W_{j,l} x_c`, slots `(c, l)` | `G = ⟨V_s · conj(V_t)⟩`, Hermitian over channels |
| S3 (j2 = j) | `V`, and `U = W_{j,l2} \|W_{j1,l1} x_c\|` for every slot `(c, j1, l1)` with `j1 ≤ j` | cross Gram `⟨V · conj(U)⟩`, sharing `l2` |
| S4 (j3 = j) | `U = W_{j,l3} \|W_{j1,l1} x_c\|` for every slot `(c, j1, l1)` with `j1 ≤ j` | `G = ⟨U_s · conj(U_t)⟩` per `l3`, Hermitian in `(s, t)` |

- **Low-pass filters are slots.** φ is a pseudo-scale with one orientation, so low-pass terms in any statistic add rows to the slot tables and need no separate code path.
- **Tensor products change only the slot tables and the level list.**
- **One weight map per level** folds mask propagation and border crop together.

A static gather then keeps the canonical entries and writes them into their segment of the buffer; the segments are assembled with one `concat`. There is no preallocation and no in-place write.

This gives a few large products instead of many small ones, which suits GPUs and keeps the JAX trace short: 48 levels for 2D+1D+1D, against about 10⁵ coefficient triples. The full Gram also computes the lower triangle and the incomparable pairs, up to about twice the necessary products; grouping slots by `j1` into block-triangular products is a later optimisation, to be justified by profiling (rule B3).

**Memory first, then reorganisation (Part IV, §22.4, C18).** Step 1 keeps STL's loop. In step 2, the first change is to compute each second-layer mean by a contraction over pixels (`einsum` or a matrix product) instead of building the full product array first: this lowers the memory peak without changing the loop (LF31). Stacking the slots of a level into one Gram product, as described above, needs a concatenation that copies the stack; it is adopted only if a realistic benchmark shows it pays, after a human validation.

### 19.4 `Layout` and `Segment`

```python
@dataclass(frozen=True)
class Segment:
    family: str                 # "mean" | "var" | "S1" | "S2" | "S3" | "S4" | "PS"
    level: tuple[int, ...]      # output level (multi-index in tensor products)
    index_names: tuple[str, ...]        # e.g. ("c1", "c2", "j1", "l1", "j2", "l2", "l3")
    index: np.ndarray           # (n_entries, len(index_names)) int32, static
    is_complex: np.ndarray      # (n_entries,) bool, static
    offset: int                 # first float in the buffer
    n_floats: int

@dataclass(frozen=True)
class Layout:                   # hashable: usable as a JAX static argument
    segments: tuple[Segment, ...]
    n_floats: int
    schema_version: int
    def full_grid_shape(self, family): ...  # STL-compatible shape
    def gather_tables(self, level): ...      # Gram → segment indices (static)
```

Every stored entry carries its full named coordinates, so selection, plotting and full-grid views never need the full-grid tensor itself.

### 19.5 Full-grid views

`stats.family("S4").full_grid()` builds, with scatter\_to\_grid, the tensor on the complete grid of indices, with NaN at the positions where no coefficient exists and, on request, the conjugate duplicates by conjugation. It serves humans, plots and comparisons with 1.0.x results. It is never called inside `scattering()`, and a test asserts it.

### 19.6 Reductions as static linear maps

Normalisation (for a fixed reference), `iso`, `angular_ft` and `scale_ft` are linear in the coefficients. Each is precomputed in NumPy as an (index, weight) table and applied with `segment_sum` or a small matrix product per segment, returning a new layout. They are therefore differentiable, compilable and backend-neutral.

### 19.7 What the compact layout does not solve

The layout removes the waste in the *output*. For large tensor products the limit is the *intermediate* stacks: at the 2D+1D+1D size of Table 1 (128 × 128 × 512 × 32 ≈ 2.7·10⁸ pixels), the first layer alone, with 16 complex orientations, takes about 34 GB in float32. `estimate_memory()` reports the peak beforehand and a clear error is raised beyond a budget. The level-organised loop is the natural unit for streaming later (LF19), and the Rust engine already bounds memory by block for HEALPix and 2D (§18).

### 19.8 The canonical set

STL stores, at `j1 = j2`, both `(l1, l2)` and `(l2, l1)`, which are complex conjugates, and both channel orders `(a, b)` and `(b, a)`. A Hermitian-packed layout stores each pair once. Since the loss sums squares over stored coefficients, the duplicates currently count twice. Two options:

- (a) keep STL's set as canonical, with packing as a pure storage choice and duplicates weighted twice in the loss;
- (b) make the non-redundant set canonical.

Option (a) keeps the losses of 1.0.0; option (b) is cleaner and changes reference values (a major). The decision is taken after 1.0.0 (LF20). Whatever is chosen, the gather of the Rust engine (§18.2) must produce it.

### 19.9 Implementation choice still open

Whether the conversion between the two representations is written once, in shared Python on a few generic primitives, or separately in each backend, is open (Part IV, §23, R1).

## 20. New mathematics

Each item below changes what ESTEL computes. Each one is written in `maths.md` and `_spec` first, lands as an opt-in option, and becomes a default through a bump of STATISTICS\_REFERENCE, decided with the maintainers (Part I, §7.1).

### 20.1 Low-pass filters everywhere, including S3/S4

- **Decision.** The low-pass filter φ is used for all coefficients, including S3 and S4, and in every geometry — not only in tensor products, as in STL.
- **Two possible extensions to S3/S4**, from §5 of the STOC deliverable:
  - *restricted*: the low-passed field enters S3/S4 only as one extra scale, adding 0.3–3.5% coefficients;
  - *all combinations*: φ may replace any of the wavelets ψ\_j1, ψ\_j2, ψ\_j3, in each block independently, adding 38–112%.
- **Why "all combinations" grows so much.** Each block gains one scale (J becomes J + 1), and S4 counts triples of scales: for a block with J = 2 the number of triples goes from 4 to 10. These factors multiply across blocks, so blocks with few scales dominate the increase. The single orientation of φ tempers it somewhat.
- **"All coefficients" corresponds to "all combinations".** This is to be confirmed with the counts above in mind (Part IV, §23, R2).
- **Consequences.** A new default changes reference values, so it bumps STATISTICS\_REFERENCE. The Rust engine implements the same terms in sync with PyTorch (Q11). In 2D the coefficient count goes from 1,474 to 1,476 for the S1/S2 terms alone (Part II, §13, G1).

### 20.2 Positive spectrum estimator for non-periodic data

The current estimator for non-periodic data can return negative bins (Part II, §13, G10). A tapered periodogram is positive by construction. To be fixed in `maths.md`: the taper, its parameter, and the normalisation of the bins.

### 20.3 NaN masks in tensor products

First for separable masks, then for general ones. The choice to make: the product of per-block valid weights, or the observed-sample mask with per-block border crops. They differ when the mask varies along another block.

### 20.4 HEALPix blocks in tensor products

The HEALPix geometry supplies the same hooks as a planar block (`_conv`, `_downsample`, `_crop_weights`) along its pixel axis. Its gauge orientations enter the product orientation index like 2D orientations do. A new geometry, hence a major version.

### 20.5 Spectrum estimators in N dimensions

The four v1.6.0 estimators and their automatic selection, generalised from the last two axes of 2D data to the axes of any block. This also removes the class-name check that today rejects 1D and tensor-product data.

### 20.6 Wavelet optimisation

The wavelet bank will be optimised later, rather than adding new families (such as Morlet). The criteria — localisation, Littlewood–Paley flatness, kernel size, cost — are to be agreed, and any change of bank is a major version.

### 20.7 Canonical set and gradients through the reference

Both are decided after 1.0.0: the canonical set (§19.8), and whether the default normalisation lets gradients flow through the reference statistics. The latter matters for implicit differentiation when the target itself depends on the parameters.

### 20.8 Exact translation invariance

STL's downsampling uses a 6×6 anti-aliasing kernel with asymmetric padding (2 pixels before, 3 after) and a stride of 2 (maths tab, M-DOWN, OP-2). The kernel's centre and the sampling grid are probably locked, so a shift of the field by one pixel does not shift the statistics' inputs by half a pixel at the next level, and exact translation invariance is lost. 1.0.0 reproduces STL and does not test it (Part I, §5.1). LF25 removes the grid locking — a centred kernel, or a polyphase scheme — provided the Littlewood–Paley property and the other constraints still hold, and then adds translation invariance to the invariance tests.

### 20.9 The partial order of tensor-product scales

STL orders the multi-index scales of a tensor product componentwise, so pairs such as (0, 2) and (1, 1) — finer in space, coarser in time — are not comparable and give no S3 or S4 coefficient (Part II, §13.2, G3). LF27 examines whether these interactions carry useful information on the STOC test cases. If they do, it defines which incomparable pairs to add, with their cost in coefficients, as a change of reference values.

### 20.10 Joint design of the smoothing kernel and the wavelets

LF6, LF22 and LF25 are one design problem: the pair (s, ψ). With φ\_J realised as the cascade of the smoothing kernels, the Littlewood–Paley sum Λ equals 1 everywhere as soon as the per-level energy partition P does (maths tab, "Filters and scales"); a dedicated low-pass kernel only acts on the last octave and cannot make up for losses between scales. The design requirements:

1. **Energy partition.** P = 1 at every level, within a tolerance set when the design is made, with both signs counted and an explicit gain convention (OP-33).
2. **Bounded aliasing.** At each downsampling, the energy that the smoothing kernel lets through above half the Nyquist frequency, which folds back, stays below a set fraction. Requirements 1 and 2 pull in opposite directions; they are compatible only if the wavelets capture the whole upper half of the band.
3. **First octave and corners.** A decision for the frequencies that no wavelet covers: the first octave and, in 2D, the corners of the frequency square. Either a residual high-frequency band, or a deficit that is documented.
4. **Centred kernels,** so that the downsampling grid does not lock (LF25, OP-2).

At the last level, the low-pass filter is applied without the final downsampling (LF6). Measured on STL's kernels (OP-32), the 1D pair already nearly meets requirement 1; the 2D pair is anisotropic and leaves the corners uncovered. The three diagnostics of Part II, §12.10 (energy partition, Littlewood–Paley sum, aliasing) are the measurements on which these requirements will be checked.
