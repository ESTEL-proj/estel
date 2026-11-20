# Part III — Step 2: later developments

[Back to the overview](overview.md)

Nothing in this part is implemented before 1.0.0. It is documented now for two reasons: so that step 1 reserves the right interfaces (Part II, §9.4), and so that each development starts from an agreed design rather than from an agent's improvisation. An item that changes no reference value enters ESTEL as a minor version, when it is ready. The corrections of step 2a land behind private flags, off by default, and the release that ends step 2a, a 1.x release, switches them on together and bumps STATISTICS\_REFERENCE once; step 2b ends with ESTEL 2.0.0. Each of these releases drops the previous defaults at once (§19.13; Part I, §7.1). The phases of step 2a order its items (§16); the order of the others is not fixed.

## Contents of this part

- [16. Register of later features](#16-register-of-later-features)
- [17. JAX backend](#17-jax-backend)
  - [17.1 Scope](#171-scope) · [17.2 What step 1 already guarantees](#172-what-step-1-already-guarantees) · [17.3 Points specific to JAX](#173-points-specific-to-jax) · [17.4 HEALPix on JAX](#174-healpix-on-jax) · [17.5 Exit criteria](#175-exit-criteria)
- [18. The Rust engine, from STL-Dev-MPI](#18-the-rust-engine-from-stl-dev-mpi)
  - [18.1 What STL-Dev-MPI is](#181-what-stl-dev-mpi-is) · [18.2 Integration rules](#182-integration-rules) · [18.3 MPI](#183-mpi) · [18.4 HEALPix spectra in Rust](#184-healpix-spectra-in-rust) · [18.5 Capability roadmap of the Rust engine](#185-capability-roadmap-of-the-rust-engine) · [18.6 The scope of the Rust engine, re-examined](#186-the-scope-of-the-rust-engine-re-examined)
- [19. Compact coefficient layout](#19-compact-coefficient-layout)
  - [19.1 Where the predictable NaNs come from](#191-where-the-predictable-nans-come-from) · [19.2 Two representations](#192-two-representations) · [19.3 Level-wise Gram products](#193-level-wise-gram-products) · [19.4 `Layout` and `Segment`](#194-layout-and-segment) · [19.5 Full-grid views](#195-full-grid-views) · [19.6 Reductions as static linear maps](#196-reductions-as-static-linear-maps) · [19.7 What the compact layout does not solve](#197-what-the-compact-layout-does-not-solve) · [19.8 The canonical set](#198-the-canonical-set) · [19.9 Where the conversion is written](#199-where-the-conversion-is-written) · [19.10 Where time and memory go](#1910-where-time-and-memory-go) · [19.11 Optimisations, by leverage](#1911-optimisations-by-leverage) · [19.12 What to put in place before 1.0.0, so that nothing is closed](#1912-what-to-put-in-place-before-100-so-that-nothing-is-closed) · [19.13 Steps 2a, 2b and 3](#1913-steps-2a-2b-and-3) · [19.14 Results of the optimisations of step 1](#1914-results-of-the-optimisations-of-step-1)
- [20. New mathematics](#20-new-mathematics)
  - [20.1 Low-pass filters everywhere, including S3/S4](#201-low-pass-filters-everywhere-including-s3s4) · [20.2 Positive spectrum estimator for non-periodic data](#202-positive-spectrum-estimator-for-non-periodic-data) · [20.3 NaN masks in tensor products](#203-nan-masks-in-tensor-products) · [20.4 HEALPix blocks in tensor products](#204-healpix-blocks-in-tensor-products) · [20.5 Spectrum estimators in N dimensions](#205-spectrum-estimators-in-n-dimensions) · [20.6 Wavelet optimisation](#206-wavelet-optimisation) · [20.7 Canonical set and gradients through the reference](#207-canonical-set-and-gradients-through-the-reference) · [20.8 Exact translation invariance](#208-exact-translation-invariance) · [20.9 The partial order of tensor-product scales](#209-the-partial-order-of-tensor-product-scales) · [20.10 Joint design of the smoothing kernel and the wavelets](#2010-joint-design-of-the-smoothing-kernel-and-the-wavelets) · [20.11 Centred covariances](#2011-centred-covariances)

## 16. Register of later features

The register gathers everything still to implement after step 1. Since 20 November 2026 it lives in the repository, in [`docs/register/`](../register/README.md): one card per entry (`docs/register/<ID>.md`), with its step, its phase, its status and what it still has to decide; a view generated from the cards (`docs/register/README.md`, by `tools/register.py`), which also gives the rules of the register; and the decision log of step 2 ([`docs/register/decisions.md`](../register/decisions.md)). The entries keep their identifiers, LFn and Wn, and their text.

## 17. JAX backend

### 17.1 Scope

The whole Python engine — L1, L2, synthesis, utilities — runs on JAX, with `jax.jit`, `jax.vmap` and differentiation to second order, on CPU, GPU and TPU. Users choose it with `estel.set_backend("jax")`; nothing else in their code changes.

### 17.2 What step 1 already guarantees

- The functional rules of Part I, §3.1, checked by AST rules and by running the whole test suite on the experimental JAX backend from phase P1.
- Static plans built in NumPy (Part I, §2.2.2).
- Data objects and statistics registered as pytrees.
- No backend import outside L0 (Part I, §2.4).

The JAX backend is therefore the implementation of the L0 protocol in `estel/backend/jax/`, plus its optimizer, with no change elsewhere. Any change it would require in L1 or L2 reveals a breach of the step-1 rules and is fixed as a bug.

### 17.3 Points specific to JAX

- **float64.** `jax_enable_x64` is switched on by `set_default_dtype("float64")`, before the first JAX computation.
- **Compile time.** Loops over levels are unrolled when JAX traces the code. Compile time is measured for every benchmark case, with a budget; the tensor products with many levels are the cases to watch.
- **Optimizer.** `optax.lbfgs` with `optax.scale_by_zoom_linesearch`. Trajectories differ from PyTorch's, so synthesis references are kept per backend, with cross-backend statistical criteria.
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

This section condenses the note *ESTEL: proposal for integrating the Rust engines of STL-Dev-MPI*, the decisions taken on it, and the decisions of 12 November 2026 (Part IV, §22): the Rust code moves into the ESTEL repository, and it is called "rust", not "native".

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

- **One repository (revised).** The Rust crates move into the ESTEL repository, under `rust/`, as one Cargo workspace (Part II, §11). maturin builds two wheels from `packages/`: `estel-rust`, optional (`pip install estel[rust]`), and `estel-sht`, required (§18.4). Both are released with `estel`, with the same version number, so they can never drift apart. A PyO3 binding (`rust/estel-py`) replaces the versioned C ABI of `stl-bridge`. ESTEL imports `estel_rust` only under `src/estel/engines/`, and installs and passes every test without it.
- **Engine, not backend.** The Rust code replaces the whole computation behind the Python frontend; it is not split into L0 primitives, which would lose its block-wise algorithm, its hand-written gradients and MPI (Part I, §2.2.4).
- **Capabilities.** The engine declares geometries, L, kernel sizes, gauges, dtypes, masks, cross channels, devices and differentiation order. It computes in float64 by default, even when ESTEL's default is float32, unless float32 is requested explicitly; `describe()` shows it, so this is never a silent cast.
- **Differentiation.** Today `diff_order=1`. Second order is written by hand in Rust later (LF15) — a JVP of the forward pass and a JVP of the VJP — and then passes the tests of Part I, §3.2.2 against the Python engine in float64.
- **Purity.** The forward pass returns the values and an opaque residual (a checkpoint for HEALPix, the pyramid for 2D); the VJP takes it back. Handles are cached per plan and device, guarded by a lock.
- **Plan mapping.** The engine's coefficient order, exported as JSON, is mapped once onto ESTEL's statistics by a static gather. The `has_fewer_convolutions` variant is not carried over, so the engine computes only coefficients that ESTEL defines.
- **Same numbers.** The engine passes the reference tests of every configuration it claims, at float64 tolerances, plus a VJP parity test against the Python engine (relative 1e-10). It never produces reference values; its own fixtures are an independent check.
- **Devices and memory.** Zero-copy float64 NumPy views on CPU; CUDA device pointers of PyTorch tensors on GPU; JAX through host memory until an XLA FFI binding exists. `estimate_memory()` asks the engine for its peak.
- **Determinism.** Bit-identical results for a fixed thread count, block size and device; its own row in the tolerances.
- **Optimizer.** The Rust L-BFGS is exposed through `lbfgs(..., impl="rust")`, tested against `torch.optim.LBFGS` on a fixed problem, and is the default synthesis optimizer when the engine is selected.
- **Version impact.** Adding the engine or a configuration is a minor; making it the default for a configuration is a minor gated by a tier-2 benchmark. The intended end state: the engine is the default wherever it covers a configuration and wins the benchmark, and the Python engine stays the readable implementation, the second-order path and the fallback. An engine change that moves values outside tolerance is a bug in the engine, never a reason to regenerate references.
- **CI.** Tier 0 builds the CPU wheel and runs the reference and VJP tests with `engine="rust"`; tier 1, once set up, runs the GPU build. Review rules A5, B2, C6, D8 and E4 apply in `rust/`, whose local `AGENTS.md` holds them. The binding and the capability schema are protected paths.
- **Growth.** Every new capability follows three steps: the mathematics first (`maths.md`, `_spec`, the Python engine); then the engine declares the capability and passes its parity tests; then it ships as a minor, and becomes a default once the benchmark shows the gain. A change of mathematics updates the oracle, the Python engine and the Rust engine in the same pull request; the Rust code follows the PyTorch version, never the reverse.

### 18.3 MPI

Distributed memory is one reason the engine was written in Rust. Each rank holds a contiguous NESTED range of the sky and the matching slice of the pyramid. Statistics are global — one `Allreduce` per level — and identical on every rank. Scaling of one evaluation (statistics and gradient), from STL-Dev-MPI `text/figures/scaling.txt`:

| Case | Ranks | Time per evaluation | Parallel efficiency |
| --- | --- | --- | --- |
| nside 1024, L = 2, Irene (EPYC 7H12) | 128 → 512 cores | 30.2 s → 8.9 s | 0.85 |
| nside 1024, L = 2, one EPYC 9655 node | 24 → 192 cores | 60.9 s → 10.2 s | 0.75 |
| nside 2048, L = 4, Tesla T4 GPUs | 2 → 3 GPUs | 474 s → 114 s | superlinear: the sky fits in device memory |

ESTEL integrates MPI in two stages:

1. **Out of process (1.x).** Distributed runs use the STL-Dev-MPI binaries under `mpirun` or Slurm. ESTEL reads and writes their plan, statistics and map formats through `estel.io`, so that a target can be prepared and a result checked in ESTEL.
2. **In process.** ESTEL runs under `mpirun` with mpi4py, an optional dependency. Each rank holds a `DataHealpix` slice (`partition=`, the STL-Dev-MPI partition: contiguous NESTED ranges, coarsest levels replicated). `scattering()` returns the global statistics on every rank, and the VJP the rank's slice of the gradient. Because 1.0.0 reserves `partition` and the `_global_sum` hook, this stage needs no API change. A test checks that 1, 2 and 4 ranks agree within 1e-13.

### 18.4 HEALPix spectra in Rust

The engine's `SpectrumEngine` is a differentiable anafast: auto, cross and spin-s spectra (EE, BB, EB), with masks, forward and VJP, on CPU and GPU (spin 0 on the device). It matches `healpy.anafast(..., iter=0)` and `healpy.map2alm_spin`, which are its tests.

- **It becomes the HEALPix spectrum method of ESTEL (LF24)**, hidden behind `estel.spectra`: users call the ESTEL API, on PyTorch or JAX arrays, and never see Rust.
- **A separate, required wheel.** The anafast code is independent of the scattering engine, so it lives in its own crate, `rust/estel-sht`, and its own wheel, `estel-sht`, which ESTEL requires once HEALPix spectra exist. The scattering engine `estel-rust` stays optional. Binary wheels are built in tier 0 for Linux (x86\_64, aarch64), macOS (arm64) and Windows.
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
| Low-pass filters in S3/S4 (§20.1) | no | to implement in Rust, in sync with PyTorch (LF6) |
| Tensor products (2D+1D, HEALPix blocks) | no | planned (LF17) |
| float32 | no | opt-in, to implement (LF16) |
| Second order | no | by hand, later (LF15) |
| MPI, distributed sky (HEALPix) | through the binaries | in process from ESTEL: next (LF4) |
| HEALPix anafast: auto, cross, spin s, masks, VJP | yes (GPU for spin 0) | exposed through `estel.spectra` (LF5, LF24); spin s on GPU later |

A generated capability matrix, `docs/engines.md`, built from the engine's declaration and the Python engine's registry, shows for each configuration which paths exist and which is the default.

### 18.6 The scope of the Rust engine, re-examined

**What it brings today** (§18.1): at `nside` 128, 0.106 s and 2.2 GB against 1.229 s and 19.8 GB for STL with automatic differentiation. The gain comes from a hand-written adjoint, which does not keep the intermediate maps, and from compiled loops.

**What changes.** Three changes bring part of that gain to the Python engine without a hand-written adjoint: the Gram form, in place since T2.12; level-wise gradients (LF36), tried on CPU without gain and measured again on GPU (LF62); and, if accepted, the strided second layer (LF38). The Python engine then keeps second-order derivatives, both backends and every geometry, which the Rust engine has only in part (Part III, §18.5).

**Proposal (LF3, "To decide").** Do not decide the scope of the Rust engine now. After the reorganisation of the first releases of step 2, measure the Python engine against STL-Dev-MPI on the same cases, in time and in peak memory. Then decide what the Rust engine is for: the candidates are CPU clusters with MPI, the spherical harmonic transform (`estel-sht`, which nothing else replaces), and whatever gap remains on GPU.

**Why it matters for 1.0.0:** nothing to change. The engine interface reserved in T1.4 fits any of these outcomes.

## 19. Compact coefficient layout

Implemented in step 1, first in phase P3 (Part II, §15.2, T2.15). Until then the statistics keep STL's full-grid storage behind accessors that do not depend on it (Part II, §12.7), so switching to the compact layout changes no public signature and no value.

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

**Done in step 1** (Part II, §15.2, T2.12): each second-layer mean is a contraction over pixels, one matrix product per level and band orientation, the band orientation being the leading axis of the second-layer maps so that no copy is needed (§19.12, point 2). Every value stays within the parity tolerance, and the memory of the computation is 39 to 49 % lower (§19.14).

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

`stats.full_grid("S4")` builds, with scatter\_to\_grid, the tensor on the complete grid of indices, with NaN at the positions where no coefficient exists and, on request, the conjugate duplicates by conjugation. It serves humans, plots and comparisons with 1.0.x results. It is never called inside `scattering()`, and a test asserts it.

### 19.6 Reductions as static linear maps

Normalisation (for a fixed reference), `iso`, `angular_fourier` and `scale_cosine` are linear in the coefficients. Each is precomputed in NumPy as an (index, weight) table and applied with `segment_sum` or a small matrix product per segment, returning a new layout. They are therefore differentiable, compilable and backend-neutral.

### 19.7 What the compact layout does not solve

The layout removes the waste in the *output*. For large tensor products the limit is the *intermediate* stacks: at the 2D+1D+1D size of Table 1 (128 × 128 × 512 × 32 ≈ 2.7·10⁸ pixels), the first layer alone, with 16 complex orientations, takes about 34 GB in float32. `estimate_memory()` reports the peak beforehand and a clear error is raised beyond a budget. The level-organised loop is the natural unit for streaming later (LF19), and the Rust engine already bounds memory by block for HEALPix and 2D (§18).

### 19.8 The canonical set

STL stores, at `j1 = j2`, both `(l1, l2)` and `(l2, l1)`, which are complex conjugates, and both channel orders `(a, b)` and `(b, a)`. A Hermitian-packed layout stores each pair once. Since the loss sums squares over stored coefficients, the duplicates currently count twice. Two options:

- (a) keep STL's set as canonical, with packing as a pure storage choice and duplicates weighted twice in the loss;
- (b) make the non-redundant set canonical.

Option (a) keeps the losses of 1.0.0; option (b) is cleaner and changes reference values (a major). The decision is taken after 1.0.0 (LF20). Whatever is chosen, the gather of the Rust engine (§18.2) must produce it.

### 19.9 Where the conversion is written

The conversion between the two representations is written once, in shared Python on a few generic primitives, unless that costs clearly in numerical efficiency; it comes with the compact layout, in step 1 (Part II, §15.2, T2.15).

### 19.10 Where time and memory go

*The figures of §19.10 to §19.12 are counts of multiplications and of stored numbers, and tests on a NumPy reconstruction of STL's loop, validated against the parity fixtures (identical kernels, statistics equal to STL's to 1e-7). Timings are NumPy on one CPU core: they compare two forms of one computation and do not predict PyTorch or JAX on a GPU. References R-… point to the review of 16 November 2026 (Part IV, §22); definitions are in `docs/maths_step2.md`.*


A count of real multiplications per pixel of the input, for the 2D loop with K = 5 (a complex kernel on a real map is two real convolutions of 25 taps):

| | L = 4 | L = 8 |
| --- | --- | --- |
| Convolutions | 1,690 | 6,210 |
| Covariances (S3 and S4) | 720 | 5,290 |
| Share of level 0 in the convolutions | 59 % | 58 % |
| Share of level 0 in the covariances | 45 % | 44 % |
| Second-layer maps alive at level 0 | 32 reals per pixel | 128 |

Three facts follow, and they hold for any J, since each level has a quarter of the pixels of the previous one:

1. **The finest level decides.** More than half of the time is spent at level 0, and the largest arrays live there. An optimisation of the coarse levels gains little.
2. **One term dominates.** The second convolution at the same scale as the first (j₂ = j, the modulus of scale j filtered again at scale j) costs L² convolutions at level 0: 800 of the 1,690 multiplications per pixel, and all 32 reals per pixel of second-layer maps there.
3. **Convolutions outweigh covariances** for L = 4; they are equal for L = 8. The covariances cost more in practice than this count says, because they run as many small operations with temporary arrays.

With gradients, automatic differentiation keeps every intermediate map: about 75 reals per pixel for L = 4, which is 600 bytes per pixel in float64. A 4096² field needs 10 GB for one batch element.

Two more facts, which the count does not show:

4. **In STL's loop, the largest arrays are temporaries, not maps.** For each S4 covariance, STL builds a product of shape (L, L, L, pixels) in complex, then multiplies it by the weights: 256 reals per pixel at level 0, against 32 for the maps it combines (measured: 3 to 8 times the memory of those maps), and the peak of the loop is set there. ESTEL's Gram form removes these temporaries (T2.12).
5. **In a synthesis, the optimiser holds more than the gradient computation.** L-BFGS keeps `history_size` pairs of vectors of the size of the field: 100 reals per pixel with STL's default of 50.

### 19.11 Optimisations, by leverage

The proposals are entries of the register (§16): their effect is in its column "Why", their step in its column "Step". By decreasing leverage:

- **Change what is computed.** The largest effect, and each is first a scientific question, since it changes values: the strided second layer (LF38), the strict scale order (LF35), pruned plans (LF37), the canonical set (LF20), a scale cut-off (LF40).
- **Compute the same numbers differently.** The Gram form, done in step 1 (§19.14); complex-to-complex convolutions (Part II, §15.2, T3.2).
- **Memory.** Level-wise gradients (LF36), the HEALPix tables (Part II, §15.2, T5.1), the length of the L-BFGS history (Part II, §15.2, T6.1).
- **Scale out.** Partitioned execution (LF34), coarse-to-fine synthesis (LF12), the scope of the Rust engine (§18.6; LF3).

#### What does not pay

- **Separable kernels.** The 5 × 5 kernels along the axes are nearly of rank 1 (99.2 % of their energy), but the diagonal ones need rank 3 for the same accuracy: 30 taps instead of 25. No gain at K = 5.
- **A scale cut-off for speed.** See LF40: it reduces the number of coefficients, not the time.
- **Optimising the coarse levels.** They are a small share of everything.
- **Float64 accumulation with the present sums.** Tested: float32 statistics differ from float64 ones by 1e-7 to 1.5e-6 with or without it; the error is in the maps. It matters only inside the Gram form, where it is a rule (M-GRAM).
- **A separable smoothing kernel in 1.0.0.** STL's 6 × 6 table is separable only to 2e-8: 12 taps instead of 36 change the values above the parity tolerance (R-E01). In step 2 it is a small gain on a small share, about 3 % of the loop.

### 19.12 What to put in place before 1.0.0, so that nothing is closed

None of these changes a value. Each exists to make a later change local. Points 1, 2, 3, 6 and 9 are in place, and point 7 is decided (Part IV, §22, R-Y03). Point 5 is a task of step 1 (Part II, §15.2, T2.18), and `estimate_memory` (point 8) comes with the diagnostics (T6.2); point 4 comes with the low-pass filter and the centring (LF6, LF33).

1. **Windows are objects with a kind** (full, slice, weights), given by the plan (issue #72, issue #72 (R-V02)). The centring of step 2 and the fast path for non-periodic data build on them.
2. **The loop reads its index sets from the plan.** Already so for S3 and S4. The strict order, the cut-off and pruned plans are then options of the plan. The same holds for the positions where a convolution is evaluated: the strided second layer is then an option too.
   **And the band orientation is the leading axis of the second-layer maps,** so that the Gram form needs no copy.
3. **One internal sum** (`_global_sum`) between the sum over pixels and the division, called once per level. Partitioned execution and MPI insert themselves there. On JAX the loop is compiled (T2.13), and the hook is read when it is compiled: whatever replaces it must also drop the compiled loops, as its tests do.
4. **The low-pass filter and the centring are described by the plan** (Part III, §20.10 and §20.11): decided, and partly in place. How the means of the maps that end with a low-pass filter are centred, for single precision, is decided with LF33.
5. **"Scale" and "level" are two different things** in the plan and in the notation (issue #71 (R-N10)): the low-pass filter without final downsampling, and voices, need it.
6. **Families are a list, not four attributes.** A low-pass index, fine-band coefficients or a new family must not change the interface of `STStatistics`.
7. **The reference serves another operator** (R-Y03): it depends on J, L and the channel pairs only.
8. **`estimate_memory`** reports the peak of the loop from the plan, with and without gradients, so that a user knows before running. Its formula is the count of §19.10.
9. **A benchmark from phase P2** (Part I, §6.3; Part II, §15.2, T2.11) records time and peak memory, with gradients, on 2D cases, on CPU; the cases of the other geometries come with the benchmark of step 2a (LF62). Every proposal is judged against it.

### 19.13 Steps 2a, 2b and 3

Step 2 has two parts (Part I, §1.3), and the field `step` of each card of the register (§16) places its entry, by the criterion of the decision log of step 2 (`docs/register/decisions.md`):

1. **Step 2a: the mathematics corrected and stabilised.** The changes of values that make the statistics what they should be: centred covariances (LF33), the low-pass filter in every family (LF6), the canonical set of coefficients (LF20), complete and correct reductions (LF32, LF26), and the other corrections that the open points of the maths tab sent to the register; with them, the benchmark of step 2 (LF62). Its changes of values make one change of reference, one bump of `STATISTICS_REFERENCE`, so that users see one change and not five.
2. **Step 2b: the advanced optimisations.** The joint design of the smoothing kernel and the wavelets, with the Littlewood–Paley condition and exact invariance (LF22, LF25; §20.10), and the Rust engine (LF3).

Step 2a ends with a 1.x release that changes the defaults and bumps `STATISTICS_REFERENCE` once; step 2b ends with ESTEL 2.0.0 (Part I, §7.1). Each release drops the previous defaults at once: no behaviour of STL is kept after the release of step 2a. So the corrections of step 2a land behind private flags, off by default, and `main` stays releasable; the release pull request of step 2a switches them on, removes the old paths and the flags, regenerates the references, and retires the STL parity tests and fixtures, a human decision recorded in that pull request (AGENTS.md, golden rule 2). The entries that change no reference value ship as minor versions when they are ready. **Step 3** holds the long-term developments.

The reorganisations that change no value come first, so that each later change is measured on a faster and smaller engine: the benchmark (T2.11), the Gram form (T2.12) and the compact layout (T2.15, done first in P3) belong to step 1 (Part IV, §22, "Efficiency in step 1").

### 19.14 Results of the optimisations of step 1

What the optimisations of step 1 brought, measured by the benchmark (Part I, §6.3) on CPU (Apple M3); the detail is in the pull requests and in `bench/results/`. Each optimisation adds its result here. The optimisation session of the benchmark of step 2 (LF62) confirms these conclusions on native x86 Linux and on GPU.

- **Kept:** the Gram form of the second layer (memory of the computation 39 to 49 % lower, #95); the compilation of the loop on JAX (2 to 15 times faster, #96); plain sums for the means without a mask (#96).
- **Tried without gain:** half of the Gram product, slices inside the Gram form, kernels as constants (#96); level-wise gradients (LF36, issue #92).
- **Deferred:** per-scale convolutions and the measurements on x86 and GPU (LF62, issue #91); a first-order gradient (LF46).

## 20. New mathematics

Each item below changes what ESTEL computes. Each one is written in `maths.md` and `_spec` first, lands as an opt-in option, and becomes a default through a bump of STATISTICS\_REFERENCE, decided with the maintainers (Part I, §7.1).

### 20.1 Low-pass filters everywhere, including S3/S4

- **Decision.** The low-pass filter φ is used for all coefficients, including S3 and S4, and in every geometry — not only in tensor products, as in STL.
- **Two possible extensions to S3/S4**, from §5 of the STOC deliverable:
  - *restricted*: the low-passed field enters S3/S4 only as one extra scale, adding 0.3–3.5% coefficients;
  - *all combinations*: φ may replace any of the wavelets ψ\_j1, ψ\_j2, ψ\_j3, in each block independently, adding 38–112%.
- **Why "all combinations" grows so much.** Each block gains one scale (J becomes J + 1), and S4 counts triples of scales: for a block with J = 2 the number of triples goes from 4 to 10. These factors multiply across blocks, so blocks with few scales dominate the increase. The single orientation of φ tempers it somewhat.
- **"All coefficients" corresponds to "all combinations".** This is to be confirmed with the counts above in mind (LF6, "To decide").
- **Consequences.** A new default changes reference values, so it bumps STATISTICS\_REFERENCE. The Rust engine implements the same terms in sync with PyTorch. In 2D the coefficient count goes from 1,474 to 1,476 for the S1/S2 terms alone (Part IV, §22, G1).

### 20.2 Positive spectrum estimator for non-periodic data

The current estimator for non-periodic data can return negative bins (maths tab, OP-21). A tapered periodogram is positive by construction. To be fixed in `maths.md`: the taper, its parameter, and the normalisation of the bins.

### 20.3 NaN masks in tensor products

First for separable masks, then for general ones. The choice to make: the product of per-block valid weights, or the observed-sample mask with per-block border crops. They differ when the mask varies along another block.

### 20.4 HEALPix blocks in tensor products

The HEALPix geometry supplies the same hooks as a planar block (`_conv`, `_downsample`, `_crop_weights`) along its pixel axis. Its gauge orientations enter the product orientation index like 2D orientations do. A new geometry, hence a major version.

### 20.5 Spectrum estimators in N dimensions

The four v1.6.0 estimators and their automatic selection, generalised from the last two axes of 2D data to the axes of any block. This also removes the class-name check that today rejects 1D and tensor-product data.

### 20.6 Wavelet optimisation

The wavelet bank will be optimised later, rather than adding new families (such as Morlet). The criteria — localisation, Littlewood–Paley flatness, kernel size, cost — are to be agreed, and any change of bank is a major version.

### 20.7 Canonical set and gradients through the reference

The canonical set is decided with LF20 (§19.8). Gradients through the reference are decided: they follow from the normalisation (maths tab, OP-18), which matters for implicit differentiation when the target itself depends on the parameters.

### 20.8 Exact translation invariance

STL's downsampling uses a 6×6 anti-aliasing kernel with asymmetric padding (2 pixels before, 3 after) and a stride of 2 (maths tab, M-DOWN, OP-2). The kernel's centre and the sampling grid are probably locked, so a shift of the field by one pixel does not shift the statistics' inputs by half a pixel at the next level, and exact translation invariance is lost. 1.0.0 reproduces STL and does not test it (Part I, §5.1). LF25 removes the grid locking — a centred kernel, or a polyphase scheme — provided the Littlewood–Paley property and the other constraints still hold, and then adds translation invariance to the invariance tests.

### 20.9 The partial order of tensor-product scales

STL orders the multi-index scales of a tensor product componentwise, so pairs such as (0, 2) and (1, 1) — finer in space, coarser in time — are not comparable and give no S3 or S4 coefficient (maths tab, M-IDX-TP). LF27 examines whether these interactions carry useful information on the STOC test cases. If they do, it defines which incomparable pairs to add, with their cost in coefficients, as a change of reference values.

### 20.10 Joint design of the smoothing kernel and the wavelets

LF6, LF22 and LF25 are one design problem: the pair (s, ψ). With φ\_J realised as the cascade of the smoothing kernels, the Littlewood–Paley sum Λ equals 1 everywhere as soon as the per-level energy partition P does (maths tab, "Filters and scales"); a dedicated low-pass kernel only acts on the last octave and cannot make up for losses between scales. The design requirements:

1. **Energy partition.** P = 1 at every level, within a tolerance set when the design is made, with both signs counted and an explicit gain convention (OP-33).
2. **Bounded aliasing.** At each downsampling, the energy that the smoothing kernel lets through above half the Nyquist frequency, which folds back, stays below a set fraction. Requirements 1 and 2 pull in opposite directions; they are compatible only if the wavelets capture the whole upper half of the band.
3. **First octave and corners.** A decision for the frequencies that no wavelet covers: the first octave and, in 2D, the corners of the frequency square. Either a residual high-frequency band, or a deficit that is documented.
4. **Centred kernels,** so that the downsampling grid does not lock (LF25, OP-2).
5. **Rotations.** In 2D, with L even, a rotation of the field by π/2 shifts the orientations by L/2, those of the last wavelet that leave [0, π) coming back conjugated (maths tab, M-RED-ISO, OP-5); the statistics must then match within a tolerance set when the design is made. STL's kernels miss it by 3e-3 to 9e-3 (OP-58). A CI test checks it, beside the invariances of Part I, §5.1.
6. **A smoothing kernel that sums to 1.** STL's 2D kernel sums to 1.00000004 (OP-59), so the constant shift of 2D data with a mask holds only to about 1e-9 per unit of the constant (Part I, §5.1); with a kernel summing to 1, that invariance is tested too.

At the last level, the low-pass filter is applied without the final downsampling (LF6). Measured on STL's kernels (OP-32), the 1D pair already nearly meets requirement 1; the 2D pair is anisotropic and leaves the corners uncovered. The three diagnostics of Part II, §12.10 (energy partition, Littlewood–Paley sum, aliasing) are the measurements on which these requirements will be checked.

### 20.11 Centred covariances

The scattering covariances are covariances (maths tab, "Moments and covariances"); STL computes raw second moments (OP-36). Step 2 computes covariances, by default, with the rules below. Their order of priority is numerical efficiency first, then simplicity.

1. **Centred at the least cost.** Cov(A, B) = M2(A, B) − M1(A) conj(M1(B)). By default the raw moments are computed as today, each first moment once per map, and the correction is applied to the arrays of coefficients, without building centred maps. For the maps that end with a low-pass filter this subtraction can lose precision in single precision (maths_step2 M-COV), and centring these maps first is the alternative: the method is chosen with LF33, the most efficient that keeps the precision. When the last filter is a low-pass filter, the first moments are already known: M1(φ\_J ⋆ |W\_j x|) is S1(j), up to window effects.
2. **Only where needed, decided by the plan.** A covariance is not centred when the plan can prove that one argument has M1 = 0: its last filter is band-pass, and the window is the whole periodic domain without mask. Otherwise it is centred. So periodic data without mask and without low-pass terms keep today's values and today's cost; any term ending with a low-pass filter is centred; with non-periodic borders or with masks, every covariance is centred.
3. **Band-pass kernels sum to zero, by requirement.** Every band-pass kernel is built with a zero sum, and its construction fails if this does not hold within rounding. Rule 2 relies on it: it is a guarantee of the filter bank, not a property looked up kernel by kernel.
4. **One window per level when centring is active.** All the maps of a level share the most restrictive border and mask, so that M1 and M2 use the same window. Centring is then exact, costs one sum per map, and is a rank-one correction of the level's Gram product (§19.3). This window policy is an option of the plan, which holds the weights of every mean: going back to one window per pair changes values, not the structure of the code.
5. **A fast path for non-periodic data without mask,** a common case in optimisations: the window is a rectangle, taken as a slice of the arrays, with no weight map.
6. **Two independent options.** `center` takes the values `none` (STL, 1.0.0), `self` (the map's own first moments; the default in step 2; gradients flow through them) or `ref` (the first moments of a reference). `norm` does not depend on it. Both may read the same reference object, which stores the normalisation scales and the first moments.
7. **The mean of the data** is removed first and stored apart, explicitly, by default; with `standardize`, the data are also divided by their standard deviation (Part IV, §22, R-A10). If it is not removed, every term ending with a low-pass filter, S1 included, is centred explicitly: this is never left implicit.

The measured sizes of the forgotten term are in OP-36.
