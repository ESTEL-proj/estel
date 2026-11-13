# Part II — Step 1: transfer from STL (ESTEL 1.0.0)

Step 1 rebuilds what STL computes inside the architecture of Part I, and proves, case by case, that ESTEL gives the same numbers. It adds no mathematics. Everything it leaves out is listed (§10), and everything it prepares for step 2 is reserved explicitly (§9.4).

[Back to the overview](overview.md)

## Contents of this part

- [9. Sources and scope of the transfer](#9-sources-and-scope-of-the-transfer)
  - [9.1 Sources](#91-sources) · [9.2 The transfer rule](#92-the-transfer-rule) · [9.3 Scope of ESTEL 1.0.0](#93-scope-of-estel-100) · [9.4 Interfaces reserved in 1.0.0 for step 2](#94-interfaces-reserved-in-100-for-step-2)
- [10. What is not transferred](#10-what-is-not-transferred)
- [11. Package tree](#11-package-tree)
- [12. Function inventory per module](#12-function-inventory-per-module)
  - [12.1 Tags](#121-tags) · [12.2 Public namespace and configuration](#122-public-namespace-and-configuration) · [12.3 L0 primitives](#123-l0-primitives) · [12.4 DataClasses](#124-dataclasses) · [12.5 Wavelets](#125-wavelets) · [12.6 Scattering](#126-scattering) · [12.7 Statistics](#127-statistics) · [12.8 Spectra](#128-spectra) · [12.9 Synthesis](#129-synthesis) · [12.10 Utilities](#1210-utilities) · [12.11 The oracle](#1211-the-oracle)
- [13. Mathematics extracted from STL, and open points](#13-mathematics-extracted-from-stl-and-open-points)
  - [13.1 Principle](#131-principle) · [13.2 Open points in STL's mathematics](#132-open-points-in-stls-mathematics)
- [14. Parity protocol with STL](#14-parity-protocol-with-stl)
  - [14.1 Principle](#141-principle) · [14.2 The harness](#142-the-harness) · [14.3 Comparison](#143-comparison) · [14.4 Coverage](#144-coverage) · [14.5 From parity to reference values](#145-from-parity-to-reference-values)
- [15. Roadmap to 1.0.0](#15-roadmap-to-100)
  - [15.1 Phases](#151-phases) · [15.2 Work breakdown of step 1](#152-work-breakdown-of-step-1)

## 9. Sources and scope of the transfer

### 9.1 Sources

| Source | Version | What ESTEL takes | What it does not take |
| --- | --- | --- | --- |
| STL-Dev `master` | `8eaf769` (v1.6.0), frozen | 2D kernel classes; the unified 2D cross-spectrum operator, with its four estimators (`fft`, `pbc_mask`, `nan_mask`, `pbc_nan_mask`) and their automatic selection; ST operator and statistics | the Fourier-space class; legacy modules |
| STL-Dev `2D+1D` | `8f51143`, frozen | dimension-generic planar kernel code; the 1D class; tensor products of 1D and 2D blocks with multi-index scales; low-pass φ and box term in tensor products; Gaussian start | its cross-spectrum refactor, with the 1D and tensor-product spectra: later (Part III, LF23) |
| STL-Dev `HealpixUpdate` | `90eb1a9`, frozen | the HEALPix kernel class: gauge-equivariant convolution and downsampling, ported into ESTEL (§9.3) | its spherical-harmonic spectra: later (Part III, LF24); the Python pyramid engine (`PyramidScattering`, `LocalScattering`, `AdjointOps`), superseded by STL-Dev-MPI (Part III, §18) |
| STL-Dev-MPI | `6f0f870` | in step 1, no code. Its exact HEALPix formulas document `maths.md`, and its fixtures give an independent check | the Rust engine, moved into the ESTEL repository in step 2 (Part III, §18) |
| STOC deliverable 1 | 30 October 2026 | the description of tensor products, low-pass filters, missing values and non-periodic domains; the test cases | its candidate features (§5 of the deliverable): step 2 |
| GRID4EARTH review rules | `d38b775` | the review rubric, adapted in Part I, §8.3 | rules specific to healpix-geo |
| healpix-analyse | `aac5e26` (v2026.10.2, its latest pushed version) | no runtime code: the parts STL uses are ported into ESTEL (§9.3). The package is a pinned, test-only dependency, used to prove parity | everything else |

### 9.2 The transfer rule

**The philosophy of step 1.** ESTEL first reproduces STL, with its problems; most of them are fixed afterwards, each as a deliberate, traced change with its reference values before and after. This guarantees that ESTEL starts from a base whose every number has been checked against STL.

- **Exact reproduction.** The same inputs give the same numbers, up to rounding (§14).
- **Frozen sources.** STL is frozen at the commits of §9.1. Changes made to STL afterwards are not ported in step 1; if useful, they enter ESTEL later as step-2 features.
- **Branch conflicts.** When two branches implement the same thing differently, step 1 follows the one this document names. For spectra, 1.0.0 contains only the 2D cross spectra, with the v1.6.0 estimators of `master`. The 1D and tensor-product spectra of `2D+1D` come later (Part III, LF23), and so does generalising the v1.6.0 estimators to every geometry (LF10).
- **No silent fixes.** STL behaviours that look accidental are reproduced and recorded as open points (§13). Fixing them is step 2 work.
- **No silent drops.** Anything STL computes and ESTEL does not is listed in §10.
- **Where STL fails, ESTEL raises an error.** STL is reproduced wherever it gives a result. Where it gives no usable value — a division by zero when no pixel is valid, a crash when the sizes are not divisible by 2^(J−1) — ESTEL raises a clear error at construction or at call time, and parity is compared only where STL works (maths tab, OP-3, OP-8).
- **HEALPix ported, not imported.** `HealpixUpdate` holds two implementations: the kernel class, built on healpix-analyse, and the `TorchGeometry` used by the pyramid engine; its tests show they agree (`test_torch_conv_matches_library`, `test_torch_decimation_matches_library`). Step 1 reproduces the kernel class, computing its convolution and downsampling with NumPy tables ported from `TorchGeometry` (§9.3).
- **Two exceptions.** Step 1 adds two capabilities STL lacks, as infrastructure rather than mathematics: synthesis checkpoints (§12.9) and the experimental JAX backend (Part I, §2.2.5).

### 9.3 Scope of ESTEL 1.0.0

| Area | In 1.0.0, as in STL | Later (Part III register) |
| --- | --- | --- |
| Geometry | `Data1D`, `Data2D`, `DataHealpix`, `DataTensorProd` with 1D and 2D blocks | HEALPix blocks in tensor products (LF9) |
| Data | batch, multi-channel, cross-channel matrix, pbc per block, static NaN masks in 1D, 2D and HEALPix | NaN masks in tensor products (LF8) |
| Wavelets | 2D bump-steerable 5×5; 1D analytic wavelet with its conjugate (L = 2) in tensor products; HEALPix gauge-oriented kernels; anti-aliasing filter H | wavelet optimisation (LF22) |
| Low-pass | φ and box term in tensor products, S1/S2 only | low-pass in every geometry and in S3/S4 (LF6) |
| Statistics | vanilla ScatCov: mean, var, S1–S4; normalisation; iso, angular\_ft, scale\_ft; selection | canonical coefficient set (LF20); reductions on tensor products redefined (LF26) |
| Storage | STL-shaped full-grid tensors, behind accessors that do not depend on storage | compact layout (LF14) |
| Spectra | 2D only, as in `master` v1.6.0: four estimators with automatic selection. `compute_PS` is True by default for 2D data; for other geometries it is unavailable in 1.0.0 and raises a clear error | 1D and tensor-product spectra (LF23); HEALPix spectra (LF24); v1.6.0 estimators in every geometry (LF10); positive non-periodic estimator (LF7) |
| Synthesis | from maps or stats, `mean_field`, `running_shape`, `pbc_running`, `nbatch`, Gaussian start, frozen masked pixels; the built-in L-BFGS of PyTorch; checkpoints (new, §12.9). Synthesis tests and parity come after 1.0.0 | Rust L-BFGS (LF3); coarse-to-fine, per-iteration filtering, ensembles (LF12) |
| Backends and engines | PyTorch; JAX experimental (every primitive, tested on every PR, not covered by the 1.0.0 guarantees); float32 default, float64 path; Python engine only | JAX stable, with its synthesis optimiser (LF1); Rust engine (LF3); MPI (LF4) |
| Differentiation | first and second order through PyTorch autograd, tested in float64 | second order in the Rust engine (LF15) |
| Utilities | STL's plotting and diagnostics, ported to the new objects | improvements, `gaussianise`, surrogates (LF11) |

**HEALPix in step 1.** STL's HEALPix path calls three classes of healpix-analyse (`HealPixConv`, `HealPixDown`, `HEALPixSHT`). The package is young, has no stated API stability, and pulls heavy unrelated dependencies (`umap-learn` and `healpix-plot`, and through them numba, scikit-learn and cartopy). ESTEL therefore ports what it needs (Part I, §1.6): the convolution and the downsampling are computed with index and weight tables built in NumPy, ported from `TorchGeometry` and applied with `gather_weighted` and `segment_sum`, so they work on every backend. HEALPix spectra are not in 1.0.0 (LF24), so no SHT is ported in step 1. healpix-analyse remains a test-only dependency, pinned to `aac5e26`, to prove parity.

### 9.4 Interfaces reserved in 1.0.0 for step 2

| Reserved interface | In 1.0.0 | Used in step 2 by |
| --- | --- | --- |
| L0 protocol and backend registry | PyTorch; `"jax"` registered as experimental, with a warning | the JAX backend (Part III, §17) |
| `engine=` argument, engine registry and protocol, `diff_order` | only `"python"`, order 2 | the Rust engine (Part III, §18) |
| `STStatistics` accessors (`family`, `select`, `to_vector`, `full_grid`) | backed by full-grid tensors | the compact layout (Part III, §19), with no API change |
| `DataHealpix(cell_ids=…, partition=…)` | whole sky by default; partial ranges accepted | MPI in process (Part III, §18.3) |
| `allreduce` hook in the reductions of L2 | identity | MPI in process (Part III, §18.3) |
| `estel.io`, with a versioned schema | ESTEL format | the JSON formats of STL-Dev-MPI (Part III, §18.3) |
| `method=` of `CrossSpectrumOperator` for 1D, tensor-product and HEALPix data | accepted names, raising a clear error | LF23, LF24 |

Each reserved interface has a test in 1.0.0 that exercises its trivial implementation, so it cannot rot before it is used.

## 10. What is not transferred

| STL code | Why | What replaces it |
| --- | --- | --- |
| `STL_2D_FFT_Torch` (Fourier-space wavelets) | obsolete in ESTEL | real-space kernels; its `bump_steerable_bank` is ported to build them (maths tab, OP-23) |
| The `has_fewer_convolutions` path and the `SC` switch | only vanilla ScatCov is kept; also removed from the Rust engine's bridge (Q12) | — |
| The dead modules `StlData`, `DataType1`, `DataType2`, `WaveletTransform`, `cs_op.py`, `LossOptim` | unused or broken | — |
| `inplace=` arguments | incompatible with JAX and second-order differentiation | functional updates (`with_array`) |
| `mean`, `cov`, `standardize` duplicated in each wavelet operator | duplication | one implementation in the mother DataClass |
| Dynamic class creation in the tensor-product class (`self.__class__ = …`) | opaque to readers and tools | composition: a tensor-product object holds one array and the geometry of each block |
| Exact class-name checks (`type(data).__name__ != …`, three places) | break subclasses | `isinstance` or declared capabilities |
| The mutable `conv_history` list on data objects | hidden state | crop widths in the static plan |
| The Python pyramid engine of `HealpixUpdate` | superseded by its validated Rust port | the Rust engine of STL-Dev-MPI (Part III, §18) |
| The runtime dependency on healpix-analyse | young research code with heavy dependencies (Part I, §1.6) | the ported HEALPix convolution and downsampling (§9.3) |
| The 1D and tensor-product spectra of `2D+1D`; the HEALPix spectra of `HealpixUpdate` | outside the scope of 1.0.0 (Part IV, §22.4, C22) | later: LF23, LF24 |
| Morlet wavelets (`WType="Morlet"`) of the 2D kernel class | only bump-steerable wavelets are transferred; a general optimisation of the wavelets comes later, and may or may not include them | bump-steerable wavelets; LF22 (maths tab, OP-24) |

WPH coefficients were declared in STL but never implemented; they are out of the scope of ESTEL.

## 11. Package tree

The tree of ESTEL 1.0.0. Items marked `(step 2)` are created later; items marked `(reserved)` exist in 1.0.0 with a trivial implementation (§9.4). LFn refers to the register of later features (Part III, §16).

```text
estel/                             # one repository for Python and Rust (Part IV, §22.4, C11)
├── AGENTS.md                      # canonical agent context (Part I, §8.1)
├── CLAUDE.md                      # "@AGENTS.md" import + Claude-specific notes
├── .github/
│   ├── copilot-instructions.md    # generated mirror of AGENTS.md
│   ├── CODEOWNERS                 # protected paths → the three maintainers
│   ├── pull_request_template.md   # version-impact checklist
│   └── workflows/                 # tier0.yml, bench.yml, release.yml; tier1-gpu.yml after 1.0.0;
│                                  # rust-wheels.yml (step 2)
├── api/
│   └── manifest.json              # PROTECTED: machine-readable public API
├── docs/
│   ├── maths.md                   # PROTECTED: exact formulas, each with an M- ID
│   ├── reference.md               # the ESTEL reference document, written with 1.0.0,
│   │                              # with the register of later features
│   ├── policy/                    # PROTECTED: VERSIONING.md, PR_POLICY.md, TOLERANCES.md
│   ├── agents/                    # architecture, jax_rules, differentiation, testing,
│   │   │                          # review_rules, storage
│   │   └── tuning/                # synthesis, scales_and_memory, masks_and_borders,
│   │                              # power_spectrum
│   ├── validation/                # stl_parity.md (step 1 report)
│   └── user/                      # guides and API reference
├── src/estel/
│   ├── __init__.py                # public namespace = api/manifest.json
│   ├── _config.py                 # global backend, dtype, device defaults
│   ├── backend/                   # L0: the only code importing torch / jax
│   │   ├── _protocol.py  _registry.py
│   │   ├── torch/                 # primitives, optim.py
│   │   └── jax/                   # primitives (experimental from P1); optim.py (optax) in step 2, LF1
│   ├── engines/                   # _protocol.py, _registry.py (reserved);
│   │   │                          # python.py: the Python engine
│   │   └── rust.py                # (step 2, LF3) adapter to the estel-rust wheel
│   ├── data/                      # _base.py (DataClass), _planar.py, _healpix.py,
│   │                              # _tensor_prod.py, _masks.py
│   ├── wavelets/                  # operator.py, _banks.py, _healpix_bank.py,
│   │                              # _healpix_geometry.py (ported from TorchGeometry)
│   ├── scattering/                # _plan.py, operator.py, operator_tensor_prod.py
│   ├── statistics/                # statistics.py, _reductions.py, _normalization.py, io.py,
│   │                              # _layout.py (step 2, LF14)
│   ├── spectra/                   # operator.py, _estimators.py (2D), _bins.py
│   ├── synthesis/                 # api.py, _loss.py, _init.py, _checkpoints.py
│   ├── functional/                # pure functions: scattering(), cross_spectrum(), loss()
│   ├── io/                        # versioned formats; STL-Dev-MPI JSON (step 2)
│   ├── diagnostics/               # describe, estimate_memory, frequency_response,
│   │                              # littlewood_paley
│   ├── plot/  utils/
│   └── _spec/                     # PROTECTED: the NumPy float64 oracle
├── rust/                          # (step 2, LF3, LF5) Cargo workspace, from STL-Dev-MPI
│   ├── AGENTS.md                  # local rules: review rules A5, B2, C6, D8, E4
│   ├── estel-core/  estel-gpu/  estel-mpi/
│   ├── estel-sht/                 # Rust SHT and anafast (LF5)
│   └── estel-py/                  # PyO3 binding, replaces the stl-bridge C ABI
├── packages/                      # (step 2) maturin builds of the estel-rust (optional) and
│                                  # estel-sht wheels, released with estel, same version
├── tests/
│   ├── reference/                 # PROTECTED: reference values + MANIFEST.sha256
│   ├── tolerances.toml            # PROTECTED
│   ├── backend/  geometry/  scattering/  statistics/  spectra/  synthesis/
│   ├── spec/  invariance/  differentiation/  api/  reserved/
│   └── engines/                   # (step 2) engine parity
├── bench/                         # cases.toml, thresholds.toml (PROTECTED), run.py, results/
└── tools/
    ├── api_manifest.py  check_layers.py  check_agent_mirrors.py  check_maths_ids.py
    ├── version_impact.py
    ├── make_reference.py          # PROTECTED: regenerates tests/reference (human-run)
    └── stl_parity/                # step 1 only: parity harness (§14), not shipped
```

## 12. Function inventory per module

### 12.1 Tags

| Tag | Meaning |
| --- | --- |
| **F** | Frontend: public, listed in `api/manifest.json`, frozen within a major |
| **I** | Internal Python, backend-neutral (L1 or L2) |
| **G** | Geometry hook: declared by the mother DataClass and overridden by a subclass — the *only* methods subclasses override |
| **P** | Primitive: implemented once per backend in `estel/backend/<name>/` (L0) |
| **N** | NumPy-only plan construction: static, runs once when an operator is built |
| **R** | Reference implementation in `estel/_spec`: NumPy float64, the oracle |

The "Origin" column points to the STL code being ported; "new" means there is no STL counterpart. Entries marked *(step 2)* or *(reserved)* follow the conventions of §11. Names tagged I or N live in private modules or start with an underscore, and class names follow PEP 8 (Part I, §2.4). Where this inventory is silent, STL's behaviour is the default, and any doubt is raised with the maintainers.

### 12.2 Public namespace and configuration

`estel/__init__.py`, `_config.py`, `backend/_registry.py` (signatures decided on issue #39):

| Function / class | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `set_backend(name: str) -> None`, `get_backend() -> str` | F | Global backend choice: `"torch"` (the default) and, experimental, `"jax"` (with a warning). `get_backend()` returns the name, not the backend object. Operators record the backend when built and refuse arrays from another one | new (replaces the module globals of `torch_backend.py`) |
| `set_default_dtype(dtype: Literal["float32", "float64"]) -> None`, `set_default_device(device: str) -> None` | F | Global defaults. The dtype is float32 by default; `"float64"` enables float64 on every backend (Part I, §3.1, rule 7). The device is `"cpu"`, `"cuda"` or `"cuda:N"`, checked against the backend; until it is set, it is CUDA if available, else CPU, as in STL | `bk._DEFAULT_DTYPE`, `bk._DEFAULT_DEVICE` |
| `BackendMismatchError` | F | Raised when arrays of two backends are mixed (Part I, §2.2.3); a subclass of `TypeError`, whose message says how to fix it. Public because it can occur in 1.0.0; `EngineUnsupportedError` stays internal | new |
| `__version__`, `STATISTICS_REFERENCE`, `SYNTHESIS_REFERENCE` | F | Package version and the two reference counters (Part I, §7.1) | new |

### 12.3 L0 primitives

`estel/backend/`. Implemented for PyTorch and, experimental, for JAX in 1.0.0, with the same contract. The contracts that affect values — padding modes, correlation rather than convolution, strides — are specified in the maths tab (M-CONV-2D and following); the docstrings of `_protocol.py` cite them.

| Function | Tag | Contract | Origin |
| --- | --- | --- | --- |
| `xp` (Array API namespace) | P | Elementwise ops, reductions, `reshape`, `concat`, `stack`, `where`, `take`, `fft` extension, through `array_api_compat.array_namespace` | `torch_backend.py` |
| `asarray(np_array, dtype, device)` / `to_numpy(x)` | P | Moving plans and results between NumPy and the backend | `Base_DataClass._to_array` |
| `pad(x, widths, axes, mode)` | P | mode ∈ {`circular`, `replicate`, `zeros`}; absent from the Array API | `_get_padding_mode` |
| `conv_axes(x, k, axes, stride=1)` | P | Valid real correlation along the given axes, batched over the others; complex = two real calls, in L1. Applied directly along a block's axes, without moving them | `_conv_circular`, `_semicomplex_conv_circular` |
| `gather_weighted(x, axis, index, weight)` | P | `y[..., p, k] = Σ_m weight[p,k,m] · x[..., index[p,k,m]]`: a sparse matrix in padded CSR form, for the HEALPix stencils and decimation of step 1 | `TorchGeometry` |
| `segment_sum(x, axis, segment_ids, num_segments)` | P | Bin averages, HEALPix parent binning, reductions | bin masks, `_bin_to_parents` |
| `scatter_to_grid(values, index, shape, fill)` | P | Builds a new array on the complete grid of indices: values at the static positions `index`, `fill` (NaN) everywhere else; out of place. Its inverse, `gather_from_grid(grid, index)`, reads the existing coefficients back. Assembles the statistics in step 1 (Part I, §3.1.1); builds full-grid views in step 2 | new |
| `complex_to_real_pairs(z)` / `real_pairs_to_complex(r)` | P | Complex values as trailing (re, im) floats, and back | new |
| `stop_gradient(x)` | P | `x.detach()` / `jax.lax.stop_gradient` | `.detach()` calls |
| `random_normal(key, shape, dtype)`, `make_key(seed)` | P | Explicit keys on every backend | `torch.randn` in `Synthesis.py` |
| `jit(fn, static_argnames)` | P | Identity in PyTorch; `jax.jit` in the JAX backend | new |
| `value_and_grad(fn)`, `vjp`, `jvp`, `hvp` | P | Synthesis and the differentiation tests; `torch.func` / `jax` | `loss.backward()` |
| `lbfgs(...)` | P | The built-in L-BFGS of each backend: `torch.optim.LBFGS(line_search_fn="strong_wolfe")`; `optax.lbfgs` in step 2; the Rust L-BFGS (`impl="rust"`) when the Rust engine is selected (Part III, §18.2) | `optimize_lbfgs` |
| `deterministic(enabled)` | P | `torch.use_deterministic_algorithms`; on JAX, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, set before JAX starts. On in tests | new |

### 12.4 DataClasses

`estel/data/`. The mother class holds all shared logic; subclasses override only the **G** hooks.

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `DataClass(array, mask=None, pbc=..., ...)` | F | Frozen dataclass and pytree. Validates once (rule B1). NaNs in `array` become `mask` at construction; the mask and its reweighting maps are then static (Part I, §2.3). `pbc` must be an explicit boolean | `Base_DataClass.__post_init__` |
| `.array`, `.mask`, `.shape`, `.n_batch`, `.n_channels`, `.backend` | F | Read-only accessors | `Base_DataClass` |
| `.copy(empty=False)`, `.__getitem__` (batch and channel slicing only) | F |  | `Base_DataClass.copy`, `__getitem__` |
| `.with_array(new_array)` | F | Functional replacement of in-place updates | replaces `inplace=True` |
| `modulus(x)`, `divide(x, y, eps, pow)` | I | Safe modulus and the regularised division of STL, differentiable twice (Part I, §3.2.3) | `modulus`, `divide` |
| `mean_valid(x, weights, axes)`, `cov_valid(x, y, weights, axes)` | I | Weighted means over valid pixels, one implementation for every geometry; the denominators are precomputed in the plan | `mean`, `cov`, `square_mean` of each wavelet operator (three copies) |
| `standardize`, `unstandardize` | I |  | `standardize` |
| `propagate_mask_conv(mask, footprint)` | I | An output is masked when its kernel footprint touches a masked input | `_build_reweighting_maps…`, `_find_mask` |
| `_spatial_axes()` | G | Axes the geometry acts on | the `# conv_dims dependant` markers |
| `_conv(x, kernels, level)` | G | Planar: `pad` + `conv_axes`. HEALPix: `gather_weighted` with the stencils ported from `TorchGeometry` | `apply` of each wavelet operator |
| `_downsample(x, weights, level)` | G | Planar: strided normalised convolution with H. HEALPix: decimation with `gather_weighted`, then parent binning with `segment_sum` | `_downsample_tensor`, `_down_one_level`, `_downsample_masked` |
| `_crop_weights(level, border)` | G | Planar: removal of non-periodic borders, widths from the plan. HEALPix: identity | `_crop`, `_get_crop_border_size_*` |
| `_nyquist_filter(x)` | G | Pre-filter before synthesis | `apply_nyquist_filter`, `apply_bandlimit` |
| `_spectrum_geometry()` | G | 2D: FFT frequency grids. Other geometries: later (LF23, LF24) | `CS_operator_*` initialisation |
| `Data1D`, `Data2D` | F | Explicit planar classes sharing `_planar.py`, differing by `ndim` | `STL_1D_Kernel_Torch`, `STL_2D_Kernel_Torch` |
| `DataHealpix(array, depth, cell_ids=None, partition=None, ...)` | F | NESTED ordering, partial sky through `cell_ids`; `depth`, not `nside` (rule A4). `partition` is reserved (§9.4) | `STL_Healpix_Kernel_Torch` |
| `DataTensorProd(array, blocks, pbc)` | F | `blocks = (Data1D, Data2D, …)` in axis order. It **composes** block geometries (`BlockSpec`: class, axes, pbc) instead of creating classes at run time: one array, all blocks on the same backend | `STL_TensorProd_Kernel_Torch`, `DimHandler` |
| `DataTensorProd._conv`, `_downsample`, `_crop_weights` | G | Dispatch to the geometry of each block, applied along that block's own axes, without `swapaxes`. Phase P3 benchmarks it against STL's `swapaxes`, which is kept only where it is measurably faster | `_as_data_class_of`, `_from_data_class_of` |

### 12.5 Wavelets

`estel/wavelets/`:

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `WaveletOperator(data, J, L=None, wavelet="bump_steerable")` | F | Generic operator, without a DataClass in its name. Builds the bank with a branch on `data.ndim` (1D, 2D) or the HEALPix geometry. Validates its parameters at construction (J compatible with the data size, L ≥ 1, L within the tested range) | `WaveletOperator2Dkernel_torch`, `…1Dkernel_torch`, `…HealpixKernel_torch` |
| `.apply(data, j)` | F | ψ\_j ⋆ x at level j: complex output, L orientations | `apply` |
| `.lowpass(data, j)` | F | φ\_j ⋆ x | `lowpass`, `lowpass_dg` |
| `.downsample(data, level_out)` | F | Normalised downsampling to a coarser level | `downsample` |
| `bump_steerable_2d(L, size)`, `bump_steerable_1d(xi0, size, conjugate)` | N | Kernels built in NumPy, cropped to 5×5 or 5 taps. The Littlewood–Paley property is guaranteed by construction and tested in CI over a grid of parameters plus randomly drawn ones, covering the top of the range of L; it is not checked at run time | `_build_bump_steerable_wavelet_kernel`, `bump_steerable_bank` (ported from the Fourier-space class) |
| `smoothing_kernel_H(ndim)` | N | Anti-aliasing kernel; STL's unused `sigma_smooth` is dropped | `_get_smooth_kernel` |
| `healpix_kernel_weights(depth, L, kernel_size)` | N | Gauge-oriented kernel weights | `_wavelet_kernel` |
| `healpix_stencils(cell_ids, depth)`, `healpix_decimation(cell_ids, depth)` | N | Index and weight tables for `gather_weighted`, ported from `TorchGeometry` (step 1) | `TorchGeometry`, `stl-core` |

`frequency_response` and `littlewood_paley`, which only serve plots and checks, are in `estel.diagnostics` (§12.10).

### 12.6 Scattering

`estel/scattering/`:

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `STOperator(data, J=None, L=None, *, norm="store_ref", compute_cross_matrix=None, compute_PS=None, engine="python", **wavelet_kwargs)` | F | Single-domain operator (1D, 2D, HEALPix). Every computation option is set here, once. Reductions and selections are `STStatistics` methods only (rule A6). `compute_PS` is True by default for 2D data and unavailable elsewhere in 1.0.0. `engine` is reserved | `ST_Operator.__init__` |
| `STOperator.apply(data, ref=None) -> STStatistics` | F | The only call-time arguments: the data and an optional reference. The normalisation kind is fixed at construction | `ST_Operator.apply` |
| `STOperator.from_statistics(stats)` | F | Rebuild the operator that produced a statistics object | `from_ST_Statistics` |
| `STOperatorTensorProd(data, J=(…), L=(…), *, lowpass="ST", …)` | F | Tensor-product operator, with STL's low-pass options (`"ST"`, `"PS"`, both, or none). Overrides only the hooks below | `ST_Operator_TensorProd` |
| `_scale_index_set(plan)` | I | Mother: `j1 ≤ j2 ≤ j3` on integers. Tensor product: the componentwise partial order on multi-indices (§13.2, G3), plus the low-pass index per block | `_scales_S1_S2`, the loops of `_compute_ST_loop` |
| `_levels(plan)` | I | Resolution levels in execution order. Mother: `0…J`; tensor product: the product of block levels | `_compute_ST_loop` |
| `_first_layer(x, plan)` | I | W\_j x and \|W\_j x\| at every level, built functionally | `_compute_ST_loop` |
| `_second_layer(u, plan, level)` | I | The S3 and S4 means at one level, following the structure of STL's loop (Part I, §3.1, rule 8), returned as lists and stacked; no preallocated tensor filled in place (Part I, §3.1.1) | the `cov` calls of the loop |
| `_normalize(values, plan, ref)` | I | Normalisation only: `store_ref`, `load_ref`, batch mean, with the optional reference. Reductions are `STStatistics` methods | `_finalize_compute`, `to_norm`, `_normalize_scatcov` |
| `ScatteringPlan` | N | Static description: J, L, index sets, levels, crop widths, masks and reweighting maps, mean denominators; hashable | new |
| `scattering(x, plan, arrays, ref=None)` (in `functional`) | I/F | The pure function behind `apply`, public for `jit`, `grad` and `vmap` users, and for downstream implicit differentiation | new |

### 12.7 Statistics

`estel/statistics/`. The accessors do not depend on how statistics are stored: in 1.0.0 they read STL-shaped full-grid tensors; in step 2 they read the compact layout, with no API change.

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `STStatistics(values, plan, meta)` | F | Pytree: arrays as leaves, plan static | `ST_Statistics.__init__` |
| `.family(name) -> FamilyView` | F | `"mean"`, `"var"`, `"S1"`, `"S2"`, `"S3"`, `"S4"`, `"PS"` | attributes `.S1 … .S4` |
| `FamilyView.full_grid()` | F | The full-grid tensor, in STL's shape. The storage itself in 1.0.0; a view rebuilt on demand in step 2 | current storage |
| `FamilyView.select(**index)` | F | Structural selection by named index: `j1=…`, `l3=…`, `channel=(a, b)`, `block=…`; the result keeps every `STStatistics` method | `select` |
| `.to_vector(mask_st=None)` | F | Flat real vector for losses, without the structural NaN. `mask_st`, STL's name, is an optional boolean array over that vector: True keeps a coefficient, as in STL. It drops coefficients found unreliable by a boolean test; in a synthesis it is computed once on the target and reused at every iteration | `to_flatten`, `mask_st` |
| `.iso()`, `.angular_ft(harmonics)`, `.scale_ft(harmonics, dj)` | F | Each returns a new `STStatistics` | `to_iso`, `to_angular_ft`, `to_scale_ft` |
| `.save(path)`, `STStatistics.load(path)` | F | npz plus plan JSON, with a schema version and the reference counters | new |
| `.to_xarray()` | F | *(step 2, LF13)* Export for inspection | new |
| `Layout`, `Segment` | I/N | *(step 2, LF14)* The compact layout (Part III, §19) | new |

### 12.8 Spectra

`estel/spectra/`. In 1.0.0 only 2D spectra are provided, as in `master` v1.6.0; the other geometries come later (Part IV, §22.4):

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `CrossSpectrumOperator(data, *, n_bins=None, method="auto", compute_cross_matrix=None)` | F | Generic operator, with a single `method=` argument for every geometry. In 1.0.0 it accepts 2D data only; `method="auto"` picks the estimator from pbc and NaN at construction | `master`: `CS_operator_2D_Kernel_Torch`, `get_CS_op` |
| `.apply(data) -> Spectra` | F | Binned auto and cross spectra (real part, as in STL) | `apply` |
| `method=` names: `"auto"`, `"fft"`, `"pbc_mask"`, `"nan_mask"`, `"pbc_nan_mask"` | F | Public names, listed in the manifest | `master` v1.6.0 |
| `_estimators.py`: `estimate_fft`, `estimate_pbc_mask`, `estimate_nan_mask`, `estimate_pbc_nan_mask` | I | The v1.6.0 estimators, on 2D data as in `master` | `master` v1.6.0 |
| `_bins.py`: `log_gaussian_bins`, `legacy_bins` | N | Bins built once | `_build_log_gaussian_bin_masks`, `_build_legacy_bin_masks` |
| 1D, tensor-product and HEALPix spectra, through the same `method=` | I | *(later, LF23, LF24)* | `2D+1D`; `HealpixUpdate` |
| `estimate_tapered`; the v1.6.0 estimators in every geometry; the Rust anafast | I | *(step 2, LF7, LF10, LF5)* | new |

### 12.9 Synthesis

`estel/synthesis/`:

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `synthesize_from_maps(target, *, n_batch=1, pbc_running=None, running_shape=None, init="gaussian", running_mask=None, mean_field=True, compute_PS=None, max_iter=..., seed=0, st_kwargs=None, checkpoint_every=None, checkpoint_dir=None, resume_from=None)` | F |  | `synthesize_from_maps` |
| `synthesize_from_stats(stats, *, n_batch, pbc_running, running_shape=None, init=None, running_mask=None, mean_field=True, max_iter=..., seed=0, checkpoint_every=None, checkpoint_dir=None, resume_from=None)` | F |  | `synthesize_from_stats` |
| `SynthesisResult` | F | Field, loss history, checkpoints, run manifest | new (STL returns a tensor) |
| Checkpoints: `checkpoint_every`, `checkpoint_dir`, `resume_from` | F | The current field and optimiser state are saved at regular intervals; if a run stops, the last checkpoint is usable as a result; a run can restart from a checkpoint instead of a Gaussian start, to add iterations. Not in STL: added in step 1 (Part IV, §22.4) | new |
| `gaussian_start(target, shape, key)` | I | Gaussian start, as in STL | `gaussian_start` |
| `loss(values, target_values, weights)` | I | On the flattened statistics | the loss inside `optimize_*` |
| `run_lbfgs(fun, x0, options)` | I | Calls the backend's built-in optimizer, with frozen masked pixels | `optimize_lbfgs`, `optimize_from_maps`, `optimize_from_stats` |

### 12.10 Utilities

`estel/plot/`, `estel/diagnostics/`, `estel/utils/`. In 1.0.0 these port STL's existing utilities to the new objects; their improvement comes later (LF11).

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `plot.coefficients(stats, family, ...)` | F | Coefficient plots | `ST_Statistics.plot_coeff` |
| `plot.cross_spectrum(spectra, ...)` | F | Spectrum plots | `plot_cross_spectrum` |
| `plot.nyquist_filter(data)` | F | The pre-filter, shown | `apply_nyquist_filter(plot=True)` |
| `diagnostics.describe(obj) -> dict` | F | Plan summary: coefficient counts per family, memory estimate, levels, backend, engine | new |
| `diagnostics.estimate_memory(shape, J, L, ...) -> dict` | F | Peak and output memory **before** building an operator | new |
| `diagnostics.frequency_response(op, N)` | F | Fourier transform of the effective filters (kernels cascaded through the downsamplings) on an N grid, for plots and checks | `filters_fourier`, `_build_wavelet_kernel_from_ifft_crop` |
| `diagnostics.littlewood_paley(op, N)` | F | Littlewood–Paley sum of the bank, built on `frequency_response`: a diagnostic, and the basis of the CI test of §12.5 | new |

All utilities take public objects, return values or a matplotlib `Figure`, never call `plt.show()`, and import no backend.

### 12.11 The oracle

`estel/_spec/`:

| Function | Tag | Purpose |
| --- | --- | --- |
| `spec_wavelets_*`, `spec_scattering(x, J, L, pbc, mask)`, `spec_tensor_prod(...)`, `spec_spectra(...)`, `spec_healpix(...)` | R | Direct sums, explicit loops, float64, no optimisation. Each docstring lists the `M-` IDs it implements. For HEALPix spectra (step 2, LF24), a naive float64 SHT at small sizes, cross-checked against healpy |

## 13. Mathematics extracted from STL, and open points

### 13.1 Principle

In step 1, `maths.md` is written from what STL computes, not from what it should compute. Its reference version during the review is the maths tab of this document. Reading the code turned up the points below; each is recorded in the maths tab and reproduced as is, unless this section says otherwise. Changing any of them is step 2 work.

### 13.2 Open points in STL's mathematics

The points fall into four groups, handled differently. They keep their IDs G1 to G18; the last column links them to the open points OP-n of the maths tab, or to the register of later features (Part III, §16).

**1. Implicit conventions** — applied by STL but documented nowhere. Reproduced, and documented exactly in `maths.md`.

| ID | Point | In step 1 | Related |
| --- | --- | --- | --- |
| G8 | Border crops: four methods in STL (`largest_scale_second_layer`, `layer_flexible`, `fully_flexible`, `zero`) | STL's default per family (fully flexible for planar data), with its width formulas | OP-13 |
| G11 | Gaussian start: the Tukey taper on non-periodic axes, the white-noise fallback | as in STL | syntheses section of the maths tab, to write |
| G12 | HEALPix wavelets: gauges, kernel interpolation, decimation radius and σ | the definitions of the STL-Dev-MPI `CLAUDE.md` §3, which give the same numbers as the PyTorch code (Q10) | HEALPix section, to write |
| G13 | The 1D bump-steerable `xi0`, and the conjugate wavelet used for L = 2 | as in STL | 1D section, to write |
| G14 | The Nyquist pre-filter of syntheses, per geometry | as in STL | syntheses section, to write |
| G15 | `var`: N or N−1, and how mask weights enter | `var` is the non-centred second moment; population normalisation when standardised | OP-11 |
| G16 | Defaults that depend on the data size: J, `n_bins` as a function of log N | as in STL; the default J can violate the size constraint | OP-8 |
| G17 | The regularised division `z / sqrt(\|z\| + 1e-8)` of the cross S1 terms (STL's `divide(epsilon=1e-8, pow=0.5)`), shared by STL and STL-Dev-MPI | recorded as an M-S1 convention | OP-16; a better regularisation later (LF28) |

**2. Inconsistencies** — reproduced where STL is self-consistent, decided where it is not.

| ID | Point | In step 1 | Related |
| --- | --- | --- | --- |
| G1 | Deliverable Table 1 counts 1,474 coefficients for plain 2D; its formula gives 1,476 with the low-pass term | follow STL: 1,474, since plain 2D has no low-pass in STL | LF6 |
| G7 | Masked downsampling: a 0.33 validity threshold in 2D, `_reweight_from_fraction` in HEALPix | one rule per geometry, both documented | OP-12 |
| G18 | Two HEALPix implementations in `HealpixUpdate` (healpix-analyse and `TorchGeometry`) | the kernel class is the reference; ESTEL computes it with tables ported from `TorchGeometry`, and their agreement with healpix-analyse is a test | HEALPix section, to write |

**3. Unclear definitions** — decided case by case.

| ID | Point | In step 1 | Related |
| --- | --- | --- | --- |
| G3 | **The partial order of tensor-product scales.** In a tensor product each scale is a multi-index, one scale per block, for example (j\_space, j\_time). STL orders them componentwise: (a1, b1) ≤ (a2, b2) only if a1 ≤ a2 and b1 ≤ b2. Pairs such as (0, 2) and (1, 1) — finer in space but coarser in time — are not comparable, and STL computes no S3 or S4 coefficient for them. Interactions between a component fine in space and coarse in time and one coarse in space and fine in time are therefore never measured | reproduced, and written down explicitly in the tensor-product section of the maths tab, with this consequence | whether this choice is relevant is examined later (LF27) |
| G4 | `iso`, `angular_ft`, `scale_ft` on tensor products: the product orientation mixes rotations with the direction of propagation of the 1D wavelet | STL's behaviour, documented | definitions in step 2 (LF26); see also OP-5 |
| G5 | Cross-channel S1: allocated as `Nc × Nc`, while S1 is a mean modulus | defined from the code: the mean of z₁·conj(z₂) / √((\|z₁\| + ε)(\|z₂\| + ε)), which reduces to the auto S1 when the channels are equal (maths tab, M-S1) | OP-16 |

**4. Missing features and deferred decisions** — step 2.

| ID | Point | In step 1 | Related |
| --- | --- | --- | --- |
| G2 | The canonical set: complex-conjugate duplicates at `j1 = j2`, and both channel orders, are stored and counted twice by the loss | STL's set, as is | LF20; OP-14 |
| G6 | Normalisation: whether gradients flow through the reference | STL's behaviour | LF21; OP-18 |
| G9 | Low-pass filters in S3/S4 | absent, as in STL | LF6 |
| G10 | Negative bins of the non-periodic spectrum estimator | reproduced | LF7; OP-21 |

## 14. Parity protocol with STL

### 14.1 Principle

Step 1 is validated by comparison: ESTEL and STL run on the same inputs, and their outputs are compared number by number. STL is never a dependency of ESTEL; the comparison lives in a separate harness that is retired after 1.0.0.

### 14.2 The harness

- **Location.** `tools/stl_parity/`, in the ESTEL repository but not shipped and not imported by the library.
- **Environment.** A container image that pins each STL commit of §9.1, healpix-analyse `aac5e26`, and their PyTorch version. The image is archived with the parity report, so the comparison can be rerun after STL's dependencies have moved on.
- **Which STL for what.** `master` for every 2D result (statistics and spectra); `2D+1D` for 1D and tensor-product statistics; `HealpixUpdate` for HEALPix statistics.
- **Inputs.** The reference inputs of Part I, §5.1, the test cases of the STOC deliverable at reduced sizes, and the cases of STL's certification notebooks (2D kernel, smoothing kernel design, 2D cross and power spectra, 1D kernel, 2D+1D kernel, HEALPix kernel), converted into scripts.
- **Parity fixtures.** STL's results enter the tests as fixtures, produced once in the container and stored in the repository:
    - each comparison case is a small script in `tools/stl_parity/cases/` that names its STL branch, its seeded inputs and its parameters;
    - `make parity-fixtures` runs the cases in the container, on a maintainer's machine, and writes `tests/parity/fixtures/<case>.npz`, with `tests/parity/fixtures/MANIFEST.json` recording for each file the STL commit, the container image digest, the case and a checksum;
    - the parity tests compare ESTEL with these files on every PR, in float64, with the tolerances of §14.3, without Docker;
    - STL runs with `torch.set_default_dtype(torch.float64)`. STL builds its wavelet kernels at torch's default dtype (it calls `torch.fft.fftfreq` without a dtype), so without this setting they would be built in float32 while the rest of STL computes in float64. STL's code is unchanged, and the setting is recorded in `MANIFEST.json`;
    - `tests/parity/fixtures/` is protected: only a maintainer regenerates it. An agent that needs a new case writes the case and its test, and asks for the fixture in its PR.

### 14.3 Comparison

- **Same shapes.** In step 1, ESTEL stores statistics in STL's full-grid shapes (§12.7). Outputs are compared element by element, and the NaN positions must match exactly.
- **Precision.** Both libraries run in float64 for parity, which STL supports, so that only rounding separates them.
- **Tolerances.** rtol 1e-10 in float64; 1e-5 in float32, as a secondary check; with an absolute tolerance atol = 0.01 × rtol × max|expected| (1e-12 in float64, 1e-7 in float32), so that values that are zero in theory do not fail on rounding. A larger difference is documented individually in the parity report, explained, and approved by the CODEOWNERS.
- **Gradients.** First-order gradients of a loss are compared between the two libraries; second order is tested in ESTEL only.
- **Syntheses.** Not compared in step 1: synthesis parity and synthesis tests come after 1.0.0 (Part IV, §24.1).

### 14.4 Coverage

A matrix of geometries (1D, 2D, HEALPix, 2D+1D, 2D+1D+1D at small size) crossed with options: pbc on and off, masks, cross channels, each normalisation mode, each reduction, each 2D spectrum estimator, each low-pass setting of the tensor-product operator. A single-block tensor product is also compared with the plain geometry, which is the reference. Each cell of the matrix is a parity case, and a phase of the roadmap (§15) ends only when its cells are green.

### 14.5 From parity to reference values

1. The `_spec` oracle produces the reference values (Part I, §4.2).
2. The same cases are run through STL and compared with the oracle, which checks the oracle itself.
3. The three CODEOWNERS sign off the parity report, `docs/validation/stl_parity.md`, which is archived with this transfer document.
4. The reference values are frozen with the 1.0.0 tag, with `STATISTICS_REFERENCE = 1`.

## 15. Roadmap to 1.0.0

### 15.1 Phases

During step 1, pull requests merge when CI passes; the three CODEOWNERS review step 1 as a whole before the freeze (Part I, §7.1), and the AI review is advisory. Each phase ends when its parity cases (§14.4) and its tests are green.

| Phase | Content | Exit criterion |
| --- | --- | --- |
| P0 Bootstrap | One public repository, under Apache-2.0 (Part I, §1.5); policies, agent files, tier-0 CI, manifest and check tools; `docs/maths.md` exported from the maths tab; parity harness skeleton and its container | all tools run on an empty package |
| P1 Backend | L0 for PyTorch and, experimental, for JAX; the conformance suite on both; the reserved engine and storage interfaces | conformance passes in float32 and float64 |
| P2 Planar | `DataClass`, `Data1D`, `Data2D`, `WaveletOperator`, `ScatteringPlan`, `STOperator`, `STStatistics`, normalisation, reductions; `_spec` for planar data | parity with `master` on 2D statistics and with `2D+1D` on 1D statistics (without spectrum); second-order tests; peak memory measured against STL |
| P3 Tensor products | `DataTensorProd`, `STOperatorTensorProd`, low-pass and box term as in STL; block dispatch along axes, benchmarked against `swapaxes` | parity with `2D+1D` on tensor products |
| P4 Spectra | the 2D estimators of v1.6.0, with automatic selection | parity with `master` on 2D spectra |
| P5 HEALPix | `DataHealpix`, with tables ported from `TorchGeometry` (no healpix-analyse at run time) | parity with `HealpixUpdate` on HEALPix statistics |
| P6 Synthesis and utilities | synthesis API with checkpoints; ported plotting and diagnostics, `frequency_response`, `littlewood_paley` | API tests green (synthesis tests and parity after 1.0.0) |
| P7 Benchmark | the tier-2 benchmark baseline on the agreed cases | baseline recorded; thresholds set (a factor 2 to begin with) |
| P8 Freeze | references from `_spec`, parity report, review by the three CODEOWNERS, the ESTEL reference document in `docs/reference.md`, tag 1.0.0, release on PyPI | 1.0.0 published; the merge policy of Part I, §7.3 switched on |

### 15.2 Work breakdown of step 1

Each task becomes a GitHub issue, to be picked by an agent. A task names its inputs and its acceptance test; it is done when that test passes and CI is green. Tasks of the same phase can run in parallel, unless one names another as input. The maths tab does not need to be complete to start: its 2D part is enough for P0 and P1, and each other section is written at the start of the phase that needs it (T2.0, T3.0, T5.0, T6.0).

| ID | Phase | Task | Inputs | Done when |
| --- | --- | --- | --- | --- |
| T0.1 | P0 | Create the repository with the Apache-2.0 licence files (`LICENSE`, `NOTICE`); set CODEOWNERS (`Eralys`, `jmdelouis`, `pablo-richard`) and branch protection | Part I, §1.5, §7 | repository public, protections active |
| T0.2 | P0 | `pyproject.toml`, lock file, tier-0 workflow | Part I, §1.6, §6.1 | CI green on an empty package |
| T0.3 | P0 | Agent context and policy files, from the drafts | Part I, §8; draft files | the mirror check passes |
| T0.4 | P0 | Check tools: `api_manifest`, `check_layers`, `check_maths_ids`, `version_impact` | Part I, §2.4, §7.2 | each tool fails on a crafted violation |
| T0.5 | P0 | Export `docs/maths.md` from the maths tab (its 2D part, enough to start) | maths tab | `check_maths_ids` runs on the 2D IDs |
| T0.6 | P0 | Parity harness container, with the STL commits and healpix-analyse `aac5e26` | §9.1, §14.2 | STL runs in the container in float64 |
| T0.7 | P0 | CI speed: tests run in parallel (`pytest-xdist`), locally and in CI; a PR that changes only `docs/`, `.github/` or `changes/` runs only the check tools, while every required check still reports a result | Part I, §6.1 | tier-0 wall time on a code PR measured before and after; a documentation-only PR has all required checks green in a few minutes |
| T0.8 | P0 | Parity fixtures: `tools/stl_parity/cases/`, `make parity-fixtures`, `MANIFEST.json`, the comparison helpers of `tests/parity/`, and one example case (the 2D bump-steerable kernel at `master`) | §14.2 | the example fixture is produced in the container and its test passes in tier 0 |
| T1.1 | P1 | `_protocol.py`, with value contracts citing `maths.md` | §12.3; maths tab, M-CONV-2D | contracts documented for every primitive |
| T1.2 | P1 | PyTorch primitives | §12.3 | conformance suite green in float32 and float64 |
| T1.3 | P1 | Experimental JAX backend: every L0 primitive, registered with a warning; tier 0 runs the whole test suite on both backends, in parallel jobs | Part I, §3.1 | conformance suite green on JAX in float32 and float64; all tests run on both backends |
| T1.4 | P1 | Reserved engine interface: engine protocol and registry (only `"python"`, order 2), capability declaration, `EngineUnsupportedError`. The other reserved interfaces moved: storage accessors, the `allreduce` hook and the `io` schema to T2.5, `partition` to T5.1 | §9.4 | reserved tests green |
| T1.5 | P1 | Backend registry and global configuration: `_registry.py`, `_config.py`; public `set_backend`, `get_backend`, `set_default_dtype`, `set_default_device`, listed in the manifest; mixing backends raises `BackendMismatchError` | §12.2; Part I, §2.2.3, §3.1 rule 7 | functions public and tested; "jax" selectable as an experimental backend, with a warning |
| T2.0 | P2 | Write the 1D section of the maths tab, from the STL code at the frozen commits, and add it to `docs/maths.md` | STL `2D+1D`; maths tab | reviewed by a maintainer before T2.4 checks 1D parity |
| T2.1 | P2 | `DataClass`, `Data1D`, `Data2D`, static masks and reweighting maps | §12.4; Part I, §2.3; maths M-DOWN-MASK, M-MASK | unit tests; masks equal STL's |
| T2.2 | P2 | Wavelet banks (port of `bump_steerable_bank`) and H | maths M-WAV-2D, M-H | kernels equal STL's to rounding; Littlewood–Paley CI test |
| T2.3 | P2 | `ScatteringPlan`: index sets, levels, borders, mean denominators | maths §1.3–1.5 | plan tests |
| T2.4 | P2 | `STOperator`: STL's loop structure, in functional style | maths §1.6–1.7 and the 1D section; Part I, §3.1.1 | parity on 1D and 2D statistics, masked cases included (maths tab, OP-26, OP-27) |
| T2.5 | P2 | `STStatistics`, normalisation, reductions, `to_vector(mask_st)`; the reserved storage accessors, the `allreduce` hook of the reductions and the versioned `io` schema (from T1.4) | maths §1.8; §12.7 | parity on normalised and reduced statistics |
| T2.6 | P2 | `_spec` for planar data | maths tab | spec parity |
| T2.7 | P2 | Differentiation tests, to second order | Part I, §3.2 | gradcheck and gradgradcheck green in float64 |
| T2.8 | P2 | Invariance tests | Part I, §5.1 | invariance family green |
| T3.0 | P3 | Write the tensor-product section of the maths tab, with G3 written out explicitly | STL `2D+1D`; §13.2 | reviewed by a maintainer before T3.2 |
| T3.1 | P3 | `DataTensorProd`, dispatch along each block's axes | §12.4 | unit tests; benchmark against `swapaxes` recorded |
| T3.2 | P3 | `STOperatorTensorProd`, low-pass and box term | maths, tensor-product section | parity on tensor products |
| T4.1 | P4 | 2D spectrum estimators and automatic selection | maths §1.9; §12.8 | parity with `master` |
| T5.0 | P5 | Write the HEALPix section of the maths tab | STL `HealpixUpdate`; STL-Dev-MPI `CLAUDE.md` §3 | reviewed by a maintainer before T5.1 |
| T5.1 | P5 | Port of the `TorchGeometry` tables; `DataHealpix`, with its reserved `partition` argument (from T1.4) | maths, HEALPix section; §9.3 | tables equal healpix-analyse's; parity with `HealpixUpdate` |
| T6.0 | P6 | Write the synthesis section of the maths tab: Gaussian start, Nyquist pre-filter, loss | STL `Synthesis.py` | reviewed by a maintainer before T6.1 |
| T6.1 | P6 | Synthesis API with checkpoints | §12.9 | API tests: a run writes checkpoints and resumes from them |
| T6.2 | P6 | Utilities and diagnostics | §12.10 | unit tests |
| T7.1 | P7 | Benchmark cases and baseline | Part I, §6.3 | baseline recorded |
| T8.1 | P8 | Reference values from `_spec`; tolerance calibration | Part I, §5 | references frozen with their checksums |
| T8.2 | P8 | Parity report, review, ESTEL reference document, release on PyPI | §14.5; Part IV, §22.4 | 1.0.0 published |
