# Part II — Step 1: transfer from STL (ESTEL 1.0.0)

Step 1 rebuilds what STL computes inside the architecture of Part I, and proves, case by case, that ESTEL gives the same numbers. It adds no mathematics. Everything it leaves out is listed (§10), and everything it prepares for step 2 is reserved explicitly (§9.4).

[Back to the overview](overview.md)

## Contents of this part

- [9. Sources and scope of the transfer](#9-sources-and-scope-of-the-transfer)
  - [9.1 Sources](#91-sources) · [9.2 The transfer rule](#92-the-transfer-rule) · [9.3 Scope of ESTEL 1.0.0](#93-scope-of-estel-100) · [9.4 Interfaces reserved in 1.0.0 for step 2](#94-interfaces-reserved-in-100-for-step-2)
- [10. What is not transferred](#10-what-is-not-transferred)
- [11. Package tree](#11-package-tree)
- [12. Function inventory per module](#12-function-inventory-per-module)
  - [12.1 Tags](#121-tags) · [12.2 Public namespace and configuration](#122-public-namespace-and-configuration) · [12.3 L0 primitives](#123-l0-primitives) · [12.4 Data classes](#124-data-classes) · [12.5 Wavelets](#125-wavelets) · [12.6 Scattering](#126-scattering) · [12.7 Statistics](#127-statistics) · [12.8 Spectra](#128-spectra) · [12.9 Synthesis](#129-synthesis) · [12.10 Utilities](#1210-utilities) · [12.11 The oracle](#1211-the-oracle)
- [13. Mathematics extracted from STL, and open points](#13-mathematics-extracted-from-stl-and-open-points)
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
- **Where STL fails, ESTEL raises an error.** STL is reproduced wherever it gives a result. Where it gives no usable value at all, such as a crash when the sizes are not divisible by 2^(J−1) (maths tab, OP-8), ESTEL raises a clear error at construction. A mean with no valid pixel is not such a case: as in STL, its coefficient is NaN, decided statically by the plan (OP-3, revised).
- **HEALPix ported, not imported.** `HealpixUpdate` holds two implementations: the kernel class, built on healpix-analyse, and the `TorchGeometry` used by the pyramid engine; its tests show they agree (`test_torch_conv_matches_library`, `test_torch_decimation_matches_library`). Step 1 reproduces the kernel class, computing its convolution and downsampling with NumPy tables ported from `TorchGeometry` (§9.3).
- **Two exceptions.** Step 1 adds two capabilities STL lacks, as infrastructure rather than mathematics: synthesis checkpoints (§12.9) and the experimental JAX backend (Part I, §2.2.5).

### 9.3 Scope of ESTEL 1.0.0

| Area | In 1.0.0, as in STL | Later (Part III register) |
| --- | --- | --- |
| Geometry | `Data1D`, `Data2D`, `DataHealpix`, `DataTensorProd` with 1D and 2D blocks | HEALPix blocks in tensor products (LF9) |
| Data | batch, multi-channel, cross-channel matrix, pbc per block, static NaN masks in 1D, 2D and HEALPix | NaN masks in tensor products (LF8) |
| Wavelets | 2D bump-steerable 5×5; 1D analytic wavelet with its conjugate (L = 2) in tensor products; HEALPix gauge-oriented kernels; smoothing kernel (the anti-aliasing filter of the downsampling) | wavelet optimisation (LF22) |
| Low-pass | φ and box term in tensor products, S1/S2 only | low-pass in every geometry and in S3/S4 (LF6) |
| Statistics | vanilla ScatCov: mean, var, S1–S4; normalisation; iso, angular\_fourier, scale\_cosine; selection | canonical coefficient set (LF20); reductions on tensor products redefined (LF26) |
| Storage | the compact layout (Part III, §19; T2.15): one real buffer holding every coefficient STL computes and only those, real entries as one float, STL's conjugate duplicates kept; STL-shaped full-grid views built on demand, behind accessors that do not depend on storage | packing of the conjugate duplicates, with the canonical set (LF20) |
| Spectra | 2D only, as in `master` v1.6.0: four estimators with automatic selection. `compute_PS` is True by default for 2D data; for other geometries it is unavailable in 1.0.0 and raises a clear error | 1D and tensor-product spectra (LF23); HEALPix spectra (LF24); v1.6.0 estimators in every geometry (LF10); positive non-periodic estimator (LF7) |
| Synthesis | periodic fields only: from maps or stats, `mean_field`, `running_shape`, `nbatch`, Gaussian start; the built-in L-BFGS of PyTorch; checkpoints (new, §12.9). One comparison with STL (T6.1); synthesis tests and parity come after 1.0.0 | non-periodic and masked syntheses (LF47); Rust L-BFGS (LF3); coarse-to-fine, per-iteration filtering, ensembles (LF12) |
| Backends and engines | PyTorch; JAX experimental (every primitive, tested on every PR, not covered by the 1.0.0 guarantees); float32 default, float64 path; Python engine only | JAX stable, with its synthesis optimiser (LF1); Rust engine (LF3); MPI (LF4) |
| Differentiation | first and second order through PyTorch autograd, tested in float64 | second order in the Rust engine (LF15) |
| Utilities | STL's plotting and diagnostics, ported to the new objects | improvements, `gaussianise`, surrogates (LF11) |

**HEALPix in step 1.** STL's HEALPix path calls three classes of healpix-analyse (`HealPixConv`, `HealPixDown`, `HEALPixSHT`). The package is young, has no stated API stability, and pulls heavy unrelated dependencies (`umap-learn` and `healpix-plot`, and through them numba, scikit-learn and cartopy). ESTEL therefore ports what it needs (Part I, §1.6): the convolution and the downsampling are computed with index and weight tables built in NumPy, ported from `TorchGeometry` and applied with `gather_weighted` and `segment_sum`, so they work on every backend. HEALPix spectra are not in 1.0.0 (LF24), so no SHT is ported in step 1. healpix-analyse remains a test-only dependency, pinned to `aac5e26`, to prove parity.

### 9.4 Interfaces reserved in 1.0.0 for step 2

| Reserved interface | In 1.0.0 | Used in step 2 by |
| --- | --- | --- |
| L0 protocol and backend registry | PyTorch; `"jax"` registered as experimental, with a warning | the JAX backend (Part III, §17) |
| `engine=` argument, engine registry and protocol, `diff_order` | only `"python"`, order 2 | the Rust engine (Part III, §18) |
| `STStatistics` accessors (`family`, `select`, `to_vector`, `full_grid`) | backed by the compact layout (Part III, §19; T2.15), the full-grid tensors being views built on demand | the canonical set (LF20, Part III, §19.8), with no API change |
| `DataHealpix(cell_ids=…, partition=…)` | whole sky by default; partial ranges accepted | MPI in process (Part III, §18.3) |
| `_global_sum`, internal: the hook between the weighted pixel sums of the means and their division, called once per level on the stacked sums (issue #17) | identity | MPI in process, as an MPI Allreduce with a sum (Part III, §18.3) |
| `estel.io`, with a versioned schema | ESTEL format | the JSON formats of STL-Dev-MPI (Part III, §18.3) |
| `method=` of `CrossSpectrumOperator` for 1D, tensor-product and HEALPix data | accepted names, raising a clear error | LF23, LF24 |

Each reserved interface has a test in 1.0.0 that exercises its trivial implementation, so it cannot rot before it is used.

## 10. What is not transferred

| STL code | Why | What replaces it |
| --- | --- | --- |
| `STL_2D_FFT_Torch` (Fourier-space wavelets) | obsolete in ESTEL | real-space kernels; its `bump_steerable_bank` is ported to build them (maths tab, OP-23) |
| The `has_fewer_convolutions` path and the `SC` switch | only vanilla ScatCov is kept; also removed from the Rust engine's bridge | — |
| The dead modules `StlData`, `DataType1`, `DataType2`, `WaveletTransform`, `cs_op.py`, `LossOptim` | unused or broken | — |
| `inplace=` arguments | incompatible with JAX and second-order differentiation | functional updates (`with_array`) |
| `mean`, `cov`, `standardize` duplicated in each wavelet operator | duplication | one implementation in the mother class `Data` |
| Dynamic class creation in the tensor-product class (`self.__class__ = …`) | opaque to readers and tools | composition: a tensor-product object holds one array and the geometry of each block |
| Exact class-name checks (`type(data).__name__ != …`, three places) | break subclasses | `isinstance` or declared capabilities |
| The mutable `conv_history` list on data objects | hidden state | crop widths in the static plan |
| The Python pyramid engine of `HealpixUpdate` | superseded by its validated Rust port | the Rust engine of STL-Dev-MPI (Part III, §18) |
| The runtime dependency on healpix-analyse | young research code with heavy dependencies (Part I, §1.6) | the ported HEALPix convolution and downsampling (§9.3) |
| The 1D and tensor-product spectra of `2D+1D`; the HEALPix spectra of `HealpixUpdate` | outside the scope of 1.0.0 (Part IV, §22, C22) | later: LF23, LF24 |
| Morlet wavelets (`WType="Morlet"`) of the 2D kernel class | only bump-steerable wavelets are transferred; a general optimisation of the wavelets comes later, and may or may not include them | bump-steerable wavelets; LF22 (maths tab, OP-24) |
| The public wavelet operator (`data.get_wavelet_op()`) | ESTEL computes scattering statistics; the wavelet operator is internal (Part IV, §22, "Wavelet operator, internal") | `STOperator`, which uses the internal `WaveletOperator` (§12.5) |

WPH coefficients were declared in STL but never implemented; they are out of the scope of ESTEL.

## 11. Package tree

The tree of ESTEL 1.0.0. Items marked `(step 2)` are created later; items marked `(reserved)` exist in 1.0.0 with a trivial implementation (§9.4). LFn refers to the register of later features (Part III, §16).

```text
estel/                             # one repository for Python and Rust (Part IV, §22, C11)
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
│   ├── reference.md               # the ESTEL reference document, written with 1.0.0
│   ├── register/                  # PROTECTED: the register of later features, one card
│   │                              # per entry, its view and its decision log
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
│   ├── data/                      # _base.py (Data), _planar.py, _healpix.py,
│   │                              # _tensor_prod.py, _masks.py
│   ├── wavelets/                  # _operator.py, _banks.py, _healpix_bank.py,
│   │                              # _healpix_geometry.py (ported from TorchGeometry)
│   ├── scattering/                # _plan.py, operator.py, operator_tensor_prod.py
│   ├── statistics/                # _statistics.py, _grid.py, _reductions.py, _normalization.py,
│   │                              # _layout.py (the compact layout, T2.15)
│   ├── spectra/                   # operator.py, _estimators.py (2D), _bins.py
│   ├── synthesis/                 # api.py, _loss.py, _init.py, _checkpoints.py
│   ├── functional/                # pure functions: scattering(), cross_spectrum(), loss()
│   ├── io/                        # _schema.py: the versioned schema, used by STStatistics.save
│   │                              # and load; STL-Dev-MPI JSON (step 2)
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
| **G** | Geometry hook: declared by the mother class `Data` and overridden by a subclass — the *only* methods subclasses override |
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
| `gather_weighted(x, axis, index, weight)` | P | `y[..., p, k] = Σ_m weight[p,k,m] · x[..., index[p,k,m]]`: a sparse matrix in padded CSR form, for the static linear maps of the statistics (normalisation, reductions, selection; Part III, §19.6) and for the HEALPix stencils and decimation of step 1 | `TorchGeometry` |
| `segment_sum(x, axis, segment_ids, num_segments)` | P | Bin averages, HEALPix parent binning, reductions | bin masks, `_bin_to_parents` |
| `scatter_to_grid(values, index, shape, fill)` | P | Builds a new array on the complete grid of indices: values at the static positions `index`, `fill` (NaN) everywhere else; out of place. Its inverse, `gather_from_grid(grid, index)`, reads the existing coefficients back. Builds the full-grid views of the statistics on demand, for `full_grid`, from the compact buffer; never called inside the computation (Part I, §3.1.1; Part III, §19.5) | new |
| `complex_to_real_pairs(z)` / `real_pairs_to_complex(r)` | P | Complex values as trailing (re, im) floats, and back | new |
| `stop_gradient(x)` | P | `x.detach()` / `jax.lax.stop_gradient` | `.detach()` calls |
| `random_normal(key, shape, dtype)`, `make_key(seed)` | P | Explicit keys on every backend | `torch.randn` in `Synthesis.py` |
| `jit(fn, static_argnames)` | P | Identity in PyTorch; `jax.jit` in the JAX backend | new |
| `register_pytree(cls, flatten, unflatten)` | P | Registers a class as a pytree: on JAX through `jax.tree_util`; on PyTorch through `torch.utils._pytree`, imported lazily, with a warning if no such API exists. Used for `Data`, and later `STStatistics` (Part I, §3.1, rule 9) | new |
| `value_and_grad(fn)`, `vjp`, `jvp`, `hvp` | P | Synthesis and the differentiation tests; `torch.func` / `jax` | `loss.backward()` |
| `lbfgs(...)` | P | The built-in L-BFGS of each backend: `torch.optim.LBFGS(line_search_fn="strong_wolfe")`; `optax.lbfgs` in step 2; the Rust L-BFGS (`impl="rust"`) when the Rust engine is selected (Part III, §18.2) | `optimize_lbfgs` |
| `deterministic(enabled)` | P | `torch.use_deterministic_algorithms`; on JAX, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, set before JAX starts. On in tests | new |

### 12.4 Data classes

`estel/data/`. The mother class holds all shared logic; subclasses override only the **G** hooks. Those of filtering and downsampling (`_conv`, `_downsample`, `_crop_weights`, `_lowpass`) are called by `WaveletOperator` only (§12.5; Part I, §2.3); `_nyquist_filter` and `_spectrum_geometry` serve the syntheses and the spectra.

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `Data(array, mask=None, pbc=..., ...)` | F | Frozen dataclass and pytree. Validates once (rule B1). NaNs in `array` become `mask` at construction; the mask and its reweighting maps are then static (Part I, §2.3). `pbc` must be an explicit boolean | `Base_DataClass.__post_init__` |
| `.array`, `.mask`, `.shape`, `.n_batch`, `.n_channels`, `.backend` | F | Read-only accessors | `Base_DataClass` |
| `.copy(empty=False)`, `.__getitem__` (batch and channel slicing only) | F |  | `Base_DataClass.copy`, `__getitem__` |
| `.with_array(new_array)` | F | Functional replacement of in-place updates | replaces `inplace=True` |
| `modulus(x)`, `divide(x, y, eps, pow)` | I | Safe modulus and the regularised division of STL, differentiable twice (Part I, §3.2.3) | `modulus`, `divide` |
| `mean_valid(x, weights, axes)`, `cov_valid(x, y, weights, axes)` | I | Weighted means over valid pixels, one implementation for every geometry; the denominators are precomputed in the plan. `cov_valid` returns the raw second moment in 1.0.0, as STL; it accepts optional first moments, used by the centred covariances of step 2 (LF33) | `mean`, `cov`, `square_mean` of each wavelet operator (three copies) |
| `standardize`, `unstandardize` | I |  | `standardize` |
| `propagate_mask_conv(mask, footprint)` | I | An output is masked when its kernel footprint touches a masked input | `_build_reweighting_maps…`, `_find_mask` |
| `_spatial_axes()` | G | Axes the geometry acts on | the `# conv_dims dependant` markers |
| `_conv(x, tables, level)` | G | Filtering by the wavelets at `level`, with the static tables of that level given by `WaveletOperator`. Planar: `pad` + `conv_axes` with the kernel bank, the same at every level. HEALPix: `gather_weighted` with the stencils of the level, ported from `TorchGeometry` | `apply` of each wavelet operator |
| `_downsample(x, weights, level)` | G | One step from `level` to `level + 1`, with the static weights of the step given by `WaveletOperator`. Planar: strided normalised convolution with the smoothing kernel. HEALPix: decimation with `gather_weighted`, then parent binning with `segment_sum` | `_downsample_tensor`, `_down_one_level`, `_downsample_masked` |
| `_crop_weights(level, border)` | G | Planar: removal of non-periodic borders, widths from the plan. HEALPix: identity | `_crop`, `_get_crop_border_size_*` |
| `_lowpass(x, plan, block)` | G | Applies the low-pass filter described by the plan. In 1.0.0, as in STL: the data downsampled J times, returned unchanged, since the low-pass filter is the cascade of the smoothing kernels. Later: the smoothing kernel at level J−1 without the final downsampling, or a dedicated kernel (LF6, Part III §20.10) | `Tensor_Product_Kernel_Torch.lowpass` |
| `_nyquist_filter(x)` | G | Pre-filter before synthesis | `apply_nyquist_filter`, `apply_bandlimit` |
| `_spectrum_geometry()` | G | 2D: FFT frequency grids. Other geometries: later (LF23, LF24) | `CS_operator_*` initialisation |
| `Data1D`, `Data2D` | F | Explicit planar classes sharing `_planar.py`, differing by `ndim` | `STL_1D_Kernel_Torch`, `STL_2D_Kernel_Torch` |
| `DataHealpix(array, depth, cell_ids=None, partition=None, ...)` | F | NESTED ordering, partial sky through `cell_ids`; `depth`, not `nside` (rule A4). `partition` is reserved (§9.4) | `STL_Healpix_Kernel_Torch` |
| `DataTensorProd(array, blocks, mask=None, *, pbc)` | F | `blocks = (Data1D, Data2D, …)` in axis order. It **composes** block geometries (`BlockSpec`: class, axes, pbc) instead of creating classes at run time: one array, all blocks on the same backend | `STL_TensorProd_Kernel_Torch`, `DimHandler` |
| `DataTensorProd._conv`, `_downsample`, `_crop_weights` | G | Dispatch to the geometry of each block, applied along that block's own axes, without `swapaxes`. Phase P3 benchmarks it against STL's `swapaxes`, which is kept only where it is measurably faster | `_as_data_class_of`, `_from_data_class_of` |

### 12.5 Wavelets

`estel/wavelets/`:

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `WaveletOperator(data, J, L=None, wavelet="bump_steerable")` | I | Internal, in `wavelets/_operator.py`: not public, unlike STL's `data.get_wavelet_op()` (Part IV, §22, "Wavelet operator, internal"). Generic operator, without a data class in its name, and the only way to the geometry (Part I, §2.3): level by level, the convolution by the wavelets, the downsampling, the low-pass filter, the shape of each level, and the rules of borders (M-BORDER) and of mask propagation (M-MASK). `STOperator` builds one, and its plan and loop go through it; a tensor product composes those of its blocks. A static part in the plan, arrays in the filters, pure functions. Builds the bank with a branch on `data.ndim` (1D, 2D) or the HEALPix geometry. Validates its parameters at construction (J compatible with the data size, an integer L ≥ 1, as STL accepts) | `WaveletOperator2Dkernel_torch`, `…1Dkernel_torch`, `…HealpixKernel_torch` |
| `.apply(x, j)` | I | ψ\_j ⋆ x, on a map at level j: complex output, L orientations | `apply` |
| `.lowpass(x, j)` | I | The last step of the low-pass filter φ\_j, from a map at level j − 1. In 1.0.0, as in STL, one smoothing step with its downsampling, so that φ\_j is the cascade of the smoothing kernels (M-LP); in step 2, the smoothing without the final downsampling (LF6). Implemented with its first user, the tensor products (T3.2) | `lowpass`, `lowpass_dg` |
| `.downsample(x, level, history=())` | I | One normalised downsampling step, from `level` to `level + 1`, with the masks and reweighting maps of the plan (M-DOWN, M-DOWN-MASK). With missing values these maps depend on the map's history, the scales of the convolutions applied to it: `()` for the field, `(j,)` for first-layer data convolved at scale j (the two families of maps of M-DOWN-MASK) | `downsample` |
| `bump_steerable_2d(L, size)`, `bump_steerable_1d(xi0, size, conjugate)` | N | Kernels built in NumPy, cropped to 5×5 or 5 taps. STL's kernels are reproduced as they are. They do not satisfy the Littlewood–Paley condition exactly, so 1.0.0 does not test it; a Littlewood–Paley requirement, tested in CI, comes with the wavelet optimisation (LF22); that test includes the low-pass filter, without which the Littlewood–Paley sum is not meaningful (LF6). Every band-pass kernel sums to zero by construction (its mean is subtracted), and the construction checks it: the centring rule of step 2 relies on it (Part III, §20.11) | `_build_bump_steerable_wavelet_kernel`, `bump_steerable_bank` (ported from the Fourier-space class) |
| `smoothing_kernel(ndim)` | N | Smoothing kernel, the anti-aliasing filter applied before each downsampling; STL's unused `sigma_smooth` is dropped | `_get_smooth_kernel` |
| `healpix_kernel_weights(depth, L, kernel_size)` | N | Gauge-oriented kernel weights | `_wavelet_kernel` |
| `healpix_stencils(cell_ids, depth)`, `healpix_decimation(cell_ids, depth)` | N | Index and weight tables for `gather_weighted`, ported from `TorchGeometry` (step 1) | `TorchGeometry`, `stl-core` |

`frequency_response` and `littlewood_paley`, which only serve plots and checks, are in `estel.diagnostics` (§12.10).

### 12.6 Scattering

`estel/scattering/`:

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `STOperator(data, J=None, L=None, *, norm="self", ref_batch_mean=False, standardize=False, cross_matrix=None, compute_PS=None, engine="python")` | F | Single-domain operator (1D, 2D, HEALPix). Every computation option is set here, once. Reductions and selections are `STStatistics` methods only (rule A6). `norm` is `"self"`, `"ref"` or `"none"` (STL's `store_ref`, `load_ref` and `vanilla`). `ref_batch_mean` (STL's `norm_batch_mean`) averages the `norm="self"` reference over the batch; False by default, as in STL (maths tab, M-NORM). `compute_PS` is True by default for 2D data and unavailable elsewhere in 1.0.0. `engine` is reserved: only `"python"`, checked in the engine registry. `standardize` removes the mean and divides by the standard deviation before the transform, as STL's apply(standardize=True); both values are kept with the statistics. **Channels:** the number of channels is read from the example data and fixed at construction, with `cross_matrix` (STL's `compute_cross_matrix`): `None` means every pair c₁ ≤ c₂, as STL's code; a given matrix must be Nc × Nc, else a clear error; applying the operator to data with another number of channels raises a clear error | `ST_Operator.__init__` |
| `STOperator.apply(data, ref=None) -> STStatistics` | F | The only call-time arguments: the data and an optional reference. The normalisation kind is fixed at construction. As in STL, an operator built with `norm="self"` keeps the reference of the last statistics it computed, replacing the previous one with a `UserWarning`; the reference is internal (`_NormReference`). With `norm="ref"`, `ref` is an `STStatistics` computed with `norm="self"`; without it, the operator uses the reference it holds. With `norm="self"` the reference is part of the computation and gradients flow through it; with `norm="ref"` it is a constant | `ST_Operator.apply` |
| `STOperator.from_statistics(stats)` | F | Rebuild the operator that produced a statistics object: the same plan, holding the reference the statistics were normalised with. With `STOperator(data, ...)`, the only public constructors | `from_ST_Statistics` |
| `STOperator.plan`, `STOperator.filters` | F | The plan and the filters of the operator, the arguments of `estel.functional.scattering` | new |
| `STOperatorTensorProd(data, J=(…), L=(…), *, lowpass="ST", …)` | F | Tensor-product operator, with STL's low-pass options (`"ST"`, `"PS"`, both, or none). Overrides only the hooks below | `ST_Operator_TensorProd` |
| `_scale_index_set(plan)` | I | Mother: `j1 ≤ j2 ≤ j3` on integers. Tensor product: the componentwise partial order on multi-indices (maths tab, M-IDX-TP), plus the low-pass index per block | `_scales_S1_S2`, the loops of `_compute_ST_loop` |
| `_levels(plan)` | I | Resolution levels in execution order. Mother: `0…J`; tensor product: the product of block levels | `_compute_ST_loop` |
| `_first_layer(x, plan)` | I | W\_j x and \|W\_j x\| at every level, built functionally | `_compute_ST_loop` |
| `_second_layer(u, plan, level)` | I | The S3 and S4 means at one level, following the structure of STL's loop (Part I, §3.1, rule 8), returned as lists and stacked; no preallocated tensor filled in place (Part I, §3.1.1) | the `cov` calls of the loop |
| `_normalize(values, plan, ref)` | I | Normalisation only: `norm="self"`, `norm="ref"`, batch mean, with the optional reference. Reductions are `STStatistics` methods. The reference object stores the normalisation scales and, from step 2, the first moments used by centring; the two options are independent (Part III, §20.11) | `_finalize_compute`, `to_norm`, `_normalize_scatcov` |
| `STPlan` | F/N | Static description: J, L, index sets, the static part of the `WaveletOperator` (the levels and their shapes, the low-pass filter and how it is realised, the rules of borders and masks, and the tables they give), crop widths derived from the plan's list of operations, masks and reweighting maps, mean denominators, the channel pairs, the normalisation and standardisation options, the centring of each covariance (`none` in 1.0.0) and the window policy; hashable. Public and opaque: its name is in the manifest, it has no public field | new |
| `STFilters` | F/N | The numerical arrays of an operator, on its backend and dtype: the arrays of the `WaveletOperator` (wavelets, smoothing kernel, geometry tables) and everything derived to apply them (propagated masks, mean weights, border windows). Public and opaque, as `STPlan` | new |
| `scattering(data, plan, filters, ref=None) -> STStatistics` (in `functional`) | F | The pure function behind `apply`, public for `jit`, `grad` and `vmap` users, and for downstream implicit differentiation. `plan` and `filters` are `op.plan` and `op.filters`; `ref` as in `apply`, required with `norm="ref"`. It keeps nothing: the operator, not the function, stores the `norm="self"` reference | new |

### 12.7 Statistics

`estel/statistics/`. The accessors do not depend on how statistics are stored: they read the compact layout (Part III, §19; T2.15), from which `full_grid` rebuilds STL-shaped full-grid tensors on demand, so the storage changed with no API change. In ESTEL, "reduction" means only `iso`, `angular_fourier` and `scale_cosine`.

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `STStatistics` | F | Returned by `STOperator.apply`, `estel.functional.scattering` and `STStatistics.load`; not built directly. Pytree: arrays as leaves; plan, layout and reductions applied static | `ST_Statistics.__init__` |
| `.full_grid(family)` | F | The tensor of `family` (`"mean"`, `"var"`, `"S1"` … `"S4"`, later `"PS"`), in STL's shape, NaN where no coefficient exists. A view rebuilt on demand from the compact layout, never inside the computation (Part III, §19.5). For tensor products, a view over 2 GiB is refused, with an error that says what to use instead (`to_vector`, `select`); the full grid plays no role in their computation or storage (Part IV, §22, row of 22 November 2026; LF65) | attributes `.S1 … .S4` |
| `.select(family=None, **index) -> STStatistics` | F | One selection method. `family` is a name (`"S3"`) or a sequence of names, `None` for all. `**index` restricts along named axes: an int drops the axis, a sequence or a slice keeps it; `channel=(a, b)` is shorthand for `c1=a, c2=b`. A family that does not have a named axis is left out of the result: `select(j3=2)` returns S3 and S4 restricted to j3 = 2, without mean, var, S1 or S2. An axis name that no selected family has raises an error listing the valid names. The axes are `(c,)` for mean and var, `(c1, c2, j, l)` for S1 and S2, `(c1, c2, j2, j3, l2, l3)` for S3 and `(c1, c2, j1, j2, j3, l1, l2, l3)` for S4; `iso` turns the orientations of S3 into `dl` and those of S4 into `dl2, dl3`, `angular_fourier` prefixes the axes it transforms with `h`, `scale_cosine` turns the scales into `(k, a)` and `(k, a, b)`. The result keeps every `STStatistics` method (`to_vector`, `reduce`, `full_grid`, `save`). On tensor products, `select` takes `family` and the channel axes only (`c1`, `c2`, `channel`); a scale or orientation axis raises an error, since naming the axes of the blocks is LF44 | `select` (a stub in STL) |
| `.to_vector(mask_st=None)` | F | Flat real vector for losses, STL's `to_flatten(flatten_complex=True)`: families in the order mean, var, S1 … S4, each batch element in turn, complex values as (real, imaginary) pairs, of which only the real part of S1 and S2 (and PS) is kept, as in STL, including for the cross-channel terms (maths tab, M-FLAT, OP-37). The positions where no coefficient exists (structural, empty means, OP-30, missing scales of `scale_cosine`) are left out, decided statically. `mask_st`, STL's name, is an optional boolean array over that vector: True keeps a coefficient, as in STL. It drops coefficients found unreliable by a boolean test; in a synthesis it is computed once on the target and reused at every iteration | `to_flatten`, `mask_st` |
| `.reduce(*, iso=False, angular_fourier=False, angular_harmonics=None, scale_cosine=False, scale_harmonics=None, dj=None)` | F | Applies the selected reductions in the order of M-NORM (`iso`, then `angular_fourier`, then `scale_cosine`) in one call, and returns a new `STStatistics`. `angular_harmonics` (STL's `harmonics_angle`, 1 to L), `scale_harmonics` (STL's `harmonics_scale`) and `dj` (1 to J) default to all; `angular_fourier` is allowed without `iso`, and `scale_cosine` needs `iso`, as in STL; reduced statistics are not reduced again | `to_iso`, `to_angular_ft`, `to_scale_ft` |
| `.save(path)`, `STStatistics.load(path)` | F | Through `estel.io`, which owns the versioned schema (`estel/io/_schema.py`): one npz archive with the arrays and a JSON document holding the schema version, the reference counters, the plan's parameters, the layout and the reductions applied. There is no `statistics/io.py` | new |
| `.to_xarray()` | F | *(step 2, LF13)* Export for inspection | new |
| `Layout`, `Segment` | I/N | The compact layout (Part III, §19.4; T2.15), in `_layout.py`: the buffer's segments, one per family and level, with their named index tables; static and hashable. Normalisation, reductions and selection are static linear maps on the buffer (Part III, §19.6) | new |

### 12.8 Spectra

`estel/spectra/`. In 1.0.0 only 2D spectra are provided, as in `master` v1.6.0; the other geometries come later (Part IV, §22, C22):

| Function | Tag | Purpose | Origin |
| --- | --- | --- | --- |
| `CrossSpectrumOperator(data, *, n_bins=None, method="auto", cross_matrix=None)` | F | Generic operator, with a single `method=` argument for every geometry. In 1.0.0 it accepts 2D data only; `method="auto"` picks the estimator from pbc and NaN at construction | `master`: `CS_operator_2D_Kernel_Torch`, `get_CS_op` |
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
| `synthesize_from_maps(target, *, n_batch=1, running_shape=None, init="gaussian", mean_field=True, compute_PS=None, max_iter=..., seed=0, st_kwargs=None, checkpoint_every=None, checkpoint_dir=None, resume_from=None)` | F | Periodic fields only in 1.0.0: a target with missing values or with `pbc=False` raises a clear error (Part III, §16, LF47) | `synthesize_from_maps` |
| `synthesize_from_stats(stats, *, n_batch, running_shape=None, init=None, mean_field=True, max_iter=..., seed=0, checkpoint_every=None, checkpoint_dir=None, resume_from=None)` | F | Periodic running fields only in 1.0.0 (LF47) | `synthesize_from_stats` |
| `SynthesisResult` | F | Field, loss history, checkpoints, run manifest | new (STL returns a tensor) |
| Checkpoints: `checkpoint_every`, `checkpoint_dir`, `resume_from` | F | The current field and optimiser state are saved at regular intervals; if a run stops, the last checkpoint is usable as a result; a run can restart from a checkpoint instead of a Gaussian start, to add iterations. Not in STL: added in step 1 (Part IV, §22, review of 12 November 2026) | new |
| `gaussian_start(target, shape, key)` | I | Gaussian start, as in STL | `gaussian_start` |
| `loss(values, target_values, weights)` | I | On the flattened statistics | the loss inside `optimize_*` |
| `run_lbfgs(fun, x0, options)` | I | Calls the backend's built-in optimizer | `optimize_lbfgs`, `optimize_from_maps`, `optimize_from_stats` |

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
| `diagnostics.littlewood_paley(op, N)` | F | Littlewood–Paley sum of the bank, built on `frequency_response`: a diagnostic, without a pass/fail criterion in 1.0.0; the basis of the CI test that comes with LF22. It reports the sum with and without the low-pass filter, in the convention of the maths tab (Λ) | new |
| `diagnostics.energy_partition(op, N)` | F | The per-level energy partition P of the smoothing kernel and the wavelets (maths tab, "Filters and scales"); shows where energy is lost or counted twice between scales | new |
| `diagnostics.aliasing(op, N)` | F | The fraction of energy that each downsampling folds back (Part III, §20.10) | new |

All utilities take public objects, return values or a matplotlib `Figure`, never call `plt.show()`, and import no backend.

### 12.11 The oracle

`estel/_spec/`:

| Function | Tag | Purpose |
| --- | --- | --- |
| `spec_wavelets_*`, `spec_scattering(x, J, L, pbc, mask)`, `spec_tensor_prod(...)`, `spec_spectra(...)`, `spec_healpix(...)` | R | Direct sums, explicit loops, float64, no optimisation. Each docstring lists the `M-` IDs it implements. For HEALPix spectra (step 2, LF24), a naive float64 SHT at small sizes, cross-checked against healpy |

## 13. Mathematics extracted from STL, and open points

In step 1, `maths.md` is written from what STL computes, not from what it should compute. Reading the code turned up open points: each is recorded in the maths tab (OP-n, section by section, with its status) and reproduced as is, unless its status says otherwise. Changing any of them is step 2 work.

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
- **Tolerances.** rtol 1e-10 in float64; 1e-5 in float32, as a secondary check; with an absolute tolerance atol = 0.1 × rtol × max|expected| (1e-11 in float64, 1e-6 in float32), so that values that are zero in theory do not fail on rounding. For `mean`, whose theoretical value may be zero, the scale is the standard deviation of the data: atol = 0.1 × rtol × sqrt(max(var)). A larger difference is documented individually in the parity report, explained, and approved by the CODEOWNERS.
- **Gradients.** First-order gradients of a loss are compared between the two libraries; second order is tested in ESTEL only.
- **Syntheses.** One comparison in step 1: for periodic planar data, a synthesis started from the same field as STL, with the same settings, in float64, follows STL's loss over its first iterations, within a tolerance set by the task (T6.1). Full synthesis parity and synthesis tests come after 1.0.0 (Part III, §16).

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
| P2 Planar | `Data`, `Data1D`, `Data2D`, `WaveletOperator`, `STPlan`, `STOperator`, `STStatistics`, normalisation, reductions; `_spec` for planar data; then, in this order: the benchmark (T2.11), the optimisations that leave every value within the parity tolerance (T2.12 to T2.14; Part I, §3.1, rule 8), the consistency of the estimators (T2.16), and the periodic planar syntheses (T6.0, T6.1) | parity with `master` on 2D statistics and with `2D+1D` on 1D statistics (without spectrum); second-order tests; peak memory measured against STL; the benchmark recorded before and after each optimisation, gradients included; a planar synthesis follows STL's loss (T6.1) |
| P3 Tensor products | the compact layout (T2.15), before `STOperatorTensorProd`; `DataTensorProd`, `STOperatorTensorProd`, low-pass and box term as in STL; block dispatch along axes, benchmarked against `swapaxes` | parity with `2D+1D` on tensor products |
| P4 Spectra | the 2D estimators of v1.6.0, with automatic selection | parity with `master` on 2D spectra |
| P5 HEALPix | `DataHealpix`, with tables ported from `TorchGeometry` (no healpix-analyse at run time) | parity with `HealpixUpdate` on HEALPix statistics |
| P6 Synthesis and utilities | synthesis API with checkpoints, done in P2 for periodic planar data (T6.0, T6.1), each later geometry adding its syntheses in its phase; ported plotting and diagnostics, `frequency_response`, `littlewood_paley` | API tests green (synthesis tests and parity after 1.0.0) |
| P8 Freeze | references from `_spec`, parity report, review by the three CODEOWNERS, the ESTEL reference document in `docs/reference.md`, tag 1.0.0, release on PyPI | 1.0.0 published; the merge policy of Part I, §7.3 switched on |

### 15.2 Work breakdown of step 1

Each task becomes a GitHub issue, to be picked by an agent. A task names its inputs and its acceptance test; it is done when that test passes and CI is green. Tasks of the same phase can run in parallel, unless one names another as input. The maths tab does not need to be complete to start: its 2D part is enough for P0 and P1, and each other section is written at the start of the phase that needs it (T2.0, T3.0, T5.0, T6.0). The column "Issue" gives the GitHub issue of a task; "Merged" gives the date and the pull request that completed it, and is filled at the end of each working cycle.

| ID | Phase | Task | Inputs | Done when | Issue | Merged |
| --- | --- | --- | --- | --- | --- | --- |
| T0.1 | P0 | Create the repository with the Apache-2.0 licence files (`LICENSE`, `NOTICE`); set CODEOWNERS (`Eralys`, `jmdelouis`, `pablo-richard`) and branch protection | Part I, §1.5, §7 | repository public, protections active | — | 12 Nov 2026, first commits |
| T0.2 | P0 | `pyproject.toml`, lock file, tier-0 workflow | Part I, §1.6, §6.1 | CI green on an empty package | — | 12 Nov 2026, #1 |
| T0.3 | P0 | Agent context and policy files, from the drafts | Part I, §8; draft files | the mirror check passes | — | 12 Nov 2026, #4 |
| T0.4 | P0 | Check tools: `api_manifest`, `check_layers`, `check_maths_ids`, `version_impact` | Part I, §2.4, §7.2 | each tool fails on a crafted violation | — | 12 Nov 2026, #2 |
| T0.5 | P0 | Export `docs/maths.md` from the maths tab (its 2D part, enough to start) | maths tab | `check_maths_ids` runs on the 2D IDs | — | 12 Nov 2026, first commits |
| T0.6 | P0 | Parity harness container, with the STL commits and healpix-analyse `aac5e26` | §9.1, §14.2 | STL runs in the container in float64 | #7 | 12 Nov 2026, #34 |
| T0.7 | P0 | CI speed: tests run in parallel (`pytest-xdist`), locally and in CI; a PR that changes only `docs/`, `.github/` or `changes/` runs only the check tools, while every required check still reports a result | Part I, §6.1 | tier-0 wall time on a code PR measured before and after; a documentation-only PR has all required checks green in a few minutes | #45 | 13 Nov 2026, #46 |
| T0.8 | P0 | Parity fixtures: `tools/stl_parity/cases/`, `make parity-fixtures`, `MANIFEST.json`, the comparison helpers of `tests/parity/`, and one example case (the 2D bump-steerable kernel at `master`) | §14.2 | the example fixture is produced in the container and its test passes in tier 0 | #56 | 13 Nov 2026, #57 |
| T1.1 | P1 | `_protocol.py`, with value contracts citing `maths.md` | §12.3; maths tab, M-CONV-2D | contracts documented for every primitive | #8 | 12 Nov 2026, #35 |
| T1.2 | P1 | PyTorch primitives | §12.3 | conformance suite green in float32 and float64 | #9 | 12 Nov 2026, #37 |
| T1.3 | P1 | Experimental JAX backend: every L0 primitive, registered with a warning; tier 0 runs the whole test suite on both backends, in parallel jobs | Part I, §3.1 | conformance suite green on JAX in float32 and float64; all tests run on both backends | #10 | 13 Nov 2026, #41 |
| T1.4 | P1 | Reserved engine interface: engine protocol and registry (only `"python"`, order 2), capability declaration, `EngineUnsupportedError`. The other reserved interfaces moved: storage accessors, the `_global_sum` hook and the `io` schema to T2.5, `partition` to T5.1 | §9.4 | reserved tests green | #11 | 13 Nov 2026, #47 |
| T1.5 | P1 | Backend registry and global configuration: `_registry.py`, `_config.py`; public `set_backend`, `get_backend`, `set_default_dtype`, `set_default_device`, listed in the manifest; mixing backends raises `BackendMismatchError` | §12.2; Part I, §2.2.3, §3.1 rule 7 | functions public and tested; "jax" selectable as an experimental backend, with a warning | #39 | 13 Nov 2026, #54 |
| T2.0 | P2 | Write the 1D section of the maths tab, from the STL code at the frozen commits, and add it to `docs/maths.md` | STL `2D+1D`; maths tab | reviewed by a maintainer before T2.4 checks 1D parity | #12 | 13 Nov 2026, #52 |
| T2.1 | P2 | `Data`, `Data1D`, `Data2D`, static masks and reweighting maps | §12.4; Part I, §2.3; maths M-DOWN-MASK, M-MASK | unit tests; masks equal STL's | #13 | 13 Nov 2026, #58 |
| T2.2 | P2 | Wavelet banks, 2D and 1D (port of `bump_steerable_bank` and `bump_steerable_1d`), and the smoothing kernel | maths M-WAV-2D, M-SMOOTH-2D | 1D and 2D kernels equal STL's to rounding (parity fixtures: L = 4 in 2D, L = 1 and 2 in 1D) | #14 | 13 Nov 2026, #61 |
| T2.3 | P2 | `STPlan`: index sets, levels, borders, mean denominators | maths §1.3–1.5 | plan tests | #15 | 13 Nov 2026, #63 |
| T2.4 | P2 | `STOperator`: STL's loop structure, in functional style | maths §1.6–1.7 and the 1D section; Part I, §3.1.1 | parity on 1D and 2D statistics, masked cases included (maths tab, OP-26, OP-27) | #16 | 14 Nov 2026, #67 |
| T2.5 | P2 | `STStatistics`, normalisation, reductions, `to_vector(mask_st)`; the reserved storage accessors, the `_global_sum` hook of the pixel sums and the versioned `io` schema (from T1.4); makes `STOperator`, `estel.functional.scattering` and `STStatistics` public together | maths §1.8; §12.7 | parity on normalised and reduced statistics | #17 | 16 Nov 2026, #68 |
| T2.6 | P2 | `_spec` for planar data | maths tab | spec parity | #18 | 16 Nov 2026, #82 |
| T2.7 | P2 | Differentiation tests, to second order | Part I, §3.2 | gradcheck and gradgradcheck green in float64 | #19 | 16 Nov 2026, #84 |
| T2.8 | P2 | Invariance tests | Part I, §5.1 | invariance family green | #20 | 16 Nov 2026, #86 |
| T2.9 | P2 | Pytree registration: an L0 primitive `register_pytree(cls, flatten, unflatten)` on both backends, used to register `Data` now and `STStatistics` later (Part I, §3.1, rule 9). On JAX through `jax.tree_util`; on PyTorch through `torch.utils._pytree`, imported lazily with a fallback, and only a warning if no such API exists, since nothing in ESTEL needs pytrees on PyTorch; on JAX, any registration error is raised | Part I, §3.1 rule 9; §12.3, §12.4 | `jax.jit` and `jax.vmap` accept a `Data` object and return one; the PyTorch behaviour is unchanged; conformance test on both backends | #59 | 13 Nov 2026, #60 |
| T2.10 | P2 | `WaveletOperator` (§12.5), internal, the only way to the wavelets and to the geometry of the pyramid (Part I, §2.3): `STOperator`, its plan and its loop go through it, with no geometry formula left in `scattering/`, checked in CI (Part I, §2.4); the hooks of filtering and downsampling receive the tables of the level; the low-pass filter comes with T3.2 | Part I, §2.3, §2.4; §12.4 to §12.6; maths M-WAV-2D, M-WAV-1D, M-DOWN, M-DOWN-MASK, M-MASK, M-BORDER | statistics unchanged: STL parity and spec parity green, no tolerance changed; its functions tested, to second order | #65 | 16 Nov 2026, #87 |
| T2.11 | P2 | Benchmark of step 1, outside CI, on CPU: three 2D cases at a realistic size (512 × 512, one channel, J = 7, L = 4, periodic, without mask; the same, non-periodic, with a mask; 256 × 256, two channels with cross terms), each measuring the time of the forward pass, the time of the forward pass with the gradient of a loss, and the peak resident memory of a process that runs one case. Each run checks its results. ESTEL and STL are measured in the same environment, the parity container, so that their times compare; ESTEL also outside it. Results are recorded with a description of the machine | Part I, §6.1, §6.3; Part III, §19.10, §19.12 | the three cases run with one command; their first results, ESTEL and STL, recorded | #89 | 17 Nov 2026, #94 |
| T2.12 | P2 | The Gram form of the second layer: one matrix product per level and band orientation, by blocks of pixels summed in float64 in single precision, each map set to zero outside its own window and each mean divided by the number of pixels of its pair's window, so that STL's windows are reproduced exactly, masks and borders included (M-GRAM); the band orientation as the leading axis of the second-layer maps, so that no copy is needed; the weights of the means stored once per distinct level, border and mask. First, the benchmark compares the gradient of each case with that of `main`, within a tolerance set by the task (Part I, §6.3) | Part III, §19.3, §19.12; maths_step2 M-GRAM; LF31 | every value within the parity tolerance (Part I, §3.1, rule 8: STL parity, the benchmark's comparison of the gradient with `main`, spec parity, differentiation and invariance tests green, no tolerance changed); benchmark before and after | #90 | 17 Nov 2026, #95 |
| T2.13 | P2 | Small optimisations, together: windows without weights, a plain sum or a slice where there is no mask (maths_step2 M-WIN, without the common window, which changes values); the Hermitian symmetry computed once, S4 at (ℓ₂, ℓ₁) and the channel order (c₂, c₁) obtained by conjugation, the full grid staying STL's (maths M-CHAN; maths_step2 M-CANON); real arithmetic for the auto S1 and S2 and the diagonal of S4, their storage changing only with T2.15; compilation of the whole computation on JAX by default; small kernels baked in as compiled constants (LF29). The PR measures each one separately, and drops any that the benchmark does not show to gain | Part III, §19.11, §19.12; Part I, §2.2.2, §3.1, rule 4 | every value within the parity tolerance (Part I, §3.1, rule 8: STL parity, the benchmark's comparison of the gradient with `main`, spec parity, differentiation and invariance tests green, no tolerance changed); benchmark before and after | #91 | 17 Nov 2026, #96 |
| T2.14 | P2 | Closed without change (issue #92): level-wise gradients (LF36) lowered no memory after the Gram form and slowed the gradient by 12 to 53 %; a first-order gradient on PyTorch (LF46) lowers the memory with the gradient by about 30 % but depends on the use, which 1.0.0 does not do (Part IV, §22, "Global optimisations only in 1.0.0"). The benchmark gains a fourth case, `planar-open` | Part I, §3.2; maths_step2 M-GRAD; LF36, LF46 | the specification records both results; `planar-open` in `bench/cases.toml` | #92 | 17 Nov 2026, #97 |
| T2.15 | P3 | Done in P3, before T3.2 (moved from P2: on planar data the statistics take 0.2 to 0.5 MB against 0.5 to 0.8 GB for the computation, so the gain cannot be measured there; its segments for 1D and 2D data come with this task; those of the tensor products, with their multi-index levels, are added by T3.6, before their loop). The compact coefficient layout, with STL's set of coefficients (Part III, §19.8, option a), real entries as one float and the conjugate duplicates kept, without packing; normalisation, reductions and selection as static linear maps on the compact buffer (Part III, §19.6): the accessors, `full_grid` and `to_vector` unchanged; the conversion between the computing arrays and the compact buffer is written once, in shared Python, unless that costs clearly in numerical efficiency (Part III, §19.9). The task also updates the texts that place it after 1.0.0 (§9.3, §12.7, `docs/agents/storage.md`) | Part III, §19 | every value within the parity tolerance (Part I, §3.1, rule 8: STL parity, the benchmark's comparison of the gradient with `main`, spec parity, differentiation and invariance tests green, no tolerance changed); benchmark before and after; output memory before and after | #93 | — |
| T2.16 | P2 | Consistency of the estimators (Part I, §5.1): a stationary, non-Gaussian process drawn by FFT; its statistics estimated periodic, on the same field read as non-periodic, and on the same field with a mask; the paired gaps compared with the statistical error measured over seeded draws, at two sizes at least, with a threshold corrected for the number of comparisons and a positive control | Part I, §5.1; maths M-BORDER, M-MASK, M-DOWN-MASK | the test in the full tier; the measured gaps reported, and any systematic one recorded as an open point of `maths.md` | #98 | 18 Nov 2026, #106 |
| T2.17 | P2 | Speed and health of the checks: the full tier back under its job limit on main; then the time of `make check` and of the fast CI jobs, cold (a fresh worktree) and warm, made shorter, without weakening any test | Part I, §6.1; AGENTS.md | the full tier green on main; times before and after, cold and warm, locally and in CI; no test removed, skipped or weakened | #104 | 18 Nov 2026, #105 |
| T2.18 | P2 | "Scale" and "level" told apart in the plan, the code and the notation: a scale indexes a wavelet, a level a resolution (Part III, §19.12, point 5). In 1.0.0 scale j is computed at level j, so no value changes | Part III, §19.12; maths M-NOT; issue #71 | every value unchanged, no tolerance changed; the plan and the loop name scales and levels apart | #115 | — |
| T3.0 | P3 | Write the tensor-product section of the maths tab, with the partial order of scales written out explicitly (M-IDX-TP) | STL `2D+1D`; maths tab, section 3 | reviewed by a maintainer before T3.2 | #21 | 17 Nov 2026, #103 |
| T3.1 | P3 | `DataTensorProd`, dispatch along each block's axes; its `WaveletOperator` composed from those of the blocks | §12.4, §12.5 | unit tests; the benchmark against `swapaxes` comes with T3.5 | #22 | — |
| T3.2 | P3 | `STOperatorTensorProd`, low-pass and box term, with `WaveletOperator.lowpass` (§12.5), the level of the low-pass filter read from the plan. The loop follows STL's: the blocks in their order, the result of the first blocks kept for every scale that starts with them (maths §3.2, §3.5); the second-layer means are computed in the Gram form from the start (maths_step2 M-GRAM). It writes its values into the segments of T3.6. `lowpass` takes `"ST"` or `None` (maths M-LP) | maths, tensor-product section; T2.15, T3.1, T3.3, T3.4, T3.6 | parity on tensor products against the fixtures of T3.4, and spec parity against the oracle of T3.3; the 2D+1D case added to the benchmark (Part I, §6.3): periodic, one channel, ESTEL only, sized so that the forward pass with the gradient takes at least 1 s on CPU; on that case, the memory of the output in the compact layout against the full grid (Part III, §19.1) | #23 | — |
| T3.3 | P3 | The oracle `_spec` for tensor products (maths, section 3, M-NOT-TP to M-NORM-TP): direct sums, explicit loops, float64, NumPy only, independent of the library (Part I, §4.2), each block brought to its level first, as the reference of the checks of T3.0 | maths, tensor-product section; Part I, §4.2; §12.11; T3.4 | every `M-` ID of section 3 has its `_spec` function; the oracle agrees with the fixtures of T3.4 to the parity tolerance (§14.3) | #126 | — |
| T3.4 | P3 | Parity cases of tensor products at STL `2D+1D` (§14.2): the cases of `tools/stl_parity/cases/` for the matrix of §14.4, 2D+1D in full and 2D+1D+1D at the smallest sizes that exercise every branch, a single block against the plain geometry, `lowpass` `"ST"` and `None`; and their fixtures | §14.2, §14.4; maths, tensor-product section | the fixtures produced in the container and recorded in `MANIFEST.json`; a test checks the NaN positions of each against maths M-IDX-TP | #127 | — |
| T3.5 | P3 | The trials of P3, on the 2D+1D case of the benchmark, after T3.2, each kept only if the benchmark shows a gain, its result in Part III, §19.14: the dispatch along each block's axes against `swapaxes` (§12.4); complex-to-complex convolutions instead of pairs of real ones, in the second and later blocks (Part I, §3.1, rule 6; maths M-W1-TP) | §12.4; Part I, §3.1, rules 6 and 8, §6.3; T3.2 | every value within the parity tolerance (Part I, §3.1, rule 8); the 2D+1D case of the benchmark before and after each trial | #128 | — |
| T3.6 | P3 | The statistics of tensor products on the compact buffer, before their loop (T3.2): the segments of tensor products, with their multi-index levels, built from the per-block index sets (maths M-IDX-TP), with nothing the size of the full grid on any of their paths (positions, hashing keys, copies, outputs of the reductions, `select`); their normalisation (maths M-NORM-TP) and flattening (M-FLAT) as static maps on the buffer; `full_grid`, refused beyond 2 GiB; `select` by family and channels only (LF44); the error of the reductions (OP-44); the files of `estel.io` | Part III, §19; maths M-IDX-TP, M-LP, M-S-TP, M-NORM-TP; T2.15, T3.4 | without the operator, from the values of the fixtures of T3.4 placed in the layout: the positions of the layout equal the finite positions of every fixture, family by family; the normalised statistics, their references and the two flat vectors equal those of `statistics_tp`; a fast test checks that building the layout of a small three-block tensor product allocates nothing the size of its full grid | #137 | — |
| T4.1 | P4 | 2D spectrum estimators and automatic selection. With `compute_PS`, the scattering operator builds its own spectrum operator, whose auto and cross spectra enter `STStatistics` and its `norm="self"` reference, as in STL | maths §1.9; §12.8 | parity with `master` | #24 | — |
| T5.0 | P5 | Write the HEALPix section of the maths tab | STL `HealpixUpdate`; STL-Dev-MPI `CLAUDE.md` §3 | reviewed by a maintainer before T5.1 | #25 | 18 Nov 2026, #102 |
| T5.1 | P5 | Port of the `TorchGeometry` tables, behind `WaveletOperator`; `DataHealpix`, with its reserved `partition` argument (from T1.4). Still to decide in the task: the edge of a partial sky (maths OP-50); `depth` (§12.4) or `nside`, and `nside` required with `cell_ids` (OP-91); the order of `cell_ids` (OP-90); the tables compared with healpix-analyse's as operators (OP-92); the kernel in float32 and in float64, as on PyTorch, and the run of healpix-analyse that parity needs, its cache disabled and, for float64, without its cast through float32 (OP-46); the scattering loop of `HealpixUpdate` checked against that of `master` (maths §4). With Erwan: the geometry functions to port, each with a reference test, and the memory of the gather tables (32-bit indices, the largest `nside` of 1.0.0). After the implementation that follows STL, tried and kept only if the benchmark shows a gain: the gather tables built per pixel batch, the kernel folded into the interpolation as one sparse operator per gauge, the geometry cached on disk (they could make `nside` above 256 possible on one GPU, with 2 to 3 times fewer reads, to verify). The Python classes `PatchScattering`, `PyramidScattering`, `LocalScattering`, `S3Multiscale` and `AdjointOps` of `HealpixUpdate` are never transferred: the Rust port replaces them (§10; LF3) | maths, HEALPix section; §9.3, §12.5 | tables equal healpix-analyse's as operators; parity with `HealpixUpdate` | #26 | — |
| T6.0 | P6 | Write the synthesis section of the maths tab: Gaussian start, Nyquist pre-filter, loss | STL `Synthesis.py` | reviewed by a maintainer before T6.1 | #27 | 17 Nov 2026, #100 |
| T6.1 | P6 | Synthesis API with checkpoints, reproducing STL's syntheses (maths §5.9; what comes after step 1 is LF58). Still to decide in the task, with Erwan: the default length of the L-BFGS history (STL: 50 pairs), on a simple case with few iterations | §12.9; maths §5 | API tests: a run writes checkpoints and resumes from them; for periodic planar data, a synthesis started from the same field as STL, with the same settings, in float64, follows STL's loss over its first iterations only, within a tolerance set by the task, against a parity fixture, in the full tier, as fast as it can be (W2 extends it after step 1); a reference synthesis checked by eye by a maintainer | #28 | — |
| T6.2 | P6 | Utilities and diagnostics | §12.10 | unit tests | #29 | — |
| T6.3 | P6 | The L0 primitives that have no user at the end of P6 removed; the signature of `gather_weighted` checked against the HEALPix tables of T5.1 | §12.3; T5.1 | the protocol, both backends and the conformance suite hold only primitives with a user; every test green | #116 | — |
| T8.1 | P8 | Reference values from `_spec`; tolerance calibration | Part I, §5 | references frozen with their checksums | #31 | — |
| T8.2 | P8 | Parity report, review, ESTEL reference document, release on PyPI | §14.5; Part IV, §22, review of 12 November 2026 | 1.0.0 published | #32 | — |
