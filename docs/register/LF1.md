+++
id = "LF1"
title = "JAX backend"
category = "Structure"
step = "2a"
phase = "2a.X"
status = "ready"
decide = []
depends = ["LF62"]
changes_values = "no; minor (JAX made stable; experimental since 1.0.0)"
issue = 0
+++

## Goal

`estel.set_backend("jax")` without a warning: statistics, spectra, syntheses (with `optax.lbfgs`) and differentiation to second order on JAX, in every geometry of 1.0.0, HEALPix included, covered by the guarantees of 1.x on CPU and GPU (Part III, §17.1; the decisions below). HEALPix spectra are not part of it (LF24). LF2 (HEALPix on JAX) is merged into this entry.

## Decisions

- 25 November 2026, Pablo Richard: the devices of the guarantee (question 1, answer (a), with GPU defined). The stable guarantee covers CPU, tested on every pull request, and GPU, by the full tier run by hand on the GPU node of the benchmark of step 2 (LF62) before each release. TPU is outside the guarantee: ESTEL cannot select a TPU, its devices being `cpu` and `cuda:N` (`_DEVICE` in `src/estel/backend/_registry.py`; Part II §12.2); supporting it would be a new card. Task 5 also updates Part III §17.1 ("on CPU, GPU and TPU"), protected text, with a human's go when that task is done (`docs/register/decisions.md`).
- 25 November 2026, Pablo Richard: JAX on each pull request (question 2, answer (a)). The tests marked `jax_smoke`, as in step 1, and one L-BFGS synthesis case marked `jax_smoke`: the configuration of the scattering parity smoke case (`scattering_2d`, `periodic`: 32 × 32, J = 3), so that its forward executables are already compiled in the job, 2 to 3 iterations, in float64; checkpoints and resume stay in the full tier. The conformance tests of `TestLbfgs`, the quadratic `test_solves_least_squares` among them, run on JAX once their `_lbfgs_not_provided` gate is removed. The pull request that adds the case reports the time of the JAX fast job before and after (golden rule 13): JAX times are reported, not enforced per test, and the job has its own budget of 120 s (Part I §5.4). The reasons of the former recommendation, corrected: the JAX full tier runs on every push to `main`, in jobs a merge does not require (the job `full-jax` of `.github/workflows/tier0.yml`), not only nightly; and the AST checks (`tools/check_layers.py`) do not cover `.item()` or `float(x)` (`docs/register/decisions.md`).
- 25 November 2026, Pablo Richard: LF2 (HEALPix on JAX) is merged into this entry, and its card removed; the row of the decision log keeps every citation of LF2 valid. This entry takes from LF2: the HEALPix tests on JAX in both dtypes, to second order (task 3); the peak memory of `gather_weighted` on HEALPix (task 4); its inputs (M-CONV-HEALPIX, M-DOWN-HEALPIX, M-MASK-HEALPIX, Part II §9.3, Part I §3.1 rule 4) and its Source (C9); the exclusion of HEALPix spectra (LF24), in the Goal (`docs/register/decisions.md`).

## Inputs

Part III §17.1 to §17.5; Part I §2.2.5, §3.1 (rule 4: large constants are arguments), §3.2.2, §5.2 (the rows "vs reference", and "PyTorch vs JAX"), §5.4, §6.1; Part II §9.3 ("HEALPix in step 1"), §12.2, §12.3 (`lbfgs`, `gather_weighted`, `segment_sum`), §12.9; maths M-SYN-LBFGS, M-CONV-HEALPIX, M-DOWN-HEALPIX, M-MASK-HEALPIX; Part IV §22, "JAX in step 1"; the cards LF62, W2, W3, LF24.
Modules: `src/estel/backend/jax/_primitives.py` (`lbfgs`, `not_provided`, `LBFGS_NOT_PROVIDED`, `gather_weighted` and `segment_sum`, implemented since T1.3), a new `src/estel/backend/jax/_optim.py`, `src/estel/backend/_registry.py` (`JAX_WARNING`), `src/estel/synthesis/`, the HEALPix data and tables of step 1 (T5.1), `tests/backend/test_conformance.py`, `tests/backend/test_jax.py`, `tests/conftest.py`, the synthesis and HEALPix tests, `pyproject.toml` (the extra `jax`, the marker `jax_smoke`).

## Tasks

1. `JaxBackend.lbfgs` by `optax.lbfgs` with `optax.scale_by_zoom_linesearch`, with the settings of M-SYN-LBFGS where optax has them; `optax` in the extra `jax`; its state saved and resumed by the checkpoints. The settings of M-SYN-LBFGS with no optax counterpart are implemented by the loop around it: `max_iter`, `max_eval`, `tolerance_grad` and `tolerance_change`; the line search stops after at most 25 evaluations in PyTorch's strong Wolfe search, 20 by default in optax's `lbfgs`. The conformance test of a quadratic exists already (`TestLbfgs.test_solves_least_squares`), with the rest of `TestLbfgs`, gated by `_lbfgs_not_provided` (`tests/backend/test_conformance.py`): they run on both backends once the gate is removed.
2. Syntheses on JAX: the synthesis tests on both backends; the L-BFGS synthesis case marked `jax_smoke` (the decision on question 2), with the time of the JAX fast job before and after in its pull request; the references per backend and the cross-backend criteria with W2 and W3, when they land.
3. HEALPix on JAX: the HEALPix tests on JAX in float32 and float64, reference and second-order tests included (the full tier runs them from step 1): fix what fails in L0; a fix needed in L1 or L2 is a bug of step 1 (§17.2).
4. The compile times of the JAX cases of the benchmark of step 2 (LF62), HEALPix included, within their budgets in `bench/thresholds.toml`; what exceeds them fixed. On the HEALPix cases, the peak memory of `gather_weighted`, whose `take` builds every stencil value before the sum unless XLA fuses it, against PyTorch's; a lighter gather if over budget.
5. [human] Declared stable, per the decisions: the warning (`JAX_WARNING`) and every `not_provided` guard removed: `JaxBackend.not_provided` and `LBFGS_NOT_PROVIDED` (`src/estel/backend/jax/_primitives.py`), `_check_backend` (`src/estel/synthesis/_api.py`), `test_lbfgs_names_its_step` (`tests/backend/test_jax.py`), `NOT_PROVIDED_ERRORS` and `_lbfgs_not_provided` (`tests/backend/test_conformance.py`), `_unavailable` and the guard of `test_compute_ps_puts_the_spectrum_in_the_loss` (`tests/synthesis/test_api.py`), that of `test_shape_follows_the_target_or_running_shape` (`tests/synthesis/test_start.py`) and that of `tests/parity/test_synthesis.py`; Part I §2.2.5 and Part II §12.2 updated, and Part III §17.1 (CPU and GPU, not TPU) with a human's go; `changes/<PR>.minor.md`.

## Done when

The exit criteria of §17.5: the conformance suite passes on JAX in float32 and float64; every reference test passes on JAX within its tolerances, the rows "vs reference" of Part I §5.2 (`tests/tolerances.toml`, created by T8.1); the differentiation tests pass in both modes to second order; the compile times are within their budgets. And: a periodic planar synthesis runs on JAX, writes checkpoints and resumes from them; the L-BFGS synthesis case runs in the fast tier on JAX; every HEALPix test passes on JAX in both dtypes, the second-order tests included, and the HEALPix cases of LF62 compile within budget on JAX; the full tier passes on JAX on the GPU node of LF62.

## Feature

JAX backend

## Why

`jit`, `vmap`, GPU, functional ecosystem

## Builds on (reserved in 1.0.0)

L0 protocol, functional rules

## To decide

—

## Details

§17

## Source

C9

## Step before 20 November 2026

?
