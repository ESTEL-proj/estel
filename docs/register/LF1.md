+++
id = "LF1"
title = "JAX backend"
category = "Structure"
step = "2a"
phase = "2a.X"
status = "decision"
decide = ["devices covered by the stable guarantee", "JAX tests on every pull request once stable"]
depends = ["LF62"]
changes_values = "no; minor (JAX made stable; experimental since 1.0.0)"
issue = 0
+++

## Goal

`estel.set_backend("jax")` without a warning: statistics, spectra, syntheses (with `optax.lbfgs`) and differentiation to second order on JAX, covered by the guarantees of 1.x (Part III, §17.1).

## Brief

Step 1 gives what §17.2 lists and every L0 primitive but `lbfgs`. What remains: the optimiser, syntheses on JAX, the compile-time budgets (LF62) and the declaration, with two choices.
**Question 1. Which devices does "stable" cover?** §17.1 names CPU, GPU and TPU; tier 0 tests CPU, GPU CI (tier 1) is W1, in step 2b, and nothing tests TPU.
(a) CPU, tested on every pull request, and GPU, checked on the node of the benchmark of step 2 (LF62) before the release; TPU documented as untested.
(b) CPU only until tier 1 (W1) runs; GPU and TPU stay experimental.
(c) CPU, GPU and TPU: needs a TPU runner, planned nowhere.
**Recommendation:** (a), because the GPU node exists in step 2a and a guarantee is worth what is tested.
**Question 2. What runs on JAX for each pull request once it is stable?**
(a) As in step 1: the tests marked `jax_smoke` in the fast tier; the whole suite on both backends in the full tier (`main`, nightly, `full-ci`).
(b) The whole fast tier on JAX too, in a parallel job: JAX-only breaks caught before merge, but JAX's compile times push that job against the 2-minute budget (golden rule 13).
**Recommendation:** (a), because the AST rules check the functional rules on every pull request and the nightly full tier catches the rest within a day.

## Inputs

Part III §17.1 to §17.5; Part I §2.2.5, §3.2.2, §5.2 (the row "PyTorch vs JAX"), §6.1; Part II §12.3 (`lbfgs`), §12.9; maths M-SYN-LBFGS; Part IV §22, "JAX in step 1"; the cards LF62, W2, W3, LF2.
Modules: `src/estel/backend/jax/_primitives.py` (`lbfgs`, `not_provided`), a new `src/estel/backend/jax/_optim.py`, `src/estel/backend/_registry.py` (`JAX_WARNING`), `estel/synthesis/`, `tests/backend/test_conformance.py`, `tests/conftest.py`, `pyproject.toml` (the extra `jax`, the marker `jax_smoke`).

## Tasks

1. `JaxBackend.lbfgs` by `optax.lbfgs` with `optax.scale_by_zoom_linesearch`, with the settings of M-SYN-LBFGS where optax has them; `optax` in the extra `jax`; its state saved and resumed by the checkpoints; conformance: a quadratic minimised on both backends.
2. Syntheses on JAX: the synthesis tests on both backends; the references per backend and the cross-backend criteria with W2 and W3, when they land.
3. The compile times of the JAX cases of the benchmark of step 2 (LF62) within their budgets in `bench/thresholds.toml`; what exceeds them fixed.
4. [human] Declared stable, per questions 1 and 2: the warning and `not_provided` removed; Part I §2.2.5 and Part II §12.2 updated; `changes/<PR>.minor.md`.

## Done when

The exit criteria of §17.5: the conformance suite passes on JAX in float32 and float64; every reference test passes on JAX within the row "PyTorch vs JAX" of `tests/tolerances.toml`; the differentiation tests pass in both modes to second order; the compile times are within their budgets. And a periodic planar synthesis runs on JAX, writes checkpoints and resumes from them.

## Feature

JAX backend

## Why

`jit`, `vmap`, TPU and GPU, functional ecosystem

## Builds on (reserved in 1.0.0)

L0 protocol, functional rules

## To decide

—

## Details

§17

## Source

—

## Step before 20 November 2026

?
