+++
id = "LF7"
title = "Positive spectrum estimator for non-periodic data"
category = "New feature"
step = "2a"
phase = "2a.3"
status = "study"
decide = ["the estimator for non-periodic data: taper, better crop or multitaper", "whether negative bins exist (OP-21), and the parameter, from the study", "method auto: the default for non-periodic data, the old names kept"]
depends = []
changes_values = "yes, if it becomes the default; reference bump if default"
issue = 0
+++

## Goal

Spectra of non-periodic 2D data, with or without missing values, from an estimator whose auto bins are positive by construction and whose border leakage is measured, chosen by `method="auto"`; for every geometry whose non-periodic spectra exist when it lands (2D in 1.0.0; 1D and tensor products once LF23 or LF10 has landed).

## Brief

- **Question 1. The estimator for non-periodic data, with and without a mask** (Part III §20.2: the taper, its parameter, the normalisation of the bins; the definition to write in `docs/maths_step2.md`).
  - (a) tapered periodogram: the periodogram of w·m·x (w a separable Tukey window of fraction α, m the observed mask), normalised by Σ(w·m)², binned with W_b as `fft`; one FFT per map, no band maps, so no even-size constraint (OP-20);
  - (b) `pbc_mask` kept, with a crop at a larger energy fraction than 60 % or its "smooth" mask: no new estimator, fewer pixels per bin;
  - (c) multitaper (K sine tapers): lower variance, K FFTs per map.
  - **Recommendation:** none before the study of question 2.
- **Question 2. Is there a defect, and which parameter?** By M-PS-2D the auto bins of `pbc_mask` are sums of M_b f_b² ≥ 0: the negative bins that §20.2 attributes to OP-21 must first be seen; the 60 % crop may leave a leakage bias instead. Study (at most a day): the process of T2.16 (`tests/consistency/`), N = 128 and 256, 64 draws, each field read periodic (`fft`, the reference) and non-periodic, without and with its five holes; per bin, the paired bias (z) and RMS against `fft`, and the count of negative auto bins, for `pbc_mask` (hard, smooth), (a) with α ∈ {0.1, 0.25, 0.5} and (c) with K = 3.
  - **Recommendation:** status `study`; then the estimator with the smallest RMS among those with no bin beyond the threshold of T2.16 (|z| ≤ 6.1); close LF7 if that is `pbc_mask` and it has no negative auto bin.
- **Question 3. Its place.**
  - (a) `method="auto"` selects it for non-periodic data; `"pbc_mask"` and `"pbc_nan_mask"` stay selectable until 2.0.0, since removing a name of the manifest is a major;
  - (b) it replaces their computation under their names.
  - **Recommendation:** (a): the decisions of 20 November 2026 make a correction the default, with no opt-in, and (a) breaks no public name.

## Inputs

- Read: Part III §20.2; maths M-PS-2D ("Estimators", "Choice of estimator"), OP-20, OP-21; Part II §12.8 (`estimate_tapered`); Part I §5.1, the row "Consistency of the estimators"; Part II §15.2, T2.16.
- Modules: `src/estel/spectra/` (`_estimators.py`, `_bins.py`, from T4.1) and the selection of `method="auto"`; `src/estel/_spec/planar.py`; `tests/consistency/test_estimators.py`; `api/manifest.json` (the method names; protected).

## Tasks

1. The study of question 2: its numbers and a recommendation on the issue.
2. [human] Questions 1 and 3 decided; the M- definition in `docs/maths_step2.md` [human review].
3. `_spec`: the estimator, and its tests.
4. The estimator in `src/estel/spectra/` for 2D, and for the non-periodic 1D and tensor-product spectra if LF23 or LF10 has landed; `method="auto"` selects it; its name in `api/manifest.json` [human].
5. [human] References regenerated, `STATISTICS_REFERENCE` bumped, the STL parity tests of non-periodic spectra that break removed.
6. Its cost on the benchmark of step 2 (LF62), once it exists.

## Done when

- `make test-spec` passes with the new definition; every auto bin is ≥ 0 on non-periodic and masked fields of 16 × 16 and 32 × 32 (fast tier);
- on the paired fields of T2.16, the non-periodic and masked spectra agree with the periodic `fft` one within its threshold (|z| ≤ 6.1) on every bin the study validated (full tier);
- `STATISTICS_REFERENCE` bumped with CODEOWNER approval.

## Feature

Positive spectrum estimator for non-periodic data

## Why

today's estimator gives negative bins

## Builds on (reserved in 1.0.0)

estimator registry

## To decide

—

## Details

§20.2

## Source

OP-21

## Step before 20 November 2026

?
