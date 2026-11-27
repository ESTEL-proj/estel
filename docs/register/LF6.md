+++
id = "LF6"
title = "Low-pass filters in every geometry and S3/S4"
category = "New feature"
step = "2a"
phase = "2a.1"
status = "ready"
decide = []
depends = ["LF33"]
changes_values = "yes (new default); reference bump; next 1.x minor"
issue = 0
+++

## Goal

The largest scales enter every family: S1 and S2 at the low-pass scale j = J, and S3 and S4 with it, in planar 2D and 1D, tensor products and HEALPix, without STL's final downsampling, so that no size constraint remains (OP-34).

## Decisions

- 23 November 2026, Pablo Richard: all combinations, in every geometry (question 1). The low-pass filter may replace any wavelet of S3 and S4, as a first or as the last filter, in each block independently, the mixed scales of tensor products included. Counted with M-IDX-TP's formulas (φ as the scale J_a of each block, one orientation), which reproduce the totals of the STOC deliverable (1,474; 587,324; 18,513,572): plain 2D with J = 4, L = 4 gains 196 coefficients, 1,474 → 1,670 (+13.3 %; the restricted set, the low-pass filter as the last filter only, would add 178, +12.1 %, not the 0.3 to 3.5 % quoted before); 2D+1D gains 38.2 %, 2D+1D+1D 111.8 %. The fully low-passed field alone in tensor products (+2.4 % and +0.9 %) is the new entry LF66, step 3 (`docs/register/decisions.md`).
- 23 November 2026, Pablo Richard: the half-pixel shift (question 2, answer (a)). The 6-tap smoothing kernel and M-DOWN's padding (2 before, 3 after) are kept without the final downsampling: no periodic mean moves; with a window, the crop counts each side apart (LF60); centred kernels come with LF25 (step 2b) (`docs/register/decisions.md`).
- 23 November 2026, Pablo Richard: the low-pass filter in the plan (question 3, answer (a)). One more scale j = J in the index sets, with one orientation, at level J − 1, realised as `WaveletPlan.lowpass_realization` says, with its crop; the plan marks each mean centred or not; `STStatistics` keeps its interface (§19.12, points 4 to 6) (`docs/register/decisions.md`).

## Inputs

- maths_step2 M-LP2, M-COV, M-WIN; Part III §20.1, §20.10 (last paragraph), §20.11 rules 1, 2 and 7, §19.12 points 4 to 6; maths M-LP, M-IDX-TP, M-BORDER-TP, M-SMOOTH-2D, M-SMOOTH-1D, M-DOWN, M-DOWN-MASK, M-DOWN-HPX; OP-2, OP-34, OP-42, OP-43; Part II §12.4 and §12.5 (`_lowpass`, `.lowpass`); Part IV §22, G1.
- Modules: `src/estel/wavelets/_operator.py` (`WaveletPlan.lowpass_realization`, `lowpass_level`), `src/estel/data/_base.py`, `src/estel/data/_planar.py`, `src/estel/scattering/_plan.py`, `src/estel/scattering/_loop.py`, `src/estel/statistics/_grid.py`, `src/estel/_spec/cartesian.py`; the tensor-product and HEALPix modules.

## Tasks

Waits for LF33: every low-pass term is centred. Task 2 feeds LF60's study, whose result sets the crop of task 3.
1. M-LP2 completed: the decisions of 23 November 2026, with the formulas of the low-pass filter as a first filter, and what it leaves unwritten, the smoothing without stride on masked data (M-DOWN-MASK, its threshold) and on HEALPix (M-DOWN-HPX's Gaussian without the decimation) [human review].
2. `_spec`: the low-pass terms of M-LP2, centred (LF33's `_spec`), in every geometry of `_spec`; tests.
3. Planar 2D and 1D: the low-pass scale in the plan, the wavelet operator and the loop; references regenerated, `STATISTICS_REFERENCE` bumped, broken STL parity tests removed [human].
4. Tensor products: the low-pass filter without the final downsampling (OP-34 gone) and in S3 and S4, all combinations, the mixed scales included (OP-43); references, as in 3 [human].
5. HEALPix; references, as in 3 [human].
6. Cost on the benchmark of step 2 (LF62).

## Done when

- In planar 2D and 1D, tensor products and HEALPix, S1 and S2 at j = J and every S3 and S4 with the low-pass filter (all combinations) equal `_spec` within the parity tolerance, in float32 and float64.
- The coefficient counts follow §20.1: plain 2D gains S1[J] and S2[J], 1,474 → 1,476 before S3 and S4 (G1), and 1,670 with S3 and S4 (J = 4, L = 4); the 2D+1D and 2D+1D+1D cases of the STOC deliverable gain 38.2 % and 111.8 %.
- A tensor-product block whose size is not divisible by 2^(J_a) has its low-pass filter, without warning (M-LP, case F: size 12, J = 3).
- On periodic data, S2[J] equals the variance of Φ x^(J−1) computed apart in NumPy; x → x + c leaves every low-pass term unchanged, masks included.
- `STStatistics` keeps its interface: the low-pass filter is one more scale, NaN at orientations other than 0, kept by `to_vector`. References regenerated, removed parity tests listed in the PR, cost recorded on LF62.

## Feature

Low-pass filters in every geometry and in S3/S4. At the last level the low-pass filter is applied without STL's final downsampling, which removes a size constraint (OP-34); this is a traced change of reference values. It is realised with the smoothing kernel; a dedicated low-pass kernel φ remains possible

## Why

the largest scales enter every statistic

## Builds on (reserved in 1.0.0)

index sets of the plan

## To decide

—

## Details

§20.1, §20.10, §20.11

## Source

—

## Step before 20 November 2026

2a; completed by LF22 (2b)
