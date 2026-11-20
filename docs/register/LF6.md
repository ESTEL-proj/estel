+++
id = "LF6"
title = "Low-pass filters in every geometry and S3/S4"
category = "New feature"
step = "2a"
phase = "2a.1"
status = "decision"
decide = ["S3 and S4: low-pass as last filter only, or also first", "half-pixel shift without the final downsampling (M-LP2)", "how the plan describes the low-pass filter and the centring"]
depends = ["LF33"]
changes_values = "yes (new default); reference bump; major decided with the maintainers"
issue = 0
+++

## Goal

The largest scales enter every family: S1 and S2 at the low-pass scale j = J, and S3 and S4 with it, in planar 2D and 1D, tensor products and HEALPix, without STL's final downsampling, so that no size constraint remains (OP-34).

## Brief

**Question 1. Which S3 and S4 take the low-pass filter?**
- (a) Restricted: as the last filter only (M-LP2): +0.3 to +3.5 % coefficients.
- (b) All combinations: also as a first filter, with arms U_J = |Φ x|: +38 to +112 %, mostly in S4, the dominant cost (§20.1).
- **Recommendation:** (a), because it brings the couplings with the largest scales into S3 and S4 for a few per cent; (b) can be a later card if a use needs it.

**Question 2. The half-pixel shift of the even smoothing kernel, without the final downsampling (OP-2).**
- (a) Keep the 6-tap kernel and M-DOWN's padding, 2 before and 3 after: no periodic mean moves; with a window, the crop counts each side apart (LF60).
- (b) A symmetric kernel now, such as s ⋆ [½, ½] (7 taps): no shift, but another filter, changed again by LF22 and LF25.
- (c) Wait for the centred kernels of LF25: the entry moves to step 2b.
- **Recommendation:** (a), because no periodic mean moves, and centring the kernels depends on the filters, which the step criterion puts in 2b (LF25).

**Question 3. How the plan describes the low-pass filter and the centring (§19.12, point 4).**
- (a) One more scale j = J in the index sets, with one orientation, at level J − 1 (scale and level apart, point 5), realised as `WaveletPlan.lowpass_realization` says, with its crop; the plan marks each mean centred or not (§20.11, rule 2).
- (b) A separate low-pass family in `STStatistics`: changes its interface, which point 6 excludes.
- **Recommendation:** (a), because §19.12 points 2, 5 and 6 reserve it, and `lowpass_realization` and `lowpass_level` exist already.

## Inputs

- maths_step2 M-LP2, M-COV, M-WIN; Part III §20.1, §20.10 (last paragraph), §20.11 rules 1, 2 and 7, §19.12 points 4 to 6; maths M-LP, M-IDX-TP, M-BORDER-TP, M-SMOOTH-2D, M-SMOOTH-1D, M-DOWN, M-DOWN-MASK, M-DOWN-HPX; OP-2, OP-34, OP-42, OP-43; Part II §12.4 and §12.5 (`_lowpass`, `.lowpass`); Part IV §22, G1.
- Modules: `src/estel/wavelets/_operator.py` (`WaveletPlan.lowpass_realization`, `lowpass_level`), `src/estel/data/_base.py`, `src/estel/data/_planar.py`, `src/estel/scattering/_plan.py`, `src/estel/scattering/_loop.py`, `src/estel/statistics/_grid.py`, `src/estel/_spec/planar.py`; the tensor-product and HEALPix modules.

## Tasks

Waits for LF33: every low-pass term is centred. Task 2 feeds LF60's study, whose result sets the crop of task 3.
1. M-LP2 completed: the answers to questions 1 to 3, and what it leaves unwritten, the smoothing without stride on masked data (M-DOWN-MASK, its threshold) and on HEALPix (M-DOWN-HPX's Gaussian without the decimation) [human review].
2. `_spec`: the low-pass terms of M-LP2, centred (LF33's `_spec`), in every geometry of `_spec`; tests.
3. Planar 2D and 1D: the low-pass scale in the plan, the wavelet operator and the loop; references regenerated, `STATISTICS_REFERENCE` bumped, broken STL parity tests removed [human].
4. Tensor products: the low-pass filter without the final downsampling (OP-34 gone) and in S3 and S4 (OP-43); references, as in 3 [human].
5. HEALPix; references, as in 3 [human].
6. Cost on the benchmark of step 2 (LF62).

## Done when

- In planar 2D and 1D, tensor products and HEALPix, S1 and S2 at j = J and the S3 and S4 of the set chosen in question 1 equal `_spec` within the parity tolerance, in float32 and float64.
- The coefficient counts follow §20.1: plain 2D gains S1[J] and S2[J], 1,474 → 1,476 before S3 and S4 (G1).
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

which combinations in S3 and S4: the low-pass filter as the last filter only (+0.3 to +3.5 % coefficients) or also as a first filter (+38 to +112 %, §20.1); the half-pixel shift of the even smoothing kernel without the final downsampling (maths_step2 M-LP2); how the plan describes the low-pass filter and the centring (§19.12, point 4)

## Details

§20.1, §20.10, §20.11

## Source

—

## Step before 20 November 2026

2a; completed by LF22 (2b)
