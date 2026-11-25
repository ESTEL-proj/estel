+++
id = "LF8"
title = "NaN masks in tensor products"
category = "New feature"
step = "2a"
phase = "2a.3"
status = "decision"
decide = []
depends = []
changes_values = "no (new capability); minor"
issue = 0
+++

## Goal

Tensor-product data with missing values: `DataTensorProd` takes a mask, or an array with NaN, where 1.0.0 raises at construction (OP-41, OP-80), and the statistics are computed with masks as weights, as on planar 2D, 1D and HEALPix data: gaps in space–time data. First for separable masks, then for general ones (§20.3). The NaN-aware spectra of tensor products (`nan_mask`, `pbc_nan_mask`) are LF10's, which waits for this entry.

## Decisions

- 25 November 2026, Pablo Richard: this entry moves from step 2b to step 2a, phase 2a.3 (spectra and masks), with LF10's question 3: LF10 keeps the NaN-aware estimators on tensor products, and depends on this entry (`docs/register/decisions.md`).
- 25 November 2026, Pablo Richard: the observed-sample mask with per-block border crops (question 1, answer (b), §20.3): M-MASK on the product support. A pixel is invalid when the support E₀ × … × E\_{B−1} of its product wavelet touches a missing sample; each downsampling along block a thresholds the invalid fraction along that block, with its τ (M-DOWN-MASK: 0.33 in 2D, 0.5 in 1D), and the order of the blocks is part of the definition (block order, as STL's loop, M-W1-TP), since with the threshold the masked downsamplings along two blocks no longer commute; a covariance takes the union of the masks of its arguments; the borders are cropped per block (M-BORDER-TP), apart from the mask. No statistic reads a missing sample, whatever the mask. The product of per-block valid weights (answer (a)) gives the same weights on a separable mask, and lets in pixels that read a missing sample when the mask varies along another block (28 % of (b)'s invalid pixels of the first layer, on 32 × 32 × 32 with a disc of radius 4 moving one pixel per time step, in a NumPy check). Its per-block storage is kept as an optimisation of (b) for the plans whose mask is separable, with the same values, only if the benchmark of step 2 (LF62) shows its gain (`docs/register/decisions.md`).

## Inputs

- Read: Part III §20.3; maths M-DATA-TP ("Missing data"), M-NOT-TP, M-W1-TP, M-BORDER-TP, M-MASK, M-DOWN-MASK, M-SMOOTH-1D (τ = 0.5), M-MEAN; OP-12, OP-26, OP-35, OP-41, OP-80; maths_step2 M-WIN; Part I §3.2.3 (rule 4); Part IV §22, the row Q12 of 18 November 2026; the cards LF10, LF52, LF54, LF62.
- Modules: `src/estel/data/_tensor_prod.py` (the errors that name LF8; `_downsample`, without maps), `src/estel/data/_planar.py` (its error on tensor products), `src/estel/wavelets/_tensor_prod.py`, the tensor-product plan and loop of `src/estel/scattering/` (`_tensor_prod.py`, `_loop_tp.py`), `src/estel/_spec/tensor_prod.py`, `tests/geometry/test_tensor_prod.py`, `tests/scattering/test_operator_tp.py`, `bench/_cases.py` (masks refused on tensor products).

## Tasks

LF52, in the same phase, may replace the threshold by continuous weights: its rule applies along each block.
1. [human] The rule of the decision in `docs/maths_step2.md`, with an `M-` ID: the masks of the layers on the product support, the downsampling of the mask along each block with its τ, in block order, the reweighting maps, the covariances, the windows with the per-block borders; the shapes of the mask that `DataTensorProd` takes (its place is reserved in the signature), in `api/manifest.json` if it records them.
2. The oracle: masks in `src/estel/_spec/tensor_prod.py`, without mask today, and its tests.
3. `DataTensorProd` takes a mask or an array with NaN, the errors that name LF8 removed; the plan builds the masks and the reweighting maps per level, and the loop uses them as weights; `changes/<PR>.minor.md`.
4. The cost on the 2D+1D cases of the benchmark of step 2 (LF62), with a mask; on a separable mask, the per-block storage tried there, kept only if it shows its gain, with the same values.

## Done when

On 2D+1D data of 16 × 16 × 16 with J = (2, 2), periodic and not, with a separable mask and with a mask that varies in time, every statistic matches the oracle in float64; a product of one block gives the statistics of `Data2D` or `Data1D` with the same mask, to rounding; data with NaN or a mask no longer raise in tensor products; the second-order and JAX tests pass with a mask; values without a mask unchanged (the reference tests pass without regeneration).

## Feature

NaN masks in tensor products

## Why

gaps in space–time data

## Builds on (reserved in 1.0.0)

masks as weights

## To decide

—

## Details

§20.3

## Source

—

## Step before 20 November 2026

?
