+++
id = "LF8"
title = "NaN masks in tensor products"
category = "New feature"
step = "2a"
phase = "2a.3"
status = "decision"
decide = ["per-block valid weights multiplied, or the observed-sample mask with per-block border crops (§20.3)"]
depends = []
changes_values = "no (new capability); minor"
issue = 0
+++

## Goal

Tensor-product data with missing values: `DataTensorProd` takes a mask, or an array with NaN, where 1.0.0 raises at construction (OP-41, OP-80), and the statistics are computed with masks as weights, as on planar 2D, 1D and HEALPix data: gaps in space–time data. First for separable masks, then for general ones (§20.3). The NaN-aware spectra of tensor products (`nan_mask`, `pbc_nan_mask`) are LF10's, which waits for this entry.

## Decisions

- 25 November 2026, Pablo Richard: this entry moves from step 2b to step 2a, phase 2a.3 (spectra and masks), with LF10's question 3: LF10 keeps the NaN-aware estimators on tensor products, and depends on this entry (`docs/register/decisions.md`).

## Brief

**Question 1. Which weight does a pixel of a tensor product get from a mask?** (§20.3; M-DATA-TP, "Missing data"; OP-41.) In 2D and 1D a pixel of a layer is invalid when the support E of its wavelet touches an invalid pixel (M-MASK); each downsampling thresholds the invalid fraction (M-DOWN-MASK: τ = 0.33 in 2D, 0.5 in 1D) and fills the invalid input with 0; a covariance takes the union of the masks of its arguments; the border is cropped apart (M-BORDER); a mask is a weight, never an index (Part I §3.2.3, rule 4). A product wavelet has the support E₀ × … × E\_{B−1}, applied block by block (M-W1-TP), and each block is downsampled along its own axes (M-NOT-TP).
- (a) The product of per-block valid weights, read here as w = ∏\_a w\_a, where w\_a is M-MASK and M-DOWN-MASK applied along the axes of block a alone, with that block's τ, every other coordinate fixed; the border of each non-periodic block a factor of its weight (M-BORDER-TP). On a separable mask (a land mask fixed in time, missing time steps), every w\_a depends on block a only: the plan stores per-block maps, a few numbers per pixel of each block and level, and a mean is a sequence of weighted sums along the blocks. On a mask that varies along another block, the w\_a are full-size maps, and a pixel whose product support reaches a missing sample only through an offset in two blocks at once keeps the weight 1: its value reads the 0 that fills that sample, and enters the means. (a) also needs a rule with no analogue in 2D: how w\_a is carried through the downsamplings along the other blocks.
- (b) The observed-sample mask with per-block border crops: M-MASK on the product support. A pixel is invalid when E₀ × … × E\_{B−1} touches a missing sample; each downsampling along block a thresholds the invalid fraction along that block, with its τ; a covariance takes the union; the borders are cropped per block (M-BORDER-TP), apart from the mask. No statistic reads a missing sample, whatever the mask. The plan holds full-size maps per level, as for 2D data with a mask: the levels of a product are multi-indices, so the maps of one kind add up to at most 8/3 of the size of the field in 2D+1D (4/3 in 2D), and the means use the windows "weights" of M-WIN. With the threshold, the masked downsamplings along two blocks no longer commute, so the order of the blocks is part of the definition (block order, as STL's loop, M-W1-TP).
- **What separates them.** On a separable mask the two give the same weights, layer by layer and level by level; they differ only when the mask varies along another block. A NumPy check of the masks of the first layer (2D+1D, 32 × 32 × 32, periodic, a disc of radius 4): with the disc fixed in time, the same masks; with the disc moving one pixel per time step, (a) keeps 1,664 pixels that (b) marks invalid, 28 % of (b)'s invalid set, each reading a missing sample. The order of the thresholded downsamplings along two 1D blocks changed the mask for 1,727 of 2,000 random masks, and for none of 2,000 separable ones.
- **Recommendation:** (b), because it is the rule of every other geometry read on the product support, so that no statistic reads a missing sample whatever the mask. Since (b) equals (a) on separable masks, the order of §20.3, separable masks first, becomes the order of the tests, and (a)'s per-block storage an optimisation of (b) for a plan whose mask is separable, with the same values, kept if the benchmark of step 2 (LF62) shows its gain.

LF52, in the same phase, may replace the threshold by continuous weights: its rule applies along each block in either option.

## Inputs

- Read: Part III §20.3; maths M-DATA-TP ("Missing data"), M-NOT-TP, M-W1-TP, M-BORDER-TP, M-MASK, M-DOWN-MASK, M-SMOOTH-1D (τ = 0.5), M-MEAN; OP-12, OP-26, OP-35, OP-41, OP-80; maths_step2 M-WIN; Part I §3.2.3 (rule 4); Part IV §22, the row Q12 of 18 November 2026; the cards LF10, LF52, LF54, LF62.
- Modules: `src/estel/data/_tensor_prod.py` (the errors that name LF8; `_downsample`, without maps), `src/estel/data/_planar.py` (its error on tensor products), `src/estel/wavelets/_tensor_prod.py`, the tensor-product plan and loop of `src/estel/scattering/` (`_tensor_prod.py`, `_loop_tp.py`), `src/estel/_spec/tensor_prod.py`, `tests/geometry/test_tensor_prod.py`, `tests/scattering/test_operator_tp.py`, `bench/_cases.py` (masks refused on tensor products).

## Tasks

1. [human] Question 1 answered; the rule in `docs/maths_step2.md`, with an `M-` ID: the masks of the layers, the downsampling of the mask along each block with its τ and the order of the blocks, the reweighting maps, the covariances, the windows with the per-block borders; the shapes of the mask that `DataTensorProd` takes (its place is reserved in the signature), in `api/manifest.json` if it records them.
2. The oracle: masks in `src/estel/_spec/tensor_prod.py`, without mask today, and its tests.
3. `DataTensorProd` takes a mask or an array with NaN, the errors that name LF8 removed; the plan builds the masks and the reweighting maps per level, and the loop uses them as weights; `changes/<PR>.minor.md`.
4. The cost on the 2D+1D cases of the benchmark of step 2 (LF62), with a mask.

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
