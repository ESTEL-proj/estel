# maths.md — Mathematics of step 1

[Back to the overview](transfer/overview.md)

**Status:** first draft, 12 November 2026. Extracted from the STL code by reading it; **not yet checked numerically**.

**Role of this file.** This file is the reference version of the mathematics of step 1; it changes only through pull requests that a maintainer requests and merges.

**Scope.** Step 1 (transfer). This version covers planar 2D data. 1D, tensor products, HEALPix and syntheses will follow as further sections of this tab.

**Sources.**

- Scattering statistics: STL-Dev `2D+1D` at `8f51143` (`STL_2D_Kernel_Torch.py`, `ST_Operator.py`, `ST_Statistics.py`, `torch_backend.py`).
- Power spectra: STL-Dev `master` at `8eaf769` (v1.6.0, `CS_operator_2D_Kernel_Torch`).

**Rule.** In step 1 this tab describes what STL computes, including choices that look accidental. Points that need a decision are marked **⚠ OP-n** and listed in the last sub-section of each section (§1.11 for 2D). Each entry gives the definition, its conventions, the STL function it comes from (**Source**), the oracle function that will implement it (**Oracle**, in `estel._spec`) and the tests that use it.

## Contents of this tab

- [1. Planar 2D](#1-planar-2d)
  - [1.0 Notation — M-NOT](#10-notation--m-not)
  - [1.1 Data — M-DATA-2D](#11-data--m-data-2d)
  - [1.2 Wavelets](#12-wavelets): [M-CONV-2D — "Convolution" is a correlation](#m-conv-2d--convolution-is-a-correlation) · [M-WAV-2D — Bump-steerable wavelets](#m-wav-2d--bump-steerable-wavelets)
  - [1.3 Downsampling](#13-downsampling): [M-H — Anti-aliasing kernel](#m-h--anti-aliasing-kernel) · [M-DOWN — Downsampling without mask](#m-down--downsampling-without-mask) · [M-DOWN-MASK — Downsampling with missing data](#m-down-mask--downsampling-with-missing-data)
  - [1.4 Masks of the scattering layers — M-MASK](#14-masks-of-the-scattering-layers--m-mask)
  - [1.5 Borders — M-BORDER](#15-borders--m-border)
  - [1.6 Moments and first layer](#16-moments-and-first-layer): [M-MEAN — Masked mean](#m-mean--masked-mean) · [M-MOM — mean and var](#m-mom--mean-and-var) · [M-W1 — First layer](#m-w1--first-layer)
  - [1.7 Scattering statistics — vanilla ScatCov](#17-scattering-statistics--vanilla-scatcov): [M-S1](#m-s1) · [M-S2](#m-s2) · [M-S3](#m-s3) · [M-S4](#m-s4) · [M-CHAN — Storage of channel pairs](#m-chan--storage-of-channel-pairs)
  - [1.8 Normalisation and reductions](#18-normalisation-and-reductions): [M-NORM](#m-norm) · [M-RED-ISO — Isotropy](#m-red-iso--isotropy) · [M-RED-ANG — Angular Fourier transform](#m-red-ang--angular-fourier-transform) · [M-RED-SCALE — Scale cosine transform](#m-red-scale--scale-cosine-transform) · [M-FLAT — Flattening](#m-flat--flattening)
  - [1.9 Power spectrum — M-PS-2D (from master, v1.6.0)](#19-power-spectrum--m-ps-2d-from-master-v160): [Frequencies and bins](#frequencies-and-bins) · [Estimators](#estimators) · [Choice of estimator](#choice-of-estimator)
  - [1.10 Defaults — M-DEF](#110-defaults--m-def)
  - [1.11 Open points found in the 2D code](#111-open-points-found-in-the-2d-code): [A. Values and correctness](#a-values-and-correctness) · [B. Conventions to document](#b-conventions-to-document) · [C. Transfer and API](#c-transfer-and-api)
- [2. 1D](#2-1d) — not started
- [3. Tensor products](#3-tensor-products) — not started
- [4. HEALPix](#4-healpix) — not started
- [5. Syntheses](#5-syntheses) — not started

## 1. Planar 2D

### 1.0 Notation — M-NOT

- **Field.** x\[b, c, n\], with batch index b = 0 … N\_b−1, channel c = 0 … N\_c−1, and pixel n = (n₁, n₂) with 0 ≤ nᵢ < Nᵢ. Axis 1 is the first array axis (rows), axis 2 the second (columns).
- **Resolution level.** j ≥ 0; level j has Nᵢ / 2ʲ pixels per axis. x⁽ʲ⁾ is x downsampled j times (§1.3).
- **Scales and orientations.** J scales, j = 0 … J−1; L orientations, ℓ = 0 … L−1.
- **Valid-pixel mean** of an array a with invalid mask m (True = invalid), computed after the border crop of §1.5:

```latex
\langle a \rangle_m = \frac{\sum_{n:\ \neg m[n]} a[n]}{\#\{n : \neg m[n]\}}
```

- **Complex conjugate** z̄; modulus |z|.

### 1.1 Data — M-DATA-2D

- **Shape.** Arrays are `(Nb, Nc, N1, N2)`; a 2D array is promoted to `(1, 1, N1, N2)`, a 3D one to `(1, Nc, N1, N2)`. Data must be real.
- **Size.** N₁ and N₂ must be divisible by 2^(J−1). The constraint comes from the downsampling of the scattering loop, which goes down to level J−1 and raises an error at any step with an odd size; nothing pads or resizes the input beforehand. The spectrum operator itself accepts any size (but see OP-20). The default J can violate the constraint for sizes that are not powers of two: for N = 200, J = 5 requires divisibility by 16. **⚠ OP-8**
- **Periodicity.** `pbc` is a boolean per field. The data class allows `pbc=None`, but every convolution requires a boolean, so the scattering transform fails on `pbc=None`, while the spectrum operator treats `None` as periodic. **⚠ OP-4**
- **Missing data.** A single boolean mask `mask_full_res` of shape `(N1, N2)` (True = missing), shared by every batch element and every channel. If none is given, it is detected from the NaN of the example data, which then must be a single 2D map. **⚠ OP-26**
- **Source:** `Base_DataClass`, `STL_2D_Kernel_Torch`, `ST_Operator.apply`, `get_wavelet_op`.

### 1.2 Wavelets

#### M-CONV-2D — "Convolution" is a correlation

STL's spatial filtering uses `torch.nn.functional.conv2d`, which computes a **cross-correlation**, without flipping the kernel. With p = ⌊K/2⌋ and ã the array a padded by p pixels on each side:

```latex
(w \star a)[n] = \sum_{m} w[m]\, \tilde a[n + m - p]
```

The padding is **circular** if `pbc=True`, **replicate** (edge values repeated) if `pbc=False`. The output keeps the input size.

- A complex kernel applied to a real array is computed as two real correlations (real and imaginary parts).
- **Consequence.** For the Hermitian kernels of the next entry (ψ\[−m\] = conj(ψ\[m\])) and a real field, the correlation equals the complex conjugate of the true convolution. Moduli are unchanged; the imaginary parts of cross-S2, S3 and S4 have the opposite sign to the textbook convolution definition. ESTEL reproduces the correlation. **⚠ OP-1**
- **Source:** `WaveletOperator2Dkernel_torch._conv_circular`, `_semicomplex_conv_circular`, `_get_padding_mode`.

#### M-WAV-2D — Bump-steerable wavelets

**In Fourier space**, on an N × N grid with N = 256 (a constant, independent of the data size), with integer frequencies ω = (ω₁, ω₂), ωᵢ ∈ \[−N/2, N/2):

- **Orientation of filter ℓ:** θ\_ℓ = πℓ/L + π/2. The rotated frequency is q = ω R(θ\_ℓ) (row vector times the rotation matrix), so filter ℓ responds to frequencies along the direction θ\_ℓ measured from axis 1. Orientation 0 therefore responds to oscillations along axis 2. **⚠ OP-22**
- **Radial part.** Two constants:
  - ξ₀ = N/(2√2), that is 1/(2√2) ≈ 0.354 cycles per pixel;
  - c = 2.5, the width factor.

The relative distance to ξ₀ is

```latex
r = \frac{\lvert \lVert q \rVert - \xi_0 \rvert}{\xi_0}
```

and the radial bump is

```latex
B(r) = \exp\Big(-\frac{c\,r^2}{1 - c\,r^2}\Big) \ \text{if } r < c^{-1/2}, \qquad B(r) = 0 \ \text{otherwise}
```

- **Angular part.** With φ the angle of q: A(φ) = cos^(L−1) φ if |φ| < π/2, 0 otherwise. The filter is one-sided (analytic).
- **Normalisation.**

```latex
\hat\psi_\ell = \frac{B \cdot A}{\max \lvert B \cdot A \rvert} \cdot \frac{c_L}{c_4}, \qquad c_L = \frac{2^{L-1}\,(L-1)!}{\sqrt{L\,(2(L-1))!}}
```

- **Width factor.** c = 2.5 was tuned so that the Littlewood–Paley sum is flat for L = 4 only. **⚠ OP-9**

**In pixel space**, a single K × K kernel per orientation, K = 5 (odd):

1. inverse FFT of ψ̂\_ℓ, centred (`fftshift`);
2. crop to the central K × K pixels;
3. subtract the kernel's mean, so that each kernel sums to 0;
4. no renormalisation after the crop. **⚠ OP-10**

Only the finest scale (j = 0) is built. Coarser scales are obtained by applying the same kernel to downsampled data (§1.3), so the effective filter at scale j is the cascade "smooth and downsample j times, then ψ\_ℓ".

- **Morlet.** `WType="Morlet"` is also available in STL's kernel class (built the same way from a Gaussian bank). **⚠ OP-24**
- **Source:** `_build_wavelet_kernel_from_ifft_crop`, `_build_bump_steerable_wavelet_kernel`. The Fourier-space bank comes from `WaveletOperator2D_FFT_torch.bump_steerable_bank` and `bump_steerable_2d`, in the Fourier-space class that ESTEL does not transfer: these two functions must be ported into `wavelets/banks.py`. **⚠ OP-23**
- **Oracle:** `spec_wavelets_2d(L, K)`. **Tests:** kernel values against STL; Littlewood–Paley sum of the effective filters (CI).

### 1.3 Downsampling

#### M-H — Anti-aliasing kernel

A fixed 6 × 6 kernel H ("cropped raised cosine, bandwidth 0.71, beta 0.35"), separable and symmetric, summing to 1:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| 0.00238073 | −0.00275097 | −0.02402609 | −0.02402609 | −0.00275097 | 0.00238073 |
| −0.00275097 | 0.0031788 | 0.0277626 | 0.0277626 | 0.0031788 | −0.00275097 |
| −0.02402609 | 0.0277626 | 0.2424694 | 0.2424694 | 0.0277626 | −0.02402609 |
| (rows 4–6 mirror rows 3–1) |  |  |  |  |  |

- The constructor argument `sigma_smooth` exists but has **no effect**: the kernel is hard-coded. **⚠ OP-25**
- **Source:** `_get_smooth_kernel` (the 2×2, 3×3 and 5×5 alternatives in the code are disabled).

#### M-DOWN — Downsampling without mask

One step, from level j to j+1, where x̃⁽ʲ⁾ is x⁽ʲ⁾ padded by **2 pixels before and 3 after** on each axis (circular or replicate, as in M-CONV-2D), keeping one pixel in two on each axis (stride 2):

```latex
x^{(j+1)}[n] = \sum_{m_1, m_2 = 0}^{5} H[m]\ \tilde x^{(j)}[2n + m]
```

Several levels are reached by repeating the step.

- Because H has an even size, the padding is asymmetric, and a comment in the code questions it. Each step may shift the grid by half a pixel. **⚠ OP-2**
- Complex arrays are downsampled part by part (real and imaginary).
- **Source:** `_downsample_tensor`, `downsample`.

#### M-DOWN-MASK — Downsampling with missing data

With a mask, each step is "normalised": invalid input pixels are set to 0, the result is computed as in M-DOWN, then multiplied by a reweighting map:

```latex
x^{(j+1)} = \big(\text{M-DOWN of } x^{(j)} \cdot \mathbb 1_{\text{valid}}\big) \cdot \frac{1}{1 - f^{(j+1)}}
```

where f⁽ʲ⁺¹⁾ is the fraction of invalid input contributing to each output pixel: the invalid mask, as 0/1, downsampled with the same kernel H.

- **Threshold.** An output pixel is declared invalid if f > τ, with τ = 0.33 (`downsample_nan_weight_threshold`). **⚠ OP-12**
- **Invalid output pixels** are filled with `replace_nan_value`, NaN by default; the code notes that this breaks back-propagation. ESTEL uses weights instead of NaN values; statistics only read valid pixels, so the values are unchanged. **⚠ OP-27**
- **Two families of maps.** The fractions are precomputed for the field itself ("smooth" maps, from the full-resolution mask) and for first-layer data convolved at scale j ("wavelet" maps, from the layer-1 mask of §1.4), for every target level, and for both padding modes.
- **Source:** `_build_reweighting_maps_and_scattering_layer_masks`, `downsample`.

### 1.4 Masks of the scattering layers — M-MASK

Let E be the K × K support of the wavelets (all ones), shared by all orientations.

- **Layer 0:** the full-resolution mask m⁽⁰⁾; at level j > 0, the invalid pixels after thresholded downsampling (M-DOWN-MASK).
- **Layer 1** (data convolved once, at scale j, level j): an output pixel is invalid if the wavelet support E touches an invalid input pixel, that is m₁⁽ʲ⁾ = (E ⋆ m⁽ʲ⁾) > 0.
- **Layer 2** (first convolved at j₂, modulus, downsampled to j₃, convolved again at j₃): m₂ = (E ⋆ m) > 0, where m is the layer-1 mask at j₃ if j₂ = j₃, or the downsampled-and-thresholded layer-1 mask of j₂ otherwise.
- **Covariances** use the union of the masks of their two arguments, or the larger one when one contains the other.
- **Source:** `_build_reweighting_maps_and_scattering_layer_masks`, `_find_mask`, `cov`.

### 1.5 Borders — M-BORDER

Without periodicity, pixels near the borders are contaminated by the padding, and are removed before every mean. The border width β (pixels on each side, at the current level) depends on the convolution history of the data. The default method is **fully flexible**:

- **no convolution, or `pbc=True`:** β = 0;
- **one convolution at scale j**, data at level d: β = ⌈ 2^(j−d) · ⌊K/2⌋ ⌉;
- **two convolutions** at j₂ then j₃, data at level d: β₁ = ⌈ 2^(j₂−j₃) · ⌊K/2⌋ ⌉, then β = ⌈ 2^(j₃−d) · (β₁ + ⌊K/2⌋) ⌉.

The two other methods in the code (`largest_scale_second_layer`, `layer_flexible`) are not the default. A covariance uses the larger border of its two arguments. If the border is too large for the array, it is silently reduced to ⌊(min N − 1)/2⌋, with a printed warning, once. **⚠ OP-13**

- **Source:** `_get_crop_border_size_fully_flexible`, `_crop`, `mean`, `cov`.

### 1.6 Moments and first layer

#### M-MEAN — Masked mean

⟨a⟩\_m as in M-NOT: the sum over valid pixels divided by their number, after the border crop. If no pixel is valid, the division by zero is not caught. **⚠ OP-3**

- **Source:** `torch_backend.maskmean`, `WaveletOperator2Dkernel_torch.mean`.

#### M-MOM — mean and var

```latex
\text{mean}_{b,c} = \langle x_{b,c} \rangle, \qquad \text{var}_{b,c} = \langle x_{b,c}\, \bar x_{b,c} \rangle = \langle x_{b,c}^2 \rangle
```

- `var` is **not centred**: it is the second moment, not the variance, unless the data was standardised first (`standardize=True`, off by default). The same holds for every "covariance" in STL: `cov(a, b)` = ⟨a · conj(b)⟩, without subtracting means. **⚠ OP-11**
- With standardisation: μ = ⟨x⟩, then σ² = ⟨(x − μ)²⟩, with population normalisation (divided by the number of pixels, not minus one); optionally averaged over the batch (`mean_field`).
- **Source:** `ST_Operator.apply`, `cov`, `standardize`.

#### M-W1 — First layer

At each scale j, on the field downsampled to level j:

```latex
W_{j,\ell}\, x_c = \psi_\ell \star x_c^{(j)}, \qquad U_{j,\ell}\, x_c = \lvert W_{j,\ell}\, x_c \rvert
```

### 1.7 Scattering statistics — vanilla ScatCov

All means below are masked means at the level of the data (§1.6), with the border of §1.5. Channel pairs (c₁, c₂) are those of `compute_cross_matrix`, by default the upper triangle c₁ ≤ c₂. Storage conventions are in M-CHAN.

#### M-S1

- **Auto** (c₁ = c₂ = c): S1\[c, c, j, ℓ\] = ⟨ |W\_{j,ℓ} x\_c| ⟩.
- **Cross** (c₁ < c₂), with z\_c = W\_{j,ℓ} x\_c and ε = 10⁻⁸:

```latex
S_1[c_1,c_2,j,\ell] = \Big\langle \frac{z_{c_1}}{(\lvert z_{c_1}\rvert + \varepsilon)^{1/2}}\ \overline{\frac{z_{c_2}}{(\lvert z_{c_2}\rvert + \varepsilon)^{1/2}}} \Big\rangle
```

For c₁ = c₂ this formula reduces to the auto term, up to ε; this defines what a cross-channel S1 means. **⚠ OP-16**

- **Source:** `_compute_ST_loop`; `STL_2D_Kernel_Torch.divide(epsilon=1e-8, pow=0.5)`.

#### M-S2

- **Auto:** S2\[c, c, j, ℓ\] = ⟨ |W\_{j,ℓ} x\_c|² ⟩.
- **Cross:** S2\[c₁, c₂, j, ℓ\] = ⟨ W\_{j,ℓ} x\_{c₁} · conj(W\_{j,ℓ} x\_{c₂}) ⟩.

#### M-S3

For j₂ ≤ j₃ (**including** j₂ = j₃), with D\_{j₂→j₃} the downsampling from level j₂ to level j₃ (M-DOWN or M-DOWN-MASK):

```latex
S_3[c_1,c_2,j_2,j_3,\ell_2,\ell_3] = \Big\langle \big(\psi_{\ell_3} \star D_{j_2\to j_3} U_{j_2,\ell_2} x_{c_1}\big)\ \overline{W_{j_3,\ell_3}\, x_{c_2}} \Big\rangle
```

- Both channel orders are computed separately. **⚠ OP-14**
- STL's own comments call these indices (j₁, j₂), while the code uses (j₂, j₃); this tab follows the code. **⚠ OP-15**

#### M-S4

For j₁ ≤ j₂ ≤ j₃:

```latex
S_4[c_1,c_2,j_1,j_2,j_3,\ell_1,\ell_2,\ell_3] = \Big\langle \big(\psi_{\ell_3} \star D_{j_1\to j_3} U_{j_1,\ell_1} x_{c_1}\big)\ \overline{\big(\psi_{\ell_3} \star D_{j_2\to j_3} U_{j_2,\ell_2} x_{c_2}\big)} \Big\rangle
```

#### M-CHAN — Storage of channel pairs

- **S1 and S2:** only c₁ ≤ c₂ is filled; (c₂, c₁) stays NaN, since it is the conjugate.
- **S3 and S4:** both (c₁, c₂) and (c₂, c₁) are computed and stored.
- Conjugate duplicates at j₁ = j₂ inside S4 are also stored. This is the canonical-set question, LF20. **⚠ OP-14**
- **Source:** `_compute_and_store_cross_cov` (`redundant_channels`).

**Sihao variant** (`has_fewer_convolutions=True`): not transferred.

### 1.8 Normalisation and reductions

#### M-NORM

With `norm="store_ref"` (the default), the reference is the statistics being computed, and it is **stored in the operator** (`var_ref`, `S2_ref_sqrt_chan_diag`, `PS_ref_sqrt_chan_diag`) and kept for later calls. Each new call with `store_ref` recomputes it and replaces the stored one, with a printed message. With `norm="load_ref"`, the reference stored in the operator is used, unless one is passed to `apply`; it is an error if none exists. The typical use: `store_ref` on the target, then `load_ref` on the fields compared with it, so that all are normalised by the same reference. With `norm_batch_mean=True` (the default), the reference is first averaged over the batch:

```latex
\sigma_c(j,\ell) = \sqrt{S_2^{\text{ref}}[c,c,j,\ell]}, \qquad \text{var}^{\text{ref}} = \text{var}
```

| Statistic | Divided by |
| --- | --- |
| var | var\_ref |
| mean | not normalised |
| S1\[c₁, c₂, j, ℓ\] | √(σ\_{c₁}(j, ℓ) · σ\_{c₂}(j, ℓ)) |
| S2\[c₁, c₂, j, ℓ\] | σ\_{c₁}(j, ℓ) · σ\_{c₂}(j, ℓ) |
| S3\[c₁, c₂, j₂, j₃, ℓ₂, ℓ₃\] | σ\_{c₁}(j₂, ℓ₂) · σ\_{c₂}(j₃, ℓ₃) |
| S4\[c₁, c₂, j₁, j₂, j₃, ℓ₁, ℓ₂, ℓ₃\] | σ\_{c₁}(j₁, ℓ₁) · σ\_{c₂}(j₂, ℓ₂) |
| PS\[c₁, c₂, b\] | σᴾˢ\_{c₁}(b) · σᴾˢ\_{c₂}(b), with σᴾˢ\_c(b) = √PS\_ref\[c, c, b\] |

Gradients through the reference: not specified by STL (LF21). **⚠ OP-18**

- **Source:** `ST_Statistics.to_norm`, `_normalize_scatcov`, `_get_sqrt_chan_diag`.

#### M-RED-ISO — Isotropy

- **S1, S2:** averaged over ℓ.
- **S3 and S4:** averaged over the first orientation, at fixed orientation differences modulo L:

```latex
S_3^{\text{iso}}[\dots,\delta] = \frac{1}{L} \sum_{\ell_2 - \ell_1 \equiv \delta\ (\mathrm{mod}\ L)} S_3[\dots,\ell_1,\ell_2], \qquad S_4^{\text{iso}}[\dots,\delta_2,\delta_3] = \frac{1}{L} \sum_{\ell_1} S_4[\dots,\ell_1, \ell_1+\delta_2, \ell_1+\delta_3]
```

- **The catch.** The orientations cover \[0, π). For a real field, the filter at ℓ + L (rotated by π) gives the conjugate response: W\_{ℓ+L} x = conj(W\_ℓ x). The cyclic sum therefore mixes some S3 and S4 values with what should be their conjugates, on the index pairs that wrap around. Real parts are unaffected; imaginary parts may partly cancel. To check numerically. **⚠ OP-5**
- **Source:** `to_iso`.

#### M-RED-ANG — Angular Fourier transform

An orthonormal discrete Fourier transform over the orientation axes of S3 and S4 (`norm="ortho"`), keeping the first `harmonics_angle` harmonics. The default is L, so nothing is truncated, despite a code comment mentioning "zeroth, first and second harmonic". **⚠ OP-17** The same periodicity question as M-RED-ISO applies.

- **Source:** `to_angular_ft`.

#### M-RED-SCALE — Scale cosine transform

Only after `iso`. The scales are reparametrised as (j₁, a, b) with j₂ = j₁ + a and j₃ = j₁ + b, 0 ≤ a ≤ b < `dj` (default and maximum J). An orthonormal DCT-II along j₁ is applied:

```latex
F[k, j] = \alpha_k \cos\Big(\frac{k\pi (j + 1/2)}{J}\Big), \qquad \alpha_0 = \sqrt{1/J},\ \alpha_{k>0} = \sqrt{2/J}
```

Missing entries are set to 0 for the transform and back to NaN afterwards; the first `harmonics_scale` harmonics (default J) are kept.

- **Source:** `to_scale_ft`.

#### M-FLAT — Flattening

Flattening removes the structural NaN, then keeps the coefficients where `mask_st` is True. With `flatten_complex`, complex coefficients are split into real and imaginary parts, and the null imaginary parts of S1, S2 and PS are dropped.

- **Source:** `to_flatten`.

### 1.9 Power spectrum — M-PS-2D (from master, v1.6.0)

#### Frequencies and bins

- **Radial frequencies** k ∈ {0, 1/N, …, 0.5} in cycles per pixel, with N\_max = min(N₁, N₂)/2. Each FFT mode (k₁, k₂), in mode units, is assigned to the radial index round(√(k₁² + k₂²)); modes beyond N\_max are dropped. For a rectangular image this mixes mode numbers of two different lengths, so the binning is not isotropic. **⚠ OP-6**
- **Range.** J\_PS = log₂ N\_max − 1 by default (equal to the ST default J = log₂ N − 2 for square images), J\_min = 0:

```latex
k_{\min} = \frac{0.5/\sqrt2}{2^{J_{PS}-1}}, \qquad k_{\max} = \frac{0.5/\sqrt2}{2^{J_{\min}}}
```

- **Number of bins:** n\_b = max(4, round(3 log₂(k\_max / k\_min))) by default, about 3 per octave.
- **Log-Gaussian windows** (`gaussian_rings`, the default): edges evenly spaced in log k between k\_min and k\_max; centre c\_b the middle of each log interval, width σ\_b the log interval itself, and W\_b(0) = 0:

```latex
W_b(k) = \exp\Big(-\frac{(\ln k - \ln c_b)^2}{2\sigma_b^2}\Big)
```

A "legacy" binning, built from smooth bump functions, also exists.

- **Bin normalisation:** Z\_b = Σ\_k N\_k W\_b(k), where N\_k is the number of FFT modes at radial index k (rFFT modes counted twice, except the zero and Nyquist columns).

#### Estimators

All use the orthonormal 2D FFT, x̂ = rFFT2(x) / √(N₁N₂). **Only the real part** of cross-spectra is kept. **⚠ OP-19**

**`fft`** (periodic, no missing data): the radial spectrum is

```latex
P(k) = \frac{1}{N_k} \sum_{\kappa \to k} w_\kappa\, \mathrm{Re}\big(\hat x_{c_1}(\kappa)\, \overline{\hat x_{c_2}(\kappa)}\big)
```

then binned:

```latex
PS_b = \frac{\sum_k N_k W_b(k)\, P(k)}{Z_b}
```

**Band maps**, used by the next three estimators: f\_b = irFFT2(x̂ · √W\_b(|κ|)). The inverse transform uses the width 2(K\_x − 1), which is wrong for odd sizes; ESTEL requires even sizes. **⚠ OP-20**

**`pbc_mask`** (non-periodic):

```latex
PS_b = \frac{N_{\text{pix}}}{\sum_n M_b[n]}\ \frac{1}{Z_b} \sum_n M_b[n]\, f_{b,c_1}[n]\, f_{b,c_2}[n]
```

- M\_b is a border mask per bin, "hard" by default ("smooth" is available).
- Its width is the radius containing **60 %** of the energy of the bin's point-spread function, computed from a Hankel transform of W\_b (`energy_fraction=0.60`), per bin by default (`flexible_crop`) or the largest one (`largest_crop`). **⚠ OP-21**

**`nan_mask`** (periodic, missing data):

- missing pixels are first filled with the map's mean, then band-filtered;
- the estimate sums over observed pixels only:

```latex
PS_b = \frac{N_{\text{pix}}}{N_{\text{obs}}}\ \frac{1}{Z_b} \sum_{n\ \text{observed}} f_{b,c_1}[n]\, f_{b,c_2}[n]
```

- a pixel counts as observed only if it is valid in every batch element and channel.

The filling leaks into neighbouring pixels through the filtering. **⚠ OP-7**

**`pbc_nan_mask`** (non-periodic, missing data): as `pbc_mask`, with M\_b multiplied by the observed mask.

#### Choice of estimator

Automatic, when the operator is built, from the example data. `pbc=None` counts as periodic (compare OP-4).

| Data | Estimator |
| --- | --- |
| NaN present, `pbc=False` | `pbc_nan_mask` |
| NaN present, otherwise | `nan_mask` |
| no NaN, `pbc=False` | `pbc_mask` |
| no NaN, otherwise | `fft` |

- **Source (master, `8eaf769`):** `CS_operator_2D_Kernel_Torch` (`_build_fft_index`, `_cross_pk`, `_bin_pk`, `_build_log_gaussian_bin_masks`, `estimate_pbc_border_crop`, `_apply_fft`, `_apply_pbc_mask`, `_apply_nan_mask`, `_apply_pbc_nan_mask`); `STL_2D_Kernel_Torch.get_CS_op`.

### 1.10 Defaults — M-DEF

| Parameter | Default |
| --- | --- |
| J | ⌊log₂ min(N₁, N₂)⌋ − 2 (can violate the size constraint, OP-8) |
| L | 4 |
| K (kernel size) | 5 |
| Wavelet type | bump-steerable |
| τ (NaN threshold) | 0.33 |
| Border method | fully flexible |
| `norm` | `store_ref`, with `norm_batch_mean=True` |
| `compute_PS` | True |
| `compute_cross_matrix` | upper triangle (all pairs c₁ ≤ c₂) |
| `standardize` | False |
| `replace_nan_value` | NaN |
| PS binning | log-Gaussian, about 3 bins per octave |

### 1.11 Open points found in the 2D code

**Rule for these open points.** In 1.0.0, ESTEL reproduces STL wherever STL gives a result, problems included; most open points are fixed afterwards, each as a traced change. Where STL gives no usable value (OP-3, OP-8), ESTEL raises a clear error instead.

Grouped in three categories: (A) points that affect values or correctness, (B) conventions to document, (C) transfer and API points.

#### A. Values and correctness

| # | Point | Proposal |
| --- | --- | --- |
| OP-1 | Filtering is a correlation, not a convolution; complex coefficients are the conjugates of the textbook definition | reproduce, document |
| OP-2 | Even-sized H with asymmetric padding (2/3): possible half-pixel shift per level, questioned by a `TODO` | reproduce for parity; revisit in a major |
| OP-3 | Masked mean divides by zero when no pixel is valid | decided: ESTEL raises a clear error; STL gives no usable value here, so parity is not concerned |
| OP-4 | `pbc=None` breaks the scattering transform but means "periodic" for spectra | require an explicit boolean in ESTEL |
| OP-5 | `iso` and `angular_ft` treat orientations as cyclic over \[0, π), mixing conjugate values on wrapped pairs | reproduced in 1.0.0; a problem to fix afterwards (LF32) |
| OP-6 | Radial binning of the spectrum is not isotropic for rectangular images | restrict PS to square images in 1.0.0, or document |
| OP-7 | `nan_mask` fills missing pixels with the mean before filtering, leaking near mask edges | reproduce, document; better estimator in step 2 |
| OP-8 | The default J does not ensure that the sizes are divisible by 2^(J−1), so the default operator can fail on sizes that are not powers of two | decided: ESTEL rejects the combination at construction with a clear message; STL fails here, so parity is not concerned |

#### B. Conventions to document

| # | Point |
| --- | --- |
| OP-9 | Width factor 2.5 tuned for L = 4 only: the valid range of L to set |
| OP-10 | Kernels cropped to 5×5 and centred, not renormalised; effective filters are cascades |
| OP-11 | `var` and every STL "covariance" are non-centred second moments |
| OP-12 | NaN threshold τ = 0.33; reweighting 1/(1−f) |
| OP-13 | Borders too large for the array are silently reduced |
| OP-14 | S1/S2 store c₁ ≤ c₂ only, S3/S4 both orders; conjugate duplicates (LF20) |
| OP-15 | Scale indices of S3 named (j₁, j₂) in STL's comments, (j₂, j₃) in the code |
| OP-16 | Definition of cross-channel S1, with ε = 10⁻⁸ |
| OP-17 | `angular_ft` keeps all L harmonics by default, contrary to a code comment |
| OP-18 | Gradients through the normalisation reference unspecified (LF21) |
| OP-19 | Only the real part of cross-spectra is kept |
| OP-20 | The spectrum code requires even sizes (inverse FFT width) |
| OP-21 | Spectrum border mask at 60 % of the point-spread energy, hard by default |
| OP-22 | Orientation convention: θ\_ℓ = πℓ/L + π/2 from axis 1 |

#### C. Transfer and API

| # | Point |
| --- | --- |
| OP-23 | The bump-steerable bank lives in STL's Fourier-space class; port it to `wavelets/banks.py` |
| OP-24 | STL's kernel class also offers Morlet wavelets, decided: not transferred; a general optimisation of the wavelets comes later (LF22) |
| OP-25 | `sigma_smooth` has no effect: drop it from the API |
| OP-26 | A missing-data mask must be one 2D map, shared by all batch elements and channels; auto-detection fails on batched examples |
| OP-27 | Missing pixels are filled with NaN during downsampling by default, which breaks gradients; ESTEL uses weights, with identical values, verified by parity cases with masks from phase P2 |

## 2. 1D

## 3. Tensor products

## 4. HEALPix

## 5. Syntheses
