# maths.md — Mathematics of step 1

[Back to the overview](transfer/overview.md)

**Status:** draft, 13 November 2026. Sections 1 (planar 2D, reviewed against `master`) and 2 (1D) are written from the STL code by reading it; **not yet checked numerically** (task T2.6).

**Role of this file.** This file is the reference version of the mathematics of step 1; it changes only through pull requests that a maintainer requests and merges.

**Scope.** Step 1 (transfer). This version covers planar 2D and 1D data. Tensor products, HEALPix and syntheses will follow as further sections.

**Sources.**

- Planar 2D, scattering statistics and power spectra: STL-Dev `master` at `8eaf769` (v1.6.0: `STL_2D_Kernel_Torch.py`, `ST_Operator.py`, `ST_Statistics.py`, `torch_backend.py`, `CS_operator_2D_Kernel_Torch`).
- 1D: STL-Dev `2D+1D` at `8f51143` (`STL_1D_Kernel_Torch.py`, and the classes it inherits from in that branch). For the 2D scattering statistics the two branches compute the same values (their code was compared function by function); `master` is the reference.

**Rule.** In step 1 this tab describes what STL computes, including choices that look accidental. Points that need a decision are marked **⚠ OP-n** and listed in the last sub-section of each section (§1.11 for 2D, §2.6 for 1D). Each entry gives the definition, its conventions, the STL function it comes from (**Source**), the oracle function that will implement it (**Oracle**, in `estel._spec`) and the tests that use it.

## Contents of this tab

- [1. Planar 2D](#1-planar-2d)
  - [1.0 Notation — M-NOT](#10-notation--m-not)
  - [1.1 Data — M-DATA-2D](#11-data--m-data-2d)
  - [1.2 Wavelets](#12-wavelets): [M-CONV-2D — "Convolution" is a correlation](#m-conv-2d--convolution-is-a-correlation) · [M-WAV-2D — Bump-steerable wavelets](#m-wav-2d--bump-steerable-wavelets)
  - [1.3 Downsampling](#13-downsampling): [M-SMOOTH-2D — Smoothing kernel](#m-smooth-2d--smoothing-kernel) · [M-DOWN — Downsampling without mask](#m-down--downsampling-without-mask) · [M-DOWN-MASK — Downsampling with missing data](#m-down-mask--downsampling-with-missing-data)
  - [1.4 Masks of the scattering layers — M-MASK](#14-masks-of-the-scattering-layers--m-mask)
  - [1.5 Borders — M-BORDER](#15-borders--m-border)
  - [1.6 Moments and first layer](#16-moments-and-first-layer): [M-MEAN — Masked mean](#m-mean--masked-mean) · [M-MOM — mean and var](#m-mom--mean-and-var) · [M-W1 — First layer](#m-w1--first-layer)
  - [1.7 Scattering statistics — vanilla ScatCov](#17-scattering-statistics--vanilla-scatcov): [M-S1](#m-s1) · [M-S2](#m-s2) · [M-S3](#m-s3) · [M-S4](#m-s4) · [M-CHAN — Storage of channel pairs](#m-chan--storage-of-channel-pairs)
  - [1.8 Normalisation and reductions](#18-normalisation-and-reductions): [M-NORM](#m-norm) · [M-RED-ISO — Isotropy](#m-red-iso--isotropy) · [M-RED-ANG — Angular Fourier transform](#m-red-ang--angular-fourier-transform) · [M-RED-SCALE — Scale cosine transform](#m-red-scale--scale-cosine-transform) · [M-FLAT — Flattening](#m-flat--flattening)
  - [1.9 Power spectrum — M-PS-2D (from master, v1.6.0)](#19-power-spectrum--m-ps-2d-from-master-v160): [Frequencies and bins](#frequencies-and-bins) · [Estimators](#estimators) · [Choice of estimator](#choice-of-estimator)
  - [1.10 Defaults — M-DEF](#110-defaults--m-def)
  - [1.11 Open points found in the 2D code](#111-open-points-found-in-the-2d-code): [A. Values and correctness](#a-values-and-correctness) · [B. Conventions to document](#b-conventions-to-document) · [C. Transfer and API](#c-transfer-and-api)
- [2. 1D](#2-1d)
  - [2.0 Data — M-DATA-1D](#20-data--m-data-1d)
  - [2.1 Wavelets — M-WAV-1D](#21-wavelets--m-wav-1d)
  - [2.2 Downsampling](#22-downsampling): [M-SMOOTH-1D — Smoothing kernel](#m-smooth-1d--smoothing-kernel)
  - [2.3 Masks, borders, moments and first layer](#23-masks-borders-moments-and-first-layer)
  - [2.4 Scattering statistics, normalisation and reductions](#24-scattering-statistics-normalisation-and-reductions)
  - [2.5 Defaults — M-DEF-1D](#25-defaults--m-def-1d)
  - [2.6 Open points found in the 1D code](#26-open-points-found-in-the-1d-code)
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

**Filters and scales.**

- s: the smoothing kernel, the filter of one downsampling step (M-SMOOTH-2D, M-SMOOTH-1D).
- S\_j: the cumulated smoothing of j downsampling steps, Ŝ\_j(ω) = ŝ(ω) ŝ(2ω) ⋯ ŝ(2^(j−1) ω), up to the sampling of each step.
- ψ\_ℓ: a wavelet kernel, applied at a given level. ψ\_(j,ℓ): the effective wavelet seen from full resolution, ψ̂\_(j,ℓ)(ω) = ψ̂\_ℓ(2^j ω) Ŝ\_j(ω).
- φ\_J: the low-pass filter, the effective filter of scale J. In STL and in 1.0.0, φ̂\_J = Ŝ\_J. A dedicated low-pass kernel φ may replace this realisation (LF6); there is none in 1.0.0.
- Λ: the Littlewood–Paley sum, Λ(ω) = |φ̂\_J(ω)|² + ½ Σ\_(j,ℓ) (|ψ̂\_(j,ℓ)(ω)|² + |ψ̂\_(j,ℓ)(−ω)|²). Both signs of ω are counted because the wavelets are analytic. Λ = 1 everywhere would mean that the coefficients keep all the energy of the data.
- Per-level energy partition: P(ω) = |ŝ(ω)|² + ½ Σ\_ℓ (|ψ̂\_ℓ(ω)|² + |ψ̂\_ℓ(−ω)|²). If P = 1 at every level, then Λ = 1 follows by cascade, with φ̂\_J = Ŝ\_J. A dedicated φ only acts on the last octave: it cannot make up for a partition that differs from 1 at the intermediate levels.

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
- **Source:** `WaveletOperator2Dkernel_torch._conv2d_circular`, `_semicomplex_conv2d_circular`, `_get_padding_mode`.

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
5. in STL, steps 1 to 4 run at torch's default dtype, float32 unless changed (`torch.fft.fftfreq` is called without a dtype), and the kernel is cast to complex128 afterwards. ESTEL builds the kernel in float64; the parity harness runs STL with float64 as torch's default dtype (Part II, §14.2). **⚠ OP-31**

Only the finest scale (j = 0) is built. Coarser scales are obtained by applying the same kernel to downsampled data (§1.3), so the effective filter at scale j is the cascade "smooth and downsample j times, then ψ\_ℓ".

- **Morlet.** `WType="Morlet"` is also available in STL's kernel class (built the same way from a Gaussian bank). **⚠ OP-24**
- **Source:** `_build_wavelet_kernel_from_ifft_crop`, `_build_bump_steerable_wavelet_kernel`. The Fourier-space bank comes from `WaveletOperator2D_FFT_torch.bump_steerable_bank` and `bump_steerable_2d`, in the Fourier-space class that ESTEL does not transfer: these two functions must be ported into `wavelets/banks.py`. **⚠ OP-23**
- **Oracle:** `spec_wavelets_2d(L, K)`. **Tests:** kernel values against STL; Littlewood–Paley sum of the effective filters (CI).

### 1.3 Downsampling

#### M-SMOOTH-2D — Smoothing kernel

The smoothing kernel is the anti-aliasing filter applied before each downsampling (M-DOWN, M-DOWN-MASK). It is a fixed 6 × 6 kernel ("cropped raised cosine, bandwidth 0.71, beta 0.35"), separable and symmetric, summing to 1:

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| 0.00238073 | −0.00275097 | −0.02402609 | −0.02402609 | −0.00275097 | 0.00238073 |
| −0.00275097 | 0.0031788 | 0.0277626 | 0.0277626 | 0.0031788 | −0.00275097 |
| −0.02402609 | 0.0277626 | 0.2424694 | 0.2424694 | 0.0277626 | −0.02402609 |
| (rows 4–6 mirror rows 3–1) |  |  |  |  |  |

- The constructor argument `sigma_smooth` exists but has **no effect**: the kernel is hard-coded. **⚠ OP-25**
- **Source:** `_get_smooth_kernel` (the 2×2, 3×3 and 5×5 alternatives in the code are disabled).
- **Terminology.** The smoothing kernel s is the filter of one downsampling step. The low-pass filter φ\_J (tensor-product section, §3, which will define its ID) is the effective filter of scale J. STL realises φ\_J as the cascade of the smoothing kernels of J downsamplings, which is the realisation that closes the energy sum of the pyramid; a dedicated low-pass kernel φ remains possible (LF6, LF22). In ESTEL, "low-pass filter" always means φ\_J, never s.

#### M-DOWN — Downsampling without mask

One step, from level j to j+1, where x̃⁽ʲ⁾ is x⁽ʲ⁾ padded by **2 pixels before and 3 after** on each axis (circular or replicate, as in M-CONV-2D), keeping one pixel in two on each axis (stride 2), with s the smoothing kernel (M-SMOOTH-2D):

```latex
x^{(j+1)}[n] = \sum_{m_1, m_2 = 0}^{5} s[m]\ \tilde x^{(j)}[2n + m]
```

Several levels are reached by repeating the step.

- Because the smoothing kernel has an even size, the padding is asymmetric, and a comment in the code questions it. Each step may shift the grid by half a pixel. **⚠ OP-2**
- Complex arrays are downsampled part by part (real and imaginary).
- **Source:** `_downsample_tensor`, `downsample`.

#### M-DOWN-MASK — Downsampling with missing data

With a mask, each step is "normalised": invalid input pixels are set to 0, the result is computed as in M-DOWN, then multiplied by a reweighting map:

```latex
x^{(j+1)} = \big(\text{M-DOWN of } x^{(j)} \cdot \mathbb 1_{\text{valid}}\big) \cdot \frac{1}{1 - f^{(j+1)}}
```

where f⁽ʲ⁺¹⁾ is the fraction of invalid input contributing to each output pixel: the invalid mask, as 0/1, downsampled with the same smoothing kernel.

- **Threshold.** An output pixel is declared invalid if f > τ, with τ = 0.33 (`downsample_nan_weight_threshold`). **⚠ OP-12**
- **Invalid output pixels** are filled with `replace_nan_value`, NaN by default; the code notes that this breaks back-propagation. ESTEL uses weights instead of NaN values; statistics only read valid pixels, so the values are unchanged. **⚠ OP-27**
- **Two families of maps.** The fractions are precomputed for the field itself ("smooth" maps, from the full-resolution mask) and for first-layer data convolved at scale j ("wavelet" maps, from the layer-1 mask of §1.4), for every target level, and for both padding modes.
- **Source:** `_build_reweighting_maps_and_scattering_layer_masks`, `downsample`.

### 1.4 Masks of the scattering layers — M-MASK

Let E be the K × K support of the wavelets (all ones), shared by all orientations.

- **Layer 0:** the full-resolution mask m⁽⁰⁾; at level j > 0, the invalid pixels after thresholded downsampling (M-DOWN-MASK).
- **Layer 1** (data convolved once, at scale j, level j): an output pixel is invalid if the wavelet support E touches an invalid input pixel, that is m₁⁽ʲ⁾ = (E ⋆ m⁽ʲ⁾) > 0.
- **Layer 2** (first convolved at j₂, modulus, downsampled to j₃, convolved again at j₃): m₂ = (E ⋆ m) > 0, where m is the layer-1 mask at j₃ if j₂ = j₃, or the downsampled-and-thresholded layer-1 mask of j₂ otherwise.
- **Covariances** use the union of the masks of their two arguments, or the larger one when one contains the other. **⚠ OP-35**
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

⟨a⟩\_m as in M-NOT: the sum over valid pixels divided by their number, after the border crop. If no pixel is valid, STL divides by zero and the coefficient is NaN. ESTEL keeps this NaN, decided statically: the masks and the borders tell, when the operator is built, which means have no valid pixel. These coefficients are not computed; they are NaN in the full grid, like the structural NaN, with no division 0/0 and so no NaN in gradients, and `to_vector` leaves them out. ESTEL warns once, when the operator is built, listing them, and raises an error only if a whole family has no computable coefficient. **⚠ OP-3**

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

- **Source:** `ST_Operator.apply` (the scattering loop); `STL_2D_Kernel_Torch.divide(epsilon=1e-8, pow=0.5)`.

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
- **Storage type.** S1 to S4 are stored as complex numbers, including the auto S1 and S2, which are real; `mean` and `var` are real.
- **Channels without auto-statistics.** A channel whose diagonal entry of `compute_cross_matrix` is False, but which appears in a cross pair, has no auto S1 or S2; with `store_ref`, its normalisation then gives NaN, and STL only warns. A channel in no pair at all is an error. **⚠ OP-30**
- **Source:** `_compute_and_store_cross_cov` (`redundant_channels`), `ST_Operator.apply`.

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

**Other values of `norm`.** `norm="vanilla"` computes the statistics without any normalisation. `norm` must be given: `None` in both the constructor and `apply` is an error. With `store_ref`, passing a reference to `apply` is an error.

**Order of operations.** `apply` normalises first, then applies, in this order and only if requested, `iso`, `angular_ft`, `scale_ft` and the flattening. Normalising after any reduction is an error.

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
- **Range.** J\_PS = log₂ N\_max − 1 by default (equal to the ST default J = log₂ N − 2 for square images), J\_min = 0. When the scattering operator is given an explicit J, its spectrum uses that J as J\_PS; the default applies only when J is not given:

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

**Rule for these open points.** In 1.0.0, ESTEL reproduces STL wherever STL gives a result, problems included; most open points are fixed afterwards, each as a traced change. Where STL gives no usable value (OP-8), ESTEL raises a clear error instead; a mean with no valid pixel (OP-3) gives a NaN coefficient, as in STL.

Grouped in three categories: (A) points that affect values or correctness, (B) conventions to document, (C) transfer and API points.

#### A. Values and correctness

| # | Point | Proposal |
| --- | --- | --- |
| OP-1 | Filtering is a correlation, not a convolution; complex coefficients are the conjugates of the textbook definition | reproduce, document |
| OP-2 | Even-sized smoothing kernel with asymmetric padding (2/3): possible half-pixel shift per level, questioned by a `TODO` | reproduce for parity; revisit in a major |
| OP-3 | Masked mean divides by zero when no pixel is valid | revised: the coefficient is NaN, as in STL, decided statically by the plan; ESTEL warns once, listing such coefficients, and raises only if a whole family is empty |
| OP-4 | `pbc=None` breaks the scattering transform but means "periodic" for spectra | require an explicit boolean in ESTEL |
| OP-5 | `iso` and `angular_ft` treat orientations as cyclic over \[0, π), mixing conjugate values on wrapped pairs | reproduced in 1.0.0; a problem to fix afterwards (LF32) |
| OP-6 | Radial binning of the spectrum is not isotropic for rectangular images | restrict PS to square images in 1.0.0, or document |
| OP-7 | `nan_mask` fills missing pixels with the mean before filtering, leaking near mask edges | reproduce, document; better estimator in step 2 |
| OP-8 | The default J does not ensure that the sizes are divisible by 2^(J−1), so the default operator can fail on sizes that are not powers of two | decided: ESTEL rejects the combination at construction with a clear message; STL fails here, so parity is not concerned |
| OP-31 | STL builds its wavelet kernels at torch's default dtype, float32 unless changed, although the rest of its computation is in float64; its kernels therefore have float32 precision | ESTEL builds them in float64; the parity harness runs STL with float64 as torch's default dtype (Part II, §14.2) |
| OP-35 | STL computes S3 on the layer-2 mask alone, assuming it contains the layer-1 mask; for some masked 1D data the assumption fails and STL's S3 is NaN | ESTEL uses the union of the two masks, as the formula defines; both agree wherever STL gives a value |

#### B. Conventions to document

| # | Point |
| --- | --- |
| OP-9 | Width factor 2.5 tuned for L = 4 only: the valid range of L to set |
| OP-10 | Kernels cropped to 5×5 and centred, not renormalised; effective filters are cascades |
| OP-11 | `var` and every STL "covariance" are non-centred second moments |
| OP-12 | NaN threshold τ = 0.33 in 2D, 0.5 in 1D; reweighting 1/(1−f); a pixel whose missing fraction equals τ exactly is decided by the last bit of the smoothing kernel: ESTEL reproduces STL's correctly rounded smoothing kernel (`math.fsum`), and in float32 such ties may be decided differently from STL |
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
| OP-30 | A channel with cross statistics but no auto statistics gets NaN after `store_ref` normalisation; STL only warns |
| OP-32 | The per-level energy partition P of STL's kernels is only close to 1. Measured on the 5-tap kernels: in 1D, within 0.1 of 1 up to ω = 3π/4, falling to 0.23 at the Nyquist frequency; in 2D (L = 4, J = 5, with φ\_J), Λ lies between 0.22 and 1.76 in the disc of radius π, is anisotropic (more energy on the diagonals than on the axes), and is close to 0 in the corners of the frequency square |
| OP-33 | Gain convention: the analytic wavelets have a maximum gain of 1, so on real data they measure half the energy of their band; Λ counts both signs to compensate. Normalised statistics do not depend on it |
| OP-34 | Tensor products (section 3, when written): the low-pass filter of a block is silently dropped when the block size is not divisible by 2^(J\_b), because STL obtains it by one more downsampling than the coarsest wavelet |

#### C. Transfer and API

| # | Point |
| --- | --- |
| OP-23 | The bump-steerable bank lives in STL's Fourier-space class; port it to `wavelets/banks.py` |
| OP-24 | STL's kernel class also offers Morlet wavelets, decided: not transferred; a general optimisation of the wavelets comes later (LF22) |
| OP-25 | `sigma_smooth` has no effect: drop it from the API |
| OP-26 | A missing-data mask must be one 2D map, shared by all batch elements and channels; auto-detection fails on batched examples |
| OP-27 | Missing pixels are filled with NaN during downsampling by default, which breaks gradients; ESTEL uses weights, with identical values, verified by parity cases with masks from phase P2 |

## 2. 1D

**Source:** STL-Dev `2D+1D` at `8f51143`, the only branch with 1D data. The 1D classes inherit almost everything from the 2D classes of that branch; this section only states what differs. Every entry of section 1 not mentioned below applies to 1D data with a single spatial axis: a pixel is n with 0 ≤ n < N, and every orientation axis has size L.

### 2.0 Data — M-DATA-1D

- **Shape.** Arrays are `(Nb, Nc, N)`. A 1D array is promoted to `(1, 1, N)`; a 2D array is promoted to `(1, Nc, N)`: its first axis is read as channels, not as a batch. **⚠ OP-29**
- **Size.** N must be divisible by 2^(J−1), as in M-DATA-2D, for the same reason (**⚠ OP-8**).
- **Default J.** J = ⌊log₂ N⌋ − 2, computed by the data class (`get_wavelet_op`), as in 2D. Only the 1D wavelet operator, called directly, requires J.
- **Periodicity and missing data.** As in M-DATA-2D. The missing-data mask is one 1D array of shape `(N,)`, shared by every batch element and channel (**⚠ OP-26**).
- **Spectrum.** STL computes a 1D power spectrum by default (`compute_PS=True`, with its `CS_operator_1D_Kernel_Torch`). ESTEL 1.0.0 has no 1D spectrum (Part II, §9.3): 1D statistics are computed and compared without it (LF23).
- **Source:** `STL_1D_Kernel_Torch`, `get_wavelet_op` (inherited), `ST_Operator.apply`.

### 2.1 Wavelets — M-WAV-1D

**Filtering** is the correlation of M-CONV-2D along the single axis (`conv1d`), with circular or replicate padding of ⌊K/2⌋ pixels on each side; a complex kernel on real data is two real correlations.

**In Fourier space**, on a grid of N = 256 points (a constant, independent of the data size), with integer frequencies ω ∈ {−N/2, …, N/2 − 1} and centre ξ₀ = N/2:

```latex
\hat\psi(\omega) = \exp\Big(-\frac{r^2}{1 - r^2}\Big) \ \text{if } r < 1, \qquad \hat\psi(\omega) = 0 \ \text{otherwise}, \qquad r = \frac{\lvert \omega - \xi_0 \rvert}{\xi_0}
```

- **Differences with M-WAV-2D.** No width factor (c = 1 instead of 2.5), no angular part, no normalisation (neither by the maximum nor by c_L/c_4).
- **Position of the bump.** ξ₀ = N/2 is the Nyquist frequency. The bump's support, 0 < ω < N, meets the grid only for 1 ≤ ω ≤ N/2 − 1: the filter is the lower half of a bump whose maximum, at Nyquist, is not on the grid. It vanishes at negative frequencies, so it is analytic. **⚠ OP-28**
- **In pixel space:** as in M-WAV-2D, a single kernel of K = 5 taps: inverse FFT, centred (`fftshift`), cropped to the central K taps, mean subtracted, not renormalised (**⚠ OP-10**), cast to complex. As in 2D, STL builds it at torch's default dtype (**⚠ OP-31**). The bank is computed for J scales, but only j = 0 is used; coarser scales come from downsampling, as in 2D.
- **Orientations.** L = 1 by default; L ∈ {1, 2}. With L = 2 the second filter is the conjugate of the first, ψ₁ = conj(ψ₀), which covers the negative frequencies; tensor products use L = 2 for their 1D blocks. On real data W₁x = conj(W₀x), so the L = 2 statistics are redundant there. With L = 1, `iso` and `angular_ft` change nothing, and OP-5 does not arise.
- **Morlet.** Also available in the 1D class; not transferred (**⚠ OP-24**).
- **Source:** `WaveletOperator1Dkernel_torch.bump_steerable_1d`, `bump_steerable_bank`, `_build_wavelet_kernel_from_ifft_crop`, `__init__`; `_conv_circular` (inherited).
- **Oracle:** `spec_wavelets_1d(L, K)`. **Tests:** kernel values against STL, for L = 1 and L = 2.

### 2.2 Downsampling

#### M-SMOOTH-1D — Smoothing kernel

The 1D smoothing kernel, the anti-aliasing filter applied before each downsampling, is the third row of the 2D table of M-SMOOTH-2D divided by its sum (0.49241182):

```text
s = [-0.04879268, 0.05638086, 0.49241182, 0.49241182, 0.05638086, -0.04879268]
```

It sums to 1, and the 2D smoothing kernel of M-SMOOTH-2D is exactly s ⊗ s (for example 0.49241182² = 0.2424694). The code doubts this choice in a `TODO`; the check above confirms that it is the 1D factor of the 2D kernel.

**Downsampling** with this kernel:

- **Without a mask:** M-DOWN along one axis: padding 2 before and 3 after (**⚠ OP-2**), stride 2.
- **With a mask:** M-DOWN-MASK along one axis, with the threshold **τ = 0.5** in 1D (`downsample_nan_weight_threshold=0.5`), instead of 0.33 in 2D. **⚠ OP-12**
- `sigma_smooth` has no effect, as in 2D (**⚠ OP-25**).
- **Source:** `WaveletOperator1Dkernel_torch._get_smooth_kernel`, `__init__`; `_downsample_tensor`, `downsample` (inherited).

### 2.3 Masks, borders, moments and first layer

M-MASK, M-BORDER, M-MEAN, M-MOM and M-W1 apply unchanged, with E the support of K = 5 taps and one axis: the fully flexible border widths use ⌊K/2⌋ = 2, and the border is cropped at both ends of the axis.

- **Source:** `_build_reweighting_maps_and_scattering_layer_masks`, `_get_crop_border_size_fully_flexible`, `_crop`, `mean`, `cov` (inherited, written for any number of axes).

### 2.4 Scattering statistics, normalisation and reductions

M-S1 to M-S4, M-CHAN, M-NORM, M-RED-ISO, M-RED-ANG, M-RED-SCALE and M-FLAT apply unchanged, with ℓ ∈ {0, …, L − 1}. The full-grid shapes are S1 and S2 `(Nb, Nc, Nc, J, L)`, S3 `(Nb, Nc, Nc, J, J, L, L)` and S4 `(Nb, Nc, Nc, J, J, J, L, L, L)`; with the default L = 1, every orientation axis has size 1. Without a spectrum, the normalisation has no PS term.

- **Source:** `ST_Operator._compute_ST_loop`, `_finalize_compute`; `ST_Statistics` (`2D+1D`).

### 2.5 Defaults — M-DEF-1D

| Parameter | Default |
| --- | --- |
| J | ⌊log₂ N⌋ − 2 (can violate the size constraint, OP-8) |
| L | 1 (2 for the 1D blocks of tensor products) |
| K (kernel size) | 5 |
| Wavelet type | bump-steerable |
| τ (NaN threshold) | 0.5 |
| Border method | fully flexible |
| `compute_PS` | True in STL; unavailable for 1D data in ESTEL 1.0.0 |
| Other parameters | as in M-DEF |

### 2.6 Open points found in the 1D code

The rule of §1.11 applies. OP-2, OP-8, OP-10, OP-12, OP-24, OP-25 and OP-26 also concern 1D data, as noted above.

| # | Point | Proposal |
| --- | --- | --- |
| OP-28 | The 1D wavelet is centred on the Nyquist frequency, so only the lower half of its bump exists on the grid; it has no width factor and no normalisation, unlike the 2D wavelet | reproduce; examine with the wavelet optimisation (LF22). The 1D wavelet is the upper half of a two-channel bank whose lower half is the smoothing kernel: their per-level energy partition is within 0.1 of 1 up to 3π/4 (OP-32) |
| OP-29 | A 2D array given as 1D data is read as `(Nc, N)`, not `(Nb, N)` | reproduce, document |

## 3. Tensor products

## 4. HEALPix

## 5. Syntheses
