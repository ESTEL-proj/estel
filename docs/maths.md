# maths.md — Mathematics of step 1

[Back to the overview](transfer/overview.md)

**Status:** draft, 16 November 2026. Sections 1 (planar 2D, reviewed against `master`) and 2 (1D) are written from the STL code by reading it; **not yet checked numerically** (task T2.6). Sections 3 (tensor products), 4 (HEALPix) and 5 (syntheses) are drafts written in the same way; each is verified by the first task of its phase (T3.0, T5.0, T6.0).

**Role of this file.** This file is the reference version of the mathematics of step 1; it changes only through pull requests that a maintainer requests and merges.

**Scope.** Step 1 (transfer): planar 2D, 1D, tensor products, HEALPix and syntheses. The definitions proposed for step 2 are in `docs/maths_step2.md`.

**Sources.**

- Planar 2D, scattering statistics and power spectra: STL-Dev `master` at `8eaf769` (v1.6.0: `STL_2D_Kernel_Torch.py`, `ST_Operator.py`, `ST_Statistics.py`, `torch_backend.py`, `CS_operator_2D_Kernel_Torch`).
- 1D: STL-Dev `2D+1D` at `8f51143` (`STL_1D_Kernel_Torch.py`, and the classes it inherits from in that branch). For the 2D scattering statistics the two branches compute the same values (their code was compared function by function); `master` is the reference.
- Tensor products: STL-Dev `2D+1D` at `8f51143` (`Tensor_Product_Kernel_Torch.py`, `ST_Operator_Tensor_Product.py`, `ST_Statistics.py`).
- HEALPix: STL-Dev `HealpixUpdate` at `90eb1a9` (`STL_Healpix_Kernel_Torch.py`, `TorchGeometry.py`), with healpix-analyse at `aac5e26` (`convol.py`, `down.py`).
- Syntheses: STL-Dev `master` at `8eaf769` (`Synthesis.py`).

**Rule.** In step 1 this tab describes what STL computes, including choices that look accidental. Points that need a decision are marked **⚠ OP-n** and listed in the last sub-section of each section: §1.11 (2D), §2.6 (1D), §3.10 (tensor products), §4.6 (HEALPix) and §5.9 (syntheses). Each entry gives the definition, its conventions, the STL function it comes from (**Source**), the oracle function that will implement it (**Oracle**, in `estel._spec`) and the tests that use it.

## Contents of this tab

- [1. Planar 2D](#1-planar-2d)
  - [1.0 Notation — M-NOT](#10-notation--m-not)
  - [1.1 Data — M-DATA-2D](#11-data--m-data-2d)
  - [1.2 Wavelets](#12-wavelets): [M-CONV-2D — "Convolution" is a correlation](#m-conv-2d--convolution-is-a-correlation) · [M-WAV-2D — Bump-steerable wavelets](#m-wav-2d--bump-steerable-wavelets)
  - [1.3 Downsampling](#13-downsampling): [M-SMOOTH-2D — Smoothing kernel](#m-smooth-2d--smoothing-kernel) · [M-DOWN — Downsampling without mask](#m-down--downsampling-without-mask) · [M-DOWN-MASK — Downsampling with missing data](#m-down-mask--downsampling-with-missing-data)
  - [1.4 Masks of the scattering layers — M-MASK](#14-masks-of-the-scattering-layers--m-mask)
  - [1.5 Borders — M-BORDER](#15-borders--m-border)
  - [1.6 Moments and first layer](#16-moments-and-first-layer): [M-MEAN — Masked mean](#m-mean--masked-mean) · [M-MOM — mean and var](#m-mom--mean-and-var) · [M-STD — Standardisation](#m-std--standardisation) · [M-W1 — First layer](#m-w1--first-layer)
  - [1.7 Scattering statistics — vanilla ScatCov](#17-scattering-statistics--vanilla-scatcov): [M-S1](#m-s1) · [M-S2](#m-s2) · [M-S3](#m-s3) · [M-S4](#m-s4) · [M-CHAN — Storage of channel pairs](#m-chan--storage-of-channel-pairs)
  - [1.8 Normalisation and reductions](#18-normalisation-and-reductions): [M-ORDER — Order of the operations of `apply`](#m-order--order-of-the-operations-of-apply) · [M-NORM](#m-norm) · [M-RED-ISO — Isotropy](#m-red-iso--isotropy) · [M-RED-ANG — Angular Fourier transform](#m-red-ang--angular-fourier-transform) · [M-RED-SCALE — Scale cosine transform](#m-red-scale--scale-cosine-transform) · [M-FLAT — Flattening](#m-flat--flattening)
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
- [3. Tensor products](#3-tensor-products)
  - [3.0 Notation — M-NOT-TP](#30-notation--m-not-tp)
  - [3.1 Data — M-DATA-TP](#31-data--m-data-tp)
  - [3.2 First layer — M-W1-TP](#32-first-layer--m-w1-tp)
  - [3.3 Low-pass filter — M-LP](#33-low-pass-filter--m-lp)
  - [3.4 Index sets — M-IDX-TP](#34-index-sets--m-idx-tp)
  - [3.5 Statistics — M-S-TP](#35-statistics--m-s-tp)
  - [3.6 Borders — M-BORDER-TP](#36-borders--m-border-tp)
  - [3.7 Normalisation and reductions — M-NORM-TP](#37-normalisation-and-reductions--m-norm-tp)
  - [3.8 Defaults — M-DEF-TP](#38-defaults--m-def-tp)
  - [3.9 Counts](#39-counts)
  - [3.10 Open points found in the tensor-product code](#310-open-points-found-in-the-tensor-product-code)
- [4. HEALPix](#4-healpix)
  - [4.0 Data — M-DATA-HPX](#40-data--m-data-hpx)
  - [4.1 Convolution — M-CONV-HPX](#41-convolution--m-conv-hpx)
  - [4.2 Wavelet — M-WAV-HPX](#42-wavelet--m-wav-hpx)
  - [4.3 Downsampling — M-DOWN-HPX](#43-downsampling--m-down-hpx)
  - [4.4 Masks and borders — M-MASK-HPX](#44-masks-and-borders--m-mask-hpx)
  - [4.5 Defaults — M-DEF-HPX](#45-defaults--m-def-hpx)
  - [4.6 Open points found in the HEALPix code](#46-open-points-found-in-the-healpix-code)
- [5. Syntheses](#5-syntheses)
  - [5.1 Target — M-SYN-TARGET](#51-target--m-syn-target)
  - [5.2 Running field — M-SYN-INIT](#52-running-field--m-syn-init)
  - [5.3 Nyquist pre-filter — M-SYN-NYQ](#53-nyquist-pre-filter--m-syn-nyq)
  - [5.4 Loss — M-SYN-LOSS](#54-loss--m-syn-loss)
  - [5.5 Optimiser — M-SYN-LBFGS](#55-optimiser--m-syn-lbfgs)
  - [5.6 Result — M-SYN-OUT](#56-result--m-syn-out)
  - [5.7 Operators — M-SYN-OPS](#57-operators--m-syn-ops)
  - [5.8 What ESTEL adds](#58-what-estel-adds)
  - [5.9 Open points found in the synthesis code](#59-open-points-found-in-the-synthesis-code)

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

**Moments and covariances.**

- M1(A) = ⟨A⟩\_w: the first moment, the mean of A over a window w (the valid pixels after the border crop).
- M2(A, B) = ⟨A conj(B)⟩\_w: the raw second moment.
- Cov(A, B) = M2(A, B) − M1(A) conj(M1(B)): the covariance, with M1 and M2 on the same window.
- The scattering covariances are covariances by definition: S2 = Cov(W\_{j,ℓ} x, W\_{j,ℓ} x), S3 = Cov(ψ\_{ℓ₃} ⋆ |W\_{j₂,ℓ₂} x|, W\_{j₃,ℓ₃} x) and S4 = Cov(ψ\_{ℓ₃} ⋆ |W\_{j₁,ℓ₁} x|, ψ\_{ℓ₃} ⋆ |W\_{j₂,ℓ₂} x|), with ψ at scale j3. S1 = M1(|W\_{j,ℓ} x|) is a first moment. STL, and ESTEL 1.0.0, compute the raw moments M2 instead (OP-36); the two are equal when M1 of one argument is zero.
- Zero-mean property: a band-pass kernel sums to zero, so a field filtered by it has M1 = 0 exactly when w is the whole periodic domain, without mask. This does not hold with a cropped border or a mask. It does not hold either when the last filter is the low-pass filter φ\_J: M1(φ\_J ⋆ |W\_{j,ℓ} x|) is S1(j), and M1(φ\_J ⋆ x) is the mean of the data.

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
- **Rotations and energies.** The kernels of orientations ℓ and ℓ + L/2 are not rotations of each other by 90 degrees: on STL's own kernels (L = 4) they differ by 4.8e-3 (ℓ = 0) and 1.9e-3 (ℓ = 1), relative to the largest coefficient. The design grid has the frequency −N/2 and not +N/2, and the bump reaches the edge of the grid; setting that unpaired frequency to zero makes the rotation exact. The energies Σ |ψ\_ℓ|² also differ by 3.2 % between the orientations along the axes and the others. **⚠ OP-58**
- **Source:** `_build_wavelet_kernel_from_ifft_crop`, `_build_bump_steerable_wavelet_kernel`. The Fourier-space bank comes from `WaveletOperator2D_FFT_torch.bump_steerable_bank` and `bump_steerable_2d`, in the Fourier-space class that ESTEL does not transfer: these two functions must be ported into `wavelets/banks.py`. **⚠ OP-23**
- **Oracle:** `spec_wavelets_2d(L, K)`. **Tests:** kernel values against STL; Littlewood–Paley sum of the effective filters (CI).

### 1.3 Downsampling

#### M-SMOOTH-2D — Smoothing kernel

The smoothing kernel is the anti-aliasing filter applied before each downsampling (M-DOWN, M-DOWN-MASK). It is a fixed 6 × 6 kernel ("cropped raised cosine, bandwidth 0.71, beta 0.35"), symmetric and separable to 2e-8; as STL writes it, it sums to 1.00000004 (**⚠ OP-59**):

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| 0.00238073 | −0.00275097 | −0.02402609 | −0.02402609 | −0.00275097 | 0.00238073 |
| −0.00275097 | 0.0031788 | 0.0277626 | 0.0277626 | 0.0031788 | −0.00275097 |
| −0.02402609 | 0.0277626 | 0.2424694 | 0.2424694 | 0.0277626 | −0.02402609 |
| (rows 4–6 mirror rows 3–1) |  |  |  |  |  |

- The constructor argument `sigma_smooth` exists but has **no effect**: the kernel is hard-coded. **⚠ OP-25**
- **Source:** `_get_smooth_kernel` (the 2×2, 3×3 and 5×5 alternatives in the code are disabled).
- **Terminology.** The smoothing kernel s is the filter of one downsampling step. The low-pass filter φ\_J (M-LP) is the effective filter of scale J. STL realises φ\_J as the cascade of the smoothing kernels of J downsamplings, which is the realisation that closes the energy sum of the pyramid; a dedicated low-pass kernel φ remains possible (LF6, LF22). In ESTEL, "low-pass filter" always means φ\_J, never s.

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
- With standardisation (`standardize=True`): see M-STD.
- **Source:** `ST_Operator.apply`, `cov`, `standardize`.

#### M-STD — Standardisation

With `standardize=True` (off by default), the field is centred and scaled before the transform, for each batch element and channel, with the masked mean ⟨·⟩ of M-MEAN:

```latex
\mu_{b,c} = \langle x_{b,c} \rangle, \qquad \sigma_{b,c}^2 = \big\langle (x_{b,c} - \mu_{b,c})^2 \big\rangle, \qquad x'_{b,c} = \frac{x_{b,c} - \mu_{b,c}}{\sigma_{b,c}}
```

- The field is centred first, then σ² is the second moment of the centred field: this is the one place where STL computes a true variance.
- Population normalisation: divided by the number of valid pixels, not minus one.
- μ and σ are kept with the statistics (`mean_pre_std`, `std_pre_std`).
- STL's function has a `mean_field` option, which averages μ and σ over the batch. The operator always calls it with `mean_field=False`; the syntheses call it with their own `mean_field` (M-SYN-TARGET).
- The inverse, used by the syntheses: x = x′ σ + μ.
- **Source:** `WaveletOperator2Dkernel_torch.standardize`, `unstandardize`; `ST_Operator.apply`.

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
- **Default pairs.** When `compute_cross_matrix` is not given, STL's code uses the upper triangle of the Nc × Nc matrix: every pair c₁ ≤ c₂, so all auto and all cross terms. STL's docstring says "auto-statistics only"; ESTEL follows the code. **⚠ OP-38**
- **Source:** `_compute_and_store_cross_cov` (`redundant_channels`), `ST_Operator.apply`.

**Sihao variant** (`has_fewer_convolutions=True`): not transferred.

### 1.8 Normalisation and reductions

#### M-ORDER — Order of the operations of `apply`

1. promotion of the array to `(Nb, Nc, …)`, and channel pairs from `compute_cross_matrix`;
2. standardisation, if asked (M-STD);
3. `mean` and `var` (M-MOM), on the possibly standardised field;
4. the scattering loop: S1 to S4 (M-S1 to M-S4);
5. the power spectrum, if asked, on the same field;
6. normalisation (M-NORM);
7. reductions, in this order and only if asked: `iso`, `angular_ft`, `scale_ft`;
8. flattening, if asked (M-FLAT).

- With standardisation, `mean` is 0 and `var` is 1 up to rounding, by construction.
- **Source:** `ST_Operator.apply`, `_finalize_compute`.

#### M-NORM

With `norm="store_ref"` (the default), the reference is the statistics being computed, and it is **stored in the operator** (`var_ref`, `S2_ref_sqrt_chan_diag`, `PS_ref_sqrt_chan_diag`) and kept for later calls. Each new call with `store_ref` recomputes it and replaces the stored one, with a printed message. With `norm="load_ref"`, the reference stored in the operator is used, unless one is passed to `apply`; it is an error if none exists. The typical use: `store_ref` on the target, then `load_ref` on the fields compared with it, so that all are normalised by the same reference. With `norm_batch_mean=True`, the reference is first averaged over the batch; the default is `norm_batch_mean=False`, as in STL, whose `ST_Operator.apply` passes False:

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

Gradients through the reference follow from the mode (LF21, closed; OP-18): with `store_ref` the reference is part of the computation and gradients flow through it; with `load_ref` it is a constant, through which no gradient flows.

- **Source:** `ST_Statistics.to_norm`, `_normalize_scatcov`, `_get_sqrt_chan_diag`.

#### M-RED-ISO — Isotropy

- **S1, S2:** averaged over ℓ.
- **S3 and S4:** averaged over the first orientation, at fixed orientation differences modulo L:

```latex
S_3^{\text{iso}}[\dots,\delta] = \frac{1}{L} \sum_{\ell_2 - \ell_1 \equiv \delta\ (\mathrm{mod}\ L)} S_3[\dots,\ell_1,\ell_2], \qquad S_4^{\text{iso}}[\dots,\delta_2,\delta_3] = \frac{1}{L} \sum_{\ell_1} S_4[\dots,\ell_1, \ell_1+\delta_2, \ell_1+\delta_3]
```

- **Conjugation.** The orientations cover \[0, π). For a real field, the filter at ℓ + L (rotated by π) gives the conjugate response: W\_{ℓ+L} x = conj(W\_ℓ x). The sums above mix no value with a conjugate of another: each orientation of the last wavelet appears once in each sum, and only the orientations of the arms wrap, which the modulus makes periodic with no conjugate (tested against STL to 1e-18). What is true is narrower: a rotation of the field by a multiple of π/L brings one orientation of the last wavelet back conjugated, so the real parts of the reduced S3 and S4 are invariant and their imaginary parts are not. **⚠ OP-5**
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

Flattening builds one vector from the statistics.

- **Order:** `mean`, `var`, S1, S2, S3, S4, then PS if it was computed; within each family, the order of the stored tensor (row-major).
- **Missing entries:** the NaN entries are removed (`keepnans=False`, the default): the structural NaN, and the means with no valid pixel.
- **Batch:** by default the batch axis is flattened with the rest. With `keep_batch_dim=True`, the result is `(Nb, n)`, and the NaN must be at the same positions for every batch element. With `mean_along_batch=True`, every family is first averaged over the batch.
- **Complex values:** two forms. With `flatten_complex=False`, the default, a complex vector, with `mean` and `var` cast to complex. With `flatten_complex=True`, a real vector: each complex coefficient becomes two reals (real part, imaginary part), except in S1, S2 and PS, of which only the real part is kept. That part is null only for the auto terms: the imaginary part of the cross-channel terms is dropped too. **⚠ OP-37**
- **`mask_st`:** an optional boolean array over the flattened vector; True keeps a coefficient. Its length must be the vector's.
- **Which form is used where.** `to_vector` of ESTEL is the real form. STL's syntheses use the complex form, with `mean_along_batch` set by their `mean_field` (M-SYN-LOSS). The two forms do not carry the same information for cross-channel S1 and S2. **⚠ OP-39**

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
| `norm` | `store_ref`, with `norm_batch_mean=False` |
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
| OP-5 | `iso` and `angular_ft` average over orientations that cover \[0, π): the real parts of the reduced S3 and S4 are invariant under rotations by multiples of π/L, their imaginary parts are not | reproduce; decide whether `iso` returns the real parts only (LF32) |
| OP-6 | Radial binning of the spectrum is not isotropic for rectangular images | restrict PS to square images in 1.0.0, or document |
| OP-7 | `nan_mask` fills missing pixels with the mean before filtering, leaking near mask edges | reproduce, document; better estimator in step 2 |
| OP-8 | The default J does not ensure that the sizes are divisible by 2^(J−1), so the default operator can fail on sizes that are not powers of two | decided: ESTEL rejects the combination at construction with a clear message; STL fails here, so parity is not concerned |
| OP-31 | STL builds its wavelet kernels at torch's default dtype, float32 unless changed, although the rest of its computation is in float64; its kernels therefore have float32 precision | ESTEL builds them in float64; the parity harness runs STL with float64 as torch's default dtype (Part II, §14.2) |
| OP-35 | STL computes S3 on the layer-2 mask alone, assuming it contains the layer-1 mask; for some masked 1D data the assumption fails and STL's S3 is NaN | ESTEL uses the union of the two masks, as the formula defines; both agree wherever STL gives a value |
| OP-58 | The 2D kernels are not rotations of each other (5e-3 at 90 degrees, from the unpaired frequency −N/2 of the design grid), and their energies differ by 3.2 % between axes and diagonals | reproduce; a requirement of the joint design (Part III, §20.10) |
| OP-59 | The 2D smoothing table sums to 1.00000004 and is separable only to 2e-8; a mean is multiplied by that sum at each downsampling | reproduce; a separable kernel is a change of values, for step 2 |

#### B. Conventions to document

| # | Point |
| --- | --- |
| OP-9 | Width factor 2.5 tuned for L = 4 only: the valid range of L to set |
| OP-10 | Kernels cropped to 5×5 and centred, not renormalised; effective filters are cascades |
| OP-11 | `var` and every STL "covariance" are non-centred second moments (OP-36) |
| OP-12 | NaN threshold τ = 0.33 in 2D, 0.5 in 1D; reweighting 1/(1−f); a pixel whose missing fraction equals τ exactly is decided by the last bit of the smoothing kernel: ESTEL reproduces STL's correctly rounded smoothing kernel (`math.fsum`), and in float32 such ties may be decided differently from STL |
| OP-13 | Borders too large for the array are silently reduced |
| OP-14 | S1/S2 store c₁ ≤ c₂ only, S3/S4 both orders; conjugate duplicates (LF20) |
| OP-15 | Scale indices of S3 named (j₁, j₂) in STL's comments, (j₂, j₃) in the code |
| OP-16 | Definition of cross-channel S1, with ε = 10⁻⁸ |
| OP-17 | `angular_ft` keeps all L harmonics by default, contrary to a code comment |
| OP-18 | Gradients through the normalisation reference: decided, they follow from the mode (LF21, closed): with store_ref the reference is part of the computation and gradients flow through it; with load_ref it is a constant |
| OP-19 | Only the real part of cross-spectra is kept |
| OP-20 | The spectrum code requires even sizes (inverse FFT width) |
| OP-21 | Spectrum border mask at 60 % of the point-spread energy, hard by default |
| OP-22 | Orientation convention: θ\_ℓ = πℓ/L + π/2 from axis 1 |
| OP-30 | A channel with cross statistics but no auto statistics gets NaN after `store_ref` normalisation; STL only warns |
| OP-32 | The per-level energy partition P of STL's kernels is only close to 1. Measured on the 5-tap kernels: in 1D, within 0.1 of 1 up to ω = 3π/4, falling to 0.23 at the Nyquist frequency; in 2D (L = 4, J = 5, with φ\_J), Λ lies between 0.22 and 1.76 in the disc of radius π, is anisotropic (more energy on the diagonals than on the axes), and is close to 0 in the corners of the frequency square |
| OP-33 | Gain convention: the analytic wavelets have a maximum gain of 1, so on real data they measure half the energy of their band; Λ counts both signs to compensate. Normalised statistics do not depend on it |
| OP-34 | Tensor products (M-LP): the low-pass filter of a block is silently dropped when the block size is not divisible by 2^(J\_a), because STL obtains it by one more downsampling than the coarsest wavelet |
| OP-36 | STL computes raw second moments, not covariances; its `cov` has a `remove_mean` argument that was never implemented. This is exact for periodic data without mask. Otherwise the forgotten term M1(A)·conj(M1(B)), relative to the covariance, measured on a test field (orders of magnitude): from 1e-6 at j = 0 to 1e-2 at the coarsest scale for non-periodic data; about ten times more with a mask covering 20 % of the pixels; of order 1 for any term whose last filter is a low-pass filter |
| OP-37 | When flattening to a real vector, STL keeps only the real part of S1, S2 and PS, assuming they are real; this holds for the auto terms, but the imaginary part of the cross-channel terms is dropped, with the information it carries |
| OP-38 | Default of `compute_cross_matrix`: STL's code computes every pair c₁ ≤ c₂; its docstring says auto statistics only. ESTEL follows the code |
| OP-39 | The real flattening drops the imaginary part of the cross-channel S1 and S2; STL's syntheses use the complex flattening, which keeps it, so `to_vector` and the synthesis loss do not see the same coefficients; to decide before T6.1 |

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

It sums to 1 exactly. The 2D smoothing kernel of M-SMOOTH-2D is s ⊗ s to 2e-8 relative, the precision of STL's literals (for example 0.49241182² = 0.2424694; OP-59). The code doubts this choice in a `TODO`; this shows that s is the 1D factor of the 2D kernel, up to that precision.

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

**Reductions in 1D.** `iso`, `angular_ft` and `scale_ft` apply to 1D statistics as to 2D ones. With L = 1, `iso` and `angular_ft` change no value. With L = 2, ℓ = 1 is the conjugate filter, not a rotation: `iso` then averages a coefficient with the one built on the conjugate filter, whose meaning is not established. **⚠ OP-40** `scale_ft` requires `iso` first, in 1D as in 2D.

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
| OP-40 | With L = 2 in 1D, `iso` averages a coefficient with its counterpart on the conjugate filter; the meaning is not established | reproduce; treat with LF32 |

## 3. Tensor products

**Status:** draft, written from the STL code by reading it, not checked numerically; verified by task T3.0.

**Source:** STL-Dev `2D+1D` at `8f51143`: `Tensor_Product_Kernel_Torch.py`, `ST_Operator_Tensor_Product.py`, `ST_Statistics.py`. Entries of sections 1 and 2 not mentioned here apply block by block.

### 3.0 Notation — M-NOT-TP

- **Blocks.** A tensor-product field has B blocks, a = 0 … B−1. Block a is a planar geometry of section 1 (2D) or 2 (1D), with its own sizes N⁽ᵃ⁾, periodicity `pbc[a]`, number of scales J\_a and number of orientations L\_a.
- **Multi-indices.** A scale is j = (j₀, …, j\_{B−1}), an orientation ℓ = (ℓ₀, …, ℓ\_{B−1}), a level d = (d₀, …, d\_{B−1}). x⁽ᵈ⁾ is the field downsampled d\_a times along block a.
- **Partial order.** j ≤ j′ means j\_a ≤ j′\_a for every block a. It is a partial order: two scales may be incomparable.
- **Block convolution.** ⋆\_a is the filtering of M-CONV-2D or of §2.1 along the axes of block a only.

### 3.1 Data — M-DATA-TP

- **Shape.** Arrays are `(Nb, Nc, *N⁽⁰⁾, …, *N⁽ᴮ⁻¹⁾)`: the spatial axes of the blocks, in block order, with no axis between them.
- **Structure.** The list of block geometries is part of the data: for example `[2D, 1D]` for space and time, `[2D, 1D, 1D]` for space, depth and time.
- **Size.** Each size of block a must be divisible by 2^(J\_a−1), as in M-DATA-2D, block by block.
- **Default J.** J\_a = ⌊log₂ min N⁽ᵃ⁾⌋ − 2, for each block. A short block gets few scales: 8 time steps give J = 1.
- **Default L.** The default of each block's geometry: 4 for a 2D block; **2 for a 1D block**, not 1 as for 1D data alone (M-WAV-1D).
- **Periodicity.** One boolean per block.
- **Missing data.** Not supported: with a mask, every mean raises `NotImplementedError`. **⚠ OP-41**
- **Source:** `STL_TensorProd_Kernel_Torch`, `get_wavelet_op`; `Wavelet_Operator_TensorProd_Kernel_Torch.__init__`, `cov`.

### 3.2 First layer — M-W1-TP

The wavelet of scale j and orientation ℓ is the product of the block wavelets. It is applied block after block, in block order, on the field brought to level j:

```latex
W_{j,\ell}\, x_c = \psi^{(B-1)}_{\ell_{B-1}} \star_{B-1} \Big( \cdots \big( \psi^{(0)}_{\ell_0} \star_0 x_c^{(j)} \big) \cdots \Big), \qquad U_{j,\ell}\, x_c = \lvert W_{j,\ell}\, x_c \rvert
```

- **One modulus, after all the blocks.** The intermediate results are complex: the second and later block convolutions filter a complex array with a complex kernel, which STL does in one complex convolution. In ESTEL this is four real convolutions (Part I, §3.1, rule 6), or the complex-to-complex convolution of LF30.
- **Orientation axes.** One per block, placed in block order in front of the spatial axes: `(Nb, Nc, L₀, …, L_{B−1}, spatial axes)`.
- **Why L = 2 in 1D blocks.** The product of analytic wavelets covers one sign combination of the block frequencies. With L\_a = 2 in a 1D block, its conjugate filter gives the other sign, so the products cover every combination that differs for a real field.
- **Order of the blocks.** Convolutions along different axes commute exactly in theory; in floating point their order changes the rounding. STL's order is block 0 first. **[to verify, T3.2]** that ESTEL's dispatch along axes gives STL's values at the parity tolerance.
- **Source:** `Wavelet_Operator_TensorProd_Kernel_Torch.apply`; `ST_Operator_TensorProd._compute_ST_loop`.

### 3.3 Low-pass filter — M-LP

With `lowpass="ST"` (STL's default), each block may contribute a low-pass filter φ⁽ᵃ⁾\_J besides its wavelets.

- **Index.** The low-pass filter of block a is the scale j\_a = J\_a, after its wavelets j\_a = 0 … J\_a − 1. It has one orientation, stored at index 0; the other orientation entries are NaN.
- **Realisation.** The cascade of the smoothing kernels: the field at level J\_a along block a, with no further computation.
- **Existence.** Block a has a low-pass filter only if each of its sizes is divisible by 2^(J\_a), one more downsampling than its wavelets need. Otherwise STL warns and gives the block no low-pass filter. **⚠ OP-34**
- **Mixed scales.** A scale with the low-pass filter in some blocks and a wavelet in at least one other is a zero-mean filter on a periodic domain, because of its wavelet factor. S1 and S2 are computed as for the wavelets.
- **The scale with the low-pass filter in every block** is the only filter without zero mean. With X the field at level (J₀, …, J\_{B−1}):

```latex
S_1 = \big\langle \lvert X - \langle X \rangle \rvert \big\rangle, \qquad S_2 = \big\langle \lvert X - \langle X \rangle \rvert^2 \big\rangle
```

  S2 is then the variance of the coarse-grained field. This is the only centred covariance of STL. **⚠ OP-42**
- **No low-pass term in S3 and S4.** They are expected to nearly vanish when their last filter contains a low-pass filter, and are not computed. Since j₁ ≤ j₂ ≤ j₃ componentwise and the low-pass filter is the coarsest scale of its block, this removes every S3 and S4 that involves a low-pass filter. **⚠ OP-43**
- **Other values of `lowpass`.** `"PS"` adds a low-pass bin to the power spectrum, `("ST", "PS")` both, `None` neither. ESTEL 1.0.0 has no tensor-product spectrum: `lowpass` takes `"ST"` or `None`.
- **Source:** `ST_Operator_TensorProd.__init__`, `_scales_S1_S2`, `_compute_ST_loop`; `Wavelet_Operator_TensorProd_Kernel_Torch.lowpass`, `lowpass_dg`.

### 3.4 Index sets — M-IDX-TP

- **Wavelet scales:** 𝒥\_w = {j : 0 ≤ j\_a ≤ J\_a − 1 for every a}.
- **All scales:** 𝒥 = {j : 0 ≤ j\_a ≤ J′\_a − 1}, with J′\_a = J\_a + 1 if block a has a low-pass filter, J\_a otherwise.
- **S1 and S2:** every j ∈ 𝒥.
- **S3:** j₃ ∈ 𝒥\_w and j₂ ≤ j₃ (componentwise, equality included).
- **S4:** j₃ ∈ 𝒥\_w and j₁ ≤ j₂ ≤ j₃.
- The loops enumerate each set in lexicographic order; the order has no effect on the values.
- For B = 1 this is the index set of sections 1 and 2.
- **Source:** `_compute_ST_loop` (`J_range`, `j3_range`, `j2_range`).

*Whether the componentwise order is the right one is LF27. The names of the indices follow the rule decided in the review of 16 November 2026 (Part IV, R-A01): j₃ is the wavelet applied last.*

### 3.5 Statistics — M-S-TP

M-S1 to M-S4 with multi-indices. D\_{j₂→j₃} downsamples block by block (M-DOWN along each block), and the second wavelet transform is the full tensor wavelet of scale j₃:

```latex
S_3[c_1,c_2,j_2,j_3,\ell_2,\ell_3] = \Big\langle \big( W_{j_3,\ell_3}\, D_{j_2\to j_3} U_{j_2,\ell_2} x_{c_1} \big)\ \overline{W_{j_3,\ell_3}\, x_{c_2}} \Big\rangle
```

```latex
S_4[c_1,c_2,j_1,j_2,j_3,\ell_1,\ell_2,\ell_3] = \Big\langle \big( W_{j_3,\ell_3}\, D_{j_1\to j_3} U_{j_1,\ell_1} x_{c_1} \big)\ \overline{\big( W_{j_3,\ell_3}\, D_{j_2\to j_3} U_{j_2,\ell_2} x_{c_2} \big)} \Big\rangle
```

Here W\_{j₃,ℓ₃} applied to a field already at level j₃ means the block convolutions of M-W1-TP without further downsampling.

- **Shapes** of the full-grid tensors, with one scale axis and one orientation axis per block:

| Family | Shape |
| --- | --- |
| S1, S2 | `(Nb, Nc, Nc, J′₀ … J′_{B−1}, L₀ … L_{B−1})` |
| S3 | `(Nb, Nc, Nc, J₀ … (j₂), J₀ … (j₃), L₀ … (ℓ₂), L₀ … (ℓ₃))` |
| S4 | `(Nb, Nc, Nc, (j₁), (j₂), (j₃), (ℓ₁), (ℓ₂), (ℓ₃))`, each group with one axis per block |

- **A correction in the frozen commit.** Before `8f51143`, S4 paired the orientations of ψ₁ and ψ₂ in every block but the first, instead of combining them: its entries with ℓ₁ ≠ ℓ₂ in those blocks were copies. The commit fixes it. Parity is with `8f51143` only.
- **Channel pairs, the cross S1, storage:** as M-S1 and M-CHAN.
- **The variant with fewer convolutions** raises `NotImplementedError` in the tensor-product loop; it is not transferred.
- **Source:** `_compute_ST_loop`.

### 3.6 Borders — M-BORDER-TP

- Each block has its own border width, computed by M-BORDER (or its 1D form) from the convolution history and the level of that block alone.
- A covariance uses, block by block, the larger border of its two arguments.
- The crop removes the border of block a along the axes of block a.
- A border too large for a block is reduced as in M-BORDER, with one printed warning (OP-13).
- **Source:** `_get_crop_border_size_from_children`, `_crop`, `cov`.

### 3.7 Normalisation and reductions — M-NORM-TP

- **Normalisation:** M-NORM with multi-indices. S3 and S4 are normalised with the wavelet scales of the S2 reference only: the low-pass scales of S2 have no counterpart in S3 and S4.
- **Reductions:** `iso`, `angular_ft` and `scale_ft` are written for one scale and one orientation axis per layer. On tensor-product statistics they fail. ESTEL raises a clear error. **⚠ OP-44**
- **Flattening:** M-FLAT applies unchanged.
- **Source:** `ST_Statistics._normalize_scatcov`, `to_iso`, `to_angular_ft`, `to_scale_ft`.

### 3.8 Defaults — M-DEF-TP

| Parameter | Default |
| --- | --- |
| J\_a | ⌊log₂ min N⁽ᵃ⁾⌋ − 2 |
| L\_a | 4 for a 2D block, 2 for a 1D block |
| `lowpass` | `"ST"` |
| `compute_PS` | True in STL; unavailable for tensor products in ESTEL 1.0.0 |
| Masks | not supported |
| Other parameters | those of each block's geometry (M-DEF, M-DEF-1D) |

### 3.9 Counts

For reference, the number of non-redundant coefficients of the STOC deliverable (Table 1): 2D, 1,474; 2D+1D space and time, 587,324; 2D+1D+1D, 18,513,572. S4 holds nearly all of them. The full-grid tensors allocate 3.9, 13 and 26 times more (Part III, §19.1).

---

### 3.10 Open points found in the tensor-product code

The rule of §1.11 applies. OP-34 also concerns tensor products.

| # | Point | Proposal |
| --- | --- | --- |
| OP-41 | Missing data are not supported in tensor products | ESTEL raises a clear error at construction; LF8 |
| OP-42 | The only centred statistics of STL are S1 and S2 of the scale with the low-pass filter in every block; the mixed scales rely on the zero mean of a wavelet factor, exact only on a periodic domain (OP-36) | reproduce; LF33 |
| OP-43 | No S3 or S4 involves a low-pass filter; they are assumed to nearly vanish | reproduce; LF6 |
| OP-44 | The reductions fail on tensor-product statistics | ESTEL raises a clear error; LF26 |
| OP-57 | `conv_history = [[]] * nblock` makes every block share one list | **[to verify, T3.1]** whether any value depends on it |

## 4. HEALPix

**Status:** draft, written from the STL and healpix-analyse code by reading it, not checked numerically; verified by task T5.0.

**Source:** STL-Dev `HealpixUpdate` at `90eb1a9` (`STL_Healpix_Kernel_Torch.py`), and healpix-analyse at `aac5e26` (`convol.py`, `down.py`), which holds the definitions of the convolution and of the downsampling. `TorchGeometry.py` of the same branch reproduces that geometry in vectorised code, for a sphere only; it is the code ESTEL ports (Part II, §9.3). The scattering loop, the statistics, the normalisation, the reductions and the flattening are those of section 1. **[to verify, T5.1]** that the loop of `HealpixUpdate` computes the same statistics as that of `master`.

### 4.0 Data — M-DATA-HPX

- **Shape.** Arrays are `(Nb, Nc, Npix)`: one pixel axis.
- **Ordering.** NESTED, required by the downsampling.
- **Resolution.** `nside` at full resolution; level d has `nside / 2^d`. In healpix-geo vocabulary, the depth is log₂ `nside`.
- **Coverage.** `cell_ids`, the NESTED indices of the pixels, or the full sky when they are not given. `pbc` means "full sky": True when the map has 12 `nside`² pixels, False for a partial sky.
- **Default J:** max(1, ⌊log₂ `nside`⌋ − 1): the coarsest level has `nside` = 4. This is not the planar default.
- **Missing data.** One mask of shape `(Npix,)`, shared by the batch and the channels, as OP-26; detected from the NaN of a single map.
- **Source:** `STL_Healpix_Kernel_Torch`, `get_wavelet_op`.

### 4.1 Convolution — M-CONV-HPX

The convolution brings the data to a fixed stencil, rotated at each pixel.

1. **Stencil.** A K × K grid of points around the North Pole, K = 5, with offsets (u, v) ∈ {−2, …, 2}². The point (u, v) is at colatitude √(u² + v²)·α and azimuth atan2(v, u), with α = √(4π / (12 `nside`²)), the mean pixel size at the current level.
2. **Rotation.** For pixel k, at colatitude θ\_k and longitude φ\_k, and gauge g = 0 … L−1, the stencil is carried by R = R\_gauge(α\_g) · R\_z(φ\_k) · R\_y(θ\_k), where R\_gauge turns around the pixel's own direction by the angle

```latex
\alpha_g = \alpha_{\text{base}} + s\,\frac{\pi g}{L}
```

   With the default gauge `"cosmo"`: α\_base = −φ\_k and s = +1 in the northern hemisphere; α\_base = +φ\_k and s = −1 in the southern one.
3. **Interpolation.** The value of the field at each rotated point is the bilinear interpolation of its 4 nearest pixels (HEALPix `get_interpol`). On a partial sky the weights of the available pixels are renormalised.
4. **Sum.** y\[g, k\] = Σ\_p W\[p\] · x\_interp\[g, k, p\], over the K² stencil points p.

- **Orientations are gauges.** The stencil turns, the kernel does not: L is the number of gauges, covering \[0, π).
- **Singularities.** Every gauge has two singular points; for `"cosmo"` they are the poles, where the orientation is not defined, and the orientation convention flips across the equator. **⚠ OP-47**
- **Pixel centres** are computed on an ellipsoid, `"WGS84"` by default: the Earth, not a sphere. **⚠ OP-45**
- **Precision.** healpix-analyse casts the kernel through float32, and stores the interpolation weights of its disk cache in float32. **⚠ OP-46**
- A complex input is not supported; the second layer convolves a modulus, which is real.
- **Source:** healpix-analyse `HealPixConv`, `_build_rotation_matrices`, `_local_kernel_grid`, `_get_interp_weights`, `_bind_support_batched`; `WaveletOperatorHealpixKernel_torch.apply`, `_get_conv`.

### 4.2 Wavelet — M-WAV-HPX

One complex kernel on the stencil offsets (u, v), with u the offset along the second stencil axis:

```latex
\psi[u, v] = \exp\Big(-\frac{u^2 + v^2}{2}\Big)\, \exp\Big(i\,\frac{\pi u}{2}\Big)
```

then the mean is subtracted, and the kernel is divided by Σ |ψ|.

- A Gaussian envelope times a plane wave: a Morlet-like kernel, not the bump-steerable wavelet of M-WAV-2D. Its normalisation (the sum of the moduli) is not that of the planar kernels (none). A comment in STL plans to align the two. **⚠ OP-48**
- The real and imaginary parts are two output channels of one convolution.
- **Source:** `WaveletOperatorHealpixKernel_torch._wavelet_kernel`.

### 4.3 Downsampling — M-DOWN-HPX

One step takes level d to d + 1. The coarse pixels are the NESTED parents of the fine ones. For a coarse pixel of centre n₀, with γ the angular distance from n₀ to the centre of a fine pixel:

```latex
x^{(d+1)}[n_0] = \frac{\sum_{\gamma \le \rho} e^{-\gamma^2 / 2\sigma^2}\, x^{(d)}[n]}{\sum_{\gamma \le \rho} e^{-\gamma^2 / 2\sigma^2}}
```

- **Radius:** ρ = 3 fine pixel sizes, a pixel size being √(4π / N\_pix) at the fine level. **Width:** σ = ρ / 2.
- The weights sum to 1 for each coarse pixel. On a partial sky, only the available fine pixels enter the sums.
- This is the smoothing of HEALPix: a Gaussian of the distance, not the 6-tap kernel of M-SMOOTH-2D. **⚠ OP-49**
- STL's `_smooth_kernel` (a K × K Gaussian) serves a separate smoothing function, which the scattering loop does not call; it is not transferred.
- **With a mask:** as M-DOWN-MASK. The invalid pixels are set to 0, each step is multiplied by the reweighting map 1/(1 − f), and a coarse pixel is invalid if f > τ, with τ = 0.33.
- **Source:** healpix-analyse `HealPixDown._build_smooth_matrix`; `WaveletOperatorHealpixKernel_torch._down_one_level`, `_downsample_masked`.

### 4.4 Masks and borders — M-MASK-HPX

- **Masks of the layers:** as M-MASK. "The wavelet support touches an invalid pixel" becomes: a convolution of the invalid map with a kernel of ones over the stencil, taken over all the L gauges, is positive.
- **Borders:** none, β = 0, on the full sky and on a partial sky alike. The edge of a partial sky is not cropped. **⚠ OP-50**
- **A second way to handle NaN** exists in STL, `nan_aware_stats`, which uses `nanmean` without precomputed masks. It is not transferred: ESTEL has one way, masks as weights.
- **Source:** `_build_masks`, `_support_mask`, `_get_crop_border_size_zero`.

### 4.5 Defaults — M-DEF-HPX

| Parameter | Default |
| --- | --- |
| J | max(1, ⌊log₂ `nside`⌋ − 1) |
| L (number of gauges) | 4 |
| K (stencil size) | 5 |
| Gauge | `"cosmo"` |
| Ellipsoid | `"WGS84"` in STL (OP-45) |
| τ (NaN threshold) | 0.33 |
| Border | none |
| Decimation | radius 3 pixels, σ = radius / 2 |
| `compute_PS` | the angular spectrum in STL; unavailable for HEALPix in ESTEL 1.0.0 (LF24) |

### 4.6 Open points found in the HEALPix code

The rule of §1.11 applies.

| # | Point | Proposal |
| --- | --- | --- |
| OP-45 | The pixel centres are computed on the WGS84 ellipsoid by default; the code ESTEL ports (`TorchGeometry`) supports the sphere only | to decide: ESTEL on the sphere only, and the parity harness runs STL with `ellipsoid="sphere"` |
| OP-46 | healpix-analyse casts the kernel through float32 and caches its interpolation weights in float32 | to decide, as OP-31: what the harness sets (cache disabled), and whether ESTEL reproduces the float32 kernel |
| OP-47 | Gauge singularities at the poles; the orientation convention flips across the equator | reproduce; document |
| OP-48 | The HEALPix wavelet is not the planar one (Gaussian times plane wave, divided by the sum of moduli), so coefficients are not comparable between geometries | reproduce; LF22 |
| OP-49 | The decimation is a Gaussian of radius 3 pixels and σ = radius/2 (G12); it is not the planar smoothing kernel | reproduce; Part III, §20.10 |
| OP-50 | No border is cropped on a partial sky | reproduce; to examine |


## 5. Syntheses

**Status:** draft, written from the STL code by reading it, not checked numerically; verified by task T6.0.

**Source:** STL-Dev `master` at `8eaf769`, `Synthesis.py`. A synthesis looks for a field u whose statistics match those of a target. Step 1 does not compare synthesised fields with STL's (Part II, §14.3); this section fixes the definitions.

### 5.1 Target — M-SYN-TARGET

1. The target is promoted to `(Nb, Nc, …)`.
2. If it has no NaN and `prefilter_Nyquist=True` (the default), it is filtered by M-SYN-NYQ.
3. It is standardised (M-STD) with `mean_field` (True by default): μ and σ are then averaged over the batch.
4. Its statistics are computed with `norm="store_ref"` and `norm_batch_mean=mean_field`.
5. They are reweighted (M-SYN-LOSS) and flattened, with `mean_along_batch=mean_field`.

- **Source:** `optimize_from_maps`.

### 5.2 Running field — M-SYN-INIT

- **Shape:** `(nbatch, Nc, …)`, with the target's spatial shape unless `running_shape` is given.
- **Start:** a Gaussian white noise of unit variance, or `init_running` if given; then M-SYN-NYQ, if `prefilter_Nyquist=True`.
- **Missing data:** the missing pixels of the running field are set to 10¹⁰, and their gradient is set to 0 at every iteration, so they never move; they are NaN in the result. **⚠ OP-53**
- **Randomness:** STL seeds the global generator of PyTorch (`seed`, 26 by default in the user-level functions). ESTEL uses explicit keys (Part I, §3.1, rule 5). **⚠ OP-55**
- **Taper.** The former point G11 mentioned a Tukey taper of the Gaussian start on non-periodic axes. STL's syntheses at the frozen commits apply none; the only cosine taper is the border mask of the spectrum estimator. **[to verify, T6.0]** whether it comes from another code base (STL-Dev-MPI) and should be reproduced.
- **Source:** `ScatteringMatchModel.__init__`.

### 5.3 Nyquist pre-filter — M-SYN-NYQ

In the 2D Fourier plane, with integer frequencies (k₁, k₂), the modes with k₁² + k₂² > (min(N₁, N₂)/2)² are set to 0.

- It removes the corners of the frequency square, which no wavelet covers (OP-32). For a rectangular field, the radius is that of the shorter side. **⚠ OP-54**
- **Source:** `apply_nyquist_filter`.

### 5.4 Loss — M-SYN-LOSS

With Φ(u) the flattened statistics of the running field, computed with `norm="load_ref"` and the target's reference:

```latex
\mathcal L(u) = \sum_i \big\lvert w_i\, \Phi_i(u) - w_i\, \Phi_i(x_{\text{target}}) \big\rvert^2
```

- **Weights.** w = 1, except 3.5 for S3 and 3.5² for S4 (`adhoc_weights`, the default). **⚠ OP-51**
- **Vector.** The complex form of M-FLAT; |·| is the complex modulus. **⚠ OP-39**
- **Batch.** With `mean_field=True`, the statistics of the running fields are averaged over the batch before the comparison: the number of running fields need not be that of the targets.
- **From statistics.** When the target is given as statistics rather than as fields, the same loss, divided by Nb if `mean_field=True`.
- **Source:** `ScatteringMatchModel.forward`, `reweight`, `optimize_from_maps`, `optimize_from_stats`.

### 5.5 Optimiser — M-SYN-LBFGS

L-BFGS of PyTorch, in one call:

| Setting | Value |
| --- | --- |
| step size (`lr`) | 1.0 |
| `max_iter` | 100 |
| `history_size` | 50 |
| line search | strong Wolfe |
| `tolerance_grad` | 10⁻¹² |
| `tolerance_change` | 10⁻¹⁵ |
| function evaluations | PyTorch's default cap, 1.25 × `max_iter` |

- These are the values that the `lbfgs` primitive of ESTEL leaves open (issue #28).
- **Source:** `optimize_lbfgs`.

### 5.6 Result — M-SYN-OUT

The optimised field is de-standardised with the target's μ and σ (M-STD), its missing pixels are set to NaN, and the batch and channel axes of size 1 are removed.

### 5.7 Operators — M-SYN-OPS

The user-level functions build two operators, one for the target and one for the running field, so that each has its own mask.

- **J.** For each, the default J minus 1 if the field is not periodic; the synthesis uses the smaller of the two. **⚠ OP-52**
- **Spectrum.** Used in the loss only if neither field has NaN.
- **Reference.** The target operator's reference is copied into the running operator.
- **Source:** `synthesize_from_maps`, `synthesize_from_stats`.

### 5.8 What ESTEL adds

Checkpoints (Part II, §9.2 and §12.9): the state of the optimiser, the running field and the key are saved every `checkpoint_every` iterations, and a run resumed from a checkpoint continues the same trajectory on the same device. No STL counterpart.

### 5.9 Open points found in the synthesis code

The rule of §1.11 applies. OP-39 also concerns syntheses.

| # | Point | Proposal |
| --- | --- | --- |
| OP-51 | Weights 3.5 for S3 and 3.5² for S4 in the loss, without a stated reason | reproduce as the default; document |
| OP-52 | The syntheses use the default J minus 1 for non-periodic fields | reproduce |
| OP-53 | Missing pixels of the running field are set to 10¹⁰ and frozen | ESTEL uses weights: the pixels are never read; to verify that the trajectory is unchanged |
| OP-54 | The Nyquist pre-filter uses the radius of the shorter side for rectangular fields | reproduce |
| OP-55 | Global seed, and a default seed of 26 in the user-level functions | ESTEL takes an explicit key; no default seed |
| OP-56 | In `synthesize_from_maps`, the default `prefilter_Nyquist` calls `init_running.isnan.any()`, which fails when an initial field is given | ESTEL computes the intended value |
