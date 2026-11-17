# maths.md — Mathematics of step 1

[Back to the overview](transfer/overview.md)

**Status:** draft, 16 November 2026. Sections 1 (planar 2D, reviewed against `master`) and 2 (1D) are written from the STL code by reading it, and checked numerically by the oracle `estel._spec`, which agrees with ESTEL to the parity tolerance (task T2.6); ESTEL itself agrees with STL's fixtures. Sections 3 (tensor products) and 4 (HEALPix) are drafts written in the same way; each is verified by the first task of its phase (T3.0, T5.0). Section 5 (syntheses) is checked against STL's code for periodic syntheses, and numerically where it says so (task T6.0); its non-periodic and masked parts come after 1.0.0 (Part III, §16, LF47).

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

where f⁽ʲ⁺¹⁾ is the fraction of invalid input contributing to each output pixel: the invalid mask of x⁽ʲ⁾, after its own thresholding, as 0/1, downsampled with the same smoothing kernel; it is not the fraction f⁽ʲ⁾ downsampled again.

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

The two other methods in the code (`largest_scale_second_layer`, `layer_flexible`) are not the default. A covariance uses the larger border of its two arguments. If the border is too large for the array, that is if 2β ≥ min N at that level, it is silently reduced to ⌊(min N − 1)/2⌋, with a printed warning, once. **⚠ OP-13**

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
- STL's function has a `mean_field` option, which averages μ over the batch and takes for σ the square root of the batch mean of the variances (M-SYN-TARGET). The operator always calls it with `mean_field=False`; the syntheses call it with their own `mean_field` (M-SYN-TARGET).
- The inverse, used by the syntheses: x = x′ σ + μ.
- **Source:** `WaveletOperator2Dkernel_torch.standardize`, `unstandardize`; `ST_Operator.apply`.

#### M-W1 — First layer

At each scale j, on the field downsampled to level j:

```latex
W_{j,\ell}\, x_c = \psi_\ell \star x_c^{(j)}, \qquad U_{j,\ell}\, x_c = \lvert W_{j,\ell}\, x_c \rvert
```

### 1.7 Scattering statistics — vanilla ScatCov

All means below are masked means at the level of the data (§1.6), with the border of §1.5. Channel pairs (c₁, c₂) are those of `cross_matrix` (STL's `compute_cross_matrix`), by default the upper triangle c₁ ≤ c₂. Storage conventions are in M-CHAN.

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
- **Channels without auto-statistics.** A channel whose diagonal entry of `cross_matrix` is False, but which appears in a cross pair, has no auto S1 or S2; with `norm="self"` (STL's `store_ref`), its normalisation then gives NaN, and STL only warns. A channel in no pair at all is an error. **⚠ OP-30**
- **Default pairs.** When `cross_matrix` (STL's `compute_cross_matrix`) is not given, STL's code uses the upper triangle of the Nc × Nc matrix: every pair c₁ ≤ c₂, so all auto and all cross terms. STL's docstring says "auto-statistics only"; ESTEL follows the code. **⚠ OP-38**
- **Source:** `_compute_and_store_cross_cov` (`redundant_channels`), `ST_Operator.apply`.

**Sihao variant** (`has_fewer_convolutions=True`): not transferred.

### 1.8 Normalisation and reductions

#### M-ORDER — Order of the operations of `apply`

1. promotion of the array to `(Nb, Nc, …)`, and channel pairs from `cross_matrix`;
2. standardisation, if asked (M-STD);
3. `mean` and `var` (M-MOM), on the possibly standardised field;
4. the scattering loop: S1 to S4 (M-S1 to M-S4);
5. the power spectrum, if asked, on the same field;
6. normalisation (M-NORM);
7. reductions, in this order and only if asked: `iso`, `angular_fourier`, `scale_cosine` (STL's `to_iso`, `to_angular_ft`, `to_scale_ft`);
8. flattening, if asked (M-FLAT).

- With standardisation, `mean` is 0 and `var` is 1 up to rounding, by construction.
- **Source:** `ST_Operator.apply`, `_finalize_compute`.

#### M-NORM

With `norm="self"` (STL's `store_ref`; the default), the reference is the statistics being computed, and it is **stored in the operator** (`var_ref`, `S2_ref_sqrt_chan_diag`, `PS_ref_sqrt_chan_diag`) and kept for later calls. Each new call with `norm="self"` recomputes it and replaces the stored one, with a printed message. With `norm="ref"` (STL's `load_ref`), the reference stored in the operator is used, unless one is passed to `apply`; it is an error if none exists. The typical use: `norm="self"` on the target, then `norm="ref"` on the fields compared with it, so that all are normalised by the same reference. With `ref_batch_mean=True` (STL's `norm_batch_mean`), the reference is first averaged over the batch; the default is `ref_batch_mean=False`, as in STL, whose `ST_Operator.apply` passes False:

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

**Other values of `norm`.** `norm="none"` (STL's `vanilla`) computes the statistics without any normalisation. `norm` must be given: `None` in both the constructor and `apply` is an error. With `norm="self"`, passing a reference to `apply` is an error.

**Order of operations.** `apply` normalises first, then applies, in this order and only if requested, `iso`, `angular_fourier`, `scale_cosine` and the flattening. Normalising after any reduction is an error.

Gradients through the reference follow from the mode (LF21, closed; OP-18): with `norm="self"` the reference is part of the computation and gradients flow through it; with `norm="ref"` it is a constant, through which no gradient flows.

- **Source:** `ST_Statistics.to_norm`, `_normalize_scatcov`, `_get_sqrt_chan_diag`.

#### M-RED-ISO — Isotropy

- **S1, S2:** averaged over ℓ.
- **S3 and S4:** averaged over the first orientation, at fixed orientation differences modulo L:

```latex
S_3^{\text{iso}}[\dots,\delta] = \frac{1}{L} \sum_{\ell_3 - \ell_2 \equiv \delta\ (\mathrm{mod}\ L)} S_3[\dots,\ell_2,\ell_3], \qquad S_4^{\text{iso}}[\dots,\delta_2,\delta_3] = \frac{1}{L} \sum_{\ell_1} S_4[\dots,\ell_1, \ell_1+\delta_2, \ell_1+\delta_3]
```

- **Conjugation.** The orientations cover \[0, π). For a real field, the filter at ℓ + L (rotated by π) gives the conjugate response: W\_{ℓ+L} x = conj(W\_ℓ x). The sums above mix no value with a conjugate of another: each orientation of the last wavelet appears once in each sum, and only the orientations of the arms wrap, which the modulus makes periodic with no conjugate (tested against STL to 1e-18). What is true is narrower: a rotation of the field by a multiple of π/L brings one orientation of the last wavelet back conjugated, so the real parts of the reduced S3 and S4 are invariant and their imaginary parts are not. **⚠ OP-5**
- **Source:** `to_iso`.

#### M-RED-ANG — Angular Fourier transform

An orthonormal discrete Fourier transform over the orientation axes of S3 and S4 (`norm="ortho"`), keeping the first `angular_harmonics` (STL's `harmonics_angle`) harmonics. The default is L, so nothing is truncated, despite a code comment mentioning "zeroth, first and second harmonic". **⚠ OP-17** After `iso` it acts on the orientation differences: δ for S3, δ₂ and δ₃ for S4. The same periodicity question as M-RED-ISO applies.

- **Source:** `to_angular_ft`.

#### M-RED-SCALE — Scale cosine transform

Only after `iso`. The scales of S4 are reparametrised as (j₁, a, b) with j₂ = j₁ + a and j₃ = j₁ + b, 0 ≤ a ≤ b < `dj` (default and maximum J); those of S3 as (j₂, a) with j₃ = j₂ + a, 0 ≤ a < `dj`. An orthonormal DCT-II along the first scale (j₁ for S4, j₂ for S3) is applied:

```latex
F[k, j] = \alpha_k \cos\Big(\frac{k\pi (j + 1/2)}{J}\Big), \qquad \alpha_0 = \sqrt{1/J},\ \alpha_{k>0} = \sqrt{2/J}
```

Missing entries are set to 0 for the transform; afterwards, a harmonic is NaN wherever the entry at the same position was missing, as STL does: (k, a, b) where (j₁ = k, a, b) was missing for S4, (k, a) where (j₂ = k, a) was missing for S3; the first `scale_harmonics` (STL's `harmonics_scale`) harmonics (default J) are kept.

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
| `norm` | `"self"` (STL's `store_ref`), with `ref_batch_mean=False` (STL's `norm_batch_mean`) |
| `compute_PS` | True |
| `cross_matrix` (STL's `compute_cross_matrix`) | upper triangle (all pairs c₁ ≤ c₂) |
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
| OP-5 | `iso` and `angular_fourier` average over orientations that cover \[0, π): the real parts of the reduced S3 and S4 are invariant under rotations by multiples of π/L, their imaginary parts are not | reproduce; decide whether `iso` returns the real parts only (LF32) |
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
| OP-17 | `angular_fourier` (STL's `to_angular_ft`) keeps all L harmonics by default, contrary to a code comment |
| OP-18 | Gradients through the normalisation reference: decided, they follow from the mode (LF21, closed): with `norm="self"` the reference is part of the computation and gradients flow through it; with `norm="ref"` it is a constant |
| OP-19 | Only the real part of cross-spectra is kept |
| OP-20 | The spectrum code requires even sizes (inverse FFT width) |
| OP-21 | Spectrum border mask at 60 % of the point-spread energy, hard by default |
| OP-22 | Orientation convention: θ\_ℓ = πℓ/L + π/2 from axis 1 |
| OP-30 | A channel with cross statistics but no auto statistics gets NaN after `norm="self"` normalisation (STL's `store_ref`); STL only warns |
| OP-32 | The per-level energy partition P of STL's kernels is only close to 1. Measured on the 5-tap kernels: in 1D, within 0.1 of 1 up to ω = 3π/4, falling to 0.23 at the Nyquist frequency; in 2D (L = 4, J = 5, with φ\_J), Λ lies between 0.22 and 1.76 in the disc of radius π, is anisotropic (more energy on the diagonals than on the axes), and is close to 0 in the corners of the frequency square |
| OP-33 | Gain convention: the analytic wavelets have a maximum gain of 1, so on real data they measure half the energy of their band; Λ counts both signs to compensate. Normalised statistics do not depend on it |
| OP-34 | Tensor products (M-LP): the low-pass filter of a block is silently dropped when the block size is not divisible by 2^(J\_a), because STL obtains it by one more downsampling than the coarsest wavelet |
| OP-36 | STL computes raw second moments, not covariances; its `cov` has a `remove_mean` argument that was never implemented. This is exact for periodic data without mask. Otherwise the forgotten term M1(A)·conj(M1(B)), relative to the covariance, measured on a test field (orders of magnitude): from 1e-6 at j = 0 to 1e-2 at the coarsest scale for non-periodic data; about ten times more with a mask covering 20 % of the pixels; of order 1 for any term whose last filter is a low-pass filter |
| OP-37 | When flattening to a real vector, STL keeps only the real part of S1, S2 and PS, assuming they are real; this holds for the auto terms, but the imaginary part of the cross-channel terms is dropped, with the information it carries |
| OP-38 | Default of `cross_matrix` (STL's `compute_cross_matrix`): STL's code computes every pair c₁ ≤ c₂; its docstring says auto statistics only. ESTEL follows the code |
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
- **Orientations.** L = 1 by default; L ∈ {1, 2}. With L = 2 the second filter is the conjugate of the first, ψ₁ = conj(ψ₀), which covers the negative frequencies; tensor products use L = 2 for their 1D blocks. On real data W₁x = conj(W₀x), so the L = 2 statistics are redundant there. With L = 1, `iso` and `angular_fourier` change nothing, and OP-5 does not arise.
- **Morlet.** Also available in the 1D class; not transferred (**⚠ OP-24**).
- **Source:** `WaveletOperator1Dkernel_torch.bump_steerable_1d`, `bump_steerable_bank`, `_build_wavelet_kernel_from_ifft_crop`, `__init__`; `_conv_circular` (inherited).
- **Oracle:** `spec_wavelets_1d(L, K)`. **Tests:** kernel values against STL, for L = 1 and L = 2.

### 2.2 Downsampling

#### M-SMOOTH-1D — Smoothing kernel

The 1D smoothing kernel, the anti-aliasing filter applied before each downsampling, is the third row of the 2D table of M-SMOOTH-2D divided by its sum (0.49241182), the correctly rounded sum of the row (`math.fsum`; OP-12):

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

**Reductions in 1D.** `iso`, `angular_fourier` and `scale_cosine` apply to 1D statistics as to 2D ones. With L = 1, `iso` and `angular_fourier` change no value. With L = 2, ℓ = 1 is the conjugate filter, not a rotation: `iso` then averages a coefficient with the one built on the conjugate filter, whose meaning is not established. **⚠ OP-40** `scale_cosine` requires `iso` first, in 1D as in 2D.

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

**Status:** checked against STL's code, entry by entry, and numerically in the parity container where an entry says so (task T3.0, 17 November 2026). Not implemented yet: the oracle and the tests come with T3.2.

**Source:** STL-Dev `2D+1D` at `8f51143`: `Tensor_Product_Kernel_Torch.py`, `ST_Operator_Tensor_Product.py`, `ST_Statistics.py`, and what they call in `ST_Operator.py` (`apply`, `_finalize_compute`) and in the block operators (`STL_2D_Kernel_Torch.py`, `STL_1D_Kernel_Torch.py`). Entries of sections 1 and 2 not mentioned here apply block by block.

**Numerical checks.** They ran in the parity container (environment `2d1d`, PyTorch 2.15.0, float64), with `compute_PS=False`. The reference is an evaluation of the formulas of this section written apart from STL's loop: it takes from STL only the kernels of the block operators and their convolution and downsampling along one block, brings a field to its level in every block first, then applies the block convolutions. It is compared with the statistics of `ST_Operator_TensorProd`, without normalisation, on seven cases, with two maps of two channels unless said:

| Case | Blocks | Sizes | Periodic | J |
| --- | --- | --- | --- | --- |
| A | 2D, 1D | 16 × 16, 16 | yes, yes | (2, 2) |
| B | 2D, 1D | 16 × 16, 16 | no, no | (2, 2) |
| C | 2D, 1D | 16 × 16, 8 | yes, no | default: (2, 1) |
| D | 2D, 1D, 1D | 16 × 16, 8, 16 | no, yes, no | default: (2, 1, 2) |
| E | 1D, 2D | 16, 16 × 16 | no, yes | (2, 2) |
| F | 2D, 1D | 16 × 16, 12 | yes, yes | (2, 3): no low-pass filter in block 1 |
| G | 2D, 1D | 32 × 16, 32 | no, no | (3, 3); one map, one channel |

In the seven cases, S1, S2, S3 and S4 have their NaN at the same entries as the reference, and differ from it by at most 1.2e-15 of the largest coefficient of the family. "Checked" below names what a check adds to this comparison.

### 3.0 Notation — M-NOT-TP

- **Blocks.** A tensor-product field has B blocks, a = 0 … B−1. Block a is a planar geometry of section 1 (2D) or 2 (1D), with its own sizes N⁽ᵃ⁾, periodicity `pbc[a]`, number of scales J\_a and number of orientations L\_a. L = L₀ ⋯ L\_{B−1} is the number of product orientations.
- **Multi-indices.** A scale is j = (j₀, …, j\_{B−1}), an orientation ℓ = (ℓ₀, …, ℓ\_{B−1}), a level d = (d₀, …, d\_{B−1}). x⁽ᵈ⁾ is the field downsampled d\_a times along block a, by M-DOWN along the axes of that block (M-SMOOTH-2D or M-SMOOTH-1D); the downsamplings of different blocks act on different axes and commute.
- **Level of a scale.** The level of scale j is j itself: j\_a downsamplings along block a for a wavelet, and J\_a for the low-pass filter of block a (M-LP).
- **Partial order.** j ≤ j′ means j\_a ≤ j′\_a for every block a. It is a partial order: two scales may be incomparable (M-IDX-TP).
- **Block convolution.** ⋆\_a is the filtering of M-CONV-2D or of §2.1 along the axes of block a only, with the padding of that block (circular if `pbc[a]`, replicate otherwise).

### 3.1 Data — M-DATA-TP

- **Shape.** Arrays are `(Nb, Nc, *N⁽⁰⁾, …, *N⁽ᴮ⁻¹⁾)`: the spatial axes of the blocks, in block order, with no axis between them. An array with the spatial axes only is promoted to `(1, 1, …)`, one with a single leading axis to `(1, Nc, …)`. Data must be real.
- **Structure.** The list of block geometries is part of the data: for example `[2D, 1D]` for space and time, `[2D, 1D, 1D]` for space, depth and time. STL accepts any list of 2D and 1D blocks, in any order: `[1D, 2D]`, `[1D, 1D]`, `[2D, 2D]` and a single block also run. **⚠ OP-82**
- **Size.** Each size of block a must be divisible by 2^(J\_a−1), as in M-DATA-2D, block by block. Nothing checks it when the operator is built: the computation fails at the first downsampling that meets an odd size (**⚠ OP-8**).
- **J and L.** Lists with one entry per block; an entry `None`, or no list at all, gives the default of the block. **⚠ OP-83**
- **Default J.** J\_a = ⌊log₂ min N⁽ᵃ⁾⌋ − 2, for each block. A short block gets few scales: 8 time steps give J = 1.
- **Default L.** The default of each block's geometry: 4 for a 2D block; **2 for a 1D block**, not 1 as for 1D data alone (M-WAV-1D). A 1D block accepts L = 1 or 2, as in section 2.
- **Periodicity.** One boolean per block. Without it, the first convolution fails (**⚠ OP-4**).
- **Missing data.** Not supported. An operator built from data that hold a NaN, or with a mask, fails at construction: the operator of the first block raises a `NotImplementedError` whose message asks for a 2D mask. The `NotImplementedError` written in the covariance of the tensor-product operator is never reached. **⚠ OP-41** An operator built from data without NaN and applied to data that hold one raises nothing: every statistic is NaN. **⚠ OP-80**
- **Checked:** the default J of a 1D block is 1, 1, 2, 2, 3, 4 for N = 8, 12, 16, 31, 32, 64, and that of a 32 × 8 block is 1; the defaults of a 32 × 32 × 8 field are J = (3, 1) and L = (4, 2), and L = 1 for 1D data alone; arrays of shape (16, 16, 16) and (3, 16, 16, 16) give statistics with (Nb, Nc) = (1, 1) and (1, 3); a size 12 with J = 3 runs and a size 10 fails in the downsampling (`ValueError`); the three cases of missing data behave as written (0 finite value out of 33,938 in the third).
- **Source:** `STL_TensorProd_Kernel_Torch`, `get_wavelet_op`; `Wavelet_Operator_TensorProd_Kernel_Torch.__init__`; `ST_Operator.apply`.

### 3.2 First layer — M-W1-TP

The wavelet of scale j and orientation ℓ is the product of the block wavelets. It is applied block after block, in block order, on the field brought to level j:

```latex
W_{j,\ell}\, x_c = \psi^{(B-1)}_{\ell_{B-1}} \star_{B-1} \Big( \cdots \big( \psi^{(0)}_{\ell_0} \star_0 x_c^{(j)} \big) \cdots \Big), \qquad U_{j,\ell}\, x_c = \lvert W_{j,\ell}\, x_c \rvert
```

- **Block wavelets.** The kernels of the blocks are those of sections 1 and 2: one kernel of K = 5 taps per axis and per block, the same at every level (M-WAV-2D; M-WAV-1D with L = 2).
- **One modulus, after all the blocks.** The intermediate results are complex: the second and later block convolutions filter a complex array with a complex kernel, which STL does in one complex convolution. In ESTEL this is four real convolutions (Part I, §3.1, rule 6), or the complex-to-complex convolution of LF30.
- **Orientation axes.** One per block, placed in block order in front of the spatial axes: `(Nb, Nc, L₀, …, L_{B−1}, spatial axes)`.
- **Why L = 2 in 1D blocks.** The product of analytic wavelets covers one sign combination of the block frequencies. With L\_a = 2 in a 1D block, its conjugate filter gives the other sign, so the products cover every combination that differs for a real field.
- **How STL reaches it.** STL does not bring the field to level j first. It filters along block 0 a field at level j₀ in block 0 and at level 0 in the other blocks, downsamples the complex result along block 1 to level j₁ (M-DOWN, on the real and the imaginary parts), filters along block 1, and so on: the result of the blocks 0 … a is kept and serves every scale that starts with (j₀, …, j\_a). A downsampling along a block and a filtering along another act on different axes and commute exactly, with circular and with replicate padding: STL's values are those of the formula up to rounding.
- **Order of the blocks.** Convolutions along different axes commute exactly in theory; in floating point their order changes the rounding. STL's order is block 0 first. ESTEL's dispatch along axes is compared with STL's values at the parity tolerance in T3.2.
- **Checked:** the kernels of the blocks of a `[2D, 1D]` operator are bit for bit those of the 2D operator and of the 1D operator with L = 2, whose second kernel is the conjugate of the first. The comparison of the seven cases, in which the reference follows the formula and STL its own order, bounds the effect of the order of the operations by 1.2e-15. With the blocks exchanged (`[2D, 1D]` on a field, `[1D, 2D]` on the same field with its axes permuted; periodic and not), S1 to S4 agree to 1.0e-15 and are not bit for bit equal.
- **Source:** `Wavelet_Operator_TensorProd_Kernel_Torch.apply`, `downsample`; `ST_Operator_TensorProd._compute_ST_loop` (`l_data_multi_res`, `data_l1`).

### 3.3 Low-pass filter — M-LP

With `lowpass="ST"` (STL's default), each block may contribute a low-pass filter φ⁽ᵃ⁾\_J besides its wavelets.

- **Index.** The low-pass filter of block a is the scale j\_a = J\_a, after its wavelets j\_a = 0 … J\_a − 1. It has one orientation, stored at index 0; the other orientation entries are NaN.
- **Realisation.** The cascade of the smoothing kernels: the field at level J\_a along block a, with no further computation. In a scale that mixes it with wavelets of earlier blocks, it is the complex result of these blocks that is downsampled to level J\_a (M-W1-TP).
- **Existence.** Block a has a low-pass filter only if each of its sizes is divisible by 2^(J\_a), one more downsampling than its wavelets need. Otherwise STL warns (`UserWarning`) and gives the block no low-pass filter; there is then no scale with the low-pass filter in every block. **⚠ OP-34**
- **Mixed scales.** A scale with the low-pass filter in some blocks and a wavelet in at least one other is a zero-mean filter on a periodic domain, because of its wavelet factor. S1 and S2 are computed as for the wavelets, without centring.
- **The scale with the low-pass filter in every block** is the only filter without zero mean. With X the field at level (J₀, …, J\_{B−1}):

```latex
S_1 = \big\langle \lvert X - \langle X \rangle \rvert \big\rangle, \qquad S_2 = \big\langle \lvert X - \langle X \rangle \rvert^2 \big\rangle
```

  S2 is then the variance of the coarse-grained field. This is the only centred covariance of STL. ⟨·⟩ runs over every pixel of X, periodic or not (M-BORDER-TP). **⚠ OP-42**
- **Cross-channel terms.** As M-S1 and M-S2, on the filtered field of the scale; for the scale with the low-pass filter in every block, on X − ⟨X⟩.
- **No low-pass term in S3 and S4.** They are expected to nearly vanish when their last filter contains a low-pass filter, and are not computed. Since j₁ ≤ j₂ ≤ j₃ componentwise and the low-pass filter is the coarsest scale of its block, this removes every S3 and S4 that involves a low-pass filter. The wavelet scales are computed as without low-pass filters. **⚠ OP-43**
- **Other values of `lowpass`.** `"PS"` adds a low-pass bin to the power spectrum, `("ST", "PS")` both, `None` neither; `True` and `False` are read as `"ST"` and `None`, and any other value raises a `ValueError`. ESTEL 1.0.0 has no tensor-product spectrum: `lowpass` takes `"ST"` or `None`.
- **Checked:** in the cases with a low-pass filter in every block, S2 of the last scale equals the variance of X over all its pixels, computed apart, to 8.4e-16; S1 at a low-pass scale holds one value at orientation 0 of that block and NaN at the others. The mean of a mixed scale, relative to the mean of its modulus, is below 8.1e-17 when every block is periodic (A, F) and reaches 2 to 7 % when one is not (B, C, D, E, G). Case F: a size 12 with J = 3 gives the warning, the flags (yes, no), S1 of shape `(…, 3, 3, 4, 2)` and no centred scale. With `lowpass=None`, S1 and S2 are bit for bit the wavelet scales of the `"ST"` run, and S3 and S4 are bit for bit the same. `True`, `False`, `"PS"` and `("ST", "PS")` give the flags and the shapes of `"ST"`, `None`, `None` and `"ST"`.
- **Source:** `ST_Operator_TensorProd.__init__`, `_scales_S1_S2`, `_compute_ST_loop`; `Wavelet_Operator_TensorProd_Kernel_Torch.lowpass`, `lowpass_dg`.

### 3.4 Index sets — M-IDX-TP

- **Wavelet scales:** 𝒥\_w = {j : 0 ≤ j\_a ≤ J\_a − 1 for every a}, J₀ ⋯ J\_{B−1} scales.
- **All scales:** 𝒥 = {j : 0 ≤ j\_a ≤ J′\_a − 1}, with J′\_a = J\_a + 1 if block a has a low-pass filter, J\_a otherwise.
- **S1 and S2:** every j ∈ 𝒥.
- **S3:** j₃ ∈ 𝒥\_w and j₂ ≤ j₃ (componentwise, equality included).
- **S4:** j₃ ∈ 𝒥\_w and j₁ ≤ j₂ ≤ j₃.

**The partial order, written out.**

- **Definition.** For two scales, j ≤ j′ if and only if j\_a ≤ j′\_a in every block a: j′ is at least as coarse as j in every block. The relation is reflexive, antisymmetric and transitive.
- **Not total.** With two blocks or more, some scales are incomparable: neither (0, 1) ≤ (1, 0) nor (1, 0) ≤ (0, 1). In space and time, (0, 1) is finer in space and coarser in time than (1, 0): there is no S3 between these two scales, in either direction, and no S4 that holds both.
- **A product of block index sets.** (j₂, j₃) is a pair of S3 if and only if j₂,ₐ ≤ j₃,ₐ in every block, and (j₁, j₂, j₃) a triple of S4 if and only if j₁,ₐ ≤ j₂,ₐ ≤ j₃,ₐ in every block. The index set of a tensor product is therefore the Cartesian product of the index sets of its blocks (sections 1 and 2), for S3 and for S4.
- **Sizes.** S3 has ∏ₐ J\_a (J\_a + 1) / 2 pairs of scales, out of (∏ₐ J\_a)²; S4 has ∏ₐ J\_a (J\_a + 1)(J\_a + 2) / 6 triples, out of (∏ₐ J\_a)³. For J = (2, 2): 4 wavelet scales, 9 pairs out of 16, 16 triples out of 64. The pairs with j₃ = (1, 1) are the four j₂; with j₃ = (0, 1), j₂ ∈ {(0, 0), (0, 1)}; with j₃ = (1, 0), j₂ ∈ {(0, 0), (1, 0)}; with j₃ = (0, 0), j₂ = (0, 0).
- **Low-pass scales.** The low-pass scale of a block is above each of its wavelet scales, but it enters no pair and no triple: S3 and S4 take j₃ in 𝒥\_w, hence j₁ and j₂ too (M-LP).
- **Enumeration.** The loops enumerate each set in lexicographic order, the last block varying fastest. Lexicographic order extends the partial order: j ≤ j′ implies that j comes first, so every j₂ ≤ j₃ has been met when j₃ is reached. The order has no effect on the values.
- **Storage.** The full-grid tensors have one scale axis per block and per filter (M-S-TP); the entries outside the sets are NaN.
- **One block.** For B = 1 and `lowpass=None` this is the index set of sections 1 and 2. With `lowpass="ST"`, S1 and S2 of a single block have one more scale than in sections 1 and 2 (OP-82).
- **Checked:** the finite entries of S1 to S4 are those of the sets, entry by entry, in the seven cases. Their numbers per channel pair equal the formulas of §3.9 for (16, 16, 16) with J = (2, 2): 45, 45, 576, 8,192; (32, 32, 16) with J = (3, 2): 65, 65, 1,152, 20,480; (16, 16, 12) with J = (2, 3) and no low-pass filter in block 1: 54, 54, 1,152, 20,480; (16, 16, 8, 16) with J = (2, 1, 2): 135, 135, 2,304, 65,536. A tensor product of one 2D block with `lowpass=None` gives bit for bit the statistics of the 2D operator (16 × 16, periodic).
- **Source:** `_compute_ST_loop` (`J_range`, `j3_range`, `j2_range`, `is_lowpass`).

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

- **`mean` and `var`.** M-MOM over the spatial axes of all the blocks together, without border; the standardisation M-STD likewise.
- **S1 and S2.** M-S1 and M-S2 on U\_{j,ℓ}, for every scale of 𝒥; the low-pass scales follow M-LP.
- **How STL reaches the second layer.** As for the first layer (M-W1-TP), STL interleaves the downsamplings and the block convolutions: the modulus U\_{j₂} is brought to level j₃ along block 0 and filtered along block 0, the complex result is downsampled along block 1 and filtered along block 1, and so on, and the result of the blocks 0 … a serves every j₃ that starts with the same indices. The values are those of the formulas up to rounding.
- **Shapes** of the full-grid tensors, with one scale axis and one orientation axis per block:

| Family | Shape |
| --- | --- |
| S1, S2 | `(Nb, Nc, Nc, J′₀ … J′_{B−1}, L₀ … L_{B−1})` |
| S3 | `(Nb, Nc, Nc, J₀ … (j₂), J₀ … (j₃), L₀ … (ℓ₂), L₀ … (ℓ₃))` |
| S4 | `(Nb, Nc, Nc, (j₁), (j₂), (j₃), (ℓ₁), (ℓ₂), (ℓ₃))`, each group with one axis per block |

- **A correction in the frozen commit.** Before `8f51143`, S4 paired the orientations of ψ₁ and ψ₂ in every block but the first, instead of combining them: its entries with ℓ₁ ≠ ℓ₂ in those blocks were copies. The commit fixes it. Parity is with `8f51143` only.
- **Channel pairs, the cross S1, storage:** as M-S1 and M-CHAN. S1 and S2 are stored as complex numbers, with a null imaginary part for the auto terms.
- **The variant with fewer convolutions** raises `NotImplementedError` in the tensor-product loop, after the S1 and S2 of the first scale; it is not transferred.
- **Checked:** the comparison of the seven cases covers S1 to S4, auto and cross-channel, entry by entry, hence the place of every scale and orientation axis and the S4 of the frozen commit. For a 32 × 32 × 8 field with the defaults, the shapes are S1 and S2 `(1, 1, 1, 4, 2, 4, 2)`, S3 `(1, 1, 1, 3, 1, 3, 1, 4, 2, 4, 2)` and S4 `(1, 1, 1, 3, 1, 3, 1, 3, 1, 4, 2, 4, 2, 4, 2)`. With `standardize=True`, `mean` is 2e-17 and `var` − 1 is 2e-16.
- **Source:** `_compute_ST_loop` (`data_l1m`, `data_l1m_multi_res`, `data_l1m_l2`); `ST_Operator.apply`.

### 3.6 Borders — M-BORDER-TP

- Each block has its own border width, computed by M-BORDER (or its 1D form) from the convolution history and the level of that block alone: 0 if the block is periodic.
- **In a non-periodic block,** with ⌊K/2⌋ = 2: β\_a = 2 for a first-layer field at a wavelet scale; β\_a = ⌈2^(j₂,ₐ − j₃,ₐ) · 2⌉ + 2 for a second-layer field, that is 4, 3, 3, … for j₃,ₐ − j₂,ₐ = 0, 1, 2, …
- **The low-pass filter is not a convolution** in this count: along a block taken at its low-pass scale, β\_a = 0, although each step of the cascade pads the block by replication. The field X of M-LP and the data (`mean`, `var`) have no border in any block. **⚠ OP-81**
- A covariance uses, block by block, the larger border of its two arguments.
- The crop removes the border of block a along the axes of block a.
- A border too large for a block (2β\_a ≥ the smallest size of the block at that level) is reduced as in M-BORDER, block by block, with one printed warning for the operator (OP-13).
- **Checked:** the reference of the cases B, C, D, E and G computes its borders from the rules above, the reduction included (it occurs in each of them: a border of 4 on 8 pixels becomes 3), and agrees with STL to 1.2e-15.
- **Source:** `_get_crop_border_size_from_children`, `_crop`, `cov`; `mean`, `square_mean` (inherited from the 2D operator).

### 3.7 Normalisation and reductions — M-NORM-TP

- **Reference.** As in M-NORM: the square root of the channel diagonal of S2, with the shape of S2 without one channel axis, `(Nb, Nc, J′₀ …, L₀ …)`; it holds the low-pass scales, and NaN where S2 does.
- **Normalisation:** M-NORM with multi-indices. S1 and S2 are normalised at every scale of 𝒥; the normalised auto S2 of the scale with the low-pass filter in every block is 1 with `norm="self"`. S3 and S4 are normalised with the wavelet scales of the S2 reference only: the low-pass scales of S2 have no counterpart in S3 and S4. S3 is divided by the reference of (c₁, j₂, ℓ₂) times that of (c₂, j₃, ℓ₃); S4 by that of (c₁, j₁, ℓ₁) times that of (c₂, j₂, ℓ₂).
- **Reductions:** `iso`, `angular_fourier` and `scale_cosine` are written for one scale and one orientation axis per layer. On tensor-product statistics, STL's `to_iso` and `to_scale_ft` raise a `TypeError`, and so does `to_angular_ft` with its default number of harmonics; asked through the operator, the error comes after the whole computation. Two cases are not clean failures: `to_iso` has already averaged S1 and S2 over their last orientation axis when it fails, and `to_angular_ft` with an integer number of harmonics runs, as a Fourier transform over the last two (S3) or three (S4) orientation axes, which belong to different blocks. ESTEL raises a clear error. **⚠ OP-44**
- **Flattening:** M-FLAT applies unchanged.
- **`select`** is an empty method in STL (it returns 1), and nothing names the blocks.
- **Checked** (16 × 16 × 16, J = (2, 2), block 0 not periodic, two maps of two channels): the reference has shape `(2, 2, 3, 3, 4, 2)`, and `(1, 2, 3, 3, 4, 2)` with the batch mean; the normalised S1 to S4 equal those computed without normalisation, divided as written, bit for bit; the normalised S2 of the centred scale is 1 − 2e-16. The five outcomes of the reductions are those written; `to_flatten` returns 8,860 complex coefficients for one map of one channel (33,938 with the NaN).
- **Source:** `ST_Statistics._normalize_scatcov`, `to_norm`, `to_iso`, `to_angular_ft`, `to_scale_ft`, `to_flatten`, `select`.

### 3.8 Defaults — M-DEF-TP

| Parameter | Default |
| --- | --- |
| J\_a | ⌊log₂ min N⁽ᵃ⁾⌋ − 2 |
| L\_a | 4 for a 2D block, 2 for a 1D block |
| K (kernel size) | 5 in every block |
| Wavelet type | bump-steerable in every block |
| Border method | fully flexible in every block |
| `lowpass` | `"ST"` |
| `norm` | as in M-DEF |
| `compute_PS` | True in STL; unavailable for tensor products in ESTEL 1.0.0 |
| Masks | not supported |
| Other parameters | those of each block's geometry (M-DEF, M-DEF-1D) |

**Checked:** an operator built with no argument on a 32 × 32 × 8 field has these values.

### 3.9 Counts

**What STL computes.** Per map and per channel pair, with λ\_a = 1 if block a has a low-pass filter and 0 otherwise, and L = ∏ₐ L\_a:

| Family | Coefficients computed | Entries of the full-grid tensor |
| --- | --- | --- |
| S1, S2 | ∏ₐ (J\_a L\_a + λ\_a) | ∏ₐ (J\_a + λ\_a) L\_a |
| S3 | L² ∏ₐ J\_a (J\_a + 1) / 2 | L² ∏ₐ J\_a² |
| S4 | L³ ∏ₐ J\_a (J\_a + 1)(J\_a + 2) / 6 | L³ ∏ₐ J\_a³ |

Checked against the finite entries of STL's tensors (M-IDX-TP). For 16 × 16 × 16 with J = (2, 2): 8,192 coefficients of S4 in a tensor of 32,768.

**Non-redundant coefficients.** For reference, the number of non-redundant coefficients of the STOC deliverable (Table 1): 2D, 1,474; 2D+1D space and time, 587,324; 2D+1D+1D, 18,513,572. S4 holds nearly all of them. The full-grid tensors allocate 3.9, 13 and 26 times more (Part III, §19.1). These numbers are quoted, not derived here.

---

### 3.10 Open points found in the tensor-product code

The rule of §1.11 applies. OP-4, OP-8, OP-13 and OP-34 also concern tensor products, as noted above. OP-80 to OP-83 come from the check of T3.0.

| # | Point | Proposal |
| --- | --- | --- |
| OP-41 | Missing data are not supported in tensor products: the construction fails in the operator of the first block, with a `NotImplementedError` that asks for a 2D mask | ESTEL raises a clear error at construction; LF8 |
| OP-42 | The only centred statistics of STL are S1 and S2 of the scale with the low-pass filter in every block; the mixed scales rely on the zero mean of a wavelet factor, exact only on a periodic domain (OP-36). Measured: the mean of a mixed scale is 2 to 7 % of the mean of its modulus when a block is not periodic, on 16 to 32 pixels | reproduce; LF33 |
| OP-43 | No S3 or S4 involves a low-pass filter; they are assumed to nearly vanish | reproduce; LF6 |
| OP-44 | The reductions are not defined on tensor-product statistics. In STL they raise a `TypeError`, after the computation; `to_iso` fails with S1 and S2 already averaged over one axis, and `to_angular_ft` with an integer number of harmonics returns a transform over orientation axes of different blocks, without error | ESTEL raises a clear error, before the computation; LF26 |
| OP-57 | `conv_history = [[]] * nblock` makes every block share one list. Closed (T3.0): no value depends on it. STL never modifies these lists in place, it replaces them; with one list per block given explicitly, S1 to S4 are bit for bit the same (case B) | nothing to reproduce |
| OP-80 | An operator built from data without NaN and applied to data that hold a NaN returns NaN for every statistic, without error; with the spectrum, the spectrum raises a `ValueError` | to decide with the errors on masks (Q12) |
| OP-81 | Along a block taken at its low-pass scale the border is 0, also when the block is not periodic: the cascade of smoothing kernels, padded by replication at each step, is not counted as a convolution. The variance of the coarse-grained field and the mixed scales therefore use pixels that the padding has reached | reproduce; examine with LF6 and LF33 |
| OP-82 | STL accepts a tensor product of one block. With `lowpass=None` its statistics are bit for bit those of the plain geometry; with the default `lowpass="ST"`, S1 and S2 have one more scale. J and L are then lists of one entry | to decide with the parity matrix (Q12) whether ESTEL accepts one block |
| OP-83 | Arguments of the operator. A block shorter than 8 pixels has a default J of 0, and the construction fails with an `IndexError` in the kernel code. A list J that holds `None` is completed in place: the caller's list is modified. An integer J raises a `TypeError`, an integer L an `AssertionError` | ESTEL validates the arguments at construction, with clear errors, and leaves the caller's objects unchanged |

## 4. HEALPix

**Status:** 17 November 2026 (task T5.0). Every entry is checked against the code of STL `HealpixUpdate` and of healpix-analyse, and numerically, in the parity container, where an entry says "Checked": the data conventions, the wavelet, the convolution on the full sky and on a partial sky, the gauges, the downsampling, the masks and the defaults. The numerical checks run in float64, at `nside` 8 and 16 (64 for the gauges), with `ellipsoid="sphere"` and the cache of healpix-analyse disabled unless said; their references are written in NumPy from the definitions below, with healpy for the pixel centres and the bilinear interpolation. Not checked: the scattering loop of `HealpixUpdate` against that of `master` (T5.1), the angular spectrum (LF24), and the definitions of the STL-Dev-MPI `CLAUDE.md` §3, a private repository that this check could not read. **[to verify]**

**Source:** STL-Dev `HealpixUpdate` at `90eb1a9` (`STL_Healpix_Kernel_Torch.py`), and healpix-analyse at `aac5e26` (`convol.py`, `down.py`, `healpix_interp.py`), which holds the definitions of the convolution and of the downsampling. `TorchGeometry.py` of the same branch reproduces that geometry in vectorised code, for a sphere and the gauges `"phi"` and `"cosmo"` only; it is the code ESTEL ports (Part II, §9.3). The scattering loop, the statistics, the normalisation, the reductions and the flattening are those of section 1: `ST_Operator.py` of `HealpixUpdate` differs from that of `master` in the indexing of the pixel axes (one axis instead of two), in the options of the spectrum and in methods that the loop does not call. **[to verify, T5.1]** that the two loops compute the same statistics.

### 4.0 Data — M-DATA-HPX

- **Shape.** Arrays are `(Nb, Nc, Npix)`: one pixel axis. A 1D array is promoted to `(1, 1, Npix)` and a 2D array to `(1, Nc, Npix)`, as in M-DATA-2D.
- **Ordering.** NESTED, required by the downsampling.
- **Resolution.** `nside`, a power of 2, at full resolution; level d has `nside / 2^d`. In healpix-geo vocabulary, the depth is log₂ `nside`; healpix-analyse calls it `level`. For a full sky, `nside` is read from Npix = 12 `nside`²; for a partial sky it must be given.
- **Coverage.** `cell_ids`, the NESTED indices of the pixels, or the full sky when they are not given. `pbc` means "full sky": True when the map has 12 `nside`² pixels, False for a partial sky. A partial sky of 12 n² pixels given without `nside` is taken for a full sky of `nside` = n. **⚠ OP-91**
- **Order of `cell_ids`.** Ascending. With `cell_ids` in another order, healpix-analyse returns the values of the convolution in a wrong order and reads the input of the downsampling in a wrong order, without an error. **⚠ OP-90**
- **Default J:** max(1, ⌊log₂ `nside`⌋ − 1): the coarsest level has `nside` = 4 (for `nside` ≥ 4). This is not the planar default.
- **Missing data.** One mask of shape `(Npix,)`, shared by the batch and the channels, as OP-26; detected from the NaN of the example data, which then must be a single 1D map (a batch raises an error).
- **Means.** Every pixel has the same weight (HEALPix pixels have equal areas): ⟨·⟩ is the plain mean over the pixels, or over the valid pixels (M-MEAN).
- **Checked:** J = 1, 1, 2, 3, 4 for `nside` = 2, 4, 8, 16, 32; `pbc` True for a full sky and False for a disc; a map of 47 pixels without `nside` raises, and a disc of 48 = 12 × 2² pixels without `nside` becomes a full sky of `nside` 2; the mask is inferred from the NaN of one 1D map (17 pixels), and a `(2, Npix)` example with NaN raises; for one map of `nside` 8 the statistics have the shapes of section 1 with J = 2, L = 4 (`mean` `(1, 1)`, S1 `(1, 1, 1, 2, 4)`, S4 `(1, 1, 1, 2, 2, 2, 4, 4, 4)`).
- **Source:** `STL_Healpix_Kernel_Torch` (`__post_init__`, `_infer_N0`, `get_wavelet_op`), `ST_Operator.apply`.

### 4.1 Convolution — M-CONV-HPX

The convolution brings the data to a fixed stencil, rotated at each pixel.

1. **Stencil.** A K × K grid of points around the North Pole, K = 5, with offsets (u, v) ∈ {−2, …, 2}²; in the flattened stencil, v runs along the first axis and u along the second, p = (v + 2) K + (u + 2). The point (u, v) is at colatitude √(u² + v²)·α and azimuth atan2(v, u), with α = √(4π / (12 `nside`²)), the mean pixel size at the current level.
2. **Rotation.** For pixel k, at colatitude θ\_k and longitude φ\_k, and gauge g = 0 … L−1, the stencil is carried by R = R\_gauge(α\_g) · R\_z(φ\_k) · R\_y(θ\_k). R\_z R\_y carries the pole to the pixel, the u axis of the stencil onto e\_θ (towards the south) and the v axis onto e\_φ (towards the east). R\_gauge turns around the pixel's own direction, counter-clockwise seen from outside the sphere, by the angle

```latex
\alpha_g = \alpha_{\text{base}} + s\,\frac{\pi g}{L}
```

   so that the u axis of gauge g is cos α\_g · e\_θ + sin α\_g · e\_φ. With the default gauge `"cosmo"`: α\_base = −φ\_k and s = +1 in the northern hemisphere (θ\_k ≤ π/2, the equator included); α\_base = +φ\_k and s = −1 in the southern one. With the gauge `"phi"`: α\_base = 0 and s = +1.
3. **Interpolation.** The value of the field at each rotated point is the bilinear interpolation of its 4 nearest pixels (HEALPix `get_interpol`, on the sphere); the 4 weights sum to 1.
4. **Sum.** y\[g, k\] = Σ\_p W\[p\] · x\_interp\[g, k, p\], over the K² stencil points p, with W the kernel of M-WAV-HPX. There is no bias.

- **Shapes.** `(Nb, Nc, Npix)` gives `(Nb, Nc, L, Npix)`, complex; every map is convolved on its own. A complex input is not supported (STL raises); the second layer convolves a modulus, which is real.
- **Orientations are gauges.** The stencil turns, the kernel does not: L is the number of gauges, covering \[0, π).
- **The gauge `"cosmo"`.** Its u axis of gauge 0 is cos φ · e\_θ − sin φ · e\_φ in the north and cos φ · e\_θ + sin φ · e\_φ in the south. It is regular at both poles: around the North Pole the u axis tends to the Cartesian direction +x, around the South Pole to −x. It is discontinuous across the equator: at longitude φ, the u axes of gauge g on the two sides make the angle 2φ − 2πg/L, so the orientations agree, as directions modulo π, at four longitudes only. The southern frame is the mirror image of the northern one through the equatorial plane, with u reversed: for a map mirrored through the equator, the convolution at the mirrored pixel is the complex conjugate of the convolution of the map (ψ\[−u, v\] is the conjugate of ψ\[u, v\]), so its modulus is the same. The pixels of the equatorial ring follow the northern rule and do not have this symmetry. The gauge `"phi"` is singular at the two poles. **⚠ OP-47**
- **Partial sky.** A neighbour that is not in `cell_ids` is dropped, and the weights of the remaining neighbours of that stencil point are renormalised to sum to 1. A stencil point with no neighbour in `cell_ids` takes the 4 neighbours of the stencil centre instead: it reads the value of the pixel itself. Nothing is cropped, and nothing marks the pixels whose stencil is truncated. **⚠ OP-50**
- **Pixel centres** are computed by healpix-geo on an ellipsoid, `"WGS84"` by default: the Earth, not a sphere. The longitudes are unchanged; the latitude of a centre becomes the geodetic latitude whose authalic latitude is that of the sphere, at most 0.128° away (at ±45°). It is then used as a latitude on the unit sphere: the rotation, the stencil and the interpolation are those of the sphere. With `ellipsoid="sphere"` the centres are those of HEALPix. **⚠ OP-45**
- **Precision.** healpix-analyse casts the kernel through float32, whatever the dtype of the operator: the kernel used differs from that of M-WAV-HPX by 5e-8 of its largest modulus, and its sum is 9e-9 instead of 0. Its disk cache, enabled by default in STL, stores the interpolation weights in float32: the first operator of a geometry computes them in float64, and every later one, in the same process or not, reads them back in float32. **⚠ OP-46**
- **Checked** (sphere, cache disabled): STL's `apply` equals steps 1 to 4 written in NumPy with the float32 kernel, to 2e-14 of the largest modulus (`nside` 8 and 16, full sky), and to 4e-8 with the float64 kernel; on a disc of 154 pixels (`nside` 16), to 9e-15 with the partial-sky rule above, where 121 pixels have a truncated stencil and 1320 of the 15400 stencil points have no neighbour; on the pixels with a complete stencil the partial sky equals the full sky (4e-16), on the others the relative rms difference is 0.48 for a white noise. The rotation matrices of healpix-analyse equal the formula of step 2 to 1e-15, for `"phi"` and `"cosmo"`. Gauge `"cosmo"` at `nside` 64: around each pole the u axis of gauge 0 is within 0.52° of ±x; across the equator the angle is 2φ − 2πg/L to 1.2° (rings 2 `nside` ± 1), from 1.2° to 180°; mirror relation to 4e-14 off the equatorial ring, not on it. With the default `"WGS84"`: `apply` equals the same formula with the WGS84 centres to 2e-14, and differs from the sphere by a relative rms of 2.0 % (`nside` 8) and 4.2 % (`nside` 16) for a white noise. With the cache enabled, a second operator has the weights of the first cast through float32 (difference 3e-8), and S3 and S4 change by up to 2.5e-7 in relative value.
- **Source:** healpix-analyse `HealPixConv`, `_build_rotation_matrices`, `_local_kernel_grid`, `_get_interp_weights`, `_bind_support_batched`, `healpix_interp.get_interp_weights`; `WaveletOperatorHealpixKernel_torch.apply`, `_get_conv`.

### 4.2 Wavelet — M-WAV-HPX

One complex kernel on the stencil offsets (u, v) of M-CONV-HPX, with u the offset along the second stencil axis:

```latex
\psi[u, v] = \exp\Big(-\frac{u^2 + v^2}{2}\Big)\, \exp\Big(i\,\frac{\pi u}{2}\Big)
```

then the mean is subtracted, and the kernel is divided by Σ |ψ|, taken after the subtraction.

- A Gaussian envelope, of width 1 pixel, times a plane wave along u, of period 4 pixels: a Morlet-like kernel, not the bump-steerable wavelet of M-WAV-2D. Its normalisation (the sum of the moduli) is not that of the planar kernels (none). A comment in STL plans to align the two. **⚠ OP-48**
- ψ\[−u, v\] is the conjugate of ψ\[u, v\], and ψ\[u, −v\] = ψ\[u, v\].
- The same kernel serves every level and every gauge. The real and imaginary parts are two output channels of one convolution.
- STL's two wavelet types, `"HealpixWavelet"` (the default) and `"Morlet"`, give this same kernel.
- **Checked:** STL's kernel equals the formula to 3e-19; its sum is 5e-17 and the sum of its moduli is 1; the two symmetries hold to 3e-20; both wavelet types give it; the kernel that the convolution uses is its cast through float32 (M-CONV-HPX).
- **Source:** `WaveletOperatorHealpixKernel_torch._wavelet_kernel`.

### 4.3 Downsampling — M-DOWN-HPX

One step takes level d to d + 1. The coarse pixels are the NESTED parents of the fine ones, in ascending order. For a coarse pixel of centre n₀, with γ the angular distance from n₀ to the centre of a fine pixel:

```latex
x^{(d+1)}[n_0] = \frac{\sum_{\gamma \le \rho} e^{-\gamma^2 / 2\sigma^2}\, x^{(d)}[n]}{\sum_{\gamma \le \rho} e^{-\gamma^2 / 2\sigma^2}}
```

- **Radius:** ρ = 3 fine pixel sizes, a pixel size being √(4π / N\_pix) at the fine level, with N\_pix that of the full sky. **Width:** σ = ρ / 2.
- The sums run over every fine pixel within ρ of n₀, not only over the 4 children: 26 to 32 fine pixels on the full sky, the 4 children carrying 0.28 to 0.30 of the weight. The weights sum to 1 for each coarse pixel.
- **Partial sky.** Only the available fine pixels enter the sums, and the weights are renormalised over them; the coarse pixels are the parents of the available fine pixels. A coarse pixel can be computed from a few fine pixels: nothing marks it. **⚠ OP-50**
- This is the smoothing of HEALPix: a Gaussian of the distance, not the 6-tap kernel of M-SMOOTH-2D. **⚠ OP-49**
- **Pixel centres:** as in M-CONV-HPX, on `"WGS84"` by default; γ is then the distance on the unit sphere between the points whose latitudes are the WGS84 ones. **⚠ OP-45**
- Several levels are reached by repeating the step. Complex arrays are downsampled part by part (real and imaginary).
- healpix-analyse has two fallbacks that a full sky and the partial skies checked here never reach: if every fine pixel within ρ has a null weight, uniform weights over the available ones; if none is available, the value 0, with a warning.
- STL's `_smooth_kernel` (a K × K Gaussian) serves a separate smoothing function, `apply_smooth`, which the scattering loop does not call; neither does it call the plain average of the 4 children (`smooth=False`) nor the max-pooling of healpix-analyse. They are not transferred.
- **With a mask:** as M-DOWN-MASK, with this step in place of M-DOWN. The invalid pixels are set to 0, each step is multiplied by the reweighting map 1/(1 − f), with f the invalid 0/1 mask of the level decimated by this step, and a coarse pixel is invalid if f > τ, with τ = 0.33; the invalid pixels of the result are NaN.
- **Checked** (sphere): the matrix of `HealPixDown` equals the formula written in NumPy to 6e-16 (`nside` 16 to 8 and 8 to 4); STL's `downsample` applies it to each map (4e-15), two levels are the step repeated (7e-16), and a complex array is downsampled part by part (9e-16). On a disc of 154 fine pixels: 44 coarse pixels, the sorted parents, computed from 10 to 32 fine pixels each, equal to the formula restricted to the available pixels (2e-16); 29 of the 44 differ from the full-sky values. With the default `"WGS84"`: the matrix equals the same formula with the WGS84 latitudes (3e-16) and differs from that of the sphere by up to 1.1e-2 per weight. With a mask (`nside` 16, a disc of 94 NaN, two levels): equal to M-DOWN-MASK to 7e-16, NaN at the same pixels; the largest f kept is 0.326.
- **Source:** healpix-analyse `HealPixDown._build_smooth_matrix`; `WaveletOperatorHealpixKernel_torch._down_one_level`, `_downsample_masked`.

### 4.4 Masks and borders — M-MASK-HPX

- **Masks of the layers:** as M-MASK, with the decimation of M-DOWN-HPX and with "the wavelet support touches an invalid pixel" read as: a convolution of the invalid 0/1 map with a kernel of ones over the stencil (M-CONV-HPX) is above 10⁻¹² for at least one of the L gauges. The support of a pixel is therefore the set of pixels that the interpolation reads, over the K² stencil points and the L gauges.
- **Reweighting maps:** 1/(1 − f), with 1 − f not below 10⁻⁸, and 0 on the invalid pixels; one family for the field and one for the first-layer data convolved at scale j, as in M-DOWN-MASK.
- **Covariances:** as M-MASK: the mask of the deeper layer; at equal depth with different histories, the union.
- **Borders:** none, β = 0, on the full sky and on a partial sky alike. The edge of a partial sky is not cropped: the pixels whose stencil or whose decimation is truncated (M-CONV-HPX, M-DOWN-HPX) enter every mean. **⚠ OP-50**
- **A second way to handle NaN** exists in STL, `nan_aware_stats`, which uses `nanmean` without precomputed masks. It is not transferred: ESTEL has one way, masks as weights.
- **Checked** (`nside` 16, J = 3, a disc of 94 NaN, sphere): the masks of STL equal those of M-MASK rebuilt in NumPy from the stencil and the decimation above, for layer 0 (94, 26 and 9 invalid pixels at levels 0, 1, 2), layer 1 (244, 112, 70), the downsampled layer 1 and layer 2 (from 95 to 460); the test "above 10⁻¹²" selects the same pixels as "a neighbour of positive weight is invalid"; the reweighting maps equal 1/(1 − f) to 1e-15; the masked mean is the sum over the valid pixels divided by their number (5e-18); the border is 0 with a mask and on a partial sky, where the mean runs over all the pixels.
- **Source:** `_build_masks`, `_support_mask`, `_reweight_from_fraction`, `_find_mask`, `cov`, `_get_crop_border_size_zero`.

### 4.5 Defaults — M-DEF-HPX

| Parameter | Default |
| --- | --- |
| J | max(1, ⌊log₂ `nside`⌋ − 1) |
| L (number of gauges) | 4 |
| K (stencil size) | 5 |
| Wavelet | `"HealpixWavelet"`; envelope of width 1 pixel, period 4 pixels |
| Gauge | `"cosmo"` |
| Ellipsoid | `"WGS84"` in STL (OP-45) |
| Cache of the interpolation weights | enabled in STL, in float32 (OP-46) |
| τ (NaN threshold) | 0.33 |
| Border | none |
| Decimation | radius 3 pixels, σ = radius / 2, weights of sum 1 |
| `compute_PS` | True in STL: the angular spectrum; unavailable for HEALPix in ESTEL 1.0.0 (LF24) |

- **Checked:** the wavelet operator that STL builds by default has L = 4, K = 5, the gauge `"cosmo"`, the ellipsoid `"WGS84"`, τ = 0.33, the cache enabled, no option passed to the decimation and a border of 0.

### 4.6 Open points found in the HEALPix code

The rule of §1.11 applies. OP-90 to OP-92 come from the check of T5.0. OP-45, OP-46 and OP-50 hold the facts for the points to settle Q7, Q8 and Q10 (Part IV, §23); they are not decided here.

| # | Point | Proposal |
| --- | --- | --- |
| OP-45 | The pixel centres are computed on the WGS84 ellipsoid by default: their latitudes are geodetic latitudes (at most 0.128° from those of the sphere), used as latitudes of a sphere by a convolution and a decimation that are otherwise spherical. A convolution changes by a relative rms of 2 % (`nside` 8) to 4 % (`nside` 16) for a white noise. The code ESTEL ports (`TorchGeometry`) supports the sphere only | to decide (Q7): ESTEL on the sphere only, and the parity harness runs STL with `ellipsoid="sphere"` |
| OP-46 | healpix-analyse casts the kernel through float32 (5e-8 of its largest modulus; its sum becomes 9e-9) and caches its interpolation weights in float32: with the cache, enabled by default, the first operator of a geometry has float64 weights and the later ones float32 weights (S3 and S4 change by up to 2.5e-7). The decimation is in float64 | to decide (Q8), as OP-31: what the harness sets (cache disabled), and whether ESTEL reproduces the float32 kernel |
| OP-47 | The gauge `"cosmo"` is regular at the poles and discontinuous across the equator, where the orientations of the two sides differ by 2φ − 2πg/L; the southern frame is the mirror image of the northern one, so the moduli are symmetric under a mirror through the equator, except on the equatorial ring. The reading of the draft (singular at the poles) was that of the gauge `"phi"` | reproduce; document |
| OP-48 | The HEALPix wavelet is not the planar one (Gaussian times plane wave, divided by the sum of moduli), so coefficients are not comparable between geometries | reproduce; LF22 |
| OP-49 | The decimation is a Gaussian of radius 3 pixels and σ = radius/2 (G12), over 26 to 32 fine pixels of which the 4 children carry 0.28 to 0.30 of the weight; it is not the planar smoothing kernel | reproduce; Part III, §20.10 |
| OP-50 | No border is cropped on a partial sky. At the edge, the convolution renormalises its interpolation weights over the available pixels and replaces a stencil point without any available pixel by the pixel itself; the decimation renormalises over the available fine pixels. These pixels enter every mean (on a disc of 154 pixels at `nside` 16: 121 pixels with a truncated stencil, a relative rms difference of 0.48 from the full-sky values) | reproduce; to decide (Q10) |
| OP-90 | `cell_ids` must be in ascending order: otherwise the convolution of healpix-analyse (and of `TorchGeometry`) returns its values permuted once more by the inverse sort order, and the decimation reads its input as if it were sorted; no error is raised (measured: differences of the order of the values) | ESTEL requires ascending `cell_ids`, or sorts them itself; to decide with the API of T5.1 |
| OP-91 | A partial sky of 12 n² pixels given without `nside` is taken for a full sky of `nside` = n | ESTEL requires `nside` whenever `cell_ids` is given; to confirm |
| OP-92 | The gather tables of `TorchGeometry` equal those of healpix-analyse as linear operators (to 1e-13, and the outputs to 2e-14), not entry by entry: about 1 % of the index entries differ (10924 of 1228800 at `nside` 16) | T5.1 compares the tables as operators; to confirm |


## 5. Syntheses

**Status:** 17 November 2026 (task T6.0). For periodic syntheses, the only ones of 1.0.0 (Part IV, §22.5; LF47), every entry is checked against STL's code (`Synthesis.py` and what it calls), and numerically, in the parity container, where an entry says "Checked": the Nyquist pre-filter, the start, the target, the weights and the value of the loss, the result, the operators and the settings passed to L-BFGS. The comparison of whole syntheses comes with T6.1. The non-periodic and masked parts are not checked; they are marked "after 1.0.0 (Part III, §16, LF47)".

**Source:** STL-Dev `master` at `8eaf769`, `Synthesis.py`. A synthesis looks for a field u whose statistics match those of a target. Step 1 does not compare synthesised fields with STL's (Part II, §14.3), except in the one comparison of T6.1; this section fixes the definitions.

**Levels.** `Synthesis.py` has two levels. The user-level functions (`synthesize_from_maps`, `synthesize_from_stats`) build the operators (M-SYN-OPS) and call the mid-level ones (`optimize_from_maps`, `optimize_from_stats`), which build the target, the running field and the loss, and call L-BFGS. Two defaults differ between the levels: the seed (26 at the user level, none at the mid level) and the spectrum (in the loss at the user level, not at the mid level of `optimize_from_maps`). This section gives the user level.

### 5.1 Target — M-SYN-TARGET

From maps:

1. The target is promoted to `(Nb, Nc, …)` as in M-DATA-2D: a 3D array is one map of Nc channels, `(1, Nc, N1, N2)`, never a batch.
2. If it has no NaN and `prefilter_Nyquist=True` (the default), it is filtered by M-SYN-NYQ, before the standardisation.
3. It is standardised (M-STD) with `mean_field` (True by default). With `mean_field=True`, one μ and one σ per channel serve the whole batch (formula below): μ is the batch mean of the means, and σ² the batch mean of the second moments of the fields centred by that μ, so σ is the square root of a mean of variances, not a mean of standard deviations. With `mean_field=False`, μ and σ are those of M-STD, per map, and the number of running fields must be Nb.
4. Its statistics are computed on the standardised field, without a second standardisation (`standardize=False`), with the spectrum, with `norm="self"` and `ref_batch_mean=mean_field` (STL's `store_ref` and `norm_batch_mean`). The batch mean of its `mean` is then 0 and that of its `var` is 1, up to rounding.
5. They are reweighted (M-SYN-LOSS) and flattened, with `mean_along_batch=mean_field`.

```latex
\mu_c = \frac{1}{N_b} \sum_b \langle x_{b,c} \rangle, \qquad \sigma_c^2 = \frac{1}{N_b} \sum_b \big\langle (x_{b,c} - \mu_c)^2 \big\rangle \qquad (\texttt{mean\_field=True})
```

From statistics: the target is a statistics object computed beforehand with `norm="self"`. Its normalisation, its `ref_batch_mean` and its standardisation are those the caller chose: with `standardize=True`, each map was standardised on its own (M-STD), whatever `mean_field`. The synthesis only reweights and flattens it (step 5). **⚠ OP-72**

- STL reweights the target's statistics in place. From statistics, the object of the caller is therefore modified: S3 and S4 stay multiplied by their weights, `mean_pre_std` and `std_pre_std` are replaced by their batch means if `mean_field=True`, and `compute_PS` is set to False if the spectrum is not used. A second synthesis from the same object has another target. **⚠ OP-70**
- **Checked** (32 × 32, J = 3, L = 4; 1, 2 and 3 targets; 1 and 2 channels; `mean_field` True and False): μ and σ as above (3 targets of different amplitudes: STL 1.857454, the formula 1.857454, the mean of the standard deviations 1.728709); |mean| ≤ 2.4e-17 and var = 1 exactly for one target; after one synthesis from statistics, the caller's S3 and S4 are multiplied by 3.5 and 12.25, and a second call starts from a loss of 1.6e2 instead of 0.96.
- **Source:** `optimize_from_maps`, `optimize_from_stats`, `standardize`.

### 5.2 Running field — M-SYN-INIT

- **Shape:** `(nbatch, Nc, …)`, with the target's spatial shape unless `running_shape` is given.
- **Start:** a Gaussian white noise of unit variance, or `init_running` if given; then M-SYN-NYQ, if `prefilter_Nyquist=True`.
- **Not standardised.** The running field is in the standardised units of the target, and it is never standardised itself: the loss compares its `mean` with 0 and its `var` with 1 (M-SYN-LOSS). After the pre-filter, the start has a variance below 1, close to the fraction of modes that the filter keeps (M-SYN-NYQ). **⚠ OP-73**
- **Missing data:** the missing pixels of the running field are set to 10¹⁰, and their gradient is set to 0 at every iteration, so they never move; they are NaN in the result. **⚠ OP-53** After 1.0.0 (Part III, §16, LF47).
- **Randomness:** STL seeds the global generator of PyTorch (`seed`, 26 by default in the user-level functions, no seeding by default in the mid-level ones), then draws the whole batch in one call, `torch.randn((nbatch, Nc, N1, N2))`, in the dtype and on the device of the running operator; nothing else is drawn between the seeding and this call. ESTEL uses explicit keys (Part I, §3.1, rule 5). **⚠ OP-55**
- **Taper.** The former point G11 mentioned a Tukey taper of the Gaussian start on non-periodic axes. STL's syntheses at the frozen commits apply none; the only cosine taper is the border mask of the spectrum estimator. **[to verify]** whether it comes from another code base (STL-Dev-MPI) and should be reproduced. After 1.0.0 (Part III, §16, LF47).
- **Checked** (same cases as M-SYN-TARGET): the start of a default synthesis is the pre-filter of `torch.randn((nbatch, Nc, N1, N2))` drawn after `torch.manual_seed(26)`, to 8.9e-16; its variance is 0.77 to 0.78 at 32 × 32, where the filter keeps 795 modes of 1024.
- **Source:** `ScatteringMatchModel.__init__`, `optimize_from_maps`, `optimize_from_stats`.

### 5.3 Nyquist pre-filter — M-SYN-NYQ

In the 2D Fourier plane, with integer frequencies (k₁, k₂), where kᵢ runs over the Nᵢ values of the discrete Fourier transform (−Nᵢ/2 included when Nᵢ is even), the modes with k₁² + k₂² > (min(N₁, N₂)/2)² are set to 0:

```latex
\hat y(k_1, k_2) = \hat x(k_1, k_2)\; \mathbb 1\Big[k_1^2 + k_2^2 \le \big(\min(N_1, N_2)/2\big)^2\Big]
```

- The modes on the circle are kept: (±N/2, 0) and (0, ±N/2) for a square field of even side. The mode 0 is kept, so the mean is unchanged.
- The set of kept modes is symmetric under k → −k, so the filtered field is real; STL takes the real part of the inverse transform. The filter is a projection: applying it twice changes nothing.
- It acts on the last two axes, on every batch element and channel alike.
- It removes the corners of the frequency square, which no wavelet covers (OP-32). It keeps a fraction of the modes that tends to π/4 for a large square field: 195 of 256 at 16 × 16, 795 of 1024 at 32 × 32.
- For a rectangular field, the radius is that of the shorter side, in mode numbers on both axes: on the longer axis the cut is at a lower spatial frequency. **⚠ OP-54**
- **Checked:** against the definition above written in NumPy, on 16 × 16, 32 × 32, 16 × 24, 24 × 16, 15 × 15, 15 × 20 and a `(2, 3, 16, 16)` batch: largest difference 8.9e-16. Modes on the circle kept and corners removed (16 × 16); idempotent to 5e-16; mean unchanged to 3e-17; on 16 × 24, the largest kept mode number on the long axis is 8.
- **Source:** `apply_nyquist_filter`.

### 5.4 Loss — M-SYN-LOSS

With Φ(u) the flattened statistics of the running field, computed with `norm="ref"` (STL's `load_ref`) and the target's reference, without standardisation, and with w the weights:

```latex
\mathcal L(u) = \sum_i \big\lvert w_i\, \Phi_i(u) - w_i\, \Phi_i(x_{\text{target}}) \big\rvert^2
```

- **Families.** Φ holds `mean`, `var`, S1, S2, S3, S4 and, in the periodic case, PS (M-SYN-OPS), in the order of M-FLAT, without the structural NaN. `mean` and `var` are in the loss: they are what ties the mean and the variance of the running field to those of the standardised target.
- **Weights.** w = 1, except 3.5 for S3 and 3.5² for S4 (`adhoc_weights`, the default). They multiply the statistics of both fields before the flattening, so a difference in S3 counts with 3.5² in the loss, and one in S4 with 3.5⁴. **⚠ OP-51**
- **Vector.** The complex form of M-FLAT; |·| is the complex modulus. Since |z|² = (Re z)² + (Im z)², the loss is the squared Euclidean norm of a complete real vector: the real and the imaginary parts of every coefficient, those of S1, S2 and PS included. `to_vector` of ESTEL stays STL's `to_flatten(flatten_complex=True)`, as in STL (Part IV, §22.5, Q14 closed): it leaves out the imaginary parts of S1, S2 and PS, which are null for the auto terms and not for the cross-channel terms of S1 and S2. With one channel the two vectors give the same loss; with cross-channel terms they do not, and the loss of 1.0.0 is the complete one. **⚠ OP-39**
- **Gradient.** STL writes |z|² as `abs() ** 2`; PyTorch's gradient of it is 0 at z = 0, as that of (Re z)² + (Im z)².
- **Batch.** With `mean_field=True`, every family is averaged over the batch, for the target and for the running fields, before the comparison: the number of running fields need not be that of the targets. With `mean_field=False`, the sum also runs over the batch, and the two numbers must be equal.
- **From statistics.** When the target is given as statistics rather than as fields, the same loss, divided by the target's Nb (not by `nbatch`) if `mean_field=True`. The factor does not move the minimum, but it scales the gradient, hence the first step of L-BFGS and what its tolerances mean. **⚠ OP-71**
- **Checked** (same cases as M-SYN-TARGET): the value STL gives to L-BFGS at the start equals the formula above, computed from statistics obtained without `Synthesis.py`, to 5e-16 relative in the four cases from maps (for one target: 5.3935191911, on 768 complex coefficients: 1, 1, 12, 12, 96, 640 and 6 for `mean`, `var`, S1, S2, S3, S4 and PS) and to 3e-16 in the two from statistics, with the division by Nb. Without the weights the loss would be 2.51 instead of 5.39. With two channels: the complex form (3038 coefficients) and the complete real vector (6076) give the same loss to 1e-16, and `to_flatten(flatten_complex=True)` (5982) a loss 4.9 % smaller; with one channel the two agree to 2e-16. The gradient at the start is finite.
- **Source:** `ScatteringMatchModel.forward`, `reweight`, `optimize_from_maps`, `optimize_from_stats`.

### 5.5 Optimiser — M-SYN-LBFGS

L-BFGS of PyTorch (`torch.optim.LBFGS`), in one call of `step`, on one parameter: the whole running batch `(nbatch, Nc, N1, N2)`.

| Setting | Value |
| --- | --- |
| step size (`lr`) | 1.0 |
| `max_iter` | 100 |
| `history_size` | 50 |
| line search | strong Wolfe |
| `tolerance_grad` | 10⁻¹² |
| `tolerance_change` | 10⁻¹⁵ |
| function evaluations (`max_eval`) | not passed: PyTorch's default, ⌊5 × `max_iter` / 4⌋, 125 for 100 |

- `lr`, `max_iter` and `history_size` are arguments of the synthesis functions, with these defaults; the line search and the two tolerances are fixed in `optimize_lbfgs`.
- **Stopping.** PyTorch stops at the first of: `max_iter` iterations, `max_eval` evaluations of the loss, or one of the two tolerances. The cap on evaluations can come first: with `max_iter=7`, a run stopped after 4 iterations and 8 evaluations. The number STL prints as "iterations of synthesis" is the number of evaluations of the loss. **⚠ OP-74**
- These are the values that the `lbfgs` primitive of ESTEL leaves open (issue #28).
- **Checked:** the settings of the optimiser that a default `synthesize_from_maps` builds (PyTorch 2.15.0) are those of the table, with `max_eval` = 125, and 8, 12 and 37 for `max_iter` = 7, 10 and 30; one parameter of shape `(2, 1, 32, 32)` for two running fields, one call of `step`; a default run made 100 iterations and 104 evaluations, and STL printed 104.
- **Source:** `optimize_lbfgs`.

### 5.6 Result — M-SYN-OUT

The optimised field is de-standardised, x = u σ + μ (M-STD), its missing pixels are set to NaN (after 1.0.0: Part III, §16, LF47), and axes are removed.

- **From maps:** μ and σ are those of the target (M-SYN-TARGET): of the filtered target if it was pre-filtered, one pair per channel if `mean_field=True`, one per map otherwise.
- **From statistics:** only if the statistics were computed with `standardize=True`; μ and σ are then `mean_pre_std` and `std_pre_std`, averaged over the batch if `mean_field=True`. Here σ is a mean of standard deviations, unlike from maps. **⚠ OP-72**
- **Axes.** The batch axis is removed if `nbatch` = 1. The channel axis is removed, from maps, if the target was given as a 2D array, and kept otherwise, even with one channel; from statistics, if Nc = 1. **⚠ OP-75**
- **Checked:** with the optimisation replaced by the identity, the result equals u σ + μ with μ and σ recomputed from the definitions above, to 2e-15 (from maps: five cases; from statistics: `mean_field` True and False). Shapes from maps: a `(N, N)` target gives `(N, N)` for `nbatch` = 1 and `(2, N, N)` for 2; `(1, N, N)` gives `(1, N, N)`; `(3, 1, N, N)` with `nbatch` = 2 gives `(2, 1, N, N)`. From statistics with Nc = 1 and `nbatch` = 2: `(2, N, N)`.
- **Source:** `optimize_from_maps`, `optimize_from_stats`, `unstandardize`.

### 5.7 Operators — M-SYN-OPS

The user-level functions build two operators, one for the target and one for the running field, so that each has its own mask (after 1.0.0: Part III, §16, LF47). In the periodic case without missing data the two are distinct objects with the same settings.

- **J.** For each, the default J (M-DEF) minus 1 if the field is not periodic; the synthesis uses the smaller of the two. **⚠ OP-52** After 1.0.0 (Part III, §16, LF47). In the periodic case, J is the default of M-DEF; it is passed explicitly to both operators, so the spectrum uses it as J\_PS (M-PS-2D). From statistics, J is that of the target's statistics.
- **Other settings.** From maps, the user-level function passes only J, the number of bins and `has_fewer_convolutions`: L, the wavelet type and the kernel size are the defaults of M-DEF. From statistics, the running operator takes J, the number of bins and `has_fewer_convolutions` from the target's statistics, but not L nor the wavelet type: statistics computed with another L than the default make the synthesis fail on a shape mismatch. **⚠ OP-76**
- **Spectrum.** Used in the loss only if neither field has NaN: always in the periodic case of 1.0.0, with the `fft` estimator (M-PS-2D). The number of bins is the smaller of the two fields' defaults. From statistics, also only if the target's statistics hold a spectrum.
- **Reference.** The target operator's reference (`var_ref`, `S2_ref_sqrt_chan_diag` and, with the spectrum, `PS_ref_sqrt_chan_diag`) is copied into the running operator; from statistics, it is read from the target's statistics.
- **Checked** (targets of 32 × 32, 64 × 32 and 48 × 48, periodic): both operators have J = 3 (the default), L = 4, bump-steerable wavelets, 6 bins, J\_PS = 3, the `fft` estimator and the spectrum in the loss; they are two objects, and the running one holds the target's three reference arrays; the seed is 26, the pre-filter on, the weights 3.5 and 12.25. From statistics computed with J = 2: the running operator has J = 2, the same 4 bins and the target's reference. From statistics computed with L = 8: STL raises a size error (4 against 8).
- **Source:** `synthesize_from_maps`, `synthesize_from_stats`.

### 5.8 What ESTEL adds

Checkpoints (Part II, §9.2 and §12.9): the state of the optimiser, the running field and the key are saved every `checkpoint_every` iterations, and a run resumed from a checkpoint continues the same trajectory on the same device. No STL counterpart.

### 5.9 Open points found in the synthesis code

The rule of §1.11 applies. OP-39 also concerns syntheses; for the loss it is closed by Q14 (Part IV, §22.5; M-SYN-LOSS). OP-70 to OP-76 come from the check of T6.0.

| # | Point | Proposal |
| --- | --- | --- |
| OP-51 | Weights 3.5 for S3 and 3.5² for S4 in the loss, without a stated reason | reproduce as the default; document |
| OP-52 | The syntheses use the default J minus 1 for non-periodic fields | reproduce. After 1.0.0 (Part III, §16, LF47) |
| OP-53 | Missing pixels of the running field are set to 10¹⁰ and frozen | ESTEL uses weights: the pixels are never read; to verify that the trajectory is unchanged. After 1.0.0 (Part III, §16, LF47) |
| OP-54 | The Nyquist pre-filter uses the radius of the shorter side for rectangular fields | reproduce |
| OP-55 | Global seed, and a default seed of 26 in the user-level functions | ESTEL takes an explicit key; no default seed |
| OP-56 | In `synthesize_from_maps`, the default `prefilter_Nyquist` calls `init_running.isnan.any()`, which fails when an initial field is given | ESTEL computes the intended value. After 1.0.0 (Part III, §16, LF47) |
| OP-70 | A synthesis from statistics modifies the caller's statistics in place: S3 and S4 stay multiplied by their weights, `mean_pre_std` and `std_pre_std` are replaced by their batch means, `compute_PS` can be set to False; a second synthesis from the same object has another target (measured: a starting loss of 1.6e2 instead of 0.96) | ESTEL is functional and leaves its inputs unchanged: the first synthesis is reproduced, not the effect on later ones; to confirm |
| OP-71 | From statistics with `mean_field=True`, the loss is divided by the target's Nb; from maps it is not. The minimum is the same, the trajectory of L-BFGS is not | reproduce; to examine |
| OP-72 | The standardisation differs between the two entries. From maps with `mean_field=True`: one μ and one σ for the batch, σ the square root of the mean variance, on the filtered target. From statistics: each map standardised on its own when the statistics were computed, never filtered, and the result de-standardised with the batch means of μ and of σ | reproduce; document; to examine |
| OP-73 | The running field is never standardised, and the pre-filter leaves the start with a variance near π/4 (0.78 at 32 × 32) against a target `var` of 1 | reproduce |
| OP-74 | PyTorch's cap on evaluations, ⌊5 × `max_iter` / 4⌋, can stop a run before `max_iter` iterations; STL prints the number of evaluations as "iterations" | reproduce the cap; ESTEL names the two counts apart |
| OP-75 | From maps the channel axis of the result is removed only if the target was a 2D array; from statistics, whenever Nc = 1 | to decide with the API of T6.1 |
| OP-76 | `synthesize_from_stats` builds the running operator with the default L and wavelet type, whatever those of the target's statistics: it fails for statistics computed with another L | ESTEL takes every setting of the operator from the statistics; to confirm |
