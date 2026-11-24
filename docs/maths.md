# maths.md — Mathematics of step 1

[Back to the overview](transfer/overview.md)

**Status:** draft, 16 November 2026. Sections 1 (planar 2D, reviewed against `master`) and 2 (1D) are written from the STL code by reading it, and checked numerically by the oracle `estel._spec`, which agrees with ESTEL to the parity tolerance (task T2.6); ESTEL itself agrees with STL's fixtures. Section 3 (tensor products) is checked against STL's code (task T3.0), and numerically by the oracle, which agrees with STL's parity fixtures to the parity tolerance (tasks T3.3 and T3.4). Section 4 (HEALPix) is checked against STL's code, and numerically where it says so (task T5.0), its scattering loop against that of `master` and against section 1 (task T5.2). Section 1.9 (the power spectrum) is checked against STL's code, entry by entry, and numerically in the parity container (task T4.0). Section 5 (syntheses) is checked against STL's code for periodic syntheses, and numerically where it says so (task T6.0), and for full-sky HEALPix syntheses without the prefilter and without the spectrum (task T5.2); its non-periodic and masked parts come after 1.0.0 (Part III, §16, LF47). It has no oracle: the syntheses are validated against STL's trajectory and by their own tests (task T6.1; Part IV, §22).

**Role of this file.** This file is the reference version of the mathematics of step 1; it changes only through pull requests that a maintainer requests and merges.

**Scope.** Step 1 (transfer): planar 2D, 1D, tensor products, HEALPix and syntheses. The definitions proposed for step 2 are in `docs/maths_step2.md`.

**Sources.**

- Planar 2D, scattering statistics and power spectra: STL-Dev `master` at `8eaf769` (v1.6.0: `STL_2D_Kernel_Torch.py`, `ST_Operator.py`, `ST_Statistics.py`, `torch_backend.py`, `CS_operator_2D_Kernel_Torch`).
- 1D: STL-Dev `2D+1D` at `8f51143` (`STL_1D_Kernel_Torch.py`, and the classes it inherits from in that branch). For the 2D scattering statistics the two branches compute the same values (their code was compared function by function); `master` is the reference.
- Tensor products: STL-Dev `2D+1D` at `8f51143` (`Tensor_Product_Kernel_Torch.py`, `ST_Operator_Tensor_Product.py`, `ST_Statistics.py`).
- HEALPix: STL-Dev `HealpixUpdate` at `90eb1a9` (`STL_Healpix_Kernel_Torch.py`, `TorchGeometry.py`, `ST_Operator.py`, `ST_Statistics.py`, `Synthesis.py`), with healpix-analyse at `aac5e26` (`convol.py`, `down.py`).
- Syntheses: STL-Dev `master` at `8eaf769` (`Synthesis.py`); for the HEALPix path, STL-Dev `HealpixUpdate` at `90eb1a9` (`Synthesis.py`).

**Rule.** In step 1 this tab describes what STL computes, including choices that look accidental. Points that need a decision are marked **⚠ OP-n** and listed in the last sub-section of each section: §1.11 (2D), §2.6 (1D), §3.10 (tensor products), §4.6 (HEALPix) and §5.9 (syntheses). Each entry gives the definition, its conventions, the STL function it comes from (**Source**), the oracle function that will implement it (**Oracle**, in `estel._spec`) and the tests that use it.

## Contents of this tab

- [1. Planar 2D](#1-planar-2d)
  - [1.0 Notation — M-NOT](#10-notation--m-not) · [1.1 Data — M-DATA-2D](#11-data--m-data-2d) · [1.2 Wavelets](#12-wavelets) — [M-CONV-2D](#m-conv-2d--convolution-is-a-correlation), [M-WAV-2D](#m-wav-2d--bump-steerable-wavelets) · [1.3 Downsampling](#13-downsampling) — [M-SMOOTH-2D](#m-smooth-2d--smoothing-kernel), [M-DOWN](#m-down--downsampling-without-mask), [M-DOWN-MASK](#m-down-mask--downsampling-with-missing-data) · [1.4 Masks of the scattering layers — M-MASK](#14-masks-of-the-scattering-layers--m-mask) · [1.5 Borders — M-BORDER](#15-borders--m-border) · [1.6 Moments and first layer](#16-moments-and-first-layer) — [M-MEAN](#m-mean--masked-mean), [M-MOM](#m-mom--mean-and-var), [M-STD](#m-std--standardisation), [M-W1](#m-w1--first-layer) · [1.7 Scattering statistics — vanilla ScatCov](#17-scattering-statistics--vanilla-scatcov) — [M-S1](#m-s1), [M-S2](#m-s2), [M-S3](#m-s3), [M-S4](#m-s4), [M-CHAN](#m-chan--storage-of-channel-pairs) · [1.8 Normalisation and reductions](#18-normalisation-and-reductions) — [M-ORDER](#m-order--order-of-the-operations-of-apply), [M-NORM](#m-norm), [M-RED-ISO](#m-red-iso--isotropy), [M-RED-ANG](#m-red-ang--angular-fourier-transform), [M-RED-SCALE](#m-red-scale--scale-cosine-transform), [M-FLAT](#m-flat--flattening) · [1.9 Power spectrum — M-PS-2D (from master, v1.6.0)](#19-power-spectrum--m-ps-2d-from-master-v160) · [1.10 Defaults — M-DEF](#110-defaults--m-def) · [1.11 Open points found in the 2D code](#111-open-points-found-in-the-2d-code)
- [2. 1D](#2-1d)
  - [2.0 Data — M-DATA-1D](#20-data--m-data-1d) · [2.1 Wavelets — M-WAV-1D](#21-wavelets--m-wav-1d) · [2.2 Downsampling](#22-downsampling) — [M-SMOOTH-1D](#m-smooth-1d--smoothing-kernel) · [2.3 Masks, borders, moments and first layer](#23-masks-borders-moments-and-first-layer) · [2.4 Scattering statistics, normalisation and reductions](#24-scattering-statistics-normalisation-and-reductions) · [2.5 Defaults — M-DEF-1D](#25-defaults--m-def-1d) · [2.6 Open points found in the 1D code](#26-open-points-found-in-the-1d-code)
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
- **Resolution level.** d ≥ 0; level d has Nᵢ / 2ᵈ pixels per axis. x⁽ᵈ⁾ is x downsampled d times (§1.3).
- **Scales and orientations.** J scales, j = 0 … J−1; L orientations, ℓ = 0 … L−1. Scale j is computed at level d = j (§1.2).
- **Valid-pixel mean** of an array a with invalid mask m (True = invalid), computed after the border crop of §1.5:

```latex
\langle a \rangle_m = \frac{\sum_{n:\ \neg m[n]} a[n]}{\#\{n : \neg m[n]\}}
```

- **Complex conjugate** z̄; modulus |z|.

**Filters and scales.**

- s: the smoothing kernel, the filter of one downsampling step (M-SMOOTH-2D, M-SMOOTH-1D).
- S\_d: the cumulated smoothing of d downsampling steps, Ŝ\_d(ω) = ŝ(ω) ŝ(2ω) ⋯ ŝ(2^(d−1) ω), up to the sampling of each step.
- ψ\_ℓ: a wavelet kernel, applied at a given level. ψ\_(j,ℓ): the effective wavelet seen from full resolution, ψ̂\_(j,ℓ)(ω) = ψ̂\_ℓ(2^d ω) Ŝ\_d(ω), with d = j the level of scale j.
- φ\_J: the low-pass filter, the effective filter of scale J. In STL and in 1.0.0, φ̂\_J = Ŝ\_d with d = J. A dedicated low-pass kernel φ may replace this realisation (LF6); there is none in 1.0.0.
- Λ: the Littlewood–Paley sum, Λ(ω) = |φ̂\_J(ω)|² + ½ Σ\_(j,ℓ) (|ψ̂\_(j,ℓ)(ω)|² + |ψ̂\_(j,ℓ)(−ω)|²). Both signs of ω are counted because the wavelets are analytic. Λ = 1 everywhere would mean that the coefficients keep all the energy of the data.
- Per-level energy partition: P(ω) = |ŝ(ω)|² + ½ Σ\_ℓ (|ψ̂\_ℓ(ω)|² + |ψ̂\_ℓ(−ω)|²). If P = 1 at every level, then Λ = 1 follows by cascade, with φ̂\_J = Ŝ\_d and d = J. A dedicated φ only acts on the last octave: it cannot make up for a partition that differs from 1 at the intermediate levels.

**Moments and covariances.**

- M1(A) = ⟨A⟩\_w: the first moment, the mean of A over a window w (the valid pixels after the border crop).
- M2(A, B) = ⟨A conj(B)⟩\_w: the raw second moment.
- Cov(A, B) = M2(A, B) − M1(A) conj(M1(B)): the covariance, with M1 and M2 on the same window.
- The scattering covariances are covariances by definition: S2 = Cov(W\_{j,ℓ} x, W\_{j,ℓ} x), S3 = Cov(ψ\_{ℓ₃} ⋆ |W\_{j₂,ℓ₂} x|, W\_{j₃,ℓ₃} x) and S4 = Cov(ψ\_{ℓ₃} ⋆ |W\_{j₁,ℓ₁} x|, ψ\_{ℓ₃} ⋆ |W\_{j₂,ℓ₂} x|), with ψ at scale j3. S1 = M1(|W\_{j,ℓ} x|) is a first moment. STL, and ESTEL 1.0.0, compute the raw moments M2 instead (OP-36); the two are equal when M1 of one argument is zero.
- Zero-mean property: a band-pass kernel sums to zero, so a field filtered by it has M1 = 0 exactly when w is the whole periodic domain, without mask. This does not hold with a cropped border or a mask. It does not hold either when the last filter is the low-pass filter φ\_J: M1(φ\_J ⋆ |W\_{j,ℓ} x|) is S1(j), and M1(φ\_J ⋆ x) is the mean of the data.

**Notation to harmonise (W5, issue #71).** The review of 16 November 2026 found the 17 problems below in the notation of this file, each with a recommendation. Only R-N10 is decided and applied (T2.18); the others are the card W5. The full text of the remarks, and a single table of symbols proposed in their place (R-N18), are in issue #71. This table goes once the notation is decided and applied.

| Remark | What collides | Recommendation |
| --- | --- | --- |
| R-N01 | φ is an angle and the low-pass filter | keep φ for the low-pass filter; write the angle ϑ |
| R-N02 | c has four meanings: the channel index, the width factor (2.5), the constant c\_L and the bin centre c\_b | keep c for the channel; κ for the width factor, a\_L for the constant; rename the bin centre with the spectra |
| R-N03 | S is the statistics and the cumulated smoothing (S\_d, Ŝ\_d) | write the cumulated smoothing s⁽ᵈ⁾, as x⁽ᵈ⁾ |
| R-N04 | w is a kernel and a window; W and W\_b exist too | keep w for the window; write a generic kernel g |
| R-N05 | two notations for one mean, ⟨a⟩\_m and ⟨A⟩\_w; m has three meanings | one notation, ⟨·⟩\_w; keep m for the mask; write a kernel offset t |
| R-N06 | σ has three meanings: the normalisation, the standard deviation of the data and the width of a bin | keep σ for the standard deviation; write √(S2ʳᵉᶠ) in M-NORM |
| R-N07 | b is the batch index and the spectral bin | keep b for the batch; rename the bin index with the spectra |
| R-N08 | N is the size of the data and the design grid (256) | write the design grid M |
| R-N09 | ω is an integer mode and an angular frequency | ω is always an angular frequency; write the integer mode ν |
| R-N10 | j is a scale and a resolution level | j is a scale, d a level, everywhere; applied (T2.18) |
| R-N11 | three notations for the conjugate | z̄ in displayed formulas, `conj(·)` in text; never \* |
| R-N12 | "layer" is the layers of the architecture and the number of wavelet convolutions | keep "layer" for the architecture; write "scattering order 0, 1, 2"; not "depth" |
| R-N13 | W, U and D are used before they are defined; S3 and S4 are defined twice | define them once, in the notation section; one definition per statistic |
| R-N14 | the indices of the reductions (ℓ₁, ℓ₂, δ; k for a harmonic) | rewrite them with the numbering decided (Part IV, §22, R-A01); write a harmonic h |
| R-N15 | ψ̂\_ℓ is the design profile and the transform of the kernel | write the design profile Ψ\_ℓ; ψ\_ℓ is the kernel, ψ̂\_ℓ its transform |
| R-N16 | "kernel", "filter" and "wavelet" are sometimes used for one another | say it once: a kernel is applied at one level; a filter is the effective object seen from full resolution; a wavelet is a band-pass kernel or filter |
| R-N17 | the conventions of the conjugate and of the direction of the scales differ between the papers | state both in the notation section, with the correspondence to Cheng et al. and Mousset et al. |

### 1.1 Data — M-DATA-2D

- **Shape.** Arrays are `(Nb, Nc, N1, N2)`; a 2D array is promoted to `(1, 1, N1, N2)`, a 3D one to `(1, Nc, N1, N2)`. Data must be real.
- **Size.** N₁ and N₂ must be divisible by 2^(J−1). The constraint comes from the downsampling of the scattering loop, which goes down to level d = J−1 and raises an error at any step with an odd size; nothing pads or resizes the input beforehand. The spectrum operator itself accepts any size (but see OP-20). The default J can violate the constraint for sizes that are not powers of two: for N = 200, J = 5 requires divisibility by 16. **⚠ OP-8**
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

Only the finest scale (j = 0) is built. Coarser scales are obtained by applying the same kernel to downsampled data (§1.3), so the effective filter at scale j is the cascade "smooth and downsample d = j times, then ψ\_ℓ".

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

One step, from level d to d+1, where x̃⁽ᵈ⁾ is x⁽ᵈ⁾ padded by **2 pixels before and 3 after** on each axis (circular or replicate, as in M-CONV-2D), keeping one pixel in two on each axis (stride 2), with s the smoothing kernel (M-SMOOTH-2D):

```latex
x^{(d+1)}[n] = \sum_{m_1, m_2 = 0}^{5} s[m]\ \tilde x^{(d)}[2n + m]
```

Several levels are reached by repeating the step.

- Because the smoothing kernel has an even size, the padding is asymmetric, and a comment in the code questions it. Each step may shift the grid by half a pixel. **⚠ OP-2**
- Complex arrays are downsampled part by part (real and imaginary).
- **Source:** `_downsample_tensor`, `downsample`.

#### M-DOWN-MASK — Downsampling with missing data

With a mask, each step is "normalised": invalid input pixels are set to 0, the result is computed as in M-DOWN, then multiplied by a reweighting map:

```latex
x^{(d+1)} = \big(\text{M-DOWN of } x^{(d)} \cdot \mathbb 1_{\text{valid}}\big) \cdot \frac{1}{1 - f^{(d+1)}}
```

where f⁽ᵈ⁺¹⁾ is the fraction of invalid input contributing to each output pixel: the invalid mask of x⁽ᵈ⁾, after its own thresholding, as 0/1, downsampled with the same smoothing kernel; it is not the fraction f⁽ᵈ⁾ downsampled again.

- **Threshold.** An output pixel is declared invalid if f > τ, with τ = 0.33 (`downsample_nan_weight_threshold`). **⚠ OP-12**
- **Invalid output pixels** are filled with `replace_nan_value`, NaN by default; the code notes that this breaks back-propagation. ESTEL uses weights instead of NaN values; statistics only read valid pixels, so the values are unchanged. **⚠ OP-27**
- **Two families of maps.** The fractions are precomputed for the field itself ("smooth" maps, from the full-resolution mask) and for first-layer data convolved at scale j ("wavelet" maps, from the layer-1 mask of §1.4), for every target level, and for both padding modes.
- **Source:** `_build_reweighting_maps_and_scattering_layer_masks`, `downsample`.

### 1.4 Masks of the scattering layers — M-MASK

Let E be the K × K support of the wavelets (all ones), shared by all orientations.

- **Layer 0:** the full-resolution mask m⁽⁰⁾; at level d > 0, the invalid pixels after thresholded downsampling (M-DOWN-MASK).
- **Layer 1** (data convolved once, at scale j, level d = j): an output pixel is invalid if the wavelet support E touches an invalid input pixel, that is m₁⁽ᵈ⁾ = (E ⋆ m⁽ᵈ⁾) > 0.
- **Layer 2** (first convolved at scale j₂, at level d = j₂; modulus; downsampled to level d = j₃; convolved again at scale j₃): m₂ = (E ⋆ m) > 0, where m is the layer-1 mask at level d = j₃ if j₂ = j₃, or the layer-1 mask of scale j₂, downsampled and thresholded to level d = j₃, otherwise.
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

At each scale j, on the field downsampled to level d = j:

```latex
W_{j,\ell}\, x_c = \psi_\ell \star x_c^{(d)}, \qquad U_{j,\ell}\, x_c = \lvert W_{j,\ell}\, x_c \rvert
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

For j₂ ≤ j₃ (**including** j₂ = j₃), with D\_{j₂→j₃} the downsampling from the level of scale j₂ to the level of scale j₃ (M-DOWN or M-DOWN-MASK):

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
- **Spectrum.** PS is stored as S1 and S2: c₁ ≤ c₂ only, (c₂, c₁) NaN, complex numbers with a null imaginary part; inside the scattering operator it has the pairs of the statistics, and STL's spectrum operator alone defaults to the auto spectra only (§1.9). **⚠ OP-94**
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
| PS\[c₁, c₂, b\] | σᴾˢ\_{c₁}(b) · σᴾˢ\_{c₂}(b), with σᴾˢ\_c(b) = √PS\_ref\[c, c, b\] (checked, §1.9) |

**Other values of `norm`.** `norm="none"` (STL's `vanilla`) computes the statistics without any normalisation. `norm` must be given: `None` in both the constructor and `apply` is an error. With `norm="self"`, passing a reference to `apply` is an error.

**`SpectrumOperator`** (Part II, §12.8) normalises its spectrum in the same way, with the same `norm` and `ref_batch_mean`: PS divided by σᴾˢ\_{c₁}(b) σᴾˢ\_{c₂}(b), the reference being its own spectrum with `norm="self"` (decision of 24 November 2026). STL's spectrum operator normalises nothing, as `norm="none"`.

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

- **Order:** `mean`, `var`, S1, S2, S3, S4, then PS if it was computed (checked, §1.9); within each family, the order of the stored tensor (row-major).
- **Missing entries:** the NaN entries are removed (`keepnans=False`, the default): the structural NaN, and the means with no valid pixel.
- **Batch:** by default the batch axis is flattened with the rest. With `keep_batch_dim=True`, the result is `(Nb, n)`, and the NaN must be at the same positions for every batch element. With `mean_along_batch=True`, every family is first averaged over the batch.
- **Complex values:** two forms. With `flatten_complex=False`, the default, a complex vector, with `mean` and `var` cast to complex. With `flatten_complex=True`, a real vector: each complex coefficient becomes two reals (real part, imaginary part), except in S1, S2 and PS, of which only the real part is kept. That part is null only for the auto terms: the imaginary part of the cross-channel terms is dropped too. **⚠ OP-37**
- **`mask_st`:** an optional boolean array over the flattened vector; True keeps a coefficient. Its length must be the vector's.
- **Which form is used where.** `to_vector` of ESTEL is the real form. STL's syntheses use the complex form, with `mean_along_batch` set by their `mean_field` (M-SYN-LOSS). The two forms do not carry the same information for cross-channel S1 and S2. **⚠ OP-39**

- **Source:** `to_flatten`.

### 1.9 Power spectrum — M-PS-2D (from master, v1.6.0)

**Status:** checked against STL's code, entry by entry, and numerically in the parity container where an entry says so (task T4.0, 23 November 2026). The options ESTEL keeps were decided by the maintainer on #148 (Part IV, §22); those not kept are described below, marked "not transferred", and listed in Part II, §10. ESTEL's oracle comes with T4.2, the spectrum as a family of `STStatistics` with T4.4, the operator with T4.1.

**Numerical checks.** They ran in the parity container (environment `master`, PyTorch 2.15.0, float64). The reference evaluates the formulas of this section in NumPy, apart from STL's code: it uses the full complex FFT rather than STL's half-plane layout, SciPy's Bessel function J₀ and `quad`, and builds every window, border and mask itself. It is compared with `CS_operator_2D_Kernel_Torch.apply` on five sizes, 16 × 16, 32 × 32, 32 × 16, 16 × 32 and 64 × 32 (default J), with two maps of one channel and two maps of two correlated channels (every pair c₁ ≤ c₂), fields of non-zero mean; for the estimators with missing data, the maps miss a block of 12 pixels and one isolated pixel. Every estimator runs with the log-Gaussian bins, and with the legacy bins at 16 × 16 and 32 × 16; the non-periodic ones with each mask and each crop mode. In every case the spectrum differs from the reference by at most 1.2e-15 of the largest value of its channel pair. "Checked" below names what a check adds to this comparison.

#### Frequencies and bins

- **Fourier modes and radial frequencies.** A Fourier mode is the vector of integers 𝐤 = (k₁, k₂), in mode units, with k₁ ∈ {−⌊N₁/2⌋, …, ⌈N₁/2⌉ − 1} and k₂ ∈ {0, …, ⌊N₂/2⌋} (the half plane of the real FFT along the last axis). The radial frequency is the scalar k, in the same units: the mode 𝐤 belongs to the radial frequency k = round(|𝐤|) = round(√(k₁² + k₂²)) (never a tie: a sum of two squares of integers is never the square of a half-integer), and the radial frequencies are the integers k = 0 … N\_max, with N\_max = ⌊min(N₁, N₂)/2⌋; modes with round(|𝐤|) > N\_max are dropped. A mode enters every formula below through round(|𝐤|), not through its exact modulus. Every frequency of this section (k, k\_min, k\_max, c\_b, and the edges, whose logarithms are e\_b) is in mode units. **Cycles per pixel.** A frequency k is k / (2 N\_max) cycles per pixel, that is k / N for an even square image of side N. STL computes in cycles per pixel, from 0 to 0.5: its `k_min`, `k_max` and `bin_centers` are k\_min / (2 N\_max), k\_max / (2 N\_max) and c\_b / (2 N\_max); the windows, n\_b and the borders do not depend on the unit, since only ln k − ln c\_b and k\_max / k\_min enter them. For a rectangular image this mixes mode numbers of two different lengths, so the binning is not isotropic: a mode (k₁, 0) of the longer axis counts as the radial frequency k₁, that is k₁ / min(N₁, N₂) cycles per pixel, although its frequency is k₁ / max(N₁, N₂). **⚠ OP-6**
- **Range.** J\_PS = ⌊log₂ N\_max⌋ − 1 by default (equal to the ST default J = ⌊log₂ min(N₁, N₂)⌋ − 2, M-DEF), and J\_min = 0. When the scattering operator is given an explicit J, its spectrum uses that J as J\_PS; the default applies only when J is not given. STL requires 0 ≤ J\_min ≤ J\_PS − 2 (a `ValueError` at construction), so J\_PS ≥ 2. **⚠ OP-93**

```latex
k_{\min} = \frac{N_{\max}}{\sqrt2\, 2^{J_{PS}-1}}, \qquad k_{\max} = \frac{N_{\max}}{\sqrt2\, 2^{J_{\min}}}
```

- **Number of bins:** n\_b = max(4, round(3 log₂(k\_max / k\_min))) = max(4, round(3 (J\_PS − 1 − J\_min))) by default, about 3 per octave.
- **Checked:** for a 64 × 64 field, the scattering operator with J not given, 3, 4 and 5 builds its spectrum with J\_PS = 4, 3, 4, 5 and n\_b = 9, 6, 9, 12; with J\_min = 1, n\_b = 6, k\_min = 2.83 and k\_max = 11.31 (STL's 0.0442 and 0.1768 cycles per pixel); with n\_b given (7), it is used. A scattering operator with J = 1 fails (OP-93).
- **Log-Gaussian windows** (`gaussian_rings`, the default). The edges e\_0 … e\_{n\_b} are evenly spaced in ln k from ln k\_min to ln k\_max; the centre of bin b is ln c\_b = (e\_b + e\_{b+1}) / 2, and the width of every bin is Δ = e\_{b+1} − e\_b. For the radial frequencies k ≥ 1:

```latex
W_b(k) = \exp\Big(-\frac{(\ln k - \ln c_b)^2}{2\Delta^2}\Big), \qquad W_b(0) = 0
```

  The windows have no compact support: every bin has a positive weight at every non-zero frequency.
- **Legacy windows** (`binning_type="legacy"`): the Littlewood-Paley windows of STL v1.x, of compact support, computed by numerical quadrature (`scipy.integrate.quad`) once per bin and radial frequency; a bin can then hold no Fourier mode (source `_build_legacy_bin_masks`). **⚠ OP-95** **Not transferred** (decision of the maintainer on #148): ESTEL builds the log-Gaussian windows only.
- **Checked:** the log-Gaussian windows and centres equal the reference to 1.2e-15, the legacy ones to 3.3e-16.
- **Bin normalisation:** Z\_b = Σ\_k N\_k W\_b(k), where N\_k is the number of Fourier modes 𝐤 with round(|𝐤|) = k: in STL's half-plane layout, every column counted twice except the zero column and, when N₂ is even, the Nyquist column, which is the number of such modes in the full plane. **Checked:** N\_k equals the count of the full plane exactly.

#### Estimators

All use the orthonormal 2D FFT, x̂ = FFT2(x) / √(N₁N₂), on the half plane of the last axis (`rfft2`). **Only the real part** of cross-spectra is computed. **⚠ OP-19**

**`fft`** (periodic, no missing data): the radial spectrum is, with w\_𝐤 the half-plane weight of the mode (2, or 1 for the zero and Nyquist columns),

```latex
P(k) = \frac{1}{N_k} \sum_{\mathbf{k}:\ \mathrm{round}(|\mathbf{k}|) = k} w_{\mathbf{k}}\, \mathrm{Re}\big(\hat x_{c_1}(\mathbf{k})\, \overline{\hat x_{c_2}(\mathbf{k})}\big)
```

then binned:

```latex
PS_b = \frac{\sum_k N_k W_b(k)\, P(k)}{Z_b}
```

- **Checked:** to 6.9e-16 with the log-Gaussian windows and 4.1e-16 with the legacy ones. STL also computes it on odd sizes, and agrees with the reference there (31 × 32, 32 × 31, 33 × 33: 3.3e-16); ESTEL does not (OP-20, below).

**Band maps**, used by the next three estimators:

```latex
f_b = \mathrm{FFT2}^{-1}_{\text{ortho}}\Big(\hat x(\mathbf{k})\, \sqrt{W_b(\mathrm{round}(|\mathbf{k}|))}\Big), \qquad \sqrt{W_b} = 0\ \text{for}\ \mathrm{round}(|\mathbf{k}|) > N_{\max}
```

computed with the inverse real FFT (`irfft2`) of width 2(K\_x − 1), where K\_x = ⌊N₂/2⌋ + 1 is the length of the half-plane axis. That width is N₂ only for even N₂: for an odd last size the band maps have N₂ − 1 columns and the three estimators fail with a `RuntimeError` (a shape mismatch); an odd N₁ runs. **ESTEL requires even sizes for every estimator, `fft` included**, and raises a clear error at construction on an odd size, where STL computes `fft` on any size (decision of the maintainer on #148). **⚠ OP-20** **Checked:** the band maps computed on the full plane are real to 1.2e-15; (32, 31) and (33, 33) fail as written, (31, 32) runs.

**`pbc_mask`** (non-periodic):

```latex
PS_b = \frac{N_{\text{pix}}}{\sum_n M_b[n]}\ \frac{1}{Z_b} \sum_n M_b[n]\, f_{b,c_1}[n]\, f_{b,c_2}[n]
```

- **The border mask** M\_b, of border β\_b pixels, at n = (n₁, n₂):
  - `"hard"` (the default): M\_b[n] = 1 if β\_b ≤ n₁ < N₁ − β\_b and β\_b ≤ n₂ < N₂ − β\_b, else 0;
  - `"smooth"`: M\_b[n] = ∏ᵢ ½ (1 − cos(π min(min(nᵢ, Nᵢ − 1 − nᵢ) / β\_b, 1))), a raised cosine, null on the outermost pixels; M\_b = 1 if β\_b = 0. **Not transferred** (decision of the maintainer on #148); the non-periodic estimator is redesigned later (LF7).
- **The border** β\_b is the radius that holds **60 %** of the energy of g\_b, the kernel of bin b in pixel space: the 2D inverse Fourier transform of the radial window W\_b, a Hankel transform, evaluated at the integer radii ρ = 0 … N\_max (in pixels) by a Riemann sum over the radial frequencies (in cycles per pixel, the frequency k / (2 N\_max) and the step 1 / (2 N\_max)):

```latex
g_b(\rho) = \frac{\pi}{2 N_{\max}^2} \sum_{k=0}^{N_{\max}} W_b(k)\, k\, J_0\Big(\frac{\pi \rho k}{N_{\max}}\Big), \qquad
Q_b(\rho) = \int_0^\rho 2\pi \rho'\, g_b(\rho')^2\, d\rho'
```

  Q\_b is the cumulative trapezoidal rule on the integer radii, Q\_b(0) = 0; β\_b is the smallest ρ with Q\_b(ρ) ≥ 0.60 Q\_b(N\_max), and 0 for a bin whose window is null on the grid (Q\_b = 0). The fraction is that of the energy within ρ ≤ N\_max, not of the total, and it is computed from W\_b, whereas each band map carries √W\_b. **⚠ OP-96** (reproduced in 1.0.0; its fix belongs with LF7.) The fraction 0.60 (`energy_fraction`) is not reachable from the constructor. **⚠ OP-21**
- **The Bessel function J₀.** STL computes it with `torch.special.bessel_j0`. NumPy has no Bessel function J₀, and SciPy is not a dependency of ESTEL: ESTEL implements the Bessel function J₀ in NumPy, within the plan, without adding SciPy (decision of the maintainer on #148). Its argument π ρ k / N\_max runs from 0 to π N\_max, about 804 at 512 pixels, so a power series alone does not do. Required accuracy: an absolute error of at most 1e-12 on that range, computed in float64 in the plan whatever the dtype of the data. SciPy's `scipy.special.j0` gives the same borders as STL. STL's `torch.special.bessel_j0` is itself accurate to only 3.8e-7 in float64 on [2.76, 14.37] (PyTorch 2.15.0); the borders do not depend on it, their margin being 3.2e-5 (below).
- **Crop modes:** `flexible_crop` (the default) uses β\_b for bin b, `largest_crop` uses max\_b β\_b for every bin. `largest_crop` is **not transferred** (decision of the maintainer on #148), for the same reason as the smooth mask (LF7): ESTEL uses β\_b for bin b.
- **Checked:** the borders equal the reference in 229 configurations: sizes 16 to 512, every J\_PS from 2 to log₂ N + 1 and every J\_min, both kinds of windows. In none of them does Q\_b(ρ) / Q\_b(N\_max) come closer to 0.60 than 3.2e-5, at any radius: an error of 1e-12 on the Bessel function J₀ leaves every border unchanged. The borders are not monotone in b: [4, 3, 3, 3] at 16 × 16, [5, 4, 3, 3, 2, 3] at 32 × 32 (default J); [13, 13, 10, 17, 16, 7, 10, 8, 6, 5, 4, 3, 3, 2, 3] at 64 × 64 with J\_PS = 6. Their largest value is about 0.3 N\_max at the default J (10, 19, 38 and 75 pixels at 64, 128, 256 and 512), and up to 0.75 N\_max at 16 × 16 (legacy windows: [5, 5, 5, 6]). The spectrum equals the reference to 8.6e-16 for each mask and each crop mode.

**`nan_mask`** (periodic, missing data):

- each map (each batch element and channel) has its missing pixels filled with the mean of its pixels valid in that map, then is band-filtered; STL accepts missing pixels that differ per map, ESTEL one mask shared by every map (M-DATA-2D) **⚠ OP-98**;
- a pixel is observed only if it is valid in every batch element and channel; the estimate sums over these pixels only:

```latex
PS_b = \frac{N_{\text{pix}}}{N_{\text{obs}}}\ \frac{1}{Z_b} \sum_{n\ \text{observed}} f_{b,c_1}[n]\, f_{b,c_2}[n]
```

- no observed pixel raises a `ValueError`.

The filling leaks into neighbouring pixels through the filtering. **⚠ OP-7** **Checked:** to 9.5e-16; on data without missing pixels it equals `fft` to 6.4e-16.

**`pbc_nan_mask`** (non-periodic, missing data): as `pbc_mask`, with M\_b multiplied by the observed mask O (as in `nan_mask`) in the sum and in its prefactor, N\_pix / Σ\_n M\_b[n] O[n]; a bin whose product is null everywhere raises a `ValueError`. **Checked:** to 1.1e-15, for each mask and each crop mode.

**Data with missing pixels** given to `fft` or `pbc_mask` raise a `ValueError` at `apply`. **Checked:** a scattering operator built on data without NaN fails so on data with NaN. In ESTEL the missing pixels are static, known when the operator is built: the error comes at construction. **⚠ OP-98**

#### Choice of estimator

Automatic, when the operator is built, from the example data. `pbc=None` counts as periodic (compare OP-4). The scattering operator's `spectrum_method` overrides it.

| Data | Estimator |
| --- | --- |
| NaN present, `pbc=False` | `pbc_nan_mask` |
| NaN present, otherwise | `nan_mask` |
| no NaN, `pbc=False` | `pbc_mask` |
| no NaN, otherwise | `fft` |

- **Checked:** the four rows, with `pbc` True, None and False; `spectrum_method="pbc_mask"` on periodic data gives `pbc_mask`.

#### Channel pairs and layout

- **Shape** (Nb, Nc, Nc, n\_b), stored as complex numbers (complex128 in float64) whose imaginary part is exactly 0, as S1 and S2 (M-CHAN).
- **Pairs.** Only c₁ ≤ c₂ is computed, for the pairs that `cross_matrix` holds; the other entries, (c₂, c₁) included, are NaN. The spectrum is real and symmetric in the two channels: PS\[c₂, c₁\] would equal PS\[c₁, c₂\].
- **Default pairs.** Inside the scattering operator, the spectrum has the pairs of the scattering statistics (M-CHAN; default: every pair c₁ ≤ c₂). STL's spectrum operator alone defaults to the auto spectra only. In ESTEL, `cross_matrix=None` means every pair c₁ ≤ c₂ on `SpectrumOperator` as on `STOperator`, and inside `STOperator` one matrix selects the pairs of the scattering statistics and of the spectrum alike (decision of the maintainer on #148). **⚠ OP-94**
- **Checked:** NaN exactly at c₁ > c₂, auto spectra only by default; the scattering operator's spectrum equals, bit for bit, that of the spectrum operator with the upper triangle, for the four estimators (32 × 32 periodic and not, with and without missing pixels; 32 × 16 non-periodic).

#### In the scattering operator

- **Construction.** STL's scattering operator always builds its spectrum operator, even with `compute_PS=False`, passing it J when J is given, and `n_bins`, `spectrum_method` and `Jmin` when they are given. **⚠ OP-93** ESTEL's `STOperator` builds its spectrum operator only when `compute_PS=True`: J = 1 with `compute_PS=False` is accepted, and with `compute_PS=True` a J < 2 + J\_min raises a clear error at construction, which says to pass `compute_PS=False` or to use a larger image. When the spectrum is not computed (`compute_PS=False`, or data that are not 2D), a value of `n_bins`, `Jmin` or `spectrum_method` other than its default raises a clear error, which says that these options act only with `compute_PS=True` (decisions of the maintainer on #148).
- **Field.** The spectrum is computed on the field at full resolution, after the standardisation when `standardize` is set (M-STD). **Checked:** with `standardize=True` it equals the spectrum of the standardised field to 2.5e-16, and differs from that of the raw field by 93 % of its largest value.
- **Normalisation** (M-NORM): σᴾˢ\_c(b) = √PS\_ref\[c, c, b\], per batch element, or of the batch mean of PS\_ref\[c, c, b\] with `ref_batch_mean=True`. A channel without auto spectrum gives NaN normalised cross spectra (OP-30). **Checked:** `norm="self"`, `norm="self"` with the batch mean and `norm="ref"` on another field give the formula bit for bit; the stored reference has shape (Nb, Nc, n\_b), and (1, Nc, n\_b) with the batch mean.
- **Reductions.** `iso`, `angular_fourier` and `scale_cosine` leave the spectrum as it is (read in `to_iso`, `to_angular_ft`, `to_scale_ft`).
- **Flattening** (M-FLAT): the spectrum comes last, after S4. **Checked:** for the five cases above, the complex vector ends with the finite entries of PS in row-major order, and the real vector with their real parts (one real number each).

#### Options

Decided by the maintainer on #148 (Part IV, §22). Every option is set at construction; STL's per-call overrides of `apply` (`spectrum_method`, `get_crop_border_size_method`, `pbc_mask_type`) are not transferred.

| Option | ESTEL | Default | STL |
| --- | --- | --- | --- |
| number of bins | `n_bins` | `None`: max(4, round(3 (J\_PS − 1 − J\_min))) | `n_bins` |
| J\_PS | `J` of `SpectrumOperator`; inside `STOperator`, its own J | `None`: ⌊log₂ N\_max⌋ − 1 | `J` |
| J\_min | `Jmin` | 0, with 0 ≤ J\_min ≤ J\_PS − 2 | `Jmin` |
| estimator | `method` of `SpectrumOperator`, `spectrum_method` of `STOperator` | `"auto"`: the choice above | `spectrum_method` |
| channel pairs | `cross_matrix` | `None`: every pair c₁ ≤ c₂ (OP-94) | `compute_cross_spectrum_matrix`, default the auto spectra |
| spectrum in the scattering statistics | `compute_PS` of `STOperator`, at construction only | True for 2D data | `compute_PS`, also in `apply` |
| reference of `norm="ref"` | `apply(data, ref=…)` | — | `PS_ref_sqrt_chan_diag` |
| bin centres c\_b | `STStatistics.bin_centers`, the coordinate of the axis `bin` of `"PS"`, in cycles per pixel as STL: c\_b / (2 N\_max) | — | `bin_centers`, in cycles per pixel: c\_b / (2 N\_max) |
| fraction of the border energy | fixed at 0.60 (OP-21) | — | `energy_fraction` of `estimate_pbc_border_crop` |

Not transferred: the legacy windows (`binning_type="legacy"`), `largest_crop`, the smooth mask (`pbc_mask_type="smooth"`), the per-call overrides; `build_pbc_mask` is internal to the plan; `plot_cross_spectrum` goes with STL's plotting utilities (Part II, §9.3); `device` and `dtype` are those of ESTEL's backend.

- **Source (master, `8eaf769`):** `CS_operator_2D_Kernel_Torch` (`__init__`, `_build_fft_index`, `_build_bin_windows`, `_build_log_gaussian_bin_masks`, `_build_legacy_bin_masks`, `estimate_pbc_border_crop`, `build_pbc_mask`, `_build_hard_pbc_mask`, `_build_smooth_pbc_mask`, `apply`, `_cross_pk`, `_bin_pk`, `_apply_fft`, `_apply_pbc_mask`, `_apply_nan_mask`, `_apply_pbc_nan_mask`); `STL_2D_Kernel_Torch.get_CS_op`; `ST_Operator.__init__`, `ST_Operator.apply`; `ST_Statistics.to_norm`, `_get_sqrt_chan_diag`, `to_flatten`.

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
| PS estimator | automatic, from `pbc` and the missing pixels of the example data (§1.9) |
| PS range | J\_PS = the J of the operator, J\_min = 0 |
| PS binning | log-Gaussian, n\_b = max(4, round(3 (J\_PS − 1 − J\_min))), about 3 bins per octave |

### 1.11 Open points found in the 2D code

**Rule for these open points.** In 1.0.0, ESTEL reproduces STL wherever STL gives a result, problems included; most open points are fixed afterwards, each as a traced change. Where STL gives no usable value (OP-8), ESTEL raises a clear error instead; a mean with no valid pixel (OP-3) gives a NaN coefficient, as in STL.

**Status of a point.** Each table gives it: `reproduced`, ESTEL reproduces STL in 1.0.0, the point included; `differs`, ESTEL does otherwise, and the proposal or the entry says how; `open`, still to decide, by the task or the entry of the register that the proposal names; `later (LFn)`, deferred to the register (Part III, §16); `closed`, no longer a point. A point stays in its table whatever its status.

Grouped in three categories: (A) points that affect values or correctness, (B) conventions to document, (C) transfer and API points.

#### A. Values and correctness

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-1 | Filtering is a correlation, not a convolution; complex coefficients are the conjugates of the textbook definition | reproduce, document | reproduced |
| OP-2 | Even-sized smoothing kernel with asymmetric padding (2/3): possible half-pixel shift per level, questioned by a `TODO` | reproduce for parity; revisit in a major | reproduced |
| OP-3 | Masked mean divides by zero when no pixel is valid | revised: the coefficient is NaN, as in STL, decided statically by the plan; ESTEL warns once, listing such coefficients, and raises only if a whole family is empty | reproduced |
| OP-4 | `pbc=None` breaks the scattering transform but means "periodic" for spectra | require an explicit boolean in ESTEL | differs |
| OP-5 | `iso` and `angular_fourier` average over orientations that cover \[0, π): the real parts of the reduced S3 and S4 are invariant under rotations by multiples of π/L, their imaginary parts are not | reproduce; decide whether `iso` returns the real parts only (LF32) | reproduced |
| OP-6 | Radial binning of the spectrum is not isotropic for rectangular images | reproduce and document in 1.0.0, without restricting the tests to square images; fixed in step 2 (LF49) | reproduced |
| OP-7 | `nan_mask` fills missing pixels with the mean before filtering, leaking near mask edges | reproduce, document; better estimator in step 2 | reproduced |
| OP-8 | The default J does not ensure that the sizes are divisible by 2^(J−1), so the default operator can fail on sizes that are not powers of two | decided: ESTEL rejects the combination at construction with a clear message; STL fails here, so parity is not concerned | differs |
| OP-31 | STL builds its wavelet kernels at torch's default dtype, float32 unless changed, although the rest of its computation is in float64; its kernels therefore have float32 precision | ESTEL builds them in float64; the parity harness runs STL with float64 as torch's default dtype (Part II, §14.2) | differs |
| OP-35 | STL computes S3 on the layer-2 mask alone, assuming it contains the layer-1 mask; for some masked 1D data the assumption fails and STL's S3 is NaN | ESTEL uses the union of the two masks, as the formula defines; both agree wherever STL gives a value | differs |
| OP-58 | The 2D kernels are not rotations of each other (5e-3 at 90 degrees, from the unpaired frequency −N/2 of the design grid), and their energies differ by 3.2 % between axes and diagonals | reproduce; a requirement of the joint design (Part III, §20.10) | reproduced |
| OP-59 | The 2D smoothing table sums to 1.00000004 and is separable only to 2e-8; a mean is multiplied by that sum at each downsampling | reproduce; a separable kernel is a change of values, for step 2 | reproduced |
| OP-60 | With missing data and J ≥ 2, the gradient of STL's statistics with respect to the field is NaN at the valid pixels, although every statistic is finite: at every valid pixel of the benchmark case `planar-masked` (512 × 512, J = 7, non-periodic, 10.1 % missing: 235 584 pixels of 235 584), and 0 at the missing ones. The cause is the default `replace_nan_value=NaN` of `ST_Operator` (OP-27): after each downsampling, `downsample` writes NaN in the coarse pixels declared invalid (M-DOWN-MASK). These pixels go through the wavelet convolution, the modulus (`modulus`, `data.array.abs()`) and the products of `cov`, and `maskmean` then removes them from the mean with `torch.where(~mask, x, 0.0)`. In the backward pass that `where` sends them a zero gradient, which the modulus and the products multiply by a NaN local derivative (the sign of a NaN, or the NaN factor): 0 × NaN = NaN, first in the backward pass of `abs`, and the transposed convolutions and downsamplings spread it to the valid pixels. The values are not concerned. With `replace_nan_value=None`, which STL's syntheses pass, or 0, STL's statistics are the same bit for bit and its gradient is finite, equal to ESTEL's to 4e-16 of its largest value on that case (T2.16) | ESTEL keeps a finite gradient: it uses weights, never NaN values (OP-27, M-MEAN); where STL's gradient is NaN, ESTEL's is compared with STL run with `replace_nan_value=None` | differs |
| OP-61 | A reduced border (OP-13) biases the coarsest non-periodic means. On a 64 × 64 field with J = 5 the level of scale 4 has 4 × 4 pixels: the border of M-BORDER (2 pixels for S1 and S2, more for S3 and S4) is too large and is reduced to 1 pixel, so the mean runs over 2 × 2 pixels that the padding still reaches. Measured on a stationary log-normal process, the same periodic fields read with `pbc=False` and with `pbc=True` (512 seeded draws, raw statistics, L = 4): at j = 4 the non-periodic S2 is 18.8 to 24.7 % below the periodic one according to the orientation (7.5 to 11.5 standard errors of the paired difference), and S1 10.2 to 13.7 % below; 446 of the 2 064 real numbers of S3 and S4 whose mean is at scale 4 are beyond 6.1 standard errors as well, the threshold of the test, and no coefficient of the finer scales is (largest 2.8). STL gives the same values on the same fields (`master`, parity container: equal to ESTEL to 5e-16 of the largest coefficient), so the gap is a property of STL's estimator. At 128 × 128 and 256 × 256 with J = 5, the means with a reduced border keep less than a quarter of the pixels of their level, and no gap is seen on the others (64 draws; at 128 × 128 the non-periodic means of scale 4, with a full border of 2 pixels on an 8 × 8 level, stay within 2.1 standard errors with 256 draws, on six seeds). `tests/consistency/` keeps the case as its positive control (T2.16) | reproduce in step 1. For step 2: a mean whose border cannot be cropped in full is not computed (a NaN coefficient with one warning, as a mean with no valid pixel, OP-3), or J is limited so that every level holds its border. A less conservative border crop, tuned so that no mean is too small or empty, is to test too (LF57) | reproduced |
| OP-62 | Two formulas take a scale for the level at which it is computed, which is exact in 1.0.0, where d = j: the border of M-BORDER, β = ⌈ 2^(j−d) · ⌊K/2⌋ ⌉ after one convolution, β₁ = ⌈ 2^(j₂−j₃) · ⌊K/2⌋ ⌉ and β = ⌈ 2^(j₃−d) · (β₁ + ⌊K/2⌋) ⌉ after two, whose exponents are differences of levels written with scales; and the size condition of M-DATA-2D, divisible by 2^(J−1), the level of the last scale. When a scale is no longer computed at its own level (voices, LF41), both read the level of the scale from the plan, where T2.18 put it (issue #115) | keep in 1.0.0, where d = j; write them with the level of the scale when it differs | later (LF41) |

#### B. Conventions to document

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-9 | Width factor 2.5 tuned for L = 4 only: the valid range of L to set | reproduce in 1.0.0; a robust definition of the width factor for every L, tested, in step 2 (LF50) | reproduced |
| OP-10 | Kernels cropped to 5×5 and centred, not renormalised; effective filters are cascades | reproduce, document; LF51 | reproduced |
| OP-11 | `var` and every STL "covariance" are non-centred second moments (OP-36) | reproduce, document; changed in step 2 (LF33) | reproduced |
| OP-12 | NaN threshold τ = 0.33 in 2D, 0.5 in 1D; reweighting 1/(1−f); a pixel whose missing fraction equals τ exactly is decided by the last bit of the smoothing kernel: ESTEL reproduces STL's correctly rounded smoothing kernel (`math.fsum`), and in float32 such ties may be decided differently from STL | reproduce, document; LF52 | reproduced |
| OP-13 | Borders too large for the array are silently reduced | reproduce, document; its effect is measured in OP-61 | reproduced |
| OP-14 | S1/S2 store c₁ ≤ c₂ only, S3/S4 both orders; conjugate duplicates (LF20) | reproduce; LF20 | reproduced |
| OP-15 | Scale indices of S3 named (j₁, j₂) in STL's comments, (j₂, j₃) in the code | document; ESTEL numbers the wavelets in the order they are applied (Part IV, §22, R-A01) | reproduced |
| OP-16 | Definition of cross-channel S1, with ε = 10⁻⁸ | reproduce; LF28 | reproduced |
| OP-17 | `angular_fourier` (STL's `to_angular_ft`) keeps all L harmonics by default, contrary to a code comment | reproduce, document | reproduced |
| OP-18 | Gradients through the normalisation reference: decided, they follow from the mode (LF21, closed): with `norm="self"` the reference is part of the computation and gradients flow through it; with `norm="ref"` it is a constant | decided: nothing else to design | closed |
| OP-19 | Only the real part of cross-spectra is kept | reproduce, document; LF53 | reproduced |
| OP-20 | The spectrum code requires even sizes (inverse FFT width): the three band-map estimators fail when the last size is odd; STL computes `fft` on any size (§1.9) | decided on #148: in 1.0.0 the spectrum requires even sizes for every estimator, `fft` included, with a clear error at construction; LF61, OP-97 | differs |
| OP-21 | Spectrum border mask at 60 % of the point-spread energy, hard by default | reproduce; LF7 | reproduced |
| OP-22 | Orientation convention: θ\_ℓ = πℓ/L + π/2 from axis 1 | reproduce, document | reproduced |
| OP-30 | A channel with cross statistics but no auto statistics gets NaN after `norm="self"` normalisation (STL's `store_ref`); STL only warns | reproduce, document | reproduced |
| OP-32 | The per-level energy partition P of STL's kernels is only close to 1. Measured on the 5-tap kernels: in 1D, within 0.1 of 1 up to ω = 3π/4, falling to 0.23 at the Nyquist frequency; in 2D (L = 4, J = 5, with φ\_J), Λ lies between 0.22 and 1.76 in the disc of radius π, is anisotropic (more energy on the diagonals than on the axes), and is close to 0 in the corners of the frequency square | reproduce, document; LF22 | reproduced |
| OP-33 | Gain convention: the analytic wavelets have a maximum gain of 1, so on real data they measure half the energy of their band; Λ counts both signs to compensate. Normalised statistics do not depend on it | reproduce, document | reproduced |
| OP-34 | Tensor products (M-LP): the low-pass filter of a block is silently dropped when the block size is not divisible by 2^(J\_a), because STL obtains it by one more downsampling than the coarsest wavelet | reproduce, document; LF6 | reproduced |
| OP-36 | STL computes raw second moments, not covariances; its `cov` has a `remove_mean` argument that was never implemented. This is exact for periodic data without mask. Otherwise the forgotten term M1(A)·conj(M1(B)), relative to the covariance, measured on a test field (orders of magnitude): from 1e-6 at j = 0 to 1e-2 at the coarsest scale for non-periodic data; about ten times more with a mask covering 20 % of the pixels; of order 1 for any term whose last filter is a low-pass filter | reproduce, document; LF33 | reproduced |
| OP-37 | When flattening to a real vector, STL keeps only the real part of S1, S2 and PS, assuming they are real; this holds for the auto terms, but the imaginary part of the cross-channel terms is dropped, with the information it carries | reproduce, document; LF53 | reproduced |
| OP-38 | Default of `cross_matrix` (STL's `compute_cross_matrix`): STL's code computes every pair c₁ ≤ c₂; its docstring says auto statistics only. ESTEL follows the code | follow STL's code | reproduced |
| OP-39 | The real flattening drops the imaginary part of the cross-channel S1 and S2; STL's syntheses use the complex flattening, which keeps it, so `to_vector` and the synthesis loss do not see the same coefficients | reproduce: the synthesis loss keeps the complex form and `to_vector` stays as it is (Part IV, §22, "The vector of the synthesis loss") | reproduced |
| OP-96 | The border of the non-periodic spectrum (§1.9) is computed from the point-spread function of W\_b, whereas each band map carries √W\_b; its 60 % are of the energy within ρ ≤ N\_max, not of the total; the borders are not monotone in the bin (64 × 64, J\_PS = 6: 13, 13, 10, 17, 16, 7, …) | reproduced in 1.0.0 (decision on #148); its fix belongs with LF7 | reproduced |

#### C. Transfer and API

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-23 | The bump-steerable bank lives in STL's Fourier-space class; port it to `wavelets/banks.py` | ported to `wavelets/_banks.py` (T2.2) | closed |
| OP-24 | STL's kernel class also offers Morlet wavelets, decided: not transferred; a general optimisation of the wavelets comes later (LF22) | not transferred | closed |
| OP-25 | `sigma_smooth` has no effect: drop it from the API | dropped from the API | differs |
| OP-26 | A missing-data mask must be one 2D map, shared by all batch elements and channels; auto-detection fails on batched examples | reproduce, document; LF54 | reproduced |
| OP-27 | Missing pixels are filled with NaN during downsampling by default, which breaks gradients; ESTEL uses weights, with identical values, verified by parity cases with masks from phase P2 | ESTEL uses weights | differs |
| OP-93 | STL's scattering operator always builds its spectrum operator, even with `compute_PS=False`, and that operator requires 0 ≤ J\_min ≤ J\_PS − 2: every 2D scattering operator with J = 1 fails at construction (`ValueError`), the default one of an 8 × 8 image included. ESTEL's `STOperator` accepts J = 1 with `compute_PS=False` | decided on #148: ESTEL builds the spectrum only when it computes it, and raises a clear error at construction when `compute_PS=True` and J < 2 + J\_min, saying to pass `compute_PS=False` or to use a larger image; STL gives no value there, so parity is not concerned | differs |
| OP-94 | The default channel pairs of the spectrum: inside STL's scattering operator, those of the statistics (every pair c₁ ≤ c₂, M-CHAN); STL's spectrum operator alone, the auto spectra only | decided on #148: `cross_matrix=None` means every pair c₁ ≤ c₂ on `SpectrumOperator` as on `STOperator`; the parity cases pass the matrix explicitly | differs |
| OP-95 | The legacy windows of the spectrum (§1.9), the Littlewood-Paley windows of STL v1.x, have compact support: when J\_PS is large or J\_min > 0, a bin holds no Fourier mode, Z\_b = 0 and its coefficient is NaN (0 / 0), with a border of 0. Checked: 44 of 229 configurations (16 to 128 pixels, every J\_PS and J\_min), for example 32 × 32 with J\_PS = 5 or 64 × 64 with J\_PS = 6 (one bin each); never at the default J with J\_min = 0 | decided on #148: the legacy windows are not transferred (Part II, §10) | closed |
| OP-97 | Sizes: the spectrum requires even sizes (OP-20), the scattering loop sizes divisible by 2^(J−1) (OP-8). How to accept sizes as general as possible for every statistic | to take up after 1.0.0 (LF61) | open |
| OP-98 | The missing pixels of the spectrum: STL's spectrum operator reads them from the NaN of each map at `apply`, so that they may differ per batch element and channel, and `fft` or `pbc_mask` fail at `apply` on data with NaN. ESTEL's missing pixels are one static mask shared by every map (M-DATA-2D, OP-26): data whose missing pixels differ per map are out of reach (the configuration `nan_per_map` of `spectrum_2d`: given the union of its missing pixels, ESTEL equals the oracle to 1.2e-15, and STL's values differ by up to 29 %), and a forced `fft` or `pbc_mask` with missing pixels raises at construction | ESTEL keeps one static mask in 1.0.0; masks per map are LF54 (PR #161) | differs |

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

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-28 | The 1D wavelet is centred on the Nyquist frequency, so only the lower half of its bump exists on the grid; it has no width factor and no normalisation, unlike the 2D wavelet | reproduce; examine with the wavelet optimisation (LF22). The 1D wavelet is the upper half of a two-channel bank whose lower half is the smoothing kernel: their per-level energy partition is within 0.1 of 1 up to 3π/4 (OP-32) | reproduced |
| OP-29 | A 2D array given as 1D data is read as `(Nc, N)`, not `(Nb, N)` | reproduce, document; LF55 | reproduced |
| OP-40 | With L = 2 in 1D, `iso` averages a coefficient with its counterpart on the conjugate filter; the meaning is not established | reproduce; treat with LF32 | reproduced |

## 3. Tensor products

**Status:** checked against STL's code, entry by entry, and numerically in the parity container where an entry says so (task T3.0, 17 November 2026). Checked numerically by the oracle `estel._spec`, which agrees with STL's parity fixtures to the parity tolerance (tasks T3.3 and T3.4). ESTEL's operator, and its tests against the oracle and the fixtures, come with T3.2.

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
- **Level of a scale.** The level of scale j is d = j: j\_a downsamplings along block a for a wavelet, and J\_a for the low-pass filter of block a (M-LP).
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

The wavelet of scale j and orientation ℓ is the product of the block wavelets. It is applied block after block, in block order, on the field brought to level d = j:

```latex
W_{j,\ell}\, x_c = \psi^{(B-1)}_{\ell_{B-1}} \star_{B-1} \Big( \cdots \big( \psi^{(0)}_{\ell_0} \star_0 x_c^{(d)} \big) \cdots \Big), \qquad U_{j,\ell}\, x_c = \lvert W_{j,\ell}\, x_c \rvert
```

- **Block wavelets.** The kernels of the blocks are those of sections 1 and 2: one kernel of K = 5 taps per axis and per block, the same at every level (M-WAV-2D; M-WAV-1D with L = 2).
- **One modulus, after all the blocks.** The intermediate results are complex: the second and later block convolutions filter a complex array with a complex kernel, which STL does in one complex convolution. In ESTEL this is four real convolutions (Part I, §3.1, rule 6); complex-to-complex convolutions, tried with T3.5, gained less (Part III, §19.14).
- **Orientation axes.** One per block, placed in block order in front of the spatial axes: `(Nb, Nc, L₀, …, L_{B−1}, spatial axes)`.
- **Why L = 2 in 1D blocks.** The product of analytic wavelets covers one sign combination of the block frequencies. With L\_a = 2 in a 1D block, its conjugate filter gives the other sign, so the products cover every combination that differs for a real field.
- **How STL reaches it.** STL does not bring the field to level d = j first. It filters along block 0 a field at level d₀ = j₀ in block 0 and at level 0 in the other blocks, downsamples the complex result along block 1 to level d₁ = j₁ (M-DOWN, on the real and the imaginary parts), filters along block 1, and so on: the result of the blocks 0 … a is kept and serves every scale that starts with (j₀, …, j\_a). A downsampling along a block and a filtering along another act on different axes and commute exactly, with circular and with replicate padding: STL's values are those of the formula up to rounding.
- **Order of the blocks.** Convolutions along different axes commute exactly in theory; in floating point their order changes the rounding. STL's order is block 0 first. ESTEL's dispatch along axes is compared with STL's values at the parity tolerance in T3.2.
- **Checked:** the kernels of the blocks of a `[2D, 1D]` operator are bit for bit those of the 2D operator and of the 1D operator with L = 2, whose second kernel is the conjugate of the first. The comparison of the seven cases, in which the reference follows the formula and STL its own order, bounds the effect of the order of the operations by 1.2e-15. With the blocks exchanged (`[2D, 1D]` on a field, `[1D, 2D]` on the same field with its axes permuted; periodic and not), S1 to S4 agree to 1.0e-15 and are not bit for bit equal.
- **Source:** `Wavelet_Operator_TensorProd_Kernel_Torch.apply`, `downsample`; `ST_Operator_TensorProd._compute_ST_loop` (`l_data_multi_res`, `data_l1`).

### 3.3 Low-pass filter — M-LP

With `lowpass="ST"` (STL's default), each block may contribute a low-pass filter φ⁽ᵃ⁾\_J besides its wavelets.

- **Index.** The low-pass filter of block a is the scale j\_a = J\_a, after its wavelets j\_a = 0 … J\_a − 1. It has one orientation, stored at index 0; the other orientation entries are NaN.
- **Realisation.** The cascade of the smoothing kernels: the field at level d\_a = J\_a along block a, with no further computation. In a scale that mixes it with wavelets of earlier blocks, it is the complex result of these blocks that is downsampled to level d\_a = J\_a (M-W1-TP).
- **Existence.** Block a has a low-pass filter only if each of its sizes is divisible by 2^(J\_a), one more downsampling than its wavelets need. Otherwise STL warns (`UserWarning`) and gives the block no low-pass filter; there is then no scale with the low-pass filter in every block. **⚠ OP-34**
- **Mixed scales.** A scale with the low-pass filter in some blocks and a wavelet in at least one other is a zero-mean filter on a periodic domain, because of its wavelet factor. S1 and S2 are computed as for the wavelets, without centring.
- **The scale with the low-pass filter in every block** is the only filter without zero mean. With X the field at level d = (J₀, …, J\_{B−1}):

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

Here W\_{j₃,ℓ₃} applied to a field already at level d = j₃ means the block convolutions of M-W1-TP without further downsampling.

- **`mean` and `var`.** M-MOM over the spatial axes of all the blocks together, without border; the standardisation M-STD likewise.
- **S1 and S2.** M-S1 and M-S2 on U\_{j,ℓ}, for every scale of 𝒥; the low-pass scales follow M-LP.
- **How STL reaches the second layer.** As for the first layer (M-W1-TP), STL interleaves the downsamplings and the block convolutions: the modulus U\_{j₂} is brought along block 0 to the level of scale j₃ in that block and filtered along block 0, the complex result is downsampled along block 1 and filtered along block 1, and so on, and the result of the blocks 0 … a serves every j₃ that starts with the same indices. The values are those of the formulas up to rounding.
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

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-41 | Missing data are not supported in tensor products: the construction fails in the operator of the first block, with a `NotImplementedError` that asks for a 2D mask | ESTEL raises a clear error at construction; LF8 | differs |
| OP-42 | The only centred statistics of STL are S1 and S2 of the scale with the low-pass filter in every block; the mixed scales rely on the zero mean of a wavelet factor, exact only on a periodic domain (OP-36). Measured: the mean of a mixed scale is 2 to 7 % of the mean of its modulus when a block is not periodic, on 16 to 32 pixels | reproduce; LF33 | reproduced |
| OP-43 | No S3 or S4 involves a low-pass filter; they are assumed to nearly vanish | reproduce; LF6 | reproduced |
| OP-44 | The reductions are not defined on tensor-product statistics. In STL they raise a `TypeError`, after the computation; `to_iso` fails with S1 and S2 already averaged over one axis, and `to_angular_ft` with an integer number of harmonics returns a transform over orientation axes of different blocks, without error | ESTEL raises a clear error, before the computation; LF26 | differs |
| OP-57 | `conv_history = [[]] * nblock` makes every block share one list. Closed (T3.0): no value depends on it. STL never modifies these lists in place, it replaces them; with one list per block given explicitly, S1 to S4 are bit for bit the same (case B) | nothing to reproduce | closed |
| OP-80 | An operator built from data without NaN and applied to data that hold a NaN returns NaN for every statistic, without error; with the spectrum, the spectrum raises a `ValueError` | ESTEL raises a `ValueError`, as the spectrum does; data with missing values in tensor products after step 1 (LF8) | differs |
| OP-81 | Along a block taken at its low-pass scale the border is 0, also when the block is not periodic: the cascade of smoothing kernels, padded by replication at each step, is not counted as a convolution. The variance of the coarse-grained field and the mixed scales therefore use pixels that the padding has reached | reproduce; examine with LF6 and LF33; LF60 | reproduced |
| OP-82 | STL accepts a tensor product of one block. With `lowpass=None` its statistics are bit for bit those of the plain geometry; with the default `lowpass="ST"`, S1 and S2 have one more scale. J and L are then lists of one entry | ESTEL accepts one block, in 1.0.0 and after | reproduced |
| OP-83 | Arguments of the operator. A block shorter than 8 pixels has a default J of 0, and the construction fails with an `IndexError` in the kernel code. A list J that holds `None` is completed in place: the caller's list is modified. An integer J raises a `TypeError`, an integer L an `AssertionError` | ESTEL validates the arguments at construction, with clear errors, and leaves the caller's objects unchanged | differs |

## 4. HEALPix

**Status:** 17 November 2026 (task T5.0); the scattering loop on 24 November 2026 (task T5.2); the interpolation of M-CONV-HPX, step 3, written as a definition and checked against healpy on 24 November 2026 (task T5.4, the oracle). Every entry is checked against the code of STL `HealpixUpdate` and of healpix-analyse, and numerically, in the parity container, where an entry says "Checked": the data conventions, the wavelet, the convolution on the full sky and on a partial sky, the gauges, the downsampling, the masks and the defaults. The numerical checks run in float64, at `nside` 8 and 16 (64 for the gauges), with `ellipsoid="sphere"` and the cache of healpix-analyse disabled unless said; their references are written in NumPy from the definitions below, with healpy for the pixel centres and the bilinear interpolation. The scattering loop of `HealpixUpdate` is checked against that of `master` and against section 1 (T5.2, "Scattering loop" below). Not checked: the angular spectrum (LF24). The model of STL-Dev-MPI's `CLAUDE.md` §3 (commit `d397650`) was read, not run, and agrees with this section: the bilinear interpolation of the 4 nearest pixels, the gauge `"cosmo"`, the decimation by a Gaussian of radius 3 fine pixels with σ = radius / 2 and weights of sum 1, dg(j) = j by default, and the form of S1 to S4; its patch-local engine with hand-written reverse passes is not transferred in step 1 (Part II, §10; T5.1).

**Source:** STL-Dev `HealpixUpdate` at `90eb1a9` (`STL_Healpix_Kernel_Torch.py`), and healpix-analyse at `aac5e26` (`convol.py`, `down.py`, `healpix_interp.py`), which holds the definitions of the convolution and of the downsampling. `TorchGeometry.py` of the same branch reproduces that geometry in vectorised code, for a sphere and the gauges `"phi"` and `"cosmo"` only; it is the code ESTEL ports (Part II, §9.3). The scattering loop, the statistics, the normalisation, the reductions and the flattening are those of section 1, read from `ST_Operator.py` and `ST_Statistics.py` of the same branch (T5.2).

**Scattering loop** (task T5.2). `ST_Operator.py` of `HealpixUpdate` differs from that of `master` by 116 changed lines, `ST_Statistics.py` by 7. Block by block:

| Changed block | Does the loop reach it? | Can it change a value? |
| --- | --- | --- |
| `wavelet_op_kwargs`, a new constructor argument passed to `get_wavelet_op` | at construction | only through the options it carries, five of which change S1 to S4: `ellipsoid` and `conv_cache_dir` (OP-45, OP-46), `gauge_type`, `nan_aware_stats` (with NaN and no mask) and `down_kwargs` (the decimation); none is transferred (Part II, §10). Without it, the operator is the default one |
| `spectrum_method` renamed `power_spectrum_method`; the spectrum operator built only with `compute_PS=True`, and an error at `apply` if the spectrum is asked without one | no: spectrum only | no value of S1 to S4 |
| ten new methods (`build_plan`, `apply_pyramid`, `apply_patched`, `apply_local` and their reverse passes): the pyramid engine (Part II, §10) | no: `apply` calls none of them | no |
| two new fields of `ST_Statistics`, `data_example` and `pix_shape`, filled by `apply` | yes | no: the syntheses read them, the statistics do not |
| six hunks holding eight index expressions of the loop: `x[:, c, :, :, :]` becomes `x[:, c]`, `x[:, :, None, :, :, :]` becomes `x[:, :, None]`, and so on (three of them in the variant with fewer convolutions, not transferred: STL l. 682, 709, 710) | yes | no: on planar arrays the trailing axes are implicit and the result is the same. This change is what lets the loop run on one pixel axis |
| an assertion message, a comment translated | — | no |

What the loop calls outside these files changes in the same way: `torch_backend.maskmean` accepts any set of trailing axes (`dim=-1` on HEALPix, the same sums for `(-2, -1)`), and the base data class infers `N0` through `_infer_N0` (the trailing axes for planar data, `nside` for HEALPix).

**Checked, the two loops.** The `ST_Operator.py` of `master`, loaded in the environment of `HealpixUpdate`, and that of `HealpixUpdate`, on the same 2D kernel, give the same statistics bit for bit (difference 0), at 32 × 32, J = 3: periodic, two maps of two channels; non-periodic; a mask of 42 NaN; without normalisation, and with `norm="self"` followed by `iso`, `angular_fourier` and `scale_cosine`. On HEALPix data the loop of `master` raises `IndexError` (too many indices) at the first S1.

**Checked, the loop against section 1.** A reference written in NumPy from section 1 (M-MEAN to M-S4, M-CHAN), fed with the geometry of `HealpixUpdate`'s own wavelet operator: its healpix-analyse convolutions and decimations (`_get_conv`, `_get_down`), its masks and its reweighting maps. The masked downsampling is that of M-DOWN-MASK, and the means are taken over the masks of M-MASK, the union for every covariance. The normalisation, the reductions and the flattening of the reference are those of the oracle `estel._spec` (M-NORM to M-FLAT), applied to its statistics. Seven cases, with L = 4, the sphere, the cache disabled and `compute_PS=False`:

| Case | Sky | `nside` | Maps × channels | J | Missing data, pairs |
| --- | --- | --- | --- | --- | --- |
| A | full | 8 | 2 × 1 | 2 | — |
| B | full | 16 | 1 × 1 | 3 | — |
| C | disc of 357 pixels (`cell_ids`) | 16 | 1 × 1 | 3 | — |
| D | full | 16 | 1 × 1 | 3 | 90 NaN, the mask inferred |
| E | full | 8 | 2 × 2 | 2 | every pair c₁ ≤ c₂ |
| F | full | 8 | 1 × 2 | 2 | `cross_matrix` without the auto statistics of channel 0 |
| G | full | 16 | 1 × 2 | 3 | the 90 pixels of D, the mask given |

In each case the comparison covers:

- the statistics without normalisation, and with `standardize=True`, with `mean_pre_std` and `std_pre_std`;
- `norm="self"` and `norm="ref"` (a second field normalised by the reference of the first), each with and without `ref_batch_mean`;
- `iso`; `angular_fourier`; `iso` with `angular_fourier` and `scale_cosine`; `iso` with `scale_cosine`;
- the four vectors of `to_flatten` (complex and real, with and without `mean_along_batch`).

S1 to S4, `var` and the vectors have their NaN at the same entries as the reference, and differ from it by at most 1.0e-15 of the largest coefficient of the family; `mean` differs by at most 6e-17. With missing data (D, G), STL takes the covariance S3 over the mask of its deeper argument (M-MASK), and the reference over the union: they agree, so the layer-2 mask of (j₂, j₃) contains the layer-1 mask of j₃ there. The NaN of the field enter the first convolution as they are, and in D and G reach no mean; with other masks a weight at the level of rounding can carry one into a statistic (OP-102).

**Source of the loop:** `ST_Operator.__init__`, `apply`; `ST_Statistics.__init__`; `torch_backend.maskmean`; `Base_DataClass._infer_N0`.

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
3. **Interpolation.** The value of the field at each rotated point (θ, φ) is the bilinear interpolation of 4 pixels read on the rings of HEALPix (`get_interpol`, on the sphere); the 4 weights sum to 1. The rings r = 1 … 4 `nside` − 1 run from north to south; ring r, at colatitude θ\_r, holds N\_r pixels at the longitudes φ\_r + 2πk/N\_r, k = 0 … N\_r − 1 (Górski et al. 2005): N\_r = 4r in the northern cap (r < `nside`), 4 `nside` in the equatorial belt, 4(4 `nside` − r) in the southern cap; φ\_r = π/N\_r on the rings shifted by half a pixel (the caps, and the rings of the belt with r − `nside` even), 0 on the others. In the RING ordering of HEALPix, the pixels of each ring come in this order, ring after ring.
   - **On a ring**, the point reads the two pixels whose longitudes enclose φ: with t = (φ − φ\_r) N\_r / 2π, φ taken in \[0, 2π), the pixels k = ⌊t⌋ and k + 1, modulo N\_r, with the weights 1 − (t − ⌊t⌋) and t − ⌊t⌋: linear in longitude.
   - **Between two rings**, θ\_r ≤ θ < θ\_{r+1}: the two pixels of each ring, the weights of ring r multiplied by 1 − w and those of ring r + 1 by w, with w = (θ − θ\_r)/(θ\_{r+1} − θ\_r): linear in colatitude, not in cos θ.
   - **Near a pole**, θ < θ\_1: the North Pole takes the mean of the 4 pixels of ring 1, and the point is interpolated between the pole and ring 1, with w = θ/θ\_1: the two pixels of ring 1 that enclose φ get w (1 − (t − ⌊t⌋)) + (1 − w)/4 and w (t − ⌊t⌋) + (1 − w)/4, the two others (1 − w)/4 each. South of the last ring, θ > θ\_{4 nside − 1}, the same with the South Pole and w = (π − θ)/(π − θ\_{4 nside − 1}).
   - The edges of the base faces need no rule of their own: the rings run round the whole sphere.
   - **Checked** (task T5.4): the oracle's weights equal those of healpy 1.20.1, `get_interp_weights(nest=True)`, as linear functionals (the weight of each pixel, summed over the 4 slots), to 2e-14 at `nside` 16 and 6e-14 at `nside` 32, on dense samples at `nside` 1 to 32: uniform points, points within 0.3 rad of each pole and on the poles, the pixel centres, and points on every ring colatitude. Where a point falls on a pixel longitude or on a ring, a null weight may sit on another pixel (OP-92).
4. **Sum.** y\[g, k\] = Σ\_p W\[p\] · x\_interp\[g, k, p\], over the K² stencil points p, with W the kernel of M-WAV-HPX. There is no bias.

- **Shapes.** `(Nb, Nc, Npix)` gives `(Nb, Nc, L, Npix)`, complex; every map is convolved on its own. A complex input is not supported (STL raises); the second layer convolves a modulus, which is real.
- **Orientations are gauges.** The stencil turns, the kernel does not: L is the number of gauges, covering \[0, π).
- **The gauge `"cosmo"`.** Its u axis of gauge 0 is cos φ · e\_θ − sin φ · e\_φ in the north and cos φ · e\_θ + sin φ · e\_φ in the south. It is regular at both poles: around the North Pole the u axis tends to the Cartesian direction +x, around the South Pole to −x. It is discontinuous across the equator: at longitude φ, the u axes of gauge g on the two sides make the angle 2φ − 2πg/L, so the orientations agree, as directions modulo π, at four longitudes only. The southern frame is the mirror image of the northern one through the equatorial plane, with u reversed: for a map mirrored through the equator, the convolution at the mirrored pixel is the complex conjugate of the convolution of the map (ψ\[−u, v\] is the conjugate of ψ\[u, v\]), so its modulus is the same. The pixels of the equatorial ring follow the northern rule and do not have this symmetry. The gauge `"phi"` is singular at the two poles. **⚠ OP-47**
- **Partial sky.** A neighbour that is not in `cell_ids` is dropped, and so is a neighbour whose interpolation weight is below 10⁻¹², the threshold of the support test of M-MASK-HPX. The weights of the remaining neighbours of that stencil point are renormalised to sum to 1 exactly, with no floor on their sum. A stencil point with no remaining neighbour takes the neighbours of the stencil centre instead: it reads the value of the pixel itself. Nothing is cropped, and nothing marks the pixels whose stencil is truncated. **⚠ OP-50** On the full sky the threshold changes nothing beyond rounding. On a partial sky, STL drops only the neighbours outside `cell_ids`, so at a stencil point that falls on a pixel longitude or on a ring its values depend on its rounding. **⚠ OP-110**
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
- **Weights at the level of rounding.** The support test leaves valid a pixel that reads an invalid one with a weight below 10⁻¹²; with STL's default `replace_nan_value=NaN` that pixel gets NaN in the statistics. ESTEL's masks are weights, and its values stay finite. **⚠ OP-102**
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

The rule of §1.11 applies. OP-90 to OP-92 come from the check of T5.0; the check of the scattering loop (T5.2) found none; OP-102 comes from the review of T5.2 and T5.3; OP-110 from the oracle of T5.4 and the port of T5.1, decided on #165. OP-45 is decided: ESTEL computes on the sphere only. OP-46, OP-50 and OP-90 to OP-92 were decided on 24 November 2026 (Part IV, §22).

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-45 | The pixel centres are computed on the WGS84 ellipsoid by default: their latitudes are geodetic latitudes (at most 0.128° from those of the sphere), used as latitudes of a sphere by a convolution and a decimation that are otherwise spherical. A convolution changes by a relative rms of 2 % (`nside` 8) to 4 % (`nside` 16) for a white noise. The code ESTEL ports (`TorchGeometry`) supports the sphere only | ESTEL on the sphere only, and the parity harness runs STL with `ellipsoid="sphere"` | differs |
| OP-46 | healpix-analyse casts the kernel through float32 (5e-8 of its largest modulus; its sum becomes 9e-9) and caches its interpolation weights in float32: with the cache, enabled by default, the first operator of a geometry has float64 weights and the later ones float32 weights (S3 and S4 change by up to 2.5e-7). The decimation is in float64 | ESTEL builds the kernel and its tables in float64, then casts them to the dtype of the data, as the planar kernels (OP-31); the parity harness runs healpix-analyse with its cache disabled and its kernel kept in float64 (Part II, §14.2) | differs |
| OP-47 | The gauge `"cosmo"` is regular at the poles and discontinuous across the equator, where the orientations of the two sides differ by 2φ − 2πg/L; the southern frame is the mirror image of the northern one, so the moduli are symmetric under a mirror through the equator, except on the equatorial ring. The reading of the draft (singular at the poles) was that of the gauge `"phi"` | reproduce; document | reproduced |
| OP-48 | The HEALPix wavelet is not the planar one (Gaussian times plane wave, divided by the sum of moduli), so coefficients are not comparable between geometries | reproduce; LF22 | reproduced |
| OP-49 | The decimation is a Gaussian of radius 3 pixels and σ = radius/2, over 26 to 32 fine pixels of which the 4 children carry 0.28 to 0.30 of the weight; it is not the planar smoothing kernel | reproduce; Part III, §20.10 | reproduced |
| OP-50 | No border is cropped on a partial sky. At the edge, the convolution renormalises its interpolation weights over the available pixels and replaces a stencil point without any available pixel by the pixel itself; the decimation renormalises over the available fine pixels. These pixels enter every mean (on a disc of 154 pixels at `nside` 16: 121 pixels with a truncated stencil, a relative rms difference of 0.48 from the full-sky values) | reproduce, document; a treatment of the edge later (LF67) | later (LF67) |
| OP-90 | `cell_ids` must be in ascending order: otherwise the convolution of healpix-analyse (and of `TorchGeometry`) returns its values permuted once more by the inverse sort order, and the decimation reads its input as if it were sorted; no error is raised (measured: differences of the order of the values) | ESTEL requires `cell_ids` strictly increasing, with a clear error at construction (Part II, §12.4) | differs |
| OP-91 | A partial sky of 12 n² pixels given without `nside` is taken for a full sky of `nside` = n | ESTEL infers `depth` on a full sky only, and requires it whenever `cell_ids` is given (Part II, §12.4) | differs |
| OP-92 | The gather tables of `TorchGeometry` equal those of healpix-analyse as linear operators (to 1e-13, and the outputs to 2e-14), not entry by entry: about 1 % of the index entries differ (10924 of 1228800 at `nside` 16). The entries that differ lie in groups of 4 neighbours equal as linear functionals (3121 groups at `nside` 16, to 1.3e-13): a rotated point falls on a pixel centre or a ring boundary and a zero weight fills an arbitrary slot, or the same neighbours come in another order | the tables are compared as operators (T5.1) | closed |
| OP-102 | With STL's default `replace_nan_value=NaN`, the invalid pixels of a level hold NaN. A valid pixel whose interpolation reads an invalid pixel with a weight at the level of rounding, below the 10⁻¹² of the support test (M-MASK-HPX; the zero weights of OP-92), is not masked but gets NaN: in S1 and S2, and through the normalisation in S3 and S4. Found by the review of #171: at `nside` 16, level 1, pixel 188 reads pixel 167 with a weight of 8.6e-16 at gauge 2 (T5.3 gives the numbers on #164). The same happens at level 0, through the NaN of the field itself, which `replace_nan_value` does not reach: on a disc of 359 pixels at `nside` 16 with 2 missing pixels, the valid cell 60 reads the missing cell 39 with a weight of 1.87e-15 at gauge 2 | ESTEL uses masks as weights and gives finite values; the parity fixtures run STL with `replace_nan_value=None`, the setting STL's own syntheses pass, as for OP-60 (decision on #163), and apply STL's operator, built on the field with its NaN, to the field with 0 at the missing pixels, so that STL never computes on NaN (decision on #164) | differs |
| OP-110 | On a partial sky, STL's rule of M-CONV-HPX is not decided by the weights alone where a stencil point falls on a pixel longitude or on a ring. One of its 4 weights is null in exact arithmetic and a few 1e-17 after rounding, and which pixel holds it is arbitrary (OP-92). When the pixels of non-zero weight are not in `cell_ids` and the pixel of rounding weight is, the point reads that pixel with the weight 1, or reads the pixel itself, depending on the rounding (STL's `TorchGeometry` and healpix-analyse differ there by 8.9e-3). Found by the oracle of T5.4 on the fixtures of T5.3, and by the port of T5.1: on the meridians φ = 0 and π/4, 1 pixel of 50 at `nside` 8 (cell 307) and 5 of 209 at `nside` 16 (cells 60, 1219, 1231, 2291, 2300) differ from STL by up to 0.084 and 0.133 of the largest modulus. The full sky is not affected | ESTEL applies the threshold of M-CONV-HPX: a neighbour of weight below 10⁻¹² counts as absent. STL's values at these points are set by its rounding, so the parity on a partial sky leaves out the ill-conditioned rows, and the statistics of a partial sky are compared with the oracle only (decision on #165). The edge of a partial sky is redefined later (LF67) | differs |


## 5. Syntheses

**Status:** 17 November 2026 (task T6.0). For periodic syntheses, the only ones of 1.0.0 (Part IV, §22, "Periodic syntheses only in 1.0.0"; LF47), every entry is checked against STL's code (`Synthesis.py` and what it calls), and numerically, in the parity container, where an entry says "Checked": the Nyquist pre-filter, the start, the target, the weights and the value of the loss, the result, the operators and the settings passed to L-BFGS. The comparison of whole syntheses comes with T6.1. The non-periodic and masked parts are not checked; they are marked "after 1.0.0 (Part III, §16, LF47)". For full-sky HEALPix syntheses, without the prefilter and without the spectrum (OP-77), every entry is checked against the code of `HealpixUpdate`, and numerically where its HEALPix item says "Checked" (task T5.2, 24 November 2026).

**Source:** STL-Dev `master` at `8eaf769`, `Synthesis.py`; for HEALPix, STL-Dev `HealpixUpdate` at `90eb1a9`, `Synthesis.py`. A synthesis looks for a field u whose statistics match those of a target. Step 1 does not compare synthesised fields with STL's (Part II, §14.3), except in the one comparison of T6.1; this section fixes the definitions.

**Levels.** `Synthesis.py` has two levels. The user-level functions (`synthesize_from_maps`, `synthesize_from_stats`) build the operators (M-SYN-OPS) and call the mid-level ones (`optimize_from_maps`, `optimize_from_stats`), which build the target, the running field and the loss, and call L-BFGS. Two defaults differ between the levels: the seed (26 at the user level, none at the mid level) and the spectrum (in the loss at the user level, not at the mid level of `optimize_from_maps`). This section gives the user level.

**HEALPix** (task T5.2). `Synthesis.py` of `HealpixUpdate` differs from that of `master` by 869 changed lines (814 added, 55 removed), ignoring the CRLF line endings of `master`'s file. Most of them are two new syntheses by patches, not transferred (Part II, §10; decision on #163). In the functions of the two levels above, the changes only carry the geometry, and the formulas do not change:

- the number of pixel axes is read from the data (`NDIM_PIX`, one on HEALPix) instead of being 2;
- the fields of the synthesis are built from the target, or from the statistics' copy of the data (`new_like`), so that they carry its `nside` and its `cell_ids`;
- from statistics, the shape of the running field is read from `pix_shape`, Npix on HEALPix, instead of `N0`;
- the prefilter goes through the data type (`apply_bandlimit`): on a full HEALPix sky, a round trip through the spherical-harmonic transform; on a partial sky, nothing (OP-77);
- `synthesize_from_maps` takes `compute_PS` (None by default), decides it before any spectrum operator is built, and raises an error for `compute_PS=True` on data with NaN.

One change is not one of geometry: `synthesize_from_maps` of `HealpixUpdate` fixes OP-56 (`_contains_nan(init_running)`), while `synthesize_from_stats` keeps it (`init_running.isnan.any()`).

The numerical checks run in float64, in the parity container, on full skies at `nside` 8 (J = 2, L = 4), with `prefilter_Nyquist=False` and `compute_PS=False`. The optimisation is replaced by the identity unless said. They run on the sphere with the cache disabled: the defaults of the HEALPix wavelet operator are changed at run time for this purpose, since the user level gives no other way (M-SYN-OPS). They also run with STL's defaults where said.

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
- **HEALPix.** As above, with one pixel axis: a target `(Npix,)` is `(1, 1, Npix)`, `(Nc, Npix)` is `(1, Nc, Npix)`, and `(Nb, Nc, Npix)` stays a batch. The means of the standardisation are plain means over the pixels (M-DATA-HPX).
  - **Checked** (1, 2 and 3 targets; 1 and 2 channels; `mean_field` True and False): μ and σ are those of the formula above. For 3 targets of different amplitudes, STL's σ is 2.142642, and so is the formula's; the mean of the standard deviations is 1.993640. For one target, |mean| is 6.9e-18 and var − 1 is −1.1e-16. After one synthesis from statistics, the caller's S3 are multiplied by 3.5 (OP-70).
- **Source:** `optimize_from_maps`, `optimize_from_stats`, `standardize`.

### 5.2 Running field — M-SYN-INIT

- **Shape:** `(nbatch, Nc, …)`, with the target's spatial shape unless `running_shape` is given.
- **Start:** a Gaussian white noise of unit variance, or `init_running` if given; then M-SYN-NYQ, if `prefilter_Nyquist=True`.
- **Not standardised.** The running field is in the standardised units of the target, and it is never standardised itself: the loss compares its `mean` with 0 and its `var` with 1 (M-SYN-LOSS). After the pre-filter, the start has a variance below 1, close to the fraction of modes that the filter keeps (M-SYN-NYQ). **⚠ OP-73**
- **Missing data:** the missing pixels of the running field are set to 10¹⁰, and their gradient is set to 0 at every iteration, so they never move; they are NaN in the result. **⚠ OP-53** After 1.0.0 (Part III, §16, LF47).
- **Randomness:** STL seeds the global generator of PyTorch (`seed`, 26 by default in the user-level functions, no seeding by default in the mid-level ones), then draws the whole batch in one call, `torch.randn((nbatch, Nc, N1, N2))`, in the dtype and on the device of the running operator; nothing else is drawn between the seeding and this call. ESTEL uses explicit keys (Part I, §3.1, rule 5). **⚠ OP-55**
- **Taper.** An earlier list of open points mentioned a Tukey taper of the Gaussian start on non-periodic axes. STL's syntheses at the frozen commits apply none; the only cosine taper is the border mask of the spectrum estimator. **[to verify]** whether it comes from another code base (STL-Dev-MPI) and should be reproduced. After 1.0.0 (Part III, §16, LF47).
- **Checked** (same cases as M-SYN-TARGET): the start of a default synthesis is the pre-filter of `torch.randn((nbatch, Nc, N1, N2))` drawn after `torch.manual_seed(26)`, to 8.9e-16; its variance is 0.77 to 0.78 at 32 × 32, where the filter keeps 795 modes of 1024.
- **HEALPix.** Shape `(nbatch, Nc, Npix)`. From maps, on a full sky without a mask, PyTorch's generator draws other numbers between the seeding and the start. healpix-analyse's convolution (`HealPixConv`) draws its learnable weights when it is built (`kaiming_uniform_`, L × 2 × K² = 200 uniform numbers), and `set_kernel` then overwrites them. STL builds each convolution at its first use, and the target's statistics build the J convolutions of the target operator after the seeding. So, from maps, the start is drawn after J × 200 uniform draws. From statistics, the running operator builds its convolutions after the start, which is then drawn right after the seeding, as above. With a mask, the convolutions that build the masks run when the operator is built, before the seeding (#164). **⚠ OP-99**
  - **Checked:** from maps, in the six cases of M-SYN-LOSS, the start equals `torch.randn((nbatch, Nc, Npix))` drawn after `torch.manual_seed(26)` and J × 200 uniform draws (difference 0). A `randn` right after the seeding differs from it by up to 5.3. From statistics, the start equals that plain `randn` (difference 0).
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
- **HEALPix.** `HealpixUpdate` band-limits HEALPix maps through the spherical-harmonic transform of healpix-analyse, not by this filter; ESTEL synthesises full-sky HEALPix fields without a prefilter, and without the spectrum in the loss. **⚠ OP-77** With `prefilter_Nyquist=True`, the data type does the band-limit (`apply_bandlimit`): `HEALPixSHT.map2alm`, then `alm2map`, on a full sky only. A partial sky is returned unchanged. Not checked numerically: it is out of the scope of 1.0.0.
- **Source:** `apply_nyquist_filter`.

### 5.4 Loss — M-SYN-LOSS

With Φ(u) the flattened statistics of the running field, computed with `norm="ref"` (STL's `load_ref`) and the target's reference, without standardisation, and with w the weights:

```latex
\mathcal L(u) = \sum_i \big\lvert w_i\, \Phi_i(u) - w_i\, \Phi_i(x_{\text{target}}) \big\rvert^2
```

- **Families.** Φ holds `mean`, `var`, S1, S2, S3, S4 and, in the periodic case, PS (M-SYN-OPS), in the order of M-FLAT, without the structural NaN. `mean` and `var` are in the loss: they are what ties the mean and the variance of the running field to those of the standardised target.
- **Weights.** w = 1, except 3.5 for S3 and 3.5² for S4 (`adhoc_weights`, the default). They multiply the statistics of both fields before the flattening, so a difference in S3 counts with 3.5² in the loss, and one in S4 with 3.5⁴. **⚠ OP-51**
- **Vector.** The complex form of M-FLAT; |·| is the complex modulus. Since |z|² = (Re z)² + (Im z)², the loss is the squared Euclidean norm of a complete real vector: the real and the imaginary parts of every coefficient, those of S1, S2 and PS included. `to_vector` of ESTEL stays STL's `to_flatten(flatten_complex=True)`, as in STL (Part IV, §22, "The vector of the synthesis loss"): it leaves out the imaginary parts of S1, S2 and PS, which are null for the auto terms and not for the cross-channel terms of S1 and S2. With one channel the two vectors give the same loss; with cross-channel terms they do not, and the loss of 1.0.0 is the complete one. **⚠ OP-39**
- **Gradient.** STL writes |z|² as `abs() ** 2`; PyTorch's gradient of it is 0 at z = 0, as that of (Re z)² + (Im z)².
- **Batch.** With `mean_field=True`, every family is averaged over the batch, for the target and for the running fields, before the comparison: the number of running fields need not be that of the targets. With `mean_field=False`, the sum also runs over the batch, and the two numbers must be equal.
- **From statistics.** When the target is given as statistics rather than as fields, the same loss, divided by the target's Nb (not by `nbatch`) if `mean_field=True`. The factor does not move the minimum, but it scales the gradient, hence the first step of L-BFGS and what its tolerances mean. **⚠ OP-71**
- **Checked** (same cases as M-SYN-TARGET): the value STL gives to L-BFGS at the start equals the formula above, computed from statistics obtained without `Synthesis.py`, to 5e-16 relative in the four cases from maps (for one target: 5.3935191911, on 768 complex coefficients: 1, 1, 12, 12, 96, 640 and 6 for `mean`, `var`, S1, S2, S3, S4 and PS) and to 3e-16 in the two from statistics, with the division by Nb. Without the weights the loss would be 2.51 instead of 5.39. With two channels: the complex form (3038 coefficients) and the complete real vector (6076) give the same loss to 1e-16, and `to_flatten(flatten_complex=True)` (5982) a loss 4.9 % smaller; with one channel the two agree to 2e-16. The gradient at the start is finite.
- **HEALPix.** The same loss. Φ holds `mean`, `var`, S1 to S4, without PS (`compute_PS=False`, OP-77).
  - **Checked** on six cases from maps:
    - one target in a batch of 1, and in a batch of 2;
    - 3 targets for 2 running fields;
    - 2 targets with `mean_field=False`;
    - one target of 2 channels;
    - 2 targets of 2 channels with `mean_field=False`.

    The value STL gives to L-BFGS at the start equals the formula, computed from statistics obtained without `Synthesis.py`, to 8e-16 relative. For one channel at `nside` 8 there are 322 complex coefficients: 1, 1, 8, 8, 48 and 256 for `mean`, `var`, S1, S2, S3 and S4. The check from statistics (`mean_field` True and False) agrees to 0, with the division by Nb. With two channels, the complex form (1268 coefficients) and the complete real vector (2536) give the same loss (5.647983115), and `to_flatten(flatten_complex=True)` (2484) gives 4.9595, 12 % less. The gradient at the start is finite. With the target operator's settings, the loss at the standardised target itself is 0.
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
- **HEALPix.** `optimize_lbfgs` is the same in `HealpixUpdate`.
  - **Checked:** a HEALPix synthesis from maps with `max_iter=7` builds L-BFGS with the settings of the table and `max_eval` = 8. It ran 8 evaluations of the loss, and STL printed 8. The loss went from 1.42 to 0.34, and the result is finite.
- **Source:** `optimize_lbfgs`.

### 5.6 Result — M-SYN-OUT

The optimised field is de-standardised, x = u σ + μ (M-STD), its missing pixels are set to NaN (after 1.0.0: Part III, §16, LF47), and axes are removed.

- **From maps:** μ and σ are those of the target (M-SYN-TARGET): of the filtered target if it was pre-filtered, one pair per channel if `mean_field=True`, one per map otherwise.
- **From statistics:** only if the statistics were computed with `standardize=True`; μ and σ are then `mean_pre_std` and `std_pre_std`, averaged over the batch if `mean_field=True`. Here σ is a mean of standard deviations, unlike from maps. **⚠ OP-72**
- **Axes.** The batch axis is removed if `nbatch` = 1. The channel axis is removed, from maps, if the target was given as a 2D array, and kept otherwise, even with one channel; from statistics, if Nc = 1. **⚠ OP-75**
- **Checked:** with the optimisation replaced by the identity, the result equals u σ + μ with μ and σ recomputed from the definitions above, to 2e-15 (from maps: five cases; from statistics: `mean_field` True and False). Shapes from maps: a `(N, N)` target gives `(N, N)` for `nbatch` = 1 and `(2, N, N)` for 2; `(1, N, N)` gives `(1, N, N)`; `(3, 1, N, N)` with `nbatch` = 2 gives `(2, 1, N, N)`. From statistics with Nc = 1 and `nbatch` = 2: `(2, N, N)`.
- **From statistics without standardisation.** With `mean_field=True`, the default, statistics computed with `standardize=False` make the synthesis fail before it starts. STL averages `mean_pre_std` over the batch, but that field is `None`, so it raises `AttributeError`. This holds in both branches: `master` on a 2D field, `HealpixUpdate` on a HEALPix field. In STL, the rule above, no de-standardisation, is reached only with `mean_field=False`. ESTEL follows the rule above whatever `mean_field`, as it does since T6.1: the field is returned in the units of the statistics, on planar and HEALPix data alike. It does not reproduce STL's failure, a recorded exception to Part II, §9.2 (decision on #163). **⚠ OP-100**
- **HEALPix.** The same rules, with one pixel axis. The result is built on the target's geometry (`new_like`).
  - **Checked:** with the optimisation replaced by the identity, the result equals u σ + μ to 1.3e-15, in the six cases of M-SYN-LOSS and from statistics with `mean_field` True and False. Shapes from maps:
    - `(Npix,)` gives `(Npix,)` for `nbatch` = 1, and `(2, Npix)` for 2;
    - `(2, Npix)`, two channels, gives `(2, Npix)` for `nbatch` = 1;
    - `(3, 1, Npix)` with `nbatch` = 2 gives `(2, 1, Npix)`.

    From statistics with Nc = 1 and `nbatch` = 2: `(2, Npix)`.
- **Source:** `optimize_from_maps`, `optimize_from_stats`, `unstandardize`.

### 5.7 Operators — M-SYN-OPS

The user-level functions build two operators, one for the target and one for the running field, so that each has its own mask (after 1.0.0: Part III, §16, LF47). In the periodic case without missing data the two are distinct objects with the same settings.

- **J.** For each, the default J (M-DEF) minus 1 if the field is not periodic; the synthesis uses the smaller of the two. **⚠ OP-52** After 1.0.0 (Part III, §16, LF47). In the periodic case, J is the default of M-DEF; it is passed explicitly to both operators, so the spectrum uses it as J\_PS (M-PS-2D). From statistics, J is that of the target's statistics.
- **Other settings.** From maps, the user-level function passes only J, the number of bins and `has_fewer_convolutions`: L, the wavelet type and the kernel size are the defaults of M-DEF. From statistics, the running operator takes J, the number of bins and `has_fewer_convolutions` from the target's statistics, but not L nor the wavelet type: statistics computed with another L than the default make the synthesis fail on a shape mismatch. **⚠ OP-76**
- **Spectrum.** Used in the loss only if neither field has NaN: always in the periodic case of 1.0.0, with the `fft` estimator (M-PS-2D). The number of bins is the smaller of the two fields' defaults. From statistics, also only if the target's statistics hold a spectrum; the running operator then takes J and the number of bins of the target's statistics, and in STL not J\_min. **⚠ OP-76**
- **Reference.** The target operator's reference (`var_ref`, `S2_ref_sqrt_chan_diag` and, with the spectrum, `PS_ref_sqrt_chan_diag`) is copied into the running operator; from statistics, it is read from the target's statistics.
- **Checked** (targets of 32 × 32, 64 × 32 and 48 × 48, periodic): both operators have J = 3 (the default), L = 4, bump-steerable wavelets, 6 bins, J\_PS = 3, the `fft` estimator and the spectrum in the loss; they are two objects, and the running one holds the target's three reference arrays; the seed is 26, the pre-filter on, the weights 3.5 and 12.25. From statistics computed with J = 2: the running operator has J = 2, the same 4 bins and the target's reference. From statistics computed with L = 8: STL raises a size error (4 against 8).
- **HEALPix.**
  - **What the user level passes.** On a full sky, J is the default of M-DEF-HPX. From maps, the user level passes J, `has_fewer_convolutions` and `compute_PS`, and the number of bins only with the spectrum: L, the wavelet type and K are the defaults of M-DEF-HPX. The running operator also gets `replace_nan_value=None`. With `compute_PS=False`, no spectrum operator is built. With `compute_PS=None`, the default, a full sky without NaN has the spectrum in the loss (the angular spectrum, 14 bins at `nside` 8), which ESTEL leaves out (OP-77).
  - **No `wavelet_op_kwargs`.** STL's user-level syntheses build their operators without `wavelet_op_kwargs`, from maps and from statistics alike. On HEALPix they therefore always run on the `"WGS84"` ellipsoid with healpix-analyse's cache on, and nothing changes it. From statistics, this holds even when the statistics were computed with other settings: the running operator, like L in OP-76, does not take the ellipsoid or the cache settings of the statistics. With the cache, when the disk cache holds no entry for this geometry (M-CONV-HPX), the target operator computes its interpolation weights in float64, and the running operator reads them back from the cache in float32 (OP-46): at the standardised target, the loss is 1.1e-15 instead of 0 (`nside` 8). This is a precision of OP-45 and OP-46, not a new open point. ESTEL computes on the sphere (OP-45), and the comparison of T5.3 and T5.8 calls the mid level with operators built on the sphere and without the cache.
  - **`running_shape` fails on HEALPix.** A running field of another resolution is wrapped in the target's geometry (`new_like` in `ScatteringMatchModel`), which then fails; `synthesize_from_stats(..., running_shape=(3072,))` fails the same way. ESTEL refuses `running_shape` on HEALPix data with a clear error in 1.0.0, and keeps it for planar data (decision on #163). **⚠ OP-101**
  - **Checked:** in the six cases of M-SYN-LOSS, both operators have J = 2, L = 4, the wavelet type `"HealpixWavelet"` and `compute_PS=False`, with no spectrum operator. They are two objects, and the running one holds the target's `var_ref` and `S2_ref_sqrt_chan_diag`. The seed is 26 and the weights are 3.5 and 12.25. With STL's defaults, both operators are on `"WGS84"` with the cache on, and so is the running operator from statistics. A target at `nside` 8 with `running_shape=(3072,)` raises `ValueError: Expected 768 pixels, got 3072.`, from maps and from statistics (statistics of `nside` 8).
- **Source:** `synthesize_from_maps`, `synthesize_from_stats`.

### 5.8 What ESTEL adds

Checkpoints (Part II, §9.2 and §12.9): the state of the optimiser, the running field and the key are saved every `checkpoint_every` iterations, and a run resumed from a checkpoint continues the same trajectory on the same device. No STL counterpart.

### 5.9 Open points found in the synthesis code

The rule of §1.11 applies. OP-39 also concerns syntheses; for the loss it is closed (Part IV, §22, "The vector of the synthesis loss"; M-SYN-LOSS). OP-70 to OP-76 come from the check of T6.0, and OP-99 to OP-101 from that of T5.2.

| # | Point | Proposal | Status |
| --- | --- | --- | --- |
| OP-51 | Weights 3.5 for S3 and 3.5² for S4 in the loss, without a stated reason | reproduce as the default; document; LF56 | reproduced |
| OP-52 | The syntheses use the default J minus 1 for non-periodic fields | reproduce. After 1.0.0 (Part III, §16, LF47) | later (LF47) |
| OP-53 | Missing pixels of the running field are set to 10¹⁰ and frozen | ESTEL uses weights: the pixels are never read; to verify that the trajectory is unchanged. After 1.0.0 (Part III, §16, LF47) | later (LF47) |
| OP-54 | The Nyquist pre-filter uses the radius of the shorter side for rectangular fields | reproduce | reproduced |
| OP-55 | Global seed, and a default seed of 26 in the user-level functions | ESTEL takes an explicit key; no default seed | differs |
| OP-56 | In `synthesize_from_maps`, the default `prefilter_Nyquist` calls `init_running.isnan.any()`, which fails when an initial field is given | ESTEL computes the intended value. After 1.0.0 (Part III, §16, LF47) | later (LF47) |
| OP-70 | A synthesis from statistics modifies the caller's statistics in place: S3 and S4 stay multiplied by their weights, `mean_pre_std` and `std_pre_std` are replaced by their batch means, `compute_PS` can be set to False; a second synthesis from the same object has another target (measured: a starting loss of 1.6e2 instead of 0.96) | ESTEL is functional and leaves its inputs unchanged: the first synthesis is reproduced, not the effect on later ones; confirmed | differs |
| OP-71 | From statistics with `mean_field=True`, the loss is divided by the target's Nb; from maps it is not. The minimum is the same, the trajectory of L-BFGS is not | reproduce; to examine in step 2 (LF58) | reproduced |
| OP-72 | The standardisation differs between the two entries. From maps with `mean_field=True`: one μ and one σ for the batch, σ the square root of the mean variance, on the filtered target. From statistics: each map standardised on its own when the statistics were computed, never filtered, and the result de-standardised with the batch means of μ and of σ | reproduce; document; to examine in step 2 (LF58) | reproduced |
| OP-73 | The running field is never standardised, and the pre-filter leaves the start with a variance near π/4 (0.78 at 32 × 32) against a target `var` of 1 | reproduce; LF59 | reproduced |
| OP-74 | PyTorch's cap on evaluations, ⌊5 × `max_iter` / 4⌋, can stop a run before `max_iter` iterations; STL prints the number of evaluations as "iterations" | reproduce the cap; ESTEL names the two counts apart | reproduced |
| OP-75 | From maps the channel axis of the result is removed only if the target was a 2D array; from statistics, whenever Nc = 1 | reproduce in 1.0.0; one rule for both entries after step 1 (LF58) | later (LF58) |
| OP-76 | `synthesize_from_stats` builds the running operator with the default L and wavelet type, whatever those of the target's statistics: it fails for statistics computed with another L. With the spectrum it takes J and the number of bins of the statistics, not J\_min: from statistics computed with J\_min > 0, the running spectrum has other bins than the target's, and the synthesis runs without error (checked on 24 November 2026: J\_min 1 for the target, 0 for the running operator) | ESTEL takes every setting of the operator from the statistics, J\_min of the spectrum included; confirmed | differs |
| OP-77 | On HEALPix data, `HealpixUpdate` band-limits the target and the start of a synthesis through the spherical-harmonic transform of healpix-analyse (`HEALPixSHT.map2alm`, then `alm2map`), and puts the angular power spectrum in the loss by default (`compute_PS=None`, True without NaN); ESTEL 1.0.0 ports neither the transform nor the HEALPix spectra (Part II, §9.3; LF24) | ESTEL synthesises full-sky HEALPix fields without the prefilter and without the spectrum; the comparison with STL runs `HealpixUpdate` with `prefilter_Nyquist=False` and `compute_PS=False` (T5.2, T5.8); both come with a spherical-harmonic transform (LF24) | differs |
| OP-99 | On HEALPix, on a full sky without a mask, a synthesis from maps draws its start after other numbers of PyTorch's global generator. Each `HealPixConv` draws its learnable weights when it is built (`kaiming_uniform_`, 200 numbers for L = 4 and K = 5), and `set_kernel` then overwrites them. The target operator builds its J convolutions after the seeding, so the start depends on how many convolutions were built before it (measured: a plain `randn` after the seed differs from STL's start by up to 5.3). From statistics, the start follows the seeding directly | ESTEL takes an explicit key (OP-55). The comparison of T5.8 starts from STL's start, stored by T5.3 (#164) | differs |
| OP-100 | `synthesize_from_stats` with `mean_field=True`, the default, fails on statistics computed with `standardize=False`: it averages `mean_pre_std`, which is `None` (`AttributeError`), in `master` (2D) and `HealpixUpdate` (HEALPix) alike. With `mean_field=False` the field is returned without de-standardisation (M-SYN-OUT; read from the code) | ESTEL keeps its behaviour since T6.1 (M-SYN-OUT): it de-standardises only statistics computed with `standardize=True`, and otherwise returns the field in the units of the statistics, whatever `mean_field`, on planar and HEALPix data alike. STL's failure is not reproduced: a recorded exception to Part II, §9.2 (decision on #163) | differs |
| OP-101 | On HEALPix, `running_shape` (a running field of another resolution) fails: the running field is wrapped in the target's geometry (`new_like`). Measured: a target at `nside` 8 with `running_shape=(3072,)` raises `ValueError: Expected 768 pixels, got 3072.`, from maps and from statistics alike. STL has no working HEALPix synthesis at another depth | ESTEL keeps `running_shape` for planar fields, and refuses it on HEALPix data with a clear error in 1.0.0: STL never computes a HEALPix synthesis at another depth (Part II, §9.2; golden rule 11; decision on #163) | differs |
