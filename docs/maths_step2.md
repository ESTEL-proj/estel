# maths_step2.md — Mathematics proposed for step 2

[Back to the overview](transfer/overview.md) · [Mathematics of step 1](maths.md)

**Status:** proposals, 16 November 2026. Step 2 is described in prose in Part III of the specification (§19 and §20); this file gives the definitions that an implementation and an oracle need. Nothing here is decided unless the decision log (Part IV, §22) records it, and nothing here describes what ESTEL 1.0.0 computes: that is `docs/maths.md`.

**Role of this file.** Like `docs/maths.md`, it changes only through pull requests that a maintainer requests and merges. Its IDs are defined here; `tools/check_maths_ids.py` checks those of `docs/maths.md` only.

**Conventions.**

- **Notation:** that of `docs/maths.md`, with the blocks "Filters and scales" and "Moments and covariances" of its §1.0.
- **Each entry says** what is decided, what is proposed, and what is open. An entry marked **new** is not yet in the register of later features (Part III, §16). References R-… point to the review of 16 November 2026 in Part IV.
- **Numbers** come from counts and from tests on a NumPy reconstruction of STL's loop. The reconstruction is validated: its kernels are identical to those of the parity fixtures, and its statistics agree with STL's to 1e-7 on the periodic fixture. Timings are from NumPy on one CPU core: they compare forms of a computation, they do not predict PyTorch or JAX on a GPU. **[to verify]** marks what a benchmark or the parity container must confirm.
- **Every entry changes values** with respect to 1.0.0, unless stated: each is a traced change of `STATISTICS_REFERENCE`.

## Contents

- [6.1 Windows — M-WIN](#61-windows--m-win)
- [6.2 Centred covariances — M-COV](#62-centred-covariances--m-cov)
- [6.3 Gram form — M-GRAM](#63-gram-form--m-gram)
- [6.4 Low-pass filter in every family — M-LP2](#64-low-pass-filter-in-every-family--m-lp2)
- [6.5 Canonical coefficient set — M-CANON](#65-canonical-coefficient-set--m-canon)
- [6.6 Strict scale order — M-STRICT](#66-strict-scale-order--m-strict)
- [6.7 Isotropic reduction and conjugation — M-ISO2](#67-isotropic-reduction-and-conjugation--m-iso2)
- [6.8 Partitioned sums — M-PART](#68-partitioned-sums--m-part)
- [6.9 Level-wise gradients — M-GRAD](#69-level-wise-gradients--m-grad)
- [6.10 Requirements on the filters — M-FB](#610-requirements-on-the-filters--m-fb)
- [6.11 Strided second layer — M-STRIDE](#611-strided-second-layer--m-stride)

---

## 6.1 Windows — M-WIN

*Decided in principle (Part III, §20.11, rules 4 and 5). This entry makes it precise.*

A window w is the set of pixels of a level that enter a mean, with |w| their number. A mean is ⟨a⟩\_w = Σ\_{n ∈ w} a\[n\] / |w|.

- **Three kinds,** decided by the plan:
  - **full:** every pixel; the mean is a plain sum. Periodic data without mask.
  - **slice:** a rectangle, the level minus a border of β pixels on each side; the mean is a sum over a slice of the array. Non-periodic data without mask.
  - **weights:** a 0/1 map. Data with a mask.
- **One window per level** when centring is active. At level d, every map has at most two wavelet convolutions in its history, the second at level d. With M-BORDER, the largest border at a level is reached when both convolutions are at that level:

```latex
\beta_d = 2\,\lfloor K/2 \rfloor
```

  that is 4 pixels for K = 5, at every level, against 2 for a map convolved once. The common window costs 2 more pixels per side on the single-convolution means, and nothing on the others at j₁ = j.
- With a mask, the common window is the intersection of the valid sets of all the maps of the level: the layer-2 masks of M-MASK contain the layer-1 mask whenever the downsampling threshold does not re-validate a pixel. **[to verify]** the exact rule on the masks of the fixtures (see OP-35).
- **Changes values?** Not for periodic data without mask. Elsewhere yes, through the border of the single-convolution means.

## 6.2 Centred covariances — M-COV

*Decided (Part III, §20.11; LF33). This entry gives the formulas.*

At a level, let A₁, …, A\_P be the maps whose moments are needed, and w the window. With

```latex
m_i = \langle A_i \rangle_w, \qquad G_{ik} = \langle A_i\, \overline{A_k} \rangle_w
```

the covariance, in the two modes of `center`, is

```latex
C^{\text{self}}_{ik} = G_{ik} - m_i\, \overline{m_k}, \qquad
C^{\text{ref}}_{ik} = G_{ik} - m_i\, \overline{\mu_k} - \mu_i\, \overline{m_k} + \mu_i\, \overline{\mu_k}
```

where μ are the first moments of the reference.

- **Rule.** C\_ik = G\_ik, with no correction, when the plan proves that m\_i = 0 or m\_k = 0: the last filter of that map is band-pass and w is full. Otherwise the correction is applied.
- **Cost.** One sum per map for m, and an outer product on arrays of coefficients.
- **Numerical stability.** G − m m^H subtracts two close numbers when the mean dominates the fluctuations. Measured in float32, on the maps that end with a low-pass filter:

| Field | (second moment) / (covariance) | coefficients corrected, float32 sums | coefficients corrected, float64 sums | maps centred first, float32 |
| --- | --- | --- | --- | --- |
| lognormal, red spectrum | 2 to 4 | 3e-7 | 9e-8 | 9e-8 |
| white noise, 1024² | up to 2,170 | 2e-4 | 8e-8 | 8e-8 |
| white noise, 2048², J = 5 | up to 550 | 8e-5 | 4e-8 | 4e-8 |

  (errors relative to the scale of the covariance.) So the rule "correct coefficients, never maps" of Part III, §20.11 needs an amendment (Part IV, R-E02):
  - band-pass maps with a border or a mask: the mean is small, the coefficients are corrected;
  - maps that end with a low-pass filter: the maps are centred, with a mean computed in float64. They live at the coarsest level, so this is free.
- **Bias.** With `self`, the estimator is biased by a factor of order 1 − 1/N\_eff, with N\_eff the number of independent samples in w. It is not corrected.
- **Gradients.** With `self`, they flow through m. With `ref`, μ is a constant.
- **Statistics.** S2\[j, ℓ\] = C between the band map and itself; S3\[j₁, ℓ₁; j, ℓ\] = C between an arm and the band map; S4 = C between two arms. S1 is a first moment and is unchanged.
- **Orders of magnitude** of the correction relative to the covariance: OP-36.

## 6.3 Gram form — M-GRAM

*In the register as LF31. This entry defines it.*

Fix a level j, which is also the scale of the band, and one band orientation ℓ. Let the rows of the matrix A ∈ ℂ^(P × n) be the arms: the P = (j + 1) L maps ψ\_ℓ ⋆ D\_{j₁→j} U\_{j₁,ℓ₁} for j₁ ≤ j and all ℓ₁, each with its n pixels flattened. Let b ∈ ℂⁿ be the band map W\_{j,ℓ} x, and W the diagonal matrix of the window weights. Then

```latex
G = \frac{A\, W\, A^{\mathsf H}}{\lvert w \rvert}, \qquad g = \frac{A\, W\, \overline{b}}{\lvert w \rvert}
```

hold every S4 and every S3 of that band and orientation: S4\[j₁, ℓ₁, j₂, ℓ₂; j, ℓ\] = G at the rows (j₁, ℓ₁) and (j₂, ℓ₂), and S3\[j₁, ℓ₁; j, ℓ\] = g at the row (j₁, ℓ₁).

- **One matrix product** per level and band orientation replaces the loops over (j₁, j₂) and the broadcast over (ℓ₁, ℓ₂). With a full or slice window, W is absent.
- **G is Hermitian:** only its upper triangle is computed and stored (M-CANON).
- **Centring** is the rank-one correction G − m m^H (M-COV).
- **Channels.** For a pair (c₁, c₂), G = A\_{c₁} W A\_{c₂}^H / |w|.
- **Gradient.** For a real loss whose derivative with respect to G is Θ, the derivative with respect to A is (Θ + Θ^H) A W / |w|: one more matrix product (M-GRAD).
- **Changes values?** No: the same numbers, up to the order of the sums (2e-15 in the test below). With a tolerance, not a reference change.
- **Measured,** NumPy on one CPU core, 256² pixels, L = 4, the S4 of one band:

| Form | time, 4 arms | extra memory | time, 12 arms | extra memory |
| --- | --- | --- | --- | --- |
| present: broadcast product of shape (L, L, L, n), times the weights, then a sum | 49 ms | 8.0 × the maps | 261 ms | 3.0 × |
| the same without the weights | 20 ms | 5.0 × | 113 ms | 1.7 × |
| Gram | 3.2 ms | 0.25 × | 14 ms | 0.25 × |

- **Rules.**
  - **Layout:** the band orientation is the leading axis of the second-layer maps; the arms of one orientation then form the matrix A without a copy.
  - **Windows:** STL's window for a pair is the intersection of the windows of its two maps: the union of their masks, and the larger of their two borders. With every map set to zero outside its own window, the product A A^H sums each pair over that intersection, and the number of its pixels is the product M M^T of the 0/1 windows M of the maps. So G = (A A^H) / (M M^T), element by element, reproduces STL's per-pair windows exactly, masks and borders included, without the common window of M-WIN; a pair with no valid pixel gives NaN, as in STL (OP-3). With a window common to the level, M M^T is one number and the formula above holds.
  - **Single precision:** one product over all the pixels accumulates in the precision of the data. Relative error of a mean in float32: 2e-8 at 16,000 pixels, 2e-6 at 4 million, against 1e-8 for the pairwise sum used today. Products by blocks of pixels, added in float64, stay at 3e-9 at every size. The Gram form is computed by blocks.
- **Tensor products:** the present form has one operation per triple of scales, 22,400 for 2D+1D+1D with J = (6, 4, 4), each with a temporary of 4,096 numbers per pixel. The Gram form is a prerequisite there (Part IV, R-T06).
- **[to verify]** by a benchmark on PyTorch and JAX, CPU and GPU.

## 6.4 Low-pass filter in every family — M-LP2

*Decided in principle (LF6; Part III, §20.1, §20.10, §20.11). Several choices below are open.*

- **Index.** In every geometry, the low-pass filter is the scale j = J, after the wavelets j = 0 … J−1, with one orientation.
- **Realisation.** The smoothing kernel applied at level J−1, **without the final downsampling** (decided, 13 November 2026). For a map y at level J−1:

```latex
\Phi\, y = s \star y
```

  with the padding of M-DOWN, and no stride. The effective filter is that of STL's cascade; the statistics are means over four times more pixels in 2D, and no size constraint is added (OP-34).
- **Statistics.** With the mean of the data removed first (Part III, §20.11, rule 7):

```latex
S_1[J] = \langle \lvert \Phi\, x^{(J-1)} \rvert \rangle, \qquad S_2[J] = \mathrm{Cov}(\Phi x^{(J-1)}, \Phi x^{(J-1)})
```
```latex
S_3[j_1,\ell_1; J] = \mathrm{Cov}\big(\Phi\, D\, U_{j_1,\ell_1} x,\ \Phi\, x^{(J-1)}\big), \qquad
S_4[j_1,\ell_1,j_2,\ell_2; J] = \mathrm{Cov}\big(\Phi\, D\, U_{j_1,\ell_1} x,\ \Phi\, D\, U_{j_2,\ell_2} x\big)
```

  All are centred: their last filter is a low-pass filter (M-COV). The first moment of Φ D U\_{j₁,ℓ₁} x is S1\[j₁, ℓ₁\] up to window effects.
- **Open: which combinations.** Two sets were counted in the STOC deliverable (§5):
  - **restricted:** the low-pass filter as the last filter only, as above: +0.3 to +3.5 % coefficients;
  - **all combinations:** also as a first filter, with arms U\_J = |Φ x|: +38 to +112 %.
- **Open: the half-pixel shift.** The smoothing kernel has an even size, so Φ y is shifted by half a pixel with respect to y (OP-2). For a mean this has no effect on a periodic domain; with a window it moves the border by one pixel. To settle with the centred kernels of LF25.
- **Open: a dedicated kernel φ.** It remains possible (Part III, §20.10); the plan describes the realisation.

## 6.5 Canonical coefficient set — M-CANON

*In the register as LF20, "decision after 1.0.0". This entry lists the redundancies and counts them.*

- **Hermitian symmetry of S4.** S4\[c₁, c₂; j₁, ℓ₁, j₂, ℓ₂; j, ℓ\] is the conjugate of S4\[c₂, c₁; j₂, ℓ₂, j₁, ℓ₁; j, ℓ\]. For an auto term (c₁ = c₂), the full-grid tensor of STL stores both (ℓ₁, ℓ₂) and (ℓ₂, ℓ₁) at j₁ = j₂: (j + 1) L (L − 1) / 2 duplicates per band and orientation. The diagonal entries, j₁ = j₂ and ℓ₁ = ℓ₂, are real.
- **S1 and S2,** auto terms: real. Cross terms: (c₁, c₂) and (c₂, c₁) are conjugates, already stored once.
- **S3 and S4, cross terms:** both channel orders are computed in STL (OP-14); with the Hermitian symmetry, one order of S4 suffices. S3 has no such symmetry: its two arguments play different roles.
- **Counts,** auto terms, one channel:

| J, L | STL (non-NaN) | canonical | of which S4: STL, canonical |
| --- | --- | --- | --- |
| 4, 4 | 1,474 | 1,234 | 1,280, 1,040 |
| 6, 4 | 3,970 | 3,466 | 3,584, 3,080 |
| 8, 4 | 8,322 | 7,458 | 7,680, 6,816 |
| 6, 8 | 30,114 | 25,410 | 28,672, 23,968 |

  The first line is the count of the STOC deliverable (Table 1), which is STL's count, duplicates included. The canonical set removes 11 to 19 % of S4.
- **Open (LF20):** whether the canonical set is what `to_vector` returns, and how real and imaginary parts are counted for the entries that are real.

## 6.6 Strict scale order — M-STRICT

***New proposal (R-P02).*** *Not in the register.*

STL computes the second layer for j₁ ≤ j₂ ≤ j, **equality with the band included**: the modulus of scale j is filtered again at the same scale j. The proposal is the set j₁ ≤ j₂ **<** j, and j₁ < j for S3.

- **In the papers.** Cheng et al. (2024) require the second wavelet to be at a strictly lower frequency than both arms in S4, and allow equality in S3. Mousset et al. (2024) write j₁ ≤ j₂ ≤ j₃.
- **Why it matters numerically.** The second convolution at the same scale happens at the level of that scale, the finest for j = 0. In a count of multiplications per pixel (K = 5, L = 4):

| Index set | convolutions | covariances | total | second-layer maps to keep |
| --- | --- | --- | --- | --- |
| STL, j₂ ≤ j | 1,690 | 720 | 2,410 | 57 reals per pixel, of which 32 at level 0 |
| strict, j₂ < j | 620 | 180 | 800 | 14 reals per pixel, none at level 0 |

  The cost is divided by three and the memory by four. Level 0 keeps only the first layer.
- **Number of coefficients:** 51 to 70 % of the canonical set, for J = 4 to 8.
- **What is lost, measured.** Magnitudes normalised by √(S2 S2), medians over lognormal and Gaussian test fields (256², J = 5, L = 4):

| Coefficients | lognormal | Gaussian |
| --- | --- | --- |
| S3, j₁ = j (dropped) | 0.036 | 0.003 |
| S3, j₁ < j (kept) | 0.103 | 0.004 |
| S4, one arm at the band scale (dropped) | 0.0047 | 0.0005 |
| S4, both arms at the band scale (dropped) | 0.0038 | 0.0030 |
| S4, both arms finer (kept) | 0.0135 | 0.0007 |

  The dropped S3 terms and the dropped S4 terms with one arm at the band scale are about a third of the kept ones, and ten times their Gaussian level: they carry non-Gaussian information. Their magnitude does not depend on the orientation difference, so keeping one orientation is not a middle way.
- **Position.** A trade-off, not a candidate default. M-STRIDE gives the same gain and drops nothing. The two combine, to a factor of about 6, for the cases that need it, after a validation on syntheses.
- **Changes values?** It removes coefficients; those kept are unchanged.

## 6.7 Isotropic reduction and conjugation — M-ISO2

*In the register as LF32 (OP-5). A test shows that STL's definition needs no change; what is open is the imaginary part.*

The orientations ℓ = 0 … L−1 cover \[0, π). For a real field, extend them to ℓ ∈ ℤ: a filter turned by π gives the conjugate response.

- **Band orientation:** W\_{j,ℓ+L} x = conj(W\_{j,ℓ} x), and ψ\_{ℓ+L} ⋆ u = conj(ψ\_ℓ ⋆ u) for a real u.
- **Arm orientation:** U\_{j₁,ℓ₁+L} = U\_{j₁,ℓ₁}: the modulus removes the conjugation, so an arm orientation is defined modulo L, with no conjugate.

**What STL computes.** M-RED-ISO sums the coefficients whose orientation differences are equal modulo L. Each band orientation ℓ = 0 … L−1 appears once in such a sum, and only arm orientations wrap. It is therefore the average over the band orientation, with arm orientations relative to the band:

```latex
S_3^{\text{iso}}[\delta] = \frac{1}{L} \sum_{\ell=0}^{L-1} S_3[j_1, \ell - \delta; j, \ell], \qquad
S_4^{\text{iso}}[\delta_2, \delta_3] = \frac{1}{L} \sum_{\ell=0}^{L-1} S_4[j_1, \ell - \delta_3, j_2, \ell - \delta_3 + \delta_2; j, \ell]
```

Tested: this agrees with STL's loops to 1e-18. No value is mixed with a conjugate of another.

**What is and is not invariant.** A rotation of the field by π/L shifts every orientation by one, and the band orientation that passes L comes back conjugated: one term of the sum is replaced by its conjugate.

- The **real part** of the average is invariant under rotations by multiples of π/L.
- The **imaginary part** is not: it depends on which half-plane of orientations is called \[0, π). For a field whose statistics are invariant under a rotation by π, its expectation is zero.
- **Measured** on a lognormal field turned by 90 degrees (128², J = 3): the real parts change by 3e-3, which is the defect of the kernels themselves (Part IV, R-N19), and the imaginary parts by more than their own size. The imaginary parts are about a tenth of the real parts.

**Open.** Whether `iso` returns the real parts only. It would halve the reduced S3 and S4, and remove from a loss a term that is noise for isotropic fields and is not invariant in any case. It is a change of values.

## 6.8 Partitioned sums — M-PART

***New proposal (R-P04).*** *It would cover LF4 (MPI) and LF19 (memory beyond one device).*

Every statistic is a ratio of two sums over pixels. Split the domain Ω of a level into parts Ω\_p:

```latex
\langle a \rangle_w = \frac{\sum_p \sum_{n \in \Omega_p \cap w} a[n]}{\lvert w \rvert}
```

- **Exact.** The sums over parts add up to the sum over the domain; |w| is global and known by the plan. The internal `_global_sum` is the point where the parts are added.
- **Halo.** To compute a map on Ω\_p, a part needs the field on a halo around it. For a map at level d, the halo at full resolution grows like 2^d times the kernel sizes: about 2^d (3 + β\_d) pixels on each side, from the smoothing kernels down to level d and the wavelet borders there.
- **Two regimes.** At fine levels the halo is thin compared with a part: the parts are processed independently. At coarse levels the halo reaches the size of a part, but the maps of the whole domain are small: from a level d\* on, the pyramid is gathered and processed as one piece. In the cost model of M-STRICT, level 0 alone is 59 % of the convolutions.
- **One mechanism, four uses:** a field larger than the memory of one device (streaming over parts), several GPUs, MPI, and HEALPix at high resolution, where the gather tables themselves do not fit (R-H04).
- **Open:** the choice of d\*, the handling of masks across parts, and the gradient, which needs the adjoint of the halo exchange.

## 6.9 Level-wise gradients — M-GRAD

***New proposal (R-P03).*** *It would bring to the Python engine what the Rust engine does (Part III, §18.1).*

A synthesis needs the gradient of a loss of the statistics with respect to the field. Automatic differentiation keeps every intermediate map until the backward pass: in reals per pixel, for L = 4, about 57 for the second-layer maps, 11 for the first layer and 7 for the moduli brought to coarser levels.

- **Structure.** The statistics of band (j, ℓ) depend on the field only through the maps of level j. With the Gram form, the derivative of the loss with respect to those maps is one matrix product (M-GRAM).
- **Proposal.** Keep only the pyramids of the field and of the moduli; for each level and band orientation, recompute its second layer, form its contribution to the gradient, and release it. The second-layer maps are never all in memory.
- **Order of magnitude:** from about 75 to about 25 reals per pixel, a factor of 3, at the price of a second forward computation of the second layer. This counts the maps kept, not the temporary arrays of the present loop, which are larger and which M-GRAM removes first. With M-STRIDE the second-layer maps are four times smaller, and the gains combine.
- **Not counted here:** the history of L-BFGS, 100 reals per pixel with STL's default of 50 pairs (Part IV, R-P16).
- **Means.** `torch.utils.checkpoint` and `jax.checkpoint` recompute a block during the backward pass; no hand-written adjoint is needed, so the rule "no `custom_vjp` in the Python engine" (Part I, §3.2) holds. Second-order derivatives stay available.
- **[to verify]** by a benchmark of peak memory and time, against the figures of STL-Dev-MPI (19.8 GB against 2.2 GB at `nside` 128).

## 6.10 Requirements on the filters — M-FB

*Decided as requirements (Part III, §20.10). This entry gives the quantities to measure; they are the three diagnostics of Part II, §12.10.*

With ŝ the transform of the smoothing kernel, and E(ω) = ½ Σ\_ℓ (|ψ̂\_ℓ(ω)|² + |ψ̂\_ℓ(−ω)|²) the energy response of the wavelets of one level:

- **Energy partition:** P(ω) = |ŝ(ω)|² + E(ω). Requirement: |P − 1| ≤ ε\_P on the band of the level.
- **Aliasing:** the fraction of the transmitted energy that folds back at a downsampling, for a field of flat spectrum:

```latex
a = \frac{\int_{\lVert \omega \rVert_\infty > \pi/2} \lvert \hat s(\omega) \rvert^2\, d\omega}{\int \lvert \hat s(\omega) \rvert^2\, d\omega}
```

  Requirement: a ≤ ε\_a.
- **Littlewood–Paley sum:** Λ of `docs/maths.md`, with the low-pass filter. If P = 1 at every level, Λ = 1.
- **Coverage:** the set where E + |ŝ|² is below a threshold at level 0: the first octave and, in 2D, the corners.
- **Values for STL's kernels** (OP-32): in 1D, |P − 1| ≤ 0.1 up to 3π/4, and P falls to 0.23 at the Nyquist frequency; |ŝ|² is 0.47 at π/2, where the aliasing requirement wants 0 and the partition requirement 0.4. In 2D, Λ lies between 0.22 and 1.76 in the disc and near 0 in the corners.
- **Open:** ε\_P and ε\_a, the gain convention (OP-33), and whether a residual high-frequency band is added.

## 6.11 Strided second layer — M-STRIDE

***New proposal (R-P15).*** *Not in the register. The strongest lever on cost that keeps every coefficient.*

**Definition.** For band j, let Γ be the grid of level j + 1: one pixel in two per axis of level j. The arms and the band map are evaluated on Γ only, and the means run over Γ ∩ w:

```latex
S_4[j_1,\ell_1,j_2,\ell_2; j,\ell] = \big\langle (\psi_\ell \star D U_{j_1,\ell_1} x)\ \overline{(\psi_\ell \star D U_{j_2,\ell_2} x)} \big\rangle_{\Gamma \cap w}
```

and the same for S3. The convolution ψ\_ℓ ⋆ is computed at the positions of Γ only, with its input at level j. The first layer, S1 and S2 are unchanged.

- **Why it is sound.** A mean is the zero frequency of a product. Sampling the product on Γ adds its components at the frequencies π along each axis, which for a stationary field have random phases: an error with zero mean, not a bias.
- **Cost,** in multiplications per pixel (K = 5, L = 4):

| | convolutions | covariances | total | second-layer maps |
| --- | --- | --- | --- | --- |
| STL | 1,690 | 720 | 2,410 | 57 reals per pixel |
| strided second layer | 620 | 180 | 800 | 14 |
| strided, with the strict order (M-STRICT) | 355 | 45 | 400 | 3.6 |

  In 1D the second layer costs half instead of a quarter.
- **Error, measured** (normalised by √(S2 S2), medians):

| Field | coefficient | error of the strided mean | scatter between realisations |
| --- | --- | --- | --- |
| lognormal, S3 | 0.055 | 0.0031 | 0.023 |
| lognormal, S4 | 0.0062 | 0.00034 | 0.0031 |
| Gaussian, S3 | 0.0028 | 0.0006 | 0.0025 |
| Gaussian, S4 | 0.0008 | 0.00005 | 0.0003 |

  The error is 4 to 9 times smaller than the scatter; added in quadrature it raises the standard deviation of the estimator by 1 to 3 %.
- **No bias.** On the positive terms (the variance of an arm), the mean signed relative error is −0.009 ± 0.037 at 128² and 0.0000 ± 0.011 at 512².
- **Scaling.** The ratio of the error to the scatter is 0.11 at 128² and 0.14 at 512²: the error shrinks with the size of the field as the scatter does.
- **Limits.** The values change. A target and a running field must use the same setting. The test fields are stationary with power-law spectra: **[to verify]** on fields with strong small-scale structure, and on syntheses.
- **Open.** Whether it is an option or the default of step 2; whether the stride applies to S3 as to S4; its interplay with the windows (Γ ∩ w with a mask).
