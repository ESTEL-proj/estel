"""Consistency of the estimators (Part I, §5.1; task T2.16, issue #98; maths.md M-BORDER, M-MASK,
M-DOWN-MASK, M-MEAN).

The statistics of one stationary process, estimated in three ways on the same periodic fields,
must agree within their statistical error:

- **periodic**: the whole field, ``pbc=True``, no mask: the reference;
- **non-periodic**: the same field read with ``pbc=False``, no mask: the same pixels, the
  borders cropped (M-BORDER);
- **masked**: the same field with holes, ``pbc=True``: a fixed, seeded mask of five square holes
  of different sizes, 10.1 % of the pixels (M-MASK, M-DOWN-MASK).

**The process** is drawn by FFT, so it is periodic by construction: y = exp(g / 2), with g a
Gaussian field of unit variance and of spectrum 1 / (k0² + k²), k0 = 1/64 cycle per pixel; the
field is (y - E y) / std(y), with the moments of the log-normal law, so its mean is zero and its
variance one over the draws, not in each draw. The spectrum is a power law above k0, with the
same variance in every octave, so every scale has content, and the exponential makes S3 and S4
differ from zero. Below k0 it is flat: the correlations of g fall as exp(-2π k0 r), 3e-6 at
r = 128, so the opposite sides of a field are independent and the process is the same at both
sizes.

**No window cut from a larger field** (decision 2 of the issue named one; removed after the
review of the pull request #101). With ``pbc=False`` no mean reads across a border, so the
non-periodic estimate of a periodic field has the expectation of the estimate of a window of the
same process. But it is paired with the periodic estimate, and a window is another draw: its
standard error was 1 % of S2 at j = 0 where the paired one is 0.1 %, and a window wrongly read
with ``pbc=True`` stayed under |z| = 4. In the first version of the test the window (N = 128 with
256 draws, N = 256 with 64, cut from fields of side 4N) showed no gap either: largest |z| 3.50.

**The criterion** (issue #98, decisions 3 and 4). Over K seeded draws and for each coefficient,
real and imaginary part apart, z is the mean over the draws of the difference between an
estimate and the periodic one, divided by the standard error of that difference. The test
asserts |z| ≤ 5.5 for every compared coefficient: the threshold of decision 4, 4, corrected for
the number of comparisons (see *The threshold*). A part that is zero in every draw of both
estimates (the imaginary parts of the auto S1 and S2) is not compared, nor are ``mean`` and
``var`` of the non-periodic estimate, which have no border and equal the periodic ones.

**The rule of the compared coefficients.** A coefficient is compared when its mean keeps at least
a quarter of the pixels of its level: the pixels left by the border crop (M-BORDER) and by the
masks (M-MASK), counted by the plan, over the (N / 2^j)² pixels of the level of the mean. The
others, and the means with no pixel (OP-3), are left out.

**Two sizes, 64 draws each**: N = 128 and N = 256, with J = 5 and L = 4, raw statistics
(``norm="none"``). 64 draws is the smallest number that keeps z close to Gaussian: over nine
runs (seeds and sizes), the largest |z| was 7.0 with 16 draws (three runs above 5.5), 4.85 with
32, and 5.06 with 64. The relative standard error of S2, from j = 0 to the coarsest compared
scale, is then: non-periodic, 0.12 % to 5 % at N = 128 and 0.06 % to 1.8 % at N = 256; masked,
0.4 % to 2.8 % at N = 128 and 0.2 % to 2.5 % at N = 256. A gap smaller than that is not seen.

**Measured** (PyTorch, float64, seed 0; for each family the largest |z| and the mean of z²,
about 1 if the standard errors are right, with the number of compared real numbers):

=======  ====  ==================  ==================
size     fam.  non-periodic        masked
=======  ====  ==================  ==================
N = 128  mean  equal               0.49, 0.24 (1)
N = 128  var   equal               0.90, 0.80 (1)
N = 128  S1    1.89, 1.01 (20)     0.72, 0.20 (16)
N = 128  S2    1.95, 0.97 (20)     1.18, 0.38 (16)
N = 128  S3    3.00, 1.00 (320)    2.79, 0.86 (224)
N = 128  S4    3.55, 1.01 (2496)   3.28, 1.03 (1344)
N = 256  mean  equal               1.02, 1.05 (1)
N = 256  var   equal               1.89, 3.58 (1)
N = 256  S1    2.05, 1.61 (20)     2.51, 1.48 (16)
N = 256  S2    2.26, 1.58 (20)     2.46, 2.13 (16)
N = 256  S3    3.86, 1.33 (480)    3.75, 1.33 (320)
N = 256  S4    5.06, 1.22 (4464)   3.57, 1.24 (2560)
=======  ====  ==================  ==================

Over all the compared numbers the mean of z² is 1.00 at N = 128 and 1.24 at N = 256. The
coefficients of a family are correlated, so the mean of z² of a small family moves far from 1
from one seed to the next: it is an indication, not asserted.

**No systematic gap was found**, at either size, by either comparison: the list of decision 4 of
the issue (the gaps exempted from the threshold, each with its open point) is empty. One exists
at a smaller size, outside this test: at N = 64 with J = 5 the coarsest level has 4 x 4 pixels,
its border is too large and is reduced (OP-13), and the non-periodic S1 and S2 at j = 4 are 10
to 26 % below the periodic ones (|z| up to 11 with 256 draws). At N = 128 and 256 the means
with a reduced border keep less than a quarter of their pixels, and are not compared.

**The threshold** (review of the pull request #101). The test compares 12 356 real numbers. At
|z| ≤ 4, one independent Gaussian value in 16 000 fails, and five of the six seeds tried have a
coefficient above 4, by chance.
The threshold is therefore corrected for the number of comparisons (Bonferroni): at |z| ≤ 5.5 a
Gaussian value fails with probability 3.8e-8, so 5e-4 for the whole test. The seed, 0, was fixed
before its result was seen. Seeds 1 to 5 pass as well, with largest |z| 4.11, 4.93, 4.35, 3.54
and 4.31.

The largest |z| of seed 0, 5.06 (non-periodic, S4, N = 256), is above what Gaussian values give,
and is chance all the same: with 64 draws the means of a log-normal field have heavier tails.
Measured again with 256 draws, that coefficient is at 1.95, and at 0.27 on the 192 new draws
alone; the two others above 4 (4.85 and 4.07) fall to 2.51 and 1.70. A systematic gap would have
doubled. A coefficient above the threshold after a change is measured the same way before it is
called a gap.

Slow: 64 draws and three estimates of each, at 128 x 128 and 256 x 256 with J = 5, since the
standard error of a mean falls only as the root of the number of pixels: 24 s of CPU time on
PyTorch (4 s for N = 128, 20 s for N = 256), full tier only (issue #98, decision 7).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data2D, STOperator

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
SEED = 0
"""The seed of every draw, with the side of the field and the number of the draw; fixed before
its result was seen."""
J, L = 5, 4
K0 = 1 / 64
"""The frequency below which the spectrum of the Gaussian field is flat, in cycles per pixel."""
SIGMA = 0.5
"""The field is the exponential of ``SIGMA`` times a Gaussian field of unit variance."""
HOLES_PER_512 = (112, 88, 64, 40, 24)
"""The sides of the square holes of the mask for N = 512, scaled with N: 10.1 % of the pixels,
as in the benchmark case ``planar-masked``."""
MIN_KEPT = 0.25
"""A coefficient is compared when its mean keeps at least this fraction of the pixels of its
level."""
Z_MAX = 5.5
"""The largest |z| allowed: 4 corrected for 12 356 comparisons (Bonferroni; risk 5e-4)."""
SYSTEMATIC_GAPS: dict[str, str] = {}
"""The systematic gaps found, each with the open point of maths.md that records it (issue #98,
decision 4): none."""

pytestmark = [pytest.mark.slow, pytest.mark.maths("M-BORDER", "M-MASK", "M-DOWN-MASK", "M-MEAN")]


@dataclass(frozen=True)
class Estimate:
    """The statistics of K draws, estimated one way."""

    values: dict[str, NDArray[Any]]
    """Each family in its full grid, ``(K, ...)``, without the channel axes (one channel)."""
    kept: dict[str, NDArray[np.float64]]
    """For each family, the fraction of the pixels of its level that each mean keeps, 0 for a
    mean with no pixel; broadcasts against the family without its draw axis."""


def _draw(n: int, draw: int) -> NDArray[np.float64]:
    """One seeded draw of the process on a periodic grid of side ``n``: zero mean and unit
    variance over the draws."""
    rng = np.random.default_rng((SEED, n, draw))
    frequency = np.fft.fftfreq(n)
    spectrum = 1.0 / (K0**2 + frequency[:, None] ** 2 + frequency[None, :] ** 2)
    amplitude = np.sqrt(spectrum[:, : n // 2 + 1] / spectrum.mean())  # unit variance on the grid
    gaussian = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((n, n))) * amplitude, s=(n, n))
    mean = np.exp(SIGMA**2 / 2)
    std = np.sqrt((np.exp(SIGMA**2) - 1.0) * np.exp(SIGMA**2))
    field: NDArray[np.float64] = (np.exp(SIGMA * gaussian) - mean) / std
    return field


def _holes(n: int) -> NDArray[np.bool_]:
    """The missing pixels (True = missing): square holes of different sizes, each placed by a
    seeded uniform draw, drawn again while it touches a hole already placed."""
    rng = np.random.default_rng((SEED, n))
    missing = np.zeros((n, n), dtype=bool)
    for side in (round(side * n / 512) for side in HOLES_PER_512):
        while True:
            row, col = (int(i) for i in rng.integers(0, n - side + 1, size=2))
            hole = (slice(row, row + side), slice(col, col + side))
            if not missing[hole].any():
                missing[hole] = True
                break
    return missing


def _estimate(fields: NDArray[np.float64], mask: NDArray[np.bool_] | None, pbc: bool) -> Estimate:
    """The raw statistics of ``fields``, ``(K, N, N)``, and the fraction of the pixels each mean
    keeps, read in the plan (M-BORDER, M-MASK)."""
    n = fields.shape[-1]
    data = Data2D(fields[:, None], mask=mask, pbc=pbc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13: the means left out below
        operator = STOperator(data, J, L, norm="none", compute_PS=False)
    statistics = operator.apply(data)
    values = {}
    for family in FAMILIES:
        grid = np.asarray(statistics.full_grid(family))
        values[family] = grid[:, 0] if family in ("mean", "var") else grid[:, 0, 0]
    plan = operator.plan
    first, s3, s4 = np.zeros(J), np.zeros((J, J)), np.zeros((J, J, J))
    for j, weights in plan.first_layer.items():
        first[j] = weights.denominator / (n >> weights.level) ** 2
    for pair, weights in plan.s3.items():
        s3[pair] = weights.denominator / (n >> weights.level) ** 2
    for triple, weights in plan.s4.items():
        s4[triple] = weights.denominator / (n >> weights.level) ** 2
    moments = np.array(plan.moments.denominator / n**2)
    kept = {
        "mean": moments,
        "var": moments,
        "S1": first[:, None],
        "S2": first[:, None],
        "S3": s3[:, :, None, None],
        "S4": s4[:, :, :, None, None, None],
    }
    return Estimate(values, kept)


def _z_scores(
    estimate: Estimate, periodic: Estimate
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.bool_]]]:
    """For each family, the z-score of each real number of ``estimate`` against ``periodic``
    (real parts, then imaginary parts for a complex family, on a new first axis), and which are
    compared. The two estimates are of the same draws: the standard error is that of their
    differences."""
    scores = {}
    for family in FAMILIES:
        ours, reference = estimate.values[family], periodic.values[family]
        if np.iscomplexobj(reference):
            ours = np.stack([ours.real, ours.imag], axis=1)
            reference = np.stack([reference.real, reference.imag], axis=1)
        else:
            ours, reference = ours[:, None], reference[:, None]
        draws = reference.shape[0]
        gap = ours.mean(axis=0) - reference.mean(axis=0)
        error = (ours - reference).std(axis=0, ddof=1) / np.sqrt(draws)
        kept = np.broadcast_to(estimate.kept[family] >= MIN_KEPT, gap.shape)
        compared = kept & np.isfinite(gap) & (error > 0)
        # Where nothing varies over the draws, the two estimates are equal: a structural zero,
        # or mean and var without border or mask.
        assert not (gap[kept & np.isfinite(gap) & (error == 0)] != 0).any(), family
        z = np.divide(gap, error, out=np.full(gap.shape, np.nan), where=compared)
        scores[family] = (z, compared)
    return scores


def _comparisons(n: int, draws: int) -> dict[str, dict[str, Any]]:
    """The z-scores of the non-periodic and masked estimates against the periodic one, for ``draws``
    draws at side ``n``."""
    fields = np.stack([_draw(n, draw) for draw in range(draws)])
    periodic = _estimate(fields, None, pbc=True)
    return {
        "non-periodic": _z_scores(_estimate(fields, None, pbc=False), periodic),
        "masked": _z_scores(_estimate(fields, _holes(n), pbc=True), periodic),
    }


def test_the_mask_has_holes_of_different_sizes() -> None:
    """About 10 % of the pixels are missing, at both sizes."""
    for n in (128, 256):
        assert _holes(n).mean() == pytest.approx(0.101, abs=1e-3)


@pytest.mark.parametrize(("n", "draws"), [(128, 64), (256, 64)], ids=["N128", "N256"])
def test_the_estimates_agree_within_their_statistical_error(n: int, draws: int) -> None:
    """Every compared coefficient of the non-periodic and masked estimates is within
    ``Z_MAX`` standard errors of the periodic estimate; no systematic gap is exempted
    (``SYSTEMATIC_GAPS``)."""
    assert not SYSTEMATIC_GAPS
    outliers = []
    for name, scores in _comparisons(n, draws).items():
        for family in FAMILIES:
            z, compared = scores[family]
            # Without a border, mean and var of the non-periodic estimate equal the periodic ones.
            without_border = name == "non-periodic" and family in ("mean", "var")
            assert compared.any() or without_border, (name, family)
            for index in np.argwhere(compared & (np.abs(z) > Z_MAX)):
                where = tuple(int(i) for i in index)
                outliers.append(
                    f"{name} {family}{list(where[1:])} part {where[0]}: z = {z[where]:.2f}"
                )
    assert not outliers, (
        f"N = {n}, {draws} draws: {len(outliers)} coefficient(s) farther than {Z_MAX} standard "
        "errors from the periodic estimate (index: scales, then orientations; part 0 real, 1 "
        "imaginary):\n  " + "\n  ".join(outliers) + "\nIf src/ changed, an estimator has moved: "
        "compare with STL (tools/stl_parity) on this configuration. Otherwise measure the "
        "coefficient again with four times the draws: a systematic gap doubles its z, and is "
        "recorded as an open point of docs/maths.md and listed in SYSTEMATIC_GAPS (issue #98, "
        "decisions 4 and 5); never raise Z_MAX."
    )
