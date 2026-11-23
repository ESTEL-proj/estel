"""Consistency of the estimators (Part I, §5.1; task T2.16, issue #98, second start; maths.md
M-BORDER, M-MASK, M-DOWN-MASK, M-MEAN, OP-13, OP-61).

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
variance one over the draws, not in each draw. The spectrum is a power law above k0, so every
scale has content, and the exponential makes S3 and S4 differ from zero.

**The criterion.** Over K seeded draws and for each coefficient, real and imaginary part apart,
z is the mean over the draws of the difference between an estimate and the periodic one, divided
by the standard error of that difference: the estimates are paired, field by field. The test
asserts |z| ≤ 6.1 for every compared coefficient outside the known gap (see *The positive
control*). A part that is zero in every draw of both estimates (the imaginary parts of the auto
S1 and S2) is not compared, nor are ``mean`` and ``var`` of the non-periodic estimate, which
have no border and equal the periodic ones.

**The rule of the compared coefficients.** A coefficient is compared when its mean keeps at least
a quarter of the pixels of its level: the pixels left by the border crop (M-BORDER) and by the
masks (M-MASK), counted by the plan, over the (N / 2^j)² pixels of the level of the mean. The
others, and the means with no pixel (OP-3), are left out.

**Sizes and draws** (J = 5, L = 4, raw statistics, ``norm="none"``; sized on the issue before the
test was written): N = 64 with 512 draws, the positive control; N = 128 and N = 256 with 64 draws
each, where no gap is expected. The test compares 16 902 real numbers: 4 546 at N = 64, of which
2 072 belong to the known gap, 4 458 at N = 128 and 7 898 at N = 256.

**The threshold** comes from Student's law. With K draws, z follows Student's law with K - 1
degrees of freedom, not the Gaussian one. The test holds 14 830 real numbers to the threshold
(the 16 902 compared, without the 2 072 of the known gap), and the smallest number of draws is
64, so 63 degrees of freedom. For a family-wise risk of 1e-3 (Bonferroni), each comparison may
be beyond the threshold with probability 1e-3 / 14 830 = 6.7e-8, both tails together, which
Student's law with 63 degrees of freedom gives at |z| = 6.12: the threshold is 6.1. At 6.1 the
probability is 7.2e-8 for one comparison with 64 draws and 2.1e-9 with the 512 draws of N = 64
(1.1e-9 for a Gaussian value), so the risk of the whole test is 12 356 x 7.2e-8 + 2 474 x 2.1e-9
= 8.9e-4. The same threshold serves the positive control.

**The number of draws** at N = 128 and N = 256 is 64, the smallest of 16, 32 and 64 that keeps
that risk: at 6.1 one comparison is beyond the threshold with probability 2.0e-5 with 16 draws
and 9.2e-7 with 32, so 0.25 and 0.011 false alarms expected on the 12 356 comparisons of these
two sizes, against 8.9e-4 with 64. Measured on seeds 0 to 5 at these two sizes, the largest |z|
is 7.01 with 16 draws (three runs of 12 beyond 6.1), 4.96 with 32 and 5.06 with 64. The seed,
0, was fixed before its result was seen; seeds 1 to 5 pass as well, the control included, with
largest |z| 4.11, 4.93, 4.35, 3.54 and 4.31; over the six seeds the smallest |z| of the eight
coefficients of the control is 6.96.

**The positive control (OP-61)**: a configuration where a gap is known, to show that the test
can see one. At N = 64 with J = 5 the level of scale 4 has 4 x 4 pixels; the border of the
non-periodic means, too large for it, is reduced to 1 pixel (OP-13), and these means keep
exactly a quarter of their pixels, so the rule compares them. There the non-periodic S2 is 18.8
to 24.7 % below the periodic one according to the orientation (z from -7.5 to -11.5), and S1
10.2 to 13.7 % below (z from -7.4 to -12.1). STL (``master``, parity container) gives the same
statistics on the same fields, to 5e-16 of the largest coefficient: the gap is a property of
STL's estimator, recorded and not corrected in step 1. The test asserts that each of these
eight coefficients is beyond -6.1 and within 1.5 times its measured gap. The other coefficients
whose mean is at scale 4 (S3 and S4: 2 064 real numbers, 446 of them beyond 6.1) are exempt from
the threshold; every other compared coefficient of N = 64 is held to it. The control needs 512
draws: with 64, the |z| of the four S2 go from 1.8 to 6.3 over seeds 0 to 5, and with 256 from
5.1 to 9.4.

**Measured** (seed 0; for each family the largest |z| and the mean of z², about 1 if the
standard errors are right, with the number of compared real numbers; at N = 64 without the
known gap):

=======  ====  ==================  ==================
size     fam.  non-periodic        masked
=======  ====  ==================  ==================
N = 64   mean  equal               0.14, 0.02 (1)
N = 64   var   equal               1.08, 1.17 (1)
N = 64   S1    2.51, 1.28 (16)     1.23, 0.64 (12)
N = 64   S2    2.22, 1.01 (16)     1.06, 0.54 (12)
N = 64   S3    2.23, 0.67 (192)    2.60, 1.17 (160)
N = 64   S4    2.83, 0.77 (1216)   2.84, 0.97 (848)
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

The coefficients of a family are correlated, so the mean of z² of a small family moves far from
1 from one seed to the next: it is an indication, not asserted. No gap is found at N = 128 and
N = 256. STL gives the same statistics as ESTEL on the nine configurations of this test (the
three estimates at the three sizes, on these fields), to 2.5e-15 of the largest coefficient of
each family, with NaN at the same positions: what is measured is STL's estimator.

**What "no gap" means.** The smallest relative gap of S2 that reaches 6.1 standard errors with
64 draws (the largest over seeds 0 to 5), from j = 0 to the coarsest compared scale:
non-periodic, 1.3 % to 48 % at N = 128 (15 % at j = 3) and 0.4 % to 13 % at N = 256; masked,
2.8 % to 21 % at N = 128 and 1.6 % to 21 % at N = 256. At N = 128 the non-periodic means of
scale 4 (an 8 x 8 level, 16 pixels kept) would not show a gap as large as the control's with 64
draws; measured apart with 256 draws on the six seeds, they stay within |z| = 2.1.

The largest |z| of seed 0, 5.06 (non-periodic, S4, N = 256), is chance: measured again with 256
draws, that coefficient is at 1.95, and at 0.27 on the 192 new draws alone; the two others above
4 (4.85 and 4.07) fall to 2.51 and 1.70. A systematic gap would have doubled. A coefficient
above the threshold after a change is measured the same way before it is called a gap.

Slow: 512 draws at 64 x 64 and 64 draws at 128 x 128 and 256 x 256, three estimates of each with
J = 5, since the standard error of a mean falls only as the root of the number of pixels and of
draws: 18 s of CPU time in one process (6 s, 2 s and 10 s), and 33 to 40 s when the workers
of the full tier run the three sizes side by side on a loaded laptop; full tier only. PyTorch,
float64, one thread (``conftest.py``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data2D, STOperator
from estel.scattering._plan import PlanarPlan

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
Z_MAX = 6.1
"""The largest |z| allowed, from Student's law (module docstring, *The threshold*): never raise
it."""
CONTROL = (64, 512)
"""The side and the number of draws of the positive control (OP-61)."""
SIZES = ((128, 64), (256, 64))
"""The sides where no gap is expected, each with its number of draws."""
SCALE_AXIS = {"S1": 0, "S2": 0, "S3": 1, "S4": 2}
"""The axis of the scale of the mean of each family (j for S1 and S2, j3 for S3 and S4), in its
grid without the draws."""
OP_61_SCALE = J - 1
"""OP-61: at N = 64 the level of scale 4 has 4 x 4 pixels, and the border of the non-periodic
means, too large for it, is reduced to 1 pixel (OP-13). These means keep exactly a quarter of
their pixels, so the rule compares them."""
OP_61_GAPS = {
    "S1": (-0.111, -0.137, -0.102, -0.131),
    "S2": (-0.212, -0.247, -0.188, -0.246),
}
"""OP-61, measured (seed 0, 512 draws; STL gives the same values): the gap between the
non-periodic and the periodic estimate at N = 64 and j = 4, relative to the periodic one, for
each orientation."""
GAP_MARGIN = 1.5
"""A known gap must stay within this factor of its measured size (issue #98, decision 3)."""

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
    assert isinstance(plan, PlanarPlan)
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


@dataclass(frozen=True)
class Scores:
    """One estimate of a family against the periodic one: an entry for each real number of the
    family, the real parts then the imaginary parts of a complex family on a new first axis."""

    z: NDArray[np.float64]
    """The mean over the draws of the paired difference over its standard error; NaN where
    not compared."""
    compared: NDArray[np.bool_]
    relative_gap: NDArray[np.float64]
    """The mean difference over the mean of the periodic estimate."""


def _z_scores(estimate: Estimate, periodic: Estimate) -> dict[str, Scores]:
    """The scores of each family of ``estimate`` against ``periodic``. The two estimates are
    of the same draws, so they are paired: the standard error is that of their differences."""
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
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = gap / reference.mean(axis=0)
        scores[family] = Scores(z, compared, relative)
    return scores


def _comparisons(n: int, draws: int) -> dict[str, dict[str, Scores]]:
    """The z-scores of the non-periodic and masked estimates against the periodic one, for ``draws``
    draws at side ``n``."""
    fields = np.stack([_draw(n, draw) for draw in range(draws)])
    periodic = _estimate(fields, None, pbc=True)
    return {
        "non-periodic": _z_scores(_estimate(fields, None, pbc=False), periodic),
        "masked": _z_scores(_estimate(fields, _holes(n), pbc=True), periodic),
    }


def _known_gap(name: str, family: str, n: int, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """Which entries of a family (first axis: real and imaginary parts) belong to a known
    systematic gap, exempt from ``Z_MAX``: OP-61, the non-periodic means of scale 4 at N = 64."""
    known = np.zeros(shape, dtype=bool)
    if (name, n) == ("non-periodic", CONTROL[0]) and family in SCALE_AXIS:
        np.moveaxis(known, 1 + SCALE_AXIS[family], 0)[OP_61_SCALE] = True
    return known


def _outliers(n: int, comparisons: dict[str, dict[str, Scores]]) -> list[str]:
    """The compared coefficients farther than ``Z_MAX`` from the periodic estimate, outside the
    known gaps."""
    outliers = []
    for name, scores in comparisons.items():
        for family in FAMILIES:
            z, compared = scores[family].z, scores[family].compared
            # Without a border, mean and var of the non-periodic estimate equal the periodic ones.
            without_border = name == "non-periodic" and family in ("mean", "var")
            assert compared.any() or without_border, (name, family)
            held = compared & ~_known_gap(name, family, n, z.shape)
            for index in np.argwhere(held & (np.abs(z) > Z_MAX)):
                where = tuple(int(i) for i in index)
                outliers.append(
                    f"{name} {family}{list(where[1:])} part {where[0]}: z = {z[where]:.2f}"
                )
    return outliers


def _assert_no_outlier(n: int, draws: int, comparisons: dict[str, dict[str, Scores]]) -> None:
    outliers = _outliers(n, comparisons)
    assert not outliers, (
        f"N = {n}, {draws} draws: {len(outliers)} coefficient(s) farther than {Z_MAX} standard "
        "errors from the periodic estimate (index: scales, then orientations; part 0 real, 1 "
        "imaginary):\n  " + "\n  ".join(outliers) + "\nIf src/ changed, an estimator has moved: "
        "compare with STL (tools/stl_parity) on this configuration. Otherwise measure the "
        "coefficient again with four times the draws: a systematic gap doubles its z, and is "
        "recorded as an open point of docs/maths.md and exempted in _known_gap (issue #98); "
        "never raise Z_MAX."
    )


def test_the_mask_has_holes_of_different_sizes() -> None:
    """About 10 % of the pixels are missing, at every size."""
    for n in (CONTROL[0], *(n for n, _ in SIZES)):
        assert _holes(n).mean() == pytest.approx(0.101, abs=1e-3)


@pytest.mark.parametrize(("n", "draws"), SIZES, ids=[f"N{n}" for n, _ in SIZES])
def test_the_estimates_agree_within_their_statistical_error(n: int, draws: int) -> None:
    """Every compared coefficient of the non-periodic and masked estimates is within ``Z_MAX``
    standard errors of the periodic estimate: no gap at these sizes."""
    _assert_no_outlier(n, draws, _comparisons(n, draws))


def test_the_reduced_border_gap_is_seen_at_the_small_size() -> None:
    """The positive control (OP-61): at N = 64 the non-periodic S1 and S2 of scale 4, whose
    border is reduced (OP-13), are below the periodic ones by more than ``Z_MAX`` standard
    errors, and by at most ``GAP_MARGIN`` times the measured gap; every compared coefficient of
    the other scales, and of the masked estimate, is within ``Z_MAX``."""
    n, draws = CONTROL
    comparisons = _comparisons(n, draws)
    for family, measured in OP_61_GAPS.items():
        scores = comparisons["non-periodic"][family]
        assert scores.compared[0, OP_61_SCALE].all(), family  # real parts; kept: a quarter
        z, gap = scores.z[0, OP_61_SCALE], scores.relative_gap[0, OP_61_SCALE]
        assert (z < -Z_MAX).all(), (
            f"{family} at j = {OP_61_SCALE}, N = {n}: z = {z.round(2).tolist()}, not all below "
            f"-{Z_MAX}. The test no longer sees the known gap OP-61 (docs/maths.md, section "
            "1.11): if src/ changed, compare with STL (tools/stl_parity) on this configuration; "
            "if the gap was corrected on purpose, update OP-61 and this control."
        )
        # z < 0 above: the gap is negative.
        assert (np.abs(gap) <= GAP_MARGIN * np.abs(measured)).all(), (
            f"{family} at j = {OP_61_SCALE}, N = {n}: relative gap {gap.round(3).tolist()}, "
            f"outside {GAP_MARGIN} times the measured {list(measured)} (OP-61). The gap has "
            "grown: compare with STL (tools/stl_parity) on this configuration before touching "
            "OP_61_GAPS."
        )
    _assert_no_outlier(n, draws, comparisons)
