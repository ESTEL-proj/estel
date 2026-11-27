"""Spec parity of the statistics: ESTEL against the oracle ``estel._spec`` (Part I, §4.2 and
§5.1; task T2.6, issue #18; maths.md sections 1 and 2, without the spectrum).

ESTEL's ``STOperator`` and the oracle's ``spec_apply`` run on the same seeded inputs, in float64,
and every family is compared with ``assert_parity``: same full-grid shape, NaN at the same
positions, values within the float64 tolerances of Part II, §14.3 (rtol 1e-10, atol = 0.1 x rtol
x max|oracle|; for ``mean``, whose theoretical value may be zero, atol = 0.1 x rtol x
sqrt(max(var))). The helper's messages name STL as the expected side: here it is the oracle.

Each case compares the statistics without normalisation and their vector, the statistics
normalised by themselves, those of a second field normalised by that reference, and the
reductions.

The fast tier runs one small case per geometry and periodicity, with a mask and J = 2
(``test_small_case_equals_the_oracle``). The full matrix of Part I, §5.1 (wavelet families,
masks, pbc, cross channels, normalisation modes, reductions) runs in the full tier
(``test_matrix_equals_the_oracle``).

A disagreement is a finding about ``docs/maths.md`` or about the oracle, since ESTEL matches
STL's fixtures: do not change ESTEL, the oracle or a tolerance to make a test pass; report the
M- ID and the coefficients in an issue.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data, Data1D, Data2D, STOperator, STStatistics, _config
from estel._spec import cartesian as spec
from estel._spec import common
from estel.scattering._plan import PlanarPlan
from tests.parity._compare import assert_parity

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
APPLY = inspect.signature(STOperator.apply)
GEOMETRIES: dict[int, type[Data]] = {1: Data1D, 2: Data2D}
SIZES = {1: 64, 2: 32}
"""Pixels per axis: the smallest sizes at which the masked, non-periodic cases keep valid pixels
in every mean at J = 2, and in most of them at J = 3."""
pytestmark = pytest.mark.maths(
    "M-NOT",
    "M-DATA-2D",
    "M-DATA-1D",
    "M-CONV-2D",
    "M-DOWN",
    "M-DOWN-MASK",
    "M-MASK",
    "M-BORDER",
    "M-MEAN",
    "M-MOM",
    "M-STD",
    "M-W1",
    "M-S1",
    "M-S2",
    "M-S3",
    "M-S4",
    "M-CHAN",
    "M-ORDER",
    "M-NORM",
    "M-RED-ISO",
    "M-RED-ANG",
    "M-RED-SCALE",
    "M-FLAT",
)

# Channel 1 has cross statistics but no auto statistics (OP-30).
CROSS_MATRIX = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
CONFIGS: dict[str, dict[str, Any]] = {
    "single": {"nb": 1, "nc": 1},
    "cross": {"nb": 2, "nc": 2},
    "cross_matrix": {"nb": 2, "nc": 3, "cross_matrix": CROSS_MATRIX, "ref_batch_mean": True},
    "standardize": {"nb": 2, "nc": 2, "standardize": True},
}


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec parity is a float64 comparison (Part I, §5.1)."""
    monkeypatch.setattr(_config, "_dtype", "float64")


def _mask(ndim: int) -> NDArray[np.bool_]:
    """Missing values inside the domain and on its edge, where the padding mode matters."""
    n = SIZES[ndim]
    mask = np.zeros((n,) * ndim, dtype=bool)
    if ndim == 2:
        mask[5:7, 20:23] = True
        mask[n - 1, 0] = True
    else:
        mask[9:12] = True
        mask[n - 2] = True
    return mask


def _fields(ndim: int, nb: int, nc: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two seeded fields with a non-zero mean; channels are correlated, so that the cross
    statistics are not noise."""
    rng = np.random.default_rng(2618)
    shape = (nb, nc, *(SIZES[ndim],) * ndim)
    common = rng.normal(size=(2, nb, 1, *shape[2:]))
    fields = rng.normal(size=(2, *shape)) + 0.5 * common + 0.3
    return fields[0], fields[1]


def _compare(actual: STStatistics, expected: dict[str, Any], scale: float) -> None:
    for family in FAMILIES:
        assert_parity(
            np.asarray(actual.full_grid(family)),
            expected[family],
            scale=scale if family == "mean" else None,
        )


def _check_case(
    ndim: int,
    J: int,  # noqa: N803 (maths.md notation)
    L: int,  # noqa: N803
    pbc: bool,
    masked: bool,
    reductions: list[dict[str, Any]],
    nb: int = 2,
    nc: int = 2,
    cross_matrix: NDArray[np.bool_] | None = None,
    ref_batch_mean: bool = False,
    standardize: bool = False,
    through_apply: bool = True,
) -> None:
    """ESTEL against the oracle on one configuration: without normalisation, ``norm="self"``,
    ``norm="ref"`` on a second field, each chain of ``reductions``, and the vectors.

    The oracle runs on the target without normalisation and on the second field with
    ``norm="ref"``; the ``norm="self"`` statistics are derived from the first result. With
    ``through_apply`` each chain of reductions is one more run of ``spec_apply``; without it,
    the chain is derived from the first result too, which keeps the fast cases short."""
    target, other = _fields(ndim, nb, nc)
    mask = _mask(ndim) if masked else None
    cls = GEOMETRIES[ndim]
    data, second = cls(target, mask=mask, pbc=pbc), cls(other, mask=mask, pbc=pbc)
    options: dict[str, Any] = {"cross_matrix": cross_matrix, "standardize": standardize}
    estel_options = {**options, "compute_PS": False if ndim == 2 else None}

    def oracle(array: NDArray[np.float64], **more: Any) -> dict[str, Any]:
        return spec.spec_apply(array, ndim, pbc, mask, J, L, **options, **more)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30: tested elsewhere
        operator = STOperator(data, J, L, **estel_options)
        raw = operator.apply(data, norm="none")
        stored = operator.apply(data, norm="self", ref_batch_mean=ref_batch_mean)
        loaded = operator.apply(second, norm="ref")

    expected = oracle(target, norm="none", flatten=True)
    scale = float(np.sqrt(np.max(expected["var"])))
    _compare(raw, expected, scale)
    assert_parity(np.asarray(raw.to_vector()), expected["vector"])
    # mask_st (M-FLAT): a fixed boolean array over the vector, every third coefficient kept.
    mask_st = np.arange(expected["vector"].size) % 3 == 0
    assert_parity(
        np.asarray(raw.to_vector(mask_st)),
        common.spec_flatten(expected, flatten_complex=True, mask_st=mask_st),
    )
    if standardize:
        assert raw._standardization is not None
        for key, removed in zip(("mean_pre_std", "std_pre_std"), raw._standardization, strict=True):
            assert_parity(np.asarray(removed), expected[key])

    # norm="self": the reference, and the statistics divided by it (M-NORM).
    reference = common.spec_reference(expected, ref_batch_mean)
    assert stored._reference is not None
    assert_parity(np.asarray(stored._reference.var), reference["var"])
    assert_parity(np.asarray(stored._reference.s2_sqrt), reference["sigma"])
    normalized = common.spec_normalize(expected, reference)
    _compare(stored, normalized, scale)
    # norm="ref": a second field, normalised by the reference of the first.
    _compare(loaded, oracle(other, norm="ref", ref=reference), scale)

    # The reductions and the vector of the reduced statistics (M-FLAT): through the whole of
    # apply (M-ORDER), or derived from the first oracle result, function by function.
    for chain in reductions:
        reduced = stored.reduce(**chain)
        if through_apply:
            expected = oracle(
                target, norm="self", ref_batch_mean=ref_batch_mean, flatten=True, **chain
            )
        else:
            expected = _reduced(normalized, chain)
        _compare(reduced, expected, scale)
        assert_parity(np.asarray(reduced.to_vector()), expected["vector"])


def _reduced(normalized: dict[str, Any], chain: dict[str, Any]) -> dict[str, Any]:
    """The oracle's reductions of ``chain`` applied to normalised statistics, in the order of
    M-ORDER, and the vector of the result."""
    out = normalized
    iso = bool(chain.get("iso"))
    if iso:
        out = common.spec_iso(out)
    if chain.get("angular_fourier"):
        out = common.spec_angular_fourier(out, chain.get("angular_harmonics"), after_iso=iso)
    if chain.get("scale_cosine"):
        out = common.spec_scale_cosine(out, chain.get("scale_harmonics"), chain.get("dj"))
    return {**out, "vector": common.spec_flatten(out, flatten_complex=True)}


@pytest.mark.parametrize("pbc", [True, False], ids=["periodic", "open"])
@pytest.mark.parametrize("ndim", [2, 1], ids=["2d", "1d"])
def test_small_case_equals_the_oracle(ndim: int, pbc: bool) -> None:
    """One small case per geometry and periodicity, with a mask, two channels and J = 2: the
    statistics, both normalisations, and the three reductions in a row, truncated. Two runs of
    the oracle; the reductions are derived from the first, function by function."""
    L = 4 if ndim == 2 else 2  # noqa: N806
    chain = {
        "iso": True,
        "angular_fourier": True,
        "angular_harmonics": L - 1,
        "scale_cosine": True,
        "scale_harmonics": 1,
        "dj": 2,
    }
    _check_case(ndim, 2, L, pbc, masked=True, reductions=[chain], through_apply=False)


def _chains(J: int, L: int) -> list[dict[str, Any]]:  # noqa: N803
    """Every chain of reductions ESTEL allows, with and without truncation."""
    angular = {"angular_fourier": True}
    scale = {"scale_cosine": True}
    return [
        {"iso": True},
        angular,
        {"iso": True, **angular},
        {"iso": True, **scale},
        {
            "iso": True,
            **angular,
            "angular_harmonics": max(1, L - 1),
            **scale,
            "scale_harmonics": J - 1,
            "dj": J - 1,
        },
    ]


@pytest.mark.slow
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("masked", [False, True], ids=["full", "masked"])
@pytest.mark.parametrize("pbc", [True, False], ids=["periodic", "open"])
@pytest.mark.parametrize(("ndim", "L"), [(2, 4), (2, 3), (1, 1), (1, 2)])
def test_matrix_equals_the_oracle(ndim: int, L: int, pbc: bool, masked: bool, config: str) -> None:  # noqa: N803
    """The matrix of Part I, §5.1, at J = 3: each wavelet family (2D with L = 4 and L = 3, 1D
    with L = 1 and L = 2), with and without mask, periodic and not, for one channel, two
    channels with every pair, three channels with a custom ``cross_matrix`` and a batch-averaged
    reference, and standardised data; every normalisation mode and every chain of reductions.

    Slow: 64 cases of 0.1 to 3 s each, where the oracle loops over the coefficients of J = 3.
    """
    settings = {k: v for k, v in CONFIGS[config].items()}
    _check_case(ndim, 3, L, pbc, masked, _chains(3, L), **settings)


@pytest.mark.maths("M-DEF", "M-DEF-1D")
@pytest.mark.parametrize("ndim", [2, 1], ids=["2d", "1d"])
def test_defaults_equal_the_oracle(ndim: int) -> None:
    target, _ = _fields(ndim, 1, 2)
    data = GEOMETRIES[ndim](target, pbc=True)
    operator = STOperator(data, compute_PS=False if ndim == 2 else None)
    plan = operator.plan
    assert isinstance(plan, PlanarPlan)
    defaults = spec.spec_defaults(target.shape[2:])
    assert defaults["cross_matrix"] == "upper triangle"
    found = {
        "J": operator.J,
        "L": operator.L,
        "K": plan.kernel_size,
        "nan_threshold": plan.nan_threshold,
        # The normalisation is chosen at apply: its defaults are those of M-DEF.
        "norm": APPLY.parameters["norm"].default,
        "ref_batch_mean": APPLY.parameters["ref_batch_mean"].default,
        "standardize": operator.standardize,
        "cross_matrix": "upper triangle",
    }
    assert found == defaults
    assert np.array_equal(plan.cross_matrix, np.triu(np.ones((2, 2), dtype=bool)))


@pytest.mark.parametrize("ndim", [2, 1], ids=["2d", "1d"])
def test_data_are_promoted_as_in_the_oracle(ndim: int) -> None:
    """M-DATA-2D, M-DATA-1D: the promotion to ``(Nb, Nc, *spatial)`` (OP-29) and the mask read
    from the NaN of a single map (OP-26)."""
    cls, n = GEOMETRIES[ndim], SIZES[ndim]
    rng = np.random.default_rng(0)
    for lead in ((), (3,), (2, 3)):
        array = rng.normal(size=(*lead, *(n,) * ndim))
        promoted, mask = spec.spec_data(array, ndim, 2, pbc=True)
        data = cls(array, pbc=True)
        assert data.shape == promoted.shape
        assert np.array_equal(np.asarray(data.array), promoted)
        assert data.mask is None
        assert mask is None
    holes = rng.normal(size=(n,) * ndim)
    holes[(3,) * ndim] = np.nan
    _, mask = spec.spec_data(holes, ndim, 2, pbc=False)
    found = cls(holes, pbc=False).mask
    assert found is not None
    assert mask is not None
    assert np.array_equal(found, mask)
