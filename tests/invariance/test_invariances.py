"""Invariances of the scattering statistics (Part I, §5.1; task T2.8, issue #20; maths.md M-MOM,
M-S1 to M-S4, M-NORM, M-CHAN, M-WAV-1D).

What the statistics of 1D and 2D planar data must satisfy whatever the field, with ``pbc`` on and
off, through ``STOperator.apply`` and through ``estel.functional.scattering``:

- **scaling**, x → a·x with a > 0 and a < 0: mean ∝ a; var, S2 and S4 ∝ a²; the auto S1 ∝ |a|;
  S3 ∝ a·|a|, since its first argument goes through a modulus and its second does not (M-S3).
  With ``norm="self"`` the divisors of M-NORM follow, so var, the auto S1, S2 and S4 are
  unchanged and S3 is multiplied by the sign of a; mean, which is not normalised, stays ∝ a;
- **constant shift**, x → x + constant: S1 to S4 unchanged, and every family with
  ``standardize``;
- **time reversal** in 1D with L = 2: the statistics of the reversed field are those of the field
  with the two orientations swapped (ψ₁ = conj(ψ₀), M-WAV-1D);
- **batch independence**: the statistics of a batch are those of each of its maps alone;
- **channel permutation**: permuting the channels, with ``cross_matrix`` permuted consistently,
  permutes the statistics; a pair of S1 or S2 that moves below the diagonal is stored as its
  conjugate (M-CHAN);
- **all-valid mask**: a mask without missing values gives the statistics of no mask.

Left out, where STL's definitions make the property approximate (Part I, §5.1): the scaling of
the cross S1, whose ε breaks it by about 1e-8 (OP-16), and the constant shift of 2D data with a
mask, where the 2D smoothing kernel, which sums to 1.00000004, leaves about 1e-9 per unit of the
constant (OP-59). Rotations come with the joint design of the filters (Part III, §20.10; OP-58),
translation is not tested in 1.0.0 (LF25), and the tensor-product invariances come with P3.

Everything else holds to rounding (relative errors from 1e-17 to 1e-14, measured), so the two
sides are compared in float64 with ``assert_parity`` and the float64 tolerances of Part II, §14.3
(rtol 1e-10, atol = 0.1 x rtol x max|expected|; for ``mean``, whose theoretical value may be
zero, atol = 0.1 x rtol x the standard deviation of the data). The helper's messages name STL as
the expected side: here it is the transformed statistics of the original field.

The fast tier runs one case per property, at J = 2 (``_cases``).

Slow: the matrix of each property, 6 to 16 configurations (geometry, ``pbc``, mask, operator or
function) at J = 3: 132 cases, 9 s of CPU time on PyTorch and 90 s on JAX, which the fast tier
has no room for (issue #20, decision 4).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data, Data1D, Data2D, STOperator
from tests.parity._compare import assert_parity

Statistics = dict[str, NDArray[Any]]
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
SCATTERING = ("S1", "S2", "S3", "S4")
ORIENTATION_AXES = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}
"""The number of orientation axes of each family: its last axes (Part II, §12.7)."""
GEOMETRIES: dict[int, type[Data]] = {1: Data1D, 2: Data2D}
SIZES = {1: 64, 2: 32}
"""Pixels per axis: the smallest sizes at which the masked, non-periodic cases keep valid pixels
in every mean at J = 2, and in most of them at J = 3."""
ORIENTATIONS = {1: 2, 2: 4}
# Channel 1 has cross statistics but no auto statistics (OP-30).
CROSS_MATRIX = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
PERMUTATION = (2, 0, 1)
"""New channel i is old channel ``PERMUTATION[i]``: pairs move across the diagonal."""
pytestmark = pytest.mark.maths(
    "M-MOM", "M-STD", "M-S1", "M-S2", "M-S3", "M-S4", "M-NORM", "M-CHAN", "M-WAV-1D", "M-MASK"
)


@dataclass(frozen=True)
class Case:
    """One configuration: the geometry, the periodicity, whether values are missing, and whether
    the statistics come from ``STOperator.apply`` or from ``estel.functional.scattering``."""

    ndim: int
    pbc: bool
    masked: bool
    via: str
    J: int = 3

    @property
    def name(self) -> str:
        parts = (
            f"{self.ndim}d",
            "periodic" if self.pbc else "open",
            "masked" if self.masked else "full",
            self.via,
            f"J{self.J}",
        )
        return "-".join(parts)


MATRIX = tuple(
    Case(ndim, pbc, masked, via)
    for ndim, pbc, masked, via in product(
        (1, 2), (True, False), (False, True), ("operator", "functional")
    )
)


def _cases(fast: Case, keep: Callable[[Case], bool] = lambda case: True) -> list[Any]:
    """The cases of a property: ``fast``, at J = 2, in the fast tier, and the configurations of
    ``MATRIX`` that ``keep`` retains, at J = 3, in the full tier."""
    assert fast.J == 2, fast
    assert keep(fast), fast
    slow = [
        pytest.param(case, marks=pytest.mark.slow, id=case.name) for case in MATRIX if keep(case)
    ]
    return [pytest.param(fast, id=fast.name), *slow]


def _fields(case: Case, nb: int = 1, nc: int = 2) -> NDArray[np.float64]:
    """Seeded fields with a non-zero mean; the channels are correlated, so that the cross
    statistics are not noise."""
    rng = np.random.default_rng(2028)
    spatial = (SIZES[case.ndim],) * case.ndim
    common = rng.normal(size=(nb, 1, *spatial))
    fields: NDArray[np.float64] = rng.normal(size=(nb, nc, *spatial)) + 0.5 * common + 0.3
    return fields


def _mask(case: Case) -> NDArray[np.bool_] | None:
    """The missing values of a masked case, inside the domain and on its edge, where the padding
    mode matters; not symmetric under reversal."""
    if not case.masked:
        return None
    n = SIZES[case.ndim]
    mask = np.zeros((n,) * case.ndim, dtype=bool)
    if case.ndim == 2:
        mask[5:7, 20:23] = True
        mask[n - 1, 0] = True
    else:
        mask[9:12] = True
        mask[n - 2] = True
    return mask


def _statistics(
    case: Case, array: NDArray[np.float64], mask: NDArray[np.bool_] | None, **options: Any
) -> Statistics:
    """The full grid of every family for ``array``, as NumPy, without normalisation unless
    ``options`` say otherwise."""
    data = GEOMETRIES[case.ndim](array, mask=mask, pbc=case.pbc)
    settings: dict[str, Any] = {"norm": "none", **options}
    if case.ndim == 2:
        settings["compute_PS"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30: tested elsewhere
        operator = STOperator(data, case.J, ORIENTATIONS[case.ndim], **settings)
    if case.via == "operator":
        values = operator.apply(data)
    else:
        values = estel.functional.scattering(data, operator.plan, operator.filters)
    return {family: np.asarray(values.full_grid(family)) for family in FAMILIES}


def _assert_equal(
    actual: Statistics, expected: Statistics, scale: float, families: tuple[str, ...] = FAMILIES
) -> None:
    """``actual`` equals ``expected`` for ``families``, within the float64 tolerances of Part II,
    §14.3; ``scale`` is the standard deviation of the data, the scale of ``mean``."""
    for family in families:
        assert np.isfinite(expected[family]).any(), family
        assert_parity(actual[family], expected[family], scale=scale if family == "mean" else None)


def _auto(s1: NDArray[Any]) -> NDArray[Any]:
    """The auto terms of S1, ``(Nb, Nc, J, L)``."""
    return np.stack([s1[:, c, c] for c in range(s1.shape[1])], axis=1)


@pytest.mark.parametrize("a", [2.5, -1.7], ids=["positive", "negative"])
@pytest.mark.parametrize("case", _cases(Case(1, False, True, "operator", J=2)))
def test_scaling(case: Case, a: float) -> None:
    """x → a·x. Without normalisation: mean ∝ a; var, S2 and S4 ∝ a²; the auto S1 ∝ |a|;
    S3 ∝ a·|a|. With ``norm="self"``: var, the auto S1, S2 and S4 unchanged, S3 multiplied by
    the sign of a, and mean, which M-NORM does not normalise, ∝ a. The cross S1 is left out
    (OP-16)."""
    fields, mask = _fields(case), _mask(case)
    scale = abs(a) * float(np.std(fields))
    powers = {"mean": a, "var": a * a, "S1": abs(a), "S2": a * a, "S3": a * abs(a), "S4": a * a}
    normalized = {"mean": a, "var": 1.0, "S1": 1.0, "S2": 1.0, "S3": float(np.sign(a)), "S4": 1.0}
    for norm, factors in (("none", powers), ("self", normalized)):
        base = _statistics(case, fields, mask, norm=norm)
        scaled = _statistics(case, a * fields, mask, norm=norm)
        expected = {family: factors[family] * base[family] for family in FAMILIES}
        scaled["S1"], expected["S1"] = _auto(scaled["S1"]), _auto(expected["S1"])
        _assert_equal(scaled, expected, scale)


@pytest.mark.parametrize(
    "case",
    _cases(Case(2, True, False, "functional", J=2), lambda c: not (c.ndim == 2 and c.masked)),
)
def test_constant_shift(case: Case) -> None:
    """x → x + constant: S1 to S4 are unchanged, since the wavelets have a zero mean; with
    ``standardize`` every family is, since the mean is removed first. In 2D without a mask only
    (OP-59)."""
    fields, mask = _fields(case), _mask(case)
    shifted = fields + 3.0
    _assert_equal(
        _statistics(case, shifted, mask), _statistics(case, fields, mask), 1.0, SCATTERING
    )
    # Standardised data have a unit standard deviation: the scale of their mean, zero in theory.
    _assert_equal(
        _statistics(case, shifted, mask, standardize=True),
        _statistics(case, fields, mask, standardize=True),
        1.0,
    )


@pytest.mark.parametrize(
    "case", _cases(Case(1, False, True, "functional", J=2), lambda c: c.ndim == 1)
)
def test_time_reversal_swaps_the_orientations(case: Case) -> None:
    """In 1D with L = 2, ψ₁ = conj(ψ₀) is ψ₀ reversed (M-WAV-1D): the statistics of the reversed
    field, with its mask reversed, are those of the field with the two orientations swapped
    along every orientation axis."""
    fields, mask = _fields(case), _mask(case)
    base = _statistics(case, fields, mask)
    reversed_mask = None if mask is None else mask[::-1].copy()
    found = _statistics(case, fields[..., ::-1].copy(), reversed_mask)
    expected = dict(base)
    for family, n_axes in ORIENTATION_AXES.items():
        expected[family] = np.flip(base[family], axis=tuple(range(-n_axes, 0)))
        assert not np.array_equal(expected[family], base[family], equal_nan=True), family
    _assert_equal(found, expected, float(np.std(fields)))


@pytest.mark.parametrize(
    "options", [{}, {"norm": "self", "standardize": True}], ids=["raw", "self"]
)
@pytest.mark.parametrize("case", _cases(Case(2, False, True, "operator", J=2)))
def test_batch_independence(case: Case, options: dict[str, Any]) -> None:
    """The statistics of a batch are those of each of its maps computed alone: without
    normalisation, and with ``standardize`` and ``norm="self"``, whose moments and reference are
    those of each map (M-STD; M-NORM without ``ref_batch_mean``)."""
    fields, mask = _fields(case, nb=2), _mask(case)
    together = _statistics(case, fields, mask, **options)
    alone = [_statistics(case, fields[b : b + 1], mask, **options) for b in range(2)]
    expected = {family: np.concatenate([one[family] for one in alone]) for family in FAMILIES}
    _assert_equal(together, expected, float(np.std(fields)))


def _permuted_matrix(matrix: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """``cross_matrix`` for the permuted channels: the same pairs, each written above the
    diagonal again."""
    upper = np.triu(matrix)
    symmetric = upper | upper.T
    permuted: NDArray[np.bool_] = np.triu(symmetric[np.ix_(PERMUTATION, PERMUTATION)])
    return permuted


def _permuted_statistics(base: Statistics) -> Statistics:
    """The statistics expected for the permuted channels. S3 and S4 store both orders of a pair;
    S1 and S2 store c₁ ≤ c₂ only, the other order being the conjugate (M-CHAN)."""
    p = list(PERMUTATION)
    expected = {family: base[family][:, p] for family in ("mean", "var")}
    for family in ("S3", "S4"):
        expected[family] = base[family][:, p][:, :, p]
    lower = np.tril(np.ones((len(p), len(p)), dtype=bool), -1)
    for family in ("S1", "S2"):
        stored = base[family]
        hermitian = np.where(lower[:, :, None, None], np.conj(np.swapaxes(stored, 1, 2)), stored)
        moved = hermitian[:, p][:, :, p]
        expected[family] = np.where(lower[:, :, None, None], np.nan, moved)
    return expected


@pytest.mark.parametrize(
    ("options", "custom"),
    [({}, True), ({"norm": "self"}, False)],
    ids=["raw-cross_matrix", "self-every_pair"],
)
@pytest.mark.parametrize("case", _cases(Case(1, True, False, "operator", J=2)))
def test_channel_permutation(case: Case, options: dict[str, Any], custom: bool) -> None:
    """Permuting three channels permutes the statistics, with ``cross_matrix`` permuted
    consistently: a custom matrix without normalisation, and every pair with ``norm="self"``."""
    fields, mask = _fields(case, nc=3), _mask(case)
    matrix = CROSS_MATRIX if custom else None
    base = _statistics(case, fields, mask, cross_matrix=matrix, **options)
    found = _statistics(
        case,
        fields[:, list(PERMUTATION)],
        mask,
        cross_matrix=_permuted_matrix(CROSS_MATRIX) if custom else None,
        **options,
    )
    _assert_equal(found, _permuted_statistics(base), float(np.std(fields)))


@pytest.mark.parametrize("options", [{}, {"standardize": True}], ids=["raw", "standardize"])
@pytest.mark.parametrize(
    "case", _cases(Case(2, False, False, "functional", J=2), lambda c: not c.masked)
)
def test_all_valid_mask_equals_no_mask(case: Case, options: dict[str, Any]) -> None:
    """A mask without missing values takes the masked path (M-DOWN-MASK, M-MEAN) and gives the
    statistics of no mask."""
    fields = _fields(case)
    all_valid = np.zeros(fields.shape[2:], dtype=bool)
    _assert_equal(
        _statistics(case, fields, all_valid, **options),
        _statistics(case, fields, None, **options),
        float(np.std(fields)),
    )
