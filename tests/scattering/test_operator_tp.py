"""``STOperatorTensorProd``: its construction, its errors, and what it shares with
``STOperator`` (task T3.2, issue #23; Part II, §12.6; maths.md M-DATA-TP, M-DEF-TP, M-LP,
M-IDX-TP, OP-8, OP-13, OP-34, OP-41, OP-82, OP-83).

The values are compared with STL in ``tests/parity/test_tensor_prod.py`` and with the oracle in
``tests/spec/test_tensor_prod_operator.py``; these tests check the operator itself: a tensor
product of one block with ``lowpass=None`` gives the statistics of the plain geometry (OP-82),
``estel.functional.scattering`` is ``apply``, ``from_statistics`` rebuilds the operator, a wrong
reference is refused before the loop, and the arguments are validated at construction with
errors that say what to do.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data1D, Data2D, DataTensorProd, STOperator, STOperatorTensorProd, _config
from estel.backend import _registry
from estel.functional import _scattering
from estel.statistics._layout import tensor_prod_gather, tensor_prod_layout
from tests.parity._compare import assert_parity

pytestmark = [
    pytest.mark.maths("M-DATA-TP", "M-DEF-TP", "M-LP", "M-IDX-TP"),
    # OP-13 on these small open blocks; tested in the last test.
    pytest.mark.filterwarnings("ignore:.*too small for a border crop:UserWarning"),
]


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return values


def _space_time(shape: tuple[int, ...] = (1, 1, 8, 8, 8), seed: int = 0) -> DataTensorProd:
    return DataTensorProd(_field(shape, seed), [Data2D, Data1D], pbc=[True, False])


@pytest.mark.parametrize(
    ("cls", "shape", "L"),
    [(Data2D, (2, 2, 16, 16), 4), (Data1D, (1, 2, 32), 2)],
    ids=["2D", "1D"],
)
def test_one_block_without_lowpass_is_the_plain_geometry(
    cls: type[estel.Data],
    shape: tuple[int, ...],
    L: int,  # noqa: N803
) -> None:
    """A tensor product of one block with ``lowpass=None`` gives the statistics of the plain
    geometry, family by family (maths.md OP-82, M-IDX-TP, "One block"), here open, with cross
    channels and ``norm="self"``; with ``lowpass="ST"``, S1 and S2 have one scale more."""
    field = _field(shape)
    plain = cls(field, pbc=False)
    product = DataTensorProd(field, [cls], pbc=[False])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13
        expected = STOperator(plain, 2, L, compute_PS=False if cls is Data2D else None)
        found = STOperatorTensorProd(product, [2], [L], lowpass=None)
        with_lowpass = STOperatorTensorProd(product, [2], [L])
    stats, reference = found.apply(product), expected.apply(plain)
    for family in ("mean", "var", "S1", "S2", "S3", "S4"):
        assert_parity(_numpy(stats.full_grid(family)), _numpy(reference.full_grid(family)))
    s1 = with_lowpass.apply(product).full_grid("S1")
    assert tuple(s1.shape)[3] == 3


def test_functional_scattering_is_apply() -> None:
    """``estel.functional.scattering`` takes the plan and the filters of a tensor product: one
    pure function for both geometries, equal to ``apply``."""
    data = _space_time((2, 2, 8, 8, 8))
    operator = STOperatorTensorProd(data, norm="self")
    stats = operator.apply(data)
    found = estel.functional.scattering(data, operator.plan, operator.filters)
    np.testing.assert_array_equal(_numpy(found.to_vector()), _numpy(stats.to_vector()))


def test_from_statistics_rebuilds_the_operator(tmp_path: Any) -> None:
    """``from_statistics`` gives the operator of statistics computed, saved and loaded, with
    the reference they were normalised with; ``STOperator.from_statistics`` points to it."""
    data = _space_time((1, 1, 8, 8, 8))
    stats = STOperatorTensorProd(data, [2, 1], norm="self").apply(data)
    path = tmp_path / "stats.npz"
    stats.save(path)
    loaded = estel.STStatistics.load(path)
    rebuilt = STOperatorTensorProd.from_statistics(loaded)
    assert rebuilt.plan == stats._plan
    assert (rebuilt.J, rebuilt.L) == ((2, 1), (4, 2))
    again = rebuilt.apply(data)
    np.testing.assert_array_equal(_numpy(again.to_vector()), _numpy(stats.to_vector()))
    with pytest.raises(ValueError, match=r"use STOperatorTensorProd.from_statistics"):
        STOperator.from_statistics(stats)
    planar = STOperator(Data2D(_field((1, 1, 16, 16)), pbc=True), 2, compute_PS=False)
    with pytest.raises(ValueError, match=r"use STOperator.from_statistics"):
        STOperatorTensorProd.from_statistics(planar.apply(Data2D(_field((1, 1, 16, 16)), pbc=True)))
    assert "STOperatorTensorProd(J=[2, 1], L=[4, 2], lowpass='ST'" in repr(rebuilt)


def test_a_wrong_reference_is_refused_before_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``norm="ref"``, a reference of other blocks, J or low-pass flags, or of planar
    statistics, is refused before the loop runs (issue #23, decisions, point 4)."""
    data = _space_time()
    operator = STOperatorTensorProd(data, [2, 1], norm="ref")
    other = STOperatorTensorProd(data, [1, 1], norm="self").apply(data)
    planar = STOperator(Data2D(_field((1, 1, 16, 16)), pbc=True), 2, compute_PS=False)
    planar_stats = planar.apply(Data2D(_field((1, 1, 16, 16)), pbc=True))

    def no_loop(name: str) -> Any:
        raise AssertionError("the loop ran before the reference was checked")

    monkeypatch.setattr(_scattering, "_tensor_prod_loop", no_loop)
    with pytest.raises(ValueError, match=r"J = \[1, 1\] \(the operator has \[2, 1\]\)"):
        operator.apply(data, ref=other)
    with pytest.raises(ValueError, match=r"comes from planar statistics"):
        operator.apply(data, ref=planar_stats)
    with pytest.raises(ValueError, match=r'norm="ref" needs a reference'):
        operator.apply(data)


def test_arguments_are_checked_at_construction() -> None:
    """The errors of maths.md §3.1 and §3.10, at construction, saying what to do: planar data,
    a mask or a NaN (OP-41), the spectrum (LF23), ``lowpass``, J and L per block (OP-83), the
    sizes (OP-8), and data that differ from the plan when applied."""
    data = _space_time()
    with pytest.raises(TypeError, match=r"for 1D and 2D data, use STOperator"):
        STOperatorTensorProd(Data2D(_field((1, 1, 16, 16)), pbc=True))
    with pytest.raises(NotImplementedError, match=r"LF8"):
        DataTensorProd(np.full((8, 8, 8), np.nan), [Data2D, Data1D], pbc=[True, True])
    with pytest.raises(ValueError, match=r"no power spectrum in ESTEL 1.0.0"):
        STOperatorTensorProd(data, compute_PS=True)
    with pytest.raises(ValueError, match=r'pass "ST"'):
        STOperatorTensorProd(data, lowpass="PS")
    with pytest.raises(ValueError, match=r'pass "ST"'):
        STOperatorTensorProd(data, lowpass=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"one J per block"):
        STOperatorTensorProd(data, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"pass one per block"):
        STOperatorTensorProd(data, [2])
    with pytest.raises(ValueError, match=r"divisible by 2\^\(J_1-1\) = 16"):
        STOperatorTensorProd(data, [1, 5])
    with pytest.raises(ValueError, match=r"L = 1 or L = 2"):
        STOperatorTensorProd(data, None, [4, 3])
    J = [2, None]  # noqa: N806
    assert STOperatorTensorProd(data, J).J == (2, 1)
    assert J == [2, None]  # the caller's list is not completed in place (OP-83)
    operator = STOperatorTensorProd(data)
    with pytest.raises(ValueError, match=r"build a new STOperatorTensorProd"):
        operator.apply(DataTensorProd(_field((1, 1, 8, 8, 8)), [Data2D, Data1D], pbc=[True, True]))
    with pytest.raises(ValueError, match=r"Build a new\s+STOperatorTensorProd"):
        operator.apply(_space_time((1, 2, 8, 8, 8)))


def test_warnings_of_the_lowpass_filter_and_of_the_borders() -> None:
    """A block whose sizes are not divisible by 2^J_a warns and has no low-pass filter
    (OP-34); a border too large for a block is reduced with one warning for the operator
    (OP-13)."""
    data = DataTensorProd(_field((1, 1, 8, 8, 12)), [Data2D, Data1D], pbc=[True, True])
    with pytest.warns(UserWarning, match=r"block 1 .* has no low-pass filter") as record:
        operator = STOperatorTensorProd(data, [2, 3])
    assert record[0].filename == __file__
    assert operator.plan.lowpass_blocks == (True, False)
    assert tuple(operator.apply(data).full_grid("S1").shape[3:]) == (3, 3, 4, 2)
    open_data = DataTensorProd(_field((1, 1, 8, 8, 8)), [Data2D, Data1D], pbc=[False, False])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        STOperatorTensorProd(open_data, [2, 2])
    reduced = [w for w in caught if "too small for a border crop" in str(w.message)]
    assert len(reduced) == 1
    # Both warnings point at the line that builds the operator.
    assert reduced[0].filename == __file__


def test_tables_of_the_loop_grow_with_the_chunks() -> None:
    """The index and denominator tables of the loop, in the plan (``TensorProdPlan.loop``), and
    of its gather (``tensor_prod_gather``)
    hold one entry per chunk of a segment, and the pack of S4 one row per kind of chunk, never
    one per coefficient (issue #23, decisions, point 4): here 2D + 1D + 1D, 16⁴, J = (2, 2, 2),
    269,508 coefficients. The crops are the only pixel arrays, of the shape of one block. The
    operator is built, not applied."""
    data = DataTensorProd(
        np.zeros((1, 1, 16, 16, 16, 16)), [Data2D, Data1D, Data1D], pbc=[True, False, True]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operator = STOperatorTensorProd(data, [2, 2, 2], norm="none")
    loop = operator.plan.loop
    n_chunks = sum(
        s.s3_rows.size + s.s4_rows.size + s.s3_denominators.size + s.s4_denominators.size
        for s in loop.second_layer
    )
    pack = tensor_prod_gather(operator.plan)
    tables = n_chunks + pack.within.size + pack.kinds.size + pack.starts.size
    n_floats = tensor_prod_layout(operator.plan).n_floats  # 531,652
    # 16,694 here: 310 chunks of S3 and S4 (rows and denominators), and the pack of S4, whose
    # table of the floats within a chunk is one row of L³ = 4,096 orientations per kind.
    assert tables < n_floats / 20
    for second in loop.second_layer:
        for a, crops in enumerate(second.arm_crops):
            if crops is not None:
                assert crops.shape[0] == len(second.arms)
                assert crops.ndim == 1 + len(operator.plan.wavelets.axes[a])
