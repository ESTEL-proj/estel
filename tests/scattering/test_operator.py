"""The internal ``STOperator`` and its loop (task T2.4, issue #16; Part II, §12.6; Part I,
§3.1.1; maths.md M-S1 to M-S4, M-CHAN, M-MEAN, OP-3, OP-27, OP-30).

Values are compared with STL in ``tests/parity/test_scattering.py``, in float64. These tests run
on the run's backend and dtype, and check what parity does not: the full-grid shapes and dtypes
and the NaN of absent coefficients, that missing values are never read, that the computation is
pure and batch elements independent, that gradients are finite to second order with missing
values, and the errors of the constructor and of ``apply``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data1D, Data2D, DataClass, _config
from estel.backend._protocol import Backend, DType
from estel.scattering._loop import ScatteringValues, channel_pairs
from estel.scattering._operator import STOperator

pytestmark = pytest.mark.maths("M-S1", "M-S2", "M-S3", "M-S4", "M-CHAN", "M-MEAN")
FAMILIES = ("S1", "S2", "S3", "S4")


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    field: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return field


def _hole(n: int, ndim: int) -> NDArray[np.bool_]:
    mask = np.zeros((n,) * ndim, dtype=bool)
    mask[(slice(n // 4, n // 4 + 3),) * ndim] = True
    return mask


def _operator(data: DataClass, J: int | None, **kwargs: Any) -> STOperator:  # noqa: N803
    options: dict[str, Any] = {"norm": "vanilla", "compute_PS": False if data.ndim == 2 else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13: tested with the plan
        return STOperator(data, J, **{**options, **kwargs})


def _numpy(values: ScatteringValues) -> dict[str, NDArray[Any]]:
    return {k: np.asarray(v) for k, v in values._asdict().items() if v is not None}


# ---------------------------------------------------------------------- shapes, dtypes, NaN


@pytest.mark.parametrize(("cls", "n", "L"), [(Data2D, 16, 4), (Data1D, 32, 1), (Data1D, 32, 2)])
def test_full_grids_have_stls_shapes_dtypes_and_nan(
    cls: type[DataClass],
    n: int,
    L: int,  # noqa: N803 (maths.md notation)
    dtype: str,
) -> None:
    J, nb, nc = 3, 2, 2  # noqa: N806
    data = cls(_field((nb, nc) + (n,) * cls.ndim), pbc=True)
    values = _numpy(_operator(data, J, L=L).apply(data))
    complex_dtype = {"float32": np.complex64, "float64": np.complex128}[dtype]
    assert values["mean"].shape == values["var"].shape == (nb, nc)
    assert values["mean"].dtype == values["var"].dtype == np.dtype(dtype)
    shapes = {
        "S1": (nb, nc, nc, J, L),
        "S2": (nb, nc, nc, J, L),
        "S3": (nb, nc, nc, J, J, L, L),
        "S4": (nb, nc, nc, J, J, J, L, L, L),
    }
    for family, shape in shapes.items():
        assert values[family].shape == shape, family
        assert values[family].dtype == complex_dtype, family
    # M-CHAN: (c2, c1) of S1 and S2 is NaN, both orders of S3 and S4 are computed.
    for family in ("S1", "S2"):
        assert np.isnan(values[family][:, 1, 0]).all()
        assert np.isfinite(values[family][:, [0, 0, 1], [0, 1, 1]]).all()
        # The auto terms are real, stored complex.
        np.testing.assert_array_equal(values[family][:, [0, 1], [0, 1]].imag, 0.0)
    for j2 in range(J):
        for j3 in range(J):
            s3 = values["S3"][:, :, :, j2, j3]
            assert np.isfinite(s3).all() if j2 <= j3 else np.isnan(s3).all()
    for index in np.ndindex(J, J, J):
        j1, j2, j3 = index
        s4 = values["S4"][:, :, :, j1, j2, j3]
        assert np.isfinite(s4).all() if j1 <= j2 <= j3 else np.isnan(s4).all()


def test_empty_means_are_nan_and_the_rest_finite() -> None:
    """The means with no valid pixel (OP-3) are NaN in the full grid, without being computed;
    every other coefficient within the index sets is finite."""
    n = 64
    mask = np.zeros((n, n), dtype=bool)
    mask[30:33, 30:34] = True
    data = Data2D(_field((n, n)), mask=mask, pbc=False)
    operator = _operator(data, 4)
    plan = operator.plan
    assert plan.empty_s3_pairs
    assert plan.empty_s4_triples
    values = _numpy(operator.apply(data))
    for j2, j3 in plan.s3_pairs:
        s3 = values["S3"][0, 0, 0, j2, j3]
        assert np.isnan(s3).all() if (j2, j3) in plan.empty_s3_pairs else np.isfinite(s3).all()
    for t in plan.s4_triples:
        s4 = values["S4"][(0, 0, 0, *t)]
        assert np.isnan(s4).all() if t in plan.empty_s4_triples else np.isfinite(s4).all()


def test_custom_cross_matrix(dtype: str) -> None:
    """A channel without auto statistics but in a cross pair (OP-30); entries below the
    diagonal are ignored, as in STL."""
    matrix = np.array([[True, False, True], [True, False, True], [True, True, True]])
    data = Data2D(_field((1, 3, 16, 16)), pbc=True)
    values = _numpy(_operator(data, 2, compute_cross_matrix=matrix).apply(data))
    computed_s1 = ~np.isnan(values["S1"][0, :, :, 0, 0])
    np.testing.assert_array_equal(
        computed_s1, np.triu(matrix) & ~np.eye(3, dtype=bool) | np.diag([1, 0, 1]).astype(bool)
    )
    computed_s3 = ~np.isnan(values["S3"][0, :, :, 0, 0, 0, 0])
    upper = np.triu(matrix)
    np.testing.assert_array_equal(computed_s3, upper | upper.T)


def test_channel_pairs_follow_stls_order() -> None:
    pairs = channel_pairs(np.array([[True, True, False], [False, False, True], [True, True, True]]))
    assert pairs.auto == (0, 2)
    assert pairs.cross == ((0, 1), (1, 2))
    assert pairs.ordered == ((0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2))


# ---------------------------------------------------------------------- missing values, purity


@pytest.mark.parametrize(("cls", "n"), [(Data2D, 32), (Data1D, 64)])
@pytest.mark.parametrize("pbc", [True, False])
def test_missing_values_are_never_read(cls: type[DataClass], n: int, pbc: bool) -> None:
    """NaN, 0 or a huge value at the missing pixels give the same statistics (OP-27)."""
    mask = _hole(n, cls.ndim)
    field = _field((2, 2) + (n,) * cls.ndim)
    operator = _operator(cls(field, mask=mask, pbc=pbc), 3, standardize=True)
    results = [
        _numpy(operator.apply(cls(np.where(mask, fill, field), mask=mask, pbc=pbc)))
        for fill in (np.nan, 0.0, 1e30)
    ]
    for other in results[1:]:
        for key, value in results[0].items():
            np.testing.assert_array_equal(other[key], value, err_msg=key)
    assert all(
        np.isfinite(results[0][k]).all() for k in ("mean", "var", "mean_pre_std", "std_pre_std")
    )


def test_apply_is_pure_and_batch_elements_are_independent(backend: Backend, dtype: str) -> None:
    field = _field((3, 2, 16, 16))
    data = Data2D(field, pbc=False)
    before = backend.to_numpy(data.array).copy()
    operator = _operator(data, 3)
    first = _numpy(operator.apply(data))
    np.testing.assert_array_equal(backend.to_numpy(data.array), before)
    again = _numpy(operator.apply(data))
    single = _numpy(operator.apply(data[1]))
    rtol = 1e-5 if dtype == "float32" else 1e-12
    for key, value in first.items():
        np.testing.assert_array_equal(again[key], value, err_msg=key)
        np.testing.assert_allclose(single[key][0], value[1], rtol=rtol, atol=rtol, err_msg=key)


def test_standardize_returns_the_moments_removed(dtype: str) -> None:
    field = 5.0 + 2.0 * _field((2, 1, 32, 32))
    data = Data2D(field, pbc=True)
    values = _numpy(_operator(data, 2, standardize=True).apply(data))
    rtol = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(values["mean_pre_std"], field.mean(axis=(-2, -1)), rtol=rtol)
    np.testing.assert_allclose(values["std_pre_std"], field.std(axis=(-2, -1)), rtol=rtol)
    np.testing.assert_allclose(values["var"], 1.0, rtol=10 * rtol)
    np.testing.assert_allclose(values["mean"], 0.0, atol=10 * rtol)
    assert "mean_pre_std" not in _numpy(_operator(data, 2).apply(data))


# ---------------------------------------------------------------------- differentiation


@pytest.mark.parametrize(("cls", "n"), [(Data2D, 16), (Data1D, 32)])
def test_gradients_are_finite_with_missing_values(
    cls: type[DataClass], n: int, backend: Backend, dtype: DType
) -> None:
    """First and second order through the whole loop, with exact zeros (the missing pixels,
    filled with 0) and NaN in the input there (Part I, §3.2.3): finite, and zero at the missing
    pixels. Gradient values are checked in T2.7."""
    mask = _hole(n, cls.ndim)
    field = np.where(mask, np.nan, _field((1, 1) + (n,) * cls.ndim))
    data = cls(field, mask=mask, pbc=False)
    operator = _operator(data, 2)
    xp = backend.xp

    def loss(array: Any) -> Any:
        values = operator.apply(data.with_array(array))
        total = xp.sum(values.var)
        for family in FAMILIES:
            pairs = backend.complex_to_real_pairs(getattr(values, family))
            total = total + xp.sum(xp.where(xp.isnan(pairs), xp.zeros_like(pairs), pairs) ** 2)
        return total

    _, grad = backend.value_and_grad(loss)(data.array)
    grad = backend.to_numpy(grad)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[..., mask], 0.0)
    direction = backend.asarray(np.where(mask, 0.0, 1.0) * np.ones(data.shape), dtype)
    hessian_v = backend.to_numpy(backend.hvp(loss, data.array, direction))
    assert np.isfinite(hessian_v).all()


# ---------------------------------------------------------------------- errors


def test_norms_other_than_vanilla_come_with_t25() -> None:
    data = Data2D(_field((16, 16)), pbc=True)
    for norm in ("store_ref", "load_ref"):
        with pytest.raises(NotImplementedError, match=r"T2.5"):
            STOperator(data, 2, norm=norm, compute_PS=False)
    with pytest.raises(NotImplementedError, match=r"T2.5"):
        STOperator(data, 2, compute_PS=False)  # the default, store_ref
    with pytest.raises(ValueError, match="vanilla"):
        STOperator(data, 2, norm="self", compute_PS=False)
    operator = _operator(data, 2)
    with pytest.raises(ValueError, match="no reference"):
        operator.apply(data, ref=object())


def test_compute_ps() -> None:
    with pytest.raises(NotImplementedError, match=r"T4.1"):
        STOperator(Data2D(_field((16, 16)), pbc=True), 2, norm="vanilla")
    with pytest.raises(ValueError, match="LF23"):
        STOperator(Data1D(_field((32,)), pbc=True), 2, norm="vanilla", compute_PS=True)
    STOperator(Data1D(_field((32,)), pbc=True), 2, norm="vanilla", compute_PS=False)


def test_defaults_follow_m_def() -> None:
    operator = _operator(Data2D(_field((32, 64)), pbc=True), None)
    assert (operator.J, operator.L) == (3, 4)
    operator = _operator(Data1D(_field((64,)), pbc=True), None)
    assert (operator.J, operator.L) == (4, 1)
    with pytest.raises(ValueError, match="default J"):
        _operator(Data1D(_field((4,)), pbc=True), None)
    with pytest.raises(ValueError, match="L = 2"):
        _operator(Data1D(_field((32,)), pbc=True), 2, L=3)


def test_apply_refuses_other_data(dtype: str) -> None:
    data = Data2D(_field((2, 16, 16)), pbc=True)
    operator = _operator(data, 2)
    for other in (
        Data2D(_field((2, 16, 16)), pbc=False),
        Data2D(_field((2, 32, 32)), pbc=True),
        Data2D(_field((2, 16, 16)), mask=_hole(16, 2), pbc=True),
        Data1D(_field((2, 16)), pbc=True),
    ):
        with pytest.raises(ValueError, match="build a new STOperator"):
            operator.apply(other)
    with pytest.raises(TypeError, match="DataClass"):
        operator.apply(_field((2, 16, 16)))  # type: ignore[arg-type]
    other_dtype = "float64" if dtype == "float32" else "float32"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_config, "_dtype", other_dtype)
        mismatched = Data2D(_field((2, 16, 16)), pbc=True)
    with pytest.raises(TypeError, match="set_default_dtype"):
        operator.apply(mismatched)


def test_cross_matrix_errors() -> None:
    data = Data2D(_field((3, 16, 16)), pbc=True)
    with pytest.raises(ValueError, match="OP-30"):
        _operator(data, 2, compute_cross_matrix=np.diag([True, False, True]))
    with pytest.raises(TypeError, match="boolean"):
        _operator(data, 2, compute_cross_matrix=np.ones((3, 3)))
    with pytest.raises(ValueError, match="channel"):
        _operator(data, 2, compute_cross_matrix=np.ones((2, 2), dtype=bool))
    operator = _operator(data, 2, compute_cross_matrix=np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="channel"):
        operator.apply(Data2D(_field((2, 16, 16)), pbc=True))
    with pytest.raises(TypeError, match="standardize"):
        _operator(data, 2, standardize=1)
