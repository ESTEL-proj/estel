"""``STStatistics``: selection, flattening, reductions, normalisation references, storage and
pytrees (task T2.5, issue #17; Part II, §12.7; maths.md M-NORM, M-RED-ISO, M-RED-ANG,
M-RED-SCALE, M-FLAT, OP-3, OP-30, OP-37).

Values are compared with STL in ``tests/parity/test_statistics.py``, in float64. These tests run
on the run's backend and dtype, and check what parity does not: that the static layout predicts
every NaN, the semantics of ``select`` and ``to_vector(mask_st)``, the errors of ``reduce``,
the life of the reference in the operator, the gradients through it, the round trip through
``estel.io``, and the pytree registration.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data, Data1D, Data2D, STOperator, STStatistics, _config
from estel.backend._protocol import Backend, DType
from estel.io import _schema
from estel.scattering._plan import PlanarPlan
from tests._backends import backend_name

pytestmark = pytest.mark.maths("M-NORM", "M-RED-ISO", "M-RED-ANG", "M-RED-SCALE", "M-FLAT")
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
jax_only = pytest.mark.skipif(backend_name() != "jax", reason="names JAX's tree utilities")
torch_only = pytest.mark.skipif(backend_name() != "torch", reason="names PyTorch's pytrees")


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    field: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return field


def _operator(data: Data, J: int, **kwargs: Any) -> STOperator:  # noqa: N803
    options: dict[str, Any] = {"compute_PS": False if data.ndim == 2 else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30
        return STOperator(data, J, **{**options, **kwargs})


def _masked_2d(nb: int = 2, nc: int = 2, seed: int = 0) -> Data2D:
    """Open 2D data with a hole that leaves some means empty at J = 4 (OP-3)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:33, 30:34] = True
    return Data2D(_field((nb, nc, 64, 64), seed), mask=mask, pbc=False)


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(x)


def _tol(dtype: str) -> float:
    return 1e-5 if dtype == "float32" else 1e-12


def _check_layout(stats: STStatistics) -> None:
    """The static layout predicts every NaN of every family, on every batch element."""
    for grid in stats._grids:
        values = _numpy(stats.full_grid(grid.name))
        assert values.shape[1:] == grid.shape, grid.name
        expected = np.broadcast_to(~grid.present, values.shape)
        np.testing.assert_array_equal(np.isnan(values), expected, err_msg=grid.name)


# ---------------------------------------------------------------------- static layout


def test_layout_predicts_every_nan() -> None:
    """Structural NaN, empty means (OP-3), channels without auto statistics after normalisation
    (OP-30), missing scales of scale_cosine: all static, through every operation."""
    data = _masked_2d(nc=3)
    matrix = np.array([[True, False, True], [False, False, True], [False, False, True]])
    for norm in ("none", "self"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # OP-30, tested below
            stats = _operator(data, 4, cross_matrix=matrix).apply(data, norm=norm)
        assert isinstance(stats._plan, PlanarPlan)
        assert stats._plan.empty_s3_pairs
        for reduced in (
            stats,
            stats.reduce(iso=True),
            stats.reduce(angular_fourier=True, angular_harmonics=2),
            stats.reduce(iso=True, angular_fourier=True, angular_harmonics=3),
            stats.reduce(iso=True, scale_cosine=True, scale_harmonics=3, dj=3),
            stats.reduce(iso=True, angular_fourier=True, scale_cosine=True),
            stats.select(j3=[1, 3]),
            stats.select("S4", channel=(0, 2), l1=1),
        ):
            _check_layout(reduced)
    # OP-30: after normalisation, the cross statistics of channel 1 do not exist.
    s1 = stats._grids[2]
    assert s1.name == "S1"
    assert not s1.present[1, 2].any()
    assert s1.present[0, 2].all()


# ---------------------------------------------------------------------- full_grid and select


def test_full_grid_shapes_and_unknown_family() -> None:
    data = Data1D(_field((2, 2, 64)), pbc=True)
    stats = _operator(data, 3, L=2).apply(data)
    shapes = {
        "mean": (2, 2),
        "var": (2, 2),
        "S1": (2, 2, 2, 3, 2),
        "S2": (2, 2, 2, 3, 2),
        "S3": (2, 2, 2, 3, 3, 2, 2),
        "S4": (2, 2, 2, 3, 3, 3, 2, 2, 2),
    }
    for family, shape in shapes.items():
        assert tuple(stats.full_grid(family).shape) == shape
    with pytest.raises(ValueError, match="hold mean, var, S1, S2, S3, S4"):
        stats.full_grid("PS")


def test_select_restricts_along_named_axes() -> None:
    data = Data2D(_field((2, 3, 32, 32)), pbc=True)
    stats = _operator(data, 3).apply(data)
    full = {f: _numpy(stats.full_grid(f)) for f in FAMILIES}

    by_scale = stats.select(j3=2)  # S3 and S4 only, the axis dropped
    assert [g.name for g in by_scale._grids] == ["S3", "S4"]
    np.testing.assert_array_equal(_numpy(by_scale.full_grid("S3")), full["S3"][:, :, :, :, 2])
    np.testing.assert_array_equal(_numpy(by_scale.full_grid("S4")), full["S4"][:, :, :, :, :, 2])
    assert by_scale._grids[0].axes == ("c1", "c2", "j2", "l2", "l3")

    kept = stats.select("S1", j=[2, 0], l=slice(1, 3))  # sequences and slices keep the axis
    np.testing.assert_array_equal(
        _numpy(kept.full_grid("S1")), full["S1"][:, :, :, [2, 0]][..., 1:3]
    )
    assert kept._grids[0].coords[2:] == ((2, 0), (1, 2))

    pair = stats.select(channel=(0, 2))  # c1, c2: mean and var are left out
    assert [g.name for g in pair._grids] == ["S1", "S2", "S3", "S4"]
    np.testing.assert_array_equal(_numpy(pair.full_grid("S2")), full["S2"][:, 0, 2])

    moments = stats.select(("mean", "var"), c=1)
    np.testing.assert_array_equal(_numpy(moments.full_grid("var")), full["var"][:, 1])

    # The result keeps every method: reductions along untouched axes, and to_vector.
    np.testing.assert_array_equal(
        _numpy(stats.select(c1=[0], c2=[2]).reduce(iso=True).full_grid("S3")),
        _numpy(stats.reduce(iso=True).select(c1=[0], c2=[2]).full_grid("S3")),
    )
    assert _numpy(pair.to_vector()).ndim == 1


def test_select_errors() -> None:
    data = Data2D(_field((1, 2, 16, 16)), pbc=True)
    stats = _operator(data, 2).apply(data)
    with pytest.raises(ValueError, match="valid axis names: c1, c2, j, l"):
        stats.select("S1", j3=0)
    with pytest.raises(
        ValueError, match=r"no selected family \(mean, var, S1, S2, S3, S4\) has an axis 'x'"
    ):
        stats.select(x=0)
    with pytest.raises(ValueError, match="no selected family has every axis"):
        stats.select(j=0, j3=0)
    with pytest.raises(ValueError, match="hold mean, var"):
        stats.select("PS")
    with pytest.raises(ValueError, match="not both"):
        stats.select(channel=(0, 1), c1=0)
    with pytest.raises(ValueError, match="pair of channels"):
        stats.select(channel=0)
    with pytest.raises(IndexError, match="2 position"):
        stats.select(j=2)
    with pytest.raises(TypeError, match="sequence of integers"):
        stats.select(j=[[0]])
    with pytest.raises(ValueError, match="apply the reductions before selecting"):
        stats.select(l2=[0, 1]).reduce(iso=True)


# ---------------------------------------------------------------------- to_vector


def _vector_from_grids(stats: STStatistics) -> NDArray[Any]:
    """STL's ``to_flatten(flatten_complex=True)``, written with NumPy and ``isnan``."""
    parts = []
    for grid in stats._grids:
        values = _numpy(stats.full_grid(grid.name))
        if np.iscomplexobj(values):
            pairs = np.stack([values.real, values.imag], axis=-1)
            values = pairs[..., 0] if grid.name in ("S1", "S2") else pairs
        flat = values.reshape(-1)
        parts.append(flat[~np.isnan(flat)])
    return np.concatenate(parts)


def test_to_vector_is_stls_flattening(backend: Backend) -> None:
    data = _masked_2d(nc=2)
    stats = _operator(data, 4).apply(data)
    reduced_all = stats.reduce(
        iso=True,
        angular_fourier=True,
        angular_harmonics=2,
        scale_cosine=True,
        scale_harmonics=2,
        dj=3,
    )
    for reduced in (stats, reduced_all, stats.select(j2=1)):
        vector = _numpy(reduced.to_vector())
        np.testing.assert_array_equal(vector, _vector_from_grids(reduced))
        assert np.isfinite(vector).all()
    # OP-37: the imaginary parts of the cross S1 and S2 are not zero, and are dropped.
    assert np.abs(_numpy(stats.full_grid("S2"))[:, 0, 1].imag).max() > 0


def test_to_vector_mask_st(backend: Backend) -> None:
    data = Data1D(_field((2, 1, 64)), pbc=True)
    stats = _operator(data, 3).apply(data)
    vector = _numpy(stats.to_vector())
    mask_st = np.arange(vector.shape[0]) % 3 == 0
    np.testing.assert_array_equal(_numpy(stats.to_vector(mask_st)), vector[mask_st])
    as_backend = backend.asarray(mask_st, "bool")
    np.testing.assert_array_equal(_numpy(stats.to_vector(as_backend)), vector[mask_st])
    with pytest.raises(ValueError, match="entries, but the vector has"):
        stats.to_vector(mask_st[:-1])
    with pytest.raises(TypeError, match="1D boolean array"):
        stats.to_vector(mask_st.astype(int))


# ---------------------------------------------------------------------- reductions


def test_reduce_errors_and_axis_names() -> None:
    """Each error of reduce says what to do instead (R-S08); the reduced axes are named."""
    data = Data2D(_field((1, 1, 16, 16)), pbc=True)
    stats = _operator(data, 2).apply(data)
    with pytest.raises(ValueError, match="no reduction selected"):
        stats.reduce()
    alone: list[tuple[dict[str, Any], str]] = [
        ({"angular_harmonics": 2}, "angular_fourier"),
        ({"iso": True, "scale_harmonics": 2}, "scale_cosine"),
        ({"iso": True, "dj": 1}, "scale_cosine"),
    ]
    for kwargs, transform in alone:
        with pytest.raises(ValueError, match=f"pass {transform}=True as well"):
            stats.reduce(**kwargs)
    with pytest.raises(ValueError, match="without iso=True"):
        stats.reduce(scale_cosine=True)
    with pytest.raises(ValueError, match=r"already reduced \(iso\); call reduce once"):
        stats.reduce(iso=True).reduce(angular_fourier=True)
    with pytest.raises(ValueError, match="between 1 and L = 4"):
        stats.reduce(angular_fourier=True, angular_harmonics=5)
    with pytest.raises(ValueError, match="between 1 and J = 2"):
        stats.reduce(iso=True, scale_cosine=True, dj=0)
    with pytest.raises(TypeError, match="True or False"):
        stats.reduce(iso=1)  # type: ignore[arg-type]
    reduced = stats.reduce(iso=True, angular_fourier=True, scale_cosine=True)
    names = {g.name: g.axes for g in reduced._grids}
    assert names["S3"] == ("c1", "c2", "k", "a", "hdl")
    assert names["S4"] == ("c1", "c2", "k", "a", "b", "hdl2", "hdl3")
    angular = stats.reduce(angular_fourier=True)
    assert {g.name: g.axes for g in angular._grids}["S3"][-2:] == ("hl2", "hl3")


def test_iso_is_the_sum_of_m_red_iso(dtype: str) -> None:
    """iso S1 is the mean over l; iso S3 at d is (1/L) Σ_{l2} S3[.., l2, l2 + d mod L]
    (M-RED-ISO), written here with NumPy."""
    data = Data2D(_field((1, 1, 32, 32)), pbc=True)
    stats = _operator(data, 3).apply(data)
    iso = stats.reduce(iso=True)
    s1 = _numpy(stats.full_grid("S1"))
    np.testing.assert_allclose(
        _numpy(iso.full_grid("S1")), s1.mean(axis=-1), rtol=_tol(dtype), atol=_tol(dtype)
    )
    s3 = _numpy(stats.full_grid("S3"))
    expected = np.stack(
        [np.mean([s3[..., m, (m + d) % 4] for m in range(4)], axis=0) for d in range(4)], -1
    )
    np.testing.assert_allclose(
        _numpy(iso.full_grid("S3")), expected, rtol=_tol(dtype), atol=_tol(dtype)
    )


# ---------------------------------------------------------------------- normalisation


def test_norm_self_keeps_and_replaces_the_reference(dtype: str) -> None:
    data = Data2D(_field((2, 2, 32, 32)), pbc=True)
    operator = _operator(data, 3)
    first = operator.apply(data)
    assert operator._reference is first._reference
    np.testing.assert_allclose(_numpy(first.full_grid("var")), 1.0, rtol=10 * _tol(dtype))
    s2 = _numpy(first.full_grid("S2"))
    np.testing.assert_allclose(s2[:, [0, 1], [0, 1]].real, 1.0, rtol=10 * _tol(dtype))
    with pytest.warns(UserWarning, match="replaces the reference"):
        second = operator.apply(data.with_array(2 * data.array))
    assert operator._reference is second._reference


def test_ref_batch_mean(dtype: str) -> None:
    data = Data1D(_field((3, 1, 64)), pbc=True)
    stats = _operator(data, 3).apply(data, ref_batch_mean=True)
    reference = stats._reference
    assert reference is not None
    assert tuple(reference.var.shape) == (1, 1)
    np.testing.assert_allclose(
        _numpy(reference.var)[0],
        _numpy(_operator(data, 3).apply(data, norm="none").full_grid("var")).mean(axis=0),
        rtol=_tol(dtype),
    )
    per_element = _operator(data, 3).apply(data)._reference
    assert per_element is not None
    assert tuple(per_element.var.shape) == (3, 1)


def test_norm_ref(dtype: str) -> None:
    data = Data2D(_field((2, 2, 32, 32)), pbc=False)
    other = data.with_array(data.array * 3 + 1)
    target = _operator(data, 3).apply(data)
    loading = _operator(data, 3)
    # The target normalised by its own reference is its norm="self" statistics.
    again = loading.apply(data, ref=target, norm="ref")
    assert (again._norm, again._ref_batch_mean) == ("ref", False)
    for family in FAMILIES:
        np.testing.assert_allclose(
            _numpy(again.full_grid(family)),
            _numpy(target.full_grid(family)),
            rtol=_tol(dtype),
            atol=_tol(dtype),
            err_msg=family,
        )
    assert loading._reference is None  # norm="ref" stores nothing
    loaded = loading.apply(other, ref=target, norm="ref")
    # from_statistics: an operator holding the reference of its statistics.
    rebuilt = STOperator.from_statistics(loaded)
    np.testing.assert_array_equal(
        _numpy(rebuilt.apply(other, norm="ref").to_vector()), _numpy(loaded.to_vector())
    )
    smaller = _operator(data[0], 3).apply(data[0])  # a reference of one batch element
    loading.apply(other, ref=smaller, norm="ref")  # broadcasts
    three = Data2D(_field((3, 2, 32, 32)), pbc=False)
    with pytest.raises(ValueError, match="batch element"):
        _operator(three, 3).apply(three, ref=target, norm="ref")
    with pytest.raises(ValueError, match="same J, L and channels"):
        _operator(data, 2).apply(data, ref=target, norm="ref")


def test_one_operator_serves_every_normalisation(dtype: str) -> None:
    """One operator applied with ``norm="self"`` on a target, then ``norm="ref"`` on another
    field, gives the statistics of two operators, one per normalisation, the second given the
    target's statistics (STL's ``store_ref`` then ``load_ref``; task T2.19). ``norm="none"``
    reads and stores no reference."""
    data = Data2D(_field((2, 2, 32, 32)), pbc=False)
    other = data.with_array(data.array * 3 + 1)
    target = _operator(data, 3).apply(data)
    pair = _operator(data, 3).apply(other, ref=target, norm="ref")
    operator = _operator(data, 3)
    first = operator.apply(data)
    stored = operator._reference
    second = operator.apply(other, norm="ref")
    np.testing.assert_array_equal(_numpy(first.to_vector()), _numpy(target.to_vector()))
    np.testing.assert_array_equal(_numpy(second.to_vector()), _numpy(pair.to_vector()))
    raw = operator.apply(other, norm="none")
    assert raw._reference is None
    assert operator._reference is stored
    assert stored is not None
    expected = _numpy(raw.full_grid("var")) / _numpy(stored.var)
    np.testing.assert_allclose(_numpy(second.full_grid("var")), expected, rtol=_tol(dtype))


def _normalisation_error(case: str) -> tuple[dict[str, Any], type[Exception], str]:
    """The arguments of ``apply`` and of ``functional.scattering`` for one invalid
    normalisation (maths.md M-NORM; issue #192, decisions, point 2), the error and its
    message."""
    data = Data2D(_field((1, 2, 32, 32)), pbc=True)
    target = _operator(data, 3).apply(data)
    raw = _operator(data, 3).apply(data, norm="none")
    loaded = _operator(data, 3).apply(data, ref=target, norm="ref")
    cases: dict[str, tuple[dict[str, Any], type[Exception], str]] = {
        "unknown norm": ({"norm": "store_ref"}, ValueError, "'self', 'ref', 'none'"),
        "norm not a string": ({"norm": None}, ValueError, "'self', 'ref', 'none'"),
        "ref without a reference": ({"norm": "ref"}, ValueError, 'norm="ref" needs a reference'),
        "ref with norm=self": ({"ref": target}, ValueError, 'norm="self" takes no reference'),
        "ref with norm=none": (
            {"ref": target, "norm": "none"},
            ValueError,
            'norm="none" takes no reference',
        ),
        "ref not statistics": (
            {"ref": object(), "norm": "ref"},
            TypeError,
            "pass STStatistics",
        ),
        "ref of norm=none": ({"ref": raw, "norm": "ref"}, ValueError, 'computed with norm="none"'),
        "ref of norm=ref": ({"ref": loaded, "norm": "ref"}, ValueError, 'computed with norm="ref"'),
        "ref_batch_mean with norm=ref": (
            {"ref": target, "norm": "ref", "ref_batch_mean": True},
            ValueError,
            'ref_batch_mean=True with norm="ref"',
        ),
        "ref_batch_mean with norm=none": (
            {"norm": "none", "ref_batch_mean": True},
            ValueError,
            'ref_batch_mean=True with norm="none"',
        ),
        "ref_batch_mean not a boolean": ({"ref_batch_mean": 1}, TypeError, "pass True or False"),
    }
    return cases[case]


@pytest.mark.parametrize(
    "case",
    [
        "unknown norm",
        "norm not a string",
        "ref without a reference",
        "ref with norm=self",
        "ref with norm=none",
        "ref not statistics",
        "ref of norm=none",
        "ref of norm=ref",
        "ref_batch_mean with norm=ref",
        "ref_batch_mean with norm=none",
        "ref_batch_mean not a boolean",
    ],
)
def test_invalid_normalisations_are_refused(case: str) -> None:
    """Each invalid normalisation of M-NORM raises a clear error through ``apply`` and through
    ``functional.scattering``, before the loop, and the operator keeps the reference it held."""
    options, error, message = _normalisation_error(case)
    data = Data2D(_field((1, 2, 32, 32), seed=1), pbc=True)
    operator = _operator(data, 3)
    with pytest.raises(error, match=message):
        operator.apply(data, **options)
    assert operator._reference is None
    with pytest.raises(error, match=message):
        estel.functional.scattering(data, operator.plan, operator.filters, **options)
    if case == "ref without a reference":
        # The advice of apply: compute a reference first, or pass one.
        with pytest.raises(ValueError, match=r'Call STOperator.apply\(data, norm="self"\) first'):
            operator.apply(data, norm="ref")


def test_reference_of_other_data_is_accepted(dtype: str) -> None:
    """A reference computed on data of another size, mask and periodicity normalises these
    statistics, through ``apply`` and ``functional.scattering``: the syntheses need it."""
    target = _masked_2d(nb=1, nc=2)  # 64 x 64, a hole, open
    reference = _operator(target, 3).apply(target)
    data = Data2D(_field((2, 2, 32, 32), seed=1), pbc=True)  # 32 x 32, no mask, periodic
    loading = _operator(data, 3)
    raw = _operator(data, 3).apply(data, norm="none")
    stored = reference._reference
    assert stored is not None
    expected_var = _numpy(raw.full_grid("var")) / _numpy(stored.var)
    for loaded in (
        loading.apply(data, ref=reference, norm="ref"),
        estel.functional.scattering(data, loading.plan, loading.filters, ref=reference, norm="ref"),
    ):
        np.testing.assert_allclose(
            _numpy(loaded.full_grid("var")), expected_var, rtol=_tol(dtype), atol=0
        )
        _check_layout(loaded)


@pytest.mark.parametrize(
    ("J", "L", "nc", "matrix", "message"),
    [
        (2, 4, 2, None, r"J = 2 \(the operator has 3\)"),
        (3, 2, 2, None, r"L = 2 \(the operator has 4\)"),
        (3, 4, 3, None, r"number of channels = 3 \(the operator has 2\)"),
        (3, 4, 2, [[True, True], [False, False]], r"no normalisation scale for channel\(s\) \[1\]"),
    ],
)
def test_reference_of_another_plan_is_refused(
    J: int,  # noqa: N803
    L: int,  # noqa: N803
    nc: int,
    matrix: list[list[bool]] | None,
    message: str,
) -> None:
    """A reference without the J, L or channels the operator normalises raises, through
    ``apply`` and ``functional.scattering``, naming what differs."""
    source = Data2D(_field((1, nc, 32, 32)), pbc=True)
    cross = None if matrix is None else np.array(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-30
        reference = _operator(source, J, L=L, cross_matrix=cross).apply(source)
    data = Data2D(_field((1, 2, 32, 32), seed=1), pbc=True)
    loading = _operator(data, 3)
    with pytest.raises(ValueError, match=message):
        loading.apply(data, ref=reference, norm="ref")
    with pytest.raises(ValueError, match=message):
        estel.functional.scattering(data, loading.plan, loading.filters, ref=reference, norm="ref")


def test_channels_without_auto_statistics_warn_at_a_normalised_apply() -> None:
    """OP-30: the warning comes with the normalisation, chosen at ``apply`` (task T2.19)."""
    data = Data2D(_field((1, 3, 16, 16)), pbc=True)
    matrix = np.array([[True, False, True], [False, False, True], [False, False, True]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        operator = STOperator(data, 2, cross_matrix=matrix, compute_PS=False)
        operator.apply(data, norm="none")
    with pytest.warns(UserWarning, match=r"channel\(s\) \[1\]"):
        operator.apply(data)


@pytest.mark.slow
def test_gradients_flow_through_norm_self_not_norm_ref(backend: Backend, dtype: DType) -> None:
    """With norm="self" the reference is part of the computation; with norm="ref" it is a
    constant (M-NORM, LF21). Both are finite to second order with missing values and empty
    means, since nothing absent reaches them.

    Slow: Hessian-vector products through two normalised operators (second order)."""
    data = _masked_2d(nb=1, nc=2)
    storing = _operator(data, 4)
    loading = _operator(data, 4)
    xp = backend.xp

    def store_loss(array: Any) -> Any:
        return xp.sum(storing.apply(data.with_array(array)).to_vector() ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the reference is replaced at each call
        reference = storing.apply(data)

        def load_loss(array: Any) -> Any:
            stats = loading.apply(data.with_array(array), ref=reference, norm="ref")
            return xp.sum(stats.to_vector() ** 2)

        _, grad_store = backend.value_and_grad(store_loss)(data.array)
        _, grad_load = backend.value_and_grad(load_loss)(data.array)
        assert data.mask is not None
        direction = backend.asarray(np.where(data.mask, 0.0, 1.0) * np.ones(data.shape), dtype)
        hessian_v = backend.to_numpy(backend.hvp(store_loss, data.array, direction))
    grad_store, grad_load = backend.to_numpy(grad_store), backend.to_numpy(grad_load)
    assert np.isfinite(grad_store).all()
    assert np.isfinite(grad_load).all()
    assert np.isfinite(hessian_v).all()
    np.testing.assert_array_equal(grad_store[..., data.mask], 0.0)
    # The two losses agree at this point, their gradients do not: norm="self" differentiates
    # the reference as well.
    assert np.abs(grad_store - grad_load).max() > 1e-3 * np.abs(grad_load).max()


# ---------------------------------------------------------------------- functional


def test_functional_scattering_is_apply(backend: Backend) -> None:
    data = Data1D(_field((2, 2, 64)), pbc=False)
    operator = _operator(data, 3)
    stats = estel.functional.scattering(data, operator.plan, operator.filters, norm="none")
    np.testing.assert_array_equal(
        _numpy(stats.to_vector()), _numpy(operator.apply(data, norm="none").to_vector())
    )
    storing = _operator(data, 3)
    stored = estel.functional.scattering(data, storing.plan, storing.filters)
    assert storing._reference is None  # the pure function keeps nothing
    with pytest.raises(ValueError, match="needs a reference"):
        estel.functional.scattering(data, storing.plan, storing.filters, norm="ref")
    loaded = estel.functional.scattering(
        data, storing.plan, storing.filters, ref=stored, norm="ref"
    )
    np.testing.assert_allclose(
        _numpy(loaded.to_vector()), _numpy(stored.to_vector()), rtol=1e-5, atol=1e-5
    )
    with pytest.raises(ValueError, match="build a new STOperator"):
        estel.functional.scattering(
            Data1D(_field((1, 2, 32)), pbc=False), operator.plan, operator.filters
        )
    # 2D data with the power spectrum, the pure function of SpectrumOperator removed (T4.6): the
    # family PS of the pure function is that of apply, normalised by itself.
    planar = Data2D(_field((2, 2, 16, 16)), pbc=True)
    with_ps = STOperator(planar, 2, 1, compute_PS=True)
    np.testing.assert_array_equal(
        _numpy(estel.functional.scattering(planar, with_ps.plan, with_ps.filters).full_grid("PS")),
        _numpy(with_ps.apply(planar).full_grid("PS")),
    )


def test_plan_and_filters_are_opaque_and_static() -> None:
    data = Data2D(_field((1, 2, 16, 16)), pbc=True)
    first, second = _operator(data, 2), _operator(data, 2)
    assert isinstance(first.plan, estel.STPlan)
    assert isinstance(first.filters, estel.STFilters)
    assert first.plan == second.plan
    assert hash(first.plan) == hash(second.plan)
    assert first.plan != _operator(data, 2, standardize=False).plan
    assert first.plan != _operator(data, 2, cross_matrix=np.eye(2, dtype=bool)).plan


# ---------------------------------------------------------------------- storage


@pytest.mark.dtype
def test_save_and_load_round_trip(tmp_path: Path, dtype: str) -> None:
    data = _masked_2d(nc=2)
    stats = _operator(data, 4, standardize=True).apply(data, ref_batch_mean=True)
    reduced = stats.reduce(
        iso=True,
        angular_fourier=True,
        angular_harmonics=2,
        scale_cosine=True,
        scale_harmonics=3,
        dj=2,
    ).select(c1=[0, 1])
    for original in (stats, reduced):
        path = tmp_path / "stats.npz"
        original.save(path)
        loaded = STStatistics.load(path)
        assert loaded._plan == original._plan
        assert loaded._grids == original._grids
        assert loaded._reductions == original._reductions
        assert (loaded._norm, loaded._ref_batch_mean) == ("self", True)
        for grid in original._grids:
            np.testing.assert_array_equal(
                _numpy(loaded.full_grid(grid.name)), _numpy(original.full_grid(grid.name))
            )
            assert str(loaded.full_grid(grid.name).dtype) == str(
                original.full_grid(grid.name).dtype
            )
        np.testing.assert_array_equal(_numpy(loaded.to_vector()), _numpy(original.to_vector()))
        assert loaded._reference is not None
        assert original._reference is not None
        np.testing.assert_array_equal(
            _numpy(loaded._reference.s2_sqrt), _numpy(original._reference.s2_sqrt)
        )
        assert loaded._standardization is not None
    rebuilt = STOperator.from_statistics(STStatistics.load(tmp_path / "stats.npz"))
    assert rebuilt.plan == stats._plan
    assert rebuilt._dtype == dtype


def test_load_checks_the_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = Data1D(_field((1, 1, 32)), pbc=True)
    stats = _operator(data, 2).apply(data)
    path = tmp_path / "stats.npz"
    stats.save(path)
    np.savez(tmp_path / "plain.npz", x=np.zeros(3))
    with pytest.raises(ValueError, match="not written by ESTEL"):
        STStatistics.load(tmp_path / "plain.npz")
    (tmp_path / "text.npz").write_text("not an archive")
    with pytest.raises(ValueError, match="is not an ESTEL file"):
        STStatistics.load(tmp_path / "text.npz")
    _schema.write(tmp_path / "other.npz", "estel.other", {}, {})
    with pytest.raises(ValueError, match=r"not 'estel\.statistics'"):
        STStatistics.load(tmp_path / "other.npz")
    written = _schema.SCHEMA_VERSION
    monkeypatch.setattr(_schema, "SCHEMA_VERSION", written + 1)
    with pytest.raises(ValueError, match=f"schema version {written}"):
        STStatistics.load(path)
    monkeypatch.setattr(_schema, "SCHEMA_VERSION", written)
    monkeypatch.setattr(_schema, "STATISTICS_REFERENCE", 7)
    with pytest.warns(UserWarning, match="STATISTICS_REFERENCE"):
        STStatistics.load(path)


def test_statistics_are_immutable_and_not_built_directly() -> None:
    data = Data1D(_field((1, 1, 32)), pbc=True)
    stats = _operator(data, 2).apply(data)
    with pytest.raises(AttributeError, match="immutable"):
        stats._buffer = None
    with pytest.raises(TypeError, match="computed, not built"):
        STStatistics()
    assert "STStatistics(mean(1, 1)" in repr(stats)


# ---------------------------------------------------------------------- pytrees


@jax_only
def test_jit_returns_statistics() -> None:
    import jax

    data = Data2D(_field((1, 2, 16, 16)), mask=None, pbc=True)
    operator = _operator(data, 2)
    plan, filters = operator.plan, operator.filters
    compiled = jax.jit(lambda d, f: estel.functional.scattering(d, plan, f))
    stats = compiled(data, filters)
    assert isinstance(stats, STStatistics)
    expected = _numpy(operator.apply(data).to_vector())
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(_numpy(stats.to_vector()), expected, rtol=1e-5, atol=1e-6 * scale)
    leaves, treedef = jax.tree_util.tree_flatten(stats)
    # The compact buffer, the two arrays of the reference, and the two moments removed by the
    # standardisation, on by default (OP-120).
    assert len(leaves) == 1 + 2 + 2
    again = jax.tree_util.tree_unflatten(treedef, leaves)
    assert again._grids == stats._grids
    # norm is a Python value, static under jit as the plan is (task T2.19).
    raw = jax.jit(lambda d, f: estel.functional.scattering(d, plan, f, norm="none"))(data, filters)
    assert (raw._norm, raw._reference) == ("none", None)
    assert len(jax.tree_util.tree_leaves(raw)) == 1 + 2


@torch_only
def test_torch_pytree_flattens_statistics() -> None:
    from torch.utils import _pytree

    data = Data1D(_field((1, 1, 32)), pbc=True)
    stats = _operator(data, 2, standardize=True).apply(data)
    leaves, spec = _pytree.tree_flatten(stats)
    assert len(leaves) == 1 + 2 + 2  # the compact buffer, reference, standardisation
    again = _pytree.tree_unflatten(leaves, spec)
    assert isinstance(again, STStatistics)
    np.testing.assert_array_equal(_numpy(again.to_vector()), _numpy(stats.to_vector()))
