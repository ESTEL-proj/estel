"""Conformance suite of the L0 primitives: every backend against NumPy float64.

Task T1.2 (Part II, §12.3; Part I, §5.1, "Backend conformance"). Each primitive is checked
for its values against a naive NumPy float64 transcription of its contract in
``estel.backend._protocol``, and for its derivatives: reverse mode (``vjp``,
``value_and_grad``), forward mode (``jvp``) and second order (``hvp``), against the exact
Jacobian that NumPy builds column by column for the linear and bilinear primitives. Error
cases follow the "Edge cases" section of each contract.

Inputs are drawn in float64 and rounded to the dtype of the run, so that the NumPy reference
sees exactly the values the backend sees; the remaining differences are the backend's
rounding. Tolerances are the starting values of Part I, §5.2 (``docs/agents/testing.md``):
float32 rtol 1e-4 / atol 1e-6, float64 CPU rtol 1e-10 / atol 1e-12, with ``atol`` relative
to the largest expected magnitude, since §5.2 compares normalised values.
``tests/tolerances.toml`` replaces this table when T8.1 creates it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import array_api_compat
import numpy as np
import pytest

from estel.backend._protocol import Backend, DType

Array = Any
NDArray = np.ndarray[Any, Any]

TOLERANCES = {"float32": (1e-4, 1e-6), "float64": (1e-10, 1e-12)}
COMPLEX_OF: dict[str, DType] = {"float32": "complex64", "float64": "complex128"}

# M-H: the anti-aliasing kernel of the downsampling, separable; its 1D factor.
H_1D = np.array([-0.04853, 0.05664, 0.49189, 0.49189, 0.05664, -0.04853])


# ------------------------------------------------------------------ helpers


def rounded(a: NDArray, dtype: DType) -> NDArray:
    """``a`` rounded to ``dtype`` and back to float64: what both sides compute with."""
    return np.asarray(a).astype(dtype).astype(np.float64)


def draw(shape: tuple[int, ...], dtype: DType, seed: int) -> NDArray:
    return rounded(np.random.default_rng(seed).standard_normal(shape), dtype)


def assert_close(backend: Backend, actual: Array, expected: NDArray, dtype: DType) -> None:
    got = backend.to_numpy(actual)
    expected = np.asarray(expected)
    assert got.shape == expected.shape, f"shape {got.shape}, expected {expected.shape}"
    rtol, atol = TOLERANCES[dtype]
    finite = np.abs(expected[np.isfinite(expected)])
    scale = max(1.0, float(finite.max())) if finite.size else 1.0
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol * scale, equal_nan=True)


def index_array(backend: Backend, values: Any, dtype: DType = "int64") -> Array:
    return backend.asarray(np.asarray(values), dtype)


# ------------------------------------------------------------------ NumPy references


def np_pad(x: NDArray, widths: list[tuple[int, int]], axes: list[int], mode: str) -> NDArray:
    full = [(0, 0)] * x.ndim
    for axis, width in zip(axes, widths, strict=True):
        full[axis % x.ndim] = width
    np_mode = {"circular": "wrap", "replicate": "edge", "zeros": "constant"}[mode]
    return np.pad(x, full, mode=np_mode)  # type: ignore[call-overload,no-any-return]


def np_conv(x: NDArray, k: NDArray, axes: list[int], stride: int | list[int] = 1) -> NDArray:
    """``y[..., o, n, ...] = Σ_m k[o, m] · x[..., s·n + m, ...]``, term by term."""
    d = len(axes)
    first = axes[0] % x.ndim
    strides = [stride] * d if isinstance(stride, int) else list(stride)
    support = k.shape[k.ndim - d :]
    out_axes = k.shape[: k.ndim - d]
    pre, post = x.shape[:first], x.shape[first + d :]
    out = tuple((x.shape[first + i] - support[i]) // strides[i] + 1 for i in range(d))
    y = np.zeros(pre + out_axes + out + post)
    for m in np.ndindex(*support):
        window = [slice(None)] * x.ndim
        for i in range(d):
            window[first + i] = slice(m[i], m[i] + strides[i] * (out[i] - 1) + 1, strides[i])
        part = x[tuple(window)].reshape(pre + (1,) * len(out_axes) + out + post)
        weight = k[(..., *m)].reshape((1,) * len(pre) + out_axes + (1,) * (d + len(post)))
        y = y + weight * part
    return y


def np_gather_weighted(x: NDArray, axis: int, index: NDArray, weight: NDArray) -> NDArray:
    axis = axis % x.ndim
    taken = np.take(x, index, axis=axis)  # pre + (P, Q, M) + post
    post = x.ndim - axis - 1
    return (taken * weight.reshape(weight.shape + (1,) * post)).sum(axis=axis + 2)  # type: ignore[no-any-return]


def np_segment_sum(x: NDArray, axis: int, ids: NDArray, num: int) -> NDArray:
    moved = np.moveaxis(x, axis, 0)
    y = np.zeros((num, *moved.shape[1:]), dtype=x.dtype)
    np.add.at(y, ids, moved)
    return np.moveaxis(y, 0, axis)


def np_scatter(
    values: NDArray, index: list[NDArray], shape: tuple[int, ...], fill: float
) -> NDArray:
    batch = values.shape[:-1]
    grid = np.full((*batch, math.prod(shape)), fill, dtype=values.dtype)
    grid[..., np.ravel_multi_index(tuple(index), shape)] = values
    return grid.reshape(*batch, *shape)


def np_gather(grid: NDArray, index: list[NDArray]) -> NDArray:
    d = len(index)
    shape = grid.shape[grid.ndim - d :]
    flat = np.ravel_multi_index(tuple(index), shape)
    return grid.reshape(*grid.shape[: grid.ndim - d], -1)[..., flat]


def pairs(z: NDArray) -> NDArray:
    return np.stack((z.real, z.imag), axis=-1)


def unpairs(r: NDArray) -> NDArray:
    return r[..., 0] + 1j * r[..., 1]


# ------------------------------------------------------------------ linear cases
#
# Every primitive is linear in each of its floating arguments (bilinear ones are linear in
# each argument with the other fixed), so its Jacobian is the matrix whose columns are the
# reference applied to unit vectors. Complex primitives are checked through real pairs,
# which also checks complex_to_real_pairs and real_pairs_to_complex.


@dataclass
class Linear:
    name: str
    shape: tuple[int, ...]
    apply: Callable[[Backend, Array, DType], Array]
    reference: Callable[[NDArray, DType], NDArray]
    marks: list[Any] = field(default_factory=list)


GW_INDEX = np.random.default_rng(10).integers(0, 7, size=(4, 2, 3))
SEG_IDS = np.array([0, 3, 3, 1, 0, 4, 1, 3])  # segment 2 is empty
GRID_INDEX = [np.array([0, 2, 1, 2, 0]), np.array([3, 0, 1, 2, 0])]


def _fixed(shape: tuple[int, ...], seed: int) -> Callable[[DType], NDArray]:
    return lambda dtype: draw(shape, dtype, seed)


K_2D = _fixed((3, 3, 2), 21)
K_1D = _fixed((2, 3), 22)
K_3D = _fixed((2, 2, 2), 23)
K_4D = _fixed((2, 2, 2, 2), 24)
X_CONV = _fixed((2, 7, 6), 25)
GW_WEIGHT = _fixed((4, 2, 3), 26)
X_GW = _fixed((2, 7, 3), 27)
COMPLEX_FACTOR = 0.6 - 1.3j


def _asarray(b: Backend, a: NDArray, dtype: DType) -> Array:
    return b.asarray(a, dtype)


LINEAR_CASES = [
    Linear(
        "pad-circular",
        (2, 5, 6),
        lambda b, x, dt: b.pad(x, [(2, 1), (1, 3)], [1, 2], "circular"),
        lambda x, dt: np_pad(x, [(2, 1), (1, 3)], [1, 2], "circular"),
    ),
    Linear(
        "pad-replicate",
        (3, 4),
        lambda b, x, dt: b.pad(x, [(3, 2)], [-1], "replicate"),
        lambda x, dt: np_pad(x, [(3, 2)], [-1], "replicate"),
    ),
    Linear(
        "pad-zeros",
        (2, 3, 4),
        lambda b, x, dt: b.pad(x, [(1, 0), (2, 2)], [0, 2], "zeros"),
        lambda x, dt: np_pad(x, [(1, 0), (2, 2)], [0, 2], "zeros"),
    ),
    Linear(
        "pad-complex",
        (2, 5, 2),
        lambda b, x, dt: b.complex_to_real_pairs(
            b.pad(b.real_pairs_to_complex(x), [(2, 2)], [1], "circular")
        ),
        lambda x, dt: pairs(np_pad(unpairs(x), [(2, 2)], [1], "circular")),
    ),
    Linear(
        "conv-2d-in-x",
        (2, 7, 6),
        lambda b, x, dt: b.conv_axes(x, _asarray(b, K_2D(dt), dt), [1, 2]),
        lambda x, dt: np_conv(x, K_2D(dt), [1, 2]),
    ),
    Linear(
        "conv-2d-in-k",
        (3, 3, 2),
        lambda b, k, dt: b.conv_axes(_asarray(b, X_CONV(dt), dt), k, [1, 2]),
        lambda k, dt: np_conv(X_CONV(dt), k, [1, 2]),
    ),
    Linear(
        "conv-1d-middle-axis-stride-2",
        (2, 9, 3),
        lambda b, x, dt: b.conv_axes(x, _asarray(b, K_1D(dt), dt), [1], stride=2),
        lambda x, dt: np_conv(x, K_1D(dt), [1], 2),
    ),
    Linear(
        "conv-3d-per-axis-stride",
        (1, 5, 4, 4),
        lambda b, x, dt: b.conv_axes(x, _asarray(b, K_3D(dt), dt), [-3, -2, -1], [1, 2, 1]),
        lambda x, dt: np_conv(x, K_3D(dt), [1, 2, 3], [1, 2, 1]),
    ),
    Linear(
        "conv-4d",
        (3, 4, 3, 3, 2),
        lambda b, x, dt: b.conv_axes(x, _asarray(b, K_4D(dt), dt), [0, 1, 2, 3]),
        lambda x, dt: np_conv(x, K_4D(dt), [0, 1, 2, 3]),
    ),
    Linear(
        "down-m-down",
        (1, 8, 8),
        lambda b, x, dt: b.conv_axes(
            b.pad(x, [(2, 3), (2, 3)], [1, 2], "replicate"),
            _asarray(b, rounded(np.outer(H_1D, H_1D), dt), dt),
            [1, 2],
            stride=2,
        ),
        lambda x, dt: np_conv(
            np_pad(x, [(2, 3), (2, 3)], [1, 2], "replicate"),
            rounded(np.outer(H_1D, H_1D), dt),
            [1, 2],
            2,
        ),
    ),
    Linear(
        "gather-weighted-in-x",
        (2, 7, 3),
        lambda b, x, dt: b.gather_weighted(
            x, 1, index_array(b, GW_INDEX), _asarray(b, GW_WEIGHT(dt), dt)
        ),
        lambda x, dt: np_gather_weighted(x, 1, GW_INDEX, GW_WEIGHT(dt)),
    ),
    Linear(
        "gather-weighted-in-weight",
        (4, 2, 3),
        lambda b, w, dt: b.gather_weighted(
            _asarray(b, X_GW(dt), dt), -2, index_array(b, GW_INDEX, "int32"), w
        ),
        lambda w, dt: np_gather_weighted(X_GW(dt), 1, GW_INDEX, w),
    ),
    Linear(
        "segment-sum",
        (3, 8),
        lambda b, x, dt: b.segment_sum(x, -1, index_array(b, SEG_IDS), 5),
        lambda x, dt: np_segment_sum(x, 1, SEG_IDS, 5),
    ),
    Linear(
        "segment-sum-complex",
        (8, 2, 2),
        lambda b, x, dt: b.complex_to_real_pairs(
            b.segment_sum(b.real_pairs_to_complex(x), 0, index_array(b, SEG_IDS, "int32"), 5)
        ),
        lambda x, dt: pairs(np_segment_sum(unpairs(x), 0, SEG_IDS, 5)),
    ),
    Linear(
        "scatter-to-grid",
        (2, 5),
        lambda b, v, dt: b.scatter_to_grid(
            v, [index_array(b, i) for i in GRID_INDEX], (3, 4), fill=0.0
        ),
        lambda v, dt: np_scatter(v, GRID_INDEX, (3, 4), 0.0),
    ),
    Linear(
        "gather-from-grid",
        (2, 3, 4),
        lambda b, g, dt: b.gather_from_grid(g, [index_array(b, i, "int32") for i in GRID_INDEX]),
        lambda g, dt: np_gather(g, GRID_INDEX),
    ),
    Linear(
        "complex-arithmetic",
        (3, 2),
        lambda b, x, dt: b.complex_to_real_pairs(b.real_pairs_to_complex(x) * COMPLEX_FACTOR),
        lambda x, dt: pairs(unpairs(x) * COMPLEX_FACTOR),
    ),
]


def jacobian(case: Linear, x: NDArray, dtype: DType) -> NDArray:
    """The Jacobian ``J[out, in]`` of a linear case, from NumPy, in float64."""
    columns = []
    for i in range(x.size):
        unit = np.zeros(x.size)
        unit[i] = 1.0
        columns.append(case.reference(unit.reshape(x.shape), dtype).ravel())
    return np.stack(columns, axis=1)


@pytest.fixture(params=LINEAR_CASES, ids=lambda case: case.name)
def case(request: pytest.FixtureRequest) -> Linear:
    return request.param  # type: ignore[no-any-return]


class TestLinearPrimitives:
    def test_values(self, backend: Backend, case: Linear, dtype: DType) -> None:
        x = draw(case.shape, dtype, 1)
        assert_close(
            backend,
            case.apply(backend, backend.asarray(x, dtype), dtype),
            case.reference(x, dtype),
            dtype,
        )

    def test_dtype_is_kept(self, backend: Backend, case: Linear, dtype: DType) -> None:
        x = backend.asarray(draw(case.shape, dtype, 1), dtype)
        assert backend.to_numpy(case.apply(backend, x, dtype)).dtype == np.dtype(dtype)

    def test_input_is_not_modified(self, backend: Backend, case: Linear, dtype: DType) -> None:
        x_np = draw(case.shape, dtype, 1)
        x = backend.asarray(x_np, dtype)
        case.apply(backend, x, dtype)
        np.testing.assert_array_equal(backend.to_numpy(x), x_np.astype(dtype))

    def test_vjp(self, backend: Backend, case: Linear, dtype: DType) -> None:
        x = draw(case.shape, dtype, 1)
        jac = jacobian(case, x, dtype)
        out_shape = case.reference(x, dtype).shape
        c = draw(out_shape, dtype, 2)
        out, vjp_fn = backend.vjp(
            lambda a: case.apply(backend, a, dtype), backend.asarray(x, dtype)
        )
        assert_close(backend, out, case.reference(x, dtype), dtype)
        (cotangent,) = vjp_fn(backend.asarray(c, dtype))
        assert_close(backend, cotangent, (jac.T @ c.ravel()).reshape(case.shape), dtype)

    def test_jvp(self, backend: Backend, case: Linear, dtype: DType) -> None:
        x = draw(case.shape, dtype, 1)
        t = draw(case.shape, dtype, 3)
        jac = jacobian(case, x, dtype)
        out_shape = case.reference(x, dtype).shape
        out, tangent = backend.jvp(
            lambda a: case.apply(backend, a, dtype),
            (backend.asarray(x, dtype),),
            (backend.asarray(t, dtype),),
        )
        assert_close(backend, out, case.reference(x, dtype), dtype)
        assert_close(backend, tangent, (jac @ t.ravel()).reshape(out_shape), dtype)

    def test_value_and_grad(self, backend: Backend, case: Linear, dtype: DType) -> None:
        """f(x) = Σ sin(Jx): grad = Jᵀ cos(Jx)."""
        xp = backend.xp
        x = 0.3 * draw(case.shape, dtype, 1)
        x = rounded(x, dtype)
        jac = jacobian(case, x, dtype)
        value, grad = backend.value_and_grad(
            lambda a: xp.sum(xp.sin(case.apply(backend, a, dtype)))
        )(backend.asarray(x, dtype))
        jx = jac @ x.ravel()
        assert_close(backend, value, np.asarray(np.sin(jx).sum()), dtype)
        assert_close(backend, grad, (jac.T @ np.cos(jx)).reshape(case.shape), dtype)

    def test_hvp(self, backend: Backend, case: Linear, dtype: DType) -> None:
        """Second order: f(x) = Σ sin(Jx) has H·v = -Jᵀ (sin(Jx) ⊙ Jv)."""
        xp = backend.xp
        x = rounded(0.3 * draw(case.shape, dtype, 1), dtype)
        v = draw(case.shape, dtype, 4)
        jac = jacobian(case, x, dtype)
        hv = backend.hvp(
            lambda a: xp.sum(xp.sin(case.apply(backend, a, dtype))),
            backend.asarray(x, dtype),
            backend.asarray(v, dtype),
        )
        expected = -jac.T @ (np.sin(jac @ x.ravel()) * (jac @ v.ravel()))
        assert_close(backend, hv, expected.reshape(case.shape), dtype)

    def test_second_order_through_vjp(self, backend: Backend, case: Linear, dtype: DType) -> None:
        """The gradient of g(x) = ½‖Jᵀ cos(Jx)‖², whose inner gradient comes from vjp."""
        xp = backend.xp
        x = rounded(0.3 * draw(case.shape, dtype, 1), dtype)
        jac = jacobian(case, x, dtype)

        def g(a: Array) -> Array:
            _, inner = backend.value_and_grad(
                lambda b: xp.sum(xp.sin(case.apply(backend, b, dtype)))
            )(a)
            return 0.5 * xp.sum(inner * inner)

        _, grad = backend.value_and_grad(g)(backend.asarray(x, dtype))
        jx = jac @ x.ravel()
        inner = jac.T @ np.cos(jx)
        expected = -jac.T @ (np.sin(jx) * (jac @ inner))
        assert_close(backend, grad, expected.reshape(case.shape), dtype)


# ------------------------------------------------------------------ name and namespace


def test_name(backend: Backend) -> None:
    assert backend.name in ("torch", "jax")


def test_xp_is_the_array_api_namespace(backend: Backend, dtype: DType) -> None:
    x = backend.asarray(np.zeros(3), dtype)
    assert array_api_compat.array_namespace(x) is backend.xp


def test_xp_values_used_by_the_statistics(backend: Backend, dtype: DType) -> None:
    """Masked means with {0, 1} weights (M-MEAN) and the orthonormal FFTs (M-PS-2D, M-RED-ANG)."""
    xp = backend.xp
    a = draw((2, 6, 8), dtype, 5)
    w = (np.random.default_rng(6).random((6, 8)) > 0.3).astype(np.float64)
    ta, tw = backend.asarray(a, dtype), backend.asarray(w, dtype)
    mean = xp.sum(tw * ta, axis=(-2, -1)) / xp.sum(tw)
    assert_close(backend, mean, (w * a).sum(axis=(-2, -1)) / w.sum(), dtype)
    assert_close(
        backend,
        backend.complex_to_real_pairs(xp.fft.rfft2(ta, norm="ortho")),
        pairs(np.fft.rfft2(a, norm="ortho")),
        dtype,
    )
    assert_close(
        backend,
        backend.complex_to_real_pairs(xp.fft.fft(ta, axis=-1, norm="ortho")),
        pairs(np.fft.fft(a, axis=-1, norm="ortho")),
        dtype,
    )


# ------------------------------------------------------------------ asarray and to_numpy


class TestConversion:
    def test_roundtrip(self, backend: Backend, dtype: DType) -> None:
        a = draw((3, 4), dtype, 7)
        x = backend.asarray(a, dtype)
        back = backend.to_numpy(x)
        assert back.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(back, a.astype(dtype))

    def test_zero_d_stays_zero_d(self, backend: Backend, dtype: DType) -> None:
        assert backend.to_numpy(backend.asarray(np.array(2.5), dtype)).shape == ()

    def test_float64_to_float32_rounds_to_nearest(self, backend: Backend) -> None:
        a = np.random.default_rng(8).standard_normal(50)
        np.testing.assert_array_equal(
            backend.to_numpy(backend.asarray(a, "float32")), a.astype(np.float32)
        )

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("bool", "int64"),
            ("int32", "float64"),
            ("int64", "complex128"),
            ("bool", "float32"),
            ("int64", "int32"),
            ("float32", "complex64"),
        ],
    )
    def test_widening_casts(self, backend: Backend, source: str, target: DType) -> None:
        a = np.array([0, 1, 1, 0]).astype(source)
        back = backend.to_numpy(backend.asarray(a, target))
        assert back.dtype == np.dtype(target)
        np.testing.assert_array_equal(back, a.astype(target))

    @pytest.mark.parametrize(
        ("source", "target"),
        [("complex128", "float64"), ("float64", "int64"), ("int64", "bool"), ("float32", "bool")],
    )
    def test_narrowing_casts_raise(self, backend: Backend, source: str, target: DType) -> None:
        with pytest.raises(TypeError, match="drops information"):
            backend.asarray(np.ones(3, dtype=source), target)

    def test_asarray_takes_numpy_only(self, backend: Backend, dtype: DType) -> None:
        with pytest.raises(TypeError, match=r"numpy\.ndarray"):
            backend.asarray([1.0, 2.0], dtype)
        x = backend.asarray(np.ones(2), dtype)
        with pytest.raises(TypeError, match=r"numpy\.ndarray"):
            backend.asarray(x, dtype)

    def test_no_shared_memory(self, backend: Backend, dtype: DType) -> None:
        a = np.ones(4, dtype=dtype)
        x = backend.asarray(a, dtype)
        a[0] = 7.0
        assert backend.to_numpy(x)[0] == 1.0
        back = backend.to_numpy(x)
        back[1] = 9.0
        assert backend.to_numpy(x)[1] == 1.0


# ------------------------------------------------------------------ pad


class TestPad:
    def test_zero_widths_return_a_copy(self, backend: Backend, dtype: DType) -> None:
        a = draw((3, 4), dtype, 9)
        assert_close(
            backend, backend.pad(backend.asarray(a, dtype), [(0, 0)], [0], "circular"), a, dtype
        )

    def test_circular_width_equal_to_the_length(self, backend: Backend, dtype: DType) -> None:
        a = draw((2, 3), dtype, 9)
        y = backend.pad(backend.asarray(a, dtype), [(3, 3)], [1], "circular")
        assert_close(backend, y, np_pad(a, [(3, 3)], [1], "circular"), dtype)

    def test_zeros_on_an_empty_axis(self, backend: Backend, dtype: DType) -> None:
        y = backend.pad(backend.asarray(np.zeros((2, 0)), dtype), [(1, 2)], [1], "zeros")
        assert_close(backend, y, np.zeros((2, 3)), dtype)

    @pytest.mark.parametrize(
        ("widths", "axes", "mode", "match"),
        [
            ([(1, 1)], [0, 1], "circular", "one \\(before, after\\) pair per axis"),
            ([(1, 1), (1, 1)], [1, -1], "circular", "repeat an axis"),
            ([(-1, 1)], [0], "zeros", "non-negative"),
            ([(4, 0)], [1], "circular", "wrap more than once"),
            ([(1, 1)], [3], "zeros", "out of range"),
            ([(1, 1)], [0], "reflect", "not one of"),
        ],
    )
    def test_rejects(
        self, backend: Backend, dtype: DType, widths: Any, axes: Any, mode: Any, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend.pad(backend.asarray(np.ones((2, 3)), dtype), widths, axes, mode)

    @pytest.mark.parametrize("mode", ["circular", "replicate"])
    def test_rejects_an_empty_axis(self, backend: Backend, dtype: DType, mode: Any) -> None:
        with pytest.raises(ValueError, match="non-empty axis"):
            backend.pad(backend.asarray(np.zeros((2, 0)), dtype), [(1, 1)], [1], mode)

    def test_rejects_integers(self, backend: Backend) -> None:
        with pytest.raises(TypeError, match="floating"):
            backend.pad(backend.asarray(np.ones(3), "int64"), [(1, 1)], [0], "zeros")


# ------------------------------------------------------------------ conv_axes


class TestConvAxes:
    def test_is_a_correlation_m_conv_2d(self, backend: Backend, dtype: DType) -> None:
        """M-CONV-2D: pad by ⌊K/2⌋, then correlate: (w ⋆ a)[n] = Σ_m w[m] ã[n + m - p]."""
        a = draw((6, 5), dtype, 11)
        w = draw((3, 3), dtype, 12)  # not symmetric: a flipped kernel would differ
        y = backend.conv_axes(
            backend.pad(backend.asarray(a, dtype), [(1, 1), (1, 1)], [0, 1], "circular"),
            backend.asarray(w, dtype),
            [0, 1],
        )
        expected = np.zeros((6, 5))
        for n1, n2 in itertools.product(range(6), range(5)):
            for m1, m2 in itertools.product(range(3), range(3)):
                expected[n1, n2] += w[m1, m2] * a[(n1 + m1 - 1) % 6, (n2 + m2 - 1) % 5]
        assert_close(backend, y, expected, dtype)

    def test_output_shape_with_kernel_axes(self, backend: Backend, dtype: DType) -> None:
        """(Nb, Nc, N1, N2) with (L, K, K) gives (Nb, Nc, L, M1, M2)."""
        x = backend.asarray(np.ones((2, 3, 8, 7)), dtype)
        k = backend.asarray(np.ones((4, 3, 3)), dtype)
        assert backend.to_numpy(backend.conv_axes(x, k, [2, 3])).shape == (2, 3, 4, 6, 5)
        k2 = backend.asarray(np.ones((2, 4, 3, 3)), dtype)
        assert backend.to_numpy(backend.conv_axes(x, k2, [2, 3], 2)).shape == (2, 3, 2, 4, 3, 3)

    def test_rejects_complex(self, backend: Backend, dtype: DType) -> None:
        z = backend.asarray(np.ones((5, 5), dtype=complex), COMPLEX_OF[dtype])
        k = backend.asarray(np.ones((3, 3)), dtype)
        with pytest.raises(TypeError, match="one call for each"):
            backend.conv_axes(z, k, [0, 1])

    def test_rejects_mixed_dtypes(self, backend: Backend) -> None:
        x = backend.asarray(np.ones((5, 5)), "float32")
        k = backend.asarray(np.ones((3, 3)), "float64")
        with pytest.raises(TypeError, match="share one dtype"):
            backend.conv_axes(x, k, [0, 1])

    @pytest.mark.parametrize(
        ("x_shape", "k_shape", "axes", "stride", "match"),
        [
            ((5, 5), (3,), [0, 1], 1, "fewer than the 2 filtered axes"),
            ((5, 4, 5), (3, 3), [0, 2], 1, "contiguous"),
            ((5, 5), (3, 3), [1, 0], 1, "contiguous"),
            ((5, 5), (3, 3), [0, 0], 1, "repeat an axis"),
            ((5, 5), (3, 3), [0, 1], 0, "at least 1"),
            ((5, 5), (3, 3), [0, 1], [1, 2, 1], "one per filtered axis"),
            ((2, 5), (3, 3), [0, 1], 1, "smaller than the kernel support"),
            ((5, 5), (3,), [], 1, "at least one axis"),
        ],
    )
    def test_rejects(
        self,
        backend: Backend,
        dtype: DType,
        x_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        axes: list[int],
        stride: Any,
        match: str,
    ) -> None:
        x = backend.asarray(np.ones(x_shape), dtype)
        k = backend.asarray(np.ones(k_shape), dtype)
        with pytest.raises(ValueError, match=match):
            backend.conv_axes(x, k, axes, stride)


# ------------------------------------------------------------------ sparse operations


class TestSparse:
    def test_gather_weighted_padding_entries(self, backend: Backend, dtype: DType) -> None:
        """Rows padded with weight 0 contribute nothing; a NaN read there still gives NaN."""
        x = np.array([1.0, 2.0, np.nan])
        index = np.array([[[0, 1]], [[1, 2]]])
        weight = np.array([[[0.5, 0.0]], [[2.0, 0.0]]])
        y = backend.to_numpy(
            backend.gather_weighted(
                backend.asarray(x, dtype),
                0,
                index_array(backend, index),
                backend.asarray(weight, dtype),
            )
        )
        assert y[0, 0] == 0.5
        assert np.isnan(y[1, 0])

    def test_gather_weighted_rejects(self, backend: Backend, dtype: DType) -> None:
        x = backend.asarray(np.ones((3, 4)), dtype)
        index = index_array(backend, np.zeros((2, 1, 3), dtype=int))
        with pytest.raises(ValueError, match=r"one\s+shape \(P, Q, M\)"):
            backend.gather_weighted(x, 1, index, backend.asarray(np.ones((2, 1, 2)), dtype))
        with pytest.raises(TypeError, match="int32 or int64"):
            backend.gather_weighted(
                x,
                1,
                backend.asarray(np.zeros((2, 1, 3)), dtype),
                backend.asarray(np.ones((2, 1, 3)), dtype),
            )
        other: DType = "float64" if dtype == "float32" else "float32"
        weight = backend.asarray(np.ones((2, 1, 3)), other)
        with pytest.raises(TypeError, match="share one dtype"):
            backend.gather_weighted(x, 1, index, weight)

    def test_segment_sum_empty_segment_and_zero_segments(
        self, backend: Backend, dtype: DType
    ) -> None:
        x = backend.asarray(np.ones((2, 8)), dtype)
        y = backend.to_numpy(backend.segment_sum(x, 1, index_array(backend, SEG_IDS), 6))
        np.testing.assert_array_equal(y[0], [2.0, 2.0, 0.0, 3.0, 1.0, 0.0])
        empty = backend.segment_sum(
            backend.asarray(np.ones((2, 0)), dtype),
            1,
            index_array(backend, np.zeros(0, dtype=int)),
            0,
        )
        assert backend.to_numpy(empty).shape == (2, 0)

    def test_segment_sum_rejects(self, backend: Backend, dtype: DType) -> None:
        x = backend.asarray(np.ones((2, 8)), dtype)
        with pytest.raises(ValueError, match="non-negative"):
            backend.segment_sum(x, 1, index_array(backend, SEG_IDS), -1)
        with pytest.raises(ValueError, match="one id per entry"):
            backend.segment_sum(x, 0, index_array(backend, SEG_IDS), 5)

    def test_scatter_fill_and_roundtrip(self, backend: Backend, dtype: DType) -> None:
        v = draw((2, 5), dtype, 13)
        index = [index_array(backend, i) for i in GRID_INDEX]
        grid = backend.scatter_to_grid(backend.asarray(v, dtype), index, (3, 4))
        assert_close(backend, grid, np_scatter(v, GRID_INDEX, (3, 4), np.nan), dtype)
        assert np.isnan(backend.to_numpy(grid)).sum() == 2 * (12 - 5)
        assert_close(backend, backend.gather_from_grid(grid, index), v, dtype)

    def test_scatter_complex_fill(self, backend: Backend, dtype: DType) -> None:
        cdtype = COMPLEX_OF[dtype]
        v = np.array([1 + 2j, 3 - 1j])
        grid = backend.to_numpy(
            backend.scatter_to_grid(
                backend.asarray(v, cdtype), [index_array(backend, [0, 2])], (3,)
            )
        )
        assert grid.dtype == np.dtype(cdtype)
        assert np.isnan(grid[1].real)
        assert grid[1].imag == 0.0
        np.testing.assert_array_equal(grid[[0, 2]], v.astype(cdtype))

    def test_no_gradient_from_the_fill(self, backend: Backend, dtype: DType) -> None:
        xp = backend.xp
        index = [index_array(backend, i) for i in GRID_INDEX]
        v = draw((5,), dtype, 14)

        def f(a: Array) -> Array:
            grid = backend.scatter_to_grid(a, index, (3, 4))
            return xp.sum(backend.gather_from_grid(grid, index) ** 2)

        _, grad = backend.value_and_grad(f)(backend.asarray(v, dtype))
        assert_close(backend, grad, 2 * v, dtype)

    def test_grid_rejects(self, backend: Backend, dtype: DType) -> None:
        v = backend.asarray(np.ones((2, 5)), dtype)
        index = [index_array(backend, i) for i in GRID_INDEX]
        with pytest.raises(ValueError, match="one per grid axis"):
            backend.scatter_to_grid(v, index, (3, 4, 2))
        with pytest.raises(ValueError, match="values has 3"):
            backend.scatter_to_grid(backend.asarray(np.ones((2, 3)), dtype), index, (3, 4))
        with pytest.raises(ValueError, match="must all be \\(n,\\)"):
            backend.scatter_to_grid(v, [index[0], index_array(backend, [0, 1])], (3, 4))
        with pytest.raises(ValueError, match="0-d"):
            backend.scatter_to_grid(backend.asarray(np.array(1.0), dtype), index[:1], (3,))
        with pytest.raises(ValueError, match="one per grid axis"):
            backend.gather_from_grid(backend.asarray(np.ones(4), dtype), index)


# ------------------------------------------------------------------ complex values


class TestComplexPairs:
    def test_dtypes_and_values(self, backend: Backend, dtype: DType) -> None:
        cdtype = COMPLEX_OF[dtype]
        z = unpairs(draw((3, 2, 2), dtype, 15))
        r = backend.complex_to_real_pairs(backend.asarray(z, cdtype))
        assert backend.to_numpy(r).dtype == np.dtype(dtype)
        assert_close(backend, r, pairs(z), dtype)
        back = backend.to_numpy(backend.real_pairs_to_complex(r))
        assert back.dtype == np.dtype(cdtype)
        np.testing.assert_array_equal(back, z.astype(cdtype))

    def test_rejects(self, backend: Backend, dtype: DType) -> None:
        cdtype = COMPLEX_OF[dtype]
        with pytest.raises(TypeError, match="complex64 or complex128"):
            backend.complex_to_real_pairs(backend.asarray(np.ones(3), dtype))
        with pytest.raises(TypeError, match="one call for each"):
            backend.real_pairs_to_complex(backend.asarray(np.ones((3, 2)), cdtype))
        with pytest.raises(ValueError, match="length 2"):
            backend.real_pairs_to_complex(backend.asarray(np.ones((3, 3)), dtype))


# ------------------------------------------------------------------ differentiation


class TestDifferentiation:
    def test_stop_gradient(self, backend: Backend, dtype: DType) -> None:
        """f(x) = Σ sg(x)·x² has gradient 2·x·x, not 3·x², and second derivative 2·x."""
        xp = backend.xp
        x = draw((4,), dtype, 16)

        def f(a: Array) -> Array:
            return xp.sum(backend.stop_gradient(a) * a * a)

        value, grad = backend.value_and_grad(f)(backend.asarray(x, dtype))
        assert_close(backend, value, np.asarray((x**3).sum()), dtype)
        assert_close(backend, grad, 2 * x * x, dtype)
        v = draw((4,), dtype, 17)
        hv = backend.hvp(f, backend.asarray(x, dtype), backend.asarray(v, dtype))
        assert_close(backend, hv, 2 * x * v, dtype)
        assert_close(backend, backend.stop_gradient(backend.asarray(x, dtype)), x, dtype)

    def test_value_and_grad_extra_arguments(self, backend: Backend, dtype: DType) -> None:
        xp = backend.xp
        x, y = draw((3,), dtype, 18), draw((3,), dtype, 19)
        value, grad = backend.value_and_grad(lambda a, b: xp.sum(a * a * b))(
            backend.asarray(x, dtype), backend.asarray(y, dtype)
        )
        assert_close(backend, value, np.asarray((x * x * y).sum()), dtype)
        assert_close(backend, grad, 2 * x * y, dtype)

    def test_vjp_and_jvp_of_several_primals(self, backend: Backend, dtype: DType) -> None:
        x, y = draw((3,), dtype, 18), draw((3,), dtype, 19)
        tx, ty, c = draw((3,), dtype, 20), draw((3,), dtype, 21), draw((3,), dtype, 22)
        a, b = backend.asarray(x, dtype), backend.asarray(y, dtype)
        out, vjp_fn = backend.vjp(lambda p, q: p * q, a, b)
        assert_close(backend, out, x * y, dtype)
        ga, gb = vjp_fn(backend.asarray(c, dtype))
        assert_close(backend, ga, c * y, dtype)
        assert_close(backend, gb, c * x, dtype)
        _, tangent = backend.jvp(
            lambda p, q: p * q, (a, b), (backend.asarray(tx, dtype), backend.asarray(ty, dtype))
        )
        assert_close(backend, tangent, tx * y + x * ty, dtype)

    def test_rejects_complex_and_non_scalar(self, backend: Backend, dtype: DType) -> None:
        xp = backend.xp
        cdtype = COMPLEX_OF[dtype]
        z = backend.asarray(np.ones(3, dtype=complex), cdtype)
        x = backend.asarray(np.ones(3), dtype)
        with pytest.raises(TypeError):
            backend.value_and_grad(lambda a: xp.sum(a))(z)
        with pytest.raises(TypeError):
            backend.value_and_grad(lambda a: xp.sum(a * (1 + 1j)))(x)
        with pytest.raises(ValueError, match="0-d real array"):
            backend.value_and_grad(lambda a: a * a)(x)
        with pytest.raises(TypeError):
            backend.vjp(lambda a: a, z)
        with pytest.raises(TypeError):
            backend.vjp(lambda a: a * (1 + 1j), x)
        with pytest.raises(TypeError):
            backend.jvp(lambda a: a * (1 + 1j), (x,), (x,))
        with pytest.raises(ValueError, match="one tangent"):
            backend.jvp(lambda a: a, (x,), (x, x))
        with pytest.raises(ValueError, match="0-d real array"):
            backend.hvp(lambda a: a, x, x)

    def test_jit_keeps_values(self, backend: Backend, dtype: DType) -> None:
        xp = backend.xp
        x = draw((5,), dtype, 23)

        def f(a: Array, scale: float) -> Array:
            return xp.sin(a) * scale

        jitted = backend.jit(f, static_argnames=("scale",))
        assert_close(backend, jitted(backend.asarray(x, dtype), scale=2.0), 2 * np.sin(x), dtype)


# ------------------------------------------------------------------ randomness


class TestRandom:
    def test_same_key_same_values(self, backend: Backend, dtype: DType) -> None:
        key = backend.make_key(1234)
        a = backend.to_numpy(backend.random_normal(key, (3, 50), dtype))
        b = backend.to_numpy(backend.random_normal(backend.make_key(1234), (3, 50), dtype))
        c = backend.to_numpy(backend.random_normal(backend.make_key(1235), (3, 50), dtype))
        assert a.dtype == np.dtype(dtype)
        assert a.shape == (3, 50)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            a, backend.to_numpy(backend.random_normal(key, (3, 50), dtype))
        )
        assert not np.array_equal(a, c)

    def test_standard_normal(self, backend: Backend, dtype: DType) -> None:
        s = backend.to_numpy(backend.random_normal(backend.make_key(7), (20000,), dtype))
        assert abs(s.mean()) < 0.03  # 4 standard errors of the mean
        assert abs(s.std() - 1.0) < 0.03

    @pytest.mark.parametrize("seed", [-1, 2**63, 1.5, True, "1"])
    def test_rejects_bad_seeds(self, backend: Backend, seed: Any) -> None:
        with pytest.raises(ValueError, match="seed"):
            backend.make_key(seed)

    @pytest.mark.parametrize("bad", ["complex64", "int32", "bool"])
    def test_rejects_non_real_dtypes(self, backend: Backend, bad: DType) -> None:
        with pytest.raises(TypeError):
            backend.random_normal(backend.make_key(0), (3,), bad)

    def test_rejects_foreign_keys(self, backend: Backend, dtype: DType) -> None:
        with pytest.raises(TypeError, match="make_key"):
            backend.random_normal(0, (3,), dtype)


# ------------------------------------------------------------------ optimization

A_LSQ = np.random.default_rng(30).standard_normal((20, 5))
B_LSQ = np.random.default_rng(31).standard_normal(20)


def _least_squares(backend: Backend, dtype: DType) -> Callable[[Array], Array]:
    xp = backend.xp
    a = backend.asarray(rounded(A_LSQ, dtype), dtype)
    b = backend.asarray(rounded(B_LSQ, dtype), dtype)
    return lambda x: xp.sum((xp.matmul(a, x) - b) ** 2)


def _smooth(backend: Backend, dtype: DType) -> Callable[[Array], Array]:
    xp = backend.xp
    a = backend.asarray(rounded(A_LSQ, dtype), dtype)
    b = backend.asarray(rounded(B_LSQ, dtype), dtype)
    return lambda x: xp.sum(xp.sin(xp.matmul(a, x) - b) ** 2) + 0.1 * xp.sum(x**4)


class TestLbfgs:
    def test_solves_least_squares(self, backend: Backend, dtype: DType) -> None:
        x0_np = np.zeros(5)
        x0 = backend.asarray(x0_np, dtype)
        result = backend.lbfgs(
            _least_squares(backend, dtype),
            x0,
            max_iter=50,
            history_size=10,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )
        a, b = rounded(A_LSQ, dtype), rounded(B_LSQ, dtype)
        solution = np.linalg.lstsq(a, b, rcond=None)[0]
        rtol = TOLERANCES[dtype][0]
        np.testing.assert_allclose(
            backend.to_numpy(result.x), solution, rtol=100 * rtol, atol=100 * rtol
        )  # an optimum is reached to √ε of the loss: rtol of §5.2 times the condition number
        np.testing.assert_array_equal(backend.to_numpy(x0), x0_np)
        losses = backend.to_numpy(result.losses)
        assert losses.dtype == np.dtype(dtype)
        assert 1 <= losses.shape[0] <= 50
        assert np.all(np.diff(losses) <= 0)

    def test_losses_follow_the_iterates(self, backend: Backend, dtype: DType) -> None:
        fun = _smooth(backend, dtype)
        result = backend.lbfgs(
            fun,
            backend.asarray(np.zeros(5), dtype),
            max_iter=4,
            history_size=3,
            tolerance_grad=1e-30,
            tolerance_change=1e-30,
        )
        losses = backend.to_numpy(result.losses)
        # At most max_iter iterations; the backend's evaluation budget may stop it earlier.
        assert 1 <= losses.shape[0] <= 4
        assert np.all(np.diff(losses) <= 0)
        assert_close(backend, fun(result.x), np.asarray(losses[-1], dtype=np.float64), dtype)

    def test_resume_continues_the_run(self, backend: Backend, dtype: DType) -> None:
        fun = _smooth(backend, dtype)
        settings: dict[str, Any] = {
            "history_size": 3,
            "tolerance_grad": 1e-30,
            "tolerance_change": 1e-30,
        }
        x0 = backend.asarray(np.zeros(5), dtype)
        whole = backend.lbfgs(fun, x0, max_iter=8, **settings)
        first = backend.lbfgs(fun, x0, max_iter=4, **settings)
        second = backend.lbfgs(fun, first.x, max_iter=4, state=first.state, **settings)
        assert_close(backend, second.x, backend.to_numpy(whole.x).astype(np.float64), dtype)
        resumed = np.concatenate([backend.to_numpy(first.losses), backend.to_numpy(second.losses)])
        assert_close(backend, whole.losses, resumed.astype(np.float64), dtype)

    def test_stops_at_a_stationary_point(self, backend: Backend, dtype: DType) -> None:
        xp = backend.xp
        result = backend.lbfgs(
            lambda x: xp.sum(x * x),
            backend.asarray(np.zeros(3), dtype),
            max_iter=10,
            history_size=5,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )
        assert backend.to_numpy(result.losses).shape == (0,)

    def test_rejects(self, backend: Backend, dtype: DType) -> None:
        fun = _least_squares(backend, dtype)
        x0 = backend.asarray(np.zeros(5), dtype)
        common: dict[str, Any] = {"tolerance_grad": 1e-7, "tolerance_change": 1e-9}
        with pytest.raises(ValueError, match="at least 1"):
            backend.lbfgs(fun, x0, max_iter=0, history_size=5, **common)
        with pytest.raises(ValueError, match="at least 1"):
            backend.lbfgs(fun, x0, max_iter=5, history_size=0, **common)
        with pytest.raises(TypeError, match="state"):
            backend.lbfgs(fun, x0, max_iter=5, history_size=5, state={"n_iter": 3}, **common)
        with pytest.raises(TypeError):
            backend.lbfgs(
                fun,
                backend.asarray(np.zeros(5), COMPLEX_OF[dtype]),
                max_iter=5,
                history_size=5,
                **common,
            )


def test_deterministic_can_be_switched(backend: Backend) -> None:
    backend.deterministic(True)
    backend.deterministic(False)
