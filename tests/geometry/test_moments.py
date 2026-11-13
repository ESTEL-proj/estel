"""The internal moment functions of the DataClass (task T2.4, issue #16; Part II, §12.4;
maths.md M-NOT, M-MEAN, M-MOM, M-S1): the safe modulus, STL's regularised division, the means
over valid pixels and the standardisation, against NumPy float64, on the run's backend and dtype.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel.backend._protocol import Backend, DType
from estel.data._moments import cov_valid, divide, mean_valid, modulus, standardize

pytestmark = pytest.mark.maths("M-MEAN", "M-MOM")
RTOL = {"float32": 1e-5, "float64": 1e-12}
COMPLEX: dict[str, DType] = {"float32": "complex64", "float64": "complex128"}


def _complex(rng: np.random.Generator, shape: tuple[int, ...]) -> NDArray[np.complex128]:
    out: NDArray[np.complex128] = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return out


def _weights(rng: np.random.Generator, shape: tuple[int, ...]) -> NDArray[np.float64]:
    w = (rng.random(shape) > 0.3).astype(np.float64)
    w[0, 0] = 1.0
    return w


def test_modulus_is_abs(backend: Backend, dtype: DType) -> None:
    rng = np.random.default_rng(0)
    z = _complex(rng, (3, 8))
    z[0, :3] = 0.0
    for values, kind in ((z, COMPLEX[dtype]), (z.real, dtype)):
        got = backend.to_numpy(modulus(backend.asarray(values, kind)))
        assert str(got.dtype) == dtype
        np.testing.assert_allclose(got, np.abs(values), rtol=RTOL[dtype], atol=0)


def test_modulus_has_finite_derivatives_at_zero(backend: Backend, dtype: DType) -> None:
    """The double ``where`` of Part I, §3.2.3: gradient and Hessian-vector product finite, and
    zero, where the argument is exactly 0."""
    x = np.array([0.0, 0.0, 1.5, -2.0, 0.0, 0.5])

    def f(a: Any) -> Any:
        xp = backend.xp
        pairs = xp.reshape(a, (3, 2))
        return xp.sum(modulus(backend.real_pairs_to_complex(pairs)))

    value, grad = backend.value_and_grad(f)(backend.asarray(x, dtype))
    grad = backend.to_numpy(grad)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[:2], 0.0)
    hessian_v = backend.to_numpy(
        backend.hvp(f, backend.asarray(x, dtype), backend.asarray(x, dtype))
    )
    assert np.isfinite(hessian_v).all()
    np.testing.assert_allclose(
        float(backend.to_numpy(value)), 3.0, rtol=RTOL[dtype]
    )  # 0 + 2.5 + 0.5


def test_divide_is_stls_regularised_division(backend: Backend, dtype: DType) -> None:
    rng = np.random.default_rng(1)
    z = _complex(rng, (4, 5))
    y = np.abs(z)
    got = divide(backend.asarray(z, COMPLEX[dtype]), backend.asarray(y, dtype), 1e-8, 0.5)
    np.testing.assert_allclose(backend.to_numpy(got), z / (y + 1e-8) ** 0.5, rtol=RTOL[dtype])
    with pytest.raises(TypeError, match="modulus"):
        divide(backend.asarray(z, COMPLEX[dtype]), backend.asarray(z, COMPLEX[dtype]))


def test_mean_and_cov_over_valid_pixels(backend: Backend, dtype: DType) -> None:
    rng = np.random.default_rng(2)
    x, y = _complex(rng, (2, 3, 6, 7)), _complex(rng, (2, 3, 6, 7))
    w = _weights(rng, (6, 7))
    valid = w > 0
    weights = backend.asarray(w, dtype)
    kind = COMPLEX[dtype]
    got_mean = mean_valid(backend.asarray(x, kind), weights, float(w.sum()), (-2, -1))
    got_cov = cov_valid(
        backend.asarray(x, kind), backend.asarray(y, kind), weights, w.sum(), (-2, -1)
    )
    np.testing.assert_allclose(backend.to_numpy(got_mean), x[..., valid].mean(-1), rtol=RTOL[dtype])
    expected = (x * np.conj(y))[..., valid].mean(-1)
    np.testing.assert_allclose(backend.to_numpy(got_cov), expected, rtol=RTOL[dtype])
    real = cov_valid(
        backend.asarray(x.real, dtype), backend.asarray(y.real, dtype), weights, w.sum(), (-2, -1)
    )
    np.testing.assert_allclose(
        backend.to_numpy(real), (x.real * y.real)[..., valid].mean(-1), rtol=10 * RTOL[dtype]
    )


def test_standardize_as_stl(backend: Backend, dtype: DType) -> None:
    """μ = ⟨x⟩, sigma² = ⟨(x - μ)²⟩ over valid pixels, population normalisation, per map (M-MOM)."""
    rng = np.random.default_rng(3)
    x = 2.0 + 3.0 * rng.standard_normal((2, 3, 8, 8))
    w = _weights(rng, (8, 8))
    valid = w > 0
    out = standardize(backend.asarray(x, dtype), backend.asarray(w, dtype), w.sum(), (-2, -1))
    mean = x[..., valid].mean(-1)
    std = np.sqrt(((x - mean[..., None, None])[..., valid] ** 2).mean(-1))
    np.testing.assert_allclose(backend.to_numpy(out.mean), mean, rtol=RTOL[dtype])
    np.testing.assert_allclose(backend.to_numpy(out.std), std, rtol=RTOL[dtype])
    np.testing.assert_allclose(
        backend.to_numpy(out.x),
        (x - mean[..., None, None]) / std[..., None, None],
        rtol=10 * RTOL[dtype],
        atol=10 * RTOL[dtype],
    )
