"""DataClasses are pytrees whose only leaf is the array (Part I, §2.3 and §3.1, rule 9; T2.9).

Run on the backend and dtype of the test run. ``backend.jit`` is ``jax.jit`` on JAX and the
identity on PyTorch, so the tests that use it check both: JAX traces the DataClass, PyTorch
behaves as before. The tests that name a backend's tree utilities run only in its run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from estel import Data1D, Data2D, DataClass
from estel.backend._protocol import Backend, DType
from estel.data._base import _StaticMask
from tests._backends import backend_name

jax_only = pytest.mark.skipif(backend_name() != "jax", reason="names JAX's tree utilities")
torch_only = pytest.mark.skipif(backend_name() != "torch", reason="names PyTorch's pytrees")


def _field(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def _mask(shape: tuple[int, ...], *missing: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for position in missing:
        mask[position] = True
    return mask


def _same_metadata(a: DataClass, b: DataClass) -> bool:
    masks_equal = (a.mask is None and b.mask is None) or (
        a.mask is not None and b.mask is not None and np.array_equal(a.mask, b.mask)
    )
    return type(a) is type(b) and masks_equal and (a.pbc, a.backend) == (b.pbc, b.backend)


CASES = [
    pytest.param(Data2D, (2, 3, 8, 6), (1, 2), id="Data2D-masked"),
    pytest.param(Data2D, (8, 6), None, id="Data2D"),
    pytest.param(Data1D, (2, 1, 16), (5,), id="Data1D-masked"),
]


@pytest.mark.parametrize(("cls", "shape", "missing"), CASES)
def test_jit_takes_a_dataclass_and_returns_one(
    backend: Backend, dtype: DType, cls: type[DataClass], shape: tuple[int, ...], missing: Any
) -> None:
    values = _field(shape)
    spatial = shape[-cls.ndim :]
    mask = None if missing is None else _mask(spatial, missing)
    data = cls(values, mask=mask, pbc=False)
    out = backend.jit(lambda d: d.with_array(2 * d.array))(data)
    assert _same_metadata(out, data)
    np.testing.assert_allclose(
        backend.to_numpy(out.array), 2 * backend.to_numpy(data.array), rtol=0, atol=0
    )


def test_gradients_reach_the_array_through_jit(backend: Backend, dtype: DType) -> None:
    xp = backend.xp
    data = Data2D(_field((4, 4)), mask=_mask((4, 4), (0, 0)), pbc=True)
    square = backend.jit(lambda d: d.with_array(d.array * d.array))
    _, grad = backend.value_and_grad(lambda a: xp.sum(square(data.with_array(a)).array))(data.array)
    np.testing.assert_allclose(backend.to_numpy(grad), 2 * backend.to_numpy(data.array), rtol=1e-6)


def test_static_masks_compare_by_value() -> None:
    a, b = _mask((4, 4), (0, 1)), _mask((4, 4), (0, 1))
    assert _StaticMask(a) == _StaticMask(b)
    assert hash(_StaticMask(a)) == hash(_StaticMask(b))
    assert _StaticMask(a) != _StaticMask(_mask((4, 4), (1, 0)))
    assert _StaticMask(a) != _StaticMask(_mask((2, 8), (0, 1)))


def test_static_mask_hashes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hash is computed when the wrapper is made, never again."""
    calls: list[int] = []
    packbits = np.packbits

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return packbits(*args, **kwargs)

    monkeypatch.setattr(np, "packbits", counting)
    static = _StaticMask(_mask((4, 4), (0, 1)))
    assert len(calls) == 1
    first = hash(static)
    assert all(hash(static) == first for _ in range(3))
    assert len(calls) == 1


def test_static_mask_cannot_be_written() -> None:
    """A writable mask is copied and frozen, so the hash cannot go stale; the caller's
    array is left as it was."""
    given = _mask((4, 4), (0, 1))
    static = _StaticMask(given)
    assert static.mask is not given
    assert given.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        static.mask[2, 2] = True
    given[2, 2] = True  # writing the caller's array does not reach the wrapper
    assert not static.mask[2, 2]
    assert static == _StaticMask(_mask((4, 4), (0, 1)))


def test_static_mask_copies_a_read_only_view_of_a_writable_array() -> None:
    base = _mask((4, 4), (0, 1))
    view = base[:]
    view.flags.writeable = False
    static = _StaticMask(view)
    base[3, 3] = True
    assert not static.mask[3, 3]


def test_static_mask_keeps_the_mask_of_a_dataclass() -> None:
    """A DataClass mask is read-only and owns its data: wrapped without a copy."""
    data = Data2D(_field((4, 4)), mask=_mask((4, 4), (0, 0)), pbc=True)
    assert data.mask is not None
    assert _StaticMask(data.mask).mask is data.mask
    with pytest.raises(ValueError, match="read-only"):
        data.mask[1, 1] = True


@jax_only
def test_jax_compiles_once_per_mask(dtype: DType) -> None:
    """Equal masks reuse the compiled function; a new mask is new static data, so it compiles
    again (a NumPy mask as such would make JAX fail at the second call)."""
    import jax

    traces: list[int] = []

    @jax.jit
    def double(d: Data2D) -> DataClass:
        traces.append(1)
        return d.with_array(2 * d.array)

    first = Data2D(_field((4, 4), 1), mask=_mask((4, 4), (0, 0)), pbc=True)
    same = Data2D(_field((4, 4), 2), mask=_mask((4, 4), (0, 0)), pbc=True)
    other = Data2D(_field((4, 4), 3), mask=_mask((4, 4), (1, 1)), pbc=True)
    double(first)
    double(same)
    assert len(traces) == 1
    assert double(other).mask is other.mask
    assert len(traces) == 2


@jax_only
@pytest.mark.parametrize(("cls", "shape", "missing"), CASES)
def test_vmap_takes_a_dataclass_and_returns_one(
    backend: Backend, dtype: DType, cls: type[DataClass], shape: tuple[int, ...], missing: Any
) -> None:
    import jax

    spatial = shape[-cls.ndim :]
    mask = None if missing is None else _mask(spatial, missing)
    data = cls(_field(shape), mask=mask, pbc=True)
    scale = backend.asarray(np.arange(1.0, data.n_batch + 1), dtype)
    out = jax.vmap(lambda d, s: d.with_array(s * d.array))(data, scale)
    assert _same_metadata(out, data)
    assert out.shape == data.shape
    expected = backend.to_numpy(data.array) * np.arange(1.0, data.n_batch + 1).reshape(
        (-1,) + (1,) * (len(data.shape) - 1)
    )
    np.testing.assert_allclose(backend.to_numpy(out.array), expected, rtol=1e-6)


@jax_only
def test_jax_tree_util_sees_the_array_as_the_only_leaf() -> None:
    import jax

    data = Data1D(_field((2, 8)), mask=_mask((8,), (3,)), pbc=False)
    leaves, treedef = jax.tree_util.tree_flatten(data)
    assert len(leaves) == 1
    assert leaves[0] is data.array
    assert _same_metadata(jax.tree_util.tree_unflatten(treedef, leaves), data)


@torch_only
def test_torch_pytree_sees_the_array_as_the_only_leaf() -> None:
    import torch.utils._pytree as pytree

    data = Data2D(_field((3, 5, 5)), mask=_mask((5, 5), (2, 2)), pbc=True)
    leaves, spec = pytree.tree_flatten(data)
    assert len(leaves) == 1
    assert leaves[0] is data.array
    assert _same_metadata(pytree.tree_unflatten(leaves, spec), data)
    doubled = pytree.tree_map(lambda x: 2 * x, data)
    assert type(doubled) is Data2D
    assert _same_metadata(doubled, data)
