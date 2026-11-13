"""Unit tests of the DataClasses (Part II, §12.4; maths.md M-DATA-2D, M-DATA-1D; T2.1).

Run on the backend and dtype of the test run (``backend`` and ``dtype`` fixtures).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import estel
from estel import Data1D, Data2D, DataClass
from estel._config import default_dtype
from estel.backend._protocol import Backend, DType


def _np(backend: Backend, x: Any) -> np.ndarray:
    return np.asarray(backend.to_numpy(x))


def _field(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


# ---------------------------------------------------------------------- shapes


@pytest.mark.parametrize(
    ("cls", "shape", "promoted"),
    [
        (Data2D, (8, 6), (1, 1, 8, 6)),
        (Data2D, (3, 8, 6), (1, 3, 8, 6)),
        (Data2D, (2, 3, 8, 6), (2, 3, 8, 6)),
        (Data1D, (16,), (1, 1, 16)),
        (Data1D, (3, 16), (1, 3, 16)),  # read as channels, as in STL (OP-29)
        (Data1D, (2, 3, 16), (2, 3, 16)),
    ],
)
def test_arrays_are_promoted_to_batch_channel_spatial(
    cls: type[DataClass], shape: tuple[int, ...], promoted: tuple[int, ...], backend: Backend
) -> None:
    field = _field(shape)
    data = cls(field, pbc=True)
    assert data.shape == promoted
    assert (data.n_batch, data.n_channels) == promoted[:2]
    np.testing.assert_allclose(
        _np(backend, data.array), field.reshape(promoted), rtol=1e-6, atol=1e-6
    )
    assert data._spatial_axes() == tuple(range(2, 2 + cls.ndim))


@pytest.mark.parametrize(
    ("cls", "shape"), [(Data2D, (8,)), (Data2D, (1, 1, 1, 8, 8)), (Data1D, (1, 1, 1, 8))]
)
def test_other_numbers_of_axes_are_rejected(cls: type[DataClass], shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="accepted shapes"):
        cls(np.zeros(shape), pbc=True)


def test_empty_axes_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty axis"):
        Data2D(np.zeros((0, 4, 4)), pbc=True)


def test_ndim_of_each_geometry() -> None:
    assert (Data1D.ndim, Data2D.ndim) == (1, 2)
    assert (Data1D._nan_threshold, Data2D._nan_threshold) == (0.5, 0.33)  # M-DOWN-MASK, OP-12


# ---------------------------------------------------------------------- arrays and dtypes


def test_numpy_input_gets_the_default_dtype(backend: Backend) -> None:
    data = Data2D(np.ones((4, 4), dtype=np.float64), pbc=True)
    assert data.array.dtype == getattr(backend.xp, default_dtype())
    assert data.backend == estel.get_backend()


def test_integer_input_is_converted(backend: Backend) -> None:
    data = Data1D(np.arange(8), pbc=False)
    np.testing.assert_array_equal(_np(backend, data.array)[0, 0], np.arange(8))


def test_backend_arrays_keep_their_dtype(backend: Backend, dtype: DType) -> None:
    array = backend.asarray(_field((4, 4)), dtype)
    data = Data2D(array, pbc=True)
    assert data.array.dtype == getattr(backend.xp, dtype)


def test_complex_data_are_rejected(backend: Backend, dtype: DType) -> None:
    with pytest.raises(TypeError, match="real"):
        Data2D(np.ones((4, 4), dtype=complex), pbc=True)
    complex_dtype: DType = "complex64" if dtype == "float32" else "complex128"
    with pytest.raises(TypeError, match="float32 or float64"):
        Data2D(backend.asarray(np.ones((4, 4)), complex_dtype), pbc=True)


def test_none_and_strings_are_rejected() -> None:
    with pytest.raises(TypeError, match="needs an array"):
        Data2D(None, pbc=True)
    with pytest.raises(TypeError, match="real numbers"):
        Data2D(np.array([["a", "b"]]), pbc=True)


# ---------------------------------------------------------------------- pbc


@pytest.mark.parametrize("pbc", [None, 1, "yes"])
def test_pbc_must_be_an_explicit_boolean(pbc: Any) -> None:
    with pytest.raises(TypeError, match="pbc must be True"):
        Data2D(np.zeros((4, 4)), pbc=pbc)


def test_pbc_has_no_default() -> None:
    with pytest.raises(TypeError):
        Data2D(np.zeros((4, 4)))  # type: ignore[call-arg]


@pytest.mark.parametrize("pbc", [False, True, np.bool_(True)])
def test_pbc_is_stored_as_a_bool(pbc: Any) -> None:
    data = Data2D(np.zeros((4, 4)), pbc=pbc)
    assert data.pbc is bool(pbc)


def test_the_mother_class_has_no_geometry() -> None:
    with pytest.raises(TypeError, match=r"Data1D or estel\.Data2D"):
        DataClass(np.zeros((4, 4)), pbc=True)


# ---------------------------------------------------------------------- masks


def test_no_missing_value_means_no_mask() -> None:
    assert Data2D(_field((4, 4)), pbc=True).mask is None


def test_nan_become_the_mask() -> None:
    expected = np.zeros((4, 6), dtype=bool) | (np.arange(6) == 2)
    data = Data2D(np.where(expected, np.nan, _field((4, 6))), pbc=False)
    assert data.mask is not None
    np.testing.assert_array_equal(data.mask, expected)
    assert data.mask.dtype == np.bool_


def test_nan_of_a_single_map_in_full_shape_become_the_mask() -> None:
    expected = np.arange(8) == 5
    data = Data1D(np.where(expected, np.nan, _field((8,)))[None, None], pbc=True)
    assert data.mask is not None
    np.testing.assert_array_equal(data.mask, expected)


def test_nan_in_batched_data_need_an_explicit_mask() -> None:
    field = _field((2, 4, 4))
    field_nan = np.where(np.arange(4) == 1, np.nan, field)
    with pytest.raises(ValueError, match="OP-26"):
        Data2D(field_nan, pbc=True)
    mask = np.zeros((4, 4), dtype=bool) | (np.arange(4) == 1)
    data = Data2D(field_nan, mask=mask, pbc=True)
    assert data.mask is not None
    np.testing.assert_array_equal(data.mask, mask)


def test_explicit_mask_without_nan(backend: Backend) -> None:
    mask = np.arange(8) < 2
    data = Data1D(_field((3, 8)), mask=mask, pbc=False)
    assert data.mask is not None
    np.testing.assert_array_equal(data.mask, mask)
    from_backend = Data1D(_field((3, 8)), mask=backend.asarray(mask, "bool"), pbc=False)
    assert from_backend.mask is not None
    np.testing.assert_array_equal(from_backend.mask, mask)


def test_masks_must_be_boolean_maps_of_the_spatial_shape() -> None:
    with pytest.raises(TypeError, match="boolean"):
        Data2D(np.zeros((4, 4)), mask=np.zeros((4, 4)), pbc=True)
    with pytest.raises(ValueError, match="spatial shape"):
        Data2D(np.zeros((2, 4, 4)), mask=np.zeros((2, 4, 4), dtype=bool), pbc=True)


def test_nan_outside_the_mask_are_rejected() -> None:
    field = np.where(np.arange(4) == 3, np.nan, _field((4, 4)))
    with pytest.raises(ValueError, match="NaN outside the mask"):
        Data2D(field, mask=np.zeros((4, 4), dtype=bool), pbc=True)


def test_all_missing_is_rejected() -> None:
    with pytest.raises(ValueError, match="OP-3"):
        Data2D(np.full((4, 4), np.nan), pbc=True)
    with pytest.raises(ValueError, match="OP-3"):
        Data1D(np.zeros(4), mask=np.ones(4, dtype=bool), pbc=True)


def test_the_mask_is_a_read_only_copy() -> None:
    mask = np.arange(8) == 0
    data = Data1D(np.zeros(8), mask=mask, pbc=True)
    assert data.mask is not None
    with pytest.raises(ValueError, match="read-only"):
        data.mask[0] = False
    mask[1] = True
    assert not data.mask[1]


# ---------------------------------------------------------------------- immutability


def test_data_are_immutable() -> None:
    data = Data2D(np.zeros((4, 4)), pbc=True)
    with pytest.raises(AttributeError, match="immutable"):
        data.pbc = False  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        data._array = None
    with pytest.raises(AttributeError):
        del data._pbc


def test_with_array_replaces_the_array_only(backend: Backend, dtype: DType) -> None:
    mask = np.arange(8) == 0
    data = Data1D(_field((2, 8)), mask=mask, pbc=False)
    new = backend.asarray(np.ones(data.shape), dtype)
    replaced = data.with_array(new)
    assert type(replaced) is Data1D
    assert replaced.array is new
    assert replaced.mask is data.mask
    assert replaced.pbc is False
    assert replaced.backend == data.backend
    same = data.with_array(data.array)
    np.testing.assert_array_equal(_np(backend, same.array), _np(backend, data.array))


def test_with_array_checks_shape_and_backend(backend: Backend, dtype: DType) -> None:
    data = Data2D(np.zeros((4, 4)), pbc=True)
    with pytest.raises(ValueError, match="same shape"):
        data.with_array(backend.asarray(np.zeros((1, 1, 4, 2)), dtype))
    with pytest.raises(TypeError, match="build a new Data2D"):
        data.with_array(np.zeros((1, 1, 4, 4)))


def test_copy(backend: Backend) -> None:
    data = Data2D(_field((2, 4, 4)), pbc=True)
    copied = data.copy()
    assert copied.array is not data.array
    np.testing.assert_array_equal(_np(backend, copied.array), _np(backend, data.array))
    empty = data.copy(empty=True)
    assert empty.shape == data.shape
    np.testing.assert_array_equal(_np(backend, empty.array), 0.0)
    assert (copied.pbc, copied.mask) == (data.pbc, data.mask)


# ---------------------------------------------------------------------- slicing


def test_getitem_selects_batch_and_channels_and_keeps_the_axes(backend: Backend) -> None:
    field = _field((3, 2, 4, 4))
    data = Data2D(field, pbc=True)
    expected = _np(backend, data.array)
    assert data[1].shape == (1, 2, 4, 4)
    np.testing.assert_array_equal(_np(backend, data[1].array), expected[1:2])
    np.testing.assert_array_equal(_np(backend, data[-1, 0].array), expected[2:3, 0:1])
    np.testing.assert_array_equal(_np(backend, data[0:2, 1].array), expected[0:2, 1:2])
    np.testing.assert_array_equal(_np(backend, data[:, :].array), expected)
    assert data[np.int64(0)].shape == (1, 2, 4, 4)


def test_getitem_keeps_the_metadata() -> None:
    mask = np.zeros((4, 4), dtype=bool) | (np.arange(4) == 0)
    data = Data2D(_field((3, 2, 4, 4)), mask=mask, pbc=False)
    item = data[0, 1]
    assert type(item) is Data2D
    assert item.mask is data.mask
    assert item.pbc is False


@pytest.mark.parametrize("key", [(0, 0, 0), Ellipsis, [0, 1], 1.5, True])
def test_getitem_refuses_spatial_and_fancy_indexing(key: Any) -> None:
    data = Data2D(_field((3, 2, 4, 4)), pbc=True)
    with pytest.raises(IndexError):
        data[key]


def test_getitem_refuses_out_of_range_and_empty_selections() -> None:
    data = Data2D(_field((3, 2, 4, 4)), pbc=True)
    with pytest.raises(IndexError, match="out of range"):
        data[3]
    with pytest.raises(IndexError, match="out of range"):
        data[0, -3]
    with pytest.raises(IndexError, match="selects no element"):
        data[5:]


# ---------------------------------------------------------------------- pytree and repr


def test_tree_flatten_round_trip() -> None:
    mask = np.arange(8) == 0
    data = Data1D(_field((8,)), mask=mask, pbc=True)
    children, aux = data._tree_flatten()
    assert children == (data.array,)
    rebuilt = Data1D._tree_unflatten(aux, children)
    assert type(rebuilt) is Data1D
    assert (rebuilt.array, rebuilt.mask, rebuilt.pbc, rebuilt.backend) == (
        data.array,
        data.mask,
        data.pbc,
        data.backend,
    )


def test_repr_names_the_geometry() -> None:
    text = repr(Data1D(np.where(np.arange(4) == 0, np.nan, 1.0), pbc=False))
    assert text.startswith("Data1D(shape=(1, 1, 4), pbc=False, missing=1")
