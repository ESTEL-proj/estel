"""Tensor products of Cartesian geometries: ``DataTensorProd`` and the wavelet operator composed
from those of its blocks (task T3.1; Part II, §12.4, §12.5; maths.md M-NOT-TENSORPROD,
M-DATA-TENSORPROD, M-W1-TENSORPROD, M-BORDER-TENSORPROD, M-DEF-TENSORPROD; OP-4, OP-8, OP-41, OP-82,
OP-83).

The hooks of a tensor product act on one block, along its axes, without moving any: they are
compared with the Cartesian geometry applied to the same field with the block's axes moved here,
in the test. Run on the backend and dtype of the test run; tolerances of Part I, §5.2
(``docs/agents/testing.md``), with atol relative to the largest expected modulus.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data, Data1D, Data2D, DataHealpix, DataTensorProd, STOperator
from estel._config import FloatDType, default_device
from estel.backend._protocol import Backend, DType
from estel.data._base import DownsampleWeights
from estel.data._cartesian import DataCartesian
from estel.wavelets._banks_cartesian import smoothing_kernel_cartesian, wavelet_bank_cartesian
from estel.wavelets._operator import (
    WaveletOperator,
    WaveletOperatorSingleDomain,
    default_scales_of_cartesian,
)
from estel.wavelets._operator_tensor_prod import (
    WaveletOperatorTensorProd,
    WaveletPlanTensorProd,
    build_wavelet_filters_tensor_prod,
    build_wavelet_plan_tensor_prod,
)
from tests._backends import backend_name

jax_only = pytest.mark.skipif(backend_name() != "jax", reason="names JAX's vmap")

TOLERANCES = {"float32": (1e-4, 1e-6), "float64": (1e-10, 1e-12)}
COMPLEX: dict[str, DType] = {"float32": "complex64", "float64": "complex128"}
SIZES: dict[type[Data], tuple[int, ...]] = {Data2D: (16, 16), Data1D: (8,)}


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal(shape)


def _assert_close(backend: Backend, actual: Any, expected: Any, dtype: str) -> None:
    rtol, atol = TOLERANCES[dtype]
    got, want = backend.to_numpy(actual), backend.to_numpy(expected)
    assert got.shape == want.shape
    scale = float(np.abs(want).max()) or 1.0
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol * scale)


def _bank(backend: Backend, geometry: type[DataCartesian], dtype: str) -> Any:
    return backend.asarray(
        wavelet_bank_cartesian(geometry.ndim, 4 if geometry is Data2D else 2), COMPLEX[dtype]
    )


def _operator(
    product: DataTensorProd,
    dtype: DType,
    J: list[int] | None = None,  # noqa: N803
) -> WaveletOperatorTensorProd:
    """The wavelet operator of ``product`` with its arrays in ``dtype``."""
    plan = build_wavelet_plan_tensor_prod(product, J)
    real: FloatDType = "float64" if dtype == "float64" else "float32"
    filters = build_wavelet_filters_tensor_prod(plan, product.backend, real, default_device())
    return WaveletOperatorTensorProd.from_parts(product, plan, filters)


def _smoothing(backend: Backend, geometry: type[DataCartesian], dtype: DType) -> DownsampleWeights:
    return DownsampleWeights(backend.asarray(smoothing_kernel_cartesian(geometry.ndim), dtype))


# ---------------------------------------------------------------------- one block (OP-82)


@pytest.mark.maths("M-DATA-TENSORPROD", "M-W1-TENSORPROD", "M-BORDER-TENSORPROD")
@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize("geometry", [Data2D, Data1D])
def test_one_block_hooks_equal_the_plain_geometry(
    backend: Backend, dtype: DType, geometry: type[Data2D] | type[Data1D], pbc: bool
) -> None:
    """A tensor product of one block filters, downsamples and crops as its Cartesian geometry,
    bit for bit (OP-82)."""
    values = backend.asarray(_field((2, 3, *(16,) * geometry.ndim)), dtype)
    plain = geometry(values, pbc=pbc)
    product = DataTensorProd(values, [geometry], pbc=[pbc])
    x = plain.array
    bank, weights = _bank(backend, geometry, dtype), _smoothing(backend, geometry, dtype)
    np.testing.assert_array_equal(
        backend.to_numpy(product._conv(x, bank, 0, block=0)),
        backend.to_numpy(plain._conv(x, bank, 0)),
    )
    np.testing.assert_array_equal(
        backend.to_numpy(product._downsample(x, weights, 0, block=0)),
        backend.to_numpy(plain._downsample(x, weights, 0)),
    )
    np.testing.assert_array_equal(product._crop_weights(1, 2, block=0), plain._crop_weights(1, 2))


# ---------------------------------------------------------------------- two blocks


def _moved(backend: Backend, x: Any, axes: tuple[int, ...]) -> Any:
    """``x`` ``(Nb, Nc, *spatial)`` with ``axes`` moved last and the other spatial axes merged
    into the channels: plain data of the block's geometry."""
    xp = backend.xp
    others = tuple(a for a in range(2, len(x.shape)) if a not in axes)
    moved = xp.permute_dims(x, (0, 1, *others, *axes))
    return xp.reshape(moved, (int(x.shape[0]), -1, *(int(x.shape[a]) for a in axes)))


def _back(
    backend: Backend, y: Any, x_shape: tuple[int, ...], axes: tuple[int, ...], oriented: bool
) -> Any:
    """The inverse of :func:`_moved` on a result of the Cartesian geometry, ``(Nb, L, Nc', *block)``
    if ``oriented``, else ``(Nb, Nc', *block)``: ``(Nb, [L,] Nc, *spatial)``, the block's axes
    back in place."""
    xp = backend.xp
    others = tuple(a for a in range(2, len(x_shape)) if a not in axes)
    head = 2 if oriented else 1
    lead = tuple(int(n) for n in y.shape[:head])
    block = tuple(int(n) for n in y.shape[len(y.shape) - len(axes) :])
    y = xp.reshape(y, (*lead, x_shape[1], *(x_shape[a] for a in others), *block))
    inverse = np.argsort((0, 1, *others, *axes))
    shift = head - 1
    order = tuple(range(head)) + tuple(int(inverse[i]) + shift for i in range(1, len(x_shape)))
    return xp.permute_dims(y, order)


CASES = [
    pytest.param((Data2D, Data1D), (True, False), id="2D-1D"),
    pytest.param((Data2D, Data1D), (False, True), id="2D-1D-np-p"),
    pytest.param((Data1D, Data2D), (False, True), id="1D-2D"),
]
"""The block geometries and their ``pbc``: each geometry in each position, and each geometry
periodic and not, a non-periodic 2D block and a periodic 1D block included."""


def _product(
    backend: Backend, dtype: DType, blocks: tuple[type[Data], ...], pbc: tuple[bool, ...]
) -> DataTensorProd:
    spatial = tuple(n for g in blocks for n in SIZES[g])
    return DataTensorProd(backend.asarray(_field((2, 2, *spatial)), dtype), blocks, pbc=pbc)


@pytest.mark.dtype
@pytest.mark.maths("M-NOT-TENSORPROD", "M-W1-TENSORPROD")
@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize(("blocks", "pbc"), CASES)
def test_block_hooks_equal_the_plain_geometry_with_its_axes_moved(
    backend: Backend,
    dtype: DType,
    blocks: tuple[type[Data], ...],
    pbc: tuple[bool, ...],
    block: int,
) -> None:
    """Along a block, ``_conv`` and ``_downsample`` give the Cartesian geometry of the block
    applied with the block's axes moved last and the other axes read as channels, to
    rounding: the dispatch along the block's axes moves nothing and changes nothing else."""
    product = _product(backend, dtype, blocks, pbc)
    x, shape = product.array, product.shape
    spec = product._block_specs[block]
    moved = _moved(backend, x, spec.axes)
    plain = spec.geometry._from_parts(moved, None, spec.pbc, product.backend)
    bank = _bank(backend, spec.geometry, dtype)
    weights = _smoothing(backend, spec.geometry, dtype)
    _assert_close(
        backend,
        product._conv(x, bank, 0, block=block),
        _back(backend, plain._conv(moved, bank, 0), shape, spec.axes, True),
        dtype,
    )
    _assert_close(
        backend,
        product._downsample(x, weights, 0, block=block),
        _back(backend, plain._downsample(moved, weights, 0), shape, spec.axes, False),
        dtype,
    )


@pytest.mark.maths("M-W1-TENSORPROD")
@pytest.mark.parametrize(("filtered", "downsampled"), [(0, 1), (1, 0)])
@pytest.mark.parametrize(("blocks", "pbc"), CASES)
def test_filtering_and_downsampling_along_different_blocks_commute(
    backend: Backend,
    dtype: DType,
    blocks: tuple[type[Data], ...],
    pbc: tuple[bool, ...],
    filtered: int,
    downsampled: int,
) -> None:
    """Filtering along one block and downsampling along another act on different axes and
    commute, to rounding, with circular and replicate padding (M-W1-TENSORPROD)."""
    product = _product(backend, dtype, blocks, pbc)
    operator = _operator(product, dtype, [2, 2])
    x = product.array
    level = tuple(int(i == downsampled) for i in range(2))
    filtered_first = operator.downsample(
        operator.apply(x, 0, block=filtered, level=(0, 0)), (0, 0), block=downsampled
    )
    downsampled_first = operator.apply(
        operator.downsample(x, (0, 0), block=downsampled), 0, block=filtered, level=level
    )
    _assert_close(backend, filtered_first, downsampled_first, dtype)


@pytest.mark.maths("M-W1-TENSORPROD")
@pytest.mark.parametrize(("block", "geometry", "n_orientations"), [(0, Data2D, 4), (1, Data1D, 2)])
def test_the_kernels_of_the_blocks_are_those_of_the_cartesian_operators(
    backend: Backend, dtype: DType, block: int, geometry: type[Data], n_orientations: int
) -> None:
    """The kernels of a ``[2D, 1D]`` operator are bit for bit those of the 2D operator and of
    the 1D operator with L = 2 (M-W1-TENSORPROD)."""
    product = DataTensorProd(np.zeros((16, 16, 8)), [Data2D, Data1D], pbc=[True, True])
    operator = _operator(product, dtype)
    assert operator.filters is not None
    plain = WaveletOperatorSingleDomain(
        geometry(np.zeros(SIZES[geometry]), pbc=True), L=n_orientations
    ).plan.bank()
    np.testing.assert_array_equal(operator.plan.blocks[block].bank(), plain)
    np.testing.assert_array_equal(
        backend.to_numpy(operator.filters[block]._conv_tables(0)),
        backend.to_numpy(backend.asarray(plain, COMPLEX[dtype])),
    )


@pytest.mark.maths("M-W1-TENSORPROD")
def test_the_second_kernel_of_a_1d_block_is_the_conjugate_of_the_first() -> None:
    """With L = 2 in a 1D block, the second kernel is the conjugate of the first
    (M-W1-TENSORPROD)."""
    product = DataTensorProd(np.zeros((16, 16, 8)), [Data2D, Data1D], pbc=[True, True])
    bank = WaveletOperatorTensorProd(product).plan.blocks[1].bank()
    np.testing.assert_array_equal(bank[1], np.conj(bank[0]))


@pytest.mark.maths("M-W1-TENSORPROD")
@pytest.mark.parametrize(
    ("blocks", "spatial", "orientations"),
    [
        pytest.param((Data2D, Data1D), (16, 16, 8), (4, 2), id="2D-1D"),
        pytest.param((Data1D, Data2D), (8, 16, 16), (2, 4), id="1D-2D"),
    ],
)
def test_the_orientation_axes_follow_the_channels_in_block_order(
    backend: Backend,
    dtype: DType,
    blocks: tuple[type[Data], ...],
    spatial: tuple[int, ...],
    orientations: tuple[int, int],
) -> None:
    """The product of the block convolutions, block 0 first, is ``(Nb, Nc, L₀, L₁, *spatial)``
    (M-W1-TENSORPROD): entry ``[:, c, l₀, l₁]`` is the convolution by ψ⁽¹⁾_{l₁} along block 1 of the
    convolution by ψ⁽⁰⁾_{l₀} along block 0 of channel c, built here from the hooks of the
    data."""
    n_batch, n_channels = 2, 3
    (l_0, l_1), n_spatial = orientations, len(spatial)
    values = backend.asarray(_field((n_batch, n_channels, *spatial)), dtype)
    product = DataTensorProd(values, blocks, pbc=[True, False])
    operator = _operator(product, dtype)
    assert operator.filters is not None
    first = operator.apply(product.array, 0, block=0, level=(0, 0))
    both = operator.apply(first, 0, block=1, level=(0, 0))
    assert tuple(both.shape) == (n_batch, n_channels, l_0, l_1, *spatial)
    xp = backend.xp
    banks = [f._conv_tables(0) for f in operator.filters]
    by_block_0 = product._conv(product.array, banks[0], 0, block=0)  # (Nb, L₀, Nc, ...)
    flat = xp.reshape(by_block_0, (n_batch, l_0 * n_channels, *spatial))
    by_block_1 = product._conv(flat, banks[1], 0, block=1)  # (Nb, L₁, L₀·Nc, ...)
    expected = xp.permute_dims(
        xp.reshape(by_block_1, (n_batch, l_1, l_0, n_channels, *spatial)),
        (0, 3, 2, 1, *range(4, 4 + n_spatial)),
    )
    _assert_close(backend, both, expected, dtype)


@pytest.mark.maths("M-NOT-TENSORPROD")
def test_the_operator_of_a_tensor_prod_is_a_wavelet_operator() -> None:
    """It gives the members its root declares: the multi-index levels and their shapes; its
    means are weighted block by block, so mean_weights raises and says so."""
    product = DataTensorProd(np.zeros((8, 8, 4)), [Data2D, Data1D], pbc=[True, True])
    operator = WaveletOperatorTensorProd(product, [2, 1])
    assert isinstance(operator, WaveletOperator)
    assert operator.levels == ((0, 0), (1, 0))
    assert [operator.level_shape(d) for d in operator.levels] == [(8, 8, 4), (4, 4, 4)]
    with pytest.raises(
        NotImplementedError, match=r"M-BORDER-TENSORPROD.*estel\.STOperatorTensorProd"
    ):
        operator.mean_weights({})


@pytest.mark.maths("M-BORDER-TENSORPROD")
def test_the_crop_is_the_outer_product_of_the_crops_of_the_blocks() -> None:
    """Block by block, the border of each block along its axes; a border too large for a block
    is reduced, with one warning for the operator (OP-13)."""
    product = DataTensorProd(np.zeros((16, 16, 8)), [Data2D, Data1D], pbc=[False, False])
    operator = WaveletOperatorTensorProd(product, [2, 1])
    crop = operator.crop_weights((1, 0), (1, 2))
    plain_2d = Data2D(np.zeros((16, 16)), pbc=False)._crop_weights(1, 1)
    plain_1d = Data1D(np.zeros(8), pbc=False)._crop_weights(0, 2)
    np.testing.assert_array_equal(crop, np.multiply.outer(plain_2d, plain_1d))
    small = WaveletOperatorTensorProd(
        DataTensorProd(np.zeros((8, 8)), [Data1D, Data1D], pbc=[False, False]), [1, 1]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reduced = small.crop_weights((0, 0), (4, 4))
    assert len(caught) == 1
    assert "OP-13" in str(caught[0].message)
    expected = np.pad(np.ones(2), 3)
    np.testing.assert_array_equal(reduced, np.multiply.outer(expected, expected))


# ---------------------------------------------------------------------- data (M-DATA-TENSORPROD)


@pytest.mark.maths("M-DATA-TENSORPROD")
@pytest.mark.parametrize(
    ("shape", "promoted"),
    [((16, 16, 16), (1, 1, 16, 16, 16)), ((3, 16, 16, 16), (1, 3, 16, 16, 16))],
)
def test_shapes_are_promoted(shape: tuple[int, ...], promoted: tuple[int, ...]) -> None:
    product = DataTensorProd(np.zeros(shape), [Data2D, Data1D], pbc=[True, False])
    assert product.shape == promoted
    assert product.ndim == 3
    assert product.blocks == (Data2D, Data1D)
    assert product.pbc == (True, False)


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(lambda d: d.with_array(2 * d.array), id="with_array"),
        pytest.param(lambda d: d.copy(), id="copy"),
        pytest.param(lambda d: d[1, 0], id="slicing"),
    ],
)
def test_functional_updates_and_slicing_keep_the_blocks(
    update: Callable[[DataTensorProd], Data],
) -> None:
    product = DataTensorProd(_field((2, 2, 8, 16, 16)), [Data1D, Data2D], pbc=[False, True])
    other = update(product)
    assert isinstance(other, DataTensorProd)
    assert other._block_specs == product._block_specs


def test_the_pytree_holds_the_array_only() -> None:
    product = DataTensorProd(_field((2, 2, 8, 16, 16)), [Data1D, Data2D], pbc=[False, True])
    children, aux = product._tree_flatten()
    rebuilt = DataTensorProd._tree_unflatten(aux, children)
    assert isinstance(rebuilt, DataTensorProd)
    assert rebuilt.array is product.array
    assert (rebuilt.blocks, rebuilt.pbc) == (product.blocks, product.pbc)


@pytest.mark.maths("M-DATA-TENSORPROD", "M-DEF-TENSORPROD")
@pytest.mark.parametrize(
    ("shape", "expected"),
    [((8,), 1), ((12,), 1), ((16,), 2), ((31,), 2), ((32,), 3), ((64,), 4), ((32, 8), 1)],
)
def test_default_scales_of_a_block(shape: tuple[int, ...], expected: int) -> None:
    """The rule of Cartesian data, block by block (M-DEF-TENSORPROD), one implementation."""
    assert default_scales_of_cartesian(shape) == expected


@pytest.mark.maths("M-DATA-TENSORPROD", "M-DEF-TENSORPROD")
def test_defaults_of_a_32x32x8_field() -> None:
    product = DataTensorProd(np.zeros((32, 32, 8)), [Data2D, Data1D], pbc=[True, True])
    operator = WaveletOperatorTensorProd(product)
    assert operator.plan.J == (3, 1)
    assert operator.plan.L == (4, 2)


# ---------------------------------------------------------------------- errors


def _valid_array() -> NDArray[np.float64]:
    return np.zeros((16, 16, 8))


@pytest.mark.parametrize(
    ("blocks", "pbc", "error", "match"),
    [
        (Data2D, [True], TypeError, r"as a list"),
        ([], [], ValueError, r"at least one block"),
        ([Data2D, "1d"], [True, True], TypeError, r"estel.Data2D or estel.Data1D"),
        ([Data, Data1D], [True, True], TypeError, r"estel.Data2D or estel.Data1D"),
        ([DataTensorProd, Data1D], [True, True], TypeError, r"estel.Data2D or estel.Data1D"),
        ([Data2D, Data1D], True, TypeError, r"one boolean per block.*pbc=\[True, False\]"),
        ([Data2D, Data1D], [True], ValueError, r"1 value\(s\) for 2 block\(s\)"),
        ([Data2D, Data1D], [True, 1], TypeError, r"pbc\[1\].*written explicitly"),
    ],
)
@pytest.mark.maths("M-DATA-TENSORPROD")
def test_blocks_and_pbc_are_checked(
    blocks: Any, pbc: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        DataTensorProd(_valid_array(), blocks, pbc=pbc)


class _DataSubclass2D(Data2D):
    """A subclass of Data2D: it inherits the capability of a block without declaring it."""


@pytest.mark.maths("M-DATA-TENSORPROD")
@pytest.mark.parametrize(
    ("block", "accepted"),
    [
        (Data2D, True),
        (Data1D, True),
        (DataCartesian, False),
        (_DataSubclass2D, False),
        (DataHealpix, False),
    ],
)
def test_the_blocks_are_the_classes_that_declare_the_capability(
    block: type[Data], accepted: bool
) -> None:
    """The capability of a block (``DataCartesian._block_tensor_prod``, Part I, §2.3.1) accepts
    the classes accepted before it, Data2D and Data1D, and refuses the others with the same
    message."""
    if accepted:
        array = np.zeros((16,) * cast(int, block.ndim))
        assert DataTensorProd(array, [block], pbc=[True]).blocks == (block,)
        return
    message = (
        f"DataTensorProd: block 0 is {block!r}; each block is the class estel.Data2D or "
        "estel.Data1D, not an instance (maths.md M-DATA-TENSORPROD). HEALPix blocks come later "
        "(LF9). Pass the classes, for example blocks=[estel.Data2D, estel.Data1D] for space and "
        "time."
    )
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        DataTensorProd(np.zeros((16, 16)), [block], pbc=[True])


@pytest.mark.maths("M-DATA-TENSORPROD")
def test_the_shape_is_checked() -> None:
    with pytest.raises(ValueError, match=r"accepted shapes are \(N0_1, N0_2, N1\)"):
        DataTensorProd(np.zeros((16, 16)), [Data2D, Data1D], pbc=[True, True])


@pytest.mark.maths("M-DATA-TENSORPROD")
def test_missing_values_raise_and_name_lf8() -> None:
    """Data with a NaN, or given a mask, raise at construction (OP-41)."""
    values = _valid_array()
    values[0, 0, 0] = np.nan
    with pytest.raises(NotImplementedError, match=r"1 NaN.*LF8.*Replace the NaN"):
        DataTensorProd(values, [Data2D, Data1D], pbc=[True, True])
    with pytest.raises(NotImplementedError, match=r"LF8.*Pass mask=None"):
        DataTensorProd(
            _valid_array(), [Data2D, Data1D], np.zeros((16, 16, 8), bool), pbc=[True, True]
        )


@pytest.mark.parametrize(
    ("J", "L", "error", "match"),
    [
        (3, None, TypeError, r"J=3: a tensor product takes one J per block, as a list"),
        ([2], None, ValueError, r"1 entry\(ies\) for 2 block\(s\)"),
        ([2, 0], None, ValueError, r"J\[1\]=0.*at least 1"),
        ([2, 1.5], None, TypeError, r"J\[1\]=1.5.*integer"),
        ([2, 4], None, ValueError, r"block 1 .*divisible by 2\^\(J_1-1\) = 8.*Use J_1 ≤ 3"),
        (None, 4, TypeError, r"L=4: a tensor product takes one L per block.*L=\[4, 2\]"),
        (None, [4, 3], ValueError, r"L\[1\]=3: block 1 \(Data1D of shape \(12,\)\) has L = 1 or"),
        (None, [0, 2], ValueError, r"L\[0\]=0.*at least 1"),
    ],
)
@pytest.mark.maths("M-DATA-TENSORPROD", "M-DEF-TENSORPROD")
def test_the_arguments_of_the_operator_are_checked(
    J: Any,  # noqa: N803 (maths.md notation)
    L: Any,  # noqa: N803
    error: type[Exception],
    match: str,
) -> None:
    """Clear errors at construction (OP-83), and the sizes of each block divisible by
    2^(J_a - 1) (OP-8)."""
    product = DataTensorProd(np.zeros((16, 16, 12)), [Data2D, Data1D], pbc=[True, True])
    with pytest.raises(error, match=match):
        WaveletOperatorTensorProd(product, J, L)


@pytest.mark.maths("M-DATA-TENSORPROD", "M-DEF-TENSORPROD")
def test_defaults_that_do_not_hold_raise() -> None:
    """A block too small for its default J, and a default J the sizes cannot take (OP-83,
    OP-8): construction fails with a message that says what to pass."""
    small = DataTensorProd(np.zeros((16, 16, 4)), [Data2D, Data1D], pbc=[True, True])
    with pytest.raises(ValueError, match=r"default J_1 = ⌊log2 4⌋ - 2 = 0.*Pass J_1 explicitly"):
        WaveletOperatorTensorProd(small)
    odd = DataTensorProd(np.zeros((16, 16, 31)), [Data2D, Data1D], pbc=[True, True])
    with pytest.raises(ValueError, match=r"the default J_1 = 2 needs every size.*Use J_1 ≤ 1"):
        WaveletOperatorTensorProd(odd)
    assert WaveletOperatorTensorProd(odd, [None, 1]).plan.J == (2, 1)


def test_the_callers_lists_are_left_unchanged() -> None:
    product = DataTensorProd(np.zeros((16, 16, 8)), [Data2D, Data1D], pbc=[True, True])
    scales: list[int | None] = [None, 1]
    orientations: list[int | None] = [None, None]
    operator = WaveletOperatorTensorProd(product, scales, orientations)
    assert scales == [None, 1]
    assert orientations == [None, None]
    assert operator.plan.J == (2, 1)
    assert operator.plan.L == (4, 2)


def test_hooks_need_a_block(backend: Backend, dtype: DType) -> None:
    product = DataTensorProd(
        backend.asarray(_field((16, 16, 8)), dtype), [Data2D, Data1D], pbc=[True, True]
    )
    bank = _bank(backend, Data2D, dtype)
    with pytest.raises(TypeError, match=r"one block at a time: pass block=a"):
        product._conv(product.array, bank, 0)
    with pytest.raises(ValueError, match=r"2 block\(s\), 0 … 1"):
        product._crop_weights(0, 0, block=2)


def test_scattering_operators_of_single_domain_data_refuse_tensor_prod_data() -> None:
    product = DataTensorProd(np.zeros((16, 16, 8)), [Data2D, Data1D], pbc=[True, True])
    with pytest.raises(TypeError, match=r"a tensor product is STOperatorTensorProd"):
        STOperator(product)
    with pytest.raises(TypeError, match=r"a tensor product is STOperatorTensorProd"):
        STOperator(product, compute_PS=True)  # before any check of the spectrum


# ---------------------------------------------------------------------- pytree (Part I, §2.3)


def _block_convolution(
    plan: WaveletPlanTensorProd, filters: Any
) -> Callable[[DataTensorProd], Any]:
    """A function of a ``DataTensorProd``: the block convolutions of scale 0 along block 0
    then block 1, through the hooks of the data it is given."""

    def function(data: DataTensorProd) -> Any:
        operator = WaveletOperatorTensorProd.from_parts(data, plan, filters)
        first = operator.apply(data.array, 0, block=0, level=(0, 0))
        return operator.apply(first, 0, block=1, level=(0, 0))

    return function


@pytest.mark.slow
@pytest.mark.maths("M-W1-TENSORPROD")
def test_jit_takes_tensor_prod_data(backend: Backend, dtype: DType) -> None:
    """``backend.jit`` (``jax.jit`` on JAX, the identity on PyTorch) of a function of a
    ``DataTensorProd``: the blocks are static, the array the only leaf.

    Slow: a compilation of the block convolutions; the full tier, as decided on issue #22."""
    values = backend.asarray(_field((2, 2, 16, 16, 8)), dtype)
    product = DataTensorProd(values, [Data2D, Data1D], pbc=[False, True])
    operator = _operator(product, dtype)
    function = _block_convolution(operator.plan, operator.filters)
    out = backend.jit(function)(product)
    _assert_close(backend, out, function(product), dtype)
    doubled = backend.jit(lambda d: d.with_array(2 * d.array))(product)
    assert isinstance(doubled, DataTensorProd)
    assert doubled._block_specs == product._block_specs


@jax_only
@pytest.mark.slow
@pytest.mark.maths("M-W1-TENSORPROD")
def test_vmap_maps_a_function_of_tensor_prod_data(backend: Backend, dtype: DType) -> None:
    """``jax.vmap`` over a stack of fields, a ``DataTensorProd`` whose leaf has one more
    leading axis: each field gives the block convolutions of that field alone.

    Slow: a compilation of the block convolutions; the full tier, as decided on issue #22."""
    import jax

    values = backend.asarray(_field((1, 2, 16, 16, 8)), dtype)
    product = DataTensorProd(values, [Data2D, Data1D], pbc=[False, True])
    operator = _operator(product, dtype)
    function = _block_convolution(operator.plan, operator.filters)
    stacked = jax.tree_util.tree_map(lambda a: backend.xp.stack([a, 2 * a]), product)
    out = jax.vmap(function)(stacked)
    single = function(product)
    _assert_close(backend, out[0], single, dtype)
    _assert_close(backend, out[1], 2 * single, dtype)
