"""``DataHealpix``: its constructor, its metadata as a pytree and its geometry hooks (task T5.5,
issue #166; maths.md M-DATA-HPX, M-MASK-HPX; Part II, §12.4).

The hooks are checked here for their shapes, dtypes, errors and the rules that need no
reference; their values against the oracle are in ``tests/spec/test_healpix_operator.py`` and
against STL's fixtures in ``tests/parity/test_healpix_operator.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from estel import DataHealpix, _config
from estel.backend._protocol import Backend, DType
from estel.data._base import DownsampleWeights
from estel.data._healpix import _StaticCellIds
from estel.wavelets._operator import WaveletOperator

pytestmark = pytest.mark.maths("M-DATA-HPX")

DISC = np.array([0, 1, 2, 3, 5, 9, 12, 13, 14, 47, 100, 101])
"""A few NESTED pixels of depth 2 (``nside`` 4), strictly increasing."""


def _field(n_pix: int, lead: tuple[int, ...] = (), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((*lead, n_pix))


# ---------------------------------------------------------------------- the constructor


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((48,), (1, 1, 48)), ((3, 48), (1, 3, 48)), ((2, 3, 48), (2, 3, 48))],
)
def test_shapes_are_promoted_and_depth_read_on_the_full_sky(
    shape: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    data = DataHealpix(_field(shape[-1], shape[:-1]))
    assert data.shape == expected
    assert (data.depth, data.cell_ids, data.pbc, data.mask) == (1, None, True, None)
    assert DataHealpix(_field(48), depth=1).depth == 1


def test_a_partial_sky_has_its_cells_and_no_pbc() -> None:
    data = DataHealpix(_field(DISC.size), depth=2, cell_ids=list(DISC))
    assert data.pbc is False
    assert data.depth == 2
    np.testing.assert_array_equal(data.cell_ids, DISC)
    assert data.cell_ids is not None
    assert not data.cell_ids.flags.writeable
    # The levels hold the parents of the pixels above (M-DOWN-HPX).
    assert [data._n_pixels(d) for d in range(3)] == [12, 6, 3]


def test_pbc_is_read_only() -> None:
    data = DataHealpix(_field(12))
    with pytest.raises(AttributeError, match="immutable"):
        data.pbc = False  # type: ignore[misc]


def test_the_mask_is_read_from_the_nan_of_one_map_or_given() -> None:
    values = _field(48)
    values[[3, 7]] = np.nan
    data = DataHealpix(values)
    assert data.mask is not None
    np.testing.assert_array_equal(np.flatnonzero(data.mask), [3, 7])
    mask = np.zeros(48, dtype=bool)
    mask[3] = True
    given = DataHealpix(_field(48, (2, 1)), mask=mask)
    assert given.mask is not None
    assert int(given.mask.sum()) == 1


ERRORS: list[Any] = [
    pytest.param({"array": _field(47)}, ValueError, "not a full sky", id="npix-not-12-4^d"),
    pytest.param({"array": _field(24)}, ValueError, "not a full sky", id="npix-not-power-of-4"),
    pytest.param({"array": _field(48), "depth": 2}, ValueError, "full sky of depth 1", id="depth"),
    pytest.param({"array": _field(48), "depth": True}, TypeError, "integer", id="depth-bool"),
    pytest.param(
        {"array": _field(12), "cell_ids": np.arange(12)}, ValueError, "OP-91", id="no-depth"
    ),
    pytest.param(
        {"array": _field(3), "depth": 2, "cell_ids": [5, 3, 9]},
        ValueError,
        "np.argsort",
        id="unsorted",
    ),
    pytest.param(
        {"array": _field(3), "depth": 2, "cell_ids": [3, 3, 9]},
        ValueError,
        "strictly increasing",
        id="repeated",
    ),
    pytest.param(
        {"array": _field(2), "depth": 1, "cell_ids": [3, 48]}, ValueError, "outside", id="range"
    ),
    pytest.param(
        {"array": _field(48), "depth": 1, "cell_ids": np.arange(48)},
        ValueError,
        "omit cell_ids",
        id="whole-sphere",
    ),
    pytest.param(
        {"array": _field(2), "depth": 1, "cell_ids": [3.0, 4.0]}, TypeError, "integer", id="float"
    ),
    pytest.param(
        {"array": _field(3), "depth": 1, "cell_ids": [3, 4]}, ValueError, "one NESTED", id="size"
    ),
    pytest.param(
        {"array": np.where(np.arange(48) == 5, np.nan, _field(48, (2, 1)))},
        ValueError,
        "single map",
        id="nan-of-a-batch",
    ),
    pytest.param({"array": _field(48, (2, 2, 1))}, ValueError, "accepted shapes", id="shape"),
]


@pytest.mark.parametrize(("kwargs", "error", "match"), ERRORS)
def test_invalid_arguments_raise_and_say_what_to_do(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        DataHealpix(**kwargs)


# ---------------------------------------------------------------------- pytree


def test_jit_takes_a_partial_sky_with_a_mask_and_keeps_its_metadata(backend: Backend) -> None:
    """The array is the only leaf: ``depth``, ``cell_ids`` and the mask are static (Part I,
    §2.3; issue #166, decision 7)."""
    mask = np.zeros(DISC.size, dtype=bool)
    mask[4] = True
    data = DataHealpix(_field(DISC.size, (2, 1)), mask=mask, depth=2, cell_ids=DISC)
    out = backend.jit(lambda d: d.with_array(2 * d.array))(data)
    assert isinstance(out, DataHealpix)
    assert (out.depth, out.pbc, out.backend) == (2, False, data.backend)
    np.testing.assert_array_equal(out.cell_ids, DISC)
    np.testing.assert_array_equal(out.mask, mask)
    np.testing.assert_array_equal(backend.to_numpy(out.array), 2 * backend.to_numpy(data.array))


def test_static_cell_ids_compare_and_hash_by_value() -> None:
    a, b = _StaticCellIds(DISC, 2), _StaticCellIds(DISC.copy(), 2)
    assert a == b
    assert hash(a) == hash(b)
    assert a != _StaticCellIds(DISC[1:], 2)
    assert not a.cell_ids.flags.writeable


# ---------------------------------------------------------------------- hooks


@pytest.mark.dtype
@pytest.mark.maths("M-CONV-HPX", "M-DOWN-HPX")
def test_the_hooks_keep_the_dtype(
    monkeypatch: pytest.MonkeyPatch, backend: Backend, dtype: DType
) -> None:
    """The data, the tables and the results of ``_conv`` and ``_downsample`` are in the run's
    dtype (Part I, §3.1, rule 7): weights cast from float64, indices int32."""
    monkeypatch.setattr(_config, "_dtype", dtype)
    complex_dtype = {"float32": "complex64", "float64": "complex128"}[dtype]
    data = DataHealpix(_field(48, (1, 2)))
    operator = WaveletOperator(data, 2)
    filters = operator.filters
    assert filters is not None
    tables = filters._conv_tables(0)
    assert tables.index.dtype == backend.xp.int32
    assert tables.weight.dtype == getattr(backend.xp, dtype)
    w = operator.apply(data.array, 0)
    assert tuple(w.shape) == (1, 4, 2, 48)
    assert w.dtype == getattr(backend.xp, complex_dtype)
    down = operator.downsample(data.array, 0)
    assert tuple(down.shape) == (1, 2, 12)
    assert down.dtype == getattr(backend.xp, dtype)


@pytest.mark.maths("M-CONV-HPX", "M-DOWN-HPX")
def test_a_complex_map_is_convolved_and_decimated_part_by_part(backend: Backend) -> None:
    data = DataHealpix(_field(48, (1, 1)))
    operator = WaveletOperator(data, 2)
    xp = backend.xp
    re, im = data.array, xp.flip(data.array, axis=-1)
    z = backend.real_pairs_to_complex(xp.stack([re, im], axis=-1))

    def conv(a: Any) -> Any:
        return operator.apply(a, 0)

    def down(a: Any) -> Any:
        return operator.downsample(a, 0)

    for function in (conv, down):
        expected = backend.to_numpy(function(re)) + 1j * backend.to_numpy(function(im))
        np.testing.assert_allclose(backend.to_numpy(function(z)), expected, atol=1e-12)


@pytest.mark.maths("M-MASK-HPX")
def test_no_border_and_the_hooks_healpix_does_not_have() -> None:
    data = DataHealpix(_field(DISC.size), depth=2, cell_ids=DISC)
    np.testing.assert_array_equal(data._crop_weights(1, 3), np.ones(6))
    with pytest.raises(NotImplementedError, match="M-LP"):
        data._lowpass(data.array, None, 0)
    with pytest.raises(NotImplementedError, match="OP-77"):
        data._nyquist_filter(data.array)
    with pytest.raises(NotImplementedError, match="LF24"):
        data._spectrum_geometry()


def test_the_hooks_check_their_tables_and_shapes() -> None:
    data = DataHealpix(_field(48))
    with pytest.raises(TypeError, match="HealpixConvTables"):
        data._conv(data.array, object(), 0)
    with pytest.raises(TypeError, match="HealpixDownTables"):
        data._downsample(data.array, DownsampleWeights(object()), 0)
    with pytest.raises(ValueError, match="12 pixels of level 1"):
        data._conv(data.array, object(), 1)
