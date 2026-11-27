"""The compact coefficient layout (tasks T2.15 and T3.7, issues #93 and #139; Part III, §19.2,
§19.4 to §19.6).

The values are compared with STL through the full-grid views in ``tests/parity/``, and the
accessors in ``test_statistics.py``. These tests check what those do not: that the buffer and the
full-grid views hold the same coefficients both ways, through the normalisation, the selection
and every reduction; that real entries take one float; and that the computation builds no
full-grid view.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data, Data1D, Data2D, STOperator, STStatistics, _config
from estel.backend import _registry
from estel.functional import _scattering
from estel.scattering._plan import PlanarPlan
from estel.statistics import _grid, _layout, _normalization, _statistics
from estel.statistics import _reductions as reductions
from estel.statistics._arrays import constant

pytestmark = pytest.mark.maths("M-FLAT")


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    monkeypatch.setattr(_config, "_dtype", dtype)


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    field: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return field


def _apply(data: Data, J: int, norm: str = "self", **kwargs: Any) -> STStatistics:  # noqa: N803
    options: dict[str, Any] = {"compute_PS": False if data.ndim == 2 else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30
        return STOperator(data, J, **{**options, **kwargs}).apply(data, norm=norm)


@functools.lru_cache(maxsize=8)
def _planar(dtype: str, backend: str) -> STStatistics:
    """2D, not periodic, a mask with empty means, cross channels and a channel without auto
    statistics (OP-30); built once per dtype and backend."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[14:17, 14:18] = True
    masked = Data2D(_field((2, 3, 32, 32)), mask=mask, pbc=False)
    matrix = np.array([[True, False, True], [False, False, True], [False, False, True]])
    return _apply(masked, 3, cross_matrix=matrix)


def _line(norm: str | None) -> STStatistics:
    line = Data1D(_field((2, 2, 32), seed=1), pbc=True)
    return _apply(line, 3, L=2, **({} if norm is None else {"norm": norm}))


def _masked_without_norm() -> STStatistics:
    mask = np.zeros((32, 32), dtype=bool)
    mask[14:17, 14:18] = True
    return _apply(Data2D(_field((2, 3, 32, 32)), mask=mask, pbc=False), 3, norm="none")


CASES: dict[str, Callable[[STStatistics], STStatistics]] = {
    "planar": lambda planar: planar,
    "planar-norm-none": lambda planar: _masked_without_norm(),
    "line-norm-none": lambda planar: _line("none"),
    "line-norm-self": lambda planar: _line(None),
    "select-channels": lambda planar: planar.select(c1=[0, 2], j3=2),
    "select-S4": lambda planar: planar.select("S4", channel=(0, 2), l1=1),
    "iso": lambda planar: planar.reduce(iso=True),
    "angular": lambda planar: planar.reduce(angular_fourier=True, angular_harmonics=2),
    "all-reductions": lambda planar: planar.reduce(
        iso=True, angular_fourier=True, scale_cosine=True, dj=2
    ),
}
"""1D and 2D, periodic and not, a mask with empty means, cross channels, a channel without
auto statistics (OP-30), each normalisation, then the selection and every reduction."""


def _case(name: str, dtype: str) -> STStatistics:
    return CASES[name](_planar(dtype, _registry.get_backend()))


def _buffer_from_grids(stats: STStatistics) -> NDArray[Any]:
    """The buffer read back from the full-grid views, with NumPy: each entry of the layout at
    its position, its real part, then its imaginary part if it has one."""
    layout = stats._layout
    floats = np.zeros((int(stats._buffer.shape[0]), layout.n_floats + 1))
    columns: list[NDArray[Any]] = []
    for grid in layout.grids:
        full = np.asarray(stats.full_grid(grid.name))
        entries = layout.entries(grid.name)
        values = full.reshape(full.shape[0], -1)[
            :, _layout.flat_positions(entries.positions, grid.shape)
        ]
        pairs = np.stack([values.real, values.imag], axis=-1)
        columns.extend(
            pairs[:, i, : 1 + int(entries.is_complex[i])] for i in range(entries.n_entries)
        )
    return np.concatenate([c for c in columns], axis=1) if columns else floats[:, :0]


@pytest.mark.parametrize("case", CASES)
def test_buffer_and_full_grid_round_trip(case: str, dtype: str) -> None:
    """buffer -> full_grid -> buffer is the identity, and every position of the full grid
    outside the layout is NaN."""
    stats = _case(case, dtype)
    buffer = np.asarray(stats._buffer)
    np.testing.assert_array_equal(_buffer_from_grids(stats), buffer)
    assert np.isfinite(buffer).all()
    full = {grid.name: np.asarray(stats.full_grid(grid.name)) for grid in stats._grids}
    assert all(full[grid.name].shape[1:] == grid.shape for grid in stats._grids)
    assert all(
        np.array_equal(np.isnan(full[g.name]), np.broadcast_to(~g.present, full[g.name].shape))
        for g in stats._grids
    )


REAL_ENTRIES: dict[str, Callable[[NDArray[Any]], NDArray[Any]]] = {
    "mean": lambda p: np.zeros(len(p), bool),
    "var": lambda p: np.zeros(len(p), bool),
    "S1": lambda p: p[:, 0] != p[:, 1],
    "S2": lambda p: p[:, 0] != p[:, 1],
    "S3": lambda p: np.ones(len(p), bool),
    "S4": lambda p: ~((p[:, 0] == p[:, 1]) & (p[:, 2] == p[:, 3]) & (p[:, 5] == p[:, 6])),
}
"""Which entries of each family are complex, from their positions: mean, var, the auto S1
and S2 and the diagonal of S4 are real."""


def _two_channels() -> STStatistics:
    return _apply(Data2D(_field((1, 2, 16, 16)), pbc=True), 2, norm="none")


@pytest.mark.parametrize("family", REAL_ENTRIES)
def test_real_entries_take_one_float(family: str) -> None:
    """Mean, var, the auto S1 and S2 and the diagonal of S4 take one float, every other entry
    two; their imaginary part is 0 in the full-grid view."""
    stats = _two_channels()
    entries = stats._layout.entries(family)
    np.testing.assert_array_equal(entries.is_complex, REAL_ENTRIES[family](entries.positions))
    if family not in ("mean", "var"):
        full = np.asarray(stats.full_grid(family))
        real = entries.positions[~entries.is_complex]
        assert np.all(full[(slice(None), *real.T)].imag == 0)


def test_buffer_counts_one_float_per_real_entry() -> None:
    """The buffer holds one float per entry and one more per complex entry; S4 has one
    segment per level."""
    stats = _two_channels()
    layout = stats._layout
    n_entries = sum(layout.entries(n).n_entries for n in layout.names)
    n_complex = sum(int(layout.entries(n).is_complex.sum()) for n in layout.names)
    assert layout.n_floats == n_entries + n_complex
    assert tuple(stats._buffer.shape) == (1, layout.n_floats)
    assert {s.level for s in layout.segments if s.family == "S4"} == {(0,), (1,)}


@pytest.mark.parametrize("norm", ["self", "ref"])
def test_scattering_builds_no_full_grid_view(monkeypatch: pytest.MonkeyPatch, norm: str) -> None:
    """Part III, §19.5 and review rule X5: ``estel.functional.scattering`` calls neither a
    full-grid view nor ``scatter_to_grid``, with each normalisation."""
    data = Data2D(_field((1, 2, 16, 16)), pbc=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operator = STOperator(data, 2, compute_PS=False)
    reference = None
    if norm == "ref":
        reference = estel.functional.scattering(data, operator.plan, operator.filters)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("scattering() built a full-grid view")

    backend = _registry.backend_named(data.backend)
    monkeypatch.setattr(_layout, "full_grid", refuse)
    monkeypatch.setattr(type(backend), "scatter_to_grid", refuse)
    _scattering._loop.cache_clear()  # JAX: trace the loop again, with the patches
    try:
        stats = estel.functional.scattering(
            data, operator.plan, operator.filters, reference, norm=norm
        )
        stats.to_vector()
        stats.reduce(iso=True).to_vector()
    finally:
        _scattering._loop.cache_clear()


def test_select_repeats_and_reorders_positions(dtype: str) -> None:
    """``select`` with a repeated and reordered sequence and a reversed slice, computed on the
    entries, gives the full-grid view indexed the same way."""
    planar = _planar(dtype, _registry.get_backend())
    chosen = planar.select("S3", j3=[2, 0, 2], l2=slice(None, None, -1))
    expected = np.asarray(planar.full_grid("S3"))[:, :, :, :, [2, 0, 2]][:, :, :, :, :, ::-1]
    np.testing.assert_array_equal(np.asarray(chosen.full_grid("S3")), expected)


def test_angular_fourier_is_applied_one_axis_at_a_time(dtype: str) -> None:
    """Part III, §19.6: one map per orientation axis of S4, each output float reading at most
    L complex entries (2 L floats), rather than one map over the Kronecker product of the
    axes, which grows as L to the power of twice their number."""
    planar = _planar(dtype, _registry.get_backend())
    maps = reductions.angular_fourier(planar._layout, 3, False)
    assert len(maps) == 3
    widths = [
        p.columns.shape[1]
        for m in maps
        for p in m.parts
        if p.columns is not None and p.weights is not None
    ]
    assert widths
    plan = planar._plan
    assert isinstance(plan, PlanarPlan)
    assert max(widths) <= 2 * plan.L


def _static_elements(found: Any, seen: set[int] | None = None) -> int:
    """The elements of the NumPy arrays reachable from ``found`` through containers, named
    tuples and the attributes of objects (cached properties included), each base array once."""
    seen = set() if seen is None else seen
    if id(found) in seen or isinstance(found, str | bytes | int | float | bool | type | None):
        return 0
    seen.add(id(found))
    if isinstance(found, np.ndarray):
        base = found
        while isinstance(base.base, np.ndarray):
            base = base.base
        if base is not found and id(base) in seen:
            return 0
        seen.add(id(base))
        return int(base.size)
    if isinstance(found, dict):
        found = list(found.values())
    if isinstance(found, tuple | list | set | frozenset):
        return sum(_static_elements(item, seen) for item in found)
    return sum(_static_elements(item, seen) for item in vars(found).values())


def test_layout_tables_grow_with_the_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Part III, §19.4 and §19.6 (T3.7): on 2D data with J = 5 and L = 4, the static arrays of
    the layout of the loop and of its normalisation, its maps' caches filled, hold fewer
    elements in all than the coefficients the layout stores: one row per chunk, no table per
    coefficient. Applying the maps (the normalisation of ``norm="self"``, ``to_vector``, the
    reductions, ``select``) makes no constant that large either: their indices are built inside
    the computation, from the tables of the chunks."""
    data = Data2D(_field((1, 1, 32, 32)), pbc=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operator = STOperator(data, 5, L=4, compute_PS=False)
    plan = operator.plan
    assert isinstance(plan, PlanarPlan)
    layout = _layout.base_layout(plan)
    present = _grid.auto_scales(plan)
    tables = _normalization._norm_tables(layout, present.shape, present.tobytes(), plan.L)
    assert all(
        layout.chunk_starts(name).shape == (layout.grid(name).n_chunks,) for name in layout.names
    )
    n_coefficients = sum(g.n_chunks * g.n_orientations for g in layout.grids)
    assert n_coefficients == sum(int(np.count_nonzero(g.present)) for g in layout.grids)
    assert _static_elements((layout, tables)) < n_coefficients

    sizes: list[int] = []

    def recording(backend: Any, array: Any, like: Any, dtype: Any = None) -> Any:
        sizes.append(int(np.asarray(array).size))
        return constant(backend, array, like, dtype)

    for module in (_layout, _normalization, _statistics):
        monkeypatch.setattr(module, "constant", recording)
    stats = operator.apply(data)
    stats.to_vector()
    stats.reduce(iso=True, angular_fourier=True, scale_cosine=True).to_vector()
    stats.select("S4", l1=[1, 0]).to_vector()
    assert sizes
    assert max(sizes) < n_coefficients
