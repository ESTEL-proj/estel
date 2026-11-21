"""The compact coefficient layout (task T2.15, issue #93; Part III, §19.2, §19.4, §19.5).

The values are compared with STL through the full-grid views in ``tests/parity/``, and the
accessors in ``test_statistics.py``. These tests check what those do not: that the buffer and the
full-grid views hold the same coefficients both ways, through the normalisation, the selection
and every reduction; that real entries take one float; and that the computation builds no
full-grid view.
"""

from __future__ import annotations

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
from estel.statistics import _layout

pytestmark = pytest.mark.maths("M-FLAT")


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    monkeypatch.setattr(_config, "_dtype", dtype)


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    field: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return field


def _apply(data: Data, J: int, **kwargs: Any) -> STStatistics:  # noqa: N803
    options: dict[str, Any] = {"compute_PS": False if data.ndim == 2 else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30
        return STOperator(data, J, **{**options, **kwargs}).apply(data)


def _cases() -> list[STStatistics]:
    """1D and 2D, periodic and not, a mask with empty means, cross channels, a channel without
    auto statistics (OP-30), each normalisation, then the selection and every reduction."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[14:17, 14:18] = True
    masked = Data2D(_field((2, 3, 32, 32)), mask=mask, pbc=False)
    matrix = np.array([[True, False, True], [False, False, True], [False, False, True]])
    planar = _apply(masked, 3, cross_matrix=matrix)
    line = Data1D(_field((2, 2, 32), seed=1), pbc=True)
    return [
        planar,
        _apply(masked, 3, norm="none"),
        _apply(line, 3, L=2, norm="none"),
        _apply(line.with_array(line.array), 3, L=2),
        planar.select(c1=[0, 2], j3=2),
        planar.select("S4", channel=(0, 2), l1=1),
        planar.reduce(iso=True),
        planar.reduce(angular_fourier=True, angular_harmonics=2),
        planar.reduce(iso=True, angular_fourier=True, scale_cosine=True, dj=2),
    ]


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


def test_buffer_and_full_grid_round_trip() -> None:
    """buffer -> full_grid -> buffer is the identity, and every position of the full grid
    outside the layout is NaN, in every case."""
    for stats in _cases():
        buffer = np.asarray(stats._buffer)
        np.testing.assert_array_equal(_buffer_from_grids(stats), buffer)
        assert np.isfinite(buffer).all()
        for grid in stats._grids:
            full = np.asarray(stats.full_grid(grid.name))
            assert full.shape[1:] == grid.shape
            assert np.array_equal(np.isnan(full), np.broadcast_to(~grid.present, full.shape))


def test_real_entries_take_one_float() -> None:
    """Mean, var, the auto S1 and S2 and the diagonal of S4 take one float, every other entry
    two; their imaginary part is 0 in the full-grid view."""
    data = Data2D(_field((1, 2, 16, 16)), pbc=True)
    stats = _apply(data, 2, norm="none")
    layout = stats._layout
    expected: dict[str, Callable[[NDArray[Any]], NDArray[Any]]] = {
        "mean": lambda p: np.zeros(len(p), bool),
        "var": lambda p: np.zeros(len(p), bool),
        "S1": lambda p: p[:, 0] != p[:, 1],
        "S2": lambda p: p[:, 0] != p[:, 1],
        "S3": lambda p: np.ones(len(p), bool),
        "S4": lambda p: ~((p[:, 0] == p[:, 1]) & (p[:, 2] == p[:, 3]) & (p[:, 5] == p[:, 6])),
    }
    for name, rule in expected.items():
        entries = layout.entries(name)
        np.testing.assert_array_equal(entries.is_complex, rule(entries.positions), err_msg=name)
        if name in ("mean", "var"):
            continue
        full = np.asarray(stats.full_grid(name))
        real = entries.positions[~entries.is_complex]
        assert np.all(full[(slice(None), *real.T)].imag == 0), name
    n_entries = sum(layout.entries(n).n_entries for n in layout.names)
    n_complex = sum(int(layout.entries(n).is_complex.sum()) for n in layout.names)
    assert layout.n_floats == n_entries + n_complex
    assert tuple(stats._buffer.shape) == (1, layout.n_floats)
    assert {s.level for s in layout.segments if s.family == "S4"} == {(0,), (1,)}


def test_scattering_builds_no_full_grid_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """Part III, §19.5 and review rule X5: ``estel.functional.scattering`` calls neither a
    full-grid view nor ``scatter_to_grid``, with each normalisation."""
    data = Data2D(_field((1, 2, 16, 16)), pbc=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        storing = STOperator(data, 2, compute_PS=False)
        loading = STOperator(data, 2, norm="ref", compute_PS=False)
    reference = estel.functional.scattering(data, storing.plan, storing.filters)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("scattering() built a full-grid view")

    backend = _registry.backend_named(data.backend)
    monkeypatch.setattr(_layout, "full_grid", refuse)
    monkeypatch.setattr(type(backend), "scatter_to_grid", refuse)
    _scattering._loop.cache_clear()  # JAX: trace the loop again, with the patches
    try:
        for plan, filters, ref in (
            (storing.plan, storing.filters, None),
            (loading.plan, loading.filters, reference),
        ):
            stats = estel.functional.scattering(data, plan, filters, ref)
            stats.to_vector()
            stats.reduce(iso=True).to_vector()
    finally:
        _scattering._loop.cache_clear()
