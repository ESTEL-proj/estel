"""The statistics of tensor products on the compact buffer, without their loop (task T3.6,
issue #137; maths.md M-IDX-TP, M-LP, M-S-TP, M-NORM-TP, M-FLAT; Part II, §12.7; Part III,
§19.4 to §19.6).

These tests, written before the operator of tensor products (T3.2), take the raw statistics from the
parity fixtures of T3.4 (``scattering_tp``), or from the oracle ``spec_apply_tensor_prod(...,
norm="none")`` on the fields of ``statistics_tp``, which stores only normalised values, place
them in the layout of the plan (``layout_tensor_prod``), and read them back through
``STStatistics``: the positions of every family, the normalisation, its references and the two
flat vectors are compared with the fixtures, in float64, with the tolerances of Part II, §14.3.
"""

from __future__ import annotations

import dataclasses
import tracemalloc
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import Data1D, Data2D, DataTensorProd, STStatistics
from estel._spec import tensor_prod as spec
from estel.backend import _registry
from estel.scattering._plan import (
    STPlanSingleDomain,
    STPlanTensorProd,
    build_statistics_plan_tensor_prod,
)
from estel.statistics import _grid, _layout, _normalization, _statistics
from estel.statistics._arrays import constant
from estel.statistics._statistics import statistics_of
from tests.parity._compare import assert_parity, full_grid, load_fixture, read_manifest
from tests.statistics.test_layout import _static_elements

pytestmark = pytest.mark.maths("M-IDX-TP", "M-LP", "M-S-TP", "M-NORM-TP", "M-CHAN", "M-FLAT")
GEOMETRIES = {"2D": Data2D, "1D": Data1D}
AXES = {"2D": 2, "1D": 1}
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
MODES = {"ST": "st", None: "none"}


@cache
def _arrays(case: str) -> dict[str, NDArray[Any]]:
    return load_fixture(case)


def _parameters(case: str) -> dict[str, Any]:
    parameters: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    return parameters


def _plan(
    field: NDArray[np.float64],
    blocks: list[str],
    pbc: list[bool],
    J: Any,  # noqa: N803 (maths.md notation)
    **options: Any,
) -> STPlanTensorProd:
    data = DataTensorProd(field, [GEOMETRIES[b] for b in blocks], pbc=pbc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-34
        return build_statistics_plan_tensor_prod(data, J, **options)


def _buffer(layout: _layout.Layout, grids: dict[str, NDArray[Any]]) -> Any:
    """The buffer of ``layout`` filled from STL's full grids, with NumPy: each entry at its
    position, its real part, then its imaginary part if it has one; on the current backend, in
    float64."""
    n_batch = int(grids["mean"].shape[0])
    floats = np.full((n_batch, layout.n_floats), np.nan)
    for grid in layout.grids:
        entries = layout.entries(grid.name)
        values = grids[grid.name].reshape(n_batch, -1)[
            :, _layout.flat_positions(entries.positions, grid.shape)
        ]
        floats[:, entries.first] = values.real
        floats[:, entries.first[entries.is_complex] + 1] = values.imag[:, entries.is_complex]
    # Every position of the layout holds a coefficient of STL: none falls on a NaN.
    assert np.isfinite(floats).all()
    return _registry.current_backend().asarray(floats, "float64", None)


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _scattering_runs() -> list[Any]:
    return [
        pytest.param(name, lowpass, id=f"{name}-{MODES[lowpass]}")
        for name, config in _parameters("scattering_tp")["configs"].items()
        for lowpass in config["lowpass"]
    ]


@pytest.mark.parametrize(("name", "lowpass"), _scattering_runs())
def test_positions_equal_the_fixture(name: str, lowpass: str | None) -> None:
    """For every configuration of ``scattering_tp``, the layout built from the per-block index
    sets holds a coefficient exactly where STL's full grid is finite, family by family: the
    values placed in the buffer come back through ``full_grid`` with STL's shapes and NaN at
    the same positions (Part II, §14.3). S3 and S4 of ``lowpass=None`` are those of ``"ST"``
    (maths.md M-LP)."""
    parameters = _parameters("scattering_tp")
    config = parameters["configs"][name]
    case = parameters["cases"][config["case"]]
    arrays = _arrays("scattering_tp")
    matrix = config.get("cross_matrix")
    plan = _plan(
        arrays[f"{name}_field"],
        case["blocks"],
        case["pbc"],
        case["J"],
        lowpass=lowpass,
        cross_matrix=None if matrix is None else np.array(matrix),
    )
    prefix = f"{name}_{MODES[lowpass]}"
    grids = {
        "mean": arrays[f"{prefix}_mean"],
        "var": arrays[f"{prefix}_var"],
        **{f: full_grid(arrays, f"{prefix}_{f}") for f in ("S1", "S2")},
        **{f: full_grid(arrays, f"{name}_st_{f}") for f in ("S3", "S4")},
    }
    stats = statistics_of(_buffer(_layout.layout_tensor_prod(plan), grids), plan)
    for family in FAMILIES:
        assert_parity(_numpy(stats.full_grid(family)), grids[family])


@cache
def _statistics_tensor_prod() -> dict[str, Any]:
    """The three runs of ``statistics_tp`` (``norm="self"``, ``norm="ref"`` by its reference,
    ``norm="self"`` with ``ref_batch_mean``), from the raw statistics of the oracle on the
    fixture's fields."""
    parameters = _parameters("statistics_tp")
    arrays = _arrays("statistics_tp")
    blocks, pbc, J = parameters["blocks"], parameters["pbc"], parameters["J"]  # noqa: N806

    def run(key: str, norm: str, ref: Any = None, batch_mean: bool = False) -> STStatistics:
        raw = spec.spec_apply_tensor_prod(
            arrays[key],
            [AXES[b] for b in blocks],
            pbc,
            J,
            lowpass="ST",
            norm="none",
            standardize=False,  # STL's default (OP-120)
        )
        plan = _plan(arrays[key], blocks, pbc, J)
        buffer = _buffer(_layout.layout_tensor_prod(plan), raw)
        return statistics_of(buffer, plan, ref, norm=norm, ref_batch_mean=batch_mean)

    store = run("field", "self")
    return {
        "store": store,
        "load": run("other_field", "ref", store._reference),
        "store_batch_mean": run("field", "self", batch_mean=True),
    }


def test_normalisation_equals_the_fixture() -> None:
    """``statistics_tp``: the normalised statistics of the three runs, their references
    (``var_ref``, ``S2_ref_sqrt_chan_diag``) and the real and complex flat vectors equal STL's
    (maths.md M-NORM-TP, M-FLAT). S1 and S2 are normalised at every scale, the low-pass scales
    included, S3 and S4 by the wavelet scales of the reference only."""
    arrays = _arrays("statistics_tp")
    for run, stats in _statistics_tensor_prod().items():
        var_ref = arrays[
            "store_batch_mean_var_ref" if run == "store_batch_mean" else "store_var_ref"
        ]
        scale = float(np.sqrt(np.max(arrays[f"{run}_var"] * var_ref)))
        assert_parity(_numpy(stats.full_grid("mean")), arrays[f"{run}_mean"], scale=scale)
        assert_parity(_numpy(stats.full_grid("var")), arrays[f"{run}_var"])
        for family in ("S1", "S2", "S3", "S4"):
            assert_parity(_numpy(stats.full_grid(family)), full_grid(arrays, f"{run}_{family}"))
        if run != "load":
            assert stats._reference is not None
            assert_parity(_numpy(stats._reference.var), arrays[f"{run}_var_ref"])
            assert_parity(_numpy(stats._reference.s2_sqrt), full_grid(arrays, f"{run}_s2_ref"))
    store = _statistics_tensor_prod()["store"]
    assert_parity(_numpy(store.to_vector()), arrays["vector_real"])
    assert_parity(_numpy(store._to_complex_vector()), arrays["vector_complex"])


# ---------------------------------------------------------------------- memory


def _three_blocks() -> STPlanTensorProd:
    """2D + 1D + 1D, 16 x 16 x 16 x 16, J = (2, 2, 2), L = (4, 2, 2), with the low-pass filter
    of every block: its full-grid S4 has 2,097,152 entries for 262,144 coefficients."""
    field = np.zeros((1, 1, 16, 16, 16, 16))
    return _plan(field, ["2D", "1D", "1D"], [True, False, True], [2, 2, 2])


def test_three_block_layout_allocates_no_full_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the layout of a small three-block tensor product and the tables of its
    normalisation allocates nothing the size of its full grid, not even a boolean mask (decision
    of 22 November 2026 on #23; task T3.6), and their static arrays hold fewer elements than the
    coefficients the layout stores: one row per chunk. Applying the maps (the normalisation of
    ``norm="self"``, ``to_vector``, ``select``) makes no constant that large either."""

    def build(plan: STPlanTensorProd) -> tuple[_layout.Layout, Any]:
        present = _grid.auto_scales_tensor_prod(plan)
        layout = _layout.layout_tensor_prod(plan)
        tables = _normalization._norm_tables(
            layout,
            present.shape,
            present.tobytes(),
            int(np.prod(plan.L)),
            _normalization.blocks_tensor_prod(plan),
        )
        return layout, tables

    # A tiny tensor product first, so that the one-time costs of NumPy are not counted.
    build(_plan(np.zeros((1, 1, 8, 8, 8)), ["2D", "1D"], [True, True], [1, 1]))
    plan = _three_blocks()
    _layout.layout_tensor_prod.cache_clear()
    _normalization._norm_tables.cache_clear()
    tracemalloc.start()
    try:
        layout, tables = build(plan)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    full_grid_entries = sum(int(np.prod(g.shape, dtype=np.int64)) for g in layout.grids)
    n_coefficients = sum(layout.entries(g.name).n_entries for g in layout.grids)
    # maths.md §3.9: S1 and S2 9 x 5 x 5, S3 16² x 3³, S4 16³ x 4³ coefficients.
    assert n_coefficients == 2 + 2 * 9 * 5 * 5 + 16**2 * 3**3 + 16**3 * 4**3
    assert full_grid_entries == 2 + 2 * 3**3 * 16 + 8**2 * 16**2 + 8**3 * 16**3
    # Less than a boolean mask of the full grid, one byte per entry (2.1 MB here).
    assert peak < full_grid_entries
    assert _static_elements((layout, tables)) < n_coefficients

    sizes: list[int] = []

    def recording(backend: Any, array: Any, like: Any, dtype: Any = None) -> Any:
        sizes.append(int(np.asarray(array).size))
        return constant(backend, array, like, dtype)

    for module in (_layout, _normalization, _statistics):
        monkeypatch.setattr(module, "constant", recording)
    buffer = _registry.current_backend().asarray(
        np.random.default_rng(0).random((1, layout.n_floats)) + 0.5, "float64", None
    )
    stats = statistics_of(buffer, plan, norm="self")
    stats.to_vector()
    stats.select(("S1", "S3"), channel=(0, 0)).to_vector()
    assert sizes
    assert max(sizes) < n_coefficients


# ---------------------------------------------------------------------- accessors


@cache
def _two_channels() -> STStatistics:
    """Case C of maths.md §3 with two channels, normalised by itself: low-pass scales in both
    blocks (S1 and S2 with four orientation shapes), from the oracle."""
    store: STStatistics = _statistics_tensor_prod()["store"]
    return store


def test_full_grid_is_refused_above_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Part II, §12.7 (LF65): a full-grid view of a tensor product over the threshold, batch
    included, in its dtype, is refused with its size and what to use instead; the threshold is
    lowered here rather than allocating gigabytes."""
    stats = _two_channels()
    s4 = _layout.full_grid_bytes(stats._layout, "S4", 2, 8)
    assert s4 == 2 * 2 * 2 * (2 * 1) ** 3 * (4 * 2) ** 3 * 16
    monkeypatch.setattr(_statistics, "_FULL_GRID_LIMIT", s4 - 1)
    with pytest.raises(ValueError, match=r"GiB.*to_vector\(\).*select"):
        stats.full_grid("S4")
    assert _numpy(stats.full_grid("S3")).shape == (2, 2, 2, 2, 1, 2, 1, 4, 2, 4, 2)


def test_select_takes_family_and_channels() -> None:
    """``select`` on a tensor product: ``family`` and the channels (``c1``, ``c2``,
    ``channel``), on families with several orientation shapes too; any scale or orientation
    axis raises an error that points to LF44 (Part II, §12.7)."""
    stats = _two_channels()
    s1 = _numpy(stats.full_grid("S1"))
    chosen = stats.select(("S1", "S4"), c1=[1, 0])
    np.testing.assert_array_equal(_numpy(chosen.full_grid("S1")), s1[:, [1, 0]])
    s4 = _numpy(stats.full_grid("S4"))
    pair = stats.select("S4", channel=(1, 0))
    np.testing.assert_array_equal(_numpy(pair.full_grid("S4")), s4[:, 1, 0])
    assert [g.name for g in pair._grids] == ["S4"]
    np.testing.assert_array_equal(
        _numpy(stats.select("var", c=1).full_grid("var")), _numpy(stats.full_grid("var"))[:, 1]
    )
    expected = np.concatenate(
        [_flat(s1[:, [1, 0]], real_part=True), _flat(s4[:, [1, 0]], real_part=False)]
    )
    np.testing.assert_array_equal(_numpy(chosen.to_vector()), expected)


@pytest.mark.parametrize(
    ("family", "index", "match"),
    [
        pytest.param("S3", {"j[0]": 0}, r"LF44\). Until then, use full_grid", id="block-axis"),
        pytest.param("S3", {"l": 0}, r"in block order \(maths\.md M-S-TP\)", id="single-domain-l"),
        pytest.param("S4", {"j3": 1}, r"or to_vector\.$", id="single-domain-j3"),
        pytest.param(
            "S1",
            {"c": 0},
            r"no selected family \(S1\) has an axis 'c'; valid axis names: c1, c2 \(channel="
            r"\(a, b\) stands for c1, c2; on a tensor product, the scale and orientation axes "
            r"of the blocks have no name yet, LF44\)\.$",
            id="valid-names",
        ),
        pytest.param(
            ("S1", "mean"),
            {"c": 0, "c1": 0},
            r"no selected family has every axis of \['c', 'c1'\]: mean has c; S1 has c1, c2\.$",
            id="every-axis",
        ),
    ],
)
def test_select_errors_name_the_channels_only(
    family: str | tuple[str, ...], index: dict[str, int], match: str
) -> None:
    """On a tensor product, the errors of ``select`` name the channel axes only, the axes of
    the blocks having no public name yet (LF44, answer 2 of the maintainer on #137), and say
    what to use instead: ``full_grid`` on a small case, or ``to_vector``."""
    with pytest.raises(ValueError, match=match) as raised:
        _two_channels().select(family, **index)
    # No axis of the blocks is named, but the one the caller passed.
    message = str(raised.value).replace(f"select({next(iter(index))}=...)", "")
    assert "[0]" not in message
    assert "[1]" not in message


def _flat(grid: NDArray[Any], real_part: bool) -> NDArray[Any]:
    """M-FLAT of one family of a full grid: each batch element in turn, its finite entries in
    row-major order, the real part alone or (re, im)."""
    rows = grid.reshape(grid.shape[0], -1)
    kept = rows[:, np.isfinite(rows[0])]
    pairs = kept.real if real_part else np.stack([kept.real, kept.imag], axis=-1)
    return pairs.reshape(-1)


@pytest.mark.parametrize("reduction", ["iso", "angular_fourier", "scale_cosine"])
def test_reduce_raises_before_computing(reduction: str) -> None:
    """The reductions are not defined on tensor products (maths.md M-NORM-TP, OP-44): a clear
    error, raised before any computation, even with arguments that planar statistics refuse
    (``scale_cosine`` without ``iso``)."""
    with pytest.raises(ValueError, match=r"OP-44.*LF26"):
        _two_channels().reduce(**{reduction: True})


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """``estel.io``, schema version 5: the statistics of a tensor product, their blocks, J, L,
    low-pass flags and reference come back unchanged; the plan is rebuilt with the signature of
    ``DataTensorProd``."""
    stats = _two_channels()
    path = tmp_path / "tensor_prod.npz"
    stats.save(path)
    loaded = STStatistics.load(path)
    assert loaded._plan == stats._plan
    assert loaded._layout == stats._layout
    for family in FAMILIES:
        np.testing.assert_array_equal(
            _numpy(loaded.full_grid(family)), _numpy(stats.full_grid(family))
        )
    np.testing.assert_array_equal(_numpy(loaded.to_vector()), _numpy(stats.to_vector()))
    assert loaded._reference is not None
    assert stats._reference is not None
    assert loaded._reference.blocks == stats._reference.blocks
    np.testing.assert_array_equal(
        _numpy(loaded._reference.s2_sqrt), _numpy(stats._reference.s2_sqrt)
    )


def test_reference_of_other_tensor_prod_data_is_accepted() -> None:
    """A reference from another tensor product with the same blocks, J, L, low-pass flags and
    channels is accepted, whatever its sizes and periodicity (maths.md M-NORM-TP); a planar
    reference is refused on a tensor product, and the reference of a tensor product on planar
    statistics."""
    reference = _two_channels()._reference
    assert reference is not None
    rng = np.random.default_rng(3)
    other = _plan(rng.standard_normal((2, 2, 32, 32, 16)), ["2D", "1D"], [False, True], [2, 1])
    buffer = _registry.current_backend().asarray(
        rng.random((2, _layout.layout_tensor_prod(other).n_floats)) + 0.5, "float64", None
    )
    stats = statistics_of(buffer, other, reference, norm="ref")
    assert stats._reference is not None
    planar = estel.STOperator(
        Data2D(rng.standard_normal((2, 2, 16, 16)), pbc=True), 2, compute_PS=False
    ).apply(Data2D(rng.standard_normal((2, 2, 16, 16)), pbc=True))
    assert planar._reference is not None
    with pytest.raises(ValueError, match="planar"):
        statistics_of(buffer, other, planar._reference, norm="ref")
    assert isinstance(planar._plan, STPlanSingleDomain)
    with pytest.raises(ValueError, match="tensor product"):
        _normalization.check_reference_single_domain(
            reference, planar._plan, 2, reference.var.dtype
        )


@pytest.mark.parametrize(
    ("what", "blocks", "shape", "J"),
    [
        pytest.param("J", ["2D", "1D"], (32, 32, 16), [3, 1], id="J"),
        pytest.param("blocks", ["1D", "2D"], (8, 16, 16), [1, 2], id="blocks"),
        # 9 is odd: block 1 has no low-pass filter (M-LP).
        pytest.param("low-pass flags", ["2D", "1D"], (16, 16, 9), [2, 1], id="low-pass"),
    ],
)
def test_reference_of_other_tensor_prod_data_is_refused(
    what: str,
    blocks: list[str],
    shape: tuple[int, ...],
    J: list[int],  # noqa: N803 (maths.md notation)
) -> None:
    """A reference from a tensor product whose blocks, J or low-pass flags differ is refused
    with an error naming what differs (maths.md M-NORM-TP; Part IV, R-Y03)."""
    reference = _two_channels()._reference
    assert reference is not None
    plan = _plan(np.zeros((2, 2, *shape)), blocks, [True, True], J)
    n_floats = _layout.layout_tensor_prod(plan).n_floats
    data = _registry.current_backend().asarray(np.ones((2, n_floats)), "float64", None)
    with pytest.raises(ValueError, match=what):
        statistics_of(data, plan, reference, norm="ref")


def test_layout_refuses_a_segment_of_another_orientation_shape() -> None:
    """``Layout`` checks that the orientations of each segment are those of its chunks: a
    segment of S1 at a low-pass scale holds orientation 0 only along that block (maths.md
    M-LP), and claiming the full orientations for it is refused (task T3.6)."""
    layout = _two_channels()._layout
    index, segment = next(
        (i, s)
        for i, s in enumerate(layout.segments)
        if s.family == "S1" and s.orientations != (4, 2)
    )
    wrong = dataclasses.replace(segment, orientations=(4, 2))
    segments = (*layout.segments[:index], wrong, *layout.segments[index + 1 :])
    with pytest.raises(ValueError, match="orientations of a segment are not those of its chunks"):
        _layout.Layout(grids=layout.grids, segments=segments, n_floats=layout.n_floats)
