"""The 2D power spectrum as a family of ``STStatistics``, without its operator (task T4.4, issue
#151; maths.md §1.9, M-CHAN, M-NORM, M-FLAT; Part II, §12.7; Part III, §19.4 to §19.6).

These tests, written before the spectrum operator (T4.1), take the raw statistics of the fixture
``statistics_ps_2d`` (T4.3): the scattering statistics from the planar oracle
(``spec_apply(..., norm="none")``) on its fields, the raw spectrum of the target from
``<config>_raw_PS``, and that of the second field from ``<config>_load_PS`` times the product of
the spectrum references (``<config>_ps_ref``). They place them in the layout of a plan with the
spectrum (``build_plan(..., compute_PS=True)``, the private path until T4.1), and read them back
through ``STStatistics``: the positions, the normalisation, its references and the flat vectors
are compared with the fixture, in float64, with the tolerances of Part II, §14.3.
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data2D, STOperator, STStatistics, _config
from estel._spec import planar as spec
from estel.backend import _registry
from estel.io import _schema
from estel.scattering._plan import PlanarPlan, build_plan
from estel.statistics import _grid, _layout, _normalization, _statistics
from estel.statistics._arrays import constant
from estel.statistics._normalization import SpectrumParameters
from estel.statistics._statistics import statistics_of
from tests.parity._compare import assert_parity, load_fixture, read_manifest
from tests.statistics.test_layout import _static_elements
from tests.statistics.test_tensor_prod import _buffer, _flat

pytestmark = pytest.mark.maths("M-CHAN", "M-NORM", "M-FLAT")
CASE = "statistics_ps_2d"
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4", "PS")
SCATTERING = ("mean", "var", "S1", "S2", "S3", "S4")


@cache
def _arrays() -> dict[str, NDArray[Any]]:
    return load_fixture(CASE)


def _configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = read_manifest()[f"{CASE}.npz"]["parameters"]["configs"]
    return configs


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _matrix(config: dict[str, Any]) -> NDArray[np.bool_] | None:
    matrix = config.get("cross_matrix")
    return None if matrix is None else np.array(matrix, dtype=bool)


def _plan(name: str, field: str = "field", **options: Any) -> PlanarPlan:
    """The plan of configuration ``name`` with the spectrum, on its field ``field``."""
    config = _configs()[name]
    arrays = _arrays()
    mask = arrays[f"{name}_mask"] > 0 if f"{name}_mask" in arrays else None
    data = Data2D(arrays[f"{name}_{field}"], mask=mask, pbc=config["pbc"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13: tested elsewhere
        plan = build_plan(
            data,
            config["J"],
            4,
            cross_matrix=_matrix(config),
            compute_PS=True,
            n_bins=config.get("n_bins"),
            J_PS=config["J"],  # STL passes its J to its spectrum (OP-120)
            Jmin_PS=config.get("Jmin", 0),
            **options,
        )
    assert plan.spectrum is not None
    bins = _arrays()[f"{name}_ps_bins"]  # [J, J_PS, J_min, n_b] of STL
    assert (plan.J, plan.spectrum.J_PS, plan.spectrum.Jmin_PS, plan.spectrum.n_bins) == tuple(
        int(b) for b in bins
    )
    return plan


def _raw(name: str, field: str) -> dict[str, NDArray[Any]]:
    """The raw statistics of field ``field`` of configuration ``name``, in STL's full grids: the
    scattering statistics from the oracle, the spectrum from the fixture."""
    config = _configs()[name]
    arrays = _arrays()
    mask = arrays[f"{name}_mask"] > 0 if f"{name}_mask" in arrays else None
    raw = spec.spec_apply(
        arrays[f"{name}_{field}"],
        2,
        config["pbc"],
        mask,
        config["J"],
        4,
        norm="none",
        cross_matrix=_matrix(config),
    )
    grids = {family: raw[family] for family in SCATTERING}
    target = arrays[f"{name}_raw_PS"]
    if field == "field":
        return {**grids, "PS": target}
    # The other field: its normalised spectrum times the target's references. A pair normalised
    # to NaN (a channel without auto spectrum, OP-30) has no raw value in the fixture; any value
    # does there, since the normalisation drops it.
    ps_ref = arrays[f"{name}_ps_ref"]
    found = arrays[f"{name}_load_PS"] * ps_ref[:, :, None, :] * ps_ref[:, None, :, :]
    return {**grids, "PS": np.where(np.isfinite(target), np.nan_to_num(found), np.nan)}


@pytest.mark.parametrize("name", list(_configs()))
def test_positions_equal_the_fixture(name: str) -> None:
    """For every configuration, the layout with the spectrum holds a coefficient exactly where
    the raw full grids are finite, PS included, one chunk per channel pair c1 ≤ c2 of
    ``cross_matrix`` (maths.md §1.9, M-CHAN): placed in the buffer, the values come back through
    ``full_grid`` with STL's shapes and NaN at the same positions; PS is the fixture's raw
    spectrum, and the last segment of the buffer, at level 0."""
    plan = _plan(name)
    layout = _layout.base_layout(plan)
    grids = _raw(name, "field")
    stats = statistics_of(_buffer(layout, grids), plan)
    for family in FAMILIES:  # PS: the fixture's <config>_raw_PS (_raw)
        assert_parity(_numpy(stats.full_grid(family)), grids[family])
    last = layout.segments[-1]
    assert (last.family, last.level, last.index_names) == ("PS", (0,), ("c1", "c2", "bin"))
    assert layout.span("PS")[1] == layout.n_floats
    assert layout.grid("PS").real.all()  # one float per entry


@cache
def _runs(name: str) -> dict[str, STStatistics]:
    """The two runs of configuration ``name``: ``norm="self"`` on the target, with its
    ``ref_batch_mean``, and ``norm="ref"`` on the second field by the target's reference."""
    config = _configs()[name]
    store_plan = _plan(name, norm="self", ref_batch_mean=config["norm_batch_mean"])
    store = statistics_of(_buffer(_layout.base_layout(store_plan), _raw(name, "field")), store_plan)
    load_plan = _plan(name, "ref_field", norm="ref")
    load = statistics_of(
        _buffer(_layout.base_layout(load_plan), _raw(name, "ref_field")),
        load_plan,
        store._reference,
    )
    return {"store": store, "load": load}


@pytest.mark.parametrize("name", list(_configs()))
def test_normalisation_equals_the_fixture(name: str) -> None:
    """The normalised statistics of both runs, every family and the spectrum, their references
    (``var_ref``, ``S2_ref_sqrt_chan_diag``, ``PS_ref_sqrt_chan_diag``) and the four flat
    vectors equal STL's (maths.md M-NORM, M-FLAT, OP-30, OP-37): PS divided by
    σᴾˢ_{c1}(b) σᴾˢ_{c2}(b), last in every vector, its real part only in the real one."""
    arrays = _arrays()
    runs = _runs(name)
    for run, stats in runs.items():
        scale = float(np.sqrt(np.max(arrays[f"{name}_{run}_var"] * arrays[f"{name}_var_ref"])))
        assert_parity(_numpy(stats.full_grid("mean")), arrays[f"{name}_{run}_mean"], scale=scale)
        for family in FAMILIES[1:]:
            assert_parity(_numpy(stats.full_grid(family)), arrays[f"{name}_{run}_{family}"])
    reference = runs["store"]._reference
    assert reference is not None
    assert_parity(_numpy(reference.var), arrays[f"{name}_var_ref"])
    assert_parity(_numpy(reference.s2_sqrt), arrays[f"{name}_s2_ref"])
    assert_parity(_numpy(reference.ps_sqrt), arrays[f"{name}_ps_ref"])
    store = runs["store"]
    assert_parity(_numpy(store._to_complex_vector()), arrays[f"{name}_vector"])
    assert_parity(_numpy(store.to_vector()), arrays[f"{name}_vector_real"])
    assert_parity(
        _numpy(store._to_complex_vector(mean_along_batch=True)),
        arrays[f"{name}_vector_batch_mean"],
    )
    assert_parity(_numpy(runs["load"]._to_complex_vector()), arrays[f"{name}_load_vector"])


# ---------------------------------------------------------------------- accessors


def _periodic() -> STStatistics:
    """Two channels, every pair, normalised by themselves (``periodic``)."""
    return _runs("periodic")["store"]


def test_select_by_channel_pair_returns_the_spectrum_too() -> None:
    """``select(channel=(a, b))`` returns every family of that pair, the spectrum included, in
    one ``STStatistics`` and one vector (Part II, §12.7); a selection by ``bin`` returns the
    spectrum alone, and one on a scale axis leaves it out."""
    stats = _periodic()
    grids = {f: _numpy(stats.full_grid(f)) for f in FAMILIES}
    pair = stats.select(channel=(0, 1))
    assert [g.name for g in pair._grids] == ["S1", "S2", "S3", "S4", "PS"]
    for family in ("S1", "S2", "S3", "S4", "PS"):
        np.testing.assert_array_equal(_numpy(pair.full_grid(family)), grids[family][:, 0, 1])
    expected = np.concatenate(
        [_flat(grids[f][:, 0, 1], real_part=f in ("S1", "S2", "PS")) for f in pair._layout.names]
    )
    np.testing.assert_array_equal(_numpy(pair.to_vector()), expected)

    bins = stats.select(bin=[3, 1])
    assert [g.name for g in bins._grids] == ["PS"]
    assert bins._grids[0].axes == ("c1", "c2", "bin")
    np.testing.assert_array_equal(_numpy(bins.full_grid("PS")), grids["PS"][..., [3, 1]])
    one = stats.select("PS", channel=(1, 1), bin=2)
    np.testing.assert_array_equal(_numpy(one.full_grid("PS")), grids["PS"][:, 1, 1, 2])

    assert [g.name for g in stats.select(j3=1)._grids] == ["S3", "S4"]
    assert [g.name for g in stats.select(l=2)._grids] == ["S1", "S2"]
    with pytest.raises(ValueError, match=r"valid axis names: bin, c1, c2"):
        stats.select("PS", j=0)


def test_reduce_keeps_the_spectrum() -> None:
    """``iso``, ``angular_fourier`` and ``scale_cosine`` leave the spectrum as it is (maths.md
    §1.9, "Reductions"); it stays last in the vector."""
    stats = _periodic()
    reduced = stats.reduce(iso=True, angular_fourier=True, scale_cosine=True)
    ps = _numpy(stats.full_grid("PS"))
    np.testing.assert_array_equal(_numpy(reduced.full_grid("PS")), ps)
    vector = _numpy(reduced.to_vector())
    tail = _flat(ps, real_part=True)
    np.testing.assert_array_equal(vector[-tail.shape[0] :], tail)


def test_save_and_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``estel.io``, schema version 9: the spectrum, its reference and its description in the
    plan come back unchanged; a file of version 8 is refused, with what to do."""
    stats = _runs("spectrum_options")["store"]
    path = tmp_path / "ps.npz"
    stats.save(path)
    loaded = STStatistics.load(path)
    assert loaded._plan == stats._plan
    assert isinstance(loaded._plan, PlanarPlan)
    assert isinstance(stats._plan, PlanarPlan)
    assert loaded._plan.spectrum == stats._plan.spectrum
    assert loaded._layout == stats._layout
    for family in FAMILIES:
        np.testing.assert_array_equal(
            _numpy(loaded.full_grid(family)), _numpy(stats.full_grid(family))
        )
    assert loaded._reference is not None
    assert stats._reference is not None
    assert loaded._reference.spectrum is not None
    assert stats._reference.spectrum is not None
    expected = SpectrumParameters(n_bins=5, J_PS=3, Jmin_PS=1)
    assert loaded._reference.spectrum.parameters == stats._reference.spectrum.parameters == expected
    np.testing.assert_array_equal(
        _numpy(loaded._reference.ps_sqrt), _numpy(stats._reference.ps_sqrt)
    )
    old = tmp_path / "version8.npz"
    monkeypatch.setattr(_schema, "SCHEMA_VERSION", 8)
    stats.save(old)
    monkeypatch.undo()
    assert _schema.SCHEMA_VERSION == 9
    with pytest.raises(ValueError, match=r"schema version 8.*reads version 9"):
        STStatistics.load(old)


def _with_reference(reference: Any, **spectrum: Any) -> STStatistics:
    """Statistics of another operator, 64 x 64 and not periodic, J = 3 and J_PS = 3 (its default
    would be 4), normalised by ``reference``."""
    rng = np.random.default_rng(5)
    data = Data2D(rng.standard_normal((2, 2, 64, 64)), pbc=False)
    plan = build_plan(data, 3, 4, norm="ref", compute_PS=True, **{"J_PS": 3, **spectrum})
    buffer = _registry.current_backend().asarray(
        rng.random((2, _layout.base_layout(plan).n_floats)) + 0.5, "float64", None
    )
    return statistics_of(buffer, plan, reference)


def test_reference_of_another_operator() -> None:
    """A reference from another operator is accepted when its J, L, channels and the n_bins,
    J_PS and J_min of its spectrum match, whatever its size and periodicity, and refused
    otherwise with an error naming what differs (maths.md M-NORM, §1.9); a reference without the
    spectrum is refused on statistics with it."""
    reference = _periodic()._reference
    stats = _with_reference(reference)
    assert np.isfinite(_numpy(stats.full_grid("PS"))[:, [0, 0, 1], [0, 1, 1]]).all()
    with pytest.raises(ValueError, match=r"n_bins = 6 \(the operator has 5\)"):
        _with_reference(reference, n_bins=5)
    with pytest.raises(ValueError, match=r"Jmin_PS = 0 \(the operator has 1\)"):
        _with_reference(reference, Jmin_PS=1)
    data = Data2D(np.random.default_rng(6).standard_normal((1, 2, 32, 32)), pbc=True)
    without = STOperator(data, 3, 4, compute_PS=False).apply(data)._reference
    with pytest.raises(ValueError, match="holds no power spectrum"):
        _with_reference(without)


def test_reference_with_the_spectrum_on_statistics_without_it() -> None:
    """A reference that holds the spectrum is accepted for statistics without it, its σᴾˢ
    ignored, as STL ignores ``PS_ref`` without ``compute_PS`` (decision of the maintainer on
    #157): the result equals that of the same reference stripped of its spectrum."""
    reference = _periodic()._reference
    assert reference is not None
    assert reference.spectrum is not None
    stripped = dataclasses.replace(reference, sigma_ps=None, spectrum=None)
    rng = np.random.default_rng(8)
    data = Data2D(rng.standard_normal((2, 2, 32, 32)), pbc=True)
    plan = build_plan(data, 3, 4, norm="ref")
    assert plan.spectrum is None
    buffer = _registry.current_backend().asarray(
        rng.random((2, _layout.base_layout(plan).n_floats)) + 0.5, "float64", None
    )
    found = statistics_of(buffer, plan, reference)
    assert "PS" not in found._layout.names
    np.testing.assert_array_equal(
        _numpy(found.to_vector()), _numpy(statistics_of(buffer, plan, stripped).to_vector())
    )


def test_gradients_through_the_spectrum_reference() -> None:
    """Gradients through σᴾˢ follow M-NORM (OP-18, LF21): with ``norm="self"`` they flow through
    the reference, with ``norm="ref"`` it is a constant. The loss is the sum of the normalised
    cross spectrum PS[0, 1, b] / (σᴾˢ_0(b) σᴾˢ_1(b)), on a raw buffer x without the operator; its
    gradient with respect to the auto spectrum x₀₀(b) is -x₀₁ / (2 x₀₀ √(x₀₀ x₁₁)) when the
    reference is that of x (``norm="self"``), and 0 when the reference, computed from x too, is
    passed with ``norm="ref"``."""
    backend = _registry.current_backend()
    xp = backend.xp
    rng = np.random.default_rng(9)
    data = Data2D(rng.standard_normal((1, 2, 32, 32)), pbc=True)
    plans = {norm: build_plan(data, 3, 4, norm=norm, compute_PS=True) for norm in ("self", "ref")}
    layout = _layout.base_layout(plans["self"])
    n_bins = layout.grid("PS").n_orientations
    starts = layout.chunk_starts("PS")  # the pairs (0, 0), (0, 1), (1, 1)
    auto_0, cross, auto_1 = (int(a) + np.arange(n_bins) for a in starts)
    x = rng.random((1, layout.n_floats)) + 0.5

    def loss(buffer: Any, norm: str) -> Any:
        stats = statistics_of(buffer, plans["self"])
        if norm == "ref":
            stats = statistics_of(buffer, plans["ref"], stats._reference)
        return xp.sum(stats.select("PS", channel=(0, 1)).to_vector())

    grads = {
        norm: _numpy(
            backend.value_and_grad(lambda b, n=norm: loss(b, n))(
                backend.asarray(x, "float64", None)
            )[1]
        )
        for norm in ("self", "ref")
    }
    root = np.sqrt(x[0, auto_0] * x[0, auto_1])
    np.testing.assert_allclose(
        grads["self"][0, auto_0], -x[0, cross] / (2 * x[0, auto_0] * root), rtol=1e-12
    )
    np.testing.assert_allclose(grads["self"][0, cross], 1 / root, rtol=1e-12)
    np.testing.assert_array_equal(grads["ref"][0, auto_0], 0)
    np.testing.assert_array_equal(grads["ref"][0, auto_1], 0)
    np.testing.assert_allclose(grads["ref"][0, cross], 1 / root, rtol=1e-12)


def test_an_operator_rebuilt_from_statistics_computes_the_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Since the spectrum operator (T4.1), an operator rebuilt from statistics with the spectrum
    computes them again, the spectrum included (``STOperator.from_statistics``): on the target of
    ``periodic``, the same statistics as the fixture's ``store`` run."""
    monkeypatch.setattr(_config, "_dtype", "float64")  # the dtype of the statistics
    stats = _periodic()
    operator = STOperator.from_statistics(stats)
    data = Data2D(_arrays()["periodic_field"], pbc=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the reference held is replaced
        again = operator.apply(data)
    for family in FAMILIES:
        assert_parity(_numpy(again.full_grid(family)), _numpy(stats.full_grid(family)))


@pytest.mark.parametrize(
    ("shape", "options", "error", "match"),
    [
        pytest.param(
            (32, 32),
            {"Jmin_PS": 2},
            ValueError,
            r"0 ≤ Jmin_PS ≤ J_PS - 2.*compute_PS=False",
            id="Jmin_PS",
        ),
        # J = 1, so that the scattering statistics take the odd size (M-DATA-2D).
        pytest.param((32, 31), {"J": 1}, ValueError, r"even sizes.*OP-20", id="odd"),
        pytest.param((32, 32), {"n_bins": 0}, ValueError, r"n_bins=0: pass a positive", id="bins"),
        pytest.param((32, 32), {"n_bins": 4.0}, TypeError, r"n_bins=4\.0: pass", id="bins-type"),
        pytest.param((32, 32), {"Jmin_PS": 0.5}, TypeError, r"Jmin_PS=0\.5: pass", id="Jmin-type"),
    ],
)
def test_plan_refuses_a_spectrum_it_cannot_compute(
    shape: tuple[int, int], options: dict[str, Any], error: type[Exception], match: str
) -> None:
    """The plan refuses the spectrum with J_min outside 0 … J_PS - 2 (OP-93), on an odd size
    (OP-20) and with a number of bins that is not positive, with what to do (maths.md §1.9): a
    ``TypeError`` when ``n_bins`` or ``Jmin_PS`` is not an integer, a ``ValueError`` otherwise."""
    data = Data2D(np.zeros((1, 1, *shape)), pbc=True)
    with pytest.raises(error, match=match):
        build_plan(data, L=4, compute_PS=True, **options)
    with pytest.raises(ValueError, match="act only on the power spectrum"):
        build_plan(Data2D(np.zeros((1, 1, 32, 32)), pbc=True), None, 4, n_bins=5)


def test_spectrum_tables_grow_with_the_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """As in T3.7 (Part III, §19.4, §19.6): the static arrays of the spectrum (its family in the
    layout, the floats of σᴾˢ, the rows of its normalisation) hold fewer elements than its
    coefficients, and the maps that read it (``to_vector``, ``select``, ``reduce``) make no
    constant that large: one row per channel pair, never a table per coefficient or the size of
    its full grid. Three channels and 96 bins, so that the spectrum outweighs the tables of the
    other families: 576 coefficients, a full grid of 864. The constants are recorded from the
    normalisation (``norm="self"``) on."""
    rng = np.random.default_rng(7)
    data = Data2D(rng.standard_normal((1, 3, 32, 32)), pbc=True)
    plan = build_plan(data, 3, 4, norm="self", compute_PS=True, n_bins=96)
    layout = _layout.base_layout(plan)
    grid = layout.grid("PS")
    n_coefficients = grid.n_chunks * grid.n_orientations
    assert (n_coefficients, int(np.prod(grid.shape))) == (576, 864)
    present = _grid.auto_scales(plan)
    spectrum = _normalization.spectrum_reference_of(plan.spectrum, plan.pairs).present.tobytes()
    tables = _normalization._norm_tables(
        layout, present.shape, present.tobytes(), plan.L, None, spectrum
    )
    floats = _normalization._spectrum_tables(layout, spectrum)
    assert _static_elements((grid, floats, tables.rows["PS"])) < n_coefficients

    buffer = _registry.current_backend().asarray(
        rng.random((1, layout.n_floats)) + 0.5, "float64", None
    )
    sizes: list[int] = []

    def recording(backend: Any, array: Any, like: Any, dtype: Any = None) -> Any:
        sizes.append(int(np.asarray(array).size))
        return constant(backend, array, like, dtype)

    for module in (_layout, _normalization, _statistics):
        monkeypatch.setattr(module, "constant", recording)
    stats = statistics_of(buffer, plan)
    stats.to_vector()
    stats.select("PS").to_vector()
    stats.select("PS", bin=[5, 2], c1=[2, 0]).to_vector()
    stats.select(channel=(0, 1)).to_vector()
    stats.reduce(iso=True).select("PS")._to_complex_vector()
    assert sizes
    assert max(sizes) < n_coefficients
