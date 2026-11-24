"""``SpectrumOperator``, ``estel.functional.spectrum`` and ``STStatistics.bin_centers`` (task T4.1,
issue #24; Part II, §12.7, §12.8; maths.md §1.9, M-NORM).

Values are compared with STL in ``tests/parity/test_spectrum.py`` and with the oracle in
``tests/spec/test_spectra_operator.py``. Here: the references between the two operators, the
errors, the files, the pure function, the accessors on the spectrum alone, and
``STOperator.from_statistics`` with the spectrum.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import estel
from estel import (
    Data1D,
    Data2D,
    DataTensorProd,
    SpectrumOperator,
    STOperator,
    STStatistics,
    _config,
)
from estel.backend import _registry
from estel.io import _schema
from estel.scattering._plan import PlanarPlan
from estel.spectra._plan import SpectrumOperatorPlan

pytestmark = pytest.mark.maths("M-PS-2D", "M-NORM")


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.3
    return values


def _data(seed: int = 0, n_batch: int = 2, pbc: bool = True) -> Data2D:
    return Data2D(_field((n_batch, 2, 32, 32), seed), pbc=pbc)


def test_references_between_the_two_operators() -> None:
    """A ``SpectrumOperator`` takes the reference of an ``STOperator`` whose spectrum has the same
    ``n_bins``, J and ``Jmin``, and gives the same spectrum as with its own; an ``STOperator``
    refuses the reference of a ``SpectrumOperator``, which holds no scattering statistics, and a
    ``SpectrumOperator`` the reference of an ``STOperator`` with another spectrum."""
    target, other = _data(), _data(seed=1)
    scattering = STOperator(target).apply(target)
    alone = SpectrumOperator(target).apply(target)
    by_ref = SpectrumOperator(other, norm="ref")
    np.testing.assert_allclose(
        _numpy(by_ref.apply(other, ref=scattering).full_grid("PS")),
        _numpy(by_ref.apply(other, ref=alone).full_grid("PS")),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match=r"SpectrumOperator and holds the power spectrum alone"):
        STOperator(other, norm="ref").apply(other, ref=alone)
    with pytest.raises(ValueError, match=r"n_bins = 5 \(the operator has 6\)"):
        SpectrumOperator(other, norm="ref").apply(
            other, ref=STOperator(target, n_bins=5).apply(target)
        )
    without = STOperator(target, compute_PS=False).apply(target)
    with pytest.raises(ValueError, match=r"holds no power spectrum"):
        SpectrumOperator(other, norm="ref").apply(other, ref=without)
    with pytest.raises(ValueError, match=r'norm="ref" needs a reference'):
        SpectrumOperator(other, norm="ref").apply(other)


def test_norm_self_keeps_the_reference() -> None:
    """With ``norm="self"`` the operator keeps the reference of its last statistics, replacing it
    with a warning, as ``STOperator`` (M-NORM); ``ref_batch_mean`` averages it over the batch."""
    data = _data()
    operator = SpectrumOperator(data, ref_batch_mean=True)
    stats = operator.apply(data)
    reference = stats._reference
    assert reference is not None
    assert reference.spectrum_only
    assert tuple(_numpy(reference.ps_sqrt).shape) == (1, 2, 6)
    with pytest.warns(UserWarning, match="replaces the reference"):
        operator.apply(data)
    auto = _numpy(stats.full_grid("PS"))[:, [0, 1], [0, 1]]
    np.testing.assert_allclose(auto.real.mean(axis=0), 1.0, rtol=1e-12)


@pytest.mark.parametrize(
    ("data", "options", "error", "match"),
    [
        pytest.param(
            lambda: Data2D(_field((1, 1, 32, 31)), pbc=True), {}, ValueError, "OP-20", id="odd"
        ),
        pytest.param(lambda: Data2D(_field((1, 1, 34, 34)), pbc=True), {}, None, None, id="even"),
        pytest.param(lambda: _data(), {"J": 3, "Jmin": 2}, ValueError, "OP-93", id="J-Jmin"),
        pytest.param(lambda: _data(), {"J": 1}, ValueError, "OP-93", id="J1"),
        pytest.param(lambda: Data1D(_field((1, 1, 32)), pbc=True), {}, ValueError, "LF23", id="1D"),
        pytest.param(
            lambda: DataTensorProd(_field((1, 1, 8, 8, 8)), [Data2D, Data1D], pbc=[True, True]),
            {},
            ValueError,
            "LF23",
            id="tensor-product",
        ),
        pytest.param(
            lambda: _data(), {"method": "anafast"}, ValueError, "'auto', 'fft'", id="method"
        ),
        pytest.param(
            lambda: Data2D(_field((1, 1, 32, 32)), mask=_corner(), pbc=True),
            {"method": "fft"},
            ValueError,
            "nan_mask",
            id="fft-with-missing-pixels",
        ),
        pytest.param(
            lambda: Data2D(_field((1, 1, 16, 16)), mask=_frame(), pbc=False),
            {},
            ValueError,
            r"bin\(s\) \[0, 1, 2, 3\] have no observed pixel inside their border",
            id="pbc-nan-mask-empty-bin",
        ),
        pytest.param(
            lambda: _data(),
            {"cross_matrix": np.ones((3, 3), bool)},
            ValueError,
            r"\(2, 2\)",
            id="pairs",
        ),
        pytest.param(
            lambda: _data(), {"norm": "store_ref"}, ValueError, "'none', 'self'", id="norm"
        ),
        pytest.param(lambda: _data(), {"n_bins": 0}, ValueError, "n_bins=0", id="bins"),
        pytest.param(lambda: _data(), {"J": 3.0}, TypeError, r"J=3\.0", id="J-type"),
    ],
)
def test_construction_errors_say_what_to_do(
    data: Any, options: dict[str, Any], error: type[Exception] | None, match: str | None
) -> None:
    """Every refusal is at construction, with what to do next (golden rule 8)."""
    example = data()
    if error is None:
        SpectrumOperator(example, **options)
        return
    with pytest.raises(error, match=match) as found:
        SpectrumOperator(example, **options)
    assert "compute_PS" not in str(found.value)  # advice a SpectrumOperator can follow


def _corner() -> NDArray[np.bool_]:
    mask = np.zeros((32, 32), dtype=bool)
    mask[:3, :3] = True
    return mask


def _frame() -> NDArray[np.bool_]:
    """Only the outermost ring of a 16 x 16 map is observed: inside no border (β_b ≥ 3)."""
    mask = np.ones((16, 16), dtype=bool)
    mask[1:-1, 1:-1] = False
    return ~mask


def test_apply_refuses_other_data() -> None:
    operator = SpectrumOperator(_data(), norm="none")
    for other in (
        _data(pbc=False),
        Data2D(_field((1, 2, 16, 16)), pbc=True),
        Data2D(_field((1, 3, 32, 32)), pbc=True),
    ):
        with pytest.raises(ValueError, match=r"SpectrumOperator"):
            operator.apply(other)


def test_functional_spectrum_is_the_pure_function_of_apply() -> None:
    """``estel.functional.spectrum(data, op.plan, op.filters)`` is ``op.apply(data)``; each pure
    function refuses the plan of the other operator."""
    data = _data()
    operator = SpectrumOperator(data)
    np.testing.assert_array_equal(
        _numpy(estel.functional.spectrum(data, operator.plan, operator.filters).full_grid("PS")),
        _numpy(operator.apply(data).full_grid("PS")),
    )
    scattering = STOperator(data, compute_PS=False)
    with pytest.raises(TypeError, match=r"functional\.scattering"):
        estel.functional.spectrum(data, scattering.plan, scattering.filters)
    with pytest.raises(TypeError, match=r"op\.plan and op\.filters"):
        estel.functional.scattering(data, operator.plan, operator.filters)


def test_statistics_of_the_spectrum_alone() -> None:
    """The family ``"PS"`` alone: its full grid, ``to_vector`` (its real part), ``select``, and
    ``bin_centers`` in cycles per pixel, which follow a selection, the axis ``bin`` kept or
    dropped; ``bin_centers`` of statistics without the spectrum raises; an ``STOperator`` cannot
    be rebuilt from them."""
    data = _data()
    stats = SpectrumOperator(data, norm="none").apply(data)
    grid = _numpy(stats.full_grid("PS"))
    assert grid.shape == (2, 2, 2, 6)
    assert np.isnan(grid[:, 1, 0]).all()
    present = grid[:, [0, 0, 1], [0, 1, 1]].real
    np.testing.assert_array_equal(_numpy(stats.to_vector()), present.reshape(-1))
    centers = stats.bin_centers
    # c_b / (2 N_max) at J = 3: k_min / (2 N_max) = 1 / (√2 · 2³) cycles per pixel.
    np.testing.assert_allclose(centers[0], np.exp(np.log(1 / (8 * np.sqrt(2))) + np.log(4) / 12))
    assert centers.dtype == np.float64
    assert not centers.flags.writeable
    np.testing.assert_array_equal(stats.select(bin=[4, 1]).bin_centers, centers[[4, 1]])
    one = stats.select(bin=2)
    assert one._grids[0].axes == ("c1", "c2")
    assert one.bin_centers.shape == ()
    assert one.bin_centers == centers[2]
    assert stats.select(bin=[4, 1]).select(bin=1).bin_centers == centers[1]
    with pytest.raises(ValueError, match=r"bin_centers: these statistics hold no power spectrum"):
        _ = STOperator(data, compute_PS=False).apply(data).bin_centers
    with pytest.raises(ValueError, match=r"computed by a SpectrumOperator"):
        STOperator.from_statistics(stats)
    with_scattering = STOperator(data).apply(data)
    np.testing.assert_array_equal(with_scattering.bin_centers, centers)
    with pytest.raises(ValueError, match=r"select the family PS as well"):
        _ = with_scattering.select("S1").bin_centers


def test_bin_centers_do_not_depend_on_the_size() -> None:
    """At equal J, J_min and number of bins, the centres in cycles per pixel do not depend on the
    size of the map (Part II, §12.7)."""
    small = Data2D(_field((1, 1, 32, 32)), pbc=True)
    large = Data2D(_field((1, 1, 64, 48)), pbc=True)
    np.testing.assert_allclose(
        SpectrumOperator(small, 3).apply(small).bin_centers,
        SpectrumOperator(large, 3).apply(large).bin_centers,
        rtol=1e-14,
    )


@pytest.mark.parametrize("which", ["SpectrumOperator", "STOperator"])
def test_save_and_load_round_trip(
    which: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``estel.io``, schema version 7: the statistics of either operator with the spectrum, the
    estimator of their plan and their reference come back unchanged; a file of version 6 is
    refused, with what to do."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:9, 10:12] = True
    data = Data2D(_field((2, 2, 32, 32)), mask=mask, pbc=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13: tested with the plan
        operator: Any = (
            SpectrumOperator(data, n_bins=5, method="nan_mask")
            if which == "SpectrumOperator"
            else STOperator(data, 3, n_bins=5, spectrum_method="nan_mask")
        )
    stats = operator.apply(data)
    path = tmp_path / "ps.npz"
    stats.save(path)
    loaded = STStatistics.load(path)
    assert loaded._plan == stats._plan
    plan = loaded._plan
    assert isinstance(plan, PlanarPlan | SpectrumOperatorPlan)
    assert plan.spectrum is not None
    assert plan.spectrum.method == "nan_mask"
    assert loaded._layout == stats._layout
    for family in stats._layout.names:
        np.testing.assert_array_equal(
            _numpy(loaded.full_grid(family)), _numpy(stats.full_grid(family))
        )
    assert loaded._reference is not None
    assert stats._reference is not None
    np.testing.assert_array_equal(
        _numpy(loaded._reference.ps_sqrt), _numpy(stats._reference.ps_sqrt)
    )
    np.testing.assert_array_equal(loaded.bin_centers, stats.bin_centers)
    old = tmp_path / "version6.npz"
    monkeypatch.setattr(_schema, "SCHEMA_VERSION", 6)
    stats.save(old)
    monkeypatch.undo()
    with pytest.raises(ValueError, match=r"schema version 6.*reads version 7"):
        STStatistics.load(old)


def test_from_statistics_computes_the_spectrum_again() -> None:
    """``STOperator.from_statistics`` rebuilds an operator with the spectrum, its estimator and
    its options: from ``norm="self"`` statistics it computes them again; from ``norm="ref"``
    statistics it holds the reference they were normalised with, and normalises other data by
    it, with no ``ref``, as the original operator did."""
    target, other = _data(), _data(seed=1)
    options: dict[str, Any] = {"n_bins": 4, "Jmin": 1, "spectrum_method": "pbc_mask"}
    stats = STOperator(target, **options).apply(target)
    rebuilt = STOperator.from_statistics(stats)
    assert rebuilt.plan == stats._plan
    with pytest.warns(UserWarning, match="replaces the reference"):
        again = rebuilt.apply(target)
    np.testing.assert_array_equal(_numpy(again.to_vector()), _numpy(stats.to_vector()))
    normalised = STOperator(other, norm="ref", **options).apply(other, ref=stats)
    by_held = STOperator.from_statistics(normalised).apply(other)
    np.testing.assert_array_equal(_numpy(by_held.to_vector()), _numpy(normalised.to_vector()))


def test_reduce_leaves_the_spectrum_alone_unchanged() -> None:
    """Every reduction leaves the power spectrum unchanged (maths.md §1.9, "Reductions"): the
    statistics of a ``SpectrumOperator``, and those of an ``STOperator`` restricted to PS, come
    back as they are."""
    data = _data()
    for stats in (
        SpectrumOperator(data).apply(data),
        STOperator(data).apply(data).select("PS"),
    ):
        reduced = stats.reduce(iso=True, angular_fourier=True, scale_cosine=True)
        np.testing.assert_array_equal(
            _numpy(reduced.full_grid("PS")), _numpy(stats.full_grid("PS"))
        )
        np.testing.assert_array_equal(_numpy(reduced.to_vector()), _numpy(stats.to_vector()))
