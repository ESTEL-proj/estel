"""The power spectrum of ``STOperator``: its own options ``J_PS`` and ``Jmin_PS``, and what
``SpectrumOperator`` was tested on, reached through ``STOperator`` (tasks T4.1, T4.6, issues #24,
#191; Part II, §12.6, §12.7; maths.md §1.9, M-NORM, OP-93, OP-120).

Values are compared with STL in ``tests/parity/test_spectrum.py`` and with the oracle in
``tests/spec/test_spectra_operator.py``. Here: the spectrum does not depend on the scattering J,
the references of ``norm="self"`` and ``norm="ref"``, the errors, the statistics of the spectrum
alone (``select("PS")``) and ``bin_centers``, the files, and ``STOperator.from_statistics`` with
the spectrum. The operators use J = 1 and L = 1 wherever the scattering plays no role (decision
2 on #191; golden rule 13), so that the spectrum dominates their cost: neither reaches it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data2D, STOperator, STStatistics, _config
from estel.backend import _registry
from estel.io import _schema
from estel.scattering._plan import PlanarPlan

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


def _ps(stats: STStatistics) -> NDArray[Any]:
    return _numpy(stats.full_grid("PS"))


def test_the_spectrum_does_not_depend_on_the_scattering_j() -> None:
    """At equal ``J_PS``, the family ``"PS"`` is the same, bit for bit, for J = 1, 2 and 3, with
    ``norm="none"`` and ``norm="self"``; ``J_PS=None`` is ⌊log₂ N_max⌋ - 1 whatever J, given or
    not (maths.md §1.9, "Range"; OP-120)."""
    data = _data()
    for norm in ("none", "self"):
        found = [_ps(STOperator(data, j, 1, norm=norm, J_PS=2).apply(data)) for j in (1, 2, 3)]
        for other in found[1:]:
            np.testing.assert_array_equal(other, found[0])
    for scales in (None, 1, 2, 3):
        spectrum = STOperator(data, scales, 1, norm="none").plan.spectrum  # type: ignore[attr-defined]
        assert (spectrum.J_PS, spectrum.Jmin_PS, spectrum.n_bins) == (3, 0, 6)


def test_norm_ref_takes_a_reference_with_the_same_spectrum() -> None:
    """``norm="ref"`` takes the reference of statistics whose spectrum has the same ``n_bins``,
    ``J_PS`` and ``Jmin_PS``, and names what differs otherwise (maths.md M-NORM); without a
    reference, or by one without the spectrum, it raises with what to do."""
    target, other = _data(), _data(seed=1)
    stored = STOperator(target, 1, 1).apply(target)
    loading = STOperator(other, 1, 1, norm="ref")
    found = _ps(loading.apply(other, ref=stored))
    ps_ref = _numpy(stored._reference.ps_sqrt)  # type: ignore[union-attr]
    raw = _ps(STOperator(other, 1, 1, norm="none").apply(other))
    np.testing.assert_allclose(
        found, raw / (ps_ref[:, :, None, :] * ps_ref[:, None, :, :]), rtol=1e-12
    )
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"n_bins": 5}, r"n_bins = 6 \(the operator has 5\)"),
        ({"J_PS": 2}, r"J_PS = 3 \(the operator has 2\)"),
        ({"Jmin_PS": 1}, r"Jmin_PS = 0 \(the operator has 1\)"),
    )
    for option, match in cases:
        with pytest.raises(ValueError, match=match):
            STOperator(other, 1, 1, norm="ref", **option).apply(other, ref=stored)
    without = STOperator(target, 1, 1, compute_PS=False).apply(target)
    with pytest.raises(ValueError, match=r"holds no power spectrum"):
        loading.apply(other, ref=without)
    with pytest.raises(ValueError, match=r'norm="ref" needs a reference'):
        loading.apply(other)


def test_norm_self_keeps_the_reference_of_the_spectrum() -> None:
    """With ``norm="self"`` and ``ref_batch_mean`` the operator keeps σᴾˢ of the batch mean,
    replacing it with a warning at each call (M-NORM): the auto spectra average to 1."""
    data = _data()
    operator = STOperator(data, 1, 1, ref_batch_mean=True)
    stats = operator.apply(data)
    reference = stats._reference
    assert reference is not None
    assert tuple(_numpy(reference.ps_sqrt).shape) == (1, 2, 6)
    with pytest.warns(UserWarning, match="replaces the reference"):
        operator.apply(data)
    auto = _ps(stats)[:, [0, 1], [0, 1]]
    np.testing.assert_allclose(auto.real.mean(axis=0), 1.0, rtol=1e-12)


def _corner() -> NDArray[np.bool_]:
    mask = np.zeros((32, 32), dtype=bool)
    mask[:3, :3] = True
    return mask


def _frame() -> NDArray[np.bool_]:
    """Only the outermost ring of a 16 x 16 map is observed: inside no border (β_b ≥ 3)."""
    mask = np.ones((16, 16), dtype=bool)
    mask[1:-1, 1:-1] = False
    return ~mask


@pytest.mark.parametrize(
    ("data", "options", "error", "match"),
    [
        pytest.param(
            lambda: Data2D(_field((1, 1, 32, 31)), pbc=True),
            {},
            ValueError,
            r"even sizes.*OP-20.*compute_PS=False",
            id="odd",
        ),
        pytest.param(lambda: Data2D(_field((1, 1, 34, 34)), pbc=True), {}, None, None, id="even"),
        pytest.param(
            lambda: _data(),
            {"J_PS": 3, "Jmin_PS": 2},
            ValueError,
            r"J_PS=3, Jmin_PS=2.*OP-93.*larger J_PS.*compute_PS=False.*larger image",
            id="J_PS-Jmin_PS",
        ),
        pytest.param(lambda: _data(), {"J_PS": 1}, ValueError, "OP-93", id="J_PS-1"),
        pytest.param(
            lambda: _data(),
            {"spectrum_method": "anafast"},
            ValueError,
            "'auto', 'fft'",
            id="method",
        ),
        pytest.param(
            lambda: Data2D(_field((1, 1, 32, 32)), mask=_corner(), pbc=True),
            {"spectrum_method": "fft"},
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
        pytest.param(lambda: _data(), {"J_PS": 3.0}, TypeError, r"J_PS=3\.0", id="J_PS-type"),
    ],
)
def test_construction_errors_say_what_to_do(
    data: Any, options: dict[str, Any], error: type[Exception] | None, match: str | None
) -> None:
    """Every refusal of the spectrum is at construction, with what to do next (golden rule 8),
    J = 1 so that the scattering takes every size (OP-8). The errors of other data (1D, tensor
    products, HEALPix: LF23, LF24) are in ``tests/scattering/``."""
    example = data()
    if error is None:
        STOperator(example, 1, 1)
        return
    with pytest.raises(error, match=match):
        STOperator(example, 1, 1, **options)


def test_statistics_of_the_spectrum_alone() -> None:
    """The family ``"PS"`` alone (``select("PS")``): its full grid, ``to_vector`` (its real
    part), and ``bin_centers`` in cycles per pixel, which follow a selection that keeps the axis
    ``bin`` and raise when an integer dropped it; ``bin_centers`` of statistics without the
    spectrum raises."""
    data = _data()
    stats = STOperator(data, 1, 1, norm="none").apply(data).select("PS")
    grid = _ps(stats)
    assert grid.shape == (2, 2, 2, 6)
    assert np.isnan(grid[:, 1, 0]).all()
    present = grid[:, [0, 0, 1], [0, 1, 1]].real
    np.testing.assert_array_equal(_numpy(stats.to_vector()), present.reshape(-1))
    centers = stats.bin_centers
    # c_b / (2 N_max) at J_PS = 3: k_min / (2 N_max) = 1 / (√2 · 2³) cycles per pixel.
    np.testing.assert_allclose(centers[0], np.exp(np.log(1 / (8 * np.sqrt(2))) + np.log(4) / 12))
    assert centers.dtype == np.float64
    assert not centers.flags.writeable
    np.testing.assert_array_equal(stats.select(bin=[4, 1]).bin_centers, centers[[4, 1]])
    np.testing.assert_array_equal(stats.select(bin=[2]).bin_centers, centers[[2]])
    with pytest.raises(ValueError, match=r"dropped by select\(bin=b\); select it with a list"):
        _ = stats.select(bin=2).bin_centers
    with pytest.raises(ValueError, match=r"bin_centers: these statistics hold no power spectrum"):
        _ = STOperator(data, 1, 1, compute_PS=False).apply(data).bin_centers
    with pytest.raises(ValueError, match=r"select the family PS as well"):
        _ = STOperator(data, 1, 1).apply(data).select("S1").bin_centers
    reduced = stats.reduce(iso=True, angular_fourier=True, scale_cosine=True)
    np.testing.assert_array_equal(_ps(reduced), grid)


def test_bin_centers_do_not_depend_on_the_size() -> None:
    """At equal J_PS, J_min and number of bins, the centres in cycles per pixel do not depend on
    the size of the map (Part II, §12.7)."""
    small = Data2D(_field((1, 1, 32, 32)), pbc=True)
    large = Data2D(_field((1, 1, 64, 48)), pbc=True)
    np.testing.assert_allclose(
        STOperator(small, 1, 1, J_PS=3).apply(small).bin_centers,
        STOperator(large, 1, 1, J_PS=3).apply(large).bin_centers,
        rtol=1e-14,
    )


def test_save_and_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``estel.io``, schema version 9: statistics with the spectrum, its options (``J_PS``,
    ``Jmin_PS`` and the estimator) and its reference come back unchanged; a file of version 8,
    as ``SpectrumOperator`` wrote, is refused, with what to do (decision 6 on #191)."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:9, 10:12] = True
    data = Data2D(_field((2, 2, 32, 32)), mask=mask, pbc=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-13: tested with the plan
        operator = STOperator(data, 1, 1, n_bins=5, J_PS=4, Jmin_PS=1, spectrum_method="nan_mask")
    stats = operator.apply(data)
    path = tmp_path / "ps.npz"
    stats.save(path)
    loaded = STStatistics.load(path)
    assert loaded._plan == stats._plan
    plan = loaded._plan
    assert isinstance(plan, PlanarPlan)
    assert plan.spectrum is not None
    assert plan.spectrum.content() == {"n_bins": 5, "J_PS": 4, "Jmin_PS": 1, "method": "nan_mask"}
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
    old = tmp_path / "version8.npz"
    monkeypatch.setattr(_schema, "SCHEMA_VERSION", 8)
    stats.save(old)
    monkeypatch.undo()
    with pytest.raises(ValueError, match=r"schema version 8.*reads version 9"):
        STStatistics.load(old)


def test_from_statistics_computes_the_spectrum_again() -> None:
    """``STOperator.from_statistics`` rebuilds an operator with the spectrum, its estimator and
    its options: from ``norm="self"`` statistics it computes them again; from ``norm="ref"``
    statistics it holds the reference they were normalised with, and normalises other data by
    it, with no ``ref``, as the original operator did."""
    target, other = _data(), _data(seed=1)
    options: dict[str, Any] = {"n_bins": 4, "J_PS": 4, "Jmin_PS": 1, "spectrum_method": "pbc_mask"}
    stats = STOperator(target, 1, 1, **options).apply(target)
    rebuilt = STOperator.from_statistics(stats)
    assert rebuilt.plan == stats._plan
    with pytest.warns(UserWarning, match="replaces the reference"):
        again = rebuilt.apply(target)
    np.testing.assert_array_equal(_numpy(again.to_vector()), _numpy(stats.to_vector()))
    normalised = STOperator(other, 1, 1, norm="ref", **options).apply(other, ref=stats)
    by_held = STOperator.from_statistics(normalised).apply(other)
    np.testing.assert_array_equal(_numpy(by_held.to_vector()), _numpy(normalised.to_vector()))
