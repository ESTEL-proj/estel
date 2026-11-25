"""``STOperator`` on HEALPix data (task T5.6, issue #167; Part II, §12.6, §12.7; maths.md §4
"Scattering loop", M-DATA-HPX, M-DEF-HPX, OP-3).

Values are compared with STL in ``tests/parity/test_healpix_statistics.py`` and with the oracle
in ``tests/spec/test_healpix_scattering.py``. These tests run on the run's backend and dtype, at
``nside`` 2 to 8, and check what parity does not: the defaults of M-DEF-HPX, the power spectrum
left out and refused with LF24 (decision 2 on #167), the data ``apply`` refuses (another partial
sky of the same size: decision 4), the advice of the OP-3 messages, and the statistics through a
file and back to their operator (decision 5).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import DataHealpix, STOperator, STStatistics, _config
from estel.io import _schema

pytestmark = pytest.mark.maths("M-DATA-HPX", "M-DEF-HPX", "M-MEAN")
DISC = np.arange(40, 150)
"""A partial sky of ``nside`` 4 (depth 2) across the north cap and the equatorial faces."""


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


def _field(shape: tuple[int, ...], seed: int = 0) -> NDArray[np.float64]:
    field: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return field


def _hole(npix: int) -> NDArray[np.bool_]:
    mask = np.zeros(npix, dtype=bool)
    mask[[5, 6, 70]] = True
    return mask


def _operator(data: DataHealpix, J: int | None, **kwargs: Any) -> STOperator:  # noqa: N803
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3: tested below
        return STOperator(data, J, **{"norm": "none", **kwargs})


@pytest.mark.parametrize(("depth", "n_scales"), [(1, 1), (2, 1), (3, 2)])
def test_defaults_follow_m_def_healpix(depth: int, n_scales: int) -> None:
    """J = max(1, depth - 1) and L = 4 gauges (M-DEF-HPX), on the full sky."""
    operator = _operator(DataHealpix(_field((12 * 4**depth,))), None)
    assert n_scales == operator.J
    assert operator.L == 4


def test_defaults_on_a_partial_sky() -> None:
    """The defaults of M-DEF-HPX follow the depth, not the number of pixels, on a partial sky."""
    operator = _operator(DataHealpix(_field((DISC.size,)), depth=2, cell_ids=DISC), None)
    assert (operator.J, operator.L) == (1, 4)
    assert "pbc=False" in repr(operator)


@pytest.mark.parametrize(
    "options",
    [pytest.param({}, id="compute_PS=None"), pytest.param({"compute_PS": False}, id="False")],
)
def test_the_spectrum_is_left_out(options: dict[str, Any]) -> None:
    """On HEALPix data ``compute_PS=None`` (the default) and ``False`` leave the spectrum out
    (decision 2 on #167)."""
    data = DataHealpix(_field((48,)))
    stats = STOperator(data, 1, norm="none", **options).apply(data)
    assert "PS" not in stats._layout.names


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"compute_PS": True}, id="compute_PS=True"),
        pytest.param({"n_bins": 5}, id="n_bins"),
        pytest.param({"Jmin": 1}, id="Jmin"),
        pytest.param({"spectrum_method": "fft"}, id="spectrum_method"),
        pytest.param({"compute_PS": False, "n_bins": 5}, id="n_bins-without-spectrum"),
    ],
)
def test_the_spectrum_is_refused(options: dict[str, Any]) -> None:
    """Asking for the spectrum of HEALPix data, or giving one of its options, raises the one
    message of data that are not 2D, which names the class received and LF24, the angular
    spectrum of HEALPix data (decision 2 on #167, amended)."""
    data = DataHealpix(_field((48,)))
    with pytest.raises(ValueError, match=r"a DataHealpix, which has no power spectrum .* \(LF24\)"):
        STOperator(data, 1, norm="none", **options)


def _disc_operator() -> STOperator:
    return _operator(DataHealpix(_field((DISC.size,)), depth=2, cell_ids=DISC), 2)


def test_apply_takes_the_same_sky_with_another_batch() -> None:
    data = DataHealpix(_field((2, 1, DISC.size)), depth=2, cell_ids=DISC)
    assert _disc_operator().apply(data).full_grid("S1").shape[0] == 2


@pytest.mark.parametrize(
    ("other", "match"),
    [
        pytest.param(
            lambda: DataHealpix(_field((DISC.size,)), depth=2, cell_ids=DISC + 1),
            "lie on other pixels than those the plan was built for: their depth, cell_ids differ",
            id="other-partial-sky-same-size",
        ),
        pytest.param(
            lambda: DataHealpix(_field((DISC.size,)), depth=3, cell_ids=DISC),
            "lie on other pixels",
            id="other-depth-same-pixels",
        ),
        pytest.param(lambda: DataHealpix(_field((192,))), "build a new STOperator", id="full-sky"),
        pytest.param(
            lambda: DataHealpix(_field((DISC.size,)), _hole(DISC.size), depth=2, cell_ids=DISC),
            "build a new STOperator",
            id="other-mask",
        ),
    ],
)
def test_apply_refuses_other_skies(other: Any, match: str) -> None:
    """Another partial sky of the same size or another depth raise through the check of the
    plan's ``meta`` (``depth``, ``cell_ids``; decision 4 on #167); the full sky and another mask
    through the earlier checks."""
    operator = _disc_operator()
    with pytest.raises(ValueError, match=match):
        operator.apply(other())


def test_empty_means_give_one_advice_for_every_geometry() -> None:
    """The OP-3 warning has one advice for every geometry (the amendment to decision 2 on
    #167): a smaller J or fewer missing values. At ``nside`` 4 the mask covers the coarse level
    of S3 (j2=0, j3=1)."""
    data = DataHealpix(_field((DISC.size,)), _hole(DISC.size), depth=2, cell_ids=DISC)
    with pytest.warns(UserWarning, match="OP-3") as record:
        STOperator(data, 2, norm="none")
    assert "Use a smaller J or data with fewer missing values" in str(record[0].message)


def test_statistics_go_through_a_file_and_back_to_their_operator(tmp_path: Path) -> None:
    """``save`` stores ``depth`` and ``cell_ids`` with the mask (schema version 8), so that
    ``load`` then ``from_statistics`` rebuild the operator of a masked partial sky; the methods
    of ``STStatistics`` work on the loaded statistics, with the axes of section 1 (decision 5 on
    #167)."""
    data = DataHealpix(_field((2, 2, DISC.size)), _hole(DISC.size), depth=2, cell_ids=DISC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3: tested above
        operator = STOperator(data, 2, norm="self")
    stats = operator.apply(data)
    path = tmp_path / "healpix.npz"
    stats.save(path)
    arrays, content = _schema.read(path, _schema.STATISTICS_FORMAT)
    assert content["plan"]["meta"] == {"depth": 2}
    assert np.array_equal(arrays["plan__cell_ids"], DISC)
    assert np.array_equal(arrays["plan__mask"], _hole(DISC.size))
    loaded = STStatistics.load(path)
    rebuilt = STOperator.from_statistics(loaded)
    assert rebuilt.plan == operator.plan
    with pytest.warns(UserWarning, match="replaces the reference"):
        again = rebuilt.apply(data)
    np.testing.assert_array_equal(np.asarray(again.to_vector()), np.asarray(stats.to_vector()))
    np.testing.assert_array_equal(
        np.asarray(loaded.select("S1", c1=1, c2=1).full_grid("S1")),
        np.asarray(stats.full_grid("S1"))[:, 1, 1],
    )
    reduced = loaded.reduce(iso=True, angular_fourier=True, scale_cosine=True)
    np.testing.assert_array_equal(
        np.asarray(reduced.to_vector()),
        np.asarray(stats.reduce(iso=True, angular_fourier=True, scale_cosine=True).to_vector()),
    )
    other = DataHealpix(_field((DISC.size,)), _hole(DISC.size), depth=2, cell_ids=DISC + 1)
    with pytest.raises(ValueError, match="build a new STOperator"):
        rebuilt.apply(other)
