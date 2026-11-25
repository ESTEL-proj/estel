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


def test_defaults_follow_m_def_hpx() -> None:
    """J = max(1, depth - 1) and L = 4 gauges (M-DEF-HPX), on the full sky and a partial sky."""
    for depth, n_scales in ((1, 1), (2, 1), (3, 2)):
        data = DataHealpix(_field((12 * 4**depth,)))
        operator = _operator(data, None)
        assert n_scales == operator.J
        assert operator.L == 4
    operator = _operator(DataHealpix(_field((DISC.size,)), depth=2, cell_ids=DISC), None)
    assert (operator.J, operator.L) == (1, 4)
    assert "pbc=False" in repr(operator)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="compute_PS=None"),
        pytest.param({"compute_PS": False}, id="compute_PS=False"),
        pytest.param({"compute_PS": True}, id="compute_PS=True"),
        pytest.param({"n_bins": 5}, id="n_bins"),
        pytest.param({"Jmin": 1}, id="Jmin"),
        pytest.param({"spectrum_method": "fft"}, id="spectrum_method"),
        pytest.param({"compute_PS": False, "n_bins": 5}, id="n_bins-without-spectrum"),
    ],
)
def test_the_spectrum_is_left_out_or_refused_with_lf24(options: dict[str, Any]) -> None:
    """On HEALPix data ``compute_PS=None`` and ``False`` leave the spectrum out; asking for it,
    or giving one of its options, raises an error naming LF24, the angular spectrum of HEALPix
    data, not LF23 (decision 2 on #167)."""
    data = DataHealpix(_field((48,)))
    if options in ({}, {"compute_PS": False}):
        stats = STOperator(data, 1, norm="none", **options).apply(data)
        assert "PS" not in stats._layout.names
        return
    with pytest.raises(ValueError, match="for HEALPix data it comes later \\(LF24\\)"):
        STOperator(data, 1, norm="none", **options)


@pytest.mark.parametrize(
    ("other", "accepted"),
    [
        pytest.param(lambda: DataHealpix(_field((2, 1, DISC.size)), depth=2, cell_ids=DISC), True,
                     id="same-sky-other-batch"),
        pytest.param(lambda: DataHealpix(_field((DISC.size,)), depth=2, cell_ids=DISC + 1), False,
                     id="other-partial-sky-same-size"),
        pytest.param(lambda: DataHealpix(_field((DISC.size,)), depth=3, cell_ids=DISC), False,
                     id="other-depth-same-pixels"),
        pytest.param(lambda: DataHealpix(_field((192,))), False, id="full-sky"),
        pytest.param(
            lambda: DataHealpix(_field((DISC.size,)), _hole(DISC.size), depth=2, cell_ids=DISC),
            False,
            id="other-mask",
        ),
    ],
)  # fmt: skip
def test_apply_refuses_other_skies(other: Any, accepted: bool) -> None:
    """``apply`` takes the sky the operator was built for, whatever the batch; another partial
    sky of the same size, another depth, the full sky or another mask raise (decision 4 on
    #167): the plan compares ``depth`` and ``cell_ids`` (``Data._meta``)."""
    operator = _operator(DataHealpix(_field((DISC.size,)), depth=2, cell_ids=DISC), 2)
    data = other()
    if accepted:
        assert operator.apply(data).full_grid("S1").shape[0] == 2
        return
    with pytest.raises(ValueError, match="build a new STOperator"):
        operator.apply(data)


def test_empty_means_give_the_advice_of_healpix() -> None:
    """The OP-3 warning on HEALPix advises a smaller J or fewer missing pixels, not ``pbc``,
    which DataHealpix does not take (decision 4 on #167). At ``nside`` 4 the mask covers the
    coarse level of S3 (j2=0, j3=1)."""
    data = DataHealpix(_field((DISC.size,)), _hole(DISC.size), depth=2, cell_ids=DISC)
    with pytest.warns(UserWarning, match="OP-3") as record:
        STOperator(data, 2, norm="none")
    message = str(record[0].message)
    assert "fewer missing pixels" in message
    assert "pbc" not in message


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
