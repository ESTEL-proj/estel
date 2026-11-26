"""Parity of the 2D power spectrum with STL (tasks T4.1, T4.6, issues #24, #191; maths.md §1.9,
M-NORM, M-FLAT, OP-120; Part II, §12.8, §14.3).

``spectrum_2d`` (T4.3): STL's spectrum operator on each configuration, the automatic choice with
its default pairs (the auto spectra, passed here explicitly: OP-94) and each estimator forced
with every pair c₁ ≤ c₂, and its bins. ESTEL computes them with ``STOperator``, J = 1 and L = 1
(the smallest scattering, which does not reach the spectrum), its J_PS
the configuration's (decision 3 on #191), and ``norm="none"``, STL's operator normalising
nothing; the family ``"PS"`` is compared. The configuration ``nan_per_map``, whose
missing pixels differ per map, is left out: ESTEL's mask is one map shared by every batch element
and channel (OP-98), and the comment on #24 gives the numbers; ``tests/spec/test_spectra_operator``
compares it with the oracle on the union of its missing pixels.

``statistics_ps_2d`` (T4.3): STL's scattering operator with the spectrum, ``store_ref`` on a
target and ``load_ref`` on a second field. ESTEL computes them with ``STOperator`` (the spectrum
is on by default for 2D data), with ``J_PS`` the J of the configuration, where STL passes its
scattering J to its spectrum (OP-120); the spectrum with J = 1 and L = 1, ``norm="self"`` and
``"ref"``,
equals the fixture's normalised spectra and its reference.

Every array is compared with ``assert_parity`` (same shape, NaN at the same positions, the
tolerances of §14.3) in the run's dtype: float64, and float32 for the configurations marked
``dtype``.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from estel import Data2D, STOperator, _config
from estel.backend import _registry
from estel.backend._protocol import DType
from tests.parity._compare import assert_parity, load_fixture, read_manifest

pytestmark = pytest.mark.maths("M-PS-2D", "M-NORM", "M-FLAT")
SPECTRUM, STATISTICS = "spectrum_2d", "statistics_ps_2d"
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4", "PS")
ESTIMATORS = ("fft", "pbc_mask", "nan_mask", "pbc_nan_mask")
PER_MAP = "nan_per_map"
"""Missing pixels that differ per map: out of ESTEL's data model (OP-98; comment on #24)."""
PBC = {
    "periodic": True,
    "open": False,
    "periodic_nan": True,
    "open_nan": False,
    "rect_wide": True,
    "rect_tall": False,
    "periodic_J2": True,
    "open_Jmin1": False,
}
"""``pbc`` of each configuration of ``spectrum_2d`` (its docstring, and PR #154)."""


@pytest.fixture(autouse=True)
def _run_dtype(monkeypatch: pytest.MonkeyPatch, dtype: DType) -> None:
    """Data and operators in the run's dtype."""
    monkeypatch.setattr(_config, "_dtype", dtype)


@functools.cache
def _arrays(case: str) -> dict[str, NDArray[Any]]:
    return load_fixture(case)


def _numpy(x: Any) -> NDArray[Any]:
    return np.asarray(_registry.current_backend().to_numpy(x))


def _data(field: NDArray[Any], pbc: bool, mask: NDArray[np.bool_] | None = None) -> Data2D:
    if mask is None and np.isnan(field).any():
        mask = np.isnan(field).any(axis=(0, 1))
    return Data2D(field, mask=mask, pbc=pbc)


# ---------------------------------------------------------------------- spectrum_2d


def _spectrum_configs() -> list[Any]:
    names = sorted({k.split("_bin_scalars")[0] for k in _arrays(SPECTRUM) if "_bin_scalars" in k})
    assert set(names) == {*PBC, PER_MAP}
    marks = {"periodic": (pytest.mark.dtype,)}
    return [pytest.param(name, marks=marks.get(name, ())) for name in sorted(PBC)]


@pytest.mark.parametrize("name", _spectrum_configs())
def test_spectrum_equals_stl(name: str, dtype: DType) -> None:
    """Through ``STOperator`` with J = 1 and STL's J_PS: every estimator that accepts the field,
    forced, with every pair c₁ ≤ c₂, and the automatic choice with STL's default pairs (the auto
    spectra, passed explicitly; OP-94); and the bins:
    the estimator chosen, the centres in cycles per pixel (``bin_centers``), the windows, Z_b,
    N_k, the radial index and weight of each mode, and the borders."""
    arrays = _arrays(SPECTRUM)
    field = arrays[f"{name}_field"]
    j_ps, j_min, n_bins = (int(v) for v in arrays[f"{name}_bin_scalars"][:3])
    data = _data(field, PBC[name])
    options: dict[str, Any] = {"J_PS": j_ps, "Jmin_PS": j_min, "n_bins": n_bins, "norm": "none"}
    n_channels = int(field.shape[1])
    runs = [("auto", np.eye(n_channels, dtype=bool), f"{name}_auto")] + [
        (m, None, f"{name}_method_{m}") for m in ESTIMATORS if f"{name}_method_{m}" in arrays
    ]
    for method, pairs, key in runs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13: tested elsewhere
            operator = STOperator(data, 1, 1, spectrum_method=method, cross_matrix=pairs, **options)
        stats = operator.apply(data)
        assert_parity(_numpy(stats.full_grid("PS")), arrays[key], dtype)
        if method == "auto":
            chosen = operator._plan.spectrum.method  # type: ignore[union-attr]
            assert ESTIMATORS.index(chosen) == int(arrays[f"{name}_auto_method"][0])
            assert_parity(stats.bin_centers, arrays[f"{name}_bin_centers"])
    bins = operator._plan.spectrum.bins()  # type: ignore[union-attr]
    assert_parity(bins.windows, arrays[f"{name}_bin_windows"])
    assert_parity(bins.norm, arrays[f"{name}_bin_norm"])
    np.testing.assert_array_equal(bins.n_modes, arrays[f"{name}_n_modes_radial"])
    np.testing.assert_array_equal(bins.radial, arrays[f"{name}_radial_bin_of_k"])
    np.testing.assert_array_equal(bins.mode_weights, arrays[f"{name}_weight_of_k"])
    np.testing.assert_array_equal(bins.borders, arrays[f"{name}_crop_borders"])
    scalars = arrays[f"{name}_bin_scalars"]
    assert (bins.J, bins.Jmin, bins.n_bins, bins.n_max) == tuple(
        int(scalars[i]) for i in (0, 1, 2, 5)
    )
    assert_parity(np.array([bins.k_min, bins.k_max]) / (2 * bins.n_max), scalars[3:5])


# ---------------------------------------------------------------------- statistics_ps_2d


def _statistics_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = read_manifest()[f"{STATISTICS}.npz"]["parameters"][
        "configs"
    ]
    return configs


class _Run:
    """ESTEL on one configuration of ``statistics_ps_2d``: ``STOperator`` with the spectrum,
    ``norm="self"`` on the target and ``norm="ref"`` on the second field, with the J of the
    configuration and with J = 1, both with ``J_PS`` the J of the configuration, as STL passes
    its J to its spectrum (OP-120): ``J_PS=2`` for ``periodic_J2`` (J = 2 on 32 x 32, the
    fixture's [J, J_PS, J_min, n_b]); the other configurations give no J, and ``J_PS=None`` is
    STL's default, ⌊log₂ 16⌋ - 1 = 3."""

    def __init__(self, name: str) -> None:
        config = _statistics_configs()[name]
        arrays = _arrays(STATISTICS)
        mask = arrays[f"{name}_mask"] > 0 if f"{name}_mask" in arrays else None
        matrix = config.get("cross_matrix")
        pairs = None if matrix is None else np.array(matrix, dtype=bool)
        spectrum = {
            "n_bins": config.get("n_bins"),
            "J_PS": config["J"],
            "Jmin_PS": config.get("Jmin", 0),
        }
        method = config.get("spectrum_method", "auto")
        target = _data(arrays[f"{name}_field"], config["pbc"], mask)
        other = _data(arrays[f"{name}_ref_field"], config["pbc"], mask)
        batch_mean = config["norm_batch_mean"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13, OP-30: tested elsewhere
            self.store_op = STOperator(
                target,
                config["J"],
                4,
                ref_batch_mean=batch_mean,
                cross_matrix=pairs,
                spectrum_method=method,
                **spectrum,
            )
            self.store = self.store_op.apply(target)
            load_op = STOperator(
                other,
                config["J"],
                4,
                norm="ref",
                cross_matrix=pairs,
                spectrum_method=method,
                **spectrum,
            )
            self.load = load_op.apply(other, ref=self.store)
            options = {"cross_matrix": pairs, "spectrum_method": method, **spectrum}
            self.spectrum_store = STOperator(
                target, 1, 1, ref_batch_mean=batch_mean, **options
            ).apply(target)
            spectrum_load = STOperator(other, 1, 1, norm="ref", **options)
            self.spectrum_load = spectrum_load.apply(other, ref=self.spectrum_store)


@functools.cache
def _run(name: str, dtype: str) -> _Run:
    return _Run(name)


def _marks(name: str) -> Any:
    return pytest.param(name, marks=(pytest.mark.dtype,) if name == "periodic" else ())


@pytest.mark.parametrize("name", [_marks(n) for n in _statistics_configs()])
def test_statistics_with_the_spectrum_equal_stl(name: str, dtype: DType) -> None:
    """``STOperator`` with the spectrum: the estimator, the positions, ``store`` and ``load`` of
    every family, the three references and the four vectors (maths.md §1.9, M-NORM, M-FLAT)."""
    arrays = _arrays(STATISTICS)
    run = _run(name, dtype)
    spectrum = getattr(run.store_op.plan, "spectrum", None)
    assert spectrum is not None
    assert ESTIMATORS.index(spectrum.method) == int(arrays[f"{name}_ps_method"][0])
    for which, stats in (("store", run.store), ("load", run.load)):
        scale = float(np.sqrt(np.max(arrays[f"{name}_{which}_var"] * arrays[f"{name}_var_ref"])))
        assert_parity(_numpy(stats.full_grid("mean")), arrays[f"{name}_{which}_mean"], dtype, scale)
        for family in FAMILIES[1:]:
            assert_parity(
                _numpy(stats.full_grid(family)), arrays[f"{name}_{which}_{family}"], dtype
            )
    reference = run.store._reference
    assert reference is not None
    assert_parity(_numpy(reference.var), arrays[f"{name}_var_ref"], dtype)
    assert_parity(_numpy(reference.s2_sqrt), arrays[f"{name}_s2_ref"], dtype)
    assert_parity(_numpy(reference.ps_sqrt), arrays[f"{name}_ps_ref"], dtype)
    assert_parity(_numpy(run.store._to_complex_vector()), arrays[f"{name}_vector"], dtype)
    assert_parity(_numpy(run.store.to_vector()), arrays[f"{name}_vector_real"], dtype)
    assert_parity(
        _numpy(run.store._to_complex_vector(mean_along_batch=True)),
        arrays[f"{name}_vector_batch_mean"],
        dtype,
    )
    assert_parity(_numpy(run.load._to_complex_vector()), arrays[f"{name}_load_vector"], dtype)


@pytest.mark.parametrize("name", list(_statistics_configs()))
def test_spectrum_with_j_1_normalises_as_stl(name: str, dtype: DType) -> None:
    """The spectrum through ``STOperator`` with J = 1, which does not reach it (OP-120): with
    ``norm="self"`` it equals the fixture's ``store_PS`` and its reference ``ps_ref``; with
    ``norm="ref"``, by that reference, ``load_PS`` (maths.md M-NORM)."""
    arrays = _arrays(STATISTICS)
    run = _run(name, dtype)
    assert_parity(_numpy(run.spectrum_store.full_grid("PS")), arrays[f"{name}_store_PS"], dtype)
    reference = run.spectrum_store._reference
    assert reference is not None
    assert_parity(_numpy(reference.ps_sqrt), arrays[f"{name}_ps_ref"], dtype)
    assert_parity(_numpy(run.spectrum_load.full_grid("PS")), arrays[f"{name}_load_PS"], dtype)
