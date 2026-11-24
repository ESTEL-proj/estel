"""The synthesis API (Part II, §12.9; Part I, §5.1, "Synthesis"; tasks T6.1 and T4.5, issues #28
and #152; maths.md section 5): a synthesis runs, with the power spectrum in its loss or not,
writes checkpoints and resumes from them, and refuses what 1.0.0 does not synthesise, with
errors that say what to do.

Step 1 tests the API only (Part I, §5.1); the comparisons with STL are in
``tests/parity/test_synthesis.py``. The cases are the smallest that run the whole loss: 16 x 16
pixels (32 x 32 where the number of bins of the spectrum needs it), J = 2 or 3, a few
iterations, in float64, and one run of each entry in float32. On the JAX backend, which has no
L-BFGS before step 2 (LF1), every test checks the error a synthesis raises there instead.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import estel
from estel import Data2D, STOperator, _config
from estel.backend._protocol import Backend
from estel.statistics._arrays import device_of
from estel.synthesis._api import _running_data, _start
from estel.synthesis._init import nyquist_filter, standardize_target

pytestmark = pytest.mark.maths(
    "M-SYN-TARGET", "M-SYN-INIT", "M-SYN-NYQ", "M-SYN-LOSS", "M-SYN-LBFGS", "M-SYN-OUT", "M-SYN-OPS"
)
N = 16
SMALL: dict[str, Any] = {"compute_PS": False, "st_kwargs": {"J": 2}, "seed": 3}
WITH_PS: dict[str, Any] = {**SMALL, "compute_PS": None}
"""The default of STL's user level: the power spectrum in the loss, 4 bins at 16 x 16 with J = 2
(M-SYN-OPS)."""


@pytest.fixture(autouse=True)
def _float64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")


def _maps(n_batch: int = 1, n_channels: int = 1, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_batch, n_channels, N, N)) * 2.0 + 1.0


def _unavailable(backend: Backend) -> bool:
    """On a backend without L-BFGS, check the error of a synthesis and return True."""
    if "lbfgs" not in getattr(backend, "not_provided", frozenset()):
        return False
    with pytest.raises(NotImplementedError, match="torch backend"):
        estel.synthesize_from_maps(_maps()[0, 0], max_iter=1, **SMALL)
    return True


def test_from_maps_runs_and_reports(backend: Backend) -> None:
    """The result: the fields with STL's axes (M-SYN-OUT, OP-75), de-standardised with the
    target's mean and standard deviation, the losses, the two counts (OP-74), the manifest."""
    if _unavailable(backend):
        return
    target = _maps()
    result = estel.synthesize_from_maps(target[0, 0], max_iter=4, **SMALL)
    assert tuple(result.field.shape) == (N, N)  # a 2D array: no batch nor channel axis
    losses = result.losses
    assert result.n_iter == losses.shape[0] >= 1
    assert result.n_iter <= 4
    assert result.n_evals <= 4 * 5 // 4
    assert np.all(np.isfinite(losses))
    assert np.all(np.diff(losses) <= 0)
    assert result.checkpoints == ()
    assert not result.converged
    field = np.asarray(backend.to_numpy(result.field))
    assert abs(field.mean() - target.mean()) < 0.5 * target.std()  # the target's units
    manifest = result.manifest
    assert manifest["entry"] == "maps"
    assert manifest["seed"] == 3
    assert (manifest["J"], manifest["L"], manifest["history_size"], manifest["lr"]) == (
        2,
        4,
        50,
        1.0,
    )
    assert manifest["weights"] == {"S3": 3.5, "S4": 12.25}
    # A Data object keeps its channel axis; two fields keep their batch axis (OP-75).
    batch = estel.synthesize_from_maps(Data2D(target, pbc=True), n_batch=2, max_iter=1, **SMALL)
    assert tuple(batch.field.shape) == (2, 1, N, N)
    # The same seed gives the same run.
    again = estel.synthesize_from_maps(target[0, 0], max_iter=4, **SMALL)
    np.testing.assert_array_equal(again.losses, losses)


def test_checkpoints_are_written_and_resume_the_same_run(backend: Backend, tmp_path: Path) -> None:
    """Checkpoints every 2 iterations and at the end; a run resumed from one continues the
    same trajectory, bit for bit, with the same evaluation budget (maths.md §5.8; the state
    argument of Backend.lbfgs, decision 4 on #28); a checkpoint at max_iter is the result. With
    the spectrum in the loss, the default."""
    if _unavailable(backend):
        return
    target = _maps()[0, 0]
    whole = estel.synthesize_from_maps(target, max_iter=5, **WITH_PS)
    run = estel.synthesize_from_maps(
        target, max_iter=5, checkpoint_every=2, checkpoint_dir=tmp_path, **WITH_PS
    )
    names = [Path(p).name for p in run.checkpoints]
    assert names[0] == "checkpoint_000002.npz"
    assert names[-1] == f"checkpoint_{run.n_iter:06d}.npz"
    np.testing.assert_array_equal(run.losses, whole.losses)
    resumed = estel.synthesize_from_maps(
        target, max_iter=5, resume_from=run.checkpoints[0], **WITH_PS
    )
    np.testing.assert_array_equal(resumed.losses, whole.losses)
    np.testing.assert_array_equal(backend.to_numpy(resumed.field), backend.to_numpy(whole.field))
    assert resumed.n_evals == whole.n_evals
    assert resumed.manifest["resumed_from"] == run.checkpoints[0]
    final = estel.synthesize_from_maps(
        target, max_iter=5, resume_from=run.checkpoints[-1], **WITH_PS
    )
    np.testing.assert_array_equal(final.losses, whole.losses)
    np.testing.assert_array_equal(backend.to_numpy(final.field), backend.to_numpy(whole.field))


def test_from_stats_leaves_them_unchanged(backend: Backend) -> None:
    """From statistics computed with standardize=True: de-standardised with the batch means of
    the removed means and standard deviations (M-SYN-OUT, OP-72); the channel axis dropped for
    Nc = 1 (OP-75); the statistics of the caller unchanged (OP-70)."""
    if _unavailable(backend):
        return
    stats = _stats()
    before = {f: np.asarray(backend.to_numpy(stats.full_grid(f))) for f in ("S3", "S4")}
    result = estel.synthesize_from_stats(stats, n_batch=3, max_iter=2)
    assert tuple(result.field.shape) == (3, N, N)
    assert result.manifest["entry"] == "stats"
    assert result.manifest["J"] == 2
    for family, values in before.items():
        np.testing.assert_array_equal(backend.to_numpy(stats.full_grid(family)), values)
    field = np.asarray(backend.to_numpy(result.field))
    assert abs(field.mean() - 1.0) < 1.0  # the units of the data, not the standardised ones


def _stats(
    n_batch: int = 2,
    ref_batch_mean: bool = True,
    standardize: bool = True,
    compute_ps: bool = False,
) -> Any:
    data = Data2D(_maps(n_batch=n_batch), pbc=True)
    operator = STOperator(
        data,
        2,
        norm="self",
        ref_batch_mean=ref_batch_mean,
        standardize=standardize,
        compute_PS=compute_ps,
    )
    return operator.apply(data)


def test_float32_from_maps_resumes(
    backend: Backend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From maps in float32, the default dtype: a short run with checkpoints that ends finite,
    in float32, and a resume that continues it bit for bit, with the spectrum in the loss. That a
    run with checkpoints equals the run without is checked in float64 only."""
    monkeypatch.setattr(_config, "_dtype", "float32")
    if _unavailable(backend):
        return
    target = _maps()[0, 0]
    run = estel.synthesize_from_maps(
        target, max_iter=4, checkpoint_every=2, checkpoint_dir=tmp_path, **WITH_PS
    )
    resumed = estel.synthesize_from_maps(
        target, max_iter=4, resume_from=run.checkpoints[0], **WITH_PS
    )
    for result in (run, resumed):
        field = np.asarray(backend.to_numpy(result.field))
        assert field.dtype == np.float32
        assert np.all(np.isfinite(field))
        assert np.all(np.isfinite(result.losses))
    np.testing.assert_array_equal(resumed.losses, run.losses)
    np.testing.assert_array_equal(backend.to_numpy(resumed.field), backend.to_numpy(run.field))


def test_float32_from_stats_runs(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    """From statistics in float32 that hold the spectrum: a short run that ends finite, in
    float32, with the spectrum in the loss."""
    monkeypatch.setattr(_config, "_dtype", "float32")
    if _unavailable(backend):
        return
    from_stats = estel.synthesize_from_stats(_stats(compute_ps=True), n_batch=1, max_iter=2)
    assert from_stats.manifest["spectrum"] is not None
    assert np.all(np.isfinite(from_stats.losses))
    field = np.asarray(backend.to_numpy(from_stats.field))
    assert field.dtype == np.float32
    assert np.all(np.isfinite(field))


def test_converged_run_stops_and_stays_stopped(backend: Backend, tmp_path: Path) -> None:
    """A run stopped by a tolerance of L-BFGS reports ``converged`` (M-SYN-LBFGS); resuming
    from its last checkpoint does not iterate, and resuming from an earlier one reaches the same
    end. The start is the standardised target, moved by 1e-7: the loss is near its minimum, and
    the change of loss falls below 1e-15 in a few iterations."""
    if _unavailable(backend):
        return
    maps = _maps()
    target = standardize_target(
        backend, nyquist_filter(backend, backend.asarray(maps, "float64")), True
    )
    noise = np.random.default_rng(5).standard_normal(maps.shape)
    init = np.asarray(backend.to_numpy(target.x)) + 1e-7 * noise
    options: dict[str, Any] = {**SMALL, "init": init, "max_iter": 30}
    run = estel.synthesize_from_maps(maps, checkpoint_every=4, checkpoint_dir=tmp_path, **options)
    assert run.converged
    assert 1 <= run.n_iter < 30
    assert run.checkpoints[-1].endswith(f"checkpoint_{run.n_iter:06d}.npz")
    final = estel.synthesize_from_maps(maps, resume_from=run.checkpoints[-1], **options)
    assert final.converged
    assert final.checkpoints == ()
    np.testing.assert_array_equal(final.losses, run.losses)
    np.testing.assert_array_equal(backend.to_numpy(final.field), backend.to_numpy(run.field))
    earlier = estel.synthesize_from_maps(maps, resume_from=run.checkpoints[0], **options)
    assert earlier.converged
    np.testing.assert_array_equal(earlier.losses, run.losses)


def test_fields_stay_on_the_device_of_the_data(backend: Backend, tmp_path: Path) -> None:
    """The start, a resumed field and the result of both entries are on the device of the
    target or of the statistics (CPU in CI)."""
    if _unavailable(backend):
        return
    data = Data2D(_maps(), pbc=True)
    device = device_of(data.array)
    running = _running_data(data, (2, 1, N, N), backend)
    assert device_of(_start(backend, "gaussian", 0, running, "float64")) == device
    run = estel.synthesize_from_maps(
        data, max_iter=2, checkpoint_every=1, checkpoint_dir=tmp_path, **SMALL
    )
    resumed = estel.synthesize_from_maps(data, max_iter=3, resume_from=run.checkpoints[0], **SMALL)
    assert device_of(run.field) == device
    assert device_of(resumed.field) == device
    stats = _stats()
    from_stats = estel.synthesize_from_stats(stats, n_batch=1, max_iter=1)
    assert device_of(from_stats.field) == device_of(stats.full_grid("mean"))


def _masked() -> np.ndarray:
    masked: np.ndarray = np.array(_maps()[0, 0])
    masked[3, 4] = np.nan
    return masked


MAPS_ERRORS: dict[str, tuple[type[Exception], str, Any, dict[str, Any]]] = {
    "not-periodic": (ValueError, "LF47", lambda: Data2D(_maps(), pbc=False), {}),
    "masked": (ValueError, "LF47", lambda: Data2D(_masked(), pbc=True), {}),
    "compute-ps": (TypeError, "pass True, False or None", _maps, {"compute_PS": 1}),
    "n-batch": (ValueError, "n_batch", _maps, {"mean_field": False, "n_batch": 2}),
    "st-kwargs": (ValueError, r"takes \['J'\] only", _maps, {"st_kwargs": {"J": 2, "L": 4}}),
    "checkpoint-dir": (ValueError, "go together", _maps, {"checkpoint_every": 2}),
    "init-shape": (ValueError, "broadcast", _maps, {"init": np.zeros((3, N, N))}),
    "running-shape": (ValueError, "running_shape", _maps, {"running_shape": (N,)}),
    "no-checkpoint": (FileNotFoundError, "checkpoint", _maps, {"resume_from": "no_such.npz"}),
}


@pytest.mark.parametrize("case", sorted(MAPS_ERRORS))
def test_maps_errors_say_what_to_do(backend: Backend, case: str) -> None:
    """Non-periodic and masked targets (LF47) and settings that do not fit, each with an error
    that says what to do (golden rule 8)."""
    if _unavailable(backend):
        return
    error, match, target, options = MAPS_ERRORS[case]
    with pytest.raises(error, match=match):
        estel.synthesize_from_maps(target(), max_iter=1, **{**SMALL, **options})


@pytest.mark.parametrize(
    ("function", "advice"),
    [
        ("synthesize_from_maps", "Pass estel.Data2D maps"),
        ("synthesize_from_stats", "Pass the statistics of estel.Data2D maps"),
    ],
)
def test_tensor_products_are_refused_first(function: str, advice: str) -> None:
    """Syntheses on tensor products are not part of ESTEL 1.0.0: refused before any other
    check, on every backend, including a non-periodic product of one 2D block, whose ``ndim``
    is 2 and whose ``pbc``, ``(False,)``, is a non-empty tuple."""
    target = estel.DataTensorProd(_maps(), [Data2D], pbc=[False])
    with pytest.raises(NotImplementedError, match=rf"DataTensorProd; syntheses on .*{advice}"):
        getattr(estel, function)(target, n_batch=1, max_iter=1)


def test_checkpoint_of_another_run_is_refused(backend: Backend, tmp_path: Path) -> None:
    if _unavailable(backend):
        return
    run = estel.synthesize_from_maps(
        _maps(), max_iter=1, checkpoint_every=1, checkpoint_dir=tmp_path, **SMALL
    )
    with pytest.raises(ValueError, match="another run"):
        estel.synthesize_from_maps(
            _maps(seed=1), max_iter=2, resume_from=run.checkpoints[-1], **SMALL
        )


def _raw_stats() -> Any:
    data = Data2D(_maps(), pbc=True)
    return STOperator(data, 2, norm="none", compute_PS=False).apply(data)


def _spectrum_stats() -> Any:
    """Statistics with the 2D power spectrum, the default of STOperator on 2D data (T4.1)."""
    data = Data2D(_maps(), pbc=True)
    return STOperator(data, 2, norm="self").apply(data)


STATS_ERRORS: dict[str, tuple[type[Exception], str, Any, int]] = {
    "not-self": (ValueError, 'norm="self"', _raw_stats, 1),
    "reduced": (ValueError, "reduced", lambda: _stats().reduce(iso=True), 1),
    "reference-batch": (
        ValueError,
        r"n_batch=3.*ref_batch_mean=True",
        lambda: _stats(ref_batch_mean=False),
        3,
    ),
    "maps": (TypeError, "synthesize_from_maps", _maps, 1),
    "selected": (
        ValueError,
        r"reduced or selected.*stats\.select\(\['mean', 'var', 'S1', 'S2', 'S3', 'S4'\]\)",
        lambda: _spectrum_stats().select(["mean", "var", "S1", "S2", "S3", "PS"]),
        1,
    ),
}


@pytest.mark.parametrize("case", sorted(STATS_ERRORS))
def test_stats_errors_say_what_to_do(backend: Backend, case: str) -> None:
    """Statistics a synthesis cannot start from, and an ``n_batch`` that their reference
    cannot normalise (M-NORM), refused up front with the fix."""
    if _unavailable(backend):
        return
    error, match, stats, n_batch = STATS_ERRORS[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bad = stats()
    with pytest.raises(error, match=match):
        estel.synthesize_from_stats(bad, n_batch=n_batch, max_iter=1)


def test_results_are_not_built() -> None:
    with pytest.raises(TypeError, match="returned by"):
        estel.SynthesisResult()


# ---------------------------------------------------------------- the power spectrum (T4.5)

SPECTRUM = {"n_bins": 4, "J": 2, "Jmin": 0, "method": "fft"}
"""The spectrum of a synthesis of 16 x 16 fields with J = 2: J_PS = J, the default number of
bins of 16 x 16, max(4, round(3 (2 - 1))) = 4, J_min 0 and ``fft`` (maths.md §1.9, M-SYN-OPS)."""


def test_compute_ps_puts_the_spectrum_in_the_loss(backend: Backend) -> None:
    """``compute_PS`` None (the default) and True put the spectrum in the loss, as STL's
    ``synthesize_from_maps``; False leaves it out, as ``optimize_from_maps`` (M-SYN-OPS). The
    manifest records it, and the spectrum changes the loss. On JAX, the error of T6.1."""
    if "lbfgs" in getattr(backend, "not_provided", frozenset()):
        with pytest.raises(NotImplementedError, match="torch backend"):
            estel.synthesize_from_maps(_maps()[0, 0], max_iter=1, st_kwargs={"J": 2})
        return
    target = _maps()[0, 0]
    runs = {
        flag: estel.synthesize_from_maps(target, max_iter=2, **{**SMALL, "compute_PS": flag})
        for flag in (None, True, False)
    }
    assert runs[None].manifest["spectrum"] == runs[True].manifest["spectrum"] == SPECTRUM
    assert runs[False].manifest["spectrum"] is None
    np.testing.assert_array_equal(runs[None].losses, runs[True].losses)
    assert np.all(np.isfinite(runs[True].losses))
    assert not np.array_equal(runs[True].losses, runs[False].losses)


@pytest.mark.parametrize(
    ("running_shape", "n_bins"),
    [((32, 32), 6), ((16, 32), 4)],
)
def test_number_of_bins_is_the_smaller_default(
    backend: Backend, running_shape: tuple[int, int], n_bins: int
) -> None:
    """The number of bins is the smaller of the defaults of the target and of the synthesised
    fields, each from its own size, whatever J (M-SYN-OPS): a 32 x 32 target with J = 2 takes
    6 bins, the default of 32 x 32 (J = 3), not the 4 of J = 2; with fields of 16 x 32, 4."""
    if _unavailable(backend):
        return
    target = np.random.default_rng(2).standard_normal((32, 32))
    result = estel.synthesize_from_maps(
        target, running_shape=running_shape, max_iter=1, st_kwargs={"J": 2}
    )
    assert result.manifest["spectrum"] == {**SPECTRUM, "n_bins": n_bins}


def test_from_stats_with_the_spectrum_takes_its_settings(backend: Backend) -> None:
    """From statistics that hold the spectrum, computed with J_min = 1, the running operator
    takes J, the number of bins and J_min of the statistics, and the estimator ``fft`` (OP-76);
    from the same statistics without PS (``select``), the spectrum is not in the loss."""
    if _unavailable(backend):
        return
    data = Data2D(_maps(n_batch=2), pbc=True)
    stats = STOperator(
        data, 3, norm="self", ref_batch_mean=True, standardize=True, n_bins=5, Jmin=1
    ).apply(data)
    result = estel.synthesize_from_stats(stats, n_batch=1, max_iter=2)
    assert result.manifest["spectrum"] == {"n_bins": 5, "J": 3, "Jmin": 1, "method": "fft"}
    assert np.all(np.isfinite(result.losses))
    families = ["mean", "var", "S1", "S2", "S3", "S4"]
    without = estel.synthesize_from_stats(stats.select(families), n_batch=1, max_iter=2)
    assert without.manifest["spectrum"] is None
    assert not np.array_equal(without.losses, result.losses)


def test_checkpoint_of_another_compute_ps_is_refused(backend: Backend, tmp_path: Path) -> None:
    """The settings of a run record the spectrum: a checkpoint written with it does not resume
    a run without it, nor the converse; each resumes a run with its own settings."""
    if _unavailable(backend):
        return
    target = _maps()[0, 0]
    options = {**SMALL, "compute_PS": True}
    with_ps = estel.synthesize_from_maps(
        target, max_iter=1, checkpoint_every=1, checkpoint_dir=tmp_path / "with", **options
    )
    without = estel.synthesize_from_maps(
        target, max_iter=1, checkpoint_every=1, checkpoint_dir=tmp_path / "without", **SMALL
    )
    for written, other in ((with_ps, SMALL), (without, options)):
        with pytest.raises(ValueError, match="another run: spectrum"):
            estel.synthesize_from_maps(
                target, max_iter=2, resume_from=written.checkpoints[-1], **other
            )
    resumed = estel.synthesize_from_maps(
        target, max_iter=2, resume_from=with_ps.checkpoints[-1], **options
    )
    assert resumed.manifest["spectrum"] == SPECTRUM
    resumed = estel.synthesize_from_maps(
        target, max_iter=2, resume_from=without.checkpoints[-1], **SMALL
    )
    assert resumed.manifest["spectrum"] is None
