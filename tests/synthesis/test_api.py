"""The synthesis API (Part II, §12.9; Part I, §5.1, "Synthesis"; task T6.1, issue #28; maths.md
section 5): a synthesis runs, writes checkpoints and resumes from them, and refuses what 1.0.0
does not synthesise, with errors that say what to do.

Step 1 tests the API only (Part I, §5.1); the one comparison with STL is
``tests/parity/test_synthesis.py``. The cases are the smallest that run the whole loss: 16 x 16
pixels, J = 2, a few iterations, in float64. On the JAX backend, which has no L-BFGS before
step 2 (LF1), every test checks the error a synthesis raises there instead.
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

pytestmark = pytest.mark.maths(
    "M-SYN-TARGET", "M-SYN-INIT", "M-SYN-NYQ", "M-SYN-LOSS", "M-SYN-LBFGS", "M-SYN-OUT"
)
N = 16
SMALL: dict[str, Any] = {"compute_PS": False, "st_kwargs": {"J": 2}, "seed": 3}


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
    argument of Backend.lbfgs, decision 4 on #28); a checkpoint at max_iter is the result."""
    if _unavailable(backend):
        return
    target = _maps()[0, 0]
    whole = estel.synthesize_from_maps(target, max_iter=5, **SMALL)
    run = estel.synthesize_from_maps(
        target, max_iter=5, checkpoint_every=2, checkpoint_dir=tmp_path, **SMALL
    )
    names = [Path(p).name for p in run.checkpoints]
    assert names[0] == "checkpoint_000002.npz"
    assert names[-1] == f"checkpoint_{run.n_iter:06d}.npz"
    np.testing.assert_array_equal(run.losses, whole.losses)
    resumed = estel.synthesize_from_maps(
        target, max_iter=5, resume_from=run.checkpoints[0], **SMALL
    )
    np.testing.assert_array_equal(resumed.losses, whole.losses)
    np.testing.assert_array_equal(backend.to_numpy(resumed.field), backend.to_numpy(whole.field))
    assert resumed.n_evals == whole.n_evals
    assert resumed.manifest["resumed_from"] == run.checkpoints[0]
    final = estel.synthesize_from_maps(target, max_iter=5, resume_from=run.checkpoints[-1], **SMALL)
    np.testing.assert_array_equal(final.losses, whole.losses)
    np.testing.assert_array_equal(backend.to_numpy(final.field), backend.to_numpy(whole.field))


def test_from_stats_leaves_them_unchanged(backend: Backend) -> None:
    """From statistics computed with standardize=True: de-standardised with the batch means of
    the removed means and standard deviations (M-SYN-OUT, OP-72); the channel axis dropped for
    Nc = 1 (OP-75); the statistics of the caller unchanged (OP-70)."""
    if _unavailable(backend):
        return
    data = Data2D(_maps(n_batch=2), pbc=True)
    operator = STOperator(
        data, 2, norm="self", ref_batch_mean=True, standardize=True, compute_PS=False
    )
    stats = operator.apply(data)
    before = {f: np.asarray(backend.to_numpy(stats.full_grid(f))) for f in ("S3", "S4")}
    result = estel.synthesize_from_stats(stats, n_batch=3, max_iter=2)
    assert tuple(result.field.shape) == (3, N, N)
    assert result.manifest["entry"] == "stats"
    assert result.manifest["J"] == 2
    for family, values in before.items():
        np.testing.assert_array_equal(backend.to_numpy(stats.full_grid(family)), values)
    field = np.asarray(backend.to_numpy(result.field))
    assert abs(field.mean() - 1.0) < 1.0  # the units of the data, not the standardised ones


def test_refuses_what_it_does_not_synthesise(backend: Backend, tmp_path: Path) -> None:
    """Non-periodic and masked targets (LF47), the power spectrum before T4.1, and settings
    that do not fit, each with an error that says what to do (golden rule 8)."""
    if _unavailable(backend):
        return
    maps = _maps()
    masked = maps[0, 0].copy()
    masked[3, 4] = np.nan
    cases: list[tuple[type[Exception], str, Any, dict[str, Any]]] = [
        (ValueError, "LF47", Data2D(maps, pbc=False), SMALL),
        (ValueError, "LF47", Data2D(masked, pbc=True), SMALL),
        (NotImplementedError, "T4.1", maps, {**SMALL, "compute_PS": None}),
        (ValueError, "n_batch", maps, {**SMALL, "mean_field": False, "n_batch": 2}),
        (ValueError, "st_kwargs", maps, {**SMALL, "st_kwargs": {"norm": "ref"}}),
        (ValueError, "go together", maps, {**SMALL, "checkpoint_every": 2}),
        (ValueError, "broadcast", maps, {**SMALL, "init": np.zeros((3, N, N))}),
        (ValueError, "running_shape", maps, {**SMALL, "running_shape": (N,)}),
        (FileNotFoundError, "checkpoint", maps, {**SMALL, "resume_from": tmp_path / "x.npz"}),
    ]
    for error, match, target, options in cases:
        with pytest.raises(error, match=match):
            estel.synthesize_from_maps(target, max_iter=1, **options)
    # A checkpoint of another run.
    run = estel.synthesize_from_maps(
        maps, max_iter=1, checkpoint_every=1, checkpoint_dir=tmp_path, **SMALL
    )
    with pytest.raises(ValueError, match="another run"):
        estel.synthesize_from_maps(
            _maps(seed=1), max_iter=2, resume_from=run.checkpoints[-1], **SMALL
        )
    # Statistics a synthesis cannot start from.
    data = Data2D(maps, pbc=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stats = STOperator(data, 2, norm="self", compute_PS=False).apply(data)
        raw = STOperator(data, 2, norm="none", compute_PS=False).apply(data)
    for bad, match in ((raw, 'norm="self"'), (stats.reduce(iso=True), "reduced")):
        with pytest.raises(ValueError, match=match):
            estel.synthesize_from_stats(bad, n_batch=1, max_iter=1)
    with pytest.raises(TypeError, match="synthesize_from_maps"):
        estel.synthesize_from_stats(maps, n_batch=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="returned by"):
        estel.SynthesisResult()
