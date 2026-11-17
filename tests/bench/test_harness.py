"""Smoke test of the benchmark harness (task T2.11, issue #89, decision 7; Part I, §6.3).

The benchmark itself runs outside CI (``make bench``, ``bench/README.md``). These tests run its
harness on a tiny case, in this process, so that it cannot rot: the seeded inputs, the two
measured calls, the peak memory, the check against STL and the results file. The real cases, the
measured processes and the container never run here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bench.__main__ import BenchError, check_against_stl, format_table, results_path
from bench._cases import Case, input_mask, load_cases, read_inputs, write_inputs
from bench.worker import measure

from estel import _config

# 32 x 32 pixels, J = 2: the smallest data that runs the whole loop, with a mask and borders.
TINY = Case(
    name="tiny", shape=(32, 32), channels=1, J=2, L=4, pbc=False, seed=1, target_seed=2,
    mask_seed=3, mask_holes=(6, 3),
)  # fmt: skip


def test_cases_file_holds_the_three_cases_of_the_issue() -> None:
    settings, cases = load_cases()
    assert (settings.dtype, settings.device) == ("float32", "cpu")
    assert (settings.warmup, settings.repeats) == (1, 5)
    assert sorted(cases) == ["planar-cross", "planar-masked", "planar-periodic"]
    masked = cases["planar-masked"]
    assert not masked.pbc
    assert len(set(masked.mask_holes)) >= 3  # a few holes of different sizes
    mask = input_mask(masked)
    assert mask is not None
    assert mask.sum() == sum(side**2 for side in masked.mask_holes)  # no overlap
    assert 0.08 < mask.mean() < 0.12  # about 10 % of missing pixels
    assert input_mask(cases["planar-periodic"]) is None


def test_harness_measures_and_checks_a_tiny_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_config, "_dtype", "float32")  # the dtype of the benchmark cases
    path = tmp_path / "tiny.npz"
    digest = write_inputs(TINY, "float32", path)
    inputs = read_inputs(path)
    assert (inputs.case, inputs.digest) == (TINY, digest)
    assert inputs.field.shape == (1, 1, 32, 32)
    assert inputs.field.dtype == np.float32
    # Seeded: written again, the inputs are the same.
    assert write_inputs(TINY, "float32", tmp_path / "again.npz") == digest

    measurements, arrays = measure("estel", inputs, warmup=1, repeats=2)
    assert len(measurements["forward_runs"]) == len(measurements["gradient_runs"]) == 2
    assert 0 < measurements["forward_seconds"] < measurements["gradient_seconds"]
    assert measurements["peak_rss_bytes"] > 2**20
    assert measurements["coefficients"] == arrays["vector"].size > 0
    assert measurements["loss"] > 0
    assert arrays["gradient"].shape == inputs.field.shape
    assert not arrays["gradient"][..., inputs.mask].any()  # missing pixels are not read

    # The check against STL, with these results in the place of STL's.
    check = check_against_stl("tiny", arrays, arrays, "float32")
    assert check["gradient_relative_error"] == 0
    assert check["stl_gradient_not_finite"] == 0
    wrong = {**arrays, "S3": arrays["S3"] * (1 + 1e-3)}
    with pytest.raises(BenchError, match="ESTEL's S3 differs from STL's"):
        check_against_stl("tiny", wrong, arrays, "float32")

    # The table and the results file, which is never overwritten.
    run = {"case": "tiny", "environment": "native", "library": "estel", **measurements}
    assert "| tiny | native      | estel   |" in format_table([run])
    first = results_path(tmp_path, "2026-11-17", "machine")
    first.write_text(json.dumps(run))
    assert first.name == "2026-11-17-machine.json"
    assert results_path(tmp_path, "2026-11-17", "machine").name == "2026-11-17-machine-2.json"
