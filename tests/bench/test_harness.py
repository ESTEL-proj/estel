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
from bench.__main__ import (
    BenchError,
    check_against_main,
    check_against_stl,
    comparison_ratios,
    comparison_verdict,
    export_main,
    format_comparison,
    format_main_differences,
    format_table,
    results_path,
)
from bench._cases import Case, input_mask, load_cases, read_inputs, write_inputs
from bench.worker import measure

from estel import _config

# 32 x 32 pixels, J = 2: the smallest data that runs the whole loop, with a mask and borders.
TINY = Case(
    name="tiny", shape=(32, 32), channels=1, J=2, L=4, pbc=False, seed=1, target_seed=2,
    mask_seed=3, mask_holes=(6, 3),
)  # fmt: skip


def test_cases_file_holds_the_six_cases() -> None:
    settings, cases = load_cases()
    assert (settings.dtype, settings.device) == ("float32", "cpu")
    assert (settings.warmup, settings.repeats) == (1, 5)
    assert sorted(cases) == [
        "healpix-masked-64",
        "planar-cross",
        "planar-masked",
        "planar-open",
        "planar-periodic",
        "tensor-prod-2d1d-64",
    ]
    space_time = cases["tensor-prod-2d1d-64"]  # T3.5: ESTEL only, one J, L and pbc per block
    assert space_time.tensor_prod
    assert space_time.blocks == ("2D", "1D")
    assert (space_time.shape, space_time.J, space_time.L) == ((64, 64, 32), (4, 3), (4, 2))
    assert input_mask(space_time) is None
    masked = cases["planar-masked"]
    assert not masked.pbc
    assert len(set(masked.mask_holes)) >= 3  # a few holes of different sizes
    mask = input_mask(masked)
    assert mask is not None
    assert mask.sum() == sum(side**2 for side in masked.mask_holes)  # no overlap
    assert 0.08 < mask.mean() < 0.12  # about 10 % of missing pixels
    assert input_mask(cases["planar-periodic"]) is None
    # planar-open is planar-masked without its missing pixels (T2.14, issue #92).
    opened = cases["planar-open"]
    assert (opened.shape, opened.J, opened.pbc) == (masked.shape, masked.J, masked.pbc)
    assert (opened.seed, opened.target_seed) == (masked.seed, masked.target_seed)
    assert input_mask(opened) is None
    # T5.7 (issue #168, decision 2): a HEALPix full sky at nside 64, J = 5, ESTEL only, with
    # about 10 % of missing pixels in discs.
    sky = cases["healpix-masked-64"]
    assert sky.healpix and sky.estel_only and not sky.tensor_prod
    assert (sky.shape, sky.depth, sky.J, sky.L, sky.channels) == ((49152,), 6, 5, 4, 1)
    assert not any(case.estel_only for name, case in cases.items() if name.startswith("planar"))


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
    assert measurements["forward_first_seconds"] > 0 < measurements["gradient_first_seconds"]
    assert measurements["first_apply_seconds"] > 0
    assert 0 < measurements["forward_seconds"] < measurements["gradient_seconds"]
    assert measurements["peak_rss_bytes"] >= measurements["baseline_rss_bytes"] > 2**20
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

    # The check against main (Part II, §15.2, T2.12), with these results in the place of
    # main's: the statistics within the parity tolerance, the gradient within 1e-4.
    close = {**arrays, "gradient": arrays["gradient"] * (1 + 5e-5)}
    against_main = check_against_main("tiny", close, arrays, "float32")
    assert against_main["statistics_relative_difference"]["S4"] == 0
    assert 4e-5 < against_main["gradient_relative_difference"] < 6e-5
    assert "| tiny | 0.0e+00 |" in format_main_differences(
        [{"case": "tiny", "check_main": against_main}]
    )
    with pytest.raises(BenchError, match="S3 differs from that of main"):
        check_against_main("tiny", wrong, arrays, "float32")
    far = {**arrays, "gradient": arrays["gradient"] * (1 + 1e-3)}
    with pytest.raises(BenchError, match="gradient of the loss differs from that of main"):
        check_against_main("tiny", far, arrays, "float32")

    # The table and the results file, which is never overwritten.
    run = {"case": "tiny", "environment": "native", "library": "estel", **measurements}
    assert "| tiny | native      | estel   |" in format_table([run])
    assert "| estel (main) |" in format_table([{**run, "tree": "main"}])
    first = results_path(tmp_path, "2026-11-17", "machine")
    first.write_text(json.dumps(run))
    assert first.name == "2026-11-17-machine.json"
    assert results_path(tmp_path, "2026-11-17", "machine").name == "2026-11-17-machine-2.json"


def test_harness_measures_a_tiny_tensor_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tensor-product case (task T3.2): ``[2D, 1D]``, 8 x 8 x 8, J = (1, 1), measured on
    ESTEL only, its seeded inputs read back with their blocks."""
    monkeypatch.setattr(_config, "_dtype", "float32")
    case = Case(
        name="tiny-tp", shape=(8, 8, 8), channels=1, J=(1, 1), L=(4, 2), pbc=(True, True),
        seed=1, target_seed=2, blocks=("2D", "1D"),
    )  # fmt: skip
    path = tmp_path / "tiny-tp.npz"
    digest = write_inputs(case, "float32", path)
    inputs = read_inputs(path)
    assert (inputs.case, inputs.digest) == (case, digest)
    assert inputs.field.shape == (1, 1, 8, 8, 8)
    measurements, arrays = measure("estel", inputs, warmup=1, repeats=1)
    # Both measured; on a field this small their order is noise, so it is not compared.
    assert measurements["forward_seconds"] > 0
    assert measurements["gradient_seconds"] > 0
    assert arrays["gradient"].shape == inputs.field.shape
    assert arrays["S4"].shape[3:] == (1, 1, 1, 1, 1, 1, 4, 2, 4, 2, 4, 2)


def test_harness_measures_a_tiny_healpix_sky(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A HEALPix case (task T5.7): a full sky at nside 4, J = 2, missing pixels in two discs,
    measured on ESTEL only; its seeded inputs read back, the build timed apart."""
    monkeypatch.setattr(_config, "_dtype", "float32")
    case = Case(
        name="tiny-sky", shape=(192,), channels=1, J=2, L=4, pbc=True, seed=1, target_seed=2,
        mask_seed=3, mask_discs=(25.0, 15.0), geometry="healpix",
    )  # fmt: skip
    path = tmp_path / "tiny-sky.npz"
    digest = write_inputs(case, "float32", path)
    inputs = read_inputs(path)
    assert (inputs.case, inputs.digest) == (case, digest)
    assert inputs.field.shape == (1, 1, 192)
    assert inputs.mask is not None and 0 < inputs.mask.sum() < 192 // 4
    measurements, arrays = measure("estel", inputs, warmup=1, repeats=1)
    assert measurements["build_seconds"] > 0
    assert 0 < measurements["build_peak_rss_bytes"] <= measurements["peak_rss_bytes"]
    assert arrays["gradient"].shape == inputs.field.shape
    assert not arrays["gradient"][..., inputs.mask].any()  # missing pixels are not read


def test_earlier_cases_keep_their_digest(tmp_path: Path) -> None:
    """The fields added to the cases with T3.2 and T5.7 stay out of the description of the
    cases that do not use them, so the inputs of the earlier results keep their digest."""
    digest = write_inputs(TINY, "float32", tmp_path / "tiny.npz")
    description = str(np.load(tmp_path / "tiny.npz")["description"])
    assert not {"blocks", "geometry", "mask_discs"} & set(json.loads(description)["case"])
    assert read_inputs(tmp_path / "tiny.npz").digest == digest


def test_main_is_exported_without_a_checkout(tmp_path: Path) -> None:
    """The ESTEL the native run is compared with is the ``src`` of a commit, exported once:
    here that of HEAD, in the place of ``origin/main``."""
    source, commit = export_main("HEAD", tmp_path)
    assert source == tmp_path / f"main-{commit[:12]}" / "src"
    assert (source / "estel" / "__init__.py").is_file()
    assert export_main("HEAD", tmp_path) == (source, commit)  # exported once
    with pytest.raises(BenchError, match="git fetch origin"):
        export_main("no-such-ref", tmp_path)


def test_comparison_of_two_commits_gives_the_ratios_of_each_round() -> None:
    """The comparison mode (issue #91, decision 1): per round the ratios B / A, their median
    and range, and what the criterion reads in them."""

    def run(forward: float, gradient: float, peak: int) -> dict[str, float]:
        return {
            "build_seconds": 1.0,
            "build_peak_rss_bytes": 150,
            "forward_seconds": forward,
            "gradient_seconds": gradient,
            "peak_rss_bytes": peak,
            "baseline_rss_bytes": 100,
        }

    rounds = [
        (run(1.0, 2.0, 300), run(0.8, 2.2, 200)),
        (run(1.0, 2.0, 300), run(0.9, 1.9, 250)),
        (run(2.0, 2.0, 300), run(1.0, 2.0, 300)),
    ]
    ratios = comparison_ratios(rounds)
    assert ratios["forward"]["ratios"] == pytest.approx([0.8, 0.9, 0.5])
    assert ratios["forward"]["median"] == pytest.approx(0.8)
    assert (ratios["gradient"]["min"], ratios["gradient"]["max"]) == pytest.approx((0.95, 1.1))
    assert ratios["memory"]["ratios"] == pytest.approx([0.5, 0.75, 1.0])  # peak - baseline
    assert ratios["peak"]["median"] == pytest.approx(250 / 300)
    verdict = comparison_verdict({"tiny": ratios})
    assert verdict == {
        "below_1_in_every_round": ["tiny: forward"],
        "above_1.05_in_a_round": ["tiny: gradient"],
    }
    assert ratios["build"]["median"] == ratios["build_peak"]["median"] == 1
    assert "| 0.800 (0.500 to 0.900) | 1.000 (0.950 to 1.100) |" in format_comparison(
        {"tiny": ratios}
    )
