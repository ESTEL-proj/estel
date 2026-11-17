"""The benchmark of step 1: `make bench` (Part I, §6.3; task T2.11, issue #89).

    python -m bench [--case NAME] [--env native|container] [--backend torch|jax]

Runs the cases of ``bench/cases.toml``, each in a process of its own (``bench/worker.py``), in
two environments: ``container``, the parity container with ESTEL installed, where STL and ESTEL
are measured side by side; and ``native``, ESTEL on this machine, where the ESTEL of ``main`` is
measured too, just before. Every ESTEL run is checked against STL's statistics on the same
inputs, with the parity tolerances of Part II, §14.3, and the native run against the statistics
and the gradient of ``main``; the benchmark fails otherwise. Prints a table and writes
``bench/results/<date>-<machine>.json``. Outside CI: see ``bench/README.md``.

    python -m bench --compare A B [--rounds 5] [--case NAME] [--backend torch|jax]

compares two commits natively (task T2.13, issue #91, decision 1): in each round, A then B, or
B then A in the odd rounds, each in a fresh process; per round, the ratios B / A of the times
and of the memory; the median ratio and its range over the rounds. With A = B it gives the
noise floor of the comparison on this machine.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tests.parity._compare import assert_parity

from bench._cases import CASES_FILE, Case, Settings, load_cases, write_inputs
from bench.worker import _cpu_model, _power

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
ENVIRONMENTS = ("container", "native")
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
CONTAINER_PLATFORM = "linux/amd64"
CONTAINER_PYTHON = "/opt/venv/master/bin/python"
# A measured process that runs longer is stopped: the maintainer resizes a case on which STL
# takes more than about 30 minutes (issue #89, decision 8).
TIMEOUT_SECONDS = 45 * 60
# The gradient of the loss against that of ``main``: max |g - g_main| <= this x max |g_main|
# (Part II, §15.2, T2.12 to T2.15; issue #90, decision 2). Until the planar syntheses exist
# (T6.1), this comparison is what validates the gradients of an optimisation.
MAIN_GRADIENT_RTOL = 1e-4
MAIN_REF = "origin/main"
SCHEMA = 3  # 2: the runs of ``main`` (``tree``) and the check against them (``check_main``);
# 3: the time of the first call of each measure; the comparisons of two commits (``comparison``)
COMPARE_ROUNDS = 5
# What a comparison measures: the name of its ratio, and the value in a run.
COMPARE_MEASURES = ("forward", "gradient", "memory", "peak")
# A change is kept when a time ratio is below 1 in every round on at least one case and above
# this in no round on any case (issue #91, decision 1).
COMPARE_LOSS = 1.05


class BenchError(RuntimeError):
    """The benchmark cannot run or its results are wrong; the message says what to do."""


# ---------------------------------------------------------------------- correctness


def check_against_stl(
    case: str, actual: dict[str, NDArray[Any]], expected: dict[str, NDArray[Any]], dtype: str
) -> dict[str, Any]:
    """Compare the results of an ESTEL run with STL's on the same inputs (Part I, §6.3).

    The statistics must agree with the parity tolerances of Part II, §14.3 for ``dtype``: every
    family in STL's full-grid shape and the flat vector, same shape, NaN at the same positions;
    ``mean`` takes the standard deviation of the data as its scale. Raises ``BenchError``
    otherwise. The gradient of the loss is compared too, and the largest difference is returned,
    relative to the largest gradient of STL: §14.3 sets no tolerance for it in float32, so it is
    recorded, not enforced. Where STL's gradient is not finite (a mean without a valid pixel
    makes it NaN), nothing is compared, and the number of such pixels is returned.
    """
    scale = float(np.sqrt(np.max(expected["var"])))
    for name in (*FAMILIES, "vector"):
        try:
            assert_parity(
                actual[name], expected[name], dtype, scale=scale if name == "mean" else None
            )
        except AssertionError as error:
            raise BenchError(
                f"case {case!r}: ESTEL's {name} differs from STL's on the same inputs, so the "
                f"times of this run measure a wrong computation (Part I, §6.3). {error}"
            ) from None
    stl_gradient = expected["gradient"]
    finite = np.isfinite(stl_gradient)
    gradient_error = None
    if finite.any():
        largest = float(np.abs(stl_gradient[finite]).max())
        difference = float(np.abs(actual["gradient"][finite] - stl_gradient[finite]).max())
        gradient_error = difference / largest if largest > 0 else None
    return {
        "statistics": f"equal to STL's within the {dtype} parity tolerances (Part II, §14.3)",
        "gradient_relative_error": gradient_error,
        "stl_gradient_not_finite": int((~finite).sum()),
    }


def _relative_difference(actual: NDArray[Any], expected: NDArray[Any]) -> float | None:
    """The largest ``|actual - expected|`` over the finite values of ``expected``, relative to
    its largest modulus; None when ``expected`` has no finite value or is zero."""
    finite = np.isfinite(expected)
    if not finite.any():
        return None
    largest = float(np.abs(expected[finite]).max())
    if largest == 0:
        return None
    return float(np.abs(actual[finite] - expected[finite]).max()) / largest


def check_against_main(
    case: str,
    actual: dict[str, NDArray[Any]],
    expected: dict[str, NDArray[Any]],
    dtype: str,
    gradient_rtol: float = MAIN_GRADIENT_RTOL,
) -> dict[str, Any]:
    """Compare the results of an ESTEL run with those of the ESTEL of ``main`` on the same
    inputs (Part II, §15.2, T2.12; issue #90, decision 2).

    The statistics must agree with the parity tolerances of Part II, §14.3 for ``dtype``, as in
    :func:`check_against_stl`, and the gradient of the loss within ``gradient_rtol``:
    ``max |g - g_main| <= gradient_rtol x max |g_main|``, every value finite. Raises
    ``BenchError`` otherwise. Returns the measured differences: for each family and for the
    gradient, the largest difference relative to the largest value of ``main``.
    """
    scale = float(np.sqrt(np.max(expected["var"])))
    for name in (*FAMILIES, "vector"):
        try:
            assert_parity(
                actual[name], expected[name], dtype, scale=scale if name == "mean" else None
            )
        except AssertionError as error:
            raise BenchError(
                f"case {case!r}: {name} differs from that of main on the same inputs beyond "
                f"the {dtype} parity tolerance, so this is not an optimisation of step 1 (Part "
                f"I, §3.1, rule 8). Stop and comment on the task's issue with the case and "
                f"these numbers. {error}"
            ) from None
    gradient, main_gradient = actual["gradient"], expected["gradient"]
    difference = _relative_difference(gradient, main_gradient)
    if (
        gradient.shape != main_gradient.shape
        or not np.isfinite(gradient).all()
        or not np.isfinite(main_gradient).all()
        or (difference is not None and difference > gradient_rtol)
    ):
        raise BenchError(
            f"case {case!r}: the gradient of the loss differs from that of main: max |g - "
            f"g_main| = {difference} x max |g_main|, above {gradient_rtol:g}, or a value is not "
            f"finite ({int((~np.isfinite(gradient)).sum())} here, "
            f"{int((~np.isfinite(main_gradient)).sum())} on main). Stop and comment on the "
            "task's issue with the case and these numbers."
        )
    return {
        "statistics": f"equal to main's within the {dtype} parity tolerances (Part II, §14.3)",
        "statistics_relative_difference": {
            name: _relative_difference(actual[name], expected[name]) for name in FAMILIES
        },
        "gradient_relative_difference": difference,
        "gradient_rtol": gradient_rtol,
    }


# ---------------------------------------------------------------------- measured processes


def _load(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def _worker_arguments(
    library: str, inputs: str, out: str, backend: str, settings: Settings
) -> list[str]:
    return [
        "-m", "bench.worker", "--library", library, "--inputs", inputs, "--out", out,
        "--backend", backend, "--device", settings.device, "--threads", str(settings.threads),
        "--warmup", str(settings.warmup), "--repeats", str(settings.repeats),
    ]  # fmt: skip


def _run_process(command: list[str], what: str, env: dict[str, str] | None = None) -> None:
    try:
        completed = subprocess.run(command, cwd=REPO, timeout=TIMEOUT_SECONDS, env=env)
    except subprocess.TimeoutExpired:
        raise BenchError(
            f"{what} ran for more than {TIMEOUT_SECONDS // 60} minutes and was stopped. The "
            "case is too large for this machine: comment on the task's issue with this time, "
            "so that the maintainer resizes it."
        ) from None
    if completed.returncode != 0:
        hint = (
            " Exit code 137 is the container's memory limit: comment on the task's issue with "
            "the memory the container has, so that the maintainer resizes the case."
            if completed.returncode == 137
            else ""
        )
        raise BenchError(
            f"{what} failed with exit code {completed.returncode}: read its error above.{hint}"
        )


def run_native(
    case: Case,
    work: Path,
    backend: str,
    settings: Settings,
    main_source: Path | None = None,
    label: str | None = None,
) -> Path:
    """Measure ESTEL on this machine; returns the path of the results, without suffix.

    With ``main_source``, the ``src`` directory of an export of a commit (:func:`export_main`),
    the process imports that ``estel`` instead of the working tree's: the same worker, the same
    interpreter and libraries, the same inputs. ``label`` names the results (by default
    ``estel``, or ``estel-main`` with ``main_source``).
    """
    tree = label or ("estel" if main_source is None else "estel-main")
    out = work / f"native-{tree}-{case.name}"
    arguments = _worker_arguments(
        "estel", str(work / f"{case.name}.npz"), str(out), backend, settings
    )
    env = None
    what = f"ESTEL on case {case.name!r} (native)"
    if main_source is not None:
        # PYTHONPATH comes before the environment's site-packages, where the working tree is
        # installed: this process imports the estel of that commit.
        env = {**os.environ, "PYTHONPATH": str(main_source)}
        what = f"ESTEL of {label or 'main'} on case {case.name!r} (native)"
    _run_process([sys.executable, *arguments], what, env)
    return out


def export_main(ref: str, work: Path) -> tuple[Path, str]:
    """The ``src`` directory of ``ref`` (``origin/main``), exported once per commit under
    ``work``, and its commit. No checkout and no worktree: sessions never check out ``main``
    (AGENTS.md, rule 14)."""
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    if not commit:
        raise BenchError(
            f"No commit {ref!r} to compare with: run git fetch origin first, or pass --main "
            "REF with the commit the optimisation starts from."
        )
    destination = work / f"main-{commit[:12]}"
    if not (destination / "src" / "estel" / "__init__.py").is_file():
        destination.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", commit, "src"], capture_output=True
        )
        if (
            archive.returncode != 0
            or subprocess.run(
                ["tar", "-x", "-C", str(destination)], input=archive.stdout
            ).returncode
            != 0
        ):
            raise BenchError(
                f"Exporting src/ of {ref} ({commit[:12]}) to {destination} failed: "
                f"{archive.stderr.decode().strip() or 'read the error above'}."
            )
    return destination / "src", commit


def run_container(
    case: Case, library: str, work: Path, settings: Settings, container: str, image: str
) -> Path:
    """Measure ``library`` in the benchmark image; returns the path of the results."""
    out = work / f"container-{library}-{case.name}"
    arguments = _worker_arguments(
        library, f"/mnt/work/{case.name}.npz", f"/mnt/work/{out.name}", "torch", settings
    )
    command = [
        container, "run", "--rm", "--platform", CONTAINER_PLATFORM,
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
        "-v", f"{REPO / 'bench'}:/mnt/bench:ro", "-v", f"{work}:/mnt/work",
        image, CONTAINER_PYTHON, *arguments,
    ]  # fmt: skip
    _run_process(command, f"{library.upper()} on case {case.name!r} (container)")
    return out


def build_image(container: str, parity_image: str, image: str) -> str:
    """Build the benchmark image from the working tree (cached when nothing changed) and return
    its ID."""

    def inspect(name: str) -> subprocess.CompletedProcess[str]:
        command = [container, "image", "inspect", "--format", "{{.Id}}", name]
        return subprocess.run(command, capture_output=True, text=True)

    try:
        missing = inspect(parity_image).returncode != 0
    except FileNotFoundError:
        raise BenchError(
            f"'{container}' not found: install Docker, or run make bench CONTAINER=podman, or "
            "make bench BENCH_ENV=native to measure ESTEL alone against stored STL statistics."
        ) from None
    if missing:
        raise BenchError(f"No image '{parity_image}': build it first with make parity-image.")
    command = [
        container, "build", "--quiet", "--platform", CONTAINER_PLATFORM,
        "--build-arg", f"PARITY_IMAGE={parity_image}",
        "-f", str(REPO / "bench" / "Dockerfile"), "-t", image, str(REPO),
    ]  # fmt: skip
    if subprocess.run(command, stdout=subprocess.DEVNULL).returncode != 0:
        raise BenchError(f"Building the benchmark image '{image}' failed: read its error above.")
    return inspect(image).stdout.strip()


# ---------------------------------------------------------------------- results


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *arguments], capture_output=True, text=True
    ).stdout.strip()


def machine_name() -> str:
    """A short name of this machine for the results file, from its CPU model: no host name."""
    return re.sub(r"[^a-z0-9]+", "-", _cpu_model().lower()).strip("-") or "unknown"


def results_path(directory: Path, date: str, machine: str) -> Path:
    """``<date>-<machine>.json``, or with ``-2``, ``-3``... when that file exists: a result
    is never overwritten."""
    path = directory / f"{date}-{machine}.json"
    count = 1
    while path.exists():
        count += 1
        path = directory / f"{date}-{machine}-{count}.json"
    return path


def format_table(runs: list[dict[str, Any]]) -> str:
    """The short table of the runs: times in seconds (median), memory in GiB: the peak of
    the process, and the peak minus its baseline, the memory of the computation itself."""
    header = ("case", "environment", "library", "forward (s)", "forward + gradient (s)",
              "peak memory (GiB)", "peak - baseline (GiB)")  # fmt: skip
    rows = [
        (
            run["case"], run["environment"],
            f"{run['library']} (main)" if run.get("tree") == "main" else run["library"],
            f"{run['forward_seconds']:.3f}", f"{run['gradient_seconds']:.3f}",
            f"{run['peak_rss_bytes'] / 2**30:.2f}",
            f"{(run['peak_rss_bytes'] - run['baseline_rss_bytes']) / 2**30:.2f}",
        )
        for run in runs
    ]  # fmt: skip
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]

    def line(row: tuple[str, ...]) -> str:
        cells = [
            cell.ljust(width) if i < 3 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(row, widths, strict=True))
        ]
        return "| " + " | ".join(cells) + " |"

    rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(header), rule, *(line(row) for row in rows)])


def format_main_differences(runs: list[dict[str, Any]]) -> str:
    """The differences with ``main`` of the runs checked against it: for each family and for
    the gradient, the largest difference relative to the largest value of ``main``."""
    header = ("case", *FAMILIES, "gradient")

    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.1e}"

    rows = [
        (
            run["case"],
            *(cell(run["check_main"]["statistics_relative_difference"][f]) for f in FAMILIES),
            cell(run["check_main"]["gradient_relative_difference"]),
        )
        for run in runs
    ]
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in (header, *rows)
    ]
    rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([lines[0], rule, *lines[1:]])


def _record(out: Path, environment: str, environments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The measurements of a finished process; its description of the environment is merged
    into ``environments``."""
    run: dict[str, Any] = json.loads(out.with_suffix(".json").read_text())
    description = run.pop("environment")
    known = environments.setdefault(environment, description)
    known["versions"].update(description["versions"])
    return {"case": run.pop("case"), "environment": environment, **run}


def _reference(work: Path, case: Case, digest: str) -> dict[str, NDArray[Any]]:
    """STL's results on the inputs of ``case``, from the last container run."""
    out = work / f"container-stl-{case.name}"
    if out.with_suffix(".json").is_file() and out.with_suffix(".npz").is_file():
        stored = json.loads(out.with_suffix(".json").read_text())
        if stored.get("inputs_sha256") == digest:
            return _load(out.with_suffix(".npz"))
    raise BenchError(
        f"case {case.name!r}: no STL statistics for these inputs in {work}, so the run cannot "
        "be checked (Part I, §6.3). Run the container environment first: make bench "
        f"BENCH_CASE={case.name} BENCH_ENV=container (it needs make parity-image)."
    )


# ---------------------------------------------------------------------- comparison of two commits


def _measure_of(run: dict[str, Any], measure: str) -> float:
    """The value of ``measure`` in a run: the median time of the forward pass or of the forward
    pass with the gradient, the memory of the computation (the peak minus the baseline), or
    the peak resident memory of the process."""
    if measure == "memory":
        return float(run["peak_rss_bytes"] - run["baseline_rss_bytes"])
    if measure == "peak":
        return float(run["peak_rss_bytes"])
    return float(run[f"{measure}_seconds"])


def comparison_ratios(rounds: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """The ratios B / A of a case over the rounds of a comparison: for each measure of
    ``COMPARE_MEASURES``, the ratio of each round, their median and their range.

    ``rounds`` holds, per round, the run of A and the run of B (the measurements of
    ``bench/worker.py``). A ratio below 1 is a gain of B.
    """
    out: dict[str, Any] = {}
    for measure in COMPARE_MEASURES:
        ratios = [_measure_of(b, measure) / _measure_of(a, measure) for a, b in rounds]
        out[measure] = {
            "ratios": ratios,
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        }
    return out


def comparison_verdict(ratios: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """What the criterion of issue #91 (decision 1) reads in the ratios of every case: the
    measures, ``case: measure``, whose ratio is below 1 in every round (a gain), and those whose
    ratio is above ``COMPARE_LOSS`` in at least one round (a loss); for the time measures, and
    for the memory of the computation."""
    gains, losses = [], []
    for case, of_case in ratios.items():
        for measure in ("forward", "gradient", "memory"):
            found = of_case[measure]
            if found["max"] < 1:
                gains.append(f"{case}: {measure}")
            if measure != "memory" and found["max"] > COMPARE_LOSS:
                losses.append(f"{case}: {measure}")
    return {"below_1_in_every_round": gains, f"above_{COMPARE_LOSS}_in_a_round": losses}


def format_comparison(ratios: dict[str, dict[str, Any]]) -> str:
    """The table of a comparison: per case and measure, the median ratio B / A and its range
    over the rounds."""
    header = ("case", "forward", "forward + gradient", "memory (peak - baseline)", "peak memory")

    def cell(found: dict[str, Any]) -> str:
        return f"{found['median']:.3f} ({found['min']:.3f} to {found['max']:.3f})"

    rows = [
        (case, *(cell(of_case[measure]) for measure in COMPARE_MEASURES))
        for case, of_case in ratios.items()
    ]
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in (header, *rows)
    ]
    rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([lines[0], rule, *lines[1:]])


def _first_calls(rounds: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """The median, over the rounds, of the time of the very first call of the operator, and of
    the first call and of the later calls of each time measure, for A and for B: on JAX the
    first calls hold the compilation, of the forward pass and of the gradient."""
    out: dict[str, Any] = {}
    for side, runs in (("a", [a for a, _ in rounds]), ("b", [b for _, b in rounds])):
        applies = [
            float(run["first_apply_seconds"]) for run in runs if run.get("first_apply_seconds")
        ]
        out[f"{side}_first_apply_seconds"] = (
            statistics.median(applies) if len(applies) == len(runs) else None
        )
        for measure in ("forward", "gradient"):
            first = [run.get(f"{measure}_first_seconds") for run in runs]
            known = [float(value) for value in first if value is not None]
            out[f"{side}_{measure}_first_seconds"] = (
                statistics.median(known) if len(known) == len(first) else None
            )
            out[f"{side}_{measure}_seconds"] = statistics.median(
                run[f"{measure}_seconds"] for run in runs
            )
    return out


def run_comparison(args: argparse.Namespace) -> Path:
    """Compare the commits ``args.compare`` (A, B) natively, on the selected cases: R rounds,
    each measuring A then B, or B then A in the odd rounds, every measure in a fresh process
    (warm-up, then the timed runs, their median); per round the ratios B / A. Every run is
    checked against STL's statistics, and B against the statistics and the gradient of A."""
    settings, cases = load_cases(args.cases_file)
    if args.threads is not None:
        settings = replace(settings, threads=args.threads)
    if args.repeats is not None:
        settings = replace(settings, repeats=args.repeats)
    unknown = sorted(set(args.case or ()) - set(cases))
    if unknown:
        raise BenchError(f"No case {unknown} in {args.cases_file}: choose among {sorted(cases)}.")
    if args.rounds < 1:
        raise BenchError(f"--rounds {args.rounds}: a comparison has at least one round.")
    selected = [cases[name] for name in (args.case or cases)]
    work = args.results / ".work"
    work.mkdir(parents=True, exist_ok=True)
    sides = {}
    for side, ref in zip(("a", "b"), args.compare, strict=True):
        source, commit = export_main(ref, work)
        sides[side] = {"ref": ref, "commit": commit, "source": source}

    described: dict[str, dict[str, Any]] = {}
    ratios: dict[str, dict[str, Any]] = {}
    first_calls: dict[str, dict[str, Any]] = {}
    differences: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for case in selected:
        digest = write_inputs(case, settings.dtype, work / f"{case.name}.npz")
        expected = _reference(work, case, digest)
        rounds: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index in range(args.rounds):
            found: dict[str, dict[str, Any]] = {}
            arrays: dict[str, dict[str, NDArray[Any]]] = {}
            for side in ("a", "b") if index % 2 == 0 else ("b", "a"):
                ref = sides[side]["ref"]
                print(f"== {case.name}: round {index + 1}, {side.upper()} ({ref})", flush=True)
                out = run_native(
                    case, work, args.backend, settings, sides[side]["source"], f"compare-{side}"
                )
                run = {**_record(out, "native", described), "side": side, "round": index + 1}
                arrays[side] = _load(out.with_suffix(".npz"))
                run["check"] = check_against_stl(case.name, arrays[side], expected, settings.dtype)
                found[side] = run
            found["b"]["check_a"] = check_against_main(
                case.name, arrays["b"], arrays["a"], settings.dtype
            )
            rounds.append((found["a"], found["b"]))
            runs.extend(found[side] for side in ("a", "b"))
        ratios[case.name] = comparison_ratios(rounds)
        first_calls[case.name] = _first_calls(rounds)
        differences.append({"case": case.name, "check_main": rounds[-1][1]["check_a"]})

    date = datetime.date.today().isoformat()
    verdict = comparison_verdict(ratios)
    document = {
        "schema": SCHEMA,
        "date": date,
        "machine": args.machine or machine_name(),
        "comparison": {
            "a": {key: str(sides["a"][key]) for key in ("ref", "commit")},
            "b": {key: str(sides["b"][key]) for key in ("ref", "commit")},
            "rounds": args.rounds,
            "ratios": ratios,
            "first_calls": first_calls,
            "verdict": verdict,
        },
        "backend": args.backend,
        "settings": asdict(settings),
        "cases": {case.name: asdict(case) for case in selected},
        "environments": described,
        "runs": runs,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    path = results_path(args.results, date, f"{document['machine']}-compare")
    path.write_text(json.dumps(document, indent=2) + "\n")
    a, b = sides["a"], sides["b"]
    print(
        f"\nB = {b['ref']} ({b['commit'][:12]}) against A = {a['ref']} ({a['commit'][:12]}), "
        f"{args.backend}, native: ratios B / A, the median of {args.rounds} rounds and its range "
        "(below 1: B gains)."
    )
    print(format_comparison(ratios))
    for name, measures in verdict.items():
        print(f"{name}: {', '.join(measures) or 'none'}")
    if args.backend == "jax":
        print("\nFirst call and later calls (s), the median of the rounds:")
        for case_name, calls in first_calls.items():
            print(f"  {case_name}: " + ", ".join(
                f"{key.removesuffix('_seconds')} {value:.3f}"
                for key, value in calls.items() if value is not None
            ))  # fmt: skip
    print(
        f"\nEvery run equals STL within the {settings.dtype} parity tolerances; B equals A "
        f"within them, and its gradient within {MAIN_GRADIENT_RTOL:g}. Largest differences of "
        "the last round, relative to the largest value of A:"
    )
    print(format_main_differences(differences))
    _warn_power(described.get("native", {}).get("power") or _power())
    print(f"Results written to {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    return path


def _warn_power(power: dict[str, Any] | None) -> None:
    if power and (power["low_power_mode"] or power["source"] != "AC Power"):
        print(
            f"WARNING: measured on {power['source']}, Low Power Mode "
            f"{'on' if power['low_power_mode'] else 'off'}: the times are slower than on AC "
            "power and do not compare with other results. Do not commit them as a reference."
        )


def run_benchmark(args: argparse.Namespace) -> Path:
    settings, cases = load_cases(args.cases_file)
    if args.threads is not None:
        settings = replace(settings, threads=args.threads)
    if args.repeats is not None:
        settings = replace(settings, repeats=args.repeats)
    unknown = sorted(set(args.case or ()) - set(cases))
    if unknown:
        raise BenchError(f"No case {unknown} in {args.cases_file}: choose among {sorted(cases)}.")
    selected = [cases[name] for name in (args.case or cases)]
    environments = args.env or (["native"] if args.backend != "torch" else list(ENVIRONMENTS))
    if "container" in environments and args.backend != "torch":
        raise BenchError(
            f"--backend {args.backend}: the benchmark image has PyTorch only, as STL does. Run "
            "with --env native; STL's statistics come from the last container run."
        )
    work = args.results / ".work"
    work.mkdir(parents=True, exist_ok=True)

    described: dict[str, dict[str, Any]] = {}
    if "container" in environments:
        image_id = build_image(args.container, args.parity_image, args.image)
    main_source, main_commit = None, None
    if "native" in environments and args.main:
        main_source, main_commit = export_main(args.main, work)
    runs: list[dict[str, Any]] = []
    for case in selected:
        inputs = work / f"{case.name}.npz"
        digest = write_inputs(case, settings.dtype, inputs)
        if "container" in environments:
            print(f"== {case.name}: STL, container", flush=True)
            out = run_container(case, "stl", work, settings, args.container, args.image)
            runs.append({**_record(out, "container", described), "check": "the reference"})
        expected = _reference(work, case, digest)
        for environment in environments:
            of_main = None
            if environment == "native" and main_source is not None:
                # main first, then the working tree, one after the other on the same inputs.
                print(f"== {case.name}: ESTEL of {args.main}, native", flush=True)
                out = run_native(case, work, args.backend, settings, main_source)
                run = {**_record(out, environment, described), "tree": "main"}
                of_main = _load(out.with_suffix(".npz"))
                run["check"] = check_against_stl(case.name, of_main, expected, settings.dtype)
                runs.append(run)
            print(f"== {case.name}: ESTEL, {environment}", flush=True)
            if environment == "container":
                out = run_container(case, "estel", work, settings, args.container, args.image)
            else:
                out = run_native(case, work, args.backend, settings)
            run = _record(out, environment, described)
            actual = _load(out.with_suffix(".npz"))
            run["check"] = check_against_stl(case.name, actual, expected, settings.dtype)
            if of_main is not None:
                run["check_main"] = check_against_main(case.name, actual, of_main, settings.dtype)
            runs.append(run)
    if "container" in described:
        described["container"].update(
            image=args.image,
            image_id=image_id,
            platform=CONTAINER_PLATFORM,
            host_architecture=platform.machine(),
            emulated=platform.machine().lower() not in ("x86_64", "amd64"),
        )

    date = datetime.date.today().isoformat()
    document = {
        "schema": SCHEMA,
        "date": date,
        "machine": args.machine or machine_name(),
        "estel_commit": _git("rev-parse", "HEAD"),
        "main": None if main_commit is None else {"ref": args.main, "commit": main_commit},
        "estel_uncommitted_changes": bool(
            _git("status", "--porcelain", "--", "src", "bench", ":(exclude)bench/results")
        ),
        "backend": args.backend,
        "settings": asdict(settings),
        "cases": {case.name: asdict(case) for case in selected},
        "environments": described,
        "runs": runs,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    path = results_path(args.results, date, document["machine"])
    path.write_text(json.dumps(document, indent=2) + "\n")
    print()
    print(format_table(runs))
    print(f"\nEvery ESTEL run equals STL within the {settings.dtype} parity tolerances.")
    compared = [run for run in runs if "check_main" in run]
    if compared and main_commit is not None:
        print(
            f"\nAgainst {args.main} ({main_commit[:12]}), native: statistics within the "
            f"{settings.dtype} parity tolerances, gradient within {MAIN_GRADIENT_RTOL:g}. "
            "Largest differences, relative to the largest value of main:"
        )
        print(format_main_differences(compared))
    _warn_power(described.get("native", {}).get("power") or _power())
    print(f"Results written to {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", action="append", help="run this case only; may be repeated")
    parser.add_argument(
        "--env", action="append", choices=ENVIRONMENTS,
        help="run this environment only; may be repeated (default: both)",
    )  # fmt: skip
    parser.add_argument("--backend", default="torch", choices=("torch", "jax"),
                        help="the backend of ESTEL (default: torch)")  # fmt: skip
    parser.add_argument("--threads", type=int, help="override the threads of the cases file")
    parser.add_argument("--repeats", type=int, help="override the repeats of the cases file")
    parser.add_argument("--machine", help="name of this machine in the results file")
    parser.add_argument(
        "--main", default=MAIN_REF, metavar="REF",
        help="the commit the native run is compared with, measured just before it "
        f"(default: {MAIN_REF}); an empty string skips it",
    )  # fmt: skip
    parser.add_argument(
        "--compare", nargs=2, metavar=("A", "B"),
        help="compare two commits natively instead: per round, the ratios B / A of the times "
        "and of the memory (A = B gives the noise floor)",
    )  # fmt: skip
    parser.add_argument("--rounds", type=int, default=COMPARE_ROUNDS,
                        help=f"rounds of --compare (default: {COMPARE_ROUNDS})")  # fmt: skip
    parser.add_argument("--cases-file", type=Path, default=CASES_FILE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--container", default="docker", help="container runtime")
    parser.add_argument("--parity-image", default="estel-stl-parity")
    parser.add_argument("--image", default="estel-bench")
    args = parser.parse_args(argv)
    try:
        if args.compare:
            run_comparison(args)
        else:
            run_benchmark(args)
    except BenchError as error:
        print(f"make bench: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
