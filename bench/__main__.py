"""The benchmark of step 1: `make bench` (Part I, §6.3; task T2.11, issue #89).

    python -m bench [--case NAME] [--env native|container] [--backend torch|jax]

Runs the cases of ``bench/cases.toml``, each in a process of its own (``bench/worker.py``), in
two environments: ``container``, the parity container with ESTEL installed, where STL and ESTEL
are measured side by side; and ``native``, ESTEL on this machine. Every ESTEL run is checked
against STL's statistics on the same inputs, with the parity tolerances of Part II, §14.3, and
the benchmark fails otherwise. Prints a table and writes ``bench/results/<date>-<machine>.json``.
Outside CI: see ``bench/README.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tests.parity._compare import assert_parity

from bench._cases import CASES_FILE, Case, Settings, load_cases, write_inputs
from bench.worker import _cpu_model

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
ENVIRONMENTS = ("container", "native")
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
CONTAINER_PLATFORM = "linux/amd64"
CONTAINER_PYTHON = "/opt/venv/master/bin/python"
# A measured process that runs longer is stopped: the maintainer resizes a case on which STL
# takes more than about 30 minutes (issue #89, decision 8).
TIMEOUT_SECONDS = 45 * 60
SCHEMA = 1


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


def _run_process(command: list[str], what: str) -> None:
    try:
        completed = subprocess.run(command, cwd=REPO, timeout=TIMEOUT_SECONDS)
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


def run_native(case: Case, work: Path, backend: str, settings: Settings) -> Path:
    """Measure ESTEL on this machine; returns the path of the results, without suffix."""
    out = work / f"native-estel-{case.name}"
    arguments = _worker_arguments(
        "estel", str(work / f"{case.name}.npz"), str(out), backend, settings
    )
    _run_process([sys.executable, *arguments], f"ESTEL on case {case.name!r} (native)")
    return out


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
    """The short table of the runs: times in seconds (median), peak memory in GiB."""
    header = ("case", "environment", "library", "forward (s)", "forward + gradient (s)",
              "peak memory (GiB)")  # fmt: skip
    rows = [
        (
            run["case"], run["environment"], run["library"],
            f"{run['forward_seconds']:.3f}", f"{run['gradient_seconds']:.3f}",
            f"{run['peak_rss_bytes'] / 2**30:.2f}",
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
            print(f"== {case.name}: ESTEL, {environment}", flush=True)
            if environment == "container":
                out = run_container(case, "estel", work, settings, args.container, args.image)
            else:
                out = run_native(case, work, args.backend, settings)
            run = _record(out, environment, described)
            actual = _load(out.with_suffix(".npz"))
            run["check"] = check_against_stl(case.name, actual, expected, settings.dtype)
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
    parser.add_argument("--cases-file", type=Path, default=CASES_FILE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--container", default="docker", help="container runtime")
    parser.add_argument("--parity-image", default="estel-stl-parity")
    parser.add_argument("--image", default="estel-bench")
    try:
        run_benchmark(parser.parse_args(argv))
    except BenchError as error:
        print(f"make bench: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
