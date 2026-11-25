"""One measured process of the benchmark (task T2.11, issue #89): one case, one library.

    python -m bench.worker --library estel --inputs case.npz --out run --threads 4

`make bench` (``bench/__main__.py``) starts one such process per case, library and environment,
so that the peak resident memory is that of a process that runs this case only. It writes
``<out>.json`` (the times, the peak memory, the description of the environment) and ``<out>.npz``
(the statistics, the loss and its gradient, for the check against STL).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench._cases import Inputs, read_inputs

LIBRARIES = {"estel": "bench._estel", "stl": "bench._stl"}


def timed(
    call: Callable[[], Any], warmup: int, repeats: int
) -> tuple[list[float], Any, float | None]:
    """The wall times of ``repeats`` calls, after ``warmup`` calls that are not part of them,
    the result of the last call, and the wall time of the very first call (None without
    warm-up): on a backend that compiles, the first call holds the compilation."""
    result = None
    first = None
    for index in range(warmup):
        start = time.perf_counter()
        result = call()
        if index == 0:
            first = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return times, result, first


def peak_rss_bytes() -> int:
    """The peak resident memory of this process so far. ``ru_maxrss`` is in bytes on macOS and
    in kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def rss_bytes() -> int:
    """The resident memory of this process now: from ``/proc`` on Linux, from ``ps`` (in
    kilobytes) on macOS."""
    if sys.platform == "darwin":
        command = ["ps", "-o", "rss=", "-p", str(os.getpid())]
        return int(subprocess.run(command, capture_output=True, text=True).stdout) * 1024
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def measure(
    library: str, inputs: Inputs, warmup: int, repeats: int
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Measure a case in the current process, with ``library`` (``"estel"`` or ``"stl"``): the
    forward pass, then the forward pass with the gradient of the loss, each as ``warmup`` runs
    then ``repeats`` timed runs, and the peak resident memory after both. The first call of each
    is timed apart (``forward_first_seconds``, ``gradient_first_seconds``), and so is the very
    first call of the operator, made on the second field when the run is built
    (``first_apply_seconds``): on JAX these hold the compilation of the gradient and of the
    forward pass. The construction of the operator is timed apart (``build_seconds``), and
    the peak resident memory is read once it is built (``build_peak_rss_bytes``; task T5.7). The
    baseline is the
    resident memory once the library and the inputs are loaded, just before the operator is
    built: the peak minus the baseline is the memory of the computation itself. Returns the
    measurements and the results of the last runs as NumPy arrays."""
    module = importlib.import_module(LIBRARIES[library])
    baseline = rss_bytes()
    build_peak: list[int] = []
    run = module.Run(inputs, built=lambda: build_peak.append(peak_rss_bytes()))
    forward, statistics_, forward_first = timed(run.forward, warmup, repeats)
    gradient, loss_and_gradient, gradient_first = timed(run.gradient, warmup, repeats)
    peak = peak_rss_bytes()
    arrays: dict[str, NDArray[Any]] = run.arrays(statistics_, loss_and_gradient)
    measurements = {
        # The construction of the operator, timed apart, and the peak once it is built (#168).
        "build_seconds": run.build_seconds,
        "build_peak_rss_bytes": build_peak[0],
        "forward_seconds": statistics.median(forward),
        "forward_runs": forward,
        "forward_first_seconds": forward_first,
        # The very first call of the operator, on the second field, when the run is built.
        "first_apply_seconds": getattr(run, "first_apply_seconds", None),
        "gradient_seconds": statistics.median(gradient),
        "gradient_runs": gradient,
        "gradient_first_seconds": gradient_first,
        "peak_rss_bytes": peak,
        "baseline_rss_bytes": baseline,
        "coefficients": int(arrays["vector"].size),
        "loss": float(arrays["loss"]),
    }
    return measurements, arrays


def _cpu_model() -> str:
    if sys.platform == "darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _power() -> dict[str, Any] | None:
    """On macOS, the power source and whether Low Power Mode is on: both slow the machine, so a
    result measured on battery or in Low Power Mode does not compare with the others."""
    if sys.platform != "darwin":
        return None
    source = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    low = [line.split()[-1] == "1" for line in settings.splitlines() if "lowpowermode" in line]
    return {"source": source.partition("'")[2].partition("'")[0] or "unknown",
            "low_power_mode": any(low)}  # fmt: skip


def describe_environment(versions: dict[str, str]) -> dict[str, Any]:
    """The machine as this process sees it: CPU model and cores, memory, OS, Python and library
    versions. In a container these are the container's: the cores and the memory it is given."""
    memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {
        "cpu": _cpu_model(),
        "architecture": platform.machine(),
        "cores": os.cpu_count(),
        "memory_bytes": memory,
        "os": platform.platform(),
        "python": platform.python_version(),
        "power": _power(),
        "versions": {"numpy": np.__version__, **versions},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", choices=sorted(LIBRARIES), required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="written by bench._cases")
    parser.add_argument("--out", type=Path, required=True, help="writes <out>.json and <out>.npz")
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    inputs = read_inputs(args.inputs)
    configure = importlib.import_module(LIBRARIES[args.library]).configure
    versions = configure(args.backend, inputs.dtype, args.device, args.threads)
    measurements, arrays = measure(args.library, inputs, args.warmup, args.repeats)
    result = {
        "case": inputs.case.name,
        "library": args.library,
        "backend": args.backend,
        "dtype": inputs.dtype,
        "device": args.device,
        "threads": args.threads,
        "inputs_sha256": inputs.digest,
        **measurements,
        "environment": describe_environment(versions),
    }
    members: dict[str, Any] = dict(arrays)
    np.savez(args.out.with_suffix(".npz"), **members)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
