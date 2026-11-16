"""Shared test configuration: backends, dtypes, the two tiers and their time budget.

**Tiers** (Part I, §5.4 and §6.1), chosen with ``--tier``:

- ``full``, the default of a bare ``pytest`` (``make check-full``; CI on ``main``, nightly, and on
  a pull request labelled ``full-ci``): every test, in float32 and float64.
- ``fast`` (``make check``; CI on every other pull request): the tests not marked ``slow`` or
  ``serial``. On PyTorch, all of them in float64, and those marked ``dtype`` in float32 too. On
  the experimental JAX backend, only those marked ``jax_smoke``, in float64. The tests of the
  check tools (``tests/tools/``) run only when the change touches ``tools/``, ``tests/tools/`` or
  ``pyproject.toml`` (``TOOLS_INPUTS``, against ``ESTEL_TEST_BASE``, ``origin/main`` by default):
  every pull request runs the tools themselves on the repository (``make check-tools``).

**Time budget of the fast tier.** A test of the fast tier that takes more than 1 second (set-up
and call; ``FAST_TEST_BUDGET``) fails: make it faster, on smaller data, or mark it ``slow`` with
the reason in its docstring (a sentence starting with ``Slow:``). The time is CPU time, the test's
subprocesses included, so that the other workers' load does not count; a wall-time limit five
times larger catches a test that waits. CI enforces the budget; elsewhere a test fails at twice it
(``LOCAL_BUDGET_FACTOR``) and the summary lists those in between. On JAX the times are only
reported (see ``FAST_TEST_BUDGET``). Every test marked ``slow`` needs that sentence, in its own
docstring or in that of its class or module; collection fails otherwise. ``make check`` and CI
also check the wall time of the whole fast tier.

**Backends and dtypes.** ``ESTEL_TEST_BACKEND`` names the backend of a run
(``tests/_backends.py``): one backend per process, since the backend and JAX's 64-bit mode are
process-wide. Both dtypes run in the same process: the ``dtype`` fixture is parametrised (Part I,
§3.1, rule 7), so a test that does not take it runs once. ``ESTEL_TEST_DTYPE=float32`` or
``float64`` restricts a full-tier run to one dtype, as the JAX jobs of CI do.

**Passes** (task T0.7): the tests run in parallel, in ``pytest-xdist`` workers
(``-n auto -m "not serial"``), then, in the full tier, the tests marked ``serial`` alone, in one
process (``-p no:xdist -m serial``). Mark ``serial`` a test that starts fresh processes or switches
a process-wide setting, such as ``Backend.deterministic``. The fast tier has no serial pass: a
pytest process of its own costs more than its tests.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from estel.backend._protocol import Backend
from estel.backend._registry import set_backend
from tests._backends import backend_name, make_backend

DTYPES = ("float32", "float64")
TIERS = ("fast", "full")
# Budget of a fast-tier test, set-up and call (Part I, §5.4): seconds of CPU time, its
# subprocesses included, which the other workers do not inflate; and a limit on wall time, for a
# test that waits rather than computes. It is enforced on PyTorch, where every test runs. On the
# experimental JAX backend, which runs eagerly op by op, the first operator of a process compiles
# or loads one executable per operation: a parity case takes 1 to 3 s with JAX's compilation cache
# and 10 to 20 s without it (after a change of uv.lock). There the times are only reported; the
# same tests meet the budget on PyTorch, and the JAX job has its own total budget in CI.
FAST_TEST_BUDGET = 1.0
WALL_LIMIT_FACTOR = 5
# CI is the reference machine of the budget: its runners have equal cores. A laptop with
# efficiency cores measures the same test up to three times longer under load, so outside CI
# (``CI`` unset) a test fails at twice its budget, and the summary lists those between the two.
LOCAL_BUDGET_FACTOR = 2
SLOW_REASON = "Slow:"
REPO = Path(__file__).resolve().parents[1]
# What the tests of tests/tools/ test: the fast tier runs them only when the change touches it.
TOOLS_INPUTS = ("tools", "tests/tools", "pyproject.toml")
_SETUP_DURATION = pytest.StashKey[float]()
_CPU = pytest.StashKey[float]()
_RUN_DTYPE_TESTS = pytest.StashKey[set[str]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tier",
        choices=TIERS,
        default="full",
        help="fast: the tests of a pull request (make check); full: every test (make check-full).",
    )


def _tier(config: pytest.Config) -> str:
    tier: str = config.getoption("--tier")
    return tier


def _run_dtypes() -> tuple[str, ...]:
    value = os.environ.get("ESTEL_TEST_DTYPE")
    if value is None:
        return DTYPES
    if value not in DTYPES:
        raise pytest.UsageError(
            f"ESTEL_TEST_DTYPE={value!r}: set it to one of {DTYPES}, or unset it to run both."
        )
    return (value,)


def _is_worker_or_alone(config: pytest.Config) -> bool:
    """True in a process that runs tests: an xdist worker, or a run without workers."""
    return hasattr(config, "workerinput") or not getattr(config.option, "numprocesses", None)


def pytest_configure(config: pytest.Config) -> None:
    """Set up the backend of the run, before any test creates an array."""
    try:
        name = backend_name()
        dtypes = _run_dtypes()
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if _tier(config) == "fast" and dtypes != DTYPES:
        raise pytest.UsageError(
            "The fast tier chooses its dtypes itself (float64, and float32 for the tests marked "
            "dtype): unset ESTEL_TEST_DTYPE, or run --tier full."
        )
    if _tier(config) == "fast":
        _tools_changed()  # once, here: xdist workers inherit the answer
    _one_thread_per_worker(name)
    if name == "jax":
        # JAX keeps float64 only in its process-wide 64-bit mode, which the backend never
        # switches itself (issue #10); the JAX run switches it on at start, for both dtypes.
        # Every primitive takes an explicit dtype, so float32 tests stay in float32.
        import jax

        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
        _persistent_jax_cache(config)
    # Operators read the default backend when they are built (issue #39): make it the run's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the experimental-JAX warning
        set_backend(name)
    if _is_worker_or_alone(config):
        _warm_up(make_backend(name))


@pytest.hookimpl(optionalhook=True)  # a hook of pytest-xdist, absent with -p no:xdist
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """The workers of ``-n auto``. In the fast tier, none unless it runs the tools tests (on
    PyTorch, when tools change): its tests take a few seconds, less than the start-up of a worker
    (importing PyTorch and its first gradient, about 6 s), so one process is the fastest.
    Otherwise one per CPU, and on Apple silicon one per performance core, since workers on the
    efficiency cores slow the run down."""
    if _tier(config) == "fast" and (backend_name() == "jax" or not _tools_changed()):
        return 0
    if sys.platform == "darwin":
        cores = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if cores.isdigit():
            return int(cores)
    return None  # pytest-xdist's own count


def _tools_changed() -> bool:
    """Whether the working tree differs from ``ESTEL_TEST_BASE`` (``origin/main`` by default) in
    ``TOOLS_INPUTS``, untracked files included; when git cannot say, it counts as changed. The
    answer is kept in ``ESTEL_TOOLS_CHANGED``, which also forces it (``1`` or ``0``)."""
    known = os.environ.get("ESTEL_TOOLS_CHANGED")
    if known in ("0", "1"):
        return known == "1"
    base = os.environ.get("ESTEL_TEST_BASE", "origin/main")
    try:
        diff = subprocess.run(
            ["git", "diff", "--quiet", base, "--", *TOOLS_INPUTS],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        new = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *TOOLS_INPUTS],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        changed = True
    else:
        changed = diff.returncode != 0 or new.returncode != 0 or bool(new.stdout.strip())
    os.environ["ESTEL_TOOLS_CHANGED"] = "1" if changed else "0"
    return changed


def _persistent_jax_cache(config: pytest.Config) -> None:
    """Keep what XLA compiles in ``.pytest_cache`` (CI restores it), so that the next run, and
    the other workers, load it instead of compiling it again. It changes no result: the cache
    only skips compilations, whose output is the same."""
    import jax

    cache = getattr(config, "cache", None)  # absent with -p no:cacheprovider
    if cache is None:
        return
    directory = cache.mkdir("jax-compilation-cache")
    jax.config.update("jax_compilation_cache_dir", str(directory))  # type: ignore[no-untyped-call]
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # type: ignore[no-untyped-call]
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)  # type: ignore[no-untyped-call]


def _one_thread_per_worker(name: str) -> None:
    """Compute on one thread in each xdist worker: with ``-n auto`` each worker would otherwise
    start one compute thread per CPU, and the workers would slow each other down. On JAX, in every
    process of the run, workers or not: XLA's flags are part of the key of JAX's compilation
    cache, so the fast tier (one process) must use the flags of the full tier (workers), whose
    run on the default branch saves the cache that pull requests load."""
    if name == "torch":
        if "PYTEST_XDIST_WORKER" not in os.environ:
            return
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        # Read by XLA when JAX starts its CPU client, at the first array (_warm_up).
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        ).strip()


def _warm_up(backend: Backend) -> None:
    """Pay once per process, before the first test, what the first call of a backend costs:
    PyTorch imports ``torch.func`` and JAX starts its CPU client on first use. Otherwise the
    first test of each worker would carry it, and the time budget would fail at random. A PyTorch
    run also imports JAX, if installed, which the tests that select it need; it does not start
    JAX, whose 64-bit mode those tests set."""
    x = backend.asarray(np.ones(4), "float64")
    backend.value_and_grad(lambda a: backend.xp.sum(a * a))(x)
    if backend.name != "jax" and importlib.util.find_spec("jax") is not None:
        importlib.import_module("estel.backend.jax._primitives")


def _parametrised_names(metafunc: pytest.Metafunc) -> set[str]:
    names: set[str] = set()
    for mark in metafunc.definition.iter_markers("parametrize"):
        argnames = mark.args[0] if mark.args else mark.kwargs["argnames"]
        if isinstance(argnames, str):
            argnames = [name.strip() for name in argnames.split(",")]
        names.update(argnames)
    return names


@pytest.hookimpl(trylast=True)  # after the parametrize marks: the dtype ends the test ID
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise the ``dtype`` fixture with the dtypes of the run. A test that parametrises
    ``dtype`` itself keeps its own values, in every tier."""
    if "dtype" not in metafunc.fixturenames or "dtype" in _parametrised_names(metafunc):
        return
    metafunc.config.stash.setdefault(_RUN_DTYPE_TESTS, set()).add(metafunc.definition.nodeid)
    metafunc.parametrize("dtype", _run_dtypes())


def _is_float32_outside_the_fast_tier(item: pytest.Item, jax: bool) -> bool:
    """A float32 case of the run's dtypes: on JAX, whose smoke subset runs in float64 only; on
    PyTorch, of a test not marked ``dtype`` (on the function, its class or module, or the
    ``pytest.param`` of the case)."""
    callspec = getattr(item, "callspec", None)
    if callspec is None or callspec.params.get("dtype") != "float32":
        return False
    function = item.nodeid.split("[", 1)[0]
    if function not in item.config.stash.get(_RUN_DTYPE_TESTS, set()):
        return False
    return jax or item.get_closest_marker("dtype") is None


def _has_slow_reason(item: pytest.Item) -> bool:
    for node in item.listchain():
        obj = getattr(node, "obj", None)
        if obj is not None and SLOW_REASON in (getattr(obj, "__doc__", None) or ""):
            return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Check that every slow test says why; in the fast tier, keep only the tests of the tier."""
    missing = [
        item.nodeid
        for item in items
        if item.get_closest_marker("slow") and not _has_slow_reason(item)
    ]
    if missing:
        raise pytest.UsageError(
            f"Marked slow without a reason: {', '.join(sorted(missing))}. Add a sentence starting "
            f"with {SLOW_REASON!r} to the docstring of the test (or of its class or module) that "
            "says why it cannot run in under a second on smaller data (Part I, §5.4)."
        )
    if _tier(config) != "fast":
        return
    jax = backend_name() == "jax"
    tools = _tools_changed()
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        out = (
            item.get_closest_marker("slow") is not None
            or item.get_closest_marker("serial") is not None
            or (jax and item.get_closest_marker("jax_smoke") is None)
            or (not tools and item.path.is_relative_to(REPO / "tests" / "tools"))
            or _is_float32_outside_the_fast_tier(item, jax)
        )
        (deselected if out else kept).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
    if jax:
        # The parity case first: without JAX's compilation cache it takes about 10 s, so xdist
        # workers run the conformance suite beside it rather than after it.
        items.sort(key=lambda item: not item.path.is_relative_to(REPO / "tests" / "parity"))


def _cpu_seconds() -> float:
    """CPU time of this process and of the subprocesses it has waited for."""
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:
    """Refuse to run a ``serial`` test in an xdist worker, where other tests run beside it; time
    the set-up for the budget."""
    if item.get_closest_marker("serial") and "PYTEST_XDIST_WORKER" in os.environ:
        pytest.fail(
            f"{item.nodeid} is marked serial and must run alone: run the parallel pass with "
            '-m "not serial" and this test with -p no:xdist -m serial, as `make test` does.',
            pytrace=False,
        )
    start = _cpu_seconds()
    try:
        return (yield)
    finally:
        item.stash[_CPU] = _cpu_seconds() - start


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    start = _cpu_seconds()
    try:
        return (yield)
    finally:
        item.stash[_CPU] = item.stash.get(_CPU, 0.0) + _cpu_seconds() - start


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """In the fast tier, fail a passing test over its budget: ``FAST_TEST_BUDGET`` seconds of CPU
    time (set-up and call, its subprocesses included), so that the load of the other workers does
    not count, or ``WALL_LIMIT_FACTOR`` times as much wall time, for a test that waits."""
    report = yield
    if report.when == "call":
        # On the call report for the summary line; on the item for --junitxml, which reads it.
        cpu_seconds = ("cpu_seconds", item.stash.get(_CPU, 0.0))
        report.user_properties.append(cpu_seconds)
        item.user_properties.append(cpu_seconds)
    if _tier(item.config) != "fast" or item.get_closest_marker("slow"):
        return report
    if report.when == "setup":
        item.stash[_SETUP_DURATION] = report.duration
    elif report.when == "call" and report.passed:
        cpu = item.stash.get(_CPU, 0.0)
        wall = item.stash.get(_SETUP_DURATION, 0.0) + report.duration
        budget = FAST_TEST_BUDGET
        if backend_name() == "jax":
            limit = math.inf  # reported, not enforced: see FAST_TEST_BUDGET
        else:
            limit = budget if os.environ.get("CI") else LOCAL_BUDGET_FACTOR * budget
        if budget < cpu <= limit:
            report.user_properties.append(("over_budget", cpu))
        if cpu > limit or wall > WALL_LIMIT_FACTOR * limit:
            report.outcome = "failed"
            report.longrepr = (
                f"{item.nodeid} took {cpu:.2f} s of CPU time and {wall:.2f} s of wall time: over "
                f"the budget of a fast-tier test, {budget:.0f} s of CPU time (here {limit:.0f} s) "
                f"and {WALL_LIMIT_FACTOR * limit:.0f} s of wall time (Part I, §5.4). Make it "
                "faster, on the smallest data that exercises the behaviour, or mark it slow with "
                f"a sentence starting with {SLOW_REASON!r} in its docstring that says why; it then "
                "runs in the full tier only (make check-full)."
            )
    return report


class _Totals:
    """The time of the run's tests, for its summary line."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.tests = 0
        self.wall = 0.0
        self.cpu = 0.0
        self.slowest: tuple[float, str] = (0.0, "")
        self.over_budget: list[str] = []

    def add(self, report: pytest.TestReport) -> None:
        self.wall += report.duration
        if report.when == "call":
            self.tests += 1
            cpu = 0.0
            for key, value in report.user_properties:
                if key == "cpu_seconds" and isinstance(value, float):
                    cpu += value
                if key == "over_budget" and isinstance(value, float):
                    self.over_budget.append(f"{report.nodeid} ({value:.2f} s)")
            self.cpu += cpu
            self.slowest = max(self.slowest, (cpu, report.nodeid))


_TOTALS = _Totals()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _TOTALS.add(report)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """One line: the tier, the number of tests, their total time and the run's wall time."""
    config = terminalreporter.config
    if hasattr(config, "workerinput") or config.option.collectonly:
        return
    wall = time.perf_counter() - _TOTALS.start
    slowest, nodeid = _TOTALS.slowest
    terminalreporter.write_line(
        f"tier {_tier(config)} on {backend_name()}: {_TOTALS.tests} tests, "
        f"{_TOTALS.cpu:.1f} s of CPU time in calls, {_TOTALS.wall:.1f} s of test wall time, "
        f"{wall:.1f} s for the run; slowest {slowest:.2f} s of CPU ({nodeid or 'none'})"
    )
    if _TOTALS.over_budget:
        terminalreporter.write_line(
            f"Over the budget of a fast-tier test ({FAST_TEST_BUDGET:.0f} s of CPU; Part I, §5.4), "
            "which CI enforces on PyTorch; here reported only: "
            f"{', '.join(sorted(_TOTALS.over_budget))}",
            yellow=True,
        )


@pytest.fixture
def dtype() -> str:
    """The floating dtype of the test: parametrised by ``pytest_generate_tests``."""
    raise pytest.UsageError("The dtype fixture is parametrised by tests/conftest.py.")


@pytest.fixture
def backend() -> Backend:
    """The backend of this run (``ESTEL_TEST_BACKEND``)."""
    return make_backend(backend_name())
