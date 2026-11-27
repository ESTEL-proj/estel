# Tests, reference values, tolerances

Summary of [Part I, §5 — Tests, reference values, tolerances](../transfer/part1.md#5-tests-reference-values-tolerances). The transfer document is authoritative.

## Test families ([§5.1](../transfer/part1.md#51-test-families))

| Family | What |
| --- | --- |
| Backend conformance | each L0 primitive vs NumPy float64, values and gradients |
| Spec parity | ESTEL vs the `_spec` oracle, float64, small inputs, every combination (`make test-spec`) |
| Reference values | seeded inputs and stored outputs in `tests/reference/*.npz`, including normalisation references; every backend and dtype |
| Invariances | time reversal, scaling, constant shift, batch/channel/block permutations, all-valid mask = no mask, single-block tensor product = the Cartesian geometry of its block; rotations come with the joint design of the filters (Part III, §20.10) |
| Differentiation | the table of [differentiation.md](differentiation.md), float64 |
| Synthesis | step 1: the API (runs, checkpoints, resumes), and one comparison with STL over the first iterations (T6.1, full tier) |
| maths ↔ spec ↔ tests | every `M-` ID has a `_spec` function and a test, and vice versa (`tools/check_maths_ids.py`) |
| API | manifest up to date, public vs internal, actionable error messages |
| Reserved interfaces | each step-2 interface exercised through its trivial implementation |
| STL parity | ESTEL vs STL in float64, step 1 only: STL's results are fixtures of `tests/parity/fixtures/` (protected), made in the container by `make parity-fixtures`; the tests compare with them through `tests/parity/_compare.py` |

Every test runs in float32 and float64 (the `dtype` fixture, parametrised by `tests/conftest.py`; `ESTEL_TEST_DTYPE` restricts a full run to one), on PyTorch and on the experimental JAX backend (`ESTEL_TEST_BACKEND=torch|jax`, the `backend` fixture; `tests/_backends.py`), one backend per process. `tests/backend/test_torch.py` and `test_jax.py` run only on their own backend.

## Speed and tiers ([§5.4](../transfer/part1.md#54-test-speed-the-fast-and-the-full-tier))

Tests are as short as possible (golden rule 13; review rule C11). Two tiers, `pytest --tier fast|full`:

- **fast** (`make check`, under 60 s; CI on a PR, under 120 s per job): the tests not marked `slow` or `serial`; on PyTorch in float64, plus float32 for the tests marked `dtype`; on JAX only the tests marked `jax_smoke`, in float64. `tests/tools/` runs only when the change touches `tools/`, `tests/tools/` or `pyproject.toml` (compared with `ESTEL_TEST_BASE`, `origin/main` by default; `ESTEL_TOOLS_CHANGED=1` forces it). Each backend runs in one process and `make check` runs the check tools and both backends side by side: a worker's start-up (about 6 s for PyTorch) costs more than the tests it would share.
- **full** (`make check-full`, a bare `pytest`; CI on `main`, nightly, and on a PR labelled `full-ci`): everything, both backends, both dtypes.

Orientations: in 2D, including the 2D blocks of tensor products, tests, benchmark cases and parity cases use L ≤ 4 (golden rule 13). `tools/check_orientations.py`, part of `make check`, fails on a larger L in `tests/`, `bench/` or `tools/stl_parity/cases/` unless its line carries `# allow-large-L: #<issue>`, the issue where a human approved it. A local run with L > 4 that an agent judges essential to check a feature stays outside the repository and is reported to the human.

Markers:

- `slow`: over 1 s even on the smallest data that exercises the behaviour (large configurations, second-order gradients through the loop, fresh processes). The docstring of the test, its class or its module says why in a sentence starting with `Slow:`; collection fails without it.
- `dtype`: also run in float32 in the fast tier. Only for tests of dtype propagation through a public function, representative parity cases and precision tests; a test of logic needs one dtype.
- `jax_smoke`: run on JAX in the fast tier: the L0 conformance suite and one parity case, of the scattering statistics (the whole loop). On a cold CI runner JAX compiles every operation, and each parity case costs about 10 s.

In the fast tier a test that takes more than 1 s of CPU time (its subprocesses included) fails in CI; locally it fails at 2 s, and the run lists those between 1 and 2 s, which CI would fail. On JAX the times are reported, not enforced (its eager mode compiles one executable per operation on first use); the same tests meet the budget on PyTorch. Each pytest run ends with a line giving its tier, its number of tests, their CPU time and the slowest; `pytest --tier fast --durations=20` lists more.

The tests run in parallel, in `pytest-xdist` workers (`make test`; `PYTEST_WORKERS`, `auto` by default), then the tests marked `serial` run alone, in one process. Mark `serial` a test that starts fresh processes or switches a process-wide setting (such as `Backend.deterministic`); `tests/conftest.py` refuses to run one in a worker. A change confined to the no-test scope runs only the check tools (`make check-tools`; `tools/ci_scope.py`): `docs/`, `.github/` and `changes/`; `.gitignore`, `LICENSE`, `NOTICE` and `README.md` at the root; every `AGENTS.md` and `CLAUDE.md`, at any depth. Everything else is a code change, including `Makefile`, `pyproject.toml`, `uv.lock`, `.python-version` and `api/manifest.json`. In CI both required jobs still report success. Locally, when `python tools/ci_scope.py --base origin/main` prints `tests=false`, run `make check-tools` instead of `make check` before opening the PR. A no-test file that a test reads is a test input, and changing it runs the whole suite: a path written in `tests/` as one string (such as `"docs/maths.md"`) or joined onto the repository root (`REPO / "src" / "estel" / "backend" / "AGENTS.md"`). Fake files that tests write in a temporary repository (dictionary keys given to `write`, paths joined onto `tmp_path`) are not test inputs.

## Reference values

- Produced by `tools/make_reference.py` from `_spec` in float64, **never from ESTEL**, and only by a human.
- If a reference test fails: do not regenerate, do not loosen tolerances. Report the failing coefficients in an issue.
- A change that alters a reference value needs a `STATISTICS_REFERENCE` bump and a human decision.

## Tolerances ([§5.2](../transfer/part1.md#52-tolerances))

`tests/tolerances.toml` (protected) has one entry per dtype, device class and family. Comparisons use **normalised** coefficients (divided by the reference S2 scale). Starting values: float64 CPU rtol 1e-10 / atol 1e-12; float64 GPU 1e-9 / 1e-11; float32 1e-4 / 1e-6. A test with an explicit tolerance must cite this table.

## Protection ([§5.3](../transfer/part1.md#53-protection-of-reference-values))

- CODEOWNERS protects reference data and tolerances, the transfer specification (`docs/transfer/**`), the oracle, `docs/maths.md` and `docs/maths_step2.md`, policy and API, context files, and infrastructure (full list in the root `AGENTS.md`).
- `tests/reference/MANIFEST.sha256` is checked in CI. From 1.0.0, changing a reference file needs a `STATISTICS_REFERENCE` bump with CODEOWNER approval, or CI fails; before 1.0.0 the change is only reported.
- CI fails if the number of tests drops, or if a new `skip`/`xfail` appears without the `test-change` label. Never weaken, skip or delete a test to make CI pass.
