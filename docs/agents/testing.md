# Tests, reference values, tolerances

Summary of [Part I, §5 — Tests, reference values, tolerances](../transfer/part1.md#5-tests-reference-values-tolerances). The transfer document is authoritative.

## Test families ([§5.1](../transfer/part1.md#51-test-families))

| Family | What |
| --- | --- |
| Backend conformance | each L0 primitive vs NumPy float64, values and gradients |
| Spec parity | ESTEL vs the `_spec` oracle, float64, small inputs, every combination (`make test-spec`) |
| Reference values | seeded inputs and stored outputs in `tests/reference/*.npz`, including normalisation references; every backend and dtype |
| Invariances | rotations, time reversal, scaling, constant shift, batch/channel/block permutations, all-valid mask = no mask, single-block tensor product = plain geometry |
| Differentiation | the table of [differentiation.md](differentiation.md), float64 |
| Synthesis | step 1: the API only (runs, checkpoints, resumes) |
| maths ↔ spec ↔ tests | every `M-` ID has a `_spec` function and a test, and vice versa (`tools/check_maths_ids.py`) |
| API | manifest up to date, public vs internal, actionable error messages |
| Reserved interfaces | each step-2 interface exercised through its trivial implementation |
| STL parity | ESTEL vs STL in float64, step 1 only: STL's results are fixtures of `tests/parity/fixtures/` (protected), made in the container by `make parity-fixtures`; the tests compare with them through `tests/parity/_compare.py` |

Every test runs in float32 and float64 (`ESTEL_TEST_DTYPE`), on PyTorch and on the experimental JAX backend (`ESTEL_TEST_BACKEND=torch|jax`, the `backend` fixture; `tests/_backends.py`). `make check` runs every combination; tier 0 runs one backend per job. `tests/backend/test_torch.py` and `test_jax.py` run only on their own backend.

The tests run in parallel, in `pytest-xdist` workers (`make test`; `PYTEST_WORKERS`, `auto` by default), then the tests marked `serial` run alone, in one process. Mark `serial` a test that starts fresh processes or switches a process-wide setting (such as `Backend.deterministic`); `tests/conftest.py` refuses to run one in a worker. A change confined to the no-test scope runs only the check tools (`make check-tools`; `tools/ci_scope.py`): `docs/`, `.github/` and `changes/`; `.gitignore`, `LICENSE`, `NOTICE` and `README.md` at the root; every `AGENTS.md` and `CLAUDE.md`, at any depth. Everything else is a code change, including `Makefile`, `pyproject.toml`, `uv.lock`, `.python-version` and `api/manifest.json`. In CI both required jobs still report success. Locally, when `python tools/ci_scope.py --base origin/main` prints `tests=false`, run `make check-tools` instead of `make check` before opening the PR. A no-test file that a test reads is a test input, and changing it runs the whole suite: a path written in `tests/` as one string (such as `"docs/maths.md"`) or joined onto the repository root (`REPO / "src" / "estel" / "backend" / "AGENTS.md"`). Fake files that tests write in a temporary repository (dictionary keys given to `write`, paths joined onto `tmp_path`) are not test inputs.

## Reference values

- Produced by `tools/make_reference.py` from `_spec` in float64, **never from ESTEL**, and only by a human.
- If a reference test fails: do not regenerate, do not loosen tolerances. Report the failing coefficients in an issue.
- A change that alters a reference value needs a `STATISTICS_REFERENCE` bump and a human decision.

## Tolerances ([§5.2](../transfer/part1.md#52-tolerances))

`tests/tolerances.toml` (protected) has one entry per dtype, device class and family. Comparisons use **normalised** coefficients (divided by the reference S2 scale). Starting values: float64 CPU rtol 1e-10 / atol 1e-12; float64 GPU 1e-9 / 1e-11; float32 1e-4 / 1e-6. A test with an explicit tolerance must cite this table.

## Protection ([§5.3](../transfer/part1.md#53-protection-of-reference-values))

- CODEOWNERS protects reference data and tolerances, the transfer specification (`docs/transfer/**`), the oracle and `docs/maths.md`, policy and API, context files, and infrastructure (full list in the root `AGENTS.md`).
- `tests/reference/MANIFEST.sha256` is checked in CI. From 1.0.0, changing a reference file needs a `STATISTICS_REFERENCE` bump with CODEOWNER approval, or CI fails; before 1.0.0 the change is only reported.
- CI fails if the number of tests drops, or if a new `skip`/`xfail` appears without the `test-change` label. Never weaken, skip or delete a test to make CI pass.
