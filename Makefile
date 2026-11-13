# Tier 0 (Part I, §6.1): everything here runs on CPU, locally and in CI.
# `make check` is what CI runs on every push and pull request.

RUN := uv run --locked

.PHONY: check check-tools lint types lint-imports layers manifest mirrors mirrors-fix maths-ids workflows test \
        test-torch test-jax version-impact parity-image parity container-runtime

# Backends the tests run on (Part I, §3.1). CI runs one per job: make check TEST_BACKENDS=torch
TEST_BACKENDS ?= torch jax

check: check-tools test
	@echo "make check: tier 0 passed (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, workflows, tests float32 + float64 on $(TEST_BACKENDS))"

# Tier 0 without the test suites. CI runs only this on a change confined to the no-test scope
# of tools/ci_scope.py (docs/, .github/, changes/, root .gitignore, LICENSE, NOTICE, README.md,
# every AGENTS.md and CLAUDE.md; task T0.7); run it instead of `make check` for such a change.
check-tools: lint types lint-imports layers manifest mirrors maths-ids workflows
	@echo "make check-tools: check tools passed (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, workflows)"

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

types:
	$(RUN) mypy

# Import contracts of Part I, §2.4 (pyproject.toml); custom contract types live in tools/.
lint-imports:
	PYTHONPATH=tools $(RUN) lint-imports --no-cache

# AST rules of Part I, §2.4: no class-name comparisons, no __class__ writes, no in-place
# array updates in L1/L2.
layers:
	$(RUN) python tools/check_layers.py

manifest:
	$(RUN) python tools/api_manifest.py --check

# Regenerate the Copilot mirror after changing AGENTS.md: make mirrors-fix
mirrors:
	$(RUN) python tools/check_agent_mirrors.py

mirrors-fix:
	$(RUN) python tools/check_agent_mirrors.py --fix

# --allow-unimplemented until task T2.6 writes the oracle for planar data.
maths-ids:
	$(RUN) python tools/check_maths_ids.py --allow-unimplemented

# Review rule E5: actions pinned to a commit SHA, checkouts without persisted credentials,
# a top-level permissions block in every workflow.
workflows:
	$(RUN) python tools/check_workflows.py

test: $(addprefix test-,$(TEST_BACKENDS))

# pytest-xdist workers of the parallel pass (task T0.7): a number, or auto (one per CPU).
PYTEST_WORKERS ?= auto

# The whole suite on one backend, in float32 then float64. The JAX run switches JAX's 64-bit
# mode on at start (tests/conftest.py). Each run has two passes: the tests in parallel, then
# the tests marked serial (fresh processes, process-wide settings) alone, in one process.
test-torch test-jax: test-%:
	ESTEL_TEST_BACKEND=$* ESTEL_TEST_DTYPE=float32 $(RUN) pytest -n $(PYTEST_WORKERS) -m "not serial"
	ESTEL_TEST_BACKEND=$* ESTEL_TEST_DTYPE=float32 $(RUN) pytest -p no:xdist -m serial
	ESTEL_TEST_BACKEND=$* ESTEL_TEST_DTYPE=float64 $(RUN) pytest -n $(PYTEST_WORKERS) -m "not serial"
	ESTEL_TEST_BACKEND=$* ESTEL_TEST_DTYPE=float64 $(RUN) pytest -p no:xdist -m serial

# STL parity harness (Part II, §14; tools/stl_parity/README.md). Step 1 only, not in tier 0.
CONTAINER ?= docker
PARITY_IMAGE ?= estel-stl-parity

parity-image: container-runtime
	$(CONTAINER) build --platform linux/amd64 -t $(PARITY_IMAGE) tools/stl_parity

parity: container-runtime
	$(CONTAINER) run --rm --platform linux/amd64 $(PARITY_IMAGE)

container-runtime:
	@command -v $(CONTAINER) >/dev/null 2>&1 || { \
		echo "'$(CONTAINER)' not found: install Docker, or run with CONTAINER=podman."; exit 1; }

# Runs in CI on pull requests; locally: make version-impact BASE=origin/main PR=12
BASE ?= origin/main
version-impact:
	$(RUN) python tools/version_impact.py --base $(BASE) $(if $(PR),--pr $(PR)) $(if $(LABELS),--labels $(LABELS))
