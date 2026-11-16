# Tier 0 (Part I, §6.1): everything here runs on CPU, locally and in CI.
# Two tiers (Part I, §5.4; tests/conftest.py). `make check` runs the fast tier, as CI does on a
# pull request: the check tools, the tests not marked slow in float64 on PyTorch and those marked
# dtype in float32 too, and the tests marked jax_smoke on JAX. `make check-full` runs everything,
# as CI does on main, nightly and on a pull request labelled full-ci: both backends, both dtypes,
# slow tests included.

RUN := uv run --locked

.PHONY: check check-full check-tools lint types lint-imports layers manifest mirrors mirrors-fix \
        maths-ids workflows test test-fast test-torch test-jax test-fast-torch test-fast-jax \
        version-impact parity-image parity parity-fixtures container-runtime

# Backends the tests run on (Part I, §3.1). CI runs one per job: make test-fast TEST_BACKENDS=torch
TEST_BACKENDS ?= torch jax

# Wall-time budget of `make check`, in seconds (Part I, §5.4). CI is the reference machine and
# fails a fast-tier job above 120 s (.github/workflows/tier0.yml). Machines differ, so here
# `make check` warns above the budget and fails above twice it, as the per-test budget does.
CHECK_BUDGET ?= 60
CHECK_BUDGET_ADVICE := The summary line of each pytest run names its slowest test; pytest --tier fast --durations=20 lists more. Make them faster or mark them slow; a PR that slows the fast tier by more than 10 % says why in its description, with the before and after times.

check:
	@start=$$(date +%s); \
	$(MAKE) --no-print-directory check-tools test-fast || exit $$?; \
	elapsed=$$(( $$(date +%s) - start )); \
	if [ $$elapsed -gt $$(( 2 * $(CHECK_BUDGET) )) ]; then \
		echo "make check: $$elapsed s, over twice its budget of $(CHECK_BUDGET) s (Part I, §5.4). $(CHECK_BUDGET_ADVICE)"; \
		exit 1; \
	elif [ $$elapsed -gt $(CHECK_BUDGET) ]; then \
		echo "make check: WARNING: $$elapsed s, over its budget of $(CHECK_BUDGET) s (Part I, §5.4). $(CHECK_BUDGET_ADVICE)"; \
	fi; \
	echo "make check: fast tier passed in $$elapsed s, budget $(CHECK_BUDGET) s (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, workflows, fast-tier tests on $(TEST_BACKENDS))"

check-full: check-tools test
	@echo "make check-full: full tier passed (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, workflows, every test in float32 and float64 on $(TEST_BACKENDS))"

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
test-fast: $(addprefix test-fast-,$(TEST_BACKENDS))

# pytest-xdist workers of the parallel pass (task T0.7): a number, or auto (one per CPU). On Apple
# silicon, one per performance core: workers on the efficiency cores slow the run down.
PYTEST_WORKERS ?= $(or $(shell sysctl -n hw.perflevel0.physicalcpu 2>/dev/null),auto)
# One dtype only, in the full tier (CI splits the JAX run in two jobs): make test-jax TEST_DTYPE=float32
TEST_DTYPE ?=
DTYPE_ENV := $(if $(TEST_DTYPE),ESTEL_TEST_DTYPE=$(TEST_DTYPE))

# One run per backend, float32 and float64 in the same process (tests/conftest.py); the JAX run
# switches JAX's 64-bit mode on at start. Each run has two passes: the tests in parallel, then the
# tests marked serial (fresh processes, process-wide settings) alone, in one process. pytest exits
# with 5 when a pass selects no test, as the serial pass of the fast tier may: that is a success.
test-torch test-jax test-fast-torch test-fast-jax: TIER = $(if $(findstring fast,$@),fast,full)
test-torch test-jax test-fast-torch test-fast-jax: BACKEND = $(lastword $(subst -, ,$@))
test-torch test-jax test-fast-torch test-fast-jax:
	ESTEL_TEST_BACKEND=$(BACKEND) $(DTYPE_ENV) $(RUN) pytest -n $(PYTEST_WORKERS) --tier $(TIER) -m "not serial"
	ESTEL_TEST_BACKEND=$(BACKEND) $(DTYPE_ENV) $(RUN) pytest -p no:xdist --tier $(TIER) -m serial; \
	status=$$?; [ $$status -eq 0 ] || [ $$status -eq 5 ]

# STL parity harness (Part II, §14; tools/stl_parity/README.md). Step 1 only, not in tier 0.
CONTAINER ?= docker
PARITY_IMAGE ?= estel-stl-parity

parity-image: container-runtime
	$(CONTAINER) build --platform linux/amd64 -t $(PARITY_IMAGE) tools/stl_parity

parity: container-runtime
	$(CONTAINER) run --rm --platform linux/amd64 $(PARITY_IMAGE)

# Parity fixtures (Part II, §14.2; task T0.8): runs the cases of tools/stl_parity/cases/ in the
# image and writes tests/parity/fixtures/ (protected: maintainers only). The cases are mounted,
# not copied, so adding one needs no new image. One case: make parity-fixtures CASES=<name>
CASES ?=
PARITY_FIXTURES := tests/parity/fixtures

parity-fixtures: container-runtime
	@image_id=$$($(CONTAINER) image inspect --format '{{.Id}}' $(PARITY_IMAGE)) || { \
		echo "No image '$(PARITY_IMAGE)': build it first with make parity-image."; exit 1; }; \
	mkdir -p $(PARITY_FIXTURES) && \
	$(CONTAINER) run --rm --platform linux/amd64 --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-e PARITY_IMAGE_ID="$$image_id" \
		-v "$(CURDIR)/tools/stl_parity:/mnt/stl_parity:ro" \
		-v "$(CURDIR)/$(PARITY_FIXTURES):/mnt/fixtures" \
		$(PARITY_IMAGE) /bin/sh /mnt/stl_parity/run_fixtures.sh $(CASES)

container-runtime:
	@command -v $(CONTAINER) >/dev/null 2>&1 || { \
		echo "'$(CONTAINER)' not found: install Docker, or run with CONTAINER=podman."; exit 1; }

# Runs in CI on pull requests; locally: make version-impact BASE=origin/main PR=12
BASE ?= origin/main
version-impact:
	$(RUN) python tools/version_impact.py --base $(BASE) $(if $(PR),--pr $(PR)) $(if $(LABELS),--labels $(LABELS))
