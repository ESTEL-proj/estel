# Tier 0 (Part I, §6.1): everything here runs on CPU, locally and in CI.
# Two tiers (Part I, §5.4; tests/conftest.py). `make check` runs the fast tier, as CI does on a
# pull request: the check tools, the tests not marked slow in float64 on PyTorch and those marked
# dtype in float32 too, and the tests marked jax_smoke on JAX. `make check-full` runs everything,
# as CI does on main, nightly and on a pull request labelled full-ci: both backends, both dtypes,
# slow tests included.

# Python writes its bytecode (__pycache__), whatever the session's environment says (task
# T2.17): the shells of coding agents set PYTHONDONTWRITEBYTECODE, and every run then compiled
# every module of the tests, of ESTEL and of its dependencies again, about 3 s of a warm
# `make check`. Python checks each bytecode file against the size and the time of its source.
unexport PYTHONDONTWRITEBYTECODE

# CI's JAX fast-tier job runs with RUN="uv run --no-sync": its environment leaves torch out.
RUN ?= uv run --locked

.PHONY: check check-full check-tools lint types lint-imports layers manifest mirrors mirrors-fix \
        maths-ids retired-names identifiers orientations workflows register register-check \
        test test-fast test-torch test-jax test-fast-torch test-fast-jax \
        test-spec version-impact clean-worktrees parity-image parity parity-fixtures container-runtime bench \
        docs prove

# Backends the tests run on (Part I, §3.1). CI runs one per job: make test-fast TEST_BACKENDS=torch
TEST_BACKENDS ?= torch jax

# Wall-time budget of `make check`, in seconds (Part I, §5.4). CI is the reference machine and
# fails a fast-tier job above 120 s (.github/workflows/tier0.yml). Machines differ, so here
# `make check` warns above the budget and fails above twice it, as the per-test budget does.
CHECK_BUDGET ?= 60
CHECK_BUDGET_ADVICE := The summary line of each pytest run names its slowest test; pytest --tier fast --durations=20 lists more. Make them faster or mark them slow; a PR that slows the fast tier by more than 10 % says why in its description, with the before and after times.

# The parts of `make check` run side by side, each in one process (the fast tier runs its tests
# without xdist workers, tests/conftest.py), and each part's output follows when it ends.
CHECK_PARTS = check-tools $(addprefix test-fast-,$(TEST_BACKENDS))
CHECK_LOGS := .pytest_cache/make-check

check:
	@start=$$(date +%s); mkdir -p $(CHECK_LOGS); pids=""; \
	for part in $(CHECK_PARTS); do \
		$(MAKE) --no-print-directory $$part > $(CHECK_LOGS)/$$part.log 2>&1 & pids="$$pids $$!"; \
	done; \
	failed=""; set -- $$pids; \
	for part in $(CHECK_PARTS); do \
		wait $$1 || failed="$$failed $$part"; shift; \
		echo "--- make $$part"; cat $(CHECK_LOGS)/$$part.log; \
	done; \
	if [ -n "$$failed" ]; then echo "make check: failed:$$failed"; exit 1; fi; \
	elapsed=$$(( $$(date +%s) - start )); \
	if [ $$elapsed -gt $$(( 2 * $(CHECK_BUDGET) )) ]; then \
		echo "make check: $$elapsed s, over twice its budget of $(CHECK_BUDGET) s (Part I, §5.4). $(CHECK_BUDGET_ADVICE)"; \
		exit 1; \
	elif [ $$elapsed -gt $(CHECK_BUDGET) ]; then \
		echo "make check: WARNING: $$elapsed s, over its budget of $(CHECK_BUDGET) s (Part I, §5.4). $(CHECK_BUDGET_ADVICE)"; \
	fi; \
	echo "make check: fast tier passed in $$elapsed s, budget $(CHECK_BUDGET) s (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, retired names, identifiers, orientations, workflows, register, fast-tier tests on $(TEST_BACKENDS))"

check-full: check-tools test
	@echo "make check-full: full tier passed (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, retired names, identifiers, orientations, workflows, register, every test in float32 and float64 on $(TEST_BACKENDS))"

# Tier 0 without the test suites. CI runs only this on a change confined to the no-test scope
# of tools/ci_scope.py (docs/, .github/, changes/, root .gitignore, LICENSE, NOTICE, README.md,
# every AGENTS.md and CLAUDE.md; task T0.7); run it instead of `make check` for such a change.
check-tools: lint types lint-imports layers manifest mirrors maths-ids retired-names identifiers orientations workflows register-check
	@echo "make check-tools: check tools passed (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, retired names, identifiers, orientations, workflows, register)"

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

# Caches shared by every worktree of this machine, outside the repository (task T2.17): a
# session works in a worktree of its own, whose first `make check` would otherwise start with
# no cache. mypy's cache and JAX's compilation cache (tests/conftest.py) are kept there. Neither
# can give a stale result: mypy checks each entry against the source file (size, time, then
# hash), its options and its version, and JAX names each entry by a hash of the program, of
# its version and of its flags. Another place: ESTEL_CACHE_DIR=<directory>; none shared (the
# caches stay in the worktree): ESTEL_CACHE_DIR= (empty), which is also what CI does, since its
# workflow restores and saves .mypy_cache.
ifeq ($(origin ESTEL_CACHE_DIR),undefined)
ESTEL_CACHE_DIR := $(if $(CI),,$(or $(XDG_CACHE_HOME),$(HOME)/.cache)/estel)
endif
export ESTEL_CACHE_DIR
MYPY_CACHE_DIR := $(if $(ESTEL_CACHE_DIR),$(ESTEL_CACHE_DIR)/mypy,.mypy_cache)

types:
	$(RUN) mypy --cache-dir "$(MYPY_CACHE_DIR)"

# Import contracts of Part I, §2.4 (pyproject.toml); custom contract types live in tools/.
lint-imports:
	PYTHONPATH=tools $(RUN) lint-imports --no-cache

# AST rules of Part I, §2.4: no class-name comparisons, no __class__ writes, no in-place
# array updates in L1/L2, no geometry hook used outside estel.wavelets and estel.data.
layers:
	$(RUN) python tools/check_layers.py

manifest:
	$(RUN) python tools/api_manifest.py --check

# Regenerate the Copilot mirror after changing AGENTS.md: make mirrors-fix
mirrors:
	$(RUN) python tools/check_agent_mirrors.py

mirrors-fix:
	$(RUN) python tools/check_agent_mirrors.py --fix

# Every M- ID of docs/maths.md has an oracle function and a test, except those the tool lists as
# not yet implemented (sections 3 to 5 and M-PS-2D), which their tasks remove from the list.
maths-ids:
	$(RUN) python tools/check_maths_ids.py

# Public names retired by a renaming task stay out of the specification, AGENTS.md and the
# docstrings of public names (Part IV, §22.5, R-A12); the list is in the tool. And the rule of names
# of Part I, §2.4.1, through the table tools/renamed_names.toml (phase P7, T7.2), strict since
# T7.12: a name that breaks it fails, as a row not done and the old name of a row done.
retired-names:
	$(RUN) python tools/check_retired_names.py --strict

# Every identifier of the lists of the specification that is cited (OP-n, LFn, Wn, Qn, Fn, Tn.m,
# R-…, Gn) has a row in its list or in the decision log, Part IV, §22 (issue #107).
identifiers:
	$(RUN) python tools/check_identifiers.py

# In 2D, tests, benchmark cases and parity cases use L ≤ 4 (golden rule 13; issue #133): no
# integer L above 4 in tests/, bench/ or tools/stl_parity/cases/ unless its line carries
# `# allow-large-L: #<issue>`.
orientations:
	$(RUN) python tools/check_orientations.py

# Review rule E5: actions pinned to a commit SHA, checkouts without persisted credentials,
# a top-level permissions block in every workflow.
workflows:
	$(RUN) python tools/check_workflows.py

# The register of later features (docs/register/): every card valid, and the view README.md up to date.
# Regenerate the view after changing a card: make register
register-check:
	$(RUN) python tools/register.py --check

register:
	$(RUN) python tools/register.py --write

test: $(addprefix test-,$(TEST_BACKENDS))
test-fast: $(addprefix test-fast-,$(TEST_BACKENDS))

# pytest-xdist workers of the parallel pass (task T0.7): a number, or auto, which
# tests/conftest.py resolves (pytest_xdist_auto_num_workers): none in the fast tier when it leaves
# the tools tests out, otherwise one per CPU (one per performance core on Apple silicon).
PYTEST_WORKERS ?= auto
# One dtype only, in the full tier (CI splits the JAX run in two jobs): make test-jax TEST_DTYPE=float32
TEST_DTYPE ?=
DTYPE_ENV := $(if $(TEST_DTYPE),ESTEL_TEST_DTYPE=$(TEST_DTYPE))
# One part of the full tier (CI splits the JAX float64 run in two jobs; tests/conftest.py,
# _shard): make test-jax TEST_DTYPE=float64 TEST_SHARD=1/2. The parts are disjoint and together
# hold every test; the tests marked serial run in every part.
TEST_SHARD ?=
DTYPE_ENV += $(if $(TEST_SHARD),ESTEL_TEST_SHARD=$(TEST_SHARD))

# One run per backend, float32 and float64 in the same process (tests/conftest.py); the JAX run
# switches JAX's 64-bit mode on at start. The tests run in parallel, in xdist workers; in the full
# tier, the tests marked serial (fresh processes, process-wide settings) then run alone, in one
# process. The fast tier leaves them out: a process of their own costs more than they do.
test-torch test-jax test-fast-torch test-fast-jax: TIER = $(if $(findstring fast,$@),fast,full)
test-torch test-jax test-fast-torch test-fast-jax: BACKEND = $(lastword $(subst -, ,$@))
test-torch test-jax test-fast-torch test-fast-jax:
	ESTEL_TEST_BACKEND=$(BACKEND) $(DTYPE_ENV) $(RUN) pytest -n $(PYTEST_WORKERS) --dist loadgroup --tier $(TIER) -m "not serial"
	$(if $(filter full,$(TIER)),ESTEL_TEST_BACKEND=$(BACKEND) $(DTYPE_ENV) $(RUN) pytest -p no:xdist --tier full -m serial)

# Spec parity (Part I, §5.1; task T2.6): ESTEL against the oracle estel._spec, in float64, on
# PyTorch. The fast tier, one small case per geometry and periodicity, is also part of
# `make check`; the full matrix, part of `make check-full`: make test-spec SPEC_TIER=full
SPEC_TIER ?= fast
test-spec:
	ESTEL_TEST_BACKEND=torch $(RUN) pytest tests/spec -n $(PYTEST_WORKERS) --dist loadgroup --tier $(SPEC_TIER) -m "not serial"

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

# Benchmark of step 1 (Part I, §6.3; task T2.11; bench/README.md): outside CI. Measures the cases
# of bench/cases.toml, STL and ESTEL in the parity container (build it first: make
# parity-image) and ESTEL natively, checks every ESTEL run against STL, prints a table and writes
# bench/results/<date>-<machine>.json. One case: make bench BENCH_CASE=planar-periodic; one
# environment: make bench BENCH_ENV=native (or container); other options: BENCH_ARGS.
BENCH_CASE ?=
BENCH_ENV ?=
BENCH_ARGS ?=

bench:
	$(RUN) python -m bench --container $(CONTAINER) --parity-image $(PARITY_IMAGE) \
		$(if $(BENCH_CASE),--case $(BENCH_CASE)) $(if $(BENCH_ENV),--env $(BENCH_ENV)) $(BENCH_ARGS)

container-runtime:
	@command -v $(CONTAINER) >/dev/null 2>&1 || { \
		echo "'$(CONTAINER)' not found: install Docker, or run with CONTAINER=podman."; exit 1; }

# The user documentation, docs/user/, as Read the Docs builds it (.readthedocs.yaml): every
# warning is an error. Not part of `make check` or `make check-tools`, which have a time budget.
# The quickstart runs at build time; its outputs are cached in $(DOCS_BUILD)/.jupyter_cache.
DOCS_BUILD := docs/user/_build

docs:
	$(RUN) --group docs sphinx-build -W --keep-going -b html docs/user $(DOCS_BUILD)/html

# The proof of phase P7 (task T7.2; tools/prove_unchanged.py): the outputs of a fixed set of cases
# on origin/main and on the working tree, bitwise equal, on CPU with one thread, both backends,
# float32 and float64; plan dumps, files and function bodies compared modulo the table. Minutes,
# outside CI. Exit 0: bitwise equal; 1: within the parity tolerance; 2: beyond it. One case:
# make prove PROVE_ARGS="--case 2d-periodic"; another base: PROVE_ARGS="--base <ref>"
PROVE_ARGS ?=

prove:
	$(RUN) python tools/prove_unchanged.py $(PROVE_ARGS)

# Runs in CI on pull requests; locally: make version-impact BASE=origin/main PR=12
BASE ?= origin/main
version-impact:
	$(RUN) python tools/version_impact.py --base $(BASE) $(if $(PR),--pr $(PR)) $(if $(LABELS),--labels $(LABELS))

# Remove the worktrees of merged pull requests and their local branches (issue #109): never the
# main checkout, the current worktree, or one with changes; never forced; needs gh logged in.
# One line per worktree, removed or kept and why. Standard library only: no environment needed.
clean-worktrees:
	python3 tools/clean_worktrees.py
