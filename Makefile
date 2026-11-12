# Tier 0 (Part I, §6.1): everything here runs on CPU, locally and in CI.
# `make check` is what CI runs on every push and pull request.

RUN := uv run --locked

.PHONY: check lint types lint-imports layers manifest mirrors maths-ids test test-float32 \
        test-float64 version-impact

check: lint types lint-imports layers manifest mirrors maths-ids test
	@echo "make check: tier 0 passed (lint, types, import contracts, AST layer rules, manifest, mirrors, maths IDs, tests float32 + float64)"

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

# --allow-missing-mirror until task T0.3 generates .github/copilot-instructions.md.
mirrors:
	$(RUN) python tools/check_agent_mirrors.py --allow-missing-mirror

# --allow-unimplemented until task T2.6 writes the oracle for planar data.
maths-ids:
	$(RUN) python tools/check_maths_ids.py --allow-unimplemented

test: test-float32 test-float64

test-float32:
	ESTEL_TEST_DTYPE=float32 $(RUN) pytest

test-float64:
	ESTEL_TEST_DTYPE=float64 $(RUN) pytest

# Runs in CI on pull requests; locally: make version-impact BASE=origin/main PR=12
BASE ?= origin/main
version-impact:
	$(RUN) python tools/version_impact.py --base $(BASE) $(if $(PR),--pr $(PR)) $(if $(LABELS),--labels $(LABELS))
