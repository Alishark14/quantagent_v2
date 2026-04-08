# QuantAgent v2 — make targets for tests, evals, and backtests.
#
# Eval targets shell out to a tiny harness that points the runner at
# the live Claude pipeline. They will fail with a helpful message if
# ANTHROPIC_API_KEY is not set.

.PHONY: help test test-unit test-integration test-smoke test-eval test-eval-full

help:
	@echo "QuantAgent v2 — common targets:"
	@echo ""
	@echo "  make test               Run the full pytest suite"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests only"
	@echo ""
	@echo "  make test-smoke         Eval smoke test (2 scenarios per category, 1 run each)"
	@echo "  make test-eval CATEGORY=clear_setups   Targeted eval (one category, 3 runs)"
	@echo "  make test-eval-full     Golden master eval (all scenarios, 3 runs each)"
	@echo ""

test:
	pytest -q

test-unit:
	pytest tests/unit -q

test-integration:
	pytest tests/integration -q

# ── Eval targets ─────────────────────────────────────────────────────────────
#
# CI runs `test-smoke --mock` so it costs $0 and needs no API key.
# `test-eval` and `test-eval-full` hit real Claude — they require
# ANTHROPIC_API_KEY in the environment.

test-smoke:
	@python -m backtesting.evals.run_smoke --mock

test-eval:
	@if [ -z "$(CATEGORY)" ]; then \
		echo "ERROR: CATEGORY is required. Example: make test-eval CATEGORY=clear_setups"; \
		exit 2; \
	fi
	@python -m backtesting.evals.run_eval --category $(CATEGORY) --runs 3

test-eval-full:
	@python -m backtesting.evals.run_eval_full --runs 3
