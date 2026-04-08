"""Shared CLI plumbing for the eval harness scripts.

The three ``run_smoke.py`` / ``run_eval.py`` / ``run_eval_full.py``
modules each call a different ``EvalRunner`` method, but they all need
the same supporting machinery: build a :class:`PipelineAdapter`,
generate the JSON+HTML report, print a human-readable summary, and
turn the overall pass-rate into a CI exit code. That logic lives here
so the three entry-point scripts stay tiny and uniform.

The shared exit-code rule (``score > 0.50``) is the CI gate from
ARCHITECTURE.md §31.4.7 — anything below 50 % overall fails the build.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from backtesting.evals.framework import EvalReport
    from backtesting.evals.pipeline_adapter import PipelineAdapter


# Load .env at import time so the eval harness scripts (run_smoke.py,
# run_eval.py, run_eval_full.py) pick up ANTHROPIC_API_KEY and other
# secrets without requiring a manually exported shell environment.
load_dotenv()


# CI gate threshold (overall pass-rate). >0.50 passes, ≤0.50 fails.
PASS_THRESHOLD = 0.50


def configure_logging(verbose: bool = False) -> None:
    """Best-effort root logger setup so harness output is readable."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_adapter(mock: bool, model: str | None = None) -> "PipelineAdapter":
    """Construct a PipelineAdapter for the chosen mode.

    Imported lazily so ``--help`` runs of the CLI don't pull in the
    anthropic SDK or fail when ``ANTHROPIC_API_KEY`` is unset.
    """
    from backtesting.evals.pipeline_adapter import PipelineAdapter

    if mock:
        return PipelineAdapter(mode="mock")
    kwargs: dict = {"mode": "live"}
    if model:
        kwargs["model"] = model
    return PipelineAdapter(**kwargs)


def write_report(report: "EvalReport") -> tuple[str, str]:
    """Persist the JSON + HTML report and return their paths."""
    from backtesting.evals.reporter import generate_eval_report

    return generate_eval_report(report)


def print_summary(report: "EvalReport", *, header: str) -> None:
    """Print a human-readable summary of an EvalReport to stdout."""
    pct = report.overall_pass_rate * 100
    passed = sum(1 for r in report.scenario_results if r.pass_fail == "PASS")
    failed = report.total_scenarios - passed

    print()
    print("=" * 72)
    print(header)
    print("=" * 72)
    print(f"  Total scenarios:    {report.total_scenarios}")
    print(f"  Runs per scenario:  {report.runs_per_scenario}")
    print(f"  Overall pass rate:  {pct:5.1f}%  ({passed} pass / {failed} fail)")
    print(f"  Consistency stdev:  {report.consistency_avg_stdev:.4f}")
    print(f"  Duration:           {report.duration_seconds:.1f}s")
    print(f"  Model:              {report.model_id}")
    print()
    print("  By category:")
    if not report.by_category:
        print("    (no scenarios graded)")
    for stat in report.by_category:
        cat_pct = stat.pass_rate * 100
        print(
            f"    {stat.category:<22} "
            f"{stat.passed:>3} / {stat.total:<3}  ({cat_pct:5.1f}%)"
        )
    print()

    if report.top_failures:
        print("  Top failures:")
        for r in report.top_failures[:5]:
            reasons = "; ".join(r.failure_reasons) or "(no reason)"
            print(f"    [{r.category}] {r.scenario_id} — {reasons}")
        print()


def exit_code_for(report: "EvalReport") -> int:
    """0 if overall pass rate beats the gate, 1 otherwise."""
    return 0 if report.overall_pass_rate > PASS_THRESHOLD else 1


def print_report_paths(json_path: str, html_path: str) -> None:
    print(f"  JSON report:  {json_path}")
    print(f"  HTML report:  {html_path}")
    print()
