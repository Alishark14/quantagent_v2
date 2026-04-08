"""Tests for backtesting.evals.reporter — JSON + HTML eval reports."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from backtesting.evals.framework import (
    CategoryStats,
    EvalReport,
    ScenarioResult,
)
from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.reporter import generate_eval_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(direction: str = "LONG", conviction: float = 0.75) -> EvalOutput:
    return EvalOutput(
        direction=direction,
        conviction=conviction,
        sl_price=99.0,
        tp1_price=102.0,
        tp2_price=104.0,
        position_size_pct=0.1,
        reasoning="bull flag",
        latency_ms=2300.0,
        model_id="claude-sonnet-4-6",
    )


def _scenario_result(
    sid: str,
    category: str,
    pass_fail: str,
    reasons: list[str] | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        scenario_name=f"Scenario {sid}",
        category=category,
        runs=[_make_output()],
        direction_match=pass_fail == "PASS",
        conviction_in_range=pass_fail == "PASS",
        action_match=pass_fail == "PASS",
        consistency_stdev=0.0,
        pass_fail=pass_fail,
        failure_reasons=reasons or [],
    )


def _make_report(
    overall: float = 0.6,
    cats: list[tuple[str, int, int]] | None = None,
    fail_results: int = 0,
) -> EvalReport:
    cats = cats or [("clear_setups", 5, 4), ("clear_avoids", 5, 2)]
    by_category = [
        CategoryStats(category=c, total=t, passed=p) for c, t, p in cats
    ]
    results: list[ScenarioResult] = []
    for c, total, passed in cats:
        for i in range(passed):
            results.append(_scenario_result(f"{c}_pass_{i}", c, "PASS"))
        for i in range(total - passed):
            results.append(
                _scenario_result(
                    f"{c}_fail_{i}",
                    c,
                    "FAIL",
                    reasons=[f"action SHORT != expected LONG"],
                )
            )
    failures = [r for r in results if r.pass_fail == "FAIL"][:fail_results or 10]
    return EvalReport(
        timestamp="2026-04-08T00:00:00+00:00",
        total_scenarios=len(results),
        runs_per_scenario=3,
        overall_pass_rate=overall,
        by_category=by_category,
        consistency_avg_stdev=0.012,
        scenario_results=results,
        top_failures=failures,
        model_id="2026.04.2.6.0-alpha.1",
        prompt_versions={"indicator_agent": "3.2", "pattern_agent": "2.1"},
        duration_seconds=42.5,
    )


# ---------------------------------------------------------------------------
# Files created with the right names
# ---------------------------------------------------------------------------


def test_generate_eval_report_creates_both_files(tmp_path):
    report = _make_report()
    json_path, html_path = generate_eval_report(
        report,
        output_dir=tmp_path,
        run_date=date(2026, 4, 8),
    )
    assert Path(json_path).exists()
    assert Path(html_path).exists()
    assert Path(json_path).name == "2026-04-08_eval.json"
    assert Path(html_path).name == "2026-04-08_eval.html"


def test_generate_eval_report_default_run_date(tmp_path):
    """Omitting `run_date` falls back to today (filename pattern check)."""
    import re

    json_path, html_path = generate_eval_report(_make_report(), output_dir=tmp_path)
    assert re.match(r"\d{4}-\d{2}-\d{2}_eval\.json$", Path(json_path).name)
    assert re.match(r"\d{4}-\d{2}-\d{2}_eval\.html$", Path(html_path).name)


# ---------------------------------------------------------------------------
# JSON shape
# ---------------------------------------------------------------------------


def test_json_report_contents(tmp_path):
    report = _make_report()
    json_path, _ = generate_eval_report(
        report, output_dir=tmp_path, run_date=date(2026, 4, 8)
    )
    payload = json.loads(Path(json_path).read_text())

    for key in (
        "timestamp",
        "total_scenarios",
        "runs_per_scenario",
        "overall_pass_rate",
        "by_category",
        "consistency_avg_stdev",
        "scenario_results",
        "top_failures",
        "model_id",
        "prompt_versions",
        "duration_seconds",
    ):
        assert key in payload

    assert payload["total_scenarios"] == 10
    assert payload["overall_pass_rate"] == 0.6
    assert len(payload["by_category"]) == 2
    assert payload["model_id"] == "2026.04.2.6.0-alpha.1"


def test_json_report_with_previous_includes_regressions(tmp_path):
    """If previous_report has a higher pass rate in some category, that
    category appears in the regressions list."""
    previous = _make_report(cats=[("clear_setups", 5, 5), ("clear_avoids", 5, 5)])
    current = _make_report(cats=[("clear_setups", 5, 1), ("clear_avoids", 5, 5)])
    json_path, _ = generate_eval_report(
        current,
        output_dir=tmp_path,
        run_date=date(2026, 4, 8),
        previous_report=previous,
    )
    payload = json.loads(Path(json_path).read_text())
    assert "regressions" in payload
    assert any(r["category"] == "clear_setups" for r in payload["regressions"])
    # The unchanged category is NOT in the regressions list
    assert not any(r["category"] == "clear_avoids" for r in payload["regressions"])


def test_no_regressions_when_pass_rate_unchanged(tmp_path):
    same = _make_report()
    json_path, _ = generate_eval_report(
        same,
        output_dir=tmp_path,
        run_date=date(2026, 4, 8),
        previous_report=same,
    )
    payload = json.loads(Path(json_path).read_text())
    assert payload.get("regressions", []) == []


# ---------------------------------------------------------------------------
# HTML shape
# ---------------------------------------------------------------------------


def test_html_report_contains_expected_sections(tmp_path):
    report = _make_report(overall=0.65, fail_results=3)
    _, html_path = generate_eval_report(
        report, output_dir=tmp_path, run_date=date(2026, 4, 8)
    )
    text = Path(html_path).read_text()
    assert text.startswith("<!doctype html>")
    assert "QuantAgent Eval Report" in text
    assert "Overall Score" in text
    assert "By Category" in text
    assert "Top Failures" in text
    # Pass rate hero
    assert "65.0%" in text
    # Each category appears
    assert "clear_setups" in text
    assert "clear_avoids" in text


def test_html_report_escapes_unsafe_strings(tmp_path):
    """A failure reason with HTML injection-y values must be escaped."""
    nasty_result = _scenario_result(
        "x",
        "clear_setups",
        "FAIL",
        reasons=["<script>alert(1)</script>"],
    )
    report = EvalReport(
        timestamp="2026-04-08T00:00:00+00:00",
        total_scenarios=1,
        runs_per_scenario=1,
        overall_pass_rate=0.0,
        by_category=[CategoryStats("clear_setups", 1, 0)],
        consistency_avg_stdev=0.0,
        scenario_results=[nasty_result],
        top_failures=[nasty_result],
        model_id="test",
        prompt_versions={},
        duration_seconds=1.0,
    )
    _, html_path = generate_eval_report(
        report, output_dir=tmp_path, run_date=date(2026, 4, 8)
    )
    text = Path(html_path).read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_html_report_renders_regressions_when_provided(tmp_path):
    previous = _make_report(cats=[("clear_setups", 5, 5)])
    current = _make_report(cats=[("clear_setups", 5, 1)])
    _, html_path = generate_eval_report(
        current,
        output_dir=tmp_path,
        run_date=date(2026, 4, 8),
        previous_report=previous,
    )
    text = Path(html_path).read_text()
    assert "Regressions" in text
    # The drop is from 100% to 20% = -80pp; check for the magnitude
    assert "-80" in text or "-80.0" in text


def test_html_report_no_failures_section_message(tmp_path):
    report = _make_report(cats=[("clear_setups", 5, 5)], overall=1.0)
    _, html_path = generate_eval_report(
        report, output_dir=tmp_path, run_date=date(2026, 4, 8)
    )
    text = Path(html_path).read_text()
    assert "No failures" in text
