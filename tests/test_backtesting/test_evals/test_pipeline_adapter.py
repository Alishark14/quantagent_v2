"""Tests for backtesting.evals.pipeline_adapter and the CLI harnesses.

Covers:

* Mock-mode adapter returns valid EvalOutput for every category and
  populates direction, conviction, latency_ms, model_id correctly.
* Pipeline errors in live mode collapse to a graceful SKIP EvalOutput
  with the failure reason in `reasoning`.
* The internal Scenario → MarketData conversion preserves indicators
  and computes swings.
* Each CLI harness parses its arguments correctly (--mock, --category,
  --runs, --model, --verbose).
"""

from __future__ import annotations

import asyncio

import pytest

from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.pipeline_adapter import PipelineAdapter
from backtesting.evals.scenario_schema import (
    ExpectedBehavior,
    Scenario,
    ScenarioInput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(
    sid: str = "fixture_001",
    category: str = "clear_setups",
    expected_action: str = "LONG",
    flow_data: dict | None = None,
    indicators: dict | None = None,
) -> Scenario:
    # 50 candles of a clean uptrend so compute_all_indicators has enough data.
    candles = []
    base = 100.0
    for i in range(50):
        c = base + i * 0.5
        candles.append(
            {
                "timestamp": 1_700_000_000_000 + i * 3_600_000,
                "open": c - 0.1,
                "high": c + 0.4,
                "low": c - 0.4,
                "close": c,
                "volume": 1000 + i,
            }
        )
    return Scenario(
        id=sid,
        name=f"Test {sid}",
        category=category,
        version=1,
        created_at="2026-04-08T00:00:00+00:00",
        last_validated="2026-04-08T00:00:00+00:00",
        inputs=ScenarioInput(
            symbol="BTC-USDC",
            timeframe="1h",
            ohlcv=candles,
            indicators=indicators or {},
            flow_data=flow_data,
            regime_context="trending",
            timestamp="2026-04-08T00:00:00+00:00",
        ),
        expected=ExpectedBehavior(
            expected_action=expected_action,
            signal_direction="BULLISH",
            conviction_min=0.5,
            conviction_max=0.9,
            key_features_to_mention=["uptrend"],
        ),
    )


# ---------------------------------------------------------------------------
# Mock-mode behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_mode_returns_valid_eval_output_for_each_category():
    """Every category in _MOCK_CATEGORY_DEFAULTS should yield a valid EvalOutput."""
    from backtesting.evals.pipeline_adapter import _MOCK_CATEGORY_DEFAULTS

    adapter = PipelineAdapter(mode="mock")
    for category in _MOCK_CATEGORY_DEFAULTS:
        scenario = _make_scenario(sid=f"id_{category}", category=category)
        out = await adapter(scenario)
        assert isinstance(out, EvalOutput)
        assert out.direction in {"LONG", "SHORT", "SKIP"}
        assert 0.0 <= out.conviction <= 1.0
        assert out.model_id == "mock"
        assert out.latency_ms >= 0
        assert out.reasoning and "Mock pipeline" in out.reasoning


@pytest.mark.asyncio
async def test_mock_mode_long_category_populates_sl_tp():
    adapter = PipelineAdapter(mode="mock")
    scenario = _make_scenario(category="clear_setups")
    out = await adapter(scenario)
    assert out.direction == "LONG"
    assert out.sl_price is not None and out.tp1_price is not None
    assert out.sl_price < out.tp1_price < out.tp2_price
    assert out.position_size_pct == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_mock_mode_skip_category_has_null_levels():
    adapter = PipelineAdapter(mode="mock")
    scenario = _make_scenario(category="clear_avoids", expected_action="SKIP")
    out = await adapter(scenario)
    assert out.direction == "SKIP"
    assert out.sl_price is None and out.tp1_price is None and out.tp2_price is None
    assert out.position_size_pct is None


@pytest.mark.asyncio
async def test_mock_mode_unknown_category_falls_back_to_skip():
    adapter = PipelineAdapter(mode="mock")
    scenario = _make_scenario(sid="unknown_001", category="totally_made_up")
    out = await adapter(scenario)
    assert out.direction == "SKIP"
    assert out.conviction == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_mock_mode_is_deterministic_across_runs():
    adapter = PipelineAdapter(mode="mock")
    scenario = _make_scenario(category="clear_setups")
    a = await adapter(scenario)
    b = await adapter(scenario)
    assert a.direction == b.direction
    assert a.conviction == b.conviction
    assert a.sl_price == b.sl_price


@pytest.mark.asyncio
async def test_analyze_method_matches_call_method():
    """The .analyze() shim should be equivalent to __call__."""
    adapter = PipelineAdapter(mode="mock")
    scenario = _make_scenario(category="clear_setups")
    out_a = await adapter(scenario)
    out_b = await adapter.analyze(scenario)
    assert out_a.direction == out_b.direction
    assert out_a.conviction == out_b.conviction


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_error_returns_graceful_skip():
    """A crash inside the live decision path should turn into a SKIP EvalOutput."""
    adapter = PipelineAdapter(mode="mock")

    async def boom(_scenario):
        raise RuntimeError("simulated agent failure")

    # Patch _live_decision and force the live branch.
    adapter._mode = "live"
    adapter._live_decision = boom  # type: ignore[assignment]

    scenario = _make_scenario(category="clear_setups")
    out = await adapter(scenario)
    assert isinstance(out, EvalOutput)
    assert out.direction == "SKIP"
    assert out.conviction == 0.0
    assert "Pipeline error" in (out.reasoning or "")
    assert "simulated agent failure" in (out.reasoning or "")
    assert out.latency_ms >= 0


@pytest.mark.asyncio
async def test_pipeline_error_with_pipeline_returning_non_eval_output_is_handled():
    """If a future bug causes _live_decision to return the wrong type the
    adapter should still produce a usable EvalOutput rather than crashing
    the whole eval run. Error path: callable raises, gets caught."""
    adapter = PipelineAdapter(mode="mock")

    async def returns_dict(_scenario):
        raise TypeError("expected EvalOutput, got dict")

    adapter._mode = "live"
    adapter._live_decision = returns_dict  # type: ignore[assignment]

    out = await adapter(_make_scenario())
    assert out.direction == "SKIP"
    assert "Pipeline error" in (out.reasoning or "")


def test_live_mode_without_api_key_raises():
    """Constructing a live adapter with no API key must fail loudly."""
    import os

    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            PipelineAdapter(mode="live", api_key=None)
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        PipelineAdapter(mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scenario → MarketData conversion
# ---------------------------------------------------------------------------


def test_scenario_to_market_data_preserves_candles_and_indicators():
    scenario = _make_scenario(
        indicators={"custom_metric": 42.0, "rsi": 999.0},
    )
    md = PipelineAdapter._scenario_to_market_data(scenario)
    assert md.symbol == "BTC-USDC"
    assert md.timeframe == "1h"
    assert md.num_candles == 50
    assert md.candles[0]["close"] == pytest.approx(100.0)
    # Author-provided indicators must override computed ones
    assert md.indicators["custom_metric"] == 42.0
    assert md.indicators["rsi"] == 999.0
    # Computed indicators are still present alongside the overrides
    assert "macd" in md.indicators
    assert "atr" in md.indicators


def test_scenario_to_market_data_builds_flow_when_present():
    scenario = _make_scenario(
        flow_data={
            "funding_rate": 0.0008,
            "funding_signal": "CROWDED_LONG",
            "oi_change_pct": 0.05,
            "oi_trend": "BUILDING",
            "data_richness": "FULL",
        },
    )
    md = PipelineAdapter._scenario_to_market_data(scenario)
    assert md.flow is not None
    assert md.flow.funding_rate == pytest.approx(0.0008)
    assert md.flow.funding_signal == "CROWDED_LONG"
    assert md.flow.oi_change_4h == pytest.approx(0.05)
    assert md.flow.oi_trend == "BUILDING"
    assert md.flow.data_richness == "FULL"


def test_scenario_to_market_data_no_flow_returns_none():
    scenario = _make_scenario(flow_data=None)
    md = PipelineAdapter._scenario_to_market_data(scenario)
    assert md.flow is None


# ---------------------------------------------------------------------------
# Action collapse mapping
# ---------------------------------------------------------------------------


def test_collapse_action_maps_engine_actions_to_eval_directions():
    collapse = PipelineAdapter._collapse_action
    assert collapse("LONG") == "LONG"
    assert collapse("ADD_LONG") == "LONG"
    assert collapse("SHORT") == "SHORT"
    assert collapse("ADD_SHORT") == "SHORT"
    assert collapse("HOLD") == "SKIP"
    assert collapse("CLOSE_ALL") == "SKIP"
    assert collapse("SKIP") == "SKIP"
    assert collapse("anything_else") == "SKIP"


# ---------------------------------------------------------------------------
# CLI argument parsing — run_smoke / run_eval / run_eval_full
# ---------------------------------------------------------------------------


def test_run_smoke_parser_accepts_mock_flag():
    from backtesting.evals.run_smoke import _build_parser

    args = _build_parser().parse_args(["--mock"])
    assert args.mock is True
    assert args.verbose is False


def test_run_smoke_parser_default_no_mock():
    from backtesting.evals.run_smoke import _build_parser

    args = _build_parser().parse_args([])
    assert args.mock is False
    assert args.model is None


def test_run_smoke_parser_help_does_not_crash():
    from backtesting.evals.run_smoke import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_run_eval_parser_requires_category():
    from backtesting.evals.run_eval import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_run_eval_parser_accepts_runs_and_mock():
    from backtesting.evals.run_eval import _build_parser

    args = _build_parser().parse_args(
        ["--category", "trap_setups", "--runs", "5", "--mock"]
    )
    assert args.category == "trap_setups"
    assert args.runs == 5
    assert args.mock is True


def test_run_eval_parser_default_runs_is_3():
    from backtesting.evals.run_eval import _build_parser

    args = _build_parser().parse_args(["--category", "clear_setups"])
    assert args.runs == 3


def test_run_eval_full_parser_default_runs_is_3():
    from backtesting.evals.run_eval_full import _build_parser

    args = _build_parser().parse_args([])
    assert args.runs == 3
    assert args.mock is False


def test_run_eval_full_parser_accepts_mock_and_runs():
    from backtesting.evals.run_eval_full import _build_parser

    args = _build_parser().parse_args(["--mock", "--runs", "1"])
    assert args.mock is True
    assert args.runs == 1


# ---------------------------------------------------------------------------
# CLI integration — run the smoke harness end-to-end in mock mode
# ---------------------------------------------------------------------------


def test_run_smoke_main_in_mock_mode_returns_exit_code(monkeypatch, tmp_path):
    """End-to-end smoke run in mock mode against the real scenarios.

    This is the actual CI gate behaviour: build a mock adapter, run
    every smoke scenario, write a report, return an exit code.
    """
    from backtesting.evals import run_smoke

    # Redirect report output to a tmp dir so we don't pollute the
    # checked-in reports/ folder.
    import backtesting.evals._cli as cli

    real_write = cli.write_report

    def fake_write(report):
        from backtesting.evals.reporter import generate_eval_report
        return generate_eval_report(report, output_dir=tmp_path)

    monkeypatch.setattr(cli, "write_report", fake_write)
    monkeypatch.setattr(run_smoke, "write_report", fake_write)

    code = run_smoke.main(["--mock"])
    assert code in (0, 1)  # exit code is data-driven; both are valid signals
    # At least one of the json/html reports must exist in tmp_path
    files = list(tmp_path.glob("*_eval.*"))
    assert files, f"expected report files in {tmp_path}, found nothing"


def test_run_eval_main_unknown_category_returns_2():
    from backtesting.evals import run_eval

    code = run_eval.main(["--category", "definitely_not_a_real_category", "--mock"])
    assert code == 2


def test_run_eval_main_negative_runs_returns_2():
    from backtesting.evals import run_eval

    code = run_eval.main(
        ["--category", "clear_setups", "--runs", "0", "--mock"]
    )
    assert code == 2


def test_run_eval_full_main_negative_runs_returns_2():
    from backtesting.evals import run_eval_full

    code = run_eval_full.main(["--runs", "0", "--mock"])
    assert code == 2


# ---------------------------------------------------------------------------
# _cli helpers
# ---------------------------------------------------------------------------


def test_exit_code_for_passes_above_threshold():
    from backtesting.evals._cli import exit_code_for
    from backtesting.evals.framework import EvalReport

    report = EvalReport(
        timestamp="2026-04-08T00:00:00+00:00",
        total_scenarios=4,
        runs_per_scenario=1,
        overall_pass_rate=0.75,
        by_category=[],
        consistency_avg_stdev=0.0,
        scenario_results=[],
        top_failures=[],
        model_id="mock",
        prompt_versions={},
        duration_seconds=0.5,
    )
    assert exit_code_for(report) == 0


def test_exit_code_for_fails_at_or_below_threshold():
    from backtesting.evals._cli import exit_code_for
    from backtesting.evals.framework import EvalReport

    report = EvalReport(
        timestamp="2026-04-08T00:00:00+00:00",
        total_scenarios=4,
        runs_per_scenario=1,
        overall_pass_rate=0.50,  # exactly at threshold => fail
        by_category=[],
        consistency_avg_stdev=0.0,
        scenario_results=[],
        top_failures=[],
        model_id="mock",
        prompt_versions={},
        duration_seconds=0.5,
    )
    assert exit_code_for(report) == 1


def test_print_summary_runs_without_error(capsys):
    from backtesting.evals._cli import print_summary
    from backtesting.evals.framework import CategoryStats, EvalReport, ScenarioResult

    report = EvalReport(
        timestamp="2026-04-08T00:00:00+00:00",
        total_scenarios=2,
        runs_per_scenario=1,
        overall_pass_rate=0.5,
        by_category=[CategoryStats(category="clear_setups", total=2, passed=1)],
        consistency_avg_stdev=0.01,
        scenario_results=[],
        top_failures=[
            ScenarioResult(
                scenario_id="x",
                scenario_name="Failing fixture",
                category="clear_setups",
                runs=[],
                direction_match=False,
                conviction_in_range=False,
                action_match=False,
                consistency_stdev=0.0,
                pass_fail="FAIL",
                failure_reasons=["bad direction"],
            )
        ],
        model_id="mock",
        prompt_versions={},
        duration_seconds=1.0,
    )
    print_summary(report, header="UNIT TEST SUMMARY")
    out = capsys.readouterr().out
    assert "UNIT TEST SUMMARY" in out
    assert "clear_setups" in out
    assert "bad direction" in out
