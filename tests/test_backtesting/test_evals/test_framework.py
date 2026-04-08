"""Tests for backtesting.evals.framework — EvalRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtesting.evals.framework import (
    CategoryStats,
    EvalReport,
    EvalRunner,
    ScenarioResult,
)
from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.scenario_schema import (
    ExpectedBehavior,
    Scenario,
    ScenarioInput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(
    sid: str = "test_001",
    category: str = "clear_setups",
    expected_action: str = "LONG",
    conviction_min: float | None = None,
    conviction_max: float | None = None,
    signal_direction: str | None = "BULLISH",
) -> Scenario:
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
            ohlcv=[
                {"timestamp": i * 1000, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}
                for i in range(50)
            ],
            indicators={"rsi": 55.0},
            flow_data=None,
            regime_context="trending",
            timestamp="2026-04-08T00:00:00+00:00",
        ),
        expected=ExpectedBehavior(
            expected_action=expected_action,
            signal_direction=signal_direction,
            conviction_min=conviction_min,
            conviction_max=conviction_max,
            key_features_to_mention=["bull flag"],
        ),
    )


def _make_output(direction: str = "LONG", conviction: float = 0.75) -> EvalOutput:
    return EvalOutput(
        direction=direction,
        conviction=conviction,
        sl_price=99.0,
        tp1_price=102.0,
        tp2_price=104.0,
        position_size_pct=0.1,
        reasoning="bull flag",
        latency_ms=100.0,
        model_id="mock",
    )


@pytest.fixture
def tmp_scenarios_dir(tmp_path):
    """Build a 4-scenario manifest spanning 2 categories."""
    scenarios = [
        _make_scenario("s1", category="clear_setups", expected_action="LONG"),
        _make_scenario("s2", category="clear_setups", expected_action="LONG"),
        _make_scenario(
            "s3",
            category="clear_avoids",
            expected_action="SKIP",
            signal_direction=None,
        ),
        _make_scenario(
            "s4",
            category="clear_avoids",
            expected_action="SKIP",
            signal_direction=None,
        ),
    ]
    for s in scenarios:
        cat_dir = tmp_path / s.category
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / f"{s.id}.json").write_text(s.model_dump_json())

    manifest = {
        "version": 1,
        "last_validated": "2026-04-08T00:00:00+00:00",
        "scenarios": [
            {"id": s.id, "name": s.name, "category": s.category, "path": f"{s.category}/{s.id}.json"}
            for s in scenarios
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------


def test_load_scenarios_returns_all(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)
    scenarios = runner.load_scenarios()
    assert len(scenarios) == 4
    assert [s.id for s in scenarios] == ["s1", "s2", "s3", "s4"]


def test_load_scenarios_filters_by_category(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)
    setups = runner.load_scenarios(category="clear_setups")
    assert {s.id for s in setups} == {"s1", "s2"}


def test_categories_lists_distinct(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)
    cats = runner.categories()
    assert set(cats) == {"clear_setups", "clear_avoids"}


def test_load_scenarios_missing_manifest_raises(tmp_path):
    runner = EvalRunner(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        runner.load_scenarios()


def test_load_scenarios_skips_missing_files(tmp_path, caplog):
    """A manifest entry pointing at a non-existent file logs a warning."""
    import logging

    manifest = {
        "version": 1,
        "last_validated": "2026-04-08T00:00:00+00:00",
        "scenarios": [
            {"id": "ghost", "name": "Ghost", "category": "x", "path": "x/ghost.json"}
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    runner = EvalRunner(tmp_path)
    with caplog.at_level(logging.WARNING):
        out = runner.load_scenarios()
    assert out == []
    assert any("missing" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_scenario — pipeline invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_scenario_with_callable_pipeline():
    runner = EvalRunner(Path("/tmp"))  # dir not used by run_scenario
    scenario = _make_scenario()

    async def pipeline(s: Scenario) -> EvalOutput:
        return _make_output()

    runs = await runner.run_scenario(scenario, pipeline, runs=3)
    assert len(runs) == 3
    assert all(r.direction == "LONG" for r in runs)


@pytest.mark.asyncio
async def test_run_scenario_with_object_pipeline():
    class Pipeline:
        async def analyze(self, scenario: Scenario) -> EvalOutput:
            return _make_output()

    runner = EvalRunner(Path("/tmp"))
    runs = await runner.run_scenario(_make_scenario(), Pipeline(), runs=2)
    assert len(runs) == 2


@pytest.mark.asyncio
async def test_run_scenario_raises_on_non_eval_output():
    runner = EvalRunner(Path("/tmp"))

    async def bad_pipeline(s: Scenario):
        return {"direction": "LONG"}

    with pytest.raises(TypeError, match="EvalOutput"):
        await runner.run_scenario(_make_scenario(), bad_pipeline, runs=1)


@pytest.mark.asyncio
async def test_run_scenario_zero_runs_raises():
    runner = EvalRunner(Path("/tmp"))

    async def pipeline(s):
        return _make_output()

    with pytest.raises(ValueError, match="runs must be"):
        await runner.run_scenario(_make_scenario(), pipeline, runs=0)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def test_grade_passes_on_correct_action_and_conviction():
    runner = EvalRunner(Path("/tmp"))
    scenario = _make_scenario(conviction_min=0.6, conviction_max=0.9)
    runs = [_make_output(direction="LONG", conviction=0.75)]
    result = runner.grade(scenario, runs)
    assert result.pass_fail == "PASS"
    assert result.action_match is True
    assert result.direction_match is True
    assert result.conviction_in_range is True
    assert result.failure_reasons == []


def test_grade_fails_on_wrong_action():
    runner = EvalRunner(Path("/tmp"))
    scenario = _make_scenario()  # expects LONG
    runs = [_make_output(direction="SHORT", conviction=0.8)]
    result = runner.grade(scenario, runs)
    assert result.pass_fail == "FAIL"
    assert result.action_match is False
    assert any("action" in r for r in result.failure_reasons)


def test_grade_fails_on_conviction_below_min():
    runner = EvalRunner(Path("/tmp"))
    scenario = _make_scenario(conviction_min=0.7)
    runs = [_make_output(conviction=0.5)]
    result = runner.grade(scenario, runs)
    assert result.pass_fail == "FAIL"
    assert result.conviction_in_range is False
    assert any("below min" in r for r in result.failure_reasons)


def test_grade_fails_on_conviction_above_max():
    runner = EvalRunner(Path("/tmp"))
    scenario = _make_scenario(conviction_max=0.4)
    runs = [_make_output(conviction=0.8)]
    result = runner.grade(scenario, runs)
    assert result.pass_fail == "FAIL"
    assert result.conviction_in_range is False


def test_grade_consistency_stdev_across_multiple_runs():
    runner = EvalRunner(Path("/tmp"))
    scenario = _make_scenario(conviction_min=0.5, conviction_max=0.9)
    runs = [
        _make_output(conviction=0.7),
        _make_output(conviction=0.8),
        _make_output(conviction=0.6),
    ]
    result = runner.grade(scenario, runs)
    assert result.consistency_stdev > 0  # variance present
    assert result.pass_fail == "PASS"


def test_grade_signal_direction_none_means_any():
    """signal_direction=None on the expected behaviour means 'any acceptable'."""
    runner = EvalRunner(Path("/tmp"))
    scenario = _make_scenario(
        expected_action="SKIP",
        signal_direction=None,
    )
    runs = [_make_output(direction="SKIP", conviction=0.3)]
    result = runner.grade(scenario, runs)
    assert result.direction_match is True
    assert result.pass_fail == "PASS"


def test_grade_no_runs_returns_failure():
    runner = EvalRunner(Path("/tmp"))
    result = runner.grade(_make_scenario(), [])
    assert result.pass_fail == "FAIL"
    assert "no runs" in result.failure_reasons[0]


# ---------------------------------------------------------------------------
# Tiered run modes (smoke / category / full)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_smoke_executes_two_per_category(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)

    async def pipeline(s):
        # Always returns the right action for each scenario
        return _make_output(
            direction=s.expected.expected_action,
            conviction=0.75 if s.expected.expected_action == "LONG" else 0.3,
        )

    report = await runner.run_smoke(pipeline)
    # 2 categories × 2 scenarios per category = 4 scenarios
    assert report.total_scenarios == 4
    assert report.runs_per_scenario == 1
    # All should pass with the perfect mock pipeline
    assert report.overall_pass_rate == 1.0


@pytest.mark.asyncio
async def test_run_category_filters(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)

    async def pipeline(s):
        return _make_output(direction=s.expected.expected_action, conviction=0.75)

    report = await runner.run_category("clear_setups", pipeline, runs_per_scenario=2)
    assert report.total_scenarios == 2
    assert all(r.category == "clear_setups" for r in report.scenario_results)


@pytest.mark.asyncio
async def test_run_full_aggregates_categories(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)

    async def pipeline(s):
        return _make_output(direction=s.expected.expected_action, conviction=0.75)

    report = await runner.run_full(pipeline, runs_per_scenario=1)
    assert report.total_scenarios == 4
    cats_in_report = {c.category for c in report.by_category}
    assert cats_in_report == {"clear_setups", "clear_avoids"}
    for cat_stat in report.by_category:
        assert cat_stat.pass_rate == 1.0


@pytest.mark.asyncio
async def test_run_full_handles_pipeline_crash(tmp_scenarios_dir):
    """A scenario whose pipeline raises is recorded as FAIL, not propagated."""
    runner = EvalRunner(tmp_scenarios_dir)

    async def crashing_pipeline(s):
        raise RuntimeError("LLM API down")

    report = await runner.run_full(crashing_pipeline, runs_per_scenario=1)
    assert report.total_scenarios == 4
    assert report.overall_pass_rate == 0.0
    assert all(r.pass_fail == "FAIL" for r in report.scenario_results)
    assert any("LLM API down" in r.failure_reasons[0] for r in report.scenario_results)


@pytest.mark.asyncio
async def test_run_full_records_top_failures(tmp_scenarios_dir):
    """Top failures list contains failed scenarios in order."""
    runner = EvalRunner(tmp_scenarios_dir)

    async def half_right(s):
        # Always returns LONG → setups pass, avoids fail
        return _make_output(direction="LONG", conviction=0.75)

    report = await runner.run_full(half_right, runs_per_scenario=1)
    assert report.overall_pass_rate == 0.5
    assert len(report.top_failures) == 2
    for fail in report.top_failures:
        assert fail.category == "clear_avoids"
        assert fail.action_match is False


# ---------------------------------------------------------------------------
# Report payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_to_dict_is_json_serialisable(tmp_scenarios_dir):
    runner = EvalRunner(tmp_scenarios_dir)

    async def pipeline(s):
        return _make_output(direction=s.expected.expected_action, conviction=0.75)

    report = await runner.run_full(pipeline, runs_per_scenario=1)
    d = report.to_dict()
    json.dumps(d)  # must not raise
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
        assert key in d


def test_category_stats_pass_rate():
    s = CategoryStats(category="x", total=10, passed=7)
    assert s.pass_rate == 0.7
    assert CategoryStats(category="x", total=0, passed=0).pass_rate == 0.0
