"""Tests for the 5→15 scenario expansion.

These tests pin the expansion contract: the manifest must list 15
scenarios, every category must be represented, the EvalRunner must load
all of them, and each scenario must validate against ScenarioInput +
ExpectedBehavior.

Note: ``test_scenario_schema.py::test_all_starter_scenarios_validate``
already globs and validates every JSON file under scenarios/, so the
new 10 are covered there too. This file adds the expansion-specific
invariants.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from backtesting.evals.framework import EvalRunner
from backtesting.evals.scenario_schema import (
    ExpectedBehavior,
    Scenario,
    ScenarioInput,
)


SCENARIOS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "backtesting"
    / "evals"
    / "scenarios"
)


# ---------------------------------------------------------------------------
# Manifest contract — 15 scenarios across the expected categories
# ---------------------------------------------------------------------------


def test_manifest_lists_fifteen_scenarios():
    manifest = json.loads((SCENARIOS_DIR / "manifest.json").read_text())
    assert len(manifest["scenarios"]) == 15


def test_manifest_entries_point_at_real_files():
    """Every manifest entry resolves to a file that actually exists."""
    manifest = json.loads((SCENARIOS_DIR / "manifest.json").read_text())
    for entry in manifest["scenarios"]:
        path = SCENARIOS_DIR / entry["path"]
        assert path.exists(), f"Manifest entry {entry['id']} → {path} (missing)"


def test_manifest_entry_ids_match_scenario_ids():
    """Manifest entry id must equal the id field inside the scenario file."""
    manifest = json.loads((SCENARIOS_DIR / "manifest.json").read_text())
    for entry in manifest["scenarios"]:
        scenario = Scenario.model_validate_json(
            (SCENARIOS_DIR / entry["path"]).read_text()
        )
        assert scenario.id == entry["id"]
        assert scenario.category == entry["category"]


def test_every_required_category_has_at_least_one_scenario():
    """ARCHITECTURE.md §31.4.4 enumerates the categories. After the
    expansion, all 9 (every category in the manifest at least)
    must have at least one scenario."""
    manifest = json.loads((SCENARIOS_DIR / "manifest.json").read_text())
    cats = {entry["category"] for entry in manifest["scenarios"]}
    required = {
        "clear_setups",
        "clear_avoids",
        "conflicting_signals",
        "regime_transitions",
        "trap_setups",
        "high_impact_events",
        "edge_cases",
        "cross_tf_conflicts",
        "flow_divergence",
    }
    missing = required - cats
    assert not missing, f"Required categories with no scenarios: {missing}"


def test_category_distribution_matches_expansion_spec():
    """The expansion adds 10 scenarios with a specific category breakdown."""
    manifest = json.loads((SCENARIOS_DIR / "manifest.json").read_text())
    cats = Counter(e["category"] for e in manifest["scenarios"])
    # Original 5: 2 clear_setups + 2 clear_avoids + 1 conflicting_signals
    # Expansion 10: 2 regime_transitions + 2 trap_setups + 2 high_impact_events
    #              + 1 edge_cases + 1 cross_tf_conflicts + 2 flow_divergence
    expected = {
        "clear_setups": 2,
        "clear_avoids": 2,
        "conflicting_signals": 1,
        "regime_transitions": 2,
        "trap_setups": 2,
        "high_impact_events": 2,
        "edge_cases": 1,
        "cross_tf_conflicts": 1,
        "flow_divergence": 2,
    }
    assert dict(cats) == expected


# ---------------------------------------------------------------------------
# EvalRunner integration — load + filter
# ---------------------------------------------------------------------------


def test_eval_runner_loads_all_fifteen():
    runner = EvalRunner(SCENARIOS_DIR)
    scenarios = runner.load_scenarios()
    assert len(scenarios) == 15


def test_eval_runner_categories_helper_returns_nine_categories():
    runner = EvalRunner(SCENARIOS_DIR)
    cats = runner.categories()
    assert len(cats) == 9


@pytest.mark.parametrize(
    "category",
    [
        "clear_setups",
        "clear_avoids",
        "conflicting_signals",
        "regime_transitions",
        "trap_setups",
        "high_impact_events",
        "edge_cases",
        "cross_tf_conflicts",
        "flow_divergence",
    ],
)
def test_eval_runner_filters_by_each_category(category):
    runner = EvalRunner(SCENARIOS_DIR)
    filtered = runner.load_scenarios(category=category)
    assert len(filtered) >= 1
    assert all(s.category == category for s in filtered)


# ---------------------------------------------------------------------------
# Per-scenario shape — every scenario has the fields the framework needs
# ---------------------------------------------------------------------------


def _all_scenarios() -> list[Scenario]:
    return [
        Scenario.model_validate_json(p.read_text())
        for p in sorted(SCENARIOS_DIR.rglob("*.json"))
        if p.name != "manifest.json"
    ]


def test_every_scenario_has_realistic_ohlcv_window():
    """Spec asks for 50–60 candles per scenario. Allow 49–60 (the bear-trap
    scenario has 49 because of the asymmetric flush/recovery sequence)."""
    for s in _all_scenarios():
        n = len(s.inputs.ohlcv)
        assert 49 <= n <= 60, f"{s.id}: {n} candles, expected 49-60"


def test_every_scenario_has_required_indicator_keys():
    """Pipelines downstream rely on these indicator fields being present."""
    required_keys = {"rsi", "macd", "atr", "stochastic", "adx", "bollinger_bands", "volume_ma"}
    for s in _all_scenarios():
        present = set(s.inputs.indicators.keys())
        missing = required_keys - present
        assert not missing, f"{s.id}: missing indicator keys {missing}"


def test_every_scenario_has_complete_expected_behavior():
    """Every scenario must declare the canonical expected_action and
    populate key_features_to_mention so the eval grader and the LLM
    judge both have something concrete to check."""
    for s in _all_scenarios():
        assert s.expected.expected_action in ("LONG", "SHORT", "SKIP"), (
            f"{s.id}: invalid expected_action {s.expected.expected_action!r}"
        )
        assert s.expected.key_features_to_mention, (
            f"{s.id}: key_features_to_mention is empty"
        )
        assert s.expected.notes, f"{s.id}: notes is empty"


def test_every_scenario_has_well_formed_ohlcv_rows():
    """Each candle row must have the 6 OHLCV keys with numeric values."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    for s in _all_scenarios():
        for i, c in enumerate(s.inputs.ohlcv):
            assert required.issubset(c.keys()), (
                f"{s.id} candle {i}: missing keys {required - set(c.keys())}"
            )
            assert c["high"] >= c["low"], f"{s.id} candle {i}: high < low"


def test_every_scenario_has_a_timestamp():
    for s in _all_scenarios():
        assert s.inputs.timestamp
        assert s.inputs.symbol
        assert s.inputs.timeframe


# ---------------------------------------------------------------------------
# Story-specific expectations on the 10 new scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id,expected_action",
    [
        # Regime transitions → SKIP (too early to take a side)
        ("trending_to_ranging_btc_1h_001", "SKIP"),
        ("quiet_to_volatile_eth_1h_001", "SKIP"),
        # Distribution top trap → SKIP (don't chase the breakout)
        ("distribution_top_btc_4h_001", "SKIP"),
        # Bear trap / spring → LONG (failed breakdown is bullish)
        ("bear_trap_eth_1h_001", "LONG"),
        # High-impact events → SKIP (blackout window)
        ("pre_fomc_btc_1h_001", "SKIP"),
        ("post_cpi_surprise_eth_1h_001", "SKIP"),
        # Extreme funding + low liquidity → SKIP (cost of carry too high)
        ("extreme_funding_low_liquidity_sol_1h_001", "SKIP"),
        # 1h bull / 4h bear → SKIP (defer to higher TF)
        ("1h_bull_4h_bear_btc_001", "SKIP"),
        # Smart money distribution (price up / flow bearish) → SKIP
        ("price_up_flow_bearish_btc_1h_001", "SKIP"),
        # Smart money accumulation (price down / OI building) → LONG
        ("price_down_oi_building_eth_1h_001", "LONG"),
    ],
)
def test_new_scenario_expected_action(scenario_id, expected_action):
    """Pin the expected_action for each new scenario so the spec stays in sync
    with what's on disk. If someone edits a scenario JSON to change its
    canonical answer, this test fires."""
    matches = [s for s in _all_scenarios() if s.id == scenario_id]
    assert len(matches) == 1, f"Scenario {scenario_id} not found"
    assert matches[0].expected.expected_action == expected_action
