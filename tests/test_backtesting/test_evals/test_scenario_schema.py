"""Tests for the eval scenario Pydantic schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backtesting.evals.scenario_schema import (
    ExpectedBehavior,
    Scenario,
    ScenarioInput,
)


SCENARIOS_DIR = Path(__file__).parent.parent.parent.parent / "backtesting" / "evals" / "scenarios"


# ---------------------------------------------------------------------------
# Round-trip on the 5 starter scenarios
# ---------------------------------------------------------------------------


def test_all_starter_scenarios_validate():
    """Every JSON file under scenarios/ (excluding manifest) loads cleanly."""
    files = [
        p for p in SCENARIOS_DIR.rglob("*.json")
        if p.name != "manifest.json"
    ]
    assert len(files) >= 5, "Expected at least 5 starter scenarios"

    for path in files:
        scenario = Scenario.model_validate_json(path.read_text())
        assert scenario.id
        assert scenario.category
        assert len(scenario.inputs.ohlcv) >= 30, (
            f"{scenario.id}: needs ≥30 candles, got {len(scenario.inputs.ohlcv)}"
        )
        assert scenario.expected.expected_action in ("LONG", "SHORT", "SKIP")


def test_starter_scenarios_cover_required_categories():
    """Spec calls for clear_setups, clear_avoids, and conflicting_signals."""
    files = [p for p in SCENARIOS_DIR.rglob("*.json") if p.name != "manifest.json"]
    cats = {Scenario.model_validate_json(p.read_text()).category for p in files}
    assert "clear_setups" in cats
    assert "clear_avoids" in cats
    assert "conflicting_signals" in cats


# ---------------------------------------------------------------------------
# Round-trip + serialization
# ---------------------------------------------------------------------------


def _candle(ts: int) -> dict:
    return {"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}


def _make_scenario(**overrides) -> Scenario:
    base = {
        "id": "test_001",
        "name": "Test scenario",
        "category": "clear_setups",
        "version": 1,
        "created_at": "2026-04-08T00:00:00+00:00",
        "last_validated": "2026-04-08T00:00:00+00:00",
        "inputs": {
            "symbol": "BTC-USDC",
            "timeframe": "1h",
            "ohlcv": [_candle(i * 1000) for i in range(50)],
            "indicators": {"rsi": 55.0},
            "flow_data": None,
            "regime_context": "trending",
            "timestamp": "2026-04-08T00:00:00+00:00",
        },
        "expected": {
            "expected_action": "LONG",
            "key_features_to_mention": ["bull flag"],
        },
    }
    base.update(overrides)
    return Scenario.model_validate(base)


def test_scenario_round_trip_via_json():
    s = _make_scenario()
    raw = s.model_dump_json()
    s2 = Scenario.model_validate_json(raw)
    assert s2.id == s.id
    assert s2.inputs.symbol == "BTC-USDC"
    assert s2.expected.expected_action == "LONG"
    assert len(s2.inputs.ohlcv) == 50


def test_scenario_extra_fields_allowed():
    """Schema uses extra='allow' so future fields don't break old loaders."""
    s = _make_scenario()
    raw = s.model_dump()
    raw["future_field"] = "from_v2"
    s2 = Scenario.model_validate(raw)
    assert hasattr(s2, "future_field") or "future_field" in s2.model_extra


def test_missing_required_id_raises():
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "name": "x",
                "category": "y",
                "created_at": "z",
                "last_validated": "z",
                "inputs": {
                    "symbol": "BTC-USDC",
                    "timeframe": "1h",
                    "ohlcv": [],
                    "timestamp": "z",
                },
                "expected": {"expected_action": "SKIP"},
            }
        )


def test_expected_behavior_defaults():
    eb = ExpectedBehavior(expected_action="SKIP")
    assert eb.signal_direction is None
    assert eb.conviction_min is None
    assert eb.conviction_max is None
    assert eb.key_features_to_mention == []


def test_scenario_input_requires_symbol_and_timeframe():
    with pytest.raises(ValidationError):
        ScenarioInput(
            ohlcv=[],
            timestamp="2026-04-08T00:00:00+00:00",
        )
