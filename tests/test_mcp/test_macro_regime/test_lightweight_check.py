"""Tests for the LightweightCheck (no LLM, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mcp.macro_regime.data_fetcher import EconomicEvent, MacroSnapshot
from mcp.macro_regime.lightweight_check import (
    DVOL_DELTA_THRESHOLD_PCT,
    DXY_DELTA_THRESHOLD_PCT,
    HL_OI_DELTA_THRESHOLD_PCT,
    LightweightCheck,
    VIX_DELTA_THRESHOLD_PCT,
    WEEKEND_DVOL_DELTA_THRESHOLD_PCT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(**fields) -> MacroSnapshot:
    base = dict(
        fetched_at="2026-04-08T12:00:00Z",
        vix=18.0,
        vix_timestamp="2026-04-08T11:00:00Z",
        dxy=104.0,
        dxy_timestamp="2026-04-08T11:00:00Z",
        dvol=55.0,
        dvol_timestamp="2026-04-08T11:00:00Z",
        fear_greed_value=50,
        fear_greed_classification="Neutral",
        btc_dominance=51.0,
        hl_total_oi=1_000_000.0,
        hl_avg_funding=0.0001,
        economic_calendar=[],
        available_sources={"vix", "dxy", "dvol", "fear_greed", "btc_dominance", "hyperliquid"},
    )
    base.update(fields)
    return MacroSnapshot(**base)


_WEDNESDAY_NOON = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
_SATURDAY_NOON = datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def check(tmp_path: Path) -> LightweightCheck:
    return LightweightCheck(
        snapshot_path=tmp_path / "macro_regime_snapshot.json",
        clock=lambda: _WEDNESDAY_NOON,
    )


# ---------------------------------------------------------------------------
# No-trigger baselines
# ---------------------------------------------------------------------------


def test_first_run_no_previous_no_trigger(check: LightweightCheck):
    snap = _snap()
    result = check.run(snap, previous=None)
    assert result.should_trigger_deep is False
    assert result.reasons == []
    # Snapshot persisted to disk for the next tick.
    assert check._snapshot_path.exists()


def test_no_changes_no_trigger(check: LightweightCheck):
    prev = _snap()
    curr = _snap()
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is False
    assert result.reasons == []


# ---------------------------------------------------------------------------
# Per-rule triggers
# ---------------------------------------------------------------------------


def test_vix_5pct_move_triggers(check: LightweightCheck):
    prev = _snap(vix=20.0)
    curr = _snap(vix=21.5)  # +7.5%
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is True
    assert any("VIX" in r for r in result.reasons)


def test_vix_below_threshold_no_trigger(check: LightweightCheck):
    prev = _snap(vix=20.0)
    curr = _snap(vix=20.5)  # +2.5%
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is False


def test_dxy_1pct_move_triggers(check: LightweightCheck):
    prev = _snap(dxy=100.0)
    curr = _snap(dxy=101.5)  # +1.5%
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is True
    assert any("DXY" in r for r in result.reasons)


def test_dvol_10pct_move_triggers(check: LightweightCheck):
    prev = _snap(dvol=50.0)
    curr = _snap(dvol=56.0)  # +12%
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is True
    assert any("DVOL" in r for r in result.reasons)


def test_hl_oi_10pct_move_triggers(check: LightweightCheck):
    prev = _snap(hl_total_oi=1_000_000.0)
    curr = _snap(hl_total_oi=1_120_000.0)  # +12%
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is True
    assert any("HL_OI" in r for r in result.reasons)


def test_fear_greed_category_change_triggers(check: LightweightCheck):
    prev = _snap(fear_greed_classification="Fear")
    curr = _snap(fear_greed_classification="Extreme Fear")
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is True
    assert any("Fear&Greed" in r for r in result.reasons)


def test_fear_greed_same_category_case_insensitive(check: LightweightCheck):
    prev = _snap(fear_greed_classification="Neutral")
    curr = _snap(fear_greed_classification="neutral")
    result = check.run(curr, previous=prev)
    assert result.should_trigger_deep is False


def test_economic_event_within_24h_triggers(check: LightweightCheck):
    upcoming = EconomicEvent(
        name="FOMC_ANNOUNCEMENT",
        timestamp="2026-04-09T10:00:00Z",  # 22 hours away
        impact="HIGH",
    )
    snap = _snap(economic_calendar=[upcoming])
    result = check.run(snap, previous=_snap())
    assert result.should_trigger_deep is True
    assert any("FOMC_ANNOUNCEMENT" in r for r in result.reasons)


def test_economic_event_beyond_24h_no_trigger(check: LightweightCheck):
    upcoming = EconomicEvent(
        name="FOMC_ANNOUNCEMENT",
        timestamp="2026-04-15T18:00:00Z",  # 7 days away
        impact="HIGH",
    )
    snap = _snap(economic_calendar=[upcoming])
    result = check.run(snap, previous=_snap())
    assert result.should_trigger_deep is False


def test_medium_impact_event_does_not_trigger(check: LightweightCheck):
    upcoming = EconomicEvent(
        name="ECB_SPEAKER",
        timestamp="2026-04-09T10:00:00Z",
        impact="MEDIUM",
    )
    snap = _snap(economic_calendar=[upcoming])
    result = check.run(snap, previous=_snap())
    assert result.should_trigger_deep is False


# ---------------------------------------------------------------------------
# Weekend / staleness behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def weekend_check(tmp_path: Path) -> LightweightCheck:
    return LightweightCheck(
        snapshot_path=tmp_path / "macro_regime_snapshot.json",
        clock=lambda: _SATURDAY_NOON,
    )


def test_weekend_with_stale_tradfi_skips_vix_dxy(weekend_check: LightweightCheck):
    # VIX timestamp is from the previous Friday — > 24h stale on Saturday noon.
    prev = _snap(
        vix=20.0,
        vix_timestamp="2026-04-09T22:00:00Z",  # ~38h before Saturday noon
        dxy=100.0,
        dxy_timestamp="2026-04-09T22:00:00Z",
    )
    curr = _snap(
        vix=25.0,  # +25% — would normally trigger
        vix_timestamp="2026-04-09T22:00:00Z",
        dxy=110.0,  # +10% — would normally trigger
        dxy_timestamp="2026-04-09T22:00:00Z",
    )
    result = weekend_check.run(curr, previous=prev)
    assert result.should_trigger_deep is False
    assert all("VIX" not in r and "DXY" not in r for r in result.reasons)


def test_weekend_dvol_15pct_spike_triggers(weekend_check: LightweightCheck):
    prev = _snap(
        vix_timestamp="2026-04-09T22:00:00Z",
        dxy_timestamp="2026-04-09T22:00:00Z",
        dvol=50.0,
    )
    curr = _snap(
        vix_timestamp="2026-04-09T22:00:00Z",
        dxy_timestamp="2026-04-09T22:00:00Z",
        dvol=60.0,  # +20%
    )
    result = weekend_check.run(curr, previous=prev)
    assert result.should_trigger_deep is True
    assert any("DVOL" in r for r in result.reasons)


def test_weekend_dvol_below_15pct_no_trigger(weekend_check: LightweightCheck):
    # Stale tradfi → only DVOL/F&G/OI count.
    prev = _snap(
        vix_timestamp="2026-04-09T22:00:00Z",
        dxy_timestamp="2026-04-09T22:00:00Z",
        dvol=50.0,
    )
    curr = _snap(
        vix_timestamp="2026-04-09T22:00:00Z",
        dxy_timestamp="2026-04-09T22:00:00Z",
        dvol=55.0,  # +10%, below 15% weekend threshold
    )
    result = weekend_check.run(curr, previous=prev)
    assert result.should_trigger_deep is False


def test_weekend_with_fresh_tradfi_uses_normal_thresholds(weekend_check: LightweightCheck):
    # Saturday but VIX timestamp < 24h old → normal mode.
    fresh_ts = (_SATURDAY_NOON - timedelta(hours=4)).isoformat().replace("+00:00", "Z")
    prev = _snap(vix=20.0, vix_timestamp=fresh_ts)
    curr = _snap(vix=22.0, vix_timestamp=fresh_ts)  # +10%
    result = weekend_check.run(curr, previous=prev)
    assert result.should_trigger_deep is True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "snap.json"
    check = LightweightCheck(snapshot_path=path, clock=lambda: _WEDNESDAY_NOON)
    snap = _snap(vix=22.0)
    check.save_snapshot(snap)
    loaded = check.load_previous()
    assert loaded is not None
    assert loaded.vix == 22.0


def test_load_previous_missing_returns_none(tmp_path: Path):
    check = LightweightCheck(snapshot_path=tmp_path / "missing.json")
    assert check.load_previous() is None


def test_load_previous_corrupt_returns_none(tmp_path: Path):
    path = tmp_path / "snap.json"
    path.write_text("not json {")
    check = LightweightCheck(snapshot_path=path)
    assert check.load_previous() is None


def test_run_loads_previous_from_disk_when_not_provided(tmp_path: Path):
    path = tmp_path / "snap.json"
    check = LightweightCheck(snapshot_path=path, clock=lambda: _WEDNESDAY_NOON)
    # Save a baseline.
    check.save_snapshot(_snap(vix=20.0))
    # Run with previous=None — should load from disk and detect the +10% move.
    result = check.run(_snap(vix=22.0), previous=None)
    assert result.should_trigger_deep is True
    assert any("VIX" in r for r in result.reasons)


def test_threshold_constants_match_spec():
    # §13.2.1 sanity: the literal numbers exist as constants.
    assert VIX_DELTA_THRESHOLD_PCT == 5.0
    assert DXY_DELTA_THRESHOLD_PCT == 1.0
    assert DVOL_DELTA_THRESHOLD_PCT == 10.0
    assert HL_OI_DELTA_THRESHOLD_PCT == 10.0
    assert WEEKEND_DVOL_DELTA_THRESHOLD_PCT == 15.0


def test_check_result_to_dict_smoke(check: LightweightCheck):
    snap = _snap(vix=22.0)
    result = check.run(snap, previous=_snap(vix=20.0))
    payload = result.to_dict()
    assert payload["should_trigger_deep"] is True
    assert payload["snapshot"]["vix"] == 22.0
    assert isinstance(payload["reasons"], list)
