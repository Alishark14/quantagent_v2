"""Tests for the MacroRegimeManager agent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm.base import LLMResponse
from mcp.macro_regime.agent import (
    BLACKOUT_POST_BUFFER_MINUTES,
    BLACKOUT_PRE_BUFFER_MINUTES,
    BlackoutWindow,
    MacroAdjustments,
    MacroRegime,
    MacroRegimeManager,
    build_assessment_prompt,
    build_blackout_windows,
    load_macro_regime,
)
from mcp.macro_regime.data_fetcher import EconomicEvent, MacroSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FIXED_NOW = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)


def _snap(events=None, **fields) -> MacroSnapshot:
    base = dict(
        fetched_at="2026-04-08T12:00:00Z",
        vix=18.0,
        dxy=104.0,
        dvol=55.0,
        fear_greed_value=25,
        fear_greed_classification="Extreme Fear",
        btc_dominance=51.3,
        hl_total_oi=1_000_000.0,
        hl_avg_funding=0.0001,
        economic_calendar=events or [],
        available_sources={"vix", "dxy", "dvol", "fear_greed", "btc_dominance", "hyperliquid"},
    )
    base.update(fields)
    return MacroSnapshot(**base)


def _llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=100,
        output_tokens=50,
        cost=0.0,
        model="claude-test",
        latency_ms=10.0,
        cached_input_tokens=0,
    )


def _mock_llm(content: str) -> AsyncMock:
    llm = AsyncMock()
    llm.generate_text = AsyncMock(return_value=_llm_response(content))
    return llm


_VALID_LLM_JSON = json.dumps(
    {
        "regime": "RISK_OFF",
        "confidence": 0.82,
        "reasoning": "VIX elevated, DXY strengthening, F&G in extreme fear.",
        "adjustments": {
            "conviction_threshold_boost": 0.1,
            "max_concurrent_positions_override": 1,
            "position_size_multiplier": 0.7,
            "avoid_assets": ["TSLA-USDC", "NVDA-USDC"],
            "prefer_assets": ["GOLD-USDC", "BTC-USDC"],
        },
    }
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_deep_happy_path_writes_file(tmp_path: Path):
    out = tmp_path / "macro_regime.json"
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(_VALID_LLM_JSON),
        output_path=out,
        clock=lambda: _FIXED_NOW,
    )
    snap = _snap()
    regime = await agent.run_deep(snap)

    assert regime.error is None
    assert regime.regime == "RISK_OFF"
    assert regime.confidence == pytest.approx(0.82)
    assert regime.adjustments.conviction_threshold_boost == pytest.approx(0.1)
    assert regime.adjustments.position_size_multiplier == pytest.approx(0.7)
    assert "TSLA-USDC" in regime.adjustments.avoid_assets
    assert regime.generated_at == "2026-04-08T12:00:00Z"
    assert regime.expires == "2026-04-09T12:00:00Z"
    assert out.exists()
    written = json.loads(out.read_text())
    assert written["regime"] == "RISK_OFF"


@pytest.mark.asyncio
async def test_dry_run_does_not_write(tmp_path: Path):
    out = tmp_path / "macro_regime.json"
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(_VALID_LLM_JSON),
        output_path=out,
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap(), dry_run=True)
    assert regime.error is None
    assert not out.exists()


@pytest.mark.asyncio
async def test_atomic_write_leaves_no_tmp(tmp_path: Path):
    out = tmp_path / "macro_regime.json"
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(_VALID_LLM_JSON),
        output_path=out,
        clock=lambda: _FIXED_NOW,
    )
    await agent.run_deep(_snap())
    assert out.exists()
    assert not (tmp_path / "macro_regime.json.tmp").exists()


# ---------------------------------------------------------------------------
# LLM parse failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_call_failure_returns_error(tmp_path: Path):
    llm = AsyncMock()
    llm.generate_text = AsyncMock(side_effect=RuntimeError("anthropic 429"))
    agent = MacroRegimeManager(
        llm_provider=llm, output_path=tmp_path / "out.json", clock=lambda: _FIXED_NOW
    )
    regime = await agent.run_deep(_snap())
    assert regime.error and "llm_call_failed" in regime.error
    assert not (tmp_path / "out.json").exists()


@pytest.mark.asyncio
async def test_llm_returns_unparseable_json(tmp_path: Path):
    agent = MacroRegimeManager(
        llm_provider=_mock_llm("not json at all"),
        output_path=tmp_path / "out.json",
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap())
    assert regime.error and "llm_parse_failed" in regime.error
    assert not (tmp_path / "out.json").exists()


@pytest.mark.asyncio
async def test_llm_returns_invalid_regime_value(tmp_path: Path):
    bad = json.dumps({"regime": "PARTY_TIME", "confidence": 0.9, "reasoning": "x"})
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(bad),
        output_path=tmp_path / "out.json",
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap())
    assert regime.error and "payload_invalid" in regime.error


@pytest.mark.asyncio
async def test_llm_returns_confidence_out_of_range(tmp_path: Path):
    bad = json.dumps({"regime": "RISK_ON", "confidence": 1.5, "reasoning": "x"})
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(bad),
        output_path=tmp_path / "out.json",
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap())
    assert regime.error and "payload_invalid" in regime.error


@pytest.mark.asyncio
async def test_llm_response_in_code_fence_is_parsed(tmp_path: Path):
    fenced = f"```json\n{_VALID_LLM_JSON}\n```"
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(fenced),
        output_path=tmp_path / "out.json",
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap())
    assert regime.error is None
    assert regime.regime == "RISK_OFF"


@pytest.mark.asyncio
async def test_llm_response_with_prose_prefix_is_parsed(tmp_path: Path):
    prose = f"Here is the assessment:\n{_VALID_LLM_JSON}\n"
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(prose),
        output_path=tmp_path / "out.json",
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap())
    assert regime.error is None
    assert regime.regime == "RISK_OFF"


@pytest.mark.asyncio
async def test_position_size_multiplier_is_clamped(tmp_path: Path):
    bad = json.dumps(
        {
            "regime": "RISK_ON",
            "confidence": 0.5,
            "reasoning": "x",
            "adjustments": {
                "conviction_threshold_boost": 5.0,  # clamp to 0.5
                "position_size_multiplier": 99.0,  # clamp to 2.0
            },
        }
    )
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(bad),
        output_path=tmp_path / "out.json",
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap())
    assert regime.error is None
    assert regime.adjustments.position_size_multiplier == 2.0
    assert regime.adjustments.conviction_threshold_boost == 0.5


# ---------------------------------------------------------------------------
# Blackout windows
# ---------------------------------------------------------------------------


def test_build_blackout_windows_high_impact_only():
    ref = _FIXED_NOW
    events = [
        EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-09T14:00:00Z", "HIGH"),
        EconomicEvent("ECB_SPEAKER", "2026-04-09T16:00:00Z", "MEDIUM"),
        EconomicEvent("CPI", "2026-04-10T08:00:00Z", "HIGH"),
    ]
    windows = build_blackout_windows(events, reference=ref)
    assert len(windows) == 2
    reasons = {w.reason for w in windows}
    assert reasons == {"FOMC_ANNOUNCEMENT", "CPI"}
    for w in windows:
        assert w.action == "execution_block"


def test_blackout_window_pre_post_buffers_default():
    ref = _FIXED_NOW
    events = [EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-09T14:00:00Z", "HIGH")]
    windows = build_blackout_windows(events, reference=ref)
    w = windows[0]
    start = datetime.fromisoformat(w.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(w.end.replace("Z", "+00:00"))
    assert (end - start) == timedelta(
        minutes=BLACKOUT_PRE_BUFFER_MINUTES + BLACKOUT_POST_BUFFER_MINUTES
    )
    event_dt = datetime(2026, 4, 9, 14, 0, tzinfo=timezone.utc)
    assert start == event_dt - timedelta(minutes=BLACKOUT_PRE_BUFFER_MINUTES)
    assert end == event_dt + timedelta(minutes=BLACKOUT_POST_BUFFER_MINUTES)


def test_blackout_excludes_events_beyond_lookahead():
    ref = _FIXED_NOW
    events = [EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-15T14:00:00Z", "HIGH")]
    windows = build_blackout_windows(events, reference=ref, lookahead_hours=48)
    assert windows == []


def test_blackout_excludes_long_past_events():
    ref = _FIXED_NOW
    events = [EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-01T14:00:00Z", "HIGH")]
    windows = build_blackout_windows(events, reference=ref)
    assert windows == []


def test_blackout_window_contains_check():
    w = BlackoutWindow(
        start="2026-04-09T13:00:00Z",
        end="2026-04-09T14:30:00Z",
        reason="FOMC_ANNOUNCEMENT",
    )
    assert w.contains(datetime(2026, 4, 9, 13, 30, tzinfo=timezone.utc))
    assert not w.contains(datetime(2026, 4, 9, 15, 0, tzinfo=timezone.utc))


@pytest.mark.asyncio
async def test_blackout_windows_appear_in_written_output(tmp_path: Path):
    out = tmp_path / "macro_regime.json"
    events = [EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-09T14:00:00Z", "HIGH")]
    agent = MacroRegimeManager(
        llm_provider=_mock_llm(_VALID_LLM_JSON),
        output_path=out,
        clock=lambda: _FIXED_NOW,
    )
    regime = await agent.run_deep(_snap(events=events))
    assert len(regime.blackout_windows) == 1
    written = json.loads(out.read_text())
    assert len(written["blackout_windows"]) == 1
    assert written["blackout_windows"][0]["reason"] == "FOMC_ANNOUNCEMENT"


# ---------------------------------------------------------------------------
# Assessment prompt
# ---------------------------------------------------------------------------


def test_build_assessment_prompt_includes_core_data():
    snap = _snap()
    prompt = build_assessment_prompt(snap, urgency="normal")
    assert "VIX" in prompt
    assert "DXY" in prompt
    assert "Extreme Fear" in prompt
    assert "Hyperliquid total OI" in prompt
    assert "Urgency: normal" in prompt
    assert "JSON" in prompt


def test_build_assessment_prompt_emergency_includes_symbols():
    snap = _snap()
    prompt = build_assessment_prompt(
        snap,
        urgency="emergency",
        triggering_symbols=["BTC-USDC", "ETH-USDC"],
        reasons=["VIX spiked 8%"],
    )
    assert "Urgency: emergency" in prompt
    assert "BTC-USDC" in prompt
    assert "ETH-USDC" in prompt
    assert "VIX spiked 8%" in prompt


def test_build_assessment_prompt_handles_missing_data():
    snap = MacroSnapshot(fetched_at="2026-04-08T12:00:00Z")
    prompt = build_assessment_prompt(snap)
    assert "(unavailable)" in prompt
    assert "(none — all data sources failed to fetch)" in prompt


# ---------------------------------------------------------------------------
# load_macro_regime helper
# ---------------------------------------------------------------------------


def test_load_macro_regime_round_trip(tmp_path: Path):
    payload = {
        "regime": "RISK_OFF",
        "confidence": 0.8,
        "reasoning": "test",
        "adjustments": {
            "conviction_threshold_boost": 0.1,
            "position_size_multiplier": 0.7,
            "avoid_assets": ["X"],
            "prefer_assets": ["Y"],
        },
        "blackout_windows": [
            {
                "start": "2026-04-09T13:00:00Z",
                "end": "2026-04-09T14:30:00Z",
                "reason": "FOMC_ANNOUNCEMENT",
                "action": "execution_block",
            }
        ],
        "generated_at": "2026-04-08T12:00:00Z",
        "expires": "2026-04-09T12:00:00Z",
    }
    p = tmp_path / "macro_regime.json"
    p.write_text(json.dumps(payload))
    regime = load_macro_regime(p)
    assert regime is not None
    assert regime.regime == "RISK_OFF"
    assert regime.adjustments.position_size_multiplier == 0.7
    assert len(regime.blackout_windows) == 1


def test_load_macro_regime_missing_returns_none(tmp_path: Path):
    assert load_macro_regime(tmp_path / "missing.json") is None


def test_load_macro_regime_corrupt_returns_none(tmp_path: Path):
    p = tmp_path / "macro_regime.json"
    p.write_text("not json")
    assert load_macro_regime(p) is None


# ---------------------------------------------------------------------------
# MacroAdjustments / dataclass smoke
# ---------------------------------------------------------------------------


def test_macro_adjustments_from_dict_defaults():
    adj = MacroAdjustments.from_dict(None)
    assert adj.conviction_threshold_boost == 0.0
    assert adj.position_size_multiplier == 1.0
    assert adj.avoid_assets == []


def test_macro_regime_to_dict_smoke():
    r = MacroRegime(regime="NEUTRAL", confidence=0.5, reasoning="x")
    payload = r.to_dict()
    assert payload["regime"] == "NEUTRAL"
    assert payload["confidence"] == 0.5
    assert payload["adjustments"]["position_size_multiplier"] == 1.0
