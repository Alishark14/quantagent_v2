"""Tests for blackout-window enforcement.

Two surfaces:

  1. ConvictionAgent loads `macro_regime.json` at cycle start and
     forces conviction=0.0 when a blackout window is active. RISK_OFF
     overlay stamps the threshold boost + size multiplier on the
     output. Expired / missing files are the safe default — no overlay.
  2. Sentinel checks the same file before emitting `SetupDetected`
     and suppresses the emission silently when a blackout is active.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.conviction.agent import ConvictionAgent
from engine.events import InProcessBus, SetupDetected
from engine.types import (
    AdapterCapabilities,
    ConvictionOutput,
    MarketData,
    OrderResult,
    SignalOutput,
)
from exchanges.base import ExchangeAdapter
from llm.base import LLMProvider, LLMResponse
from sentinel.monitor import SentinelMonitor


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


_FIXED_NOW = datetime(2026, 4, 9, 13, 30, tzinfo=timezone.utc)
# A blackout that contains _FIXED_NOW: FOMC at 14:00, pre-buffer 60min, post 30min
# → window 13:00–14:30. Half an hour into it.
_FOMC_AT = datetime(2026, 4, 9, 14, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_macro_payload(
    *,
    regime: str = "NEUTRAL",
    confidence: float = 0.5,
    blackout: bool = False,
    expired: bool = False,
    threshold_boost: float = 0.0,
    size_multiplier: float = 1.0,
    avoid: list[str] | None = None,
    prefer: list[str] | None = None,
    blackout_reason: str = "FOMC_ANNOUNCEMENT",
) -> dict:
    """Build a macro_regime.json payload pinned around _FIXED_NOW."""
    expires_dt = _FIXED_NOW + timedelta(hours=24)
    if expired:
        expires_dt = _FIXED_NOW - timedelta(hours=1)
    payload = {
        "regime": regime,
        "confidence": confidence,
        "reasoning": "test",
        "adjustments": {
            "conviction_threshold_boost": threshold_boost,
            "max_concurrent_positions_override": None,
            "position_size_multiplier": size_multiplier,
            "avoid_assets": avoid or [],
            "prefer_assets": prefer or [],
        },
        "blackout_windows": [],
        "generated_at": _iso(_FIXED_NOW - timedelta(hours=2)),
        "expires": _iso(expires_dt),
    }
    if blackout:
        payload["blackout_windows"] = [
            {
                "start": _iso(_FOMC_AT - timedelta(minutes=60)),
                "end": _iso(_FOMC_AT + timedelta(minutes=30)),
                "reason": blackout_reason,
                "action": "execution_block",
            }
        ]
    return payload


def _write_macro(tmp_path: Path, payload: dict | None) -> Path:
    p = tmp_path / "macro_regime.json"
    if payload is not None:
        p.write_text(json.dumps(payload))
    return p


# ---------- ConvictionAgent helpers ----------


_HIGH_CONVICTION_RESPONSE = json.dumps(
    {
        "conviction_score": 0.78,
        "direction": "LONG",
        "regime": "TRENDING_UP",
        "regime_confidence": 0.82,
        "signal_quality": "HIGH",
        "contradictions": [],
        "reasoning": "Two agents bullish",
        "factual_weight": 0.4,
        "subjective_weight": 0.6,
    }
)


class _RecordingLLM(LLMProvider):
    def __init__(self, response: str = _HIGH_CONVICTION_RESPONSE) -> None:
        self._response = response
        self.call_count = 0
        self.last_user_prompt: str | None = None

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_user_prompt = user_prompt
        return LLMResponse(
            content=self._response,
            input_tokens=100,
            output_tokens=50,
            cost=0.0,
            model="claude-test",
            latency_ms=10.0,
            cached_input_tokens=0,
        )

    async def generate_vision(self, **kwargs) -> LLMResponse:
        raise NotImplementedError


def _make_market_data() -> MarketData:
    candles = [
        {
            "timestamp": 1700000000 + i * 3600,
            "open": 65000.0 + i,
            "high": 65020.0 + i,
            "low": 64980.0 + i,
            "close": 65000.0 + i,
            "volume": 1000.0,
        }
        for i in range(50)
    ]
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=candles,
        num_candles=50,
        lookback_description="~2 days",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={
            "rsi": 55.0,
            "macd": {"macd": 1.0, "signal": 0.9, "histogram": 0.1, "histogram_direction": "rising", "cross": "none"},
            "roc": 0.5,
            "stochastic": {"k": 50.0, "d": 50.0, "zone": "neutral"},
            "williams_r": -50.0,
            "atr": 400.0,
            "adx": {"adx": 25.0, "plus_di": 20.0, "minus_di": 15.0, "classification": "TRENDING"},
            "bollinger_bands": {"upper": 66000.0, "middle": 65000.0, "lower": 64000.0, "width": 2000.0, "width_percentile": 50.0},
            "volume_ma": {"ma": 1000.0, "current": 1100.0, "ratio": 1.1, "spike": False},
            "volatility_percentile": 50.0,
        },
        swing_highs=[65500.0, 66000.0],
        swing_lows=[63100.0, 62500.0],
    )


def _make_signals() -> list[SignalOutput]:
    return [
        SignalOutput(
            agent_name="indicator_agent",
            signal_type="llm",
            direction="BULLISH",
            confidence=0.6,
            reasoning="momentum",
            signal_category="directional",
            data_richness="full",
            contradictions="none",
            key_levels={"resistance": 66000.0, "support": 64000.0},
            pattern_detected=None,
            raw_output="...",
        ),
    ]


def _make_conviction_agent(
    macro_path: Path, llm: _RecordingLLM | None = None
) -> ConvictionAgent:
    return ConvictionAgent(
        llm_provider=llm or _RecordingLLM(),
        macro_regime_path=macro_path,
        clock=lambda: _FIXED_NOW,
    )


# ---------------------------------------------------------------------------
# ConvictionAgent: blackout window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blackout_forces_conviction_zero(tmp_path: Path):
    path = _write_macro(tmp_path, _make_macro_payload(regime="RISK_OFF", blackout=True))
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    out = await agent.evaluate(
        signals=_make_signals(),
        market_data=_make_market_data(),
        memory_context="",
    )
    assert out.conviction_score == 0.0
    assert out.direction == "SKIP"
    assert out.macro_blackout_reason == "FOMC_ANNOUNCEMENT"
    assert "Blackout window active" in out.reasoning
    assert "FOMC_ANNOUNCEMENT" in out.reasoning
    # LLM should NOT have been called — blackout short-circuits.
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_blackout_outside_window_proceeds(tmp_path: Path):
    payload = _make_macro_payload(regime="RISK_OFF", blackout=True)
    # Move the blackout window 5h into the future so _FIXED_NOW is outside it.
    future = _FIXED_NOW + timedelta(hours=5)
    payload["blackout_windows"] = [
        {
            "start": _iso(future - timedelta(minutes=60)),
            "end": _iso(future + timedelta(minutes=30)),
            "reason": "FOMC_ANNOUNCEMENT",
            "action": "execution_block",
        }
    ]
    path = _write_macro(tmp_path, payload)
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    out = await agent.evaluate(_make_signals(), _make_market_data(), "")
    # LLM was called normally; conviction is the LLM's value.
    assert llm.call_count == 1
    assert out.conviction_score == pytest.approx(0.78)
    assert out.direction == "LONG"
    # No blackout reason stamped.
    assert out.macro_blackout_reason is None


# ---------------------------------------------------------------------------
# ConvictionAgent: RISK_OFF overlay (no blackout)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_off_stamps_boost_and_multiplier(tmp_path: Path):
    payload = _make_macro_payload(
        regime="RISK_OFF",
        threshold_boost=0.1,
        size_multiplier=0.7,
        avoid=["TSLA-USDC"],
        prefer=["BTC-USDC"],
    )
    path = _write_macro(tmp_path, payload)
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    out = await agent.evaluate(_make_signals(), _make_market_data(), "")
    assert out.macro_regime == "RISK_OFF"
    assert out.macro_threshold_boost == pytest.approx(0.1)
    assert out.macro_position_size_multiplier == pytest.approx(0.7)
    # Score itself is preserved (data moat fidelity).
    assert out.conviction_score == pytest.approx(0.78)


@pytest.mark.asyncio
async def test_risk_off_injects_macro_context_into_prompt(tmp_path: Path):
    payload = _make_macro_payload(
        regime="RISK_OFF",
        threshold_boost=0.1,
        size_multiplier=0.7,
        avoid=["TSLA-USDC"],
    )
    path = _write_macro(tmp_path, payload)
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    await agent.evaluate(_make_signals(), _make_market_data(), "")
    assert llm.last_user_prompt is not None
    assert "MACRO REGIME OVERLAY" in llm.last_user_prompt
    assert "RISK_OFF" in llm.last_user_prompt
    assert "TSLA-USDC" in llm.last_user_prompt


@pytest.mark.asyncio
async def test_neutral_regime_does_not_inject_context(tmp_path: Path):
    path = _write_macro(tmp_path, _make_macro_payload(regime="NEUTRAL"))
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    await agent.evaluate(_make_signals(), _make_market_data(), "")
    assert "MACRO REGIME OVERLAY" not in (llm.last_user_prompt or "")


# ---------------------------------------------------------------------------
# ConvictionAgent: missing / expired files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_macro_file_proceeds_normally(tmp_path: Path):
    path = tmp_path / "macro_regime.json"  # never written
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    out = await agent.evaluate(_make_signals(), _make_market_data(), "")
    assert llm.call_count == 1
    assert out.macro_blackout_reason is None
    assert out.macro_regime == "NEUTRAL"  # default
    assert out.macro_threshold_boost == 0.0


@pytest.mark.asyncio
async def test_expired_macro_file_ignored(tmp_path: Path):
    path = _write_macro(
        tmp_path, _make_macro_payload(regime="RISK_OFF", blackout=True, expired=True)
    )
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    out = await agent.evaluate(_make_signals(), _make_market_data(), "")
    # Even though the file declares a blackout, the file is expired
    # → ignored. The LLM call goes through and produces a normal score.
    assert llm.call_count == 1
    assert out.conviction_score == pytest.approx(0.78)
    assert out.macro_blackout_reason is None


@pytest.mark.asyncio
async def test_corrupt_macro_file_proceeds_normally(tmp_path: Path):
    path = tmp_path / "macro_regime.json"
    path.write_text("not json {")
    llm = _RecordingLLM()
    agent = _make_conviction_agent(path, llm=llm)
    out = await agent.evaluate(_make_signals(), _make_market_data(), "")
    assert llm.call_count == 1
    assert out.macro_blackout_reason is None


# ---------------------------------------------------------------------------
# Sentinel: blackout suppression
# ---------------------------------------------------------------------------


class _MockSentinelAdapter(ExchangeAdapter):
    """Minimal adapter that always returns volume-spike candles to trigger setup."""

    def __init__(self) -> None:
        self._candles = self._build_candles()

    @staticmethod
    def _build_candles() -> list[dict]:
        candles = []
        for i in range(50):
            base = 65000.0 + i * 5.0
            candles.append({
                "timestamp": 1700000000 + i * 3600,
                "open": base - 15,
                "high": base + 50,
                "low": base - 50,
                "close": base,
                "volume": 1000.0 + (5000.0 if i == 49 else 0),
            })
        return candles

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market"], supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100):
        return self._candles[:limit]

    async def get_ticker(self, symbol):
        return {"last": 65290.0}

    async def get_balance(self):
        return 10000.0

    async def get_positions(self, symbol=None):
        return []

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=True, order_id="m-1", fill_price=65290.0, fill_size=size, error=None)

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=True, order_id="l-1", fill_price=price, fill_size=size, error=None)

    async def place_sl_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id="sl-1", fill_price=None, fill_size=None, error=None)

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id="tp-1", fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol, order_id):
        return True

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=True, order_id="c-1", fill_price=65290.0, fill_size=0.1, error=None)

    async def modify_sl(self, symbol, new_price):
        return OrderResult(success=True, order_id="sl-m", fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=True, order_id="tp-m", fill_price=None, fill_size=None, error=None)

    async def get_funding_rate(self, symbol):
        return 0.0001

    async def get_open_interest(self, symbol):
        return 500_000_000.0


def _make_sentinel(macro_path: Path, bus: InProcessBus) -> SentinelMonitor:
    return SentinelMonitor(
        adapter=_MockSentinelAdapter(),
        event_bus=bus,
        symbol="BTC-USDC",
        threshold=0.15,  # low threshold so volume spike triggers
        candle_window=50,
        macro_regime_path=macro_path,
        clock=lambda: _FIXED_NOW,
    )


@pytest.mark.asyncio
async def test_sentinel_suppresses_setup_during_blackout(tmp_path: Path):
    path = _write_macro(tmp_path, _make_macro_payload(blackout=True))
    bus = InProcessBus()
    received: list[SetupDetected] = []
    bus.subscribe(SetupDetected, lambda e: received.append(e))

    sentinel = _make_sentinel(path, bus)
    score, conds = await sentinel.check_once()
    # Setup score is high enough to fire …
    assert score >= 0.15
    # … but no event was emitted.
    assert received == []
    # Daily budget was NOT consumed (we returned BEFORE incrementing).
    assert sentinel.daily_triggers_remaining == sentinel._daily_budget


@pytest.mark.asyncio
async def test_sentinel_emits_setup_outside_blackout(tmp_path: Path):
    # No blackout → emission proceeds normally.
    path = _write_macro(tmp_path, _make_macro_payload(blackout=False))
    bus = InProcessBus()
    received: list[SetupDetected] = []
    bus.subscribe(SetupDetected, lambda e: received.append(e))

    sentinel = _make_sentinel(path, bus)
    score, _ = await sentinel.check_once()
    assert score >= 0.15
    assert len(received) == 1
    assert received[0].symbol == "BTC-USDC"


@pytest.mark.asyncio
async def test_sentinel_no_macro_file_emits_normally(tmp_path: Path):
    path = tmp_path / "macro_regime.json"  # never written
    bus = InProcessBus()
    received: list[SetupDetected] = []
    bus.subscribe(SetupDetected, lambda e: received.append(e))

    sentinel = _make_sentinel(path, bus)
    await sentinel.check_once()
    assert len(received) == 1


@pytest.mark.asyncio
async def test_sentinel_expired_macro_file_emits_normally(tmp_path: Path):
    path = _write_macro(
        tmp_path, _make_macro_payload(blackout=True, expired=True)
    )
    bus = InProcessBus()
    received: list[SetupDetected] = []
    bus.subscribe(SetupDetected, lambda e: received.append(e))

    sentinel = _make_sentinel(path, bus)
    await sentinel.check_once()
    # Expired blackout → no suppression.
    assert len(received) == 1
