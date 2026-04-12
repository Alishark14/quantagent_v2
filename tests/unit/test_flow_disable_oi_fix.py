"""Tests for FlowAgent disable, OI deque fix, and ConvictionAgent prompt changes.

Change 1: FlowAgent disabled → 3 signals only, no flow in signals_block.
Change 2: OI deque sized for WebSocket push rate; reasoning uses dynamic lookback.
Change 3: ConvictionAgent prompt version 1.3 with CONSENSUS FLOOR.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.config import FeatureFlags
from engine.data.flow.crypto import CryptoFlowProvider, _maxlen_for_lookback
from engine.data.flow.signal_agent import FlowSignalAgent
from engine.signals.registry import SignalRegistry
from engine.types import FlowOutput, MarketData, SignalOutput


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signal(name: str, direction: str = "BULLISH") -> SignalOutput:
    return SignalOutput(
        agent_name=name,
        signal_type="llm",
        direction=direction,
        confidence=0.7,
        reasoning="test",
        signal_category="directional",
        data_richness="full",
        contradictions="",
        key_levels={},
        pattern_detected=None,
        raw_output="test",
    )


def _make_flow_output(**overrides) -> FlowOutput:
    defaults = dict(
        funding_rate=0.001,
        funding_signal="NEUTRAL",
        oi_change_4h=None,
        oi_trend="STABLE",
        nearest_liquidation_above=None,
        nearest_liquidation_below=None,
        gex_regime=None,
        gex_flip_level=None,
        data_richness="PARTIAL",
    )
    defaults.update(overrides)
    return FlowOutput(**defaults)


def _make_market_data(candles_count: int = 20) -> MarketData:
    candles = [
        {"timestamp": i, "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0 + (i * 0.01), "volume": 1000}
        for i in range(candles_count)
    ]
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=candles,
        num_candles=candles_count,
        lookback_description="~1 day",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={"atr": 5.0, "rsi": 50.0},
        swing_highs=[],
        swing_lows=[],
        flow=_make_flow_output(),
    )


# ── Change 1: FlowAgent disabled ────────────────────────────────────


class StubProducer:
    """Minimal stub implementing the SignalProducer interface."""

    def __init__(self, agent_name: str, enabled: bool = True):
        self._name = agent_name
        self._enabled = enabled

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        return self._enabled

    def requires_vision(self) -> bool:
        return False

    async def analyze(self, data):
        return _make_signal(self._name)


def _make_flags(flow_enabled: bool, tmp_path) -> FeatureFlags:
    """Create FeatureFlags from a temp YAML file."""
    p = tmp_path / "features.yaml"
    p.write_text(f"flow_signal_agent: {'true' if flow_enabled else 'false'}\n")
    return FeatureFlags(p)


def test_flow_signal_agent_disabled_by_feature_flag(tmp_path):
    """When flow_signal_agent flag is false, is_enabled returns False."""
    flags = _make_flags(False, tmp_path)
    agent = FlowSignalAgent(flags)
    assert agent.is_enabled() is False


def test_flow_signal_agent_enabled_by_feature_flag(tmp_path):
    """When flow_signal_agent flag is true, is_enabled returns True."""
    flags = _make_flags(True, tmp_path)
    agent = FlowSignalAgent(flags)
    assert agent.is_enabled() is True


@pytest.mark.asyncio
async def test_registry_excludes_disabled_flow_agent():
    """SignalRegistry.run_all skips disabled producers."""
    registry = SignalRegistry()
    registry.register(StubProducer("indicator_agent", enabled=True))
    registry.register(StubProducer("pattern_agent", enabled=True))
    registry.register(StubProducer("trend_agent", enabled=True))
    registry.register(StubProducer("flow_signal_agent", enabled=False))

    data = _make_market_data()
    signals = await registry.run_all(data)

    names = [s.agent_name for s in signals]
    assert len(names) == 3
    assert "flow_signal_agent" not in names
    assert "indicator_agent" in names
    assert "pattern_agent" in names
    assert "trend_agent" in names


@pytest.mark.asyncio
async def test_conviction_receives_3_signals_when_flow_disabled():
    """ConvictionAgent's signals_block should not contain flow when disabled."""
    from engine.conviction.agent import _AGENT_DISPLAY_NAMES, ConvictionAgent

    # Build signal_map with 3 agents (no flow) — full dict shape
    _sig = {
        "direction": "BEARISH", "confidence": 0.7, "reasoning": "test",
        "contradictions": "", "pattern": "none", "key_levels": {},
    }
    signal_map = {
        "indicator_agent": {**_sig},
        "pattern_agent": {**_sig, "confidence": 0.6},
        "trend_agent": {**_sig, "confidence": 0.65},
    }

    # Use a dummy provider (we only test the block builder)
    llm = MagicMock()
    agent = ConvictionAgent(llm)
    block = agent._build_signals_block(signal_map)

    # Flow should show "did not produce a signal" (known agent, not in signal_map)
    assert "FlowAgent" in block
    assert "did not produce a signal" in block

    # The 3 LLM agents should appear with their data
    assert "IndicatorAgent" in block
    assert "PatternAgent" in block
    assert "TrendAgent" in block


# ── Change 2: OI deque fix ──────────────────────────────────────────


def test_maxlen_for_lookback_handles_websocket_rate():
    """Deque maxlen must be large enough for WebSocket pushes (~3s interval).

    Old formula: lookback_seconds // 30 = 240 for 2h lookback.
    New formula: max(600, lookback_seconds) = 7200 for 2h lookback.
    """
    # 2h lookback
    assert _maxlen_for_lookback(7200) >= 7200
    # 8h lookback
    assert _maxlen_for_lookback(28800) >= 28800
    # Small lookback (should floor at 600)
    assert _maxlen_for_lookback(100) == 600


@pytest.mark.asyncio
async def test_warmup_then_fetch_produces_oi_change():
    """After warmup_from_repo populates the deque, oi_change_pct must NOT be None."""
    provider = CryptoFlowProvider(lookback_seconds=7200)

    # Simulate warmup by manually populating the deque with data spanning
    # the lookback window: entries from 3h ago to now at 30s intervals.
    now = time.time()
    symbol = "BTC-USDC"
    buf = deque(maxlen=_maxlen_for_lookback(7200))
    # Fill with data from 3h ago (older than cutoff) to now
    for i in range(360):  # 3 hours at 30s intervals
        ts = now - (360 - i) * 30
        buf.append((ts, 1_000_000.0 + i * 100))  # OI slowly increasing
    provider._oi_history[symbol] = buf

    # Now compute delta — should NOT return None
    current_oi = 1_050_000.0
    oi_change, oi_trend = provider._compute_oi_delta(symbol, now, current_oi)

    assert oi_change is not None, "oi_change should not be None after warmup"
    assert oi_change > 0, "OI was increasing, should show positive change"
    assert oi_trend in ("BUILDING", "STABLE")


@pytest.mark.asyncio
async def test_on_oi_update_appends_to_deque():
    """_on_oi_update from WebSocket pushes must append to the same deque fetch reads."""
    from datetime import datetime, timezone
    from engine.types import OIUpdate

    provider = CryptoFlowProvider(lookback_seconds=7200)

    # Simulate an OI update event
    class FakeEvent:
        update = OIUpdate(
            symbol="BTC-USDC",
            open_interest=5_000_000.0,
            timestamp=datetime.now(timezone.utc),
        )

    await provider._on_oi_update(FakeEvent())

    buf = provider._oi_history.get("BTC-USDC")
    assert buf is not None
    assert len(buf) == 1
    assert buf[0][1] == 5_000_000.0


def test_deque_survives_high_frequency_pushes():
    """Deque maxlen must be large enough that 2h of 3s pushes don't evict
    the oldest entries needed for the lookback."""
    lookback = 7200
    maxlen = _maxlen_for_lookback(lookback)

    buf = deque(maxlen=maxlen)
    now = time.time()

    # Simulate 2 hours of 3-second WebSocket pushes
    push_count = lookback // 3  # 2400 pushes
    for i in range(push_count):
        ts = now - (push_count - i) * 3
        buf.append((ts, 1_000_000.0))

    # The oldest entry should still be >= lookback_seconds old
    oldest_ts = buf[0][0]
    cutoff = now - lookback
    assert oldest_ts <= cutoff, (
        f"Oldest entry ({now - oldest_ts:.0f}s ago) should be at or before "
        f"the cutoff ({lookback}s ago). Deque maxlen={maxlen} is too small."
    )


def test_reasoning_text_no_hardcoded_4h():
    """FlowSignalAgent reasoning should not say '4h OI history'."""
    agent = FlowSignalAgent(feature_flags=None)  # defaults to enabled

    md = _make_market_data()
    md.flow = _make_flow_output(oi_change_4h=None)

    result = agent._evaluate(md)
    assert "4h OI history" not in result.reasoning
    assert "4h" not in result.reasoning


# ── Change 3: ConvictionAgent prompt ─────────────────────────────────


def test_conviction_prompt_version_is_1_3():
    """Prompt version must be 1.3 after the CONSENSUS FLOOR update."""
    from engine.conviction.prompts.conviction_v1 import PROMPT_VERSION
    assert PROMPT_VERSION == "1.3"


def test_conviction_prompt_has_consensus_floor():
    """SYSTEM_PROMPT must contain the CONSENSUS FLOOR rule."""
    from engine.conviction.prompts.conviction_v1 import SYSTEM_PROMPT
    assert "CONSENSUS FLOOR" in SYSTEM_PROMPT
    assert "0.45" in SYSTEM_PROMPT


def test_conviction_prompt_no_flow_agent_scoring_rules():
    """SYSTEM_PROMPT must not reference FlowAgent in scoring rules."""
    from engine.conviction.prompts.conviction_v1 import SYSTEM_PROMPT
    assert "flow confirms" not in SYSTEM_PROMPT
    assert "FlowAgent contradicts" not in SYSTEM_PROMPT


def test_conviction_prompt_uncertainty_anchor_clarified():
    """UNCERTAINTY ANCHOR must state it doesn't apply to unanimous agreement."""
    from engine.conviction.prompts.conviction_v1 import SYSTEM_PROMPT
    assert "does NOT apply when all signal agents agree" in SYSTEM_PROMPT
