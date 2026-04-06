"""Integration tests for SignalRegistry — multiple producers running in parallel."""

from __future__ import annotations

import pytest

from engine.signals.base import SignalProducer
from engine.signals.registry import SignalRegistry
from engine.types import MarketData, SignalOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market_data() -> MarketData:
    return MarketData(
        symbol="ETH-USDC",
        timeframe="4h",
        candles=[],
        num_candles=0,
        lookback_description="~0",
        forecast_candles=3,
        forecast_description="~12 hours",
        indicators={},
        swing_highs=[],
        swing_lows=[],
    )


def _make_signal(agent: str, direction: str = "BULLISH", confidence: float = 0.7) -> SignalOutput:
    return SignalOutput(
        agent_name=agent,
        signal_type="llm",
        direction=direction,
        confidence=confidence,
        reasoning=f"reasoning from {agent}",
        signal_category="directional",
        data_richness="full",
        contradictions="",
        key_levels={},
        pattern_detected=None,
        raw_output="",
    )


class StubProducer(SignalProducer):
    def __init__(self, producer_name: str, result: SignalOutput | None = None, raises: bool = False) -> None:
        self._name = producer_name
        self._result = result
        self._raises = raises

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        return True

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        if self._raises:
            raise ValueError(f"{self._name} crashed")
        return self._result


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    @pytest.mark.asyncio
    async def test_three_producers_all_return(self) -> None:
        reg = SignalRegistry()
        reg.register(StubProducer("indicator_agent", _make_signal("indicator_agent", "BULLISH", 0.8)))
        reg.register(StubProducer("pattern_agent", _make_signal("pattern_agent", "BEARISH", 0.6)))
        reg.register(StubProducer("trend_agent", _make_signal("trend_agent", "NEUTRAL", 0.5)))

        results = await reg.run_all(_make_market_data())

        assert len(results) == 3
        names = {r.agent_name for r in results}
        assert names == {"indicator_agent", "pattern_agent", "trend_agent"}

        # Verify individual results preserved
        by_name = {r.agent_name: r for r in results}
        assert by_name["indicator_agent"].direction == "BULLISH"
        assert by_name["pattern_agent"].confidence == 0.6
        assert by_name["trend_agent"].direction == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_one_fails_others_still_return(self) -> None:
        reg = SignalRegistry()
        reg.register(StubProducer("good_1", _make_signal("good_1")))
        reg.register(StubProducer("bad", raises=True))
        reg.register(StubProducer("good_2", _make_signal("good_2")))

        results = await reg.run_all(_make_market_data())

        assert len(results) == 2
        names = {r.agent_name for r in results}
        assert names == {"good_1", "good_2"}

    @pytest.mark.asyncio
    async def test_all_fail_returns_empty(self) -> None:
        reg = SignalRegistry()
        reg.register(StubProducer("bad_1", raises=True))
        reg.register(StubProducer("bad_2", raises=True))

        results = await reg.run_all(_make_market_data())
        assert results == []
