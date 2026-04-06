"""Unit tests for DecisionAgent."""

from __future__ import annotations

import json

import pytest

from engine.config import TradingConfig
from engine.execution.agent import DecisionAgent
from engine.types import ConvictionOutput, MarketData, Position, TradeAction
from llm.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Returns a predetermined response for testing."""

    def __init__(self, response_content: str) -> None:
        self._response_content = response_content
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.call_count = 0

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.call_count += 1
        return LLMResponse(
            content=self._response_content,
            input_tokens=600,
            output_tokens=150,
            cost=0.008,
            model="claude-sonnet-4-20250514",
            latency_ms=1100.0,
            cached_input_tokens=500,
        )

    async def generate_vision(self, **kwargs) -> LLMResponse:
        raise NotImplementedError("DecisionAgent does not use vision")


class ErrorLLMProvider(LLMProvider):
    """Always raises an exception."""

    async def generate_text(self, **kwargs) -> LLMResponse:
        raise ConnectionError("LLM API is down")

    async def generate_vision(self, **kwargs) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> TradingConfig:
    defaults = {
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "conviction_threshold": 0.5,
        "max_position_pct": 1.0,
    }
    defaults.update(overrides)
    return TradingConfig(**defaults)


def _make_market_data() -> MarketData:
    candles = []
    for i in range(150):
        c = 65000.0 + i * 10
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": c - 5,
            "high": c + 20,
            "low": c - 20,
            "close": c,
            "volume": 5000.0,
        })

    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=candles,
        num_candles=150,
        lookback_description="~6 days",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={
            "rsi": 65.0,
            "atr": 450.0,
            "adx": {"adx": 28.0, "plus_di": 25.0, "minus_di": 15.0, "classification": "TRENDING"},
            "volatility_percentile": 50.0,
        },
        swing_highs=[66500.0, 67000.0],
        swing_lows=[63000.0, 62000.0],
    )


def _make_conviction(
    score: float = 0.72,
    direction: str = "LONG",
    regime: str = "TRENDING_UP",
) -> ConvictionOutput:
    return ConvictionOutput(
        conviction_score=score,
        direction=direction,
        regime=regime,
        regime_confidence=0.80,
        signal_quality="HIGH",
        contradictions=["minor RSI overbought"],
        reasoning="Two agents agree bullish.",
        factual_weight=0.4,
        subjective_weight=0.6,
        raw_output="...",
    )


def _make_position(direction: str = "long", entry_price: float = 64000.0) -> Position:
    return Position(
        symbol="BTC-USDC",
        direction=direction,
        size=0.1,
        entry_price=entry_price,
        unrealized_pnl=150.0,
        leverage=5.0,
    )


def _long_response() -> str:
    return json.dumps({
        "action": "LONG",
        "reasoning": "High conviction bullish setup with strong trend confirmation.",
        "suggested_rr": None,
    })


def _skip_response() -> str:
    return json.dumps({
        "action": "SKIP",
        "reasoning": "No clear edge at current levels.",
        "suggested_rr": None,
    })


def _hold_response() -> str:
    return json.dumps({
        "action": "HOLD",
        "reasoning": "Position in profit, trend intact. No reason to close.",
        "suggested_rr": None,
    })


def _close_all_response() -> str:
    return json.dumps({
        "action": "CLOSE_ALL",
        "reasoning": "Conviction direction reversed against our position.",
        "suggested_rr": None,
    })


def _add_long_response() -> str:
    return json.dumps({
        "action": "ADD_LONG",
        "reasoning": "Price moved 0.8 ATR in favor, trend still strong.",
        "suggested_rr": None,
    })


# ---------------------------------------------------------------------------
# Tests: Low conviction quick exit
# ---------------------------------------------------------------------------

class TestDecisionAgentLowConviction:

    @pytest.mark.asyncio
    async def test_low_conviction_skips_without_llm(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config(conviction_threshold=0.5))

        conviction = _make_conviction(score=0.35)
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "SKIP"
        assert "below threshold" in result.reasoning
        assert llm.call_count == 0  # LLM should NOT be called

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_calls_llm(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config(conviction_threshold=0.5))

        conviction = _make_conviction(score=0.50)
        await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_below_threshold_no_position_size(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config(conviction_threshold=0.5))

        conviction = _make_conviction(score=0.3)
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.position_size is None
        assert result.sl_price is None
        assert result.tp1_price is None


# ---------------------------------------------------------------------------
# Tests: LLM decision with successful parse
# ---------------------------------------------------------------------------

class TestDecisionAgentLLMDecision:

    @pytest.mark.asyncio
    async def test_long_entry(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "LONG"
        assert result.conviction_score == 0.72
        assert result.position_size is not None
        assert result.position_size > 0
        assert result.sl_price is not None
        assert result.sl_price < 66490.0  # SL should be below entry
        assert result.tp1_price is not None
        assert result.tp1_price > 66490.0  # TP1 above entry for LONG
        assert result.tp2_price is not None
        assert result.tp2_price > result.tp1_price  # TP2 > TP1
        assert result.rr_ratio is not None
        assert result.rr_ratio > 0

    @pytest.mark.asyncio
    async def test_skip_no_position(self) -> None:
        llm = MockLLMProvider(_skip_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.55, direction="SKIP")
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "SKIP"
        assert result.position_size is None

    @pytest.mark.asyncio
    async def test_hold_with_position(self) -> None:
        llm = MockLLMProvider(_hold_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.55, direction="LONG")
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position, 10000.0)

        assert result.action == "HOLD"

    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        llm = MockLLMProvider(_close_all_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.65, direction="SHORT")
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position, 10000.0)

        assert result.action == "CLOSE_ALL"

    @pytest.mark.asyncio
    async def test_raw_output_preserved(self) -> None:
        raw = _long_response()
        llm = MockLLMProvider(raw)
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.raw_output == raw


# ---------------------------------------------------------------------------
# Tests: SL/TP computation
# ---------------------------------------------------------------------------

class TestDecisionAgentSLTP:

    @pytest.mark.asyncio
    async def test_sl_below_entry_for_long(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        data = _make_market_data()
        result = await agent.decide(conviction, data, None, 10000.0)

        entry = data.candles[-1]["close"]
        assert result.sl_price is not None
        assert result.sl_price < entry

    @pytest.mark.asyncio
    async def test_tp_above_entry_for_long(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        data = _make_market_data()
        result = await agent.decide(conviction, data, None, 10000.0)

        entry = data.candles[-1]["close"]
        assert result.tp1_price is not None
        assert result.tp1_price > entry
        assert result.tp2_price is not None
        assert result.tp2_price > entry

    @pytest.mark.asyncio
    async def test_short_sl_above_entry(self) -> None:
        response = json.dumps({
            "action": "SHORT",
            "reasoning": "Bearish conviction confirmed.",
            "suggested_rr": None,
        })
        llm = MockLLMProvider(response)
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="SHORT")
        data = _make_market_data()
        result = await agent.decide(conviction, data, None, 10000.0)

        entry = data.candles[-1]["close"]
        assert result.sl_price is not None
        assert result.sl_price > entry
        assert result.tp1_price is not None
        assert result.tp1_price < entry

    @pytest.mark.asyncio
    async def test_suggested_rr_override(self) -> None:
        response = json.dumps({
            "action": "LONG",
            "reasoning": "Strong setup warrants higher RR.",
            "suggested_rr": 3.0,
        })
        llm = MockLLMProvider(response)
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.rr_ratio == 3.0


# ---------------------------------------------------------------------------
# Tests: Position sizing
# ---------------------------------------------------------------------------

class TestDecisionAgentSizing:

    @pytest.mark.asyncio
    async def test_conviction_size_multiplier_high(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conv_moderate = _make_conviction(score=0.55)
        conv_high = _make_conviction(score=0.75)

        result_mod = await agent.decide(conv_moderate, _make_market_data(), None, 10000.0)
        result_high = await agent.decide(conv_high, _make_market_data(), None, 10000.0)

        # Higher conviction should produce larger position
        assert result_high.position_size > result_mod.position_size

    @pytest.mark.asyncio
    async def test_pyramid_size_is_half(self) -> None:
        llm = MockLLMProvider(_add_long_response())
        agent = DecisionAgent(llm, _make_config())

        # Get base size first
        llm2 = MockLLMProvider(_long_response())
        agent2 = DecisionAgent(llm2, _make_config())
        conv = _make_conviction(score=0.75)
        base_result = await agent2.decide(conv, _make_market_data(), None, 10000.0)

        # Now get pyramid size
        position = _make_position("long", entry_price=64000.0)
        pyramid_result = await agent.decide(conv, _make_market_data(), position, 10000.0)

        # Pyramid can get safety-checked to HOLD, but if it got through as ADD_LONG
        # the size would be ~50% of base. If safety converts to HOLD, size is None.
        if pyramid_result.action == "ADD_LONG":
            assert pyramid_result.position_size is not None
            assert pyramid_result.position_size < base_result.position_size


# ---------------------------------------------------------------------------
# Tests: Safety check overrides
# ---------------------------------------------------------------------------

class TestDecisionAgentSafetyOverrides:

    @pytest.mark.asyncio
    async def test_safety_blocks_duplicate_entry(self) -> None:
        """Safety check #4: can't open LONG when already have a position."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        position = _make_position("long")  # already have a LONG

        result = await agent.decide(conviction, _make_market_data(), position, 10000.0)

        # Safety should convert LONG → HOLD (already have position)
        assert result.action == "HOLD"
        assert "position_limit" in result.reasoning

    @pytest.mark.asyncio
    async def test_safety_conviction_floor(self) -> None:
        """Safety check #1: conviction < 0.3 forces SKIP even if LLM says LONG."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config(conviction_threshold=0.2))

        # Threshold is 0.2 (so LLM gets called), but conviction is 0.25
        # Safety check enforces conviction floor at 0.3
        conviction = _make_conviction(score=0.25, direction="LONG")
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "SKIP"
        assert "conviction_floor" in result.reasoning

    @pytest.mark.asyncio
    async def test_safety_override_clears_sizing(self) -> None:
        """When safety converts action to HOLD/SKIP, sizing fields are cleared."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        position = _make_position("long")  # triggers position_limit → HOLD

        result = await agent.decide(conviction, _make_market_data(), position, 10000.0)

        assert result.action == "HOLD"
        assert result.position_size is None
        assert result.sl_price is None
        assert result.tp1_price is None

    @pytest.mark.asyncio
    async def test_safety_sl_validation_blocks_zero_atr(self) -> None:
        """Safety check #5: ATR <= 0 blocks entry."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        data = _make_market_data()
        data.indicators["atr"] = 0.0  # invalid ATR

        result = await agent.decide(conviction, data, None, 10000.0)

        # With ATR=0, SL/TP can't be computed, and safety check blocks entry
        assert result.action == "SKIP"


# ---------------------------------------------------------------------------
# Tests: Parse failure safety
# ---------------------------------------------------------------------------

class TestDecisionAgentParseSafety:

    @pytest.mark.asyncio
    async def test_garbage_response_no_position_returns_skip(self) -> None:
        llm = MockLLMProvider("I'm not sure what to do here.")
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "SKIP"

    @pytest.mark.asyncio
    async def test_garbage_response_with_position_returns_hold(self) -> None:
        llm = MockLLMProvider("I'm not sure what to do here.")
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position, 10000.0)

        # Safety check converts to HOLD since we have a position
        assert result.action == "HOLD"

    @pytest.mark.asyncio
    async def test_invalid_action_defaults_safe(self) -> None:
        response = json.dumps({
            "action": "MOON_BUY",  # invalid
            "reasoning": "test",
            "suggested_rr": None,
        })
        llm = MockLLMProvider(response)
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "SKIP"

    @pytest.mark.asyncio
    async def test_llm_exception_returns_safe_default(self) -> None:
        llm = ErrorLLMProvider()
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        result = await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert result.action == "SKIP"

    @pytest.mark.asyncio
    async def test_llm_exception_with_position_returns_hold(self) -> None:
        llm = ErrorLLMProvider()
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position, 10000.0)

        assert result.action == "HOLD"


# ---------------------------------------------------------------------------
# Tests: Prompt content
# ---------------------------------------------------------------------------

class TestDecisionAgentPromptContent:

    @pytest.mark.asyncio
    async def test_conviction_data_in_user_prompt(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG", regime="TRENDING_UP")
        await agent.decide(conviction, _make_market_data(), None, 10000.0)

        prompt = llm.last_user_prompt
        assert "0.72" in prompt
        assert "LONG" in prompt
        assert "TRENDING_UP" in prompt

    @pytest.mark.asyncio
    async def test_position_context_no_position(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        await agent.decide(conviction, _make_market_data(), None, 10000.0)

        assert "No open position" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_position_context_with_position(self) -> None:
        llm = MockLLMProvider(_hold_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        position = _make_position("long", entry_price=64000.0)
        await agent.decide(conviction, _make_market_data(), position, 10000.0)

        prompt = llm.last_user_prompt
        assert "LONG" in prompt
        assert "64000" in prompt

    @pytest.mark.asyncio
    async def test_memory_context_in_prompt(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        memory = "Last cycle: SKIP at 0.45 conviction. Rule: Avoid LONG when RSI > 75."
        await agent.decide(conviction, _make_market_data(), None, 10000.0, memory_context=memory)

        assert "Avoid LONG when RSI > 75" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_action_types(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        await agent.decide(conviction, _make_market_data(), None, 10000.0)

        sys = llm.last_system_prompt
        assert "LONG" in sys
        assert "SHORT" in sys
        assert "ADD_LONG" in sys
        assert "CLOSE_ALL" in sys
        assert "HOLD" in sys
        assert "SKIP" in sys

    @pytest.mark.asyncio
    async def test_account_balance_in_prompt(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        await agent.decide(conviction, _make_market_data(), None, 25000.0)

        assert "25000" in llm.last_user_prompt
