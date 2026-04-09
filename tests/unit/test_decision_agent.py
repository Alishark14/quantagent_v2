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
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "SKIP"
        assert "below threshold" in result.reasoning
        assert llm.call_count == 0  # LLM should NOT be called

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_calls_llm(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config(conviction_threshold=0.5))

        conviction = _make_conviction(score=0.50)
        await agent.decide(conviction, _make_market_data(), None)

        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_below_threshold_no_sl_or_risk_weight(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config(conviction_threshold=0.5))

        conviction = _make_conviction(score=0.3)
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.position_size is None
        assert result.sl_price is None
        assert result.tp1_price is None
        assert result.risk_weight is None


# ---------------------------------------------------------------------------
# Tests: LLM decision with successful parse
# ---------------------------------------------------------------------------

class TestDecisionAgentLLMDecision:

    @pytest.mark.asyncio
    async def test_long_entry(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "LONG"
        assert result.conviction_score == 0.72
        # DecisionAgent never sets position_size — PRM owns sizing.
        assert result.position_size is None
        assert result.sl_price is not None
        assert result.sl_price < 66490.0  # SL should be below entry
        assert result.tp1_price is not None
        assert result.tp1_price > 66490.0  # TP1 above entry for LONG
        assert result.tp2_price is not None
        assert result.tp2_price > result.tp1_price  # TP2 > TP1
        assert result.rr_ratio is not None
        assert result.rr_ratio > 0
        # Conviction 0.72 → "high" tier → risk_weight 1.15
        assert result.risk_weight == pytest.approx(1.15)

    @pytest.mark.asyncio
    async def test_skip_no_position(self) -> None:
        llm = MockLLMProvider(_skip_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.55, direction="SKIP")
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "SKIP"
        assert result.position_size is None
        # SKIP is not an entry — risk_weight stays None (PRM never runs).
        assert result.risk_weight is None

    @pytest.mark.asyncio
    async def test_hold_with_position(self) -> None:
        llm = MockLLMProvider(_hold_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.55, direction="LONG")
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position)

        assert result.action == "HOLD"

    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        llm = MockLLMProvider(_close_all_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.65, direction="SHORT")
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position)

        assert result.action == "CLOSE_ALL"

    @pytest.mark.asyncio
    async def test_raw_output_preserved(self) -> None:
        raw = _long_response()
        llm = MockLLMProvider(raw)
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        result = await agent.decide(conviction, _make_market_data(), None)

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
        result = await agent.decide(conviction, data, None)

        entry = data.candles[-1]["close"]
        assert result.sl_price is not None
        assert result.sl_price < entry

    @pytest.mark.asyncio
    async def test_tp_above_entry_for_long(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        data = _make_market_data()
        result = await agent.decide(conviction, data, None)

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
        result = await agent.decide(conviction, data, None)

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
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.rr_ratio == 3.0


# ---------------------------------------------------------------------------
# Tests: risk_weight derived from conviction (Sprint Portfolio-Risk-Manager Task 1)
#
# DecisionAgent no longer outputs dollar sizes — it attaches a deterministic
# risk_weight (0.75 / 1.0 / 1.15 / 1.3) computed from the conviction score in
# plain Python. PortfolioRiskManager consumes this weight downstream when it
# computes the actual position size in USD.
# ---------------------------------------------------------------------------

class TestDecisionAgentRiskWeight:

    @pytest.mark.asyncio
    async def test_risk_weight_low_band(self) -> None:
        """conviction 0.50 - 0.60 → risk_weight 0.75."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())
        result = await agent.decide(_make_conviction(score=0.55), _make_market_data(), None)
        assert result.risk_weight == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_risk_weight_moderate_band(self) -> None:
        """conviction 0.60 - 0.70 → risk_weight 1.0."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())
        result = await agent.decide(_make_conviction(score=0.65), _make_market_data(), None)
        assert result.risk_weight == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_risk_weight_high_band(self) -> None:
        """conviction 0.70 - 0.85 → risk_weight 1.15."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())
        result = await agent.decide(_make_conviction(score=0.78), _make_market_data(), None)
        assert result.risk_weight == pytest.approx(1.15)

    @pytest.mark.asyncio
    async def test_risk_weight_very_high_band(self) -> None:
        """conviction 0.85+ → risk_weight 1.3."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())
        result = await agent.decide(_make_conviction(score=0.90), _make_market_data(), None)
        assert result.risk_weight == pytest.approx(1.30)

    @pytest.mark.asyncio
    async def test_risk_weight_only_set_for_entry_actions(self) -> None:
        """SKIP / HOLD / CLOSE_ALL must NOT carry a risk_weight — there
        is no trade to size."""
        llm = MockLLMProvider(_skip_response())
        agent = DecisionAgent(llm, _make_config())
        result = await agent.decide(_make_conviction(score=0.72), _make_market_data(), None)
        assert result.action == "SKIP"
        assert result.risk_weight is None

    @pytest.mark.asyncio
    async def test_position_size_always_none_from_decision_agent(self) -> None:
        """DecisionAgent must NEVER output a dollar position size — that's
        PortfolioRiskManager's job. Pins the contract that the field comes
        out as None for every action including LONG/SHORT/ADD_LONG."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())
        result = await agent.decide(
            _make_conviction(score=0.75), _make_market_data(), None
        )
        assert result.action == "LONG"
        assert result.position_size is None

    def test_risk_weight_function_directly(self) -> None:
        """The pure-Python helper itself — no LLM, no async, no agent.
        Locks the boundary cases that the band tests above can't hit
        without contriving a conviction at the exact boundary."""
        from engine.execution.agent import risk_weight_from_conviction

        # Below 0.60 → 0.75 (lowest valid band; below 0.50 the engine SKIPs first)
        assert risk_weight_from_conviction(0.50) == pytest.approx(0.75)
        assert risk_weight_from_conviction(0.59) == pytest.approx(0.75)
        # 0.60 - 0.70 → 1.0
        assert risk_weight_from_conviction(0.60) == pytest.approx(1.0)
        assert risk_weight_from_conviction(0.69) == pytest.approx(1.0)
        # 0.70 - 0.85 → 1.15
        assert risk_weight_from_conviction(0.70) == pytest.approx(1.15)
        assert risk_weight_from_conviction(0.84) == pytest.approx(1.15)
        # 0.85+ → 1.3
        assert risk_weight_from_conviction(0.85) == pytest.approx(1.30)
        assert risk_weight_from_conviction(1.00) == pytest.approx(1.30)


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

        result = await agent.decide(conviction, _make_market_data(), position)

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
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "SKIP"
        assert "conviction_floor" in result.reasoning

    @pytest.mark.asyncio
    async def test_safety_override_clears_sl_tp_and_risk_weight(self) -> None:
        """When safety converts an entry action to HOLD/SKIP, the SL/TP
        levels and risk_weight must be cleared so PRM can't size a trade
        the safety check just blocked."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        position = _make_position("long")  # triggers position_limit → HOLD

        result = await agent.decide(conviction, _make_market_data(), position)

        assert result.action == "HOLD"
        assert result.position_size is None
        assert result.sl_price is None
        assert result.tp1_price is None
        assert result.risk_weight is None

    @pytest.mark.asyncio
    async def test_safety_sl_validation_blocks_zero_atr(self) -> None:
        """Safety check #5: ATR <= 0 blocks entry."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72, direction="LONG")
        data = _make_market_data()
        data.indicators["atr"] = 0.0  # invalid ATR

        result = await agent.decide(conviction, data, None)

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
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "SKIP"

    @pytest.mark.asyncio
    async def test_garbage_response_with_position_returns_hold(self) -> None:
        llm = MockLLMProvider("I'm not sure what to do here.")
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position)

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
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "SKIP"

    @pytest.mark.asyncio
    async def test_llm_exception_returns_safe_default(self) -> None:
        llm = ErrorLLMProvider()
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        result = await agent.decide(conviction, _make_market_data(), None)

        assert result.action == "SKIP"

    @pytest.mark.asyncio
    async def test_llm_exception_with_position_returns_hold(self) -> None:
        llm = ErrorLLMProvider()
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        position = _make_position("long")
        result = await agent.decide(conviction, _make_market_data(), position)

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
        await agent.decide(conviction, _make_market_data(), None)

        prompt = llm.last_user_prompt
        assert "0.72" in prompt
        assert "LONG" in prompt
        assert "TRENDING_UP" in prompt

    @pytest.mark.asyncio
    async def test_position_context_no_position(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        await agent.decide(conviction, _make_market_data(), None)

        assert "No open position" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_position_context_with_position(self) -> None:
        llm = MockLLMProvider(_hold_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        position = _make_position("long", entry_price=64000.0)
        await agent.decide(conviction, _make_market_data(), position)

        prompt = llm.last_user_prompt
        assert "LONG" in prompt
        assert "64000" in prompt

    @pytest.mark.asyncio
    async def test_memory_context_in_prompt(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        memory = "Last cycle: SKIP at 0.45 conviction. Rule: Avoid LONG when RSI > 75."
        await agent.decide(conviction, _make_market_data(), None, memory_context=memory)

        assert "Avoid LONG when RSI > 75" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_action_types(self) -> None:
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        await agent.decide(conviction, _make_market_data(), None)

        sys = llm.last_system_prompt
        assert "LONG" in sys
        assert "SHORT" in sys
        assert "ADD_LONG" in sys
        assert "CLOSE_ALL" in sys
        assert "HOLD" in sys
        assert "SKIP" in sys

    @pytest.mark.asyncio
    async def test_account_balance_not_in_prompt(self) -> None:
        """DecisionAgent must NOT see account balance — sizing is the
        PortfolioRiskManager's job (Sprint Portfolio-Risk-Manager Task 1).

        Pins the contract that the LLM prompt contains no dollar amount,
        no Balance: line, no equity reference. A future regression that
        re-adds balance to the prompt would break this assertion."""
        llm = MockLLMProvider(_long_response())
        agent = DecisionAgent(llm, _make_config())

        conviction = _make_conviction(score=0.72)
        await agent.decide(conviction, _make_market_data(), None)

        prompt = llm.last_user_prompt or ""
        assert "Balance" not in prompt
        assert "$" not in prompt
        assert "account_balance" not in prompt
