"""Unit tests for ReflectionAgent."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from engine.events import InProcessBus, RuleGenerated, TradeClosed
from engine.memory.reflection_rules import ReflectionRules
from engine.reflection.agent import ReflectionAgent, create_reflection_handler
from llm.base import LLMProvider, LLMResponse
from storage.repositories.sqlite import SQLiteRepositories


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):

    def __init__(self, response_content: str) -> None:
        self._response_content = response_content
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.call_count = 0

    async def generate_text(
        self, system_prompt: str, user_prompt: str, agent_name: str, **kwargs
    ) -> LLMResponse:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.call_count += 1
        return LLMResponse(
            content=self._response_content,
            input_tokens=600, output_tokens=150, cost=0.010,
            model="mock", latency_ms=100.0, cached_input_tokens=400,
        )

    async def generate_vision(self, **kwargs) -> LLMResponse:
        raise NotImplementedError


class ErrorLLMProvider(LLMProvider):
    async def generate_text(self, **kwargs) -> LLMResponse:
        raise ConnectionError("LLM down")

    async def generate_vision(self, **kwargs) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def _trade_data(pnl: float = 150.0, exit_reason: str = "TP1") -> dict:
    return {
        "id": "trade-001",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "direction": "LONG",
        "entry_price": 65000.0,
        "exit_price": 66500.0,
        "pnl": pnl,
        "r_multiple": 1.5,
        "exit_reason": exit_reason,
        "duration": "4h 32m",
    }


def _cycle_data() -> dict:
    return {
        "signals": [
            {"agent_name": "indicator_agent", "direction": "BULLISH", "confidence": 0.72, "pattern_detected": None},
            {"agent_name": "pattern_agent", "direction": "BULLISH", "confidence": 0.80, "pattern_detected": "ascending_triangle"},
            {"agent_name": "trend_agent", "direction": "NEUTRAL", "confidence": 0.50, "pattern_detected": None},
        ],
        "conviction": {"conviction_score": 0.72, "regime": "TRENDING_UP"},
        "indicators": {
            "rsi": 73.2,
            "atr": 450.0,
            "adx": {"adx": 31.0, "plus_di": 28.0, "minus_di": 14.0},
            "volatility_percentile": 62.0,
        },
    }


def _good_rule_response() -> str:
    return json.dumps({
        "rule": "When RSI > 70 and all agents agree BULLISH with ascending_triangle at ADX > 30, LONG has high win rate — trust the trend",
        "reasoning": "This LONG trade won because strong trend (ADX 31) supported the bullish consensus despite RSI being near overbought. The ascending triangle resolved bullishly as expected.",
        "applies_to": "BTC-USDC 1h",
        "confidence": 0.75,
    })


def _null_rule_response() -> str:
    return json.dumps({
        "rule": None,
        "reasoning": "Trade outcome was ambiguous — breakeven exit via trailing stop, no clear pattern to learn from.",
        "applies_to": None,
        "confidence": 0.0,
    })


def _loss_rule_response() -> str:
    return json.dumps({
        "rule": "When TrendAgent is NEUTRAL but IndicatorAgent and PatternAgent are BULLISH, reduce position size — trend exhaustion risk",
        "reasoning": "Trade lost because trend was exhausting despite bullish signals. TrendAgent correctly flagged NEUTRAL but was overridden by 2/3 consensus. The mixed signal warranted smaller sizing.",
        "applies_to": "all",
        "confidence": 0.65,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def repos(tmp_path):
    db_path = str(tmp_path / "test_reflection.db")
    r = SQLiteRepositories(db_path=db_path)
    await r.init_db()
    return r


# ---------------------------------------------------------------------------
# Tests: Successful reflection
# ---------------------------------------------------------------------------

class TestReflectionAgentSuccess:

    @pytest.mark.asyncio
    async def test_generates_rule_from_winning_trade(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        bus = InProcessBus()
        agent = ReflectionAgent(llm, rules, bus)

        result = await agent.reflect(_trade_data(), _cycle_data())

        assert result is not None
        assert "RSI > 70" in result["rule_text"]
        assert result["symbol"] == "BTC-USDC"
        assert result["timeframe"] == "1h"
        assert result["score"] == 0
        assert result["active"] is True
        assert result["id"] is not None

    @pytest.mark.asyncio
    async def test_rule_saved_to_repo(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        bus = InProcessBus()
        agent = ReflectionAgent(llm, rules, bus)

        await agent.reflect(_trade_data(), _cycle_data())

        saved = await rules.get_active_rules("BTC-USDC", "1h")
        assert len(saved) == 1
        assert "RSI > 70" in saved[0]["rule_text"]

    @pytest.mark.asyncio
    async def test_rule_generated_event_emitted(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        bus = InProcessBus()
        agent = ReflectionAgent(llm, rules, bus)

        events: list[RuleGenerated] = []
        bus.subscribe(RuleGenerated, lambda e: events.append(e))

        await agent.reflect(_trade_data(), _cycle_data())

        assert len(events) == 1
        assert "RSI > 70" in events[0].rule["rule_text"]

    @pytest.mark.asyncio
    async def test_loss_trade_produces_rule(self, repos) -> None:
        llm = MockLLMProvider(_loss_rule_response())
        rules = ReflectionRules(repos.rules)
        bus = InProcessBus()
        agent = ReflectionAgent(llm, rules, bus)

        result = await agent.reflect(_trade_data(pnl=-200.0, exit_reason="SL"), _cycle_data())

        assert result is not None
        assert "reduce position size" in result["rule_text"]

    @pytest.mark.asyncio
    async def test_llm_called_once(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        await agent.reflect(_trade_data(), _cycle_data())

        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Null / no rule
# ---------------------------------------------------------------------------

class TestReflectionAgentNullRule:

    @pytest.mark.asyncio
    async def test_null_rule_returns_none(self, repos) -> None:
        llm = MockLLMProvider(_null_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        result = await agent.reflect(_trade_data(pnl=0.0), _cycle_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_null_rule_not_saved(self, repos) -> None:
        llm = MockLLMProvider(_null_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        await agent.reflect(_trade_data(pnl=0.0), _cycle_data())

        saved = await rules.get_active_rules("BTC-USDC", "1h")
        assert len(saved) == 0

    @pytest.mark.asyncio
    async def test_null_rule_no_event(self, repos) -> None:
        llm = MockLLMProvider(_null_rule_response())
        rules = ReflectionRules(repos.rules)
        bus = InProcessBus()
        agent = ReflectionAgent(llm, rules, bus)

        events: list[RuleGenerated] = []
        bus.subscribe(RuleGenerated, lambda e: events.append(e))

        await agent.reflect(_trade_data(pnl=0.0), _cycle_data())

        assert len(events) == 0


# ---------------------------------------------------------------------------
# Tests: Parse failure safety
# ---------------------------------------------------------------------------

class TestReflectionAgentParseSafety:

    @pytest.mark.asyncio
    async def test_garbage_response_returns_none(self, repos) -> None:
        llm = MockLLMProvider("Just some random text, no JSON.")
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        result = await agent.reflect(_trade_data(), _cycle_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self, repos) -> None:
        llm = ErrorLLMProvider()
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        result = await agent.reflect(_trade_data(), _cycle_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_too_short_rule_returns_none(self, repos) -> None:
        response = json.dumps({
            "rule": "Short",
            "reasoning": "test",
            "applies_to": "all",
            "confidence": 0.5,
        })
        llm = MockLLMProvider(response)
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        result = await agent.reflect(_trade_data(), _cycle_data())

        assert result is None


# ---------------------------------------------------------------------------
# Tests: Prompt content
# ---------------------------------------------------------------------------

class TestReflectionPromptContent:

    @pytest.mark.asyncio
    async def test_trade_data_in_prompt(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        await agent.reflect(_trade_data(), _cycle_data())

        prompt = llm.last_user_prompt
        assert "BTC-USDC" in prompt
        assert "LONG" in prompt
        assert "65000" in prompt
        assert "66500" in prompt
        assert "TP1" in prompt

    @pytest.mark.asyncio
    async def test_signals_in_prompt(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        await agent.reflect(_trade_data(), _cycle_data())

        prompt = llm.last_user_prompt
        assert "BULLISH" in prompt
        assert "ascending_triangle" in prompt
        assert "NEUTRAL" in prompt
        assert "0.72" in prompt

    @pytest.mark.asyncio
    async def test_indicators_in_prompt(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        await agent.reflect(_trade_data(), _cycle_data())

        prompt = llm.last_user_prompt
        assert "rsi" in prompt
        assert "73.2" in prompt
        assert "atr" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_good_bad_examples(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        await agent.reflect(_trade_data(), _cycle_data())

        sys = llm.last_system_prompt
        assert "GOOD RULES" in sys
        assert "BAD RULES" in sys
        assert "Be more careful" in sys

    @pytest.mark.asyncio
    async def test_json_cycle_data_parsed(self, repos) -> None:
        """Cycle data with JSON strings (as stored in DB) should be parsed."""
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        agent = ReflectionAgent(llm, rules, InProcessBus())

        cycle = {
            "signals_json": json.dumps([
                {"agent_name": "indicator_agent", "direction": "BEARISH", "confidence": 0.6, "pattern_detected": None},
            ]),
            "conviction_json": json.dumps({"conviction_score": 0.55, "regime": "RANGING"}),
            "indicators_json": json.dumps({"rsi": 28.5}),
        }

        await agent.reflect(_trade_data(), cycle)

        prompt = llm.last_user_prompt
        assert "BEARISH" in prompt
        assert "RANGING" in prompt
        assert "28.5" in prompt


# ---------------------------------------------------------------------------
# Tests: Event handler factory
# ---------------------------------------------------------------------------

class TestReflectionEventHandler:

    @pytest.mark.asyncio
    async def test_create_handler_subscribes(self, repos) -> None:
        llm = MockLLMProvider(_good_rule_response())
        rules = ReflectionRules(repos.rules)
        bus = InProcessBus()
        agent = ReflectionAgent(llm, rules, bus)

        handler = create_reflection_handler(agent, repos.trades, repos.cycles)
        bus.subscribe(TradeClosed, handler)

        await bus.publish(TradeClosed(
            source="test", symbol="BTC-USDC", pnl=100.0, exit_reason="TP1",
        ))

        # Handler should have triggered the LLM call
        assert llm.call_count == 1
