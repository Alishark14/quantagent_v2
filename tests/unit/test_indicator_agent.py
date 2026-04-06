"""Unit tests for IndicatorAgent."""

from __future__ import annotations

import json

import pytest

from engine.config import FeatureFlags
from engine.signals.indicator_agent import IndicatorAgent
from engine.types import MarketData, SignalOutput
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
            input_tokens=500,
            output_tokens=100,
            cost=0.008,
            model="claude-sonnet-4-20250514",
            latency_ms=1200.0,
            cached_input_tokens=400,
        )

    async def generate_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: bytes,
        image_media_type: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        raise NotImplementedError("IndicatorAgent does not use vision")


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_market_data() -> MarketData:
    """Build a minimal MarketData for testing."""
    candles = []
    for i in range(50):
        c = 100.0 + i * 0.5
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": c - 0.3,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": 1000.0,
        })

    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=candles,
        num_candles=50,
        lookback_description="~2 days",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={
            "rsi": 65.3,
            "macd": {"macd": 1.2, "signal": 0.9, "histogram": 0.3, "histogram_direction": "rising", "cross": "none"},
            "roc": 1.8,
            "stochastic": {"k": 72.0, "d": 68.0, "zone": "neutral"},
            "williams_r": -28.0,
            "atr": 1.1,
            "adx": {"adx": 27.0, "plus_di": 24.0, "minus_di": 16.0, "classification": "TRENDING"},
            "bollinger_bands": {"upper": 126.0, "middle": 124.0, "lower": 122.0, "width": 4.0, "width_percentile": 55.0},
            "volume_ma": {"ma": 1000.0, "current": 1100.0, "ratio": 1.1, "spike": False},
            "volatility_percentile": 50.0,
        },
        swing_highs=[126.0, 128.0],
        swing_lows=[118.0, 115.0],
    )


def _valid_json_response() -> str:
    return json.dumps({
        "direction": "BULLISH",
        "confidence": 0.72,
        "reasoning": "RSI at 65 shows building momentum. MACD histogram rising with ADX 27 confirming trend. No overbought conditions yet.",
        "contradictions": "none",
        "key_levels": {"resistance": 126.0, "support": 118.0},
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndicatorAgent:
    def test_name(self) -> None:
        llm = MockLLMProvider("")
        agent = IndicatorAgent(llm)
        assert agent.name() == "indicator_agent"

    def test_signal_type(self) -> None:
        llm = MockLLMProvider("")
        agent = IndicatorAgent(llm)
        assert agent.signal_type() == "llm"

    def test_requires_vision_false(self) -> None:
        llm = MockLLMProvider("")
        agent = IndicatorAgent(llm)
        assert agent.requires_vision() is False

    def test_is_enabled_default(self) -> None:
        llm = MockLLMProvider("")
        agent = IndicatorAgent(llm, feature_flags=None)
        assert agent.is_enabled() is True


class TestIndicatorAgentAnalyze:
    @pytest.mark.asyncio
    async def test_successful_parse(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)
        data = _make_market_data()

        result = await agent.analyze(data)

        assert result is not None
        assert isinstance(result, SignalOutput)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.72)
        assert "momentum" in result.reasoning.lower()
        assert result.contradictions == "none"
        assert result.key_levels == {"resistance": 126.0, "support": 118.0}
        assert result.agent_name == "indicator_agent"
        assert result.signal_type == "llm"
        assert result.signal_category == "directional"
        assert result.pattern_detected is None

    @pytest.mark.asyncio
    async def test_bearish_signal(self) -> None:
        response = json.dumps({
            "direction": "BEARISH",
            "confidence": 0.85,
            "reasoning": "RSI overbought at 78, MACD bearish cross detected.",
            "contradictions": "ADX strong but momentum fading",
            "key_levels": {"resistance": 130.0, "support": 120.0},
        })
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_neutral_signal(self) -> None:
        response = json.dumps({
            "direction": "NEUTRAL",
            "confidence": 0.4,
            "reasoning": "Mixed signals. RSI mid-range, MACD flat.",
            "contradictions": "Stochastic disagrees with MACD",
            "key_levels": {"resistance": None, "support": None},
        })
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_markdown_code_block(self) -> None:
        response = '```json\n' + _valid_json_response() + '\n```'
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_markdown_code_block_no_lang(self) -> None:
        response = '```\n' + _valid_json_response() + '\n```'
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_json_with_surrounding_text(self) -> None:
        response = 'Here is my analysis:\n' + _valid_json_response() + '\nEnd.'
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_none(self) -> None:
        llm = MockLLMProvider("This is not JSON at all. Just some random text.")
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_direction_returns_none(self) -> None:
        response = json.dumps({
            "direction": "LONG",  # invalid — should be BULLISH/BEARISH/NEUTRAL
            "confidence": 0.7,
            "reasoning": "test",
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_direction_returns_none(self) -> None:
        response = json.dumps({
            "confidence": 0.7,
            "reasoning": "test",
        })
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_confidence_clamped(self) -> None:
        response = json.dumps({
            "direction": "BULLISH",
            "confidence": 1.5,  # over 1.0 — should be clamped
            "reasoning": "Very strong",
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_negative_confidence_clamped(self) -> None:
        response = json.dumps({
            "direction": "BEARISH",
            "confidence": -0.3,
            "reasoning": "test",
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockLLMProvider(response)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_empty_candles_returns_none(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)
        data = MarketData(
            symbol="BTC-USDC", timeframe="1h", candles=[], num_candles=0,
            lookback_description="", forecast_candles=3, forecast_description="",
            indicators={}, swing_highs=[], swing_lows=[],
        )

        result = await agent.analyze(data)

        assert result is None
        assert llm.call_count == 0  # LLM should not be called

    @pytest.mark.asyncio
    async def test_empty_indicators_returns_none(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)
        data = _make_market_data()
        data.indicators = {}

        result = await agent.analyze(data)

        assert result is None

    @pytest.mark.asyncio
    async def test_raw_output_preserved(self) -> None:
        raw = _valid_json_response()
        llm = MockLLMProvider(raw)
        agent = IndicatorAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.raw_output == raw


class TestGroundingInPrompt:
    @pytest.mark.asyncio
    async def test_grounding_header_in_system_prompt(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_system_prompt is not None
        assert "CONTEXT (do not override with visual impression):" in llm.last_system_prompt
        assert "BTC-USDC" in llm.last_system_prompt
        assert "RSI:" in llm.last_system_prompt
        assert "MACD:" in llm.last_system_prompt
        assert "ATR:" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_user_prompt_contains_symbol(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_user_prompt is not None
        assert "BTC-USDC" in llm.last_user_prompt
        assert "1h" in llm.last_user_prompt
        assert "3 candles" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_json_instructions(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_system_prompt is not None
        assert '"direction"' in llm.last_system_prompt
        assert '"confidence"' in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_swing_levels_in_grounding(self) -> None:
        llm = MockLLMProvider(_valid_json_response())
        agent = IndicatorAgent(llm)

        await agent.analyze(_make_market_data())

        assert "Nearest resistance:" in llm.last_system_prompt
        assert "Nearest support:" in llm.last_system_prompt


class TestExtractJson:
    def test_raw_json(self) -> None:
        text = '{"direction": "BULLISH", "confidence": 0.7}'
        result = IndicatorAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["direction"] == "BULLISH"

    def test_fenced_json(self) -> None:
        text = '```json\n{"direction": "BEARISH"}\n```'
        result = IndicatorAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["direction"] == "BEARISH"

    def test_fenced_no_lang(self) -> None:
        text = '```\n{"direction": "NEUTRAL"}\n```'
        result = IndicatorAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["direction"] == "NEUTRAL"

    def test_no_json_returns_none(self) -> None:
        text = "No JSON here, just plain text analysis."
        result = IndicatorAgent._extract_json(text)
        assert result is None

    def test_json_with_surrounding_text(self) -> None:
        text = 'Analysis: {"direction": "BULLISH", "confidence": 0.6} done.'
        result = IndicatorAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["direction"] == "BULLISH"
