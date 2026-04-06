"""Unit tests for PatternAgent."""

from __future__ import annotations

import json

import pytest

from engine.signals.pattern_agent import PatternAgent, _extract_json
from engine.types import MarketData, SignalOutput
from llm.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Mock LLM provider (vision-capable)
# ---------------------------------------------------------------------------

class MockVisionLLM(LLMProvider):
    """Returns a predetermined response and records calls."""

    def __init__(self, response_content: str) -> None:
        self._response_content = response_content
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_image_data: bytes | None = None
        self.last_image_media_type: str | None = None
        self.text_call_count = 0
        self.vision_call_count = 0

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        self.text_call_count += 1
        return LLMResponse(
            content=self._response_content,
            input_tokens=500, output_tokens=100, cost=0.008,
            model="claude-sonnet-4-20250514", latency_ms=1200.0, cached_input_tokens=400,
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
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_image_data = image_data
        self.last_image_media_type = image_media_type
        self.vision_call_count += 1
        return LLMResponse(
            content=self._response_content,
            input_tokens=800, output_tokens=120, cost=0.009,
            model="claude-sonnet-4-20250514", latency_ms=2100.0, cached_input_tokens=600,
        )


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_market_data() -> MarketData:
    candles = []
    for i in range(50):
        c = 100.0 + i * 0.5
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": c - 0.3,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": 1000.0 + i * 10,
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


def _valid_pattern_response() -> str:
    return json.dumps({
        "direction": "BULLISH",
        "confidence": 0.75,
        "reasoning": "Ascending triangle pattern forming with flat resistance at 126 and rising support. Volume increasing on up moves.",
        "pattern_detected": "ascending_triangle",
        "pattern_completion": 0.8,
        "contradictions": "none",
        "key_levels": {"resistance": 126.0, "support": 118.0},
    })


def _no_pattern_response() -> str:
    return json.dumps({
        "direction": "NEUTRAL",
        "confidence": 0.3,
        "reasoning": "No clear classical pattern visible. Price in a choppy range.",
        "pattern_detected": None,
        "pattern_completion": None,
        "contradictions": "none",
        "key_levels": {"resistance": 126.0, "support": 118.0},
    })


# ---------------------------------------------------------------------------
# Tests: interface
# ---------------------------------------------------------------------------

class TestPatternAgentInterface:
    def test_name(self) -> None:
        agent = PatternAgent(MockVisionLLM(""))
        assert agent.name() == "pattern_agent"

    def test_signal_type(self) -> None:
        agent = PatternAgent(MockVisionLLM(""))
        assert agent.signal_type() == "llm"

    def test_requires_vision_true(self) -> None:
        agent = PatternAgent(MockVisionLLM(""))
        assert agent.requires_vision() is True

    def test_is_enabled_default(self) -> None:
        agent = PatternAgent(MockVisionLLM(""), feature_flags=None)
        assert agent.is_enabled() is True


# ---------------------------------------------------------------------------
# Tests: analyze
# ---------------------------------------------------------------------------

class TestPatternAgentAnalyze:
    @pytest.mark.asyncio
    async def test_successful_parse_with_pattern(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert isinstance(result, SignalOutput)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.75)
        assert result.pattern_detected == "ascending_triangle"
        assert result.agent_name == "pattern_agent"
        assert result.signal_type == "llm"
        assert result.signal_category == "directional"
        assert "ascending triangle" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_no_pattern_detected(self) -> None:
        llm = MockVisionLLM(_no_pattern_response())
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "NEUTRAL"
        assert result.pattern_detected is None

    @pytest.mark.asyncio
    async def test_bearish_pattern(self) -> None:
        response = json.dumps({
            "direction": "BEARISH",
            "confidence": 0.82,
            "reasoning": "Head and shoulders pattern confirmed with neckline break.",
            "pattern_detected": "head_and_shoulders",
            "pattern_completion": 0.95,
            "contradictions": "RSI not yet oversold",
            "key_levels": {"resistance": 130.0, "support": 120.0},
        })
        llm = MockVisionLLM(response)
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BEARISH"
        assert result.pattern_detected == "head_and_shoulders"
        assert result.confidence == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_uses_vision_not_text(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.vision_call_count == 1
        assert llm.text_call_count == 0

    @pytest.mark.asyncio
    async def test_sends_png_image(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_image_data is not None
        assert llm.last_image_data[:8] == b"\x89PNG\r\n\x1a\n"
        assert llm.last_image_media_type == "image/png"

    @pytest.mark.asyncio
    async def test_markdown_code_block(self) -> None:
        response = "```json\n" + _valid_pattern_response() + "\n```"
        llm = MockVisionLLM(response)
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_none(self) -> None:
        llm = MockVisionLLM("I see an interesting chart but can't format JSON right now.")
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_direction_returns_none(self) -> None:
        response = json.dumps({
            "direction": "UP",
            "confidence": 0.7,
            "reasoning": "test",
            "pattern_detected": None,
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockVisionLLM(response)
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_confidence_clamped(self) -> None:
        response = json.dumps({
            "direction": "BULLISH",
            "confidence": 1.5,
            "reasoning": "Very strong pattern detected.",
            "pattern_detected": "bull_flag",
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockVisionLLM(response)
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_empty_candles_returns_none(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)
        data = MarketData(
            symbol="BTC-USDC", timeframe="1h", candles=[], num_candles=0,
            lookback_description="", forecast_candles=3, forecast_description="",
            indicators={}, swing_highs=[], swing_lows=[],
        )

        result = await agent.analyze(data)

        assert result is None
        assert llm.vision_call_count == 0

    @pytest.mark.asyncio
    async def test_raw_output_preserved(self) -> None:
        raw = _valid_pattern_response()
        llm = MockVisionLLM(raw)
        agent = PatternAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.raw_output == raw


# ---------------------------------------------------------------------------
# Tests: grounding in prompt
# ---------------------------------------------------------------------------

class TestPatternGrounding:
    @pytest.mark.asyncio
    async def test_grounding_header_in_system_prompt(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_system_prompt is not None
        assert "CONTEXT (do not override with visual impression):" in llm.last_system_prompt
        assert "BTC-USDC" in llm.last_system_prompt
        assert "RSI:" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_pattern_library(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        await agent.analyze(_make_market_data())

        assert "ascending_triangle" in llm.last_system_prompt
        assert "head_and_shoulders" in llm.last_system_prompt
        assert "bull_flag" in llm.last_system_prompt
        assert "dark_cloud_cover" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_emphasizes_math_facts(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        await agent.analyze(_make_market_data())

        assert "MATHEMATICAL FACTS" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_user_prompt_contains_symbol(self) -> None:
        llm = MockVisionLLM(_valid_pattern_response())
        agent = PatternAgent(llm)

        await agent.analyze(_make_market_data())

        assert "BTC-USDC" in llm.last_user_prompt
        assert "1h" in llm.last_user_prompt


# ---------------------------------------------------------------------------
# Tests: _extract_json (module-level)
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_raw_json(self) -> None:
        result = _extract_json('{"direction": "BULLISH"}')
        assert result is not None
        assert json.loads(result)["direction"] == "BULLISH"

    def test_fenced_json(self) -> None:
        result = _extract_json('```json\n{"direction": "BEARISH"}\n```')
        assert result is not None
        assert json.loads(result)["direction"] == "BEARISH"

    def test_no_json_returns_none(self) -> None:
        result = _extract_json("Just some text, no JSON.")
        assert result is None
