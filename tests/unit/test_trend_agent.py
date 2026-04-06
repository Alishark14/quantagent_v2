"""Unit tests for TrendAgent."""

from __future__ import annotations

import json

import pytest

from engine.signals.trend_agent import TrendAgent, _extract_json
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
            input_tokens=900, output_tokens=130, cost=0.010,
            model="claude-sonnet-4-20250514", latency_ms=2300.0, cached_input_tokens=700,
        )


# ---------------------------------------------------------------------------
# Test data
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
            "rsi": 58.0,
            "macd": {"macd": 0.8, "signal": 0.6, "histogram": 0.2, "histogram_direction": "rising", "cross": "none"},
            "roc": 1.2,
            "stochastic": {"k": 62.0, "d": 58.0, "zone": "neutral"},
            "williams_r": -38.0,
            "atr": 1.05,
            "adx": {"adx": 29.0, "plus_di": 26.0, "minus_di": 14.0, "classification": "TRENDING"},
            "bollinger_bands": {"upper": 126.0, "middle": 124.0, "lower": 122.0, "width": 4.0, "width_percentile": 45.0},
            "volume_ma": {"ma": 1000.0, "current": 1050.0, "ratio": 1.05, "spike": False},
            "volatility_percentile": 45.0,
        },
        swing_highs=[126.0, 128.0],
        swing_lows=[118.0, 115.0],
    )


def _bullish_trend_response() -> str:
    return json.dumps({
        "direction": "BULLISH",
        "confidence": 0.78,
        "reasoning": "Primary trendline rising with positive slope. Short-term trendline confirms — steeper rise. ADX 29 confirms strong trend. Price riding upper Bollinger Band.",
        "trend_regime": "CLEAN_TREND",
        "contradictions": "none",
        "key_levels": {"resistance": 128.0, "support": 118.0},
    })


def _reversal_response() -> str:
    return json.dumps({
        "direction": "BEARISH",
        "confidence": 0.62,
        "reasoning": "Primary trendline still rising but short-term trendline flattening. Divergence suggests trend exhaustion. BB width expanding — volatile regime.",
        "trend_regime": "REVERSAL",
        "contradictions": "Primary slope positive but short-term momentum fading",
        "key_levels": {"resistance": 126.0, "support": 120.0},
    })


def _neutral_response() -> str:
    return json.dumps({
        "direction": "NEUTRAL",
        "confidence": 0.35,
        "reasoning": "Both trendlines nearly flat. ADX weak. Bollinger Band squeeze forming — breakout expected but direction unclear.",
        "trend_regime": "COMPRESSION",
        "contradictions": "none",
        "key_levels": {"resistance": 126.0, "support": 122.0},
    })


# ---------------------------------------------------------------------------
# Tests: interface
# ---------------------------------------------------------------------------

class TestTrendAgentInterface:
    def test_name(self) -> None:
        agent = TrendAgent(MockVisionLLM(""))
        assert agent.name() == "trend_agent"

    def test_signal_type(self) -> None:
        agent = TrendAgent(MockVisionLLM(""))
        assert agent.signal_type() == "llm"

    def test_requires_vision_true(self) -> None:
        agent = TrendAgent(MockVisionLLM(""))
        assert agent.requires_vision() is True

    def test_is_enabled_default(self) -> None:
        agent = TrendAgent(MockVisionLLM(""), feature_flags=None)
        assert agent.is_enabled() is True


# ---------------------------------------------------------------------------
# Tests: analyze
# ---------------------------------------------------------------------------

class TestTrendAgentAnalyze:
    @pytest.mark.asyncio
    async def test_bullish_trend(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert isinstance(result, SignalOutput)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.78)
        assert result.agent_name == "trend_agent"
        assert result.signal_type == "llm"
        assert result.signal_category == "directional"
        assert result.pattern_detected is None
        assert "trendline" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_reversal_signal(self) -> None:
        llm = MockVisionLLM(_reversal_response())
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.62)
        assert "divergence" in result.reasoning.lower() or "exhaustion" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_neutral_compression(self) -> None:
        llm = MockVisionLLM(_neutral_response())
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.35)

    @pytest.mark.asyncio
    async def test_uses_vision_not_text(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.vision_call_count == 1
        assert llm.text_call_count == 0

    @pytest.mark.asyncio
    async def test_sends_png_image(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_image_data is not None
        assert llm.last_image_data[:8] == b"\x89PNG\r\n\x1a\n"
        assert llm.last_image_media_type == "image/png"

    @pytest.mark.asyncio
    async def test_markdown_code_block(self) -> None:
        response = "```json\n" + _bullish_trend_response() + "\n```"
        llm = MockVisionLLM(response)
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_none(self) -> None:
        llm = MockVisionLLM("The trend looks interesting but I can't format JSON.")
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_direction_returns_none(self) -> None:
        response = json.dumps({
            "direction": "TRENDING",
            "confidence": 0.7,
            "reasoning": "test",
            "trend_regime": "CLEAN_TREND",
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockVisionLLM(response)
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is None

    @pytest.mark.asyncio
    async def test_confidence_clamped(self) -> None:
        response = json.dumps({
            "direction": "BULLISH",
            "confidence": 1.8,
            "reasoning": "Extremely strong trend confirmed by everything.",
            "trend_regime": "CLEAN_TREND",
            "contradictions": "none",
            "key_levels": {},
        })
        llm = MockVisionLLM(response)
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_empty_candles_returns_none(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)
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
        raw = _bullish_trend_response()
        llm = MockVisionLLM(raw)
        agent = TrendAgent(llm)

        result = await agent.analyze(_make_market_data())

        assert result is not None
        assert result.raw_output == raw


# ---------------------------------------------------------------------------
# Tests: grounding in prompt
# ---------------------------------------------------------------------------

class TestTrendGrounding:
    @pytest.mark.asyncio
    async def test_grounding_header_in_system_prompt(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        await agent.analyze(_make_market_data())

        assert llm.last_system_prompt is not None
        assert "CONTEXT (do not override with visual impression):" in llm.last_system_prompt
        assert "BTC-USDC" in llm.last_system_prompt
        assert "RSI:" in llm.last_system_prompt
        assert "ADX:" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_trendline_instructions(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        await agent.analyze(_make_market_data())

        assert "PRIMARY TRENDLINE" in llm.last_system_prompt
        assert "SHORT-TERM TRENDLINE" in llm.last_system_prompt
        assert "BOLLINGER BANDS" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_has_trend_regime_options(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        await agent.analyze(_make_market_data())

        assert "CLEAN_TREND" in llm.last_system_prompt
        assert "COMPRESSION" in llm.last_system_prompt
        assert "REVERSAL" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_user_prompt_contains_symbol(self) -> None:
        llm = MockVisionLLM(_bullish_trend_response())
        agent = TrendAgent(llm)

        await agent.analyze(_make_market_data())

        assert "BTC-USDC" in llm.last_user_prompt
        assert "1h" in llm.last_user_prompt


# ---------------------------------------------------------------------------
# Tests: _extract_json
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
        result = _extract_json("Just text, no JSON here.")
        assert result is None
