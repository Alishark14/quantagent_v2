"""Unit tests for ConvictionAgent."""

from __future__ import annotations

import json

import pytest

from engine.conviction.agent import ConvictionAgent
from engine.types import ConvictionOutput, MarketData, SignalOutput
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
        self.last_temperature: float | None = None
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
        self.last_temperature = temperature
        self.call_count += 1
        return LLMResponse(
            content=self._response_content,
            input_tokens=800,
            output_tokens=200,
            cost=0.010,
            model="claude-sonnet-4-20250514",
            latency_ms=1500.0,
            cached_input_tokens=600,
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
        raise NotImplementedError("ConvictionAgent does not use vision")


class ErrorLLMProvider(LLMProvider):
    """Always raises an exception."""

    async def generate_text(self, **kwargs) -> LLMResponse:
        raise ConnectionError("LLM API is down")

    async def generate_vision(self, **kwargs) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_market_data() -> MarketData:
    """Build a minimal MarketData for testing."""
    candles = []
    for i in range(50):
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
        num_candles=50,
        lookback_description="~2 days",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={
            "rsi": 73.2,
            "macd": {"macd": 1.5, "signal": 1.0, "histogram": 0.5, "histogram_direction": "declining", "cross": "none"},
            "roc": 2.1,
            "stochastic": {"k": 82.0, "d": 78.0, "zone": "overbought"},
            "williams_r": -18.0,
            "atr": 450.0,
            "adx": {"adx": 31.0, "plus_di": 28.0, "minus_di": 14.0, "classification": "TRENDING"},
            "bollinger_bands": {"upper": 66000.0, "middle": 65000.0, "lower": 64000.0, "width": 2000.0, "width_percentile": 55.0},
            "volume_ma": {"ma": 5000.0, "current": 5500.0, "ratio": 1.1, "spike": False},
            "volatility_percentile": 50.0,
        },
        swing_highs=[65400.0, 66000.0],
        swing_lows=[63100.0, 62500.0],
    )


def _make_signals() -> list[SignalOutput]:
    """Build a set of 3 signal outputs (indicator, pattern, trend)."""
    return [
        SignalOutput(
            agent_name="indicator_agent",
            signal_type="llm",
            direction="BULLISH",
            confidence=0.72,
            reasoning="RSI at 73 shows strong momentum. MACD histogram positive but declining.",
            signal_category="directional",
            data_richness="full",
            contradictions="RSI overbought conflicts with bullish read",
            key_levels={"resistance": 65400.0, "support": 63100.0},
            pattern_detected=None,
            raw_output="...",
        ),
        SignalOutput(
            agent_name="pattern_agent",
            signal_type="llm",
            direction="BULLISH",
            confidence=0.80,
            reasoning="Ascending triangle pattern near completion at 65400 resistance.",
            signal_category="directional",
            data_richness="full",
            contradictions="Pattern forming into 4h resistance",
            key_levels={"resistance": 65400.0, "support": 64200.0},
            pattern_detected="ascending_triangle",
            raw_output="...",
        ),
        SignalOutput(
            agent_name="trend_agent",
            signal_type="llm",
            direction="NEUTRAL",
            confidence=0.50,
            reasoning="Trend exhaustion forming. OLS trendline flattening.",
            signal_category="directional",
            data_richness="full",
            contradictions="none",
            key_levels={"resistance": 65500.0, "support": 63000.0},
            pattern_detected=None,
            raw_output="...",
        ),
    ]


def _high_conviction_response() -> str:
    return json.dumps({
        "conviction_score": 0.78,
        "direction": "LONG",
        "regime": "TRENDING_UP",
        "regime_confidence": 0.82,
        "signal_quality": "HIGH",
        "contradictions": ["RSI overbought conflicts with bullish pattern"],
        "reasoning": "Two agents agree bullish. Strong ADX confirms trend. Pattern near completion provides clear entry.",
        "factual_weight": 0.4,
        "subjective_weight": 0.6,
    })


def _low_conviction_response() -> str:
    return json.dumps({
        "conviction_score": 0.28,
        "direction": "SKIP",
        "regime": "RANGING",
        "regime_confidence": 0.65,
        "signal_quality": "CONFLICTING",
        "contradictions": [
            "IndicatorAgent bullish but RSI overbought",
            "PatternAgent bullish but TrendAgent neutral",
            "Funding rate crowded long suggests reversal risk",
        ],
        "reasoning": "Significant disagreement between agents. Trend exhaustion noted. Overbought conditions with crowded funding. No clear edge.",
        "factual_weight": 0.7,
        "subjective_weight": 0.3,
    })


# ---------------------------------------------------------------------------
# Tests: Core evaluate flow
# ---------------------------------------------------------------------------

class TestConvictionAgentEvaluate:

    @pytest.mark.asyncio
    async def test_high_conviction_parse(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert isinstance(result, ConvictionOutput)
        assert result.conviction_score == pytest.approx(0.78)
        assert result.direction == "LONG"
        assert result.regime == "TRENDING_UP"
        assert result.regime_confidence == pytest.approx(0.82)
        assert result.signal_quality == "HIGH"
        assert len(result.contradictions) == 1
        assert "overbought" in result.contradictions[0].lower()
        assert result.factual_weight == pytest.approx(0.4)
        assert result.subjective_weight == pytest.approx(0.6)
        assert len(result.reasoning) > 0
        assert result.raw_output != ""

    @pytest.mark.asyncio
    async def test_low_conviction_with_contradictions(self) -> None:
        llm = MockLLMProvider(_low_conviction_response())
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.conviction_score == pytest.approx(0.28)
        assert result.direction == "SKIP"
        assert result.regime == "RANGING"
        assert result.signal_quality == "CONFLICTING"
        assert len(result.contradictions) == 3
        assert result.factual_weight == pytest.approx(0.7)
        assert result.subjective_weight == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_short_direction(self) -> None:
        response = json.dumps({
            "conviction_score": 0.65,
            "direction": "SHORT",
            "regime": "TRENDING_DOWN",
            "regime_confidence": 0.75,
            "signal_quality": "MEDIUM",
            "contradictions": [],
            "reasoning": "Bearish consensus with strong downtrend.",
            "factual_weight": 0.5,
            "subjective_weight": 0.5,
        })
        llm = MockLLMProvider(response)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.direction == "SHORT"
        assert result.regime == "TRENDING_DOWN"

    @pytest.mark.asyncio
    async def test_llm_called_once(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Parse failure safety
# ---------------------------------------------------------------------------

class TestConvictionAgentParseSafety:

    @pytest.mark.asyncio
    async def test_garbage_response_returns_safe_default(self) -> None:
        llm = MockLLMProvider("I don't know what to say. Here's some random text.")
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.conviction_score == 0.0
        assert result.direction == "SKIP"
        assert result.regime == "RANGING"
        assert result.signal_quality == "LOW"
        assert "Parse failure" in result.contradictions[0]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_safe_default(self) -> None:
        llm = MockLLMProvider("{broken json: not valid")
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.conviction_score == 0.0
        assert result.direction == "SKIP"

    @pytest.mark.asyncio
    async def test_llm_exception_returns_safe_default(self) -> None:
        llm = ErrorLLMProvider()
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.conviction_score == 0.0
        assert result.direction == "SKIP"

    @pytest.mark.asyncio
    async def test_invalid_direction_defaults_to_skip(self) -> None:
        response = json.dumps({
            "conviction_score": 0.7,
            "direction": "BUY",  # invalid
            "regime": "TRENDING_UP",
            "regime_confidence": 0.8,
            "signal_quality": "HIGH",
            "contradictions": [],
            "reasoning": "test",
            "factual_weight": 0.4,
            "subjective_weight": 0.6,
        })
        llm = MockLLMProvider(response)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.direction == "SKIP"

    @pytest.mark.asyncio
    async def test_invalid_regime_defaults_to_ranging(self) -> None:
        response = json.dumps({
            "conviction_score": 0.7,
            "direction": "LONG",
            "regime": "CRAZY_MARKET",  # invalid
            "regime_confidence": 0.8,
            "signal_quality": "HIGH",
            "contradictions": [],
            "reasoning": "test",
            "factual_weight": 0.4,
            "subjective_weight": 0.6,
        })
        llm = MockLLMProvider(response)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.regime == "RANGING"

    @pytest.mark.asyncio
    async def test_conviction_score_clamped_high(self) -> None:
        response = json.dumps({
            "conviction_score": 1.5,  # over max
            "direction": "LONG",
            "regime": "TRENDING_UP",
            "regime_confidence": 0.8,
            "signal_quality": "HIGH",
            "contradictions": [],
            "reasoning": "test",
            "factual_weight": 0.4,
            "subjective_weight": 0.6,
        })
        llm = MockLLMProvider(response)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.conviction_score == 1.0

    @pytest.mark.asyncio
    async def test_conviction_score_clamped_low(self) -> None:
        response = json.dumps({
            "conviction_score": -0.3,  # below min
            "direction": "SKIP",
            "regime": "RANGING",
            "regime_confidence": 0.5,
            "signal_quality": "LOW",
            "contradictions": [],
            "reasoning": "test",
            "factual_weight": 0.5,
            "subjective_weight": 0.5,
        })
        llm = MockLLMProvider(response)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.conviction_score == 0.0

    @pytest.mark.asyncio
    async def test_contradictions_non_list_coerced(self) -> None:
        response = json.dumps({
            "conviction_score": 0.5,
            "direction": "SKIP",
            "regime": "RANGING",
            "regime_confidence": 0.5,
            "signal_quality": "LOW",
            "contradictions": "single string instead of list",
            "reasoning": "test",
            "factual_weight": 0.5,
            "subjective_weight": 0.5,
        })
        llm = MockLLMProvider(response)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert isinstance(result.contradictions, list)
        assert len(result.contradictions) == 1


# ---------------------------------------------------------------------------
# Tests: Prompt content verification
# ---------------------------------------------------------------------------

class TestConvictionPromptContent:

    @pytest.mark.asyncio
    async def test_grounding_header_in_system_prompt(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        assert llm.last_system_prompt is not None
        assert "CONTEXT (do not override with visual impression):" in llm.last_system_prompt
        assert "BTC-USDC" in llm.last_system_prompt
        assert "RSI:" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_grounding_header_in_user_prompt(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        assert llm.last_user_prompt is not None
        assert "FACTUAL DATA" in llm.last_user_prompt
        assert "CONTEXT (do not override with visual impression):" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_all_signal_data_in_user_prompt(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        prompt = llm.last_user_prompt
        assert prompt is not None

        # IndicatorAgent signal data
        assert "IndicatorAgent:" in prompt
        assert "BULLISH" in prompt
        assert "0.72" in prompt
        assert "RSI overbought conflicts with bullish read" in prompt

        # PatternAgent signal data
        assert "PatternAgent:" in prompt
        assert "0.80" in prompt
        assert "ascending_triangle" in prompt
        assert "Pattern forming into 4h resistance" in prompt

        # TrendAgent signal data
        assert "TrendAgent:" in prompt
        assert "NEUTRAL" in prompt
        assert "0.50" in prompt

    @pytest.mark.asyncio
    async def test_flow_signal_renders_in_user_prompt(self) -> None:
        """4th signal voice (FlowSignalAgent) reaches the LLM via signals_block.

        Regression test for the v1.1→v1.2 prompt refactor: previously the
        USER_PROMPT had 3 hardcoded blocks (Indicator/Pattern/Trend) and
        a 4th SignalOutput would never reach the LLM. After the dynamic
        signals_block change, FlowAgent's voice MUST appear when present.
        """
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        flow_signal = SignalOutput(
            agent_name="flow_signal_agent",
            signal_type="flow",
            direction="BEARISH",
            confidence=0.70,
            reasoning=(
                "BEARISH divergence: price up but funding flipped negative "
                "and OI dropping — smart money exiting."
            ),
            signal_category="directional",
            data_richness="full",
            contradictions="",
            key_levels={},
            pattern_detected=None,
            raw_output="BEARISH divergence ...",
        )
        signals = _make_signals() + [flow_signal]

        await agent.evaluate(signals, _make_market_data())

        prompt = llm.last_user_prompt
        assert prompt is not None
        assert "FlowAgent:" in prompt
        assert "BEARISH" in prompt
        assert "0.70" in prompt
        assert "smart money exiting" in prompt

    @pytest.mark.asyncio
    async def test_flow_block_renders_empty_when_flow_missing(self) -> None:
        """When FlowSignalAgent isn't in the signals list, its slot still renders.

        The dynamic signals_block always renders all 4 known agents so the
        prompt structure stays predictable for the LLM. Missing slots show
        the standard "Agent did not produce a signal" reasoning.
        """
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        prompt = llm.last_user_prompt
        assert prompt is not None
        assert "FlowAgent:" in prompt  # slot rendered
        assert "Agent did not produce a signal" in prompt

    @pytest.mark.asyncio
    async def test_memory_context_in_user_prompt(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)
        memory = "Last 3 cycles: SKIP, SKIP, LONG (0.72 conviction). Rule: Avoid LONG when RSI > 65."

        await agent.evaluate(_make_signals(), _make_market_data(), memory_context=memory)

        assert llm.last_user_prompt is not None
        assert "Avoid LONG when RSI > 65" in llm.last_user_prompt
        assert "Last 3 cycles" in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_default_memory_context(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        assert llm.last_user_prompt is not None
        assert "No prior history." in llm.last_user_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_regime_instructions(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        assert llm.last_system_prompt is not None
        assert "REGIME CLASSIFICATION" in llm.last_system_prompt
        assert "TRENDING_UP" in llm.last_system_prompt
        assert "CONVICTION SCORING RULES" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_contains_json_format(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        await agent.evaluate(_make_signals(), _make_market_data())

        assert llm.last_system_prompt is not None
        assert '"conviction_score"' in llm.last_system_prompt
        assert '"direction"' in llm.last_system_prompt
        assert '"regime"' in llm.last_system_prompt


# ---------------------------------------------------------------------------
# Tests: Missing signals handling
# ---------------------------------------------------------------------------

class TestConvictionMissingSignals:

    @pytest.mark.asyncio
    async def test_missing_agent_shows_na(self) -> None:
        """When only some agents produce output, missing ones show N/A."""
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)

        # Only IndicatorAgent signal, no pattern or trend
        signals = [_make_signals()[0]]

        await agent.evaluate(signals, _make_market_data())

        prompt = llm.last_user_prompt
        assert prompt is not None
        assert "Agent did not produce a signal" in prompt

    @pytest.mark.asyncio
    async def test_empty_signals_still_works(self) -> None:
        """With zero signals, agent still produces output (all N/A)."""
        llm = MockLLMProvider(_low_conviction_response())
        agent = ConvictionAgent(llm)

        result = await agent.evaluate([], _make_market_data())

        assert isinstance(result, ConvictionOutput)
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Tests: JSON extraction
# ---------------------------------------------------------------------------

class TestConvictionExtractJson:

    def test_raw_json(self) -> None:
        text = '{"conviction_score": 0.7, "direction": "LONG"}'
        result = ConvictionAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["direction"] == "LONG"

    def test_fenced_json(self) -> None:
        text = '```json\n{"conviction_score": 0.5, "direction": "SKIP"}\n```'
        result = ConvictionAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["direction"] == "SKIP"

    def test_json_with_surrounding_text(self) -> None:
        text = 'My assessment:\n{"conviction_score": 0.8, "direction": "LONG"}\nEnd.'
        result = ConvictionAgent._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["conviction_score"] == 0.8

    def test_no_json_returns_none(self) -> None:
        text = "No JSON here at all."
        result = ConvictionAgent._extract_json(text)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Raw output preservation
# ---------------------------------------------------------------------------

class TestConvictionRawOutput:

    @pytest.mark.asyncio
    async def test_raw_output_preserved_on_success(self) -> None:
        raw = _high_conviction_response()
        llm = MockLLMProvider(raw)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.raw_output == raw

    @pytest.mark.asyncio
    async def test_raw_output_preserved_on_parse_failure(self) -> None:
        raw = "garbage response"
        llm = MockLLMProvider(raw)
        agent = ConvictionAgent(llm)

        result = await agent.evaluate(_make_signals(), _make_market_data())

        assert result.raw_output == raw


# ---------------------------------------------------------------------------
# Tests: Deterministic temperature
# ---------------------------------------------------------------------------


class TestConvictionDeterministicTemperature:
    """ConvictionAgent must call the LLM at temperature=0.0 so identical
    evidence produces identical scores. The agent is an evaluator, not
    a creative writer — non-zero sampling temperature was producing
    spreads of 0.25–0.65 on the same signal pattern.
    """

    @pytest.mark.asyncio
    async def test_temperature_is_zero(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)
        await agent.evaluate(_make_signals(), _make_market_data())
        assert llm.last_temperature == 0.0

    @pytest.mark.asyncio
    async def test_repeated_calls_use_zero_temperature(self) -> None:
        llm = MockLLMProvider(_high_conviction_response())
        agent = ConvictionAgent(llm)
        for _ in range(3):
            await agent.evaluate(_make_signals(), _make_market_data())
            assert llm.last_temperature == 0.0
        assert llm.call_count == 3
