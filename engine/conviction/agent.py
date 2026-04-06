"""ConvictionAgent: LLM meta-evaluator that scores signal consensus.

The most critical agent in the system. ConvictionAgent does NOT generate
a signal — it EVALUATES the quality and coherence of existing signals
from the Signal Layer and produces a continuous 0-1 conviction score.

On any parse failure, returns ConvictionOutput with score=0.0, direction="SKIP".
SKIP is always safe.
"""

from __future__ import annotations

import json
import logging
import re

from engine.conviction.prompts.conviction_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.data.charts import generate_grounding_header
from engine.types import ConvictionOutput, MarketData, SignalOutput
from llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"LONG", "SHORT", "SKIP"}
_VALID_REGIMES = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "BREAKOUT"}
_VALID_QUALITY = {"HIGH", "MEDIUM", "LOW", "CONFLICTING"}

_SAFE_DEFAULT = ConvictionOutput(
    conviction_score=0.0,
    direction="SKIP",
    regime="RANGING",
    regime_confidence=0.0,
    signal_quality="LOW",
    contradictions=["Parse failure — defaulting to SKIP"],
    reasoning="Could not parse LLM response. Defaulting to safe SKIP.",
    factual_weight=0.5,
    subjective_weight=0.5,
    raw_output="",
)


class ConvictionAgent:
    """Meta-evaluator that scores signal quality and coherence.

    Receives all signal agent outputs + market data, calls a single LLM,
    and produces a ConvictionOutput with conviction score, regime, and reasoning.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def evaluate(
        self,
        signals: list[SignalOutput],
        market_data: MarketData,
        memory_context: str = "No prior history.",
    ) -> ConvictionOutput:
        """Evaluate signal consensus and produce a conviction score.

        Args:
            signals: List of SignalOutput from all signal producers.
            market_data: Full MarketData with indicators, flow, parent TF.
            memory_context: Formatted string with cycle memory, rules, regime history.

        Returns:
            ConvictionOutput. On any failure, returns safe default (score=0.0, SKIP).
        """
        try:
            grounding = generate_grounding_header(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                indicators=market_data.indicators,
                flow=market_data.flow,
                parent_tf=market_data.parent_tf,
                swing_highs=market_data.swing_highs,
                swing_lows=market_data.swing_lows,
                forecast_candles=market_data.forecast_candles,
                forecast_description=market_data.forecast_description,
                num_candles=market_data.num_candles,
                lookback_description=market_data.lookback_description,
            )

            # Build signal map by agent name
            signal_map = self._build_signal_map(signals)

            # Format system prompt (uses {{ }} escaping for literal braces)
            system = SYSTEM_PROMPT.replace("{{symbol}}", market_data.symbol)
            system = system.replace("{{timeframe}}", market_data.timeframe)
            system = system.replace("{{grounding_header}}", grounding)

            # Format user prompt
            ind = signal_map.get("indicator_agent", self._empty_signal())
            pat = signal_map.get("pattern_agent", self._empty_signal())
            trend = signal_map.get("trend_agent", self._empty_signal())

            user = USER_PROMPT.format(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                grounding_header=grounding,
                ind_direction=ind["direction"],
                ind_confidence=ind["confidence"],
                ind_reasoning=ind["reasoning"],
                ind_contradictions=ind["contradictions"],
                pat_direction=pat["direction"],
                pat_confidence=pat["confidence"],
                pat_pattern=pat["pattern"],
                pat_reasoning=pat["reasoning"],
                pat_contradictions=pat["contradictions"],
                trend_direction=trend["direction"],
                trend_confidence=trend["confidence"],
                trend_reasoning=trend["reasoning"],
                trend_contradictions=trend["contradictions"],
                memory_context=memory_context,
            )

            response = await self._llm.generate_text(
                system_prompt=system,
                user_prompt=user,
                agent_name="conviction_agent",
                max_tokens=768,
                temperature=0.3,
                cache_system_prompt=True,
            )

            return self._parse_response(response)

        except Exception:
            logger.exception("ConvictionAgent: evaluation failed")
            return _SAFE_DEFAULT

    def _build_signal_map(self, signals: list[SignalOutput]) -> dict[str, dict]:
        """Map agent_name -> formatted signal dict for prompt injection."""
        result: dict[str, dict] = {}
        for signal in signals:
            result[signal.agent_name] = self._format_signal(signal)
        return result

    def _format_signal(self, signal: SignalOutput) -> dict:
        """Extract display fields from a signal for prompt formatting."""
        return {
            "direction": signal.direction or "N/A",
            "confidence": f"{signal.confidence:.2f}",
            "reasoning": signal.reasoning or "No reasoning provided.",
            "contradictions": signal.contradictions or "none",
            "pattern": signal.pattern_detected or "none",
        }

    @staticmethod
    def _empty_signal() -> dict:
        """Default signal dict when an agent didn't produce output."""
        return {
            "direction": "N/A",
            "confidence": "0.00",
            "reasoning": "Agent did not produce a signal.",
            "contradictions": "none",
            "pattern": "none",
        }

    def _parse_response(self, response: LLMResponse) -> ConvictionOutput:
        """Parse LLM response into ConvictionOutput.

        On any failure, returns safe default (score=0.0, direction=SKIP).
        """
        raw = response.content
        try:
            json_str = self._extract_json(raw)
            if json_str is None:
                logger.warning("ConvictionAgent: could not extract JSON from response")
                return _safe_default_with_raw(raw)

            parsed = json.loads(json_str)

            # Validate and extract fields
            direction = parsed.get("direction", "SKIP")
            if direction not in _VALID_DIRECTIONS:
                logger.warning(f"ConvictionAgent: invalid direction '{direction}', defaulting to SKIP")
                direction = "SKIP"

            regime = parsed.get("regime", "RANGING")
            if regime not in _VALID_REGIMES:
                logger.warning(f"ConvictionAgent: invalid regime '{regime}', defaulting to RANGING")
                regime = "RANGING"

            signal_quality = parsed.get("signal_quality", "LOW")
            if signal_quality not in _VALID_QUALITY:
                signal_quality = "LOW"

            conviction_score = float(parsed.get("conviction_score", 0.0))
            conviction_score = max(0.0, min(1.0, conviction_score))

            regime_confidence = float(parsed.get("regime_confidence", 0.0))
            regime_confidence = max(0.0, min(1.0, regime_confidence))

            factual_weight = float(parsed.get("factual_weight", 0.5))
            factual_weight = max(0.0, min(1.0, factual_weight))

            subjective_weight = float(parsed.get("subjective_weight", 0.5))
            subjective_weight = max(0.0, min(1.0, subjective_weight))

            contradictions = parsed.get("contradictions", [])
            if not isinstance(contradictions, list):
                contradictions = [str(contradictions)]

            reasoning = str(parsed.get("reasoning", ""))

            return ConvictionOutput(
                conviction_score=conviction_score,
                direction=direction,
                regime=regime,
                regime_confidence=regime_confidence,
                signal_quality=signal_quality,
                contradictions=contradictions,
                reasoning=reasoning,
                factual_weight=factual_weight,
                subjective_weight=subjective_weight,
                raw_output=raw,
            )

        except Exception:
            logger.exception("ConvictionAgent: parse failed")
            return _safe_default_with_raw(raw)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON object from text, handling markdown code blocks."""
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        return None


def _safe_default_with_raw(raw: str) -> ConvictionOutput:
    """Return the safe default ConvictionOutput but preserve raw_output."""
    return ConvictionOutput(
        conviction_score=0.0,
        direction="SKIP",
        regime="RANGING",
        regime_confidence=0.0,
        signal_quality="LOW",
        contradictions=["Parse failure — defaulting to SKIP"],
        reasoning="Could not parse LLM response. Defaulting to safe SKIP.",
        factual_weight=0.5,
        subjective_weight=0.5,
        raw_output=raw,
    )
