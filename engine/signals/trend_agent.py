"""TrendAgent: LLM vision-based trendline analysis.

Sends a trendline chart image (OLS + Bollinger Bands) to Claude Vision
for trend direction/strength/reversal assessment. Implements SignalProducer
interface. SKIP is always safe — parse failures return None.
"""

from __future__ import annotations

import json
import logging
import re

from engine.config import FeatureFlags
from engine.data.charts import generate_grounding_header, generate_trendline_chart
from engine.signals.base import SignalProducer
from engine.signals.prompts.trend_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.types import MarketData, SignalOutput
from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class TrendAgent(SignalProducer):
    """Vision-based LLM agent that analyzes OLS trendlines and Bollinger Bands.

    Generates a trendline chart PNG from MarketData, sends it to Claude
    Vision with a grounding context header, and parses the structured JSON
    response into a SignalOutput.
    """

    def __init__(
        self,
        llm: LLMProvider,
        feature_flags: FeatureFlags | None = None,
    ) -> None:
        self._llm = llm
        self._flags = feature_flags

    def name(self) -> str:
        return "trend_agent"

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        if self._flags is None:
            return True
        return self._flags.is_enabled("trend_agent")

    def requires_vision(self) -> bool:
        return True

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        """Generate trendline chart, call Vision LLM, parse response."""
        if not data.candles or not data.indicators:
            logger.warning("TrendAgent: no candles or indicators, returning None")
            return None

        try:
            chart_png = generate_trendline_chart(
                candles=data.candles,
                symbol=data.symbol,
                timeframe=data.timeframe,
            )
            if not chart_png:
                logger.warning("TrendAgent: chart generation returned empty bytes")
                return None

            grounding = generate_grounding_header(
                symbol=data.symbol,
                timeframe=data.timeframe,
                indicators=data.indicators,
                flow=data.flow,
                parent_tf=data.parent_tf,
                swing_highs=data.swing_highs,
                swing_lows=data.swing_lows,
                forecast_candles=data.forecast_candles,
                forecast_description=data.forecast_description,
                num_candles=data.num_candles,
                lookback_description=data.lookback_description,
            )

            system = SYSTEM_PROMPT.format(grounding_header=grounding)
            user = USER_PROMPT.format(
                symbol=data.symbol,
                timeframe=data.timeframe,
                forecast_candles=data.forecast_candles,
                forecast_description=data.forecast_description,
            )

            response = await self._llm.generate_vision(
                system_prompt=system,
                user_prompt=user,
                image_data=chart_png,
                image_media_type="image/png",
                agent_name=self.name(),
                max_tokens=512,
                temperature=0.3,
                cache_system_prompt=True,
            )

            return self._parse_response(response.content)

        except Exception:
            logger.exception("TrendAgent: analysis failed")
            return None

    def _parse_response(self, raw: str) -> SignalOutput | None:
        """Extract JSON from LLM response and build SignalOutput.

        Returns None on any parse failure (SKIP is safe).
        """
        try:
            json_str = _extract_json(raw)
            if json_str is None:
                logger.warning("TrendAgent: could not extract JSON from response")
                return None

            parsed = json.loads(json_str)

            direction = parsed.get("direction")
            if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                logger.warning(f"TrendAgent: invalid direction '{direction}'")
                return None

            confidence = float(parsed.get("confidence", 0))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(parsed.get("reasoning", ""))
            contradictions = str(parsed.get("contradictions", "none"))

            key_levels = parsed.get("key_levels", {})
            if not isinstance(key_levels, dict):
                key_levels = {}

            return SignalOutput(
                agent_name=self.name(),
                signal_type=self.signal_type(),
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                signal_category="directional",
                data_richness="full" if len(reasoning) > 20 else "minimal",
                contradictions=contradictions,
                key_levels=key_levels,
                pattern_detected=None,
                raw_output=raw,
            )

        except Exception:
            logger.exception("TrendAgent: parse failed")
            return None


# ---------------------------------------------------------------------------
# Shared JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Extract JSON object from text, handling markdown code blocks."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return None
