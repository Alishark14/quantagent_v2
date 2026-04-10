"""PatternAgent: LLM vision-based pattern recognition.

Sends a candlestick chart image to Claude Vision for classical chart
pattern detection. Implements SignalProducer interface. SKIP is always
safe — parse failures return None.
"""

from __future__ import annotations

import json
import logging
import re

from engine.config import FeatureFlags
from engine.data.charts import generate_candlestick_chart, generate_grounding_header
from engine.signals.base import SignalProducer
from engine.signals.prompts.pattern_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.types import MarketData, SignalOutput
from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class PatternAgent(SignalProducer):
    """Vision-based LLM agent that detects classical chart patterns.

    Generates a candlestick chart PNG from MarketData, sends it to Claude
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
        return "pattern_agent"

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        if self._flags is None:
            return True
        return self._flags.is_enabled("pattern_agent")

    def requires_vision(self) -> bool:
        return True

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        """Generate candlestick chart, call Vision LLM, parse response."""
        # ── DIAGNOSTIC: log provider state on every call ──
        logger.warning(
            "DIAG PatternAgent: analyze() entered | "
            f"llm_provider={self._llm}, type={type(self._llm).__name__}, "
            f"candles={len(data.candles) if data.candles else 0}, "
            f"indicators={bool(data.indicators)}, "
            f"flags_enabled={self.is_enabled()}"
        )

        if not data.candles or not data.indicators:
            logger.warning(
                "DIAG PatternAgent: returning None at L60 — "
                f"candles={'empty' if not data.candles else len(data.candles)}, "
                f"indicators={'empty/falsy' if not data.indicators else 'truthy'}"
            )
            return None

        try:
            # Generate chart image
            chart_png = generate_candlestick_chart(
                candles=data.candles,
                symbol=data.symbol,
                timeframe=data.timeframe,
                swing_highs=data.swing_highs,
                swing_lows=data.swing_lows,
            )
            if not chart_png:
                logger.warning(
                    "DIAG PatternAgent: returning None at L73 — "
                    "chart generation returned empty bytes"
                )
                return None

            logger.warning(
                f"DIAG PatternAgent: chart generated OK | "
                f"png_bytes={len(chart_png)}"
            )

            # Build grounding header
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

            # ── DIAGNOSTIC: confirm we reach the LLM call ──
            logger.warning(
                "DIAG PatternAgent: about to call self._llm.generate_vision() | "
                f"system_len={len(system)}, user_len={len(user)}, "
                f"image_bytes={len(chart_png)}"
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

            # ── DIAGNOSTIC: LLM returned ──
            logger.warning(
                f"DIAG PatternAgent: LLM returned | content_len={len(response.content)}"
            )

            result = self._parse_response(response.content)
            if result is None:
                logger.warning(
                    "DIAG PatternAgent: returning None at L111 — _parse_response returned None"
                )
            return result

        except Exception as exc:
            logger.warning(
                f"DIAG PatternAgent: returning None at L114 — "
                f"exception: {type(exc).__name__}: {exc}"
            )
            logger.exception("PatternAgent: analysis failed")
            return None

    def _parse_response(self, raw: str) -> SignalOutput | None:
        """Extract JSON from LLM response and build SignalOutput.

        Same extraction logic as IndicatorAgent but also reads
        pattern_detected field.
        Returns None on any parse failure (SKIP is safe).
        """
        try:
            json_str = _extract_json(raw)
            if json_str is None:
                logger.warning(
                    "DIAG PatternAgent: returning None at L126 — "
                    f"could not extract JSON from response: {raw[:200]!r}"
                )
                return None

            parsed = json.loads(json_str)

            direction = parsed.get("direction")
            if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                logger.warning(
                    f"DIAG PatternAgent: returning None at L134 — "
                    f"invalid direction '{direction}'"
                )
                return None

            confidence = float(parsed.get("confidence", 0))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(parsed.get("reasoning", ""))
            contradictions = str(parsed.get("contradictions", "none"))

            key_levels = parsed.get("key_levels", {})
            if not isinstance(key_levels, dict):
                key_levels = {}

            pattern = parsed.get("pattern_detected")
            if pattern is not None:
                pattern = str(pattern)

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
                pattern_detected=pattern,
                raw_output=raw,
            )

        except Exception as exc:
            logger.warning(
                f"DIAG PatternAgent: returning None at L167 — "
                f"parse exception: {type(exc).__name__}: {exc}"
            )
            logger.exception("PatternAgent: parse failed")
            return None


# ---------------------------------------------------------------------------
# Shared JSON extraction (same logic as IndicatorAgent)
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
