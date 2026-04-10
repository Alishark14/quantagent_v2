"""IndicatorAgent: LLM text-based signal producer.

Analyzes computed indicator values via Claude (text only, no vision).
Implements SignalProducer interface. SKIP is always safe — parse failures
return None.
"""

from __future__ import annotations

import json
import logging
import re

from engine.config import FeatureFlags
from engine.data.charts import generate_grounding_header
from engine.signals.base import SignalProducer
from engine.signals.prompts.indicator_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.types import MarketData, SignalOutput
from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class IndicatorAgent(SignalProducer):
    """Text-only LLM agent that analyzes technical indicator values.

    Receives computed indicators as grounding context, calls Claude,
    and parses the structured JSON response into a SignalOutput.
    """

    def __init__(
        self,
        llm: LLMProvider,
        feature_flags: FeatureFlags | None = None,
    ) -> None:
        self._llm = llm
        self._flags = feature_flags

    def name(self) -> str:
        return "indicator_agent"

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        if self._flags is None:
            return True
        return self._flags.is_enabled("indicator_agent")

    def requires_vision(self) -> bool:
        return False

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        """Build grounding header, call LLM, parse JSON response."""
        # ── DIAGNOSTIC: log provider state on every call ──
        logger.warning(
            "DIAG IndicatorAgent: analyze() entered | "
            f"llm_provider={self._llm}, type={type(self._llm).__name__}, "
            f"candles={len(data.candles) if data.candles else 0}, "
            f"indicators={bool(data.indicators)}, "
            f"flags_enabled={self.is_enabled()}"
        )

        if not data.candles or not data.indicators:
            logger.warning(
                "DIAG IndicatorAgent: returning None at L59 — "
                f"candles={'empty' if not data.candles else len(data.candles)}, "
                f"indicators={'empty/falsy' if not data.indicators else 'truthy'}"
            )
            return None

        try:
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
                forecast_candles=data.forecast_candles,
                forecast_description=data.forecast_description,
                symbol=data.symbol,
                timeframe=data.timeframe,
            )

            # ── DIAGNOSTIC: confirm we reach the LLM call ──
            logger.warning(
                "DIAG IndicatorAgent: about to call self._llm.generate_text() | "
                f"system_len={len(system)}, user_len={len(user)}"
            )

            response = await self._llm.generate_text(
                system_prompt=system,
                user_prompt=user,
                agent_name=self.name(),
                max_tokens=512,
                temperature=0.3,
                cache_system_prompt=True,
            )

            # ── DIAGNOSTIC: LLM returned ──
            logger.warning(
                f"DIAG IndicatorAgent: LLM returned | content_len={len(response.content)}"
            )

            result = self._parse_response(response.content)
            if result is None:
                logger.warning(
                    "DIAG IndicatorAgent: returning None at L96 — _parse_response returned None"
                )
            return result

        except Exception as exc:
            logger.warning(
                f"DIAG IndicatorAgent: returning None at L99 — "
                f"exception: {type(exc).__name__}: {exc}"
            )
            logger.exception("IndicatorAgent: analysis failed")
            return None

    def _parse_response(self, raw: str) -> SignalOutput | None:
        """Extract JSON from LLM response and build SignalOutput.

        Handles raw JSON, markdown code blocks, and partial responses.
        Returns None on any parse failure (SKIP is safe).
        """
        try:
            json_str = self._extract_json(raw)
            if json_str is None:
                logger.warning(
                    "DIAG IndicatorAgent: returning None at L110 — "
                    f"could not extract JSON from response: {raw[:200]!r}"
                )
                return None

            parsed = json.loads(json_str)

            direction = parsed.get("direction")
            if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                logger.warning(
                    f"DIAG IndicatorAgent: returning None at L118 — "
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

        except Exception as exc:
            logger.warning(
                f"DIAG IndicatorAgent: returning None at L148 — "
                f"parse exception: {type(exc).__name__}: {exc}"
            )
            logger.exception("IndicatorAgent: parse failed")
            return None

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON object from text, handling markdown code blocks."""
        # Try stripping markdown code fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        # Try finding a raw JSON object
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        return None
