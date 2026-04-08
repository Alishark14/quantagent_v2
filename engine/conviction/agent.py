"""ConvictionAgent: LLM meta-evaluator that scores signal consensus.

The most critical agent in the system. ConvictionAgent does NOT generate
a signal — it EVALUATES the quality and coherence of existing signals
from the Signal Layer and produces a continuous 0-1 conviction score.

On any parse failure, returns ConvictionOutput with score=0.0, direction="SKIP".
SKIP is always safe.

**Macro regime overlay (ARCHITECTURE §13.2.4):**

At the start of every `evaluate()` call the agent attempts to load
`macro_regime.json` from `self._macro_regime_path` (default: project
root). If the file exists AND is non-expired:

  * If the current time falls inside any `blackout_window`, the agent
    short-circuits the LLM call entirely and returns
    `conviction_score=0.0, direction="SKIP"` with a structured
    "Blackout window active: {reason}" reasoning. This is the
    §13.2.4 "don't open new positions" contract.
  * Otherwise, if `regime != "NEUTRAL"`, the agent injects a regime
    context paragraph into the LLM user prompt, applies the
    `conviction_threshold_boost` to the parsed conviction score
    (subtracts the boost so the trade only fires if intrinsic
    confidence > threshold + boost), and stamps the
    `position_size_multiplier` onto the output for DecisionAgent
    to apply when sizing.

A missing OR expired `macro_regime.json` is the safe default: the
agent proceeds with no overlay applied and logs an info message.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from engine.conviction.prompts.conviction_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.data.charts import generate_grounding_header
from engine.types import ConvictionOutput, MarketData, SignalOutput
from llm.base import LLMProvider, LLMResponse
from mcp.macro_regime.agent import MacroRegime, load_macro_regime

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"LONG", "SHORT", "SKIP"}
_VALID_REGIMES = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "BREAKOUT"}
_VALID_QUALITY = {"HIGH", "MEDIUM", "LOW", "CONFLICTING"}

# Display names rendered in the conviction prompt for known signal agents.
# Keys are ``SignalProducer.name()`` values; values are the display name
# the LLM sees in the SUBJECTIVE SIGNALS block. Insertion order is the
# rendering order — keep this stable so prompt-cache hits and eval
# calibration don't drift across cycles.
_AGENT_DISPLAY_NAMES: dict[str, str] = {
    "indicator_agent": "IndicatorAgent",
    "pattern_agent": "PatternAgent",
    "trend_agent": "TrendAgent",
    "flow_signal_agent": "FlowAgent",
}

# Agents whose signals always render a "Pattern:" line in the prompt.
# Other agents only render Pattern when ``pattern_detected`` is non-empty,
# matching the historical v1.0/v1.1 behaviour for IndicatorAgent / TrendAgent.
_PATTERN_LINE_ALWAYS_FOR: frozenset[str] = frozenset({"pattern_agent"})

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

_DEFAULT_MACRO_REGIME_PATH = Path("macro_regime.json")


class ConvictionAgent:
    """Meta-evaluator that scores signal quality and coherence.

    Receives all signal agent outputs + market data, calls a single LLM,
    and produces a ConvictionOutput with conviction score, regime, and reasoning.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        macro_regime_path: Path | str = _DEFAULT_MACRO_REGIME_PATH,
        clock=None,
    ) -> None:
        self._llm = llm_provider
        self._macro_regime_path = Path(macro_regime_path)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

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
        # ---- Macro overlay (§13.2.4) — load + blackout check FIRST ----
        macro = self._load_active_macro_regime()
        blackout = self._active_blackout(macro)
        if blackout is not None:
            logger.info(
                f"ConvictionAgent: blackout window active "
                f"({blackout.reason}), forcing conviction=0.0 for {market_data.symbol}"
            )
            return _blackout_skip(blackout.reason, macro)

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

            # Render the SUBJECTIVE SIGNALS block dynamically so the prompt
            # is N-signal aware. All known agents in `_AGENT_DISPLAY_NAMES`
            # are rendered in fixed order; any unknown agents land at the
            # bottom in alphabetical order. Missing known agents render
            # an explicit "Agent did not produce a signal" block so the
            # LLM still sees a stable structure.
            signals_block = self._build_signals_block(signal_map)

            user = USER_PROMPT.format(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                grounding_header=grounding,
                signals_block=signals_block,
                memory_context=memory_context,
            )

            # Append macro context as additional input to the LLM (NOT a hard
            # override on direction — the LLM sees it as one more signal).
            if macro is not None and macro.regime != "NEUTRAL":
                user = user + "\n\n" + _format_macro_context(macro)

            response = await self._llm.generate_text(
                system_prompt=system,
                user_prompt=user,
                agent_name="conviction_agent",
                max_tokens=768,
                temperature=0.3,
                cache_system_prompt=True,
            )

            output = self._parse_response(response)
            return self._apply_macro_overlay(output, macro)

        except Exception:
            logger.exception("ConvictionAgent: evaluation failed")
            return _SAFE_DEFAULT

    # ------------------------------------------------------------------
    # Macro regime helpers
    # ------------------------------------------------------------------

    def _load_active_macro_regime(self) -> MacroRegime | None:
        """Load `macro_regime.json` and discard if missing or expired.

        A missing or unparseable file is the safe default — the cycle
        proceeds with no overlay applied. Logged at INFO level so the
        first cycle after a deploy makes it obvious whether the file
        was found.
        """
        macro = load_macro_regime(self._macro_regime_path)
        if macro is None:
            logger.info(
                f"ConvictionAgent: no macro_regime.json at "
                f"{self._macro_regime_path} — proceeding without overlay"
            )
            return None
        if macro.error is not None:
            logger.info(
                f"ConvictionAgent: macro_regime.json carries error "
                f"{macro.error!r} — proceeding without overlay"
            )
            return None
        if _is_expired(macro, self._clock()):
            logger.info(
                f"ConvictionAgent: macro_regime.json expired at "
                f"{macro.expires} — proceeding without overlay"
            )
            return None
        return macro

    def _active_blackout(self, macro: MacroRegime | None):
        """Return the first blackout window covering `now`, or None."""
        if macro is None:
            return None
        now = self._clock()
        for window in macro.blackout_windows:
            if window.contains(now):
                return window
        return None

    def _apply_macro_overlay(
        self, output: ConvictionOutput, macro: MacroRegime | None
    ) -> ConvictionOutput:
        """Stamp the macro adjustments onto a parsed ConvictionOutput.

        Score itself is preserved (the LLM's actual judgment lands in
        the data moat unmutated). The boost + multiplier are stamped
        as new fields and read by DecisionAgent / threshold logic
        downstream.
        """
        if macro is None:
            return output
        output.macro_regime = macro.regime
        output.macro_threshold_boost = macro.adjustments.conviction_threshold_boost
        output.macro_position_size_multiplier = (
            macro.adjustments.position_size_multiplier
        )
        return output

    def _build_signal_map(self, signals: list[SignalOutput]) -> dict[str, dict]:
        """Map agent_name -> formatted signal dict for prompt injection."""
        result: dict[str, dict] = {}
        for signal in signals:
            result[signal.agent_name] = self._format_signal(signal)
        return result

    def _build_signals_block(self, signal_map: dict[str, dict]) -> str:
        """Render the SUBJECTIVE SIGNALS block from a signal_map.

        Known agents render in the fixed order declared by
        ``_AGENT_DISPLAY_NAMES``. Unknown agents (e.g. an experimental
        ML producer that nobody added to the display map yet) land at
        the bottom in alphabetical order so the prompt structure stays
        deterministic. Missing known agents render an explicit
        "Agent did not produce a signal" block — the LLM should always
        see a slot for every expected voice so silence is observable.
        """
        rendered_keys: set[str] = set()
        blocks: list[str] = []

        for key, display_name in _AGENT_DISPLAY_NAMES.items():
            sig = signal_map.get(key, self._empty_signal())
            blocks.append(self._render_signal_block(key, display_name, sig))
            rendered_keys.add(key)

        unknown_keys = sorted(set(signal_map.keys()) - rendered_keys)
        for key in unknown_keys:
            display_name = _fallback_display_name(key)
            blocks.append(self._render_signal_block(key, display_name, signal_map[key]))

        return "\n\n".join(blocks)

    @staticmethod
    def _render_signal_block(agent_key: str, display_name: str, sig: dict) -> str:
        """Render one agent's block in the prompt.

        Format mirrors the historical v1.0/v1.1 layout exactly for
        IndicatorAgent / PatternAgent / TrendAgent so existing eval
        calibration and prompt-cache hits don't drift. New agents
        (FlowAgent, future ML producers) follow the same layout.

        The Pattern line always renders for ``pattern_agent`` (matching
        the old hardcoded template) and renders for any other agent
        whose ``pattern_detected`` field is non-empty / non-"none".
        """
        lines = [
            f"{display_name}: direction={sig['direction']}, confidence={sig['confidence']}"
        ]
        pattern = sig.get("pattern", "none")
        always_pattern = agent_key in _PATTERN_LINE_ALWAYS_FOR
        if always_pattern or (pattern and pattern != "none"):
            lines.append(f"  Pattern: {pattern}")
        lines.append(f"  Reasoning: {sig['reasoning']}")
        lines.append(f"  Contradictions noted: {sig['contradictions']}")
        return "\n".join(lines)

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


def _fallback_display_name(agent_key: str) -> str:
    """Title-case an unknown agent's name for the SUBJECTIVE SIGNALS block.

    Used for any agent whose name() isn't in ``_AGENT_DISPLAY_NAMES`` —
    typically an experimental producer registered in dev. Converts
    ``"foo_bar_agent"`` → ``"FooBarAgent"``.
    """
    return "".join(part.capitalize() for part in agent_key.split("_"))


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


def _blackout_skip(reason: str, macro: MacroRegime | None) -> ConvictionOutput:
    """Return a forced-SKIP output when a blackout window is active.

    The conviction_score is hard-zero, the direction is SKIP, and the
    `macro_blackout_reason` field carries the event name (e.g.
    `"FOMC_ANNOUNCEMENT"`) so downstream logging / data moat capture
    can trace why the cycle was skipped without re-loading the macro
    file.
    """
    multiplier = 1.0
    boost = 0.0
    regime_label = "NEUTRAL"
    if macro is not None:
        multiplier = macro.adjustments.position_size_multiplier
        boost = macro.adjustments.conviction_threshold_boost
        regime_label = macro.regime
    return ConvictionOutput(
        conviction_score=0.0,
        direction="SKIP",
        regime="RANGING",
        regime_confidence=0.0,
        signal_quality="LOW",
        contradictions=[f"Blackout window active: {reason}"],
        reasoning=(
            f"Blackout window active: {reason}. "
            f"No new entries permitted."
        ),
        factual_weight=1.0,
        subjective_weight=0.0,
        raw_output="",
        macro_regime=regime_label,
        macro_threshold_boost=boost,
        macro_position_size_multiplier=multiplier,
        macro_blackout_reason=reason,
    )


def _is_expired(macro: MacroRegime, now: datetime) -> bool:
    """Compare `macro.expires` against `now`. Treat unparseable as expired."""
    if not macro.expires:
        return False  # no expiry → assume valid
    try:
        expires_dt = datetime.fromisoformat(macro.expires.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now > expires_dt


def _format_macro_context(macro: MacroRegime) -> str:
    """Render the non-NEUTRAL macro overlay as additional LLM context.

    The LLM sees this as one more signal — NOT as a hard override on
    direction. The blackout case is handled before the LLM is called,
    so this string only renders for active RISK_ON / RISK_OFF regimes.
    """
    avoid = ", ".join(macro.adjustments.avoid_assets) or "(none)"
    prefer = ", ".join(macro.adjustments.prefer_assets) or "(none)"
    return (
        "## MACRO REGIME OVERLAY\n"
        f"Current macro regime: {macro.regime} "
        f"(confidence={macro.confidence:.2f})\n"
        f"Reasoning: {macro.reasoning}\n"
        f"Conviction threshold boost: +{macro.adjustments.conviction_threshold_boost:.2f}\n"
        f"Position size multiplier: ×{macro.adjustments.position_size_multiplier:.2f}\n"
        f"Avoid assets: {avoid}\n"
        f"Prefer assets: {prefer}\n"
        "Treat this as ONE additional input — do NOT override your "
        "direction call solely on the macro regime."
    )
