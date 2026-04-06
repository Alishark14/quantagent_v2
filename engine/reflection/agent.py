"""ReflectionAgent: async post-trade analysis and rule distillation.

Runs after a trade closes. Compares entry reasoning to actual outcome
and distills ONE actionable, testable rule. Saves the rule to the
repository and emits a RuleGenerated event.

On any parse failure, returns None (never generates bad rules).
"""

from __future__ import annotations

import json
import logging
import re

from engine.events import EventBus, RuleGenerated
from engine.reflection.prompts.reflection_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.memory.reflection_rules import ReflectionRules
from llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Post-trade analyst that distills trading rules from outcomes.

    Designed to run asynchronously after TradeClosed events. Each
    reflection produces at most one rule, saved to the repository.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        rules: ReflectionRules,
        event_bus: EventBus,
    ) -> None:
        self._llm = llm_provider
        self._rules = rules
        self._bus = event_bus

    async def reflect(
        self,
        trade_data: dict,
        cycle_data: dict,
    ) -> dict | None:
        """Analyze a completed trade and distill a rule.

        Args:
            trade_data: Trade record with entry/exit prices, P&L, exit_reason, etc.
            cycle_data: Cycle record from entry time with signals, indicators, conviction.

        Returns:
            The saved rule dict, or None if no rule could be distilled.
        """
        try:
            user = self._format_user_prompt(trade_data, cycle_data)

            response = await self._llm.generate_text(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user,
                agent_name="reflection_agent",
                max_tokens=512,
                temperature=0.4,
                cache_system_prompt=True,
            )

            rule = self._parse_response(response.content, trade_data)
            if rule is None:
                return None

            # Save to repository
            rule_id = await self._rules.save_rule(rule)
            rule["id"] = rule_id

            # Emit event
            try:
                await self._bus.publish(RuleGenerated(
                    source="reflection_agent",
                    rule=rule,
                ))
            except Exception:
                logger.warning("ReflectionAgent: failed to emit RuleGenerated", exc_info=True)

            logger.info(
                f"ReflectionAgent: rule generated for {trade_data.get('symbol', '?')}: "
                f"{rule['rule_text']}"
            )
            return rule

        except Exception:
            logger.exception("ReflectionAgent: reflection failed")
            return None

    def _format_user_prompt(self, trade_data: dict, cycle_data: dict) -> str:
        """Build the user prompt from trade and cycle data."""
        # Extract signal details from cycle_data
        signals = cycle_data.get("signals") or cycle_data.get("signals_json") or []
        if isinstance(signals, str):
            try:
                signals = json.loads(signals)
            except (json.JSONDecodeError, TypeError):
                signals = []

        signal_map: dict[str, dict] = {}
        if isinstance(signals, list):
            for s in signals:
                if isinstance(s, dict):
                    signal_map[s.get("agent_name", "")] = s

        ind = signal_map.get("indicator_agent", {})
        pat = signal_map.get("pattern_agent", {})
        trend = signal_map.get("trend_agent", {})

        # Extract conviction
        conviction = cycle_data.get("conviction") or cycle_data.get("conviction_json") or {}
        if isinstance(conviction, str):
            try:
                conviction = json.loads(conviction)
            except (json.JSONDecodeError, TypeError):
                conviction = {}

        # Indicators summary
        indicators = cycle_data.get("indicators") or cycle_data.get("indicators_json") or {}
        if isinstance(indicators, str):
            try:
                indicators = json.loads(indicators)
            except (json.JSONDecodeError, TypeError):
                indicators = {}
        indicators_summary = self._format_indicators(indicators)

        # Outcome narrative
        pnl = trade_data.get("pnl", 0)
        outcome = "WIN" if pnl and pnl > 0 else "LOSS" if pnl and pnl < 0 else "BREAKEVEN"
        exit_reason = trade_data.get("exit_reason", "unknown")
        narrative = (
            f"Trade was a {outcome} (P&L: {pnl}). "
            f"Exited via {exit_reason}."
        )

        return USER_PROMPT.format(
            symbol=trade_data.get("symbol", "?"),
            timeframe=trade_data.get("timeframe", "?"),
            direction=trade_data.get("direction", "?"),
            entry_price=trade_data.get("entry_price", "?"),
            exit_price=trade_data.get("exit_price", "?"),
            pnl=pnl,
            r_multiple=trade_data.get("r_multiple", "?"),
            exit_reason=exit_reason,
            duration=trade_data.get("duration", "?"),
            conviction_score=conviction.get("conviction_score", "?"),
            regime=conviction.get("regime", "?"),
            ind_direction=ind.get("direction", "N/A"),
            ind_confidence=ind.get("confidence", "?"),
            pat_direction=pat.get("direction", "N/A"),
            pat_confidence=pat.get("confidence", "?"),
            pat_pattern=pat.get("pattern_detected", "none"),
            trend_direction=trend.get("direction", "N/A"),
            trend_confidence=trend.get("confidence", "?"),
            indicators_summary=indicators_summary,
            outcome_narrative=narrative,
        )

    @staticmethod
    def _format_indicators(indicators: dict) -> str:
        """Format indicator dict into a readable summary."""
        if not indicators:
            return "No indicators available."
        parts = []
        for key, val in indicators.items():
            if isinstance(val, dict):
                sub = ", ".join(f"{k}={v}" for k, v in val.items() if k != "classification")
                parts.append(f"{key}: {sub}")
            else:
                parts.append(f"{key}: {val}")
        return "\n".join(parts)

    def _parse_response(self, raw: str, trade_data: dict) -> dict | None:
        """Parse LLM response into a rule dict. Returns None on failure."""
        try:
            json_str = self._extract_json(raw)
            if json_str is None:
                logger.warning("ReflectionAgent: could not extract JSON from response")
                return None

            parsed = json.loads(json_str)

            rule_text = parsed.get("rule")
            if rule_text is None:
                logger.info("ReflectionAgent: LLM returned null rule (ambiguous trade)")
                return None

            rule_text = str(rule_text).strip()
            if not rule_text or len(rule_text) < 10:
                logger.warning(f"ReflectionAgent: rule too short: '{rule_text}'")
                return None

            reasoning = str(parsed.get("reasoning", ""))
            applies_to = parsed.get("applies_to", "all")
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            symbol = trade_data.get("symbol", "all")
            timeframe = trade_data.get("timeframe", "all")

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "rule_text": rule_text,
                "score": 0,
                "active": True,
                "reasoning": reasoning,
                "applies_to": applies_to,
                "confidence": confidence,
                "source_trade": trade_data.get("id"),
                "raw_output": raw,
            }

        except Exception:
            logger.exception("ReflectionAgent: parse failed")
            return None

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


def create_reflection_handler(
    agent: ReflectionAgent,
    trade_repo,
    cycle_repo,
) -> callable:
    """Create an event handler that triggers reflection on TradeClosed.

    Usage:
        handler = create_reflection_handler(agent, trade_repo, cycle_repo)
        bus.subscribe(TradeClosed, handler)
    """
    async def _on_trade_closed(event) -> None:
        try:
            # In production, fetch full trade and cycle data from repos
            trade_data = {
                "symbol": event.symbol,
                "pnl": event.pnl,
                "exit_reason": event.exit_reason,
            }
            cycle_data = {}
            await agent.reflect(trade_data, cycle_data)
        except Exception:
            logger.exception("ReflectionAgent: event handler failed")

    return _on_trade_closed
