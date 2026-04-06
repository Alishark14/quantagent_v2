"""DecisionAgent: LLM-based action selection from conviction output.

Receives a PRE-FILTERED conviction score — the hard analytical work is done.
DecisionAgent decides trade mechanics: action, sizing, risk parameters.
On any failure, defaults to HOLD (if position open) or SKIP (if no position).
"""

from __future__ import annotations

import json
import logging
import re

from engine.config import (
    DEFAULT_PROFILES,
    TradingConfig,
    TimeframeProfile,
    get_dynamic_profile,
)
from engine.execution.prompts.decision_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from engine.execution.risk_profiles import compute_position_size, compute_sl_tp
from engine.execution.safety_checks import SafetyCheckResult, run_safety_checks
from engine.types import ConvictionOutput, MarketData, Position, TradeAction
from llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"LONG", "SHORT", "ADD_LONG", "ADD_SHORT", "CLOSE_ALL", "HOLD", "SKIP"}

# Conviction-based size multipliers (per ARCHITECTURE.md sec 7.4)
_CONVICTION_SIZE_MULTIPLIER = {
    # score range -> multiplier
    "moderate": 1.0,    # 0.5 - 0.7
    "high": 1.15,       # 0.7 - 0.85
    "very_high": 1.3,   # 0.85 - 1.0
}


def _safe_default(conviction_score: float, has_position: bool, raw: str = "") -> TradeAction:
    """Return a safe default action: HOLD if position open, SKIP if not."""
    action = "HOLD" if has_position else "SKIP"
    return TradeAction(
        action=action,
        conviction_score=conviction_score,
        position_size=None,
        sl_price=None,
        tp1_price=None,
        tp2_price=None,
        rr_ratio=None,
        atr_multiplier=None,
        reasoning=f"Parse failure — defaulting to {action}.",
        raw_output=raw,
    )


class DecisionAgent:
    """LLM-based agent that selects trade actions from conviction output.

    The conviction scoring is already done by ConvictionAgent. DecisionAgent
    focuses on: action selection, SL/TP computation, position sizing, and
    safety check enforcement.
    """

    def __init__(self, llm_provider: LLMProvider, config: TradingConfig) -> None:
        self._llm = llm_provider
        self._config = config

    async def decide(
        self,
        conviction: ConvictionOutput,
        market_data: MarketData,
        current_position: Position | None,
        account_balance: float,
        memory_context: str = "",
    ) -> TradeAction:
        """Decide on a trade action given conviction and position state.

        Steps:
        1. Quick exit: conviction below threshold → SKIP without LLM call
        2. Call LLM for action decision
        3. Parse response
        4. Compute SL/TP and position size for entry actions
        5. Run safety checks (may override action)
        6. Return final TradeAction

        Args:
            conviction: ConvictionOutput from ConvictionAgent.
            market_data: Full MarketData with indicators, swings, etc.
            current_position: Existing position or None.
            account_balance: Account balance in USD.
            memory_context: Formatted string with cycle memory.

        Returns:
            TradeAction with all fields populated.
        """
        has_position = current_position is not None

        # 1. Quick exit: conviction below threshold → SKIP without LLM call
        if conviction.conviction_score < self._config.conviction_threshold:
            return TradeAction(
                action="SKIP",
                conviction_score=conviction.conviction_score,
                position_size=None,
                sl_price=None,
                tp1_price=None,
                tp2_price=None,
                rr_ratio=None,
                atr_multiplier=None,
                reasoning=f"Conviction {conviction.conviction_score:.2f} below threshold {self._config.conviction_threshold}.",
                raw_output="",
            )

        try:
            # 2. Format and call LLM
            current_price = self._get_current_price(market_data)
            atr = self._get_atr(market_data)

            position_context = self._format_position_context(current_position)

            system = SYSTEM_PROMPT.format(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
            )
            user = USER_PROMPT.format(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                conviction_score=f"{conviction.conviction_score:.2f}",
                conviction_direction=conviction.direction,
                regime=conviction.regime,
                signal_quality=conviction.signal_quality,
                contradictions=", ".join(conviction.contradictions) if conviction.contradictions else "none",
                position_context=position_context,
                account_balance=account_balance,
                current_price=f"{current_price:.2f}" if current_price else "N/A",
                atr=f"{atr:.4f}" if atr else "N/A",
                memory_context=memory_context or "No prior history.",
            )

            response = await self._llm.generate_text(
                system_prompt=system,
                user_prompt=user,
                agent_name="decision_agent",
                max_tokens=512,
                temperature=0.2,
                cache_system_prompt=True,
            )

            # 3. Parse LLM response
            action, reasoning, suggested_rr = self._parse_response(response.content, has_position)

            # 4. Compute SL/TP and position size for entry actions
            sl_price = None
            tp1_price = None
            tp2_price = None
            rr_ratio = None
            atr_multiplier = None
            position_size = None

            if action in ("LONG", "SHORT", "ADD_LONG", "ADD_SHORT") and current_price and atr > 0:
                # Get dynamic profile
                profile = self._get_profile(market_data.timeframe, conviction.regime, market_data)
                atr_multiplier = profile.atr_multiplier

                direction = "LONG" if action in ("LONG", "ADD_LONG") else "SHORT"
                sl_tp = compute_sl_tp(
                    entry_price=current_price,
                    direction=direction,
                    atr=atr,
                    profile=profile,
                    swing_highs=market_data.swing_highs,
                    swing_lows=market_data.swing_lows,
                )
                sl_price = sl_tp["sl_price"]
                tp1_price = sl_tp["tp1_price"]
                tp2_price = sl_tp["tp2_price"]
                rr_ratio = sl_tp["rr_ratio"]

                # Override RR if LLM suggested one
                if suggested_rr is not None and suggested_rr > 0:
                    rr_ratio = suggested_rr

                # Position sizing
                risk_per_trade = 0.01  # 1% risk per trade
                size_mult = self._conviction_size_multiplier(conviction.conviction_score)
                is_pyramid = action in ("ADD_LONG", "ADD_SHORT")

                position_size = compute_position_size(
                    account_balance=account_balance,
                    risk_per_trade=risk_per_trade,
                    entry_price=current_price,
                    sl_price=sl_price,
                    max_position_pct=self._config.max_position_pct,
                )
                position_size *= size_mult

                # Pyramid adds are 50% of base size
                if is_pyramid:
                    position_size *= 0.5

                position_size = round(position_size, 2)

            # 5. Run safety checks
            daily_pnl = 0.0  # Will be injected by pipeline in production
            max_daily_loss = -500.0  # Will come from config in production

            safety = run_safety_checks(
                action=action,
                current_position=current_position,
                daily_pnl=daily_pnl,
                max_daily_loss=max_daily_loss,
                swing_highs=market_data.swing_highs,
                swing_lows=market_data.swing_lows,
                atr=atr,
                conviction_score=conviction.conviction_score,
                entry_price=current_price,
            )

            if not safety.passed:
                logger.info(
                    f"DecisionAgent: safety checks adjusted {action} → {safety.adjusted_action}: "
                    f"{safety.violations}"
                )
                action = safety.adjusted_action
                reasoning = f"{reasoning} [Safety override: {'; '.join(safety.violations)}]"

                # Clear sizing for non-entry actions
                if action in ("HOLD", "SKIP"):
                    position_size = None
                    sl_price = None
                    tp1_price = None
                    tp2_price = None
                    rr_ratio = None
                    atr_multiplier = None

            return TradeAction(
                action=action,
                conviction_score=conviction.conviction_score,
                position_size=position_size,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                rr_ratio=rr_ratio,
                atr_multiplier=atr_multiplier,
                reasoning=reasoning,
                raw_output=response.content,
            )

        except Exception:
            logger.exception("DecisionAgent: decision failed")
            return _safe_default(conviction.conviction_score, has_position)

    def _parse_response(
        self, raw: str, has_position: bool
    ) -> tuple[str, str, float | None]:
        """Parse LLM response into (action, reasoning, suggested_rr).

        On failure, returns safe defaults: HOLD if position, SKIP if not.
        """
        try:
            json_str = self._extract_json(raw)
            if json_str is None:
                logger.warning("DecisionAgent: could not extract JSON from response")
                default_action = "HOLD" if has_position else "SKIP"
                return default_action, "Could not parse LLM response.", None

            parsed = json.loads(json_str)

            action = parsed.get("action", "SKIP")
            if action not in _VALID_ACTIONS:
                logger.warning(f"DecisionAgent: invalid action '{action}', defaulting to safe action")
                action = "HOLD" if has_position else "SKIP"

            reasoning = str(parsed.get("reasoning", ""))
            suggested_rr = parsed.get("suggested_rr")
            if suggested_rr is not None:
                try:
                    suggested_rr = float(suggested_rr)
                except (ValueError, TypeError):
                    suggested_rr = None

            return action, reasoning, suggested_rr

        except Exception:
            logger.exception("DecisionAgent: parse failed")
            default_action = "HOLD" if has_position else "SKIP"
            return default_action, "Parse failure — using safe default.", None

    @staticmethod
    def _format_position_context(position: Position | None) -> str:
        """Format current position state for prompt injection."""
        if position is None:
            return "No open position."
        return (
            f"Direction: {position.direction.upper()}\n"
            f"Size: {position.size}\n"
            f"Entry Price: {position.entry_price}\n"
            f"Unrealized P&L: ${position.unrealized_pnl:.2f}\n"
            f"Leverage: {position.leverage or 'N/A'}"
        )

    @staticmethod
    def _get_current_price(market_data: MarketData) -> float:
        """Get the latest close price from candles."""
        if market_data.candles:
            return float(market_data.candles[-1].get("close", 0))
        return 0.0

    @staticmethod
    def _get_atr(market_data: MarketData) -> float:
        """Extract ATR from indicators."""
        return float(market_data.indicators.get("atr", 0))

    def _get_profile(
        self, timeframe: str, regime: str, market_data: MarketData
    ) -> TimeframeProfile:
        """Get the regime-adjusted timeframe profile."""
        base = DEFAULT_PROFILES.get(timeframe)
        if base is None:
            base = DEFAULT_PROFILES["1h"]

        vol_pct = float(market_data.indicators.get("volatility_percentile", 50.0))
        return get_dynamic_profile(base, regime, vol_pct)

    @staticmethod
    def _conviction_size_multiplier(score: float) -> float:
        """Return position size multiplier based on conviction tier."""
        if score >= 0.85:
            return _CONVICTION_SIZE_MULTIPLIER["very_high"]
        elif score >= 0.7:
            return _CONVICTION_SIZE_MULTIPLIER["high"]
        else:
            return _CONVICTION_SIZE_MULTIPLIER["moderate"]

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
