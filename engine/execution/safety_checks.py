"""Mechanical safety checks: pyramid gate, daily loss limit, etc.

These are deterministic, non-negotiable, and cannot be overridden by the LLM.
They execute after DecisionAgent outputs its action but before any order is placed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from engine.types import Position

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of running all mechanical safety checks."""

    passed: bool
    original_action: str
    adjusted_action: str  # may differ if a check converts the action
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "original_action": self.original_action,
            "adjusted_action": self.adjusted_action,
            "violations": self.violations,
        }


def run_safety_checks(
    action: str,
    current_position: Position | None,
    daily_pnl: float,
    max_daily_loss: float,
    swing_highs: list[float],
    swing_lows: list[float],
    atr: float,
    conviction_score: float,
    entry_price: float | None = None,
    cost_model=None,
    symbol: str = "",
    position_size: float | None = None,
    sl_price: float | None = None,
    tp_price: float | None = None,
    direction: str = "",
    hold_hours: float = 8.0,
    min_rr: float = 1.0,
) -> SafetyCheckResult:
    """Run all 6 mechanical safety checks against a proposed action.

    Checks (in order):
    1. Conviction floor — score < 0.3 forces SKIP
    2. Daily loss limit — block new entries when daily loss exceeded
    3. Pyramid distance-to-resistance gate — ADD near S/R converts to HOLD
    4. Position count limit — 1 per symbol, block duplicate entries
    5. SL validation — entry actions require valid ATR for SL computation
    6. Cost viability — fee-adjusted R:R must meet minimum (if cost_model provided)

    Args:
        action: Proposed action (LONG, SHORT, ADD_LONG, ADD_SHORT, CLOSE_ALL, HOLD, SKIP).
        current_position: Existing position if any.
        daily_pnl: Realized + unrealized P&L for the day.
        max_daily_loss: Maximum allowed daily loss (negative number, e.g. -500).
        swing_highs: Nearest swing high levels.
        swing_lows: Nearest swing low levels.
        atr: Current ATR value.
        conviction_score: ConvictionAgent score (0-1).
        entry_price: Current/planned entry price (for pyramid gate distance check).
        cost_model: Optional ExecutionCostModel for cost viability check.
        symbol: Trading symbol (for cost check).
        position_size: Position size in USD (for cost check).
        sl_price: Stop-loss price (for cost check).
        tp_price: Take-profit price (for cost check).
        direction: Trade direction (for cost check).
        hold_hours: Expected hold duration (for cost check).
        min_rr: Minimum acceptable fee-adjusted R:R.

    Returns:
        SafetyCheckResult with pass/fail, adjusted action, and violation list.
    """
    violations: list[str] = []
    adjusted = action

    # -----------------------------------------------------------------------
    # 1. Conviction floor: score < 0.3 → force SKIP
    # -----------------------------------------------------------------------
    if conviction_score < 0.3 and action not in ("HOLD", "CLOSE_ALL", "SKIP"):
        violations.append(f"conviction_floor: score {conviction_score:.2f} < 0.3")
        adjusted = "SKIP"

    # -----------------------------------------------------------------------
    # 2. Daily loss limit: block new entries when max loss exceeded
    #    HOLD and CLOSE_ALL still permitted
    # -----------------------------------------------------------------------
    if daily_pnl <= max_daily_loss and action in ("LONG", "SHORT", "ADD_LONG", "ADD_SHORT"):
        violations.append(f"daily_loss_limit: P&L {daily_pnl:.2f} exceeds max {max_daily_loss:.2f}")
        adjusted = "SKIP" if adjusted in ("LONG", "SHORT") else "HOLD"

    # -----------------------------------------------------------------------
    # 3. Pyramid distance-to-resistance gate
    #    If ADD_LONG and price within 0.3*ATR of nearest swing high → HOLD
    #    If ADD_SHORT and price within 0.3*ATR of nearest swing low → HOLD
    # -----------------------------------------------------------------------
    if adjusted == "ADD_LONG" and entry_price is not None and swing_highs and atr > 0:
        nearest_resistance = min(swing_highs)  # closest swing high above
        distance = nearest_resistance - entry_price
        threshold = 0.3 * atr
        if distance <= threshold:
            violations.append(
                f"pyramid_gate: ADD_LONG within {distance:.2f} of resistance "
                f"{nearest_resistance:.2f} (threshold {threshold:.2f})"
            )
            adjusted = "HOLD"

    if adjusted == "ADD_SHORT" and entry_price is not None and swing_lows and atr > 0:
        nearest_support = max(swing_lows)  # closest swing low below
        distance = entry_price - nearest_support
        threshold = 0.3 * atr
        if distance <= threshold:
            violations.append(
                f"pyramid_gate: ADD_SHORT within {distance:.2f} of support "
                f"{nearest_support:.2f} (threshold {threshold:.2f})"
            )
            adjusted = "HOLD"

    # -----------------------------------------------------------------------
    # 4. Position count limit: 1 position per symbol per bot
    #    Block new entry if position already exists
    # -----------------------------------------------------------------------
    if adjusted in ("LONG", "SHORT") and current_position is not None:
        violations.append(
            f"position_limit: already have {current_position.direction} "
            f"position size {current_position.size}"
        )
        adjusted = "HOLD"

    # -----------------------------------------------------------------------
    # 5. SL validation: entry actions need valid ATR for SL computation
    #    If ATR is zero or negative, cannot compute a valid stop-loss
    # -----------------------------------------------------------------------
    if adjusted in ("LONG", "SHORT", "ADD_LONG", "ADD_SHORT") and atr <= 0:
        violations.append(f"sl_validation: ATR {atr} is not valid for SL computation")
        adjusted = "SKIP"

    # -----------------------------------------------------------------------
    # 6. Cost viability: fee-adjusted R:R must meet minimum
    #    Only if cost_model is provided and action is an entry
    # -----------------------------------------------------------------------
    if (
        cost_model is not None
        and adjusted in ("LONG", "SHORT")
        and entry_price
        and sl_price
        and tp_price
        and position_size
        and position_size > 0
    ):
        sl_dist_pct = abs(entry_price - sl_price) / entry_price if entry_price > 0 else 0
        tp_dist_pct = abs(tp_price - entry_price) / entry_price if entry_price > 0 else 0
        if sl_dist_pct > 0 and tp_dist_pct > 0:
            viable, reason = cost_model.is_trade_viable(
                symbol=symbol,
                position_value=position_size,
                sl_distance_pct=sl_dist_pct,
                tp_distance_pct=tp_dist_pct,
                direction=direction or adjusted,
                hold_hours=hold_hours,
                min_rr=min_rr,
            )
            if not viable:
                violations.append(f"cost_viability: {reason}")
                adjusted = "SKIP"

    passed = len(violations) == 0
    return SafetyCheckResult(
        passed=passed,
        original_action=action,
        adjusted_action=adjusted,
        violations=violations,
    )
