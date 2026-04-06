"""Dynamic regime-driven risk profiles.

Computes SL/TP levels and position sizes based on ATR, swing structure,
and the active TimeframeProfile (which is already regime-adjusted by
get_dynamic_profile in engine/config.py).

When an ExecutionCostModel is provided, position sizing is cost-aware:
fees depend on size, size depends on fees — solved iteratively.
"""

from __future__ import annotations

import logging

from engine.config import TimeframeProfile

logger = logging.getLogger(__name__)


def compute_sl_tp(
    entry_price: float,
    direction: str,
    atr: float,
    profile: TimeframeProfile,
    swing_highs: list[float],
    swing_lows: list[float],
) -> dict:
    """Compute SL, TP1, and TP2 from ATR, profile, and swing structure.

    Logic:
    1. ATR-based SL = entry ± (ATR * profile.atr_multiplier)
    2. Snap SL to nearest swing structure if it provides a tighter stop
       (better risk) but is at least 0.2*ATR away from entry (not too tight).
    3. TP1 at 1:1 risk/reward (50% position close).
    4. TP2 at full RR from profile (rr_min as minimum target).

    Returns dict with sl_price, tp1_price, tp2_price, rr_ratio, risk_distance.
    """
    atr_distance = atr * profile.atr_multiplier
    min_distance = atr * 0.2  # minimum SL distance to avoid noise stops

    if direction == "LONG":
        atr_sl = entry_price - atr_distance

        # Snap to swing low if it gives a tighter (higher) stop
        structural_sl = _snap_sl_long(entry_price, atr_sl, swing_lows, min_distance)
        sl_price = structural_sl

        risk_distance = entry_price - sl_price
        if risk_distance <= 0:
            risk_distance = atr_distance
            sl_price = entry_price - atr_distance

        tp1_price = entry_price + risk_distance  # 1:1 RR
        tp2_price = entry_price + risk_distance * profile.rr_min  # full RR target
        rr_ratio = profile.rr_min

    elif direction == "SHORT":
        atr_sl = entry_price + atr_distance

        structural_sl = _snap_sl_short(entry_price, atr_sl, swing_highs, min_distance)
        sl_price = structural_sl

        risk_distance = sl_price - entry_price
        if risk_distance <= 0:
            risk_distance = atr_distance
            sl_price = entry_price + atr_distance

        tp1_price = entry_price - risk_distance  # 1:1 RR
        tp2_price = entry_price - risk_distance * profile.rr_min
        rr_ratio = profile.rr_min

    else:
        return {
            "sl_price": 0.0,
            "tp1_price": 0.0,
            "tp2_price": 0.0,
            "rr_ratio": 0.0,
            "risk_distance": 0.0,
        }

    return {
        "sl_price": round(sl_price, 8),
        "tp1_price": round(tp1_price, 8),
        "tp2_price": round(tp2_price, 8),
        "rr_ratio": round(rr_ratio, 4),
        "risk_distance": round(risk_distance, 8),
    }


def _snap_sl_long(
    entry: float,
    atr_sl: float,
    swing_lows: list[float],
    min_distance: float,
) -> float:
    """For a LONG: snap SL to a swing low if it's tighter than ATR-based SL.

    Pick the highest swing low that is still below entry by at least min_distance
    and that is higher than (or equal to) the ATR-based SL (i.e., tighter stop).
    """
    candidates = [
        s for s in swing_lows
        if s >= atr_sl and (entry - s) >= min_distance
    ]
    if candidates:
        return max(candidates)  # highest = tightest valid stop
    return atr_sl


def _snap_sl_short(
    entry: float,
    atr_sl: float,
    swing_highs: list[float],
    min_distance: float,
) -> float:
    """For a SHORT: snap SL to a swing high if it's tighter than ATR-based SL.

    Pick the lowest swing high that is still above entry by at least min_distance
    and that is lower than (or equal to) the ATR-based SL (i.e., tighter stop).
    """
    candidates = [
        s for s in swing_highs
        if s <= atr_sl and (s - entry) >= min_distance
    ]
    if candidates:
        return min(candidates)  # lowest = tightest valid stop
    return atr_sl


def compute_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    sl_price: float,
    max_position_pct: float = 1.0,
    min_size_usd: float = 20.0,
    cost_model=None,
    symbol: str = "",
    direction: str = "LONG",
    hold_hours: float = 8.0,
) -> float:
    """Compute position size from risk budget and SL distance.

    When cost_model is provided, uses cost-aware sizing that accounts
    for fees, slippage, spread, and funding in the risk budget.
    Otherwise falls back to naive division.

    Args:
        account_balance: Total account value in USD.
        risk_per_trade: Fraction of account to risk (e.g. 0.01 = 1%).
        entry_price: Planned entry price.
        sl_price: Stop-loss price.
        max_position_pct: Maximum position as fraction of balance.
        min_size_usd: Minimum position size in USD.
        cost_model: Optional ExecutionCostModel for cost-aware sizing.
        symbol: Trading symbol (required if cost_model provided).
        direction: Trade direction (required if cost_model provided).
        hold_hours: Expected hold duration in hours.

    Returns:
        Position size in USD (0 if trade not viable due to costs).
    """
    if account_balance <= 0 or entry_price <= 0:
        return 0.0

    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return 0.0

    sl_pct = sl_distance / entry_price

    # Cost-aware path
    if cost_model is not None and symbol:
        result = cost_model.compute_cost_aware_position_size(
            symbol=symbol,
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            sl_distance_pct=sl_pct,
            direction=direction,
            hold_hours=hold_hours,
            max_position_pct=max_position_pct,
        )
        if not result.viable:
            logger.warning(f"Position size 0: {result.reason}")
            return 0.0
        return max(result.size, min_size_usd) if result.size > 0 else 0.0

    # Naive path (no cost model)
    risk_amount = account_balance * risk_per_trade
    position_usd = risk_amount / sl_pct

    max_position = account_balance * max_position_pct
    position_usd = min(position_usd, max_position)
    position_usd = max(position_usd, min_size_usd)
    position_usd = min(position_usd, max_position)

    return round(position_usd, 2)
