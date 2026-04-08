"""Backtest metrics — performance / risk analytics for completed runs.

ARCHITECTURE.md §31.3.6 enumerates the standardised metrics every
backtest run produces. This module is the single source of truth for
those calculations so Tier 1 (mechanical), Tier 2 (replay), and Tier 3
(LLM, future) all report identically.

Sharpe convention:
- Bucket the equity curve into end-of-day snapshots
- Daily return = eod[i] / eod[i-1] - 1
- Sharpe = mean(returns) / stdev(returns) * sqrt(365)  (crypto: 24/7)
- Returns 0.0 if there are fewer than 2 daily snapshots or stdev ≈ 0

Calmar convention:
- annual_return = (final / initial) ^ (1/years) - 1
- Calmar = annual_return / max_drawdown
- Returns 0.0 if max_drawdown ≈ 0 or years ≤ 0

R-multiple convention:
- Per-trade: trade["pnl"] / risk_amount
- risk_amount = abs(entry_price - sl_price) * size if sl_price present in
  the trade dict, else falls back to (initial_balance × risk_per_trade)
- This means metrics work with both today's bare-bones sim trade records
  *and* the richer schema we'll move to once SL is recorded per trade.

Skip rate convention:
- skip_rate = (setups_detected - setups_taken) / setups_detected
- Returns 0.0 when no setups were detected (correct: there was nothing to
  either skip or take)
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# Cap profit factor when there are zero losses — "infinity" is unfriendly
# in JSON and on the dashboard. The cap is high enough to be unmistakably
# "no losses" but small enough to render cleanly.
_PROFIT_FACTOR_CAP = 999.9
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass
class BacktestMetrics:
    """Standardised backtest performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    loss_rate: float
    avg_r_multiple: float
    profit_factor: float           # gross_profit / gross_loss (capped at 999.9)
    sharpe_ratio: float            # annualised, daily returns × √365
    calmar_ratio: float            # annual_return / max_drawdown
    max_drawdown_pct: float        # peak-to-trough %
    max_drawdown_duration: int     # equity-curve points (≈ candles)
    avg_trade_duration_hours: float
    longest_win_streak: int
    longest_loss_streak: int
    skip_rate: float               # % of detected setups skipped
    total_pnl: float
    total_fees: float
    cost_adjusted_pnl: float       # = total_pnl (already net of fees in sim)
    initial_balance: float
    final_balance: float
    return_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_metrics(
    trade_history: list[dict],
    equity_curve: list[tuple[int, float]],
    config: "BacktestConfigLike",
    setups_detected: int = 0,
    setups_taken: int = 0,
) -> BacktestMetrics:
    """Compute the full BacktestMetrics tuple.

    Args:
        trade_history: List of completed-trade dicts. Required keys:
            ``pnl``, ``fee``. Optional keys (used when present):
            ``entry_timestamp``, ``timestamp`` (close ts), ``sl_price``,
            ``entry_price``, ``size``.
        equity_curve: List of ``(timestamp_ms, equity)`` tuples in
            chronological order.
        config: Anything with ``initial_balance`` and ``risk_per_trade``
            attributes (typically ``BacktestConfig``).
        setups_detected: Count of SetupDetected events. Used for skip
            rate. 0 → skip_rate = 0 (no setups means nothing to skip).
        setups_taken: Count of cycles that actually opened a trade.

    Returns:
        BacktestMetrics with every field populated. Edge cases (zero
        trades, all winners, single trade) all return safe values.
    """
    initial_balance = float(getattr(config, "initial_balance", 0.0))
    final_balance = equity_curve[-1][1] if equity_curve else initial_balance
    return_pct = _safe_pct(final_balance - initial_balance, initial_balance)

    # ----- Trade-level aggregates -----
    n = len(trade_history)
    wins = [t for t in trade_history if t["pnl"] > 0]
    losses = [t for t in trade_history if t["pnl"] < 0]
    # ties (pnl == 0) count as neither win nor loss

    total_pnl = sum(t["pnl"] for t in trade_history)
    total_fees = sum(t.get("fee", 0.0) for t in trade_history)

    win_rate = len(wins) / n if n else 0.0
    loss_rate = len(losses) / n if n else 0.0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = _profit_factor(gross_profit, gross_loss)

    avg_r_multiple = _avg_r_multiple(trade_history, config)
    avg_duration_hours = _avg_trade_duration_hours(trade_history)
    longest_win, longest_loss = _streaks(trade_history)

    # ----- Equity-curve aggregates -----
    max_dd_decimal, max_dd_duration = _max_drawdown(equity_curve)
    sharpe = _sharpe_annualised(equity_curve)
    calmar = _calmar(equity_curve, max_dd_decimal)

    # ----- Skip rate -----
    skip_rate = (
        (setups_detected - setups_taken) / setups_detected
        if setups_detected > 0
        else 0.0
    )

    return BacktestMetrics(
        total_trades=n,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=round(win_rate, 6),
        loss_rate=round(loss_rate, 6),
        avg_r_multiple=round(avg_r_multiple, 6),
        profit_factor=round(profit_factor, 4),
        sharpe_ratio=round(sharpe, 6),
        calmar_ratio=round(calmar, 6),
        max_drawdown_pct=round(max_dd_decimal * 100, 6),
        max_drawdown_duration=max_dd_duration,
        avg_trade_duration_hours=round(avg_duration_hours, 4),
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        skip_rate=round(skip_rate, 6),
        total_pnl=round(total_pnl, 6),
        total_fees=round(total_fees, 6),
        cost_adjusted_pnl=round(total_pnl, 6),  # sim PnL is already net of fees
        initial_balance=round(initial_balance, 6),
        final_balance=round(final_balance, 6),
        return_pct=round(return_pct * 100, 6),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss > 0:
        return min(gross_profit / gross_loss, _PROFIT_FACTOR_CAP)
    if gross_profit > 0:
        return _PROFIT_FACTOR_CAP  # all wins, no losses → cap
    return 0.0  # no trades or only zero-pnl ties


def _avg_r_multiple(trade_history: list[dict], config) -> float:
    """Mean R-multiple across trades.

    Per-trade risk = ``abs(entry - sl) * size`` if both ``sl_price`` and
    ``entry_price`` are present in the trade dict; otherwise falls back
    to the configured fixed-risk amount (initial_balance × risk_per_trade).
    """
    if not trade_history:
        return 0.0

    fallback_risk = float(getattr(config, "initial_balance", 0.0)) * float(
        getattr(config, "risk_per_trade", 0.01)
    )

    rs: list[float] = []
    for t in trade_history:
        risk = _trade_risk(t, fallback_risk)
        if risk > 0:
            rs.append(t["pnl"] / risk)
    if not rs:
        return 0.0
    return sum(rs) / len(rs)


def _trade_risk(trade: dict, fallback_risk: float) -> float:
    sl = trade.get("sl_price")
    entry = trade.get("entry_price")
    size = trade.get("size")
    if sl is not None and entry is not None and size is not None:
        return abs(float(entry) - float(sl)) * float(size)
    return fallback_risk


def _avg_trade_duration_hours(trade_history: list[dict]) -> float:
    """Mean (close_ts - entry_ts) in hours.

    Uses ``entry_timestamp`` (open) and ``timestamp`` (close, set by the
    sim adapter on close). Trades missing either field are skipped.
    Returns 0.0 if no trade has timing info — better than NaN for the
    JSON / HTML report renderers.
    """
    if not trade_history:
        return 0.0
    spans_ms: list[float] = []
    for t in trade_history:
        entry = t.get("entry_timestamp")
        close = t.get("timestamp")
        # Use `is not None` rather than truthiness — entry_timestamp == 0
        # is a legitimate value (start of epoch / synthetic test fixture).
        if entry is not None and close is not None and close >= entry:
            spans_ms.append(float(close - entry))
    if not spans_ms:
        return 0.0
    return (sum(spans_ms) / len(spans_ms)) / 3_600_000.0


def _streaks(trade_history: list[dict]) -> tuple[int, int]:
    """Walk trades in chronological order; return (longest_win, longest_loss)."""
    if not trade_history:
        return 0, 0
    longest_win = 0
    longest_loss = 0
    cur_win = 0
    cur_loss = 0
    for t in trade_history:
        pnl = t["pnl"]
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
            if cur_win > longest_win:
                longest_win = cur_win
        elif pnl < 0:
            cur_loss += 1
            cur_win = 0
            if cur_loss > longest_loss:
                longest_loss = cur_loss
        else:
            # Flat trade breaks both streaks (conservative)
            cur_win = 0
            cur_loss = 0
    return longest_win, longest_loss


def _max_drawdown(equity_curve: list[tuple[int, float]]) -> tuple[float, int]:
    """Walk the curve, return (max_drawdown_decimal, duration_in_points).

    Drawdown duration = number of equity points from the peak that
    started the worst episode until equity recovers above that peak (or
    end of curve if it never recovers).
    """
    if not equity_curve:
        return 0.0, 0

    peak = -math.inf
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_recover_idx = 0

    for i, (_, eq) in enumerate(equity_curve):
        if eq > peak:
            peak = eq
            peak_idx = i
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_recover_idx = i  # current low point of the worst episode

    # Walk forward from the peak that anchored the worst DD until equity
    # recovers above that peak (or fall off the end).
    if max_dd > 0:
        anchor_peak = equity_curve[max_dd_peak_idx][1]
        recovered_at = len(equity_curve) - 1
        for i in range(max_dd_peak_idx + 1, len(equity_curve)):
            if equity_curve[i][1] >= anchor_peak:
                recovered_at = i
                break
        else:
            # Never recovered — duration is from peak to end
            recovered_at = len(equity_curve) - 1
        duration = recovered_at - max_dd_peak_idx
        return max_dd, duration

    return 0.0, 0


def _sharpe_annualised(equity_curve: list[tuple[int, float]]) -> float:
    """End-of-day equity → daily returns → annualised Sharpe."""
    if len(equity_curve) < 2:
        return 0.0

    # Bucket by UTC date, take the LAST equity sample of each day.
    eod: dict[str, float] = {}
    for ts_ms, eq in equity_curve:
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        eod[day] = eq

    sorted_days = sorted(eod.items())
    if len(sorted_days) < 2:
        return 0.0

    daily_returns: list[float] = []
    prev_eq = sorted_days[0][1]
    for _, eq in sorted_days[1:]:
        if prev_eq > 0:
            daily_returns.append((eq / prev_eq) - 1.0)
        prev_eq = eq

    if len(daily_returns) < 2:
        return 0.0
    mean = statistics.fmean(daily_returns)
    std = statistics.stdev(daily_returns)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(365.0)


def _calmar(equity_curve: list[tuple[int, float]], max_dd_decimal: float) -> float:
    if max_dd_decimal <= 0 or len(equity_curve) < 2:
        return 0.0
    first_ts, first_eq = equity_curve[0]
    last_ts, last_eq = equity_curve[-1]
    if first_eq <= 0:
        return 0.0
    seconds = (last_ts - first_ts) / 1000.0
    years = seconds / _SECONDS_PER_YEAR
    if years <= 0:
        return 0.0
    try:
        annual_return = (last_eq / first_eq) ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError):
        return 0.0
    return annual_return / max_dd_decimal


# ---------------------------------------------------------------------------
# Type alias for the config protocol
# ---------------------------------------------------------------------------


class BacktestConfigLike:
    """Structural type: any object with ``initial_balance`` and
    ``risk_per_trade`` attributes works as a config for ``calculate_metrics``.

    Importing the real ``BacktestConfig`` here would create a cycle
    (engine → metrics → engine). Duck typing keeps the dependency one-way.
    """

    initial_balance: float
    risk_per_trade: float
