"""Unit tests for backtesting.metrics — calculate_metrics + helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from backtesting.metrics import (
    BacktestMetrics,
    _max_drawdown,
    _sharpe_annualised,
    _streaks,
    calculate_metrics,
)


# ---------------------------------------------------------------------------
# Minimal config stub (the metrics module only reads two attributes)
# ---------------------------------------------------------------------------


@dataclass
class _Cfg:
    initial_balance: float = 10_000.0
    risk_per_trade: float = 0.01
    mode: str = "mechanical"


DAY_MS = 24 * 3600 * 1000


def _make_trade(pnl: float, fee: float = 0.5, ts: int = 0) -> dict:
    return {
        "timestamp": ts,
        "symbol": "BTC-USDC",
        "side": "long",
        "entry_price": 100.0,
        "exit_price": 100.0 + pnl,
        "size": 1.0,
        "fee": fee,
        "slippage": 0.0,
        "pnl": pnl,
        "reason": "tp2_hit" if pnl > 0 else "stop_hit",
    }


def _equity_from_pnls(initial: float, pnls: list[float]) -> list[tuple[int, float]]:
    """Build a daily equity curve where each trade settles one day apart."""
    out: list[tuple[int, float]] = []
    eq = initial
    for i, p in enumerate(pnls):
        eq += p
        out.append((i * DAY_MS, eq))
    return out


# ---------------------------------------------------------------------------
# Edge cases: zero / single / all-winners
# ---------------------------------------------------------------------------


def test_metrics_zero_trades():
    m = calculate_metrics(trade_history=[], equity_curve=[], config=_Cfg())
    assert isinstance(m, BacktestMetrics)
    assert m.total_trades == 0
    assert m.winning_trades == 0
    assert m.losing_trades == 0
    assert m.win_rate == 0.0
    assert m.loss_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.sharpe_ratio == 0.0
    assert m.calmar_ratio == 0.0
    assert m.max_drawdown_pct == 0.0
    assert m.max_drawdown_duration == 0
    assert m.avg_r_multiple == 0.0
    assert m.avg_trade_duration_hours == 0.0
    assert m.longest_win_streak == 0
    assert m.longest_loss_streak == 0
    assert m.skip_rate == 0.0
    assert m.total_pnl == 0.0
    assert m.return_pct == 0.0
    # Final balance is the initial balance when there's no equity curve
    assert m.final_balance == 10_000.0
    assert m.initial_balance == 10_000.0


def test_metrics_single_winning_trade():
    pnls = [50.0]
    m = calculate_metrics(
        trade_history=[_make_trade(p) for p in pnls],
        equity_curve=_equity_from_pnls(10_000.0, pnls),
        config=_Cfg(),
    )
    assert m.total_trades == 1
    assert m.winning_trades == 1
    assert m.losing_trades == 0
    assert m.win_rate == 1.0
    assert m.profit_factor == 999.9  # capped (no losses)
    assert m.longest_win_streak == 1
    assert m.longest_loss_streak == 0
    # avg_r = 50 / (10000 * 0.01) = 0.5 (fallback risk)
    assert m.avg_r_multiple == pytest.approx(0.5)
    assert m.total_pnl == pytest.approx(50.0)
    assert m.final_balance == pytest.approx(10_050.0)
    assert m.return_pct == pytest.approx(0.5)


def test_metrics_all_winners_caps_profit_factor():
    pnls = [10.0, 20.0, 5.0, 15.0]
    m = calculate_metrics(
        trade_history=[_make_trade(p) for p in pnls],
        equity_curve=_equity_from_pnls(10_000.0, pnls),
        config=_Cfg(),
    )
    assert m.profit_factor == 999.9
    assert m.winning_trades == 4
    assert m.losing_trades == 0
    assert m.win_rate == 1.0
    assert m.longest_win_streak == 4


def test_metrics_all_losers():
    pnls = [-10.0, -20.0, -5.0]
    m = calculate_metrics(
        trade_history=[_make_trade(p) for p in pnls],
        equity_curve=_equity_from_pnls(10_000.0, pnls),
        config=_Cfg(),
    )
    assert m.winning_trades == 0
    assert m.losing_trades == 3
    assert m.profit_factor == 0.0  # no profit, only loss
    assert m.longest_loss_streak == 3
    assert m.longest_win_streak == 0


# ---------------------------------------------------------------------------
# Known 10-trade scenario with manually-derivable values
# ---------------------------------------------------------------------------


KNOWN_PNLS = [10.0, -5.0, 20.0, 15.0, -10.0, 5.0, -8.0, 12.0, -3.0, 25.0]


def test_metrics_known_scenario():
    cfg = _Cfg(initial_balance=10_000.0, risk_per_trade=0.01)
    history = [_make_trade(p, fee=0.5) for p in KNOWN_PNLS]
    equity = _equity_from_pnls(10_000.0, KNOWN_PNLS)
    m = calculate_metrics(history, equity, cfg)

    # ----- Manually-derived expectations -----
    assert m.total_trades == 10
    assert m.winning_trades == 6
    assert m.losing_trades == 4
    assert m.win_rate == 0.6
    assert m.loss_rate == 0.4

    # gross_profit = 10+20+15+5+12+25 = 87
    # gross_loss = 5+10+8+3 = 26
    assert m.profit_factor == pytest.approx(87 / 26, rel=1e-4)

    # Streaks: W L W W L W L W L W → max_win=2 (idx 2-3), max_loss=1
    assert m.longest_win_streak == 2
    assert m.longest_loss_streak == 1

    # avg_r with fallback risk = 100: mean(pnls)/100 = 6.1/100 = 0.061
    assert m.avg_r_multiple == pytest.approx(0.061, abs=1e-4)

    # total_pnl = 61, total_fees = 5.0
    assert m.total_pnl == pytest.approx(61.0)
    assert m.total_fees == pytest.approx(5.0)
    assert m.cost_adjusted_pnl == pytest.approx(61.0)  # sim PnL is already net

    # final_balance = 10000 + 61 = 10061
    assert m.final_balance == pytest.approx(10_061.0)
    assert m.return_pct == pytest.approx(0.61)

    # Max drawdown: peak after trade 4 (idx 3) = 10040, trough at trade 7 (idx 6) = 10027
    # → DD = (10040 - 10027) / 10040 ≈ 0.001295
    expected_dd_pct = (10_040.0 - 10_027.0) / 10_040.0 * 100
    assert m.max_drawdown_pct == pytest.approx(expected_dd_pct, abs=1e-4)


def test_metrics_avg_r_uses_sl_when_present():
    """When trade dicts include sl_price + entry_price + size, R is per-trade."""
    cfg = _Cfg(initial_balance=10_000.0, risk_per_trade=0.01)
    history = [
        # Each trade risks $5 (entry 100, sl 95, size 1), pnl = +10 → R = +2
        {**_make_trade(10.0), "entry_price": 100.0, "sl_price": 95.0, "size": 1.0},
        # pnl = -5 → R = -1
        {**_make_trade(-5.0), "entry_price": 100.0, "sl_price": 95.0, "size": 1.0},
    ]
    equity = _equity_from_pnls(10_000.0, [10.0, -5.0])
    m = calculate_metrics(history, equity, cfg)
    assert m.avg_r_multiple == pytest.approx(0.5)  # mean(2, -1)


def test_metrics_avg_trade_duration_uses_entry_timestamp():
    cfg = _Cfg()
    history = [
        {**_make_trade(10.0, ts=2 * 3_600_000), "entry_timestamp": 0},  # 2h
        {**_make_trade(-5.0, ts=6 * 3_600_000), "entry_timestamp": 2 * 3_600_000},  # 4h
    ]
    equity = _equity_from_pnls(10_000.0, [10.0, -5.0])
    m = calculate_metrics(history, equity, cfg)
    assert m.avg_trade_duration_hours == pytest.approx(3.0)  # mean(2h, 4h)


def test_metrics_avg_trade_duration_zero_when_missing_timestamps():
    cfg = _Cfg()
    history = [_make_trade(10.0)]  # no entry_timestamp
    equity = _equity_from_pnls(10_000.0, [10.0])
    m = calculate_metrics(history, equity, cfg)
    assert m.avg_trade_duration_hours == 0.0


# ---------------------------------------------------------------------------
# Skip rate
# ---------------------------------------------------------------------------


def test_skip_rate_with_setups():
    cfg = _Cfg()
    m = calculate_metrics(
        trade_history=[_make_trade(10.0)],
        equity_curve=_equity_from_pnls(10_000.0, [10.0]),
        config=cfg,
        setups_detected=10,
        setups_taken=4,
    )
    assert m.skip_rate == pytest.approx(0.6)  # (10 - 4) / 10


def test_skip_rate_zero_when_no_setups_detected():
    m = calculate_metrics(
        trade_history=[],
        equity_curve=[],
        config=_Cfg(),
        setups_detected=0,
        setups_taken=0,
    )
    assert m.skip_rate == 0.0


def test_skip_rate_full_when_all_skipped():
    m = calculate_metrics(
        trade_history=[],
        equity_curve=[],
        config=_Cfg(),
        setups_detected=5,
        setups_taken=0,
    )
    assert m.skip_rate == 1.0


# ---------------------------------------------------------------------------
# Helper functions tested in isolation
# ---------------------------------------------------------------------------


def test_max_drawdown_simple():
    """Equity goes up to 110, down to 90 (worst), back up to 105.
    Expected DD = (110 - 90) / 110 ≈ 0.1818."""
    curve = [
        (0, 100.0),
        (1, 105.0),
        (2, 110.0),  # peak
        (3, 100.0),
        (4, 90.0),   # trough
        (5, 95.0),
        (6, 105.0),  # never recovers above 110
    ]
    max_dd, duration = _max_drawdown(curve)
    assert max_dd == pytest.approx((110 - 90) / 110)
    # Never recovers above 110 → duration = end_idx - peak_idx = 6 - 2 = 4
    assert duration == 4


def test_max_drawdown_with_recovery():
    curve = [
        (0, 100.0),
        (1, 110.0),  # peak
        (2, 90.0),   # trough
        (3, 110.0),  # recovers exactly
        (4, 120.0),
    ]
    max_dd, duration = _max_drawdown(curve)
    assert max_dd == pytest.approx((110 - 90) / 110)
    # Peak at idx 1, recovery at idx 3 → duration = 2
    assert duration == 2


def test_max_drawdown_no_drawdown_when_only_rising():
    curve = [(i, 100.0 + i) for i in range(10)]
    max_dd, duration = _max_drawdown(curve)
    assert max_dd == 0.0
    assert duration == 0


def test_max_drawdown_empty():
    assert _max_drawdown([]) == (0.0, 0)


def test_sharpe_zero_for_constant_equity():
    """Constant equity means zero stdev → Sharpe = 0 by convention."""
    curve = [(i * DAY_MS, 10_000.0) for i in range(5)]
    assert _sharpe_annualised(curve) == 0.0


def test_sharpe_single_day_returns_zero():
    """Need at least 2 daily samples for a return, 2 returns for stdev."""
    curve = [(0, 10_000.0), (DAY_MS // 4, 10_010.0)]
    # Both samples on day 0 → only 1 EOD bucket → 0 daily returns → 0
    assert _sharpe_annualised(curve) == 0.0


def test_sharpe_positive_for_steady_uptrend():
    """5 days of +1% daily returns → high Sharpe (zero variance!).
    With *zero* variance Sharpe is 0 by convention. Add a tiny wiggle
    so stdev > 0."""
    eqs = [10_000.0]
    for r in [0.01, 0.012, 0.009, 0.011, 0.010]:
        eqs.append(eqs[-1] * (1 + r))
    curve = [(i * DAY_MS, e) for i, e in enumerate(eqs)]
    sharpe = _sharpe_annualised(curve)
    assert sharpe > 0  # consistent positive returns, low variance


def test_sharpe_manual_calculation():
    """End-of-day equity series with 4 days of returns.

    Returns: r1, r2, r3, r4 from 5 EOD samples.
    Sharpe = mean(r) / stdev(r) * sqrt(365).
    """
    eqs = [100.0, 102.0, 101.0, 104.0, 103.0]  # 4 returns
    curve = [(i * DAY_MS, e) for i, e in enumerate(eqs)]

    # Manual computation
    rs = [102 / 100 - 1, 101 / 102 - 1, 104 / 101 - 1, 103 / 104 - 1]
    mean = sum(rs) / len(rs)
    var = sum((x - mean) ** 2 for x in rs) / (len(rs) - 1)
    stdev = math.sqrt(var)
    expected = (mean / stdev) * math.sqrt(365)

    assert _sharpe_annualised(curve) == pytest.approx(expected, rel=1e-6)


def test_streaks_basic():
    history = [
        {"pnl": 1.0}, {"pnl": 2.0}, {"pnl": -1.0},
        {"pnl": 3.0}, {"pnl": -2.0}, {"pnl": -3.0},
        {"pnl": 4.0},
    ]
    longest_win, longest_loss = _streaks(history)
    assert longest_win == 2  # first two
    assert longest_loss == 2  # idx 4-5


def test_streaks_zero_pnl_breaks_streaks():
    history = [
        {"pnl": 1.0}, {"pnl": 0.0}, {"pnl": 1.0},
    ]
    longest_win, _ = _streaks(history)
    assert longest_win == 1  # the zero broke the streak


def test_streaks_empty():
    assert _streaks([]) == (0, 0)
