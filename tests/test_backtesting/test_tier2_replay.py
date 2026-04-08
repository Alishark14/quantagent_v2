"""Unit tests for Tier2ReplayEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from backtesting.data_loader import ParquetDataLoader
from backtesting.forward_path import ForwardPathLoader
from backtesting.tier2_replay import (
    ReplayResult,
    SweepResult,
    Tier2ReplayEngine,
)


EXCHANGE = "hyperliquid"
MIN_MS = 60 * 1000


# ---------------------------------------------------------------------------
# Fixtures: synthetic forward paths + recorded trades
# ---------------------------------------------------------------------------


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _path_df(rows: list[tuple[int, float, float, float, float]]) -> pl.DataFrame:
    """Build a Polars DataFrame from (ts, open, high, low, close) tuples."""
    return pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [10.0] * len(rows),
        },
        schema={
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _make_trade(
    trade_id: str = "t1",
    direction: str = "LONG",
    entry_price: float = 100.0,
    sl_price: float = 95.0,
    tp1_price: float = 105.0,
    tp2_price: float = 110.0,
    size: float = 1.0,
    atr: float = 5.0,
    conviction: float = 0.7,
    pnl: float = 10.0,
    exit_price: float = 110.0,
    exit_reason: str = "tp2_hit",
    timeframe: str = "1h",
    entry_ts: int = 1_700_000_000_000,
) -> dict:
    return {
        "trade_id": trade_id,
        "symbol": "BTC-USDC",
        "timeframe": timeframe,
        "direction": direction,
        "entry_timestamp": entry_ts,
        "entry_price": entry_price,
        "size": size,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "atr_at_entry": atr,
        "conviction": conviction,
        "exit_price": exit_price,
        "exit_timestamp": entry_ts + 60 * MIN_MS,
        "exit_reason": exit_reason,
        "pnl": pnl,
    }


@pytest.fixture
def stub_loader():
    """ForwardPathLoader stub that returns whatever DataFrame the test sets."""

    class _StubLoader:
        def __init__(self) -> None:
            self.path: pl.DataFrame | None = None

        def load(
            self,
            symbol: str,
            entry_timestamp: int,
            duration_candles: int = 60,
            resolution: str = "1m",
        ) -> pl.DataFrame:
            if self.path is None:
                raise FileNotFoundError("test stub: no path set")
            return self.path

        @staticmethod
        def recommended_resolution(tf: str) -> str:
            return "1m"

    return _StubLoader()


@pytest.fixture
def engine(stub_loader) -> Tier2ReplayEngine:
    return Tier2ReplayEngine(forward_loader=stub_loader)


# ---------------------------------------------------------------------------
# Single-trade replay
# ---------------------------------------------------------------------------


def test_replay_no_modifications_matches_recorded_outcome(engine):
    """With empty modified_params and a forward path that hits TP2 cleanly,
    the counterfactual should hit TP2 too."""
    trade = _make_trade()
    # Path that walks straight up to 115 — TP2 (110) is hit cleanly
    rows = [
        (1_700_000_000_000 + i * MIN_MS, 100 + i, 100.5 + i, 99.5 + i, 100 + i)
        for i in range(20)
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(trade, modified_params={}, forward_path=fp)

    assert result.counterfactual_outcome["exit_reason"] == "tp2_hit"
    assert result.counterfactual_outcome["exit_price"] == 110.0
    # PnL = TP1 portion (5 * 0.5) + TP2 portion (10 * 0.5) = 2.5 + 5.0 = 7.5
    assert result.counterfactual_outcome["pnl"] == pytest.approx(7.5)


def test_replay_with_tighter_sl_stops_out_originally_winning_trade(engine):
    """Original trade won big. With a tighter SL, the same path stops out
    on the early dip and yields a loss."""
    trade = _make_trade(pnl=10.0, exit_reason="tp2_hit")
    # Path: dips to 98 first (would clip a tight SL at 99) then rallies
    rows = [
        (1_700_000_000_000, 100, 100, 98, 99),  # bar 0: low touches 98
        (1_700_000_000_000 + MIN_MS, 99, 115, 99, 115),
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"sl_price": 99.0},
        forward_path=fp,
    )

    assert result.counterfactual_outcome["exit_reason"] == "stop_hit"
    assert result.counterfactual_outcome["exit_price"] == 99.0
    # Loss on full size (no TP1 hit yet because SL fires first on bar 0)
    assert result.counterfactual_outcome["pnl"] == pytest.approx(-1.0)
    # delta_pnl = cf - original = -1 - 10 = -11
    assert result.delta_pnl == pytest.approx(-11.0)
    assert result.skipped is False


def test_replay_with_wider_tp_holds_to_tp2(engine):
    """Original trade exited at TP1. With a wider TP1 and a much wider TP2,
    the path now rides further and posts more PnL."""
    # The "original" trade in the recording closed at TP1 — represented as a
    # smaller PnL in the input dict. Replay just compares counterfactuals.
    trade = _make_trade(pnl=2.5, exit_reason="tp1_hit", tp1_price=105.0, tp2_price=110.0)
    # Path that rallies to 120 cleanly
    rows = [
        (1_700_000_000_000 + i * MIN_MS, 100 + i, 100.5 + i, 99.5 + i, 100 + i)
        for i in range(25)
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"tp1_price": 108.0, "tp2_price": 118.0},
        forward_path=fp,
    )
    assert result.counterfactual_outcome["exit_reason"] == "tp2_hit"
    assert result.counterfactual_outcome["exit_price"] == 118.0
    # 0.5 * (108 - 100) + 0.5 * (118 - 100) = 4 + 9 = 13
    assert result.counterfactual_outcome["pnl"] == pytest.approx(13.0)
    assert result.delta_pnl == pytest.approx(13.0 - 2.5)


def test_replay_short_with_tighter_sl_stops_on_high_breach(engine):
    trade = _make_trade(
        direction="SHORT",
        entry_price=100.0,
        sl_price=105.0,
        tp1_price=95.0,
        tp2_price=90.0,
        pnl=-2.0,
        exit_reason="stop_hit",
    )
    # Spike up to 102, then collapse — tighter SL at 101.5 fires immediately
    rows = [
        (1_700_000_000_000, 100, 102, 100, 101),
        (1_700_000_000_000 + MIN_MS, 101, 101, 88, 89),
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"sl_price": 101.5},
        forward_path=fp,
    )
    assert result.counterfactual_outcome["exit_reason"] == "stop_hit"
    assert result.counterfactual_outcome["exit_price"] == 101.5
    # Short loss: (100 - 101.5) * 1.0 = -1.5
    assert result.counterfactual_outcome["pnl"] == pytest.approx(-1.5)


def test_replay_atr_multiplier_re_derives_sl(engine):
    """Pass atr_multiplier instead of sl_price — engine recomputes SL from ATR."""
    trade = _make_trade(entry_price=100.0, atr=5.0, sl_price=92.5)
    # Path that dips to 95 — original SL at 92.5 holds, new SL at entry - 5*0.8 = 96 fires
    rows = [
        (1_700_000_000_000, 100, 100, 95, 96),
        (1_700_000_000_000 + MIN_MS, 96, 115, 96, 115),
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"atr_multiplier": 0.8},
        forward_path=fp,
    )
    # SL would be 100 - 5 * 0.8 = 96
    assert result.counterfactual_outcome["exit_reason"] == "stop_hit"
    assert result.counterfactual_outcome["exit_price"] == pytest.approx(96.0)


def test_replay_breakeven_after_tp1(engine):
    """After TP1 fires the SL snaps to entry. A subsequent dip below entry
    closes the runner at break-even."""
    trade = _make_trade(entry_price=100.0, sl_price=95.0, tp1_price=105.0, tp2_price=120.0)
    rows = [
        (1_700_000_000_000, 100, 106, 99, 105),  # bar 0: hits TP1, then dips
        # bar 1: drops to 99 — break-even SL at 100 fires on the runner
        (1_700_000_000_000 + MIN_MS, 105, 105, 99, 100),
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"breakeven_after_tp1": True},
        forward_path=fp,
    )
    # PnL = TP1 portion (5 * 0.5) + break-even portion (0 * 0.5) = 2.5
    assert result.counterfactual_outcome["pnl"] == pytest.approx(2.5)
    assert result.counterfactual_outcome["exit_reason"] == "stop_hit"


def test_replay_trailing_stop_locks_in_profit(engine):
    """A trailing stop should follow the price up and exit on the pullback."""
    trade = _make_trade(
        entry_price=100.0, sl_price=95.0, tp1_price=200.0, tp2_price=300.0, atr=2.0
    )
    # Walk up to 110 then drop to 105. Trail = 2 * 1.0 = 2.
    # After bar at close=110, trail SL = 108. Bar at low=105 hits 108 → stop.
    rows = [
        (1_700_000_000_000, 100, 102, 99, 102),
        (1_700_000_000_000 + MIN_MS, 102, 110, 102, 110),
        (1_700_000_000_000 + 2 * MIN_MS, 110, 110, 105, 105),
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"trailing_atr_mult": 1.0},
        forward_path=fp,
    )
    assert result.counterfactual_outcome["exit_reason"] == "stop_hit"
    # Trail SL after bar 1's close=110 is 108; bar 2 low=105 → fills at 108
    assert result.counterfactual_outcome["exit_price"] == pytest.approx(108.0)


def test_replay_still_open_at_end_of_forward_path(engine):
    """Edge case: forward path runs out before SL or TP fires."""
    trade = _make_trade(
        entry_price=100.0, sl_price=80.0, tp1_price=200.0, tp2_price=300.0, pnl=5.0
    )
    rows = [
        (1_700_000_000_000 + i * MIN_MS, 100 + i, 100.5 + i, 99 + i, 100 + i)
        for i in range(5)
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(trade, modified_params={}, forward_path=fp)
    assert result.counterfactual_outcome["exit_reason"] == "still_open"
    # Last close = 104; mark-to-market PnL = (104 - 100) * 1.0 = 4
    assert result.counterfactual_outcome["pnl"] == pytest.approx(4.0)
    assert result.counterfactual_outcome["exit_index"] == 4


# ---------------------------------------------------------------------------
# Conviction filter
# ---------------------------------------------------------------------------


def test_replay_conviction_threshold_filters_low_conviction_trade(engine):
    trade = _make_trade(conviction=0.55, pnl=10.0)
    fp = _path_df([(1_700_000_000_000, 100, 110, 100, 110)])
    result = engine.replay_trade(
        trade,
        modified_params={"conviction_threshold": 0.7},
        forward_path=fp,
    )
    assert result.skipped is True
    assert result.counterfactual_outcome["pnl"] == 0.0
    assert result.counterfactual_outcome["exit_reason"] == "SKIPPED_BY_CONVICTION_FILTER"
    # delta_pnl = 0 - 10 = -10 (we *avoided* a winning trade)
    assert result.delta_pnl == pytest.approx(-10.0)


def test_replay_conviction_threshold_passes_high_conviction(engine):
    trade = _make_trade(conviction=0.85)
    rows = [
        (1_700_000_000_000 + i * MIN_MS, 100 + i, 100.5 + i, 99.5 + i, 100 + i)
        for i in range(20)
    ]
    fp = _path_df(rows)
    result = engine.replay_trade(
        trade,
        modified_params={"conviction_threshold": 0.7},
        forward_path=fp,
    )
    assert result.skipped is False
    assert result.counterfactual_outcome["exit_reason"] == "tp2_hit"


# ---------------------------------------------------------------------------
# Batch + sweep
# ---------------------------------------------------------------------------


def test_replay_batch_iterates_all_trades(engine, stub_loader):
    # Two trades, same forward path stub for simplicity
    trades = [
        _make_trade(trade_id="t1", conviction=0.8, pnl=5.0),
        _make_trade(trade_id="t2", conviction=0.4, pnl=-3.0),
    ]
    rows = [
        (1_700_000_000_000 + i * MIN_MS, 100 + i, 100.5 + i, 99.5 + i, 100 + i)
        for i in range(20)
    ]
    stub_loader.path = _path_df(rows)

    results = engine.replay_batch(trades, modified_params={})
    assert len(results) == 2
    assert all(isinstance(r, ReplayResult) for r in results)
    assert {r.trade_id for r in results} == {"t1", "t2"}


def test_replay_batch_skips_missing_forward_paths(engine, stub_loader):
    """Trades whose forward path raises FileNotFoundError are dropped, not crashed."""
    trades = [_make_trade(trade_id="t1"), _make_trade(trade_id="t2")]
    stub_loader.path = None  # → loader.load() raises FileNotFoundError
    results = engine.replay_batch(trades, modified_params={})
    assert results == []


def test_parameter_sweep_produces_one_row_per_value(engine, stub_loader):
    trades = [
        _make_trade(trade_id="t1", conviction=0.85),
        _make_trade(trade_id="t2", conviction=0.55),
        _make_trade(trade_id="t3", conviction=0.40),
    ]
    rows = [
        (1_700_000_000_000 + i * MIN_MS, 100 + i, 100.5 + i, 99.5 + i, 100 + i)
        for i in range(20)
    ]
    stub_loader.path = _path_df(rows)

    sweep = engine.parameter_sweep(
        trades,
        param_name="conviction_threshold",
        param_values=[0.0, 0.5, 0.7, 0.9],
    )
    assert isinstance(sweep, SweepResult)
    assert sweep.param_name == "conviction_threshold"
    assert len(sweep.rows) == 4

    # Threshold 0.0 → none skipped
    assert sweep.rows[0].num_skipped == 0
    assert sweep.rows[0].num_trades == 3
    # Threshold 0.5 → t3 skipped
    assert sweep.rows[1].num_skipped == 1
    assert sweep.rows[1].num_trades == 2
    # Threshold 0.7 → t2 + t3 skipped
    assert sweep.rows[2].num_skipped == 2
    assert sweep.rows[2].num_trades == 1
    # Threshold 0.9 → all skipped
    assert sweep.rows[3].num_skipped == 3
    assert sweep.rows[3].num_trades == 0


def test_parameter_sweep_atr_multiplier_changes_outcomes(engine, stub_loader):
    """A sweep over ATR multipliers should produce different total_pnl rows."""
    trades = [_make_trade(trade_id="t1", atr=5.0)]
    # Path that dips to 96, then rallies to 115. Tight SL kills it; wide SL rides.
    rows = [
        (1_700_000_000_000, 100, 100, 96, 97),
        (1_700_000_000_000 + MIN_MS, 97, 115, 97, 115),
    ]
    stub_loader.path = _path_df(rows)

    sweep = engine.parameter_sweep(
        trades,
        param_name="atr_multiplier",
        param_values=[0.5, 1.5],  # SL at 97.5 vs 92.5
    )
    # Tight (0.5): SL at 97.5 → low 96 hits → loss
    # Wide (1.5): SL at 92.5 → low 96 holds → TP2 at 110 fires on bar 1 → win
    assert sweep.rows[0].total_pnl < 0
    assert sweep.rows[1].total_pnl > 0


def test_parameter_sweep_to_dict_serialises(engine, stub_loader):
    trades = [_make_trade(trade_id="t1")]
    stub_loader.path = _path_df([(1_700_000_000_000, 100, 110, 100, 110)])
    sweep = engine.parameter_sweep(trades, "conviction_threshold", [0.5])
    d = sweep.to_dict()
    import json
    json.dumps(d)  # must not raise
    assert d["param_name"] == "conviction_threshold"
    assert len(d["rows"]) == 1


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def test_replay_result_to_dict(engine):
    trade = _make_trade()
    fp = _path_df([(1_700_000_000_000, 100, 110, 100, 110)])
    result = engine.replay_trade(trade, modified_params={}, forward_path=fp)
    d = result.to_dict()
    for key in ("trade_id", "original_outcome", "counterfactual_outcome",
                "delta_pnl", "delta_r", "skipped"):
        assert key in d
    import json
    json.dumps(d)
