"""Tests for AutoMiner — overconfident-disaster + missed-opportunity detection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backtesting.evals.auto_miner import AutoMiner


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _ms_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _trade(
    trade_id: str = "t1",
    pnl: float = 0.0,
    conviction: float = 0.5,
    forward_max_r: float | None = None,
    action: str | None = None,
    symbol: str = "BTC-USDC",
    timeframe: str = "1h",
    entry_timestamp: int | None = None,
) -> dict:
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "pnl": pnl,
        "conviction": conviction,
        "forward_max_r": forward_max_r,
        "action": action,
        "entry_timestamp": entry_timestamp if entry_timestamp is not None else _ms_now(),
        "ohlcv_at_entry": [
            {"timestamp": i * 1000, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}
            for i in range(50)
        ],
        "indicators_at_entry": {"rsi": 75.0},
        "regime_at_entry": "trending",
    }


@pytest.fixture
def output_dir(tmp_path) -> Path:
    return tmp_path / "pending_review"


# ---------------------------------------------------------------------------
# scan_recent_trades — sync + async fetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_with_sync_fetcher(output_dir):
    trades = [_trade("a"), _trade("b")]
    miner = AutoMiner(trade_fetcher=lambda: trades, output_dir=output_dir)
    out = await miner.scan_recent_trades(days=7)
    assert len(out) == 2
    assert {t["trade_id"] for t in out} == {"a", "b"}


@pytest.mark.asyncio
async def test_scan_with_async_fetcher(output_dir):
    async def fetcher():
        return [_trade("c")]

    miner = AutoMiner(trade_fetcher=fetcher, output_dir=output_dir)
    out = await miner.scan_recent_trades(days=7)
    assert [t["trade_id"] for t in out] == ["c"]


@pytest.mark.asyncio
async def test_scan_filters_by_age(output_dir):
    """Trades older than `days` are dropped."""
    old_ms = int(
        (datetime.now(tz=timezone.utc).timestamp() - 30 * 86400) * 1000
    )
    fresh = _trade("fresh", entry_timestamp=_ms_now())
    stale = _trade("stale", entry_timestamp=old_ms)
    miner = AutoMiner(trade_fetcher=lambda: [fresh, stale], output_dir=output_dir)
    out = await miner.scan_recent_trades(days=7)
    assert [t["trade_id"] for t in out] == ["fresh"]


# ---------------------------------------------------------------------------
# find_overconfident_disasters
# ---------------------------------------------------------------------------


def test_find_overconfident_disasters_flags_high_conviction_losses(output_dir):
    miner = AutoMiner(trade_fetcher=lambda: [], output_dir=output_dir)
    trades = [
        _trade("disaster1", conviction=0.90, pnl=-200),  # caught
        _trade("disaster2", conviction=0.95, pnl=-50),   # caught
        _trade("winner", conviction=0.92, pnl=300),      # high conviction, won → ignored
        _trade("low_conv_loss", conviction=0.5, pnl=-100),  # low conviction → ignored
        _trade("just_below_threshold", conviction=0.84, pnl=-100),  # below 0.85 → ignored
    ]
    flagged = miner.find_overconfident_disasters(trades)
    assert {t["trade_id"] for t in flagged} == {"disaster1", "disaster2"}


def test_overconfidence_threshold_is_inclusive(output_dir):
    miner = AutoMiner(
        trade_fetcher=lambda: [],
        output_dir=output_dir,
        overconfidence_threshold=0.85,
    )
    flagged = miner.find_overconfident_disasters(
        [_trade("exact", conviction=0.85, pnl=-1)]
    )
    assert len(flagged) == 1


def test_overconfident_skips_trades_with_missing_pnl(output_dir):
    miner = AutoMiner(trade_fetcher=lambda: [], output_dir=output_dir)
    bad = {"trade_id": "x", "conviction": 0.9}  # no pnl
    assert miner.find_overconfident_disasters([bad]) == []


# ---------------------------------------------------------------------------
# find_missed_opportunities
# ---------------------------------------------------------------------------


def test_find_missed_opportunities_low_conviction(output_dir):
    miner = AutoMiner(trade_fetcher=lambda: [], output_dir=output_dir)
    trades = [
        _trade("missed1", conviction=0.40, forward_max_r=4.0),  # caught
        _trade("missed2", conviction=0.30, forward_max_r=5.5),  # caught
        _trade("low_no_path", conviction=0.30, forward_max_r=None),  # no R info → skip
        _trade("low_small_R", conviction=0.30, forward_max_r=1.5),  # < 3R → skip
        _trade("high_conviction_traded", conviction=0.8, forward_max_r=6.0),  # not skipped → ignored
    ]
    flagged = miner.find_missed_opportunities(trades)
    assert {t["trade_id"] for t in flagged} == {"missed1", "missed2"}


def test_find_missed_opportunities_skip_action(output_dir):
    """An explicit SKIP action also qualifies, regardless of conviction."""
    miner = AutoMiner(trade_fetcher=lambda: [], output_dir=output_dir)
    skipped = _trade("skipped_winner", conviction=0.55, action="SKIP", forward_max_r=4.0)
    flagged = miner.find_missed_opportunities([skipped])
    assert len(flagged) == 1
    assert flagged[0]["trade_id"] == "skipped_winner"


def test_missed_threshold_inclusive(output_dir):
    miner = AutoMiner(trade_fetcher=lambda: [], output_dir=output_dir, missed_r_threshold=3.0)
    just_at = _trade("at", conviction=0.4, forward_max_r=3.0)
    flagged = miner.find_missed_opportunities([just_at])
    assert len(flagged) == 1


# ---------------------------------------------------------------------------
# mine() — end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_writes_drafts_to_disk(output_dir):
    trades = [
        _trade("disaster_btc", conviction=0.90, pnl=-300, symbol="BTC-USDC"),
        _trade(
            "missed_eth",
            conviction=0.35,
            forward_max_r=4.5,
            symbol="ETH-USDC",
            action="SKIP",
        ),
    ]
    miner = AutoMiner(trade_fetcher=lambda: trades, output_dir=output_dir)
    written = await miner.mine(days=7)

    assert len(written) == 2
    assert all(p.exists() for p in written)

    # Each draft is loadable as a JSON dict with the right shape
    for path in written:
        draft = json.loads(path.read_text())
        assert "id" in draft
        assert "metadata" in draft
        assert draft["metadata"]["auto_mined"] is True
        assert draft["metadata"]["mining_reason"] in (
            "overconfident_disaster",
            "missed_opportunity",
        )
        assert "expected_action" in draft["expected"]
        assert draft["inputs"]["ohlcv"]


@pytest.mark.asyncio
async def test_mine_filenames_are_deterministic(output_dir):
    """Same trade twice should produce the same scenario id (idempotent)."""
    trade = _trade("dup", conviction=0.95, pnl=-50)
    miner = AutoMiner(trade_fetcher=lambda: [trade, trade], output_dir=output_dir)
    written = await miner.mine(days=7)
    assert len(written) == 2
    # Both go to the same path (overwrite each other) — that's fine
    assert written[0] == written[1]


@pytest.mark.asyncio
async def test_mine_categorises_disasters_under_trap_setups(output_dir):
    miner = AutoMiner(
        trade_fetcher=lambda: [_trade("d", conviction=0.9, pnl=-100)],
        output_dir=output_dir,
    )
    written = await miner.mine(days=7)
    assert len(written) == 1
    draft = json.loads(written[0].read_text())
    assert draft["category"] == "trap_setups"


@pytest.mark.asyncio
async def test_mine_categorises_missed_under_clear_setups(output_dir):
    miner = AutoMiner(
        trade_fetcher=lambda: [
            _trade("m", conviction=0.3, forward_max_r=4.0, action="SKIP")
        ],
        output_dir=output_dir,
    )
    written = await miner.mine(days=7)
    draft = json.loads(written[0].read_text())
    assert draft["category"] == "clear_setups"


@pytest.mark.asyncio
async def test_mine_returns_empty_when_no_matches(output_dir):
    """A clean run (no disasters, no missed opportunities) writes nothing."""
    trades = [_trade("normal", conviction=0.6, pnl=50, forward_max_r=1.5)]
    miner = AutoMiner(trade_fetcher=lambda: trades, output_dir=output_dir)
    written = await miner.mine(days=7)
    assert written == []
    # Output dir is created regardless (mkdir parents=True, exist_ok=True)
    assert output_dir.exists()
