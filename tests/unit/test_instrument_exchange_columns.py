"""Tests for instrument_type, exchange, leverage, margin_type columns.

Verifies that trade records, cycle records, and bot records include
the new multi-exchange columns populated from the pipeline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.config import TradingConfig
from engine.events import InProcessBus
from engine.pipeline import AnalysisPipeline
from engine.types import OrderResult, TradeAction


def _make_pipeline(symbol: str = "BTC-USDC") -> AnalysisPipeline:
    config = TradingConfig(symbol=symbol, timeframe="1h")
    bus = InProcessBus()
    trade_repo = AsyncMock()
    trade_repo.save_trade = AsyncMock()

    # Mock the adapter to return a name
    mock_adapter = MagicMock()
    mock_adapter.name.return_value = "hyperliquid"
    mock_ohlcv = AsyncMock()
    mock_ohlcv._adapter = mock_adapter

    pipeline = AnalysisPipeline(
        ohlcv_fetcher=mock_ohlcv,
        flow_agent=AsyncMock(),
        signal_registry=AsyncMock(),
        conviction_agent=AsyncMock(),
        decision_agent=AsyncMock(),
        event_bus=bus,
        cycle_memory=AsyncMock(),
        reflection_rules=AsyncMock(),
        cross_bot=AsyncMock(),
        regime_history=AsyncMock(),
        cycle_repo=AsyncMock(),
        config=config,
        bot_id="test-bot",
        user_id="test-user",
        is_shadow=True,
        shadow_fixed_size_usd=500.0,
        trade_repo=trade_repo,
    )
    return pipeline


@pytest.mark.asyncio
async def test_trade_record_includes_instrument_and_exchange():
    """Trade row must have instrument_type, exchange, leverage, margin_type."""
    pipeline = _make_pipeline()
    pipeline._last_market_data = None
    pipeline._last_regime = "RANGING"

    action = TradeAction(
        action="LONG", conviction_score=0.65, position_size=500.0,
        sl_price=95.0, tp1_price=110.0, tp2_price=115.0,
        rr_ratio=1.5, atr_multiplier=1.5, reasoning="test",
        raw_output="", risk_weight=1.0,
    )
    order = OrderResult(success=True, order_id="o1", fill_price=100.0,
                        fill_size=5.0, error=None)

    await pipeline.record_trade_open(action, order)

    row = pipeline._trade_repo.save_trade.call_args[0][0]
    assert row["instrument_type"] == "perpetual"
    assert row["exchange"] == "hyperliquid"
    assert row["leverage"] is None
    assert row["margin_type"] == "cross"


@pytest.mark.asyncio
async def test_trade_exchange_reads_from_adapter():
    """Exchange name comes from adapter.name(), not hardcoded."""
    pipeline = _make_pipeline()
    pipeline._last_market_data = None
    pipeline._last_regime = "RANGING"

    # Change the adapter name
    pipeline._ohlcv._adapter.name.return_value = "binance"

    action = TradeAction(
        action="SHORT", conviction_score=0.70, position_size=500.0,
        sl_price=105.0, tp1_price=90.0, tp2_price=85.0,
        rr_ratio=1.5, atr_multiplier=1.5, reasoning="test",
        raw_output="", risk_weight=1.15,
    )
    order = OrderResult(success=True, order_id="o1", fill_price=100.0,
                        fill_size=5.0, error=None)

    await pipeline.record_trade_open(action, order)

    row = pipeline._trade_repo.save_trade.call_args[0][0]
    assert row["exchange"] == "binance"


@pytest.mark.asyncio
async def test_defaults_for_existing_rows():
    """SQLite DDL defaults backfill existing rows correctly.

    This test verifies the default values match what the migration
    sets, so rows created before the migration have sensible values.
    """
    import aiosqlite
    from storage.repositories.sqlite import SQLiteRepositories

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        # Insert a minimal trade
        await repos.trades.save_trade({
            "id": "t1", "user_id": "u1", "bot_id": "b1",
            "symbol": "BTC-USDC", "timeframe": "1h", "direction": "LONG",
            "entry_price": 100.0, "status": "open",
        })

        # Read it back — defaults should be applied
        row = await repos.trades.get_trade("t1")
        assert row is not None
        # instrument_type defaults to 'perpetual' at the DDL level
        assert row.get("instrument_type") == "perpetual"
        assert row.get("exchange") == "hyperliquid"
        assert row.get("margin_type") == "cross"
        assert row.get("leverage") is None


@pytest.mark.asyncio
async def test_bot_record_includes_instrument_type():
    """Bot save_bot stores instrument_type."""
    import aiosqlite
    from storage.repositories.sqlite import SQLiteRepositories

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        await repos.bots.save_bot({
            "id": "b1", "user_id": "u1", "symbol": "BTC-USDC",
            "timeframe": "1h", "exchange": "hyperliquid",
            "instrument_type": "perpetual",
            "created_at": "2026-04-12T00:00:00Z",
        })

        bot = await repos.bots.get_bot("b1")
        assert bot is not None
        assert bot.get("instrument_type") == "perpetual"
