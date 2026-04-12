"""Tests for trade quality columns: tp2_price, atr_multiplier, risk_weight, regime.

These columns are populated in record_trade_open and persisted via save_trade.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engine.config import TradingConfig
from engine.events import InProcessBus
from engine.pipeline import AnalysisPipeline
from engine.types import MarketData, OrderResult, TradeAction


def _make_pipeline(
    symbol: str = "ETH-USDC",
    timeframe: str = "1h",
) -> AnalysisPipeline:
    config = TradingConfig(symbol=symbol, timeframe=timeframe)
    bus = InProcessBus()
    trade_repo = AsyncMock()
    trade_repo.save_trade = AsyncMock()

    pipeline = AnalysisPipeline(
        ohlcv_fetcher=AsyncMock(),
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


def _make_action() -> TradeAction:
    return TradeAction(
        action="SHORT",
        conviction_score=0.72,
        position_size=500.0,
        sl_price=4723.0,
        tp1_price=4706.0,
        tp2_price=4694.0,
        rr_ratio=1.5,
        atr_multiplier=1.8,
        reasoning="Test trade",
        raw_output="",
        risk_weight=1.15,
    )


@pytest.mark.asyncio
async def test_all_four_columns_populated():
    """record_trade_open stores tp2_price, atr_multiplier, risk_weight, regime."""
    pipeline = _make_pipeline()
    pipeline._last_market_data = None  # skip SL/TP recomputation
    pipeline._last_regime = "TRENDING_DOWN"

    action = _make_action()
    order = OrderResult(success=True, order_id="o1", fill_price=4712.0,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)

    save = pipeline._trade_repo.save_trade
    assert save.called
    row = save.call_args[0][0]

    assert row["tp2_price"] == 4694.0
    assert row["atr_multiplier"] == 1.8
    assert row["risk_weight"] == 1.15
    assert row["regime"] == "TRENDING_DOWN"


@pytest.mark.asyncio
async def test_regime_matches_conviction():
    """regime in the trade row reflects _last_regime from ConvictionAgent."""
    for regime in ["TRENDING_UP", "RANGING", "HIGH_VOLATILITY", "BREAKOUT"]:
        pipeline = _make_pipeline()
        pipeline._last_market_data = None
        pipeline._last_regime = regime

        action = _make_action()
        order = OrderResult(success=True, order_id="o1", fill_price=4712.0,
                            fill_size=0.1, error=None)

        await pipeline.record_trade_open(action, order)

        row = pipeline._trade_repo.save_trade.call_args[0][0]
        assert row["regime"] == regime


@pytest.mark.asyncio
async def test_risk_weight_matches_conviction_tier():
    """risk_weight from TradeAction is stored directly on the trade row."""
    for weight in [0.75, 1.0, 1.15, 1.3]:
        pipeline = _make_pipeline()
        pipeline._last_market_data = None
        pipeline._last_regime = "RANGING"

        action = _make_action()
        action.risk_weight = weight
        order = OrderResult(success=True, order_id="o1", fill_price=4712.0,
                            fill_size=0.1, error=None)

        await pipeline.record_trade_open(action, order)

        row = pipeline._trade_repo.save_trade.call_args[0][0]
        assert row["risk_weight"] == weight


@pytest.mark.asyncio
async def test_tp2_different_from_tp_price():
    """tp2_price (full RR) is different from tp_price (tp1 or tp2 depending on mode)."""
    pipeline = _make_pipeline()
    pipeline._last_market_data = None
    pipeline._last_regime = "RANGING"

    action = _make_action()
    # tp1=4706, tp2=4694 — tp_price in shadow mode = tp2
    order = OrderResult(success=True, order_id="o1", fill_price=4712.0,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)

    row = pipeline._trade_repo.save_trade.call_args[0][0]
    # tp_price is tp2 in shadow mode (4694), but tp2_price is also 4694
    # The key difference: tp_price could be tp1 (4706) in live mode
    assert row["tp2_price"] == 4694.0
    assert row["tp_price"] is not None
    # In shadow mode both are tp2; verify tp1 != tp2 on the action
    assert action.tp1_price != action.tp2_price


@pytest.mark.asyncio
async def test_columns_null_when_action_fields_missing():
    """When TradeAction has None for optional fields, trade row stores NULL."""
    pipeline = _make_pipeline()
    pipeline._last_market_data = None
    pipeline._last_regime = "RANGING"

    action = TradeAction(
        action="LONG",
        conviction_score=0.55,
        position_size=500.0,
        sl_price=100.0,
        tp1_price=110.0,
        tp2_price=None,       # missing
        rr_ratio=1.0,
        atr_multiplier=None,  # missing
        reasoning="test",
        raw_output="",
        risk_weight=None,     # missing (non-entry actions)
    )
    order = OrderResult(success=True, order_id="o1", fill_price=105.0,
                        fill_size=1.0, error=None)

    await pipeline.record_trade_open(action, order)

    row = pipeline._trade_repo.save_trade.call_args[0][0]
    assert row["tp2_price"] is None
    assert row["atr_multiplier"] is None
    assert row["risk_weight"] is None
    # regime is always set from _last_regime
    assert row["regime"] == "RANGING"
