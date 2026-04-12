"""Tests for SL/TP recomputation from actual fill price in record_trade_open.

Bug: DecisionAgent computes SL/TP from the last candle's close price, but the
actual fill price differs. For SHORT trades where price dropped between candle
close and fill, the SL becomes wider and TP narrower — producing RR ratios of
0.52–0.97 when the profile targets 1.0–1.5.

Fix: record_trade_open recomputes SL/TP using compute_sl_tp with the actual
fill price, same ATR, profile, and swing levels.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from engine.config import DEFAULT_PROFILES, get_dynamic_profile
from engine.events import InProcessBus
from engine.execution.risk_profiles import compute_sl_tp
from engine.pipeline import AnalysisPipeline
from engine.types import MarketData, OrderResult, TradeAction


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_market_data(
    symbol: str = "ETH-USDC",
    timeframe: str = "1h",
    close_price: float = 4715.0,
    atr: float = 8.10,
    volatility_percentile: float = 50.0,
    swing_highs: list[float] | None = None,
    swing_lows: list[float] | None = None,
) -> MarketData:
    candles = [{"timestamp": 1000, "open": close_price, "high": close_price + 5,
                "low": close_price - 5, "close": close_price, "volume": 100}]
    return MarketData(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        num_candles=1,
        lookback_description="~6 hours",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={"atr": atr, "volatility_percentile": volatility_percentile},
        swing_highs=swing_highs or [],
        swing_lows=swing_lows or [],
    )


def _make_pipeline(
    symbol: str = "ETH-USDC",
    timeframe: str = "1h",
    is_shadow: bool = True,
    shadow_fixed_size_usd: float = 500.0,
) -> AnalysisPipeline:
    """Build a minimal pipeline with mocked dependencies."""
    from engine.config import TradingConfig

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
        is_shadow=is_shadow,
        shadow_fixed_size_usd=shadow_fixed_size_usd,
        trade_repo=trade_repo,
    )
    return pipeline


def _make_action(
    direction: str = "SHORT",
    sl_price: float = 4723.10,
    tp1_price: float = 4706.90,
    tp2_price: float = 4694.45,
    rr_ratio: float = 1.5,
    position_size: float = 500.0,
) -> TradeAction:
    return TradeAction(
        action=direction,
        conviction_score=0.72,
        position_size=position_size,
        sl_price=sl_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        rr_ratio=rr_ratio,
        atr_multiplier=1.5,
        reasoning="Test trade",
        raw_output="",
        risk_weight=1.15,
    )


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sl_tp_recomputed_from_fill_price_not_candle_close():
    """SL/TP must be anchored to the actual fill price, not the stale
    candle close that DecisionAgent used."""
    pipeline = _make_pipeline()

    candle_close = 4715.0
    fill_price = 4712.54  # price dropped between candle close and fill
    atr = 8.10

    md = _make_market_data(close_price=candle_close, atr=atr)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    # Action with SL/TP computed from candle close (the bug)
    action = _make_action(direction="SHORT")

    order = OrderResult(success=True, order_id="ord-1", fill_price=fill_price,
                        fill_size=0.1, error=None)

    trade_id = await pipeline.record_trade_open(action, order)
    assert trade_id is not None

    # Verify SL/TP were recomputed — they should differ from originals
    # because fill_price != candle_close
    base = DEFAULT_PROFILES["1h"]
    profile = get_dynamic_profile(base, "RANGING", 50.0)
    expected = compute_sl_tp(
        entry_price=fill_price,
        direction="SHORT",
        atr=atr,
        profile=profile,
        swing_highs=[],
        swing_lows=[],
    )

    assert action.sl_price == expected["sl_price"]
    assert action.tp1_price == expected["tp1_price"]
    assert action.tp2_price == expected["tp2_price"]
    assert action.rr_ratio == expected["rr_ratio"]


@pytest.mark.asyncio
async def test_rr_ratio_correct_when_fill_price_differs():
    """RR ratio must match the profile target (within rounding) after
    recomputation, not the degraded ratio from stale candle-close SL/TP."""
    pipeline = _make_pipeline(timeframe="1h")

    candle_close = 4715.0
    fill_price = 4712.54
    atr = 8.10

    md = _make_market_data(close_price=candle_close, atr=atr)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    action = _make_action(direction="SHORT")
    order = OrderResult(success=True, order_id="ord-1", fill_price=fill_price,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)

    # RANGING regime: rr_min = 1h base (1.5) * RANGING mult (0.7) = 1.05
    base = DEFAULT_PROFILES["1h"]
    profile = get_dynamic_profile(base, "RANGING", 50.0)
    assert abs(action.rr_ratio - profile.rr_min) < 0.01


@pytest.mark.asyncio
async def test_short_rr_not_degraded_by_price_drop():
    """Reproduce the exact bug: SHORT entry where price dropped between
    candle close and fill. Without the fix, RR degrades to ~0.55. With
    the fix, RR matches the profile target."""
    pipeline = _make_pipeline(timeframe="1h")

    candle_close = 4715.0
    fill_price = 4712.54
    atr = 8.10

    md = _make_market_data(close_price=candle_close, atr=atr)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    # Compute what DecisionAgent would have produced (the buggy values)
    base = DEFAULT_PROFILES["1h"]
    profile = get_dynamic_profile(base, "RANGING", 50.0)
    stale_sl_tp = compute_sl_tp(
        entry_price=candle_close,
        direction="SHORT",
        atr=atr,
        profile=profile,
        swing_highs=[],
        swing_lows=[],
    )

    action = _make_action(
        direction="SHORT",
        sl_price=stale_sl_tp["sl_price"],
        tp1_price=stale_sl_tp["tp1_price"],
        tp2_price=stale_sl_tp["tp2_price"],
        rr_ratio=stale_sl_tp["rr_ratio"],
    )

    order = OrderResult(success=True, order_id="ord-1", fill_price=fill_price,
                        fill_size=0.1, error=None)

    # BEFORE fix: with stale SL/TP, the effective RR on the fill would be:
    #   risk = stale_sl - fill = (higher) - 4712.54  → wider
    #   reward = fill - stale_tp1 = 4712.54 - (lower from 4715 anchor) → narrower
    #   RR = reward / risk < 1.0  ← THE BUG

    await pipeline.record_trade_open(action, order)

    # AFTER fix: SL/TP recomputed from fill_price
    # RR should match profile target (within rounding tolerance)
    assert abs(action.rr_ratio - profile.rr_min) < 0.01
    assert action.rr_ratio >= 1.0  # never degraded below 1.0


@pytest.mark.asyncio
async def test_shadow_mode_stores_tp2_price():
    """Shadow mode has no partial exits, so tp_price in the DB should
    be tp2 (full RR target), not tp1 (1:1 RR)."""
    pipeline = _make_pipeline(is_shadow=True)

    md = _make_market_data(atr=8.10)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    action = _make_action(direction="SHORT")
    order = OrderResult(success=True, order_id="ord-1", fill_price=4712.0,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)

    # Verify save_trade was called and tp_price is tp2, not tp1
    save_call = pipeline._trade_repo.save_trade
    assert save_call.called
    trade_row = save_call.call_args[0][0]

    # tp2 is the full RR target (further from entry than tp1)
    assert trade_row["tp_price"] == action.tp2_price
    assert trade_row["tp_price"] != action.tp1_price


@pytest.mark.asyncio
async def test_non_shadow_mode_stores_tp1_price():
    """Live mode does partial exits: tp_price should be tp1 (1:1 RR)."""
    pipeline = _make_pipeline(is_shadow=False, shadow_fixed_size_usd=500.0)

    md = _make_market_data(atr=8.10)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    action = _make_action(direction="LONG")
    order = OrderResult(success=True, order_id="ord-1", fill_price=4718.0,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)

    save_call = pipeline._trade_repo.save_trade
    assert save_call.called
    trade_row = save_call.call_args[0][0]
    assert trade_row["tp_price"] == action.tp1_price


@pytest.mark.asyncio
async def test_recomputation_uses_same_atr_and_profile():
    """The recomputed SL/TP must use the same ATR and regime-adjusted profile
    that DecisionAgent used — only the entry_price changes."""
    pipeline = _make_pipeline(timeframe="4h")

    atr = 15.5
    fill_price = 100.0
    regime = "TRENDING_UP"

    md = _make_market_data(timeframe="4h", close_price=102.0, atr=atr)
    pipeline._last_market_data = md
    pipeline._last_regime = regime

    action = _make_action(direction="LONG", sl_price=80.0, tp1_price=120.0,
                          tp2_price=140.0, rr_ratio=2.0)
    order = OrderResult(success=True, order_id="ord-1", fill_price=fill_price,
                        fill_size=1.0, error=None)

    await pipeline.record_trade_open(action, order)

    # Manually recompute with the expected ATR and profile
    base = DEFAULT_PROFILES["4h"]
    profile = get_dynamic_profile(base, regime, 50.0)
    expected = compute_sl_tp(
        entry_price=fill_price,
        direction="LONG",
        atr=atr,
        profile=profile,
        swing_highs=[],
        swing_lows=[],
    )

    assert action.sl_price == expected["sl_price"]
    assert action.tp1_price == expected["tp1_price"]
    assert action.tp2_price == expected["tp2_price"]
    assert action.rr_ratio == expected["rr_ratio"]


@pytest.mark.asyncio
async def test_recomputation_with_swing_levels():
    """Swing levels from market_data should be used in the recomputation,
    which may snap the SL to structure."""
    pipeline = _make_pipeline(timeframe="1h")

    fill_price = 100.0
    atr = 5.0
    # Swing high at 103 — tighter than ATR-based SL for a SHORT
    swing_highs = [103.0]

    md = _make_market_data(close_price=102.0, atr=atr,
                           swing_highs=swing_highs, swing_lows=[])
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    action = _make_action(direction="SHORT", sl_price=110.0, tp1_price=90.0,
                          tp2_price=85.0, rr_ratio=1.5)
    order = OrderResult(success=True, order_id="ord-1", fill_price=fill_price,
                        fill_size=1.0, error=None)

    await pipeline.record_trade_open(action, order)

    # Manually compute expected
    base = DEFAULT_PROFILES["1h"]
    profile = get_dynamic_profile(base, "RANGING", 50.0)
    expected = compute_sl_tp(
        entry_price=fill_price,
        direction="SHORT",
        atr=atr,
        profile=profile,
        swing_highs=swing_highs,
        swing_lows=[],
    )

    assert action.sl_price == expected["sl_price"]
    assert action.tp1_price == expected["tp1_price"]


@pytest.mark.asyncio
async def test_no_recomputation_when_market_data_missing():
    """If _last_market_data is None (e.g. run_cycle failed before
    Stage 1), keep DecisionAgent's original SL/TP — don't crash."""
    pipeline = _make_pipeline()
    pipeline._last_market_data = None  # not set

    original_sl = 4723.10
    original_tp1 = 4706.90
    action = _make_action(direction="SHORT", sl_price=original_sl,
                          tp1_price=original_tp1)
    order = OrderResult(success=True, order_id="ord-1", fill_price=4712.0,
                        fill_size=0.1, error=None)

    trade_id = await pipeline.record_trade_open(action, order)
    assert trade_id is not None

    # Original values preserved
    assert action.sl_price == original_sl
    assert action.tp1_price == original_tp1


@pytest.mark.asyncio
async def test_no_recomputation_when_atr_zero():
    """If ATR is 0, skip recomputation and keep original SL/TP."""
    pipeline = _make_pipeline()

    md = _make_market_data(atr=0.0)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    original_sl = 4723.10
    action = _make_action(direction="SHORT", sl_price=original_sl)
    order = OrderResult(success=True, order_id="ord-1", fill_price=4712.0,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)
    assert action.sl_price == original_sl


@pytest.mark.asyncio
async def test_long_trade_sl_tp_recomputed_correctly():
    """LONG trades also benefit from the fix: if price rose between candle
    close and fill, SL was too tight and TP too generous."""
    pipeline = _make_pipeline(timeframe="1h")

    candle_close = 100.0
    fill_price = 101.5  # price rose
    atr = 3.0

    md = _make_market_data(close_price=candle_close, atr=atr)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    action = _make_action(direction="LONG", sl_price=95.0, tp1_price=105.0,
                          tp2_price=110.0)
    order = OrderResult(success=True, order_id="ord-1", fill_price=fill_price,
                        fill_size=1.0, error=None)

    await pipeline.record_trade_open(action, order)

    # Recomputed from fill_price
    base = DEFAULT_PROFILES["1h"]
    profile = get_dynamic_profile(base, "RANGING", 50.0)
    expected = compute_sl_tp(
        entry_price=fill_price,
        direction="LONG",
        atr=atr,
        profile=profile,
        swing_highs=[],
        swing_lows=[],
    )

    assert action.sl_price == expected["sl_price"]
    assert action.tp1_price == expected["tp1_price"]
    assert action.rr_ratio == expected["rr_ratio"]


@pytest.mark.asyncio
async def test_fill_price_equals_candle_close_no_change():
    """When fill price equals candle close, recomputation produces identical
    values — no harm, no regression."""
    pipeline = _make_pipeline(timeframe="1h")

    price = 4715.0
    atr = 8.10

    md = _make_market_data(close_price=price, atr=atr)
    pipeline._last_market_data = md
    pipeline._last_regime = "RANGING"

    base = DEFAULT_PROFILES["1h"]
    profile = get_dynamic_profile(base, "RANGING", 50.0)
    original = compute_sl_tp(
        entry_price=price,
        direction="SHORT",
        atr=atr,
        profile=profile,
        swing_highs=[],
        swing_lows=[],
    )

    action = _make_action(
        direction="SHORT",
        sl_price=original["sl_price"],
        tp1_price=original["tp1_price"],
        tp2_price=original["tp2_price"],
        rr_ratio=original["rr_ratio"],
    )
    order = OrderResult(success=True, order_id="ord-1", fill_price=price,
                        fill_size=0.1, error=None)

    await pipeline.record_trade_open(action, order)

    # Values unchanged
    assert action.sl_price == original["sl_price"]
    assert action.tp1_price == original["tp1_price"]
    assert action.rr_ratio == original["rr_ratio"]
