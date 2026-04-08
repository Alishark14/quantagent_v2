"""Unit tests for SimulatedExchangeAdapter."""

from __future__ import annotations

import pytest

from backtesting.sim_exchange import SimulatedExchangeAdapter
from engine.execution.cost_model import ExecutionCost, ExecutionCostModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FlatRateCostModel(ExecutionCostModel):
    """Minimal ExecutionCostModel with a fixed taker rate (for fee tests)."""

    def __init__(self, taker_rate: float = 0.0004) -> None:
        self._taker = taker_rate

    async def refresh(self, adapter) -> None:  # pragma: no cover - unused
        return

    def get_taker_rate(self, symbol: str) -> float:
        return self._taker

    def get_maker_rate(self, symbol: str) -> float:
        return 0.0001

    def estimate_slippage(self, symbol: str, size_usd: float, side: str) -> float:
        return 0.0

    def estimate_spread_cost(self, symbol: str) -> float:
        return 0.0

    def estimate_funding_cost(
        self, symbol: str, direction: str, hold_hours: float
    ) -> float:
        return 0.0


def _candle(ts: int, o: float, h: float, low: float, c: float, v: float = 10.0) -> dict:
    return {"timestamp": ts, "open": o, "high": h, "low": low, "close": c, "volume": v}


def _make_adapter(
    balance: float = 10_000.0,
    slippage: float = 0.0,
    fee_rate: float | None = None,
) -> SimulatedExchangeAdapter:
    fee_model = FlatRateCostModel(fee_rate) if fee_rate is not None else None
    return SimulatedExchangeAdapter(
        initial_balance=balance,
        slippage_pct=slippage,
        fee_model=fee_model,
    )


# ---------------------------------------------------------------------------
# Construction & ABC contract
# ---------------------------------------------------------------------------


def test_capabilities_match_perp_venue():
    a = _make_adapter()
    caps = a.capabilities()
    assert caps.native_sl_tp is True
    assert caps.supports_short is True
    assert caps.has_funding_rate is True
    assert caps.supports_partial_close is True


def test_invalid_constructor_args_raise():
    with pytest.raises(ValueError):
        SimulatedExchangeAdapter(initial_balance=-1.0)
    with pytest.raises(ValueError):
        SimulatedExchangeAdapter(initial_balance=1000.0, slippage_pct=-0.1)


def test_satisfies_exchange_adapter_abc():
    """Cannot be instantiated unless every abstract method is implemented."""
    a = _make_adapter()
    from exchanges.base import ExchangeAdapter
    assert isinstance(a, ExchangeAdapter)
    assert a.name() == "simulated"


# ---------------------------------------------------------------------------
# Market entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_market_long_opens_position():
    a = _make_adapter(balance=10_000.0)
    a.set_current_prices({"BTC-USDC": 100.0})

    res = await a.place_market_order("BTC-USDC", "buy", size=2.0)

    assert res.success is True
    assert res.fill_price == 100.0  # zero slippage
    assert res.fill_size == 2.0

    positions = await a.get_positions()
    assert len(positions) == 1
    p = positions[0]
    assert p.symbol == "BTC-USDC"
    assert p.direction == "long"
    assert p.size == 2.0
    assert p.entry_price == 100.0
    # Mark = entry → unrealized PnL is zero
    assert p.unrealized_pnl == 0.0
    # No fees configured → balance unchanged
    assert await a.get_balance() == 10_000.0


@pytest.mark.asyncio
async def test_market_order_no_price_raises():
    a = _make_adapter()
    with pytest.raises(ValueError, match="No current price"):
        await a.place_market_order("BTC-USDC", "buy", size=1.0)


@pytest.mark.asyncio
async def test_buy_slippage_increases_fill_price():
    a = _make_adapter(slippage=0.001)  # 10 bps
    a.set_current_prices({"BTC-USDC": 100.0})

    res = await a.place_market_order("BTC-USDC", "buy", size=1.0)
    assert res.fill_price == pytest.approx(100.0 * 1.001)


@pytest.mark.asyncio
async def test_sell_slippage_decreases_fill_price():
    a = _make_adapter(slippage=0.001)
    a.set_current_prices({"BTC-USDC": 100.0})

    res = await a.place_market_order("BTC-USDC", "sell", size=1.0)
    assert res.fill_price == pytest.approx(100.0 * 0.999)
    pos = (await a.get_positions())[0]
    assert pos.direction == "short"


@pytest.mark.asyncio
async def test_fee_deducted_on_entry():
    a = _make_adapter(balance=10_000.0, fee_rate=0.0004)  # 4 bps
    a.set_current_prices({"BTC-USDC": 100.0})

    await a.place_market_order("BTC-USDC", "buy", size=10.0)
    # fee = 10 * 100 * 0.0004 = 0.4
    assert await a.get_balance() == pytest.approx(10_000.0 - 0.4)


@pytest.mark.asyncio
async def test_same_direction_market_averages_entry():
    a = _make_adapter()
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=2.0)

    a.set_current_prices({"BTC-USDC": 110.0})
    await a.place_market_order("BTC-USDC", "buy", size=2.0)

    pos = (await a.get_positions())[0]
    assert pos.size == 4.0
    assert pos.entry_price == pytest.approx(105.0)  # (100+110)/2


@pytest.mark.asyncio
async def test_opposite_market_closes_then_flips_residual():
    a = _make_adapter(balance=10_000.0)
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=2.0)

    a.set_current_prices({"BTC-USDC": 110.0})
    # Sell 3 → closes 2 long (+20 PnL) and opens 1 short
    await a.place_market_order("BTC-USDC", "sell", size=3.0)

    positions = await a.get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.direction == "short"
    assert pos.size == 1.0
    assert pos.entry_price == 110.0

    history = a.get_trade_history()
    assert len(history) == 1
    closed = history[0]
    assert closed["side"] == "long"
    assert closed["size"] == 2.0
    assert closed["pnl"] == pytest.approx(20.0)
    assert await a.get_balance() == pytest.approx(10_020.0)


# ---------------------------------------------------------------------------
# SL / TP triggering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sl_triggers_on_long_when_low_breaches():
    a = _make_adapter(balance=10_000.0)
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)

    assert len(a.get_open_orders("BTC-USDC")) == 1
    assert len(await a.get_positions()) == 1

    # Next candle dips to 94 → SL hit
    a.set_current_candle("BTC-USDC", _candle(2, 100, 100, 94, 96))

    assert await a.get_positions() == []
    assert len(a.get_open_orders("BTC-USDC")) == 0
    history = a.get_trade_history()
    assert len(history) == 1
    assert history[0]["reason"] == "stop_hit"
    assert history[0]["pnl"] == pytest.approx(-5.0)  # 95 - 100
    assert history[0]["exit_price"] == 95.0


@pytest.mark.asyncio
async def test_tp_triggers_on_long_when_high_breaches():
    a = _make_adapter()
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_tp_order("BTC-USDC", "sell", size=1.0, trigger_price=110.0)

    a.set_current_candle("BTC-USDC", _candle(2, 100, 112, 100, 111))

    assert await a.get_positions() == []
    history = a.get_trade_history()
    assert history[-1]["reason"] == "take_profit_hit"
    assert history[-1]["exit_price"] == 110.0
    assert history[-1]["pnl"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_sl_triggers_on_short_when_high_breaches():
    a = _make_adapter()
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "sell", size=1.0)
    await a.place_sl_order("BTC-USDC", "buy", size=1.0, trigger_price=105.0)

    a.set_current_candle("BTC-USDC", _candle(2, 100, 106, 100, 104))

    assert await a.get_positions() == []
    h = a.get_trade_history()[-1]
    assert h["reason"] == "stop_hit"
    assert h["exit_price"] == 105.0
    assert h["pnl"] == pytest.approx(-5.0)


@pytest.mark.asyncio
async def test_sl_takes_priority_over_tp_on_same_candle():
    """If both SL and TP fall inside one candle, SL fills first (conservative)."""
    a = _make_adapter()
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)
    await a.place_tp_order("BTC-USDC", "sell", size=1.0, trigger_price=110.0)

    # Wide candle: low 90, high 115 — both SL (95) and TP (110) inside.
    a.set_current_candle("BTC-USDC", _candle(2, 100, 115, 90, 105))

    history = a.get_trade_history()
    assert len(history) == 1  # only one fill
    assert history[0]["reason"] == "stop_hit"
    assert history[0]["exit_price"] == 95.0
    # The TP order was auto-cancelled when the position closed
    assert a.get_open_orders("BTC-USDC") == []


@pytest.mark.asyncio
async def test_sl_fill_applies_slippage_against_trader():
    a = _make_adapter(slippage=0.001)
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    # Reset slippage tracking by closing & reopening at a clean entry
    await a.close_position("BTC-USDC")
    a.set_current_candle("BTC-USDC", _candle(2, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)

    a.set_current_candle("BTC-USDC", _candle(3, 100, 100, 94, 96))

    # Long SL fills at 95 * (1 - 0.001) = 94.905
    h = a.get_trade_history()[-1]
    assert h["exit_price"] == pytest.approx(94.905)


# ---------------------------------------------------------------------------
# close_position / modify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_position_realizes_pnl():
    a = _make_adapter(balance=10_000.0)
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=2.0)

    a.set_current_prices({"BTC-USDC": 110.0})
    res = await a.close_position("BTC-USDC")

    assert res.success is True
    assert await a.get_positions() == []
    assert await a.get_balance() == pytest.approx(10_020.0)  # +20 PnL
    h = a.get_trade_history()[-1]
    assert h["pnl"] == pytest.approx(20.0)
    assert h["reason"] == "manual_close"


@pytest.mark.asyncio
async def test_close_nonexistent_position_is_noop_success():
    a = _make_adapter()
    a.set_current_prices({"BTC-USDC": 100.0})
    res = await a.close_position("BTC-USDC")
    assert res.success is True
    assert res.fill_size is None


@pytest.mark.asyncio
async def test_modify_sl_replaces_existing_trigger():
    a = _make_adapter()
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)

    res = await a.modify_sl("BTC-USDC", new_price=97.0)
    assert res.success is True
    orders = a.get_open_orders("BTC-USDC")
    assert len(orders) == 1
    assert orders[0]["type"] == "stop"
    assert orders[0]["price"] == 97.0


@pytest.mark.asyncio
async def test_modify_sl_with_no_position_fails():
    a = _make_adapter()
    res = await a.modify_sl("BTC-USDC", new_price=95.0)
    assert res.success is False
    assert "No position" in res.error


# ---------------------------------------------------------------------------
# Cancel orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_removes_from_open_orders():
    a = _make_adapter()
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    res = await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)
    assert await a.cancel_order("BTC-USDC", res.order_id) is True
    assert a.get_open_orders("BTC-USDC") == []


@pytest.mark.asyncio
async def test_cancel_all_orders_returns_count():
    a = _make_adapter()
    a.set_current_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)
    await a.place_tp_order("BTC-USDC", "sell", size=1.0, trigger_price=110.0)

    n = await a.cancel_all_orders("BTC-USDC")
    assert n == 2
    assert a.get_open_orders("BTC-USDC") == []


# ---------------------------------------------------------------------------
# Multi-symbol & funding & equity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_concurrent_positions_different_symbols():
    a = _make_adapter(balance=20_000.0)
    a.set_current_prices({"BTC-USDC": 100.0, "ETH-USDC": 50.0})
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_market_order("ETH-USDC", "sell", size=4.0)

    positions = await a.get_positions()
    assert {p.symbol for p in positions} == {"BTC-USDC", "ETH-USDC"}
    btc = [p for p in positions if p.symbol == "BTC-USDC"][0]
    eth = [p for p in positions if p.symbol == "ETH-USDC"][0]
    assert btc.direction == "long"
    assert eth.direction == "short"
    assert eth.size == 4.0


@pytest.mark.asyncio
async def test_funding_rate_charges_long_credits_short():
    a = _make_adapter(balance=10_000.0)
    a.set_current_prices({"BTC-USDC": 100.0, "ETH-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=10.0)   # long 1000 notional
    await a.place_market_order("ETH-USDC", "sell", size=10.0)  # short 1000 notional

    # Positive funding: longs pay, shorts receive
    a.apply_funding("BTC-USDC", rate=0.0001)  # -0.10
    a.apply_funding("ETH-USDC", rate=0.0001)  # +0.10
    assert await a.get_balance() == pytest.approx(10_000.0)  # net zero


@pytest.mark.asyncio
async def test_funding_with_no_position_is_noop():
    a = _make_adapter(balance=10_000.0)
    a.apply_funding("BTC-USDC", rate=0.01)
    assert await a.get_balance() == 10_000.0


@pytest.mark.asyncio
async def test_equity_curve_tracks_balance_over_time():
    a = _make_adapter(balance=10_000.0)
    a.set_current_candle("BTC-USDC", _candle(1000, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)

    a.set_current_candle("BTC-USDC", _candle(2000, 100, 105, 100, 105))
    a.set_current_candle("BTC-USDC", _candle(3000, 105, 106, 100, 102))
    a.set_current_candle("BTC-USDC", _candle(4000, 102, 110, 100, 110))

    curve = a.get_equity_curve()
    timestamps = [t for t, _ in curve]
    equities = [e for _, e in curve]

    # Strictly increasing timestamps (set_current_candle coalesces dupes)
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)

    # First snapshot is the entry candle: equity = balance + 0 unrealized
    assert equities[0] == pytest.approx(10_000.0)
    # Mid candle (close=105): unrealized +5
    assert equities[1] == pytest.approx(10_005.0)
    # Final candle (close=110): unrealized +10
    assert equities[-1] == pytest.approx(10_010.0)


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_limit_order_fills_when_price_crosses():
    a = _make_adapter()
    a.set_current_prices({"BTC-USDC": 110.0})
    res = await a.place_limit_order("BTC-USDC", "buy", size=1.0, price=100.0)
    assert res.success is True
    assert len(a.get_open_orders("BTC-USDC")) == 1

    # Price drops to 99 → buy limit at 100 fills
    a.set_current_prices({"BTC-USDC": 99.0})
    assert a.get_open_orders("BTC-USDC") == []
    pos = (await a.get_positions())[0]
    assert pos.size == 1.0
    assert pos.entry_price == 100.0  # filled at limit price, not 99
