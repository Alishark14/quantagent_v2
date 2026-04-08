"""Unit tests for SimExecutor (read facade over SimulatedExchangeAdapter)."""

from __future__ import annotations

import pytest

from backtesting.sim_exchange import SimulatedExchangeAdapter
from backtesting.sim_executor import SimExecutor
from engine.execution.cost_model import ExecutionCostModel


class FlatRateCostModel(ExecutionCostModel):
    def __init__(self, taker_rate: float = 0.0004) -> None:
        self._taker = taker_rate

    async def refresh(self, adapter) -> None:  # pragma: no cover
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


def _candle(ts, o, h, low, c, v=10.0) -> dict:
    return {"timestamp": ts, "open": o, "high": h, "low": low, "close": c, "volume": v}


@pytest.fixture
def executor() -> SimExecutor:
    adapter = SimulatedExchangeAdapter(
        initial_balance=10_000.0,
        slippage_pct=0.0,
        fee_model=FlatRateCostModel(0.0004),
    )
    return SimExecutor(adapter)


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trade_history_records_completed_trades(executor):
    a = executor.adapter
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=1.0)

    assert executor.get_trade_history() == []  # nothing closed yet

    a.set_current_prices({"BTC-USDC": 110.0})
    await a.close_position("BTC-USDC")

    history = executor.get_trade_history()
    assert len(history) == 1
    record = history[0]
    # Required fields per spec
    for key in (
        "timestamp", "symbol", "side", "entry_price",
        "exit_price", "size", "fee", "slippage", "pnl",
    ):
        assert key in record
    assert record["symbol"] == "BTC-USDC"
    assert record["side"] == "long"
    assert record["entry_price"] == 100.0
    assert record["exit_price"] == 110.0


@pytest.mark.asyncio
async def test_equity_curve_proxies_adapter(executor):
    a = executor.adapter
    a.set_current_candle("BTC-USDC", _candle(1000, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    a.set_current_candle("BTC-USDC", _candle(2000, 100, 105, 100, 105))
    a.set_current_candle("BTC-USDC", _candle(3000, 105, 110, 100, 108))

    curve = executor.get_equity_curve()
    assert len(curve) >= 3
    assert all(isinstance(t, int) and isinstance(e, float) for t, e in curve)


@pytest.mark.asyncio
async def test_balance_and_equity_properties(executor):
    a = executor.adapter
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    # Fee = 100 * 1 * 0.0004 = 0.04
    assert executor.balance == pytest.approx(9_999.96)

    a.set_current_prices({"BTC-USDC": 110.0})
    # Equity = balance + 10 unrealized
    assert executor.equity() == pytest.approx(10_009.96)


# ---------------------------------------------------------------------------
# Tick driver pass-throughs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_candle_drives_sl_tp(executor):
    a = executor.adapter
    executor.on_candle("BTC-USDC", _candle(1, 100, 101, 99, 100))
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    await a.place_sl_order("BTC-USDC", "sell", size=1.0, trigger_price=95.0)

    executor.on_candle("BTC-USDC", _candle(2, 100, 100, 94, 96))

    assert await a.get_positions() == []
    assert len(executor.get_trade_history()) == 1


@pytest.mark.asyncio
async def test_on_prices_passthrough(executor):
    a = executor.adapter
    executor.on_prices({"BTC-USDC": 100.0}, timestamp=1000)
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    executor.on_prices({"BTC-USDC": 105.0}, timestamp=2000)
    assert (await a.get_positions())[0].unrealized_pnl == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_apply_funding_passthrough(executor):
    a = executor.adapter
    executor.on_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=10.0)
    bal_before = await a.get_balance()
    executor.apply_funding("BTC-USDC", rate=0.0001)
    # 10 * 100 * 0.0001 = 0.10 charged to long
    assert await a.get_balance() == pytest.approx(bal_before - 0.10)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_metrics_after_two_trades(executor):
    a = executor.adapter

    # Trade 1: long 100 → 110 = +10 PnL (minus fees)
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    a.set_current_prices({"BTC-USDC": 110.0})
    await a.close_position("BTC-USDC")

    # Trade 2: long 110 → 100 = -10 PnL
    await a.place_market_order("BTC-USDC", "buy", size=1.0)
    a.set_current_prices({"BTC-USDC": 100.0})
    await a.close_position("BTC-USDC")

    assert executor.num_trades() == 2
    assert executor.win_rate() == pytest.approx(0.5)

    # Total pnl = sum of net pnl across trades
    expected = sum(t["pnl"] for t in executor.get_trade_history())
    assert executor.total_pnl() == pytest.approx(expected)
    # Total fees > 0 because fee_model is set
    assert executor.total_fees() > 0


def test_metrics_on_empty_history(executor):
    assert executor.num_trades() == 0
    assert executor.win_rate() == 0.0
    assert executor.total_pnl() == 0.0
    assert executor.total_fees() == 0.0
