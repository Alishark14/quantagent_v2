"""Unit tests for Executor — bridge between DecisionAgent and exchange."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engine.config import TradingConfig
from engine.events import Event, InProcessBus, TradeClosed, TradeOpened
from engine.execution.executor import Executor
from engine.types import AdapterCapabilities, OrderResult, Position, TradeAction
from exchanges.base import ExchangeAdapter


# ---------------------------------------------------------------------------
# Mock adapter that tracks all calls
# ---------------------------------------------------------------------------

class MockAdapter(ExchangeAdapter):
    """Records every call for assertion. Configurable failure modes."""

    def __init__(self, sl_fails: bool = False, close_fails: bool = False) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._sl_fails = sl_fails
        self._close_fails = close_fails

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market", "limit", "stop"], supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100):
        self.calls.append(("fetch_ohlcv", (symbol, timeframe, limit)))
        return []

    async def get_ticker(self, symbol):
        self.calls.append(("get_ticker", (symbol,)))
        return {"last": 65000.0}

    async def get_balance(self):
        self.calls.append(("get_balance", ()))
        return 10000.0

    async def get_positions(self, symbol=None):
        self.calls.append(("get_positions", (symbol,)))
        return []

    async def place_market_order(self, symbol, side, size):
        self.calls.append(("place_market_order", (symbol, side, size)))
        return OrderResult(
            success=True, order_id="mkt-001",
            fill_price=65000.0, fill_size=size, error=None,
        )

    async def place_limit_order(self, symbol, side, size, price):
        self.calls.append(("place_limit_order", (symbol, side, size, price)))
        return OrderResult(success=True, order_id="lmt-001", fill_price=price, fill_size=size, error=None)

    async def place_sl_order(self, symbol, side, size, trigger_price):
        self.calls.append(("place_sl_order", (symbol, side, size, trigger_price)))
        if self._sl_fails:
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="SL rejected")
        return OrderResult(success=True, order_id="sl-001", fill_price=None, fill_size=None, error=None)

    async def place_tp_order(self, symbol, side, size, trigger_price):
        self.calls.append(("place_tp_order", (symbol, side, size, trigger_price)))
        return OrderResult(success=True, order_id="tp-001", fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel_order", (symbol, order_id)))
        return True

    async def cancel_all_orders(self, symbol):
        self.calls.append(("cancel_all_orders", (symbol,)))
        return 3

    async def close_position(self, symbol):
        self.calls.append(("close_position", (symbol,)))
        if self._close_fails:
            return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="close failed")
        return OrderResult(success=True, order_id="close-001", fill_price=65100.0, fill_size=0.1, error=None)

    async def modify_sl(self, symbol, new_price):
        self.calls.append(("modify_sl", (symbol, new_price)))
        return OrderResult(success=True, order_id="sl-mod-001", fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol, new_price):
        self.calls.append(("modify_tp", (symbol, new_price)))
        return OrderResult(success=True, order_id="tp-mod-001", fill_price=None, fill_size=None, error=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config() -> TradingConfig:
    return TradingConfig(symbol="BTC-USDC", timeframe="1h")


def _long_action(position_size: float = 650.0) -> TradeAction:
    return TradeAction(
        action="LONG",
        conviction_score=0.72,
        position_size=position_size,
        sl_price=64000.0,
        tp1_price=66000.0,
        tp2_price=67500.0,
        rr_ratio=1.5,
        atr_multiplier=1.2,
        reasoning="Strong bullish setup.",
        raw_output="...",
    )


def _short_action() -> TradeAction:
    return TradeAction(
        action="SHORT",
        conviction_score=0.68,
        position_size=650.0,
        sl_price=66000.0,
        tp1_price=64000.0,
        tp2_price=62500.0,
        rr_ratio=1.5,
        atr_multiplier=1.2,
        reasoning="Bearish reversal.",
        raw_output="...",
    )


def _add_long_action() -> TradeAction:
    return TradeAction(
        action="ADD_LONG",
        conviction_score=0.78,
        position_size=325.0,
        sl_price=64500.0,
        tp1_price=None,
        tp2_price=None,
        rr_ratio=None,
        atr_multiplier=None,
        reasoning="Pyramid: price moved in favor.",
        raw_output="...",
    )


def _close_all_action() -> TradeAction:
    return TradeAction(
        action="CLOSE_ALL",
        conviction_score=0.65,
        position_size=None,
        sl_price=None,
        tp1_price=None,
        tp2_price=None,
        rr_ratio=None,
        atr_multiplier=None,
        reasoning="Contrary signal detected.",
        raw_output="...",
    )


def _skip_action() -> TradeAction:
    return TradeAction(
        action="SKIP",
        conviction_score=0.35,
        position_size=None,
        sl_price=None,
        tp1_price=None,
        tp2_price=None,
        rr_ratio=None,
        atr_multiplier=None,
        reasoning="Below threshold.",
        raw_output="",
    )


def _hold_action() -> TradeAction:
    return TradeAction(
        action="HOLD",
        conviction_score=0.55,
        position_size=None,
        sl_price=None,
        tp1_price=None,
        tp2_price=None,
        rr_ratio=None,
        atr_multiplier=None,
        reasoning="Position intact, monitoring.",
        raw_output="...",
    )


def _call_names(adapter: MockAdapter) -> list[str]:
    """Extract just the method names from adapter call log."""
    return [c[0] for c in adapter.calls]


# ---------------------------------------------------------------------------
# Tests: LONG entry
# ---------------------------------------------------------------------------

class TestExecutorLong:

    @pytest.mark.asyncio
    async def test_long_places_market_sl_tp1_tp2(self) -> None:
        adapter = MockAdapter()
        bus = InProcessBus()
        executor = Executor(adapter, bus, _config())

        result = await executor.execute(_long_action(), "BTC-USDC")

        assert result.success is True
        assert result.fill_price == 65000.0
        names = _call_names(adapter)
        assert "place_market_order" in names
        assert "place_sl_order" in names
        assert names.count("place_tp_order") == 2  # TP1 + TP2

    @pytest.mark.asyncio
    async def test_long_market_order_is_buy(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_long_action(), "BTC-USDC")

        market_call = [c for c in adapter.calls if c[0] == "place_market_order"][0]
        assert market_call[1][1] == "buy"  # side

    @pytest.mark.asyncio
    async def test_long_sl_side_is_sell(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_long_action(), "BTC-USDC")

        sl_call = [c for c in adapter.calls if c[0] == "place_sl_order"][0]
        assert sl_call[1][1] == "sell"  # close side

    @pytest.mark.asyncio
    async def test_long_tp_splits_50_50(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        action = _long_action(position_size=650.0)
        # size = 650 / 64000 (sl_price as ref) ≈ 0.01015625
        await executor.execute(action, "BTC-USDC")

        tp_calls = [c for c in adapter.calls if c[0] == "place_tp_order"]
        assert len(tp_calls) == 2
        tp1_size = tp_calls[0][1][2]
        tp2_size = tp_calls[1][1][2]
        # TP1 should be ~half, TP2 the remainder
        total = tp1_size + tp2_size
        assert tp1_size == pytest.approx(total / 2, abs=0.00000001)

    @pytest.mark.asyncio
    async def test_long_tp1_price_and_tp2_price(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_long_action(), "BTC-USDC")

        tp_calls = [c for c in adapter.calls if c[0] == "place_tp_order"]
        assert tp_calls[0][1][3] == 66000.0  # TP1 trigger
        assert tp_calls[1][1][3] == 67500.0  # TP2 trigger

    @pytest.mark.asyncio
    async def test_long_usd_to_base_conversion(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        action = _long_action(position_size=6400.0)  # $6400 / $64000 (SL ref) = 0.1
        await executor.execute(action, "BTC-USDC")

        market_call = [c for c in adapter.calls if c[0] == "place_market_order"][0]
        assert market_call[1][2] == 0.1  # size in BTC


# ---------------------------------------------------------------------------
# Tests: SHORT entry
# ---------------------------------------------------------------------------

class TestExecutorShort:

    @pytest.mark.asyncio
    async def test_short_market_order_is_sell(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_short_action(), "BTC-USDC")

        market_call = [c for c in adapter.calls if c[0] == "place_market_order"][0]
        assert market_call[1][1] == "sell"

    @pytest.mark.asyncio
    async def test_short_sl_side_is_buy(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_short_action(), "BTC-USDC")

        sl_call = [c for c in adapter.calls if c[0] == "place_sl_order"][0]
        assert sl_call[1][1] == "buy"  # close side for short


# ---------------------------------------------------------------------------
# Tests: SL failure -> emergency close
# ---------------------------------------------------------------------------

class TestExecutorSLFailure:

    @pytest.mark.asyncio
    async def test_sl_failure_triggers_emergency_close(self) -> None:
        adapter = MockAdapter(sl_fails=True)
        executor = Executor(adapter, InProcessBus(), _config())

        result = await executor.execute(_long_action(), "BTC-USDC")

        assert result.success is False
        assert "SL placement failed" in result.error
        names = _call_names(adapter)
        assert "place_market_order" in names
        assert "place_sl_order" in names
        assert "close_position" in names  # emergency close

    @pytest.mark.asyncio
    async def test_sl_failure_no_tp_orders_placed(self) -> None:
        adapter = MockAdapter(sl_fails=True)
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_long_action(), "BTC-USDC")

        names = _call_names(adapter)
        assert "place_tp_order" not in names  # no TPs after SL failure


# ---------------------------------------------------------------------------
# Tests: CLOSE_ALL
# ---------------------------------------------------------------------------

class TestExecutorCloseAll:

    @pytest.mark.asyncio
    async def test_close_all_cancels_then_closes(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        result = await executor.execute(_close_all_action(), "BTC-USDC")

        assert result.success is True
        names = _call_names(adapter)
        # cancel_all_orders must come before close_position
        cancel_idx = names.index("cancel_all_orders")
        close_idx = names.index("close_position")
        assert cancel_idx < close_idx

    @pytest.mark.asyncio
    async def test_close_all_returns_close_result(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        result = await executor.execute(_close_all_action(), "BTC-USDC")

        assert result.fill_price == 65100.0
        assert result.order_id == "close-001"


# ---------------------------------------------------------------------------
# Tests: ADD_LONG (pyramid)
# ---------------------------------------------------------------------------

class TestExecutorPyramid:

    @pytest.mark.asyncio
    async def test_add_long_places_market_and_adjusts_sl(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        result = await executor.execute(_add_long_action(), "BTC-USDC")

        assert result.success is True
        names = _call_names(adapter)
        assert "place_market_order" in names
        assert "modify_sl" in names

    @pytest.mark.asyncio
    async def test_add_long_modify_sl_price(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        await executor.execute(_add_long_action(), "BTC-USDC")

        sl_mod = [c for c in adapter.calls if c[0] == "modify_sl"][0]
        assert sl_mod[1][1] == 64500.0  # new SL price


# ---------------------------------------------------------------------------
# Tests: SKIP / HOLD — no adapter calls
# ---------------------------------------------------------------------------

class TestExecutorNoOp:

    @pytest.mark.asyncio
    async def test_skip_no_adapter_calls(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        result = await executor.execute(_skip_action(), "BTC-USDC")

        assert result.success is True
        assert len(adapter.calls) == 0

    @pytest.mark.asyncio
    async def test_hold_no_adapter_calls(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        result = await executor.execute(_hold_action(), "BTC-USDC")

        assert result.success is True
        assert len(adapter.calls) == 0


# ---------------------------------------------------------------------------
# Tests: Event emission
# ---------------------------------------------------------------------------

class TestExecutorEvents:

    @pytest.mark.asyncio
    async def test_long_emits_trade_opened(self) -> None:
        adapter = MockAdapter()
        bus = InProcessBus()
        executor = Executor(adapter, bus, _config())

        events: list[TradeOpened] = []
        bus.subscribe(TradeOpened, lambda e: events.append(e))

        await executor.execute(_long_action(), "BTC-USDC")

        assert len(events) == 1
        assert events[0].trade_action.action == "LONG"
        assert events[0].order_result.success is True

    @pytest.mark.asyncio
    async def test_close_all_emits_trade_closed(self) -> None:
        adapter = MockAdapter()
        bus = InProcessBus()
        executor = Executor(adapter, bus, _config())

        events: list[TradeClosed] = []
        bus.subscribe(TradeClosed, lambda e: events.append(e))

        await executor.execute(_close_all_action(), "BTC-USDC")

        assert len(events) == 1
        assert events[0].symbol == "BTC-USDC"
        assert events[0].exit_reason == "CLOSE_ALL"

    @pytest.mark.asyncio
    async def test_add_long_emits_trade_opened(self) -> None:
        adapter = MockAdapter()
        bus = InProcessBus()
        executor = Executor(adapter, bus, _config())

        events: list[TradeOpened] = []
        bus.subscribe(TradeOpened, lambda e: events.append(e))

        await executor.execute(_add_long_action(), "BTC-USDC")

        assert len(events) == 1
        assert events[0].trade_action.action == "ADD_LONG"

    @pytest.mark.asyncio
    async def test_skip_no_events(self) -> None:
        adapter = MockAdapter()
        bus = InProcessBus()
        executor = Executor(adapter, bus, _config())

        events: list[Event] = []
        bus.subscribe(TradeOpened, lambda e: events.append(e))
        bus.subscribe(TradeClosed, lambda e: events.append(e))

        await executor.execute(_skip_action(), "BTC-USDC")

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_sl_failure_no_trade_opened_event(self) -> None:
        """SL failure means trade was emergency closed — no TradeOpened."""
        adapter = MockAdapter(sl_fails=True)
        bus = InProcessBus()
        executor = Executor(adapter, bus, _config())

        events: list[TradeOpened] = []
        bus.subscribe(TradeOpened, lambda e: events.append(e))

        await executor.execute(_long_action(), "BTC-USDC")

        assert len(events) == 0


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestExecutorEdgeCases:

    @pytest.mark.asyncio
    async def test_zero_position_size(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        action = _long_action(position_size=0.0)
        result = await executor.execute(action, "BTC-USDC")

        assert result.success is False
        assert "size <= 0" in result.error
        assert len(adapter.calls) == 0

    @pytest.mark.asyncio
    async def test_none_position_size(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        action = _long_action()
        action.position_size = None
        result = await executor.execute(action, "BTC-USDC")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        adapter = MockAdapter()
        executor = Executor(adapter, InProcessBus(), _config())

        action = _long_action()
        action.action = "MOON"
        result = await executor.execute(action, "BTC-USDC")

        assert result.success is True  # no-op
        assert len(adapter.calls) == 0
