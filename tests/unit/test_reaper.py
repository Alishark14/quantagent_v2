"""Unit tests for OrphanReaper — detects and handles orphan positions."""

from __future__ import annotations

import pytest

from engine.events import InProcessBus
from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from sentinel.position_manager import PositionManager
from sentinel.reaper import OrphanReaper


# ---------------------------------------------------------------------------
# Mock adapter for reaper tests
# ---------------------------------------------------------------------------

class MockReaperAdapter(ExchangeAdapter):
    """Adapter that returns configurable positions and tracks SL placements."""

    def __init__(
        self,
        positions: list[Position] | None = None,
        sl_succeeds: bool = True,
    ) -> None:
        self._positions = positions or []
        self._sl_succeeds = sl_succeeds
        self.sl_orders_placed: list[tuple[str, str, float, float]] = []

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market"], supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100):
        return []

    async def get_ticker(self, symbol):
        return {}

    async def get_balance(self):
        return 0.0

    async def get_positions(self, symbol=None):
        return list(self._positions)

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def place_sl_order(self, symbol, side, size, trigger_price):
        self.sl_orders_placed.append((symbol, side, size, trigger_price))
        if self._sl_succeeds:
            return OrderResult(success=True, order_id="emerg-sl", fill_price=None, fill_size=None, error=None)
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="SL rejected")

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol, order_id):
        return True

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def modify_sl(self, symbol, new_price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrphanReaper:

    @pytest.mark.asyncio
    async def test_no_orphans_when_all_managed(self) -> None:
        adapter = MockReaperAdapter(positions=[
            Position("BTC-USDC", "long", 0.1, 65000.0, 100.0, 5.0),
        ])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)
        pm.register_position("BTC-USDC", "long", 65000.0, 64000.0, 500.0)

        reaper = OrphanReaper(adapter, pm)
        orphans = await reaper.check()

        assert len(orphans) == 0

    @pytest.mark.asyncio
    async def test_orphan_detected_when_not_managed(self) -> None:
        adapter = MockReaperAdapter(positions=[
            Position("ETH-USDC", "long", 1.0, 3000.0, 50.0, 5.0),
        ])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)
        # ETH-USDC NOT registered with PositionManager

        reaper = OrphanReaper(adapter, pm, default_atr=100.0)
        orphans = await reaper.check()

        assert len(orphans) == 1
        assert orphans[0]["symbol"] == "ETH-USDC"

    @pytest.mark.asyncio
    async def test_emergency_sl_placed_for_orphan(self) -> None:
        adapter = MockReaperAdapter(positions=[
            Position("ETH-USDC", "long", 1.0, 3000.0, 50.0, 5.0),
        ])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)

        reaper = OrphanReaper(adapter, pm, default_atr=100.0)
        orphans = await reaper.check()

        # Emergency SL at entry - 2*ATR = 3000 - 200 = 2800
        assert len(adapter.sl_orders_placed) == 1
        symbol, side, size, price = adapter.sl_orders_placed[0]
        assert symbol == "ETH-USDC"
        assert side == "sell"  # close side for long
        assert size == 1.0
        assert price == 2800.0

    @pytest.mark.asyncio
    async def test_emergency_sl_short_position(self) -> None:
        adapter = MockReaperAdapter(positions=[
            Position("SOL-USDC", "short", 10.0, 150.0, -5.0, 5.0),
        ])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)

        reaper = OrphanReaper(adapter, pm, default_atr=10.0)
        orphans = await reaper.check()

        # Emergency SL at entry + 2*ATR = 150 + 20 = 170
        symbol, side, size, price = adapter.sl_orders_placed[0]
        assert side == "buy"  # close side for short
        assert price == 170.0

    @pytest.mark.asyncio
    async def test_emergency_sl_failure_logged(self) -> None:
        adapter = MockReaperAdapter(
            positions=[Position("ETH-USDC", "long", 1.0, 3000.0, 50.0, 5.0)],
            sl_succeeds=False,
        )
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)

        reaper = OrphanReaper(adapter, pm, default_atr=100.0)
        orphans = await reaper.check()

        assert orphans[0]["sl_order_success"] is False

    @pytest.mark.asyncio
    async def test_multiple_positions_mixed(self) -> None:
        adapter = MockReaperAdapter(positions=[
            Position("BTC-USDC", "long", 0.1, 65000.0, 100.0, 5.0),
            Position("ETH-USDC", "long", 1.0, 3000.0, 50.0, 5.0),
            Position("SOL-USDC", "short", 10.0, 150.0, -5.0, 5.0),
        ])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)
        pm.register_position("BTC-USDC", "long", 65000.0, 64000.0, 500.0)
        # ETH and SOL are orphans

        reaper = OrphanReaper(adapter, pm, default_atr=100.0)
        orphans = await reaper.check()

        assert len(orphans) == 2
        symbols = {o["symbol"] for o in orphans}
        assert symbols == {"ETH-USDC", "SOL-USDC"}

    @pytest.mark.asyncio
    async def test_no_positions_returns_empty(self) -> None:
        adapter = MockReaperAdapter(positions=[])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)

        reaper = OrphanReaper(adapter, pm)
        orphans = await reaper.check()

        assert orphans == []

    @pytest.mark.asyncio
    async def test_orphan_tracked_in_summary(self) -> None:
        adapter = MockReaperAdapter(positions=[
            Position("ETH-USDC", "long", 1.0, 3000.0, 50.0, 5.0),
        ])
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)

        reaper = OrphanReaper(adapter, pm, default_atr=100.0)
        await reaper.check()

        summary = reaper.summary()
        assert summary["orphans_found"] == 1
        assert summary["emergency_sl_placed"] == 1
