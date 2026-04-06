"""Unit tests for Sentinel PositionManager.

Critical rule: Sentinel only TIGHTENS stops — never widens.
Tests verify modify_sl is called on the adapter when SL changes,
and never called when the adjustment would widen.
"""

from __future__ import annotations

import pytest

from engine.events import InProcessBus, PositionUpdated
from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from sentinel.position_manager import ManagedPosition, PositionManager


# ---------------------------------------------------------------------------
# Mock adapter that tracks modify_sl calls
# ---------------------------------------------------------------------------

class MockSLAdapter(ExchangeAdapter):
    """Minimal adapter that records modify_sl calls."""

    def __init__(self) -> None:
        self.modify_sl_calls: list[tuple[str, float]] = []

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
        return []

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def place_sl_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol, order_id):
        return True

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        self.modify_sl_calls.append((symbol, new_price))
        return OrderResult(success=True, order_id="sl-mod", fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=True, order_id=None, fill_price=None, fill_size=None, error=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pm() -> tuple[PositionManager, InProcessBus, MockSLAdapter]:
    adapter = MockSLAdapter()
    bus = InProcessBus()
    return PositionManager(adapter, bus), bus, adapter


def _register_long(pm: PositionManager, entry: float = 65000.0, sl: float = 64000.0, atr: float = 500.0) -> None:
    pm.register_position("BTC-USDC", "long", entry, sl, atr)


def _register_short(pm: PositionManager, entry: float = 65000.0, sl: float = 66000.0, atr: float = 500.0) -> None:
    pm.register_position("BTC-USDC", "short", entry, sl, atr)


# ---------------------------------------------------------------------------
# Registration / removal
# ---------------------------------------------------------------------------

class TestPositionManagerRegistration:

    def test_register_position(self) -> None:
        pm, _, _ = _pm()
        _register_long(pm)

        pos = pm.get_position("BTC-USDC")
        assert pos is not None
        assert pos.direction == "long"
        assert pos.entry_price == 65000.0
        assert pos.current_sl == 64000.0
        assert pos.atr == 500.0
        assert pos.tp1_filled is False

    def test_managed_symbols(self) -> None:
        pm, _, _ = _pm()
        _register_long(pm)
        pm.register_position("ETH-USDC", "short", 3000.0, 3100.0, 50.0)

        assert sorted(pm.managed_symbols) == ["BTC-USDC", "ETH-USDC"]

    def test_remove_position(self) -> None:
        pm, _, _ = _pm()
        _register_long(pm)
        pm.remove_position("BTC-USDC")

        assert pm.get_position("BTC-USDC") is None
        assert pm.managed_symbols == []

    def test_remove_nonexistent_no_error(self) -> None:
        pm, _, _ = _pm()
        pm.remove_position("NONEXISTENT")

    def test_mark_tp1_filled(self) -> None:
        pm, _, _ = _pm()
        _register_long(pm)
        pm.mark_tp1_filled("BTC-USDC")

        assert pm.get_position("BTC-USDC").tp1_filled is True


# ---------------------------------------------------------------------------
# _is_tighter: the most critical invariant
# ---------------------------------------------------------------------------

class TestIsTighter:

    def test_long_higher_sl_is_tighter(self) -> None:
        pos = ManagedPosition("BTC", "long", 65000, 64000, 500)
        assert PositionManager._is_tighter(pos, 64500.0) is True

    def test_long_lower_sl_is_not_tighter(self) -> None:
        pos = ManagedPosition("BTC", "long", 65000, 64000, 500)
        assert PositionManager._is_tighter(pos, 63500.0) is False

    def test_long_same_sl_is_not_tighter(self) -> None:
        pos = ManagedPosition("BTC", "long", 65000, 64000, 500)
        assert PositionManager._is_tighter(pos, 64000.0) is False

    def test_short_lower_sl_is_tighter(self) -> None:
        pos = ManagedPosition("BTC", "short", 65000, 66000, 500)
        assert PositionManager._is_tighter(pos, 65500.0) is True

    def test_short_higher_sl_is_not_tighter(self) -> None:
        pos = ManagedPosition("BTC", "short", 65000, 66000, 500)
        assert PositionManager._is_tighter(pos, 66500.0) is False

    def test_short_same_sl_is_not_tighter(self) -> None:
        pos = ManagedPosition("BTC", "short", 65000, 66000, 500)
        assert PositionManager._is_tighter(pos, 66000.0) is False


# ---------------------------------------------------------------------------
# Break-even after TP1
# ---------------------------------------------------------------------------

class TestBreakeven:

    @pytest.mark.asyncio
    async def test_long_breakeven_after_tp1(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=2000.0)
        pm.mark_tp1_filled("BTC-USDC")

        new_sl = await pm.check_adjustments("BTC-USDC", 66000.0)

        assert new_sl == 65000.0
        assert pm.get_position("BTC-USDC").current_sl == 65000.0
        assert adapter.modify_sl_calls == [("BTC-USDC", 65000.0)]

    @pytest.mark.asyncio
    async def test_short_breakeven_after_tp1(self) -> None:
        pm, _, adapter = _pm()
        _register_short(pm, entry=65000.0, sl=66000.0, atr=2000.0)
        pm.mark_tp1_filled("BTC-USDC")

        new_sl = await pm.check_adjustments("BTC-USDC", 64000.0)

        assert new_sl == 65000.0
        assert adapter.modify_sl_calls == [("BTC-USDC", 65000.0)]

    @pytest.mark.asyncio
    async def test_no_breakeven_without_tp1(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=2000.0)

        await pm.check_adjustments("BTC-USDC", 65500.0)

        pos = pm.get_position("BTC-USDC")
        assert pos.current_sl != 65000.0
        # No modify_sl since no trailing either (move < ATR)
        assert len(adapter.modify_sl_calls) == 0


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

class TestTrailingStop:

    @pytest.mark.asyncio
    async def test_long_trailing_after_1_atr_move(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 66000.0)

        assert new_sl == 65500.0
        assert adapter.modify_sl_calls == [("BTC-USDC", 65500.0)]

    @pytest.mark.asyncio
    async def test_long_no_trailing_below_1_atr(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 65400.0)

        assert new_sl is None
        assert len(adapter.modify_sl_calls) == 0

    @pytest.mark.asyncio
    async def test_short_trailing_after_1_atr_move(self) -> None:
        pm, _, adapter = _pm()
        _register_short(pm, entry=65000.0, sl=66000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 64000.0)

        assert new_sl == 64500.0
        assert adapter.modify_sl_calls == [("BTC-USDC", 64500.0)]

    @pytest.mark.asyncio
    async def test_trailing_never_widens_long(self) -> None:
        """Price retraces after tightening -> SL must stay, no modify_sl."""
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        # First: price moves up -> trailing tightens
        await pm.check_adjustments("BTC-USDC", 66000.0)
        assert pm.get_position("BTC-USDC").current_sl == 65500.0
        assert len(adapter.modify_sl_calls) == 1

        # Second: price retraces -> SL must NOT change
        result = await pm.check_adjustments("BTC-USDC", 65600.0)
        assert result is None
        assert pm.get_position("BTC-USDC").current_sl == 65500.0
        assert len(adapter.modify_sl_calls) == 1  # no additional call

    @pytest.mark.asyncio
    async def test_trailing_never_widens_short(self) -> None:
        pm, _, adapter = _pm()
        _register_short(pm, entry=65000.0, sl=66000.0, atr=500.0)

        await pm.check_adjustments("BTC-USDC", 64000.0)
        assert pm.get_position("BTC-USDC").current_sl == 64500.0

        result = await pm.check_adjustments("BTC-USDC", 64400.0)
        assert result is None
        assert pm.get_position("BTC-USDC").current_sl == 64500.0
        assert len(adapter.modify_sl_calls) == 1

    @pytest.mark.asyncio
    async def test_trailing_would_widen_so_no_modify_sl(self) -> None:
        """If trailing computes a LOWER SL for long, it's rejected."""
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=65200.0, atr=500.0)

        # trail = 65600 - 500 = 65100, but current SL is 65200 -> not tighter
        result = await pm.check_adjustments("BTC-USDC", 65600.0)
        assert result is None
        assert len(adapter.modify_sl_calls) == 0


# ---------------------------------------------------------------------------
# Funding rate tighten
# ---------------------------------------------------------------------------

class TestFundingTighten:

    @pytest.mark.asyncio
    async def test_long_tightens_on_positive_funding(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 65000.0, funding_rate=0.0005)

        assert new_sl == pytest.approx(64150.0)
        assert adapter.modify_sl_calls == [("BTC-USDC", pytest.approx(64150.0))]

    @pytest.mark.asyncio
    async def test_short_tightens_on_negative_funding(self) -> None:
        pm, _, adapter = _pm()
        _register_short(pm, entry=65000.0, sl=66000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 65000.0, funding_rate=-0.0005)

        assert new_sl == pytest.approx(65850.0)
        assert adapter.modify_sl_calls == [("BTC-USDC", pytest.approx(65850.0))]

    @pytest.mark.asyncio
    async def test_long_no_tighten_on_negative_funding(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 65000.0, funding_rate=-0.0005)

        assert new_sl is None
        assert len(adapter.modify_sl_calls) == 0

    @pytest.mark.asyncio
    async def test_no_tighten_on_neutral_funding(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 65000.0, funding_rate=0.00005)

        assert new_sl is None
        assert len(adapter.modify_sl_calls) == 0

    @pytest.mark.asyncio
    async def test_no_tighten_on_none_funding(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 65000.0, funding_rate=None)

        assert new_sl is None
        assert len(adapter.modify_sl_calls) == 0


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------

class TestPositionManagerEvents:

    @pytest.mark.asyncio
    async def test_emits_position_updated_on_sl_change(self) -> None:
        pm, bus, _ = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        events: list[PositionUpdated] = []
        bus.subscribe(PositionUpdated, lambda e: events.append(e))

        await pm.check_adjustments("BTC-USDC", 66000.0)

        assert len(events) == 1
        assert events[0].symbol == "BTC-USDC"

    @pytest.mark.asyncio
    async def test_no_event_when_no_change(self) -> None:
        pm, bus, _ = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        events: list[PositionUpdated] = []
        bus.subscribe(PositionUpdated, lambda e: events.append(e))

        await pm.check_adjustments("BTC-USDC", 65200.0)

        assert len(events) == 0


# ---------------------------------------------------------------------------
# Combined / priority / invariants
# ---------------------------------------------------------------------------

class TestPositionManagerCombined:

    @pytest.mark.asyncio
    async def test_trailing_beats_funding_when_tighter(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        new_sl = await pm.check_adjustments("BTC-USDC", 66500.0, funding_rate=0.0005)

        assert new_sl == 66000.0  # trailing wins
        assert adapter.modify_sl_calls == [("BTC-USDC", 66000.0)]

    @pytest.mark.asyncio
    async def test_breakeven_and_trailing_take_tightest(self) -> None:
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)
        pm.mark_tp1_filled("BTC-USDC")

        # BE: 65000, trailing: 66000-500=65500 -> trailing is tighter
        new_sl = await pm.check_adjustments("BTC-USDC", 66000.0)

        assert new_sl == 65500.0
        assert adapter.modify_sl_calls == [("BTC-USDC", 65500.0)]

    @pytest.mark.asyncio
    async def test_nonexistent_symbol_returns_none(self) -> None:
        pm, _, adapter = _pm()
        new_sl = await pm.check_adjustments("NONEXISTENT", 65000.0)
        assert new_sl is None
        assert len(adapter.modify_sl_calls) == 0

    @pytest.mark.asyncio
    async def test_never_widens_long_sl_multi_check(self) -> None:
        """Run multiple price scenarios — SL must only go up or stay same."""
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64500.0, atr=500.0)

        for price in [65100, 65300, 65000, 64800, 65600, 66000, 65200]:
            await pm.check_adjustments("BTC-USDC", float(price), funding_rate=0.0005)
            pos = pm.get_position("BTC-USDC")
            assert pos.current_sl >= 64500.0

    @pytest.mark.asyncio
    async def test_never_widens_short_sl_multi_check(self) -> None:
        """Run multiple price scenarios — SL must only go down or stay same."""
        pm, _, adapter = _pm()
        _register_short(pm, entry=65000.0, sl=65500.0, atr=500.0)

        for price in [64900, 64700, 65100, 64000, 63500, 64200]:
            await pm.check_adjustments("BTC-USDC", float(price), funding_rate=-0.0005)
            pos = pm.get_position("BTC-USDC")
            assert pos.current_sl <= 65500.0

    @pytest.mark.asyncio
    async def test_modify_sl_call_count_matches_adjustments(self) -> None:
        """modify_sl is called exactly once per actual SL change."""
        pm, _, adapter = _pm()
        _register_long(pm, entry=65000.0, sl=64000.0, atr=500.0)

        # No change
        await pm.check_adjustments("BTC-USDC", 65200.0)
        assert len(adapter.modify_sl_calls) == 0

        # Trailing triggers
        await pm.check_adjustments("BTC-USDC", 66000.0)
        assert len(adapter.modify_sl_calls) == 1

        # Price continues up -> another tighten
        await pm.check_adjustments("BTC-USDC", 67000.0)
        assert len(adapter.modify_sl_calls) == 2

        # Price retraces -> no change
        await pm.check_adjustments("BTC-USDC", 66200.0)
        assert len(adapter.modify_sl_calls) == 2  # still 2
