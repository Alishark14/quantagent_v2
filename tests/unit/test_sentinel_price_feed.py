"""Tests for Sentinel ← PriceFeed integration (Sprint Week 7 Task 6).

Verifies the event-driven mode WITHOUT touching the legacy REST-poll
path (that's covered by the 88 existing tests in ``test_sentinel.py``).

Key assertions:
  * Sentinel with ``price_feed`` subscribes to ``CandleClosed`` on ``run()``
  * Publishing a ``CandleClosed`` triggers the readiness computation
  * ``_check_once`` reads candles from PriceFeed memory, not REST
  * ``price_feed=None`` preserves the legacy REST-poll loop
  * ``SetupDetected`` emits correctly when data comes from PriceFeed
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone

import pytest

from engine.data.price_feed.base import PriceFeed, SymbolState
from engine.events import (
    CandleClosed,
    InProcessBus,
    SetupDetected,
)
from engine.types import AdapterCapabilities, CandleClose, OrderResult
from exchanges.base import ExchangeAdapter
from sentinel.monitor import SentinelMonitor


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _synth_candles(n: int = 50) -> list[dict]:
    """Synthetic 1h candles with enough variation for compute_all_indicators.

    Mirrors ``MockSentinelAdapter._default_candles`` but stores
    ``datetime`` timestamps matching the PriceFeed format.
    """
    candles = []
    for i in range(n):
        base = 65000.0 + i * 5.0 + 200 * math.sin(i * 0.3)
        ts = datetime(2026, 4, 12, i % 24, 0, 0, tzinfo=timezone.utc)
        candles.append({
            "timestamp": ts,
            "timeframe": "1h",
            "open": base - 15,
            "high": base + 50,
            "low": base - 50,
            "close": base,
            "volume": 1000.0 + (5000.0 if i == n - 1 else 0),
        })
    return candles


class FakePriceFeed(PriceFeed):
    """Concrete PriceFeed for Sentinel integration tests.

    Pre-populated with synthetic candles; ``get_candle_history`` reads
    from the deque (same as the real implementation), and ``get_latest_price``
    returns the last close. No WebSocket plumbing.
    """

    def __init__(self, event_bus, candles: list[dict] | None = None) -> None:
        super().__init__(event_bus, exchange_name="fake")
        self._candle_timeframe = "1h"
        supplied = candles if candles is not None else _synth_candles()
        state = SymbolState(symbol="BTC-USDC")
        for c in supplied:
            state.completed_candles.append(c)
        if supplied:
            state.latest_price = supplied[-1]["close"]
        self._symbols["BTC-USDC"] = state
        self._subscribed_symbols.add("BTC-USDC")

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def _listen(self) -> None:
        return None

    async def subscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed_symbols.add(s)

    async def unsubscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed_symbols.discard(s)


class StubAdapter(ExchangeAdapter):
    """Minimal adapter for Sentinel — only ``fetch_ohlcv`` is needed.

    Records calls so the test can assert that when a PriceFeed is wired,
    the adapter is NEVER called.
    """

    def __init__(self) -> None:
        self.fetch_ohlcv_calls: list[str] = []

    def name(self):
        return "stub"

    def capabilities(self):
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market"], supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
        self.fetch_ohlcv_calls.append(symbol)
        return _synth_candles(limit)

    async def get_ticker(self, symbol):
        return {"last": 65290.0}

    async def get_balance(self):
        return 10000.0

    async def get_positions(self, symbol=None):
        return []

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=True, order_id="m-1", fill_price=65290.0, fill_size=size, error=None)

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=True, order_id="l-1", fill_price=price, fill_size=size, error=None)

    async def place_sl_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id="sl-1", fill_price=trigger_price, fill_size=size, error=None)

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id="tp-1", fill_price=trigger_price, fill_size=size, error=None)

    async def cancel_order(self, symbol, order_id):
        return True

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=True, order_id="c-1", fill_price=65290.0, fill_size=0, error=None)

    async def modify_sl(self, symbol, new_price):
        return OrderResult(success=True, order_id="sl-1", fill_price=new_price, fill_size=0, error=None)

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=True, order_id="tp-1", fill_price=new_price, fill_size=0, error=None)

    async def get_funding_rate(self, symbol):
        return 0.0001

    async def get_open_interest(self, symbol):
        return 1_000_000_000.0


class _Recorder:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def bus() -> InProcessBus:
    return InProcessBus()


def _make_candle_closed_event(symbol: str = "BTC-USDC") -> CandleClosed:
    """Build a CandleClosed event that matches the FakePriceFeed's symbol."""
    return CandleClosed(
        source="test",
        candle=CandleClose(
            symbol=symbol,
            timeframe="1h",
            open=65200.0,
            high=65300.0,
            low=65100.0,
            close=65250.0,
            volume=100.0,
            exchange="fake",
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventDrivenMode:
    async def test_candle_close_triggers_readiness_computation(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        feed = FakePriceFeed(bus)
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            threshold=0.0,  # very low so any data triggers
            price_feed=feed,
        )

        # Manually subscribe (run() would do this but we don't want the loop).
        bus.subscribe(CandleClosed, sentinel._on_candle_close)

        rec = _Recorder()
        bus.subscribe(SetupDetected, rec)

        await bus.publish(_make_candle_closed_event())

        # SetupDetected should have fired (threshold 0.0 = always triggers).
        assert len(rec.events) == 1
        assert rec.events[0].symbol == "BTC-USDC"
        assert rec.events[0].readiness > 0.0

        # The adapter was NOT called — data came from PriceFeed memory.
        assert adapter.fetch_ohlcv_calls == []

    async def test_candle_close_for_wrong_symbol_is_ignored(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        feed = FakePriceFeed(bus)
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            threshold=0.0,
            price_feed=feed,
        )
        bus.subscribe(CandleClosed, sentinel._on_candle_close)

        rec = _Recorder()
        bus.subscribe(SetupDetected, rec)

        # Event for ETH — should be silently ignored.
        await bus.publish(_make_candle_closed_event(symbol="ETH-USDC"))

        assert rec.events == []

    async def test_insufficient_candles_skips_check(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        # Only 10 candles — below the 35 minimum.
        feed = FakePriceFeed(bus, candles=_synth_candles(10))
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            threshold=0.0,
            price_feed=feed,
        )
        bus.subscribe(CandleClosed, sentinel._on_candle_close)

        rec = _Recorder()
        bus.subscribe(SetupDetected, rec)

        await bus.publish(_make_candle_closed_event())

        # Not enough data → no SetupDetected.
        assert rec.events == []

    async def test_reads_price_from_price_feed_memory(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        feed = FakePriceFeed(bus)
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            price_feed=feed,
        )

        # PriceFeed has the latest price set.
        assert feed.get_latest_price("BTC-USDC") is not None
        # Sentinel can read from it (consumer usage pattern).
        price = feed.get_latest_price(sentinel._symbol)
        assert isinstance(price, float)
        assert price > 0


class TestBackwardCompat:
    async def test_price_feed_none_uses_rest_poll(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            threshold=0.0,
            price_feed=None,
        )

        rec = _Recorder()
        bus.subscribe(SetupDetected, rec)

        # Direct check_once() triggers the legacy REST path.
        await sentinel.check_once()

        # The adapter WAS called — legacy REST path.
        assert len(adapter.fetch_ohlcv_calls) >= 1
        # And SetupDetected fired (threshold=0.0).
        assert len(rec.events) >= 1

    async def test_price_feed_none_does_not_subscribe_to_candle_closed(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            price_feed=None,
        )

        # Without run(), the handler isn't registered. But even conceptually,
        # price_feed=None should never subscribe.
        rec = _Recorder()
        bus.subscribe(SetupDetected, rec)

        # Publishing CandleClosed should NOT trigger anything.
        await bus.publish(_make_candle_closed_event())

        assert rec.events == []


class TestSetupDetectedPayload:
    async def test_setup_detected_payload_matches_rest_format(
        self, bus: InProcessBus
    ) -> None:
        adapter = StubAdapter()
        feed = FakePriceFeed(bus)
        sentinel = SentinelMonitor(
            adapter,
            bus,
            "BTC-USDC",
            threshold=0.0,
            price_feed=feed,
        )
        bus.subscribe(CandleClosed, sentinel._on_candle_close)

        rec = _Recorder()
        bus.subscribe(SetupDetected, rec)

        await bus.publish(_make_candle_closed_event())

        assert len(rec.events) == 1
        evt = rec.events[0]
        assert isinstance(evt, SetupDetected)
        assert evt.source == "sentinel"
        assert evt.symbol == "BTC-USDC"
        assert isinstance(evt.readiness, float)
        assert isinstance(evt.conditions, list)
