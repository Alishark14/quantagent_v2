"""Unit tests for engine/data/price_feed/base.py.

Sprint Week 7 Task 2. Exercises the PriceFeed ABC's state container
and helper methods via a concrete StubPriceFeed that doesn't touch any
network. Verifies that every ``_update_*`` method both mutates the
``SymbolState`` and emits the matching event on the bus.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engine.data.price_feed.base import PriceFeed, SymbolState
from engine.events import (
    CandleClosed,
    FundingUpdated,
    InProcessBus,
    OpenInterestUpdated,
    PriceUpdated,
)


# ---------------------------------------------------------------------------
# StubPriceFeed — concrete subclass that stubs out all network plumbing
# ---------------------------------------------------------------------------


class StubPriceFeed(PriceFeed):
    """Minimal concrete PriceFeed for unit tests.

    ``subscribe()`` seeds a ``SymbolState`` entry. ``connect`` /
    ``disconnect`` / ``_listen`` / ``unsubscribe`` are no-ops —
    tests call ``_update_*`` directly.
    """

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def _listen(self) -> None:
        return None

    async def subscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            if s not in self._symbols:
                self._symbols[s] = SymbolState(symbol=s)
            self._subscribed_symbols.add(s)

    async def unsubscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed_symbols.discard(s)
            self._symbols.pop(s, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> InProcessBus:
    return InProcessBus()


@pytest.fixture
async def feed(bus: InProcessBus) -> StubPriceFeed:
    f = StubPriceFeed(event_bus=bus, exchange_name="stub")
    await f.subscribe(["BTC-USDC", "ETH-USDC"])
    return f


class _Recorder:
    """Handler helper that captures every event passed to it."""

    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# _update_price
# ---------------------------------------------------------------------------


class TestUpdatePrice:
    async def test_mutates_symbol_state(self, feed: StubPriceFeed) -> None:
        ts = datetime(2026, 4, 12, 10, 0, 0, tzinfo=timezone.utc)
        await feed._update_price(
            "BTC-USDC", 65000.0, bid=64999.5, ask=65000.5, size=0.1, timestamp=ts
        )
        state = feed.get_symbol_state("BTC-USDC")
        assert state is not None
        assert state.latest_price == 65000.0
        assert state.latest_bid == 64999.5
        assert state.latest_ask == 65000.5
        assert state.last_update == ts

    async def test_emits_price_updated(self, feed: StubPriceFeed, bus: InProcessBus) -> None:
        rec = _Recorder()
        bus.subscribe(PriceUpdated, rec)
        await feed._update_price("BTC-USDC", 65000.0, bid=64999.0, ask=65001.0)
        assert len(rec.events) == 1
        event = rec.events[0]
        assert isinstance(event, PriceUpdated)
        assert event.update.symbol == "BTC-USDC"
        assert event.update.price == 65000.0
        assert event.update.bid == 64999.0
        assert event.update.ask == 65001.0
        assert event.update.exchange == "stub"
        assert event.source == "price_feed:stub"

    async def test_lazy_state_for_unsubscribed_symbol(
        self, feed: StubPriceFeed, bus: InProcessBus
    ) -> None:
        """A tick for an unsubscribed symbol still lands — never drop data."""
        rec = _Recorder()
        bus.subscribe(PriceUpdated, rec)
        await feed._update_price("SOL-USDC", 142.0)
        assert feed.get_latest_price("SOL-USDC") == 142.0
        assert len(rec.events) == 1

    async def test_partial_update_keeps_previous_bid_ask(self, feed: StubPriceFeed) -> None:
        await feed._update_price("BTC-USDC", 65000.0, bid=64999.5, ask=65000.5)
        await feed._update_price("BTC-USDC", 65010.0)  # no bid/ask this tick
        state = feed.get_symbol_state("BTC-USDC")
        assert state.latest_price == 65010.0
        assert state.latest_bid == 64999.5
        assert state.latest_ask == 65000.5


# ---------------------------------------------------------------------------
# _complete_candle
# ---------------------------------------------------------------------------


class TestCompleteCandle:
    async def test_appends_to_deque(self, feed: StubPriceFeed) -> None:
        ts = datetime(2026, 4, 12, 10, 0, 0, tzinfo=timezone.utc)
        await feed._complete_candle(
            "BTC-USDC",
            timeframe="1h",
            open_price=65000.0,
            high=65500.0,
            low=64800.0,
            close=65400.0,
            volume=123.45,
            timestamp=ts,
        )
        state = feed.get_symbol_state("BTC-USDC")
        assert len(state.completed_candles) == 1
        candle = state.completed_candles[-1]
        assert candle["open"] == 65000.0
        assert candle["high"] == 65500.0
        assert candle["low"] == 64800.0
        assert candle["close"] == 65400.0
        assert candle["volume"] == 123.45
        assert candle["timeframe"] == "1h"
        assert candle["timestamp"] == ts
        assert state.building_candle is None

    async def test_emits_candle_closed(
        self, feed: StubPriceFeed, bus: InProcessBus
    ) -> None:
        rec = _Recorder()
        bus.subscribe(CandleClosed, rec)
        await feed._complete_candle(
            "BTC-USDC",
            timeframe="1h",
            open_price=65000.0,
            high=65500.0,
            low=64800.0,
            close=65400.0,
            volume=10.0,
        )
        assert len(rec.events) == 1
        event = rec.events[0]
        assert isinstance(event, CandleClosed)
        assert event.candle.symbol == "BTC-USDC"
        assert event.candle.timeframe == "1h"
        assert event.candle.close == 65400.0
        assert event.candle.exchange == "stub"

    async def test_deque_respects_maxlen(self, feed: StubPriceFeed) -> None:
        # Maxlen is 500 — push 600 candles and assert only the last 500 survive.
        for i in range(600):
            await feed._complete_candle(
                "BTC-USDC",
                timeframe="1m",
                open_price=float(i),
                high=float(i) + 1,
                low=float(i) - 1,
                close=float(i),
                volume=1.0,
            )
        state = feed.get_symbol_state("BTC-USDC")
        assert len(state.completed_candles) == 500
        # Oldest surviving candle should be index 100 (600 - 500).
        assert state.completed_candles[0]["open"] == 100.0
        assert state.completed_candles[-1]["open"] == 599.0

    async def test_candle_close_syncs_latest_price_if_empty(
        self, feed: StubPriceFeed
    ) -> None:
        """Candle close should set latest_price if no tick has arrived yet."""
        await feed._complete_candle(
            "BTC-USDC", timeframe="1h", open_price=1, high=2, low=0.5, close=1.5, volume=1
        )
        assert feed.get_latest_price("BTC-USDC") == 1.5

    async def test_candle_close_does_not_overwrite_existing_price(
        self, feed: StubPriceFeed
    ) -> None:
        await feed._update_price("BTC-USDC", 65000.0)
        await feed._complete_candle(
            "BTC-USDC",
            timeframe="1h",
            open_price=64000,
            high=65200,
            low=63900,
            close=64500,
            volume=1,
        )
        # Candle close is historical — don't rewind the live price.
        assert feed.get_latest_price("BTC-USDC") == 65000.0


# ---------------------------------------------------------------------------
# _update_funding / _update_oi
# ---------------------------------------------------------------------------


class TestFundingAndOIUpdates:
    async def test_funding_update_mutates_and_emits(
        self, feed: StubPriceFeed, bus: InProcessBus
    ) -> None:
        rec = _Recorder()
        bus.subscribe(FundingUpdated, rec)
        next_ts = datetime(2026, 4, 12, 16, 0, 0, tzinfo=timezone.utc)
        await feed._update_funding(
            "BTC-USDC", funding_rate=0.0001, next_funding_time=next_ts
        )
        assert feed.get_funding_rate("BTC-USDC") == 0.0001
        assert len(rec.events) == 1
        event = rec.events[0]
        assert isinstance(event, FundingUpdated)
        assert event.update.funding_rate == 0.0001
        assert event.update.next_funding_time == next_ts
        assert event.update.exchange == "stub"

    async def test_oi_update_mutates_and_emits(
        self, feed: StubPriceFeed, bus: InProcessBus
    ) -> None:
        rec = _Recorder()
        bus.subscribe(OpenInterestUpdated, rec)
        await feed._update_oi(
            "BTC-USDC", open_interest=1_500_000_000.0, oi_change_pct=0.0135
        )
        assert feed.get_open_interest("BTC-USDC") == 1_500_000_000.0
        assert len(rec.events) == 1
        event = rec.events[0]
        assert isinstance(event, OpenInterestUpdated)
        assert event.update.open_interest == 1_500_000_000.0
        assert event.update.oi_change_pct == 0.0135


# ---------------------------------------------------------------------------
# Read methods — unknown symbol handling and history slicing
# ---------------------------------------------------------------------------


class TestReads:
    async def test_get_latest_price_unknown_symbol(self, feed: StubPriceFeed) -> None:
        assert feed.get_latest_price("DOGE-USDC") is None

    async def test_get_latest_candle_unknown_symbol(self, feed: StubPriceFeed) -> None:
        assert feed.get_latest_candle("DOGE-USDC") is None

    async def test_get_latest_candle_empty_history(self, feed: StubPriceFeed) -> None:
        assert feed.get_latest_candle("BTC-USDC") is None

    async def test_get_funding_rate_unknown_symbol(self, feed: StubPriceFeed) -> None:
        assert feed.get_funding_rate("DOGE-USDC") is None

    async def test_get_open_interest_unknown_symbol(self, feed: StubPriceFeed) -> None:
        assert feed.get_open_interest("DOGE-USDC") is None

    async def test_get_candle_history_returns_requested_count(
        self, feed: StubPriceFeed
    ) -> None:
        for i in range(10):
            await feed._complete_candle(
                "BTC-USDC",
                timeframe="1h",
                open_price=float(i),
                high=float(i),
                low=float(i),
                close=float(i),
                volume=1.0,
            )
        # Asking for 5 returns the last 5.
        history = feed.get_candle_history("BTC-USDC", count=5)
        assert len(history) == 5
        assert [c["open"] for c in history] == [5.0, 6.0, 7.0, 8.0, 9.0]

    async def test_get_candle_history_count_exceeds_available(
        self, feed: StubPriceFeed
    ) -> None:
        for i in range(3):
            await feed._complete_candle(
                "BTC-USDC",
                timeframe="1h",
                open_price=float(i),
                high=1,
                low=0,
                close=float(i),
                volume=1.0,
            )
        history = feed.get_candle_history("BTC-USDC", count=100)
        assert len(history) == 3

    async def test_get_candle_history_unknown_symbol(self, feed: StubPriceFeed) -> None:
        assert feed.get_candle_history("DOGE-USDC", count=10) == []

    async def test_get_candle_history_zero_count(self, feed: StubPriceFeed) -> None:
        await feed._complete_candle(
            "BTC-USDC", timeframe="1h", open_price=1, high=2, low=0.5, close=1.5, volume=1
        )
        assert feed.get_candle_history("BTC-USDC", count=0) == []

    async def test_get_latest_candle_returns_most_recent(
        self, feed: StubPriceFeed
    ) -> None:
        for i in range(3):
            await feed._complete_candle(
                "BTC-USDC",
                timeframe="1h",
                open_price=float(i),
                high=1,
                low=0,
                close=float(i * 10),
                volume=1.0,
            )
        latest = feed.get_latest_candle("BTC-USDC")
        assert latest is not None
        assert latest["close"] == 20.0


# ---------------------------------------------------------------------------
# Subscription lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_subscribe_initialises_symbol_state(self, bus: InProcessBus) -> None:
        feed = StubPriceFeed(event_bus=bus, exchange_name="stub")
        assert feed.get_symbol_state("BTC-USDC") is None
        await feed.subscribe(["BTC-USDC"])
        state = feed.get_symbol_state("BTC-USDC")
        assert state is not None
        assert state.symbol == "BTC-USDC"
        assert state.latest_price is None
        assert "BTC-USDC" in feed.subscribed_symbols

    async def test_unsubscribe_drops_state(self, feed: StubPriceFeed) -> None:
        await feed._update_price("BTC-USDC", 65000.0)
        await feed.unsubscribe(["BTC-USDC"])
        assert feed.get_symbol_state("BTC-USDC") is None
        assert "BTC-USDC" not in feed.subscribed_symbols

    async def test_connect_sets_flag(self, feed: StubPriceFeed) -> None:
        assert feed.is_connected() is False
        await feed.connect()
        assert feed.is_connected() is True
        await feed.disconnect()
        assert feed.is_connected() is False

    async def test_exchange_name_property(self, feed: StubPriceFeed) -> None:
        assert feed.exchange_name == "stub"

    async def test_cannot_instantiate_abstract_base(self, bus: InProcessBus) -> None:
        with pytest.raises(TypeError):
            PriceFeed(event_bus=bus, exchange_name="abstract")  # type: ignore[abstract]
