"""Unit tests for RESTFallbackManager.

Sprint Week 7 Task 4. Uses a `FakePriceFeed` subclass of the ABC and a
`FakeAdapter` so the test never touches a real WebSocket or REST API.

Coverage:
  * starts in CONNECTED when WebSocket connect succeeds
  * starts in FALLBACK when initial connect raises
  * RECONNECTING is recorded in state history when ws disconnect callback fires
  * transitions to FALLBACK after the disconnect callback (max retries
    already exhausted by the feed)
  * REST polling emits PriceUpdated / FundingUpdated / OpenInterestUpdated
    via the wrapped PriceFeed's helpers (consumers can't tell WS vs REST)
  * REST polling fires CandleClosed exactly once per actual new candle
  * recovery: WebSocket comes back during fallback → state CONNECTED
  * health metrics increment on each transition
  * stop() cleanly cancels the REST poll task and disconnects the feed
"""

from __future__ import annotations

import asyncio

import pytest

from engine.data.price_feed.base import PriceFeed, SymbolState
from engine.data.price_feed.fallback import ConnectionState, RESTFallbackManager
from engine.events import (
    CandleClosed,
    FundingUpdated,
    InProcessBus,
    OpenInterestUpdated,
    PriceUpdated,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePriceFeed(PriceFeed):
    """Concrete PriceFeed stand-in with no WebSocket plumbing.

    ``connect_should_fail`` toggles the next ``connect()`` call between
    success and a synthetic ``RuntimeError``. ``connect_calls`` counts
    invocations so tests can assert the manager retried.
    """

    def __init__(self, event_bus) -> None:
        super().__init__(event_bus, exchange_name="hyperliquid")
        self._candle_timeframe = "1h"
        self.connect_should_fail = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.subscribe_calls: list[list[str]] = []

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_should_fail:
            raise RuntimeError("synthetic ws connect failure")
        self._connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    async def _listen(self) -> None:
        return None

    async def subscribe(self, symbols: list[str]) -> None:
        self.subscribe_calls.append(list(symbols))
        for s in symbols:
            if s not in self._symbols:
                self._symbols[s] = SymbolState(symbol=s)
            self._subscribed_symbols.add(s)

    async def unsubscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._symbols.pop(s, None)
            self._subscribed_symbols.discard(s)


class FakeAdapter:
    """Minimal ExchangeAdapter substitute for fallback poll tests."""

    def __init__(
        self,
        ohlcv_responses: dict | None = None,
        funding_responses: dict | None = None,
        oi_responses: dict | None = None,
    ) -> None:
        self.ohlcv_responses = ohlcv_responses or {}
        self.funding_responses = funding_responses or {}
        self.oi_responses = oi_responses or {}
        self.ohlcv_calls: list[dict] = []
        self.funding_calls: list[str] = []
        self.oi_calls: list[str] = []

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100, since=None
    ) -> list[dict]:
        self.ohlcv_calls.append(
            {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        )
        result = self.ohlcv_responses.get(symbol)
        if isinstance(result, Exception):
            raise result
        return result or []

    async def get_funding_rate(self, symbol: str):
        self.funding_calls.append(symbol)
        result = self.funding_responses.get(symbol)
        if isinstance(result, Exception):
            raise result
        return result

    async def get_open_interest(self, symbol: str):
        self.oi_calls.append(symbol)
        result = self.oi_responses.get(symbol)
        if isinstance(result, Exception):
            raise result
        return result


class _Recorder:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> InProcessBus:
    return InProcessBus()


@pytest.fixture
def feed(bus: InProcessBus) -> FakePriceFeed:
    return FakePriceFeed(bus)


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


def make_manager(
    bus: InProcessBus,
    feed: FakePriceFeed,
    adapter: FakeAdapter,
    *,
    max_reconnect_attempts: int = 5,
    rest_poll_interval: float = 0.01,
    ws_recovery_check_interval: float = 0.0,
) -> RESTFallbackManager:
    return RESTFallbackManager(
        price_feed=feed,
        adapter=adapter,
        event_bus=bus,
        max_reconnect_attempts=max_reconnect_attempts,
        rest_poll_interval=rest_poll_interval,
        ws_recovery_check_interval=ws_recovery_check_interval,
    )


# ---------------------------------------------------------------------------
# Start / state machine
# ---------------------------------------------------------------------------


class TestStart:
    async def test_starts_connected_when_ws_ok(
        self, bus: InProcessBus, feed: FakePriceFeed, adapter: FakeAdapter
    ) -> None:
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC", "ETH-USDC"])

        assert mgr.state == ConnectionState.CONNECTED
        assert feed.connect_calls == 1
        assert feed.subscribe_calls == [["BTC-USDC", "ETH-USDC"]]
        # Manager should have wired the disconnect callback + max retries.
        assert feed._on_disconnect_callback is not None
        assert feed._max_reconnect_attempts == 5

        await mgr.stop()

    async def test_initial_connect_failure_enters_fallback(
        self, bus: InProcessBus, feed: FakePriceFeed
    ) -> None:
        feed.connect_should_fail = True
        adapter = FakeAdapter(
            ohlcv_responses={
                "BTC-USDC": [
                    {
                        "timestamp": 1_712_000_000_000,
                        "open": 65000,
                        "high": 65100,
                        "low": 64900,
                        "close": 65050,
                        "volume": 5,
                    }
                ]
            },
            funding_responses={"BTC-USDC": 0.0001},
            oi_responses={"BTC-USDC": 1_000_000_000},
        )
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC"])

        assert mgr.state == ConnectionState.FALLBACK
        assert mgr.get_health()["fallback_activations"] == 1
        await mgr.stop()


class TestDisconnectCallback:
    async def test_disconnect_callback_records_reconnecting_then_fallback(
        self, bus: InProcessBus, feed: FakePriceFeed, adapter: FakeAdapter
    ) -> None:
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC"])
        assert mgr.state == ConnectionState.CONNECTED

        # Simulate the feed exhausting its retries — it would call this.
        await feed._fire_on_disconnect()

        assert ConnectionState.RECONNECTING in mgr.state_history
        assert mgr.state == ConnectionState.FALLBACK
        assert mgr.get_health()["reconnect_count"] == 1
        assert mgr.get_health()["fallback_activations"] == 1

        await mgr.stop()


# ---------------------------------------------------------------------------
# REST polling
# ---------------------------------------------------------------------------


class TestRestPolling:
    async def test_rest_poll_emits_events_via_price_feed_helpers(
        self, bus: InProcessBus, feed: FakePriceFeed
    ) -> None:
        price_rec = _Recorder()
        funding_rec = _Recorder()
        oi_rec = _Recorder()
        candle_rec = _Recorder()
        bus.subscribe(PriceUpdated, price_rec)
        bus.subscribe(FundingUpdated, funding_rec)
        bus.subscribe(OpenInterestUpdated, oi_rec)
        bus.subscribe(CandleClosed, candle_rec)

        adapter = FakeAdapter(
            ohlcv_responses={
                "BTC-USDC": [
                    {
                        "timestamp": 1_712_000_000_000,
                        "open": 65000,
                        "high": 65100,
                        "low": 64900,
                        "close": 65050,
                        "volume": 5,
                    }
                ]
            },
            funding_responses={"BTC-USDC": 0.0002},
            oi_responses={"BTC-USDC": 1_500_000_000},
        )

        # Force the manager into fallback by failing the initial WS connect.
        feed.connect_should_fail = True
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC"])
        assert mgr.state == ConnectionState.FALLBACK

        # Give the poll loop a couple of cycles to fire.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if (
                price_rec.events
                and funding_rec.events
                and oi_rec.events
                and candle_rec.events
            ):
                break

        assert len(price_rec.events) >= 1
        assert price_rec.events[0].update.price == 65050.0
        assert price_rec.events[0].update.exchange == "hyperliquid"

        assert len(funding_rec.events) >= 1
        assert funding_rec.events[0].update.funding_rate == 0.0002

        assert len(oi_rec.events) >= 1
        assert oi_rec.events[0].update.open_interest == 1_500_000_000.0

        # Exactly one CandleClosed event for that one new candle, even
        # though the poll loop ran multiple iterations.
        assert len(candle_rec.events) == 1
        assert candle_rec.events[0].candle.close == 65050.0

        await mgr.stop()


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    async def test_ws_recovery_transitions_back_to_connected(
        self, bus: InProcessBus, feed: FakePriceFeed
    ) -> None:
        adapter = FakeAdapter(
            ohlcv_responses={"BTC-USDC": []},  # don't care
            funding_responses={"BTC-USDC": None},
            oi_responses={"BTC-USDC": None},
        )

        # First connect fails → enter fallback.
        feed.connect_should_fail = True
        mgr = make_manager(
            bus,
            feed,
            adapter,
            rest_poll_interval=0.01,
            ws_recovery_check_interval=0.0,  # probe every loop iteration
        )
        await mgr.start(["BTC-USDC"])
        assert mgr.state == ConnectionState.FALLBACK
        initial_connect_calls = feed.connect_calls

        # Now let WS recover — next connect succeeds.
        feed.connect_should_fail = False

        for _ in range(50):
            await asyncio.sleep(0.01)
            if mgr.state == ConnectionState.CONNECTED:
                break

        assert mgr.state == ConnectionState.CONNECTED
        assert feed.connect_calls > initial_connect_calls
        # Recovery counts as a reconnect.
        assert mgr.get_health()["reconnect_count"] >= 1
        await mgr.stop()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_health_metrics_track_each_transition(
        self, bus: InProcessBus, feed: FakePriceFeed, adapter: FakeAdapter
    ) -> None:
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC"])

        h = mgr.get_health()
        assert h["fallback_activations"] == 0
        assert h["reconnect_count"] == 0
        assert h["ws_uptime_seconds"] >= 0.0

        # Trigger a fallback via the disconnect callback.
        await feed._fire_on_disconnect()
        h2 = mgr.get_health()
        assert h2["fallback_activations"] == 1
        assert h2["reconnect_count"] == 1
        # Once we're in FALLBACK, ws_uptime is frozen at 0 (the spec field
        # doesn't accumulate across cycles for this minimal implementation).
        assert h2["ws_uptime_seconds"] == 0.0

        await mgr.stop()


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


class TestStop:
    async def test_stop_disconnects_feed_and_cancels_poll_task(
        self, bus: InProcessBus, feed: FakePriceFeed, adapter: FakeAdapter
    ) -> None:
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC"])
        # Force into fallback so the poll task is running.
        await feed._fire_on_disconnect()
        assert mgr._rest_poll_task is not None
        assert not mgr._rest_poll_task.done()

        await mgr.stop()

        assert mgr.state == ConnectionState.DISCONNECTED
        assert feed.disconnect_calls == 1
        # Poll task should be either done or cancelled.
        assert mgr._rest_poll_task is None
        # Disconnect callback was cleared so a stray fire is a no-op.
        assert feed._on_disconnect_callback is None

    async def test_stop_idempotent(
        self, bus: InProcessBus, feed: FakePriceFeed, adapter: FakeAdapter
    ) -> None:
        mgr = make_manager(bus, feed, adapter)
        await mgr.start(["BTC-USDC"])
        await mgr.stop()
        # Second stop must not raise.
        await mgr.stop()
        assert mgr.state == ConnectionState.DISCONNECTED
