"""Unit tests for HyperliquidPriceFeed.

Sprint Week 7 Task 3. All tests use a `FakeWebSocket` stub — no real
network calls. Verifies:

  * ``subscribe`` emits the 3 expected Hyperliquid subscription frames
  * ``trades`` messages produce ``PriceUpdated`` with the right fields
  * ``candle`` messages produce ``CandleClosed`` on open-time advance
  * ``activeAssetCtx`` messages produce both ``FundingUpdated`` and
    ``OpenInterestUpdated`` with a computed ``oi_change_pct``
  * Malformed frames are logged and skipped (no crash, no events)
  * Reconnect fires after a ``ConnectionClosed`` and resubscribes
  * Symbol mapping handles native perps and HIP-3 aliases (including
    ``WTIOIL-USDC → xyz:CL`` which a naive split would get wrong)
  * ``unsubscribe`` sends the right frames and drops state
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
from websockets.exceptions import ConnectionClosed

from engine.data.price_feed.hyperliquid import HyperliquidPriceFeed
from engine.events import (
    CandleClosed,
    FundingUpdated,
    InProcessBus,
    OpenInterestUpdated,
    PriceUpdated,
)


# ---------------------------------------------------------------------------
# FakeWebSocket — minimal async-iterable stub
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """In-memory WebSocket substitute.

    ``push(frame)`` queues an incoming frame (str/bytes/dict) for the
    listen loop to consume. ``close()`` sets a sentinel that ends the
    async iteration gracefully. ``fail()`` makes the next ``__anext__``
    raise ``ConnectionClosed`` — use this to exercise reconnect paths.
    ``sent`` records every frame the feed sent out so tests can assert
    subscription-message shape.
    """

    _CLOSE_SENTINEL = object()
    _FAIL_SENTINEL = object()

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.closed: bool = False

    async def send(self, frame: str) -> None:
        if self.closed:
            raise ConnectionClosed(None, None)
        self.sent.append(frame)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self._queue.put(self._CLOSE_SENTINEL)

    def push(self, frame) -> None:
        self._queue.put_nowait(frame)

    def fail(self) -> None:
        self._queue.put_nowait(self._FAIL_SENTINEL)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is self._CLOSE_SENTINEL:
            raise StopAsyncIteration
        if item is self._FAIL_SENTINEL:
            raise ConnectionClosed(None, None)
        return item


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)


@pytest.fixture
def bus() -> InProcessBus:
    return InProcessBus()


def make_feed(bus: InProcessBus, ws: FakeWebSocket) -> HyperliquidPriceFeed:
    """Build a feed bound to a pre-created FakeWebSocket."""

    async def connect_stub(url: str) -> FakeWebSocket:
        return ws

    feed = HyperliquidPriceFeed(
        event_bus=bus, candle_timeframe="1h", ws_connect=connect_stub
    )
    # Keep reconnect delays tight so reconnect tests finish fast.
    feed._reconnect_delay = 0.01
    feed._max_reconnect_delay = 0.01
    return feed


async def drain_events(iterations: int = 5) -> None:
    """Yield control so the listen task can process queued frames."""
    for _ in range(iterations):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------


class TestResolveCoin:
    def test_native_perp(self) -> None:
        assert HyperliquidPriceFeed.resolve_coin("BTC-USDC") == "BTC"
        assert HyperliquidPriceFeed.resolve_coin("ETH-USDC") == "ETH"
        assert HyperliquidPriceFeed.resolve_coin("SOL-USDC") == "SOL"
        assert HyperliquidPriceFeed.resolve_coin("HYPE-USDC") == "HYPE"

    def test_hip3_straight_alias(self) -> None:
        # HIP-3 tokens whose SYMBOL_MAP value matches their base name.
        assert HyperliquidPriceFeed.resolve_coin("GOLD-USDC") == "xyz:GOLD"
        assert HyperliquidPriceFeed.resolve_coin("SP500-USDC") == "xyz:SP500"
        assert HyperliquidPriceFeed.resolve_coin("TSLA-USDC") == "xyz:TSLA"

    def test_hip3_renamed_alias_wtioil_is_cl(self) -> None:
        """Regression: WTIOIL-USDC → xyz:CL, not xyz:WTIOIL. The spec's
        ``split("-")[0]`` fallback would produce ``WTIOIL`` and break —
        this is why resolve_coin reads SYMBOL_MAP as the source of truth.
        """
        assert HyperliquidPriceFeed.resolve_coin("WTIOIL-USDC") == "xyz:CL"

    def test_unknown_symbol_fallback(self) -> None:
        # Unknown symbol — fall back to `split("-")[0]`.
        assert HyperliquidPriceFeed.resolve_coin("UNKNOWN-USDC") == "UNKNOWN"


# ---------------------------------------------------------------------------
# Subscription frames
# ---------------------------------------------------------------------------


class TestSubscribe:
    async def test_subscribe_sends_three_frames_per_symbol(
        self, bus: InProcessBus
    ) -> None:
        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        sent = [json.loads(f) for f in ws.sent]
        types = [msg["subscription"]["type"] for msg in sent]
        assert "trades" in types
        assert "candle" in types
        assert "activeAssetCtx" in types
        for msg in sent:
            assert msg["method"] == "subscribe"
            assert msg["subscription"]["coin"] == "BTC"
        candle_msg = next(m for m in sent if m["subscription"]["type"] == "candle")
        assert candle_msg["subscription"]["interval"] == "1h"

        assert "BTC-USDC" in feed.subscribed_symbols
        assert feed.get_symbol_state("BTC-USDC") is not None

        await feed.disconnect()

    async def test_subscribe_hip3_uses_xyz_prefix(self, bus: InProcessBus) -> None:
        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["WTIOIL-USDC"])
        await drain_events()

        sent = [json.loads(f) for f in ws.sent]
        assert all(msg["subscription"]["coin"] == "xyz:CL" for msg in sent)

        await feed.disconnect()


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------


class TestTradesMessage:
    async def test_trades_emits_price_updated(self, bus: InProcessBus) -> None:
        rec = _Recorder()
        bus.subscribe(PriceUpdated, rec)

        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        ws.push(
            json.dumps(
                {
                    "channel": "trades",
                    "data": [
                        {
                            "coin": "BTC",
                            "side": "B",
                            "px": "65000.5",
                            "sz": "0.1",
                            "time": 1_712_000_000_000,
                        },
                        {
                            "coin": "BTC",
                            "side": "A",
                            "px": "65001.5",
                            "sz": "0.25",
                            "time": 1_712_000_000_500,
                        },
                    ],
                }
            )
        )
        await drain_events(10)

        assert len(rec.events) == 2
        first, second = rec.events
        assert first.update.symbol == "BTC-USDC"
        assert first.update.price == 65000.5
        assert first.update.size == 0.1
        assert first.update.exchange == "hyperliquid"
        assert first.update.timestamp == datetime(
            2024, 4, 1, 19, 33, 20, tzinfo=timezone.utc
        )
        assert second.update.price == 65001.5
        assert feed.get_latest_price("BTC-USDC") == 65001.5

        await feed.disconnect()

    async def test_trades_for_unsubscribed_coin_is_ignored(
        self, bus: InProcessBus
    ) -> None:
        rec = _Recorder()
        bus.subscribe(PriceUpdated, rec)

        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        ws.push(
            json.dumps(
                {
                    "channel": "trades",
                    "data": [
                        {"coin": "ETH", "px": "3200", "sz": "1", "time": 1_712_000_000_000}
                    ],
                }
            )
        )
        await drain_events()
        assert rec.events == []
        await feed.disconnect()

    async def test_trades_malformed_entries_skipped(self, bus: InProcessBus) -> None:
        rec = _Recorder()
        bus.subscribe(PriceUpdated, rec)

        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        ws.push(
            json.dumps(
                {
                    "channel": "trades",
                    "data": [
                        {"coin": "BTC", "px": "not-a-number", "sz": "1", "time": 0},
                        {"coin": "BTC"},  # no px
                        {"coin": "BTC", "px": "65000", "sz": "0.1", "time": 0},  # valid
                    ],
                }
            )
        )
        await drain_events()

        assert len(rec.events) == 1
        assert rec.events[0].update.price == 65000.0
        await feed.disconnect()


class TestCandleMessage:
    async def test_candle_close_detected_on_open_time_advance(
        self, bus: InProcessBus
    ) -> None:
        rec = _Recorder()
        bus.subscribe(CandleClosed, rec)

        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        # First candle at t=1_712_000_000_000 (1h)
        ws.push(
            json.dumps(
                {
                    "channel": "candle",
                    "data": {
                        "s": "BTC",
                        "i": "1h",
                        "t": 1_712_000_000_000,
                        "T": 1_712_003_600_000,
                        "o": "65000",
                        "h": "65500",
                        "l": "64800",
                        "c": "65400",
                        "v": "123.45",
                    },
                }
            )
        )
        await drain_events()
        # No close yet — still building.
        assert rec.events == []

        # Same candle updated — still building.
        ws.push(
            json.dumps(
                {
                    "channel": "candle",
                    "data": {
                        "s": "BTC",
                        "i": "1h",
                        "t": 1_712_000_000_000,
                        "T": 1_712_003_600_000,
                        "o": "65000",
                        "h": "65550",
                        "l": "64800",
                        "c": "65420",
                        "v": "140.0",
                    },
                }
            )
        )
        await drain_events()
        assert rec.events == []

        # New candle open-time = the old one is now complete.
        ws.push(
            json.dumps(
                {
                    "channel": "candle",
                    "data": {
                        "s": "BTC",
                        "i": "1h",
                        "t": 1_712_003_600_000,
                        "T": 1_712_007_200_000,
                        "o": "65420",
                        "h": "65500",
                        "l": "65300",
                        "c": "65480",
                        "v": "5.0",
                    },
                }
            )
        )
        await drain_events()

        assert len(rec.events) == 1
        closed = rec.events[0]
        assert closed.candle.symbol == "BTC-USDC"
        assert closed.candle.timeframe == "1h"
        # The flushed candle reflects the LAST update before the advance —
        # open 65000, close 65420, high 65550, low 64800.
        assert closed.candle.open == 65000.0
        assert closed.candle.close == 65420.0
        assert closed.candle.high == 65550.0
        assert closed.candle.low == 64800.0
        assert closed.candle.volume == 140.0

        # And the PriceFeed now has one completed candle in history.
        history = feed.get_candle_history("BTC-USDC", count=10)
        assert len(history) == 1
        assert history[0]["close"] == 65420.0

        await feed.disconnect()


class TestActiveAssetCtx:
    async def test_active_asset_ctx_emits_funding_and_oi(
        self, bus: InProcessBus
    ) -> None:
        funding_rec = _Recorder()
        oi_rec = _Recorder()
        bus.subscribe(FundingUpdated, funding_rec)
        bus.subscribe(OpenInterestUpdated, oi_rec)

        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        ws.push(
            json.dumps(
                {
                    "channel": "activeAssetCtx",
                    "data": {
                        "coin": "BTC",
                        "ctx": {
                            "funding": "0.0001",
                            "openInterest": "1000000000",
                            "markPx": "65000",
                        },
                    },
                }
            )
        )
        await drain_events()

        assert len(funding_rec.events) == 1
        assert funding_rec.events[0].update.funding_rate == 0.0001
        assert feed.get_funding_rate("BTC-USDC") == 0.0001

        assert len(oi_rec.events) == 1
        assert oi_rec.events[0].update.open_interest == 1_000_000_000.0
        assert oi_rec.events[0].update.oi_change_pct is None  # no previous

        # Second snapshot — oi_change_pct should be computed.
        ws.push(
            json.dumps(
                {
                    "channel": "activeAssetCtx",
                    "data": {
                        "coin": "BTC",
                        "ctx": {
                            "funding": "0.0002",
                            "openInterest": "1010000000",
                        },
                    },
                }
            )
        )
        await drain_events(20)

        assert len(oi_rec.events) == 2
        assert oi_rec.events[1].update.open_interest == 1_010_000_000.0
        assert oi_rec.events[1].update.oi_change_pct == pytest.approx(0.01)

        await feed.disconnect()


class TestMalformedMessages:
    async def test_invalid_json_is_skipped(self, bus: InProcessBus) -> None:
        rec = _Recorder()
        bus.subscribe(PriceUpdated, rec)

        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        ws.push("not-json-at-all")
        ws.push(json.dumps({"channel": "unknown_channel", "data": {}}))
        ws.push(
            json.dumps(
                {
                    "channel": "subscriptionResponse",
                    "data": {"method": "subscribe", "ok": True},
                }
            )
        )
        await drain_events()

        # No crash, no spurious events.
        assert rec.events == []
        assert feed.is_connected() is True
        await feed.disconnect()


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------


class TestReconnect:
    async def test_reconnect_fires_and_resubscribes(self, bus: InProcessBus) -> None:
        ws_a = FakeWebSocket()
        ws_b = FakeWebSocket()
        instances = [ws_a, ws_b]
        calls: list[str] = []

        async def connect_stub(url: str) -> FakeWebSocket:
            calls.append(url)
            return instances.pop(0)

        feed = HyperliquidPriceFeed(
            event_bus=bus, candle_timeframe="1h", ws_connect=connect_stub
        )
        feed._reconnect_delay = 0.01
        feed._max_reconnect_delay = 0.01

        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()
        assert len(calls) == 1
        assert len(ws_a.sent) == 3  # trades + candle + activeAssetCtx

        # Force a disconnect on the first ws.
        ws_a.fail()

        # Give the reconnect loop time to reopen.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(calls) >= 2 and len(ws_b.sent) >= 3:
                break

        assert len(calls) == 2
        # Second connection re-sent all 3 subscription frames.
        sent_b = [json.loads(f) for f in ws_b.sent]
        types_b = [msg["subscription"]["type"] for msg in sent_b]
        assert sorted(types_b) == sorted(["trades", "candle", "activeAssetCtx"])
        # Backoff advanced.
        assert feed._reconnect_delay == pytest.approx(0.01)

        await feed.disconnect()


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    async def test_unsubscribe_sends_frames_and_drops_state(
        self, bus: InProcessBus
    ) -> None:
        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        # Reset the send log so we only see unsubscribe frames.
        ws.sent.clear()
        await feed.unsubscribe(["BTC-USDC"])
        await drain_events()

        sent = [json.loads(f) for f in ws.sent]
        assert len(sent) == 3
        for msg in sent:
            assert msg["method"] == "unsubscribe"
            assert msg["subscription"]["coin"] == "BTC"

        assert "BTC-USDC" not in feed.subscribed_symbols
        assert feed.get_symbol_state("BTC-USDC") is None

        await feed.disconnect()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bootstrap history
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Minimal ExchangeAdapter stand-in for bootstrap tests.

    ``responses`` maps symbol → either a list[dict] (success) or an
    Exception (raises). ``calls`` records every fetch_ohlcv invocation
    so tests can assert on what got requested.
    """

    def __init__(self, responses: dict) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100, since=None
    ) -> list[dict]:
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "limit": limit})
        result = self.responses.get(symbol)
        if isinstance(result, Exception):
            raise result
        return result or []


def _make_bootstrap_candle(ts_ms: int, close: float) -> dict:
    return {
        "timestamp": ts_ms,
        "open": close - 5,
        "high": close + 10,
        "low": close - 10,
        "close": close,
        "volume": 1.5,
    }


class TestBootstrapHistory:
    async def test_bootstrap_seeds_completed_candles_deque(
        self, bus: InProcessBus
    ) -> None:
        ws = FakeWebSocket()
        # 3 historical 1h candles for BTC, ascending by timestamp.
        history = [
            _make_bootstrap_candle(1_711_900_000_000, 64000.0),
            _make_bootstrap_candle(1_711_903_600_000, 64500.0),
            _make_bootstrap_candle(1_711_907_200_000, 65000.0),
        ]
        adapter = _StubAdapter({"BTC-USDC": history})

        async def connect_stub(url: str) -> FakeWebSocket:
            return ws

        feed = HyperliquidPriceFeed(
            event_bus=bus,
            candle_timeframe="1h",
            bootstrap_adapter=adapter,
            ws_connect=connect_stub,
        )
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        # adapter was called once for BTC-USDC with the right kwargs
        assert adapter.calls == [
            {"symbol": "BTC-USDC", "timeframe": "1h", "limit": 100}
        ]

        # The deque now holds 3 candles in the same order REST returned them.
        candles = feed.get_candle_history("BTC-USDC", count=10)
        assert len(candles) == 3
        assert [c["close"] for c in candles] == [64000.0, 64500.0, 65000.0]

        # Each normalised candle has datetime timestamps + the timeframe field.
        for c in candles:
            assert isinstance(c["timestamp"], datetime)
            assert c["timeframe"] == "1h"

        # latest_price seeded from the most recent close so consumers
        # don't see None before the first WS tick.
        assert feed.get_latest_price("BTC-USDC") == 65000.0

        await feed.disconnect()

    async def test_bootstrap_failure_for_one_symbol_does_not_block_others(
        self, bus: InProcessBus
    ) -> None:
        ws = FakeWebSocket()
        adapter = _StubAdapter(
            {
                "BTC-USDC": RuntimeError("rate limited"),
                "ETH-USDC": [
                    _make_bootstrap_candle(1_711_907_200_000, 3200.0),
                    _make_bootstrap_candle(1_711_910_800_000, 3210.0),
                ],
            }
        )

        async def connect_stub(url: str) -> FakeWebSocket:
            return ws

        feed = HyperliquidPriceFeed(
            event_bus=bus,
            candle_timeframe="1h",
            bootstrap_adapter=adapter,
            ws_connect=connect_stub,
        )
        await feed.connect()
        await feed.subscribe(["BTC-USDC", "ETH-USDC"])
        await drain_events()

        # BTC: bootstrap failed, deque empty
        assert feed.get_candle_history("BTC-USDC", count=10) == []
        # ETH: bootstrap succeeded, deque populated
        eth = feed.get_candle_history("ETH-USDC", count=10)
        assert len(eth) == 2
        assert eth[-1]["close"] == 3210.0
        # The feed remains connected — no crash.
        assert feed.is_connected() is True

        await feed.disconnect()

    async def test_no_adapter_skips_bootstrap_gracefully(
        self, bus: InProcessBus
    ) -> None:
        ws = FakeWebSocket()
        feed = make_feed(bus, ws)  # no bootstrap_adapter
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        # Deque starts empty — that's fine, the WS will fill it over time.
        assert feed.get_candle_history("BTC-USDC", count=10) == []
        # WS subscribe frames still went out.
        assert len(ws.sent) == 3

        await feed.disconnect()

    async def test_websocket_candles_append_after_bootstrapped_history(
        self, bus: InProcessBus
    ) -> None:
        rec = _Recorder()
        bus.subscribe(CandleClosed, rec)

        ws = FakeWebSocket()
        # 2 historical candles, then we'll push 2 WS candles via the listen loop.
        history = [
            _make_bootstrap_candle(1_711_907_200_000, 64000.0),
            _make_bootstrap_candle(1_711_910_800_000, 64500.0),
        ]
        adapter = _StubAdapter({"BTC-USDC": history})

        async def connect_stub(url: str) -> FakeWebSocket:
            return ws

        feed = HyperliquidPriceFeed(
            event_bus=bus,
            candle_timeframe="1h",
            bootstrap_adapter=adapter,
            ws_connect=connect_stub,
        )
        await feed.connect()
        await feed.subscribe(["BTC-USDC"])
        await drain_events()

        # WS pushes a building candle, then a NEW open-time so the building
        # candle gets flushed via _complete_candle.
        ws.push(
            json.dumps(
                {
                    "channel": "candle",
                    "data": {
                        "s": "BTC", "i": "1h",
                        "t": 1_711_914_400_000, "T": 1_711_918_000_000,
                        "o": "64500", "h": "65000", "l": "64400", "c": "64900",
                        "v": "10",
                    },
                }
            )
        )
        ws.push(
            json.dumps(
                {
                    "channel": "candle",
                    "data": {
                        "s": "BTC", "i": "1h",
                        "t": 1_711_918_000_000, "T": 1_711_921_600_000,
                        "o": "64900", "h": "65100", "l": "64850", "c": "65050",
                        "v": "12",
                    },
                }
            )
        )
        await drain_events(10)

        # The deque now holds: 2 bootstrapped + 1 newly closed WS candle (the
        # first one became "previous" when the second one arrived). The second
        # WS candle is still building, not yet in the deque.
        candles = feed.get_candle_history("BTC-USDC", count=10)
        assert len(candles) == 3
        closes = [c["close"] for c in candles]
        # Time-ordered: 64000 (bootstrap), 64500 (bootstrap), 64900 (WS-flushed).
        assert closes == [64000.0, 64500.0, 64900.0]

        # And exactly one CandleClosed event was emitted (for the WS flush).
        assert len(rec.events) == 1
        assert rec.events[0].candle.close == 64900.0

        await feed.disconnect()


class TestLifecycle:
    async def test_disconnect_cancels_listen_and_closes_ws(
        self, bus: InProcessBus
    ) -> None:
        ws = FakeWebSocket()
        feed = make_feed(bus, ws)
        await feed.connect()
        assert feed.is_connected() is True
        await feed.disconnect()
        assert feed.is_connected() is False
        assert ws.closed is True
