"""Unit tests for EventBus (InProcessBus) implementation."""

import asyncio

import pytest

from engine.events import (
    CycleCompleted,
    DataReady,
    Event,
    EventBus,
    InProcessBus,
    SetupDetected,
    SignalsReady,
    create_event_bus,
)
from engine.types import MarketData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market_data() -> MarketData:
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=[],
        num_candles=0,
        lookback_description="~0",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={},
        swing_highs=[],
        swing_lows=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubscribeAndPublish:
    @pytest.mark.asyncio
    async def test_handler_receives_event(self) -> None:
        bus = InProcessBus()
        received: list[Event] = []

        async def handler(event: DataReady) -> None:
            received.append(event)

        bus.subscribe(DataReady, handler)
        event = DataReady(source="test", market_data=_make_market_data())
        await bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_sync_handler_works(self) -> None:
        bus = InProcessBus()
        received: list[Event] = []

        def handler(event: SetupDetected) -> None:
            received.append(event)

        bus.subscribe(SetupDetected, handler)
        event = SetupDetected(source="test", symbol="ETH-USDC", readiness=0.9)
        await bus.publish(event)

        assert len(received) == 1


class TestMultipleHandlers:
    @pytest.mark.asyncio
    async def test_all_handlers_receive_event(self) -> None:
        bus = InProcessBus()
        results_a: list[Event] = []
        results_b: list[Event] = []
        results_c: list[Event] = []

        async def handler_a(e: DataReady) -> None:
            results_a.append(e)

        async def handler_b(e: DataReady) -> None:
            results_b.append(e)

        async def handler_c(e: DataReady) -> None:
            results_c.append(e)

        bus.subscribe(DataReady, handler_a)
        bus.subscribe(DataReady, handler_b)
        bus.subscribe(DataReady, handler_c)

        event = DataReady(source="test", market_data=_make_market_data())
        await bus.publish(event)

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert len(results_c) == 1


class TestHandlerErrorIsolation:
    @pytest.mark.asyncio
    async def test_error_does_not_crash_publisher(self) -> None:
        bus = InProcessBus()
        results: list[Event] = []

        async def good_handler_before(e: DataReady) -> None:
            results.append(("before", e))

        async def bad_handler(e: DataReady) -> None:
            raise RuntimeError("handler exploded")

        async def good_handler_after(e: DataReady) -> None:
            results.append(("after", e))

        bus.subscribe(DataReady, good_handler_before)
        bus.subscribe(DataReady, bad_handler)
        bus.subscribe(DataReady, good_handler_after)

        event = DataReady(source="test", market_data=_make_market_data())
        await bus.publish(event)

        # Both good handlers ran despite the bad one raising
        assert len(results) == 2
        assert bus.handler_errors == 1


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self) -> None:
        bus = InProcessBus()
        received: list[Event] = []

        async def handler(e: SetupDetected) -> None:
            received.append(e)

        bus.subscribe(SetupDetected, handler)

        event1 = SetupDetected(source="test", symbol="BTC-USDC", readiness=0.8)
        await bus.publish(event1)
        assert len(received) == 1

        bus.unsubscribe(SetupDetected, handler)

        event2 = SetupDetected(source="test", symbol="BTC-USDC", readiness=0.9)
        await bus.publish(event2)
        assert len(received) == 1  # still 1 — handler not called

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_handler_is_noop(self) -> None:
        bus = InProcessBus()

        async def handler(e: Event) -> None:
            pass

        # Should not raise
        bus.unsubscribe(DataReady, handler)


class TestEventTypeFiltering:
    @pytest.mark.asyncio
    async def test_subscribe_to_one_type_ignores_other(self) -> None:
        bus = InProcessBus()
        received: list[Event] = []

        async def handler(e: DataReady) -> None:
            received.append(e)

        bus.subscribe(DataReady, handler)

        # Publish a different event type
        await bus.publish(SignalsReady(source="test", signals=[]))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_two_types_independent(self) -> None:
        bus = InProcessBus()
        data_events: list[Event] = []
        setup_events: list[Event] = []

        async def data_handler(e: DataReady) -> None:
            data_events.append(e)

        async def setup_handler(e: SetupDetected) -> None:
            setup_events.append(e)

        bus.subscribe(DataReady, data_handler)
        bus.subscribe(SetupDetected, setup_handler)

        await bus.publish(DataReady(source="test", market_data=_make_market_data()))
        await bus.publish(SetupDetected(source="test", symbol="X", readiness=0.5))

        assert len(data_events) == 1
        assert len(setup_events) == 1


class TestNoSubscribers:
    @pytest.mark.asyncio
    async def test_publish_with_no_subscribers_no_error(self) -> None:
        bus = InProcessBus()
        # Should not raise
        await bus.publish(DataReady(source="test", market_data=_make_market_data()))
        assert bus.total_published == 1


class TestMetrics:
    @pytest.mark.asyncio
    async def test_total_published_count(self) -> None:
        bus = InProcessBus()
        for _ in range(5):
            await bus.publish(SetupDetected(source="test", symbol="X", readiness=0.5))

        metrics = bus.get_metrics()
        assert metrics["total_published"] == 5
        assert metrics["per_type_counts"]["SetupDetected"] == 5

    @pytest.mark.asyncio
    async def test_per_type_counts(self) -> None:
        bus = InProcessBus()
        await bus.publish(DataReady(source="test", market_data=_make_market_data()))
        await bus.publish(DataReady(source="test", market_data=_make_market_data()))
        await bus.publish(SetupDetected(source="test", symbol="X", readiness=0.5))

        metrics = bus.get_metrics()
        assert metrics["per_type_counts"]["DataReady"] == 2
        assert metrics["per_type_counts"]["SetupDetected"] == 1
        assert metrics["total_published"] == 3

    @pytest.mark.asyncio
    async def test_handler_error_count(self) -> None:
        bus = InProcessBus()

        async def bad(e: Event) -> None:
            raise ValueError("boom")

        bus.subscribe(CycleCompleted, bad)

        await bus.publish(CycleCompleted(source="test", symbol="X", action="SKIP", conviction=0.1))
        await bus.publish(CycleCompleted(source="test", symbol="X", action="SKIP", conviction=0.2))

        assert bus.get_metrics()["handler_errors"] == 2


class TestFactory:
    def test_create_memory_bus(self) -> None:
        bus = create_event_bus("memory")
        assert isinstance(bus, InProcessBus)
        assert isinstance(bus, EventBus)

    def test_create_redis_bus_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            create_event_bus("redis")

    def test_create_unknown_bus_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown bus backend"):
            create_event_bus("kafka")
