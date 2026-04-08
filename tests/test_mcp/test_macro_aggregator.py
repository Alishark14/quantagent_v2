"""Tests for the MacroEventAggregator swarm-consensus pipeline.

ARCHITECTURE §13.2.5: when 5+ active Sentinels across DIFFERENT
symbols emit `VolumeAnomaly` or `ExtremeMove` within a 60-second
window, the aggregator emits `MacroReassessmentRequired`. Same-symbol
duplicates do not count toward the 5-symbol threshold; the window
resets after emission and a 10-minute cooldown blocks re-trigger.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engine.events import (
    ExtremeMove,
    InProcessBus,
    MacroReassessmentRequired,
    VolumeAnomaly,
)
from sentinel.macro_aggregator import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MIN_UNIQUE_SYMBOLS,
    DEFAULT_WINDOW_SECONDS,
    MacroEventAggregator,
)


_BASE = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)


class _Clock:
    """Tickable clock so tests can advance time deterministically."""

    def __init__(self, start: datetime = _BASE) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _vol(symbol: str, severity: float = 0.5) -> VolumeAnomaly:
    return VolumeAnomaly(source="sentinel", symbol=symbol, severity=severity)


def _move(symbol: str, severity: float = 0.5) -> ExtremeMove:
    return ExtremeMove(source="sentinel", symbol=symbol, severity=severity)


@pytest.fixture
def bus():
    return InProcessBus()


@pytest.fixture
def emissions(bus: InProcessBus):
    received: list[MacroReassessmentRequired] = []
    bus.subscribe(MacroReassessmentRequired, lambda e: received.append(e))
    return received


# ---------------------------------------------------------------------------
# Threshold + window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_4_symbols_no_trigger(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    for sym in ("BTC", "ETH", "SOL", "AVAX"):
        await agg.handle_event(_vol(sym))
    assert emissions == []
    assert agg.metrics.events_received == 4
    assert agg.metrics.triggers_emitted == 0


@pytest.mark.asyncio
async def test_5_unique_symbols_within_window_triggers(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    for sym, sev in (("BTC", 0.6), ("ETH", 0.5), ("SOL", 0.7), ("AVAX", 0.4), ("ARB", 0.8)):
        await agg.handle_event(_vol(sym, sev))
        clock.advance(5)  # 5s spacing → all within 60s

    assert len(emissions) == 1
    event = emissions[0]
    assert sorted(event.triggering_symbols) == ["ARB", "AVAX", "BTC", "ETH", "SOL"]
    assert event.anomaly_types == ["VolumeAnomaly"]
    assert len(event.severity_scores) == 5
    assert event.triggered_at  # ISO timestamp populated
    assert agg.metrics.triggers_emitted == 1


@pytest.mark.asyncio
async def test_5_events_same_symbol_no_trigger(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    for _ in range(5):
        await agg.handle_event(_vol("BTC"))
        clock.advance(2)
    assert emissions == []


@pytest.mark.asyncio
async def test_5_events_2_symbols_no_trigger(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    for sym in ("BTC", "ETH", "BTC", "ETH", "BTC"):
        await agg.handle_event(_vol(sym))
        clock.advance(2)
    assert emissions == []


@pytest.mark.asyncio
async def test_5_symbols_spread_over_90s_no_trigger(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    # 5 events, 25s apart → spans 100s. The first one ages out before
    # the 5th arrives, so the window only ever holds 4 unique symbols.
    for sym in ("BTC", "ETH", "SOL", "AVAX", "ARB"):
        await agg.handle_event(_vol(sym))
        clock.advance(25)
    assert emissions == []


# ---------------------------------------------------------------------------
# Mixed event types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_volume_and_extreme_events_count_toward_threshold(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    await agg.handle_event(_vol("BTC", 0.5))
    clock.advance(3)
    await agg.handle_event(_move("ETH", 0.6))
    clock.advance(3)
    await agg.handle_event(_vol("SOL", 0.4))
    clock.advance(3)
    await agg.handle_event(_move("AVAX", 0.7))
    clock.advance(3)
    await agg.handle_event(_vol("ARB", 0.3))
    assert len(emissions) == 1
    types = set(emissions[0].anomaly_types)
    assert types == {"VolumeAnomaly", "ExtremeMove"}


@pytest.mark.asyncio
async def test_same_symbol_both_event_types_counts_once(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    await agg.handle_event(_vol("BTC"))
    clock.advance(1)
    await agg.handle_event(_move("BTC"))
    clock.advance(1)
    await agg.handle_event(_vol("ETH"))
    clock.advance(1)
    await agg.handle_event(_move("SOL"))
    clock.advance(1)
    await agg.handle_event(_vol("AVAX"))
    # 4 unique symbols (BTC counted once even though both types fired).
    assert emissions == []


@pytest.mark.asyncio
async def test_severity_payload_uses_max_per_symbol(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    await agg.handle_event(_vol("BTC", 0.3))
    clock.advance(1)
    await agg.handle_event(_move("BTC", 0.9))  # higher → wins for BTC
    clock.advance(1)
    for sym in ("ETH", "SOL", "AVAX", "ARB"):
        await agg.handle_event(_vol(sym, 0.5))
        clock.advance(1)
    assert len(emissions) == 1
    by_symbol = dict(zip(emissions[0].triggering_symbols, emissions[0].severity_scores))
    assert by_symbol["BTC"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Cooldown + window reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_blocks_immediate_retrigger(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    # First swarm → fires.
    for sym in ("BTC", "ETH", "SOL", "AVAX", "ARB"):
        await agg.handle_event(_vol(sym))
        clock.advance(1)
    assert len(emissions) == 1

    # Immediately fire 5 more anomalies on different symbols within
    # the cooldown period. Window resets after emission so we have to
    # rebuild from scratch.
    clock.advance(30)
    for sym in ("MATIC", "OP", "LINK", "DOGE", "PEPE"):
        await agg.handle_event(_vol(sym))
        clock.advance(1)
    # Still only the original emission.
    assert len(emissions) == 1
    assert agg.metrics.triggers_suppressed_cooldown >= 1


@pytest.mark.asyncio
async def test_retrigger_allowed_after_cooldown(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock, cooldown_seconds=10.0)
    for sym in ("BTC", "ETH", "SOL", "AVAX", "ARB"):
        await agg.handle_event(_vol(sym))
        clock.advance(1)
    assert len(emissions) == 1

    # Wait past cooldown and fire a fresh swarm.
    clock.advance(15)
    for sym in ("MATIC", "OP", "LINK", "DOGE", "PEPE"):
        await agg.handle_event(_vol(sym))
        clock.advance(1)
    assert len(emissions) == 2


@pytest.mark.asyncio
async def test_window_reset_after_emission(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock, cooldown_seconds=0.0)
    for sym in ("BTC", "ETH", "SOL", "AVAX", "ARB"):
        await agg.handle_event(_vol(sym))
        clock.advance(1)
    assert len(emissions) == 1
    assert len(agg._pending) == 0  # window cleared after emission

    # 4 fresh symbols immediately after — should NOT re-trigger because
    # the previous batch is gone, even with cooldown=0.
    for sym in ("MATIC", "OP", "LINK", "DOGE"):
        await agg.handle_event(_vol(sym))
        clock.advance(1)
    assert len(emissions) == 1

    # Adding the 5th unique symbol fires again (cooldown is 0).
    await agg.handle_event(_vol("PEPE"))
    assert len(emissions) == 2


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_wires_to_bus(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    agg.subscribe()
    # Publish through the real bus instead of calling handle_event.
    for sym in ("BTC", "ETH", "SOL", "AVAX", "ARB"):
        clock.advance(1)
        await bus.publish(_vol(sym))
    assert len(emissions) == 1


@pytest.mark.asyncio
async def test_unsubscribe_stops_handling(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    agg.subscribe()
    agg.unsubscribe()
    for sym in ("BTC", "ETH", "SOL", "AVAX", "ARB"):
        await bus.publish(_vol(sym))
        clock.advance(1)
    assert emissions == []
    assert agg.metrics.events_received == 0


# ---------------------------------------------------------------------------
# Defensive paths + constants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_without_symbol_ignored(bus, emissions):
    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    await agg.handle_event(_vol(""))  # empty symbol
    assert agg.metrics.events_received == 0
    assert agg.metrics.events_dropped_oldest == 0


@pytest.mark.asyncio
async def test_unrelated_event_type_ignored(bus, emissions):
    """A handler invocation with the wrong type should not crash."""

    class _Unrelated:
        symbol = "BTC"
        severity = 0.5

    clock = _Clock()
    agg = MacroEventAggregator(bus, clock=clock)
    await agg.handle_event(_Unrelated())
    assert agg.metrics.events_received == 0


def test_default_constants_match_spec():
    assert DEFAULT_WINDOW_SECONDS == 60.0
    assert DEFAULT_MIN_UNIQUE_SYMBOLS == 5
    assert DEFAULT_COOLDOWN_SECONDS == 600.0
