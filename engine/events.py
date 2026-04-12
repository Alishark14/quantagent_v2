"""Event Bus: typed events, publish/subscribe for inter-module communication."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

from engine.types import (
    CandleClose,
    ConvictionOutput,
    FundingUpdate,
    MarketData,
    OIUpdate,
    OrderResult,
    Position,
    PriceUpdate,
    SignalOutput,
    TradeAction,
)


@dataclass
class Event:
    """Base event. All events carry a timestamp and source module identifier."""

    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DataReady(Event):
    """Emitted when the data layer has assembled a complete MarketData package."""

    market_data: MarketData = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class SignalsReady(Event):
    """Emitted when all SignalProducers have returned their outputs."""

    signals: list[SignalOutput] = field(default_factory=list)


@dataclass
class ConvictionScored(Event):
    """Emitted when the ConvictionAgent has scored signal consensus."""

    conviction: ConvictionOutput = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class TradeOpened(Event):
    """Emitted when a new trade has been executed on the exchange."""

    trade_action: TradeAction = field(default_factory=lambda: None)  # type: ignore[arg-type]
    order_result: OrderResult = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class TradeClosed(Event):
    """Emitted when a position has been fully closed.

    The `trade_id` and entry-side fields are optional for back-compat —
    the executor's CLOSE_ALL emission doesn't know the trade record id
    yet. Downstream tracking handlers (e.g. `ForwardMaxRStamper`) that
    need to look up the trade should fall back to a no-op + warning
    when `trade_id` is None instead of crashing.
    """

    symbol: str = ""
    pnl: float = 0.0
    exit_reason: str = ""
    trade_id: str | None = None
    direction: str | None = None  # "LONG" | "SHORT"
    entry_price: float | None = None
    sl_price: float | None = None
    entry_timestamp_ms: int | None = None
    timeframe: str | None = None


@dataclass
class PositionUpdated(Event):
    """Emitted when Sentinel detects a position state change."""

    symbol: str = ""
    position: Position = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class SetupDetected(Event):
    """Emitted when Sentinel detects a tradeable setup."""

    symbol: str = ""
    readiness: float = 0.0
    conditions: list[str] = field(default_factory=list)


@dataclass
class RuleGenerated(Event):
    """Emitted when the ReflectionAgent distills a new trading rule."""

    rule: dict = field(default_factory=dict)


@dataclass
class FactorsUpdated(Event):
    """Emitted when the Overnight Quant MCP produces new alpha factors."""

    filepath: str = ""


@dataclass
class MacroUpdated(Event):
    """Emitted when the Macro Manager MCP produces a new regime assessment."""

    filepath: str = ""


@dataclass
class VolumeAnomaly(Event):
    """Emitted by Sentinel when a single-symbol volume anomaly is detected.

    The aggregator (`MacroEventAggregator`) collects these alongside
    `ExtremeMove` events and escalates to `MacroReassessmentRequired`
    when 5+ unique symbols fire within a 60-second window.

    `severity` is normalised to [0.0, 1.0] so the aggregator can rank
    triggers across symbols / event types.
    """

    symbol: str = ""
    severity: float = 0.0  # 0.0–1.0
    detail: str = ""


@dataclass
class ExtremeMove(Event):
    """Emitted by Sentinel when a price move >N×ATR is detected.

    Companion to `VolumeAnomaly` — both feed the
    `MacroEventAggregator` swarm-consensus pipeline (§13.2.5).
    """

    symbol: str = ""
    severity: float = 0.0  # 0.0–1.0
    move_pct: float = 0.0
    detail: str = ""


@dataclass
class MacroReassessmentRequired(Event):
    """Emitted by `MacroEventAggregator` on swarm consensus.

    Per ARCHITECTURE §13.2.5: when 5+ active Sentinels across DIFFERENT
    symbols fire `VolumeAnomaly` or `ExtremeMove` within a 60-second
    window, the aggregator publishes this event. A handler subscribes
    and triggers the Macro Regime Manager's emergency mode (out of
    process — the wiring to the MCP runner is infrastructure work for
    a future task).

    The payload tells the LLM what the swarm is reacting to so the
    emergency assessment can be calibrated to the observed stress.
    """

    triggering_symbols: list[str] = field(default_factory=list)
    anomaly_types: list[str] = field(default_factory=list)
    severity_scores: list[float] = field(default_factory=list)
    triggered_at: str = ""  # ISO 8601 UTC, mirrors `timestamp` for downstream JSON


@dataclass
class CycleCompleted(Event):
    """Emitted at the end of every analysis cycle for tracking."""

    symbol: str = ""
    action: str = ""
    conviction: float = 0.0


@dataclass
class SetupResult(Event):
    """Emitted by BotManager after a SetupDetected-spawned TraderBot completes.

    Closes the feedback loop from `SetupDetected` (Sentinel → BotManager
    → TraderBot → pipeline) back to the Sentinel so it can adjust its
    escalation state. The Sentinel never gets to see the analysis
    pipeline's actual decision otherwise — it just fires events and
    forgets them. Without this feedback, escalation-after-SKIP would
    require Sentinel to subscribe to TradeOpened with a heuristic
    timeout, which is fragile.

    `outcome` is the canonical TRADE / SKIP classification used by
    Sentinel's escalation logic:
      * "TRADE" — pipeline opened a new position (LONG / SHORT /
        ADD_LONG / ADD_SHORT) AND the order succeeded.
      * "SKIP"  — anything else (SKIP, HOLD, CLOSE_ALL, errored bot,
        failed order). CLOSE_ALL is classified SKIP because it closes
        a position rather than opening one — there's no new-entry
        rationale to reset Sentinel's escalation.

    `action` is the literal TradeAction.action string for traceability.
    `bot_id` lets the Sentinel correlate this back to a specific spawn.
    """

    symbol: str = ""
    outcome: str = "SKIP"  # "TRADE" | "SKIP"
    action: str = ""  # literal TradeAction.action
    bot_id: str = ""
    conviction_score: float = 0.0


# ---------------------------------------------------------------------------
# PriceFeed events (Sprint Week 7 — Event-Driven Refactor Phase 1)
#
# Emitted by `engine/data/price_feed/*` implementations. Each event wraps
# a payload dataclass from `engine/types.py`. Subscribers use the event
# class for class-based dispatch, matching the existing pattern used by
# DataReady / SignalsReady / TradeClosed.
# ---------------------------------------------------------------------------


@dataclass
class PriceUpdated(Event):
    """Emitted by a PriceFeed on every tick (trades channel)."""

    update: PriceUpdate = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class CandleClosed(Event):
    """Emitted by a PriceFeed when a candle finalises."""

    candle: CandleClose = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class FundingUpdated(Event):
    """Emitted by a PriceFeed on funding rate changes."""

    update: FundingUpdate = field(default_factory=lambda: None)  # type: ignore[arg-type]


@dataclass
class OpenInterestUpdated(Event):
    """Emitted by a PriceFeed on open interest changes."""

    update: OIUpdate = field(default_factory=lambda: None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Event Bus — abstract interface and in-process implementation
# ---------------------------------------------------------------------------


class EventBus(ABC):
    """Abstract Event Bus. Modules publish/subscribe via typed events."""

    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    def subscribe(self, event_type: type[Event], handler: Callable) -> None: ...

    @abstractmethod
    def unsubscribe(self, event_type: type[Event], handler: Callable) -> None: ...


class InProcessBus(EventBus):
    """Single-process Event Bus using asyncio for parallel handler dispatch."""

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()
        # Metrics
        self.total_published: int = 0
        self.per_type_counts: dict[str, int] = defaultdict(int)
        self.handler_errors: int = 0

    def subscribe(self, event_type: type[Event], handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[Event], handler: Callable) -> None:
        handlers = self._handlers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        event_type = type(event)

        async with self._lock:
            self.total_published += 1
            self.per_type_counts[event_type.__name__] += 1

        handlers = list(self._handlers.get(event_type, []))
        if not handlers:
            return

        async def _safe_call(handler: Callable) -> None:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    f"Handler {handler.__qualname__} failed for {event_type.__name__}"
                )
                async with self._lock:
                    self.handler_errors += 1

        await asyncio.gather(*[_safe_call(h) for h in handlers])

    def get_metrics(self) -> dict:
        return {
            "total_published": self.total_published,
            "per_type_counts": dict(self.per_type_counts),
            "handler_errors": self.handler_errors,
        }


def create_event_bus(backend: str = "memory") -> EventBus:
    """Factory for EventBus instances."""
    if backend == "memory":
        return InProcessBus()
    elif backend == "redis":
        raise NotImplementedError("Redis EventBus planned for multi-server")
    else:
        raise ValueError(f"Unknown bus backend: {backend}")
