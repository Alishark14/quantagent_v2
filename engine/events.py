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
    ConvictionOutput,
    MarketData,
    OrderResult,
    Position,
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
    """Emitted when a position has been fully closed."""

    symbol: str = ""
    pnl: float = 0.0
    exit_reason: str = ""


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
class CycleCompleted(Event):
    """Emitted at the end of every analysis cycle for tracking."""

    symbol: str = ""
    action: str = ""
    conviction: float = 0.0


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
