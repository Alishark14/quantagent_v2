"""Abstract PriceFeed base class + per-symbol state container.

Sprint Week 7 Task 2. The contract is exchange-agnostic: subclasses
(``HyperliquidPriceFeed`` in Task 3, future ``BinancePriceFeed`` etc.)
own the WebSocket connection plumbing and the venue-specific message
parsing, while this ABC owns:

    * in-memory ``SymbolState`` per subscribed symbol (price, bid/ask,
      a rolling deque of completed candles, funding, open interest),
    * the sync ``get_*`` reads that Sentinel / SLTPMonitor / signal
      agents use — ZERO network calls on the hot path,
    * the ``_update_*`` helpers that mutate ``SymbolState`` AND publish
      the matching event on the ``EventBus`` so everything downstream
      stays decoupled.

Subclasses implement ``connect`` / ``disconnect`` / ``_listen`` /
``subscribe`` / ``unsubscribe`` and call the ``_update_*`` helpers from
their message-loop parser. They must NOT poke at ``self._symbols``
directly — all state mutation flows through the helpers so the bus and
the memory stay in lock-step.
"""

from __future__ import annotations

import logging
import asyncio
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from engine.events import (
    CandleClosed,
    EventBus,
    FundingUpdated,
    OpenInterestUpdated,
    PriceUpdated,
)
from engine.types import (
    CandleClose,
    FundingUpdate,
    OIUpdate,
    PriceUpdate,
)

logger = logging.getLogger(__name__)

# Rolling window of completed candles kept per symbol. Sentinel's
# readiness computation only needs ~50 candles, so 500 is deep enough
# for the widest parent-TF windows without blowing up memory.
_CANDLE_HISTORY_MAXLEN = 500


@dataclass
class SymbolState:
    """In-memory real-time state for one symbol.

    Everything a PriceFeed learns from the WebSocket for a single
    symbol lives here. Sentinel / SLTPMonitor / signal agents read from
    this via the ``PriceFeed.get_*`` methods — they never touch the
    dataclass directly so the state container can evolve without
    breaking consumers.
    """

    symbol: str
    latest_price: float | None = None
    latest_bid: float | None = None
    latest_ask: float | None = None
    building_candle: dict | None = None  # partial candle currently being built
    completed_candles: deque = field(
        default_factory=lambda: deque(maxlen=_CANDLE_HISTORY_MAXLEN)
    )
    funding_rate: float | None = None
    open_interest: float | None = None
    last_update: datetime | None = None


class PriceFeed(ABC):
    """Abstract base for real-time exchange data feeds.

    Subclasses own the WebSocket connection and venue-specific message
    parsing. This base class owns per-symbol state and event emission.
    """

    def __init__(self, event_bus: EventBus, exchange_name: str) -> None:
        self._event_bus = event_bus
        self._exchange_name = exchange_name
        self._symbols: dict[str, SymbolState] = {}
        self._connected: bool = False
        self._subscribed_symbols: set[str] = set()
        # Health hooks consumed by RESTFallbackManager (Sprint Week 7
        # Task 4). The base class owns the storage + the firing helper;
        # subclasses (HyperliquidPriceFeed) call ``_fire_on_disconnect``
        # from inside their reconnect logic when they're about to give
        # up so the manager can switch to REST polling. Default values
        # keep both no-ops so feeds without a manager retry forever as
        # before — back-compat with the existing test suite.
        self._on_disconnect_callback: Callable[[], Any] | None = None
        self._max_reconnect_attempts: int | None = None

    def set_on_disconnect_callback(
        self, callback: Callable[[], Any] | None
    ) -> None:
        """Register a callback fired after the feed exhausts its retries.

        ``callback`` may be a sync function or a coroutine function — the
        helper detects coroutines and awaits them. Set to ``None`` to
        clear the hook.
        """
        self._on_disconnect_callback = callback

    def set_max_reconnect_attempts(self, n: int | None) -> None:
        """Cap how many consecutive reconnects a subclass attempts.

        ``None`` (the default) means retry forever — the back-compat
        behaviour. Setting an int makes the subclass call
        ``_fire_on_disconnect`` after that many failures and exit its
        listen loop so the wrapping manager can take over.
        """
        self._max_reconnect_attempts = n

    async def _fire_on_disconnect(self) -> None:
        """Fire the disconnect callback (sync or async, error-safe)."""
        if self._on_disconnect_callback is None:
            return
        try:
            result = self._on_disconnect_callback()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "PriceFeed %s: on_disconnect_callback failed",
                self._exchange_name,
            )

    # ------------------------------------------------------------------
    # Abstract methods — subclasses own connection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Open the WebSocket and start the listen loop."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Cancel the listen loop and close the WebSocket."""

    @abstractmethod
    async def _listen(self) -> None:
        """Infinite WebSocket receive loop. Parses messages and calls ``_update_*``."""

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to the given symbols.

        Must initialise a ``SymbolState`` for each new symbol and
        register it in ``self._subscribed_symbols``.
        """

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from the given symbols and drop their ``SymbolState``."""

    # ------------------------------------------------------------------
    # Concrete reads — sync, zero network calls
    # ------------------------------------------------------------------

    def get_latest_price(self, symbol: str) -> float | None:
        state = self._symbols.get(symbol)
        return state.latest_price if state is not None else None

    def get_latest_candle(self, symbol: str) -> dict | None:
        """Return the most recently completed candle for ``symbol`` or None."""
        state = self._symbols.get(symbol)
        if state is None or not state.completed_candles:
            return None
        return state.completed_candles[-1]

    def get_candle_history(self, symbol: str, count: int) -> list[dict]:
        """Return the most recent ``count`` completed candles, oldest first.

        Returns an empty list for unknown symbols, and fewer than
        ``count`` entries if the buffer hasn't filled yet.
        """
        state = self._symbols.get(symbol)
        if state is None or count <= 0:
            return []
        if count >= len(state.completed_candles):
            return list(state.completed_candles)
        # deque slicing isn't supported — materialise the tail via iteration.
        total = len(state.completed_candles)
        return list(state.completed_candles)[total - count : total]

    def get_funding_rate(self, symbol: str) -> float | None:
        state = self._symbols.get(symbol)
        return state.funding_rate if state is not None else None

    def get_open_interest(self, symbol: str) -> float | None:
        state = self._symbols.get(symbol)
        return state.open_interest if state is not None else None

    def get_symbol_state(self, symbol: str) -> SymbolState | None:
        return self._symbols.get(symbol)

    def is_connected(self) -> bool:
        return self._connected

    @property
    def exchange_name(self) -> str:
        return self._exchange_name

    @property
    def subscribed_symbols(self) -> set[str]:
        return set(self._subscribed_symbols)

    # ------------------------------------------------------------------
    # Concrete mutators — update state AND emit bus events
    # ------------------------------------------------------------------

    def _ensure_state(self, symbol: str) -> SymbolState:
        """Return the existing SymbolState or create a fresh one.

        Subclasses should populate ``self._symbols`` via ``subscribe()``
        before receiving messages, but WebSocket races make it safer to
        lazy-init on first update so no tick is ever dropped.
        """
        state = self._symbols.get(symbol)
        if state is None:
            state = SymbolState(symbol=symbol)
            self._symbols[symbol] = state
        return state

    async def _update_price(
        self,
        symbol: str,
        price: float,
        bid: float | None = None,
        ask: float | None = None,
        size: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Apply a tick to ``SymbolState`` and emit ``PriceUpdated``."""
        ts = timestamp or datetime.now(timezone.utc)
        state = self._ensure_state(symbol)
        state.latest_price = price
        if bid is not None:
            state.latest_bid = bid
        if ask is not None:
            state.latest_ask = ask
        state.last_update = ts

        payload = PriceUpdate(
            symbol=symbol,
            price=price,
            bid=bid,
            ask=ask,
            size=size,
            exchange=self._exchange_name,
            timestamp=ts,
        )
        await self._publish(
            PriceUpdated(source=f"price_feed:{self._exchange_name}", update=payload)
        )

    async def _complete_candle(
        self,
        symbol: str,
        timeframe: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a completed candle to the deque and emit ``CandleClosed``."""
        ts = timestamp or datetime.now(timezone.utc)
        state = self._ensure_state(symbol)
        candle_dict = {
            "timestamp": ts,
            "timeframe": timeframe,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        state.completed_candles.append(candle_dict)
        state.building_candle = None
        state.last_update = ts
        # Keep the latest price in sync if we haven't seen a tick yet —
        # candle close doubles as a synthetic price tick for consumers
        # that only care about closing prices.
        if state.latest_price is None:
            state.latest_price = close

        payload = CandleClose(
            symbol=symbol,
            timeframe=timeframe,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            exchange=self._exchange_name,
            timestamp=ts,
        )
        await self._publish(
            CandleClosed(source=f"price_feed:{self._exchange_name}", candle=payload)
        )

    async def _update_funding(
        self,
        symbol: str,
        funding_rate: float,
        next_funding_time: datetime | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Apply a funding snapshot and emit ``FundingUpdated``."""
        ts = timestamp or datetime.now(timezone.utc)
        state = self._ensure_state(symbol)
        state.funding_rate = funding_rate
        state.last_update = ts

        payload = FundingUpdate(
            symbol=symbol,
            funding_rate=funding_rate,
            next_funding_time=next_funding_time,
            exchange=self._exchange_name,
            timestamp=ts,
        )
        await self._publish(
            FundingUpdated(source=f"price_feed:{self._exchange_name}", update=payload)
        )

    async def _update_oi(
        self,
        symbol: str,
        open_interest: float,
        oi_change_pct: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Apply an OI snapshot and emit ``OpenInterestUpdated``.

        ``oi_change_pct`` may be computed by the caller (relative to the
        previous OI on the same symbol) or by the subclass itself — the
        ABC doesn't derive it because different venues push at different
        cadences.
        """
        ts = timestamp or datetime.now(timezone.utc)
        state = self._ensure_state(symbol)
        state.open_interest = open_interest
        state.last_update = ts

        payload = OIUpdate(
            symbol=symbol,
            open_interest=open_interest,
            oi_change_pct=oi_change_pct,
            exchange=self._exchange_name,
            timestamp=ts,
        )
        await self._publish(
            OpenInterestUpdated(source=f"price_feed:{self._exchange_name}", update=payload)
        )

    async def _publish(self, event) -> None:
        """Publish an event, swallowing any handler error.

        Matches the ``TrackingModule`` pattern: event emission is
        fire-and-forget and must never block the WebSocket receive loop.
        The bus already isolates handler exceptions but a bus-level
        failure (e.g. malformed handler list) would still propagate, so
        we wrap it here as a final safety net.
        """
        try:
            await self._event_bus.publish(event)
        except Exception:
            logger.exception(
                "PriceFeed %s: failed to publish %s",
                self._exchange_name,
                type(event).__name__,
            )
