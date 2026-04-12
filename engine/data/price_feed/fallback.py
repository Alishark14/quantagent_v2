"""RESTFallbackManager — keeps the data flow alive when WebSocket dies.

Sprint Week 7 Task 4. Wraps a ``PriceFeed`` and a REST
``ExchangeAdapter`` so consumers downstream of the bus never see a
data gap even when the WebSocket goes away. The state machine:

    DISCONNECTED ──start()──▶ CONNECTED
                      │
                      └──(connect failed)──▶ FALLBACK

    CONNECTED  ──(ws drop + retries exhausted)──▶ RECONNECTING ──▶ FALLBACK
    FALLBACK   ──(periodic ws reconnect succeeds)──▶ CONNECTED
    any state  ──stop()──▶ DISCONNECTED

Behaviour highlights:

* The fallback REST poll uses the SAME ``_update_*`` / ``_complete_candle``
  helpers on the wrapped PriceFeed, so consumers see ``PriceUpdated`` /
  ``CandleClosed`` / ``FundingUpdated`` / ``OpenInterestUpdated`` events
  on the bus regardless of whether the data came from the WebSocket or
  REST. The source string ("price_feed:hyperliquid") is unchanged.
* REST poll cadence is intentionally TIGHTER (5s) than the legacy
  Sentinel REST loop (30s) — this is a degraded mode and we want
  faster updates, accepting the heavier API load until the WebSocket
  recovers.
* Every ``_rest_poll_check_interval`` seconds (default 30s) the manager
  attempts a fresh ``feed.connect()`` + ``feed.subscribe()``. On success
  the REST loop exits cleanly and the state goes back to CONNECTED.
* The manager tracks state transitions in ``_state_history`` so unit
  tests can verify the RECONNECTING beat is reached even when it's
  immediately followed by FALLBACK in the same handler call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from enum import Enum

from engine.data.price_feed.base import PriceFeed
from engine.events import EventBus
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    CONNECTED = "connected"          # WebSocket healthy
    RECONNECTING = "reconnecting"    # WebSocket down, awaiting recovery
    FALLBACK = "fallback"            # WebSocket failed, REST polling
    DISCONNECTED = "disconnected"    # Intentionally stopped


class RESTFallbackManager:
    """Wraps a ``PriceFeed`` + REST adapter; falls back to polling on WS death."""

    def __init__(
        self,
        price_feed: PriceFeed,
        adapter: ExchangeAdapter,
        event_bus: EventBus,
        *,
        max_reconnect_attempts: int = 5,
        rest_poll_interval: float = 5.0,
        ws_recovery_check_interval: float = 30.0,
    ) -> None:
        self._price_feed = price_feed
        self._adapter = adapter
        self._event_bus = event_bus
        self._max_reconnect_attempts = max_reconnect_attempts
        self._rest_poll_interval = rest_poll_interval
        self._ws_recovery_check_interval = ws_recovery_check_interval

        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._state_history: list[ConnectionState] = [ConnectionState.DISCONNECTED]
        # ``_stopped`` is the source of truth for "user explicitly called
        # stop()". State alone isn't enough because the initial state IS
        # DISCONNECTED before start(), and we need _enter_fallback to work
        # from that initial state on a connect failure during start().
        self._stopped: bool = False
        self._subscribed_symbols: list[str] = []
        self._rest_poll_task: asyncio.Task | None = None
        # Per-symbol last-seen REST candle open-time so we only fire
        # _complete_candle once per actual candle close (not per poll).
        self._last_seen_candle_open_time: dict[str, int] = {}
        # ws-uptime tracking is monotonic-clock based to avoid wall-clock skew.
        self._connect_time: float | None = None
        self._health: dict = {
            "ws_uptime_seconds": 0.0,
            "reconnect_count": 0,
            "fallback_activations": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        return self._state

    def get_state(self) -> ConnectionState:
        return self._state

    def get_health(self) -> dict:
        h = dict(self._health)
        if self._state == ConnectionState.CONNECTED and self._connect_time is not None:
            h["ws_uptime_seconds"] = time.monotonic() - self._connect_time
        return h

    @property
    def state_history(self) -> list[ConnectionState]:
        """Read-only state-transition log for tests/observability."""
        return list(self._state_history)

    async def start(self, symbols: list[str]) -> None:
        """Wire callbacks, attempt initial connect; fall back on failure."""
        self._stopped = False
        self._subscribed_symbols = list(symbols)
        self._price_feed.set_on_disconnect_callback(self._on_ws_disconnect)
        self._price_feed.set_max_reconnect_attempts(self._max_reconnect_attempts)
        try:
            await self._price_feed.connect()
            await self._price_feed.subscribe(symbols)
        except Exception:
            logger.warning(
                "RESTFallbackManager: initial WebSocket connect failed, "
                "entering REST fallback",
                exc_info=True,
            )
            await self._enter_fallback()
            return
        self._record_state(ConnectionState.CONNECTED)
        self._connect_time = time.monotonic()

    async def stop(self) -> None:
        """Cancel the REST poll task (if any) and disconnect the feed."""
        self._stopped = True
        self._record_state(ConnectionState.DISCONNECTED)
        if self._rest_poll_task is not None:
            self._rest_poll_task.cancel()
            try:
                await self._rest_poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._rest_poll_task = None
        try:
            await self._price_feed.disconnect()
        except Exception:
            logger.debug(
                "RESTFallbackManager: error disconnecting price feed", exc_info=True
            )
        # Clear callback so a stray fire after stop doesn't reactivate us.
        try:
            self._price_feed.set_on_disconnect_callback(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _record_state(self, new_state: ConnectionState) -> None:
        self._state = new_state
        self._state_history.append(new_state)

    async def _on_ws_disconnect(self) -> None:
        """Callback fired by the wrapped PriceFeed after retries are exhausted."""
        if self._stopped:
            return
        self._record_state(ConnectionState.RECONNECTING)
        self._health["reconnect_count"] += 1
        # Freeze ws-uptime accumulator so subsequent get_health calls don't
        # keep ticking based on a stale connect_time.
        self._connect_time = None
        await self._enter_fallback()

    async def _enter_fallback(self) -> None:
        """Move into FALLBACK and spawn the REST poll task (idempotent)."""
        if self._stopped:
            return
        if self._state == ConnectionState.FALLBACK:
            return
        self._record_state(ConnectionState.FALLBACK)
        self._health["fallback_activations"] += 1
        logger.warning(
            "RESTFallbackManager: WebSocket down, switching to REST fallback "
            "(poll every %.1fs, ws recovery probe every %.1fs)",
            self._rest_poll_interval,
            self._ws_recovery_check_interval,
        )
        if self._rest_poll_task is None or self._rest_poll_task.done():
            self._rest_poll_task = asyncio.create_task(self._rest_poll_loop())

    # ------------------------------------------------------------------
    # REST polling
    # ------------------------------------------------------------------

    async def _rest_poll_loop(self) -> None:
        """Drive REST polls + periodic WS recovery probes while in FALLBACK."""
        last_recovery_attempt = time.monotonic()
        try:
            while self._state == ConnectionState.FALLBACK:
                for symbol in list(self._subscribed_symbols):
                    if self._state != ConnectionState.FALLBACK:
                        return
                    await self._poll_symbol(symbol)

                # Periodic WS recovery probe — checked AFTER polling so
                # the first iteration runs a poll before retrying the WS.
                now = time.monotonic()
                if now - last_recovery_attempt >= self._ws_recovery_check_interval:
                    last_recovery_attempt = now
                    if await self._try_ws_recovery():
                        return

                try:
                    await asyncio.sleep(self._rest_poll_interval)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("RESTFallbackManager: REST poll loop crashed")

    async def _try_ws_recovery(self) -> bool:
        """Attempt to bring the WebSocket back. Returns True on success."""
        try:
            await self._price_feed.connect()
            await self._price_feed.subscribe(self._subscribed_symbols)
        except Exception:
            logger.debug(
                "RESTFallbackManager: WebSocket recovery probe failed",
                exc_info=True,
            )
            return False
        self._record_state(ConnectionState.CONNECTED)
        self._connect_time = time.monotonic()
        self._health["reconnect_count"] += 1
        logger.info(
            "RESTFallbackManager: WebSocket recovered, exiting REST fallback"
        )
        return True

    async def _poll_symbol(self, symbol: str) -> None:
        """Pull OHLCV / funding / OI for one symbol and feed the PriceFeed."""
        timeframe = getattr(self._price_feed, "_candle_timeframe", "1h")

        # ---- OHLCV ----
        candles: list[dict] = []
        try:
            candles = await self._adapter.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=2,
            )
        except Exception:
            logger.warning(
                "RESTFallbackManager: ohlcv fetch failed for %s", symbol, exc_info=True
            )

        if candles:
            await self._apply_candle(symbol, candles[-1], timeframe)

        # ---- Funding ----
        try:
            funding = await self._adapter.get_funding_rate(symbol)
        except Exception:
            logger.warning(
                "RESTFallbackManager: funding fetch failed for %s",
                symbol,
                exc_info=True,
            )
            funding = None
        if funding is not None:
            try:
                await self._price_feed._update_funding(
                    symbol=symbol, funding_rate=float(funding)
                )
            except (TypeError, ValueError):
                pass

        # ---- Open Interest ----
        try:
            oi = await self._adapter.get_open_interest(symbol)
        except Exception:
            logger.warning(
                "RESTFallbackManager: oi fetch failed for %s", symbol, exc_info=True
            )
            oi = None
        if oi is not None:
            try:
                await self._price_feed._update_oi(
                    symbol=symbol, open_interest=float(oi)
                )
            except (TypeError, ValueError):
                pass

    async def _apply_candle(self, symbol: str, raw: dict, timeframe: str) -> None:
        """Push the latest REST candle through PriceFeed helpers."""
        ts_ms = raw.get("timestamp")
        if isinstance(ts_ms, (int, float)):
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        elif isinstance(ts_ms, datetime):
            ts = ts_ms
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts = datetime.now(timezone.utc)
            ts_ms = None

        try:
            close_price = float(raw["close"])
        except (KeyError, TypeError, ValueError):
            return

        # Always emit a price update so SLTPMonitor's tick path stays warm.
        await self._price_feed._update_price(
            symbol=symbol, price=close_price, timestamp=ts
        )

        # If this is a NEW completed candle, also flush it via _complete_candle
        # so the deque + CandleClosed event stay consistent with WS behaviour.
        last_seen = self._last_seen_candle_open_time.get(symbol)
        if ts_ms is None:
            return
        if last_seen is not None and int(ts_ms) <= last_seen:
            return
        try:
            await self._price_feed._complete_candle(
                symbol=symbol,
                timeframe=timeframe,
                open_price=float(raw["open"]),
                high=float(raw["high"]),
                low=float(raw["low"]),
                close=close_price,
                volume=float(raw.get("volume", 0) or 0),
                timestamp=ts,
            )
            self._last_seen_candle_open_time[symbol] = int(ts_ms)
        except (KeyError, TypeError, ValueError):
            return
