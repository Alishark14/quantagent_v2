"""HyperliquidPriceFeed — concrete WebSocket PriceFeed for Hyperliquid.

Sprint Week 7 Task 3. Wraps a single ``wss://api.hyperliquid.xyz/ws``
connection and multiplexes three subscription types per symbol:

    * ``trades``          — tick price feed
    * ``candle``          — OHLCV with close detection on open-time advance
    * ``activeAssetCtx``  — funding rate + open interest snapshots

On every message the appropriate ``_update_*`` / ``_complete_candle``
helper on the ABC runs, which both mutates ``SymbolState`` and
publishes an event on the Event Bus so downstream consumers
(``Sentinel``, ``SLTPMonitor``, signal agents) never see REST calls.

Reconnection is handled inline in the listen loop with exponential
backoff (1s → 2s → 4s → ... capped at 60s). On reconnect the feed
re-sends every subscription for every symbol in ``_subscribed_symbols``
so a dropped connection is invisible to consumers.

Symbol mapping: internal symbols are ``BASE-USDC`` (e.g. ``BTC-USDC``,
``GOLD-USDC``). Hyperliquid's WebSocket uses coin names (``BTC``) for
native perps and a ``xyz:<DEPLOYER_SYMBOL>`` prefix for HIP-3 tokens
(``xyz:GOLD``, ``xyz:CL``). The mapping is derived at construction
from ``exchanges.hyperliquid.SYMBOL_MAP`` so aliased HIP-3 tokens
(``WTIOIL-USDC → xyz:CL``) resolve correctly — a naive ``split("-")``
would produce ``WTIOIL`` and break.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed

from engine.data.price_feed.base import PriceFeed, SymbolState
from engine.events import EventBus
from exchanges.base import ExchangeAdapter
from exchanges.hyperliquid import SYMBOL_MAP

logger = logging.getLogger(__name__)


# Type alias for the ws-connect factory injected by tests. Any awaitable
# that takes a URL string and returns an async-iterable send/close object
# works — the production default is ``websockets.connect``.
WSConnectFactory = Callable[[str], Awaitable[Any]]


class HyperliquidPriceFeed(PriceFeed):
    """Concrete Hyperliquid WebSocket PriceFeed."""

    MAINNET_WS = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"

    def __init__(
        self,
        event_bus: EventBus,
        testnet: bool = False,
        candle_timeframe: str = "1h",
        *,
        bootstrap_adapter: ExchangeAdapter | None = None,
        bootstrap_count: int = 100,
        ws_connect: WSConnectFactory | None = None,
    ) -> None:
        super().__init__(event_bus, exchange_name="hyperliquid")
        self._ws_url = self.TESTNET_WS if testnet else self.MAINNET_WS
        self._ws: Any | None = None
        self._candle_timeframe = candle_timeframe
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._listen_task: asyncio.Task | None = None
        self._symbol_to_coin: dict[str, str] = {}
        self._coin_to_symbol: dict[str, str] = {}
        # Open-time (ms) of the candle currently being built per symbol.
        # Hyperliquid's `candle` channel pushes the same candle repeatedly
        # with updated OHLC as ticks arrive; we detect close by observing
        # an open-time advance.
        self._building_candle_open_time: dict[str, int] = {}
        # Test injection point — defaults to the real websockets.connect.
        self._ws_connect: WSConnectFactory = ws_connect or websockets.connect
        self._stop: bool = False
        # REST adapter used to seed `completed_candles` from history before
        # the WebSocket starts pushing candle closes. Without this, Sentinel
        # would wait 50+ candle periods (50h on a 1h chart) before the deque
        # is deep enough to compute readiness. None disables bootstrapping.
        self._bootstrap_adapter: ExchangeAdapter | None = bootstrap_adapter
        self._bootstrap_count: int = bootstrap_count
        self._bootstrapped_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Symbol → coin mapping
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_coin(symbol: str) -> str:
        """Resolve an internal symbol (``BTC-USDC``) to a Hyperliquid coin.

        Native perps → ``BTC`` / ``ETH`` / ``SOL``.
        HIP-3 tokens → ``xyz:GOLD`` / ``xyz:CL`` / ``xyz:SP500``.

        For unknown symbols, falls back to ``symbol.split("-")[0]`` —
        good enough for future native perps that haven't been added to
        SYMBOL_MAP yet.
        """
        ccxt_sym = SYMBOL_MAP.get(symbol)
        if ccxt_sym is not None:
            base = ccxt_sym.split("/")[0]  # "BTC" or "XYZ-GOLD"
            if base.startswith("XYZ-"):
                return f"xyz:{base[4:]}"
            return base
        return symbol.split("-")[0] if "-" in symbol else symbol

    def _subscription_messages(self, coin: str, *, method: str) -> list[dict]:
        """Build the three subscription messages for a single coin.

        ``method`` is ``"subscribe"`` or ``"unsubscribe"`` — the payload
        shape is identical for both.
        """
        return [
            {"method": method, "subscription": {"type": "trades", "coin": coin}},
            {
                "method": method,
                "subscription": {
                    "type": "candle",
                    "coin": coin,
                    "interval": self._candle_timeframe,
                },
            },
            {"method": method, "subscription": {"type": "activeAssetCtx", "coin": coin}},
        ]

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket and start the listen loop."""
        await self._open_ws()
        self._stop = False
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self._listen())

    async def disconnect(self) -> None:
        """Cancel the listen loop and close the WebSocket."""
        self._stop = True
        self._connected = False
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listen_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("HyperliquidPriceFeed: error closing WebSocket", exc_info=True)
            self._ws = None

    async def _open_ws(self) -> None:
        """Open a fresh WebSocket connection.

        The reconnect backoff is NOT reset here — it's reset by
        ``_inline_reconnect`` after a successful reconnect + resubscribe
        completes. Keeping the reset out of the raw connect step lets
        tests override ``_reconnect_delay`` before ``connect()`` without
        having the initial ``_open_ws`` stomp it back to 1.0.
        """
        self._ws = await self._ws_connect(self._ws_url)
        self._connected = True
        logger.info("HyperliquidPriceFeed connected to %s", self._ws_url)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to ``trades`` / ``candle`` / ``activeAssetCtx`` for each symbol.

        Order of operations is deliberate:

        1. Register ``SymbolState`` + coin mapping for every new symbol.
        2. Bootstrap historical candles via REST (if a bootstrap adapter
           was injected) — *before* sending WS subscribe frames so the
           deque is fully seeded before any ``CandleClosed`` event can
           append a fresh candle. Otherwise a fast WS push could land in
           the deque before historical bars and break the time ordering
           that consumers like Sentinel depend on.
        3. Send the 3 subscribe frames per symbol.
        """
        new_symbols: list[str] = []
        for symbol in symbols:
            coin = self.resolve_coin(symbol)
            self._symbol_to_coin[symbol] = coin
            self._coin_to_symbol[coin] = symbol
            if symbol not in self._symbols:
                self._symbols[symbol] = SymbolState(symbol=symbol)
                new_symbols.append(symbol)
            self._subscribed_symbols.add(symbol)

        # Bootstrap REST history for symbols we haven't seeded yet.
        if self._bootstrap_adapter is not None and new_symbols:
            unseeded = [s for s in new_symbols if s not in self._bootstrapped_symbols]
            if unseeded:
                await self._bootstrap_history(
                    self._bootstrap_adapter,
                    symbols=unseeded,
                    count=self._bootstrap_count,
                )

        if self._ws is None:
            return

        for symbol in symbols:
            coin = self._symbol_to_coin[symbol]
            for msg in self._subscription_messages(coin, method="subscribe"):
                try:
                    await self._ws.send(json.dumps(msg))
                except Exception:
                    logger.warning(
                        "HyperliquidPriceFeed: failed to send subscribe for %s (%s)",
                        symbol,
                        msg["subscription"]["type"],
                        exc_info=True,
                    )

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe and drop ``SymbolState`` + mapping entries."""
        for symbol in symbols:
            coin = self._symbol_to_coin.pop(symbol, None)
            if coin is not None:
                self._coin_to_symbol.pop(coin, None)
                if self._ws is not None:
                    for msg in self._subscription_messages(coin, method="unsubscribe"):
                        try:
                            await self._ws.send(json.dumps(msg))
                        except Exception:
                            logger.warning(
                                "HyperliquidPriceFeed: failed to send unsubscribe for %s",
                                symbol,
                                exc_info=True,
                            )
            self._symbols.pop(symbol, None)
            self._subscribed_symbols.discard(symbol)
            self._building_candle_open_time.pop(symbol, None)
            self._bootstrapped_symbols.discard(symbol)

    async def _bootstrap_history(
        self,
        adapter: ExchangeAdapter,
        *,
        symbols: list[str] | None = None,
        count: int = 100,
    ) -> None:
        """Seed ``completed_candles`` from REST so consumers don't wait N hours.

        Without this, Sentinel's 50-candle readiness window starts empty
        and would block setup detection for 50 candle periods (50 hours
        on a 1h chart) after every restart. The bootstrap pulls the
        latest ``count`` candles via the adapter's ``fetch_ohlcv``,
        normalises them into the same dict shape ``_complete_candle``
        produces, and appends them to the per-symbol deque.

        Per-symbol failures are logged-and-swallowed: a single 4xx for
        one symbol shouldn't block the rest of the portfolio. The
        WebSocket will accumulate fresh candles over time regardless.

        ``symbols`` defaults to every currently subscribed symbol; the
        ``subscribe()`` flow passes only newly added symbols so a second
        ``subscribe()`` call doesn't re-pull history for symbols that
        already have it.
        """
        target = list(symbols) if symbols is not None else list(self._subscribed_symbols)
        for symbol in target:
            try:
                raw_candles = await adapter.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self._candle_timeframe,
                    limit=count,
                )
            except Exception:
                logger.warning(
                    "HyperliquidPriceFeed: bootstrap fetch failed for %s",
                    symbol,
                    exc_info=True,
                )
                continue

            if not raw_candles:
                logger.warning(
                    "HyperliquidPriceFeed: bootstrap returned no candles for %s",
                    symbol,
                )
                continue

            state = self._symbols.get(symbol)
            if state is None:
                continue

            seeded = 0
            for raw in raw_candles:
                normalised = self._normalise_bootstrap_candle(raw)
                if normalised is not None:
                    state.completed_candles.append(normalised)
                    seeded += 1

            if seeded > 0:
                # Seed the latest_price too so consumers don't get None
                # before the first WebSocket tick lands.
                if state.latest_price is None:
                    state.latest_price = state.completed_candles[-1]["close"]
                self._bootstrapped_symbols.add(symbol)
                logger.info(
                    "HyperliquidPriceFeed: bootstrapped %d candles for %s",
                    seeded,
                    symbol,
                )

    def _normalise_bootstrap_candle(self, raw: dict) -> dict | None:
        """Convert a ``fetch_ohlcv`` dict to the WebSocket candle shape.

        ``fetch_ohlcv`` returns ``timestamp`` as a millisecond int and
        omits ``timeframe``; ``_complete_candle`` produces dicts with a
        ``datetime`` ``timestamp`` and an explicit ``timeframe``. This
        normaliser bridges the two so consumers reading
        ``get_candle_history`` see one consistent shape regardless of
        whether the candle came from REST or WebSocket.

        Returns ``None`` for unparseable rows so the caller can skip
        them without crashing the bootstrap loop.
        """
        try:
            ts_value = raw.get("timestamp")
            if isinstance(ts_value, datetime):
                ts = ts_value
            elif isinstance(ts_value, (int, float)):
                ts = datetime.fromtimestamp(ts_value / 1000, tz=timezone.utc)
            else:
                return None
            return {
                "timestamp": ts,
                "timeframe": self._candle_timeframe,
                "open": float(raw["open"]),
                "high": float(raw["high"]),
                "low": float(raw["low"]),
                "close": float(raw["close"]),
                "volume": float(raw.get("volume", 0) or 0),
            }
        except (KeyError, TypeError, ValueError):
            return None

    async def _resubscribe_all(self) -> None:
        """Re-send every subscription for every symbol (used on reconnect)."""
        symbols = list(self._subscribed_symbols)
        if not symbols or self._ws is None:
            return
        for symbol in symbols:
            coin = self._symbol_to_coin.get(symbol) or self.resolve_coin(symbol)
            self._symbol_to_coin[symbol] = coin
            self._coin_to_symbol[coin] = symbol
            for msg in self._subscription_messages(coin, method="subscribe"):
                try:
                    await self._ws.send(json.dumps(msg))
                except Exception:
                    logger.warning(
                        "HyperliquidPriceFeed: failed to resubscribe %s",
                        symbol,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Listen loop + reconnection
    # ------------------------------------------------------------------

    async def _listen(self) -> None:
        """Receive messages until stopped; reconnect inline on disconnects."""
        while not self._stop:
            if self._ws is None:
                # No connection to listen to — break so caller can resurrect.
                return
            try:
                async for raw in self._ws:
                    await self._handle_message(raw)
                # Clean exit from `async for` means the server closed the
                # connection without raising — treat as a disconnect.
            except ConnectionClosed:
                logger.warning("HyperliquidPriceFeed: connection closed")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("HyperliquidPriceFeed: listen loop error")

            if self._stop:
                return

            # Reconnection loop — exponential backoff + resubscribe.
            await self._inline_reconnect()

    async def _inline_reconnect(self) -> None:
        """Close the dead ws, sleep with backoff, reopen, re-subscribe.

        Lives inside ``_listen`` so the same task owns the full connection
        lifecycle — avoids the footgun of spawning a second listen task
        from inside the first one on reconnect.

        Honours ``self._max_reconnect_attempts`` (set by the wrapping
        ``RESTFallbackManager`` via the base class setter): once that many
        consecutive reconnect attempts fail, fires the
        ``on_disconnect_callback`` and returns so the listen loop exits
        and the manager can take over with REST polling. ``None`` (the
        default) preserves the legacy retry-forever behaviour.
        """
        self._connected = False
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            logger.debug("HyperliquidPriceFeed: error closing stale ws", exc_info=True)
        finally:
            self._ws = None

        attempts = 0
        while not self._stop:
            if (
                self._max_reconnect_attempts is not None
                and attempts >= self._max_reconnect_attempts
            ):
                logger.warning(
                    "HyperliquidPriceFeed: %d reconnect attempts exhausted, "
                    "firing on_disconnect_callback",
                    attempts,
                )
                await self._fire_on_disconnect()
                return

            attempts += 1
            delay = self._reconnect_delay
            logger.warning(
                "HyperliquidPriceFeed reconnecting in %.1fs (attempt %d)...",
                delay,
                attempts,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )
            try:
                await self._open_ws()
            except Exception:
                logger.warning(
                    "HyperliquidPriceFeed: reconnect attempt failed", exc_info=True
                )
                continue
            try:
                await self._resubscribe_all()
            except Exception:
                logger.warning(
                    "HyperliquidPriceFeed: resubscribe failed after reconnect",
                    exc_info=True,
                )
                # Keep the connection anyway — the next iteration of the
                # outer listen loop will pull whatever messages the server
                # already sent for coins that *are* subscribed.
            return

    # ------------------------------------------------------------------
    # Message parsing — routed per channel
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: Any) -> None:
        """Decode a WebSocket frame and dispatch by channel.

        Never crashes on malformed input — every failure is logged at
        warning and swallowed so the WebSocket loop keeps running.
        """
        try:
            if isinstance(raw, (bytes, bytearray)):
                payload = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, str):
                payload = json.loads(raw)
            elif isinstance(raw, dict):
                payload = raw  # tests pass pre-parsed dicts
            else:
                return
        except Exception:
            logger.warning(
                "HyperliquidPriceFeed: failed to parse frame", exc_info=True
            )
            return

        if not isinstance(payload, dict):
            return

        channel = payload.get("channel")
        data = payload.get("data")

        try:
            if channel == "trades":
                await self._handle_trades(data)
            elif channel == "candle":
                await self._handle_candle(data)
            elif channel == "activeAssetCtx":
                await self._handle_active_asset_ctx(data)
            elif channel in ("subscriptionResponse", "pong", "error", None):
                # Acks / heartbeats / error envelopes are not fatal.
                return
            # Unknown channels — ignore silently.
        except Exception:
            logger.warning(
                "HyperliquidPriceFeed: error handling channel=%s", channel, exc_info=True
            )

    async def _handle_trades(self, data: Any) -> None:
        if not isinstance(data, list):
            return
        for trade in data:
            if not isinstance(trade, dict):
                continue
            coin = trade.get("coin")
            symbol = self._coin_to_symbol.get(coin) if isinstance(coin, str) else None
            if symbol is None:
                continue
            try:
                price = float(trade["px"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                size = float(trade["sz"]) if trade.get("sz") is not None else None
            except (TypeError, ValueError):
                size = None
            ts = _parse_ms_timestamp(trade.get("time"))
            await self._update_price(symbol, price=price, size=size, timestamp=ts)

    async def _handle_candle(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        coin = data.get("s")
        symbol = self._coin_to_symbol.get(coin) if isinstance(coin, str) else None
        if symbol is None:
            return
        open_time = data.get("t")
        if not isinstance(open_time, (int, float)):
            return
        try:
            o = float(data["o"])
            h = float(data["h"])
            low = float(data["l"])
            c = float(data["c"])
            v = float(data.get("v", 0) or 0)
        except (KeyError, TypeError, ValueError):
            return

        timeframe = data.get("i") or self._candle_timeframe
        state = self._ensure_state(symbol)
        prev_open_time = self._building_candle_open_time.get(symbol)
        candle_ts = _parse_ms_timestamp(open_time)

        new_building = {
            "timestamp": candle_ts,
            "timeframe": timeframe,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "volume": v,
        }

        if prev_open_time is None:
            # First observation for this symbol — just start tracking.
            self._building_candle_open_time[symbol] = int(open_time)
            state.building_candle = new_building
            return

        if int(open_time) > prev_open_time:
            # Previous candle is now complete — flush it.
            prev = state.building_candle
            if prev is not None:
                await self._complete_candle(
                    symbol,
                    timeframe=prev.get("timeframe", timeframe),
                    open_price=prev["open"],
                    high=prev["high"],
                    low=prev["low"],
                    close=prev["close"],
                    volume=prev["volume"],
                    timestamp=prev["timestamp"],
                )
            self._building_candle_open_time[symbol] = int(open_time)
            state.building_candle = new_building
            return

        # Same candle, updated OHLC — refresh the partial.
        state.building_candle = new_building

    async def _handle_active_asset_ctx(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        coin = data.get("coin")
        symbol = self._coin_to_symbol.get(coin) if isinstance(coin, str) else None
        if symbol is None:
            return
        ctx = data.get("ctx")
        if not isinstance(ctx, dict):
            return

        funding_raw = ctx.get("funding")
        try:
            funding = float(funding_raw) if funding_raw is not None else None
        except (TypeError, ValueError):
            funding = None
        if funding is not None:
            await self._update_funding(symbol, funding_rate=funding)

        oi_raw = ctx.get("openInterest")
        try:
            oi = float(oi_raw) if oi_raw is not None else None
        except (TypeError, ValueError):
            oi = None
        if oi is not None:
            state = self._ensure_state(symbol)
            prev = state.open_interest
            oi_change_pct: float | None = None
            if prev is not None and prev != 0:
                oi_change_pct = (oi - prev) / prev
            await self._update_oi(
                symbol, open_interest=oi, oi_change_pct=oi_change_pct
            )


def _parse_ms_timestamp(value: Any) -> datetime:
    """Convert an int/float millisecond epoch to a UTC datetime.

    Returns ``datetime.now(tz=utc)`` on anything unparseable — the caller
    has no better signal to fall back on, and the wall-clock timestamp is
    still useful for ordering.
    """
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            pass
    return datetime.now(timezone.utc)
