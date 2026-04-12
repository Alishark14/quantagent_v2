"""CryptoFlowProvider: funding rate, OI, liquidation data.

Fetches crypto-specific positioning data from the exchange adapter.
Code-only, zero LLM cost. All data is factual/deterministic.

OI history buffer
=================

The provider maintains a per-symbol in-memory deque of
``(timestamp_seconds, oi_value)`` snapshots. On each ``fetch()`` call
the current OI is appended. When the buffer spans the configured
lookback window the provider computes ``oi_change_4h`` (a fractional
delta over that window — the field name is historical, not literally
"4h") and derives ``oi_trend`` ("BUILDING" / "DROPPING" / "STABLE").
During the cold-start warmup window (< full lookback of data) both
fields stay ``None`` / ``"STABLE"`` — ``FlowSignalAgent`` handles
``None`` correctly by falling through to NEUTRAL.

The buffer is keyed by symbol so a single provider instance can serve
multiple bots (or a multi-symbol bot) without cross-contamination. The
deque maxlen scales with the lookback window assuming one snapshot per
30-second Sentinel poll: ``maxlen = lookback_seconds // 30``.

Configurable lookback
=====================

The lookback window is configurable per provider via either the
``lookback_seconds`` constructor arg or the
``set_lookback_for_timeframe(tf)`` helper, which maps a bot timeframe
to a sensible 2× multiple. The default is 2 hours so 1h timeframe bots
detect divergences while they still matter (the previous hardcoded
4-hour window meant a divergence on a 1h chart was already two candles
old by the time the rule could fire).

Persistence
===========

When an ``OISnapshotRepository`` is wired in, every fresh snapshot is
also written to the ``oi_snapshots`` table, and ``warmup_from_repo()``
bulk-loads recent snapshots into the deques on startup. This makes the
multi-hour cold-start penalty a one-time cost across the entire
deployment instead of paying it on every restart.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone

from engine.data.flow.base import FlowProvider
from exchanges.base import ExchangeAdapter
from storage.repositories.base import OISnapshotRepository

logger = logging.getLogger(__name__)

# Funding rate thresholds for signal classification
_CROWDED_LONG_THRESHOLD = 0.01   # > +0.01% = crowded long
_CROWDED_SHORT_THRESHOLD = -0.01  # < -0.01% = crowded short

# Defaults: 2-hour lookback @ 30s Sentinel scan = 240-entry deque.
_DEFAULT_OI_LOOKBACK_SECONDS = 7_200
_OI_TREND_THRESHOLD = 0.02     # ±2% to classify BUILDING / DROPPING

# Map a bot timeframe to a 2× lookback in seconds. Anything not in this
# map falls through to the default. The 2× factor is the rule of thumb
# we settled on after the previous fixed 4h window was eating signal
# freshness on 15m / 30m / 1h bots.
_TIMEFRAME_LOOKBACK_SECONDS: dict[str, int] = {
    "15m": 1_800,    # 30 minutes
    "30m": 3_600,    # 1 hour
    "1h": 7_200,     # 2 hours
    "4h": 28_800,    # 8 hours
    "1d": 172_800,   # 48 hours
}

# Backward-compat aliases — pre-existing tests import these names. They
# now reflect the new defaults (2h / 240) but keep the same identifiers
# so the test fixtures that use them as relative offsets still work.
_OI_LOOKBACK_SECONDS = _DEFAULT_OI_LOOKBACK_SECONDS
_OI_BUFFER_MAXLEN = _DEFAULT_OI_LOOKBACK_SECONDS // 30


def _maxlen_for_lookback(lookback_seconds: int) -> int:
    """Pick a deque maxlen that safely holds the full lookback window.

    When a WebSocket PriceFeed is wired, ``_on_oi_update`` pushes
    snapshots every ~3 seconds (from Hyperliquid's ``activeAssetCtx``
    channel). The old formula ``lookback // 30`` assumed 30-second
    Sentinel polls and produced a 240-entry deque for a 2h lookback —
    WebSocket pushes filled it in ~12 minutes, evicting all warmed
    historical data and keeping ``_compute_oi_delta`` permanently in
    the cold-start path. Using ``lookback_seconds`` directly gives one
    entry per second of lookback — plenty of headroom for any push
    rate while staying trivial in memory (~16 bytes per entry).
    """
    return max(600, lookback_seconds)


class CryptoFlowProvider(FlowProvider):
    """Fetches funding rate and open interest from crypto exchange adapters.

    Maintains a per-symbol OI history buffer so ``oi_change_4h`` and
    ``oi_trend`` can be computed from live data rather than being
    hardcoded to ``None`` / ``"STABLE"``.
    """

    def __init__(
        self,
        lookback_seconds: int | None = None,
        oi_repo: OISnapshotRepository | None = None,
        price_feed=None,
        event_bus=None,
    ) -> None:
        # symbol → deque of (timestamp_seconds, oi_value)
        self._oi_history: dict[str, deque[tuple[float, float]]] = {}
        # Track which symbols have logged the "buffer warm" message
        self._oi_warm_logged: set[str] = set()

        self._lookback_seconds: int = (
            int(lookback_seconds)
            if lookback_seconds is not None
            else _DEFAULT_OI_LOOKBACK_SECONDS
        )
        self._buffer_maxlen: int = _maxlen_for_lookback(self._lookback_seconds)

        # Optional persistence backing — None disables both warmup
        # bulk-load and per-fetch INSERT. Tests + ad-hoc spawns leave
        # it None; production wires it via main.py.
        self._oi_repo = oi_repo

        # Sprint Week 7: optional PriceFeed for memory-first reads.
        # When wired, ``fetch()`` reads funding rate and OI from PriceFeed
        # memory (populated by WebSocket pushes) instead of making REST
        # calls. Falls back to adapter REST when PriceFeed returns None
        # (e.g. no WebSocket data for that symbol yet).
        self._price_feed = price_feed

        # When both PriceFeed and an EventBus are provided, subscribe to
        # ``OpenInterestUpdated`` events so the OI history buffer fills
        # from WebSocket pushes (every few seconds) instead of only from
        # pipeline ``fetch()`` calls (once per Sentinel cycle). This
        # eliminates the multi-hour cold-start penalty after restarts
        # even without the OI snapshot table warmup.
        self._event_bus = event_bus
        if event_bus is not None and price_feed is not None:
            from engine.events import OpenInterestUpdated
            event_bus.subscribe(OpenInterestUpdated, self._on_oi_update)

    def set_price_feed(self, price_feed, event_bus=None) -> None:
        """Wire a PriceFeed after construction (for late-binding in main.py).

        When the PriceFeed is created after ``_make_bot_factory`` (which
        captures the shared CryptoFlowProvider in a closure), main.py
        calls this setter so the provider can start reading funding / OI
        from memory and subscribing to ``OpenInterestUpdated`` events.
        Idempotent — calling twice with the same feed is a no-op.
        """
        if price_feed is self._price_feed:
            return
        self._price_feed = price_feed
        if event_bus is not None and price_feed is not None:
            from engine.events import OpenInterestUpdated
            event_bus.subscribe(OpenInterestUpdated, self._on_oi_update)
            self._event_bus = event_bus

    def name(self) -> str:
        return "crypto"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_lookback_for_timeframe(self, timeframe: str) -> None:
        """Reconfigure the lookback window from a bot timeframe string.

        Resets per-symbol deques because their previous ``maxlen`` was
        sized for the old lookback. The deques will refill from the next
        ``fetch()`` (and from ``warmup_from_repo()`` on the next call).
        Call this BEFORE warmup so the bulk-load uses the right window.
        """
        new_lookback = _TIMEFRAME_LOOKBACK_SECONDS.get(
            timeframe, _DEFAULT_OI_LOOKBACK_SECONDS
        )
        self._lookback_seconds = new_lookback
        self._buffer_maxlen = _maxlen_for_lookback(new_lookback)
        self._oi_history.clear()
        self._oi_warm_logged.clear()
        logger.info(
            f"CryptoFlowProvider: lookback set for timeframe={timeframe!r} → "
            f"{new_lookback}s ({self._buffer_maxlen} snapshot maxlen)"
        )

    @property
    def lookback_seconds(self) -> int:
        return self._lookback_seconds

    # ------------------------------------------------------------------
    # Persistence integration
    # ------------------------------------------------------------------

    async def warmup_from_repo(self) -> int:
        """Populate per-symbol deques from the OI snapshot table.

        Loads every snapshot newer than ``now - lookback_seconds`` and
        appends them to the per-symbol deques in chronological order.
        Returns the total number of snapshots loaded so callers can log
        a meaningful banner. No-op (returns 0) when no repo is wired.

        Errors are logged and swallowed — a warmup failure must NOT
        block startup. The provider falls back to the legacy cold-start
        path if the bulk-load can't run.
        """
        if self._oi_repo is None:
            return 0
        # Query 1.5× the lookback so the oldest loaded snapshot is
        # provably OLDER than the cold-start cutoff (`now - lookback`).
        # Without the over-fetch the bulk-load returns entries strictly
        # younger than the cutoff and the cold-start guard in
        # ``_compute_oi_delta`` would still fire on the very next
        # ``fetch()``, defeating the whole point of persistence.
        warmup_window = int(self._lookback_seconds * 1.5)
        try:
            rows = await self._oi_repo.get_recent_snapshots(warmup_window)
        except Exception:
            logger.exception(
                "CryptoFlowProvider: warmup_from_repo bulk-load failed; "
                "falling back to cold start"
            )
            return 0

        loaded = 0
        for row in rows:
            symbol = row["symbol"]
            ts = float(row["timestamp"])
            oi = float(row["oi_value"])
            if symbol not in self._oi_history:
                self._oi_history[symbol] = deque(maxlen=self._buffer_maxlen)
            self._oi_history[symbol].append((ts, oi))
            loaded += 1

        if loaded > 0:
            logger.info(
                f"CryptoFlowProvider: warmed {loaded} snapshots across "
                f"{len(self._oi_history)} symbols from oi_snapshots table"
            )
        return loaded

    # ------------------------------------------------------------------
    # FlowProvider contract
    # ------------------------------------------------------------------

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Fetch funding rate and OI from exchange adapter.

        When a ``PriceFeed`` is wired, reads are served from PriceFeed
        memory (populated by WebSocket pushes) with a transparent REST
        fallback for any symbol that doesn't have a cached value yet.
        Each data point is independently try/excepted — one failure
        does not block the other.
        """
        result: dict = {}

        # Funding rate — PriceFeed memory first, REST fallback
        try:
            rate = await self._read_funding(symbol, adapter)
            if rate is not None:
                result["funding_rate"] = rate
                if rate > _CROWDED_LONG_THRESHOLD:
                    result["funding_signal"] = "CROWDED_LONG"
                elif rate < _CROWDED_SHORT_THRESHOLD:
                    result["funding_signal"] = "CROWDED_SHORT"
                else:
                    result["funding_signal"] = "NEUTRAL"
        except Exception as e:
            logger.warning(f"CryptoFlowProvider: funding rate unavailable for {symbol}: {e}")

        # Open Interest — PriceFeed memory first, REST fallback
        try:
            oi = await self._read_oi(symbol, adapter)
            if oi is not None:
                result["open_interest"] = oi
                await self._record_oi_snapshot(symbol, float(oi))
                now = time.time()
                oi_change, oi_trend = self._compute_oi_delta(symbol, now, float(oi))
                result["oi_change_4h"] = oi_change
                result["oi_trend"] = oi_trend
                buf = self._oi_history.get(symbol)
                logger.debug(
                    f"CryptoFlowProvider {symbol}: deque_len={len(buf) if buf else 0}, "
                    f"lookback_s={self._lookback_seconds}, oi_change={oi_change}"
                )
        except Exception as e:
            logger.warning(f"CryptoFlowProvider: OI unavailable for {symbol}: {e}")

        return result

    async def _read_funding(
        self, symbol: str, adapter: ExchangeAdapter
    ) -> float | None:
        """Read funding rate from PriceFeed memory, REST fallback."""
        if self._price_feed is not None:
            pf_rate = self._price_feed.get_funding_rate(symbol)
            if pf_rate is not None:
                return pf_rate
        return await adapter.get_funding_rate(symbol)

    async def _read_oi(
        self, symbol: str, adapter: ExchangeAdapter
    ) -> float | None:
        """Read open interest from PriceFeed memory, REST fallback."""
        if self._price_feed is not None:
            pf_oi = self._price_feed.get_open_interest(symbol)
            if pf_oi is not None:
                return pf_oi
        return await adapter.get_open_interest(symbol)

    async def _record_oi_snapshot(
        self, symbol: str, oi_value: float, ts: float | None = None
    ) -> None:
        """Append an OI observation to the history buffer + persistence.

        Used by both ``fetch()`` (on every pipeline cycle) and
        ``_on_oi_update()`` (on every WebSocket push). The in-memory
        deque is bounded by ``maxlen`` so duplicate entries from both
        paths are harmless — the buffer just fills faster, and
        ``_compute_oi_delta`` is time-based not count-based.
        """
        now = ts if ts is not None else time.time()
        if symbol not in self._oi_history:
            self._oi_history[symbol] = deque(maxlen=self._buffer_maxlen)
        self._oi_history[symbol].append((now, oi_value))

        if self._oi_repo is not None:
            try:
                await self._oi_repo.insert_snapshot(
                    symbol,
                    datetime.fromtimestamp(now, tz=timezone.utc),
                    float(oi_value),
                )
            except Exception:
                logger.debug(
                    "CryptoFlowProvider: insert_snapshot failed for %s",
                    symbol,
                    exc_info=True,
                )

    async def _on_oi_update(self, event) -> None:
        """Receive pushed OI from PriceFeed — feed directly into the history buffer.

        Subscribed to ``OpenInterestUpdated`` events when both
        ``price_feed`` and ``event_bus`` are wired. Each push appends to
        the per-symbol deque so the buffer fills continuously from
        WebSocket data rather than waiting for the pipeline's
        ``fetch()`` cycle.
        """
        payload = getattr(event, "update", None)
        if payload is None:
            return
        symbol = payload.symbol
        oi = payload.open_interest
        ts = payload.timestamp
        epoch = ts.timestamp() if isinstance(ts, datetime) else time.time()
        await self._record_oi_snapshot(symbol, float(oi), epoch)

    def _compute_oi_delta(
        self, symbol: str, now: float, current_oi: float
    ) -> tuple[float | None, str]:
        """Compute oi_change and oi_trend from the history buffer.

        Returns (None, "STABLE") during the warmup window when the
        buffer doesn't span the configured lookback yet.
        """
        buf = self._oi_history.get(symbol)
        if not buf or len(buf) < 2:
            return None, "STABLE"

        cutoff = now - self._lookback_seconds

        # Find the oldest entry at or after the lookback cutoff.
        # The deque is ordered by time (append-only), so scan from left.
        old_ts, old_oi = buf[0]
        if old_ts > cutoff:
            # Buffer doesn't span the lookback yet — cold start
            return None, "STABLE"

        # Walk forward to find the entry closest to (but >= cutoff)
        best_ts, best_oi = old_ts, old_oi
        for ts, oi_val in buf:
            if ts >= cutoff:
                best_ts, best_oi = ts, oi_val
                break

        # Log when buffer first reaches full lookback depth
        if symbol not in self._oi_warm_logged:
            self._oi_warm_logged.add(symbol)
            logger.info(
                f"CryptoFlowProvider: OI history buffer warm for {symbol} "
                f"({len(buf)} snapshots, {self._lookback_seconds}s lookback)"
            )

        if best_oi == 0:
            return None, "STABLE"

        oi_change = (current_oi - best_oi) / best_oi

        if oi_change > _OI_TREND_THRESHOLD:
            oi_trend = "BUILDING"
        elif oi_change < -_OI_TREND_THRESHOLD:
            oi_trend = "DROPPING"
        else:
            oi_trend = "STABLE"

        return oi_change, oi_trend
