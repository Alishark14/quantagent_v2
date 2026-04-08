"""CacheManager: high-level caching with thundering herd protection,
epoch-aligned TTLs, file-system chart caching, and metrics.

Usage:
    cache = CacheManager()
    data = await cache.get_or_fetch("ohlcv:BTC:1h", fetch_fn, ttl=compute_ttl("1h"))
    chart = await cache.get_or_fetch("chart:BTC:1h:12345", render_fn, ttl=compute_ttl("1h"))

LLM responses are NEVER cached. Only deterministic data.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from storage.cache.base import CacheBackend
from storage.cache.file_cache import FileCacheBackend
from storage.cache.memory import MemoryCacheBackend
from storage.cache.metrics import CacheMetrics
from storage.cache.ttl import (
    FLOW_TTL,
    NEWS_TTL,
    REGIME_TTL,
    SENTIMENT_TTL,
    TIMEFRAME_SECONDS,
    compute_ttl,
    expected_candle_close,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL constants (seconds) — re-exported for backwards compatibility
# ---------------------------------------------------------------------------

# OHLCV: use compute_ttl(timeframe) for epoch-aligned expiry.
# These fixed values are kept as fallbacks / documentation.
TTL_OHLCV: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Flow data (funding, OI): 5 minutes (per ARCHITECTURE.md §18.1)
TTL_FLOW: int = FLOW_TTL  # 300

# External API responses
TTL_EXTERNAL_API: int = SENTIMENT_TTL  # 3600

# Asset metadata (deployer fees, growth mode flags)
TTL_ASSET_META: int = 86400  # 24 hours

# Orderbook snapshots (very short — stale books are dangerous)
TTL_ORDERBOOK: int = 5

# Funding rate cache
TTL_FUNDING: int = 60


# ---------------------------------------------------------------------------
# Cache key builders
# ---------------------------------------------------------------------------

def ohlcv_key(symbol: str, timeframe: str) -> str:
    return f"ohlcv:{symbol}:{timeframe}"


def flow_key(symbol: str) -> str:
    return f"flow:{symbol}"


def meta_key(exchange: str) -> str:
    return f"meta:{exchange}"


def orderbook_key(symbol: str) -> str:
    return f"orderbook:{symbol}"


def funding_key(symbol: str) -> str:
    return f"funding:{symbol}"


def chart_key(symbol: str, timeframe: str, timestamp: float) -> str:
    return f"chart:{symbol}:{timeframe}:{int(timestamp)}"


# Chart key prefix for routing
_CHART_PREFIX = "chart:"


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

class CacheManager:
    """High-level cache with thundering herd protection and dual backends.

    - Memory backend: JSON/numerical data (OHLCV, flow, metadata)
    - File backend: chart images (zlib-compressed, /tmp/quantagent/charts/)
    - Thundering herd: per-key asyncio.Lock ensures exactly 1 fetch per key
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        chart_backend: FileCacheBackend | None = None,
    ) -> None:
        self._backend = backend or MemoryCacheBackend()
        self._chart_backend = chart_backend or FileCacheBackend()
        self._metrics = CacheMetrics()
        self._locks: dict[str, asyncio.Lock] = {}

    def _select_backend(self, key: str) -> CacheBackend:
        """Route chart:* keys to file backend, everything else to memory."""
        if key.startswith(_CHART_PREFIX):
            return self._chart_backend
        return self._backend

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    @property
    def chart_backend(self) -> FileCacheBackend:
        return self._chart_backend

    async def get(self, key: str) -> Any | None:
        """Get a value. Records hit/miss in metrics."""
        be = self._select_backend(key)
        value = await be.get(key)
        if value is not None:
            self._metrics.record_hit()
        else:
            self._metrics.record_miss()
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL."""
        be = self._select_backend(key)
        await be.set(key, value, ttl=ttl)
        self._metrics.record_set()

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[Any]],
        ttl: int | float | None = None,
    ) -> Any:
        """Cache-aside with thundering herd protection.

        Fast path: check cache before acquiring any lock.
        On miss: acquire per-key lock, double-check, fetch if still miss.
        Guarantees exactly 1 fetch_fn call per cache key regardless of
        concurrent callers (Promise Caching pattern).
        """
        be = self._select_backend(key)
        effective_ttl = int(ttl) if ttl is not None else None

        # Fast path — no lock overhead on cache hit
        cached = await be.get(key)
        if cached is not None:
            self._metrics.record_hit()
            return cached

        # Slow path — acquire per-key lock
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Double-check after acquiring lock
            cached = await be.get(key)
            if cached is not None:
                self._metrics.record_hit()
                return cached

            self._metrics.record_miss()
            value = await fetch_fn()
            if value is not None:
                await be.set(key, value, ttl=effective_ttl)
                self._metrics.record_set()

            return value

        # Cleanup lock if no other waiters
        # (unreachable due to return inside async with, but kept for clarity)

    async def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache."""
        be = self._select_backend(key)
        result = await be.delete(key)
        if result:
            self._metrics.record_delete()
        return result

    async def flush_all(self) -> None:
        """Clear both backends."""
        await self._backend.clear()
        await self._chart_backend.clear()
        self._locks.clear()
        self._metrics.record_flush()
        logger.info("CacheManager: flushed all entries")

    async def invalidate_pattern(self, prefix: str) -> int:
        """Invalidate all keys starting with prefix.

        Works with MemoryCacheBackend (iterates buckets).
        """
        count = 0
        if isinstance(self._backend, MemoryCacheBackend):
            keys_to_delete = [
                k for k in self._backend._key_ttl
                if k.startswith(prefix)
            ]
            for key in keys_to_delete:
                await self._backend.delete(key)
                count += 1
                self._metrics.record_delete()
        return count
