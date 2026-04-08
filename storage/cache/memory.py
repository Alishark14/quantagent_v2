"""In-process cache using cachetools.TTLCache.

Single-server only. For multi-server deployments, use RedisCacheBackend.
Thread-safe via cachetools internal locking.
"""

from __future__ import annotations

import logging
from typing import Any

from cachetools import TTLCache

from storage.cache.base import CacheBackend

logger = logging.getLogger(__name__)

# Default: 1024 entries, 1 hour TTL
_DEFAULT_MAX_SIZE = 1024
_DEFAULT_TTL = 3600


class MemoryCacheBackend(CacheBackend):
    """In-process TTL cache backed by cachetools.TTLCache.

    Each key is stored with the TTL specified at set() time.
    Since TTLCache uses a single TTL for all entries, we use
    a dict of TTLCaches bucketed by TTL value.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._max_size = max_size
        # Bucket by TTL for different expiry periods
        self._buckets: dict[int, TTLCache] = {}
        # Key -> TTL mapping for lookups across buckets
        self._key_ttl: dict[str, int] = {}

    def _get_bucket(self, ttl: int) -> TTLCache:
        """Get or create a TTLCache bucket for this TTL value."""
        if ttl not in self._buckets:
            self._buckets[ttl] = TTLCache(maxsize=self._max_size, ttl=ttl)
        return self._buckets[ttl]

    async def get(self, key: str) -> Any | None:
        ttl = self._key_ttl.get(key)
        if ttl is None:
            return None
        bucket = self._buckets.get(ttl)
        if bucket is None:
            return None
        return bucket.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl or _DEFAULT_TTL
        # Remove from old bucket if TTL changed
        old_ttl = self._key_ttl.get(key)
        if old_ttl is not None and old_ttl != effective_ttl:
            old_bucket = self._buckets.get(old_ttl)
            if old_bucket is not None:
                old_bucket.pop(key, None)

        bucket = self._get_bucket(effective_ttl)
        bucket[key] = value
        self._key_ttl[key] = effective_ttl

    async def delete(self, key: str) -> bool:
        ttl = self._key_ttl.pop(key, None)
        if ttl is None:
            return False
        bucket = self._buckets.get(ttl)
        if bucket is not None:
            bucket.pop(key, None)
        return True

    async def clear(self) -> None:
        for bucket in self._buckets.values():
            bucket.clear()
        self._key_ttl.clear()

    async def has(self, key: str) -> bool:
        return await self.get(key) is not None

    @property
    def total_entries(self) -> int:
        """Total entries across all buckets (excluding expired)."""
        return sum(len(b) for b in self._buckets.values())
