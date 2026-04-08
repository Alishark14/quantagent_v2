"""File-system cache backend for large binary data (chart images).

Stores values as zlib-compressed files in a dedicated cache directory.
NOT suitable for Redis — chart blobs (200-500KB) degrade Redis
performance for fast JSON/numerical lookups.

TTL is enforced by checking file mtime + ttl against current time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Any

from storage.cache.base import CacheBackend

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "/tmp/quantagent/charts"


class FileCacheBackend(CacheBackend):
    """File-system cache with zlib compression and mtime-based TTL."""

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._ttls: dict[str, int] = {}  # key -> ttl for expiry check

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to a safe filesystem path."""
        # Sanitize: replace unsafe chars, hash if too long
        safe = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        if len(safe) > 200:
            safe = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{safe}.zc"

    def _ensure_dir(self) -> None:
        """Create cache directory on first write if needed."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def get(self, key: str) -> Any | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        # Check TTL via mtime
        ttl = self._ttls.get(key)
        if ttl is not None:
            mtime = path.stat().st_mtime
            if time.time() > mtime + ttl:
                # Expired
                path.unlink(missing_ok=True)
                self._ttls.pop(key, None)
                return None

        try:
            compressed = path.read_bytes()
            data = zlib.decompress(compressed)
            return pickle.loads(data)
        except Exception:
            logger.warning(f"FileCacheBackend: failed to read {key}", exc_info=True)
            path.unlink(missing_ok=True)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._ensure_dir()
        path = self._key_to_path(key)
        try:
            raw = pickle.dumps(value)
            compressed = zlib.compress(raw)
            path.write_bytes(compressed)
            if ttl is not None:
                self._ttls[key] = ttl
        except Exception:
            logger.warning(f"FileCacheBackend: failed to write {key}", exc_info=True)

    async def delete(self, key: str) -> bool:
        path = self._key_to_path(key)
        self._ttls.pop(key, None)
        if path.exists():
            path.unlink()
            return True
        return False

    async def clear(self) -> None:
        self._ttls.clear()
        if self._cache_dir.exists():
            for f in self._cache_dir.iterdir():
                if f.is_file() and f.suffix == ".zc":
                    f.unlink()

    async def has(self, key: str) -> bool:
        return await self.get(key) is not None
