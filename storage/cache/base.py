"""Abstract cache backend interface.

All cache backends implement this ABC. The engine never touches
cache internals — it goes through CacheManager which delegates here.

LLM responses are NEVER cached. Only deterministic data (OHLCV,
flow, asset metadata, external API results).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """Abstract cache backend: get/set/delete/clear."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value by key. Returns None on miss."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL in seconds."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached entries."""
        ...

    @abstractmethod
    async def has(self, key: str) -> bool:
        """Check if key exists and hasn't expired."""
        ...
