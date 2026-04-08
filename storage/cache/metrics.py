"""Cache performance metrics: hits, misses, hit rate."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CacheMetrics:
    """Tracks cache hit/miss statistics."""

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.sets: int = 0
        self.deletes: int = 0
        self.flushes: int = 0

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_set(self) -> None:
        self.sets += 1

    def record_delete(self) -> None:
        self.deletes += 1

    def record_flush(self) -> None:
        self.flushes += 1

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction (0.0 to 1.0). Returns 0.0 if no requests."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def summary(self) -> dict:
        """Return a metrics summary dict."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "sets": self.sets,
            "deletes": self.deletes,
            "flushes": self.flushes,
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.flushes = 0
