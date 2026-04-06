"""Bot health, API health, and infrastructure monitoring.

Subscribes to ALL events and counts them. Tracks error rates
and provides a health summary for the dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HealthTracker:
    """Counts events, tracks errors, monitors system health."""

    def __init__(self) -> None:
        self.event_counts: dict[str, int] = {}
        self.error_count: int = 0
        self.errors: list[dict] = []
        self._start_time: datetime = datetime.now(timezone.utc)

    def on_any_event(self, event) -> None:
        """Count every event by type name."""
        name = type(event).__name__
        self.event_counts[name] = self.event_counts.get(name, 0) + 1

    def record_error(self, source: str, error: str) -> None:
        """Record an error from any component."""
        self.error_count += 1
        self.errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "error": error,
        })
        # Keep only last 100 errors in memory
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]

        logger.warning(f"HealthTracker: error from {source}: {error}")

    @property
    def uptime_seconds(self) -> float:
        """Seconds since tracker was created."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    @property
    def total_events(self) -> int:
        return sum(self.event_counts.values())

    def summary(self) -> dict:
        """Return a health summary."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "total_events": self.total_events,
            "event_counts": dict(self.event_counts),
            "error_count": self.error_count,
            "recent_errors": self.errors[-5:],
        }
