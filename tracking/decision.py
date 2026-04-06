"""Per-cycle decision capture and signal quality tracking.

Subscribes to CycleCompleted, ConvictionScored, SignalsReady.
Records every decision for signal quality analysis and conviction calibration.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DecisionTracker:
    """Records cycle-level decisions and signal quality data."""

    def __init__(self) -> None:
        self.cycles: list[dict] = []
        self.conviction_scores: list[float] = []
        self.action_counts: dict[str, int] = {}
        self.signal_counts: dict[str, int] = {}

    def on_cycle_completed(self, event) -> None:
        """Record a completed analysis cycle."""
        record = {
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
            "symbol": event.symbol,
            "action": event.action,
            "conviction": event.conviction,
        }
        self.cycles.append(record)
        self.conviction_scores.append(event.conviction)
        self.action_counts[event.action] = self.action_counts.get(event.action, 0) + 1

        logger.debug(
            f"DecisionTracker: cycle — {event.symbol} {event.action} "
            f"(conviction={event.conviction:.2f})"
        )

    def on_signals_ready(self, event) -> None:
        """Count signals by agent name for quality tracking."""
        for signal in event.signals:
            name = signal.agent_name
            self.signal_counts[name] = self.signal_counts.get(name, 0) + 1

    def on_conviction_scored(self, event) -> None:
        """Record conviction details for calibration analysis."""
        if event.conviction:
            self.conviction_scores.append(event.conviction.conviction_score)

    @property
    def avg_conviction(self) -> float:
        """Average conviction score across all recorded cycles."""
        if not self.conviction_scores:
            return 0.0
        return sum(self.conviction_scores) / len(self.conviction_scores)

    def summary(self) -> dict:
        """Return a summary of decision metrics."""
        return {
            "total_cycles": len(self.cycles),
            "avg_conviction": self.avg_conviction,
            "action_counts": dict(self.action_counts),
            "signal_counts": dict(self.signal_counts),
        }
