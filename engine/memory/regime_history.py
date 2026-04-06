"""Loop 4: regime ring buffer for regime transition tracking.

Tracks regime classifications over time as a ring buffer of the last N entries.
Regime transition points are the most actionable moments — ConvictionAgent
uses this to detect when a market shifts from RANGING to BREAKOUT, etc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RegimeHistory:
    """In-memory ring buffer of regime classifications."""

    def __init__(self, max_size: int = 20) -> None:
        self._buffer: list[dict] = []
        self._max = max_size

    def add(self, regime: str, confidence: float) -> None:
        """Add a regime classification to the buffer."""
        self._buffer.append({
            "regime": regime,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._buffer) > self._max:
            self._buffer.pop(0)

    def get_history(self) -> list[dict]:
        """Get the full regime history buffer."""
        return list(self._buffer)

    def detect_transition(self) -> str | None:
        """Detect if regime just changed between the last two entries.

        Returns "PREV -> CURR" string if a transition occurred, None otherwise.
        """
        if len(self._buffer) < 2:
            return None
        prev = self._buffer[-2]["regime"]
        curr = self._buffer[-1]["regime"]
        if prev != curr:
            return f"{prev} -> {curr}"
        return None

    def current_regime(self) -> str | None:
        """Get the most recent regime classification."""
        if not self._buffer:
            return None
        return self._buffer[-1]["regime"]

    def regime_streak(self) -> int:
        """Count how many consecutive entries share the current regime."""
        if not self._buffer:
            return 0
        current = self._buffer[-1]["regime"]
        count = 0
        for entry in reversed(self._buffer):
            if entry["regime"] == current:
                count += 1
            else:
                break
        return count

    def format_for_prompt(self) -> str:
        """Format regime history as context string for agent prompts."""
        if not self._buffer:
            return "No regime history."
        transition = self.detect_transition()
        recent = self._buffer[-5:] if len(self._buffer) >= 5 else self._buffer
        lines = [f"- {r['regime']} (conf={r['confidence']})" for r in recent]
        header = f"Regime history (last {len(recent)}):\n" + "\n".join(lines)
        if transition:
            header += f"\nREGIME TRANSITION: {transition}"
        return header
