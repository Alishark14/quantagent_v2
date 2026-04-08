"""MacroEventAggregator — swarm-consensus pipeline for emergency assessments.

Per ARCHITECTURE §13.2.5: rather than hardcoding single-asset thresholds
("if BTC drops 8%"), the Macro Regime Manager is triggered by distributed
Sentinel consensus. When 5+ active Sentinels across DIFFERENT assets emit
`VolumeAnomaly` or `ExtremeMove` within a 60-second window, this aggregator
fires a `MacroReassessmentRequired` event.

Key invariants:

  * Events from the same symbol only count ONCE per window — one volatile
    asset must not be able to trigger an emergency assessment alone.
  * The window resets after emission so the same swarm doesn't re-fire
    on every new tick.
  * A 10-minute cooldown prevents the LLM from being hammered if stress
    persists across multiple discrete swarms.
  * Subscribe + emit are async-bus aware (matches `InProcessBus`'s
    contract: handlers may be sync or async, errors are caught upstream).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from engine.events import (
    EventBus,
    ExtremeMove,
    MacroReassessmentRequired,
    VolumeAnomaly,
)

logger = logging.getLogger(__name__)


# Defaults pulled out so tests can pin the literal §13.2.5 numbers.
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_MIN_UNIQUE_SYMBOLS = 5
DEFAULT_COOLDOWN_SECONDS = 600.0  # 10 minutes


@dataclass
class _PendingEvent:
    """One queued anomaly event held inside the sliding window."""

    symbol: str
    anomaly_type: str  # "VolumeAnomaly" | "ExtremeMove"
    severity: float
    received_at: datetime


@dataclass
class AggregatorMetrics:
    """Counters exposed for observability / tests."""

    events_received: int = 0
    events_dropped_oldest: int = 0
    triggers_emitted: int = 0
    triggers_suppressed_cooldown: int = 0


class MacroEventAggregator:
    """Sliding-window swarm-consensus aggregator.

    Construction is cheap. Call :meth:`subscribe` to wire to an
    `EventBus`, or call :meth:`handle_event` directly from tests.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        min_unique_symbols: int = DEFAULT_MIN_UNIQUE_SYMBOLS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock=None,
    ) -> None:
        self._bus = event_bus
        self._window = float(window_seconds)
        self._min_symbols = int(min_unique_symbols)
        self._cooldown = float(cooldown_seconds)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

        # Sliding window — keyed by (symbol, anomaly_type) so dupes from
        # the same Sentinel collapse but VolumeAnomaly + ExtremeMove for
        # the same symbol still both contribute (per §13.2.5: the swarm
        # is symbol-distinct, but a single symbol firing both event
        # types is still one symbol — we only count UNIQUE SYMBOLS at
        # emission time, not unique (symbol, type) pairs). Use an
        # OrderedDict so we can drop expired events in O(1) per drop.
        self._pending: OrderedDict[tuple[str, str], _PendingEvent] = OrderedDict()
        self._last_trigger: datetime | None = None
        self.metrics = AggregatorMetrics()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def subscribe(self) -> None:
        """Register this aggregator's handler for VolumeAnomaly + ExtremeMove."""
        self._bus.subscribe(VolumeAnomaly, self.handle_event)
        self._bus.subscribe(ExtremeMove, self.handle_event)

    def unsubscribe(self) -> None:
        self._bus.unsubscribe(VolumeAnomaly, self.handle_event)
        self._bus.unsubscribe(ExtremeMove, self.handle_event)

    # ------------------------------------------------------------------
    # Core handler
    # ------------------------------------------------------------------

    async def handle_event(self, event: Any) -> None:
        """Receive one anomaly event and possibly emit MacroReassessmentRequired.

        Sync usage from tests is also supported — `await` is harmless
        when the bus's gather already wraps results.
        """
        anomaly_type = type(event).__name__
        if anomaly_type not in ("VolumeAnomaly", "ExtremeMove"):
            return  # subscribed only to these but be defensive
        symbol = getattr(event, "symbol", "") or ""
        if not symbol:
            return

        self.metrics.events_received += 1
        now = self._clock()
        self._evict_expired(now)

        # Insert / refresh — most-recent occurrence wins on a re-fire
        # within the window. OrderedDict.move_to_end + reassignment
        # gives us LRU ordering for the eviction loop.
        key = (symbol, anomaly_type)
        self._pending[key] = _PendingEvent(
            symbol=symbol,
            anomaly_type=anomaly_type,
            severity=float(getattr(event, "severity", 0.0) or 0.0),
            received_at=now,
        )
        self._pending.move_to_end(key)

        # Cooldown check happens AFTER ingest so the window stays warm
        # during the cooldown — once cooldown expires, an already-hot
        # window can fire immediately.
        if self._in_cooldown(now):
            self.metrics.triggers_suppressed_cooldown += 1
            return

        unique_symbols = {ev.symbol for ev in self._pending.values()}
        if len(unique_symbols) < self._min_symbols:
            return

        # ---- Trigger! Build payload from the current window contents ----
        await self._emit_reassessment(now)

    # ------------------------------------------------------------------
    # Window maintenance
    # ------------------------------------------------------------------

    def _evict_expired(self, now: datetime) -> None:
        """Drop pending entries older than the window."""
        cutoff = now - timedelta(seconds=self._window)
        to_drop: list[tuple[str, str]] = []
        for key, event in self._pending.items():
            if event.received_at < cutoff:
                to_drop.append(key)
            else:
                # OrderedDict insertion order ≈ recency → first fresh
                # entry means everything after is also fresh.
                break
        for key in to_drop:
            self._pending.pop(key, None)
            self.metrics.events_dropped_oldest += 1

    def _in_cooldown(self, now: datetime) -> bool:
        if self._last_trigger is None:
            return False
        return (now - self._last_trigger).total_seconds() < self._cooldown

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    async def _emit_reassessment(self, now: datetime) -> None:
        """Build + publish the MacroReassessmentRequired event."""
        # Distinct-symbol payload — one entry per symbol, picking the
        # highest severity across (VolumeAnomaly, ExtremeMove) for that
        # symbol so the LLM sees the worst observation per asset.
        per_symbol: dict[str, _PendingEvent] = {}
        per_symbol_types: dict[str, set[str]] = {}
        for event in self._pending.values():
            existing = per_symbol.get(event.symbol)
            if existing is None or event.severity > existing.severity:
                per_symbol[event.symbol] = event
            per_symbol_types.setdefault(event.symbol, set()).add(event.anomaly_type)

        triggering_symbols = sorted(per_symbol.keys())
        anomaly_types = sorted({
            t for types in per_symbol_types.values() for t in types
        })
        severity_scores = [per_symbol[s].severity for s in triggering_symbols]

        triggered_at_iso = _iso(now)
        reassessment = MacroReassessmentRequired(
            source="macro_event_aggregator",
            timestamp=now,
            triggering_symbols=triggering_symbols,
            anomaly_types=anomaly_types,
            severity_scores=severity_scores,
            triggered_at=triggered_at_iso,
        )

        try:
            await self._bus.publish(reassessment)
        except Exception:
            logger.exception(
                "MacroEventAggregator: failed to publish MacroReassessmentRequired"
            )
            return

        # Reset window + start cooldown — per §13.2.5 we must NOT
        # re-trigger immediately even if more anomalies arrive.
        self._pending.clear()
        self._last_trigger = now
        self.metrics.triggers_emitted += 1
        logger.info(
            f"MacroEventAggregator: MacroReassessmentRequired emitted "
            f"({len(triggering_symbols)} symbols: {triggering_symbols}, "
            f"types={anomaly_types})"
        )


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
