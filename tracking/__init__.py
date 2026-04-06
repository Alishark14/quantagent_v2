"""TrackingModule: pure observer that subscribes to ALL events.

If tracking fails, the trade still executes. All handlers are wrapped
in _safe() to catch and log errors without propagating them.
"""

from __future__ import annotations

import logging

from engine.events import (
    ConvictionScored,
    CycleCompleted,
    DataReady,
    EventBus,
    RuleGenerated,
    SignalsReady,
    TradeClosed,
    TradeOpened,
)
from tracking.data_moat import DataMoatCapture
from tracking.decision import DecisionTracker
from tracking.financial import FinancialTracker
from tracking.health import HealthTracker

logger = logging.getLogger(__name__)


class TrackingModule:
    """Subscribes to all events and delegates to specialized trackers.

    Every handler is wrapped in _safe() — tracking failures never
    propagate to the pipeline.
    """

    def __init__(
        self,
        financial: FinancialTracker | None = None,
        decision: DecisionTracker | None = None,
        health: HealthTracker | None = None,
        data_moat: DataMoatCapture | None = None,
    ) -> None:
        self.financial = financial or FinancialTracker()
        self.decision = decision or DecisionTracker()
        self.health = health or HealthTracker()
        self.data_moat = data_moat or DataMoatCapture()

    def subscribe_all(self, bus: EventBus) -> None:
        """Subscribe all tracking handlers to the event bus."""
        # Health tracks everything
        bus.subscribe(DataReady, self._safe(self.health.on_any_event))
        bus.subscribe(SignalsReady, self._safe(self.health.on_any_event))
        bus.subscribe(ConvictionScored, self._safe(self.health.on_any_event))
        bus.subscribe(TradeOpened, self._safe(self.health.on_any_event))
        bus.subscribe(TradeClosed, self._safe(self.health.on_any_event))
        bus.subscribe(CycleCompleted, self._safe(self.health.on_any_event))
        bus.subscribe(RuleGenerated, self._safe(self.health.on_any_event))

        # Financial
        bus.subscribe(TradeOpened, self._safe(self.financial.on_trade_opened))
        bus.subscribe(TradeClosed, self._safe(self.financial.on_trade_closed))

        # Decision
        bus.subscribe(CycleCompleted, self._safe(self.decision.on_cycle_completed))
        bus.subscribe(SignalsReady, self._safe(self.decision.on_signals_ready))
        bus.subscribe(ConvictionScored, self._safe(self.decision.on_conviction_scored))

        # Data Moat
        bus.subscribe(CycleCompleted, self._safe(self.data_moat.on_cycle_completed))
        bus.subscribe(TradeOpened, self._safe(self.data_moat.on_trade_opened))
        bus.subscribe(TradeClosed, self._safe(self.data_moat.on_trade_closed))
        bus.subscribe(RuleGenerated, self._safe(self.data_moat.on_rule_generated))

        logger.info("TrackingModule: subscribed to all events")

    def summary(self) -> dict:
        """Return a combined summary from all trackers."""
        return {
            "financial": self.financial.summary(),
            "decision": self.decision.summary(),
            "health": self.health.summary(),
            "data_moat": self.data_moat.summary(),
        }

    @staticmethod
    def _safe(handler: callable) -> callable:
        """Wrap a handler so exceptions are logged but never propagate."""
        def wrapped(event) -> None:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    f"TrackingModule: handler {handler.__qualname__} failed "
                    f"for {type(event).__name__}"
                )
        return wrapped
