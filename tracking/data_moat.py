"""6-layer data capture for ML training pipeline.

Links all 6 data moat layers per trade cycle:
  L0: Raw market data (candles, order book)
  L1: Sensory inputs (indicators, flow, chart images)
  L2: Cognitive process (prompts, LLM responses, reasoning)
  L3: Action data (orders, fills, SL/TP)
  L4: Outcome data (MFE, MAE, R-multiple, P&L)
  L5: Reflection data (rules, scores)

Every cycle records are linked by cycle_id and trade_id.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataMoatCapture:
    """Captures and links all 6 data moat layers per cycle/trade."""

    def __init__(self) -> None:
        self.cycles_captured: list[dict] = []
        self.trades_captured: list[dict] = []
        self.layer_counts: dict[str, int] = {
            "L0_market": 0,
            "L1_sensory": 0,
            "L2_cognitive": 0,
            "L3_action": 0,
            "L4_outcome": 0,
            "L5_reflection": 0,
        }

    def capture_cycle(
        self,
        cycle_id: str,
        market_data: dict | None = None,
        signals: list[dict] | None = None,
        conviction: dict | None = None,
        action: str | None = None,
    ) -> None:
        """Capture layers L0-L2 from a completed analysis cycle."""
        record = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_market_data": market_data is not None,
            "has_signals": signals is not None and len(signals) > 0,
            "has_conviction": conviction is not None,
            "action": action,
        }
        self.cycles_captured.append(record)

        if market_data is not None:
            self.layer_counts["L0_market"] += 1
        if signals:
            self.layer_counts["L1_sensory"] += 1
        if conviction is not None:
            self.layer_counts["L2_cognitive"] += 1

        logger.debug(f"DataMoat: cycle captured — {cycle_id}")

    def capture_trade(
        self,
        trade_id: str,
        cycle_id: str,
        action: dict | None = None,
        outcome: dict | None = None,
        reflection: dict | None = None,
    ) -> None:
        """Capture layers L3-L5 from a completed trade."""
        record = {
            "trade_id": trade_id,
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_action": action is not None,
            "has_outcome": outcome is not None,
            "has_reflection": reflection is not None,
        }
        self.trades_captured.append(record)

        if action is not None:
            self.layer_counts["L3_action"] += 1
        if outcome is not None:
            self.layer_counts["L4_outcome"] += 1
        if reflection is not None:
            self.layer_counts["L5_reflection"] += 1

        logger.debug(f"DataMoat: trade captured — {trade_id}")

    def on_cycle_completed(self, event) -> None:
        """Event handler for CycleCompleted — captures L0-L2."""
        self.capture_cycle(
            cycle_id=f"cycle-{event.timestamp.isoformat()}",
            action=event.action,
        )

    def on_trade_opened(self, event) -> None:
        """Event handler for TradeOpened — captures L3."""
        action_data = event.trade_action.to_dict() if event.trade_action else None
        self.capture_trade(
            trade_id=event.order_result.order_id if event.order_result else "?",
            cycle_id="pending",
            action=action_data,
        )

    def on_trade_closed(self, event) -> None:
        """Event handler for TradeClosed — captures L4."""
        self.capture_trade(
            trade_id=f"close-{event.symbol}",
            cycle_id="pending",
            outcome={"pnl": event.pnl, "exit_reason": event.exit_reason},
        )

    def on_rule_generated(self, event) -> None:
        """Event handler for RuleGenerated — captures L5."""
        self.layer_counts["L5_reflection"] += 1

    def summary(self) -> dict:
        """Return data moat capture summary."""
        return {
            "cycles_captured": len(self.cycles_captured),
            "trades_captured": len(self.trades_captured),
            "layer_counts": dict(self.layer_counts),
        }
