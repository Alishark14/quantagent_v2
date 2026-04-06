"""Per-trade, per-bot, per-portfolio financial metrics.

Subscribes to TradeOpened and TradeClosed events. Records entry/exit
details for later aggregation (win rate, Sharpe, drawdown, etc.).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class FinancialTracker:
    """Records trade-level financial data from events."""

    def __init__(self) -> None:
        self.trades_opened: list[dict] = []
        self.trades_closed: list[dict] = []
        self.total_pnl: float = 0.0
        self.trade_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0

    def on_trade_opened(self, event) -> None:
        """Record a new trade entry."""
        record = {
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
            "source": event.source,
            "action": event.trade_action.action if event.trade_action else "?",
            "conviction_score": event.trade_action.conviction_score if event.trade_action else 0,
            "position_size": event.trade_action.position_size if event.trade_action else None,
            "sl_price": event.trade_action.sl_price if event.trade_action else None,
            "tp1_price": event.trade_action.tp1_price if event.trade_action else None,
            "tp2_price": event.trade_action.tp2_price if event.trade_action else None,
            "fill_price": event.order_result.fill_price if event.order_result else None,
            "fill_size": event.order_result.fill_size if event.order_result else None,
            "order_id": event.order_result.order_id if event.order_result else None,
        }
        self.trades_opened.append(record)
        logger.info(f"FinancialTracker: trade opened — {record['action']}")

    def on_trade_closed(self, event) -> None:
        """Record a trade exit and update running totals."""
        record = {
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
            "symbol": event.symbol,
            "pnl": event.pnl,
            "exit_reason": event.exit_reason,
        }
        self.trades_closed.append(record)
        self.trade_count += 1
        self.total_pnl += event.pnl

        if event.pnl > 0:
            self.win_count += 1
        elif event.pnl < 0:
            self.loss_count += 1

        logger.info(
            f"FinancialTracker: trade closed — {event.symbol} "
            f"P&L={event.pnl:.2f} ({event.exit_reason})"
        )

    @property
    def win_rate(self) -> float:
        """Win rate as fraction (0-1). Returns 0 if no trades."""
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    def summary(self) -> dict:
        """Return a summary of financial metrics."""
        return {
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "trades_opened": len(self.trades_opened),
            "trades_closed": len(self.trades_closed),
        }
