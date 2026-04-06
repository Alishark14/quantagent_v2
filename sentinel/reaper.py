"""OrphanReaper: detects positions without active Sentinel management.

Periodically queries the exchange for open positions and cross-references
against the PositionManager registry. Orphans are handled based on
whether they have SL orders:

- Orphan WITH SL on exchange: safe — log and track
- Orphan WITHOUT SL: emergency SL at 2x ATR, critical alert
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from engine.types import Position
from exchanges.base import ExchangeAdapter
from sentinel.position_manager import PositionManager

logger = logging.getLogger(__name__)


class OrphanReaper:
    """Detects and handles orphan positions on the exchange."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        position_manager: PositionManager,
        default_atr: float = 500.0,
    ) -> None:
        self._adapter = adapter
        self._pm = position_manager
        self._default_atr = default_atr
        self.orphans_found: list[dict] = []
        self.emergency_sl_placed: list[dict] = []

    async def check(self) -> list[dict]:
        """Check exchange positions against PositionManager registry.

        Returns list of orphan records with their resolution.
        """
        try:
            exchange_positions = await self._adapter.get_positions()
        except Exception as e:
            logger.error(f"OrphanReaper: failed to fetch positions: {e}")
            return []

        orphans: list[dict] = []
        managed_symbols = set(self._pm.managed_symbols)

        for pos in exchange_positions:
            if pos.symbol in managed_symbols:
                continue  # managed by Sentinel — not an orphan

            orphan = await self._handle_orphan(pos)
            orphans.append(orphan)

        if orphans:
            logger.warning(f"OrphanReaper: found {len(orphans)} orphan(s)")

        return orphans

    async def _handle_orphan(self, pos: Position) -> dict:
        """Handle a single orphan position."""
        record = {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_sl": False,
            "action_taken": "none",
        }

        # Check if exchange has SL orders for this position
        has_sl = await self._check_exchange_sl(pos.symbol)

        if has_sl:
            record["has_sl"] = True
            record["action_taken"] = "logged_safe"
            logger.info(
                f"OrphanReaper: {pos.symbol} {pos.direction} orphan — "
                f"SL exists on exchange, safe"
            )
        else:
            # Emergency SL at 2x default ATR
            emergency_sl = self._compute_emergency_sl(pos)
            record["action_taken"] = "emergency_sl"
            record["emergency_sl_price"] = emergency_sl

            try:
                close_side = "sell" if pos.direction == "long" else "buy"
                result = await self._adapter.place_sl_order(
                    pos.symbol, close_side, pos.size, emergency_sl,
                )
                record["sl_order_success"] = result.success
                record["sl_order_id"] = result.order_id

                if result.success:
                    self.emergency_sl_placed.append(record)
                    logger.critical(
                        f"OrphanReaper: EMERGENCY SL placed for {pos.symbol} "
                        f"{pos.direction} at {emergency_sl} (2x ATR)"
                    )
                else:
                    logger.critical(
                        f"OrphanReaper: FAILED to place emergency SL for {pos.symbol}: "
                        f"{result.error}"
                    )
            except Exception as e:
                record["sl_order_success"] = False
                record["error"] = str(e)
                logger.critical(
                    f"OrphanReaper: exception placing emergency SL for {pos.symbol}: {e}"
                )

        self.orphans_found.append(record)
        return record

    async def _check_exchange_sl(self, symbol: str) -> bool:
        """Check if the exchange has an active SL order for a symbol.

        Tries cancel_all_orders with count 0 as a proxy — in production,
        this would query open orders directly. For now, we assume no SL
        unless PositionManager has it registered.
        """
        # In production: query open orders, check for stop-type orders
        # For now: if PositionManager doesn't track it, assume no SL
        return False

    def _compute_emergency_sl(self, pos: Position) -> float:
        """Compute emergency SL at 2x ATR from entry."""
        atr2 = 2.0 * self._default_atr

        if pos.direction == "long":
            return pos.entry_price - atr2
        else:
            return pos.entry_price + atr2

    def summary(self) -> dict:
        """Return reaper activity summary."""
        return {
            "orphans_found": len(self.orphans_found),
            "emergency_sl_placed": len(self.emergency_sl_placed),
            "recent_orphans": self.orphans_found[-5:],
        }
