"""Sentinel PositionManager: SL/TP adjustment between TraderBot lifecycles.

CRITICAL RULE: The Sentinel only TIGHTENS stops — it never widens them
or changes trade direction. Only a TraderBot (via full LLM analysis)
can decide to CLOSE_ALL or ADD to a position.

Managed adjustments:
- Trailing stop: tighten SL when price moves 1 ATR in favor
- Break-even: move SL to entry price after TP1 is filled
- Funding tighten: tighten SL by 0.3 ATR when funding flips against position

When SL changes, PositionManager calls adapter.modify_sl() directly
and emits PositionUpdated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from engine.events import EventBus, PositionUpdated
from engine.types import Position
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    """Internal state for a position being managed by Sentinel."""

    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    current_sl: float
    atr: float
    tp1_filled: bool = False


class PositionManager:
    """Manages SL adjustments for open positions between TraderBot lifecycles.

    The Sentinel calls check_adjustments() periodically with the latest
    price and funding rate. If any adjustment is warranted and is tighter
    than the current SL, the manager calls adapter.modify_sl() and emits
    PositionUpdated.
    """

    def __init__(self, adapter: ExchangeAdapter, event_bus: EventBus) -> None:
        self._adapter = adapter
        self._bus = event_bus
        self._positions: dict[str, ManagedPosition] = {}

    def register_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        atr: float,
    ) -> None:
        """Register a new position for Sentinel management."""
        self._positions[symbol] = ManagedPosition(
            symbol=symbol,
            direction=direction.lower(),
            entry_price=entry_price,
            current_sl=sl_price,
            atr=atr,
        )
        logger.info(
            f"PositionManager: registered {direction} {symbol} "
            f"entry={entry_price} SL={sl_price} ATR={atr}"
        )

    def remove_position(self, symbol: str) -> None:
        """Remove a position from management (after close)."""
        if symbol in self._positions:
            del self._positions[symbol]
            logger.info(f"PositionManager: removed {symbol}")

    def mark_tp1_filled(self, symbol: str) -> None:
        """Mark TP1 as filled — enables break-even SL adjustment."""
        pos = self._positions.get(symbol)
        if pos:
            pos.tp1_filled = True
            logger.info(f"PositionManager: TP1 filled for {symbol}")

    def get_position(self, symbol: str) -> ManagedPosition | None:
        """Get managed position state."""
        return self._positions.get(symbol)

    @property
    def managed_symbols(self) -> list[str]:
        return list(self._positions.keys())

    async def check_adjustments(
        self,
        symbol: str,
        current_price: float,
        funding_rate: float | None = None,
    ) -> float | None:
        """Check if SL should be tightened for a managed position.

        If tighter SL is found, calls adapter.modify_sl() on the exchange
        and emits PositionUpdated. Returns the new SL price if adjusted,
        or None if no change.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        new_sl = pos.current_sl
        reason = ""

        # 1. Break-even after TP1
        be_sl = self._check_breakeven(pos)
        if be_sl is not None and self._is_tighter(pos, be_sl):
            new_sl = be_sl
            reason = "break-even after TP1"

        # 2. Trailing stop (1 ATR move -> trail at 1 ATR distance)
        trail_sl = self._check_trailing(pos, current_price)
        if trail_sl is not None and self._is_tighter(pos, trail_sl):
            if self._is_tighter_than(pos, trail_sl, new_sl):
                new_sl = trail_sl
                reason = "trailing stop"

        # 3. Funding rate tighten (flip against position)
        fund_sl = self._check_funding_tighten(pos, funding_rate)
        if fund_sl is not None and self._is_tighter(pos, fund_sl):
            if self._is_tighter_than(pos, fund_sl, new_sl):
                new_sl = fund_sl
                reason = "funding rate shift"

        # Only update if actually tighter than current
        if new_sl != pos.current_sl and self._is_tighter(pos, new_sl):
            old_sl = pos.current_sl
            pos.current_sl = new_sl

            logger.info(
                f"PositionManager: {symbol} SL tightened {old_sl:.2f} -> {new_sl:.2f} "
                f"({reason})"
            )

            # Call adapter to modify SL on the exchange
            try:
                result = await self._adapter.modify_sl(symbol, new_sl)
                if not result.success:
                    logger.warning(
                        f"PositionManager: modify_sl failed for {symbol}: {result.error}"
                    )
            except Exception:
                logger.warning("PositionManager: modify_sl call failed", exc_info=True)

            # Emit event
            try:
                await self._bus.publish(PositionUpdated(
                    source="position_manager",
                    symbol=symbol,
                    position=Position(
                        symbol=symbol,
                        direction=pos.direction,
                        size=0.0,
                        entry_price=pos.entry_price,
                        unrealized_pnl=0.0,
                        leverage=None,
                    ),
                ))
            except Exception:
                logger.warning("PositionManager: failed to emit PositionUpdated", exc_info=True)

            return new_sl

        return None

    def _check_breakeven(self, pos: ManagedPosition) -> float | None:
        """After TP1 filled, move SL to entry price (break-even)."""
        if not pos.tp1_filled:
            return None
        return pos.entry_price

    def _check_trailing(
        self, pos: ManagedPosition, current_price: float
    ) -> float | None:
        """If price moved >= 1 ATR in favor, trail SL at 1 ATR from current price.

        For LONG: price must be above entry + 1 ATR, trail SL = price - 1 ATR
        For SHORT: price must be below entry - 1 ATR, trail SL = price + 1 ATR
        """
        if pos.atr <= 0:
            return None

        if pos.direction == "long":
            favorable_move = current_price - pos.entry_price
            if favorable_move >= pos.atr:
                return current_price - pos.atr
        elif pos.direction == "short":
            favorable_move = pos.entry_price - current_price
            if favorable_move >= pos.atr:
                return current_price + pos.atr

        return None

    def _check_funding_tighten(
        self, pos: ManagedPosition, funding_rate: float | None
    ) -> float | None:
        """Tighten SL by 0.3 ATR when funding flips against position.

        For LONG: funding > 0 means crowded long (against us) -> tighten
        For SHORT: funding < 0 means crowded short (against us) -> tighten
        """
        if funding_rate is None or pos.atr <= 0:
            return None

        tighten_amount = 0.3 * pos.atr

        if pos.direction == "long" and funding_rate > 0.0001:
            return pos.current_sl + tighten_amount
        elif pos.direction == "short" and funding_rate < -0.0001:
            return pos.current_sl - tighten_amount

        return None

    @staticmethod
    def _is_tighter(pos: ManagedPosition, new_sl: float) -> bool:
        """Check if new_sl is tighter (closer to price) than current.

        For LONG: tighter means higher SL (closer to current price from below)
        For SHORT: tighter means lower SL (closer to current price from above)
        """
        if pos.direction == "long":
            return new_sl > pos.current_sl
        elif pos.direction == "short":
            return new_sl < pos.current_sl
        return False

    @staticmethod
    def _is_tighter_than(pos: ManagedPosition, new_sl: float, reference_sl: float) -> bool:
        """Check if new_sl is tighter than a reference SL (not just current)."""
        if pos.direction == "long":
            return new_sl > reference_sl
        elif pos.direction == "short":
            return new_sl < reference_sl
        return False
