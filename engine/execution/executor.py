"""Executor: bridge between DecisionAgent output and exchange orders.

Translates TradeAction into concrete exchange operations:
  LONG/SHORT  -> market order + SL + TP1 (50%) + TP2 (50%)
  ADD_*       -> pyramid market order + adjust SL
  CLOSE_ALL   -> cancel all orders + market close
  HOLD/SKIP   -> no-op

If SL placement fails after opening a position, the Executor immediately
closes the position (no unprotected positions).
"""

from __future__ import annotations

import logging

from engine.config import TradingConfig
from engine.events import EventBus, TradeClosed, TradeOpened
from engine.types import OrderResult, TradeAction
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)

_NO_OP_RESULT = OrderResult(
    success=True, order_id=None, fill_price=None, fill_size=None, error=None,
)


class Executor:
    """Executes trade actions on the exchange and manages SL/TP placement."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        event_bus: EventBus,
        config: TradingConfig,
    ) -> None:
        self._adapter = adapter
        self._bus = event_bus
        self._config = config

    async def execute(self, action: TradeAction, symbol: str) -> OrderResult:
        """Execute a TradeAction on the exchange.

        Returns:
            OrderResult from the primary order (market entry/close).
            For SKIP/HOLD returns a no-op success result.
        """
        if action.action in ("SKIP", "HOLD"):
            return _NO_OP_RESULT

        try:
            if action.action in ("LONG", "SHORT"):
                return await self._open_position(action, symbol)
            elif action.action in ("ADD_LONG", "ADD_SHORT"):
                return await self._add_to_position(action, symbol)
            elif action.action == "CLOSE_ALL":
                return await self._close_position(action, symbol)
            else:
                logger.warning(f"Executor: unknown action '{action.action}'")
                return _NO_OP_RESULT

        except Exception as e:
            logger.error(f"Executor: execute failed for {symbol}: {e}", exc_info=True)
            return OrderResult(
                success=False, order_id=None, fill_price=None,
                fill_size=None, error=str(e),
            )

    async def _open_position(self, action: TradeAction, symbol: str) -> OrderResult:
        """Open a new position: market order + SL + TP1 (50%) + TP2 (50%)."""
        side = "buy" if action.action == "LONG" else "sell"
        close_side = "sell" if side == "buy" else "buy"

        # Convert USD position size to base asset units
        size = self._usd_to_base_size(action.position_size, action.sl_price, symbol)
        if size <= 0:
            return OrderResult(
                success=False, order_id=None, fill_price=None,
                fill_size=None, error="Computed size <= 0",
            )

        # 1. Market order
        entry_result = await self._adapter.place_market_order(symbol, side, size)
        if not entry_result.success:
            logger.error(f"Executor: market order failed for {symbol}: {entry_result.error}")
            return entry_result

        fill_price = entry_result.fill_price or 0
        fill_size = entry_result.fill_size or size
        logger.info(f"Executor: {action.action} {symbol} filled at {fill_price} size={fill_size}")

        # 2. SL order — CRITICAL: if this fails, emergency close
        if action.sl_price:
            sl_result = await self._adapter.place_sl_order(
                symbol, close_side, fill_size, action.sl_price,
            )
            if not sl_result.success:
                logger.error(
                    f"Executor: SL placement FAILED for {symbol} — emergency close! "
                    f"Error: {sl_result.error}"
                )
                await self._adapter.close_position(symbol)
                return OrderResult(
                    success=False,
                    order_id=entry_result.order_id,
                    fill_price=fill_price,
                    fill_size=fill_size,
                    error=f"SL placement failed: {sl_result.error}. Position emergency closed.",
                )

        # 3. TP1 at 50% of position
        half_size = round(fill_size / 2, 8)
        remaining_size = round(fill_size - half_size, 8)

        if action.tp1_price and half_size > 0:
            tp1_result = await self._adapter.place_tp_order(
                symbol, close_side, half_size, action.tp1_price,
            )
            if not tp1_result.success:
                logger.warning(f"Executor: TP1 placement failed for {symbol}: {tp1_result.error}")

        # 4. TP2 at remaining 50%
        if action.tp2_price and remaining_size > 0:
            tp2_result = await self._adapter.place_tp_order(
                symbol, close_side, remaining_size, action.tp2_price,
            )
            if not tp2_result.success:
                logger.warning(f"Executor: TP2 placement failed for {symbol}: {tp2_result.error}")

        # 5. Emit TradeOpened event
        try:
            await self._bus.publish(TradeOpened(
                source="executor",
                trade_action=action,
                order_result=entry_result,
            ))
        except Exception:
            logger.warning("Executor: failed to emit TradeOpened event", exc_info=True)

        return entry_result

    async def _add_to_position(self, action: TradeAction, symbol: str) -> OrderResult:
        """Pyramid: add to existing position at 50% of base size, adjust SL."""
        side = "buy" if action.action == "ADD_LONG" else "sell"
        close_side = "sell" if side == "buy" else "buy"

        size = self._usd_to_base_size(action.position_size, action.sl_price, symbol)
        if size <= 0:
            return OrderResult(
                success=False, order_id=None, fill_price=None,
                fill_size=None, error="Computed pyramid size <= 0",
            )

        # Market order for the add
        add_result = await self._adapter.place_market_order(symbol, side, size)
        if not add_result.success:
            logger.error(f"Executor: pyramid order failed for {symbol}: {add_result.error}")
            return add_result

        logger.info(
            f"Executor: {action.action} {symbol} filled at "
            f"{add_result.fill_price} size={add_result.fill_size}"
        )

        # Adjust SL to new level if provided
        if action.sl_price:
            sl_result = await self._adapter.modify_sl(symbol, action.sl_price)
            if not sl_result.success:
                logger.warning(f"Executor: SL adjustment failed for {symbol}: {sl_result.error}")

        # Emit TradeOpened (pyramid counts as trade opened)
        try:
            await self._bus.publish(TradeOpened(
                source="executor",
                trade_action=action,
                order_result=add_result,
            ))
        except Exception:
            logger.warning("Executor: failed to emit TradeOpened event", exc_info=True)

        return add_result

    async def _close_position(self, action: TradeAction, symbol: str) -> OrderResult:
        """Close entire position: cancel all orders, then market close."""
        # 1. Cancel all open orders for this symbol
        try:
            cancelled = await self._adapter.cancel_all_orders(symbol)
            logger.info(f"Executor: cancelled {cancelled} orders for {symbol}")
        except Exception:
            logger.warning(f"Executor: cancel_all_orders failed for {symbol}", exc_info=True)

        # 2. Market close
        close_result = await self._adapter.close_position(symbol)
        if not close_result.success:
            logger.error(f"Executor: close_position failed for {symbol}: {close_result.error}")
            return close_result

        logger.info(f"Executor: CLOSE_ALL {symbol} at {close_result.fill_price}")

        # 3. Emit TradeClosed event
        try:
            await self._bus.publish(TradeClosed(
                source="executor",
                symbol=symbol,
                pnl=0.0,  # actual P&L computed by tracking system
                exit_reason="CLOSE_ALL",
            ))
        except Exception:
            logger.warning("Executor: failed to emit TradeClosed event", exc_info=True)

        return close_result

    def _usd_to_base_size(
        self,
        position_size_usd: float | None,
        reference_price: float | None,
        symbol: str,
    ) -> float:
        """Convert USD position size to base asset units.

        Uses SL price as reference (available before market fill).
        Falls back to 0 if inputs are invalid.
        """
        if not position_size_usd or position_size_usd <= 0:
            return 0.0
        if not reference_price or reference_price <= 0:
            return 0.0
        return round(position_size_usd / reference_price, 8)
