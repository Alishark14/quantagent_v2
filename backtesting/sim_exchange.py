"""SimulatedExchangeAdapter — drop-in fake exchange for backtesting.

Implements the full ``ExchangeAdapter`` ABC so that the live engine
(SignalProducers, ConvictionAgent, DecisionAgent, Executor, Sentinel) cannot
tell it is running on simulated state instead of a real venue. This is the
single most important property of the backtest framework — same code path,
fake exchange underneath. See ARCHITECTURE.md §31.3.5.

Capabilities and limitations:
- Single-account, netting-mode positions (one position per symbol).
- Native SL/TP simulated against candle high/low (priority: SL > TP on the
  same candle, conservative).
- Slippage applied to every market/SL/TP fill (limits fill at limit price).
- Fees taken from a real ``ExecutionCostModel`` if injected, otherwise zero.
- Funding applied via ``apply_funding(symbol, rate)`` whenever the
  BacktestEngine decides (default every 8h).
- ``fetch_ohlcv`` delegates to a ``ParquetDataLoader`` (offline backtest)
  OR to a real ``ExchangeAdapter`` injected as ``data_adapter`` (shadow
  mode — live data feed, fake fills). If both are provided,
  ``data_adapter`` wins and is used for read-only methods (ohlcv, ticker,
  orderbook, funding rate, open interest, meta) so Sentinel + signal
  agents see real market state.

The adapter never reaches over the network for ORDER methods. The only
network traffic in shadow mode comes from the injected ``data_adapter``
servicing read-only data calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter

if TYPE_CHECKING:  # avoid runtime cycles; both modules are pure-Python
    from backtesting.data_loader import ParquetDataLoader
    from engine.execution.cost_model import ExecutionCostModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state types
# ---------------------------------------------------------------------------


@dataclass
class _SimPosition:
    """In-memory position record."""

    symbol: str
    direction: str  # "long" | "short"
    size: float
    entry_price: float
    entry_fees: float = 0.0  # cumulative entry fees paid (not yet realised)
    funding_paid: float = 0.0
    opened_at_ms: int = 0


@dataclass
class _SimOrder:
    """In-memory pending order (limit / stop / take-profit)."""

    order_id: str
    symbol: str
    order_type: str  # "limit" | "stop" | "take_profit"
    side: str  # "buy" | "sell"
    size: float
    price: float
    created_at_ms: int = 0


@dataclass
class AssetMeta:
    """Per-symbol exchange metadata used by the sim."""

    tick_size: float = 0.01
    lot_size: float = 0.001
    min_notional: float = 10.0


# ---------------------------------------------------------------------------
# SimulatedExchangeAdapter
# ---------------------------------------------------------------------------


class SimulatedExchangeAdapter(ExchangeAdapter):
    """Fake exchange that satisfies ``ExchangeAdapter``."""

    def __init__(
        self,
        initial_balance: float,
        slippage_pct: float = 0.0005,
        fee_model: "ExecutionCostModel | None" = None,
        data_loader: "ParquetDataLoader | None" = None,
        data_adapter: "ExchangeAdapter | None" = None,
        spread_pct: float = 0.0001,
        asset_meta: dict[str, AssetMeta] | None = None,
        name: str = "simulated",
    ) -> None:
        if initial_balance < 0:
            raise ValueError(f"initial_balance must be >= 0, got {initial_balance}")
        if slippage_pct < 0:
            raise ValueError(f"slippage_pct must be >= 0, got {slippage_pct}")

        self._name = name
        self._initial_balance = float(initial_balance)
        self._balance = float(initial_balance)
        self._slippage_pct = float(slippage_pct)
        self._spread_pct = float(spread_pct)
        self._fee_model = fee_model
        self._data_loader = data_loader
        self._data_adapter = data_adapter
        self._asset_meta: dict[str, AssetMeta] = asset_meta or {}
        self._default_meta = AssetMeta()

        # Mutable state
        self._positions: dict[str, _SimPosition] = {}
        self._open_orders: dict[str, _SimOrder] = {}
        self._trade_history: list[dict] = []
        self._current_prices: dict[str, float] = {}
        self._current_candles: dict[str, dict] = {}
        self._equity_curve: list[tuple[int, float]] = []
        self._next_order_id: int = 1
        self._clock_ms: int = 0

    # ------------------------------------------------------------------
    # ExchangeAdapter ABC: identity + capabilities
    # ------------------------------------------------------------------

    def name(self) -> str:
        return self._name

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True,
            supports_short=True,
            market_hours=None,
            asset_types=["perpetual", "spot"],
            margin_type="cross",
            has_funding_rate=True,
            has_oi_data=False,
            max_leverage=50.0,
            order_types=["market", "limit", "stop", "take_profit"],
            supports_partial_close=True,
        )

    # ------------------------------------------------------------------
    # ExchangeAdapter ABC: data
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: int | None = None,
    ) -> list[dict]:
        """Delegate to data_adapter (shadow mode) or data_loader (backtest).

        ``data_adapter`` takes priority when both are configured — it
        services live data feeds in shadow mode without ParquetDataLoader
        having to know anything about live market state.
        """
        if self._data_adapter is not None:
            return await self._data_adapter.fetch_ohlcv(
                symbol, timeframe, limit, since
            )
        if self._data_loader is None:
            raise RuntimeError(
                "SimulatedExchangeAdapter has no data source. Construct with "
                "`data_loader=ParquetDataLoader(...)` for offline backtests "
                "or `data_adapter=<real ExchangeAdapter>` for shadow mode."
            )

        # The pipeline asks for the most recent `limit` candles relative to
        # the simulated clock — i.e. up to (but not including) the current
        # candle the BacktestEngine has handed us. If the clock is unset,
        # serve everything the loader has.
        if self._clock_ms == 0:
            # Best effort: serve a wide window. Tests inject the loader with
            # a tiny dataset; this avoids special-casing.
            start = datetime(1970, 1, 2, tzinfo=timezone.utc)
            end = datetime(2100, 1, 1, tzinfo=timezone.utc)
        else:
            from storage.cache.ttl import TIMEFRAME_SECONDS

            period_ms = TIMEFRAME_SECONDS.get(timeframe, 3600) * 1000
            end_ms = self._clock_ms + 1  # inclusive of current bar
            start_ms = end_ms - period_ms * max(limit, 1) * 2  # generous window
            start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
            end = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

        candles = self._data_loader.load_as_market_data(symbol, timeframe, start, end)
        if since is not None:
            candles = [c for c in candles if c["timestamp"] >= since]
        return candles[-limit:] if limit > 0 else candles

    async def get_ticker(self, symbol: str) -> dict:
        if self._data_adapter is not None:
            return await self._data_adapter.get_ticker(symbol)
        price = self._current_prices.get(symbol)
        if price is None:
            return {}
        half_spread = price * self._spread_pct / 2
        return {
            "bid": price - half_spread,
            "ask": price + half_spread,
            "last": price,
            "volume": 0.0,
        }

    async def get_balance(self) -> float:
        return self._balance

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        out: list[Position] = []
        for sym, pos in self._positions.items():
            if symbol is not None and sym != symbol:
                continue
            mark = self._current_prices.get(sym, pos.entry_price)
            out.append(
                Position(
                    symbol=sym,
                    direction=pos.direction,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    unrealized_pnl=self._unrealized_pnl(pos, mark),
                    leverage=None,
                )
            )
        return out

    # ------------------------------------------------------------------
    # ExchangeAdapter ABC: orders
    # ------------------------------------------------------------------

    async def place_market_order(
        self, symbol: str, side: str, size: float
    ) -> OrderResult:
        if size <= 0:
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"size must be > 0, got {size}",
            )
        if symbol not in self._current_prices:
            # Per spec: market order on a symbol with no current price is a
            # backtest-config bug. Fail loudly.
            raise ValueError(
                f"No current price for {symbol}; "
                f"call set_current_prices() or set_current_candle() first"
            )

        side = side.lower()
        if side not in ("buy", "sell"):
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"side must be buy/sell, got {side}",
            )

        mark = self._current_prices[symbol]
        fill_price = self._apply_slippage(mark, side)
        new_direction = "long" if side == "buy" else "short"
        order_id = self._gen_order_id("mkt")

        existing = self._positions.get(symbol)
        if existing is None or existing.direction == new_direction:
            self._open_or_add_position(symbol, new_direction, size, fill_price)
        else:
            # Opposing side: close (and flip if residual)
            close_size = min(size, existing.size)
            self._close_or_reduce_position(
                symbol, close_size, fill_price, reason="market_close",
            )
            residual = size - close_size
            if residual > 1e-12:
                self._open_or_add_position(
                    symbol, new_direction, residual, fill_price
                )

        return OrderResult(
            success=True,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=size,
            error=None,
        )

    async def place_limit_order(
        self, symbol: str, side: str, size: float, price: float
    ) -> OrderResult:
        if size <= 0 or price <= 0:
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"invalid size/price: {size}/{price}",
            )
        side = side.lower()
        if side not in ("buy", "sell"):
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"side must be buy/sell, got {side}",
            )
        order = _SimOrder(
            order_id=self._gen_order_id("lmt"),
            symbol=symbol,
            order_type="limit",
            side=side,
            size=size,
            price=price,
            created_at_ms=self._clock_ms,
        )
        self._open_orders[order.order_id] = order
        # A limit may already be marketable when placed; check immediately.
        self._check_limit_fills_for(symbol)
        return OrderResult(
            success=True,
            order_id=order.order_id,
            fill_price=None,
            fill_size=None,
            error=None,
        )

    async def place_sl_order(
        self, symbol: str, side: str, size: float, trigger_price: float
    ) -> OrderResult:
        return self._place_trigger_order(symbol, side, size, trigger_price, "stop")

    async def place_tp_order(
        self, symbol: str, side: str, size: float, trigger_price: float
    ) -> OrderResult:
        return self._place_trigger_order(
            symbol, side, size, trigger_price, "take_profit"
        )

    def _place_trigger_order(
        self, symbol: str, side: str, size: float, trigger_price: float, kind: str
    ) -> OrderResult:
        if size <= 0 or trigger_price <= 0:
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"invalid size/trigger: {size}/{trigger_price}",
            )
        side = side.lower()
        if side not in ("buy", "sell"):
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"side must be buy/sell, got {side}",
            )
        order = _SimOrder(
            order_id=self._gen_order_id(kind),
            symbol=symbol,
            order_type=kind,
            side=side,
            size=size,
            price=trigger_price,
            created_at_ms=self._clock_ms,
        )
        self._open_orders[order.order_id] = order
        return OrderResult(
            success=True,
            order_id=order.order_id,
            fill_price=None,
            fill_size=None,
            error=None,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        order = self._open_orders.get(order_id)
        if order is None or order.symbol != symbol:
            return False
        del self._open_orders[order_id]
        return True

    async def cancel_all_orders(self, symbol: str) -> int:
        ids = [oid for oid, o in self._open_orders.items() if o.symbol == symbol]
        for oid in ids:
            del self._open_orders[oid]
        return len(ids)

    async def close_position(self, symbol: str) -> OrderResult:
        pos = self._positions.get(symbol)
        if pos is None:
            # Idempotent: closing a non-existent position is a no-op success.
            return OrderResult(
                success=True, order_id=None, fill_price=None, fill_size=None,
                error=None,
            )
        if symbol not in self._current_prices:
            raise ValueError(
                f"No current price for {symbol}; cannot close position"
            )
        # Use the close-side: long → sell, short → buy
        close_side = "sell" if pos.direction == "long" else "buy"
        mark = self._current_prices[symbol]
        fill_price = self._apply_slippage(mark, close_side)
        order_id = self._gen_order_id("close")
        size = pos.size
        self._close_or_reduce_position(symbol, size, fill_price, reason="manual_close")
        return OrderResult(
            success=True,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=size,
            error=None,
        )

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        return self._modify_trigger(symbol, new_price, "stop")

    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult:
        return self._modify_trigger(symbol, new_price, "take_profit")

    def _modify_trigger(
        self, symbol: str, new_price: float, kind: str
    ) -> OrderResult:
        pos = self._positions.get(symbol)
        if pos is None:
            return OrderResult(
                success=False, order_id=None, fill_price=None, fill_size=None,
                error=f"No position on {symbol} to modify {kind}",
            )
        # Find existing trigger of this kind on this symbol; replace.
        existing = [
            oid for oid, o in self._open_orders.items()
            if o.symbol == symbol and o.order_type == kind
        ]
        for oid in existing:
            del self._open_orders[oid]
        side = "sell" if pos.direction == "long" else "buy"
        order = _SimOrder(
            order_id=self._gen_order_id(kind),
            symbol=symbol,
            order_type=kind,
            side=side,
            size=pos.size,
            price=new_price,
            created_at_ms=self._clock_ms,
        )
        self._open_orders[order.order_id] = order
        return OrderResult(
            success=True,
            order_id=order.order_id,
            fill_price=None,
            fill_size=None,
            error=None,
        )

    # ------------------------------------------------------------------
    # ExchangeAdapter optional methods
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> float | None:
        if self._data_adapter is not None:
            return await self._data_adapter.get_funding_rate(symbol)
        return 0.0  # neutral by default; BacktestEngine drives apply_funding

    async def get_open_interest(self, symbol: str) -> float | None:
        if self._data_adapter is not None:
            return await self._data_adapter.get_open_interest(symbol)
        return None

    async def fetch_meta(self) -> list[dict]:
        if self._data_adapter is not None:
            return await self._data_adapter.fetch_meta()
        return [
            {
                "symbol": sym,
                "tick_size": meta.tick_size,
                "lot_size": meta.lot_size,
                "min_notional": meta.min_notional,
            }
            for sym, meta in self._asset_meta.items()
        ]

    async def fetch_orderbook(self, symbol: str, limit: int = 10) -> dict:
        if self._data_adapter is not None:
            return await self._data_adapter.fetch_orderbook(symbol, limit)
        price = self._current_prices.get(symbol)
        if price is None:
            return {"bids": [], "asks": []}
        half = price * self._spread_pct / 2
        # Synthetic single-level book; size is large enough that the engine's
        # slippage walk hits the configured slippage_pct, not depth limits.
        return {
            "bids": [[price - half, 1e9]],
            "asks": [[price + half, 1e9]],
        }

    async def fetch_user_fees(self) -> dict:
        if self._data_adapter is not None:
            return await self._data_adapter.fetch_user_fees()
        return {"tier": 0, "staking_discount": 0.0, "referral_discount": 0.0}

    # ------------------------------------------------------------------
    # Backtest control surface (NOT part of the ABC)
    # ------------------------------------------------------------------

    def set_current_prices(
        self, prices: dict[str, float], timestamp: int | None = None
    ) -> None:
        """Update the simulated last-price for one or more symbols.

        Called by the BacktestEngine on every tick. Triggers limit-order
        fills and snapshots equity.
        """
        if not prices:
            return
        self._current_prices.update(prices)
        if timestamp is not None:
            self._clock_ms = int(timestamp)
        for symbol in prices:
            self._check_limit_fills_for(symbol)
        self._snapshot_equity()

    def set_current_candle(self, symbol: str, candle: dict) -> None:
        """Hand the sim a fully-formed candle bar.

        Triggers SL/TP checks against the candle's high/low (with SL
        winning ties), then limit-order fills against the close, then
        snapshots equity.
        """
        for k in ("timestamp", "open", "high", "low", "close"):
            if k not in candle:
                raise ValueError(f"candle missing required key: {k}")
        self._current_candles[symbol] = candle
        self._current_prices[symbol] = float(candle["close"])
        self._clock_ms = int(candle["timestamp"])
        self._check_sl_tp_triggers(symbol, candle)
        self._check_limit_fills_for(symbol)
        self._snapshot_equity()

    def apply_funding(self, symbol: str, rate: float) -> None:
        """Apply one funding payment to the open position on ``symbol``.

        Convention: positive rate means longs pay, shorts receive (matches
        Hyperliquid / Binance perp funding).
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return
        mark = self._current_prices.get(symbol, pos.entry_price)
        notional = pos.size * mark
        sign = -1.0 if pos.direction == "long" else 1.0
        delta = sign * rate * notional
        self._balance += delta
        pos.funding_paid += -delta  # accumulated cost (positive = cost)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "type": o.order_type,
                "side": o.side,
                "size": o.size,
                "price": o.price,
                "created_at_ms": o.created_at_ms,
            }
            for o in self._open_orders.values()
            if symbol is None or o.symbol == symbol
        ]

    def get_trade_history(self) -> list[dict]:
        return list(self._trade_history)

    def get_equity_curve(self) -> list[tuple[int, float]]:
        return list(self._equity_curve)

    def equity(self) -> float:
        """Current total equity = balance + sum unrealized PnL."""
        upnl = 0.0
        for sym, pos in self._positions.items():
            mark = self._current_prices.get(sym, pos.entry_price)
            upnl += self._unrealized_pnl(pos, mark)
        return self._balance + upnl

    @property
    def balance(self) -> float:
        return self._balance

    # ------------------------------------------------------------------
    # Internals: positions
    # ------------------------------------------------------------------

    def _open_or_add_position(
        self,
        symbol: str,
        direction: str,
        size: float,
        fill_price: float,
    ) -> None:
        fee = self._charge_fee(symbol, size, fill_price)
        existing = self._positions.get(symbol)
        if existing is None:
            self._positions[symbol] = _SimPosition(
                symbol=symbol,
                direction=direction,
                size=size,
                entry_price=fill_price,
                entry_fees=fee,
                opened_at_ms=self._clock_ms,
            )
        else:
            # Weighted-average entry price.
            new_size = existing.size + size
            existing.entry_price = (
                existing.entry_price * existing.size + fill_price * size
            ) / new_size
            existing.size = new_size
            existing.entry_fees += fee

    def _close_or_reduce_position(
        self,
        symbol: str,
        close_size: float,
        fill_price: float,
        reason: str,
    ) -> None:
        pos = self._positions[symbol]
        close_size = min(close_size, pos.size)
        if close_size <= 0:
            return

        # PnL on the closed portion
        if pos.direction == "long":
            gross_pnl = (fill_price - pos.entry_price) * close_size
        else:
            gross_pnl = (pos.entry_price - fill_price) * close_size

        # Pro-rated entry fee + fresh exit fee
        portion = close_size / pos.size
        entry_fee_share = pos.entry_fees * portion
        exit_fee = self._charge_fee(symbol, close_size, fill_price)
        total_fee = entry_fee_share + exit_fee

        # Realise PnL on balance (gross_pnl already deducted via fees)
        self._balance += gross_pnl
        net_pnl = gross_pnl - total_fee

        # Slippage cost on this exit fill
        intended = self._current_prices.get(symbol, fill_price)
        slippage_cost = abs(intended - fill_price) * close_size

        self._trade_history.append(
            {
                "timestamp": self._clock_ms,
                "entry_timestamp": pos.opened_at_ms,
                "symbol": symbol,
                "side": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": fill_price,
                "size": close_size,
                "fee": total_fee,
                "slippage": slippage_cost,
                "pnl": net_pnl,
                "reason": reason,
            }
        )

        # Update or remove the position
        pos.size -= close_size
        pos.entry_fees -= entry_fee_share
        if pos.size <= 1e-12:
            del self._positions[symbol]
            # Cancel any leftover SL/TP orders for this (now-flat) symbol
            stale = [
                oid for oid, o in self._open_orders.items()
                if o.symbol == symbol and o.order_type in ("stop", "take_profit")
            ]
            for oid in stale:
                del self._open_orders[oid]

    def _unrealized_pnl(self, pos: _SimPosition, mark: float) -> float:
        if pos.direction == "long":
            return (mark - pos.entry_price) * pos.size
        return (pos.entry_price - mark) * pos.size

    # ------------------------------------------------------------------
    # Internals: order matching
    # ------------------------------------------------------------------

    def _check_sl_tp_triggers(self, symbol: str, candle: dict) -> None:
        """Process SL first, then TP. Closing the position auto-cancels both."""
        high = float(candle["high"])
        low = float(candle["low"])

        # Snapshot orders to avoid mutating during iteration.
        sl_orders = [
            o for o in self._open_orders.values()
            if o.symbol == symbol and o.order_type == "stop"
        ]
        for o in sl_orders:
            if o.order_id not in self._open_orders:
                continue  # already cancelled by an earlier fill in this loop
            if self._sl_triggered(o, high, low):
                self._fill_trigger_order(o)

        tp_orders = [
            o for o in self._open_orders.values()
            if o.symbol == symbol and o.order_type == "take_profit"
        ]
        for o in tp_orders:
            if o.order_id not in self._open_orders:
                continue
            if self._tp_triggered(o, high, low):
                self._fill_trigger_order(o)

    def _sl_triggered(self, order: _SimOrder, high: float, low: float) -> bool:
        # SL on a long is a sell stop below current → triggers when low ≤ price.
        # SL on a short is a buy stop above current → triggers when high ≥ price.
        if order.side == "sell":
            return low <= order.price
        return high >= order.price

    def _tp_triggered(self, order: _SimOrder, high: float, low: float) -> bool:
        # TP on a long is a sell limit above → triggers when high ≥ price.
        # TP on a short is a buy limit below → triggers when low ≤ price.
        if order.side == "sell":
            return high >= order.price
        return low <= order.price

    def _fill_trigger_order(self, order: _SimOrder) -> None:
        pos = self._positions.get(order.symbol)
        if pos is None:
            # Stale order with no position behind it; drop.
            self._open_orders.pop(order.order_id, None)
            return
        # Fill at trigger price ± slippage (always against the trader)
        fill_price = self._apply_slippage(order.price, order.side)
        close_size = min(order.size, pos.size)
        self._close_or_reduce_position(
            order.symbol,
            close_size,
            fill_price,
            reason=f"{order.order_type}_hit",
        )
        # Remove this order regardless of whether the close fully drained
        # the position (the close path also auto-cancels siblings).
        self._open_orders.pop(order.order_id, None)

    def _check_limit_fills_for(self, symbol: str) -> None:
        price = self._current_prices.get(symbol)
        if price is None:
            return
        limit_orders = [
            o for o in self._open_orders.values()
            if o.symbol == symbol and o.order_type == "limit"
        ]
        for o in limit_orders:
            if o.order_id not in self._open_orders:
                continue
            triggered = (
                (o.side == "buy" and price <= o.price)
                or (o.side == "sell" and price >= o.price)
            )
            if not triggered:
                continue
            # Limits fill at the limit price (no slippage)
            new_direction = "long" if o.side == "buy" else "short"
            existing = self._positions.get(symbol)
            if existing is None or existing.direction == new_direction:
                self._open_or_add_position(symbol, new_direction, o.size, o.price)
            else:
                close_size = min(o.size, existing.size)
                self._close_or_reduce_position(
                    symbol, close_size, o.price, reason="limit_fill"
                )
                residual = o.size - close_size
                if residual > 1e-12:
                    self._open_or_add_position(
                        symbol, new_direction, residual, o.price
                    )
            self._open_orders.pop(o.order_id, None)

    # ------------------------------------------------------------------
    # Internals: fees, slippage, equity, ids
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "buy":
            return price * (1.0 + self._slippage_pct)
        return price * (1.0 - self._slippage_pct)

    def _charge_fee(self, symbol: str, size: float, price: float) -> float:
        if self._fee_model is None:
            return 0.0
        rate = self._fee_model.get_taker_rate(symbol)
        notional = size * price
        fee = notional * rate
        self._balance -= fee
        return fee

    def _snapshot_equity(self) -> None:
        eq = self.equity()
        # Coalesce duplicate timestamps (e.g. set_current_prices then
        # set_current_candle in the same tick) — keep the latest value.
        if self._equity_curve and self._equity_curve[-1][0] == self._clock_ms:
            self._equity_curve[-1] = (self._clock_ms, eq)
        else:
            self._equity_curve.append((self._clock_ms, eq))

    def _gen_order_id(self, kind: str) -> str:
        oid = f"sim-{kind}-{self._next_order_id:08d}"
        self._next_order_id += 1
        return oid
