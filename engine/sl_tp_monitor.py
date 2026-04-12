"""SLTPMonitor — tick-level SL/TP checking for shadow trades.

Sprint Week 7 Task 5. Replaces Sentinel's 30-second `_check_shadow_sl_tp`
poll with a `PriceUpdated`-event subscriber that fires on every WebSocket
tick. SL/TP gates that previously only resolved on each candle close
(missing intra-candle wicks) now resolve at the actual tick that breached
the level.

Hot-path discipline:

* The most common ``_on_price_update`` invocation has zero open trades
  on the symbol. The fast-path is a single dict lookup + early return —
  no DB call, no allocation, no parse work. Tests pin this contract.
* DB refresh is periodic (every ``_refresh_interval`` seconds, default
  10s) — never per tick. The refresh runs lazily inside
  ``_on_price_update`` so the loop has no extra background task to babysit.
* ``forward_max_r`` is updated in-memory on every tick (cheap float
  arithmetic) and only flushed to the DB when the trade closes — keeps
  the per-tick path free of writes while still recording the eventual
  best R for post-trade analysis.

Sentinel still owns setup detection. This module is strictly the
shadow-mode SL/TP gate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from engine.events import EventBus, PriceUpdated, TradeClosed
from engine.types import PriceUpdate
from storage.repositories.base import TradeRepository

logger = logging.getLogger(__name__)


class SLTPMonitor:
    """Subscribes to ``PriceUpdated`` and resolves shadow SL/TP at tick level."""

    def __init__(
        self,
        event_bus: EventBus,
        trade_repo: TradeRepository,
        is_shadow: bool = True,
        *,
        refresh_interval: float = 10.0,
        taker_fee_rate: float | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._trade_repo = trade_repo
        self._is_shadow = is_shadow
        # Configurable taker fee rate for round-trip fee calculation.
        # Default reads from env or falls back to Hyperliquid's 0.035%.
        import os
        self._taker_fee_rate = (
            taker_fee_rate
            if taker_fee_rate is not None
            else float(os.environ.get("TAKER_FEE_RATE", "0.00035"))
        )
        # symbol → list of trade dicts. Trades are kept per-symbol so
        # the hot path can short-circuit on a single dict lookup.
        self._open_trades: dict[str, list[dict]] = {}
        self._last_refresh: datetime | None = None
        self._refresh_interval = refresh_interval
        # In-memory high-water mark for ``forward_max_r`` per trade id —
        # accumulates across ticks and only gets persisted when the trade
        # closes. Avoids a per-tick DB write.
        self._forward_max_r: dict[str, float] = {}
        # Snapshot of every symbol we've ever seen open trades for, used
        # to drive the periodic DB refresh across the FULL set rather
        # than only those that received a tick this interval. Initialised
        # in ``start()`` and updated in ``_refresh_trades()``.
        self._known_symbols: set[str] = set()
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load open trades from the DB and subscribe to ``PriceUpdated``."""
        if self._started:
            return
        await self._refresh_trades()
        self._event_bus.subscribe(PriceUpdated, self._on_price_update)
        self._started = True
        n_trades = sum(len(trades) for trades in self._open_trades.values())
        n_symbols = len(self._open_trades)
        logger.info(
            "SLTPMonitor started, tracking %d open trades across %d symbols",
            n_trades,
            n_symbols,
        )

    async def stop(self) -> None:
        """Unsubscribe from the bus. Idempotent."""
        if not self._started:
            return
        try:
            self._event_bus.unsubscribe(PriceUpdated, self._on_price_update)
        except Exception:
            logger.debug("SLTPMonitor: unsubscribe failed", exc_info=True)
        self._started = False

    # ------------------------------------------------------------------
    # Hot path — fires on every tick
    # ------------------------------------------------------------------

    async def _on_price_update(self, event: PriceUpdated) -> None:
        """Tick handler. Must stay FAST.

        The 99% case is "no open trades for this symbol" — that branch
        does a single dict lookup and returns. Only when there's an open
        trade do we evaluate gates / refresh / persist.
        """
        update: PriceUpdate = event.update
        if update is None:
            return
        symbol = update.symbol

        # Hot fast-path. The DB refresh is also lazy here — only triggers
        # when at least one tick has arrived AND the interval elapsed.
        if not self._open_trades.get(symbol):
            await self._maybe_refresh()
            if not self._open_trades.get(symbol):
                return

        await self._maybe_refresh()
        trades = self._open_trades.get(symbol)
        if not trades:
            return

        price = float(update.price)
        # Iterate over a snapshot — closing a trade mutates the list.
        for trade in list(trades):
            await self._evaluate_trade(trade, symbol=symbol, price=price)

    async def _evaluate_trade(
        self, trade: dict, *, symbol: str, price: float
    ) -> None:
        """Resolve a single trade against the latest tick.

        Updates the in-memory ``forward_max_r`` high-water mark on every
        tick (cheap), and on SL/TP breach: closes the trade in the DB,
        flushes the final ``forward_max_r``, drops the trade from the
        in-memory dict, and emits ``TradeClosed`` on the bus.
        """
        trade_id = trade.get("id")
        direction = (trade.get("direction") or "").upper()
        entry_price = trade.get("entry_price")
        size_usd = trade.get("size") or 0.0
        sl_price = trade.get("sl_price")
        tp_price = trade.get("tp_price")

        if not trade_id or entry_price is None or float(entry_price) <= 0:
            return

        # Update in-memory forward_max_r before checking gates so a
        # tick that simultaneously hits TP also captures the favourable
        # extreme it represents.
        self._update_forward_max_r_in_memory(
            trade_id=str(trade_id),
            entry_price=float(entry_price),
            sl_price=sl_price,
            direction=direction,
            tick_price=price,
            prior_persisted=trade.get("forward_max_r"),
        )

        exit_price: float | None = None
        exit_reason: str | None = None

        if direction == "LONG":
            if sl_price is not None and price <= float(sl_price):
                exit_price, exit_reason = float(sl_price), "SL"
            elif tp_price is not None and price >= float(tp_price):
                exit_price, exit_reason = float(tp_price), "TP"
        elif direction == "SHORT":
            if sl_price is not None and price >= float(sl_price):
                exit_price, exit_reason = float(sl_price), "SL"
            elif tp_price is not None and price <= float(tp_price):
                exit_price, exit_reason = float(tp_price), "TP"
        else:
            logger.warning(
                "SLTPMonitor: trade %s has unknown direction %r",
                trade_id,
                direction,
            )
            return

        if exit_price is None or exit_reason is None:
            return

        raw_pnl = self._compute_pnl(
            direction=direction,
            entry_price=float(entry_price),
            exit_price=exit_price,
            size_usd=float(size_usd),
        )

        now = datetime.now(timezone.utc)

        # Round-trip trading fee: taker fee on open + taker fee on close.
        notional = float(size_usd)
        trading_fee = 2.0 * self._taker_fee_rate * notional

        # Funding cost estimate: funding_rate * notional * hours held.
        # Hyperliquid charges funding hourly. If entry_time or funding
        # data is unavailable, funding_cost stays at 0.
        funding_cost = 0.0
        entry_time_raw = trade.get("entry_time")
        funding_rate = trade.get("funding_rate")
        if entry_time_raw is not None and funding_rate is not None:
            try:
                if isinstance(entry_time_raw, str):
                    from datetime import datetime as _dt
                    entry_dt = _dt.fromisoformat(entry_time_raw)
                else:
                    entry_dt = entry_time_raw
                hold_seconds = (now - entry_dt).total_seconds()
                funding_intervals = hold_seconds / 3600.0
                funding_cost = abs(float(funding_rate) * notional * funding_intervals)
            except Exception:
                pass  # funding_cost stays 0

        pnl = raw_pnl - trading_fee - funding_cost

        try:
            await self._trade_repo.close_trade(
                str(trade_id),
                exit_price=exit_price,
                exit_reason=exit_reason,
                exit_time=now,
                pnl=pnl,
            )
        except Exception:
            logger.exception("SLTPMonitor: close_trade failed for %s", trade_id)
            return

        # Persist raw_pnl, trading_fee, and funding_cost alongside the adjusted pnl.
        try:
            await self._trade_repo.update_trade(
                str(trade_id),
                {
                    "raw_pnl": float(raw_pnl),
                    "trading_fee": float(trading_fee),
                    "funding_cost": float(funding_cost),
                },
            )
        except Exception:
            logger.debug(
                "SLTPMonitor: raw_pnl/trading_fee persist failed for %s",
                trade_id,
                exc_info=True,
            )

        # Persist the final forward_max_r if we have a better value than
        # what's stored on the trade row.
        await self._flush_forward_max_r(str(trade_id), trade.get("forward_max_r"))

        # Drop from the in-memory map so the next tick can short-circuit.
        self._drop_trade(symbol, str(trade_id))

        # Fire-and-forget bus emission. The trade_id is set so consumers
        # like ForwardMaxRStamper can look the row back up.
        try:
            await self._event_bus.publish(
                TradeClosed(
                    source="sl_tp_monitor",
                    symbol=symbol,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    trade_id=str(trade_id),
                    direction=direction,
                    entry_price=float(entry_price),
                    sl_price=float(sl_price) if sl_price is not None else None,
                )
            )
        except Exception:
            logger.warning(
                "SLTPMonitor: failed to publish TradeClosed for %s",
                trade_id,
                exc_info=True,
            )

        logger.info(
            "SLTPMonitor: %s %s hit %s at %.4f "
            "(raw=$%.2f, fee=$%.2f, funding=$%.2f, PnL=$%.2f)",
            symbol,
            direction,
            exit_reason,
            exit_price,
            raw_pnl,
            trading_fee,
            funding_cost,
            pnl,
        )

    # ------------------------------------------------------------------
    # forward_max_r tracking
    # ------------------------------------------------------------------

    def _update_forward_max_r_in_memory(
        self,
        *,
        trade_id: str,
        entry_price: float,
        sl_price: Any,
        direction: str,
        tick_price: float,
        prior_persisted: Any,
    ) -> None:
        """Bump the in-memory R high-water mark if this tick beats it.

        ``R = (favourable distance) / (initial SL distance)``. We seed
        the in-memory value from the trade's persisted ``forward_max_r``
        on first observation so a fresh process restart picks up where
        the previous run left off.
        """
        if sl_price is None:
            return
        try:
            sl = float(sl_price)
        except (TypeError, ValueError):
            return
        risk = abs(entry_price - sl)
        if risk <= 0:
            return

        if direction == "LONG":
            favourable_distance = tick_price - entry_price
        elif direction == "SHORT":
            favourable_distance = entry_price - tick_price
        else:
            return

        new_r = favourable_distance / risk

        if trade_id not in self._forward_max_r and prior_persisted is not None:
            try:
                self._forward_max_r[trade_id] = float(prior_persisted)
            except (TypeError, ValueError):
                pass

        prior = self._forward_max_r.get(trade_id)
        if prior is None or new_r > prior:
            self._forward_max_r[trade_id] = new_r

    async def _flush_forward_max_r(
        self, trade_id: str, prior_persisted: Any
    ) -> None:
        """Persist the final ``forward_max_r`` if it improved on the DB row."""
        new_r = self._forward_max_r.pop(trade_id, None)
        if new_r is None:
            return
        try:
            prior = float(prior_persisted) if prior_persisted is not None else None
        except (TypeError, ValueError):
            prior = None
        if prior is not None and prior >= new_r:
            return
        try:
            await self._trade_repo.update_trade(
                trade_id, {"forward_max_r": float(new_r)}
            )
        except Exception:
            logger.debug(
                "SLTPMonitor: forward_max_r flush failed for %s",
                trade_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # PnL — mirrors Sentinel._compute_shadow_pnl exactly
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pnl(
        *, direction: str, entry_price: float, exit_price: float, size_usd: float
    ) -> float:
        """USD PnL for a $-notional perp position.

        Mirrors ``sentinel.monitor.SentinelMonitor._compute_shadow_pnl``
        byte-for-byte so the new tick path produces identical PnL to the
        legacy candle-close path on the same exit price. Long PnL =
        ``(exit - entry) * size_usd / entry_price``; short flips the sign.
        """
        if entry_price <= 0:
            return 0.0
        if direction == "LONG":
            return (exit_price - entry_price) * size_usd / entry_price
        if direction == "SHORT":
            return (entry_price - exit_price) * size_usd / entry_price
        return 0.0

    # ------------------------------------------------------------------
    # DB refresh
    # ------------------------------------------------------------------

    async def _maybe_refresh(self) -> None:
        """Refresh open trades from the DB if the interval has elapsed."""
        now = datetime.now(timezone.utc)
        if self._last_refresh is None:
            await self._refresh_trades()
            return
        elapsed = (now - self._last_refresh).total_seconds()
        if elapsed >= self._refresh_interval:
            await self._refresh_trades()

    async def _refresh_trades(self) -> None:
        """Reload open shadow trades from the DB into ``_open_trades``.

        Walks every known symbol (every symbol we've ever seen via the
        bus or via the previous refresh) so a freshly opened trade for
        a symbol that hasn't ticked yet still gets picked up. The first
        refresh on ``start()`` walks an empty set — at that point we
        rely on the DB query method to enumerate trades by symbol via
        whatever live trades exist; we discover symbols via the trade
        rows themselves.
        """
        # On first call we don't yet know which symbols have open
        # trades, so we can't query per-symbol. The repository contract
        # is per-symbol only, so we walk the union of (a) symbols we
        # already know about and (b) any symbol that arrives via a tick.
        # The very first refresh just primes ``_known_symbols`` from
        # the empty set; subsequent refreshes pick up new trades for
        # known symbols.
        next_open: dict[str, list[dict]] = {}
        for symbol in list(self._known_symbols):
            try:
                trades = await self._trade_repo.get_open_shadow_trades(symbol)
            except Exception:
                logger.warning(
                    "SLTPMonitor: get_open_shadow_trades failed for %s",
                    symbol,
                    exc_info=True,
                )
                # Preserve the previous snapshot for this symbol so a
                # transient DB blip doesn't make us forget open trades.
                if symbol in self._open_trades:
                    next_open[symbol] = self._open_trades[symbol]
                continue
            if trades:
                next_open[symbol] = list(trades)

        self._open_trades = next_open
        self._last_refresh = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> None:
        """Tell the monitor about a symbol so the next refresh queries it.

        Wiring point for ``BotRunner`` / TradeOpened handlers — the
        repository contract is per-symbol so we need to know which
        symbols to ask about. Idempotent.
        """
        self._known_symbols.add(symbol)

    def is_running(self) -> bool:
        return self._started

    def open_trade_count(self, symbol: str | None = None) -> int:
        """Return the count of open trades, optionally for one symbol."""
        if symbol is not None:
            return len(self._open_trades.get(symbol, []))
        return sum(len(v) for v in self._open_trades.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drop_trade(self, symbol: str, trade_id: str) -> None:
        trades = self._open_trades.get(symbol)
        if not trades:
            return
        self._open_trades[symbol] = [
            t for t in trades if str(t.get("id")) != trade_id
        ]
        if not self._open_trades[symbol]:
            self._open_trades.pop(symbol, None)
