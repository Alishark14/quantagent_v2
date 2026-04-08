"""SimExecutor — thin facade over SimulatedExchangeAdapter for backtests.

The SimulatedExchangeAdapter already records the trade history and equity
curve internally (it's the only place with full state). SimExecutor exists
as the public read surface that the BacktestEngine and metrics modules
talk to:

- ``get_trade_history()`` — list of completed-trade dicts
- ``get_equity_curve()`` — list of ``(timestamp_ms, equity)`` snapshots
- pass-through helpers for the BacktestEngine tick loop

Keeping these on the executor (not the adapter) means the live engine code
that consumes ``ExchangeAdapter`` never sees backtest-only methods, and a
future production executor can implement the same read surface without
inheriting from ``ExchangeAdapter``.
"""

from __future__ import annotations

import logging

from backtesting.sim_exchange import SimulatedExchangeAdapter

logger = logging.getLogger(__name__)


class SimExecutor:
    """Read/observation facade over a ``SimulatedExchangeAdapter``."""

    def __init__(self, adapter: SimulatedExchangeAdapter) -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Read surface (used by metrics, reporter, dashboard)
    # ------------------------------------------------------------------

    def get_trade_history(self) -> list[dict]:
        """All closed trades, in close-time order.

        Each record:
        ``{timestamp, symbol, side, entry_price, exit_price, size, fee,
        slippage, pnl, reason}``
        """
        return self._adapter.get_trade_history()

    def get_equity_curve(self) -> list[tuple[int, float]]:
        """Time series of ``(timestamp_ms, equity)``.

        Equity = balance + sum of unrealised PnL on open positions.
        Snapshots happen automatically each time the BacktestEngine
        advances the simulated clock via ``set_current_prices`` /
        ``set_current_candle`` on the adapter.
        """
        return self._adapter.get_equity_curve()

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return self._adapter.get_open_orders(symbol)

    @property
    def balance(self) -> float:
        return self._adapter.balance

    def equity(self) -> float:
        return self._adapter.equity()

    @property
    def adapter(self) -> SimulatedExchangeAdapter:
        return self._adapter

    # ------------------------------------------------------------------
    # Tick driver — convenience wrappers the BacktestEngine calls
    # ------------------------------------------------------------------

    def on_candle(self, symbol: str, candle: dict) -> None:
        """Hand a candle to the underlying adapter (drives SL/TP + equity)."""
        self._adapter.set_current_candle(symbol, candle)

    def on_prices(
        self, prices: dict[str, float], timestamp: int | None = None
    ) -> None:
        self._adapter.set_current_prices(prices, timestamp=timestamp)

    def apply_funding(self, symbol: str, rate: float) -> None:
        self._adapter.apply_funding(symbol, rate)

    # ------------------------------------------------------------------
    # Aggregate metrics (cheap to compute on demand)
    # ------------------------------------------------------------------

    def total_pnl(self) -> float:
        return sum(t["pnl"] for t in self._adapter.get_trade_history())

    def total_fees(self) -> float:
        return sum(t["fee"] for t in self._adapter.get_trade_history())

    def num_trades(self) -> int:
        return len(self._adapter.get_trade_history())

    def win_rate(self) -> float:
        history = self._adapter.get_trade_history()
        if not history:
            return 0.0
        wins = sum(1 for t in history if t["pnl"] > 0)
        return wins / len(history)
