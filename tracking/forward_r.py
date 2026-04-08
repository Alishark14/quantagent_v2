"""ForwardMaxRStamper — stamp `forward_max_r` on every closed trade.

`forward_max_r` is the maximum favourable excursion (in R-multiples)
that price reached over the N candles after a trade entry. The
auto-miner uses it to detect "missed opportunity" trades — setups the
engine skipped or treated with low conviction that *would have* paid
≥ 3R if the engine had taken them. ARCHITECTURE.md §31.4.5.

This module ships:

- :func:`compute_forward_max_r`: a pure helper. Takes direction +
  entry price + risk + a forward OHLCV path and returns the
  max-favourable-excursion in R units. Used by both the stamper and
  any backfill script.

- :class:`ForwardMaxRStamper`: orchestration class. Loads the forward
  path from Parquet via :class:`ForwardPathLoader`, calls the helper,
  and persists the result via the trade repository's ``update_trade``.
  Has both a sync-friendly ``stamp_trade(trade_id)`` method and an
  async event-bus handler ``on_trade_closed(event)``.

Failure modes are designed to be silent at the pipeline boundary:
missing Parquet data, missing trade record, missing entry/risk
fields, or unsupported direction all log a warning and return
``None`` instead of raising. Trade execution must NEVER be impacted
by tracking failures (CLAUDE.md rule #11 — "tracking failures never
propagate").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from backtesting.forward_path import ForwardPathLoader
    from engine.events import TradeClosed
    from storage.repositories.base import TradeRepository


# Fallback risk fraction when neither sl_price nor an explicit risk is
# carried on the trade dict. 1% of entry is the engine's default risk
# per trade — keeps the metric meaningful even on records that
# predate the SL-on-trade-record schema enrichment.
_DEFAULT_RISK_FRACTION = 0.01

# Default forward window: 60 bars at the recommended high-resolution
# timeframe for the trade (1m for ≤1h trades, 5m for higher TFs).
_DEFAULT_FORWARD_CANDLES = 60


# ---------------------------------------------------------------------------
# Pure computation helper
# ---------------------------------------------------------------------------


def compute_forward_max_r(
    direction: str,
    entry_price: float,
    risk: float,
    forward_path: Any,
) -> float | None:
    """Compute max favourable excursion in R-multiples.

    Args:
        direction: ``"LONG"`` / ``"long"`` or ``"SHORT"`` / ``"short"``.
        entry_price: Trade entry price (must be > 0).
        risk: Per-unit risk (entry-to-sl distance, must be > 0).
        forward_path: A polars DataFrame OR a list of OHLCV dicts —
            anything with a ``high`` and ``low`` column / key. Empty
            input returns 0.0 (no excursion observed).

    Returns:
        Max R reached, or ``None`` if inputs are invalid.

    Notes:
        - LONG: ``max((high - entry) / risk)`` across the forward path.
        - SHORT: ``max((entry - low) / risk)`` across the forward path.
        - Result is clamped at 0 — a forward path that only moved
          adversely returns 0 R, never negative.
    """
    if entry_price is None or entry_price <= 0:
        logger.debug("compute_forward_max_r: invalid entry_price")
        return None
    if risk is None or risk <= 0:
        logger.debug("compute_forward_max_r: invalid risk")
        return None
    if direction is None:
        return None

    direction_upper = direction.upper()
    if direction_upper not in ("LONG", "SHORT"):
        logger.debug(f"compute_forward_max_r: unsupported direction {direction!r}")
        return None

    highs, lows = _extract_highs_lows(forward_path)
    if not highs or not lows:
        return 0.0

    if direction_upper == "LONG":
        best_excursion = max(highs) - entry_price
    else:
        best_excursion = entry_price - min(lows)

    return max(0.0, best_excursion / risk)


def _extract_highs_lows(forward_path: Any) -> tuple[list[float], list[float]]:
    """Pull high/low arrays from either a polars DF or a list of dicts."""
    if forward_path is None:
        return [], []

    # Polars DataFrame path
    if hasattr(forward_path, "is_empty") and hasattr(forward_path, "get_column"):
        try:
            if forward_path.is_empty():
                return [], []
            highs = forward_path.get_column("high").to_list()
            lows = forward_path.get_column("low").to_list()
            return [float(x) for x in highs], [float(x) for x in lows]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"compute_forward_max_r: polars extract failed: {e}")
            return [], []

    # List-of-dicts path
    try:
        highs = [float(row["high"]) for row in forward_path]
        lows = [float(row["low"]) for row in forward_path]
        return highs, lows
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"compute_forward_max_r: list extract failed: {e}")
        return [], []


# ---------------------------------------------------------------------------
# Stamper class
# ---------------------------------------------------------------------------


class ForwardMaxRStamper:
    """Loads a forward path, computes forward_max_r, persists it on the trade.

    Designed to be used in two ways:

    1. **Subscribed to TradeClosed events** via :class:`TrackingModule`.
       When the executor emits a `TradeClosed` event with a `trade_id`,
       the stamper looks up the trade record, computes the metric, and
       writes it back via ``repo.update_trade``.

    2. **Standalone backfill** via ``stamp_trade(trade_id)`` or
       ``stamp_trade_dict(trade_dict)`` — useful for the auto-mining
       script that runs against historical trades.

    Both paths are fail-safe: any error (missing Parquet, missing
    fields, repo failures) logs a warning and returns ``None`` instead
    of propagating.
    """

    def __init__(
        self,
        repo: "TradeRepository",
        forward_path_loader: "ForwardPathLoader",
        forward_candles: int = _DEFAULT_FORWARD_CANDLES,
        default_risk_fraction: float = _DEFAULT_RISK_FRACTION,
    ) -> None:
        """
        Args:
            repo: Anything implementing the TradeRepository ABC. Must
                support ``get_trade(id)`` and ``update_trade(id, dict)``.
            forward_path_loader: Loads N high-resolution candles after
                a trade entry. The stamper picks ``1m`` resolution for
                ≤ 1h trades, ``5m`` for higher TFs.
            forward_candles: How many candles to look forward. Default
                60 — about 1 hour at 1m resolution, 5 hours at 5m.
            default_risk_fraction: Fallback risk-per-unit when the
                trade dict has no ``sl_price`` and no explicit risk.
                Defaults to 1% of entry — matches the engine's default
                risk-per-trade so back-filled R values stay comparable
                to live ones.
        """
        self._repo = repo
        self._loader = forward_path_loader
        self._forward_candles = int(forward_candles)
        self._default_risk_fraction = float(default_risk_fraction)

    # ------------------------------------------------------------------
    # Event handler — TradeClosed → stamp
    # ------------------------------------------------------------------

    async def on_trade_closed(self, event: "TradeClosed") -> float | None:
        """TrackingModule subscribes this to the event bus.

        Reads ``event.trade_id`` and dispatches to ``stamp_trade``. If
        the trade_id isn't on the event (older emission sites that
        don't yet thread it through), logs and returns None.
        """
        trade_id = getattr(event, "trade_id", None)
        if trade_id is None:
            logger.debug(
                "ForwardMaxRStamper: TradeClosed event missing trade_id "
                f"(symbol={event.symbol}, reason={event.exit_reason}). "
                "Skipping forward_max_r stamp — emission site needs to be updated."
            )
            return None
        return await self.stamp_trade(trade_id)

    # ------------------------------------------------------------------
    # Public stamping API
    # ------------------------------------------------------------------

    async def stamp_trade(self, trade_id: str) -> float | None:
        """Look up the trade by id, compute forward_max_r, persist it.

        Returns the computed value, or ``None`` on any failure.
        """
        try:
            trade = await self._repo.get_trade(trade_id)
        except Exception as e:
            logger.warning(
                f"ForwardMaxRStamper: get_trade({trade_id}) failed: {e}"
            )
            return None
        if trade is None:
            logger.warning(
                f"ForwardMaxRStamper: trade {trade_id} not found in repo"
            )
            return None
        return await self.stamp_trade_dict(trade)

    async def stamp_trade_dict(self, trade: dict) -> float | None:
        """Compute + persist for a trade dict the caller already has."""
        value = self._compute_for_trade(trade)
        if value is None:
            return None

        trade_id = trade.get("id") or trade.get("trade_id")
        if not trade_id:
            logger.warning(
                "ForwardMaxRStamper: trade dict missing id; computed "
                f"forward_max_r={value:.3f} but cannot persist"
            )
            return value

        try:
            ok = await self._repo.update_trade(trade_id, {"forward_max_r": value})
        except Exception as e:
            logger.warning(
                f"ForwardMaxRStamper: update_trade({trade_id}) failed: {e}"
            )
            return value
        if not ok:
            logger.warning(
                f"ForwardMaxRStamper: update_trade({trade_id}) returned False"
            )
        return value

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_for_trade(self, trade: dict) -> float | None:
        """Load the forward path and run :func:`compute_forward_max_r`."""
        symbol = trade.get("symbol")
        timeframe = trade.get("timeframe")
        direction = trade.get("direction")
        entry_price = self._safe_float(trade.get("entry_price"))
        entry_timestamp_ms = self._coerce_entry_timestamp(trade)

        if not symbol or not timeframe or direction is None:
            logger.warning(
                "ForwardMaxRStamper: trade missing symbol/timeframe/direction "
                f"(id={trade.get('id') or trade.get('trade_id')})"
            )
            return None
        if entry_price is None:
            logger.warning(
                f"ForwardMaxRStamper: trade {trade.get('id')} has no entry_price"
            )
            return None
        if entry_timestamp_ms is None:
            logger.warning(
                f"ForwardMaxRStamper: trade {trade.get('id')} has no usable entry timestamp"
            )
            return None

        risk = self._derive_risk(trade, entry_price)
        if risk is None or risk <= 0:
            logger.warning(
                f"ForwardMaxRStamper: trade {trade.get('id')} could not derive risk"
            )
            return None

        try:
            from backtesting.forward_path import ForwardPathLoader  # noqa: F401

            resolution = self._loader.recommended_resolution(timeframe)
            forward_path = self._loader.load(
                symbol=symbol,
                entry_timestamp=entry_timestamp_ms,
                duration_candles=self._forward_candles,
                resolution=resolution,
            )
        except FileNotFoundError as e:
            logger.warning(
                f"ForwardMaxRStamper: no forward path data for {symbol} {timeframe} "
                f"at {entry_timestamp_ms}: {e}"
            )
            return None
        except Exception as e:
            logger.warning(
                f"ForwardMaxRStamper: forward path load failed for "
                f"{symbol} {timeframe}: {e}"
            )
            return None

        return compute_forward_max_r(
            direction=direction,
            entry_price=entry_price,
            risk=risk,
            forward_path=forward_path,
        )

    def _derive_risk(self, trade: dict, entry_price: float) -> float | None:
        """Best-effort risk-per-unit derivation from a trade dict.

        Order of preference:
        1. Explicit ``risk`` field
        2. ``sl_price`` field → ``|entry - sl|``
        3. Default risk fraction (currently 1% of entry)
        """
        risk = self._safe_float(trade.get("risk"))
        if risk and risk > 0:
            return risk

        sl = self._safe_float(trade.get("sl_price"))
        if sl is not None and sl > 0:
            distance = abs(entry_price - sl)
            if distance > 0:
                return distance

        return entry_price * self._default_risk_fraction

    @staticmethod
    def _coerce_entry_timestamp(trade: dict) -> int | None:
        """Trades store entry_time as TIMESTAMPTZ / ISO string. Coerce to ms."""
        # Prefer an explicit ms-precision field if the caller provided one.
        for key in ("entry_timestamp_ms", "entry_timestamp"):
            v = trade.get(key)
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                pass

        # Fall back to ISO entry_time
        entry_time = trade.get("entry_time")
        if entry_time is None:
            return None
        try:
            from datetime import datetime
            if isinstance(entry_time, str):
                # Tolerate trailing Z and missing tzinfo
                cleaned = entry_time.replace("Z", "+00:00")
                dt = datetime.fromisoformat(cleaned)
            else:
                dt = entry_time
            return int(dt.timestamp() * 1000)
        except (TypeError, ValueError) as e:
            logger.debug(f"_coerce_entry_timestamp: {e}")
            return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
