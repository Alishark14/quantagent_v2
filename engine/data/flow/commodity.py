"""CommodityFlowProvider: CFTC Commitment-of-Traders positioning for commodities.

The CFTC publishes a weekly snapshot showing how commercial hedgers
(Producer / Merchant) and managed money (hedge funds) are positioned in
US futures markets. For our HIP-3 commodity universe (gold, silver,
WTI, brent) this is the closest thing to "smart money vs. crowd"
positioning that's reliably available and free.

This provider wraps the ``cot_reports`` library (pandas DataFrame
in / dict out), maintains a 52-week rolling history per symbol, and
exposes four signals that FlowSignalAgent consumes:

* ``cot_speculator_percentile`` — where the current managed-money net
  position sits in the last 52-week distribution. > 90 means
  speculators are extremely crowded long → contrarian BEARISH lean;
  < 10 means extreme crowded short → contrarian BULLISH lean.
* ``cot_commercial_net`` — raw producer/merchant net position for
  operator introspection.
* ``cot_managed_money_net`` — raw managed-money net position.
* ``cot_weekly_change_pct`` — week-over-week change in MM net
  (how fast the speculative book is moving).
* ``cot_divergence`` — ``commercial_net - managed_money_net``. When
  commercials and speculators disagree sharply, commercials are
  usually right (they're hedging real physical exposure).
* ``cot_divergence_abs_percentile`` — where ``abs(cot_divergence)``
  sits in its own 52-week distribution. Let the FlowSignalAgent rule
  fire only when the divergence is in the top 20%.

Scope
=====

BTC / ETH / altcoin / FX / equity symbols return an empty dict
immediately so FlowAgent's merge step never sees COT fields for them.
The CFTC only publishes COT for listed futures, so there is no
equivalent data for crypto, and loading a per-crypto COT row would
silently populate fields that have no meaning for those markets.

Caching and persistence
=======================

Weekly refresh timing: COT data drops Friday evening US Eastern with
the report date set to the prior Tuesday. The provider's in-memory
rolling buffer is keyed by symbol and holds the most recent 52
``(report_date, mm_net, commercial_net, total_oi)`` tuples. On
startup ``warmup_from_repo()`` bulk-loads every commodity symbol's
history from the ``cot_cache`` table so the percentile math works on
the very first ``fetch()`` call after restart. When the in-memory
data is older than one week, the next ``fetch()`` call triggers a
single live pull via ``cot_reports.cot_year``, extracts the most
recent row per supported symbol, writes them back to the cache table,
and appends them to the in-memory history.

Error handling
==============

``cot_reports`` is a thin wrapper around a CFTC ZIP endpoint that
occasionally 503s. A live-pull failure is logged and the provider
falls back to whatever data the in-memory buffer already has — the
signal rules will still fire as long as at least 2 weeks of history
are loaded from the DB. A totally empty buffer + failed fetch returns
an empty dict so FlowSignalAgent's ``None`` guards short-circuit the
COT rules gracefully.

``cot_reports`` is imported lazily inside the refresh path so a
missing dependency (the library isn't installed in every environment)
does not break module import — the provider just stays in
DB-cache-only mode and logs a warning once per refresh attempt.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from engine.data.flow.base import FlowProvider
from exchanges.base import ExchangeAdapter
from storage.repositories.base import COTCacheRepository

logger = logging.getLogger(__name__)

# Canonical internal symbol → ordered list of (substring, priority)
# match patterns against ``Market_and_Exchange_Names``. The first
# matching row wins. Substring matching keeps us robust to the CFTC
# occasionally renaming legacy contracts (e.g. "CRUDE OIL, LIGHT SWEET"
# vs. "WTI-PHYSICAL") without needing a code change every quarter.
_SYMBOL_TO_CFTC_PATTERNS: dict[str, list[str]] = {
    "GOLD-USDC": ["GOLD - COMMODITY EXCHANGE"],
    "SILVER-USDC": ["SILVER - COMMODITY EXCHANGE"],
    "WTIOIL-USDC": [
        "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
        "WTI-PHYSICAL",
        "WTI FINANCIAL CRUDE OIL",
        "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE",
    ],
    "BRENTOIL-USDC": [
        "BRENT LAST DAY - NEW YORK MERCANTILE EXCHANGE",
        "BRENT CRUDE OIL LAST DAY",
    ],
}

# Symbols we treat as "commodity" for rule gating + warmup loops. Kept
# as an ordered tuple so tests can iterate deterministically.
COMMODITY_SYMBOLS: tuple[str, ...] = tuple(_SYMBOL_TO_CFTC_PATTERNS.keys())

# CFTC Disaggregated Futures column names. Pinned here because a
# schema change at the CFTC would silently break all the math — if
# the provider can't find these columns it logs a warning and skips
# the row instead of computing off misaligned data.
_COL_MARKET = "Market_and_Exchange_Names"
_COL_REPORT_DATE = "Report_Date_as_YYYY-MM-DD"
_COL_MM_LONG = "M_Money_Positions_Long_All"
_COL_MM_SHORT = "M_Money_Positions_Short_All"
_COL_PM_LONG = "Prod_Merc_Positions_Long_All"
_COL_PM_SHORT = "Prod_Merc_Positions_Short_All"
_COL_OI = "Open_Interest_All"

# Rolling window for percentile / divergence computations.
_HISTORY_WEEKS = 52

# Refresh cadence. COT drops Friday evening US Eastern; refreshing
# once per calendar week is sufficient. We track the report_date of
# the newest in-memory row and refresh when the current UTC time is
# more than 7 days past that date.
_REFRESH_INTERVAL_DAYS = 7


class CommodityFlowProvider(FlowProvider):
    """Rolling 52-week CFTC COT cache for gold / silver / WTI / brent."""

    def __init__(self, cot_repo: COTCacheRepository | None = None) -> None:
        self._cot_repo = cot_repo
        # symbol → list of dicts
        # {report_date: date, managed_money_net, commercial_net, total_oi}
        # ascending by report_date so the tail() == newest.
        self._history: dict[str, list[dict]] = {
            sym: [] for sym in _SYMBOL_TO_CFTC_PATTERNS
        }
        # Guards concurrent refresh: if 4 commodity bots all fire a
        # Sentinel scan at the same second we only want ONE CFTC pull.
        self._refresh_lock = asyncio.Lock()
        # Track which year the last successful live pull covered so
        # we can decide whether to roll over into a new-year fetch at
        # January. The CFTC year-file is addressed by calendar year.
        self._last_live_pull_year: int | None = None

    def name(self) -> str:
        return "commodity"

    # ------------------------------------------------------------------
    # FlowProvider contract
    # ------------------------------------------------------------------

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Return COT enrichment for ``symbol`` or an empty dict.

        Crypto / FX / equity symbols short-circuit to ``{}`` so
        FlowAgent's merge step never overwrites anything for them.
        """
        if symbol not in _SYMBOL_TO_CFTC_PATTERNS:
            return {}

        # Refresh first so the in-memory buffer is as fresh as
        # possible before we compute the derived metrics. Refresh is
        # a no-op when the newest row is already within the 7-day
        # interval.
        await self._maybe_refresh()

        rows = self._history.get(symbol, [])
        if len(rows) < 2:
            # Need at least 2 weeks of history to compute ANY derived
            # metric (weekly change + percentile both need more than
            # one point). Return empty dict so FlowSignalAgent's None
            # guards short-circuit.
            return {}

        return self._compute_metrics(rows)

    # ------------------------------------------------------------------
    # Persistence integration
    # ------------------------------------------------------------------

    async def warmup_from_repo(self) -> int:
        """Bulk-load each commodity symbol's 52-week history from the DB.

        Returns the total number of rows loaded across every symbol,
        for logging. No-op (returns 0) when no repo is wired. Errors
        are logged and swallowed — a warmup failure must NOT block
        startup, the provider just starts with empty history and
        paying the next refresh will rebuild from the live feed.
        """
        if self._cot_repo is None:
            return 0
        total = 0
        for symbol in _SYMBOL_TO_CFTC_PATTERNS:
            try:
                rows = await self._cot_repo.get_recent(symbol, limit=_HISTORY_WEEKS)
            except Exception:
                logger.exception(
                    f"CommodityFlowProvider: warmup get_recent failed for {symbol}"
                )
                continue
            self._history[symbol] = [
                {
                    "report_date": _coerce_date(r["report_date"]),
                    "managed_money_net": _as_float(r.get("managed_money_net")),
                    "commercial_net": _as_float(r.get("commercial_net")),
                    "total_oi": _as_float(r.get("total_oi")),
                }
                for r in rows
                if _coerce_date(r.get("report_date")) is not None
            ]
            total += len(self._history[symbol])
        if total > 0:
            logger.info(
                f"CommodityFlowProvider: warmed {total} COT snapshots "
                f"across {len(COMMODITY_SYMBOLS)} symbols"
            )
        return total

    # ------------------------------------------------------------------
    # Refresh loop — once per ≥7 days, guarded by an asyncio.Lock
    # ------------------------------------------------------------------

    async def _maybe_refresh(self) -> None:
        """Refresh the in-memory buffer from CFTC if any symbol is stale.

        Runs at most one live pull across all commodity symbols, even
        under concurrent Sentinel scans, thanks to the refresh lock.
        On failure the error is logged and the provider keeps serving
        whatever data it already had.
        """
        async with self._refresh_lock:
            if not self._any_symbol_stale():
                return

            try:
                new_rows = await asyncio.to_thread(self._fetch_latest_rows)
            except ImportError:
                logger.warning(
                    "CommodityFlowProvider: cot_reports library not "
                    "installed — running in DB-cache-only mode"
                )
                return
            except Exception as e:
                logger.warning(
                    f"CommodityFlowProvider: cot_reports fetch failed "
                    f"({e}); keeping cached history"
                )
                return

            if not new_rows:
                return

            for symbol, row in new_rows.items():
                self._append_row(symbol, row)
                if self._cot_repo is not None:
                    try:
                        await self._cot_repo.upsert(
                            symbol,
                            row["report_date"],
                            managed_money_net=row["managed_money_net"],
                            commercial_net=row["commercial_net"],
                            total_oi=row["total_oi"],
                            raw_json=None,
                        )
                    except Exception:
                        logger.debug(
                            f"CommodityFlowProvider: cot_cache upsert failed for {symbol}",
                            exc_info=True,
                        )

    def _any_symbol_stale(self) -> bool:
        """True when at least one tracked symbol needs a live refresh."""
        cutoff = datetime.now(timezone.utc).date() - timedelta(
            days=_REFRESH_INTERVAL_DAYS
        )
        for symbol in _SYMBOL_TO_CFTC_PATTERNS:
            rows = self._history.get(symbol)
            if not rows:
                return True
            newest = rows[-1]["report_date"]
            if newest is None or newest < cutoff:
                return True
        return False

    def _fetch_latest_rows(self) -> dict[str, dict]:
        """Pull the current year's Disaggregated report and return one row per symbol.

        Runs in a worker thread via ``asyncio.to_thread`` so the
        synchronous cot_reports / pandas call doesn't block the event
        loop. Imports cot_reports lazily so a missing dependency
        surfaces as an ``ImportError`` the caller handles.
        """
        import cot_reports  # type: ignore[import-not-found]  # lazy

        year = datetime.now(timezone.utc).year
        df = cot_reports.cot_year(
            year, cot_report_type="disaggregated_fut", verbose=False
        )
        self._last_live_pull_year = year

        out: dict[str, dict] = {}
        for symbol, patterns in _SYMBOL_TO_CFTC_PATTERNS.items():
            row = self._extract_latest_row(df, patterns)
            if row is not None:
                out[symbol] = row
        return out

    @staticmethod
    def _extract_latest_row(df, patterns: list[str]) -> dict | None:
        """Find the most recent CFTC row for ``patterns`` in ``df``.

        Pure helper so tests can pass a hand-built DataFrame and
        exercise the parsing logic without the cot_reports lib.
        """
        if df is None or len(df) == 0:
            return None

        market_col = _COL_MARKET
        if market_col not in df.columns:
            return None

        try:
            for pattern in patterns:
                subset = df[
                    df[market_col].astype(str).str.contains(
                        pattern, case=False, na=False
                    )
                ]
                if len(subset) == 0:
                    continue

                # CFTC returns one row per (contract, weekly report).
                # Two steps: (1) narrow to the MOST RECENT weekly
                # report date so we return TODAY's positioning, not
                # the week in the year with the biggest OI, and
                # (2) among ties on that date (regular + micro etc.)
                # pick the contract with the largest Open Interest.
                if _COL_REPORT_DATE in subset.columns:
                    try:
                        subset = subset.sort_values(
                            _COL_REPORT_DATE, ascending=False
                        )
                        newest_date = subset.iloc[0][_COL_REPORT_DATE]
                        subset = subset[
                            subset[_COL_REPORT_DATE] == newest_date
                        ]
                    except Exception:
                        pass
                if _COL_OI in subset.columns and len(subset) > 1:
                    try:
                        subset = subset.sort_values(
                            _COL_OI, ascending=False
                        )
                    except Exception:
                        pass

                latest = subset.iloc[0]
                report_date = _coerce_date(latest.get(_COL_REPORT_DATE))
                mm_long = _as_float(latest.get(_COL_MM_LONG))
                mm_short = _as_float(latest.get(_COL_MM_SHORT))
                pm_long = _as_float(latest.get(_COL_PM_LONG))
                pm_short = _as_float(latest.get(_COL_PM_SHORT))
                oi = _as_float(latest.get(_COL_OI))

                if (
                    report_date is None
                    or mm_long is None
                    or mm_short is None
                    or pm_long is None
                    or pm_short is None
                ):
                    continue

                return {
                    "report_date": report_date,
                    "managed_money_net": mm_long - mm_short,
                    "commercial_net": pm_long - pm_short,
                    "total_oi": oi,
                }
        except Exception:
            logger.debug(
                "CommodityFlowProvider: _extract_latest_row raised; "
                "returning None",
                exc_info=True,
            )
            return None
        return None

    def _append_row(self, symbol: str, row: dict) -> None:
        """Append a new weekly row, deduping by report_date, enforcing cap."""
        history = self._history.setdefault(symbol, [])
        rd = row["report_date"]
        # Dedupe: if the most recent row already has this report_date
        # just replace its values. Avoids counting the same weekly
        # report twice after a manual refresh.
        if history and history[-1]["report_date"] == rd:
            history[-1] = row
            return
        history.append(row)
        if len(history) > _HISTORY_WEEKS:
            del history[: len(history) - _HISTORY_WEEKS]

    # ------------------------------------------------------------------
    # Derived metrics — pure functions
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(rows: list[dict]) -> dict:
        """Compute the FlowOutput enrichment dict from a history list.

        ``rows`` must be ascending by report_date and contain at least
        2 entries (the caller guarantees this). Missing numeric fields
        silently fall through to None.
        """
        latest = rows[-1]
        previous = rows[-2]
        mm_net = _as_float(latest.get("managed_money_net"))
        comm_net = _as_float(latest.get("commercial_net"))
        prev_mm = _as_float(previous.get("managed_money_net"))

        out: dict[str, Any] = {}
        if mm_net is not None:
            out["cot_managed_money_net"] = mm_net
        if comm_net is not None:
            out["cot_commercial_net"] = comm_net
        if mm_net is not None and comm_net is not None:
            out["cot_divergence"] = comm_net - mm_net

        if prev_mm is not None and prev_mm != 0 and mm_net is not None:
            out["cot_weekly_change_pct"] = (
                (mm_net - prev_mm) / abs(prev_mm)
            ) * 100.0

        # Speculator percentile — where does the current mm_net sit
        # in the full 52-week distribution? ``percentileofscore`` rank
        # implementation below handles ties by counting "<=" so a
        # fresh max correctly lands at 100.
        if mm_net is not None:
            mm_history = [
                _as_float(r.get("managed_money_net")) for r in rows
            ]
            mm_history = [v for v in mm_history if v is not None]
            if len(mm_history) >= 2:
                out["cot_speculator_percentile"] = _percentile_of_score(
                    mm_history, mm_net
                )

        # Divergence abs percentile — where does abs(current) sit in
        # the distribution of abs(historical)? Used by the divergence
        # rules to fire only when the spread is in the top 20%.
        if "cot_divergence" in out:
            divergences = [
                abs(_as_float(r.get("commercial_net")) - _as_float(r.get("managed_money_net")))
                for r in rows
                if _as_float(r.get("commercial_net")) is not None
                and _as_float(r.get("managed_money_net")) is not None
            ]
            if len(divergences) >= 2:
                out["cot_divergence_abs_percentile"] = _percentile_of_score(
                    divergences, abs(out["cot_divergence"])
                )

        return out


# ----------------------------------------------------------------------
# Module-private helpers
# ----------------------------------------------------------------------


def _as_float(value) -> float | None:
    """Coerce to float, returning None on anything that can't cast."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # Pandas often produces NaN for missing numeric cells — filter.
    if f != f:  # NaN check
        return None
    return f


def _coerce_date(value) -> date | None:
    """Coerce mixed inputs (str, datetime, pandas Timestamp) to a ``date``."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip()).date()
        except ValueError:
            # Some CFTC rows format dates as "YYYY-MM-DD" already,
            # others as "MM/DD/YYYY" historically. Try the common
            # alternate form as a defensive fallback.
            for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                try:
                    return datetime.strptime(value.strip(), fmt).date()
                except ValueError:
                    continue
    # pandas Timestamp exposes .to_pydatetime / .date; try that path
    # without importing pandas here (keeps the helper pandas-free so
    # tests can exercise it with plain datetimes).
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            return None
    return None


def _percentile_of_score(values: list[float], target: float) -> float:
    """Return the percentile rank of ``target`` in ``values`` (0-100).

    Uses the "weak" / "≤" convention: ties count as below so a fresh
    all-time high lands at exactly 100 and a fresh all-time low lands
    at ``(1/n)*100``. Matches ``scipy.stats.percentileofscore(kind="weak")``
    and is good enough for the "top 20% / bottom 10%" bucketing the
    rules need.
    """
    if not values:
        return 0.0
    below_or_equal = sum(1 for v in values if v <= target)
    return (below_or_equal / len(values)) * 100.0
