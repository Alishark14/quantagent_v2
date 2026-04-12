"""EquityFlowProvider: FINRA RegSHO daily short-volume for US equities.

FINRA publishes one pipe-delimited file per trading day at
``https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt``
listing off-exchange short volume for every NMS-listed ticker. For
our HIP-3 equity universe (TSLA / NVDA / GOOGL) this is the closest
thing to "institutional positioning" data that's reliably free and
refreshes daily.

This provider wraps the daily fetch, maintains a 20-day rolling
history per ticker, and exposes three SVR signals plus a market-hours
flag that FlowSignalAgent consumes:

* ``short_volume_ratio`` — today's ``short_volume / total_volume``.
  Large-cap US equities typically sit between 0.40 and 0.60; above
  0.65 means institutions are aggressively shorting off-exchange.
* ``svr_zscore`` — how far today's SVR is from the 20-day rolling
  mean in standard deviations. > 2 is "unusual short activity"; < -1.5
  means shorts stepped away (contrarian bullish tell).
* ``svr_trend`` — "RISING" / "FALLING" / "STABLE", computed from the
  5-day vs 20-day SVR averages.
* ``market_open`` — True when US cash equities are in session. HIP-3
  oracle tracking is weaker outside session hours so the engine
  should be aware even when a fresh RegSHO row is unavailable.

Scope
=====

TSLA-USDC / NVDA-USDC / GOOGL-USDC only. Every crypto / FX / commodity
symbol short-circuits to ``{}`` and FlowAgent never sees RegSHO
fields for them.

Persistence
===========

A new ``regsho_cache`` table holds one row per ``(symbol, trade_date)``.
``warmup_from_repo()`` bulk-loads the last 20 trading days per ticker
so the Z-score math is live on cycle #1 after restart. On every live
fetch the provider writes the new day back to the table. A FINRA
outage (404, DNS, HTTP 5xx, parse error) falls back transparently to
whatever DB-cached history is already loaded — the Z-score +
``market_open`` rule still fires from the cached tail.

The HTTP client is injectable so tests can stub FINRA with
``httpx.MockTransport`` and never hit the network.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from engine.data.flow.base import FlowProvider
from exchanges.base import ExchangeAdapter
from storage.repositories.base import RegSHOCacheRepository

logger = logging.getLogger(__name__)

_FINRA_CDN_URL = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
)

# Canonical internal symbol → FINRA ticker. Only TSLA / NVDA / GOOGL
# today because those are the three HIP-3 equities we trade. Any
# other symbol short-circuits to an empty dict.
_SYMBOL_TO_TICKER: dict[str, str] = {
    "TSLA-USDC": "TSLA",
    "NVDA-USDC": "NVDA",
    "GOOGL-USDC": "GOOGL",
}

EQUITY_SYMBOLS: tuple[str, ...] = tuple(_SYMBOL_TO_TICKER.keys())

# Rolling window sizes. 20 trading days matches the "short-term vs
# longer-term" horizon the Z-score is meant to describe. 5 days is
# the short moving average for the trend classifier.
_HISTORY_DAYS = 20
_TREND_SHORT_WINDOW = 5
_TREND_LONG_WINDOW = 20

# Stable threshold for the FALLING classification — the 5-day SVR
# must be at least 5% below the 20-day average to count. Matches the
# spec in the task prompt.
_TREND_FALLING_RATIO = 0.95

# HTTP timeout for the FINRA CDN — files are small (~5 MB), usually
# respond in < 1s, give 15s headroom.
_HTTP_TIMEOUT = 15.0

# Search window for "most recent available" file. If today's file
# hasn't published yet (weekend, holiday, or pre-6PM-ET), walk
# backwards one day at a time up to this many attempts.
_MAX_BACKOFF_DAYS = 5


class EquityFlowProvider(FlowProvider):
    """FINRA RegSHO short-volume enrichment for TSLA / NVDA / GOOGL."""

    def __init__(
        self,
        regsho_repo: RegSHOCacheRepository | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._regsho_repo = regsho_repo
        # symbol → list[dict]
        # {trade_date: date, short_volume: int, total_volume: int,
        #  short_volume_ratio: float}
        # Ascending by trade_date so tail() == newest.
        self._history: dict[str, list[dict]] = {
            sym: [] for sym in _SYMBOL_TO_TICKER
        }
        self._client: httpx.AsyncClient | None = http_client
        self._owns_client: bool = http_client is None
        # One refresh at a time across all equity bots.
        self._refresh_lock = asyncio.Lock()
        # Cache the trade_date of the last successful live fetch so
        # we skip the HTTP round-trip for every subsequent Sentinel
        # scan on the same day.
        self._last_fetched_date: date | None = None

    def name(self) -> str:
        return "equity"

    # ------------------------------------------------------------------
    # FlowProvider contract
    # ------------------------------------------------------------------

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Return RegSHO enrichment for ``symbol`` or an empty dict.

        Non-equity symbols short-circuit. Equity symbols always
        receive at least ``market_open`` even if the FINRA fetch has
        not populated any history yet — the "outside market hours"
        rule can fire on its own.
        """
        if symbol not in _SYMBOL_TO_TICKER:
            return {}

        await self._maybe_refresh()

        out: dict[str, Any] = {
            "market_open": _is_us_market_open(),
        }

        rows = self._history.get(symbol, [])
        if len(rows) >= 2:
            out.update(self._compute_metrics(rows))
        return out

    async def close(self) -> None:
        """Release the internal HTTP client. Safe to call multiple times."""
        if self._owns_client and self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Persistence integration
    # ------------------------------------------------------------------

    async def warmup_from_repo(self) -> int:
        """Bulk-load each equity symbol's 20-day history from the DB.

        Returns the total number of rows loaded (for logging). No-op
        when no repo is wired. Errors are logged and swallowed.
        """
        if self._regsho_repo is None:
            return 0
        total = 0
        for symbol in _SYMBOL_TO_TICKER:
            try:
                rows = await self._regsho_repo.get_recent(
                    symbol, limit=_HISTORY_DAYS
                )
            except Exception:
                logger.exception(
                    f"EquityFlowProvider: warmup get_recent failed for {symbol}"
                )
                continue
            self._history[symbol] = [
                {
                    "trade_date": _coerce_date(r.get("trade_date")),
                    "short_volume": _as_int(r.get("short_volume")),
                    "total_volume": _as_int(r.get("total_volume")),
                    "short_volume_ratio": _as_float(
                        r.get("short_volume_ratio")
                    ),
                }
                for r in rows
                if _coerce_date(r.get("trade_date")) is not None
            ]
            total += len(self._history[symbol])
        if total > 0:
            logger.info(
                f"EquityFlowProvider: warmed {total} RegSHO snapshots "
                f"across {len(EQUITY_SYMBOLS)} symbols"
            )
        return total

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------

    async def _maybe_refresh(self) -> None:
        """Pull the newest RegSHO file if the in-memory buffer is stale.

        Guarded by an asyncio.Lock so concurrent Sentinel scans across
        the three equity bots never cause three simultaneous network
        hits. Fetch failure is logged and the provider keeps serving
        from the cached history.
        """
        async with self._refresh_lock:
            if not self._needs_refresh():
                return

            try:
                rows_by_symbol, trade_date = await self._fetch_latest_regsho()
            except Exception as e:
                logger.warning(
                    f"EquityFlowProvider: FINRA fetch failed ({e}); "
                    "keeping cached history"
                )
                return

            if not rows_by_symbol:
                return

            self._last_fetched_date = trade_date
            for symbol, row in rows_by_symbol.items():
                self._append_row(symbol, row)
                if self._regsho_repo is not None:
                    try:
                        await self._regsho_repo.upsert(
                            symbol,
                            row["trade_date"],
                            short_volume=row["short_volume"],
                            total_volume=row["total_volume"],
                            short_volume_ratio=row["short_volume_ratio"],
                        )
                    except Exception:
                        logger.debug(
                            f"EquityFlowProvider: regsho_cache upsert failed for {symbol}",
                            exc_info=True,
                        )

    def _needs_refresh(self) -> bool:
        """True when we haven't already pulled today's file (ET trading day)."""
        trading_day = _most_recent_trading_day()
        if (
            self._last_fetched_date is not None
            and self._last_fetched_date >= trading_day
        ):
            return False
        # Also skip if every symbol already has a row dated at or
        # after the most recent trading day — warmup may have loaded
        # it from DB before the process started.
        for symbol in _SYMBOL_TO_TICKER:
            rows = self._history.get(symbol)
            if not rows:
                return True
            newest = rows[-1].get("trade_date")
            if newest is None or newest < trading_day:
                return True
        self._last_fetched_date = trading_day
        return False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        return self._client

    async def _fetch_latest_regsho(
        self,
    ) -> tuple[dict[str, dict], date | None]:
        """Download the most recent available RegSHO file.

        Walks backwards from the current ET trading day up to
        ``_MAX_BACKOFF_DAYS`` attempts if 404/5xx, so a holiday or
        pre-publish fetch still lands on the most recent good file.
        Returns a dict keyed by internal symbol plus the trade_date
        the file was dated for.
        """
        client = await self._get_client()
        attempts = 0
        trading_day = _most_recent_trading_day()

        while attempts < _MAX_BACKOFF_DAYS:
            url = _FINRA_CDN_URL.format(
                yyyymmdd=trading_day.strftime("%Y%m%d")
            )
            try:
                resp = await client.get(url)
            except httpx.HTTPError as e:
                logger.debug(
                    f"EquityFlowProvider: FINRA GET {url} raised {e}"
                )
                trading_day = _previous_weekday(trading_day)
                attempts += 1
                continue

            if resp.status_code == 200 and resp.text:
                parsed = self._parse_regsho_text(resp.text)
                if parsed:
                    return parsed, trading_day

            trading_day = _previous_weekday(trading_day)
            attempts += 1

        return {}, None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_regsho_text(text: str) -> dict[str, dict]:
        """Parse the FINRA pipe-delimited CSV into per-symbol rows.

        File shape (header + one row per ticker):

            Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
            20260411|TSLA|31503665|139485|96692786|B,Q,N

        Malformed rows are silently skipped so a single bad line
        can't abort the whole parse.
        """
        wanted_tickers = set(_SYMBOL_TO_TICKER.values())
        reverse_map = {v: k for k, v in _SYMBOL_TO_TICKER.items()}
        out: dict[str, dict] = {}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("date|"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            ticker = parts[1].strip().upper()
            if ticker not in wanted_tickers:
                continue
            date_str = parts[0].strip()
            try:
                trade_date = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                continue
            short_vol = _as_int(parts[2])
            total_vol = _as_int(parts[4])
            if short_vol is None or total_vol is None or total_vol <= 0:
                continue
            svr = short_vol / total_vol
            internal = reverse_map[ticker]
            out[internal] = {
                "trade_date": trade_date,
                "short_volume": short_vol,
                "total_volume": total_vol,
                "short_volume_ratio": svr,
            }
        return out

    # ------------------------------------------------------------------
    # Buffer mutation
    # ------------------------------------------------------------------

    def _append_row(self, symbol: str, row: dict) -> None:
        """Append or replace the newest-date row; keep history capped."""
        history = self._history.setdefault(symbol, [])
        td = row["trade_date"]
        if history and history[-1]["trade_date"] == td:
            history[-1] = row
            return
        history.append(row)
        if len(history) > _HISTORY_DAYS:
            del history[: len(history) - _HISTORY_DAYS]

    # ------------------------------------------------------------------
    # Derived metrics — pure functions
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(rows: list[dict]) -> dict:
        """Build the FlowOutput enrichment dict from a history list.

        Caller guarantees at least 2 ascending rows. Missing fields
        silently fall through to None.
        """
        latest = rows[-1]
        svrs = [
            r["short_volume_ratio"]
            for r in rows
            if r.get("short_volume_ratio") is not None
        ]
        if not svrs:
            return {}

        out: dict[str, Any] = {
            "short_volume_ratio": latest.get("short_volume_ratio"),
        }

        # Z-score vs the 20-day rolling mean / stddev. We exclude the
        # most recent entry from the rolling stats so the Z-score
        # isn't self-referential (the "how unusual is today vs the
        # prior window" reading the rule set expects).
        if len(svrs) >= 3:
            history = svrs[:-1]
            mean = sum(history) / len(history)
            var = sum((x - mean) ** 2 for x in history) / len(history)
            std = var ** 0.5
            if std > 0 and svrs[-1] is not None:
                out["svr_zscore"] = (svrs[-1] - mean) / std

        # Trend — compare 5-day avg vs 20-day avg. Needs at least 5
        # points for a meaningful short-window average.
        if len(svrs) >= 5:
            short_avg = sum(svrs[-_TREND_SHORT_WINDOW:]) / min(
                _TREND_SHORT_WINDOW, len(svrs)
            )
            long_avg = sum(svrs[-_TREND_LONG_WINDOW:]) / min(
                _TREND_LONG_WINDOW, len(svrs)
            )
            if short_avg > long_avg:
                out["svr_trend"] = "RISING"
            elif short_avg < long_avg * _TREND_FALLING_RATIO:
                out["svr_trend"] = "FALLING"
            else:
                out["svr_trend"] = "STABLE"

        return out


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _is_us_market_open(now: datetime | None = None) -> bool:
    """Return True when US cash equities are in session (9:30-16:00 ET).

    Uses stdlib ``zoneinfo`` — no pytz dependency. Weekends return
    False. The ``now`` parameter is for deterministic testing; in
    production the caller passes None and we use the wall clock.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    current = now.astimezone(et) if now is not None else datetime.now(et)
    if current.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    market_open = current.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= current <= market_close


def _most_recent_trading_day(now: datetime | None = None) -> date:
    """Return the most recent weekday (today if weekday, else last Friday).

    This is a best-effort approximation — FINRA also skips exchange
    holidays, but we don't ship a holiday calendar here. The refresh
    loop's backoff walk handles holiday 404s by walking one day
    further back on each miss.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    current = now.astimezone(et) if now is not None else datetime.now(et)
    d = current.date()
    while d.weekday() >= 5:  # weekend → step back to Friday
        d -= timedelta(days=1)
    return d


def _previous_weekday(d: date) -> date:
    """Return the weekday immediately before ``d``."""
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _as_int(value) -> int | None:
    """Coerce to int, accepting decimal strings.

    FINRA's RegSHO file format publishes short_volume and
    total_volume as DECIMAL strings (e.g. ``31331839.752390``) —
    presumably because fractional share trading is counted through
    the same off-exchange feed. A strict ``int("31331839.752390")``
    would ValueError, so we parse as float first and truncate. The
    sub-unit precision is meaningless for signal math anyway.
    """
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _coerce_date(value) -> date | None:
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
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
                try:
                    return datetime.strptime(value.strip(), fmt).date()
                except ValueError:
                    continue
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            return None
    return None
