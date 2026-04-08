"""Macro data fetcher for the Macro Regime Manager.

Pulls VIX, DXY, DVOL, Fear & Greed, BTC Dominance, Hyperliquid total
OI/funding, and the upcoming economic calendar from a handful of
public APIs. Each fetch is wrapped in a try/except — partial data is
better than no data, so a single API outage NEVER crashes the run.

Per ARCHITECTURE §13.2.2 the data sources are:

  CBOE VIX           Yahoo Finance       continuous (US hours)
  DXY                Yahoo Finance       continuous (forex hours)
  Deribit DVOL       Deribit public API  24/7
  Fear & Greed       Alternative.me      daily
  BTC Dominance      CoinGecko           daily
  HL OI + funding    Hyperliquid API     24/7
  Economic calendar  hardcoded + API     weekly

The fetcher is intentionally synchronous in its public surface
(``fetch()`` returns a dataclass, not an awaitable) — the runner
calls it once per cron tick. ``httpx`` is used for the actual
HTTP calls because it's already a project dependency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_RATE_LIMIT_SECONDS = 1.0  # 1s between API calls

# Public endpoints — no auth required.
_YAHOO_QUOTE_URL = (
    "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
)
_DERIBIT_DVOL_URL = (
    "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_dvol"
)
_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
_HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

_VIX_SYMBOL = "^VIX"
_DXY_SYMBOL = "DX-Y.NYB"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EconomicEvent:
    """One scheduled high/medium-impact macro event."""

    name: str  # e.g. "FOMC_ANNOUNCEMENT", "CPI", "NFP"
    timestamp: str  # ISO 8601 UTC
    impact: str = "HIGH"  # HIGH | MEDIUM
    source: str = "hardcoded"

    def parsed_time(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None


@dataclass
class MacroSnapshot:
    """One unified macro snapshot.

    Every numeric field is optional — a partial snapshot is fine. The
    ``available_sources`` set tells callers which fetches succeeded so
    LightweightCheck can apply its weekend-fallback logic correctly.
    """

    fetched_at: str  # ISO 8601 UTC of when fetch() was called
    vix: float | None = None
    vix_timestamp: str | None = None
    dxy: float | None = None
    dxy_timestamp: str | None = None
    dvol: float | None = None
    dvol_timestamp: str | None = None
    fear_greed_value: int | None = None  # 0-100
    fear_greed_classification: str | None = None  # e.g. "extreme fear"
    btc_dominance: float | None = None  # 0-100 percent
    hl_total_oi: float | None = None  # USD
    hl_avg_funding: float | None = None  # average funding across markets
    economic_calendar: list[EconomicEvent] = field(default_factory=list)
    available_sources: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def fetched_at_dt(self) -> datetime:
        try:
            return datetime.fromisoformat(self.fetched_at.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return datetime.now(tz=timezone.utc)

    def has_data(self, source: str) -> bool:
        return source in self.available_sources

    def to_dict(self) -> dict:
        return {
            "fetched_at": self.fetched_at,
            "vix": self.vix,
            "vix_timestamp": self.vix_timestamp,
            "dxy": self.dxy,
            "dxy_timestamp": self.dxy_timestamp,
            "dvol": self.dvol,
            "dvol_timestamp": self.dvol_timestamp,
            "fear_greed_value": self.fear_greed_value,
            "fear_greed_classification": self.fear_greed_classification,
            "btc_dominance": self.btc_dominance,
            "hl_total_oi": self.hl_total_oi,
            "hl_avg_funding": self.hl_avg_funding,
            "economic_calendar": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp,
                    "impact": e.impact,
                    "source": e.source,
                }
                for e in self.economic_calendar
            ],
            "available_sources": sorted(self.available_sources),
            "errors": dict(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MacroSnapshot":
        events = [
            EconomicEvent(
                name=str(e.get("name", "")),
                timestamp=str(e.get("timestamp", "")),
                impact=str(e.get("impact", "MEDIUM")),
                source=str(e.get("source", "hardcoded")),
            )
            for e in (payload.get("economic_calendar") or [])
        ]
        return cls(
            fetched_at=str(payload.get("fetched_at") or _now_iso()),
            vix=_opt_float(payload.get("vix")),
            vix_timestamp=payload.get("vix_timestamp"),
            dxy=_opt_float(payload.get("dxy")),
            dxy_timestamp=payload.get("dxy_timestamp"),
            dvol=_opt_float(payload.get("dvol")),
            dvol_timestamp=payload.get("dvol_timestamp"),
            fear_greed_value=_opt_int(payload.get("fear_greed_value")),
            fear_greed_classification=payload.get("fear_greed_classification"),
            btc_dominance=_opt_float(payload.get("btc_dominance")),
            hl_total_oi=_opt_float(payload.get("hl_total_oi")),
            hl_avg_funding=_opt_float(payload.get("hl_avg_funding")),
            economic_calendar=events,
            available_sources=set(payload.get("available_sources") or []),
            errors=dict(payload.get("errors") or {}),
        )


# ---------------------------------------------------------------------------
# Hardcoded economic calendar
# ---------------------------------------------------------------------------


def _hardcoded_calendar(reference: datetime, lookahead_days: int = 30) -> list[EconomicEvent]:
    """Return well-known scheduled events within ``lookahead_days``.

    These are the *known* high-impact macro events. The list is small
    by design: we'd rather miss a low-impact event than fabricate
    timestamps for a release we can't reliably look up. The runner
    will be re-run more often than the calendar changes, and a
    successful API fetch overlays freshly-published dates.
    """
    # Hardcoded 2026 schedule for the events we always care about.
    # Times are the official US-Eastern release time converted to UTC
    # (UTC = ET + 4h during DST, ET + 5h otherwise — these dates use
    # the standard release times during DST).
    catalog: list[EconomicEvent] = [
        # FOMC announcements (2pm ET = 18:00 UTC during DST)
        EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-29T18:00:00Z", "HIGH"),
        EconomicEvent("FOMC_ANNOUNCEMENT", "2026-06-17T18:00:00Z", "HIGH"),
        EconomicEvent("FOMC_ANNOUNCEMENT", "2026-07-29T18:00:00Z", "HIGH"),
        EconomicEvent("FOMC_ANNOUNCEMENT", "2026-09-16T18:00:00Z", "HIGH"),
        # CPI (8:30am ET = 12:30 UTC during DST)
        EconomicEvent("CPI", "2026-04-14T12:30:00Z", "HIGH"),
        EconomicEvent("CPI", "2026-05-13T12:30:00Z", "HIGH"),
        EconomicEvent("CPI", "2026-06-10T12:30:00Z", "HIGH"),
        # NFP (8:30am ET = 12:30 UTC during DST, first Friday)
        EconomicEvent("NFP", "2026-05-01T12:30:00Z", "HIGH"),
        EconomicEvent("NFP", "2026-06-05T12:30:00Z", "HIGH"),
        EconomicEvent("NFP", "2026-07-03T12:30:00Z", "HIGH"),
    ]
    cutoff = reference + timedelta(days=lookahead_days)
    out: list[EconomicEvent] = []
    for ev in catalog:
        ts = ev.parsed_time()
        if ts is None:
            continue
        # Allow events up to 30 minutes in the past so a check that
        # runs *just after* an event still surfaces it (the post-event
        # buffer is 30 minutes per §13.2.4).
        if ts >= reference - timedelta(minutes=30) and ts <= cutoff:
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# MacroDataFetcher
# ---------------------------------------------------------------------------


class MacroDataFetcher:
    """Fetches a unified macro snapshot from public APIs.

    Construction is cheap (no network calls). The actual work happens
    inside :meth:`fetch`. Tests inject a fake ``http_client`` (anything
    with a ``get(url, timeout=)`` method that returns a response with
    ``json()`` and ``raise_for_status()``) to avoid hitting the real
    network.
    """

    def __init__(
        self,
        http_client: Any = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        rate_limit_seconds: float = _DEFAULT_RATE_LIMIT_SECONDS,
        sleep_func: Any = None,
        clock: Any = None,
        calendar_lookahead_days: int = 30,
    ) -> None:
        self._client = http_client
        self._owns_client = False
        self._timeout = float(timeout)
        self._rate_limit = float(rate_limit_seconds)
        self._sleep = sleep_func or time.sleep
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._lookahead_days = int(calendar_lookahead_days)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fetch(self) -> MacroSnapshot:
        """Pull every supported source. Partial failures are tolerated."""
        snapshot = MacroSnapshot(fetched_at=_iso(self._clock()))

        try:
            client = self._resolve_client()
        except Exception as e:
            logger.warning(f"MacroDataFetcher: no HTTP client available: {e}")
            snapshot.errors["http_client"] = str(e)
            # Even with no client we can still emit the hardcoded calendar.
            snapshot.economic_calendar = _hardcoded_calendar(
                self._clock(), lookahead_days=self._lookahead_days
            )
            if snapshot.economic_calendar:
                snapshot.available_sources.add("calendar")
            return snapshot

        # Each fetcher: try, log on error, sleep before next call.
        fetchers: list[tuple[str, Any]] = [
            ("vix", lambda: self._fetch_yahoo(client, _VIX_SYMBOL)),
            ("dxy", lambda: self._fetch_yahoo(client, _DXY_SYMBOL)),
            ("dvol", lambda: self._fetch_dvol(client)),
            ("fear_greed", lambda: self._fetch_fear_greed(client)),
            ("btc_dominance", lambda: self._fetch_btc_dominance(client)),
            ("hyperliquid", lambda: self._fetch_hyperliquid(client)),
        ]

        for i, (name, fn) in enumerate(fetchers):
            if i > 0 and self._rate_limit > 0:
                try:
                    self._sleep(self._rate_limit)
                except Exception:  # noqa: BLE001 — never let a sleep error kill the run
                    pass
            try:
                payload = fn()
            except Exception as e:  # noqa: BLE001 — public-API blast radius is unknown
                logger.warning(f"MacroDataFetcher: {name} fetch failed: {e}")
                snapshot.errors[name] = str(e)
                continue
            if payload is None:
                continue
            self._apply(snapshot, name, payload)

        # Calendar is always available (hardcoded baseline).
        snapshot.economic_calendar = _hardcoded_calendar(
            self._clock(), lookahead_days=self._lookahead_days
        )
        if snapshot.economic_calendar:
            snapshot.available_sources.add("calendar")

        if self._owns_client and hasattr(client, "close"):
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

        return snapshot

    # ------------------------------------------------------------------
    # Per-source fetchers
    # ------------------------------------------------------------------

    def _fetch_yahoo(self, client: Any, symbol: str) -> dict | None:
        url = _YAHOO_QUOTE_URL.format(symbol=symbol)
        data = self._get_json(client, url)
        result = (
            (data or {})
            .get("quoteResponse", {})
            .get("result", [])
        )
        if not result:
            return None
        row = result[0]
        price = row.get("regularMarketPrice")
        if price is None:
            return None
        ts_epoch = row.get("regularMarketTime")
        ts_iso = _epoch_to_iso(ts_epoch)
        return {"price": float(price), "timestamp": ts_iso}

    def _fetch_dvol(self, client: Any) -> dict | None:
        data = self._get_json(client, _DERIBIT_DVOL_URL)
        result = (data or {}).get("result", {})
        price = result.get("index_price")
        if price is None:
            return None
        return {"price": float(price), "timestamp": _now_iso(self._clock)}

    def _fetch_fear_greed(self, client: Any) -> dict | None:
        data = self._get_json(client, _FNG_URL)
        rows = (data or {}).get("data", [])
        if not rows:
            return None
        row = rows[0]
        value = row.get("value")
        if value is None:
            return None
        return {
            "value": int(value),
            "classification": row.get("value_classification"),
        }

    def _fetch_btc_dominance(self, client: Any) -> dict | None:
        data = self._get_json(client, _COINGECKO_GLOBAL_URL)
        market_cap_pct = (
            (data or {}).get("data", {}).get("market_cap_percentage", {})
        )
        btc = market_cap_pct.get("btc")
        if btc is None:
            return None
        return {"value": float(btc)}

    def _fetch_hyperliquid(self, client: Any) -> dict | None:
        # Hyperliquid /info uses POST with a JSON body. Tests inject a
        # fake client with `.post(url, json=...)` returning a response
        # whose .json() is a list of meta dicts. If the client doesn't
        # have post(), fall back to .get() (some test doubles).
        body = {"type": "metaAndAssetCtxs"}
        if hasattr(client, "post"):
            response = client.post(
                _HYPERLIQUID_INFO_URL, json=body, timeout=self._timeout
            )
        else:
            response = client.get(_HYPERLIQUID_INFO_URL, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()

        # The HL response shape is [meta, [ctx0, ctx1, ...]]; each ctx
        # has openInterest and funding. We sum OI and average funding.
        if not isinstance(data, list) or len(data) < 2:
            return None
        contexts = data[1]
        if not isinstance(contexts, list):
            return None
        total_oi = 0.0
        funding_sum = 0.0
        funding_count = 0
        for ctx in contexts:
            if not isinstance(ctx, dict):
                continue
            try:
                oi = float(ctx.get("openInterest") or 0.0)
                mark = float(ctx.get("markPx") or 0.0)
                total_oi += oi * mark
            except (TypeError, ValueError):
                pass
            funding = ctx.get("funding")
            if funding is not None:
                try:
                    funding_sum += float(funding)
                    funding_count += 1
                except (TypeError, ValueError):
                    pass
        avg_funding = funding_sum / funding_count if funding_count else None
        return {"total_oi": total_oi, "avg_funding": avg_funding}

    # ------------------------------------------------------------------
    # Snapshot assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _apply(snapshot: MacroSnapshot, source: str, payload: dict) -> None:
        if source == "vix":
            snapshot.vix = payload["price"]
            snapshot.vix_timestamp = payload.get("timestamp")
            snapshot.available_sources.add("vix")
        elif source == "dxy":
            snapshot.dxy = payload["price"]
            snapshot.dxy_timestamp = payload.get("timestamp")
            snapshot.available_sources.add("dxy")
        elif source == "dvol":
            snapshot.dvol = payload["price"]
            snapshot.dvol_timestamp = payload.get("timestamp")
            snapshot.available_sources.add("dvol")
        elif source == "fear_greed":
            snapshot.fear_greed_value = payload["value"]
            snapshot.fear_greed_classification = payload.get("classification")
            snapshot.available_sources.add("fear_greed")
        elif source == "btc_dominance":
            snapshot.btc_dominance = payload["value"]
            snapshot.available_sources.add("btc_dominance")
        elif source == "hyperliquid":
            snapshot.hl_total_oi = payload.get("total_oi")
            snapshot.hl_avg_funding = payload.get("avg_funding")
            snapshot.available_sources.add("hyperliquid")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:  # pragma: no cover — httpx is a project dep
            raise RuntimeError("httpx not available") from e
        self._client = httpx.Client(timeout=self._timeout)
        self._owns_client = True
        return self._client

    def _get_json(self, client: Any, url: str) -> Any:
        response = client.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _now_iso(clock: Any = None) -> str:
    if callable(clock):
        try:
            return _iso(clock())
        except Exception:  # noqa: BLE001
            pass
    return _iso(datetime.now(tz=timezone.utc))


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return _iso(datetime.fromtimestamp(float(value), tz=timezone.utc))
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_iso(value: str | None) -> datetime | None:
    """Tolerate Z suffix + missing tzinfo."""
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def filter_calendar_within(
    events: Iterable[EconomicEvent], reference: datetime, hours: float
) -> list[EconomicEvent]:
    """Return events within `hours` hours of `reference` (forward only)."""
    horizon = reference + timedelta(hours=hours)
    out: list[EconomicEvent] = []
    for ev in events:
        ts = ev.parsed_time()
        if ts is None:
            continue
        if reference - timedelta(minutes=30) <= ts <= horizon:
            out.append(ev)
    return out
