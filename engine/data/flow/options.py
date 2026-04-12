"""OptionsEnrichment: Deribit-backed options flow provider for BTC / ETH.

Deribit handles ~85%+ of listed BTC/ETH options volume and exposes a
public read-only REST API (``https://www.deribit.com/api/v2/public/``)
that needs no auth for market data. This provider pulls four options-
derived signals that are independent from spot/perp price action and
merges them into :class:`FlowOutput`:

* ``put_call_ratio`` — total put OI / total call OI across all active
  strikes + expiries. > 1 means the book is tilted toward downside
  hedging; < 0.5 means complacent call dominance.
* ``dvol`` — Deribit's implied-volatility index, current value.
* ``dvol_change_24h`` — % change in DVOL over the last 24 hours.
* ``skew_25d`` — 25-delta put IV minus 25-delta call IV, approximated
  using the nearest-monthly-expiry book summary. Positive skew means
  the market is paying for downside protection.
* ``gex_regime`` — POSITIVE / NEGATIVE gamma classification from the
  sum of ``oi * gamma * contract_size * underlying_price`` across all
  active instruments. Positive = market makers dampen moves (ranging),
  negative = market makers amplify moves (trending).

Scope: BTC-USDC and ETH-USDC only. Every other symbol short-circuits to
an empty dict so altcoins (SOL-USDC, DOGE-USDC, ...) never pay the
network round-trip — Deribit does not list options for them and the
upstream call would 404 anyway.

Caching: 15-minute in-memory TTL per symbol. Options books move far
slower than perp funding; 15 min is well under the rate-limit budget
and far above the Sentinel scan frequency so we aren't hammering
Deribit from every cycle.

Errors are logged-and-swallowed: a Deribit outage must NEVER crash the
analysis pipeline — the provider returns an empty dict and FlowAgent
falls through to ``None`` on every options field. FlowSignalAgent's
rule set already handles ``None`` gracefully.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from engine.data.flow.base import FlowProvider
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)

_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

# TTL for the per-symbol cache. 15 minutes keeps us well under the
# Deribit 20 req/sec public rate limit even with 22 Sentinel symbols
# scanning every 30s (BTC + ETH only = 2 calls per 15 min total).
_CACHE_TTL_SECONDS = 15 * 60

# 24h window for dvol_change_24h. DVOL candles are hourly so 24 points
# is enough for a one-day delta with no interpolation.
_DVOL_LOOKBACK_SECONDS = 24 * 60 * 60

# HTTP timeout — Deribit public API normally responds in < 1s; 10s gives
# plenty of headroom for a congested network without hanging a Sentinel
# cycle if the endpoint is actually down.
_HTTP_TIMEOUT = 10.0

# Map internal symbol → Deribit currency code. Only BTC and ETH are
# supported because Deribit does not list public options for altcoins
# and the upstream calls would return empty books for anything else.
_SYMBOL_CURRENCY_MAP: dict[str, str] = {
    "BTC-USDC": "BTC",
    "ETH-USDC": "ETH",
}

# Per-option GEX formula needs a contract size. Deribit BTC / ETH
# options are 1 contract = 1 underlying unit, so we use 1.0 — this
# makes the GEX number approximately "gamma-weighted OI in underlying
# units × spot" which is enough for a sign-only POSITIVE / NEGATIVE
# regime classification. Exact dollar GEX would require index price
# per expiry; we don't need that for v1.
_CONTRACT_SIZE = 1.0

# Rough proxy for "25-delta put / call" when we don't have the Greeks
# to interpolate: step through the book summary sorted by strike and
# take the strike ~7% OTM from spot on each side. This is intentionally
# approximate for v1 — a full Black-Scholes interpolation lives on the
# roadmap once we wire a pricing model.
_APPROX_25D_OTM_PCT = 0.07


class OptionsEnrichment(FlowProvider):
    """Fetches Deribit options data and produces FlowOutput enrichment."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        cache_ttl_seconds: int = _CACHE_TTL_SECONDS,
    ) -> None:
        # Allow injecting a mocked client for tests. When None we lazily
        # build one per-instance on first use.
        self._client: httpx.AsyncClient | None = http_client
        self._owns_client: bool = http_client is None
        self._cache_ttl_seconds = cache_ttl_seconds

        # Per-symbol cache — stores (expires_at_epoch_seconds, data_dict).
        self._cache: dict[str, tuple[float, dict]] = {}

    def name(self) -> str:
        return "options"

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Fetch options enrichment for ``symbol``.

        Returns an empty dict for anything outside BTC/ETH — Deribit
        does not list public options for altcoins and FlowAgent merges
        empty dicts without touching the existing fields. On any
        Deribit error the provider logs a warning and returns ``{}``.
        """
        currency = _SYMBOL_CURRENCY_MAP.get(symbol)
        if currency is None:
            return {}

        now = time.time()
        cached = self._cache.get(symbol)
        if cached is not None:
            expires_at, data = cached
            if now < expires_at:
                return dict(data)

        try:
            data = await self._fetch_from_deribit(currency)
        except Exception as e:
            logger.warning(
                f"OptionsEnrichment: Deribit fetch failed for {symbol}: {e}"
            )
            return {}

        self._cache[symbol] = (now + self._cache_ttl_seconds, dict(data))
        return dict(data)

    async def close(self) -> None:
        """Release the internal HTTP client. Safe to call multiple times."""
        if self._owns_client and self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Internals — network + parsing
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        return self._client

    async def _fetch_from_deribit(self, currency: str) -> dict:
        """Call the two Deribit endpoints and assemble the enrichment dict.

        Each individual sub-call is try/excepted so a partial outage
        (e.g. DVOL available but book summary down) still surfaces
        whatever data it can. Missing fields end up as None in the
        returned dict and FlowAgent merges them accordingly.
        """
        client = await self._get_client()
        result: dict[str, Any] = {}

        # 1. Book summary by currency — OI per instrument, greeks, IV.
        try:
            book = await self._get_json(
                client,
                "get_book_summary_by_currency",
                {"currency": currency, "kind": "option"},
            )
        except Exception as e:
            logger.warning(
                f"OptionsEnrichment: get_book_summary_by_currency failed "
                f"for {currency}: {e}"
            )
            book = None

        if book:
            result.update(self._parse_book_summary(book, currency))

        # 2. Volatility index — DVOL current + 24h change.
        try:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - _DVOL_LOOKBACK_SECONDS * 1000
            dvol = await self._get_json(
                client,
                "get_volatility_index_data",
                {
                    "currency": currency,
                    "resolution": 3600,
                    "start_timestamp": start_ms,
                    "end_timestamp": now_ms,
                },
            )
        except Exception as e:
            logger.warning(
                f"OptionsEnrichment: get_volatility_index_data failed "
                f"for {currency}: {e}"
            )
            dvol = None

        if dvol:
            dvol_current, dvol_change_24h = self._parse_dvol(dvol)
            if dvol_current is not None:
                result["dvol"] = dvol_current
            if dvol_change_24h is not None:
                result["dvol_change_24h"] = dvol_change_24h

        return result

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: dict,
    ) -> Any:
        """GET a Deribit endpoint and return ``response.result``.

        Deribit wraps every response in ``{"jsonrpc", "result", ...}``;
        we strip the envelope so callers work with the raw payload.
        """
        url = f"{_DERIBIT_BASE}/{endpoint}"
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if "result" not in payload:
            raise ValueError(f"Deribit {endpoint}: missing 'result' in response")
        return payload["result"]

    # ------------------------------------------------------------------
    # Parsers — pure functions so tests can hit them with fixture data
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_book_summary(
        book: list[dict], currency: str
    ) -> dict:
        """Compute put/call ratio, skew, and GEX regime from the raw book.

        ``book`` is Deribit's ``get_book_summary_by_currency`` result:
        a list of per-instrument dicts with at least ``instrument_name``,
        ``open_interest``, ``underlying_price`` and (when available)
        ``greeks.gamma`` / ``mark_iv``. Parsing is tolerant of missing
        fields — any instrument that can't be decoded is skipped.
        """
        total_call_oi = 0.0
        total_put_oi = 0.0
        gex_sum = 0.0
        saw_any_gamma = False

        # Per-expiry aggregation for the 25-delta skew approximation.
        # Keyed by expiry tag (the "DDMMMYY" slug from the instrument
        # name), each value is a list of (strike, put_or_call, iv) tuples.
        per_expiry: dict[str, list[tuple[float, str, float]]] = {}

        spot_price: float | None = None

        for row in book:
            try:
                instrument = str(row.get("instrument_name") or "")
                kind = _classify_instrument(instrument)
                if kind is None:
                    continue
                expiry_tag, strike = _parse_expiry_and_strike(instrument)
                if strike is None:
                    continue

                oi = _safe_float(row.get("open_interest"))
                if oi is None:
                    oi = 0.0

                underlying = _safe_float(row.get("underlying_price"))
                if underlying is not None and underlying > 0:
                    spot_price = underlying

                # Put / call OI totals
                if kind == "C":
                    total_call_oi += oi
                else:
                    total_put_oi += oi

                # GEX: oi * gamma * contract_size * underlying_price
                greeks = row.get("greeks") or {}
                gamma = _safe_float(greeks.get("gamma"))
                if gamma is not None and underlying is not None:
                    saw_any_gamma = True
                    contribution = (
                        oi * gamma * _CONTRACT_SIZE * underlying
                    )
                    if kind == "C":
                        gex_sum += contribution
                    else:
                        gex_sum -= contribution

                # Collect for skew approximation
                iv = _safe_float(row.get("mark_iv"))
                if iv is not None and expiry_tag is not None:
                    per_expiry.setdefault(expiry_tag, []).append(
                        (strike, kind, iv)
                    )
            except Exception:
                # One malformed row must NOT break the whole parse.
                continue

        out: dict[str, Any] = {}

        if total_call_oi > 0:
            out["put_call_ratio"] = total_put_oi / total_call_oi

        if saw_any_gamma:
            # Sign-only classification is enough for the v1 rule set.
            # A non-zero but tiny GEX near the flip level still
            # classifies as POSITIVE / NEGATIVE — we don't have a
            # stable flip-level computation yet (that's Phase 2).
            out["gex_regime"] = (
                "POSITIVE_GAMMA" if gex_sum >= 0 else "NEGATIVE_GAMMA"
            )

        skew = _approximate_25d_skew(per_expiry, spot_price)
        if skew is not None:
            out["skew_25d"] = skew

        return out

    @staticmethod
    def _parse_dvol(dvol_response: dict) -> tuple[float | None, float | None]:
        """Extract (current_dvol, pct_change_24h) from the DVOL endpoint.

        The Deribit response wraps hourly DVOL tuples in ``data`` where
        each row is ``[ts_ms, open, high, low, close]``. We treat the
        newest ``close`` as ``current`` and the oldest ``close`` inside
        the 24h window as the anchor for the % change.
        """
        rows = dvol_response.get("data") if isinstance(dvol_response, dict) else None
        if not rows:
            return None, None

        try:
            # Deribit returns oldest-first; sort defensively.
            sorted_rows = sorted(rows, key=lambda r: r[0])
            oldest_close = _safe_float(sorted_rows[0][4])
            latest_close = _safe_float(sorted_rows[-1][4])
        except (IndexError, TypeError, ValueError):
            return None, None

        if latest_close is None:
            return None, None

        if oldest_close is None or oldest_close == 0 or latest_close is None:
            return latest_close, None

        pct = ((latest_close - oldest_close) / oldest_close) * 100.0
        return latest_close, pct


# ----------------------------------------------------------------------
# Module-private helpers
# ----------------------------------------------------------------------


def _safe_float(value) -> float | None:
    """Coerce to float, returning None on anything that can't cast."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_instrument(name: str) -> str | None:
    """Return 'C' / 'P' for a Deribit options instrument name, else None.

    Deribit format: ``BTC-28MAR25-80000-C`` / ``ETH-28MAR25-3500-P``.
    Anything that doesn't end in ``-C`` / ``-P`` is not an option — skip.
    """
    if not name:
        return None
    if name.endswith("-C"):
        return "C"
    if name.endswith("-P"):
        return "P"
    return None


def _parse_expiry_and_strike(name: str) -> tuple[str | None, float | None]:
    """Pull the expiry tag + strike out of a Deribit instrument name.

    For ``BTC-28MAR25-80000-C`` returns ``("28MAR25", 80000.0)``. Returns
    ``(None, None)`` on anything that doesn't split cleanly.
    """
    parts = name.split("-")
    if len(parts) < 4:
        return None, None
    expiry = parts[1]
    try:
        strike = float(parts[2])
    except (TypeError, ValueError):
        return expiry, None
    return expiry, strike


def _approximate_25d_skew(
    per_expiry: dict[str, list[tuple[float, str, float]]],
    spot_price: float | None,
) -> float | None:
    """Approximate 25-delta put-minus-call skew from the nearest expiry.

    V1 approach: find the nearest-in-time monthly expiry (approximated
    as the expiry with the largest total IV contribution — a proxy for
    "most-traded monthly"), then pick the put strike nearest
    ``spot * (1 - _APPROX_25D_OTM_PCT)`` and the call strike nearest
    ``spot * (1 + _APPROX_25D_OTM_PCT)``, returning their IV delta.

    Returns ``None`` when any required piece is missing — the rule set
    handles None gracefully.
    """
    if spot_price is None or spot_price <= 0 or not per_expiry:
        return None

    # Pick the expiry with the most instruments (proxy for nearest monthly).
    target_expiry = max(per_expiry.items(), key=lambda kv: len(kv[1]))[0]
    strikes = per_expiry[target_expiry]
    if not strikes:
        return None

    put_target = spot_price * (1.0 - _APPROX_25D_OTM_PCT)
    call_target = spot_price * (1.0 + _APPROX_25D_OTM_PCT)

    puts = [(s, iv) for s, kind, iv in strikes if kind == "P"]
    calls = [(s, iv) for s, kind, iv in strikes if kind == "C"]
    if not puts or not calls:
        return None

    nearest_put = min(puts, key=lambda si: abs(si[0] - put_target))
    nearest_call = min(calls, key=lambda si: abs(si[0] - call_target))

    return float(nearest_put[1] - nearest_call[1])
