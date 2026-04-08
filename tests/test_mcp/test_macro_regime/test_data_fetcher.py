"""Tests for the MacroDataFetcher.

The fetcher is wired up against fake HTTP clients — no real network
traffic. Each fetcher is exercised individually, plus a happy-path
end-to-end run, plus the partial-failure / total-failure paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from mcp.macro_regime.data_fetcher import (
    MacroDataFetcher,
    MacroSnapshot,
    EconomicEvent,
    filter_calendar_within,
    parse_iso,
    _hardcoded_calendar,
)


# ---------------------------------------------------------------------------
# Fake HTTP client infrastructure
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Any, *, status: int = 200):
        self._payload = payload
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Maps URL substring → response payload (or callable raising)."""

    def __init__(self, routes: dict[str, Any], *, post_routes: dict[str, Any] | None = None):
        self._routes = routes
        self._post_routes = post_routes or {}
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict]] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:
        self.get_calls.append(url)
        for needle, payload in self._routes.items():
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                if callable(payload):
                    return payload()
                return _FakeResponse(payload)
        raise RuntimeError(f"FakeClient: no route for {url}")

    def post(self, url: str, *, json: dict, timeout: float | None = None) -> _FakeResponse:
        self.post_calls.append((url, json))
        for needle, payload in self._post_routes.items():
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                return _FakeResponse(payload)
        raise RuntimeError(f"FakeClient: no POST route for {url}")


def _yahoo_payload(price: float, ts_epoch: int) -> dict:
    return {
        "quoteResponse": {
            "result": [
                {"regularMarketPrice": price, "regularMarketTime": ts_epoch}
            ]
        }
    }


def _hl_payload() -> list:
    return [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
        [
            {"openInterest": "100", "markPx": "70000", "funding": "0.0001"},
            {"openInterest": "200", "markPx": "3500", "funding": "-0.0002"},
        ],
    ]


_FIXED_NOW = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def fake_client():
    return _FakeClient(
        routes={
            "^VIX": _yahoo_payload(18.5, 1712577600),
            "DX-Y.NYB": _yahoo_payload(104.2, 1712577600),
            "deribit.com": {"result": {"index_price": 55.4}},
            "alternative.me": {
                "data": [{"value": "25", "value_classification": "Extreme Fear"}]
            },
            "coingecko.com": {
                "data": {"market_cap_percentage": {"btc": 51.3, "eth": 17.1}}
            },
        },
        post_routes={"hyperliquid.xyz": _hl_payload()},
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fetch_happy_path_populates_all_sources(fake_client):
    fetcher = MacroDataFetcher(
        http_client=fake_client,
        rate_limit_seconds=0.0,
        clock=lambda: _FIXED_NOW,
    )
    snap = fetcher.fetch()
    assert snap.vix == pytest.approx(18.5)
    assert snap.dxy == pytest.approx(104.2)
    assert snap.dvol == pytest.approx(55.4)
    assert snap.fear_greed_value == 25
    assert snap.fear_greed_classification == "Extreme Fear"
    assert snap.btc_dominance == pytest.approx(51.3)
    assert snap.hl_total_oi == pytest.approx(100 * 70000 + 200 * 3500)
    assert snap.hl_avg_funding == pytest.approx((0.0001 + -0.0002) / 2)
    assert {
        "vix",
        "dxy",
        "dvol",
        "fear_greed",
        "btc_dominance",
        "hyperliquid",
        "calendar",
    }.issubset(snap.available_sources)
    assert snap.errors == {}
    assert len(snap.economic_calendar) >= 1


def test_fetch_includes_hardcoded_calendar_with_fixed_clock(fake_client):
    fetcher = MacroDataFetcher(
        http_client=fake_client,
        rate_limit_seconds=0.0,
        clock=lambda: _FIXED_NOW,
    )
    snap = fetcher.fetch()
    names = {ev.name for ev in snap.economic_calendar}
    # CPI on 2026-04-14 is within the 30-day lookahead from 2026-04-08
    assert "CPI" in names


# ---------------------------------------------------------------------------
# Partial / total failure
# ---------------------------------------------------------------------------


def test_partial_failure_keeps_other_sources():
    client = _FakeClient(
        routes={
            "^VIX": RuntimeError("yahoo down"),
            "DX-Y.NYB": _yahoo_payload(103.0, 1712577600),
            "deribit.com": {"result": {"index_price": 60.0}},
            "alternative.me": {"data": [{"value": "50", "value_classification": "Neutral"}]},
            "coingecko.com": RuntimeError("coingecko 503"),
        },
        post_routes={"hyperliquid.xyz": _hl_payload()},
    )
    fetcher = MacroDataFetcher(
        http_client=client, rate_limit_seconds=0.0, clock=lambda: _FIXED_NOW
    )
    snap = fetcher.fetch()
    assert snap.vix is None
    assert snap.dxy == pytest.approx(103.0)
    assert snap.btc_dominance is None
    assert "vix" in snap.errors
    assert "btc_dominance" in snap.errors
    assert "vix" not in snap.available_sources
    assert "dxy" in snap.available_sources
    assert "calendar" in snap.available_sources


def test_total_failure_returns_calendar_only():
    client = _FakeClient(
        routes={
            "^VIX": RuntimeError("down"),
            "DX-Y.NYB": RuntimeError("down"),
            "deribit.com": RuntimeError("down"),
            "alternative.me": RuntimeError("down"),
            "coingecko.com": RuntimeError("down"),
        },
        post_routes={"hyperliquid.xyz": RuntimeError("down")},
    )
    fetcher = MacroDataFetcher(
        http_client=client, rate_limit_seconds=0.0, clock=lambda: _FIXED_NOW
    )
    snap = fetcher.fetch()
    assert snap.vix is None
    assert snap.dxy is None
    assert snap.dvol is None
    assert snap.fear_greed_value is None
    assert snap.btc_dominance is None
    assert snap.hl_total_oi is None
    # Five HTTP sources failed; calendar still present.
    assert len(snap.errors) == 6
    assert "calendar" in snap.available_sources


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_called_between_fetches(fake_client):
    sleeps: list[float] = []
    fetcher = MacroDataFetcher(
        http_client=fake_client,
        rate_limit_seconds=1.5,
        sleep_func=sleeps.append,
        clock=lambda: _FIXED_NOW,
    )
    fetcher.fetch()
    # 6 fetchers → 5 sleeps between them
    assert sleeps == [1.5] * 5


def test_zero_rate_limit_skips_sleep(fake_client):
    sleeps: list[float] = []
    fetcher = MacroDataFetcher(
        http_client=fake_client,
        rate_limit_seconds=0.0,
        sleep_func=sleeps.append,
        clock=lambda: _FIXED_NOW,
    )
    fetcher.fetch()
    assert sleeps == []


# ---------------------------------------------------------------------------
# MacroSnapshot round-trip
# ---------------------------------------------------------------------------


def test_macro_snapshot_to_from_dict_roundtrip():
    snap = MacroSnapshot(
        fetched_at="2026-04-08T12:00:00Z",
        vix=18.5,
        vix_timestamp="2026-04-08T11:55:00Z",
        dvol=55.4,
        fear_greed_value=25,
        fear_greed_classification="Extreme Fear",
        btc_dominance=51.3,
        hl_total_oi=1_000_000.0,
        hl_avg_funding=0.0001,
        economic_calendar=[
            EconomicEvent("FOMC_ANNOUNCEMENT", "2026-04-29T18:00:00Z", "HIGH")
        ],
        available_sources={"vix", "dvol", "fear_greed", "btc_dominance", "hyperliquid", "calendar"},
        errors={"dxy": "down"},
    )
    payload = snap.to_dict()
    rebuilt = MacroSnapshot.from_dict(payload)
    assert rebuilt.vix == 18.5
    assert rebuilt.fear_greed_classification == "Extreme Fear"
    assert rebuilt.errors == {"dxy": "down"}
    assert "vix" in rebuilt.available_sources
    assert rebuilt.economic_calendar[0].name == "FOMC_ANNOUNCEMENT"


def test_macro_snapshot_has_data_helper():
    snap = MacroSnapshot(fetched_at="2026-04-08T12:00:00Z", available_sources={"vix"})
    assert snap.has_data("vix")
    assert not snap.has_data("dxy")


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def test_hardcoded_calendar_only_emits_within_lookahead():
    ref = datetime(2026, 4, 8, tzinfo=timezone.utc)
    events = _hardcoded_calendar(ref, lookahead_days=30)
    for ev in events:
        ts = ev.parsed_time()
        assert ts is not None
        assert ts <= ref + timedelta(days=30)


def test_filter_calendar_within_horizon_filtering():
    ref = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    events = [
        EconomicEvent("CPI", "2026-04-08T20:00:00Z", "HIGH"),  # within 24h
        EconomicEvent("NFP", "2026-04-12T12:30:00Z", "HIGH"),  # outside 24h
    ]
    within = filter_calendar_within(events, ref, hours=24.0)
    assert len(within) == 1
    assert within[0].name == "CPI"


def test_parse_iso_handles_z_suffix():
    dt = parse_iso("2026-04-08T12:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2026


def test_parse_iso_invalid_returns_none():
    assert parse_iso("not-a-date") is None
    assert parse_iso("") is None
    assert parse_iso(None) is None


# ---------------------------------------------------------------------------
# Hyperliquid edge cases
# ---------------------------------------------------------------------------


def test_hyperliquid_malformed_payload_returns_none():
    client = _FakeClient(
        routes={
            "^VIX": _yahoo_payload(18.0, 1712577600),
            "DX-Y.NYB": _yahoo_payload(104.0, 1712577600),
            "deribit.com": {"result": {"index_price": 50.0}},
            "alternative.me": {"data": [{"value": "50", "value_classification": "Neutral"}]},
            "coingecko.com": {"data": {"market_cap_percentage": {"btc": 51.0}}},
        },
        post_routes={"hyperliquid.xyz": {"unexpected": "shape"}},
    )
    fetcher = MacroDataFetcher(
        http_client=client, rate_limit_seconds=0.0, clock=lambda: _FIXED_NOW
    )
    snap = fetcher.fetch()
    assert snap.hl_total_oi is None
    assert "hyperliquid" not in snap.available_sources


def test_no_http_client_falls_back_to_calendar_only():
    class _NoClient:
        pass

    fetcher = MacroDataFetcher(
        http_client=None,
        rate_limit_seconds=0.0,
        clock=lambda: _FIXED_NOW,
    )

    # Force the lazy resolver to fail by stubbing httpx import.
    import sys
    old_httpx = sys.modules.get("httpx")
    sys.modules["httpx"] = None  # type: ignore[assignment]
    try:
        # Patch _resolve_client to raise — simulates no httpx.
        def _bad():
            raise RuntimeError("no httpx in test")
        fetcher._resolve_client = _bad  # type: ignore[method-assign]
        snap = fetcher.fetch()
    finally:
        if old_httpx is not None:
            sys.modules["httpx"] = old_httpx
        else:
            sys.modules.pop("httpx", None)

    assert "http_client" in snap.errors
    # Calendar still works.
    assert "calendar" in snap.available_sources
