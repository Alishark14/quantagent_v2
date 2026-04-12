"""Unit tests for engine.data.flow.options.OptionsEnrichment.

The provider is code-only + network-bound, so every test either:
- exercises the pure parsers on hand-built Deribit-shaped dicts, OR
- uses an httpx.MockTransport to stub the Deribit endpoints so no
  real network I/O happens in CI.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from engine.data.flow.options import OptionsEnrichment


# ----------------------------------------------------------------------
# Fixture payloads mimicking Deribit response shapes
# ----------------------------------------------------------------------


def _make_book_summary(
    *,
    call_oi: float = 1000.0,
    put_oi: float = 800.0,
    call_gamma: float = 0.0005,
    put_gamma: float = 0.0005,
    underlying: float = 65000.0,
    include_greeks: bool = True,
    extra_rows: list[dict] | None = None,
) -> list[dict]:
    """Synthesize a plausible ``get_book_summary_by_currency`` result.

    Default: 2 instruments (one call, one put), both with gamma so
    GEX classification is POSITIVE / NEGATIVE rather than None.
    """
    rows: list[dict] = [
        {
            "instrument_name": "BTC-28MAR25-70000-C",
            "open_interest": call_oi,
            "underlying_price": underlying,
            "mark_iv": 55.0,
            "greeks": {"gamma": call_gamma} if include_greeks else {},
        },
        {
            "instrument_name": "BTC-28MAR25-60000-P",
            "open_interest": put_oi,
            "underlying_price": underlying,
            "mark_iv": 65.0,
            "greeks": {"gamma": put_gamma} if include_greeks else {},
        },
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return rows


def _make_dvol_response(
    oldest_close: float = 50.0, latest_close: float = 60.0
) -> dict:
    """Synthesize a ``get_volatility_index_data`` result (hourly rows)."""
    now_ms = int(time.time() * 1000)
    return {
        "data": [
            [now_ms - 3600 * 1000 * 24, 0, 0, 0, oldest_close],
            [now_ms - 3600 * 1000 * 12, 0, 0, 0, (oldest_close + latest_close) / 2],
            [now_ms, 0, 0, 0, latest_close],
        ],
        "continuation": None,
    }


def _deribit_envelope(result) -> dict:
    return {"jsonrpc": "2.0", "result": result, "usIn": 0, "usOut": 0}


def _make_mock_transport(
    *,
    book=None,
    dvol=None,
    book_status: int = 200,
    dvol_status: int = 200,
    book_raises: bool = False,
    dvol_raises: bool = False,
) -> httpx.MockTransport:
    """Build an httpx MockTransport wired to the two Deribit endpoints."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "get_book_summary_by_currency" in request.url.path:
            if book_raises:
                raise httpx.ConnectError("boom", request=request)
            if book_status != 200:
                return httpx.Response(book_status, json={"error": "fail"})
            return httpx.Response(200, json=_deribit_envelope(book or []))
        if "get_volatility_index_data" in request.url.path:
            if dvol_raises:
                raise httpx.ConnectError("boom", request=request)
            if dvol_status != 200:
                return httpx.Response(dvol_status, json={"error": "fail"})
            return httpx.Response(
                200, json=_deribit_envelope(dvol or {"data": []})
            )
        return httpx.Response(404, json={"error": "unknown endpoint"})

    return httpx.MockTransport(handler)


# ----------------------------------------------------------------------
# Symbol gating
# ----------------------------------------------------------------------


class TestSymbolGating:

    @pytest.mark.asyncio
    async def test_altcoin_returns_empty_no_network_call(self) -> None:
        # Transport that would EXPLODE if we touched it
        transport = _make_mock_transport(book_raises=True, dvol_raises=True)
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("SOL-USDC", adapter=None)

        assert result == {}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_btc_usdc_maps_to_btc(self) -> None:
        transport = _make_mock_transport(
            book=_make_book_summary(),
            dvol=_make_dvol_response(),
        )
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("BTC-USDC", adapter=None)
        assert "put_call_ratio" in result

    @pytest.mark.asyncio
    async def test_eth_usdc_maps_to_eth(self) -> None:
        transport = _make_mock_transport(
            book=_make_book_summary(),
            dvol=_make_dvol_response(),
        )
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("ETH-USDC", adapter=None)
        assert "put_call_ratio" in result


# ----------------------------------------------------------------------
# Parser correctness
# ----------------------------------------------------------------------


class TestBookSummaryParser:

    def test_put_call_ratio_computed_from_oi(self) -> None:
        book = _make_book_summary(call_oi=1000, put_oi=1300)
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        assert parsed["put_call_ratio"] == pytest.approx(1.3)

    def test_zero_call_oi_omits_ratio(self) -> None:
        book = [
            {
                "instrument_name": "BTC-28MAR25-60000-P",
                "open_interest": 500,
                "underlying_price": 65000.0,
                "mark_iv": 65.0,
                "greeks": {"gamma": 0.0005},
            }
        ]
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        assert "put_call_ratio" not in parsed

    def test_gex_classifies_positive(self) -> None:
        # Call gamma > put gamma → net GEX positive
        book = _make_book_summary(
            call_oi=2000, put_oi=500, call_gamma=0.001, put_gamma=0.0001
        )
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        assert parsed["gex_regime"] == "POSITIVE_GAMMA"

    def test_gex_classifies_negative(self) -> None:
        # Put gamma dominates → net GEX negative
        book = _make_book_summary(
            call_oi=500, put_oi=2000, call_gamma=0.0001, put_gamma=0.001
        )
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        assert parsed["gex_regime"] == "NEGATIVE_GAMMA"

    def test_no_gamma_means_no_regime(self) -> None:
        book = _make_book_summary(include_greeks=False)
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        assert "gex_regime" not in parsed

    def test_skew_approximation_picks_otm_put_minus_call(self) -> None:
        # Build an expiry with one call and one put ~7% OTM from 65000.
        # Put IV 75, call IV 55 → skew = +20.
        book = [
            {
                "instrument_name": "BTC-28MAR25-70000-C",
                "open_interest": 100,
                "underlying_price": 65000.0,
                "mark_iv": 55.0,
                "greeks": {"gamma": 0.0005},
            },
            {
                "instrument_name": "BTC-28MAR25-60000-P",
                "open_interest": 100,
                "underlying_price": 65000.0,
                "mark_iv": 75.0,
                "greeks": {"gamma": 0.0005},
            },
        ]
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        assert parsed["skew_25d"] == pytest.approx(20.0)

    def test_malformed_row_is_skipped(self) -> None:
        # One row missing instrument_name entirely — must not break parse
        book = _make_book_summary(
            extra_rows=[{"instrument_name": None, "open_interest": 500}]
        )
        parsed = OptionsEnrichment._parse_book_summary(book, "BTC")
        # PCR still present from the good rows
        assert "put_call_ratio" in parsed


class TestDvolParser:

    def test_dvol_current_is_latest_close(self) -> None:
        current, pct = OptionsEnrichment._parse_dvol(
            _make_dvol_response(oldest_close=50.0, latest_close=60.0)
        )
        assert current == pytest.approx(60.0)

    def test_dvol_change_is_percent(self) -> None:
        current, pct = OptionsEnrichment._parse_dvol(
            _make_dvol_response(oldest_close=50.0, latest_close=60.0)
        )
        assert pct == pytest.approx(20.0)

    def test_dvol_empty_data(self) -> None:
        current, pct = OptionsEnrichment._parse_dvol({"data": []})
        assert current is None
        assert pct is None

    def test_dvol_zero_anchor_keeps_current(self) -> None:
        # Oldest close is zero → division by zero avoided, current still set
        current, pct = OptionsEnrichment._parse_dvol(
            _make_dvol_response(oldest_close=0.0, latest_close=60.0)
        )
        assert current == pytest.approx(60.0)
        assert pct is None


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_book_summary_failure_returns_empty(self) -> None:
        transport = _make_mock_transport(
            book_raises=True, dvol=_make_dvol_response()
        )
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("BTC-USDC", adapter=None)
        # dvol still works even though book failed
        assert "dvol" in result
        assert "put_call_ratio" not in result
        await client.aclose()

    @pytest.mark.asyncio
    async def test_dvol_failure_returns_book_only(self) -> None:
        transport = _make_mock_transport(
            book=_make_book_summary(), dvol_raises=True
        )
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("BTC-USDC", adapter=None)
        assert "put_call_ratio" in result
        assert "dvol" not in result
        await client.aclose()

    @pytest.mark.asyncio
    async def test_both_endpoints_down_returns_empty(self) -> None:
        transport = _make_mock_transport(book_raises=True, dvol_raises=True)
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("BTC-USDC", adapter=None)
        assert result == {}
        await client.aclose()

    @pytest.mark.asyncio
    async def test_http_5xx_treated_as_failure(self) -> None:
        transport = _make_mock_transport(book_status=503, dvol_status=503)
        client = httpx.AsyncClient(transport=transport)
        provider = OptionsEnrichment(http_client=client)

        result = await provider.fetch("BTC-USDC", adapter=None)
        assert result == {}
        await client.aclose()


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------


class TestCache:

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_hits_cache(self) -> None:
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if "get_book_summary_by_currency" in request.url.path:
                return httpx.Response(
                    200, json=_deribit_envelope(_make_book_summary())
                )
            return httpx.Response(
                200, json=_deribit_envelope(_make_dvol_response())
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OptionsEnrichment(http_client=client, cache_ttl_seconds=900)

        first = await provider.fetch("BTC-USDC", adapter=None)
        second = await provider.fetch("BTC-USDC", adapter=None)

        # Two calls per fetch (book + dvol); second fetch must hit cache
        assert call_count["n"] == 2
        assert first == second
        await client.aclose()

    @pytest.mark.asyncio
    async def test_expired_cache_refetches(self) -> None:
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if "get_book_summary_by_currency" in request.url.path:
                return httpx.Response(
                    200, json=_deribit_envelope(_make_book_summary())
                )
            return httpx.Response(
                200, json=_deribit_envelope(_make_dvol_response())
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        # TTL = 0 means every call is stale
        provider = OptionsEnrichment(http_client=client, cache_ttl_seconds=0)

        await provider.fetch("BTC-USDC", adapter=None)
        await provider.fetch("BTC-USDC", adapter=None)

        assert call_count["n"] == 4  # 2 endpoints × 2 fetches
        await client.aclose()

    @pytest.mark.asyncio
    async def test_cache_is_per_symbol(self) -> None:
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if "get_book_summary_by_currency" in request.url.path:
                return httpx.Response(
                    200, json=_deribit_envelope(_make_book_summary())
                )
            return httpx.Response(
                200, json=_deribit_envelope(_make_dvol_response())
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OptionsEnrichment(http_client=client, cache_ttl_seconds=900)

        await provider.fetch("BTC-USDC", adapter=None)
        await provider.fetch("ETH-USDC", adapter=None)

        # BTC + ETH = 2 currencies × 2 endpoints each
        assert call_count["n"] == 4
        await client.aclose()


# ----------------------------------------------------------------------
# Identity / metadata
# ----------------------------------------------------------------------


class TestProviderIdentity:

    def test_name(self) -> None:
        assert OptionsEnrichment().name() == "options"
