"""Unit tests for EquityFlowProvider (FINRA RegSHO daily short-volume).

The provider wraps a FINRA CDN endpoint; every test stubs it with
``httpx.MockTransport`` so CI never hits the network.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
import pytest_asyncio

from engine.data.flow.equity import (
    EQUITY_SYMBOLS,
    EquityFlowProvider,
    _is_us_market_open,
    _most_recent_trading_day,
    _previous_weekday,
)
from storage.repositories.sqlite import SQLiteRepositories


# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


@pytest_asyncio.fixture
async def repos(tmp_path):
    db_path = str(tmp_path / "test_regsho.db")
    r = SQLiteRepositories(db_path=db_path)
    await r.init_db()
    return r


def _sample_regsho_text(
    *, trade_date: date = date(2026, 4, 10)
) -> str:
    """Build a small FINRA file body with TSLA / NVDA / GOOGL + noise."""
    ymd = trade_date.strftime("%Y%m%d")
    return (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        f"{ymd}|A|319927|1916|1643899|B,Q,N\n"
        f"{ymd}|AAPL|50000000|100000|200000000|B,Q,N\n"
        f"{ymd}|GOOGL|5557560|7910|19770243|B,Q,N\n"
        f"{ymd}|NVDA|138295075|421482|251919199|B,Q,N\n"
        f"{ymd}|TSLA|31503665|139485|96692786|B,Q,N\n"
    )


def _mock_transport(
    *,
    responses: dict[str, tuple[int, str]] | None = None,
    default_404: bool = True,
) -> httpx.MockTransport:
    """Return a MockTransport that serves canned responses by URL.

    ``responses`` maps URL suffix (e.g. ``"CNMSshvol20260410.txt"``)
    to ``(status_code, body)``. Unmatched requests return 404 by
    default so backoff-walk logic can exercise the negative path.
    """
    responses = responses or {}

    def handler(request: httpx.Request) -> httpx.Response:
        for suffix, (status, body) in responses.items():
            if request.url.path.endswith(suffix):
                return httpx.Response(status, text=body)
        if default_404:
            return httpx.Response(404, text="")
        return httpx.Response(500, text="")

    return httpx.MockTransport(handler)


# ----------------------------------------------------------------------
# Symbol gating
# ----------------------------------------------------------------------


class TestSymbolGating:

    @pytest.mark.asyncio
    async def test_crypto_symbol_returns_empty(self) -> None:
        provider = EquityFlowProvider()
        result = await provider.fetch("BTC-USDC", adapter=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_unknown_symbol_returns_empty(self) -> None:
        provider = EquityFlowProvider()
        result = await provider.fetch("SOL-USDC", adapter=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_equity_symbol_always_gets_market_open_flag(self) -> None:
        """Even with an empty history, an equity fetch returns at least
        the market_open flag so the closed-market rule can fire."""
        provider = EquityFlowProvider()

        # Stub refresh so no network hit
        async def _noop_refresh():
            return
        provider._maybe_refresh = _noop_refresh  # type: ignore

        result = await provider.fetch("TSLA-USDC", adapter=None)
        assert "market_open" in result
        assert isinstance(result["market_open"], bool)


# ----------------------------------------------------------------------
# FINRA file parser
# ----------------------------------------------------------------------


class TestRegSHOParser:

    def test_parses_target_tickers(self) -> None:
        text = _sample_regsho_text(trade_date=date(2026, 4, 10))
        parsed = EquityFlowProvider._parse_regsho_text(text)
        assert set(parsed.keys()) == {
            "TSLA-USDC", "NVDA-USDC", "GOOGL-USDC"
        }

    def test_computes_short_volume_ratio(self) -> None:
        text = _sample_regsho_text(trade_date=date(2026, 4, 10))
        parsed = EquityFlowProvider._parse_regsho_text(text)
        tsla = parsed["TSLA-USDC"]
        assert tsla["short_volume"] == 31_503_665
        assert tsla["total_volume"] == 96_692_786
        # 31503665 / 96692786 ≈ 0.32582
        assert tsla["short_volume_ratio"] == pytest.approx(0.32582, abs=1e-4)

    def test_ignores_non_target_tickers(self) -> None:
        """Random unrelated tickers in the same file must NOT leak in."""
        text = _sample_regsho_text()
        parsed = EquityFlowProvider._parse_regsho_text(text)
        assert "A-USDC" not in parsed
        assert "AAPL-USDC" not in parsed

    def test_skips_malformed_rows(self) -> None:
        text = (
            "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "GARBAGE LINE NO PIPES\n"
            "bad|rows|not|enough|cols\n"  # missing market column is fine
            "NOT_A_DATE|TSLA|1|2|3|B\n"
            "20260410|TSLA|abc|0|100|B\n"  # non-numeric short_volume
            "20260410|TSLA|10|0|0|B\n"  # zero total volume
            "20260410|TSLA|50|0|100|B\n"  # GOOD
            "20260410|NVDA|30|0|100|B\n"  # GOOD
        )
        parsed = EquityFlowProvider._parse_regsho_text(text)
        # The good TSLA row wins over the preceding bad ones
        assert "TSLA-USDC" in parsed
        assert parsed["TSLA-USDC"]["short_volume"] == 50
        assert "NVDA-USDC" in parsed

    def test_empty_file_returns_empty(self) -> None:
        assert EquityFlowProvider._parse_regsho_text("") == {}

    def test_parses_decimal_volume_values(self) -> None:
        """FINRA's CDN feed publishes short_volume and total_volume
        as DECIMAL strings (e.g. ``31331839.752390``) — probably
        because fractional-share trading gets routed through the
        same off-exchange pipe. The parser must accept them, not
        choke on the ``.``. Regression guard.
        """
        text = (
            "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\r\n"
            "20260410|TSLA|15931717.558290|83627|28975057.835976|B,Q,N\r\n"
            "20260410|NVDA|31331839.752390|206713|71187182.614544|B,Q,N\r\n"
        )
        parsed = EquityFlowProvider._parse_regsho_text(text)
        assert "TSLA-USDC" in parsed
        assert "NVDA-USDC" in parsed
        assert parsed["TSLA-USDC"]["short_volume"] == 15_931_717  # truncated
        assert parsed["TSLA-USDC"]["total_volume"] == 28_975_057
        # Ratio still computed correctly off the floats
        assert parsed["TSLA-USDC"]["short_volume_ratio"] == pytest.approx(
            15_931_717 / 28_975_057, abs=1e-4
        )


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------


def _history(ratios: list[float], start_date: date = date(2026, 3, 1)) -> list[dict]:
    return [
        {
            "trade_date": start_date + timedelta(days=i),
            "short_volume": int(r * 1_000_000),
            "total_volume": 1_000_000,
            "short_volume_ratio": r,
        }
        for i, r in enumerate(ratios)
    ]


class TestMetricComputation:

    def test_short_volume_ratio_matches_latest(self) -> None:
        out = EquityFlowProvider._compute_metrics(_history([0.40, 0.45, 0.52]))
        assert out["short_volume_ratio"] == pytest.approx(0.52)

    def test_zscore_omitted_when_history_is_flat(self) -> None:
        """A perfectly flat history has std=0 → z-score is undefined.
        Implementation omits the field rather than dividing by zero."""
        ratios = [0.45] * 20 + [0.70]
        out = EquityFlowProvider._compute_metrics(_history(ratios))
        assert "svr_zscore" not in out
        assert out["short_volume_ratio"] == pytest.approx(0.70)

    def test_zscore_measurable_on_nearly_flat_series(self) -> None:
        import random
        random.seed(42)
        ratios = [0.50 + random.uniform(-0.005, 0.005) for _ in range(20)]
        ratios.append(0.65)  # well above the noise floor
        out = EquityFlowProvider._compute_metrics(_history(ratios))
        assert out["svr_zscore"] is not None
        assert out["svr_zscore"] > 2.0

    def test_zscore_negative_on_fresh_drop(self) -> None:
        import random
        random.seed(123)
        ratios = [0.50 + random.uniform(-0.005, 0.005) for _ in range(20)]
        ratios.append(0.35)  # sharp drop
        out = EquityFlowProvider._compute_metrics(_history(ratios))
        assert out["svr_zscore"] < -2.0

    def test_zscore_omitted_with_too_few_points(self) -> None:
        out = EquityFlowProvider._compute_metrics(_history([0.45, 0.52]))
        assert "svr_zscore" not in out

    def test_trend_rising(self) -> None:
        # 20-day avg low, 5-day avg high → RISING
        ratios = [0.40] * 15 + [0.55, 0.56, 0.57, 0.58, 0.60]
        out = EquityFlowProvider._compute_metrics(_history(ratios))
        assert out["svr_trend"] == "RISING"

    def test_trend_falling(self) -> None:
        ratios = [0.55] * 15 + [0.40, 0.39, 0.38, 0.37, 0.36]
        out = EquityFlowProvider._compute_metrics(_history(ratios))
        assert out["svr_trend"] == "FALLING"

    def test_trend_stable(self) -> None:
        ratios = [0.50] * 20
        out = EquityFlowProvider._compute_metrics(_history(ratios))
        assert out["svr_trend"] == "STABLE"


# ----------------------------------------------------------------------
# Market hours helper
# ----------------------------------------------------------------------


class TestMarketHours:
    """Deterministic tests — pass an explicit ``now`` so the helper
    doesn't depend on when CI runs."""

    def test_monday_10am_et_open(self) -> None:
        et = ZoneInfo("America/New_York")
        now = datetime(2026, 4, 13, 10, 0, tzinfo=et)  # Monday
        assert _is_us_market_open(now) is True

    def test_monday_8am_et_closed_premarket(self) -> None:
        et = ZoneInfo("America/New_York")
        now = datetime(2026, 4, 13, 8, 0, tzinfo=et)
        assert _is_us_market_open(now) is False

    def test_monday_5pm_et_closed_afterhours(self) -> None:
        et = ZoneInfo("America/New_York")
        now = datetime(2026, 4, 13, 17, 0, tzinfo=et)
        assert _is_us_market_open(now) is False

    def test_saturday_noon_et_closed(self) -> None:
        et = ZoneInfo("America/New_York")
        now = datetime(2026, 4, 11, 12, 0, tzinfo=et)  # Saturday
        assert _is_us_market_open(now) is False

    def test_sunday_closed(self) -> None:
        et = ZoneInfo("America/New_York")
        now = datetime(2026, 4, 12, 12, 0, tzinfo=et)  # Sunday
        assert _is_us_market_open(now) is False

    def test_market_close_boundary_inclusive(self) -> None:
        et = ZoneInfo("America/New_York")
        at_close = datetime(2026, 4, 13, 16, 0, tzinfo=et)
        assert _is_us_market_open(at_close) is True

    def test_market_open_boundary_inclusive(self) -> None:
        et = ZoneInfo("America/New_York")
        at_open = datetime(2026, 4, 13, 9, 30, tzinfo=et)
        assert _is_us_market_open(at_open) is True


class TestTradingDayHelpers:

    def test_most_recent_trading_day_weekday_returns_same(self) -> None:
        et = ZoneInfo("America/New_York")
        mon = datetime(2026, 4, 13, 10, 0, tzinfo=et)
        assert _most_recent_trading_day(mon) == date(2026, 4, 13)

    def test_most_recent_trading_day_saturday_returns_friday(self) -> None:
        et = ZoneInfo("America/New_York")
        sat = datetime(2026, 4, 11, 12, 0, tzinfo=et)
        assert _most_recent_trading_day(sat) == date(2026, 4, 10)

    def test_most_recent_trading_day_sunday_returns_friday(self) -> None:
        et = ZoneInfo("America/New_York")
        sun = datetime(2026, 4, 12, 12, 0, tzinfo=et)
        assert _most_recent_trading_day(sun) == date(2026, 4, 10)

    def test_previous_weekday_monday_returns_friday(self) -> None:
        assert _previous_weekday(date(2026, 4, 13)) == date(2026, 4, 10)

    def test_previous_weekday_tuesday_returns_monday(self) -> None:
        assert _previous_weekday(date(2026, 4, 14)) == date(2026, 4, 13)


# ----------------------------------------------------------------------
# Warmup + persistence
# ----------------------------------------------------------------------


class TestWarmupAndPersistence:

    @pytest.mark.asyncio
    async def test_warmup_loads_per_symbol_history(self, repos) -> None:
        for i in range(5):
            await repos.regsho_cache.upsert(
                "TSLA-USDC",
                date(2026, 4, 1) + timedelta(days=i),
                short_volume=1_000_000 + i,
                total_volume=3_000_000,
                short_volume_ratio=(1_000_000 + i) / 3_000_000,
            )
        for i in range(3):
            await repos.regsho_cache.upsert(
                "NVDA-USDC",
                date(2026, 4, 1) + timedelta(days=i),
                short_volume=1_500_000,
                total_volume=3_000_000,
                short_volume_ratio=0.5,
            )
        provider = EquityFlowProvider(regsho_repo=repos.regsho_cache)
        loaded = await provider.warmup_from_repo()
        assert loaded == 8
        assert len(provider._history["TSLA-USDC"]) == 5
        assert len(provider._history["NVDA-USDC"]) == 3

    @pytest.mark.asyncio
    async def test_warmup_no_repo_is_no_op(self) -> None:
        provider = EquityFlowProvider()
        assert await provider.warmup_from_repo() == 0

    @pytest.mark.asyncio
    async def test_finra_failure_falls_back_to_cache(self, repos) -> None:
        """Every FINRA URL 404 → provider falls back to the cached
        20-day history and still serves SVR + market_open."""
        for i in range(5):
            await repos.regsho_cache.upsert(
                "TSLA-USDC",
                date(2026, 3, 1) + timedelta(days=i),  # old date
                short_volume=1_000_000 + i,
                total_volume=3_000_000,
                short_volume_ratio=(1_000_000 + i) / 3_000_000,
            )
        transport = _mock_transport()  # every URL → 404
        client = httpx.AsyncClient(transport=transport)
        provider = EquityFlowProvider(
            regsho_repo=repos.regsho_cache, http_client=client
        )
        await provider.warmup_from_repo()

        result = await provider.fetch("TSLA-USDC", adapter=None)
        # Still returns cached SVR + market_open even though FINRA down
        assert "short_volume_ratio" in result
        assert "market_open" in result
        await client.aclose()

    @pytest.mark.asyncio
    async def test_refresh_persists_new_rows(self, repos) -> None:
        """A successful FINRA pull writes the parsed rows back to the
        regsho_cache table so subsequent restarts pick them up."""
        trade_day = _most_recent_trading_day()
        suffix = f"CNMSshvol{trade_day.strftime('%Y%m%d')}.txt"
        transport = _mock_transport(
            responses={suffix: (200, _sample_regsho_text(trade_date=trade_day))}
        )
        client = httpx.AsyncClient(transport=transport)
        provider = EquityFlowProvider(
            regsho_repo=repos.regsho_cache, http_client=client
        )

        await provider.fetch("TSLA-USDC", adapter=None)

        tsla = await repos.regsho_cache.get_recent("TSLA-USDC")
        nvda = await repos.regsho_cache.get_recent("NVDA-USDC")
        googl = await repos.regsho_cache.get_recent("GOOGL-USDC")
        assert len(tsla) == 1
        assert len(nvda) == 1
        assert len(googl) == 1
        assert tsla[0]["short_volume"] == 31_503_665
        await client.aclose()

    @pytest.mark.asyncio
    async def test_refresh_walks_backwards_on_404(self, repos) -> None:
        """Today 404 + yesterday 200 → provider lands on yesterday."""
        today = _most_recent_trading_day()
        yesterday = _previous_weekday(today)
        yesterday_suffix = f"CNMSshvol{yesterday.strftime('%Y%m%d')}.txt"
        transport = _mock_transport(
            responses={
                yesterday_suffix: (
                    200,
                    _sample_regsho_text(trade_date=yesterday),
                )
            }
        )
        client = httpx.AsyncClient(transport=transport)
        provider = EquityFlowProvider(
            regsho_repo=repos.regsho_cache, http_client=client
        )
        await provider.fetch("TSLA-USDC", adapter=None)

        rows = await repos.regsho_cache.get_recent("TSLA-USDC")
        assert len(rows) == 1
        assert rows[0]["trade_date"] == yesterday
        await client.aclose()

    @pytest.mark.asyncio
    async def test_refresh_deduped_within_same_day(self, repos) -> None:
        """Multiple fetches in the same day cause ONE HTTP pull."""
        trade_day = _most_recent_trading_day()
        suffix = f"CNMSshvol{trade_day.strftime('%Y%m%d')}.txt"
        hits = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            hits["n"] += 1
            if request.url.path.endswith(suffix):
                return httpx.Response(
                    200, text=_sample_regsho_text(trade_date=trade_day)
                )
            return httpx.Response(404, text="")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = EquityFlowProvider(
            regsho_repo=repos.regsho_cache, http_client=client
        )
        await provider.fetch("TSLA-USDC", adapter=None)
        await provider.fetch("NVDA-USDC", adapter=None)
        await provider.fetch("GOOGL-USDC", adapter=None)
        assert hits["n"] == 1  # Only ONE fetch across all three bots
        await client.aclose()
