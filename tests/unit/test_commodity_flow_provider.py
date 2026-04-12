"""Unit tests for CommodityFlowProvider.

The provider wraps ``cot_reports`` (network-bound) and runs rolling
percentile / divergence math on a 52-week in-memory buffer. Tests
never hit the real CFTC endpoint — they either:

- call ``_extract_latest_row`` / ``_compute_metrics`` directly on
  hand-built dicts and DataFrames, OR
- monkeypatch ``_fetch_latest_rows`` so the refresh path can be
  exercised end-to-end without network I/O.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
import pytest_asyncio

from engine.data.flow.commodity import (
    COMMODITY_SYMBOLS,
    CommodityFlowProvider,
    _percentile_of_score,
)
from storage.repositories.sqlite import SQLiteRepositories


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest_asyncio.fixture
async def repos(tmp_path):
    db_path = str(tmp_path / "test_cot.db")
    r = SQLiteRepositories(db_path=db_path)
    await r.init_db()
    return r


def _history(n_weeks: int, mm_net_values: list[float]) -> list[dict]:
    """Build a synthetic 52-week history with explicit MM net values.

    Commercial net is mirrored so the divergence is 0 by default;
    individual tests override when they want divergence > 0 or < 0.
    """
    assert len(mm_net_values) == n_weeks
    base = date(2026, 1, 1)
    out = []
    for i in range(n_weeks):
        out.append(
            {
                "report_date": base + timedelta(weeks=i),
                "managed_money_net": mm_net_values[i],
                "commercial_net": -mm_net_values[i],
                "total_oi": 1_000_000.0,
            }
        )
    return out


# ----------------------------------------------------------------------
# Symbol gating
# ----------------------------------------------------------------------


class TestSymbolGating:

    @pytest.mark.asyncio
    async def test_crypto_symbol_returns_empty(self) -> None:
        provider = CommodityFlowProvider()
        # Stuff the history for GOLD so we DEFINITELY have data —
        # BTC-USDC must still get {} regardless.
        provider._history["GOLD-USDC"] = _history(5, [1, 2, 3, 4, 5])
        result = await provider.fetch("BTC-USDC", adapter=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_unknown_symbol_returns_empty(self) -> None:
        provider = CommodityFlowProvider()
        result = await provider.fetch("SOL-USDC", adapter=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty(self) -> None:
        """Commodity symbol without 2+ weeks of data returns {}."""
        provider = CommodityFlowProvider()
        # Stub the refresh path so we don't hit the network in this test
        async def _noop_refresh():
            return
        provider._maybe_refresh = _noop_refresh  # type: ignore
        result = await provider.fetch("GOLD-USDC", adapter=None)
        assert result == {}


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------


class TestMetricComputation:

    def test_managed_money_net_matches_latest(self) -> None:
        rows = _history(4, [100, 150, 200, 250])
        out = CommodityFlowProvider._compute_metrics(rows)
        assert out["cot_managed_money_net"] == 250.0

    def test_weekly_change_percent(self) -> None:
        rows = _history(2, [200.0, 250.0])
        out = CommodityFlowProvider._compute_metrics(rows)
        # (250 - 200) / |200| * 100 = 25%
        assert out["cot_weekly_change_pct"] == pytest.approx(25.0)

    def test_weekly_change_negative(self) -> None:
        rows = _history(2, [200.0, 160.0])
        out = CommodityFlowProvider._compute_metrics(rows)
        assert out["cot_weekly_change_pct"] == pytest.approx(-20.0)

    def test_weekly_change_zero_anchor_omitted(self) -> None:
        rows = _history(2, [0.0, 100.0])
        out = CommodityFlowProvider._compute_metrics(rows)
        assert "cot_weekly_change_pct" not in out

    def test_speculator_percentile_is_100_on_fresh_max(self) -> None:
        rows = _history(10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        out = CommodityFlowProvider._compute_metrics(rows)
        assert out["cot_speculator_percentile"] == pytest.approx(100.0)

    def test_speculator_percentile_is_low_on_fresh_min(self) -> None:
        rows = _history(10, [100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        out = CommodityFlowProvider._compute_metrics(rows)
        # 10 is the minimum: only 1 value (itself) is ≤ 10 → 1/10 = 10%
        assert out["cot_speculator_percentile"] == pytest.approx(10.0)

    def test_divergence_computed_from_latest(self) -> None:
        rows = _history(3, [0, 0, 100])
        # History helper mirrors commercial_net = -mm_net
        # Latest: comm = -100, mm = 100 → divergence = -200
        out = CommodityFlowProvider._compute_metrics(rows)
        assert out["cot_divergence"] == pytest.approx(-200.0)

    def test_divergence_abs_percentile(self) -> None:
        # Construct rows with fixed divergence pattern.
        rows = []
        base = date(2026, 1, 1)
        for i, (mm, comm) in enumerate(
            [
                (100, 50),   # |div| = 50
                (100, 60),   # |div| = 40
                (100, 80),   # |div| = 20
                (100, 90),   # |div| = 10
                (100, 300),  # |div| = 200 ← extreme, newest
            ]
        ):
            rows.append(
                {
                    "report_date": base + timedelta(weeks=i),
                    "managed_money_net": float(mm),
                    "commercial_net": float(comm),
                    "total_oi": 0.0,
                }
            )
        out = CommodityFlowProvider._compute_metrics(rows)
        # Latest divergence = 300 - 100 = 200; abs = 200 → max of
        # [50, 40, 20, 10, 200] → 5/5 = 100th percentile
        assert out["cot_divergence"] == pytest.approx(200.0)
        assert out["cot_divergence_abs_percentile"] == pytest.approx(100.0)


# ----------------------------------------------------------------------
# Percentile helper
# ----------------------------------------------------------------------


class TestPercentileHelper:

    def test_middle_value(self) -> None:
        assert _percentile_of_score([1, 2, 3, 4, 5], 3) == pytest.approx(60.0)

    def test_max_is_100(self) -> None:
        assert _percentile_of_score([1, 2, 3], 3) == pytest.approx(100.0)

    def test_min_is_one_over_n(self) -> None:
        # weak/≤ rank → the minimum ties with itself
        assert _percentile_of_score([1, 2, 3], 1) == pytest.approx(100 / 3)

    def test_empty_returns_zero(self) -> None:
        assert _percentile_of_score([], 5) == 0.0


# ----------------------------------------------------------------------
# Warmup + persistence
# ----------------------------------------------------------------------


class TestWarmupAndPersistence:

    @pytest.mark.asyncio
    async def test_warmup_loads_per_symbol_history(self, repos) -> None:
        # Seed the cot_cache table directly
        for i in range(5):
            await repos.cot_cache.upsert(
                "GOLD-USDC",
                date(2026, 1, 1) + timedelta(weeks=i),
                managed_money_net=float(100 + i),
                commercial_net=float(-100 - i),
                total_oi=1_000_000.0,
            )
        for i in range(3):
            await repos.cot_cache.upsert(
                "SILVER-USDC",
                date(2026, 1, 1) + timedelta(weeks=i),
                managed_money_net=float(50 + i),
                commercial_net=float(-50 - i),
                total_oi=500_000.0,
            )

        provider = CommodityFlowProvider(cot_repo=repos.cot_cache)
        loaded = await provider.warmup_from_repo()
        assert loaded == 8
        assert len(provider._history["GOLD-USDC"]) == 5
        assert len(provider._history["SILVER-USDC"]) == 3
        # Ascending by date
        dates = [r["report_date"] for r in provider._history["GOLD-USDC"]]
        assert dates == sorted(dates)

    @pytest.mark.asyncio
    async def test_warmup_no_repo_is_no_op(self) -> None:
        provider = CommodityFlowProvider()
        assert await provider.warmup_from_repo() == 0

    @pytest.mark.asyncio
    async def test_fetch_after_warmup_serves_from_memory(self, repos) -> None:
        for i in range(10):
            await repos.cot_cache.upsert(
                "GOLD-USDC",
                date.today() - timedelta(weeks=9 - i),
                managed_money_net=float(i * 10),
                commercial_net=float(-i * 10),
                total_oi=1_000_000.0,
            )
        provider = CommodityFlowProvider(cot_repo=repos.cot_cache)
        await provider.warmup_from_repo()

        # Stub the refresh path so this test does NOT hit the network
        async def _noop_refresh():
            return
        provider._maybe_refresh = _noop_refresh  # type: ignore

        result = await provider.fetch("GOLD-USDC", adapter=None)
        assert result["cot_managed_money_net"] == 90.0
        # Fresh all-time high for MM net → 100th percentile
        assert result["cot_speculator_percentile"] == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_cot_library_failure_falls_back_to_cache(self, repos) -> None:
        """When ``_fetch_latest_rows`` raises, the provider logs and
        continues serving cached history instead of crashing."""
        # Seed with one-week-old data so refresh IS needed
        for i in range(5):
            await repos.cot_cache.upsert(
                "GOLD-USDC",
                date.today() - timedelta(weeks=10 - i),
                managed_money_net=float(i * 10),
                commercial_net=float(-i * 10),
                total_oi=1_000_000.0,
            )
        provider = CommodityFlowProvider(cot_repo=repos.cot_cache)
        await provider.warmup_from_repo()

        # Force the live pull to fail
        def _boom():
            raise ConnectionError("CFTC down")
        provider._fetch_latest_rows = _boom  # type: ignore

        # Must not raise; must return cached data
        result = await provider.fetch("GOLD-USDC", adapter=None)
        assert "cot_managed_money_net" in result

    @pytest.mark.asyncio
    async def test_refresh_persists_new_rows_to_repo(self, repos) -> None:
        """A successful refresh writes the newest row back to cot_cache."""
        provider = CommodityFlowProvider(cot_repo=repos.cot_cache)

        today = date.today()
        synthetic_rows = {
            "GOLD-USDC": {
                "report_date": today,
                "managed_money_net": 12345.0,
                "commercial_net": -6789.0,
                "total_oi": 500_000.0,
            },
            "SILVER-USDC": {
                "report_date": today,
                "managed_money_net": 11111.0,
                "commercial_net": -22222.0,
                "total_oi": 250_000.0,
            },
        }

        def _fake_fetch():
            return synthetic_rows
        provider._fetch_latest_rows = _fake_fetch  # type: ignore

        # First call: fetches (because history is empty → stale)
        await provider.fetch("GOLD-USDC", adapter=None)

        gold_rows = await repos.cot_cache.get_recent("GOLD-USDC")
        silver_rows = await repos.cot_cache.get_recent("SILVER-USDC")
        assert len(gold_rows) == 1
        assert gold_rows[0]["managed_money_net"] == 12345.0
        assert len(silver_rows) == 1

    @pytest.mark.asyncio
    async def test_refresh_skipped_when_history_fresh(self, repos) -> None:
        """If every symbol's newest row is within 7 days, no live pull."""
        for sym in COMMODITY_SYMBOLS:
            await repos.cot_cache.upsert(
                sym, date.today() - timedelta(days=2),
                managed_money_net=1.0, commercial_net=1.0, total_oi=1.0,
            )
        provider = CommodityFlowProvider(cot_repo=repos.cot_cache)
        await provider.warmup_from_repo()

        calls = {"n": 0}

        def _counting_fetch():
            calls["n"] += 1
            return {}
        provider._fetch_latest_rows = _counting_fetch  # type: ignore

        # Triggering fetch on any symbol must NOT call the live fetcher
        # because every symbol is within the 7-day freshness window.
        await provider.fetch("GOLD-USDC", adapter=None)
        assert calls["n"] == 0


# ----------------------------------------------------------------------
# Extract-latest-row (DataFrame parsing)
# ----------------------------------------------------------------------


class TestExtractLatestRow:

    def test_extract_from_dataframe_matches_pattern(self) -> None:
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
                    "Report_Date_as_YYYY-MM-DD": "2026-04-01",
                    "M_Money_Positions_Long_All": 200_000,
                    "M_Money_Positions_Short_All": 50_000,
                    "Prod_Merc_Positions_Long_All": 10_000,
                    "Prod_Merc_Positions_Short_All": 120_000,
                    "Open_Interest_All": 500_000,
                }
            ]
        )
        row = CommodityFlowProvider._extract_latest_row(
            df, ["GOLD - COMMODITY EXCHANGE"]
        )
        assert row is not None
        assert row["report_date"] == date(2026, 4, 1)
        # MM net = 200k − 50k = 150k
        assert row["managed_money_net"] == pytest.approx(150_000.0)
        # Commercial net = 10k − 120k = -110k
        assert row["commercial_net"] == pytest.approx(-110_000.0)
        assert row["total_oi"] == pytest.approx(500_000.0)

    def test_extract_no_match_returns_none(self) -> None:
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "Market_and_Exchange_Names": "CORN - CBOT",
                    "Report_Date_as_YYYY-MM-DD": "2026-04-01",
                    "M_Money_Positions_Long_All": 1,
                    "M_Money_Positions_Short_All": 1,
                    "Prod_Merc_Positions_Long_All": 1,
                    "Prod_Merc_Positions_Short_All": 1,
                    "Open_Interest_All": 1,
                }
            ]
        )
        row = CommodityFlowProvider._extract_latest_row(df, ["GOLD"])
        assert row is None

    def test_extract_picks_latest_report_date_across_weeks(self) -> None:
        """``cot_year`` returns one row per (contract, weekly report).
        The extractor must return the MOST RECENT week's row, not the
        week-in-the-year that happens to have the highest OI.
        Regression guard for a bug where sort-by-OI ran across all
        weeks and returned a months-old snapshot.
        """
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
                    "Report_Date_as_YYYY-MM-DD": "2026-01-20",
                    "M_Money_Positions_Long_All": 200_000,
                    "M_Money_Positions_Short_All": 50_000,
                    "Prod_Merc_Positions_Long_All": 10_000,
                    "Prod_Merc_Positions_Short_All": 120_000,
                    "Open_Interest_All": 900_000,  # HIGHEST OI
                },
                {
                    "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
                    "Report_Date_as_YYYY-MM-DD": "2026-04-07",  # newest
                    "M_Money_Positions_Long_All": 100_000,
                    "M_Money_Positions_Short_All": 40_000,
                    "Prod_Merc_Positions_Long_All": 20_000,
                    "Prod_Merc_Positions_Short_All": 60_000,
                    "Open_Interest_All": 500_000,
                },
            ]
        )
        row = CommodityFlowProvider._extract_latest_row(
            df, ["GOLD - COMMODITY EXCHANGE"]
        )
        assert row is not None
        assert row["report_date"] == date(2026, 4, 7)
        # 100k − 40k = 60k, not 200k − 50k = 150k
        assert row["managed_money_net"] == pytest.approx(60_000.0)

    def test_extract_prefers_highest_oi_among_same_date(self) -> None:
        """When multiple contracts match a pattern on the SAME latest
        report date (e.g. regular + mini), the one with the largest
        Open Interest wins — that's the canonical listing."""
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "Market_and_Exchange_Names": "MICRO GOLD - COMMODITY EXCHANGE INC.",
                    "Report_Date_as_YYYY-MM-DD": "2026-04-01",
                    "M_Money_Positions_Long_All": 10,
                    "M_Money_Positions_Short_All": 5,
                    "Prod_Merc_Positions_Long_All": 1,
                    "Prod_Merc_Positions_Short_All": 1,
                    "Open_Interest_All": 1_000,
                },
                {
                    "Market_and_Exchange_Names": "GOLD - COMMODITY EXCHANGE INC.",
                    "Report_Date_as_YYYY-MM-DD": "2026-04-01",
                    "M_Money_Positions_Long_All": 200_000,
                    "M_Money_Positions_Short_All": 50_000,
                    "Prod_Merc_Positions_Long_All": 10_000,
                    "Prod_Merc_Positions_Short_All": 120_000,
                    "Open_Interest_All": 500_000,
                },
            ]
        )
        row = CommodityFlowProvider._extract_latest_row(
            df, ["GOLD - COMMODITY EXCHANGE"]
        )
        assert row is not None
        # Big contract wins — 150k MM net, not 5
        assert row["managed_money_net"] == pytest.approx(150_000.0)
