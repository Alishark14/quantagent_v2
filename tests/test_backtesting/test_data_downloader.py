"""Unit tests for HistoricalDataDownloader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from backtesting.data_downloader import (
    HistoricalDataDownloader,
    _enumerate_months,
    _next_month_start_ms,
)
from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _candle(ts_ms: int, price: float = 100.0) -> dict:
    return {
        "timestamp": ts_ms,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price + 0.5,
        "volume": 10.0,
    }


class MockAdapter(ExchangeAdapter):
    """Returns synthetic 1h candles for any (symbol, timeframe, since)."""

    PERIOD_MS = 3600 * 1000  # 1h

    def __init__(self, period_ms: int = PERIOD_MS) -> None:
        self.period_ms = period_ms
        self.calls: list[tuple[str, str, int, int | None]] = []

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=False, supports_short=True, market_hours=None,
            asset_types=["crypto"], margin_type="cross", has_funding_rate=False,
            has_oi_data=False, max_leverage=10.0, order_types=["market"],
            supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
        self.calls.append((symbol, timeframe, limit, since))
        # Generate `limit` candles starting at `since` aligned to period
        start = since if since is not None else 0
        # Snap to period boundary
        start = (start // self.period_ms) * self.period_ms
        return [
            _candle(start + i * self.period_ms, 100.0 + i)
            for i in range(limit)
        ]

    async def get_ticker(self, symbol):
        return {}

    async def get_balance(self):
        return 0.0

    async def get_positions(self, symbol=None):
        return []

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def place_sl_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def cancel_order(self, symbol, order_id):
        return False

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def modify_sl(self, symbol, new_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")


class EmptyAdapter(MockAdapter):
    """Returns no candles."""

    async def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
        self.calls.append((symbol, timeframe, limit, since))
        return []


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_enumerate_months_basic():
    now = datetime(2026, 4, 7, tzinfo=timezone.utc)
    months = _enumerate_months(now, months_back=3)
    assert months == [
        datetime(2026, 2, 1).date(),
        datetime(2026, 3, 1).date(),
        datetime(2026, 4, 1).date(),
    ]


def test_enumerate_months_year_boundary():
    now = datetime(2026, 2, 15, tzinfo=timezone.utc)
    months = _enumerate_months(now, months_back=4)
    assert months[0] == datetime(2025, 11, 1).date()
    assert months[-1] == datetime(2026, 2, 1).date()


def test_next_month_start_ms_normal():
    ms = _next_month_start_ms(datetime(2026, 3, 1).date())
    assert ms == _ms(datetime(2026, 4, 1, tzinfo=timezone.utc))


def test_next_month_start_ms_december():
    ms = _next_month_start_ms(datetime(2026, 12, 1).date())
    assert ms == _ms(datetime(2027, 1, 1, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Downloader tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "parquet"


@pytest.mark.asyncio
async def test_download_writes_parquet(tmp_data_dir):
    adapter = MockAdapter()
    dl = HistoricalDataDownloader(
        adapter=adapter,
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)

    stats = await dl.download(
        symbols=["BTC-USDC"],
        timeframes=["1h"],
        months_back=2,  # March + April
        now=now,
    )

    assert stats.files_written == 2
    assert stats.files_skipped == 0
    assert stats.candles_total > 0

    march = tmp_data_dir / "mockex" / "BTC-USDC" / "1h_2026-03.parquet"
    april = tmp_data_dir / "mockex" / "BTC-USDC" / "1h_2026-04.parquet"
    assert march.exists()
    assert april.exists()

    df = pl.read_parquet(march)
    assert df.columns == ["timestamp", "open", "high", "low", "close", "volume"]
    # March 2026 has 31 days × 24 hours = 744 candles
    assert df.height == 744
    # All timestamps fall inside March
    march_start = _ms(datetime(2026, 3, 1, tzinfo=timezone.utc))
    april_start = _ms(datetime(2026, 4, 1, tzinfo=timezone.utc))
    assert df["timestamp"].min() >= march_start
    assert df["timestamp"].max() < april_start


@pytest.mark.asyncio
async def test_download_resume_skips_existing(tmp_data_dir):
    adapter = MockAdapter()
    dl = HistoricalDataDownloader(
        adapter=adapter,
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)

    # First run writes both months
    stats1 = await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=2, now=now
    )
    assert stats1.files_written == 2

    calls_after_first = len(adapter.calls)

    # Second run skips both
    stats2 = await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=2, now=now
    )
    assert stats2.files_written == 0
    assert stats2.files_skipped == 2
    # No new fetches
    assert len(adapter.calls) == calls_after_first


@pytest.mark.asyncio
async def test_download_force_overwrites(tmp_data_dir):
    adapter = MockAdapter()
    dl = HistoricalDataDownloader(
        adapter=adapter,
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)

    await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=1, now=now
    )

    stats = await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=1, now=now, force=True
    )
    assert stats.files_written == 1
    assert stats.files_skipped == 0


@pytest.mark.asyncio
async def test_download_empty_response_no_file(tmp_data_dir):
    adapter = EmptyAdapter()
    dl = HistoricalDataDownloader(
        adapter=adapter,
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)

    stats = await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=1, now=now
    )

    assert stats.files_written == 0
    assert not (tmp_data_dir / "mockex" / "BTC-USDC" / "1h_2026-04.parquet").exists()


@pytest.mark.asyncio
async def test_download_invalid_timeframe_raises(tmp_data_dir):
    dl = HistoricalDataDownloader(
        adapter=MockAdapter(),
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    with pytest.raises(ValueError, match="Unknown timeframe"):
        await dl.download(
            symbols=["BTC-USDC"], timeframes=["7m"], months_back=1
        )


@pytest.mark.asyncio
async def test_download_invalid_months_back_raises(tmp_data_dir):
    dl = HistoricalDataDownloader(
        adapter=MockAdapter(),
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    with pytest.raises(ValueError, match="months_back"):
        await dl.download(
            symbols=["BTC-USDC"], timeframes=["1h"], months_back=0
        )


@pytest.mark.asyncio
async def test_download_handles_adapter_error_recorded(tmp_data_dir):
    class BrokenAdapter(MockAdapter):
        async def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
            raise RuntimeError("api down")

    dl = HistoricalDataDownloader(
        adapter=BrokenAdapter(),
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)
    stats = await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=1, now=now
    )
    assert stats.files_written == 0
    assert len(stats.errors) == 1
    assert "api down" in stats.errors[0]


@pytest.mark.asyncio
async def test_download_multi_symbol_multi_timeframe(tmp_data_dir):
    adapter = MockAdapter()
    dl = HistoricalDataDownloader(
        adapter=adapter,
        exchange_name="mockex",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)

    stats = await dl.download(
        symbols=["BTC-USDC", "ETH-USDC"],
        timeframes=["1h", "4h"],
        months_back=1,
        now=now,
    )
    # 2 symbols × 2 timeframes × 1 month = 4 files
    assert stats.files_written == 4
    for symbol in ["BTC-USDC", "ETH-USDC"]:
        for tf in ["1h", "4h"]:
            assert (
                tmp_data_dir / "mockex" / symbol / f"{tf}_2026-04.parquet"
            ).exists()


@pytest.mark.asyncio
async def test_download_defaults_exchange_name_from_adapter(tmp_data_dir):
    """When `exchange_name` is omitted, use adapter.name() as the dir."""
    adapter = MockAdapter()  # adapter.name() == "mock"
    dl = HistoricalDataDownloader(
        adapter=adapter,
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)
    await dl.download(
        symbols=["BTC-USDC"], timeframes=["1h"], months_back=1, now=now
    )
    assert (tmp_data_dir / "mock" / "BTC-USDC" / "1h_2026-04.parquet").exists()


@pytest.mark.asyncio
async def test_download_same_symbol_different_exchanges_isolated(tmp_data_dir):
    """BTC-USDC on two venues writes to two different paths."""
    now = datetime(2026, 4, 15, tzinfo=timezone.utc)

    dl_a = HistoricalDataDownloader(
        adapter=MockAdapter(),
        exchange_name="venue_a",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    dl_b = HistoricalDataDownloader(
        adapter=MockAdapter(),
        exchange_name="venue_b",
        data_dir=tmp_data_dir,
        rate_limit_sleep=0.0,
    )
    await dl_a.download(["BTC-USDC"], ["1h"], months_back=1, now=now)
    await dl_b.download(["BTC-USDC"], ["1h"], months_back=1, now=now)

    a_path = tmp_data_dir / "venue_a" / "BTC-USDC" / "1h_2026-04.parquet"
    b_path = tmp_data_dir / "venue_b" / "BTC-USDC" / "1h_2026-04.parquet"
    assert a_path.exists()
    assert b_path.exists()
    assert a_path != b_path


def test_downloader_module_has_no_exchange_specific_imports():
    """Adapter-agnostic guard: downloader must not import any concrete adapter."""
    import ast
    import backtesting.data_downloader as mod

    tree = ast.parse(open(mod.__file__).read())
    forbidden = {
        "exchanges.hyperliquid",
        "exchanges.binance",
        "exchanges.ibkr",
        "exchanges.alpaca",
        "exchanges.dydx",
        "exchanges.deribit",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, (
                    f"data_downloader.py imports {alias.name} "
                    "(must stay adapter-agnostic)"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden, (
                f"data_downloader.py imports from {node.module} "
                "(must stay adapter-agnostic)"
            )
