"""Unit tests for ParquetDataLoader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from backtesting.data_loader import ParquetDataLoader, _months_in_range


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _hourly_candles(start: datetime, hours: int, base: float = 100.0) -> pl.DataFrame:
    """Build a Polars DataFrame of `hours` 1h candles starting at `start`."""
    rows = []
    start_ms = _ms(start)
    for i in range(hours):
        ts = start_ms + i * 3600 * 1000
        rows.append(
            {
                "timestamp": ts,
                "open": float(base + i),
                "high": float(base + i + 1),
                "low": float(base + i - 1),
                "close": float(base + i + 0.5),
                "volume": 10.0 + i,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _write(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


# ---------------------------------------------------------------------------
# Helper test
# ---------------------------------------------------------------------------


def test_months_in_range_single_month():
    months = _months_in_range(
        datetime(2026, 3, 5, tzinfo=timezone.utc),
        datetime(2026, 3, 25, tzinfo=timezone.utc),
    )
    assert months == [datetime(2026, 3, 1).date()]


def test_months_in_range_cross_month():
    months = _months_in_range(
        datetime(2026, 3, 28, tzinfo=timezone.utc),
        datetime(2026, 4, 5, tzinfo=timezone.utc),
    )
    assert months == [datetime(2026, 3, 1).date(), datetime(2026, 4, 1).date()]


def test_months_in_range_three_months():
    months = _months_in_range(
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        datetime(2026, 3, 10, tzinfo=timezone.utc),
    )
    assert months == [
        datetime(2026, 1, 1).date(),
        datetime(2026, 2, 1).date(),
        datetime(2026, 3, 1).date(),
    ]


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


EXCHANGE = "hyperliquid"


@pytest.fixture
def populated_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "parquet"
    march = _hourly_candles(datetime(2026, 3, 1, tzinfo=timezone.utc), hours=24 * 31)
    april = _hourly_candles(
        datetime(2026, 4, 1, tzinfo=timezone.utc), hours=24 * 30, base=200.0
    )
    _write(data_dir / EXCHANGE / "BTC-USDC" / "1h_2026-03.parquet", march)
    _write(data_dir / EXCHANGE / "BTC-USDC" / "1h_2026-04.parquet", april)
    return data_dir


def test_load_single_month(populated_dir):
    loader = ParquetDataLoader(populated_dir)
    df = loader.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 5, tzinfo=timezone.utc),
        datetime(2026, 3, 10, tzinfo=timezone.utc),
    )
    assert df.height == 24 * 5  # 5 days
    assert df["timestamp"].min() >= _ms(datetime(2026, 3, 5, tzinfo=timezone.utc))
    assert df["timestamp"].max() < _ms(datetime(2026, 3, 10, tzinfo=timezone.utc))
    # Sorted ascending
    ts = df["timestamp"].to_list()
    assert ts == sorted(ts)


def test_load_cross_month_stitching(populated_dir):
    loader = ParquetDataLoader(populated_dir)
    df = loader.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 3, tzinfo=timezone.utc),
    )
    # 2 days from March (30, 31) + 2 days from April (1, 2) = 4 days
    assert df.height == 24 * 4
    # Sequence is contiguous
    ts = df["timestamp"].to_list()
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    assert all(d == 3600 * 1000 for d in diffs)


def test_load_missing_file_raises(tmp_path):
    loader = ParquetDataLoader(tmp_path / "parquet", exchange="hyperliquid")
    with pytest.raises(
        FileNotFoundError, match="No data for hyperliquid/BTC-USDC 2026-03"
    ):
        loader.load(
            "BTC-USDC",
            "1h",
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 5, tzinfo=timezone.utc),
        )


def test_load_invalid_timeframe(populated_dir):
    loader = ParquetDataLoader(populated_dir)
    with pytest.raises(ValueError, match="Unknown timeframe"):
        loader.load(
            "BTC-USDC",
            "7m",
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 2, tzinfo=timezone.utc),
        )


def test_load_invalid_date_range(populated_dir):
    loader = ParquetDataLoader(populated_dir)
    with pytest.raises(ValueError, match="must be after"):
        loader.load(
            "BTC-USDC",
            "1h",
            datetime(2026, 3, 5, tzinfo=timezone.utc),
            datetime(2026, 3, 5, tzinfo=timezone.utc),
        )


def test_load_naive_datetime_assumed_utc(populated_dir):
    loader = ParquetDataLoader(populated_dir)
    df = loader.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 5),  # naive
        datetime(2026, 3, 6),  # naive
    )
    assert df.height == 24


def test_load_as_market_data_format(populated_dir):
    loader = ParquetDataLoader(populated_dir)
    candles = loader.load_as_market_data(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 5, tzinfo=timezone.utc),
        datetime(2026, 3, 6, tzinfo=timezone.utc),
    )
    assert isinstance(candles, list)
    assert len(candles) == 24
    first = candles[0]
    # Same keys + types as ExchangeAdapter.fetch_ohlcv output
    assert set(first.keys()) == {"timestamp", "open", "high", "low", "close", "volume"}
    assert isinstance(first["timestamp"], int)
    assert isinstance(first["open"], float)
    assert isinstance(first["volume"], float)


def test_load_as_market_data_pipeline_compatible(populated_dir):
    """Output must be consumable by indicator/swing pipeline code."""
    from engine.data.indicators import compute_all_indicators

    loader = ParquetDataLoader(populated_dir)
    candles = loader.load_as_market_data(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 31, tzinfo=timezone.utc),
    )
    # The same compute_all_indicators that runs in the live pipeline
    indicators = compute_all_indicators(candles)
    assert isinstance(indicators, dict)
    # Should have computed at least the standard indicators
    assert len(indicators) > 0


def test_load_isolates_data_by_exchange(tmp_path):
    """Same symbol on two venues = two independent datasets."""
    data_dir = tmp_path / "parquet"
    a = _hourly_candles(datetime(2026, 3, 1, tzinfo=timezone.utc), hours=24, base=100.0)
    b = _hourly_candles(datetime(2026, 3, 1, tzinfo=timezone.utc), hours=24, base=999.0)
    _write(data_dir / "venue_a" / "BTC-USDC" / "1h_2026-03.parquet", a)
    _write(data_dir / "venue_b" / "BTC-USDC" / "1h_2026-03.parquet", b)

    loader_a = ParquetDataLoader(data_dir, exchange="venue_a")
    loader_b = ParquetDataLoader(data_dir, exchange="venue_b")

    df_a = loader_a.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 2, tzinfo=timezone.utc),
    )
    df_b = loader_b.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 2, tzinfo=timezone.utc),
    )

    # Different price series
    assert df_a["open"][0] == 100.0
    assert df_b["open"][0] == 999.0

    # venue_a's loader cannot see venue_b's data, even via the wrong path
    with pytest.raises(FileNotFoundError):
        ParquetDataLoader(data_dir, exchange="venue_a").load(
            "ETH-USDC",  # only exists nowhere
            "1h",
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 2, tzinfo=timezone.utc),
        )


def test_load_default_exchange_is_hyperliquid(tmp_path):
    """Omitting `exchange` falls back to 'hyperliquid'."""
    data_dir = tmp_path / "parquet"
    df = _hourly_candles(datetime(2026, 3, 1, tzinfo=timezone.utc), hours=24)
    _write(data_dir / "hyperliquid" / "BTC-USDC" / "1h_2026-03.parquet", df)

    loader = ParquetDataLoader(data_dir)  # no exchange arg
    out = loader.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 2, tzinfo=timezone.utc),
    )
    assert out.height == 24


def test_load_dedupes_overlapping_timestamps(tmp_path):
    """If two months somehow share a timestamp at the boundary, dedup."""
    data_dir = tmp_path / "parquet"
    feb = _hourly_candles(datetime(2026, 2, 1, tzinfo=timezone.utc), hours=24)
    # Create a "march" file that includes the same first feb candle
    march = _hourly_candles(datetime(2026, 2, 1, tzinfo=timezone.utc), hours=48)
    _write(data_dir / EXCHANGE / "BTC-USDC" / "1h_2026-02.parquet", feb)
    _write(data_dir / EXCHANGE / "BTC-USDC" / "1h_2026-03.parquet", march)

    loader = ParquetDataLoader(data_dir, exchange=EXCHANGE)
    df = loader.load(
        "BTC-USDC",
        "1h",
        datetime(2026, 2, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 3, tzinfo=timezone.utc),
    )
    # 48 hours total, no duplicates
    ts = df["timestamp"].to_list()
    assert len(ts) == len(set(ts))
