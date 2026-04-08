"""Unit tests for ForwardPathLoader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from backtesting.data_loader import ParquetDataLoader
from backtesting.forward_path import ForwardPathLoader


EXCHANGE = "hyperliquid"
MIN_MS = 60 * 1000
FIVE_MIN_MS = 5 * 60 * 1000


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _candles(start_ms: int, n: int, period_ms: int, base: float = 100.0) -> pl.DataFrame:
    rows = []
    for i in range(n):
        c = base + i * 0.1
        rows.append(
            {
                "timestamp": start_ms + i * period_ms,
                "open": c - 0.05,
                "high": c + 0.2,
                "low": c - 0.2,
                "close": c,
                "volume": 10.0,
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
# recommended_resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tf,expected",
    [
        ("15m", "1m"),
        ("30m", "1m"),
        ("1h", "1m"),
        ("4h", "5m"),
        ("1d", "5m"),
        ("unknown", "1m"),  # safe default = high fidelity
    ],
)
def test_recommended_resolution(tf, expected):
    assert ForwardPathLoader.recommended_resolution(tf) == expected


# ---------------------------------------------------------------------------
# load() — happy paths
# ---------------------------------------------------------------------------


@pytest.fixture
def loader_with_1m_data(tmp_path: Path) -> ForwardPathLoader:
    """Populate /BTC-USDC/1m_2026-03.parquet with 1 day of 1m candles."""
    data_dir = tmp_path / "parquet"
    start = _ms(datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc))
    df = _candles(start_ms=start, n=24 * 60, period_ms=MIN_MS)
    _write(data_dir / EXCHANGE / "BTC-USDC" / "1m_2026-03.parquet", df)
    return ForwardPathLoader(ParquetDataLoader(data_dir, exchange=EXCHANGE))


def test_load_returns_correct_starting_candle(loader_with_1m_data):
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))
    df = loader_with_1m_data.load("BTC-USDC", entry, duration_candles=10)
    assert df.height == 10
    assert df["timestamp"][0] == entry
    # Strictly increasing 1-minute spacing
    diffs = [df["timestamp"][i + 1] - df["timestamp"][i] for i in range(df.height - 1)]
    assert all(d == MIN_MS for d in diffs)


def test_load_default_resolution_is_1m(loader_with_1m_data):
    entry = _ms(datetime(2026, 3, 15, 6, 0, tzinfo=timezone.utc))
    df = loader_with_1m_data.load("BTC-USDC", entry)  # default duration=60
    assert df.height == 60
    span = df["timestamp"][-1] - df["timestamp"][0]
    assert span == 59 * MIN_MS


def test_load_returns_dataframe_columns(loader_with_1m_data):
    entry = _ms(datetime(2026, 3, 15, 6, 0, tzinfo=timezone.utc))
    df = loader_with_1m_data.load("BTC-USDC", entry, duration_candles=5)
    assert df.columns == ["timestamp", "open", "high", "low", "close", "volume"]


def test_load_short_when_data_runs_out(loader_with_1m_data):
    """Asking for more candles than exist after entry returns whatever's
    available — no padding, no error."""
    entry = _ms(datetime(2026, 3, 15, 23, 50, tzinfo=timezone.utc))  # 10 mins from EOD
    df = loader_with_1m_data.load("BTC-USDC", entry, duration_candles=60)
    assert df.height == 10
    assert df["timestamp"][0] == entry


# ---------------------------------------------------------------------------
# 5m resolution
# ---------------------------------------------------------------------------


def test_load_explicit_5m_resolution(tmp_path):
    data_dir = tmp_path / "parquet"
    start = _ms(datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc))
    df_5m = _candles(start_ms=start, n=24 * 12, period_ms=FIVE_MIN_MS)
    _write(data_dir / EXCHANGE / "BTC-USDC" / "5m_2026-03.parquet", df_5m)

    loader = ForwardPathLoader(ParquetDataLoader(data_dir, exchange=EXCHANGE))
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))
    df = loader.load("BTC-USDC", entry, duration_candles=12, resolution="5m")
    assert df.height == 12
    diffs = [df["timestamp"][i + 1] - df["timestamp"][i] for i in range(df.height - 1)]
    assert all(d == FIVE_MIN_MS for d in diffs)


# ---------------------------------------------------------------------------
# 1m → 5m fallback
# ---------------------------------------------------------------------------


def test_load_falls_back_from_1m_to_5m(tmp_path, caplog):
    """Only 5m data exists; asking for 1m must transparently fall back."""
    import logging

    data_dir = tmp_path / "parquet"
    start = _ms(datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc))
    df_5m = _candles(start_ms=start, n=24 * 12, period_ms=FIVE_MIN_MS)
    _write(data_dir / EXCHANGE / "BTC-USDC" / "5m_2026-03.parquet", df_5m)

    loader = ForwardPathLoader(ParquetDataLoader(data_dir, exchange=EXCHANGE))
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))

    with caplog.at_level(logging.WARNING):
        df = loader.load("BTC-USDC", entry, duration_candles=10)  # default 1m
    assert df.height == 10
    diffs = [df["timestamp"][i + 1] - df["timestamp"][i] for i in range(df.height - 1)]
    # All 5m even though we asked for 1m
    assert all(d == FIVE_MIN_MS for d in diffs)
    assert any("falling back to 5m" in rec.message for rec in caplog.records)


def test_load_5m_does_not_fall_back(tmp_path):
    """If 5m is explicitly requested and missing, we raise — no fallback."""
    loader = ForwardPathLoader(ParquetDataLoader(tmp_path / "parquet", exchange=EXCHANGE))
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))
    with pytest.raises(FileNotFoundError):
        loader.load("BTC-USDC", entry, duration_candles=10, resolution="5m")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_resolution_raises(loader_with_1m_data):
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="resolution must be one of"):
        loader_with_1m_data.load("BTC-USDC", entry, resolution="3m")


def test_zero_duration_raises(loader_with_1m_data):
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="duration_candles"):
        loader_with_1m_data.load("BTC-USDC", entry, duration_candles=0)


def test_no_data_at_either_resolution_raises(tmp_path):
    loader = ForwardPathLoader(ParquetDataLoader(tmp_path / "parquet", exchange=EXCHANGE))
    entry = _ms(datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc))
    with pytest.raises(FileNotFoundError):
        loader.load("BTC-USDC", entry, duration_candles=10)
