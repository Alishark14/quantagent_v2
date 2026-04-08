"""Unit tests for MockSignalProducer."""

from __future__ import annotations

import json

import pytest

from backtesting.mock_signals import MockSignalProducer
from engine.types import MarketData


def _md(ts: int = 1_000_000) -> MarketData:
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=[{"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}],
        num_candles=1,
        lookback_description="1 candle",
        forecast_candles=3,
        forecast_description="3h",
        indicators={},
        swing_highs=[],
        swing_lows=[],
    )


@pytest.mark.asyncio
async def test_always_long():
    p = MockSignalProducer("always_long")
    sig = await p.analyze(_md())
    assert sig.direction == "BULLISH"
    assert sig.signal_type == "ml"
    assert sig.agent_name == "mock_signal"


@pytest.mark.asyncio
async def test_always_short():
    p = MockSignalProducer("always_short")
    sig = await p.analyze(_md())
    assert sig.direction == "BEARISH"


@pytest.mark.asyncio
async def test_always_skip():
    p = MockSignalProducer("always_skip")
    sig = await p.analyze(_md())
    assert sig.direction == "NEUTRAL"


@pytest.mark.asyncio
async def test_random_seed_is_reproducible():
    p1 = MockSignalProducer("random_seed:42")
    p2 = MockSignalProducer("random_seed:42")
    seq1 = [(await p1.analyze(_md(ts=i))).direction for i in range(20)]
    seq2 = [(await p2.analyze(_md(ts=i))).direction for i in range(20)]
    assert seq1 == seq2
    # And it actually varies
    assert len(set(seq1)) > 1


@pytest.mark.asyncio
async def test_random_seed_different_seeds_diverge():
    p1 = MockSignalProducer("random_seed:1")
    p2 = MockSignalProducer("random_seed:2")
    seq1 = [(await p1.analyze(_md(ts=i))).direction for i in range(50)]
    seq2 = [(await p2.analyze(_md(ts=i))).direction for i in range(50)]
    assert seq1 != seq2


def test_random_seed_invalid_int():
    with pytest.raises(ValueError, match="random_seed mode requires int"):
        MockSignalProducer("random_seed:notanint")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown MockSignalProducer mode"):
        MockSignalProducer("totally_made_up")


@pytest.mark.asyncio
async def test_from_file_replays_records(tmp_path):
    records = [
        {"timestamp": 1000, "direction": "BULLISH", "confidence": 0.9},
        {"timestamp": 2000, "direction": "BEARISH", "confidence": 0.7},
        {"timestamp": 3000, "direction": "NEUTRAL", "confidence": 0.5},
    ]
    f = tmp_path / "signals.json"
    f.write_text(json.dumps(records))

    p = MockSignalProducer(f"from_file:{f}")
    assert (await p.analyze(_md(ts=1000))).direction == "BULLISH"
    assert (await p.analyze(_md(ts=2000))).direction == "BEARISH"
    assert (await p.analyze(_md(ts=3000))).direction == "NEUTRAL"
    # Unknown timestamp → NEUTRAL (skip)
    assert (await p.analyze(_md(ts=9999))).direction == "NEUTRAL"


def test_from_file_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        MockSignalProducer(f"from_file:{tmp_path / 'nope.json'}")


def test_from_file_bad_record(tmp_path):
    f = tmp_path / "signals.json"
    f.write_text(json.dumps([{"timestamp": 1, "direction": "INVALID"}]))
    with pytest.raises(ValueError, match="direction must be one of"):
        MockSignalProducer(f"from_file:{f}")


def test_from_file_must_be_list(tmp_path):
    f = tmp_path / "signals.json"
    f.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="JSON list"):
        MockSignalProducer(f"from_file:{f}")


def test_is_signal_producer_subclass():
    from engine.signals.base import SignalProducer
    p = MockSignalProducer("always_long")
    assert isinstance(p, SignalProducer)
    assert p.is_enabled() is True
    assert p.signal_type() == "ml"
    assert p.requires_vision() is False
