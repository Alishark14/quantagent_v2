"""Tests for bot deduplication guard in quantagent/main.py."""

from __future__ import annotations

import logging

from quantagent.main import _deduplicate_bots, _DEAD_SYMBOLS


def _bot(symbol: str, timeframe: str = "1h", bot_id: str | None = None) -> dict:
    return {
        "id": bot_id or f"bot-{symbol}-{timeframe}",
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": "hyperliquid",
        "mode": "shadow",
    }


def test_dedup_keeps_one_per_symbol():
    bots = [
        _bot("BTC-USDC", "30m"),
        _bot("BTC-USDC", "1h"),
        _bot("BTC-USDC", "4h"),
        _bot("ETH-USDC", "1h"),
    ]
    result = _deduplicate_bots(bots)
    symbols = [b["symbol"] for b in result]
    assert symbols.count("BTC-USDC") == 1
    assert symbols.count("ETH-USDC") == 1
    assert len(result) == 2


def test_dedup_prefers_matching_timeframe():
    bots = [
        _bot("BTC-USDC", "30m"),
        _bot("BTC-USDC", "4h"),
        _bot("BTC-USDC", "1h"),
    ]
    result = _deduplicate_bots(bots, preferred_timeframe="1h")
    btc = [b for b in result if b["symbol"] == "BTC-USDC"]
    assert len(btc) == 1
    assert btc[0]["timeframe"] == "1h"


def test_dedup_falls_back_to_first_when_no_preferred():
    bots = [
        _bot("BTC-USDC", "30m"),
        _bot("BTC-USDC", "4h"),
    ]
    result = _deduplicate_bots(bots, preferred_timeframe="1h")
    assert len(result) == 1
    assert result[0]["timeframe"] == "30m"  # first in list


def test_dedup_logs_warning_for_duplicates(caplog):
    bots = [
        _bot("BTC-USDC", "30m", "bot-a"),
        _bot("BTC-USDC", "1h", "bot-b"),
    ]
    with caplog.at_level(logging.WARNING, logger="quantagent"):
        _deduplicate_bots(bots)
    assert any("Duplicate bots for BTC-USDC" in r.message for r in caplog.records)


def test_dead_symbols_filtered():
    bots = [
        _bot("BTC-USDC", "1h"),
        _bot("SNDK-USDC", "1h"),
        _bot("USA500-USDC", "1h"),
        _bot("XYZ100-USDC", "1h"),
        _bot("ETH-USDC", "1h"),
    ]
    result = _deduplicate_bots(bots)
    symbols = {b["symbol"] for b in result}
    assert "SNDK-USDC" not in symbols
    assert "USA500-USDC" not in symbols
    assert "XYZ100-USDC" not in symbols
    assert "BTC-USDC" in symbols
    assert "ETH-USDC" in symbols
    assert len(result) == 2


def test_dead_symbols_logged(caplog):
    bots = [_bot("SNDK-USDC", "1h", "bot-dead")]
    with caplog.at_level(logging.WARNING, logger="quantagent"):
        result = _deduplicate_bots(bots)
    assert len(result) == 0
    assert any("dead symbol" in r.message.lower() for r in caplog.records)


def test_no_duplicates_passthrough():
    bots = [
        _bot("BTC-USDC", "1h"),
        _bot("ETH-USDC", "1h"),
        _bot("SOL-USDC", "1h"),
    ]
    result = _deduplicate_bots(bots)
    assert len(result) == 3


def test_empty_list():
    assert _deduplicate_bots([]) == []
