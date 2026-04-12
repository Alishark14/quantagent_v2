"""Tests for bot lifecycle columns: is_active, deactivated_at, last_cycle_at.

Task D from Sprint Week 7 Update 2.
"""

from __future__ import annotations

import logging

import pytest

from quantagent.main import _deduplicate_bots


# ── get_active_bots_by_mode + deactivate_bot via SQLite ──────────────


@pytest.mark.asyncio
async def test_get_active_bots_by_mode_filters_inactive(tmp_path):
    """Deactivated bots (is_active=0) must not appear in query results."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    # Create two shadow bots
    await repos.bots.save_bot({
        "id": "b-active", "user_id": "u1", "symbol": "BTC-USDC",
        "timeframe": "1h", "exchange": "hyperliquid",
        "mode": "shadow", "created_at": "2026-04-13T00:00:00Z",
    })
    await repos.bots.save_bot({
        "id": "b-inactive", "user_id": "u1", "symbol": "ETH-USDC",
        "timeframe": "1h", "exchange": "hyperliquid",
        "mode": "shadow", "created_at": "2026-04-13T00:00:00Z",
    })

    # Deactivate one
    result = await repos.bots.deactivate_bot("b-inactive")
    assert result is True

    # Query should only return the active one
    bots = await repos.bots.get_active_bots_by_mode("shadow")
    ids = [b["id"] for b in bots]
    assert "b-active" in ids
    assert "b-inactive" not in ids


@pytest.mark.asyncio
async def test_deactivate_bot_sets_fields(tmp_path):
    """deactivate_bot sets is_active=0 and deactivated_at to a timestamp."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.bots.save_bot({
        "id": "b1", "user_id": "u1", "symbol": "BTC-USDC",
        "timeframe": "1h", "exchange": "hyperliquid",
        "mode": "shadow", "created_at": "2026-04-13T00:00:00Z",
    })

    await repos.bots.deactivate_bot("b1")

    bot = await repos.bots.get_bot("b1")
    assert bot is not None
    assert bot["is_active"] == 0  # SQLite stores as INTEGER
    assert bot["deactivated_at"] is not None


@pytest.mark.asyncio
async def test_update_last_cycle_stamps_timestamp(tmp_path):
    """update_last_cycle sets last_cycle_at."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.bots.save_bot({
        "id": "b1", "user_id": "u1", "symbol": "BTC-USDC",
        "timeframe": "1h", "exchange": "hyperliquid",
        "mode": "shadow", "created_at": "2026-04-13T00:00:00Z",
    })

    # Initially null
    bot = await repos.bots.get_bot("b1")
    assert bot["last_cycle_at"] is None

    # After update
    await repos.bots.update_last_cycle("b1")
    bot = await repos.bots.get_bot("b1")
    assert bot["last_cycle_at"] is not None


# ── Deduplication prefers last_cycle_at ──────────────────────────────


def test_dedup_prefers_most_recent_last_cycle_at():
    """When multiple bots match the preferred TF, keep the one with
    the most recent last_cycle_at."""
    bots = [
        {"id": "old", "symbol": "BTC-USDC", "timeframe": "1h",
         "last_cycle_at": "2026-04-12T10:00:00Z"},
        {"id": "new", "symbol": "BTC-USDC", "timeframe": "1h",
         "last_cycle_at": "2026-04-13T10:00:00Z"},
    ]
    result = _deduplicate_bots(bots, preferred_timeframe="1h")
    assert len(result) == 1
    assert result[0]["id"] == "new"


def test_dedup_prefers_last_cycle_at_over_none():
    """A bot with a last_cycle_at beats one without."""
    bots = [
        {"id": "never-ran", "symbol": "BTC-USDC", "timeframe": "1h",
         "last_cycle_at": None},
        {"id": "ran-recently", "symbol": "BTC-USDC", "timeframe": "1h",
         "last_cycle_at": "2026-04-13T10:00:00Z"},
    ]
    result = _deduplicate_bots(bots, preferred_timeframe="1h")
    assert len(result) == 1
    assert result[0]["id"] == "ran-recently"


def test_dedup_logs_warning_for_duplicates(caplog):
    """Dedup logs a WARNING when it drops duplicate bots."""
    bots = [
        {"id": "b1", "symbol": "BTC-USDC", "timeframe": "1h"},
        {"id": "b2", "symbol": "BTC-USDC", "timeframe": "30m"},
    ]
    with caplog.at_level(logging.WARNING, logger="quantagent"):
        _deduplicate_bots(bots)
    assert any("Duplicate bots for BTC-USDC" in r.message for r in caplog.records)
