"""Tests for regime and mode columns on cycles.

Task F from Sprint Week 7 Update 2.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_cycle_includes_regime(tmp_path):
    """Cycle record stores regime as a standalone column."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.cycles.save_cycle({
        "bot_id": "b1", "symbol": "BTC-USDC", "timeframe": "1h",
        "action": "LONG", "conviction_score": 0.72,
        "regime": "TRENDING_UP",
        "mode": "shadow",
        "is_shadow": True,
    })

    cycles = await repos.cycles.get_recent_cycles("b1", include_shadow=True)
    assert len(cycles) == 1
    assert cycles[0]["regime"] == "TRENDING_UP"


@pytest.mark.asyncio
async def test_cycle_includes_mode(tmp_path):
    """Cycle record stores mode as a standalone column."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.cycles.save_cycle({
        "bot_id": "b1", "symbol": "ETH-USDC", "timeframe": "1h",
        "action": "SKIP", "conviction_score": 0.30,
        "regime": "RANGING",
        "mode": "live",
        "is_shadow": False,
    })

    cycles = await repos.cycles.get_recent_cycles("b1")
    assert len(cycles) == 1
    assert cycles[0]["mode"] == "live"


@pytest.mark.asyncio
async def test_backfill_mode_from_is_shadow(tmp_path):
    """Existing rows without mode get backfilled from is_shadow."""
    import aiosqlite
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    # Insert directly without mode (simulating pre-migration row)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO cycles (id, bot_id, symbol, timeframe, timestamp, "
            "action, conviction_score, is_shadow) "
            "VALUES ('c1', 'b1', 'BTC-USDC', '1h', '2026-04-12', 'LONG', 0.7, 1)"
        )
        await db.commit()

    # Simulate backfill (what the migration does)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE cycles SET mode = 'shadow' WHERE is_shadow = 1 AND mode = 'live'"
        )
        await db.commit()

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT mode FROM cycles WHERE id = 'c1'") as cur:
            row = dict(await cur.fetchone())
    assert row["mode"] == "shadow"


@pytest.mark.asyncio
async def test_is_shadow_queries_still_work(tmp_path):
    """Existing queries using is_shadow continue to work alongside mode."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    # Insert a shadow cycle and a live cycle
    await repos.cycles.save_cycle({
        "bot_id": "b1", "symbol": "BTC-USDC", "timeframe": "1h",
        "action": "LONG", "conviction_score": 0.72,
        "mode": "shadow", "is_shadow": True,
    })
    await repos.cycles.save_cycle({
        "bot_id": "b1", "symbol": "BTC-USDC", "timeframe": "1h",
        "action": "SKIP", "conviction_score": 0.30,
        "mode": "live", "is_shadow": False,
    })

    # Default (include_shadow=False) should only see live
    live = await repos.cycles.get_recent_cycles("b1")
    assert len(live) == 1
    assert live[0]["action"] == "SKIP"

    # include_shadow=True should see both
    all_cycles = await repos.cycles.get_recent_cycles("b1", include_shadow=True)
    assert len(all_cycles) == 2


@pytest.mark.asyncio
async def test_regime_null_for_old_rows(tmp_path):
    """Pre-migration rows have regime=NULL — this is expected and harmless."""
    import aiosqlite
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    # Insert without regime
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO cycles (id, bot_id, symbol, timeframe, timestamp, "
            "action, conviction_score, is_shadow) "
            "VALUES ('c-old', 'b1', 'BTC-USDC', '1h', '2026-04-10', 'SKIP', 0.3, 0)"
        )
        await db.commit()

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT regime FROM cycles WHERE id = 'c-old'") as cur:
            row = dict(await cur.fetchone())
    assert row["regime"] is None
