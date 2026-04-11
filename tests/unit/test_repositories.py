"""Unit tests for repository pattern (SQLite backend).

Tests cover all 5 repositories: Trade, Cycle, Rule, Bot, CrossBot.
Critically tests multi-tenant isolation on cross-bot signals.
"""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from storage.repositories.sqlite import SQLiteRepositories


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def repos(tmp_path):
    """Create a fresh SQLite repository set for each test."""
    db_path = str(tmp_path / "test.db")
    r = SQLiteRepositories(db_path=db_path)
    await r.init_db()
    return r


# ---------------------------------------------------------------------------
# TradeRepository
# ---------------------------------------------------------------------------

class TestTradeRepository:

    @pytest.mark.asyncio
    async def test_save_and_get_roundtrip(self, repos):
        trade = {
            "user_id": "user1",
            "bot_id": "bot1",
            "symbol": "BTC-USDC",
            "timeframe": "1h",
            "direction": "LONG",
            "entry_price": 65000.0,
            "size": 0.1,
            "conviction_score": 0.78,
            "status": "open",
        }
        trade_id = await repos.trades.save_trade(trade)
        assert trade_id

        fetched = await repos.trades.get_trade(trade_id)
        assert fetched is not None
        assert fetched["symbol"] == "BTC-USDC"
        assert fetched["direction"] == "LONG"
        assert fetched["entry_price"] == 65000.0
        assert fetched["status"] == "open"

    @pytest.mark.asyncio
    async def test_get_trade_not_found(self, repos):
        result = await repos.trades.get_trade("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_open_positions_filters_by_user_and_bot(self, repos):
        # User1 bot1: open
        await repos.trades.save_trade({
            "user_id": "user1", "bot_id": "bot1", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
        })
        # User1 bot1: closed (should NOT appear)
        await repos.trades.save_trade({
            "user_id": "user1", "bot_id": "bot1", "symbol": "ETH-USDC",
            "timeframe": "1h", "direction": "SHORT", "status": "closed",
        })
        # User1 bot2: open (different bot, should NOT appear)
        await repos.trades.save_trade({
            "user_id": "user1", "bot_id": "bot2", "symbol": "SOL-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
        })
        # User2 bot1: open (different user, should NOT appear)
        await repos.trades.save_trade({
            "user_id": "user2", "bot_id": "bot1", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
        })

        positions = await repos.trades.get_open_positions("user1", "bot1")
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC-USDC"
        assert positions[0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_get_trades_by_bot(self, repos):
        for i in range(5):
            await repos.trades.save_trade({
                "user_id": "user1", "bot_id": "bot1", "symbol": "BTC-USDC",
                "timeframe": "1h", "direction": "LONG", "status": "closed",
                "entry_time": f"2026-04-0{i+1}T00:00:00Z",
            })
        trades = await repos.trades.get_trades_by_bot("bot1", limit=3)
        assert len(trades) == 3
        # Ordered by entry_time DESC
        assert trades[0]["entry_time"] >= trades[1]["entry_time"]

    @pytest.mark.asyncio
    async def test_update_trade(self, repos):
        trade_id = await repos.trades.save_trade({
            "user_id": "user1", "bot_id": "bot1", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
        })
        updated = await repos.trades.update_trade(trade_id, {
            "status": "closed",
            "exit_price": 66000.0,
            "pnl": 100.0,
            "exit_reason": "TP1",
        })
        assert updated is True

        fetched = await repos.trades.get_trade(trade_id)
        assert fetched["status"] == "closed"
        assert fetched["exit_price"] == 66000.0
        assert fetched["pnl"] == 100.0
        assert fetched["exit_reason"] == "TP1"

    @pytest.mark.asyncio
    async def test_update_trade_not_found(self, repos):
        result = await repos.trades.update_trade("nonexistent", {"status": "closed"})
        assert result is False

    @pytest.mark.asyncio
    async def test_update_trade_empty_updates(self, repos):
        result = await repos.trades.update_trade("any-id", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_save_trade_with_explicit_id(self, repos):
        trade_id = await repos.trades.save_trade({
            "id": "my-custom-id",
            "user_id": "user1", "bot_id": "bot1", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
        })
        assert trade_id == "my-custom-id"
        fetched = await repos.trades.get_trade("my-custom-id")
        assert fetched is not None

    @pytest.mark.asyncio
    async def test_save_trade_persists_sl_tp(self, repos):
        trade_id = await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
            "entry_price": 65000.0,
            "sl_price": 64000.0,
            "tp_price": 67000.0,
            "is_shadow": True,
        })
        fetched = await repos.trades.get_trade(trade_id)
        assert fetched["sl_price"] == 64000.0
        assert fetched["tp_price"] == 67000.0
        assert bool(fetched["is_shadow"]) is True

    @pytest.mark.asyncio
    async def test_get_open_shadow_trades_filters(self, repos):
        from datetime import datetime, timezone
        # Open shadow on BTC — should appear
        sid = await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
            "is_shadow": True,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        })
        # Closed shadow on BTC — must NOT appear
        await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "SHORT", "status": "closed",
            "is_shadow": True,
        })
        # Open LIVE on BTC — must NOT appear (not shadow)
        await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
            "is_shadow": False,
        })
        # Open shadow on ETH — wrong symbol, must NOT appear
        await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "ETH-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
            "is_shadow": True,
        })

        rows = await repos.trades.get_open_shadow_trades("BTC-USDC")
        assert len(rows) == 1
        assert rows[0]["id"] == sid

    @pytest.mark.asyncio
    async def test_close_trade_marks_closed_with_pnl(self, repos):
        from datetime import datetime, timezone
        trade_id = await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
            "entry_price": 65000.0, "size": 500.0,
            "sl_price": 64000.0, "tp_price": 67000.0,
            "is_shadow": True,
        })
        ok = await repos.trades.close_trade(
            trade_id,
            exit_price=67000.0,
            exit_reason="TP",
            exit_time=datetime.now(timezone.utc),
            pnl=15.38,
        )
        assert ok is True
        fetched = await repos.trades.get_trade(trade_id)
        assert fetched["status"] == "closed"
        assert fetched["exit_price"] == 67000.0
        assert fetched["exit_reason"] == "TP"
        assert fetched["pnl"] == pytest.approx(15.38)

    @pytest.mark.asyncio
    async def test_close_trade_idempotent_on_already_closed(self, repos):
        from datetime import datetime, timezone
        trade_id = await repos.trades.save_trade({
            "user_id": "u", "bot_id": "b", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "closed",
            "is_shadow": True,
        })
        ok = await repos.trades.close_trade(
            trade_id, exit_price=1.0, exit_reason="SL",
            exit_time=datetime.now(timezone.utc), pnl=0.0,
        )
        assert ok is False


# ---------------------------------------------------------------------------
# CycleRepository
# ---------------------------------------------------------------------------

class TestCycleRepository:

    @pytest.mark.asyncio
    async def test_save_and_get_recent_cycles(self, repos):
        for i in range(7):
            await repos.cycles.save_cycle({
                "bot_id": "bot1", "symbol": "BTC-USDC", "timeframe": "1h",
                "timestamp": f"2026-04-01T{i:02d}:00:00Z",
                "action": "SKIP", "conviction_score": 0.3 + i * 0.1,
            })

        recent = await repos.cycles.get_recent_cycles("bot1", limit=5)
        assert len(recent) == 5
        # Ordered by timestamp DESC
        assert recent[0]["timestamp"] >= recent[1]["timestamp"]

    @pytest.mark.asyncio
    async def test_cycle_json_fields_roundtrip(self, repos):
        indicators = {"rsi": 65.2, "macd_histogram": 0.5}
        signals = [{"agent": "indicator", "direction": "BULLISH", "confidence": 0.7}]
        conviction = {"score": 0.72, "regime": "TRENDING_UP"}

        cycle_id = await repos.cycles.save_cycle({
            "bot_id": "bot1", "symbol": "BTC-USDC", "timeframe": "1h",
            "indicators": indicators,
            "signals": signals,
            "conviction": conviction,
            "action": "LONG", "conviction_score": 0.72,
        })

        recent = await repos.cycles.get_recent_cycles("bot1", limit=1)
        assert len(recent) == 1
        assert recent[0]["indicators_json"] == indicators
        assert recent[0]["signals_json"] == signals
        assert recent[0]["conviction_json"] == conviction

    @pytest.mark.asyncio
    async def test_get_recent_cycles_empty(self, repos):
        recent = await repos.cycles.get_recent_cycles("nonexistent-bot")
        assert recent == []

    @pytest.mark.asyncio
    async def test_cycle_filters_by_bot_id(self, repos):
        await repos.cycles.save_cycle({
            "bot_id": "bot1", "symbol": "BTC-USDC", "timeframe": "1h",
            "action": "LONG",
        })
        await repos.cycles.save_cycle({
            "bot_id": "bot2", "symbol": "ETH-USDC", "timeframe": "4h",
            "action": "SHORT",
        })

        bot1_cycles = await repos.cycles.get_recent_cycles("bot1")
        assert len(bot1_cycles) == 1
        assert bot1_cycles[0]["symbol"] == "BTC-USDC"

    @pytest.mark.asyncio
    async def test_save_cycle_with_datetime_timestamp(self, repos):
        """Cycle saves accept a tz-aware ``datetime`` for ``timestamp``.

        Regression: ``engine/pipeline.py`` used to stringify the timestamp via
        ``isoformat()``, which SQLite happily coerces but PostgreSQL+asyncpg
        rejects with ``DataError`` (asyncpg requires a real ``datetime`` for
        ``TIMESTAMPTZ``). The pipeline was changed to pass a raw datetime
        object; this test pins that contract on the SQLite side so the cycle
        save remains backend-agnostic.
        """
        from datetime import datetime, timezone

        ts = datetime(2026, 4, 9, 16, 43, 8, tzinfo=timezone.utc)
        cycle_id = await repos.cycles.save_cycle({
            "bot_id": "bot-dt",
            "symbol": "BTC-USDC",
            "timeframe": "1h",
            "timestamp": ts,
            "action": "SKIP",
            "conviction_score": 0.42,
            "indicators": {"rsi": 60.0},
        })
        assert cycle_id

        recent = await repos.cycles.get_recent_cycles("bot-dt", limit=1)
        assert len(recent) == 1
        # SQLite stores datetimes as the str(dt) representation; assert the
        # round-trip preserves the wall-clock instant.
        stored_ts = recent[0]["timestamp"]
        assert "2026-04-09" in str(stored_ts)
        assert "16:43:08" in str(stored_ts)
        assert recent[0]["action"] == "SKIP"
        assert recent[0]["indicators_json"] == {"rsi": 60.0}


# ---------------------------------------------------------------------------
# RuleRepository
# ---------------------------------------------------------------------------

class TestRuleRepository:

    @pytest.mark.asyncio
    async def test_save_and_get_rules(self, repos):
        await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Avoid LONG when RSI > 65 and funding > 0.03%",
        })
        await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Strong LONG only when all 3 agents agree",
        })
        # Different symbol — should not appear
        await repos.rules.save_rule({
            "symbol": "ETH-USDC", "timeframe": "1h",
            "rule_text": "ETH-specific rule",
        })

        rules = await repos.rules.get_rules("BTC-USDC", "1h")
        assert len(rules) == 2
        assert all(r["active"] is True for r in rules)

    @pytest.mark.asyncio
    async def test_update_rule_score_increment(self, repos):
        rule_id = await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Test rule", "score": 0,
        })

        result = await repos.rules.update_rule_score(rule_id, +1)
        assert result is True

        rules = await repos.rules.get_rules("BTC-USDC", "1h")
        assert rules[0]["score"] == 1

    @pytest.mark.asyncio
    async def test_update_rule_score_auto_deactivate(self, repos):
        """Rule auto-deactivates when score drops below -2."""
        rule_id = await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Bad rule", "score": 0,
        })

        # Decrement to -3 (below -2 threshold)
        await repos.rules.update_rule_score(rule_id, -3)

        # Should no longer appear in active rules
        rules = await repos.rules.get_rules("BTC-USDC", "1h")
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_update_rule_score_not_found(self, repos):
        result = await repos.rules.update_rule_score("nonexistent", 1)
        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate_rule(self, repos):
        rule_id = await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "To be deactivated",
        })

        result = await repos.rules.deactivate_rule(rule_id)
        assert result is True

        rules = await repos.rules.get_rules("BTC-USDC", "1h")
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_deactivate_rule_not_found(self, repos):
        result = await repos.rules.deactivate_rule("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_rules_filters_by_timeframe(self, repos):
        await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "1h rule",
        })
        await repos.rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "4h",
            "rule_text": "4h rule",
        })

        rules_1h = await repos.rules.get_rules("BTC-USDC", "1h")
        assert len(rules_1h) == 1
        assert rules_1h[0]["rule_text"] == "1h rule"


# ---------------------------------------------------------------------------
# BotRepository
# ---------------------------------------------------------------------------

class TestBotRepository:

    @pytest.mark.asyncio
    async def test_save_and_get_bot(self, repos):
        bot_id = await repos.bots.save_bot({
            "user_id": "user1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "config": {"max_position_pct": 0.5},
        })

        bot = await repos.bots.get_bot(bot_id)
        assert bot is not None
        assert bot["symbol"] == "BTC-USDC"
        assert bot["exchange"] == "hyperliquid"
        assert bot["config_json"] == {"max_position_pct": 0.5}

    @pytest.mark.asyncio
    async def test_get_bot_not_found(self, repos):
        result = await repos.bots.get_bot("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_bots_by_user(self, repos):
        await repos.bots.save_bot({
            "user_id": "user1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "user_id": "user1", "symbol": "ETH-USDC", "timeframe": "4h",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "user_id": "user2", "symbol": "SOL-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })

        user1_bots = await repos.bots.get_bots_by_user("user1")
        assert len(user1_bots) == 2

        user2_bots = await repos.bots.get_bots_by_user("user2")
        assert len(user2_bots) == 1

    @pytest.mark.asyncio
    async def test_update_bot_health(self, repos):
        bot_id = await repos.bots.save_bot({
            "user_id": "user1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })

        health = {"state": "running", "last_cycle": "2026-04-06T12:00:00Z"}
        result = await repos.bots.update_bot_health(bot_id, health)
        assert result is True

        bot = await repos.bots.get_bot(bot_id)
        assert bot["last_health"] == health

    @pytest.mark.asyncio
    async def test_update_bot_health_not_found(self, repos):
        result = await repos.bots.update_bot_health("nonexistent", {"state": "dead"})
        assert result is False


# ---------------------------------------------------------------------------
# CrossBotRepository — Multi-tenant isolation is critical here
# ---------------------------------------------------------------------------

class TestCrossBotRepository:

    @pytest.mark.asyncio
    async def test_save_and_get_signals(self, repos):
        await repos.cross_bot.save_signal({
            "user_id": "user1", "symbol": "BTC-USDC", "direction": "LONG",
            "conviction": 0.85, "bot_id": "bot1",
        })
        await repos.cross_bot.save_signal({
            "user_id": "user1", "symbol": "BTC-USDC", "direction": "SHORT",
            "conviction": 0.72, "bot_id": "bot2",
        })

        signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "user1")
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, repos):
        """User A's cross-bot signals MUST NOT be visible to User B."""
        await repos.cross_bot.save_signal({
            "user_id": "user_alice", "symbol": "BTC-USDC", "direction": "LONG",
            "conviction": 0.9, "bot_id": "alice_bot1",
        })
        await repos.cross_bot.save_signal({
            "user_id": "user_bob", "symbol": "BTC-USDC", "direction": "SHORT",
            "conviction": 0.8, "bot_id": "bob_bot1",
        })

        # Alice can only see her own signals
        alice_signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "user_alice")
        assert len(alice_signals) == 1
        assert alice_signals[0]["direction"] == "LONG"
        assert alice_signals[0]["user_id"] == "user_alice"

        # Bob can only see his own signals
        bob_signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "user_bob")
        assert len(bob_signals) == 1
        assert bob_signals[0]["direction"] == "SHORT"
        assert bob_signals[0]["user_id"] == "user_bob"

    @pytest.mark.asyncio
    async def test_signals_filter_by_symbol(self, repos):
        await repos.cross_bot.save_signal({
            "user_id": "user1", "symbol": "BTC-USDC", "direction": "LONG",
            "conviction": 0.85, "bot_id": "bot1",
        })
        await repos.cross_bot.save_signal({
            "user_id": "user1", "symbol": "ETH-USDC", "direction": "SHORT",
            "conviction": 0.72, "bot_id": "bot2",
        })

        btc_signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "user1")
        assert len(btc_signals) == 1
        assert btc_signals[0]["symbol"] == "BTC-USDC"

    @pytest.mark.asyncio
    async def test_signals_respects_limit(self, repos):
        for i in range(10):
            await repos.cross_bot.save_signal({
                "user_id": "user1", "symbol": "BTC-USDC", "direction": "LONG",
                "conviction": 0.5 + i * 0.05, "bot_id": f"bot{i}",
                "timestamp": f"2026-04-01T{i:02d}:00:00Z",
            })

        signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "user1", limit=3)
        assert len(signals) == 3

    @pytest.mark.asyncio
    async def test_signals_ordered_by_timestamp_desc(self, repos):
        await repos.cross_bot.save_signal({
            "user_id": "user1", "symbol": "BTC-USDC", "direction": "LONG",
            "conviction": 0.85, "bot_id": "bot1",
            "timestamp": "2026-04-01T01:00:00Z",
        })
        await repos.cross_bot.save_signal({
            "user_id": "user1", "symbol": "BTC-USDC", "direction": "SHORT",
            "conviction": 0.72, "bot_id": "bot2",
            "timestamp": "2026-04-01T02:00:00Z",
        })

        signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "user1")
        assert signals[0]["timestamp"] >= signals[1]["timestamp"]


# ---------------------------------------------------------------------------
# OISnapshotRepository
# ---------------------------------------------------------------------------


class TestOISnapshotRepository:

    @pytest.mark.asyncio
    async def test_insert_and_recent_roundtrip(self, repos):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        await repos.oi_snapshots.insert_snapshot("BTC-USDC", now, 1_000_000.0)
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now - timedelta(seconds=120), 990_000.0
        )
        rows = await repos.oi_snapshots.get_recent_snapshots(600)
        assert len(rows) == 2
        # ascending order
        assert rows[0]["oi_value"] == 990_000.0
        assert rows[1]["oi_value"] == 1_000_000.0
        assert rows[0]["symbol"] == "BTC-USDC"

    @pytest.mark.asyncio
    async def test_insert_idempotent_on_conflict(self, repos):
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc)
        await repos.oi_snapshots.insert_snapshot("BTC-USDC", ts, 1_000_000.0)
        # Same (symbol, ts) — should silently no-op
        await repos.oi_snapshots.insert_snapshot("BTC-USDC", ts, 9_999_999.0)
        rows = await repos.oi_snapshots.get_recent_snapshots(600)
        assert len(rows) == 1
        assert rows[0]["oi_value"] == 1_000_000.0  # original wins

    @pytest.mark.asyncio
    async def test_get_recent_filters_by_lookback(self, repos):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        # Inside the 60s window
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now - timedelta(seconds=30), 1.0
        )
        # Outside the 60s window
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now - timedelta(seconds=600), 2.0
        )
        rows = await repos.oi_snapshots.get_recent_snapshots(60)
        assert len(rows) == 1
        assert rows[0]["oi_value"] == 1.0

    @pytest.mark.asyncio
    async def test_cleanup_older_than(self, repos):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now, 1.0
        )
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now - timedelta(seconds=10_000), 2.0
        )
        deleted = await repos.oi_snapshots.cleanup_older_than(3_600)
        assert deleted == 1
        rows = await repos.oi_snapshots.get_recent_snapshots(86_400)
        assert len(rows) == 1
        assert rows[0]["oi_value"] == 1.0

    @pytest.mark.asyncio
    async def test_recent_orders_by_symbol_then_time(self, repos):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        await repos.oi_snapshots.insert_snapshot(
            "ETH-USDC", now - timedelta(seconds=30), 50.0
        )
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now - timedelta(seconds=60), 1.0
        )
        await repos.oi_snapshots.insert_snapshot(
            "BTC-USDC", now - timedelta(seconds=30), 2.0
        )
        rows = await repos.oi_snapshots.get_recent_snapshots(600)
        symbols = [r["symbol"] for r in rows]
        assert symbols == ["BTC-USDC", "BTC-USDC", "ETH-USDC"]


# ---------------------------------------------------------------------------
# Shadow-mode filtering — Task 2 of the Shadow Redesign sprint
# ---------------------------------------------------------------------------

class TestShadowFiltering:
    """Repo-layer shadow/live data isolation.

    These tests verify the per-row shadow flag added by Alembic 003 is
    honoured by every list-returning read method on TradeRepository,
    CycleRepository, and BotRepository, and that writes from a shadow
    bot persist `is_shadow=True` so the filtering has something to bite
    on. Live-only must be the default for every read so that production
    queries cannot accidentally surface shadow data.
    """

    @pytest.mark.asyncio
    async def test_get_active_bots_excludes_shadow_by_default(self, repos):
        """get_active_bots() with no kwarg returns only live bots."""
        await repos.bots.save_bot({
            "id": "live-1",
            "user_id": "u1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "id": "shadow-1",
            "user_id": "u1", "symbol": "ETH-USDC", "timeframe": "4h",
            "exchange": "hyperliquid", "mode": "shadow",
        })

        active = await repos.bots.get_active_bots()
        ids = {b["id"] for b in active}
        assert ids == {"live-1"}

    @pytest.mark.asyncio
    async def test_get_active_bots_includes_shadow_when_asked(self, repos):
        """get_active_bots(include_shadow=True) returns the union."""
        await repos.bots.save_bot({
            "id": "live-1",
            "user_id": "u1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "id": "shadow-1",
            "user_id": "u1", "symbol": "ETH-USDC", "timeframe": "4h",
            "exchange": "hyperliquid", "mode": "shadow",
        })

        active = await repos.bots.get_active_bots(include_shadow=True)
        ids = {b["id"] for b in active}
        assert ids == {"live-1", "shadow-1"}

    @pytest.mark.asyncio
    async def test_get_active_bots_by_mode_returns_only_matching_mode(self, repos):
        """get_active_bots_by_mode('shadow') returns shadow bots only.

        Cross-checks with mode='live' to make sure the filter is exact
        (not a substring or prefix match).
        """
        await repos.bots.save_bot({
            "id": "live-1",
            "user_id": "u1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "id": "live-2",
            "user_id": "u1", "symbol": "SOL-USDC", "timeframe": "15m",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "id": "shadow-1",
            "user_id": "u1", "symbol": "ETH-USDC", "timeframe": "4h",
            "exchange": "hyperliquid", "mode": "shadow",
        })
        await repos.bots.save_bot({
            "id": "shadow-2",
            "user_id": "u1", "symbol": "AVAX-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "mode": "shadow",
        })

        shadow_bots = await repos.bots.get_active_bots_by_mode("shadow")
        assert {b["id"] for b in shadow_bots} == {"shadow-1", "shadow-2"}

        live_bots = await repos.bots.get_active_bots_by_mode("live")
        assert {b["id"] for b in live_bots} == {"live-1", "live-2"}

    @pytest.mark.asyncio
    async def test_save_trade_persists_is_shadow_flag(self, repos):
        """A trade dict with is_shadow=True writes 1 in the column."""
        live_id = await repos.trades.save_trade({
            "user_id": "u1", "bot_id": "live-bot", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
        })
        shadow_id = await repos.trades.save_trade({
            "user_id": "u1", "bot_id": "shadow-bot", "symbol": "BTC-USDC",
            "timeframe": "1h", "direction": "LONG", "status": "open",
            "is_shadow": True,
        })

        live_trade = await repos.trades.get_trade(live_id)
        shadow_trade = await repos.trades.get_trade(shadow_id)
        assert live_trade["is_shadow"] == 0
        assert shadow_trade["is_shadow"] == 1

    @pytest.mark.asyncio
    async def test_get_trades_by_bot_excludes_shadow_by_default(self, repos):
        """get_trades_by_bot() returns 0 rows when the bot has only shadow trades.

        And get_open_positions has the same default. This is the strongest
        guarantee: even reading by an explicit bot_id, you cannot leak
        shadow trades into production code paths without opting in.
        """
        await repos.trades.save_trade({
            "id": "t-shadow", "user_id": "u1", "bot_id": "shadow-bot",
            "symbol": "BTC-USDC", "timeframe": "1h", "direction": "LONG",
            "status": "open", "is_shadow": True,
            "entry_time": "2026-04-09T00:00:00Z",
        })

        # default → shadow filtered out
        trades = await repos.trades.get_trades_by_bot("shadow-bot")
        assert trades == []

        positions = await repos.trades.get_open_positions("u1", "shadow-bot")
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_trades_by_bot_includes_shadow_when_asked(self, repos):
        """include_shadow=True surfaces shadow trades alongside live."""
        await repos.trades.save_trade({
            "id": "t-live", "user_id": "u1", "bot_id": "bot-x",
            "symbol": "BTC-USDC", "timeframe": "1h", "direction": "LONG",
            "status": "open",
            "entry_time": "2026-04-09T00:00:00Z",
        })
        await repos.trades.save_trade({
            "id": "t-shadow", "user_id": "u1", "bot_id": "bot-x",
            "symbol": "BTC-USDC", "timeframe": "1h", "direction": "SHORT",
            "status": "open", "is_shadow": True,
            "entry_time": "2026-04-09T01:00:00Z",
        })

        # default → only live
        live_only = await repos.trades.get_trades_by_bot("bot-x")
        assert {t["id"] for t in live_only} == {"t-live"}

        # opt-in → both
        both = await repos.trades.get_trades_by_bot("bot-x", include_shadow=True)
        assert {t["id"] for t in both} == {"t-live", "t-shadow"}

        # get_open_positions has the same shape
        live_pos = await repos.trades.get_open_positions("u1", "bot-x")
        assert {t["id"] for t in live_pos} == {"t-live"}
        all_pos = await repos.trades.get_open_positions(
            "u1", "bot-x", include_shadow=True
        )
        assert {t["id"] for t in all_pos} == {"t-live", "t-shadow"}

    @pytest.mark.asyncio
    async def test_get_recent_cycles_excludes_shadow_by_default(self, repos):
        """CycleRepository follows the same shadow-filtering contract."""
        await repos.cycles.save_cycle({
            "id": "c-live", "bot_id": "bot-x", "symbol": "BTC-USDC",
            "timeframe": "1h", "timestamp": "2026-04-09T00:00:00Z",
            "action": "LONG",
        })
        await repos.cycles.save_cycle({
            "id": "c-shadow", "bot_id": "bot-x", "symbol": "BTC-USDC",
            "timeframe": "1h", "timestamp": "2026-04-09T01:00:00Z",
            "action": "SHORT", "is_shadow": True,
        })

        live = await repos.cycles.get_recent_cycles("bot-x")
        assert {c["id"] for c in live} == {"c-live"}

        both = await repos.cycles.get_recent_cycles("bot-x", include_shadow=True)
        assert {c["id"] for c in both} == {"c-live", "c-shadow"}

    @pytest.mark.asyncio
    async def test_save_bot_paper_mode_auto_derives_is_shadow_true(self, repos):
        """Paper Trading Task 2 — paper bots are auto-marked is_shadow=True.

        Paper bots trade on testnet (fake money). They must be excluded
        from the live data moat — the QuantDataScientist mining job
        reads the live_* views which strip ``is_shadow=true`` rows, so
        the auto-derive guarantees no testnet fills leak into alpha
        mining without any caller having to remember to set the flag.
        """
        await repos.bots.save_bot({
            "id": "paper-1",
            "user_id": "u1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "mode": "paper",
        })

        # The repo's get_active_bots(include_shadow=False) — the default
        # — strips is_shadow=True rows. So the paper bot should be
        # absent from the live read but present in the include-shadow
        # opt-in read. This pins the auto-derive end-to-end.
        live_only = await repos.bots.get_active_bots()
        assert "paper-1" not in {b["id"] for b in live_only}

        with_shadow = await repos.bots.get_active_bots(include_shadow=True)
        paper_bot = next(b for b in with_shadow if b["id"] == "paper-1")
        # SQLite stores BOOLEAN as 0/1; postgres returns Python bools.
        # Both backends should set the flag to a truthy value.
        assert bool(paper_bot["is_shadow"]) is True
        assert paper_bot["mode"] == "paper"

    @pytest.mark.asyncio
    async def test_save_bot_live_mode_keeps_is_shadow_false(self, repos):
        """Negative control: live bots are NOT auto-marked is_shadow.

        Pins the inverse of the paper auto-derive so a future regression
        that broadens the auto-derive (e.g. typo in the mode tuple)
        gets caught immediately.
        """
        await repos.bots.save_bot({
            "id": "live-only",
            "user_id": "u1", "symbol": "ETH-USDC", "timeframe": "4h",
            "exchange": "hyperliquid", "mode": "live",
        })

        live_only = await repos.bots.get_active_bots()
        bot = next(b for b in live_only if b["id"] == "live-only")
        assert bool(bot["is_shadow"]) is False
        assert bot["mode"] == "live"

    @pytest.mark.asyncio
    async def test_get_active_bots_by_mode_paper_returns_paper_bots(self, repos):
        """``get_active_bots_by_mode('paper')`` filters by exact mode
        match — proves the existing by-mode helper works for the new
        mode value without any accessor changes."""
        await repos.bots.save_bot({
            "id": "live-1",
            "user_id": "u1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid",
        })
        await repos.bots.save_bot({
            "id": "shadow-1",
            "user_id": "u1", "symbol": "ETH-USDC", "timeframe": "4h",
            "exchange": "hyperliquid", "mode": "shadow",
        })
        await repos.bots.save_bot({
            "id": "paper-1",
            "user_id": "u1", "symbol": "SOL-USDC", "timeframe": "30m",
            "exchange": "hyperliquid", "mode": "paper",
        })
        await repos.bots.save_bot({
            "id": "paper-2",
            "user_id": "u1", "symbol": "AVAX-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "mode": "paper",
        })

        paper_bots = await repos.bots.get_active_bots_by_mode("paper")
        assert {b["id"] for b in paper_bots} == {"paper-1", "paper-2"}

        # Cross-check: shadow filter doesn't see paper bots
        shadow_bots = await repos.bots.get_active_bots_by_mode("shadow")
        assert {b["id"] for b in shadow_bots} == {"shadow-1"}

        # Live filter still works unchanged
        live_bots = await repos.bots.get_active_bots_by_mode("live")
        assert {b["id"] for b in live_bots} == {"live-1"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:

    @pytest.mark.asyncio
    async def test_get_repositories_sqlite(self, tmp_path):
        os.environ["SQLITE_DB_PATH"] = str(tmp_path / "factory_test.db")
        from storage.repositories import get_repositories

        repos = await get_repositories("sqlite")
        assert hasattr(repos, "trades")
        assert hasattr(repos, "cycles")
        assert hasattr(repos, "rules")
        assert hasattr(repos, "bots")
        assert hasattr(repos, "cross_bot")
        assert hasattr(repos, "oi_snapshots")

        # Clean up
        del os.environ["SQLITE_DB_PATH"]

    @pytest.mark.asyncio
    async def test_get_repositories_unknown_backend(self):
        from storage.repositories import get_repositories

        with pytest.raises(ValueError, match="Unknown database backend"):
            await get_repositories("mongodb")
