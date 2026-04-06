"""Unit tests for the 4 memory loops + build_memory_context."""

from __future__ import annotations

import pytest
import pytest_asyncio

from engine.memory import build_memory_context
from engine.memory.cross_bot import CrossBotSignals
from engine.memory.cycle_memory import CycleMemory
from engine.memory.reflection_rules import ReflectionRules
from engine.memory.regime_history import RegimeHistory
from storage.repositories.sqlite import SQLiteRepositories


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def repos(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    r = SQLiteRepositories(db_path=db_path)
    await r.init_db()
    return r


# ---------------------------------------------------------------------------
# Loop 1: CycleMemory
# ---------------------------------------------------------------------------

class TestCycleMemory:

    @pytest.mark.asyncio
    async def test_save_and_get_recent(self, repos) -> None:
        mem = CycleMemory(repos.cycles)

        await mem.save_cycle("bot1", {
            "symbol": "BTC-USDC", "timeframe": "1h",
            "action": "LONG", "conviction_score": 0.72,
            "timestamp": "2026-04-06T10:00:00Z",
        })
        await mem.save_cycle("bot1", {
            "symbol": "BTC-USDC", "timeframe": "1h",
            "action": "SKIP", "conviction_score": 0.35,
            "timestamp": "2026-04-06T11:00:00Z",
        })

        recent = await mem.get_recent("bot1", limit=5)
        assert len(recent) == 2
        # Ordered by timestamp desc
        assert recent[0]["action"] == "SKIP"

    @pytest.mark.asyncio
    async def test_get_recent_empty(self, repos) -> None:
        mem = CycleMemory(repos.cycles)
        recent = await mem.get_recent("nonexistent")
        assert recent == []

    @pytest.mark.asyncio
    async def test_format_for_prompt_with_data(self, repos) -> None:
        mem = CycleMemory(repos.cycles)
        cycles = [
            {"timestamp": "2026-04-06T10:00:00Z", "action": "LONG", "conviction_score": 0.72},
            {"timestamp": "2026-04-06T09:00:00Z", "action": "SKIP", "conviction_score": 0.35},
        ]
        result = mem.format_for_prompt(cycles)
        assert "Recent cycles:" in result
        assert "LONG" in result
        assert "0.72" in result
        assert "SKIP" in result

    @pytest.mark.asyncio
    async def test_format_for_prompt_empty(self, repos) -> None:
        mem = CycleMemory(repos.cycles)
        result = mem.format_for_prompt([])
        assert result == "No prior cycles."


# ---------------------------------------------------------------------------
# Loop 2: ReflectionRules
# ---------------------------------------------------------------------------

class TestReflectionRules:

    @pytest.mark.asyncio
    async def test_save_and_get_active_rules(self, repos) -> None:
        rules = ReflectionRules(repos.rules)

        await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Avoid LONG when RSI > 65 and funding > 0.03%",
        })
        await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Strong LONG only when all 3 agents agree",
        })

        active = await rules.get_active_rules("BTC-USDC", "1h")
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_increment_score(self, repos) -> None:
        rules = ReflectionRules(repos.rules)
        rule_id = await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Test rule", "score": 0,
        })

        await rules.increment_score(rule_id)
        await rules.increment_score(rule_id)

        active = await rules.get_active_rules("BTC-USDC", "1h")
        rule = [r for r in active if r["id"] == rule_id][0]
        assert rule["score"] == 2

    @pytest.mark.asyncio
    async def test_decrement_score(self, repos) -> None:
        rules = ReflectionRules(repos.rules)
        rule_id = await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Bad rule", "score": 0,
        })

        await rules.decrement_score(rule_id)

        active = await rules.get_active_rules("BTC-USDC", "1h")
        rule = [r for r in active if r["id"] == rule_id][0]
        assert rule["score"] == -1

    @pytest.mark.asyncio
    async def test_auto_deactivation_on_low_score(self, repos) -> None:
        """Rule auto-deactivates when score drops below -2 (handled by repo)."""
        rules = ReflectionRules(repos.rules)
        rule_id = await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Will be deactivated", "score": 0,
        })

        await rules.decrement_score(rule_id)
        await rules.decrement_score(rule_id)
        await rules.decrement_score(rule_id)  # score = -3, auto-deactivated

        active = await rules.get_active_rules("BTC-USDC", "1h")
        assert all(r["id"] != rule_id for r in active)

    @pytest.mark.asyncio
    async def test_manual_deactivation(self, repos) -> None:
        rules = ReflectionRules(repos.rules)
        rule_id = await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "To be manually deactivated",
        })

        await rules.deactivate_rule(rule_id)

        active = await rules.get_active_rules("BTC-USDC", "1h")
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_format_for_prompt_with_rules(self, repos) -> None:
        rules = ReflectionRules(repos.rules)
        data = [
            {"score": 3, "rule_text": "Avoid LONG when RSI > 65"},
            {"score": -1, "rule_text": "Strong LONG when all agree"},
        ]
        result = rules.format_for_prompt(data)
        assert "Learned rules:" in result
        assert "[score=3]" in result
        assert "Avoid LONG" in result

    @pytest.mark.asyncio
    async def test_format_for_prompt_empty(self, repos) -> None:
        rules = ReflectionRules(repos.rules)
        result = rules.format_for_prompt([])
        assert result == "No learned rules for this asset."


# ---------------------------------------------------------------------------
# Loop 3: CrossBotSignals
# ---------------------------------------------------------------------------

class TestCrossBotSignals:

    @pytest.mark.asyncio
    async def test_publish_and_get_signals(self, repos) -> None:
        cross = CrossBotSignals(repos.cross_bot)

        await cross.publish_signal("user1", "bot1", "BTC-USDC", "LONG", 0.85)
        await cross.publish_signal("user1", "bot2", "BTC-USDC", "SHORT", 0.72)

        signals = await cross.get_other_bot_signals("BTC-USDC", "user1")
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_user_id_isolation(self, repos) -> None:
        """User A's signals MUST NOT be visible to User B."""
        cross = CrossBotSignals(repos.cross_bot)

        await cross.publish_signal("alice", "bot1", "BTC-USDC", "LONG", 0.9)
        await cross.publish_signal("bob", "bot1", "BTC-USDC", "SHORT", 0.8)

        alice_signals = await cross.get_other_bot_signals("BTC-USDC", "alice")
        assert len(alice_signals) == 1
        assert alice_signals[0]["direction"] == "LONG"

        bob_signals = await cross.get_other_bot_signals("BTC-USDC", "bob")
        assert len(bob_signals) == 1
        assert bob_signals[0]["direction"] == "SHORT"

    @pytest.mark.asyncio
    async def test_symbol_filtering(self, repos) -> None:
        cross = CrossBotSignals(repos.cross_bot)

        await cross.publish_signal("user1", "bot1", "BTC-USDC", "LONG", 0.85)
        await cross.publish_signal("user1", "bot2", "ETH-USDC", "SHORT", 0.72)

        btc = await cross.get_other_bot_signals("BTC-USDC", "user1")
        assert len(btc) == 1
        assert btc[0]["symbol"] == "BTC-USDC"

    @pytest.mark.asyncio
    async def test_format_for_prompt_with_signals(self, repos) -> None:
        cross = CrossBotSignals(repos.cross_bot)
        data = [
            {"bot_id": "bot_4h", "direction": "SHORT", "conviction": 0.85, "timestamp": "2026-04-06T10:00:00Z"},
        ]
        result = cross.format_for_prompt(data)
        assert "Cross-bot signals:" in result
        assert "bot_4h" in result
        assert "SHORT" in result
        assert "0.85" in result

    @pytest.mark.asyncio
    async def test_format_for_prompt_empty(self, repos) -> None:
        cross = CrossBotSignals(repos.cross_bot)
        result = cross.format_for_prompt([])
        assert result == "No signals from other bots."


# ---------------------------------------------------------------------------
# Loop 4: RegimeHistory
# ---------------------------------------------------------------------------

class TestRegimeHistory:

    def test_add_and_get_history(self) -> None:
        rh = RegimeHistory(max_size=20)
        rh.add("TRENDING_UP", 0.82)
        rh.add("TRENDING_UP", 0.78)
        rh.add("RANGING", 0.65)

        history = rh.get_history()
        assert len(history) == 3
        assert history[0]["regime"] == "TRENDING_UP"
        assert history[2]["regime"] == "RANGING"

    def test_ring_buffer_overflow(self) -> None:
        rh = RegimeHistory(max_size=5)
        for i in range(10):
            rh.add(f"REGIME_{i}", 0.5)

        history = rh.get_history()
        assert len(history) == 5
        # Should have the last 5 entries
        assert history[0]["regime"] == "REGIME_5"
        assert history[4]["regime"] == "REGIME_9"

    def test_detect_transition_changed(self) -> None:
        rh = RegimeHistory()
        rh.add("RANGING", 0.7)
        rh.add("BREAKOUT", 0.8)

        transition = rh.detect_transition()
        assert transition == "RANGING -> BREAKOUT"

    def test_detect_transition_no_change(self) -> None:
        rh = RegimeHistory()
        rh.add("TRENDING_UP", 0.8)
        rh.add("TRENDING_UP", 0.82)

        transition = rh.detect_transition()
        assert transition is None

    def test_detect_transition_single_entry(self) -> None:
        rh = RegimeHistory()
        rh.add("RANGING", 0.5)

        transition = rh.detect_transition()
        assert transition is None

    def test_detect_transition_empty(self) -> None:
        rh = RegimeHistory()
        assert rh.detect_transition() is None

    def test_current_regime(self) -> None:
        rh = RegimeHistory()
        rh.add("TRENDING_UP", 0.8)
        rh.add("RANGING", 0.65)

        assert rh.current_regime() == "RANGING"

    def test_current_regime_empty(self) -> None:
        rh = RegimeHistory()
        assert rh.current_regime() is None

    def test_regime_streak(self) -> None:
        rh = RegimeHistory()
        rh.add("RANGING", 0.7)
        rh.add("TRENDING_UP", 0.8)
        rh.add("TRENDING_UP", 0.82)
        rh.add("TRENDING_UP", 0.85)

        assert rh.regime_streak() == 3

    def test_regime_streak_single(self) -> None:
        rh = RegimeHistory()
        rh.add("BREAKOUT", 0.9)

        assert rh.regime_streak() == 1

    def test_regime_streak_empty(self) -> None:
        rh = RegimeHistory()
        assert rh.regime_streak() == 0

    def test_format_for_prompt_with_data(self) -> None:
        rh = RegimeHistory()
        rh.add("TRENDING_UP", 0.8)
        rh.add("TRENDING_UP", 0.82)
        rh.add("RANGING", 0.65)

        result = rh.format_for_prompt()
        assert "Regime history" in result
        assert "TRENDING_UP" in result
        assert "RANGING" in result
        assert "REGIME TRANSITION: TRENDING_UP -> RANGING" in result

    def test_format_for_prompt_no_transition(self) -> None:
        rh = RegimeHistory()
        rh.add("TRENDING_UP", 0.8)
        rh.add("TRENDING_UP", 0.82)

        result = rh.format_for_prompt()
        assert "Regime history" in result
        assert "REGIME TRANSITION" not in result

    def test_format_for_prompt_empty(self) -> None:
        rh = RegimeHistory()
        result = rh.format_for_prompt()
        assert result == "No regime history."

    def test_format_for_prompt_limits_to_5(self) -> None:
        rh = RegimeHistory()
        for i in range(10):
            rh.add("TRENDING_UP", 0.5 + i * 0.05)

        result = rh.format_for_prompt()
        # Should show "last 5" in the header
        assert "last 5" in result
        # Count the bullet points
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# build_memory_context integration
# ---------------------------------------------------------------------------

class TestBuildMemoryContext:

    @pytest.mark.asyncio
    async def test_assembles_all_4_parts(self, repos) -> None:
        cycle_mem = CycleMemory(repos.cycles)
        rules = ReflectionRules(repos.rules)
        cross_bot = CrossBotSignals(repos.cross_bot)
        regime = RegimeHistory()

        # Populate some data
        await cycle_mem.save_cycle("bot1", {
            "symbol": "BTC-USDC", "timeframe": "1h",
            "action": "LONG", "conviction_score": 0.72,
        })
        await rules.save_rule({
            "symbol": "BTC-USDC", "timeframe": "1h",
            "rule_text": "Avoid LONG when RSI > 65",
        })
        await cross_bot.publish_signal("user1", "bot2", "BTC-USDC", "SHORT", 0.8)
        regime.add("TRENDING_UP", 0.82)

        context = await build_memory_context(
            cycle_mem, rules, cross_bot, regime,
            bot_id="bot1", symbol="BTC-USDC", timeframe="1h", user_id="user1",
        )

        assert "Recent cycles:" in context
        assert "LONG" in context
        assert "Learned rules:" in context
        assert "Avoid LONG when RSI > 65" in context
        assert "Cross-bot signals:" in context
        assert "SHORT" in context
        assert "Regime history" in context
        assert "TRENDING_UP" in context

    @pytest.mark.asyncio
    async def test_all_empty(self, repos) -> None:
        cycle_mem = CycleMemory(repos.cycles)
        rules = ReflectionRules(repos.rules)
        cross_bot = CrossBotSignals(repos.cross_bot)
        regime = RegimeHistory()

        context = await build_memory_context(
            cycle_mem, rules, cross_bot, regime,
            bot_id="bot1", symbol="BTC-USDC", timeframe="1h", user_id="user1",
        )

        assert "No prior cycles." in context
        assert "No learned rules for this asset." in context
        assert "No signals from other bots." in context
        assert "No regime history." in context

    @pytest.mark.asyncio
    async def test_parts_separated_by_double_newline(self, repos) -> None:
        cycle_mem = CycleMemory(repos.cycles)
        rules = ReflectionRules(repos.rules)
        cross_bot = CrossBotSignals(repos.cross_bot)
        regime = RegimeHistory()

        context = await build_memory_context(
            cycle_mem, rules, cross_bot, regime,
            bot_id="bot1", symbol="BTC-USDC", timeframe="1h", user_id="user1",
        )

        # Should have 3 double-newline separators between 4 parts
        assert context.count("\n\n") >= 3
