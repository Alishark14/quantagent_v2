"""Tests for `quantagent.main._make_bot_factory`.

The factory closure is the connective tissue between BotRunner /
BotManager and the engine pipeline. Before this test existed, the
factory was a `raise NotImplementedError` stub, so Sentinel could fire
SetupDetected events forever and BotManager could never spawn a bot.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.config import FeatureFlags
from engine.events import InProcessBus
from engine.trader_bot import TraderBot
from llm.base import LLMProvider, LLMResponse
from quantagent.main import _make_bot_factory


class _StubLLM(LLMProvider):
    """No-network LLM stub. The factory only stores it; we never call it."""

    async def generate_text(self, system_prompt, user_prompt, agent_name,
                            max_tokens=1024, temperature=0.3,
                            cache_system_prompt=True):
        return LLMResponse(
            content="", input_tokens=0, output_tokens=0, cost=0.0,
            model="stub", latency_ms=0.0, cached_input_tokens=0,
        )

    async def generate_vision(self, system_prompt, user_prompt, image_data,
                              image_media_type, agent_name, max_tokens=1024,
                              temperature=0.3, cache_system_prompt=True):
        return LLMResponse(
            content="", input_tokens=0, output_tokens=0, cost=0.0,
            model="stub", latency_ms=0.0, cached_input_tokens=0,
        )


def _make_repos_stub():
    """Build a repos namespace with the four repositories the pipeline reads."""
    repos = MagicMock()
    # The memory wrappers (CycleMemory, ReflectionRules, CrossBotSignals)
    # accept any object that quacks like the matching repo. MagicMock is
    # fine — the pipeline never executes inside this test.
    return repos


def _make_adapter_stub():
    """Minimal adapter stub. The factory only passes it through, never calls it."""
    return MagicMock(name="exchange-adapter-stub")


def test_factory_returns_traderbot_for_symbol():
    """The whole reason this factory exists: turn (symbol, bot_id) into
    a fully-wired TraderBot that BotManager can `await bot.run()`."""
    bus = InProcessBus()
    captured: list[str] = []

    def adapter_factory(exchange: str):
        captured.append(exchange)
        return _make_adapter_stub()

    factory = _make_bot_factory(
        repos=_make_repos_stub(),
        llm_provider=_StubLLM(),
        adapter_factory=adapter_factory,
        event_bus=bus,
        feature_flags=FeatureFlags(),
    )

    bot = factory("BTC-USDC", "test-bot-001")
    assert isinstance(bot, TraderBot)
    assert bot.bot_id == "test-bot-001"
    # adapter_factory was called for the bot's exchange (default: hyperliquid)
    assert captured == ["hyperliquid"]


def test_factory_pipeline_carries_symbol_and_bot_id():
    """The pipeline must be scoped to (symbol, bot_id) so the data moat
    captures the right keys for each cycle."""
    factory = _make_bot_factory(
        repos=_make_repos_stub(),
        llm_provider=_StubLLM(),
        adapter_factory=lambda _ex: _make_adapter_stub(),
        event_bus=InProcessBus(),
        feature_flags=FeatureFlags(),
    )
    bot = factory("ETH-USDC", "bot-eth-42")
    assert bot._pipeline._config.symbol == "ETH-USDC"
    assert bot._pipeline._bot_id == "bot-eth-42"


def test_factory_registers_only_enabled_signal_agents(tmp_path):
    """FeatureFlags gates each signal agent. Disabling a flag must keep
    that agent OUT of the registry — operators rely on this for cost
    control and A/B sweeps."""
    flag_file = tmp_path / "features.yaml"
    flag_file.write_text(
        "indicator_agent: true\n"
        "pattern_agent: false\n"
        "trend_agent: false\n"
        "flow_signal_agent: true\n"
    )
    flags = FeatureFlags(yaml_path=flag_file)

    factory = _make_bot_factory(
        repos=_make_repos_stub(),
        llm_provider=_StubLLM(),
        adapter_factory=lambda _ex: _make_adapter_stub(),
        event_bus=InProcessBus(),
        feature_flags=flags,
    )
    bot = factory("BTC-USDC", "test-bot-flags")
    registry = bot._pipeline._signal_registry
    names = {p.name() for p in registry._producers}
    assert "indicator_agent" in names
    assert "flow_signal_agent" in names
    assert "pattern_agent" not in names
    assert "trend_agent" not in names


def test_factory_calls_adapter_factory_per_invocation():
    """Each spawn must re-fetch the adapter from the factory so shadow-mode
    swap (sim adapter with data delegate) is honoured even if the runner
    re-builds bots mid-process."""
    calls = {"count": 0}

    def adapter_factory(_exchange):
        calls["count"] += 1
        return _make_adapter_stub()

    factory = _make_bot_factory(
        repos=_make_repos_stub(),
        llm_provider=_StubLLM(),
        adapter_factory=adapter_factory,
        event_bus=InProcessBus(),
        feature_flags=FeatureFlags(),
    )
    factory("BTC-USDC", "b1")
    factory("ETH-USDC", "b2")
    assert calls["count"] == 2
