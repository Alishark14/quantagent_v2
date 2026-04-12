"""Tests for sentinel_events and llm_calls analytics tables.

Task E from Sprint Week 7 Update 2.
"""

from __future__ import annotations

import pytest


# ── Sentinel Events ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sentinel_event_inserted_on_setup_detected(tmp_path):
    """After SetupDetected, a 'setup_detected' row exists in sentinel_events."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.sentinel_events.insert_event({
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "event_type": "setup_detected",
        "readiness_score": 0.85,
        "threshold": 0.70,
        "triggers_today": 3,
        "reasoning": "RSI cross + level touch",
    })

    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sentinel_events") as cur:
            rows = [dict(r) async for r in cur]

    assert len(rows) == 1
    assert rows[0]["event_type"] == "setup_detected"
    assert rows[0]["readiness_score"] == 0.85
    assert rows[0]["threshold"] == 0.70
    assert rows[0]["triggers_today"] == 3


@pytest.mark.asyncio
async def test_sentinel_event_inserted_on_skip(tmp_path):
    """SKIP events are also persisted."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.sentinel_events.insert_event({
        "symbol": "ETH-USDC",
        "timeframe": "1h",
        "event_type": "skip",
        "readiness_score": 0.55,
        "threshold": 0.70,
        "reasoning": "Below threshold",
    })

    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sentinel_events WHERE event_type = 'skip'"
        ) as cur:
            rows = [dict(r) async for r in cur]

    assert len(rows) == 1
    assert rows[0]["symbol"] == "ETH-USDC"


@pytest.mark.asyncio
async def test_sentinel_event_fire_and_forget():
    """DB error in sentinel_event insert must not crash the caller."""
    from sentinel.monitor import SentinelMonitor
    from engine.events import InProcessBus
    from unittest.mock import AsyncMock, MagicMock

    # Create a repo that always raises
    bad_repo = MagicMock()
    bad_repo.insert_event = AsyncMock(side_effect=Exception("DB down"))

    adapter = MagicMock()
    sentinel = SentinelMonitor(
        adapter=adapter,
        event_bus=InProcessBus(),
        symbol="BTC-USDC",
        sentinel_event_repo=bad_repo,
    )

    # Should not raise
    await sentinel._record_sentinel_event(
        "setup_detected", 0.85, 0.70, "test"
    )
    # Verify it was called (even though it failed)
    assert bad_repo.insert_event.called


# ── LLM Calls ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_call_inserted(tmp_path):
    """An LLM call record is persisted with tokens, latency, model."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    await repos.llm_calls.insert_call({
        "agent_name": "indicator_agent",
        "model": "claude-sonnet-4-20250514",
        "input_tokens": 5000,
        "output_tokens": 800,
        "cost_usd": 0.027,
        "latency_ms": 1200,
        "cache_hit": True,
    })

    calls = await repos.llm_calls.get_calls_by_agent("indicator_agent")
    assert len(calls) == 1
    assert calls[0]["input_tokens"] == 5000
    assert calls[0]["output_tokens"] == 800
    assert calls[0]["latency_ms"] == 1200
    assert calls[0]["model"] == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_llm_calls_queryable_by_agent(tmp_path):
    """get_calls_by_agent only returns calls for the requested agent."""
    from storage.repositories.sqlite import SQLiteRepositories

    db_path = str(tmp_path / "test.db")
    repos = SQLiteRepositories(db_path=db_path)
    await repos.init_db()

    for agent in ["indicator_agent", "pattern_agent", "conviction_agent"]:
        await repos.llm_calls.insert_call({
            "agent_name": agent,
            "model": "claude-sonnet-4-20250514",
            "input_tokens": 1000,
            "output_tokens": 200,
            "cost_usd": 0.01,
            "latency_ms": 500,
        })

    indicator_calls = await repos.llm_calls.get_calls_by_agent("indicator_agent")
    assert len(indicator_calls) == 1
    assert indicator_calls[0]["agent_name"] == "indicator_agent"

    conviction_calls = await repos.llm_calls.get_calls_by_agent("conviction_agent")
    assert len(conviction_calls) == 1


@pytest.mark.asyncio
async def test_llm_call_fire_and_forget():
    """DB error in llm_call insert must not crash the LLM provider."""
    from llm.claude import ClaudeProvider
    from unittest.mock import AsyncMock, MagicMock, patch

    bad_repo = MagicMock()
    bad_repo.insert_call = AsyncMock(side_effect=Exception("DB down"))

    provider = ClaudeProvider.__new__(ClaudeProvider)
    provider.model = "test-model"
    provider._llm_call_repo = bad_repo

    # Mock the Anthropic client to return a fake response
    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50
    mock_usage.cache_read_input_tokens = 0

    mock_content = MagicMock()
    mock_content.text = "test response"

    mock_response = MagicMock()
    mock_response.usage = mock_usage
    mock_response.content = [mock_content]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    provider.client = mock_client

    # Should not raise even though the repo insert fails
    result = await provider._call(
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "hi"}],
        agent_name="test_agent",
        max_tokens=100,
        temperature=0.3,
    )
    assert result.content == "test response"
    # Verify insert was attempted
    assert bad_repo.insert_call.called
