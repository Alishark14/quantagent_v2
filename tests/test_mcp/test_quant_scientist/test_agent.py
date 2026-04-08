"""Tests for the QuantDataScientist agent class.

The agent has a lot of moving parts (LLM call, sandbox execution,
trade fetch, decay, merge, write) — these tests use small fakes to
exercise the orchestration without spinning up a real Claude or
PostgreSQL.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm.base import LLMResponse
from mcp.quant_scientist.agent import QuantDataScientist
from mcp.quant_scientist.factor import (
    AlphaFactor,
    factors_to_nested_json,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Pretends to be an LLMProvider — returns a fixed response."""

    def __init__(self, content: str = "", raise_on_call: bool = False) -> None:
        self.content = content
        self.raise_on_call = raise_on_call
        self.calls: list[dict] = []

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "agent_name": agent_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.raise_on_call:
            raise RuntimeError("simulated LLM blowup")
        return LLMResponse(
            content=self.content,
            input_tokens=1000,
            output_tokens=400,
            cost=0.01,
            model="claude-sonnet-test",
            latency_ms=120.0,
            cached_input_tokens=0,
        )

    async def generate_vision(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _FakeTradeRepo:
    def __init__(self, trades: list[dict]) -> None:
        self._trades = trades

    async def get_trades_by_bot(self, bot_id: str, limit: int = 50):
        return list(self._trades)[:limit]

    async def get_trade(self, trade_id: str):  # pragma: no cover
        return None

    async def update_trade(self, trade_id, updates):  # pragma: no cover
        return True


def _trade(
    trade_id: str = "t1",
    pattern: str = "asc_triangle",
    symbol: str = "BTC-USDC",
    timeframe: str = "1h",
    pnl: float = 100.0,
    days_ago: int = 1,
) -> dict:
    entry = (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "id": trade_id,
        "bot_id": "bot-a",
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": "LONG",
        "pattern": pattern,
        "pnl": pnl,
        "r_multiple": 1.7 if pnl > 0 else -1.0,
        "entry_time": entry,
        "exit_time": entry,
        "status": "closed",
        "exit_reason": "tp1",
    }


# Boilerplate code that the LLM "would have written"
_LLM_CODE_OK = """\
result = [
    {
        "pattern": "asc_triangle",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "win_rate": 0.65,
        "avg_r": 1.85,
        "n": 22,
        "confidence": "high",
        "note": None,
    },
    {
        "pattern": "bearish_flag",
        "symbol": "ETH-USDC",
        "timeframe": "1h",
        "win_rate": 0.30,
        "avg_r": -0.4,
        "n": 18,
        "confidence": "high",
        "note": "AVOID",
    },
]
"""


def _llm_response(code: str = _LLM_CODE_OK, language_tag: str = "python") -> str:
    return f"Here's the analysis:\n\n```{language_tag}\n{code}\n```\n"


# ---------------------------------------------------------------------------
# Happy path — full run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_writes_alpha_factors_when_llm_returns_valid_code(tmp_path):
    output = tmp_path / "alpha_factors.json"
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response()),
        trade_repository=_FakeTradeRepo([_trade(), _trade("t2")]),
        output_path=output,
        bot_ids=["bot-a"],
    )
    report = await agent.run(dry_run=False)

    assert report.error is None
    assert report.factor_count == 2
    assert report.new_count == 2
    assert report.trades_analyzed == 2
    assert report.symbols_analyzed == 1
    assert output.exists()

    persisted = json.loads(output.read_text())
    assert "BTC-USDC" in persisted
    assert "ETH-USDC" in persisted
    assert "asc_triangle" in persisted["BTC-USDC"]["1h"]
    assert persisted["ETH-USDC"]["1h"]["bearish_flag"]["note"] == "AVOID"


@pytest.mark.asyncio
async def test_run_dry_run_does_not_write_file(tmp_path):
    output = tmp_path / "alpha_factors.json"
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response()),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=output,
        bot_ids=["bot-a"],
    )
    report = await agent.run(dry_run=True)
    assert report.error is None
    assert report.dry_run is True
    assert report.factor_count == 2
    assert not output.exists()


@pytest.mark.asyncio
async def test_run_writes_atomically_via_tmp(tmp_path):
    """Verify the agent uses an atomic tmp+rename write."""
    output = tmp_path / "alpha_factors.json"
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response()),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=output,
        bot_ids=["bot-a"],
    )
    await agent.run()
    assert output.exists()
    # No leftover .tmp file
    assert not (tmp_path / "alpha_factors.json.tmp").exists()


# ---------------------------------------------------------------------------
# Safety — output path is fixed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_only_writes_to_configured_output_path(tmp_path):
    """The agent never writes to anywhere except `output_path`."""
    output = tmp_path / "subdir" / "alpha_factors.json"
    other = tmp_path / "should_not_exist.json"
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response()),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=output,
        bot_ids=["bot-a"],
    )
    await agent.run()
    assert output.exists()
    assert not other.exists()
    # Subdir was created
    assert output.parent.is_dir()


@pytest.mark.asyncio
async def test_run_rejects_llm_code_with_forbidden_imports(tmp_path):
    """Sandbox screen catches `import os` even if the LLM tries it."""
    bad_code = "import os\nresult = []\n"
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response(code=bad_code)),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=tmp_path / "alpha_factors.json",
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.error is not None
    assert "sandbox_rejected" in report.error
    assert not (tmp_path / "alpha_factors.json").exists()


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_handles_llm_call_failure_gracefully(tmp_path):
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(raise_on_call=True),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=tmp_path / "alpha_factors.json",
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.error is not None
    assert "llm_call_failed" in report.error
    assert not (tmp_path / "alpha_factors.json").exists()


@pytest.mark.asyncio
async def test_run_handles_no_code_block_in_llm_response(tmp_path):
    """LLM returned prose with no Python code → graceful error report."""
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content="Sorry, I can't help with that."),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=tmp_path / "alpha_factors.json",
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.error == "llm_no_code_block"


@pytest.mark.asyncio
async def test_run_handles_sandbox_runtime_error(tmp_path):
    """LLM code that raises is caught and reported."""
    bad = "result = []\nx = 1 / 0\n"
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response(code=bad)),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=tmp_path / "alpha_factors.json",
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.error is not None
    assert "sandbox_execution_failed" in report.error


@pytest.mark.asyncio
async def test_run_handles_invalid_result_rows(tmp_path):
    """Rows that fail AlphaFactor validation are dropped, run still completes."""
    half_bad = """\
result = [
    {"pattern": "good", "symbol": "BTC-USDC", "timeframe": "1h",
     "win_rate": 0.65, "avg_r": 1.85, "n": 22, "confidence": "high"},
    {"pattern": "bad", "symbol": "BTC-USDC", "timeframe": "1h",
     "win_rate": 9.0, "avg_r": 1.85, "n": 22, "confidence": "high"},
    {"pattern": "missing_n", "symbol": "BTC-USDC", "timeframe": "1h",
     "win_rate": 0.5, "avg_r": 1.5, "confidence": "high"},
]
"""
    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response(code=half_bad)),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=tmp_path / "alpha_factors.json",
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.error is None
    assert report.factor_count == 1
    assert report.factors[0].pattern == "good"


@pytest.mark.asyncio
async def test_run_with_no_trades_short_circuits_cleanly(tmp_path):
    """No trades → no LLM call, decay-only pass on existing factors."""
    fake_llm = _FakeLLM(content=_llm_response())
    agent = QuantDataScientist(
        llm_provider=fake_llm,
        trade_repository=_FakeTradeRepo([]),
        output_path=tmp_path / "alpha_factors.json",
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.error is None
    assert report.trades_analyzed == 0
    # Should NOT have called the LLM when there's nothing to analyse.
    assert fake_llm.calls == []
    # File written but empty-ish
    assert (tmp_path / "alpha_factors.json").exists()


# ---------------------------------------------------------------------------
# Decay + merge integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_merges_with_existing_factors(tmp_path):
    """Existing factor on disk gets re-confirmed when LLM returns the same key."""
    output = tmp_path / "alpha_factors.json"
    discovered = "2026-04-01T00:00:00+00:00"
    existing = AlphaFactor(
        pattern="asc_triangle",
        symbol="BTC-USDC",
        timeframe="1h",
        win_rate=0.55,
        avg_r=1.6,
        n=18,
        confidence="medium",
        discovered_at=discovered,
        last_confirmed=discovered,
        decay_weight=1.0,  # Will decay quickly because last_confirmed is days old
    )
    output.write_text(json.dumps(factors_to_nested_json([existing])))

    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response()),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=output,
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.confirmed_count == 1  # asc_triangle was re-confirmed
    assert report.new_count == 1  # bearish_flag is brand new

    persisted = json.loads(output.read_text())
    leaf = persisted["BTC-USDC"]["1h"]["asc_triangle"]
    assert leaf["discovered_at"] == discovered  # preserved
    assert leaf["decay_weight"] == 1.0


@pytest.mark.asyncio
async def test_run_prunes_aged_existing_factor_not_in_new_batch(tmp_path):
    """Existing factor older than horizon is pruned even if LLM doesn't know about it."""
    output = tmp_path / "alpha_factors.json"
    long_ago = (datetime.now(tz=timezone.utc) - timedelta(days=40)).isoformat()
    aged = AlphaFactor(
        pattern="dying",
        symbol="BTC-USDC",
        timeframe="1h",
        win_rate=0.55,
        avg_r=1.6,
        n=18,
        confidence="medium",
        discovered_at=long_ago,
        last_confirmed=long_ago,
        decay_weight=1.0,  # Will compute to 0 after decay
    )
    output.write_text(json.dumps(factors_to_nested_json([aged])))

    agent = QuantDataScientist(
        llm_provider=_FakeLLM(content=_llm_response()),
        trade_repository=_FakeTradeRepo([_trade()]),
        output_path=output,
        bot_ids=["bot-a"],
    )
    report = await agent.run()
    assert report.pruned_count >= 1
    persisted = json.loads(output.read_text())
    assert "dying" not in persisted.get("BTC-USDC", {}).get("1h", {})


# ---------------------------------------------------------------------------
# Helpers — _extract_code, _summarise_trades
# ---------------------------------------------------------------------------


def test_extract_code_pulls_python_block():
    response = "Here:\n```python\nresult = []\n```\nCheers!"
    assert QuantDataScientist._extract_code(response) == "result = []"


def test_extract_code_pulls_unlabeled_block():
    response = "Here:\n```\nresult = []\n```"
    assert QuantDataScientist._extract_code(response) == "result = []"


def test_extract_code_returns_none_for_prose():
    assert QuantDataScientist._extract_code("Sorry, can't help.") is None


def test_extract_code_falls_back_to_unfenced_python():
    """If the LLM omits the fences but the body looks like Python, use it."""
    response = "result = [{'pattern': 'x', 'symbol': 'BTC', 'timeframe': '1h'}]"
    code = QuantDataScientist._extract_code(response)
    assert code is not None
    assert "result = " in code


def test_summarise_trades_handles_empty():
    s = QuantDataScientist._summarise_trades([])
    assert s == {"trade_count": 0, "unique_symbols": 0, "win_rate": 0.0, "avg_r": 0.0}


def test_summarise_trades_computes_win_rate_and_avg_r():
    trades = [
        _trade(pnl=100.0),
        _trade(pnl=-50.0),
        _trade(pnl=80.0),
        _trade(pnl=-20.0),
    ]
    s = QuantDataScientist._summarise_trades(trades)
    assert s["trade_count"] == 4
    assert s["unique_symbols"] == 1
    assert s["win_rate"] == pytest.approx(0.5)
    # r_multiples are 1.7 for winners, -1.0 for losers
    assert s["avg_r"] == pytest.approx(0.35)


def test_parse_result_drops_invalid_rows():
    rows = [
        {  # OK
            "pattern": "ok", "symbol": "BTC-USDC", "timeframe": "1h",
            "win_rate": 0.6, "avg_r": 1.7, "n": 20, "confidence": "high",
        },
        {  # Bad win_rate
            "pattern": "bad", "symbol": "BTC-USDC", "timeframe": "1h",
            "win_rate": 9.0, "avg_r": 1.7, "n": 20, "confidence": "high",
        },
        {  # Missing avg_r
            "pattern": "missing", "symbol": "BTC-USDC", "timeframe": "1h",
            "win_rate": 0.6, "n": 20, "confidence": "high",
        },
    ]
    parsed = QuantDataScientist._parse_result(rows)
    assert len(parsed) == 1
    assert parsed[0].pattern == "ok"
