"""Tests for the Quant Data Scientist CLI runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm.base import LLMResponse
from mcp.quant_scientist import runner
from mcp.quant_scientist.factor import AlphaFactorsReport


# ---------------------------------------------------------------------------
# Argparse coverage
# ---------------------------------------------------------------------------


def test_parser_defaults():
    args = runner._build_parser().parse_args([])
    assert args.dry_run is False
    assert args.db_url is None
    assert args.output == "alpha_factors.json"
    assert args.bot_id is None
    assert args.lookback_days == 30
    assert args.ohlcv_lookback_days == 180
    assert args.no_ohlcv is False


def test_parser_dry_run_flag():
    args = runner._build_parser().parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parser_repeated_bot_id():
    args = runner._build_parser().parse_args(["--bot-id", "a", "--bot-id", "b"])
    assert args.bot_id == ["a", "b"]


def test_parser_repeated_timeframe():
    args = runner._build_parser().parse_args(
        ["--timeframe", "1h", "--timeframe", "4h"]
    )
    assert args.timeframe == ["1h", "4h"]


def test_parser_help_does_not_crash():
    parser = runner._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# main() — error paths
# ---------------------------------------------------------------------------


def test_main_negative_lookback_returns_2(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    code = runner.main(["--lookback-days", "0"])
    assert code == 2


def test_main_missing_api_key_returns_2(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = runner.main(["--no-ohlcv"])
    assert code == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err


# ---------------------------------------------------------------------------
# main() — end-to-end with mocks
# ---------------------------------------------------------------------------


_LLM_CODE = """\
result = [
    {
        "pattern": "asc_triangle",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "win_rate": 0.65,
        "avg_r": 1.85,
        "n": 22,
        "confidence": "high",
    }
]
"""


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_text(self, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=f"```python\n{_LLM_CODE}\n```",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            model="claude-sonnet-test",
            latency_ms=10.0,
            cached_input_tokens=0,
        )

    async def generate_vision(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _FakeTradeRepo:
    def __init__(self, trades: list[dict]) -> None:
        self._trades = trades

    async def get_trades_by_bot(self, bot_id: str, limit: int = 50):
        return list(self._trades)[:limit]


def _trade_dict():
    return {
        "id": "t1",
        "bot_id": "bot-a",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "direction": "LONG",
        "pattern": "asc_triangle",
        "pnl": 100.0,
        "r_multiple": 1.7,
        "entry_time": "2026-04-07T00:00:00+00:00",
        "exit_time": "2026-04-07T01:00:00+00:00",
        "status": "closed",
        "exit_reason": "tp1",
    }


def test_main_end_to_end_with_mocks(monkeypatch, tmp_path, capsys):
    """Patch the lazy LLM + agent constructors and run main() in mock mode."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_llm = _FakeLLM()
    monkeypatch.setattr(runner, "_build_llm_provider", lambda: fake_llm)

    # Patch the QuantDataScientist constructor used by the runner so
    # we can inject a fake trade repo. The runner builds the agent
    # itself; we just need to swap the class for one that ignores the
    # constructor's db_url + uses our fake repo instead.
    real_class = runner.QuantDataScientist

    def patched_class(*args, **kwargs):
        kwargs.pop("db_url", None)
        kwargs["trade_repository"] = _FakeTradeRepo([_trade_dict()])
        kwargs["bot_ids"] = ["bot-a"]
        return real_class(*args, **kwargs)

    monkeypatch.setattr(runner, "QuantDataScientist", patched_class)

    output = tmp_path / "alpha_factors.json"
    code = runner.main(
        [
            "--no-ohlcv",
            "--output",
            str(output),
            "--bot-id",
            "bot-a",
        ]
    )
    assert code == 0
    assert output.exists()

    out = capsys.readouterr().out
    assert "QUANT DATA SCIENTIST" in out
    assert "Analyzed 1 trades across 1 symbols" in out
    assert "Found 1 alpha factor(s)" in out
    assert "Written to" in out
    assert fake_llm.calls == 1


def test_main_dry_run_does_not_write(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(runner, "_build_llm_provider", lambda: _FakeLLM())

    real_class = runner.QuantDataScientist

    def patched_class(*args, **kwargs):
        kwargs.pop("db_url", None)
        kwargs["trade_repository"] = _FakeTradeRepo([_trade_dict()])
        kwargs["bot_ids"] = ["bot-a"]
        return real_class(*args, **kwargs)

    monkeypatch.setattr(runner, "QuantDataScientist", patched_class)

    output = tmp_path / "alpha_factors.json"
    code = runner.main(
        ["--dry-run", "--no-ohlcv", "--output", str(output), "--bot-id", "bot-a"]
    )
    assert code == 0
    assert not output.exists()
    out = capsys.readouterr().out
    assert "DRY RUN" in out


def test_main_returns_1_when_run_reports_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _ExplodingLLM:
        async def generate_text(self, **kwargs):
            raise RuntimeError("kaboom")

        async def generate_vision(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    monkeypatch.setattr(runner, "_build_llm_provider", lambda: _ExplodingLLM())

    real_class = runner.QuantDataScientist

    def patched_class(*args, **kwargs):
        kwargs.pop("db_url", None)
        kwargs["trade_repository"] = _FakeTradeRepo([_trade_dict()])
        kwargs["bot_ids"] = ["bot-a"]
        return real_class(*args, **kwargs)

    monkeypatch.setattr(runner, "QuantDataScientist", patched_class)

    output = tmp_path / "alpha_factors.json"
    code = runner.main(
        ["--no-ohlcv", "--output", str(output), "--bot-id", "bot-a"]
    )
    assert code == 1
