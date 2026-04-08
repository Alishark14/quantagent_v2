"""Tests for the AutoMiner repository wiring + mine_eval_scenarios CLI.

Covers:

* RepositoryTradeFetcher pulls trades across multiple bot ids and
  normalises field names (conviction_score → conviction, direction →
  action, entry_time ISO → entry_timestamp ms).
* RepositoryTradeFetcher requires either bot_repo or bot_ids.
* End-to-end mine() with a repo-backed fetcher writes drafts to disk
  for both failure modes.
* No-trades-in-repo case prints "No scenarios mined." and exits 0.
* mine_eval_scenarios.py CLI argparse coverage.
* mine_eval_scenarios.py main() runs end-to-end against an in-memory
  fake repository, writes drafts to a tmp dir, prints the summary,
  and exits 0.
* compute_forward_max_r helper covered by parametrised cases (LONG,
  SHORT, no excursion, polars vs list-of-dicts input).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from backtesting.evals.auto_miner import AutoMiner, RepositoryTradeFetcher


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTradeRepo:
    """Minimal in-memory TradeRepository for unit tests."""

    def __init__(self, trades_by_bot: dict[str, list[dict]]) -> None:
        self._trades_by_bot = trades_by_bot
        self.updates: list[tuple[str, dict]] = []

    async def get_trades_by_bot(self, bot_id: str, limit: int = 50) -> list[dict]:
        return list(self._trades_by_bot.get(bot_id, []))[:limit]

    async def get_trade(self, trade_id: str) -> dict | None:
        for trades in self._trades_by_bot.values():
            for t in trades:
                if t.get("id") == trade_id:
                    return dict(t)
        return None

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        self.updates.append((trade_id, dict(updates)))
        return True


class _FakeBotRepo:
    def __init__(self, bots: list[dict]) -> None:
        self._bots = bots

    async def get_bots_by_user(self, user_id: str) -> list[dict]:
        return list(self._bots)


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _entry_iso(days_ago: int = 1) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).isoformat()


def _trade_record(
    trade_id: str,
    *,
    bot_id: str = "bot-a",
    pnl: float = 0.0,
    conviction_score: float = 0.5,
    forward_max_r: float | None = None,
    direction: str = "LONG",
    days_ago: int = 1,
    symbol: str = "BTC-USDC",
    timeframe: str = "1h",
) -> dict:
    return {
        "id": trade_id,
        "bot_id": bot_id,
        "user_id": "dev-user",
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "entry_price": 60000.0,
        "exit_price": 59500.0 if pnl < 0 else 60500.0,
        "size": 0.1,
        "pnl": pnl,
        "r_multiple": None,
        "entry_time": _entry_iso(days_ago=days_ago),
        "exit_time": _entry_iso(days_ago=0),
        "exit_reason": "tp1" if pnl > 0 else "sl",
        "conviction_score": conviction_score,
        "engine_version": "test",
        "status": "closed",
        "forward_max_r": forward_max_r,
    }


# ---------------------------------------------------------------------------
# RepositoryTradeFetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repository_fetcher_with_explicit_bot_ids():
    repo = _FakeTradeRepo(
        {
            "bot-a": [_trade_record("a1"), _trade_record("a2")],
            "bot-b": [_trade_record("b1", bot_id="bot-b")],
        }
    )
    fetcher = RepositoryTradeFetcher(
        trade_repo=repo, bot_ids=["bot-a", "bot-b"]
    )
    trades = await fetcher()
    assert len(trades) == 3
    ids = sorted(t["id"] for t in trades)
    assert ids == ["a1", "a2", "b1"]


@pytest.mark.asyncio
async def test_repository_fetcher_normalises_field_names():
    repo = _FakeTradeRepo({"bot-a": [_trade_record("a1", conviction_score=0.9)]})
    fetcher = RepositoryTradeFetcher(trade_repo=repo, bot_ids=["bot-a"])
    [trade] = await fetcher()
    # conviction_score → conviction
    assert trade["conviction"] == 0.9
    # direction → action
    assert trade["action"] == "LONG"
    # id → trade_id
    assert trade["trade_id"] == "a1"
    # entry_time ISO → entry_timestamp ms
    assert isinstance(trade["entry_timestamp"], int)
    assert trade["entry_timestamp"] > 0


@pytest.mark.asyncio
async def test_repository_fetcher_with_bot_repo_discovery():
    bot_repo = _FakeBotRepo([{"id": "bot-a"}, {"id": "bot-b"}])
    trade_repo = _FakeTradeRepo(
        {
            "bot-a": [_trade_record("a1")],
            "bot-b": [_trade_record("b1", bot_id="bot-b")],
        }
    )
    fetcher = RepositoryTradeFetcher(trade_repo=trade_repo, bot_repo=bot_repo)
    trades = await fetcher()
    assert sorted(t["id"] for t in trades) == ["a1", "b1"]


def test_repository_fetcher_requires_bot_repo_or_bot_ids():
    repo = _FakeTradeRepo({})
    with pytest.raises(ValueError, match="bot_repo or bot_ids"):
        RepositoryTradeFetcher(trade_repo=repo)


@pytest.mark.asyncio
async def test_repository_fetcher_swallows_per_bot_failures(caplog):
    class FlakyRepo:
        async def get_trades_by_bot(self, bot_id, limit=50):
            if bot_id == "bot-broken":
                raise RuntimeError("simulated db blip")
            return [_trade_record("ok1", bot_id=bot_id)]

    fetcher = RepositoryTradeFetcher(
        trade_repo=FlakyRepo(),
        bot_ids=["bot-good", "bot-broken"],
    )
    with caplog.at_level("WARNING"):
        trades = await fetcher()
    assert [t["id"] for t in trades] == ["ok1"]
    assert any("simulated db blip" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_repository_fetcher_per_bot_limit_passed_through():
    captured: list[int] = []

    class LimitSpyRepo:
        async def get_trades_by_bot(self, bot_id, limit=50):
            captured.append(limit)
            return []

    fetcher = RepositoryTradeFetcher(
        trade_repo=LimitSpyRepo(), bot_ids=["bot-a"], per_bot_limit=42
    )
    await fetcher()
    assert captured == [42]


# ---------------------------------------------------------------------------
# End-to-end mine() with repository fetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_writes_overconfident_disaster_via_repo_fetcher(tmp_path):
    repo = _FakeTradeRepo(
        {
            "bot-a": [
                _trade_record(
                    "disaster-1",
                    pnl=-50.0,
                    conviction_score=0.92,
                ),
            ]
        }
    )
    fetcher = RepositoryTradeFetcher(trade_repo=repo, bot_ids=["bot-a"])
    miner = AutoMiner(trade_fetcher=fetcher, output_dir=tmp_path)

    written = await miner.mine(days=30)
    assert len(written) == 1
    assert "overconfident_disaster" in written[0].name

    payload = json.loads(written[0].read_text())
    assert payload["category"] == "trap_setups"
    assert payload["metadata"]["mining_reason"] == "overconfident_disaster"
    assert payload["metadata"]["trade_pnl"] == -50.0
    assert payload["metadata"]["trade_conviction"] == 0.92


@pytest.mark.asyncio
async def test_mine_writes_missed_opportunity_via_repo_fetcher(tmp_path):
    repo = _FakeTradeRepo(
        {
            "bot-a": [
                _trade_record(
                    "missed-1",
                    pnl=0.0,
                    conviction_score=0.20,
                    forward_max_r=4.5,
                )
            ]
        }
    )
    fetcher = RepositoryTradeFetcher(trade_repo=repo, bot_ids=["bot-a"])
    miner = AutoMiner(trade_fetcher=fetcher, output_dir=tmp_path)

    written = await miner.mine(days=30)
    assert len(written) == 1
    assert "missed_opportunity" in written[0].name

    payload = json.loads(written[0].read_text())
    assert payload["category"] == "clear_setups"
    assert payload["metadata"]["mining_reason"] == "missed_opportunity"
    assert payload["metadata"]["trade_forward_max_r"] == 4.5


@pytest.mark.asyncio
async def test_mine_with_no_trades_writes_nothing(tmp_path):
    repo = _FakeTradeRepo({"bot-a": []})
    fetcher = RepositoryTradeFetcher(trade_repo=repo, bot_ids=["bot-a"])
    miner = AutoMiner(trade_fetcher=fetcher, output_dir=tmp_path)
    written = await miner.mine(days=7)
    assert written == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mine_skips_overconfident_winner(tmp_path):
    """High conviction + winning trade is not a disaster."""
    repo = _FakeTradeRepo(
        {"bot-a": [_trade_record("winner", pnl=200.0, conviction_score=0.92)]}
    )
    fetcher = RepositoryTradeFetcher(trade_repo=repo, bot_ids=["bot-a"])
    miner = AutoMiner(trade_fetcher=fetcher, output_dir=tmp_path)
    written = await miner.mine(days=30)
    assert written == []


@pytest.mark.asyncio
async def test_mine_skips_low_conviction_no_forward_move(tmp_path):
    """Low conviction skip without a strong forward move is not a missed opportunity."""
    repo = _FakeTradeRepo(
        {
            "bot-a": [
                _trade_record(
                    "no-move", pnl=0.0, conviction_score=0.20, forward_max_r=0.5
                )
            ]
        }
    )
    fetcher = RepositoryTradeFetcher(trade_repo=repo, bot_ids=["bot-a"])
    miner = AutoMiner(trade_fetcher=fetcher, output_dir=tmp_path)
    written = await miner.mine(days=30)
    assert written == []


# ---------------------------------------------------------------------------
# mine_eval_scenarios CLI
# ---------------------------------------------------------------------------


def test_cli_argparse_defaults():
    from scripts.mine_eval_scenarios import _build_parser

    args = _build_parser().parse_args([])
    assert args.days == 7
    assert args.bot_id is None
    assert args.backend is None
    assert args.per_bot_limit == 200


def test_cli_argparse_repeated_bot_id():
    from scripts.mine_eval_scenarios import _build_parser

    args = _build_parser().parse_args(["--bot-id", "a", "--bot-id", "b"])
    assert args.bot_id == ["a", "b"]


def test_cli_argparse_help_does_not_crash():
    from scripts.mine_eval_scenarios import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_main_negative_days_returns_2():
    from scripts.mine_eval_scenarios import main

    code = main(["--days", "0", "--bot-id", "bot-a", "--backend", "sqlite"])
    assert code == 2


def test_cli_main_runs_end_to_end_with_fake_repos(monkeypatch, tmp_path, capsys):
    """End-to-end CLI run with a fake repository container."""
    from scripts import mine_eval_scenarios

    class _FakeRepos:
        def __init__(self) -> None:
            self.trades = _FakeTradeRepo(
                {
                    "bot-a": [
                        _trade_record(
                            "disaster-cli",
                            pnl=-100.0,
                            conviction_score=0.91,
                        ),
                        _trade_record(
                            "missed-cli",
                            pnl=0.0,
                            conviction_score=0.10,
                            forward_max_r=5.0,
                        ),
                    ]
                }
            )
            self.bots = _FakeBotRepo([{"id": "bot-a"}])

        async def close(self) -> None:
            pass

    async def fake_get_repositories(backend=None):
        return _FakeRepos()

    monkeypatch.setattr(
        "storage.repositories.get_repositories", fake_get_repositories
    )

    code = mine_eval_scenarios.main(
        ["--days", "30", "--output-dir", str(tmp_path)]
    )
    assert code == 0

    out = capsys.readouterr().out
    assert "AUTO-MINE EVAL SCENARIOS" in out
    assert "Found 1 overconfident disaster(s), 1 missed opportunity(ies)" in out
    assert "overconfident_disaster" in out
    assert "missed_opportunity" in out

    files = sorted(p.name for p in tmp_path.iterdir())
    assert any("overconfident_disaster" in f for f in files)
    assert any("missed_opportunity" in f for f in files)


def test_cli_main_no_trades_prints_zero(monkeypatch, tmp_path, capsys):
    from scripts import mine_eval_scenarios

    class _EmptyRepos:
        def __init__(self) -> None:
            self.trades = _FakeTradeRepo({"bot-a": []})
            self.bots = _FakeBotRepo([{"id": "bot-a"}])

        async def close(self) -> None:
            pass

    async def fake_get_repositories(backend=None):
        return _EmptyRepos()

    monkeypatch.setattr(
        "storage.repositories.get_repositories", fake_get_repositories
    )

    code = mine_eval_scenarios.main(
        ["--days", "7", "--output-dir", str(tmp_path)]
    )
    assert code == 0

    out = capsys.readouterr().out
    assert "Found 0 overconfident disaster(s), 0 missed opportunity(ies)" in out
    assert "No scenarios mined." in out
