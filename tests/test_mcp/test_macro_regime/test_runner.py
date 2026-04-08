"""Tests for the Macro Regime Manager CLI runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llm.base import LLMResponse
from mcp.macro_regime import runner
from mcp.macro_regime.data_fetcher import EconomicEvent, MacroSnapshot


# ---------------------------------------------------------------------------
# Argparse coverage
# ---------------------------------------------------------------------------


def test_parser_requires_mode():
    parser = runner._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_check_mode():
    args = runner._build_parser().parse_args(["--mode", "check"])
    assert args.mode == "check"
    assert args.dry_run is False


def test_parser_deep_mode():
    args = runner._build_parser().parse_args(["--mode", "deep"])
    assert args.mode == "deep"


def test_parser_emergency_with_symbols():
    args = runner._build_parser().parse_args(
        [
            "--mode",
            "emergency",
            "--trigger-symbols",
            "BTC-USDC,ETH-USDC",
        ]
    )
    assert args.mode == "emergency"
    assert args.trigger_symbols == "BTC-USDC,ETH-USDC"


def test_parser_invalid_mode_rejected():
    parser = runner._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "wat"])


def test_parser_help_does_not_crash():
    parser = runner._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_parse_symbols_helper():
    assert runner._parse_symbols(None) is None
    assert runner._parse_symbols("") is None
    assert runner._parse_symbols("BTC-USDC") == ["BTC-USDC"]
    assert runner._parse_symbols(" BTC-USDC , ETH-USDC ,, ") == [
        "BTC-USDC",
        "ETH-USDC",
    ]


# ---------------------------------------------------------------------------
# End-to-end with monkeypatched fetcher + LLM
# ---------------------------------------------------------------------------


_FIXED_NOW = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)


def _stable_snap(events=None) -> MacroSnapshot:
    return MacroSnapshot(
        fetched_at="2026-04-08T12:00:00Z",
        vix=18.0,
        vix_timestamp="2026-04-08T11:00:00Z",
        dxy=104.0,
        dxy_timestamp="2026-04-08T11:00:00Z",
        dvol=55.0,
        dvol_timestamp="2026-04-08T11:00:00Z",
        fear_greed_value=50,
        fear_greed_classification="Neutral",
        btc_dominance=51.0,
        hl_total_oi=1_000_000.0,
        hl_avg_funding=0.0001,
        economic_calendar=events or [],
        available_sources={"vix", "dxy", "dvol", "fear_greed", "btc_dominance", "hyperliquid"},
    )


def _moved_snap() -> MacroSnapshot:
    snap = _stable_snap()
    snap.vix = 25.0  # +39% versus stable
    return snap


class _FakeFetcher:
    def __init__(self, snapshot: MacroSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def fetch(self) -> MacroSnapshot:
        self.calls += 1
        return self.snapshot


_VALID_LLM_JSON = json.dumps(
    {
        "regime": "RISK_OFF",
        "confidence": 0.7,
        "reasoning": "stress",
        "adjustments": {
            "conviction_threshold_boost": 0.1,
            "position_size_multiplier": 0.7,
        },
    }
)


def _fake_llm_provider() -> AsyncMock:
    llm = AsyncMock()
    llm.generate_text = AsyncMock(
        return_value=LLMResponse(
            content=_VALID_LLM_JSON,
            input_tokens=10,
            output_tokens=10,
            cost=0.0,
            model="claude-test",
            latency_ms=1.0,
            cached_input_tokens=0,
        )
    )
    return llm


def _patch_runner(monkeypatch, snapshot: MacroSnapshot, *, llm=None):
    fake_fetcher = _FakeFetcher(snapshot)
    monkeypatch.setattr(runner, "_build_fetcher", lambda: fake_fetcher)
    monkeypatch.setattr(
        runner, "_build_llm_provider", lambda: llm or _fake_llm_provider()
    )
    return fake_fetcher


# ---------- check mode ----------


def test_check_mode_no_trigger_clean_exit(tmp_path: Path, monkeypatch, capsys):
    # Pre-seed the snapshot so the first run has a baseline.
    snapshot_path = tmp_path / "snap.json"
    snapshot_path.write_text(json.dumps(_stable_snap().to_dict()))

    _patch_runner(monkeypatch, _stable_snap())
    rc = runner.main(
        [
            "--mode",
            "check",
            "--snapshot-path",
            str(snapshot_path),
            "--output",
            str(tmp_path / "macro_regime.json"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Triggered: no" in out
    # No macro_regime.json written by check mode without trigger.
    assert not (tmp_path / "macro_regime.json").exists()


def test_check_mode_trigger_escalates_to_deep(tmp_path: Path, monkeypatch, capsys):
    # Pre-seed snapshot with old VIX so the new fetch shows a +39% jump.
    snapshot_path = tmp_path / "snap.json"
    snapshot_path.write_text(json.dumps(_stable_snap().to_dict()))

    _patch_runner(monkeypatch, _moved_snap())
    out_path = tmp_path / "macro_regime.json"
    rc = runner.main(
        [
            "--mode",
            "check",
            "--snapshot-path",
            str(snapshot_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Triggered: yes" in out
    assert "escalating to Deep Analysis" in out
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["regime"] == "RISK_OFF"


# ---------- deep mode ----------


def test_deep_mode_writes_file(tmp_path: Path, monkeypatch, capsys):
    _patch_runner(monkeypatch, _stable_snap())
    out_path = tmp_path / "macro_regime.json"
    rc = runner.main(
        [
            "--mode",
            "deep",
            "--snapshot-path",
            str(tmp_path / "snap.json"),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    captured = capsys.readouterr().out
    assert "Regime: RISK_OFF" in captured
    assert "Written to" in captured


def test_deep_mode_dry_run_skips_write(tmp_path: Path, monkeypatch, capsys):
    _patch_runner(monkeypatch, _stable_snap())
    out_path = tmp_path / "macro_regime.json"
    rc = runner.main(
        [
            "--mode",
            "deep",
            "--dry-run",
            "--snapshot-path",
            str(tmp_path / "snap.json"),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    assert not out_path.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_deep_mode_llm_failure_returns_1(tmp_path: Path, monkeypatch):
    bad_llm = AsyncMock()
    bad_llm.generate_text = AsyncMock(side_effect=RuntimeError("anthropic 500"))
    _patch_runner(monkeypatch, _stable_snap(), llm=bad_llm)
    rc = runner.main(
        [
            "--mode",
            "deep",
            "--snapshot-path",
            str(tmp_path / "snap.json"),
            "--output",
            str(tmp_path / "macro_regime.json"),
        ]
    )
    assert rc == 1


def test_deep_mode_no_api_key_returns_2(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_fetcher = _FakeFetcher(_stable_snap())
    monkeypatch.setattr(runner, "_build_fetcher", lambda: fake_fetcher)
    # Use the real _build_llm_provider — it should refuse without the key.
    rc = runner.main(
        [
            "--mode",
            "deep",
            "--snapshot-path",
            str(tmp_path / "snap.json"),
            "--output",
            str(tmp_path / "macro_regime.json"),
        ]
    )
    assert rc == 2


# ---------- emergency mode ----------


def test_emergency_mode_with_symbols(tmp_path: Path, monkeypatch, capsys):
    captured_args = {}

    class _CapturingLLM:
        async def generate_text(
            self,
            system_prompt: str,
            user_prompt: str,
            agent_name: str,
            **kwargs,
        ) -> LLMResponse:
            captured_args["user_prompt"] = user_prompt
            return LLMResponse(
                content=_VALID_LLM_JSON,
                input_tokens=1,
                output_tokens=1,
                cost=0.0,
                model="claude-test",
                latency_ms=1.0,
                cached_input_tokens=0,
            )

    _patch_runner(monkeypatch, _stable_snap(), llm=_CapturingLLM())
    rc = runner.main(
        [
            "--mode",
            "emergency",
            "--trigger-symbols",
            "BTC-USDC,ETH-USDC",
            "--snapshot-path",
            str(tmp_path / "snap.json"),
            "--output",
            str(tmp_path / "macro_regime.json"),
        ]
    )
    assert rc == 0
    assert "BTC-USDC" in captured_args["user_prompt"]
    assert "Urgency: emergency" in captured_args["user_prompt"]
