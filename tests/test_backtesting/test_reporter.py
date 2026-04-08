"""Unit tests for backtesting.reporter — JSON + HTML report generators."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from backtesting.engine import BacktestConfig
from backtesting.metrics import calculate_metrics
from backtesting.reporter import (
    generate_html_report,
    generate_json_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> BacktestConfig:
    return BacktestConfig(
        symbols=["BTC-USDC"],
        timeframes=["1h"],
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 1, 31, 23, tzinfo=timezone.utc),
        initial_balance=10_000.0,
        mode="mechanical",
        exchange="hyperliquid",
    )


def _make_trade(
    pnl: float,
    ts: int = 1_700_000_000_000,
    entry_ts: int = 1_700_000_000_000 - 3_600_000,
) -> dict:
    return {
        "timestamp": ts,
        "entry_timestamp": entry_ts,
        "symbol": "BTC-USDC",
        "side": "long",
        "entry_price": 100.0,
        "exit_price": 100.0 + pnl,
        "size": 1.0,
        "fee": 0.5,
        "slippage": 0.0,
        "pnl": pnl,
        "reason": "tp2_hit" if pnl > 0 else "stop_hit",
    }


@pytest.fixture
def sample_run():
    """Realistic-looking small run: 5 trades, 5-day equity curve."""
    config = _make_config()
    trades = [
        _make_trade(10.0, ts=1_700_000_000_000),
        _make_trade(-5.0, ts=1_700_086_400_000),
        _make_trade(20.0, ts=1_700_172_800_000),
        _make_trade(-3.0, ts=1_700_259_200_000),
        _make_trade(15.0, ts=1_700_345_600_000),
    ]
    DAY_MS = 24 * 3600 * 1000
    eq = 10_000.0
    equity_curve = []
    for i, t in enumerate(trades):
        equity_curve.append((1_700_000_000_000 + i * DAY_MS, eq))
        eq += t["pnl"]
    equity_curve.append((1_700_000_000_000 + 5 * DAY_MS, eq))
    metrics = calculate_metrics(trades, equity_curve, config)
    return config, trades, equity_curve, metrics


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def test_generate_json_report_creates_file(tmp_path, sample_run):
    config, trades, equity, metrics = sample_run
    path_str = generate_json_report(
        metrics=metrics,
        config=config,
        trade_history=trades,
        equity_curve=equity,
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    path = Path(path_str)
    assert path.exists()
    assert path.name == "2026-04-07_mechanical_backtest.json"


def test_json_report_is_valid_json_with_expected_keys(tmp_path, sample_run):
    config, trades, equity, metrics = sample_run
    path = generate_json_report(
        metrics=metrics,
        config=config,
        trade_history=trades,
        equity_curve=equity,
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    with open(path) as f:
        payload = json.load(f)

    for key in (
        "report_generated_at",
        "config",
        "metrics",
        "equity_curve",
        "trade_history",
        "engine_versions",
    ):
        assert key in payload

    # Config round-tripped
    assert payload["config"]["mode"] == "mechanical"
    assert payload["config"]["symbols"] == ["BTC-USDC"]

    # All metric fields present
    metric_keys = set(metrics.to_dict().keys())
    assert set(payload["metrics"].keys()) == metric_keys

    # Equity curve has structured rows
    assert len(payload["equity_curve"]) == len(equity)
    assert payload["equity_curve"][0]["timestamp"] == equity[0][0]
    assert payload["equity_curve"][0]["equity"] == equity[0][1]

    # Trade history preserved
    assert len(payload["trade_history"]) == len(trades)
    assert payload["trade_history"][0]["pnl"] == trades[0]["pnl"]

    # Engine version stamped
    assert "engine_version" in payload["engine_versions"]
    assert payload["engine_versions"]["engine_version"]


def test_json_report_handles_empty_run(tmp_path):
    config = _make_config()
    metrics = calculate_metrics([], [], config)
    path = generate_json_report(
        metrics=metrics,
        config=config,
        trade_history=[],
        equity_curve=[],
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    with open(path) as f:
        payload = json.load(f)
    assert payload["trade_history"] == []
    assert payload["equity_curve"] == []
    assert payload["metrics"]["total_trades"] == 0


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def test_generate_html_report_creates_file(tmp_path, sample_run):
    config, trades, equity, metrics = sample_run
    path_str = generate_html_report(
        metrics=metrics,
        config=config,
        trade_history=trades,
        equity_curve=equity,
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    path = Path(path_str)
    assert path.exists()
    assert path.name == "2026-04-07_mechanical_backtest.html"
    assert path.stat().st_size > 0


def test_html_report_contains_expected_sections(tmp_path, sample_run):
    config, trades, equity, metrics = sample_run
    path = generate_html_report(
        metrics=metrics,
        config=config,
        trade_history=trades,
        equity_curve=equity,
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    html_text = Path(path).read_text()

    # Doctype + key headings
    assert html_text.startswith("<!doctype html>")
    assert "QuantAgent Backtest Report" in html_text
    assert "Performance Summary" in html_text
    assert "Equity Curve" in html_text
    assert "Drawdown" in html_text
    assert "Trades" in html_text

    # Config summary
    assert "BTC-USDC" in html_text
    assert "mechanical" in html_text

    # Each metric label present (sample a few)
    for label in ("Win rate", "Sharpe", "Max drawdown", "Profit factor", "Total PnL"):
        assert label in html_text

    # Embedded chart payloads (base64 PNGs)
    assert "data:image/png;base64," in html_text
    # At least two charts (equity + drawdown)
    assert html_text.count("data:image/png;base64,") >= 2

    # Trade table populated
    assert "<table" in html_text
    for trade in trades:
        # Format the timestamp the way the reporter does to make sure
        # at least one trade row appears
        ts_str = datetime.fromtimestamp(
            trade["timestamp"] / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M")
        assert ts_str in html_text

    # Footer carries version stamp
    assert "Engine version" in html_text


def test_html_report_handles_empty_trade_history(tmp_path):
    config = _make_config()
    metrics = calculate_metrics([], [], config)
    path = generate_html_report(
        metrics=metrics,
        config=config,
        trade_history=[],
        equity_curve=[],
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    html_text = Path(path).read_text()
    # Empty-state placeholder for trades table
    assert "No trades" in html_text
    # Charts still render (even if "No equity data" placeholder)
    assert "data:image/png;base64," in html_text


def test_html_report_escapes_potentially_unsafe_strings(tmp_path):
    """A trade with HTML-injection-y values must be escaped."""
    config = _make_config()
    nasty = _make_trade(10.0)
    nasty["symbol"] = "<script>alert(1)</script>"
    nasty["side"] = "long&unsafe"
    nasty["reason"] = "<img src=x>"
    metrics = calculate_metrics(
        [nasty],
        [(1_700_000_000_000, 10_000.0), (1_700_086_400_000, 10_010.0)],
        config,
    )
    path = generate_html_report(
        metrics=metrics,
        config=config,
        trade_history=[nasty],
        equity_curve=[(1_700_000_000_000, 10_000.0), (1_700_086_400_000, 10_010.0)],
        output_dir=tmp_path,
        run_date=date(2026, 4, 7),
    )
    html_text = Path(path).read_text()
    # Raw script tag must NOT appear; escaped form must appear
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "&lt;img src=x&gt;" in html_text


def test_html_report_overwrites_same_filename(tmp_path, sample_run):
    """Two runs on the same date+mode produce the same path (deterministic)."""
    config, trades, equity, metrics = sample_run
    p1 = generate_html_report(metrics, config, trades, equity, tmp_path, date(2026, 4, 7))
    p2 = generate_html_report(metrics, config, trades, equity, tmp_path, date(2026, 4, 7))
    assert p1 == p2
    assert Path(p1).exists()


def test_json_and_html_paths_share_naming_convention(tmp_path, sample_run):
    config, trades, equity, metrics = sample_run
    j = generate_json_report(metrics, config, trades, equity, tmp_path, date(2026, 4, 7))
    h = generate_html_report(metrics, config, trades, equity, tmp_path, date(2026, 4, 7))
    assert Path(j).stem == Path(h).stem
    assert Path(j).suffix == ".json"
    assert Path(h).suffix == ".html"


def test_generate_json_report_default_output_dir_uses_run_date(tmp_path, sample_run):
    """Smoke test of the default `backtesting/results/` dir.

    We point output_dir at a tmp path so the test doesn't pollute the
    repo, but verify the filename matches the documented pattern.
    """
    config, trades, equity, metrics = sample_run
    path_str = generate_json_report(
        metrics=metrics,
        config=config,
        trade_history=trades,
        equity_curve=equity,
        output_dir=tmp_path,  # override the default
    )
    # Filename format: YYYY-MM-DD_{mode}_backtest.json
    assert re.match(
        r"\d{4}-\d{2}-\d{2}_mechanical_backtest\.json$",
        Path(path_str).name,
    )
