"""Backtest report generators — JSON + standalone HTML.

Both writers consume the same inputs (metrics, config, trade history,
equity curve) and produce a single self-contained file each. The HTML
report embeds matplotlib charts as base64 PNGs so it can be opened
locally or attached to a Slack message without any asset fetching.

Files are saved to ``backtesting/results/{YYYY-MM-DD}_{mode}_backtest.{ext}``
relative to ``output_dir`` (default ``backtesting/results``). Both
generator functions return the absolute path to the written file.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path

# Use a non-interactive matplotlib backend for headless report generation.
# Must be set before pyplot is imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtesting.metrics import BacktestMetrics  # noqa: E402

logger = logging.getLogger(__name__)


_DEFAULT_RESULTS_DIR = Path("backtesting/results")


def generate_json_report(
    metrics: BacktestMetrics,
    config,
    trade_history: list[dict],
    equity_curve: list[tuple[int, float]],
    output_dir: str | Path = _DEFAULT_RESULTS_DIR,
    run_date: date | None = None,
) -> str:
    """Write the backtest results to JSON. Returns the absolute path."""
    output_path = _output_path(config, output_dir, "json", run_date)
    payload = {
        "report_generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": _config_to_dict(config),
        "metrics": metrics.to_dict(),
        "equity_curve": [
            {"timestamp": int(ts), "equity": float(eq)} for ts, eq in equity_curve
        ],
        "trade_history": [_trade_to_dict(t) for t in trade_history],
        "engine_versions": _engine_versions(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(output_path.resolve())


def generate_html_report(
    metrics: BacktestMetrics,
    config,
    trade_history: list[dict],
    equity_curve: list[tuple[int, float]],
    output_dir: str | Path = _DEFAULT_RESULTS_DIR,
    run_date: date | None = None,
) -> str:
    """Write the backtest results to a single self-contained HTML file."""
    output_path = _output_path(config, output_dir, "html", run_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    equity_chart_b64 = _render_equity_chart_png(equity_curve)
    drawdown_chart_b64 = _render_drawdown_chart_png(equity_curve)

    document = _build_html_document(
        metrics=metrics,
        config=config,
        trade_history=trade_history,
        equity_chart_b64=equity_chart_b64,
        drawdown_chart_b64=drawdown_chart_b64,
    )
    with output_path.open("w") as f:
        f.write(document)
    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _output_path(
    config,
    output_dir: str | Path,
    extension: str,
    run_date: date | None,
) -> Path:
    today = (run_date or datetime.now(tz=timezone.utc).date()).isoformat()
    mode = getattr(config, "mode", "unknown")
    return Path(output_dir) / f"{today}_{mode}_backtest.{extension}"


def _config_to_dict(config) -> dict:
    if is_dataclass(config):
        d = asdict(config)
        # ISO-format any datetime values for JSON safety
        for key, value in list(d.items()):
            if isinstance(value, datetime):
                d[key] = value.isoformat()
        return d
    return {"repr": repr(config)}


def _trade_to_dict(trade: dict) -> dict:
    """JSON-safe trade record (every value coerced to a serialisable type)."""
    out: dict = {}
    for key, value in trade.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _engine_versions() -> dict:
    try:
        from quantagent.version import (
            ENGINE_VERSION,
            PROMPT_VERSIONS,
        )
        return {
            "engine_version": ENGINE_VERSION,
            "prompt_versions": dict(PROMPT_VERSIONS),
        }
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to read version info")
        return {}


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------


def _render_equity_chart_png(equity_curve: list[tuple[int, float]]) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    if equity_curve:
        xs = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in equity_curve]
        ys = [eq for _, eq in equity_curve]
        ax.plot(xs, ys, color="#2e86ab", linewidth=1.5)
        ax.fill_between(xs, ys, min(ys), alpha=0.15, color="#2e86ab")
    else:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _render_drawdown_chart_png(equity_curve: list[tuple[int, float]]) -> str:
    fig, ax = plt.subplots(figsize=(10, 3))
    if equity_curve:
        xs = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in equity_curve]
        peak = -float("inf")
        dd_pct: list[float] = []
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd_pct.append((eq - peak) / peak * 100.0)
            else:
                dd_pct.append(0.0)
        ax.fill_between(xs, dd_pct, 0, color="#e63946", alpha=0.4)
        ax.plot(xs, dd_pct, color="#e63946", linewidth=1.0)
    else:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _build_html_document(
    metrics: BacktestMetrics,
    config,
    trade_history: list[dict],
    equity_chart_b64: str,
    drawdown_chart_b64: str,
) -> str:
    cfg = _config_to_dict(config)
    versions = _engine_versions()
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    css = _CSS
    header = _build_header_html(cfg, generated)
    summary = _build_metrics_summary_html(metrics)
    equity_img = (
        f'<img alt="Equity curve" src="data:image/png;base64,{equity_chart_b64}" />'
    )
    drawdown_img = (
        f'<img alt="Drawdown" src="data:image/png;base64,{drawdown_chart_b64}" />'
    )
    trade_table = _build_trade_table_html(trade_history)
    footer = _build_footer_html(versions)

    return (
        f"<!doctype html>\n"
        f"<html lang=\"en\"><head>\n"
        f"<meta charset=\"utf-8\">\n"
        f"<title>QuantAgent Backtest Report</title>\n"
        f"<style>{css}</style>\n"
        f"</head><body>\n"
        f"<main>\n"
        f"{header}\n"
        f"<section><h2>Performance Summary</h2>{summary}</section>\n"
        f"<section><h2>Equity Curve</h2>{equity_img}</section>\n"
        f"<section><h2>Drawdown</h2>{drawdown_img}</section>\n"
        f"<section><h2>Trades</h2>{trade_table}</section>\n"
        f"{footer}\n"
        f"</main>\n"
        f"</body></html>\n"
    )


def _build_header_html(cfg: dict, generated: str) -> str:
    symbols = html.escape(", ".join(cfg.get("symbols", []) or []))
    timeframes = html.escape(", ".join(cfg.get("timeframes", []) or []))
    mode = html.escape(str(cfg.get("mode", "unknown")))
    start = html.escape(str(cfg.get("start_date", "")))
    end = html.escape(str(cfg.get("end_date", "")))
    return (
        f'<header>\n'
        f'  <h1>QuantAgent Backtest Report</h1>\n'
        f'  <p class="subtitle">Generated {html.escape(generated)}</p>\n'
        f'  <dl class="config">\n'
        f'    <dt>Mode</dt><dd>{mode}</dd>\n'
        f'    <dt>Symbols</dt><dd>{symbols}</dd>\n'
        f'    <dt>Timeframes</dt><dd>{timeframes}</dd>\n'
        f'    <dt>Start</dt><dd>{start}</dd>\n'
        f'    <dt>End</dt><dd>{end}</dd>\n'
        f'  </dl>\n'
        f'</header>'
    )


def _build_metrics_summary_html(metrics: BacktestMetrics) -> str:
    cards = [
        ("Win rate", f"{metrics.win_rate * 100:.2f}%"),
        ("Total trades", f"{metrics.total_trades}"),
        ("Profit factor", f"{metrics.profit_factor:.2f}"),
        ("Sharpe (ann.)", f"{metrics.sharpe_ratio:.2f}"),
        ("Calmar", f"{metrics.calmar_ratio:.2f}"),
        ("Max drawdown", f"{metrics.max_drawdown_pct:.2f}%"),
        ("Avg R", f"{metrics.avg_r_multiple:.2f}"),
        ("Total PnL", f"{metrics.total_pnl:.2f}"),
        ("Cost-adjusted PnL", f"{metrics.cost_adjusted_pnl:.2f}"),
        ("Return", f"{metrics.return_pct:.2f}%"),
        ("Win streak", f"{metrics.longest_win_streak}"),
        ("Loss streak", f"{metrics.longest_loss_streak}"),
        ("Skip rate", f"{metrics.skip_rate * 100:.2f}%"),
        ("Avg duration (h)", f"{metrics.avg_trade_duration_hours:.2f}"),
        ("Final balance", f"{metrics.final_balance:.2f}"),
    ]
    items = "".join(
        f'<div class="card"><span class="label">{html.escape(label)}</span>'
        f'<span class="value">{html.escape(value)}</span></div>'
        for label, value in cards
    )
    return f'<div class="metrics-grid">{items}</div>'


def _build_trade_table_html(trade_history: list[dict]) -> str:
    if not trade_history:
        return '<p class="empty">No trades.</p>'

    rows: list[str] = []
    for t in trade_history:
        ts = t.get("timestamp", 0)
        ts_str = (
            datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            if ts else ""
        )
        side = str(t.get("side", ""))
        symbol = str(t.get("symbol", ""))
        entry = t.get("entry_price", "")
        exit_p = t.get("exit_price", "")
        pnl = t.get("pnl", 0.0)
        reason = str(t.get("reason", ""))
        pnl_class = "win" if pnl > 0 else ("loss" if pnl < 0 else "")
        rows.append(
            f'<tr>'
            f'<td>{html.escape(ts_str)}</td>'
            f'<td>{html.escape(symbol)}</td>'
            f'<td>{html.escape(side)}</td>'
            f'<td>{entry}</td>'
            f'<td>{exit_p}</td>'
            f'<td class="{pnl_class}">{pnl:.4f}</td>'
            f'<td>{html.escape(reason)}</td>'
            f'</tr>'
        )
    body = "\n".join(rows)
    return (
        f'<table class="trades">\n'
        f'  <thead><tr>'
        f'<th>Close time</th><th>Symbol</th><th>Side</th>'
        f'<th>Entry</th><th>Exit</th><th>PnL</th><th>Reason</th>'
        f'</tr></thead>\n'
        f'  <tbody>\n{body}\n  </tbody>\n'
        f'</table>'
    )


def _build_footer_html(versions: dict) -> str:
    eng = html.escape(str(versions.get("engine_version", "unknown")))
    prompts = versions.get("prompt_versions", {}) or {}
    prompt_str = html.escape(
        ", ".join(f"{k}={v}" for k, v in sorted(prompts.items()))
    )
    return (
        f'<footer>\n'
        f'  <p>Engine version: <code>{eng}</code></p>\n'
        f'  <p>Prompt versions: <code>{prompt_str}</code></p>\n'
        f'</footer>'
    )


_CSS = """
:root {
  color-scheme: light;
  --text: #1d1d1f;
  --muted: #6b7280;
  --bg: #fafafa;
  --card-bg: #ffffff;
  --border: #e5e7eb;
  --accent: #2e86ab;
  --win: #16a34a;
  --loss: #dc2626;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0;
  padding: 24px;
}
main { max-width: 1100px; margin: 0 auto; }
header h1 { margin: 0 0 4px; font-size: 28px; }
.subtitle { color: var(--muted); margin: 0 0 16px; }
.config { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
.config dt { color: var(--muted); }
.config dd { margin: 0; }
section { margin-top: 32px; }
section h2 { font-size: 18px; margin: 0 0 12px; }
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
}
.card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 18px; font-weight: 600; margin-top: 4px; }
img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; background: white; }
table.trades { width: 100%; border-collapse: collapse; font-size: 13px; }
table.trades th, table.trades td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
table.trades th { background: var(--card-bg); color: var(--muted); font-weight: 600; }
table.trades td.win { color: var(--win); font-weight: 600; }
table.trades td.loss { color: var(--loss); font-weight: 600; }
.empty { color: var(--muted); font-style: italic; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
footer code { background: var(--card-bg); border: 1px solid var(--border); padding: 1px 6px; border-radius: 4px; }
"""
