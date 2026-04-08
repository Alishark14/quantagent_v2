"""Eval report generators — JSON + standalone HTML.

Both writers consume an :class:`backtesting.evals.framework.EvalReport`
and produce a single self-contained file each. The HTML report is
designed to be shareable as marketing material — clean styling, no
external assets, every value escaped.

Files saved to ``backtesting/evals/reports/{YYYY-MM-DD}_eval.{json|html}``
unless ``output_dir`` is overridden.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from backtesting.evals.framework import EvalReport

logger = logging.getLogger(__name__)


_DEFAULT_RESULTS_DIR = Path(__file__).parent / "reports"


def generate_eval_report(
    report: EvalReport,
    output_dir: str | Path = _DEFAULT_RESULTS_DIR,
    run_date: date | None = None,
    previous_report: EvalReport | None = None,
) -> tuple[str, str]:
    """Write the eval report to JSON + HTML.

    Returns a ``(json_path, html_path)`` tuple of absolute paths.
    Pass ``previous_report`` to highlight regressions in the HTML.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_date = run_date or datetime.now(tz=timezone.utc).date()

    json_path = output_dir / f"{run_date.isoformat()}_eval.json"
    html_path = output_dir / f"{run_date.isoformat()}_eval.html"

    payload = report.to_dict()
    if previous_report is not None:
        payload["regressions"] = _compute_regressions(report, previous_report)

    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    with html_path.open("w") as f:
        f.write(_build_html(report, payload.get("regressions", [])))

    return str(json_path.resolve()), str(html_path.resolve())


# ---------------------------------------------------------------------------
# Regression diff
# ---------------------------------------------------------------------------


def _compute_regressions(current: EvalReport, previous: EvalReport) -> list[dict]:
    """List categories whose pass-rate dropped vs the previous run."""
    prev_by_cat = {c.category: c.pass_rate for c in previous.by_category}
    out: list[dict] = []
    for stat in current.by_category:
        prev_rate = prev_by_cat.get(stat.category)
        if prev_rate is None:
            continue
        delta = stat.pass_rate - prev_rate
        if delta < -0.05:  # 5pp drop is the threshold per ARCHITECTURE §31.4.9
            out.append(
                {
                    "category": stat.category,
                    "previous_pass_rate": round(prev_rate, 4),
                    "current_pass_rate": round(stat.pass_rate, 4),
                    "delta": round(delta, 4),
                }
            )
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _build_html(report: EvalReport, regressions: list[dict]) -> str:
    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    css = _CSS
    header = _build_header(report, generated)
    summary = _build_summary(report)
    cat_table = _build_category_table(report)
    failures = _build_failures(report)
    regressions_html = _build_regressions(regressions)

    return (
        f"<!doctype html>\n<html lang=\"en\"><head>\n"
        f"<meta charset=\"utf-8\">\n"
        f"<title>QuantAgent Eval Report</title>\n"
        f"<style>{css}</style>\n"
        f"</head><body><main>\n"
        f"{header}\n"
        f"<section><h2>Overall Score</h2>{summary}</section>\n"
        f"<section><h2>By Category</h2>{cat_table}</section>\n"
        f"{regressions_html}\n"
        f"<section><h2>Top Failures</h2>{failures}</section>\n"
        f"<footer><p>Model: <code>{html.escape(report.model_id)}</code></p>"
        f"<p>Prompts: <code>{html.escape(_format_prompts(report.prompt_versions))}</code></p>"
        f"</footer>\n"
        f"</main></body></html>\n"
    )


def _build_header(report: EvalReport, generated: str) -> str:
    return (
        f"<header>\n"
        f"<h1>QuantAgent Eval Report</h1>\n"
        f"<p class=\"subtitle\">Generated {html.escape(generated)}</p>\n"
        f"<dl class=\"meta\">\n"
        f"<dt>Total scenarios</dt><dd>{report.total_scenarios}</dd>\n"
        f"<dt>Runs per scenario</dt><dd>{report.runs_per_scenario}</dd>\n"
        f"<dt>Duration</dt><dd>{report.duration_seconds:.1f}s</dd>\n"
        f"</dl>\n"
        f"</header>"
    )


def _build_summary(report: EvalReport) -> str:
    pct = report.overall_pass_rate * 100
    cls = "good" if pct >= 75 else "warn" if pct >= 50 else "bad"
    return (
        f"<div class=\"hero {cls}\">\n"
        f"<span class=\"value\">{pct:.1f}%</span>\n"
        f"<span class=\"label\">overall pass rate</span>\n"
        f"<span class=\"sub\">consistency stdev: "
        f"{report.consistency_avg_stdev:.4f}</span>\n"
        f"</div>"
    )


def _build_category_table(report: EvalReport) -> str:
    if not report.by_category:
        return "<p class=\"empty\">No category data.</p>"
    rows = []
    for stat in report.by_category:
        pct = stat.pass_rate * 100
        cls = "good" if pct >= 75 else "warn" if pct >= 50 else "bad"
        rows.append(
            f"<tr>"
            f"<td>{html.escape(stat.category)}</td>"
            f"<td>{stat.passed} / {stat.total}</td>"
            f"<td class=\"{cls}\">{pct:.1f}%</td>"
            f"</tr>"
        )
    body = "\n".join(rows)
    return (
        f"<table class=\"cats\">\n"
        f"<thead><tr><th>Category</th><th>Passed</th><th>Pass rate</th></tr></thead>\n"
        f"<tbody>{body}</tbody>\n"
        f"</table>"
    )


def _build_regressions(regressions: list[dict]) -> str:
    if not regressions:
        return ""
    rows = []
    for r in regressions:
        rows.append(
            f"<tr>"
            f"<td>{html.escape(r['category'])}</td>"
            f"<td>{r['previous_pass_rate'] * 100:.1f}%</td>"
            f"<td>{r['current_pass_rate'] * 100:.1f}%</td>"
            f"<td class=\"bad\">{r['delta'] * 100:+.1f} pp</td>"
            f"</tr>"
        )
    return (
        f"<section><h2>⚠️ Regressions</h2>\n"
        f"<table class=\"cats\">\n"
        f"<thead><tr><th>Category</th><th>Prev</th><th>Now</th><th>Δ</th></tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n"
        f"</table></section>"
    )


def _build_failures(report: EvalReport) -> str:
    if not report.top_failures:
        return "<p class=\"empty\">No failures. 🎯</p>"
    items = []
    for r in report.top_failures:
        reasons = "; ".join(html.escape(x) for x in r.failure_reasons) or "(no reason)"
        items.append(
            f"<li>"
            f"<strong>{html.escape(r.scenario_id)}</strong> "
            f"<span class=\"category\">[{html.escape(r.category)}]</span><br>"
            f"<span class=\"name\">{html.escape(r.scenario_name)}</span><br>"
            f"<span class=\"reason\">{reasons}</span>"
            f"</li>"
        )
    return f"<ul class=\"failures\">{''.join(items)}</ul>"


def _format_prompts(prompts: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(prompts.items()))


_CSS = """
:root {
  color-scheme: light;
  --text: #1d1d1f;
  --muted: #6b7280;
  --bg: #fafafa;
  --card-bg: #ffffff;
  --border: #e5e7eb;
  --good: #16a34a;
  --warn: #f59e0b;
  --bad: #dc2626;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg); color: var(--text); margin: 0; padding: 24px;
}
main { max-width: 1100px; margin: 0 auto; }
header h1 { margin: 0 0 4px; font-size: 28px; }
.subtitle { color: var(--muted); margin: 0 0 16px; }
.meta { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; font-weight: 500; }
section { margin-top: 32px; }
section h2 { font-size: 18px; margin: 0 0 12px; }
.hero {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}
.hero .value { font-size: 56px; font-weight: 700; line-height: 1; }
.hero .label { color: var(--muted); margin-top: 4px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }
.hero .sub { color: var(--muted); margin-top: 12px; font-size: 12px; }
.hero.good .value { color: var(--good); }
.hero.warn .value { color: var(--warn); }
.hero.bad  .value { color: var(--bad); }
table.cats { width: 100%; border-collapse: collapse; font-size: 14px; }
table.cats th, table.cats td {
  padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left;
}
table.cats th { background: var(--card-bg); color: var(--muted); font-weight: 600; }
.good { color: var(--good); font-weight: 600; }
.warn { color: var(--warn); font-weight: 600; }
.bad  { color: var(--bad);  font-weight: 600; }
.failures { list-style: none; padding: 0; }
.failures li {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; margin-bottom: 8px;
}
.failures .category { color: var(--muted); font-size: 12px; }
.failures .name { color: var(--muted); font-size: 13px; }
.failures .reason { color: var(--bad); font-size: 12px; display: inline-block; margin-top: 4px; }
.empty { color: var(--muted); font-style: italic; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
footer code { background: var(--card-bg); border: 1px solid var(--border); padding: 1px 6px; border-radius: 4px; }
"""
