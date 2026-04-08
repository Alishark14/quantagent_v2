"""Per-category scoring rubrics for the LLM-as-Judge.

A rubric tells the judge what to look for in this category of scenario.
The judge prompt is generic; the rubric is what makes the score
diagnostic. Without category-specific rubrics, the judge would just
reward "any plausible-sounding analysis" — defeating the framework.

See ARCHITECTURE.md §31.4.6.

Rubrics are intentionally short and structured. Long rubrics dilute the
signal and make the judge inconsistent.
"""

from __future__ import annotations


_DEFAULT_RUBRIC = """\
General rubric (no category-specific guidance available).

Be strict. Score 3 for "got the right answer with the obvious reasoning."
Reserve 4 and 5 for analyses that go beyond the obvious. Penalise
overconfidence on ambiguous setups.
"""


CATEGORY_RUBRICS: dict[str, str] = {
    "clear_setups": """\
Clear-setup rubric.

The scenario presents an unambiguous trade with supporting evidence
across multiple dimensions (indicators + pattern + flow + parent TF).
A competent model MUST get the direction right and produce conviction
≥ 0.65.

What to look for:
- Did the model identify the specific setup type (e.g. "bull flag",
  "breakout", "support bounce")? Generic "bullish momentum" is a 2.
- Did the reasoning cite the supporting confluence (volume on the
  breakout, ADX confirming trend, MACD histogram expanding)?
- Is the conviction well-calibrated? On a 4/4 confluence setup, anything
  below 0.7 is under-confident (penalise calibration).

Common failures: missing the pattern name, ignoring volume confirmation,
treating the setup as ambiguous when it isn't.
""",

    "clear_avoids": """\
Clear-avoid rubric.

This category tests SKIP discipline. Most retail bots fail here — they
trade everything they see. The correct answer is SKIP, and the
reasoning must explain WHY skipping is correct.

What to look for:
- Direction must be SKIP. LONG or SHORT is an automatic
  directional_correctness = 1, regardless of how nice the prose is.
- Did the model name the absence of edge? "ADX < 20", "ranging",
  "low volume", "no clear pattern", "drift not trend".
- Did the model resist the temptation to imagine a setup that isn't
  there? Models that hallucinate patterns into chop deserve a 1 on
  reasoning_completeness.

Common failures: forcing a direction onto noise, calling chop a "bull
flag", overweighting tiny indicator wiggles.
""",

    "conflicting_signals": """\
Conflicting-signals rubric.

The scenario has at least two pieces of evidence pointing in opposite
directions (e.g. RSI overbought + bullish pattern). A good model
acknowledges the conflict explicitly and either skips or scores
conviction below 0.5.

What to look for:
- Did the reasoning use the word "conflict", "contradiction",
  "divergence", or equivalent?
- Did it name the SPECIFIC conflicting signals (not just "mixed
  signals")?
- Is the conviction appropriately humble? An 0.8 conviction on a
  conflicting setup is a calibration failure, no matter the prose.

Common failures: ignoring one of the two signals, confidently picking
the easier one, scoring conviction > 0.6 with no acknowledgement of
the conflict.
""",

    "regime_transitions": """\
Regime-transition rubric.

The scenario captures a market shifting between regimes (trending →
ranging, ranging → breakout). The model is judged on whether it
correctly detected the inflection.

What to look for:
- Did the reasoning name both the old and new regime?
- Did it identify what triggered the transition (volatility expansion,
  ADX rising, BB width changing)?
- Did the conviction reflect the inherent uncertainty of inflection
  points? Inflection points warrant 0.4–0.6 conviction, not 0.85.
""",

    "trap_setups": """\
Trap-setup rubric.

A trap setup looks bullish (or bearish) but is actually distribution.
Fakeouts, false breakouts, exhaustion patterns. The judge specifically
looks for whether the model identified the distribution pattern.

What to look for:
- Did the model name the trap mechanism? "Distribution at the highs",
  "false breakout", "exhaustion gap", "stop hunt".
- Did it cite contradicting evidence the visual narrative ignored?
- Models that take the bait deserve directional_correctness = 1 even
  if the conviction was modest.
""",

    "high_impact_events": """\
High-impact-event rubric.

The scenario involves an upcoming or recent macro event (FOMC, CPI,
NFP). The correct response is risk awareness — typically SKIP within
the blackout window.

What to look for:
- Did the reasoning name the event explicitly?
- Did it apply the right action (SKIP within blackout, reduced size
  outside)?
- Did it identify two-directional whipsaw risk?
""",

    "edge_cases": """\
Edge-case rubric.

Low liquidity, extreme funding, weekend gaps, exotic assets. The
correct response is usually SKIP, and the reasoning must identify the
specific edge condition.
""",

    "cross_tf_conflicts": """\
Cross-timeframe-conflict rubric.

1h says one thing, 4h says another. The model must integrate the
parent-TF context, not just the trading-TF read.
""",

    "flow_divergence": """\
Flow-divergence rubric.

Price moves one way, funding/OI/positioning move the other. The model
must read the flow data and let it modulate conviction.
""",
}


def get_rubric(category: str) -> str:
    """Return the rubric text for ``category`` (or a generic default)."""
    return CATEGORY_RUBRICS.get(category, _DEFAULT_RUBRIC)
