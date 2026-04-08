"""Structured Claude prompt for the alpha mining workflow.

The Quant Data Scientist agent does NOT analyse data itself — it
prompts Claude to write a Python analysis script which the agent
then executes in a restricted sandbox. This module assembles that
prompt.

The prompt MUST instruct Claude to:

1. Use the dataframes the sandbox provides (``trades_df`` and a
   per-symbol ``ohlcv``  mapping) instead of touching the filesystem.
2. Split historical data into 4 months discovery / 2 months validation
   per ARCHITECTURE.md §13.1.3.
3. Apply Benjamini-Hochberg FDR correction across all hypotheses.
4. Filter for ``p_adjusted < 0.05``, ``n >= 15``, and ``avg_R >= 1.5``.
5. Validate surviving factors on the out-of-sample window.
6. Assign the result to a sandbox variable (``result``) as a list of
   AlphaFactor-shaped dicts. The sandbox forbids file IO so the
   "Return JSON" wording in the spec maps to "set the result variable".

ARCHITECTURE.md §13.1.6 is the safety contract: the LLM never writes
files directly. The agent parses ``result`` and writes
``alpha_factors.json`` itself.
"""

from __future__ import annotations

from typing import Iterable

# These constants ARE the spec invariants — they have to appear
# verbatim in the prompt so test_prompts.py can assert their presence.
DEFAULT_MIN_SAMPLE_SIZE = 15
DEFAULT_MIN_AVG_R = 1.5
DEFAULT_FDR_ALPHA = 0.05  # Benjamini-Hochberg α — standard practice
DISCOVERY_WINDOW_MONTHS = 4
VALIDATION_WINDOW_MONTHS = 2

# Valid confidence values — Claude must pick from this set so the
# downstream AlphaFactor.validate() check passes.
_VALID_CONFIDENCE = ("high", "medium", "low")


SYSTEM_PROMPT = """\
You are the Quant Data Scientist for the QuantAgent v2 trading system.

Your job is to mine statistically rigorous alpha factors from historical
trade outcomes and OHLCV data, then return them as a structured Python
list. You write Python analysis code that runs in a SANDBOX with these
strict rules:

- You CANNOT import os, sys, subprocess, socket, pathlib, shutil, or open().
  No filesystem, network, or process access.
- You MUST use only the variables provided in the sandbox: `trades_df`
  (pandas DataFrame of recent closed trades), `ohlcv` (dict of
  symbol -> per-timeframe pandas DataFrame), `pd` (pandas), `np` (numpy),
  `stats` (scipy.stats), `multipletests` (statsmodels Benjamini-Hochberg
  helper), and `datetime` (datetime module).
- You MUST assign your final answer to a variable named `result` of
  type list[dict]. Each dict matches the AlphaFactor schema below.
- You MUST NOT call print(), logging, or anything else with side effects.

Statistical rigor (mandatory, ARCHITECTURE.md §13.1.3):

1. OUT-OF-SAMPLE VALIDATION. Split each symbol's OHLCV history into a
   4-month DISCOVERY window and a 2-month VALIDATION window. A factor
   only ships if it shows edge in BOTH windows.
2. BENJAMINI-HOCHBERG FDR CORRECTION. After computing raw p-values for
   every (pattern, symbol, timeframe) combination, run them through
   `multipletests(pvalues, alpha=0.05, method="fdr_bh")` and only keep
   the survivors. Raw p < 0.05 alone is forbidden.
3. MINIMUM EFFECT SIZE. Reject any factor with avg_R < 1.5 or n < 15
   even if it survives FDR. Statistical significance without economic
   significance is noise.

Output schema (the dicts you put in `result`):

    {
        "pattern": "ascending_triangle + RSI_below_50 + MACD_bullish",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "win_rate": 0.68,
        "avg_r": 1.9,
        "n": 23,
        "confidence": "high",     # one of {high, medium, low}
        "note": null              # or "AVOID" for negative-edge factors
    }

`discovered_at` and `last_confirmed` will be stamped by the agent on
your behalf — do not include them.

Write tight, defensive code: handle the empty-data case (set
`result = []` rather than crashing), guard against zero-division when
computing Sharpe, and never assume a symbol or timeframe exists in the
`ohlcv` dict without checking first.
"""


def build_analysis_prompt(
    trade_summary: dict,
    available_symbols: Iterable[str],
    timeframes: Iterable[str],
    *,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
    min_avg_r: float = DEFAULT_MIN_AVG_R,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    discovery_months: int = DISCOVERY_WINDOW_MONTHS,
    validation_months: int = VALIDATION_WINDOW_MONTHS,
) -> str:
    """Assemble the user prompt the agent sends to Claude.

    Args:
        trade_summary: Aggregate stats from the last 30 days of trades
            (count, symbols touched, win rate, etc.). Used so Claude
            knows what's in the sandbox before writing code.
        available_symbols: Symbols for which OHLCV is available in the
            ``ohlcv`` dict. Claude should iterate this list.
        timeframes: Timeframes per symbol that the sandbox loaded.
        min_sample_size: Minimum n per factor (default 15).
        min_avg_r: Minimum avg R per factor (default 1.5).
        fdr_alpha: Benjamini-Hochberg α threshold (default 0.05).
        discovery_months: Discovery window length (default 4).
        validation_months: Validation window length (default 2).

    Returns:
        The full user prompt as a single string. The system prompt is
        :data:`SYSTEM_PROMPT` and is sent separately.
    """
    symbol_list = ", ".join(sorted(set(available_symbols))) or "(none)"
    timeframe_list = ", ".join(sorted(set(timeframes))) or "(none)"

    return f"""\
TASK: Analyse the last 30 days of closed trades and 6 months of OHLCV
history. Discover alpha factors that satisfy the statistical rigor
requirements in the system prompt.

Trade summary (sandbox `trades_df`):
- total trades: {trade_summary.get("trade_count", 0)}
- unique symbols touched: {trade_summary.get("unique_symbols", 0)}
- overall win rate: {trade_summary.get("win_rate", 0):.2%}
- average R: {trade_summary.get("avg_r", 0):.2f}

Available OHLCV symbols (sandbox `ohlcv` keys): {symbol_list}
Available timeframes per symbol: {timeframe_list}

Statistical requirements (HARD RULES — ANY violation means reject the factor):
- Discovery window: {discovery_months} months
- Validation window: {validation_months} months (out-of-sample, must hold)
- Multiple-testing correction: Benjamini-Hochberg with alpha = {fdr_alpha}
- Minimum sample size per factor: n >= {min_sample_size}
- Minimum effect size: avg_r >= {min_avg_r}
- Both p_adjusted < {fdr_alpha} AND validation-window edge required

WORKFLOW:
1. From `trades_df`, extract every distinct (pattern, symbol, timeframe)
   triple that appears at least {min_sample_size} times.
2. Compute discovery-window stats per triple (n, win_rate, avg_r,
   sharpe, p-value vs chance using `stats.binomtest` or t-test).
3. Run `multipletests([p1, p2, ...], alpha={fdr_alpha}, method="fdr_bh")`
   on the full p-value list. Keep only p_adjusted < {fdr_alpha}.
4. Re-evaluate each survivor on the validation window. Drop any whose
   avg_r drops below {min_avg_r} or whose win_rate inverts.
5. Build the final list of AlphaFactor-shaped dicts and assign to
   `result`.
6. Mark `confidence` as "high" when n >= 30 AND p_adjusted < 0.01,
   "medium" when n >= 20 AND p_adjusted < {fdr_alpha}, otherwise "low".
7. Set `note = "AVOID"` for any factor whose discovery avg_r is
   negative — those are anti-patterns the trader should *skip*, not
   take.

If `trades_df` is empty or no factor passes the gates, set
`result = []`. Do NOT raise. Do NOT print. Do NOT touch the filesystem.
"""
