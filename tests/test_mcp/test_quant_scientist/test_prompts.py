"""Tests for the Quant Data Scientist prompt builder.

These tests pin the *contract* of the prompt: the spec-required
instructions (FDR correction, out-of-sample validation, minimum
effect size, sandbox safety) must appear verbatim, otherwise the LLM
might silently drop them and ship a statistically unsound factor.
"""

from __future__ import annotations

import pytest

from mcp.quant_scientist.prompts import (
    DEFAULT_FDR_ALPHA,
    DEFAULT_MIN_AVG_R,
    DEFAULT_MIN_SAMPLE_SIZE,
    DISCOVERY_WINDOW_MONTHS,
    SYSTEM_PROMPT,
    VALIDATION_WINDOW_MONTHS,
    build_analysis_prompt,
)


def _summary(**overrides):
    base = {
        "trade_count": 156,
        "unique_symbols": 5,
        "win_rate": 0.62,
        "avg_r": 1.4,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# System prompt invariants — these MUST appear so the analysis is rigorous
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_benjamini_hochberg():
    assert "Benjamini-Hochberg" in SYSTEM_PROMPT or "benjamini" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_out_of_sample_validation():
    assert "out-of-sample" in SYSTEM_PROMPT.lower() or "OUT-OF-SAMPLE" in SYSTEM_PROMPT


def test_system_prompt_mentions_minimum_effect_size():
    assert "MINIMUM EFFECT SIZE" in SYSTEM_PROMPT or "minimum effect" in SYSTEM_PROMPT.lower()


def test_system_prompt_lists_sandbox_forbidden_imports():
    """Belt-and-braces with the sandbox screen — also tell the LLM."""
    for forbidden in ("os", "sys", "subprocess", "open()"):
        assert forbidden in SYSTEM_PROMPT, f"system prompt must warn about {forbidden}"


def test_system_prompt_specifies_result_variable_contract():
    assert "result" in SYSTEM_PROMPT
    assert "list[dict]" in SYSTEM_PROMPT or "list of dict" in SYSTEM_PROMPT.lower()


def test_system_prompt_lists_alpha_factor_schema_keys():
    for key in ("pattern", "symbol", "timeframe", "win_rate", "avg_r", "n", "confidence"):
        assert key in SYSTEM_PROMPT, f"system prompt missing schema key {key!r}"


# ---------------------------------------------------------------------------
# build_analysis_prompt — content checks
# ---------------------------------------------------------------------------


def test_user_prompt_contains_data_split_constants():
    prompt = build_analysis_prompt(
        trade_summary=_summary(),
        available_symbols=["BTC-USDC", "ETH-USDC"],
        timeframes=["1h", "4h"],
    )
    assert f"{DISCOVERY_WINDOW_MONTHS} months" in prompt
    assert f"{VALIDATION_WINDOW_MONTHS} months" in prompt


def test_user_prompt_contains_fdr_alpha():
    prompt = build_analysis_prompt(_summary(), ["BTC-USDC"], ["1h"])
    assert f"alpha = {DEFAULT_FDR_ALPHA}" in prompt
    assert "fdr_bh" in prompt


def test_user_prompt_contains_minimum_sample_size():
    prompt = build_analysis_prompt(_summary(), ["BTC-USDC"], ["1h"])
    assert f"n >= {DEFAULT_MIN_SAMPLE_SIZE}" in prompt


def test_user_prompt_contains_minimum_avg_r():
    prompt = build_analysis_prompt(_summary(), ["BTC-USDC"], ["1h"])
    assert f"avg_r >= {DEFAULT_MIN_AVG_R}" in prompt


def test_user_prompt_warns_against_filesystem_access():
    prompt = build_analysis_prompt(_summary(), ["BTC-USDC"], ["1h"])
    assert "filesystem" in prompt.lower()


def test_user_prompt_lists_symbols_sorted_and_deduped():
    prompt = build_analysis_prompt(
        _summary(),
        available_symbols=["BTC-USDC", "ETH-USDC", "BTC-USDC"],
        timeframes=["1h"],
    )
    # Sorted + deduplicated
    assert "BTC-USDC, ETH-USDC" in prompt


def test_user_prompt_handles_empty_symbol_list():
    prompt = build_analysis_prompt(_summary(trade_count=0), [], [])
    assert "(none)" in prompt


def test_user_prompt_includes_trade_summary_stats():
    prompt = build_analysis_prompt(
        _summary(trade_count=42, unique_symbols=3, win_rate=0.55, avg_r=1.2),
        ["BTC-USDC"],
        ["1h"],
    )
    assert "total trades: 42" in prompt
    assert "unique symbols touched: 3" in prompt
    assert "55.00%" in prompt or "55.0%" in prompt
    assert "1.20" in prompt


def test_user_prompt_overrides_take_effect():
    prompt = build_analysis_prompt(
        _summary(),
        ["BTC-USDC"],
        ["1h"],
        min_sample_size=25,
        min_avg_r=2.0,
        fdr_alpha=0.01,
        discovery_months=3,
        validation_months=1,
    )
    assert "n >= 25" in prompt
    assert "avg_r >= 2.0" in prompt
    assert "alpha = 0.01" in prompt
    assert "3 months" in prompt
    assert "1 months" in prompt


def test_user_prompt_mentions_workflow_steps():
    """The prompt should walk the LLM through the 7-step workflow."""
    prompt = build_analysis_prompt(_summary(), ["BTC-USDC"], ["1h"])
    for keyword in ("WORKFLOW", "discovery-window", "validation window", "AVOID"):
        assert keyword in prompt, f"workflow keyword missing: {keyword!r}"


def test_user_prompt_instructs_empty_data_handling():
    prompt = build_analysis_prompt(_summary(trade_count=0), [], [])
    assert "result = []" in prompt
    assert "Do NOT raise" in prompt
