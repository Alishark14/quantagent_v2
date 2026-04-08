"""Tests for the LLM-as-Judge."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from backtesting.evals.judge import (
    JudgeScore,
    build_judge_user_prompt,
    judge_output,
    load_judge_system_prompt,
    parse_judge_response,
)
from backtesting.evals.judge_rubrics import CATEGORY_RUBRICS, get_rubric
from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.scenario_schema import (
    ExpectedBehavior,
    Scenario,
    ScenarioInput,
)


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


@dataclass
class _Resp:
    content: str


class _MockLLM:
    """Returns a canned content string for any generate_text call."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> _Resp:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "agent_name": agent_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return _Resp(content=self.content)


def _make_scenario(category: str = "clear_setups") -> Scenario:
    return Scenario(
        id="bull_flag_001",
        name="Bull flag",
        category=category,
        version=1,
        created_at="2026-04-08T00:00:00+00:00",
        last_validated="2026-04-08T00:00:00+00:00",
        inputs=ScenarioInput(
            symbol="BTC-USDC",
            timeframe="1h",
            ohlcv=[
                {"timestamp": i * 1000, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}
                for i in range(50)
            ],
            indicators={"rsi": 62.4, "atr": 145.0},
            timestamp="2026-04-08T00:00:00+00:00",
        ),
        expected=ExpectedBehavior(
            expected_action="LONG",
            signal_direction="BULLISH",
            conviction_min=0.7,
            key_features_to_mention=["bull flag", "breakout"],
        ),
    )


def _make_output(direction: str = "LONG", reasoning: str = "bull flag breakout") -> EvalOutput:
    return EvalOutput(
        direction=direction,
        conviction=0.78,
        sl_price=99.0,
        tp1_price=102.0,
        tp2_price=104.0,
        position_size_pct=0.1,
        reasoning=reasoning,
        latency_ms=2300.0,
        model_id="claude-sonnet-4-6",
    )


_GOOD_JSON_RESPONSE = json.dumps(
    {
        "directional_correctness": 4,
        "risk_identification": 3,
        "reasoning_completeness": 3,
        "confidence_calibration": 4,
        "explanations": {
            "directional_correctness": "Identified the bull flag pattern correctly",
            "risk_identification": "Mentioned overhead resistance",
            "reasoning_completeness": "Cited volume and ADX",
            "confidence_calibration": "0.78 conviction is appropriate",
        },
    }
)


# ---------------------------------------------------------------------------
# load_judge_system_prompt
# ---------------------------------------------------------------------------


def test_load_judge_system_prompt_returns_non_empty():
    prompt = load_judge_system_prompt()
    assert "QuantAgent Eval Judge" in prompt
    assert "1–5" in prompt or "1-5" in prompt
    assert "JSON" in prompt


# ---------------------------------------------------------------------------
# build_judge_user_prompt
# ---------------------------------------------------------------------------


def test_build_user_prompt_contains_scenario_and_output():
    scenario = _make_scenario()
    output = _make_output()
    rubric = "test rubric"
    body = build_judge_user_prompt(scenario, output, rubric)
    assert "bull_flag_001" in body
    assert "BTC-USDC" in body
    assert "LONG" in body
    assert "0.7800" in body
    assert "test rubric" in body


# ---------------------------------------------------------------------------
# parse_judge_response — happy paths
# ---------------------------------------------------------------------------


def test_parse_clean_json():
    score = parse_judge_response(_GOOD_JSON_RESPONSE)
    assert score.directional_correctness == 4
    assert score.risk_identification == 3
    assert score.reasoning_completeness == 3
    assert score.confidence_calibration == 4
    assert score.average == 3.5
    assert "Identified the bull flag" in score.explanations["directional_correctness"]


def test_parse_json_wrapped_in_markdown_fence():
    raw = f"```json\n{_GOOD_JSON_RESPONSE}\n```"
    score = parse_judge_response(raw)
    assert score.directional_correctness == 4


def test_parse_json_with_preamble():
    raw = f"Here is my analysis:\n\n{_GOOD_JSON_RESPONSE}\n\nDone."
    score = parse_judge_response(raw)
    assert score.directional_correctness == 4


# ---------------------------------------------------------------------------
# parse_judge_response — error paths
# ---------------------------------------------------------------------------


def test_parse_unparseable_returns_failure_score():
    score = parse_judge_response("this is not JSON at all")
    assert score.directional_correctness == 1
    assert score.risk_identification == 1
    assert "PARSE_FAILURE" in score.explanations["directional_correctness"]


def test_parse_missing_required_field_returns_failure_score():
    raw = json.dumps(
        {
            "directional_correctness": 4,
            "risk_identification": 3,
            # missing reasoning_completeness + confidence_calibration
        }
    )
    score = parse_judge_response(raw)
    assert score.directional_correctness == 1  # all 1s on missing fields


def test_parse_clamps_out_of_range_scores():
    raw = json.dumps(
        {
            "directional_correctness": 7,  # > 5
            "risk_identification": 0,  # < 1
            "reasoning_completeness": 3,
            "confidence_calibration": 4,
            "explanations": {},
        }
    )
    score = parse_judge_response(raw)
    assert score.directional_correctness == 5  # clamped down
    assert score.risk_identification == 1  # clamped up


# ---------------------------------------------------------------------------
# judge_output — full async flow with mock LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_output_uses_category_rubric_by_default():
    scenario = _make_scenario(category="clear_setups")
    output = _make_output()
    llm = _MockLLM(_GOOD_JSON_RESPONSE)

    score = await judge_output(scenario, output, llm)
    assert isinstance(score, JudgeScore)
    assert score.directional_correctness == 4

    # Mock LLM was called once with the right system prompt + a user
    # prompt that includes the rubric for this category
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["agent_name"] == "eval_judge"
    assert call["temperature"] == 0.0
    assert "QuantAgent Eval Judge" in call["system_prompt"]
    assert "Clear-setup rubric" in call["user_prompt"]


@pytest.mark.asyncio
async def test_judge_output_accepts_custom_rubric():
    llm = _MockLLM(_GOOD_JSON_RESPONSE)
    await judge_output(
        _make_scenario(),
        _make_output(),
        llm,
        rubric="CUSTOM RUBRIC TEXT",
    )
    assert "CUSTOM RUBRIC TEXT" in llm.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_judge_output_handles_malformed_response():
    """A garbage response from the LLM doesn't crash judge_output."""
    llm = _MockLLM("not even close to JSON")
    score = await judge_output(_make_scenario(), _make_output(), llm)
    assert score.directional_correctness == 1
    assert "PARSE_FAILURE" in score.explanations["directional_correctness"]


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------


def test_category_rubrics_cover_required_categories():
    """Spec mandates rubrics for at least these three categories."""
    for required in ("clear_setups", "clear_avoids", "conflicting_signals"):
        assert required in CATEGORY_RUBRICS
        assert len(CATEGORY_RUBRICS[required]) > 100


def test_get_rubric_returns_default_for_unknown_category():
    rubric = get_rubric("nonexistent_category_xyz")
    assert "no category-specific guidance" in rubric.lower()


def test_judge_score_to_dict():
    score = JudgeScore(
        directional_correctness=4,
        risk_identification=3,
        reasoning_completeness=3,
        confidence_calibration=4,
        explanations={"directional_correctness": "x"},
    )
    d = score.to_dict()
    assert d["directional_correctness"] == 4
    assert d["average"] == 3.5
    json.dumps(d)  # must serialise
