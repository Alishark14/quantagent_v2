"""LLM-as-Judge — scores eval outputs against scenario expectations.

The judge is a separate Claude call that reads the scenario, the model
output, and a category-specific rubric, then returns a 1–5 score per
dimension as JSON. See ARCHITECTURE.md §31.4.6.

Critically, the judge is NOT the same model under test. Even if both
calls use Claude, the judge runs with its own (deliberately strict)
system prompt and a separate trace, so distillation comparisons stay
meaningful.

Costs ~$0.005–$0.015 per scored scenario. The framework typically
scores only the failed scenarios on a smoke run and the full set on a
golden master run.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from backtesting.evals.judge_rubrics import get_rubric
from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.scenario_schema import Scenario

logger = logging.getLogger(__name__)


_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "judge_system_prompt.txt"
_DIMENSIONS = (
    "directional_correctness",
    "risk_identification",
    "reasoning_completeness",
    "confidence_calibration",
)


@dataclass
class JudgeScore:
    """One judge call's structured output."""

    directional_correctness: int
    risk_identification: int
    reasoning_completeness: int
    confidence_calibration: int
    explanations: dict
    raw_response: str = ""

    @property
    def average(self) -> float:
        return sum(
            (
                self.directional_correctness,
                self.risk_identification,
                self.reasoning_completeness,
                self.confidence_calibration,
            )
        ) / 4

    def to_dict(self) -> dict:
        return {
            "directional_correctness": self.directional_correctness,
            "risk_identification": self.risk_identification,
            "reasoning_completeness": self.reasoning_completeness,
            "confidence_calibration": self.confidence_calibration,
            "explanations": dict(self.explanations),
            "average": round(self.average, 4),
        }


def load_judge_system_prompt() -> str:
    """Load the strict judge system prompt from disk."""
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Judge system prompt not found at {_PROMPT_PATH}. "
            "Did you forget to create prompts/judge_system_prompt.txt?"
        )
    return _PROMPT_PATH.read_text()


def build_judge_user_prompt(
    scenario: Scenario,
    eval_output: EvalOutput,
    rubric: str,
) -> str:
    """Assemble the user message for the judge LLM call."""
    return (
        f"## SCENARIO\n"
        f"id: {scenario.id}\n"
        f"category: {scenario.category}\n"
        f"name: {scenario.name}\n"
        f"symbol: {scenario.inputs.symbol}\n"
        f"timeframe: {scenario.inputs.timeframe}\n"
        f"regime_context: {scenario.inputs.regime_context}\n"
        f"\n"
        f"### Expected behaviour\n"
        f"action: {scenario.expected.expected_action}\n"
        f"signal_direction: {scenario.expected.signal_direction}\n"
        f"conviction_min: {scenario.expected.conviction_min}\n"
        f"conviction_max: {scenario.expected.conviction_max}\n"
        f"key_features_to_mention: {scenario.expected.key_features_to_mention}\n"
        f"notes: {scenario.expected.notes or '(none)'}\n"
        f"\n"
        f"### Indicators at decision time\n"
        f"{json.dumps(scenario.inputs.indicators, indent=2)}\n"
        f"\n"
        f"## MODEL OUTPUT\n"
        f"model_id: {eval_output.model_id}\n"
        f"direction: {eval_output.direction}\n"
        f"conviction: {eval_output.conviction:.4f}\n"
        f"sl_price: {eval_output.sl_price}\n"
        f"tp1_price: {eval_output.tp1_price}\n"
        f"tp2_price: {eval_output.tp2_price}\n"
        f"reasoning: {eval_output.reasoning or '(none)'}\n"
        f"\n"
        f"## RUBRIC\n"
        f"{rubric}\n"
        f"\n"
        f"Score the four dimensions per the system prompt instructions. "
        f"Return JSON only."
    )


async def judge_output(
    scenario: Scenario,
    eval_output: EvalOutput,
    llm_provider,
    rubric: str | None = None,
) -> JudgeScore:
    """Score one ``eval_output`` against ``scenario`` via an LLM judge.

    Args:
        scenario: The scenario the output was produced from.
        eval_output: The model's output to be judged.
        llm_provider: Anything implementing ``LLMProvider`` (Claude, mock).
        rubric: Optional override. Defaults to the category rubric.

    Returns:
        ``JudgeScore`` with one int per dimension. Defensive parsing
        falls back to a 1-across-the-board score if the LLM returns
        unparseable output (the framework will then surface this as a
        judge failure rather than crashing the run).
    """
    if rubric is None:
        rubric = get_rubric(scenario.category)

    system_prompt = load_judge_system_prompt()
    user_prompt = build_judge_user_prompt(scenario, eval_output, rubric)

    response = await llm_provider.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        agent_name="eval_judge",
        max_tokens=600,
        temperature=0.0,
        cache_system_prompt=True,
    )
    raw = getattr(response, "content", str(response))
    return parse_judge_response(raw)


def parse_judge_response(raw: str) -> JudgeScore:
    """Parse a judge JSON response into a ``JudgeScore``.

    Falls back to all-1s on parse failure (with the raw response captured
    on the score so the framework can flag it for human review).
    """
    payload = _extract_json(raw)
    if payload is None:
        logger.warning(f"Judge returned unparseable response: {raw[:200]!r}")
        return _failure_score(raw)

    try:
        score = JudgeScore(
            directional_correctness=int(payload["directional_correctness"]),
            risk_identification=int(payload["risk_identification"]),
            reasoning_completeness=int(payload["reasoning_completeness"]),
            confidence_calibration=int(payload["confidence_calibration"]),
            explanations=dict(payload.get("explanations", {})),
            raw_response=raw,
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Judge response missing required fields: {e}; raw={raw[:200]!r}")
        return _failure_score(raw)

    # Clamp scores to [1, 5] — defend against off-by-one judges
    for dim in _DIMENSIONS:
        v = getattr(score, dim)
        clamped = max(1, min(5, v))
        if clamped != v:
            logger.warning(f"Judge score {dim}={v} out of range; clamping to {clamped}")
            setattr(score, dim, clamped)

    return score


def _extract_json(raw: str) -> dict | None:
    """Pull a JSON object out of the LLM response.

    Tries plain ``json.loads`` first, then a regex-find of the first
    ``{...}`` block. Models that wrap their JSON in ``markdown
    code fences`` or add a preamble are still handled.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: find the first balanced {...} block
    brace = re.search(r"\{[\s\S]*\}", raw)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _failure_score(raw: str) -> JudgeScore:
    """Conservative fallback when the judge response can't be parsed."""
    return JudgeScore(
        directional_correctness=1,
        risk_identification=1,
        reasoning_completeness=1,
        confidence_calibration=1,
        explanations={
            "directional_correctness": "PARSE_FAILURE: judge response was unparseable",
            "risk_identification": "PARSE_FAILURE",
            "reasoning_completeness": "PARSE_FAILURE",
            "confidence_calibration": "PARSE_FAILURE",
        },
        raw_response=raw,
    )
