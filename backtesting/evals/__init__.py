"""QuantAgent eval framework — model-agnostic scenario scoring.

Public API:

- :class:`EvalRunner` — load scenarios, run them through any pipeline,
  grade against expected behaviour
- :class:`EvalReport`, :class:`ScenarioResult`, :class:`CategoryStats`
- :class:`Scenario`, :class:`ScenarioInput`, :class:`ExpectedBehavior`
- :class:`EvalOutput` — the standard model output contract
- :func:`judge_output`, :class:`JudgeScore`
- :func:`get_rubric` — category → judge rubric text
- :class:`AutoMiner` — turns live trading mistakes into pending scenarios
- :func:`generate_eval_report` — JSON + HTML reporting

See ARCHITECTURE.md §31.4.
"""

from backtesting.evals.auto_miner import AutoMiner, RepositoryTradeFetcher
from backtesting.evals.framework import (
    CategoryStats,
    EvalPipeline,
    EvalReport,
    EvalRunner,
    PipelineCallable,
    ScenarioResult,
)
from backtesting.evals.judge import JudgeScore, judge_output, parse_judge_response
from backtesting.evals.judge_rubrics import CATEGORY_RUBRICS, get_rubric
from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.reporter import generate_eval_report
from backtesting.evals.scenario_schema import (
    ExpectedBehavior,
    Scenario,
    ScenarioInput,
)

__all__ = [
    "AutoMiner",
    "RepositoryTradeFetcher",
    "CATEGORY_RUBRICS",
    "CategoryStats",
    "EvalOutput",
    "EvalPipeline",
    "EvalReport",
    "EvalRunner",
    "ExpectedBehavior",
    "JudgeScore",
    "PipelineCallable",
    "Scenario",
    "ScenarioInput",
    "ScenarioResult",
    "generate_eval_report",
    "get_rubric",
    "judge_output",
    "parse_judge_response",
]
