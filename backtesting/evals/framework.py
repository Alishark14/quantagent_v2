"""EvalRunner — runs scenarios through any pipeline that satisfies a
minimal protocol and grades the outputs against the expected behaviour.

The framework is **model-agnostic** by design: ``run_scenario`` takes a
``pipeline`` argument that just needs an ``async analyze(market_data)``
method returning an ``EvalOutput``. That lets us point the same
scenarios at the live Claude pipeline, a fine-tuned student model, an
ONNX HFT model, or a deterministic mock — every consumer is graded
identically. See ARCHITECTURE.md §31.4.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.scenario_schema import Scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline protocol — anything with an async analyze() method works
# ---------------------------------------------------------------------------


class EvalPipeline(Protocol):
    """Minimal duck-typed protocol the eval framework calls.

    The argument is intentionally a plain ``Scenario`` so the eval
    framework doesn't depend on the engine's ``MarketData`` dataclass.
    Real pipelines wrap this with a thin adapter that converts a
    Scenario to MarketData; mocks just inspect the scenario directly.
    """

    async def analyze(self, scenario: Scenario) -> EvalOutput: ...


# Convenience alias for callable-style pipelines (tests + simple scripts).
PipelineCallable = Callable[[Scenario], Awaitable[EvalOutput]]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Outcome of running one scenario through one pipeline."""

    scenario_id: str
    scenario_name: str
    category: str
    runs: list[EvalOutput]
    direction_match: bool
    conviction_in_range: bool
    action_match: bool
    consistency_stdev: float  # stdev of conviction across runs (0 if 1 run)
    pass_fail: str  # "PASS" | "FAIL"
    failure_reasons: list[str] = field(default_factory=list)
    judge_score: dict | None = None  # populated separately by judge.py

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "category": self.category,
            "runs": [r.to_dict() for r in self.runs],
            "direction_match": self.direction_match,
            "conviction_in_range": self.conviction_in_range,
            "action_match": self.action_match,
            "consistency_stdev": self.consistency_stdev,
            "pass_fail": self.pass_fail,
            "failure_reasons": list(self.failure_reasons),
            "judge_score": self.judge_score,
        }


@dataclass
class CategoryStats:
    category: str
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
        }


@dataclass
class EvalReport:
    """Aggregate report across many scenarios."""

    timestamp: str
    total_scenarios: int
    runs_per_scenario: int
    overall_pass_rate: float
    by_category: list[CategoryStats]
    consistency_avg_stdev: float
    scenario_results: list[ScenarioResult]
    top_failures: list[ScenarioResult]
    model_id: str
    prompt_versions: dict
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "runs_per_scenario": self.runs_per_scenario,
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "by_category": [c.to_dict() for c in self.by_category],
            "consistency_avg_stdev": round(self.consistency_avg_stdev, 6),
            "scenario_results": [r.to_dict() for r in self.scenario_results],
            "top_failures": [r.to_dict() for r in self.top_failures],
            "model_id": self.model_id,
            "prompt_versions": dict(self.prompt_versions),
            "duration_seconds": round(self.duration_seconds, 4),
        }


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


_DEFAULT_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_SMOKE_PER_CATEGORY = 2  # scenarios per category in run_smoke
_TOP_FAILURES = 10


class EvalRunner:
    """Loads scenarios, runs a pipeline against them, scores the results."""

    def __init__(
        self,
        scenarios_dir: Path | str = _DEFAULT_SCENARIOS_DIR,
    ) -> None:
        self._scenarios_dir = Path(scenarios_dir)

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def load_scenarios(self, category: str | None = None) -> list[Scenario]:
        """Load all scenarios listed in manifest.json (optionally filtered)."""
        manifest_path = self._scenarios_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found at {manifest_path}. "
                "Did you initialise the scenarios directory?"
            )
        manifest = json.loads(manifest_path.read_text())
        scenarios: list[Scenario] = []
        for entry in manifest.get("scenarios", []):
            if category and entry.get("category") != category:
                continue
            scenario_path = self._scenarios_dir / entry["path"]
            if not scenario_path.exists():
                logger.warning(
                    f"Scenario file missing for {entry['id']}: {scenario_path}"
                )
                continue
            scenarios.append(Scenario.model_validate_json(scenario_path.read_text()))
        return scenarios

    def categories(self) -> list[str]:
        """List the distinct categories present in the manifest."""
        seen: dict[str, None] = {}
        for s in self.load_scenarios():
            seen.setdefault(s.category, None)
        return list(seen)

    # ------------------------------------------------------------------
    # Single-scenario run
    # ------------------------------------------------------------------

    async def run_scenario(
        self,
        scenario: Scenario,
        pipeline: EvalPipeline | PipelineCallable,
        runs: int = 1,
    ) -> list[EvalOutput]:
        """Run ``pipeline`` against ``scenario`` ``runs`` times.

        ``pipeline`` may be either an object with an async ``analyze``
        method or a plain async callable. Both shapes are accepted so
        tests can pass a one-line lambda.
        """
        if runs < 1:
            raise ValueError(f"runs must be >= 1, got {runs}")
        outputs: list[EvalOutput] = []
        for _ in range(runs):
            output = await self._invoke(pipeline, scenario)
            if not isinstance(output, EvalOutput):
                raise TypeError(
                    f"pipeline returned {type(output).__name__}, expected EvalOutput"
                )
            outputs.append(output)
        return outputs

    @staticmethod
    async def _invoke(
        pipeline: EvalPipeline | PipelineCallable,
        scenario: Scenario,
    ) -> EvalOutput:
        if hasattr(pipeline, "analyze"):
            return await pipeline.analyze(scenario)
        return await pipeline(scenario)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def grade(self, scenario: Scenario, runs: list[EvalOutput]) -> ScenarioResult:
        """Compare runs to expected behaviour and produce a ScenarioResult."""
        if not runs:
            return ScenarioResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                category=scenario.category,
                runs=[],
                direction_match=False,
                conviction_in_range=False,
                action_match=False,
                consistency_stdev=0.0,
                pass_fail="FAIL",
                failure_reasons=["no runs produced"],
            )

        expected = scenario.expected
        # Use the first run as the canonical sample for direction/action checks.
        # Conviction range is enforced on every run.
        first = runs[0]
        reasons: list[str] = []

        # ----- Action -----
        action_match = first.direction == expected.expected_action
        if not action_match:
            reasons.append(
                f"action {first.direction!r} != expected {expected.expected_action!r}"
            )

        # ----- Direction (looser than action — None means "any acceptable") -----
        if expected.signal_direction is None:
            direction_match = True
        else:
            direction_match = self._direction_matches(first.direction, expected.signal_direction)
            if not direction_match:
                reasons.append(
                    f"direction {first.direction!r} not consistent with "
                    f"signal_direction {expected.signal_direction!r}"
                )

        # ----- Conviction range (every run) -----
        conviction_in_range = True
        for r in runs:
            if expected.conviction_min is not None and r.conviction < expected.conviction_min:
                conviction_in_range = False
                reasons.append(
                    f"conviction {r.conviction:.3f} below min "
                    f"{expected.conviction_min:.3f}"
                )
            if expected.conviction_max is not None and r.conviction > expected.conviction_max:
                conviction_in_range = False
                reasons.append(
                    f"conviction {r.conviction:.3f} above max "
                    f"{expected.conviction_max:.3f}"
                )

        # ----- Consistency -----
        if len(runs) > 1:
            stdev = statistics.stdev(r.conviction for r in runs)
        else:
            stdev = 0.0

        passed = action_match and direction_match and conviction_in_range
        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            category=scenario.category,
            runs=runs,
            direction_match=direction_match,
            conviction_in_range=conviction_in_range,
            action_match=action_match,
            consistency_stdev=stdev,
            pass_fail="PASS" if passed else "FAIL",
            failure_reasons=reasons,
        )

    @staticmethod
    def _direction_matches(action: str, expected_signal: str) -> bool:
        """Map LONG/SHORT/SKIP onto BULLISH/BEARISH/NEUTRAL."""
        mapping = {
            "BULLISH": {"LONG"},
            "BEARISH": {"SHORT"},
            "NEUTRAL": {"SKIP"},
        }
        return action in mapping.get(expected_signal, set())

    # ------------------------------------------------------------------
    # Tiered run modes
    # ------------------------------------------------------------------

    async def run_smoke(
        self,
        pipeline: EvalPipeline | PipelineCallable,
    ) -> EvalReport:
        """Smoke test: ``_SMOKE_PER_CATEGORY`` scenarios per category, 1 run each."""
        all_scenarios = self.load_scenarios()
        per_cat: dict[str, list[Scenario]] = {}
        for s in all_scenarios:
            per_cat.setdefault(s.category, []).append(s)
        smoke = [s for group in per_cat.values() for s in group[:_SMOKE_PER_CATEGORY]]
        return await self._run_set(smoke, pipeline, runs_per_scenario=1)

    async def run_full(
        self,
        pipeline: EvalPipeline | PipelineCallable,
        runs_per_scenario: int = 3,
    ) -> EvalReport:
        """Golden master: every scenario, ``runs_per_scenario`` runs each."""
        scenarios = self.load_scenarios()
        return await self._run_set(scenarios, pipeline, runs_per_scenario)

    async def run_category(
        self,
        category: str,
        pipeline: EvalPipeline | PipelineCallable,
        runs_per_scenario: int = 3,
    ) -> EvalReport:
        """Targeted: every scenario in one category."""
        scenarios = self.load_scenarios(category=category)
        return await self._run_set(scenarios, pipeline, runs_per_scenario)

    # ------------------------------------------------------------------
    # Internal: run a set of scenarios + aggregate
    # ------------------------------------------------------------------

    async def _run_set(
        self,
        scenarios: list[Scenario],
        pipeline: EvalPipeline | PipelineCallable,
        runs_per_scenario: int,
    ) -> EvalReport:
        from time import perf_counter

        start = perf_counter()
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            try:
                runs = await self.run_scenario(scenario, pipeline, runs=runs_per_scenario)
            except Exception as e:
                logger.exception(f"Pipeline crashed on scenario {scenario.id}")
                results.append(
                    ScenarioResult(
                        scenario_id=scenario.id,
                        scenario_name=scenario.name,
                        category=scenario.category,
                        runs=[],
                        direction_match=False,
                        conviction_in_range=False,
                        action_match=False,
                        consistency_stdev=0.0,
                        pass_fail="FAIL",
                        failure_reasons=[f"pipeline error: {e}"],
                    )
                )
                continue
            results.append(self.grade(scenario, runs))
        duration = perf_counter() - start

        return self._build_report(
            results=results,
            runs_per_scenario=runs_per_scenario,
            duration=duration,
        )

    def _build_report(
        self,
        results: list[ScenarioResult],
        runs_per_scenario: int,
        duration: float,
    ) -> EvalReport:
        total = len(results)
        passed = sum(1 for r in results if r.pass_fail == "PASS")
        overall_pass_rate = passed / total if total else 0.0

        # Per-category aggregates
        cat_total: dict[str, int] = {}
        cat_passed: dict[str, int] = {}
        for r in results:
            cat_total[r.category] = cat_total.get(r.category, 0) + 1
            if r.pass_fail == "PASS":
                cat_passed[r.category] = cat_passed.get(r.category, 0) + 1
        by_category = [
            CategoryStats(
                category=cat,
                total=cat_total[cat],
                passed=cat_passed.get(cat, 0),
            )
            for cat in sorted(cat_total)
        ]

        # Consistency average across scenarios that had >1 run
        stdevs = [r.consistency_stdev for r in results if r.runs and len(r.runs) > 1]
        consistency_avg = statistics.fmean(stdevs) if stdevs else 0.0

        # Top failures (first N failed results)
        top_failures = [r for r in results if r.pass_fail == "FAIL"][:_TOP_FAILURES]

        # Pull engine + prompt versions for the audit trail
        try:
            from quantagent.version import ENGINE_VERSION, PROMPT_VERSIONS
            model_id = ENGINE_VERSION
            prompt_versions = dict(PROMPT_VERSIONS)
        except Exception:  # pragma: no cover - defensive
            model_id = "unknown"
            prompt_versions = {}

        return EvalReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            total_scenarios=total,
            runs_per_scenario=runs_per_scenario,
            overall_pass_rate=overall_pass_rate,
            by_category=by_category,
            consistency_avg_stdev=consistency_avg,
            scenario_results=results,
            top_failures=top_failures,
            model_id=model_id,
            prompt_versions=prompt_versions,
            duration_seconds=duration,
        )
