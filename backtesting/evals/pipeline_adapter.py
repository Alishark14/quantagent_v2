"""PipelineAdapter — bridges the eval framework to the real engine pipeline.

The eval framework (``backtesting.evals.framework.EvalRunner``) is
intentionally model-agnostic: it just calls
``await pipeline(scenario) -> EvalOutput``. This adapter is the glue
between that abstract contract and the real Signal → Conviction →
Decision pipeline that the engine runs in production.

Two modes:

* ``mode="mock"`` — uses :class:`MockSignalProducer`, no LLM calls,
  no Claude API key required. Returns deterministic ``EvalOutput`` so
  the same harness can run in CI without cost. Mock direction is
  derived from the *first character* of the scenario id so each
  scenario gets a stable, debuggable answer (still deterministic).

* ``mode="live"`` — instantiates :class:`ClaudeProvider`, builds a
  ``MarketData`` object from the scenario inputs (computing missing
  indicators if needed), runs IndicatorAgent / PatternAgent /
  TrendAgent in parallel via :class:`SignalRegistry`, runs
  :class:`ConvictionAgent`, then :class:`DecisionAgent`, and packages
  the result as :class:`EvalOutput`.

Every pipeline error — missing API key, parse failure, agent crash —
is caught and turned into a SKIP ``EvalOutput`` with the failure
reason in ``reasoning``. The eval framework grades that exactly the
same way it grades a real decision, so a broken pipeline shows up as
a regression instead of crashing the run.

See ARCHITECTURE.md §31.4.7 (tiered run specs) and §31.4.1 (output
contract).
"""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Literal

from backtesting.evals.output_contract import EvalOutput
from backtesting.evals.scenario_schema import Scenario
from backtesting.mock_signals import MockSignalProducer
from engine.types import FlowOutput, MarketData

logger = logging.getLogger(__name__)


_MODE = Literal["mock", "live"]


# Mock-mode "decisions" — derived deterministically from the scenario
# category so the same scenario always gets the same answer. This is
# only used in mock mode for unit tests + CI smoke runs. The mapping
# is intentionally biased toward SKIP so the smoke test still flags
# obvious regressions in clear_setups categories without needing real
# LLM calls.
_MOCK_CATEGORY_DEFAULTS: dict[str, tuple[str, float]] = {
    "clear_setups": ("LONG", 0.75),
    "clear_avoids": ("SKIP", 0.20),
    "conflicting_signals": ("SKIP", 0.30),
    "regime_transitions": ("SKIP", 0.30),
    "trap_setups": ("SKIP", 0.30),
    "high_impact_events": ("SKIP", 0.20),
    "edge_cases": ("SKIP", 0.20),
    "cross_tf_conflicts": ("SKIP", 0.30),
    "flow_divergence": ("SKIP", 0.30),
}


class PipelineAdapter:
    """Wraps the engine pipeline behind the EvalRunner callable contract."""

    def __init__(
        self,
        mode: _MODE = "mock",
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """
        Args:
            mode: "mock" (no LLM, deterministic) or "live" (real Claude calls).
            api_key: Anthropic API key for live mode. Falls back to
                ``ANTHROPIC_API_KEY`` env var when ``None``.
            model: Claude model id used in live mode.

        Note: Sprint Portfolio-Risk-Manager Task 1 stripped DecisionAgent of
        position sizing, so the eval adapter no longer needs an account
        balance — it asks DecisionAgent for trade INTENT only and reports
        ``position_size_pct=None`` from live runs (PRM is wired in Task 4).
        """
        if mode not in ("mock", "live"):
            raise ValueError(f"mode must be 'mock' or 'live', got {mode!r}")
        self._mode = mode
        self._model = model

        # Lazy-construct heavy live-mode dependencies so importing this
        # module doesn't require the anthropic SDK or an API key.
        self._llm = None
        self._registry = None
        self._conviction_agent = None
        self._decision_agent = None
        self._config = None

        if mode == "live":
            self._init_live(api_key)

    # ------------------------------------------------------------------
    # Live-mode wiring
    # ------------------------------------------------------------------

    def _init_live(self, api_key: str | None) -> None:
        """Construct LLMProvider + signal registry + conviction/decision agents."""
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PipelineAdapter(mode='live') requires ANTHROPIC_API_KEY "
                "in env or api_key= argument."
            )

        from engine.config import TradingConfig
        from engine.conviction.agent import ConvictionAgent
        from engine.data.flow import FlowSignalAgent
        from engine.execution.agent import DecisionAgent
        from engine.signals.indicator_agent import IndicatorAgent
        from engine.signals.pattern_agent import PatternAgent
        from engine.signals.registry import SignalRegistry
        from engine.signals.trend_agent import TrendAgent
        from llm.claude import ClaudeProvider

        self._llm = ClaudeProvider(api_key=api_key, model=self._model)
        self._config = TradingConfig()

        registry = SignalRegistry()
        registry.register(IndicatorAgent(self._llm))
        registry.register(PatternAgent(self._llm))
        registry.register(TrendAgent(self._llm))
        registry.register(FlowSignalAgent())
        self._registry = registry

        self._conviction_agent = ConvictionAgent(self._llm)
        self._decision_agent = DecisionAgent(self._llm, self._config)

    # ------------------------------------------------------------------
    # Public callable — what EvalRunner invokes
    # ------------------------------------------------------------------

    async def __call__(self, scenario: Scenario) -> EvalOutput:
        start = perf_counter()
        try:
            if self._mode == "mock":
                output = self._mock_decision(scenario)
            else:
                output = await self._live_decision(scenario)
        except Exception as e:  # pragma: no cover - exercised via test
            logger.exception(
                f"PipelineAdapter crashed on scenario {scenario.id}: {e}"
            )
            elapsed_ms = (perf_counter() - start) * 1000
            return EvalOutput(
                direction="SKIP",
                conviction=0.0,
                sl_price=None,
                tp1_price=None,
                tp2_price=None,
                position_size_pct=None,
                reasoning=f"Pipeline error: {type(e).__name__}: {e}",
                latency_ms=elapsed_ms,
                model_id=self._model_id(),
            )

        # Stamp latency on the way out so the framework can track it.
        elapsed_ms = (perf_counter() - start) * 1000
        output.latency_ms = elapsed_ms
        return output

    # Convenience for use as a duck-typed object pipeline (mirrors the
    # ``analyze`` shape EvalRunner._invoke also accepts).
    async def analyze(self, scenario: Scenario) -> EvalOutput:
        return await self(scenario)

    # ------------------------------------------------------------------
    # Mock-mode decision
    # ------------------------------------------------------------------

    def _mock_decision(self, scenario: Scenario) -> EvalOutput:
        """Deterministic decision derived from the scenario category.

        This isn't trying to be smart — it just produces stable output
        per scenario so the smoke harness has something to grade in CI
        without spending LLM tokens. ``MockSignalProducer`` is used to
        prove the SignalProducer wiring works end-to-end.
        """
        # Touch the mock producer once so the import is exercised in
        # tests and any future ABC drift is caught here.
        _ = MockSignalProducer(mode="always_skip", name="eval_mock")

        direction, conviction = _MOCK_CATEGORY_DEFAULTS.get(
            scenario.category, ("SKIP", 0.20)
        )
        last_close = self._last_close(scenario)
        sl, tp1, tp2 = self._mock_levels(direction, last_close)

        # Mock-mode never goes through the real DecisionAgent so it has no
        # risk_weight to carry — leave it None and let the framework grade
        # direction + conviction + SL/TP only. The legacy ``position_size_pct``
        # is preserved at 10% for entry directions so existing CI smoke
        # assertions on that field don't break.
        return EvalOutput(
            direction=direction,
            conviction=conviction,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            position_size_pct=0.10 if direction != "SKIP" else None,
            risk_weight=None,
            reasoning=(
                f"Mock pipeline: category={scenario.category!r} → "
                f"{direction} (conviction={conviction:.2f})"
            ),
            latency_ms=0.0,  # overwritten by __call__
            model_id="mock",
        )

    @staticmethod
    def _last_close(scenario: Scenario) -> float | None:
        ohlcv = scenario.inputs.ohlcv
        if not ohlcv:
            return None
        try:
            return float(ohlcv[-1]["close"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _mock_levels(
        direction: str, last_close: float | None
    ) -> tuple[float | None, float | None, float | None]:
        if direction == "SKIP" or last_close is None:
            return None, None, None
        # Plausible 1% / 2% / 3% bands so SL/TP fields are non-null
        # for entry actions in mock mode.
        if direction == "LONG":
            return last_close * 0.99, last_close * 1.02, last_close * 1.03
        return last_close * 1.01, last_close * 0.98, last_close * 0.97

    # ------------------------------------------------------------------
    # Live-mode decision
    # ------------------------------------------------------------------

    async def _live_decision(self, scenario: Scenario) -> EvalOutput:
        """Run the real Signal → Conviction → Decision pipeline."""
        assert self._registry is not None
        assert self._conviction_agent is not None
        assert self._decision_agent is not None

        market_data = self._scenario_to_market_data(scenario)

        # 1. Signal layer (3 LLM calls in parallel)
        signals = await self._registry.run_all(market_data)

        # 2. Conviction layer (1 LLM call)
        conviction = await self._conviction_agent.evaluate(
            signals=signals,
            market_data=market_data,
            memory_context="No prior history (eval scenario).",
        )

        # 3. Decision layer (1 LLM call)
        action = await self._decision_agent.decide(
            conviction=conviction,
            market_data=market_data,
            current_position=None,
            memory_context="No prior history (eval scenario).",
        )

        # 4. Translate TradeAction → EvalOutput
        return self._trade_action_to_eval_output(action)

    def _trade_action_to_eval_output(self, action) -> EvalOutput:
        # Map the engine's expanded action set down to the eval contract's
        # 3-state direction. ADD_LONG / HOLD on the long side count as LONG;
        # CLOSE_ALL / HOLD without a position counts as SKIP.
        direction = self._collapse_action(action.action)

        return EvalOutput(
            direction=direction,
            conviction=float(action.conviction_score),
            sl_price=action.sl_price,
            tp1_price=action.tp1_price,
            tp2_price=action.tp2_price,
            # DecisionAgent no longer outputs dollar sizing — sizing is the
            # PortfolioRiskManager's job (Sprint PRM Task 1). Until PRM is
            # wired into the eval adapter we report None and let the eval
            # framework grade direction + conviction + SL/TP only.
            position_size_pct=None,
            risk_weight=action.risk_weight,
            reasoning=action.reasoning,
            latency_ms=0.0,  # overwritten by __call__
            model_id=self._model_id(),
        )

    @staticmethod
    def _collapse_action(action: str) -> str:
        if action in ("LONG", "ADD_LONG"):
            return "LONG"
        if action in ("SHORT", "ADD_SHORT"):
            return "SHORT"
        # CLOSE_ALL, HOLD, SKIP, anything else → SKIP for eval purposes.
        return "SKIP"

    def _model_id(self) -> str:
        if self._mode == "mock":
            return "mock"
        return self._model

    # ------------------------------------------------------------------
    # Scenario → MarketData conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _scenario_to_market_data(scenario: Scenario) -> MarketData:
        """Build a MarketData object from a Scenario.

        Indicators are taken from ``scenario.inputs.indicators`` if
        present; missing fields are computed on the fly so live-mode
        agents always see a complete payload. Swing detection runs
        against the candles in the scenario.
        """
        from engine.data.indicators import compute_all_indicators
        from engine.data.swing_detection import find_swing_highs, find_swing_lows
        import numpy as np

        inputs = scenario.inputs
        candles = list(inputs.ohlcv)

        # Merge author-provided indicators on top of computed ones so
        # hand-labelled scenarios can override / inject custom values.
        computed: dict = {}
        try:
            computed = compute_all_indicators(candles)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                f"compute_all_indicators failed for {scenario.id}: {e}"
            )

        indicators = dict(computed)
        indicators.update(inputs.indicators or {})

        try:
            high = np.array([c["high"] for c in candles], dtype=float)
            low = np.array([c["low"] for c in candles], dtype=float)
            swing_highs = find_swing_highs(high)
            swing_lows = find_swing_lows(low)
        except Exception:  # pragma: no cover - defensive
            swing_highs = []
            swing_lows = []

        flow = PipelineAdapter._build_flow(inputs.flow_data)

        return MarketData(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe,
            candles=candles,
            num_candles=len(candles),
            lookback_description=f"{len(candles)} candles",
            forecast_candles=3,
            forecast_description="~3 hours",
            indicators=indicators,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            parent_tf=None,
            flow=flow,
        )

    @staticmethod
    def _build_flow(flow_data: dict | None) -> FlowOutput | None:
        if not flow_data:
            return None
        return FlowOutput(
            funding_rate=flow_data.get("funding_rate"),
            funding_signal=flow_data.get("funding_signal", "NEUTRAL"),
            oi_change_4h=flow_data.get("oi_change_pct") or flow_data.get("oi_change_4h"),
            oi_trend=flow_data.get("oi_trend", "STABLE"),
            nearest_liquidation_above=flow_data.get("nearest_liquidation_above"),
            nearest_liquidation_below=flow_data.get("nearest_liquidation_below"),
            gex_regime=flow_data.get("gex_regime"),
            gex_flip_level=flow_data.get("gex_flip_level"),
            data_richness=flow_data.get("data_richness", "PARTIAL"),
        )
