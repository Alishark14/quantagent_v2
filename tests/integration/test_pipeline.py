"""Integration tests for AnalysisPipeline.

Full pipeline with mock LLM, mock exchange adapter, SQLite repos.
Tests the complete Data -> Signal -> Conviction -> Execution flow.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from engine.config import TradingConfig
from engine.conviction.agent import ConvictionAgent
from engine.data.flow import FlowAgent
from engine.data.ohlcv import OHLCVFetcher
from engine.events import (
    ConvictionScored,
    CycleCompleted,
    DataReady,
    Event,
    InProcessBus,
    SignalsReady,
)
from engine.execution.agent import DecisionAgent
from engine.execution.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)
from engine.memory.cross_bot import CrossBotSignals
from engine.memory.cycle_memory import CycleMemory
from engine.memory.reflection_rules import ReflectionRules
from engine.memory.regime_history import RegimeHistory
from engine.pipeline import AnalysisPipeline
from engine.signals.base import SignalProducer
from engine.signals.registry import SignalRegistry
from engine.types import (
    AdapterCapabilities,
    MarketData,
    OrderResult,
    Position,
    SignalOutput,
    TradeAction,
)
from exchanges.base import ExchangeAdapter
from llm.base import LLMProvider, LLMResponse
from storage.repositories.sqlite import SQLiteRepositories


# ---------------------------------------------------------------------------
# Mock Exchange Adapter
# ---------------------------------------------------------------------------

class MockAdapter(ExchangeAdapter):
    """Minimal exchange adapter that returns synthetic data."""

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market", "limit", "stop"], supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict]:
        candles = []
        for i in range(limit):
            c = 65000.0 + i * 10
            candles.append({
                "timestamp": 1700000000 + i * 3600,
                "open": c - 5, "high": c + 20, "low": c - 20,
                "close": c, "volume": 5000.0,
            })
        return candles

    async def get_ticker(self, symbol: str) -> dict:
        return {"last": 66490.0, "bid": 66489.0, "ask": 66491.0}

    async def get_balance(self) -> float:
        return 10000.0

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        return []

    async def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult:
        return OrderResult(success=True, order_id="mock-1", fill_price=66490.0, fill_size=size, error=None)

    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult:
        return OrderResult(success=True, order_id="mock-2", fill_price=price, fill_size=size, error=None)

    async def place_sl_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="mock-sl", fill_price=None, fill_size=None, error=None)

    async def place_tp_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="mock-tp", fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def cancel_all_orders(self, symbol: str) -> int:
        return 0

    async def close_position(self, symbol: str) -> OrderResult:
        return OrderResult(success=True, order_id="mock-close", fill_price=66490.0, fill_size=0.1, error=None)

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="mock-sl-mod", fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="mock-tp-mod", fill_price=None, fill_size=None, error=None)

    async def get_funding_rate(self, symbol: str) -> float | None:
        return 0.0001

    async def get_open_interest(self, symbol: str) -> float | None:
        return 500000000.0


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Returns configurable responses for different agents."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._default = json.dumps({
            "direction": "BULLISH", "confidence": 0.7,
            "reasoning": "Mock signal.", "contradictions": "none",
            "key_levels": {},
        })
        self.call_log: list[str] = []

    async def generate_text(
        self, system_prompt: str, user_prompt: str, agent_name: str, **kwargs
    ) -> LLMResponse:
        self.call_log.append(agent_name)
        content = self._responses.get(agent_name, self._default)
        return LLMResponse(
            content=content, input_tokens=500, output_tokens=100,
            cost=0.008, model="mock-model", latency_ms=100.0,
            cached_input_tokens=400,
        )

    async def generate_vision(self, **kwargs) -> LLMResponse:
        agent_name = kwargs.get("agent_name", "vision")
        self.call_log.append(agent_name)
        content = self._responses.get(agent_name, self._default)
        return LLMResponse(
            content=content, input_tokens=500, output_tokens=100,
            cost=0.009, model="mock-model", latency_ms=150.0,
            cached_input_tokens=400,
        )


# ---------------------------------------------------------------------------
# Mock Signal Producer (simpler than real agents for integration tests)
# ---------------------------------------------------------------------------

class MockSignalProducer(SignalProducer):
    """Always returns a fixed signal."""

    def __init__(self, agent_name: str, direction: str = "BULLISH", confidence: float = 0.7) -> None:
        self._name = agent_name
        self._direction = direction
        self._confidence = confidence

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        return True

    def requires_vision(self) -> bool:
        return False

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        return SignalOutput(
            agent_name=self._name,
            signal_type="llm",
            direction=self._direction,
            confidence=self._confidence,
            reasoning=f"Mock {self._name} analysis.",
            signal_category="directional",
            data_richness="full",
            contradictions="none",
            key_levels={},
            pattern_detected=None,
            raw_output="mock",
        )


class FailingSignalProducer(SignalProducer):
    """Always raises an exception."""

    def name(self) -> str:
        return "failing_agent"

    def signal_type(self) -> str:
        return "llm"

    def is_enabled(self) -> bool:
        return True

    def requires_vision(self) -> bool:
        return False

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        raise RuntimeError("Agent crashed!")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _conviction_response() -> str:
    return json.dumps({
        "conviction_score": 0.72,
        "direction": "LONG",
        "regime": "TRENDING_UP",
        "regime_confidence": 0.82,
        "signal_quality": "HIGH",
        "contradictions": [],
        "reasoning": "Strong bullish consensus.",
        "factual_weight": 0.4,
        "subjective_weight": 0.6,
    })


def _low_conviction_response() -> str:
    return json.dumps({
        "conviction_score": 0.25,
        "direction": "SKIP",
        "regime": "RANGING",
        "regime_confidence": 0.5,
        "signal_quality": "CONFLICTING",
        "contradictions": ["agents disagree"],
        "reasoning": "No consensus.",
        "factual_weight": 0.5,
        "subjective_weight": 0.5,
    })


def _decision_response(action: str = "LONG") -> str:
    return json.dumps({
        "action": action,
        "reasoning": f"Mock decision: {action}",
        "suggested_rr": None,
    })


@pytest_asyncio.fixture
async def repos(tmp_path):
    db_path = str(tmp_path / "test_pipeline.db")
    r = SQLiteRepositories(db_path=db_path)
    await r.init_db()
    return r


def _build_pipeline(
    repos,
    llm: MockLLMProvider,
    signal_producers: list[SignalProducer] | None = None,
) -> tuple[AnalysisPipeline, InProcessBus]:
    """Build a full pipeline with all dependencies wired up."""
    config = TradingConfig(
        symbol="BTC-USDC", timeframe="1h",
        conviction_threshold=0.5, account_balance=10000.0,
    )
    adapter = MockAdapter()
    ohlcv = OHLCVFetcher(adapter, config)
    flow_agent = FlowAgent()

    registry = SignalRegistry()
    if signal_producers:
        for p in signal_producers:
            registry.register(p)
    else:
        registry.register(MockSignalProducer("indicator_agent", "BULLISH", 0.72))
        registry.register(MockSignalProducer("pattern_agent", "BULLISH", 0.80))
        registry.register(MockSignalProducer("trend_agent", "NEUTRAL", 0.50))

    conviction = ConvictionAgent(llm)
    decision = DecisionAgent(llm, config)
    bus = InProcessBus()

    cycle_mem = CycleMemory(repos.cycles)
    rules = ReflectionRules(repos.rules)
    cross_bot = CrossBotSignals(repos.cross_bot)
    regime = RegimeHistory()

    pipeline = AnalysisPipeline(
        ohlcv_fetcher=ohlcv,
        flow_agent=flow_agent,
        signal_registry=registry,
        conviction_agent=conviction,
        decision_agent=decision,
        event_bus=bus,
        cycle_memory=cycle_mem,
        reflection_rules=rules,
        cross_bot=cross_bot,
        regime_history=regime,
        cycle_repo=repos.cycles,
        config=config,
        bot_id="test-bot-1",
        user_id="test-user-1",
    )
    return pipeline, bus


# ---------------------------------------------------------------------------
# Tests: Successful cycle
# ---------------------------------------------------------------------------

class TestPipelineSuccessfulCycle:

    @pytest.mark.asyncio
    async def test_full_cycle_returns_action(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        action = await pipeline.run_cycle()

        assert isinstance(action, TradeAction)
        assert action.action == "LONG"
        assert action.conviction_score == 0.72

    @pytest.mark.asyncio
    async def test_llm_called_for_conviction_and_decision(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        await pipeline.run_cycle()

        assert "conviction_agent" in llm.call_log
        assert "decision_agent" in llm.call_log

    @pytest.mark.asyncio
    async def test_skip_action_when_low_conviction(self, repos) -> None:
        """Low conviction → DecisionAgent threshold check → SKIP without LLM call."""
        llm = MockLLMProvider({
            "conviction_agent": _low_conviction_response(),
            "decision_agent": _decision_response("SKIP"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"


# ---------------------------------------------------------------------------
# Tests: Agent failure handling
# ---------------------------------------------------------------------------

class TestPipelineAgentFailure:

    @pytest.mark.asyncio
    async def test_one_agent_fails_others_continue(self, repos) -> None:
        """If one signal agent crashes, the others still produce signals."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        producers = [
            MockSignalProducer("indicator_agent", "BULLISH", 0.72),
            FailingSignalProducer(),  # this one crashes
            MockSignalProducer("trend_agent", "BULLISH", 0.65),
        ]
        pipeline, bus = _build_pipeline(repos, llm, signal_producers=producers)

        action = await pipeline.run_cycle()

        # Pipeline should still complete with 2 out of 3 signals
        assert action.action in ("LONG", "SKIP")
        assert "conviction_agent" in llm.call_log

    @pytest.mark.asyncio
    async def test_all_agents_fail_returns_skip(self, repos) -> None:
        """If all signal agents crash, pipeline returns SKIP."""
        llm = MockLLMProvider({})
        producers = [FailingSignalProducer()]
        pipeline, bus = _build_pipeline(repos, llm, signal_producers=producers)

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        assert "No signals produced" in action.reasoning

    @pytest.mark.asyncio
    async def test_conviction_parse_failure_returns_skip(self, repos) -> None:
        """ConvictionAgent parse failure → score=0.0 → below threshold → SKIP."""
        llm = MockLLMProvider({
            "conviction_agent": "This is garbage, not JSON",
            "decision_agent": _decision_response("SKIP"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        action = await pipeline.run_cycle()

        # ConvictionAgent returns score=0.0 on parse failure
        # DecisionAgent sees conviction < threshold → SKIP without LLM call
        assert action.action == "SKIP"


# ---------------------------------------------------------------------------
# Tests: Cycle record persistence
# ---------------------------------------------------------------------------

class TestPipelinePersistence:

    @pytest.mark.asyncio
    async def test_cycle_record_saved(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        await pipeline.run_cycle()

        cycles = await repos.cycles.get_recent_cycles("test-bot-1", limit=10)
        assert len(cycles) == 1
        assert cycles[0]["symbol"] == "BTC-USDC"
        assert cycles[0]["action"] == "LONG"

    @pytest.mark.asyncio
    async def test_cross_bot_signal_published_for_directional(self, repos) -> None:
        """Cross-bot signal should be published when conviction is LONG or SHORT."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),  # direction: LONG
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        await pipeline.run_cycle()

        signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "test-user-1")
        assert len(signals) == 1
        assert signals[0]["direction"] == "LONG"
        assert signals[0]["bot_id"] == "test-bot-1"

    @pytest.mark.asyncio
    async def test_no_cross_bot_signal_for_skip(self, repos) -> None:
        """Cross-bot signal should NOT be published when conviction is SKIP."""
        llm = MockLLMProvider({
            "conviction_agent": _low_conviction_response(),  # direction: SKIP
            "decision_agent": _decision_response("SKIP"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        await pipeline.run_cycle()

        signals = await repos.cross_bot.get_recent_signals("BTC-USDC", "test-user-1")
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: Event emission
# ---------------------------------------------------------------------------

class TestPipelineEvents:

    @pytest.mark.asyncio
    async def test_events_emitted_at_each_stage(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        received_events: list[str] = []

        def on_data_ready(event: DataReady) -> None:
            received_events.append("DataReady")

        def on_signals_ready(event: SignalsReady) -> None:
            received_events.append("SignalsReady")

        def on_conviction_scored(event: ConvictionScored) -> None:
            received_events.append("ConvictionScored")

        def on_cycle_completed(event: CycleCompleted) -> None:
            received_events.append("CycleCompleted")

        bus.subscribe(DataReady, on_data_ready)
        bus.subscribe(SignalsReady, on_signals_ready)
        bus.subscribe(ConvictionScored, on_conviction_scored)
        bus.subscribe(CycleCompleted, on_cycle_completed)

        await pipeline.run_cycle()

        assert "DataReady" in received_events
        assert "SignalsReady" in received_events
        assert "ConvictionScored" in received_events
        assert "CycleCompleted" in received_events

    @pytest.mark.asyncio
    async def test_events_emitted_in_order(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        order: list[str] = []

        bus.subscribe(DataReady, lambda e: order.append("data"))
        bus.subscribe(SignalsReady, lambda e: order.append("signals"))
        bus.subscribe(ConvictionScored, lambda e: order.append("conviction"))
        bus.subscribe(CycleCompleted, lambda e: order.append("cycle"))

        await pipeline.run_cycle()

        assert order == ["data", "signals", "conviction", "cycle"]

    @pytest.mark.asyncio
    async def test_cycle_completed_has_correct_data(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        captured: list[CycleCompleted] = []
        bus.subscribe(CycleCompleted, lambda e: captured.append(e))

        await pipeline.run_cycle()

        assert len(captured) == 1
        assert captured[0].symbol == "BTC-USDC"
        assert captured[0].action == "LONG"
        assert captured[0].conviction == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_no_signals_ready_when_all_fail(self, repos) -> None:
        """If all agents fail, SignalsReady should NOT be emitted."""
        llm = MockLLMProvider({})
        producers = [FailingSignalProducer()]
        pipeline, bus = _build_pipeline(repos, llm, signal_producers=producers)

        received: list[str] = []
        bus.subscribe(DataReady, lambda e: received.append("DataReady"))
        bus.subscribe(SignalsReady, lambda e: received.append("SignalsReady"))

        await pipeline.run_cycle()

        assert "DataReady" in received
        assert "SignalsReady" not in received


# ---------------------------------------------------------------------------
# Tests: Regime history update
# ---------------------------------------------------------------------------

class TestPipelineRegimeHistory:

    @pytest.mark.asyncio
    async def test_regime_updated_after_cycle(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, bus = _build_pipeline(repos, llm)

        await pipeline.run_cycle()

        # Access the regime history through the pipeline's internal state
        assert pipeline._regime.current_regime() == "TRENDING_UP"
        assert len(pipeline._regime.get_history()) == 1


# ---------------------------------------------------------------------------
# Tests: DecisionAgent no longer receives account_balance
# ---------------------------------------------------------------------------
#
# Sprint Portfolio-Risk-Manager Task 1: the pipeline no longer fetches the
# account balance to feed DecisionAgent. DecisionAgent outputs trade INTENT
# only (action + SL/TP + risk_weight) and dollar sizing is owned by the
# PortfolioRiskManager (Tasks 2-4 will wire it into the same call site).
# Until PRM lands, the pipeline simply does not call adapter.get_balance()
# during the decision stage. These tests pin that the call site is gone.


class _BalanceTrackingAdapter(MockAdapter):
    """MockAdapter that counts every ``get_balance`` call.

    Used to assert the pipeline does NOT call ``adapter.get_balance()``
    while running a cycle now that DecisionAgent has been stripped of
    its balance dependency. PRM (Tasks 2-4) will reintroduce a balance
    fetch with different semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.balance_call_count = 0

    async def get_balance(self) -> float:  # type: ignore[override]
        self.balance_call_count += 1
        return 12_345.0


class TestPipelineDoesNotFetchBalanceForDecisionAgent:
    """Pin the new contract: pipeline does not fetch balance to call
    DecisionAgent. PRM will own balance access in Task 4."""

    @pytest.mark.asyncio
    async def test_pipeline_does_not_call_adapter_get_balance(self, repos) -> None:
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })

        config = TradingConfig(
            symbol="BTC-USDC", timeframe="1h",
            conviction_threshold=0.5, account_balance=0.0,
        )
        adapter = _BalanceTrackingAdapter()
        ohlcv = OHLCVFetcher(adapter, config)
        flow_agent = FlowAgent()

        registry = SignalRegistry()
        registry.register(MockSignalProducer("indicator_agent", "BULLISH", 0.72))
        registry.register(MockSignalProducer("pattern_agent", "BULLISH", 0.80))
        registry.register(MockSignalProducer("trend_agent", "BULLISH", 0.65))

        pipeline = AnalysisPipeline(
            ohlcv_fetcher=ohlcv,
            flow_agent=flow_agent,
            signal_registry=registry,
            conviction_agent=ConvictionAgent(llm),
            decision_agent=DecisionAgent(llm, config),
            event_bus=InProcessBus(),
            cycle_memory=CycleMemory(repos.cycles),
            reflection_rules=ReflectionRules(repos.rules),
            cross_bot=CrossBotSignals(repos.cross_bot),
            regime_history=RegimeHistory(),
            cycle_repo=repos.cycles,
            config=config,
            bot_id="test-bot-no-bal",
            user_id="test-user-no-bal",
        )

        action = await pipeline.run_cycle()

        assert action.action == "LONG"
        assert adapter.balance_call_count == 0, (
            "pipeline must NOT call adapter.get_balance() — DecisionAgent "
            "no longer needs it; PRM will fetch the balance in Task 4"
        )
        # DecisionAgent outputs trade INTENT only — no dollar size.
        assert action.position_size is None
        # Conviction 0.72 from the fixture → "high" tier → risk_weight 1.15
        assert action.risk_weight == pytest.approx(1.15)


# ---------------------------------------------------------------------------
# Tests: PortfolioRiskManager wired into the analysis pipeline (Task 4)
# ---------------------------------------------------------------------------
#
# Sprint Portfolio-Risk-Manager Task 4: PRM is now an optional ctor param
# on AnalysisPipeline. When wired, the pipeline calls `prm.size_trade(...)`
# AFTER DecisionAgent returns for entry actions only (LONG/SHORT/ADD_LONG/
# ADD_SHORT) and either stamps the resulting `position_size_usd` onto
# `action.position_size` or converts the action to SKIP with a PRM-attributed
# reason. Non-entry actions bypass PRM entirely. The pipeline tracks
# `_peak_equity` per-bot for the Layer 6 drawdown throttle.


class _PortfolioStateAdapter(MockAdapter):
    """MockAdapter with configurable balance + positions for PRM tests.

    Records every call so tests can assert on the call shape (PRM
    should call get_balance once + get_positions once per entry-action
    cycle, and zero times for non-entry actions).
    """

    def __init__(
        self,
        balance: float | Exception = 10_000.0,
        positions: list | Exception | None = None,
    ) -> None:
        super().__init__()
        self._balance = balance
        self._positions = positions if positions is not None else []
        self.balance_calls = 0
        self.positions_calls = 0

    async def get_balance(self) -> float:  # type: ignore[override]
        self.balance_calls += 1
        if isinstance(self._balance, Exception):
            raise self._balance
        return self._balance

    async def get_positions(self, symbol: str | None = None):  # type: ignore[override]
        self.positions_calls += 1
        if isinstance(self._positions, Exception):
            raise self._positions
        return list(self._positions)


def _build_pipeline_with_prm(
    repos,
    llm: MockLLMProvider,
    *,
    adapter: ExchangeAdapter | None = None,
    prm: PortfolioRiskManager | None = None,
    decision_response: str = '{"action": "LONG", "reasoning": "test", "suggested_rr": null}',
) -> tuple[AnalysisPipeline, _PortfolioStateAdapter]:
    """Wire a pipeline with an explicit PRM for the Task 4 wiring tests.

    Defaults to the spec PortfolioRiskConfig with both caps relaxed
    so Layer 1 wins by default — individual tests opt back into the
    spec caps when they want Layer 2/3 behavior. Returns the
    constructed adapter so tests can assert on `balance_calls` and
    `positions_calls`.
    """
    config = TradingConfig(
        symbol="BTC-USDC", timeframe="1h",
        conviction_threshold=0.5, account_balance=10000.0,
    )
    if adapter is None:
        adapter = _PortfolioStateAdapter(balance=10_000.0, positions=[])
    if prm is None:
        prm = PortfolioRiskManager(
            PortfolioRiskConfig(per_asset_cap_pct=100.0, portfolio_cap_pct=100.0)
        )

    ohlcv = OHLCVFetcher(adapter, config)
    flow_agent = FlowAgent()

    registry = SignalRegistry()
    registry.register(MockSignalProducer("indicator_agent", "BULLISH", 0.72))
    registry.register(MockSignalProducer("pattern_agent", "BULLISH", 0.80))
    registry.register(MockSignalProducer("trend_agent", "BULLISH", 0.65))

    pipeline = AnalysisPipeline(
        ohlcv_fetcher=ohlcv,
        flow_agent=flow_agent,
        signal_registry=registry,
        conviction_agent=ConvictionAgent(llm),
        decision_agent=DecisionAgent(llm, config),
        event_bus=InProcessBus(),
        cycle_memory=CycleMemory(repos.cycles),
        reflection_rules=ReflectionRules(repos.rules),
        cross_bot=CrossBotSignals(repos.cross_bot),
        regime_history=RegimeHistory(),
        cycle_repo=repos.cycles,
        config=config,
        bot_id="test-bot-prm",
        user_id="test-user-prm",
        portfolio_risk_manager=prm,
    )
    return pipeline, adapter  # type: ignore[return-value]


class TestPipelinePortfolioRiskManagerWiring:
    """Pin the contract that PRM is called with the right inputs and
    its result flows through to the TradeAction."""

    @pytest.mark.asyncio
    async def test_prm_called_for_long_action_populates_position_size(
        self, repos
    ) -> None:
        """LONG action → pipeline calls PRM → action.position_size set
        to the canonical Layer 1 sizing math against the live balance.

        We pin the exact PRM math (`risk_dollars / sl_distance_pct`)
        rather than dollar bounds because synthetic test candles have
        low volatility → tiny ATR → tiny SL distance → huge position
        notional. The fact that the leveraged notional is large is a
        function of the test fixture's synthetic price series, not a
        bug in the wiring. What matters is the math is correct.
        """
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, adapter = _build_pipeline_with_prm(repos, llm)

        action = await pipeline.run_cycle()

        assert action.action == "LONG"
        assert action.position_size is not None
        assert action.position_size > 0
        # PRM ran with the live balance and the action's risk_weight
        # set by DecisionAgent (conviction 0.72 → "high" tier → 1.15)
        assert action.risk_weight == pytest.approx(1.15)
        # Adapter was queried once for balance + once for positions
        assert adapter.balance_calls == 1
        assert adapter.positions_calls == 1
        # Pin the canonical PRM math: risk_dollars / sl_distance_pct.
        # equity * risk_pct * weight = 10000 * 0.01 * 1.15 = 115.0
        current_price = 66490.0  # from MockAdapter._fetch_ohlcv last candle
        sl_distance_pct = abs(current_price - action.sl_price) / current_price
        expected_size = round(115.0 / sl_distance_pct, 2)
        assert action.position_size == pytest.approx(expected_size)

    @pytest.mark.asyncio
    async def test_prm_skip_converts_action_to_skip(self, repos) -> None:
        """When PRM returns skipped=True, the pipeline converts the action
        to SKIP with the PRM reason appended to the reasoning."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        # Force PRM to skip via the LLM cost floor — set the multiplier
        # absurdly high so any plausible trade fails. Cost floor checks
        # `expected_profit < cycle_cost * multiplier`, so multiplier=10000
        # means min_profit = 0.025 * 10000 = $250.
        prm = PortfolioRiskManager(
            PortfolioRiskConfig(
                per_asset_cap_pct=100.0,
                portfolio_cap_pct=100.0,
                cost_floor_multiplier=10000.0,
            )
        )
        adapter = _PortfolioStateAdapter(balance=100.0, positions=[])
        pipeline, _adapter = _build_pipeline_with_prm(
            repos, llm, adapter=adapter, prm=prm
        )

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        assert "PRM" in action.reasoning
        assert "LLM cost floor" in action.reasoning
        # Position size + risk fields cleared
        assert action.position_size is None
        assert action.sl_price is None
        assert action.tp1_price is None
        assert action.risk_weight is None
        # Conviction score is preserved on the SKIP so audit logs
        # show what DecisionAgent originally intended
        assert action.conviction_score == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_prm_not_called_for_skip_action(self, repos) -> None:
        """SKIP action → DecisionAgent quick-exit → PRM never called."""
        llm = MockLLMProvider({
            "conviction_agent": _low_conviction_response(),  # 0.25 → SKIP
            "decision_agent": _decision_response("SKIP"),
        })
        pipeline, adapter = _build_pipeline_with_prm(repos, llm)

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        # PRM is gated on the action — non-entry actions never reach it
        assert adapter.balance_calls == 0
        assert adapter.positions_calls == 0

    @pytest.mark.asyncio
    async def test_prm_not_called_for_hold_action(self, repos) -> None:
        """HOLD action (existing position) → PRM never called."""
        # We need DecisionAgent to actually return HOLD. The simplest
        # path is to mock the decision_agent's response to be a HOLD
        # JSON; but DecisionAgent has its own quick-exit + safety-check
        # paths. Easier: patch the decide() method on the pipeline's
        # decision agent to return a HOLD action directly.
        from engine.types import TradeAction

        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("HOLD"),
        })
        pipeline, adapter = _build_pipeline_with_prm(repos, llm)

        async def fake_decide(**kwargs):
            return TradeAction(
                action="HOLD",
                conviction_score=0.72,
                position_size=None,
                sl_price=None,
                tp1_price=None,
                tp2_price=None,
                rr_ratio=None,
                atr_multiplier=None,
                reasoning="HOLD test",
                raw_output="",
                risk_weight=None,
            )

        pipeline._decision.decide = fake_decide  # type: ignore[method-assign]

        action = await pipeline.run_cycle()

        assert action.action == "HOLD"
        assert adapter.balance_calls == 0
        assert adapter.positions_calls == 0

    @pytest.mark.asyncio
    async def test_prm_balance_fetch_failure_skips(self, repos) -> None:
        """If get_balance() raises, the cycle SKIPs with a clear reason."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        adapter = _PortfolioStateAdapter(
            balance=RuntimeError("API down"),
            positions=[],
        )
        pipeline, _adapter = _build_pipeline_with_prm(repos, llm, adapter=adapter)

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        assert "PRM balance fetch failed" in action.reasoning

    @pytest.mark.asyncio
    async def test_prm_positions_fetch_failure_skips(self, repos) -> None:
        """If get_positions() raises, the cycle SKIPs with a clear reason."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        adapter = _PortfolioStateAdapter(
            balance=10_000.0,
            positions=RuntimeError("Positions API timeout"),
        )
        pipeline, _adapter = _build_pipeline_with_prm(repos, llm, adapter=adapter)

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        assert "PRM positions fetch failed" in action.reasoning

    @pytest.mark.asyncio
    async def test_prm_zero_balance_skips(self, repos) -> None:
        """A non-positive balance from the adapter SKIPs the cycle."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        adapter = _PortfolioStateAdapter(balance=0.0, positions=[])
        pipeline, _adapter = _build_pipeline_with_prm(repos, llm, adapter=adapter)

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        assert "non-positive balance" in action.reasoning

    @pytest.mark.asyncio
    async def test_pipeline_normalizes_positions_for_prm(self, repos) -> None:
        """Adapter Position objects → PRM dict shape with abs(size * entry_price)."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        # Pre-existing BTC long: 0.1 BTC at $50k entry → notional $5000
        positions = [
            Position(
                symbol="BTC-USDC",
                direction="long",
                size=0.1,
                entry_price=50_000.0,
                unrealized_pnl=0.0,
                leverage=1.0,
            ),
        ]
        adapter = _PortfolioStateAdapter(
            balance=10_000.0, positions=positions
        )
        # Use spec defaults so the per-asset cap actually fires:
        # cap = $10k * 15% = $1500; existing BTC = $5000 → ≥ cap → SKIP
        prm = PortfolioRiskManager(PortfolioRiskConfig())
        pipeline, _adapter = _build_pipeline_with_prm(
            repos, llm, adapter=adapter, prm=prm
        )

        action = await pipeline.run_cycle()

        assert action.action == "SKIP"
        # The skip reason should attribute it to Layer 2 (per-asset cap)
        # not Layer 3 (portfolio cap) because BTC alone already
        # exceeds the per-asset bucket
        assert "Per-asset cap" in action.reasoning
        assert "BTC-USDC" in action.reasoning

    @pytest.mark.asyncio
    async def test_pipeline_short_action_uses_prm(self, repos) -> None:
        """SHORT actions also flow through PRM."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("SHORT"),
        })
        pipeline, adapter = _build_pipeline_with_prm(repos, llm)

        # Have to also stub _decision because DecisionAgent's safety
        # check will reject SHORT when we have no SHORT setup; we
        # just want to verify PRM is called for SHORT actions.
        from engine.types import TradeAction

        async def fake_decide(**kwargs):
            return TradeAction(
                action="SHORT",
                conviction_score=0.72,
                position_size=None,
                sl_price=66_500.0,  # SHORT SL is above entry
                tp1_price=64_500.0,
                tp2_price=63_500.0,
                rr_ratio=2.0,
                atr_multiplier=2.0,
                reasoning="SHORT test",
                raw_output="",
                risk_weight=1.15,
            )

        pipeline._decision.decide = fake_decide  # type: ignore[method-assign]

        action = await pipeline.run_cycle()

        assert action.action == "SHORT"
        assert action.position_size is not None
        assert action.position_size > 0
        assert adapter.balance_calls == 1
        assert adapter.positions_calls == 1


class TestPipelinePeakEquityTracking:
    """Pin the per-bot peak equity tracker for the drawdown throttle."""

    @pytest.mark.asyncio
    async def test_peak_equity_starts_at_zero(self, repos) -> None:
        """Fresh pipeline has _peak_equity == 0."""
        llm = MockLLMProvider({})
        pipeline, _adapter = _build_pipeline_with_prm(repos, llm)
        assert pipeline._peak_equity == 0.0

    @pytest.mark.asyncio
    async def test_peak_equity_seeds_from_first_balance(self, repos) -> None:
        """First successful PRM cycle sets _peak_equity to the live balance."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        adapter = _PortfolioStateAdapter(balance=12_345.0, positions=[])
        pipeline, _adapter = _build_pipeline_with_prm(
            repos, llm, adapter=adapter
        )

        await pipeline.run_cycle()

        assert pipeline._peak_equity == pytest.approx(12_345.0)

    @pytest.mark.asyncio
    async def test_peak_equity_increases_when_balance_grows(self, repos) -> None:
        """A larger balance on a subsequent cycle bumps the peak."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        adapter = _PortfolioStateAdapter(balance=10_000.0, positions=[])
        pipeline, _ = _build_pipeline_with_prm(repos, llm, adapter=adapter)

        await pipeline.run_cycle()
        assert pipeline._peak_equity == pytest.approx(10_000.0)

        # Bump the balance and run another cycle
        adapter._balance = 12_500.0
        await pipeline.run_cycle()
        assert pipeline._peak_equity == pytest.approx(12_500.0)

    @pytest.mark.asyncio
    async def test_peak_equity_does_not_decrease(self, repos) -> None:
        """A smaller balance must NOT lower the peak (drawdown depth pin)."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        adapter = _PortfolioStateAdapter(balance=10_000.0, positions=[])
        pipeline, _ = _build_pipeline_with_prm(repos, llm, adapter=adapter)

        await pipeline.run_cycle()
        assert pipeline._peak_equity == pytest.approx(10_000.0)

        # Drop the balance — peak should stay
        adapter._balance = 8_000.0
        await pipeline.run_cycle()
        assert pipeline._peak_equity == pytest.approx(10_000.0)


class TestPipelinePrmOptional:
    """Pin the back-compat contract: PRM is OPTIONAL on the pipeline.

    Existing test fixtures and any production path that hasn't yet
    wired PRM must keep working. The pipeline logs a WARNING at
    construction so a misconfigured production path is loud, but
    cycles still complete and ``action.position_size`` stays as
    DecisionAgent emitted it (always None now that Task 1 stripped
    sizing from DecisionAgent).
    """

    @pytest.mark.asyncio
    async def test_pipeline_runs_without_prm(self, repos) -> None:
        """No PRM → cycle completes, action.position_size is None."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, _bus = _build_pipeline(repos, llm)  # no PRM
        action = await pipeline.run_cycle()
        assert action.action == "LONG"
        assert action.position_size is None  # nothing populated it

    @pytest.mark.asyncio
    async def test_construction_without_prm_logs_warning(
        self, repos, caplog
    ) -> None:
        """Constructing the pipeline without a PRM emits a WARNING.

        Loud-by-default behaviour for production misconfigurations
        (the test scaffolding intentionally doesn't pass a PRM, but
        a production bot factory absolutely should).
        """
        import logging

        llm = MockLLMProvider({})
        with caplog.at_level(logging.WARNING, logger="engine.pipeline"):
            _build_pipeline(repos, llm)

        assert any(
            "WITHOUT PortfolioRiskManager" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Tests: is_shadow flag plumbed from constructor into cycle records
# ---------------------------------------------------------------------------

class TestPipelineIsShadowFlag:
    """Pin the contract that ``AnalysisPipeline(is_shadow=...)`` flows
    into every persisted cycle's ``is_shadow`` column.

    This is the wiring that lets shadow-mode AND paper-mode bots'
    cycles get filtered out of the live data moat (the QuantDataScientist
    mining job reads the ``live_*`` views which strip is_shadow=True
    rows).
    """

    @pytest.mark.asyncio
    async def test_is_shadow_defaults_to_false(self, repos) -> None:
        """Existing test fixtures + production live runs build the
        pipeline without ``is_shadow=`` and must keep getting cycles
        with ``is_shadow=False``. Pins the default behaviour."""
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        pipeline, _bus = _build_pipeline(repos, llm)

        await pipeline.run_cycle()

        cycles = await repos.cycles.get_recent_cycles(
            "test-bot-1", include_shadow=True
        )
        assert len(cycles) >= 1
        # SQLite stores BOOLEAN as 0/1
        assert all(c["is_shadow"] == 0 for c in cycles)

    @pytest.mark.asyncio
    async def test_is_shadow_true_persists_into_cycle_record(self, repos) -> None:
        """When the pipeline is constructed with ``is_shadow=True``
        every cycle it persists must carry ``is_shadow=True``.

        This is the path Paper Trading Task 2 wires up — paper-mode
        and shadow-mode bots both go through this branch (Task 2's
        ``_make_bot_factory`` resolves the flag from process mode and
        passes it to the constructor).
        """
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })
        # Build the pipeline with explicit is_shadow=True. We do this
        # by hand instead of via _build_pipeline because the helper
        # doesn't expose the flag — keeps the helper signature unchanged
        # so existing tests don't have to thread False through.
        config = TradingConfig(
            symbol="BTC-USDC", timeframe="1h",
            conviction_threshold=0.5, account_balance=10_000.0,
        )
        adapter = MockAdapter()
        ohlcv = OHLCVFetcher(adapter, config)
        flow_agent = FlowAgent()
        registry = SignalRegistry()
        registry.register(MockSignalProducer("indicator_agent", "BULLISH", 0.72))
        registry.register(MockSignalProducer("pattern_agent", "BULLISH", 0.80))
        registry.register(MockSignalProducer("trend_agent", "NEUTRAL", 0.50))
        bus = InProcessBus()
        cycle_mem = CycleMemory(repos.cycles)
        rules = ReflectionRules(repos.rules)
        cross_bot = CrossBotSignals(repos.cross_bot)
        regime = RegimeHistory()

        pipeline = AnalysisPipeline(
            ohlcv_fetcher=ohlcv,
            flow_agent=flow_agent,
            signal_registry=registry,
            conviction_agent=ConvictionAgent(llm),
            decision_agent=DecisionAgent(llm, config),
            event_bus=bus,
            cycle_memory=cycle_mem,
            reflection_rules=rules,
            cross_bot=cross_bot,
            regime_history=regime,
            cycle_repo=repos.cycles,
            config=config,
            bot_id="paper-bot-test",
            user_id="paper-user",
            is_shadow=True,
        )

        await pipeline.run_cycle()

        # Default reads strip is_shadow rows — must be empty
        live_only = await repos.cycles.get_recent_cycles("paper-bot-test")
        assert live_only == []

        # include_shadow=True surfaces the row, with the flag set
        with_shadow = await repos.cycles.get_recent_cycles(
            "paper-bot-test", include_shadow=True
        )
        assert len(with_shadow) >= 1
        assert all(c["is_shadow"] == 1 for c in with_shadow)


# ---------------------------------------------------------------------------
# SimulatedExchangeAdapter price-feed integration
# ---------------------------------------------------------------------------


class TestSimAdapterPriceFeed:
    """Verify pipeline feeds current price to SimulatedExchangeAdapter.

    Without the price feed, place_market_order() raises ValueError
    because the sim has no current price to fill against.
    """

    @pytest.mark.asyncio
    async def test_pipeline_feeds_price_to_sim_adapter(self, repos) -> None:
        """run_cycle() sets current price on sim adapter so the executor
        can place orders without 'No current price' errors."""
        from backtesting.sim_exchange import SimulatedExchangeAdapter

        # Local data adapter that matches the full ExchangeAdapter.fetch_ohlcv
        # signature (including `since`) so SimulatedExchangeAdapter can delegate.
        class _SimDataAdapter(MockAdapter):
            async def fetch_ohlcv(
                self, symbol: str, timeframe: str,
                limit: int = 100, since: int | None = None,
            ) -> list[dict]:
                return await super().fetch_ohlcv(symbol, timeframe, limit)

        data_adapter = _SimDataAdapter()
        sim = SimulatedExchangeAdapter(
            initial_balance=10_000.0,
            data_adapter=data_adapter,
        )

        # Wire pipeline with the sim adapter
        config = TradingConfig(
            symbol="BTC-USDC", timeframe="1h",
            conviction_threshold=0.5,
        )
        ohlcv = OHLCVFetcher(sim, config)
        flow_agent = FlowAgent()

        registry = SignalRegistry()
        registry.register(MockSignalProducer("indicator_agent", "BULLISH", 0.72))
        registry.register(MockSignalProducer("pattern_agent", "BULLISH", 0.80))
        registry.register(MockSignalProducer("trend_agent", "BULLISH", 0.65))

        # LLM: high conviction → LONG decision with SL/TP
        llm = MockLLMProvider({
            "conviction_agent": _conviction_response(),
            "decision_agent": _decision_response("LONG"),
        })

        conviction = ConvictionAgent(llm)
        decision = DecisionAgent(llm, config)
        bus = InProcessBus()

        prm = PortfolioRiskManager(PortfolioRiskConfig())

        pipeline = AnalysisPipeline(
            ohlcv_fetcher=ohlcv,
            flow_agent=flow_agent,
            signal_registry=registry,
            conviction_agent=conviction,
            decision_agent=decision,
            event_bus=bus,
            cycle_memory=CycleMemory(repos.cycles),
            reflection_rules=ReflectionRules(repos.rules),
            cross_bot=CrossBotSignals(repos.cross_bot),
            regime_history=RegimeHistory(),
            cycle_repo=repos.cycles,
            config=config,
            bot_id="sim-test-bot",
            user_id="test-user",
            portfolio_risk_manager=prm,
        )

        # After run_cycle, the sim adapter should have a current price
        # (fed by the pipeline's Stage 1 price-feed logic).
        action = await pipeline.run_cycle()

        # The pipeline must have fed the price before returning
        assert "BTC-USDC" in sim._current_prices
        assert sim._current_prices["BTC-USDC"] > 0
