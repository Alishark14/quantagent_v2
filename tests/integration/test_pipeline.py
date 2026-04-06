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
