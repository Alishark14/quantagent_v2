"""Unit tests for the TrackingModule and all 4 trackers."""

from __future__ import annotations

import pytest

from engine.events import (
    ConvictionScored,
    CycleCompleted,
    InProcessBus,
    RuleGenerated,
    SignalsReady,
    TradeClosed,
    TradeOpened,
)
from engine.types import ConvictionOutput, OrderResult, SignalOutput, TradeAction
from tracking import TrackingModule
from tracking.data_moat import DataMoatCapture
from tracking.decision import DecisionTracker
from tracking.financial import FinancialTracker
from tracking.health import HealthTracker


# ---------------------------------------------------------------------------
# Helpers: create events
# ---------------------------------------------------------------------------

def _trade_opened_event(action: str = "LONG", pnl: float = 0) -> TradeOpened:
    return TradeOpened(
        source="test",
        trade_action=TradeAction(
            action=action, conviction_score=0.72, position_size=650.0,
            sl_price=64000.0, tp1_price=66000.0, tp2_price=67500.0,
            rr_ratio=1.5, atr_multiplier=1.2, reasoning="test", raw_output="",
        ),
        order_result=OrderResult(
            success=True, order_id="ord-001", fill_price=65000.0,
            fill_size=0.01, error=None,
        ),
    )


def _trade_closed_event(pnl: float = 150.0, exit_reason: str = "TP1") -> TradeClosed:
    return TradeClosed(source="test", symbol="BTC-USDC", pnl=pnl, exit_reason=exit_reason)


def _cycle_completed_event(action: str = "LONG", conviction: float = 0.72) -> CycleCompleted:
    return CycleCompleted(source="test", symbol="BTC-USDC", action=action, conviction=conviction)


def _signals_ready_event() -> SignalsReady:
    return SignalsReady(
        source="test",
        signals=[
            SignalOutput(
                agent_name="indicator_agent", signal_type="llm",
                direction="BULLISH", confidence=0.7, reasoning="test",
                signal_category="directional", data_richness="full",
                contradictions="none", key_levels={}, pattern_detected=None,
                raw_output="",
            ),
            SignalOutput(
                agent_name="pattern_agent", signal_type="llm",
                direction="BULLISH", confidence=0.8, reasoning="test",
                signal_category="directional", data_richness="full",
                contradictions="none", key_levels={}, pattern_detected="ascending_triangle",
                raw_output="",
            ),
        ],
    )


def _conviction_scored_event(score: float = 0.72) -> ConvictionScored:
    return ConvictionScored(
        source="test",
        conviction=ConvictionOutput(
            conviction_score=score, direction="LONG", regime="TRENDING_UP",
            regime_confidence=0.8, signal_quality="HIGH", contradictions=[],
            reasoning="test", factual_weight=0.4, subjective_weight=0.6,
            raw_output="",
        ),
    )


def _rule_generated_event() -> RuleGenerated:
    return RuleGenerated(
        source="test",
        rule={"rule_text": "When RSI > 70, avoid LONG", "score": 0},
    )


# ---------------------------------------------------------------------------
# FinancialTracker
# ---------------------------------------------------------------------------

class TestFinancialTracker:

    def test_on_trade_opened(self) -> None:
        ft = FinancialTracker()
        ft.on_trade_opened(_trade_opened_event())

        assert len(ft.trades_opened) == 1
        assert ft.trades_opened[0]["action"] == "LONG"
        assert ft.trades_opened[0]["fill_price"] == 65000.0

    def test_on_trade_closed_win(self) -> None:
        ft = FinancialTracker()
        ft.on_trade_closed(_trade_closed_event(pnl=150.0))

        assert ft.trade_count == 1
        assert ft.win_count == 1
        assert ft.loss_count == 0
        assert ft.total_pnl == 150.0

    def test_on_trade_closed_loss(self) -> None:
        ft = FinancialTracker()
        ft.on_trade_closed(_trade_closed_event(pnl=-200.0, exit_reason="SL"))

        assert ft.loss_count == 1
        assert ft.total_pnl == -200.0

    def test_win_rate(self) -> None:
        ft = FinancialTracker()
        ft.on_trade_closed(_trade_closed_event(pnl=100.0))
        ft.on_trade_closed(_trade_closed_event(pnl=-50.0))
        ft.on_trade_closed(_trade_closed_event(pnl=75.0))

        assert ft.win_rate == pytest.approx(2 / 3)

    def test_win_rate_empty(self) -> None:
        ft = FinancialTracker()
        assert ft.win_rate == 0.0

    def test_summary(self) -> None:
        ft = FinancialTracker()
        ft.on_trade_opened(_trade_opened_event())
        ft.on_trade_closed(_trade_closed_event(pnl=100.0))

        s = ft.summary()
        assert s["trade_count"] == 1
        assert s["trades_opened"] == 1
        assert s["total_pnl"] == 100.0


# ---------------------------------------------------------------------------
# DecisionTracker
# ---------------------------------------------------------------------------

class TestDecisionTracker:

    def test_on_cycle_completed(self) -> None:
        dt = DecisionTracker()
        dt.on_cycle_completed(_cycle_completed_event())

        assert len(dt.cycles) == 1
        assert dt.cycles[0]["action"] == "LONG"
        assert dt.action_counts["LONG"] == 1

    def test_action_counts(self) -> None:
        dt = DecisionTracker()
        dt.on_cycle_completed(_cycle_completed_event(action="LONG"))
        dt.on_cycle_completed(_cycle_completed_event(action="SKIP"))
        dt.on_cycle_completed(_cycle_completed_event(action="SKIP"))

        assert dt.action_counts["LONG"] == 1
        assert dt.action_counts["SKIP"] == 2

    def test_on_signals_ready(self) -> None:
        dt = DecisionTracker()
        dt.on_signals_ready(_signals_ready_event())

        assert dt.signal_counts["indicator_agent"] == 1
        assert dt.signal_counts["pattern_agent"] == 1

    def test_avg_conviction(self) -> None:
        dt = DecisionTracker()
        dt.on_cycle_completed(_cycle_completed_event(conviction=0.8))
        dt.on_cycle_completed(_cycle_completed_event(conviction=0.6))

        assert dt.avg_conviction == pytest.approx(0.7)

    def test_avg_conviction_empty(self) -> None:
        dt = DecisionTracker()
        assert dt.avg_conviction == 0.0

    def test_summary(self) -> None:
        dt = DecisionTracker()
        dt.on_cycle_completed(_cycle_completed_event())

        s = dt.summary()
        assert s["total_cycles"] == 1


# ---------------------------------------------------------------------------
# HealthTracker
# ---------------------------------------------------------------------------

class TestHealthTracker:

    def test_on_any_event_counts(self) -> None:
        ht = HealthTracker()
        ht.on_any_event(_trade_opened_event())
        ht.on_any_event(_trade_closed_event())
        ht.on_any_event(_cycle_completed_event())

        assert ht.total_events == 3
        assert ht.event_counts["TradeOpened"] == 1
        assert ht.event_counts["TradeClosed"] == 1
        assert ht.event_counts["CycleCompleted"] == 1

    def test_record_error(self) -> None:
        ht = HealthTracker()
        ht.record_error("test_module", "something broke")

        assert ht.error_count == 1
        assert len(ht.errors) == 1
        assert ht.errors[0]["source"] == "test_module"

    def test_error_buffer_limit(self) -> None:
        ht = HealthTracker()
        for i in range(150):
            ht.record_error("test", f"error {i}")

        assert ht.error_count == 150
        assert len(ht.errors) == 100  # capped

    def test_uptime(self) -> None:
        ht = HealthTracker()
        assert ht.uptime_seconds >= 0

    def test_summary(self) -> None:
        ht = HealthTracker()
        ht.on_any_event(_trade_opened_event())

        s = ht.summary()
        assert s["total_events"] == 1
        assert s["error_count"] == 0


# ---------------------------------------------------------------------------
# DataMoatCapture
# ---------------------------------------------------------------------------

class TestDataMoatCapture:

    def test_capture_cycle(self) -> None:
        dm = DataMoatCapture()
        dm.capture_cycle(
            cycle_id="c-001",
            market_data={"candles": []},
            signals=[{"direction": "BULLISH"}],
            conviction={"score": 0.72},
            action="LONG",
        )

        assert len(dm.cycles_captured) == 1
        assert dm.layer_counts["L0_market"] == 1
        assert dm.layer_counts["L1_sensory"] == 1
        assert dm.layer_counts["L2_cognitive"] == 1

    def test_capture_trade(self) -> None:
        dm = DataMoatCapture()
        dm.capture_trade(
            trade_id="t-001",
            cycle_id="c-001",
            action={"action": "LONG"},
            outcome={"pnl": 150.0},
            reflection={"rule": "test"},
        )

        assert len(dm.trades_captured) == 1
        assert dm.layer_counts["L3_action"] == 1
        assert dm.layer_counts["L4_outcome"] == 1
        assert dm.layer_counts["L5_reflection"] == 1

    def test_on_cycle_completed(self) -> None:
        dm = DataMoatCapture()
        dm.on_cycle_completed(_cycle_completed_event())

        assert len(dm.cycles_captured) == 1

    def test_on_trade_opened(self) -> None:
        dm = DataMoatCapture()
        dm.on_trade_opened(_trade_opened_event())

        assert dm.layer_counts["L3_action"] == 1

    def test_on_trade_closed(self) -> None:
        dm = DataMoatCapture()
        dm.on_trade_closed(_trade_closed_event())

        assert dm.layer_counts["L4_outcome"] == 1

    def test_on_rule_generated(self) -> None:
        dm = DataMoatCapture()
        dm.on_rule_generated(_rule_generated_event())

        assert dm.layer_counts["L5_reflection"] == 1

    def test_summary(self) -> None:
        dm = DataMoatCapture()
        dm.capture_cycle("c-001", market_data={})
        dm.capture_trade("t-001", "c-001", action={})

        s = dm.summary()
        assert s["cycles_captured"] == 1
        assert s["trades_captured"] == 1


# ---------------------------------------------------------------------------
# TrackingModule (integration with event bus)
# ---------------------------------------------------------------------------

class TestTrackingModule:

    @pytest.mark.asyncio
    async def test_subscribe_all_and_receive_events(self) -> None:
        bus = InProcessBus()
        tm = TrackingModule()
        tm.subscribe_all(bus)

        await bus.publish(_trade_opened_event())
        await bus.publish(_trade_closed_event(pnl=100.0))
        await bus.publish(_cycle_completed_event())
        await bus.publish(_signals_ready_event())
        await bus.publish(_conviction_scored_event())
        await bus.publish(_rule_generated_event())

        # Financial
        assert tm.financial.trade_count == 1
        assert tm.financial.total_pnl == 100.0
        assert len(tm.financial.trades_opened) == 1

        # Decision
        assert len(tm.decision.cycles) == 1
        assert tm.decision.signal_counts.get("indicator_agent") == 1

        # Health — counts all events
        assert tm.health.total_events >= 6

        # Data Moat
        assert tm.data_moat.layer_counts["L3_action"] >= 1
        assert tm.data_moat.layer_counts["L4_outcome"] >= 1

    @pytest.mark.asyncio
    async def test_tracking_failure_does_not_propagate(self) -> None:
        """If a tracker handler crashes, the event bus continues."""
        bus = InProcessBus()
        tm = TrackingModule()

        # Sabotage the financial tracker
        def broken_handler(event):
            raise RuntimeError("tracker crashed!")

        tm.financial.on_trade_opened = broken_handler
        tm.subscribe_all(bus)

        # Should not raise — _safe wraps the broken handler
        await bus.publish(_trade_opened_event())

        # Health tracker should still have counted it
        assert tm.health.event_counts.get("TradeOpened", 0) >= 1

    @pytest.mark.asyncio
    async def test_summary(self) -> None:
        bus = InProcessBus()
        tm = TrackingModule()
        tm.subscribe_all(bus)

        await bus.publish(_cycle_completed_event())

        s = tm.summary()
        assert "financial" in s
        assert "decision" in s
        assert "health" in s
        assert "data_moat" in s
        assert s["decision"]["total_cycles"] == 1

    @pytest.mark.asyncio
    async def test_multiple_trade_lifecycle(self) -> None:
        """Full lifecycle: open -> cycle -> close for 2 trades."""
        bus = InProcessBus()
        tm = TrackingModule()
        tm.subscribe_all(bus)

        # Trade 1: win
        await bus.publish(_trade_opened_event())
        await bus.publish(_cycle_completed_event())
        await bus.publish(_trade_closed_event(pnl=200.0))

        # Trade 2: loss
        await bus.publish(_trade_opened_event(action="SHORT"))
        await bus.publish(_cycle_completed_event(action="SHORT", conviction=0.65))
        await bus.publish(_trade_closed_event(pnl=-100.0, exit_reason="SL"))

        assert tm.financial.trade_count == 2
        assert tm.financial.total_pnl == 100.0
        assert tm.financial.win_rate == 0.5
        assert len(tm.decision.cycles) == 2
