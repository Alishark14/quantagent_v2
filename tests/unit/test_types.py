"""Unit tests for engine/types.py and engine/events.py dataclasses."""

from datetime import datetime, timezone

import pytest

from engine.types import (
    AdapterCapabilities,
    ConvictionOutput,
    FlowOutput,
    MarketData,
    OrderResult,
    ParentTFContext,
    Position,
    SignalOutput,
    TradeAction,
)
from engine.events import (
    CycleCompleted,
    ConvictionScored,
    DataReady,
    Event,
    FactorsUpdated,
    MacroUpdated,
    PositionUpdated,
    RuleGenerated,
    SetupDetected,
    SignalsReady,
    TradeClosed,
    TradeOpened,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable instances
# ---------------------------------------------------------------------------

@pytest.fixture
def parent_tf() -> ParentTFContext:
    return ParentTFContext(
        timeframe="4h",
        trend_direction="BULLISH",
        ma_position="ABOVE_50MA",
        adx_value=28.5,
        adx_classification="TRENDING",
        bb_width_percentile=65.0,
    )


@pytest.fixture
def flow_output() -> FlowOutput:
    return FlowOutput(
        funding_rate=0.0001,
        funding_signal="NEUTRAL",
        oi_change_4h=2.5,
        oi_trend="BUILDING",
        nearest_liquidation_above={"price": 70000.0, "size": 5_000_000},
        nearest_liquidation_below={"price": 65000.0, "size": 3_000_000},
        gex_regime=None,
        gex_flip_level=None,
        data_richness="PARTIAL",
    )


@pytest.fixture
def market_data(parent_tf: ParentTFContext, flow_output: FlowOutput) -> MarketData:
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=[
            {"timestamp": 1700000000, "open": 67000, "high": 67500,
             "low": 66800, "close": 67200, "volume": 1500},
        ],
        num_candles=150,
        lookback_description="~6 days",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={"rsi": 55.3, "macd": {"macd": 120, "signal": 100, "histogram": 20}},
        swing_highs=[67500.0, 68000.0],
        swing_lows=[66000.0, 65500.0],
        parent_tf=parent_tf,
        flow=flow_output,
        external_signals={"quanthq": "BULLISH"},
    )


@pytest.fixture
def signal_output() -> SignalOutput:
    return SignalOutput(
        agent_name="indicator_agent",
        signal_type="llm",
        direction="BULLISH",
        confidence=0.72,
        reasoning="RSI above 50, MACD bullish cross",
        signal_category="directional",
        data_richness="full",
        contradictions="Williams %R near overbought",
        key_levels={"resistance": 68000.0, "support": 66000.0},
        pattern_detected=None,
        raw_output="<full llm response>",
    )


@pytest.fixture
def conviction_output() -> ConvictionOutput:
    return ConvictionOutput(
        conviction_score=0.78,
        direction="LONG",
        regime="TRENDING_UP",
        regime_confidence=0.85,
        signal_quality="HIGH",
        contradictions=["Williams %R overbought"],
        reasoning="Strong trend with momentum confirmation",
        factual_weight=0.6,
        subjective_weight=0.4,
        raw_output="<full llm response>",
    )


@pytest.fixture
def trade_action() -> TradeAction:
    return TradeAction(
        action="LONG",
        conviction_score=0.78,
        position_size=0.5,
        sl_price=66000.0,
        tp1_price=69000.0,
        tp2_price=71000.0,
        rr_ratio=2.5,
        atr_multiplier=1.5,
        reasoning="Enter long on trend confirmation",
        raw_output="<full llm response>",
    )


@pytest.fixture
def order_result() -> OrderResult:
    return OrderResult(
        success=True,
        order_id="HL-12345",
        fill_price=67200.0,
        fill_size=0.5,
        error=None,
    )


@pytest.fixture
def position() -> Position:
    return Position(
        symbol="BTC-USDC",
        direction="long",
        size=0.5,
        entry_price=67200.0,
        unrealized_pnl=150.0,
        leverage=5.0,
    )


@pytest.fixture
def adapter_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        native_sl_tp=True,
        supports_short=True,
        market_hours=None,
        asset_types=["crypto", "hip3_commodity", "hip3_index"],
        margin_type="cross",
        has_funding_rate=True,
        has_oi_data=True,
        max_leverage=50.0,
        order_types=["market", "limit", "stop_market"],
        supports_partial_close=True,
    )


# ---------------------------------------------------------------------------
# Construction tests — every dataclass with valid data
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_parent_tf_context(self, parent_tf: ParentTFContext) -> None:
        assert parent_tf.timeframe == "4h"
        assert parent_tf.trend_direction == "BULLISH"
        assert parent_tf.adx_value == 28.5

    def test_flow_output(self, flow_output: FlowOutput) -> None:
        assert flow_output.funding_rate == 0.0001
        assert flow_output.oi_trend == "BUILDING"
        assert flow_output.data_richness == "PARTIAL"

    def test_market_data(self, market_data: MarketData) -> None:
        assert market_data.symbol == "BTC-USDC"
        assert market_data.num_candles == 150
        assert market_data.parent_tf is not None
        assert market_data.flow is not None
        assert len(market_data.candles) == 1
        assert market_data.external_signals["quanthq"] == "BULLISH"

    def test_market_data_defaults(self) -> None:
        md = MarketData(
            symbol="ETH-USDC",
            timeframe="15m",
            candles=[],
            num_candles=0,
            lookback_description="~0",
            forecast_candles=3,
            forecast_description="~45 minutes",
            indicators={},
            swing_highs=[],
            swing_lows=[],
        )
        assert md.parent_tf is None
        assert md.flow is None
        assert md.external_signals == {}

    def test_signal_output(self, signal_output: SignalOutput) -> None:
        assert signal_output.agent_name == "indicator_agent"
        assert signal_output.confidence == 0.72
        assert signal_output.direction == "BULLISH"

    def test_conviction_output(self, conviction_output: ConvictionOutput) -> None:
        assert conviction_output.conviction_score == 0.78
        assert conviction_output.direction == "LONG"
        assert conviction_output.regime == "TRENDING_UP"
        assert len(conviction_output.contradictions) == 1

    def test_trade_action(self, trade_action: TradeAction) -> None:
        assert trade_action.action == "LONG"
        assert trade_action.sl_price == 66000.0
        assert trade_action.rr_ratio == 2.5

    def test_trade_action_skip(self) -> None:
        skip = TradeAction(
            action="SKIP",
            conviction_score=0.3,
            position_size=None,
            sl_price=None,
            tp1_price=None,
            tp2_price=None,
            rr_ratio=None,
            atr_multiplier=None,
            reasoning="Low conviction",
            raw_output="",
        )
        assert skip.action == "SKIP"
        assert skip.position_size is None

    def test_order_result(self, order_result: OrderResult) -> None:
        assert order_result.success is True
        assert order_result.order_id == "HL-12345"

    def test_order_result_failure(self) -> None:
        fail = OrderResult(
            success=False,
            order_id=None,
            fill_price=None,
            fill_size=None,
            error="Insufficient margin",
        )
        assert fail.success is False
        assert fail.error == "Insufficient margin"

    def test_position(self, position: Position) -> None:
        assert position.symbol == "BTC-USDC"
        assert position.direction == "long"
        assert position.leverage == 5.0

    def test_adapter_capabilities(self, adapter_capabilities: AdapterCapabilities) -> None:
        assert adapter_capabilities.native_sl_tp is True
        assert adapter_capabilities.market_hours is None
        assert "crypto" in adapter_capabilities.asset_types


# ---------------------------------------------------------------------------
# Required fields — missing args raise TypeError
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_market_data_requires_symbol(self) -> None:
        with pytest.raises(TypeError):
            MarketData(  # type: ignore[call-arg]
                timeframe="1h",
                candles=[],
                num_candles=0,
                lookback_description="",
                forecast_candles=3,
                forecast_description="",
                indicators={},
                swing_highs=[],
                swing_lows=[],
            )

    def test_signal_output_requires_agent_name(self) -> None:
        with pytest.raises(TypeError):
            SignalOutput(  # type: ignore[call-arg]
                signal_type="llm",
                direction="BULLISH",
                confidence=0.5,
                reasoning="",
                signal_category="directional",
                data_richness="full",
                contradictions="",
                key_levels={},
                pattern_detected=None,
                raw_output="",
            )

    def test_conviction_output_requires_score(self) -> None:
        with pytest.raises(TypeError):
            ConvictionOutput(  # type: ignore[call-arg]
                direction="LONG",
                regime="TRENDING_UP",
                regime_confidence=0.8,
                signal_quality="HIGH",
                contradictions=[],
                reasoning="",
                factual_weight=0.6,
                subjective_weight=0.4,
                raw_output="",
            )

    def test_position_requires_symbol(self) -> None:
        with pytest.raises(TypeError):
            Position(  # type: ignore[call-arg]
                direction="long",
                size=1.0,
                entry_price=67000.0,
                unrealized_pnl=0.0,
                leverage=None,
            )

    def test_order_result_requires_success(self) -> None:
        with pytest.raises(TypeError):
            OrderResult(  # type: ignore[call-arg]
                order_id=None,
                fill_price=None,
                fill_size=None,
                error=None,
            )

    def test_trade_action_requires_action(self) -> None:
        with pytest.raises(TypeError):
            TradeAction(  # type: ignore[call-arg]
                conviction_score=0.5,
                position_size=None,
                sl_price=None,
                tp1_price=None,
                tp2_price=None,
                rr_ratio=None,
                atr_multiplier=None,
                reasoning="",
                raw_output="",
            )


# ---------------------------------------------------------------------------
# Serialization — to_dict() for database storage
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_parent_tf_to_dict(self, parent_tf: ParentTFContext) -> None:
        d = parent_tf.to_dict()
        assert d["timeframe"] == "4h"
        assert d["adx_value"] == 28.5
        assert isinstance(d, dict)

    def test_flow_output_to_dict(self, flow_output: FlowOutput) -> None:
        d = flow_output.to_dict()
        assert d["funding_rate"] == 0.0001
        assert d["nearest_liquidation_above"]["price"] == 70000.0

    def test_market_data_to_dict(self, market_data: MarketData) -> None:
        d = market_data.to_dict()
        assert d["symbol"] == "BTC-USDC"
        assert isinstance(d["parent_tf"], dict)
        assert isinstance(d["flow"], dict)
        assert d["parent_tf"]["timeframe"] == "4h"

    def test_market_data_to_dict_none_nested(self) -> None:
        md = MarketData(
            symbol="ETH-USDC",
            timeframe="1h",
            candles=[],
            num_candles=0,
            lookback_description="",
            forecast_candles=3,
            forecast_description="",
            indicators={},
            swing_highs=[],
            swing_lows=[],
        )
        d = md.to_dict()
        assert d["parent_tf"] is None
        assert d["flow"] is None

    def test_signal_output_to_dict(self, signal_output: SignalOutput) -> None:
        d = signal_output.to_dict()
        assert d["agent_name"] == "indicator_agent"
        assert d["confidence"] == 0.72
        assert isinstance(d["key_levels"], dict)

    def test_conviction_output_to_dict(self, conviction_output: ConvictionOutput) -> None:
        d = conviction_output.to_dict()
        assert d["conviction_score"] == 0.78
        assert isinstance(d["contradictions"], list)

    def test_trade_action_to_dict(self, trade_action: TradeAction) -> None:
        d = trade_action.to_dict()
        assert d["action"] == "LONG"
        assert d["sl_price"] == 66000.0

    def test_order_result_to_dict(self, order_result: OrderResult) -> None:
        d = order_result.to_dict()
        assert d["success"] is True
        assert d["order_id"] == "HL-12345"

    def test_position_to_dict(self, position: Position) -> None:
        d = position.to_dict()
        assert d["symbol"] == "BTC-USDC"
        assert d["direction"] == "long"

    def test_adapter_capabilities_to_dict(self, adapter_capabilities: AdapterCapabilities) -> None:
        d = adapter_capabilities.to_dict()
        assert d["native_sl_tp"] is True
        assert d["max_leverage"] == 50.0


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class TestEvents:
    def test_base_event(self) -> None:
        e = Event(source="test_module")
        assert e.source == "test_module"
        assert isinstance(e.timestamp, datetime)
        assert e.timestamp.tzinfo == timezone.utc

    def test_data_ready(self, market_data: MarketData) -> None:
        e = DataReady(source="data_module", market_data=market_data)
        assert e.market_data.symbol == "BTC-USDC"
        assert e.source == "data_module"

    def test_signals_ready(self, signal_output: SignalOutput) -> None:
        e = SignalsReady(source="signal_module", signals=[signal_output])
        assert len(e.signals) == 1
        assert e.signals[0].agent_name == "indicator_agent"

    def test_conviction_scored(self, conviction_output: ConvictionOutput) -> None:
        e = ConvictionScored(source="conviction_module", conviction=conviction_output)
        assert e.conviction.conviction_score == 0.78

    def test_trade_opened(
        self, trade_action: TradeAction, order_result: OrderResult
    ) -> None:
        e = TradeOpened(
            source="execution_module",
            trade_action=trade_action,
            order_result=order_result,
        )
        assert e.trade_action.action == "LONG"
        assert e.order_result.success is True

    def test_trade_closed(self) -> None:
        e = TradeClosed(
            source="execution_module",
            symbol="BTC-USDC",
            pnl=250.0,
            exit_reason="tp1_hit",
        )
        assert e.pnl == 250.0
        assert e.exit_reason == "tp1_hit"

    def test_position_updated(self, position: Position) -> None:
        e = PositionUpdated(
            source="sentinel_module",
            symbol="BTC-USDC",
            position=position,
        )
        assert e.position.direction == "long"

    def test_setup_detected(self) -> None:
        e = SetupDetected(
            source="sentinel_module",
            symbol="ETH-USDC",
            readiness=0.85,
            conditions=["rsi_oversold", "macd_bullish_cross"],
        )
        assert e.readiness == 0.85
        assert len(e.conditions) == 2

    def test_rule_generated(self) -> None:
        e = RuleGenerated(
            source="reflection_module",
            rule={"asset": "BTC", "rule": "Avoid longs when funding > 0.05%"},
        )
        assert "asset" in e.rule

    def test_factors_updated(self) -> None:
        e = FactorsUpdated(source="mcp_module", filepath="/data/factors_2026-04-05.json")
        assert e.filepath.endswith(".json")

    def test_macro_updated(self) -> None:
        e = MacroUpdated(source="mcp_module", filepath="/data/macro_2026-04-05.json")
        assert e.filepath.endswith(".json")

    def test_cycle_completed(self) -> None:
        e = CycleCompleted(
            source="pipeline",
            symbol="BTC-USDC",
            action="LONG",
            conviction=0.78,
        )
        assert e.action == "LONG"
        assert e.conviction == 0.78

    def test_event_inheritance(self) -> None:
        e = DataReady(source="test")
        assert isinstance(e, Event)
        e2 = SetupDetected(source="test", symbol="X", readiness=0.5)
        assert isinstance(e2, Event)

    def test_event_timestamp_auto_set(self) -> None:
        before = datetime.now(timezone.utc)
        e = Event(source="test")
        after = datetime.now(timezone.utc)
        assert before <= e.timestamp <= after
