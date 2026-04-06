"""Unit tests for SignalProducer, SignalRegistry, and ML model slots."""

from __future__ import annotations

import pytest

from engine.signals.base import SignalProducer
from engine.signals.registry import SignalRegistry
from engine.signals.ml import MLModelSlot
from engine.signals.ml.direction import DirectionModel
from engine.signals.ml.regime import RegimeModel
from engine.signals.ml.anomaly import AnomalyDetector
from engine.types import MarketData, SignalOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market_data() -> MarketData:
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=[],
        num_candles=0,
        lookback_description="~0",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={},
        swing_highs=[],
        swing_lows=[],
    )


def _make_signal(agent: str, direction: str = "BULLISH") -> SignalOutput:
    return SignalOutput(
        agent_name=agent,
        signal_type="llm",
        direction=direction,
        confidence=0.7,
        reasoning="test",
        signal_category="directional",
        data_richness="full",
        contradictions="",
        key_levels={},
        pattern_detected=None,
        raw_output="",
    )


class MockProducer(SignalProducer):
    """Configurable mock SignalProducer for testing."""

    def __init__(
        self,
        producer_name: str,
        enabled: bool = True,
        result: SignalOutput | None = None,
        raises: Exception | None = None,
        sig_type: str = "llm",
    ) -> None:
        self._name = producer_name
        self._enabled = enabled
        self._result = result
        self._raises = raises
        self._sig_type = sig_type

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return self._sig_type

    def is_enabled(self) -> bool:
        return self._enabled

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        if self._raises:
            raise self._raises
        return self._result


# ---------------------------------------------------------------------------
# SignalRegistry — register / unregister / get
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_register_and_get_enabled(self) -> None:
        reg = SignalRegistry()
        p = MockProducer("test_agent", enabled=True)
        reg.register(p)
        assert len(reg.get_enabled()) == 1
        assert reg.get_enabled()[0].name() == "test_agent"

    def test_disabled_excluded_from_get_enabled(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("enabled_one", enabled=True))
        reg.register(MockProducer("disabled_one", enabled=False))
        reg.register(MockProducer("enabled_two", enabled=True))
        enabled = reg.get_enabled()
        assert len(enabled) == 2
        names = {p.name() for p in enabled}
        assert names == {"enabled_one", "enabled_two"}

    def test_unregister(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("alpha"))
        reg.register(MockProducer("beta"))
        reg.unregister("alpha")
        assert len(reg.get_enabled()) == 1
        assert reg.get_enabled()[0].name() == "beta"

    def test_unregister_nonexistent_is_noop(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("alpha"))
        reg.unregister("nope")
        assert len(reg.get_enabled()) == 1

    def test_get_by_type(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("llm_agent", sig_type="llm"))
        reg.register(MockProducer("ml_model", sig_type="ml"))
        reg.register(MockProducer("llm_agent2", sig_type="llm"))
        assert len(reg.get_by_type("llm")) == 2
        assert len(reg.get_by_type("ml")) == 1


# ---------------------------------------------------------------------------
# SignalRegistry — run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    @pytest.mark.asyncio
    async def test_run_all_returns_results(self) -> None:
        reg = SignalRegistry()
        signal = _make_signal("agent_a")
        reg.register(MockProducer("agent_a", result=signal))
        results = await reg.run_all(_make_market_data())
        assert len(results) == 1
        assert results[0].agent_name == "agent_a"

    @pytest.mark.asyncio
    async def test_run_all_skips_disabled(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("enabled", enabled=True, result=_make_signal("enabled")))
        reg.register(MockProducer("disabled", enabled=False, result=_make_signal("disabled")))
        results = await reg.run_all(_make_market_data())
        assert len(results) == 1
        assert results[0].agent_name == "enabled"

    @pytest.mark.asyncio
    async def test_run_all_filters_none_results(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("returns_none", result=None))
        reg.register(MockProducer("returns_signal", result=_make_signal("ok")))
        results = await reg.run_all(_make_market_data())
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_all_handles_error_gracefully(self) -> None:
        reg = SignalRegistry()
        reg.register(MockProducer("good_before", result=_make_signal("good_before")))
        reg.register(MockProducer("bad", raises=RuntimeError("boom")))
        reg.register(MockProducer("good_after", result=_make_signal("good_after")))
        results = await reg.run_all(_make_market_data())
        assert len(results) == 2
        names = {r.agent_name for r in results}
        assert names == {"good_before", "good_after"}

    @pytest.mark.asyncio
    async def test_run_all_empty_registry(self) -> None:
        reg = SignalRegistry()
        results = await reg.run_all(_make_market_data())
        assert results == []


# ---------------------------------------------------------------------------
# ML Model Slots
# ---------------------------------------------------------------------------


class TestMLModelSlots:
    def test_direction_model_name(self) -> None:
        m = DirectionModel()
        assert m.name() == "direction_model"
        assert m.signal_type() == "ml"

    def test_regime_model_name(self) -> None:
        m = RegimeModel()
        assert m.name() == "regime_model"
        assert m.signal_type() == "ml"

    def test_anomaly_detector_name(self) -> None:
        m = AnomalyDetector()
        assert m.name() == "anomaly_detector"
        assert m.signal_type() == "ml"

    def test_disabled_when_no_model_loaded(self) -> None:
        m = DirectionModel()
        assert m.is_enabled() is False

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self) -> None:
        m = DirectionModel()
        result = await m.analyze(_make_market_data())
        assert result is None

    def test_disabled_when_no_flags_provided(self) -> None:
        m = RegimeModel()
        m._model = "fake_model"  # simulate loaded model
        # No flags instance => is_enabled returns False
        assert m.is_enabled() is False

    def test_requires_vision_false(self) -> None:
        m = DirectionModel()
        assert m.requires_vision() is False
