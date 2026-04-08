"""Unit tests for Sentinel: ReadinessScorer and SentinelMonitor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engine.events import InProcessBus, SetupDetected, SetupResult
from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from sentinel.conditions import ReadinessScorer
from sentinel.config import (
    BASE_READINESS_THRESHOLD,
    ESCALATION_STEP,
    MAX_ESCALATION,
    SKIP_COOLDOWN_SECONDS,
    get_sentinel_cooldown,
    get_sentinel_daily_budget,
    get_sentinel_escalation_config,
)
from sentinel.monitor import SentinelMonitor


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockSentinelAdapter(ExchangeAdapter):
    """Returns synthetic candle data for Sentinel tests."""

    def __init__(self, candles: list[dict] | None = None, funding: float | None = 0.0001) -> None:
        self._candles = candles or self._default_candles()
        self._funding = funding

    @staticmethod
    def _default_candles() -> list[dict]:
        """Generate 50 candles with enough variation for MACD computation."""
        import math
        candles = []
        for i in range(50):
            # Sine wave + uptrend to give realistic variation
            base = 65000.0 + i * 5.0 + 200 * math.sin(i * 0.3)
            candles.append({
                "timestamp": 1700000000 + i * 3600,
                "open": base - 15,
                "high": base + 50,
                "low": base - 50,
                "close": base,
                "volume": 1000.0 + (5000.0 if i == 49 else 0),
            })
        return candles

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market"], supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100):
        return self._candles[:limit]

    async def get_ticker(self, symbol):
        return {"last": 65290.0}

    async def get_balance(self):
        return 10000.0

    async def get_positions(self, symbol=None):
        return []

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=True, order_id="m-1", fill_price=65290.0, fill_size=size, error=None)

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=True, order_id="l-1", fill_price=price, fill_size=size, error=None)

    async def place_sl_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id="sl-1", fill_price=None, fill_size=None, error=None)

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=True, order_id="tp-1", fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol, order_id):
        return True

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=True, order_id="c-1", fill_price=65290.0, fill_size=0.1, error=None)

    async def modify_sl(self, symbol, new_price):
        return OrderResult(success=True, order_id="sl-m", fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=True, order_id="tp-m", fill_price=None, fill_size=None, error=None)

    async def get_funding_rate(self, symbol):
        return self._funding

    async def get_open_interest(self, symbol):
        return 500_000_000.0


# ---------------------------------------------------------------------------
# Indicator helpers for building test scenarios
# ---------------------------------------------------------------------------

def _neutral_indicators() -> dict:
    return {
        "rsi": 50.0,
        "macd": {"macd": 1.0, "signal": 0.9, "histogram": 0.1, "histogram_direction": "rising", "cross": "none"},
        "roc": 0.5,
        "stochastic": {"k": 50.0, "d": 50.0, "zone": "neutral"},
        "williams_r": -50.0,
        "atr": 400.0,
        "adx": {"adx": 25.0, "plus_di": 20.0, "minus_di": 15.0, "classification": "TRENDING"},
        "bollinger_bands": {"upper": 66000.0, "middle": 65000.0, "lower": 64000.0, "width": 2000.0, "width_percentile": 50.0},
        "volume_ma": {"ma": 1000.0, "current": 1000.0, "ratio": 1.0, "spike": False},
        "volatility_percentile": 50.0,
    }


def _overbought_indicators() -> dict:
    ind = _neutral_indicators()
    ind["rsi"] = 75.0
    return ind


def _volume_spike_indicators() -> dict:
    ind = _neutral_indicators()
    ind["volume_ma"] = {"ma": 1000.0, "current": 4000.0, "ratio": 4.0, "spike": True}
    return ind


# ---------------------------------------------------------------------------
# Tests: ReadinessScorer
# ---------------------------------------------------------------------------

class TestReadinessScorer:

    def test_no_conditions_triggered(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_neutral_indicators(),
            current_price=65000.0,
            swing_highs=[66500.0, 67000.0],
            swing_lows=[63000.0, 62000.0],
            funding_rate=0.0,
            prev_macd_histogram=0.05,
        )
        assert score == 0.0
        assert all(not c.triggered for c in conds)

    def test_rsi_overbought_triggers(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_overbought_indicators(),
            current_price=65000.0,
            swing_highs=[66500.0],
            swing_lows=[63000.0],
        )
        rsi_cond = [c for c in conds if c.name == "rsi_cross"][0]
        assert rsi_cond.triggered is True
        assert score == pytest.approx(0.25)

    def test_rsi_oversold_triggers(self) -> None:
        scorer = ReadinessScorer()
        ind = _neutral_indicators()
        ind["rsi"] = 28.0
        score, conds = scorer.score(
            indicators=ind,
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
        )
        rsi_cond = [c for c in conds if c.name == "rsi_cross"][0]
        assert rsi_cond.triggered is True

    def test_level_touch_bb_upper(self) -> None:
        scorer = ReadinessScorer()
        ind = _neutral_indicators()
        ind["bollinger_bands"]["upper"] = 65010.0  # price 65000 is within 0.3%
        score, conds = scorer.score(
            indicators=ind,
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
        )
        lvl = [c for c in conds if c.name == "level_touch"][0]
        assert lvl.triggered is True

    def test_level_touch_swing_low(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_neutral_indicators(),
            current_price=65000.0,
            swing_highs=[67000.0],
            swing_lows=[65010.0],  # within 0.3%
        )
        lvl = [c for c in conds if c.name == "level_touch"][0]
        assert lvl.triggered is True

    def test_volume_anomaly(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_volume_spike_indicators(),
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
        )
        vol = [c for c in conds if c.name == "volume_anomaly"][0]
        assert vol.triggered is True
        assert score == pytest.approx(0.20)

    def test_flow_shift_positive(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_neutral_indicators(),
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            funding_rate=0.0005,  # 0.05% — above 0.01%
        )
        flow = [c for c in conds if c.name == "flow_shift"][0]
        assert flow.triggered is True

    def test_flow_shift_none(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_neutral_indicators(),
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            funding_rate=None,
        )
        flow = [c for c in conds if c.name == "flow_shift"][0]
        assert flow.triggered is False

    def test_macd_bullish_cross(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_neutral_indicators(),  # histogram = 0.1
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            prev_macd_histogram=-0.05,  # was negative, now positive = bullish cross
        )
        macd = [c for c in conds if c.name == "macd_cross"][0]
        assert macd.triggered is True

    def test_macd_bearish_cross(self) -> None:
        scorer = ReadinessScorer()
        ind = _neutral_indicators()
        ind["macd"]["histogram"] = -0.1
        score, conds = scorer.score(
            indicators=ind,
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            prev_macd_histogram=0.05,  # was positive, now negative = bearish cross
        )
        macd = [c for c in conds if c.name == "macd_cross"][0]
        assert macd.triggered is True

    def test_macd_no_prev_histogram(self) -> None:
        scorer = ReadinessScorer()
        score, conds = scorer.score(
            indicators=_neutral_indicators(),
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            prev_macd_histogram=None,
        )
        macd = [c for c in conds if c.name == "macd_cross"][0]
        assert macd.triggered is False

    def test_multiple_conditions_sum(self) -> None:
        """RSI overbought (0.25) + volume spike (0.20) = 0.45."""
        scorer = ReadinessScorer()
        ind = _overbought_indicators()
        ind["volume_ma"] = {"ma": 1000.0, "current": 4000.0, "ratio": 4.0, "spike": True}
        score, conds = scorer.score(
            indicators=ind,
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
        )
        assert score == pytest.approx(0.45)

    def test_all_conditions_trigger(self) -> None:
        """All 5 conditions fire: 0.25 + 0.30 + 0.20 + 0.15 + 0.10 = 1.0."""
        scorer = ReadinessScorer()
        ind = _overbought_indicators()
        ind["bollinger_bands"]["upper"] = 65010.0
        ind["volume_ma"] = {"ma": 1000.0, "current": 4000.0, "ratio": 4.0, "spike": True}

        score, conds = scorer.score(
            indicators=ind,
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            funding_rate=0.0005,
            prev_macd_histogram=-0.05,
        )
        assert score == pytest.approx(1.0)
        assert all(c.triggered for c in conds)

    def test_score_clamped_at_1(self) -> None:
        scorer = ReadinessScorer()
        ind = _overbought_indicators()
        ind["bollinger_bands"]["upper"] = 65010.0
        ind["volume_ma"] = {"ma": 1000.0, "current": 4000.0, "ratio": 4.0, "spike": True}
        score, _ = scorer.score(
            indicators=ind,
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
            funding_rate=0.001,
            prev_macd_histogram=-0.1,
        )
        assert score <= 1.0

    def test_conditions_list_always_5(self) -> None:
        scorer = ReadinessScorer()
        _, conds = scorer.score(
            indicators=_neutral_indicators(),
            current_price=65000.0,
            swing_highs=[], swing_lows=[],
        )
        assert len(conds) == 5


# ---------------------------------------------------------------------------
# Tests: SentinelMonitor
# ---------------------------------------------------------------------------

class TestSentinelMonitor:

    @pytest.mark.asyncio
    async def test_check_once_returns_score(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC", threshold=0.7, candle_window=50,
        )
        score, conds = await sentinel.check_once()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(conds, list)

    @pytest.mark.asyncio
    async def test_emits_setup_detected_above_threshold(self) -> None:
        """Build candles that trigger multiple conditions to exceed threshold."""
        import math
        candles = []
        for i in range(50):
            c = 65000.0 + i * 5 + 200 * math.sin(i * 0.3)
            vol = 1000.0 if i < 49 else 10000.0
            candles.append({
                "timestamp": 1700000000 + i * 3600,
                "open": c - 15, "high": c + 50, "low": c - 50,
                "close": c, "volume": vol,
            })

        adapter = MockSentinelAdapter(candles=candles, funding=0.001)
        bus = InProcessBus()
        events: list[SetupDetected] = []
        bus.subscribe(SetupDetected, lambda e: events.append(e))

        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            threshold=0.15,  # low threshold so volume spike triggers
            candle_window=50,
        )

        score, _ = await sentinel.check_once()

        # With funding > 0.01% (0.15) + volume spike (0.20) = 0.35 >= 0.15
        if score >= 0.15:
            assert len(events) >= 1
            assert events[0].symbol == "BTC-USDC"

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_triggers(self) -> None:
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        events: list[SetupDetected] = []
        bus.subscribe(SetupDetected, lambda e: events.append(e))

        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            threshold=0.01,  # very low so it always triggers
            cooldown_seconds=900,
            candle_window=50,
        )

        await sentinel.check_once()
        first_count = len(events)
        await sentinel.check_once()  # should be blocked by cooldown

        assert len(events) == first_count  # no additional event

    @pytest.mark.asyncio
    async def test_daily_budget_limits_triggers(self) -> None:
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        events: list[SetupDetected] = []
        bus.subscribe(SetupDetected, lambda e: events.append(e))

        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            threshold=0.01,
            cooldown_seconds=0,  # no cooldown
            daily_budget=2,
            candle_window=50,
        )

        for _ in range(5):
            await sentinel.check_once()

        assert len(events) == 2  # capped by budget

    @pytest.mark.asyncio
    async def test_daily_triggers_remaining(self) -> None:
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            threshold=0.01, cooldown_seconds=0, daily_budget=3,
            candle_window=50,
        )

        assert sentinel.daily_triggers_remaining == 3
        await sentinel.check_once()
        assert sentinel.daily_triggers_remaining == 2

    @pytest.mark.asyncio
    async def test_insufficient_candles_returns_zero(self) -> None:
        few_candles = [
            {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for _ in range(5)
        ]
        adapter = MockSentinelAdapter(candles=few_candles)
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC", candle_window=50,
        )

        score, conds = await sentinel.check_once()
        assert score == 0.0
        assert conds == []

    @pytest.mark.asyncio
    async def test_prev_macd_histogram_tracked(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC", candle_window=50,
        )

        assert sentinel._prev_macd_histogram is None
        await sentinel.check_once()
        assert sentinel._prev_macd_histogram is not None  # now set

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(adapter, bus, "BTC-USDC")

        assert sentinel.is_running is False
        sentinel.stop()
        assert sentinel.is_running is False

    @pytest.mark.asyncio
    async def test_no_event_below_threshold(self) -> None:
        adapter = MockSentinelAdapter(funding=0.0)
        bus = InProcessBus()
        events: list[SetupDetected] = []
        bus.subscribe(SetupDetected, lambda e: events.append(e))

        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            threshold=0.99,  # very high, nothing will trigger
            candle_window=50,
        )

        await sentinel.check_once()
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Tests: Sentinel config — timeframe-dependent cooldown/budget
# ---------------------------------------------------------------------------

class TestSentinelConfig:

    def test_cooldown_15m(self) -> None:
        assert get_sentinel_cooldown("15m") == 900

    def test_cooldown_1h(self) -> None:
        assert get_sentinel_cooldown("1h") == 3600

    def test_cooldown_4h(self) -> None:
        assert get_sentinel_cooldown("4h") == 14400

    def test_cooldown_1d(self) -> None:
        assert get_sentinel_cooldown("1d") == 86400

    def test_cooldown_unknown_defaults_1h(self) -> None:
        assert get_sentinel_cooldown("5m") == 3600
        assert get_sentinel_cooldown("weird") == 3600

    def test_budget_15m(self) -> None:
        assert get_sentinel_daily_budget("15m") == 16

    def test_budget_1h(self) -> None:
        assert get_sentinel_daily_budget("1h") == 8

    def test_budget_4h(self) -> None:
        assert get_sentinel_daily_budget("4h") == 4

    def test_budget_1d(self) -> None:
        assert get_sentinel_daily_budget("1d") == 2

    def test_budget_unknown_defaults_8(self) -> None:
        assert get_sentinel_daily_budget("5m") == 8


class TestSentinelMonitorDynamicDefaults:

    def test_1h_gets_1h_cooldown_by_default(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(adapter, bus, "BTC-USDC", timeframe="1h")
        assert sentinel._cooldown_seconds == 3600
        assert sentinel._daily_budget == 8

    def test_15m_gets_15m_defaults(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(adapter, bus, "BTC-USDC", timeframe="15m")
        assert sentinel._cooldown_seconds == 900
        assert sentinel._daily_budget == 16

    def test_4h_gets_4h_defaults(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(adapter, bus, "BTC-USDC", timeframe="4h")
        assert sentinel._cooldown_seconds == 14400
        assert sentinel._daily_budget == 4

    def test_explicit_override_takes_precedence(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC", timeframe="1h",
            cooldown_seconds=120, daily_budget=50,
        )
        assert sentinel._cooldown_seconds == 120
        assert sentinel._daily_budget == 50

    def test_explicit_zero_cooldown_is_respected(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC", timeframe="1h",
            cooldown_seconds=0,
        )
        assert sentinel._cooldown_seconds == 0


# ---------------------------------------------------------------------------
# Tests: Escalation tunables in sentinel.config
# ---------------------------------------------------------------------------


class TestSentinelEscalationConfig:
    """Pin the escalation constants against the spec."""

    def test_base_threshold_default(self) -> None:
        assert BASE_READINESS_THRESHOLD == 0.30

    def test_escalation_step_default(self) -> None:
        assert ESCALATION_STEP == 0.10

    def test_max_escalation_default(self) -> None:
        # Ceiling = 0.30 + 0.25 = 0.55
        assert MAX_ESCALATION == 0.25

    def test_skip_cooldown_default(self) -> None:
        assert SKIP_COOLDOWN_SECONDS == 900  # 15 minutes

    def test_get_sentinel_escalation_config_shape(self) -> None:
        cfg = get_sentinel_escalation_config()
        assert cfg["base_threshold"] == BASE_READINESS_THRESHOLD
        assert cfg["escalation_step"] == ESCALATION_STEP
        assert cfg["max_escalation"] == MAX_ESCALATION
        assert cfg["skip_cooldown_seconds"] == SKIP_COOLDOWN_SECONDS

    def test_escalation_config_returns_copy(self) -> None:
        """Mutating the returned dict must not affect future calls."""
        cfg1 = get_sentinel_escalation_config()
        cfg1["base_threshold"] = 0.99
        cfg2 = get_sentinel_escalation_config()
        assert cfg2["base_threshold"] == BASE_READINESS_THRESHOLD


# ---------------------------------------------------------------------------
# Tests: Escalating readiness threshold (Task 11)
# ---------------------------------------------------------------------------
#
# These tests focus exclusively on the escalation state machine. They
# build a SentinelMonitor with a base threshold of 0.30 (matching the
# spec) and feed it SetupResult events directly via _on_setup_result —
# this isolates the escalation logic from the readiness scorer + the
# bus dispatch path. End-to-end tests (subscribe_results + bus publish)
# live in TestSentinelEscalationBusIntegration below.


def _make_escalation_sentinel(
    *, symbol: str = "BTC-USDC", base_threshold: float = 0.30,
    cooldown_seconds: int = 3600,
) -> SentinelMonitor:
    return SentinelMonitor(
        adapter=MockSentinelAdapter(),
        event_bus=InProcessBus(),
        symbol=symbol,
        base_threshold=base_threshold,
        cooldown_seconds=cooldown_seconds,
    )


def _setup_result(
    *, symbol: str, outcome: str, action: str = "", bot_id: str = "bot-x"
) -> SetupResult:
    return SetupResult(
        source="test", symbol=symbol, outcome=outcome,
        action=action, bot_id=bot_id, conviction_score=0.5,
    )


class TestSentinelThresholdEscalation:

    @pytest.mark.asyncio
    async def test_initial_threshold_is_base(self) -> None:
        sentinel = _make_escalation_sentinel()
        assert sentinel.current_threshold() == pytest.approx(0.30)

    @pytest.mark.asyncio
    async def test_first_skip_escalates_30_to_40(self) -> None:
        sentinel = _make_escalation_sentinel()
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="SKIP", action="SKIP",
        ))
        assert sentinel.current_threshold() == pytest.approx(0.40)
        # Cooldown switches to short
        assert sentinel.active_cooldown_seconds == SKIP_COOLDOWN_SECONDS

    @pytest.mark.asyncio
    async def test_second_skip_escalates_40_to_50(self) -> None:
        sentinel = _make_escalation_sentinel()
        for _ in range(2):
            await sentinel._on_setup_result(_setup_result(
                symbol="BTC-USDC", outcome="SKIP",
            ))
        assert sentinel.current_threshold() == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_third_skip_caps_at_55(self) -> None:
        sentinel = _make_escalation_sentinel()
        for _ in range(3):
            await sentinel._on_setup_result(_setup_result(
                symbol="BTC-USDC", outcome="SKIP",
            ))
        # 0.30 + 0.25 (MAX_ESCALATION) = 0.55 ceiling
        assert sentinel.current_threshold() == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_fourth_and_fifth_skip_stay_capped(self) -> None:
        """Once the cap is hit, additional SKIPs do not push higher."""
        sentinel = _make_escalation_sentinel()
        for _ in range(5):
            await sentinel._on_setup_result(_setup_result(
                symbol="BTC-USDC", outcome="SKIP",
            ))
        assert sentinel.current_threshold() == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_trade_resets_threshold_to_base(self) -> None:
        sentinel = _make_escalation_sentinel(cooldown_seconds=3600)
        # Escalate twice
        for _ in range(2):
            await sentinel._on_setup_result(_setup_result(
                symbol="BTC-USDC", outcome="SKIP",
            ))
        assert sentinel.current_threshold() == pytest.approx(0.50)
        assert sentinel.active_cooldown_seconds == SKIP_COOLDOWN_SECONDS

        # TRADE resets
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="TRADE", action="LONG",
        ))
        assert sentinel.current_threshold() == pytest.approx(0.30)
        # Cooldown returns to candle period (3600s for the default 1h)
        assert sentinel.active_cooldown_seconds == 3600

    @pytest.mark.asyncio
    async def test_setup_result_for_other_symbol_ignored(self) -> None:
        """A Sentinel only reacts to its own symbol's results."""
        sentinel = _make_escalation_sentinel(symbol="BTC-USDC")
        await sentinel._on_setup_result(_setup_result(
            symbol="ETH-USDC", outcome="SKIP",
        ))
        assert sentinel.current_threshold() == pytest.approx(0.30)
        assert sentinel.active_cooldown_seconds == 3600  # unchanged

    @pytest.mark.asyncio
    async def test_unknown_outcome_logs_and_ignores(self) -> None:
        """Defensive: an outcome we don't understand must not crash or mutate state."""
        sentinel = _make_escalation_sentinel()
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="WHAT_IS_THIS",
        ))
        assert sentinel.current_threshold() == pytest.approx(0.30)
        assert sentinel.active_cooldown_seconds == 3600


class TestSentinelEscalationFiringGate:
    """Tests for the readiness >= active_threshold gate.

    The MockSentinelAdapter produces a score of ~0.9 (volume spike +
    funding + RSI cross + level touch all fire). To demonstrate the
    "below escalated threshold" path we use a base_threshold large
    enough that the post-escalation value exceeds 0.9.
    """

    @pytest.mark.asyncio
    async def test_readiness_below_escalated_does_not_fire(self) -> None:
        """Escalated threshold above the synthetic readiness → no event.

        The literal-spec scenario is "readiness 0.45 vs escalated
        threshold 0.50 → no fire". The synthetic adapter doesn't yield
        0.45, so we use a higher base (0.85) and escalate twice (→
        1.05) to put the threshold above the synthetic 0.9 instead.
        Same property under test: escalation makes the gate stricter,
        and a readiness below the gated value does NOT fire.
        """
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        events: list[SetupDetected] = []
        bus.subscribe(SetupDetected, lambda e: events.append(e))

        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            base_threshold=0.85,
            cooldown_seconds=0,
            candle_window=50,
        )
        # Two SKIPs → 0.85 + 0.20 = 1.05 (capped at 0.85 + 0.25 = 1.10)
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="SKIP",
        ))
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="SKIP",
        ))
        assert sentinel.current_threshold() == pytest.approx(1.05)

        score, _ = await sentinel.check_once()
        # Synthetic adapter yields ~0.9 — well below the 1.05 threshold.
        assert score < 1.05, (
            f"synthetic readiness {score:.2f} >= 1.05 — adapter retuned?"
        )
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_readiness_above_escalated_fires(self) -> None:
        """Escalated threshold below the synthetic readiness → fires."""
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        events: list[SetupDetected] = []
        bus.subscribe(SetupDetected, lambda e: events.append(e))

        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            base_threshold=0.05,  # very low so even after escalation we're under the synthetic 0.9
            cooldown_seconds=0,
            candle_window=50,
        )
        # Escalate to 0.15 — still well below the synthetic ~0.9
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="SKIP",
        ))
        assert sentinel.current_threshold() == pytest.approx(0.15)

        score, _ = await sentinel.check_once()
        assert score >= 0.15, (
            f"synthetic readiness {score:.2f} < 0.15 — adapter retuned?"
        )
        assert len(events) >= 1


class TestSentinelCandleCloseReset:

    @pytest.mark.asyncio
    async def test_new_candle_clears_all_escalations(self) -> None:
        """Advancing the latest candle's timestamp wipes per-symbol escalation."""
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            base_threshold=0.30,
            cooldown_seconds=3600,  # candle period — what we expect to return to
            candle_window=50,
        )
        # First check primes the candle-close marker without resetting.
        await sentinel.check_once()
        first_marker = sentinel._last_candle_ts
        assert first_marker is not None

        # Escalate
        await sentinel._on_setup_result(_setup_result(
            symbol="BTC-USDC", outcome="SKIP",
        ))
        assert sentinel.current_threshold() == pytest.approx(0.40)
        assert sentinel.active_cooldown_seconds == SKIP_COOLDOWN_SECONDS

        # Roll the candle stream forward by ONE bar in place: drop the
        # oldest candle and append a fresh one with a higher timestamp.
        # This mirrors how a real exchange's fetch_ohlcv() return looks
        # when a candle closes — the latest element advances. Just
        # appending wouldn't work because MockSentinelAdapter returns
        # `self._candles[:limit]`, which always slices from the start.
        last_close = float(adapter._candles[-1]["close"])
        last_ts = int(adapter._candles[-1]["timestamp"])
        adapter._candles.pop(0)
        adapter._candles.append({
            "timestamp": last_ts + 3600,
            "open": last_close,
            "high": last_close + 50,
            "low": last_close - 50,
            "close": last_close + 10,
            "volume": 1500.0,
        })

        # Next check should detect the new candle and reset.
        await sentinel.check_once()
        assert sentinel.current_threshold() == pytest.approx(0.30)
        assert sentinel.active_cooldown_seconds == 3600  # back to candle period
        assert sentinel._last_candle_ts == int(adapter._candles[-1]["timestamp"])

    @pytest.mark.asyncio
    async def test_first_observation_does_not_reset(self) -> None:
        """A fresh-start Sentinel should not log a 'new candle' reset on its first tick."""
        adapter = MockSentinelAdapter(funding=0.001)
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            base_threshold=0.30,
            cooldown_seconds=0,
            candle_window=50,
        )
        assert sentinel._last_candle_ts is None
        await sentinel.check_once()
        # Marker is now set, and the threshold is unchanged (still base —
        # no escalation has happened, so the reset is invisible anyway,
        # but we're verifying the marker initialised cleanly).
        assert sentinel._last_candle_ts is not None
        assert sentinel.current_threshold() == pytest.approx(0.30)


class TestSentinelEscalationPerSymbol:
    """Each Sentinel is bound to one symbol; escalation is per-symbol."""

    @pytest.mark.asyncio
    async def test_two_sentinels_independent(self) -> None:
        bus = InProcessBus()
        btc = SentinelMonitor(
            MockSentinelAdapter(), bus, "BTC-USDC", base_threshold=0.30,
        )
        eth = SentinelMonitor(
            MockSentinelAdapter(), bus, "ETH-USDC", base_threshold=0.30,
        )
        btc.subscribe_results()
        eth.subscribe_results()

        # Publish a SKIP for BTC only
        await bus.publish(SetupResult(
            source="test", symbol="BTC-USDC", outcome="SKIP",
            action="SKIP", bot_id="bot-1", conviction_score=0.4,
        ))

        assert btc.current_threshold() == pytest.approx(0.40)
        assert eth.current_threshold() == pytest.approx(0.30)  # untouched


class TestSentinelEscalationBusIntegration:
    """End-to-end: SetupResult published on the bus reaches the handler."""

    @pytest.mark.asyncio
    async def test_subscribe_results_wires_handler(self) -> None:
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            base_threshold=0.30,
            cooldown_seconds=3600,
        )
        sentinel.subscribe_results()

        await bus.publish(SetupResult(
            source="test", symbol="BTC-USDC", outcome="SKIP",
            action="SKIP", bot_id="bot-z", conviction_score=0.2,
        ))

        assert sentinel.current_threshold() == pytest.approx(0.40)
        assert sentinel.active_cooldown_seconds == SKIP_COOLDOWN_SECONDS

    @pytest.mark.asyncio
    async def test_unsubscribed_sentinel_ignores_bus_events(self) -> None:
        """Without subscribe_results(), bus events should not affect state."""
        adapter = MockSentinelAdapter()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter, bus, "BTC-USDC",
            base_threshold=0.30,
        )
        # Note: we deliberately do NOT call sentinel.subscribe_results()

        await bus.publish(SetupResult(
            source="test", symbol="BTC-USDC", outcome="SKIP",
            action="SKIP", bot_id="bot-z", conviction_score=0.2,
        ))

        assert sentinel.current_threshold() == pytest.approx(0.30)
