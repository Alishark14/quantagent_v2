"""Unit tests for FlowSignalAgent — code-only rules-based flow interpreter."""

from __future__ import annotations

import pytest

from engine.config import FeatureFlags
from engine.data.flow.signal_agent import (
    DIVERGENCE_CONFIDENCE,
    EXTREME_FUNDING_CONFIDENCE,
    FUNDING_EXTREME_CROWDED_LONG,
    FUNDING_EXTREME_CROWDED_SHORT,
    NEUTRAL_CONFIDENCE,
    OI_BUILD_THRESHOLD_PCT,
    OI_DROP_THRESHOLD_PCT,
    PRICE_LOOKBACK_CANDLES,
    FlowSignalAgent,
)
from engine.types import FlowOutput, MarketData, SignalOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(start_close: float, end_close: float, count: int = 30) -> list[dict]:
    """Build a synthetic candle stream that drifts cleanly over the lookback.

    FlowSignalAgent reads ``close`` at index ``-(PRICE_LOOKBACK_CANDLES + 1)``
    and ``close`` at ``-1``. We want a deterministic drift of
    ``end_close / start_close - 1`` over EXACTLY that window, not over the
    full ``count`` candles, otherwise tests can't pin specific percentages.

    The first ``count - PRICE_LOOKBACK_CANDLES - 1`` candles sit flat at
    ``start_close``; the final ``PRICE_LOOKBACK_CANDLES + 1`` candles
    linearly drift from ``start_close`` to ``end_close``. The result:
    the price-drift check sees exactly ``(end-start)/start * 100`` %.
    """
    if count < PRICE_LOOKBACK_CANDLES + 1:
        raise ValueError(
            f"need at least {PRICE_LOOKBACK_CANDLES + 1} candles for "
            f"FlowSignalAgent's price-drift check; got {count}"
        )
    pre = count - (PRICE_LOOKBACK_CANDLES + 1)
    drift_steps = PRICE_LOOKBACK_CANDLES  # gaps between PRICE_LOOKBACK+1 points
    step = (end_close - start_close) / drift_steps if drift_steps else 0.0

    rows = []
    ts = 1_700_000_000_000
    for i in range(count):
        if i < pre:
            c = start_close
        else:
            c = start_close + step * (i - pre)
        rows.append(
            {
                "timestamp": ts + i * 3600 * 1000,
                "open": c - 5,
                "high": c + 10,
                "low": c - 10,
                "close": c,
                "volume": 1000.0,
            }
        )
    return rows


def _make_market_data(
    *,
    candles: list[dict] | None = None,
    flow: FlowOutput | None = None,
) -> MarketData:
    return MarketData(
        symbol="BTC-USDC",
        timeframe="1h",
        candles=candles if candles is not None else _make_candles(60_000, 60_000),
        num_candles=len(candles) if candles is not None else 30,
        lookback_description="~30 hours",
        forecast_candles=3,
        forecast_description="~3 hours",
        indicators={},
        swing_highs=[],
        swing_lows=[],
        flow=flow,
    )


def _flow(
    *,
    funding_rate: float | None = 0.005,
    funding_signal: str = "NEUTRAL",
    oi_change_4h: float | None = 0.0,
    oi_trend: str = "STABLE",
    data_richness: str = "FULL",
    put_call_ratio: float | None = None,
    dvol: float | None = None,
    dvol_change_24h: float | None = None,
    skew_25d: float | None = None,
    gex_regime: str | None = None,
    cot_speculator_percentile: float | None = None,
    cot_commercial_net: float | None = None,
    cot_managed_money_net: float | None = None,
    cot_weekly_change_pct: float | None = None,
    cot_divergence: float | None = None,
    cot_divergence_abs_percentile: float | None = None,
    short_volume_ratio: float | None = None,
    svr_zscore: float | None = None,
    svr_trend: str | None = None,
    market_open: bool | None = None,
) -> FlowOutput:
    return FlowOutput(
        funding_rate=funding_rate,
        funding_signal=funding_signal,
        oi_change_4h=oi_change_4h,
        oi_trend=oi_trend,
        nearest_liquidation_above=None,
        nearest_liquidation_below=None,
        gex_regime=gex_regime,
        gex_flip_level=None,
        data_richness=data_richness,
        put_call_ratio=put_call_ratio,
        dvol=dvol,
        dvol_change_24h=dvol_change_24h,
        skew_25d=skew_25d,
        cot_speculator_percentile=cot_speculator_percentile,
        cot_commercial_net=cot_commercial_net,
        cot_managed_money_net=cot_managed_money_net,
        cot_weekly_change_pct=cot_weekly_change_pct,
        cot_divergence=cot_divergence,
        cot_divergence_abs_percentile=cot_divergence_abs_percentile,
        short_volume_ratio=short_volume_ratio,
        svr_zscore=svr_zscore,
        svr_trend=svr_trend,
        market_open=market_open,
    )


# ---------------------------------------------------------------------------
# SignalProducer interface conformance
# ---------------------------------------------------------------------------


class TestFlowSignalAgentInterface:
    def test_name(self) -> None:
        assert FlowSignalAgent().name() == "flow_signal_agent"

    def test_signal_type_is_flow(self) -> None:
        assert FlowSignalAgent().signal_type() == "flow"

    def test_requires_vision_false(self) -> None:
        assert FlowSignalAgent().requires_vision() is False

    def test_enabled_by_default_when_no_flags(self) -> None:
        assert FlowSignalAgent().is_enabled() is True

    def test_disabled_by_unset_flag(self, tmp_path) -> None:
        # Empty features.yaml → default False for unknown keys.
        empty = tmp_path / "features.yaml"
        empty.write_text("")
        agent = FlowSignalAgent(feature_flags=FeatureFlags(yaml_path=empty))
        assert agent.is_enabled() is False

    def test_enabled_by_flag(self, tmp_path) -> None:
        cfg = tmp_path / "features.yaml"
        cfg.write_text("flow_signal_agent: true\n")
        agent = FlowSignalAgent(feature_flags=FeatureFlags(yaml_path=cfg))
        assert agent.is_enabled() is True


# ---------------------------------------------------------------------------
# Rule 1: Bearish divergence (price up + funding bearish + OI dropping)
# ---------------------------------------------------------------------------


class TestBearishDivergence:
    @pytest.mark.asyncio
    async def test_price_up_funding_negative_oi_dropping_yields_bearish(self) -> None:
        # +5% drift over 30 candles, -3% OI in 4h, funding -0.005% (slightly negative)
        market = _make_market_data(
            candles=_make_candles(60_000, 63_000, count=30),
            flow=_flow(funding_rate=-0.005, oi_change_4h=-3.0),
        )
        agent = FlowSignalAgent()

        result = await agent.analyze(market)

        assert isinstance(result, SignalOutput)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(DIVERGENCE_CONFIDENCE)
        assert "BEARISH divergence" in result.reasoning
        assert "+5.00%" in result.reasoning
        assert "-3.0%" in result.reasoning

    @pytest.mark.asyncio
    async def test_does_not_fire_when_oi_only_slightly_negative(self) -> None:
        # OI drop is only -0.5%, well above the -2% threshold → no divergence.
        market = _make_market_data(
            candles=_make_candles(60_000, 63_000, count=30),
            flow=_flow(funding_rate=-0.005, oi_change_4h=-0.5),
        )

        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_does_not_fire_when_funding_positive(self) -> None:
        market = _make_market_data(
            candles=_make_candles(60_000, 63_000, count=30),
            flow=_flow(funding_rate=0.02, oi_change_4h=-3.0),
        )

        result = await FlowSignalAgent().analyze(market)
        # Falls through to extreme-funding rule? 0.02% is well below the 0.10% extreme.
        # → falls through to NEUTRAL.
        assert result.direction == "NEUTRAL"


# ---------------------------------------------------------------------------
# Rule 2: Bullish accumulation (price down + OI building + funding ≤ 0)
# ---------------------------------------------------------------------------


class TestBullishAccumulation:
    @pytest.mark.asyncio
    async def test_price_down_oi_building_funding_negative_yields_bullish(self) -> None:
        market = _make_market_data(
            candles=_make_candles(60_000, 57_000, count=30),
            flow=_flow(funding_rate=-0.005, oi_change_4h=4.0),
        )

        result = await FlowSignalAgent().analyze(market)

        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(DIVERGENCE_CONFIDENCE)
        assert "BULLISH accumulation" in result.reasoning
        assert "-5.00%" in result.reasoning
        assert "+4.0%" in result.reasoning

    @pytest.mark.asyncio
    async def test_funding_zero_still_qualifies(self) -> None:
        # funding == 0 satisfies "≤ FUNDING_NON_POSITIVE_MAX (0.0)"
        market = _make_market_data(
            candles=_make_candles(60_000, 57_000, count=30),
            flow=_flow(funding_rate=0.0, oi_change_4h=4.0),
        )

        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "BULLISH"

    @pytest.mark.asyncio
    async def test_does_not_fire_when_oi_flat(self) -> None:
        market = _make_market_data(
            candles=_make_candles(60_000, 57_000, count=30),
            flow=_flow(funding_rate=-0.005, oi_change_4h=1.0),
        )

        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"


# ---------------------------------------------------------------------------
# Rule 3: Extreme crowded long → contrarian BEARISH
# ---------------------------------------------------------------------------


class TestExtremeCrowdedLong:
    @pytest.mark.asyncio
    async def test_extreme_funding_yields_contrarian_bearish(self) -> None:
        market = _make_market_data(
            candles=_make_candles(60_000, 60_000),
            flow=_flow(funding_rate=0.15),  # > +0.10 threshold
        )

        result = await FlowSignalAgent().analyze(market)

        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(EXTREME_FUNDING_CONFIDENCE)
        assert "Extreme crowded long" in result.reasoning

    @pytest.mark.asyncio
    async def test_at_threshold_does_not_fire(self) -> None:
        # Exactly equal to threshold → not strictly greater → NEUTRAL.
        market = _make_market_data(
            flow=_flow(funding_rate=FUNDING_EXTREME_CROWDED_LONG)
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"


# ---------------------------------------------------------------------------
# Rule 4: Extreme crowded short → contrarian BULLISH
# ---------------------------------------------------------------------------


class TestExtremeCrowdedShort:
    @pytest.mark.asyncio
    async def test_extreme_negative_funding_yields_contrarian_bullish(self) -> None:
        market = _make_market_data(
            flow=_flow(funding_rate=-0.15),  # < -0.10 threshold
        )

        result = await FlowSignalAgent().analyze(market)

        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(EXTREME_FUNDING_CONFIDENCE)
        assert "Extreme crowded short" in result.reasoning

    @pytest.mark.asyncio
    async def test_at_threshold_does_not_fire(self) -> None:
        market = _make_market_data(
            flow=_flow(funding_rate=FUNDING_EXTREME_CROWDED_SHORT),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"


# ---------------------------------------------------------------------------
# Rule 5: NEUTRAL default + edge cases
# ---------------------------------------------------------------------------


class TestNeutralDefault:
    @pytest.mark.asyncio
    async def test_normal_conditions_yield_neutral(self) -> None:
        market = _make_market_data(
            flow=_flow(funding_rate=0.005, oi_change_4h=0.5),
        )

        result = await FlowSignalAgent().analyze(market)

        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(NEUTRAL_CONFIDENCE)
        assert "No flow signal" in result.reasoning

    @pytest.mark.asyncio
    async def test_no_flow_data_returns_neutral_with_explicit_reasoning(self) -> None:
        market = _make_market_data(flow=None)
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"
        assert "No flow data available" in result.reasoning
        assert result.data_richness == "minimal"

    @pytest.mark.asyncio
    async def test_funding_none_falls_to_neutral(self) -> None:
        market = _make_market_data(
            flow=_flow(funding_rate=None, oi_change_4h=-3.0),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"
        assert "funding rate unavailable" in result.reasoning

    @pytest.mark.asyncio
    async def test_oi_change_none_documents_provider_limitation(self) -> None:
        # Production CryptoFlowProvider currently leaves oi_change_4h=None.
        # The divergence rules can't fire — agent should NEUTRAL with the
        # provider-limitation reasoning visible in the data moat.
        market = _make_market_data(
            candles=_make_candles(60_000, 63_000, count=30),
            flow=_flow(funding_rate=-0.005, oi_change_4h=None),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"
        assert "OI delta unavailable" in result.reasoning

    @pytest.mark.asyncio
    async def test_too_few_candles_falls_to_neutral(self) -> None:
        # PRICE_LOOKBACK_CANDLES + 1 candles required; pass exactly that minus 1.
        # Build raw candles directly here — _make_candles enforces the
        # ≥ PRICE_LOOKBACK_CANDLES + 1 invariant on purpose.
        few = [
            {
                "timestamp": 1_700_000_000_000 + i * 3600 * 1000,
                "open": 60_000.0,
                "high": 60_010.0,
                "low": 59_990.0,
                "close": 60_000.0 + i * 5,
                "volume": 1000.0,
            }
            for i in range(PRICE_LOOKBACK_CANDLES)
        ]
        market = _make_market_data(
            candles=few,
            flow=_flow(funding_rate=-0.005, oi_change_4h=-3.0),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "NEUTRAL"
        assert "insufficient candles" in result.reasoning


# ---------------------------------------------------------------------------
# Rule precedence
# ---------------------------------------------------------------------------


class TestRulePrecedence:
    @pytest.mark.asyncio
    async def test_divergence_takes_priority_over_extreme_funding(self) -> None:
        # All conditions for BEARISH divergence AND for crowded-short extreme
        # are met (price up, funding -0.15%, OI dropping). Divergence rule
        # is checked first → BEARISH wins, NOT BULLISH (the extreme rule).
        # Note: we use a positive funding range that satisfies divergence.
        market = _make_market_data(
            candles=_make_candles(60_000, 63_000, count=30),
            flow=_flow(funding_rate=-0.005, oi_change_4h=-3.0),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "BEARISH"
        assert "BEARISH divergence" in result.reasoning

    @pytest.mark.asyncio
    async def test_accumulation_takes_priority_over_extreme_short(self) -> None:
        # Price down, OI building, funding -0.15% (also extreme short).
        # Accumulation rule is checked before extreme rules → BULLISH from
        # accumulation, with the divergence-confidence (not extreme).
        market = _make_market_data(
            candles=_make_candles(60_000, 57_000, count=30),
            flow=_flow(funding_rate=-0.15, oi_change_4h=4.0),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(DIVERGENCE_CONFIDENCE)
        assert "BULLISH accumulation" in result.reasoning


# ---------------------------------------------------------------------------
# SignalOutput shape contract
# ---------------------------------------------------------------------------


class TestSignalOutputShape:
    @pytest.mark.asyncio
    async def test_signal_output_carries_flow_signal_type(self) -> None:
        market = _make_market_data(flow=_flow(funding_rate=0.15))
        result = await FlowSignalAgent().analyze(market)
        assert result.signal_type == "flow"
        assert result.agent_name == "flow_signal_agent"
        assert result.signal_category == "directional"

    @pytest.mark.asyncio
    async def test_data_richness_normalised_to_lowercase(self) -> None:
        market = _make_market_data(
            flow=_flow(funding_rate=0.15, data_richness="FULL"),
        )
        result = await FlowSignalAgent().analyze(market)
        assert result.data_richness == "full"

    @pytest.mark.asyncio
    async def test_raw_output_mirrors_reasoning(self) -> None:
        # Code-only agent has no LLM raw response — preserves reasoning
        # in raw_output so the data moat capture is consistent.
        market = _make_market_data(flow=_flow(funding_rate=0.15))
        result = await FlowSignalAgent().analyze(market)
        assert result.raw_output == result.reasoning


# ---------------------------------------------------------------------------
# Crash safety
# ---------------------------------------------------------------------------


class TestCrashSafety:
    @pytest.mark.asyncio
    async def test_returns_none_on_truly_unrecoverable_input(self, monkeypatch) -> None:
        agent = FlowSignalAgent()

        # Force a crash inside _evaluate by replacing it with a raiser.
        def boom(self, data):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(FlowSignalAgent, "_evaluate", boom)
        result = await agent.analyze(_make_market_data())
        assert result is None

    @pytest.mark.asyncio
    async def test_threshold_constants_match_spec(self) -> None:
        # Pin the literal numbers so a future tweak that drifts them
        # against the spec fails this test loudly. The user spec said
        # extreme funding > 0.1%, OI delta thresholds tunable.
        assert FUNDING_EXTREME_CROWDED_LONG == 0.10
        assert FUNDING_EXTREME_CROWDED_SHORT == -0.10
        assert OI_DROP_THRESHOLD_PCT == -2.0
        assert OI_BUILD_THRESHOLD_PCT == 2.0
        assert PRICE_LOOKBACK_CANDLES == 12


# ---------------------------------------------------------------------------
# Options rules (BTC / ETH only — higher priority than funding / OI)
# ---------------------------------------------------------------------------


class TestOptionsRules:
    """Options rules live BEFORE the funding/OI block in the pipeline
    order and short-circuit whenever their respective fields are
    populated. Non-BTC/ETH symbols receive FlowOutput with these
    fields = None from OptionsEnrichment and fall through unchanged.
    """

    @pytest.mark.asyncio
    async def test_bearish_hedging_fires_on_high_pcr(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(put_call_ratio=1.5, funding_rate=0.005),
        )
        result = await agent.analyze(data)
        assert result is not None
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.60)
        assert "hedging" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_complacent_longs_fires_on_low_pcr(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(put_call_ratio=0.3, funding_rate=0.005),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.55)
        assert "complacent" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_extreme_skew_fires(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(skew_25d=15.0, funding_rate=0.005),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.55)
        assert "skew" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_volatility_spike_returns_neutral(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                dvol=80.0, dvol_change_24h=25.0, funding_rate=0.005
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.55)
        assert "volatility spike" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_positive_gamma_returns_neutral(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(gex_regime="POSITIVE_GAMMA", funding_rate=0.005),
        )
        result = await agent.analyze(data)
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.50)
        assert "positive gamma" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_negative_gamma_boosts_divergence(self) -> None:
        """Negative gamma regime adds +0.10 to the divergence rule's
        confidence (capped at 0.80)."""
        agent = FlowSignalAgent()
        candles = _make_candles(60_000, 62_000)  # +3.3% → "price up"
        data = _make_market_data(
            candles=candles,
            flow=_flow(
                funding_rate=-0.002,
                oi_change_4h=-3.0,
                gex_regime="NEGATIVE_GAMMA",
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.80)  # 0.70 + 0.10
        assert "divergence" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_negative_gamma_without_divergence_falls_through(self) -> None:
        """Negative gamma alone does NOT fire anything — it only boosts
        divergence/accumulation rules. A flat market with negative
        gamma reaches the default NEUTRAL."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                funding_rate=0.005,
                oi_change_4h=0.0,
                gex_regime="NEGATIVE_GAMMA",
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.30)

    @pytest.mark.asyncio
    async def test_options_rules_none_fields_fall_through_to_funding(self) -> None:
        """Non-BTC/ETH symbols receive None options fields and the
        extreme-funding rule must still fire normally."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(funding_rate=0.15),  # above 0.10% extreme
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        # 0.55 is the extreme funding confidence; also matches
        # OPTIONS_CONTRARIAN_CONFIDENCE but the reasoning distinguishes.
        assert "crowded long" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_options_rule_priority_high_pcr_wins_over_funding(self) -> None:
        """High PCR and extreme funding fire simultaneously. Options
        rule must win because it's earlier in the pipeline."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(put_call_ratio=1.5, funding_rate=0.15),
        )
        result = await agent.analyze(data)
        assert result.confidence == pytest.approx(0.60)  # options hedging, not 0.55
        assert "hedging" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_pcr_at_threshold_does_not_fire(self) -> None:
        """PCR must be strictly greater than 1.2 — exactly 1.2 falls through."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(put_call_ratio=1.2, funding_rate=0.005),
        )
        result = await agent.analyze(data)
        assert result.direction == "NEUTRAL"  # default fall-through


# ---------------------------------------------------------------------------
# COT rules (commodity only — gated by None-guards on the COT fields)
# ---------------------------------------------------------------------------


class TestCOTRules:
    """COT rules fire only when CommodityFlowProvider has populated the
    cot_* fields — for non-commodity symbols every field is None and
    the rules short-circuit to the funding/OI block unchanged.
    """

    @pytest.mark.asyncio
    async def test_extreme_speculator_long_bearish(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=95.0,
                cot_managed_money_net=300_000.0,
                cot_commercial_net=-250_000.0,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.55)
        assert "extreme speculator long" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_extreme_speculator_short_bullish(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=5.0,
                cot_managed_money_net=-50_000.0,
                cot_commercial_net=30_000.0,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.55)
        assert "extreme speculator short" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_commercial_divergence_bullish(self) -> None:
        """Commercials MORE long than speculators + extreme
        divergence + speculators low."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=20.0,  # below 30
                cot_managed_money_net=-50_000.0,
                cot_commercial_net=150_000.0,
                cot_divergence=200_000.0,  # commercial - mm > 0
                cot_divergence_abs_percentile=90.0,  # top 10%
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.60)
        assert "follow the hedgers" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_commercial_divergence_bearish(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=85.0,  # above 70
                cot_managed_money_net=300_000.0,
                cot_commercial_net=-200_000.0,
                cot_divergence=-500_000.0,  # commercial - mm < 0
                cot_divergence_abs_percentile=95.0,  # top 5%
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.60)
        assert "smart money exiting" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_cot_rules_do_not_fire_for_btc(self) -> None:
        """A BTC flow with every COT field None must fall through to
        the funding/OI rules unchanged. Extreme funding fires instead."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(funding_rate=0.15),  # > 0.10% extreme
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert "crowded long" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_middle_speculator_percentile_falls_through(self) -> None:
        """50th-percentile speculator positioning hits neither COT rule."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=50.0,
                cot_managed_money_net=0.0,
                cot_commercial_net=0.0,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        # No COT rule fires, falls through to default NEUTRAL
        assert result.direction == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_divergence_not_extreme_does_not_fire(self) -> None:
        """Commercial-divergence rules require top 20% abs percentile."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=20.0,
                cot_divergence=50_000.0,
                cot_divergence_abs_percentile=60.0,  # below 80
                cot_managed_money_net=0.0,
                cot_commercial_net=50_000.0,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_extreme_long_has_priority_over_divergence(self) -> None:
        """Rule order: extreme-long fires first even when a divergence
        setup is technically also present."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                cot_speculator_percentile=95.0,
                cot_divergence=-500_000.0,
                cot_divergence_abs_percentile=95.0,
                cot_managed_money_net=400_000.0,
                cot_commercial_net=-100_000.0,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        # confidence 0.55 matches extreme-long, not 0.60 (divergence)
        assert result.confidence == pytest.approx(0.55)
        assert "extreme speculator long" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# RegSHO equity rules (TSLA / NVDA / GOOGL only)
# ---------------------------------------------------------------------------


class TestRegSHORules:
    """RegSHO rules fire only when EquityFlowProvider has populated
    the svr_* / market_open fields. Non-equity symbols have every
    field None and fall through cleanly."""

    @pytest.mark.asyncio
    async def test_extreme_short_volume_bearish(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                svr_zscore=2.5,
                svr_trend="STABLE",
                market_open=True,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.55)
        assert "extreme short volume" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_short_squeeze_setup_bullish(self) -> None:
        """High Z-score + rising trend + price up → squeeze wins over
        plain extreme-short rule."""
        agent = FlowSignalAgent()
        candles = _make_candles(60_000, 62_000)  # +3.3% → price up
        data = _make_market_data(
            candles=candles,
            flow=_flow(
                svr_zscore=2.5,
                svr_trend="RISING",
                market_open=True,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.60)
        assert "squeeze" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_short_volume_collapse_bullish(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(
                svr_zscore=-2.0,
                market_open=True,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BULLISH"
        assert result.confidence == pytest.approx(0.50)
        assert "collapse" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_outside_market_hours_neutral(self) -> None:
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(market_open=False, funding_rate=0.005),
        )
        result = await agent.analyze(data)
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.40)
        assert "market hours" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_regsho_rules_do_not_fire_for_crypto(self) -> None:
        """BTC-style flow with every svr_* / market_open = None must
        fall through to the funding rule unchanged."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(funding_rate=0.15),  # extreme crowded long
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert "crowded long" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_market_open_true_does_not_fire_closed_rule(self) -> None:
        """market_open=True must NOT fire the outside-market-hours
        rule; rule 4 is gated on the False branch only."""
        agent = FlowSignalAgent()
        data = _make_market_data(
            flow=_flow(market_open=True, funding_rate=0.005),
        )
        result = await agent.analyze(data)
        # Falls through to default NEUTRAL (0.30), not the closed rule
        assert result.direction == "NEUTRAL"
        assert result.confidence == pytest.approx(0.30)

    @pytest.mark.asyncio
    async def test_squeeze_beats_plain_extreme_short_when_both_fire(self) -> None:
        """Priority: squeeze (rule 2) wins over extreme-short (rule 1)
        when the squeeze conditions are all satisfied."""
        agent = FlowSignalAgent()
        candles = _make_candles(60_000, 62_000)
        data = _make_market_data(
            candles=candles,
            flow=_flow(
                svr_zscore=3.0,
                svr_trend="RISING",
                market_open=True,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BULLISH"  # squeeze, not bearish
        assert result.confidence == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_squeeze_requires_price_up(self) -> None:
        """High Z-score + RISING trend but FLAT price → bearish
        extreme-short rule fires, not the squeeze."""
        agent = FlowSignalAgent()
        # Build flat candles so price_change_pct ~= 0
        candles = _make_candles(60_000, 60_000)
        data = _make_market_data(
            candles=candles,
            flow=_flow(
                svr_zscore=3.0,
                svr_trend="RISING",
                market_open=True,
                funding_rate=0.005,
            ),
        )
        result = await agent.analyze(data)
        assert result.direction == "BEARISH"
        assert result.confidence == pytest.approx(0.55)
