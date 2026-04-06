"""Unit tests for ExecutionCostModel, HyperliquidCostModel, GenericCostModel."""

from __future__ import annotations

import pytest

from engine.execution.cost_model import ExecutionCost, ExecutionCostModel, PositionSizeResult
from engine.execution.cost_models.generic import GenericCostModel
from engine.execution.cost_models.hyperliquid import (
    HyperliquidCostModel,
    PERP_MAKER_RATES,
    PERP_TAKER_RATES,
)


# ---------------------------------------------------------------------------
# HyperliquidCostModel: taker rate calculation
# ---------------------------------------------------------------------------

class TestHyperliquidTakerRate:

    def test_standard_crypto_tier0(self) -> None:
        model = HyperliquidCostModel()
        # No HIP-3 meta -> standard rate
        assert model.get_taker_rate("BTC-USDC") == pytest.approx(0.00045)

    def test_tier0_rate(self) -> None:
        model = HyperliquidCostModel()
        model._fee_tier = 0
        assert model.get_taker_rate("BTC-USDC") == pytest.approx(0.00045)

    def test_tier3_rate(self) -> None:
        model = HyperliquidCostModel()
        model._fee_tier = 3
        assert model.get_taker_rate("BTC-USDC") == pytest.approx(0.00030)

    def test_hip3_deployer_scale_below_1(self) -> None:
        """deployer_scale < 1: multiplier = scale + 1."""
        model = HyperliquidCostModel()
        model.set_meta("GOLD-USDC", deployer_scale=0.5, is_hip3=True)
        # base 0.00045 * (0.5 + 1) = 0.00045 * 1.5 = 0.000675
        assert model.get_taker_rate("GOLD-USDC") == pytest.approx(0.000675)

    def test_hip3_deployer_scale_above_1(self) -> None:
        """deployer_scale >= 1: multiplier = scale * 2."""
        model = HyperliquidCostModel()
        model.set_meta("EXOTIC-USDC", deployer_scale=2.0, is_hip3=True)
        # base 0.00045 * 2.0 * 2 = 0.00045 * 4 = 0.0018
        assert model.get_taker_rate("EXOTIC-USDC") == pytest.approx(0.0018)

    def test_hip3_deployer_scale_exactly_1(self) -> None:
        model = HyperliquidCostModel()
        model.set_meta("TEST-USDC", deployer_scale=1.0, is_hip3=True)
        # scale >= 1: mult = 1.0 * 2 = 2
        assert model.get_taker_rate("TEST-USDC") == pytest.approx(0.00045 * 2)

    def test_growth_mode_90_percent_reduction(self) -> None:
        model = HyperliquidCostModel()
        model.set_meta("NEW-USDC", growth_mode=True)
        # 0.00045 * 0.1 = 0.000045
        assert model.get_taker_rate("NEW-USDC") == pytest.approx(0.000045)

    def test_staking_discount(self) -> None:
        model = HyperliquidCostModel()
        model._staking_discount = 0.1  # 10% discount
        # 0.00045 * 0.9 = 0.000405
        assert model.get_taker_rate("BTC-USDC") == pytest.approx(0.000405)

    def test_referral_discount(self) -> None:
        model = HyperliquidCostModel()
        model._referral_discount = 0.05  # 5% discount
        assert model.get_taker_rate("BTC-USDC") == pytest.approx(0.00045 * 0.95)

    def test_combined_hip3_growth_staking(self) -> None:
        model = HyperliquidCostModel()
        model.set_meta("COMBO-USDC", deployer_scale=0.5, growth_mode=True, is_hip3=True)
        model._staking_discount = 0.1
        # base=0.00045 * hip3(1.5) * growth(0.1) * staking(0.9)
        expected = 0.00045 * 1.5 * 0.1 * 0.9
        assert model.get_taker_rate("COMBO-USDC") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# HyperliquidCostModel: maker rate
# ---------------------------------------------------------------------------

class TestHyperliquidMakerRate:

    def test_tier0_maker(self) -> None:
        model = HyperliquidCostModel()
        assert model.get_maker_rate("BTC-USDC") == pytest.approx(0.00015)

    def test_tier4_maker_free(self) -> None:
        model = HyperliquidCostModel()
        model._fee_tier = 4
        assert model.get_maker_rate("BTC-USDC") == 0.0


# ---------------------------------------------------------------------------
# Slippage estimation
# ---------------------------------------------------------------------------

class TestSlippageEstimation:

    def test_no_orderbook_crypto_default(self) -> None:
        model = HyperliquidCostModel()
        assert model.estimate_slippage("BTC-USDC", 1000, "buy") == 0.001

    def test_no_orderbook_hip3_default(self) -> None:
        model = HyperliquidCostModel()
        model.set_meta("GOLD-USDC", is_hip3=True)
        assert model.estimate_slippage("GOLD-USDC", 1000, "buy") == 0.002

    def test_from_orderbook(self) -> None:
        model = HyperliquidCostModel()
        # Simple book: asks at 65001, 65002
        model.set_orderbook("BTC-USDC",
            bids=[[65000, 10000], [64999, 10000]],
            asks=[[65001, 5000], [65002, 5000]],
        )
        slippage = model.estimate_slippage("BTC-USDC", 3000, "buy")
        # Mid = 65000.5, fills at 65001 for 3000 -> slip ~0.0008%
        assert slippage >= 0
        assert slippage < 0.01


# ---------------------------------------------------------------------------
# Spread estimation
# ---------------------------------------------------------------------------

class TestSpreadEstimation:

    def test_no_orderbook_default(self) -> None:
        model = HyperliquidCostModel()
        assert model.estimate_spread_cost("BTC-USDC") == 0.001

    def test_from_orderbook(self) -> None:
        model = HyperliquidCostModel()
        model.set_orderbook("BTC-USDC",
            bids=[[65000, 10000]],
            asks=[[65010, 10000]],
        )
        spread = model.estimate_spread_cost("BTC-USDC")
        assert spread == pytest.approx(10 / 65000, rel=0.01)


# ---------------------------------------------------------------------------
# Funding cost
# ---------------------------------------------------------------------------

class TestFundingCost:

    def test_long_positive_funding(self) -> None:
        model = HyperliquidCostModel()
        model.set_funding("BTC-USDC", 0.0001)  # 0.01% per 8h
        # 8h hold = 1 period -> 0.01%
        cost = model.estimate_funding_cost("BTC-USDC", "LONG", 8.0)
        assert cost == pytest.approx(0.0001)

    def test_long_24h_hold(self) -> None:
        model = HyperliquidCostModel()
        model.set_funding("BTC-USDC", 0.0001)
        # 24h = 3 periods -> 0.03%
        cost = model.estimate_funding_cost("BTC-USDC", "LONG", 24.0)
        assert cost == pytest.approx(0.0003)

    def test_short_negative_funding_earns(self) -> None:
        model = HyperliquidCostModel()
        model.set_funding("BTC-USDC", 0.0001)
        # Short pays -rate (earns when funding is positive)
        cost = model.estimate_funding_cost("BTC-USDC", "SHORT", 8.0)
        assert cost == pytest.approx(-0.0001)

    def test_no_funding_data(self) -> None:
        model = HyperliquidCostModel()
        cost = model.estimate_funding_cost("BTC-USDC", "LONG", 8.0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# compute_total_cost
# ---------------------------------------------------------------------------

class TestComputeTotalCost:

    def test_all_components_summed(self) -> None:
        model = HyperliquidCostModel()
        model.set_funding("BTC-USDC", 0.0001)
        cost = model.compute_total_cost("BTC-USDC", 10000, "LONG", 8.0)

        assert isinstance(cost, ExecutionCost)
        assert cost.entry_fee > 0
        assert cost.exit_fee > 0
        assert cost.total_cost > 0
        assert cost.total_cost == pytest.approx(
            cost.entry_fee + cost.exit_fee
            + cost.entry_slippage + cost.exit_slippage
            + cost.spread_cost + cost.estimated_funding,
            rel=0.01,
        )

    def test_total_as_pct(self) -> None:
        model = HyperliquidCostModel()
        cost = model.compute_total_cost("BTC-USDC", 10000, "LONG", 8.0)
        assert cost.total_as_pct == pytest.approx(cost.total_cost / 10000)

    def test_zero_position_value(self) -> None:
        model = HyperliquidCostModel()
        cost = model.compute_total_cost("BTC-USDC", 0, "LONG", 8.0)
        assert cost.total_cost == 0
        assert cost.total_as_pct == 0


# ---------------------------------------------------------------------------
# fee_adjusted_rr
# ---------------------------------------------------------------------------

class TestFeeAdjustedRR:

    def test_lower_than_raw_rr(self) -> None:
        model = HyperliquidCostModel()
        # Raw RR: 1.5% TP / 1% SL = 1.5
        adjusted = model.compute_fee_adjusted_rr(
            "BTC-USDC", 10000, 0.01, 0.015, "LONG", 8.0,
        )
        assert adjusted < 1.5
        assert adjusted > 0

    def test_costs_reduce_rr_significantly_for_small_positions(self) -> None:
        model = HyperliquidCostModel()
        model.set_meta("GOLD-USDC", deployer_scale=0.5, is_hip3=True)
        # Small position on HIP-3: costs eat a lot of the edge
        adjusted = model.compute_fee_adjusted_rr(
            "GOLD-USDC", 200, 0.01, 0.015, "LONG", 8.0,
        )
        # Should be significantly less than 1.5
        assert adjusted < 1.0

    def test_large_btc_position_with_orderbook(self) -> None:
        """With orderbook data (low slippage), BTC RR stays close to raw."""
        model = HyperliquidCostModel()
        model.set_orderbook("BTC-USDC",
            bids=[[65000, 500000], [64999, 500000]],
            asks=[[65001, 500000], [65002, 500000]],
        )
        adjusted = model.compute_fee_adjusted_rr(
            "BTC-USDC", 10000, 0.015, 0.030, "LONG", 8.0,
        )
        # With real orderbook, slippage is tiny -> RR close to raw
        assert adjusted > 1.5


# ---------------------------------------------------------------------------
# cost_aware_position_size (iterative solver)
# ---------------------------------------------------------------------------

class TestCostAwarePositionSize:

    def test_converges(self) -> None:
        """With orderbook data (realistic), the solver converges to a viable size."""
        model = HyperliquidCostModel()
        model.set_orderbook("BTC-USDC",
            bids=[[65000, 500000], [64999, 500000]],
            asks=[[65001, 500000], [65002, 500000]],
        )
        result = model.compute_cost_aware_position_size(
            "BTC-USDC", 10000, 0.01, 0.015, "LONG", 8.0, 1.0,
        )
        assert isinstance(result, PositionSizeResult)
        assert result.size > 0
        assert result.viable is True

    def test_smaller_than_naive(self) -> None:
        """Cost-aware size should be smaller than naive (fees eat risk budget)."""
        model = HyperliquidCostModel()
        result = model.compute_cost_aware_position_size(
            "BTC-USDC", 10000, 0.01, 0.015, "LONG", 8.0, 1.0,
        )
        naive_size = (10000 * 0.01) / 0.015  # 6666.67
        assert result.size < naive_size

    def test_costs_exceed_risk_returns_zero(self) -> None:
        """Extreme HIP-3 fees + tiny balance = costs eat entire risk budget."""
        model = HyperliquidCostModel()
        model.set_meta("EXOTIC-USDC", deployer_scale=3.0, is_hip3=True)
        result = model.compute_cost_aware_position_size(
            "EXOTIC-USDC", 100, 0.005, 0.002, "LONG", 72.0, 1.0,
        )
        # Very small risk budget with extreme fees
        assert result.size == 0 or not result.viable

    def test_fee_drag_pct_computed(self) -> None:
        model = HyperliquidCostModel()
        result = model.compute_cost_aware_position_size(
            "BTC-USDC", 10000, 0.01, 0.015, "LONG", 8.0, 1.0,
        )
        assert result.fee_drag_pct > 0

    def test_invalid_sl_distance(self) -> None:
        model = HyperliquidCostModel()
        result = model.compute_cost_aware_position_size(
            "BTC-USDC", 10000, 0.01, 0.0, "LONG", 8.0, 1.0,
        )
        assert result.size == 0
        assert result.viable is False


# ---------------------------------------------------------------------------
# is_trade_viable
# ---------------------------------------------------------------------------

class TestIsTradeViable:

    def test_btc_large_position_viable(self) -> None:
        model = HyperliquidCostModel()
        viable, reason = model.is_trade_viable(
            "BTC-USDC", 5000, 0.015, 0.030, "LONG", 8.0, min_rr=1.0,
        )
        assert viable is True
        assert "viable" in reason.lower()

    def test_hip3_small_position_not_viable(self) -> None:
        model = HyperliquidCostModel()
        model.set_meta("GOLD-USDC", deployer_scale=0.5, is_hip3=True)
        viable, reason = model.is_trade_viable(
            "GOLD-USDC", 100, 0.01, 0.012, "LONG", 8.0, min_rr=1.0,
        )
        # Small position + HIP-3 fees -> net RR too low
        assert viable is False
        assert "Net R:R" in reason

    def test_high_min_rr_blocks(self) -> None:
        model = HyperliquidCostModel()
        viable, _ = model.is_trade_viable(
            "BTC-USDC", 1000, 0.01, 0.015, "LONG", 8.0, min_rr=5.0,
        )
        assert viable is False


# ---------------------------------------------------------------------------
# GenericCostModel
# ---------------------------------------------------------------------------

class TestGenericCostModel:

    def test_conservative_defaults(self) -> None:
        model = GenericCostModel()
        assert model.get_taker_rate("ANY") == 0.001
        assert model.get_maker_rate("ANY") == 0.0005
        assert model.estimate_slippage("ANY", 1000, "buy") == 0.0005
        assert model.estimate_spread_cost("ANY") == 0.001
        assert model.estimate_funding_cost("ANY", "LONG", 8.0) == 0.0

    def test_compute_total_cost(self) -> None:
        model = GenericCostModel()
        cost = model.compute_total_cost("ANY", 10000, "LONG", 8.0)
        assert cost.total_cost > 0
        # taker entry + taker exit + slippage*2 + spread
        expected = 10000 * (0.001 + 0.0005 + 0.001/2) * 2
        assert cost.total_cost == pytest.approx(expected, rel=0.01)

    def test_custom_rates(self) -> None:
        model = GenericCostModel(taker_rate=0.002, slippage=0.001)
        assert model.get_taker_rate("X") == 0.002
        assert model.estimate_slippage("X", 100, "buy") == 0.001


# ---------------------------------------------------------------------------
# Real-world scenarios
# ---------------------------------------------------------------------------

class TestRealWorldScenarios:

    def test_btc_1000_position_1h_with_orderbook_low_drag(self) -> None:
        """BTC $1000 position, 1h hold, with orderbook: should be LOW fee drag."""
        model = HyperliquidCostModel()
        model.set_orderbook("BTC-USDC",
            bids=[[65000, 500000], [64999, 500000]],
            asks=[[65001, 500000], [65002, 500000]],
        )
        result = model.compute_cost_aware_position_size(
            "BTC-USDC", 10000, 0.01, 0.015, "LONG", 8.0, 1.0,
        )
        assert result.viable is True
        assert result.fee_drag_pct < 10  # LOW

    def test_gold_200_position_1h_high_drag(self) -> None:
        """GOLD $200 position with HIP-3 deployer fees: HIGH fee drag."""
        model = HyperliquidCostModel()
        model.set_meta("GOLD-USDC", deployer_scale=0.5, is_hip3=True)
        cost = model.compute_total_cost("GOLD-USDC", 200, "LONG", 8.0)
        # Fee drag relative to a 1.5% SL
        risk = 200 * 0.015
        drag = cost.total_cost / risk * 100
        assert drag > 15  # HIGH

    def test_btc_50_position_15m_medium_drag(self) -> None:
        """BTC $50 micro position, 15m scalp: MEDIUM drag."""
        model = HyperliquidCostModel()
        cost = model.compute_total_cost("BTC-USDC", 50, "LONG", 2.0)
        risk = 50 * 0.015
        drag = cost.total_cost / risk * 100
        assert drag > 5  # at least MEDIUM for such a small position
