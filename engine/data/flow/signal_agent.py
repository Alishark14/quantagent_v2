"""FlowSignalAgent: rules-based SignalProducer that interprets order flow.

The FlowAgent (in this same package) is a *data fetcher* — it pulls funding,
OI, and liquidation snapshots from exchange adapters and aggregates them
into a :class:`FlowOutput`. That output used to be dumped raw into the
grounding header so the three LLM signal agents (Indicator, Pattern, Trend)
each had to interpret it themselves.

This module elevates flow into its own *signal voice*. ``FlowSignalAgent``
implements :class:`SignalProducer` and runs alongside the LLM agents in
the ``SignalRegistry``. It is **code-only** — zero LLM calls, deterministic,
~microsecond latency. The ConvictionAgent then receives 4 ``SignalOutput``s
(indicator + pattern + trend + flow) instead of 3, and the LLM agents stop
having to look at OI/GEX/liquidation in their grounding context.

Rules
=====

The interpretation logic is driven by ordered rules. The first matching
rule wins. All thresholds are module-level constants so they can be tuned
without touching the agent's flow control.

1. **Bearish divergence** (price up + funding bearish + OI dropping) →
   smart money exiting while retail is still buying. Drop conviction on
   any LONG setup.

2. **Bullish accumulation** (price down + funding ≤ 0 + OI building) →
   smart money accumulating during a pullback. Often the highest-quality
   contrarian LONG signal.

3. **Extreme funding crowded long** (funding > +0.10%) → contrarian
   BEARISH lean, regardless of price action. The market is paying through
   the nose to be long; squeezes happen at extremes.

4. **Extreme funding crowded short** (funding < -0.10%) → contrarian
   BULLISH lean.

5. **Default NEUTRAL** — no clear signal. Confidence kept low so this
   doesn't drown out the other agents during normal conditions.

OI history limitation
=====================

Production today: ``CryptoFlowProvider`` populates ``funding_rate`` and
``open_interest`` but does NOT populate ``oi_change_4h`` because it has
no per-symbol OI history buffer. As a result, the BEARISH-divergence and
BULLISH-accumulation rules will fall through to NEUTRAL in the live
engine until the OI-history backfill lands. The unit tests use synthetic
``FlowOutput``s with ``oi_change_4h`` populated to exercise both paths.

Once a future task plumbs OI history through ``CryptoFlowProvider`` (or
adds an alternative provider that does), the divergence rules start
firing automatically — no change required to this agent.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

from engine.config import FeatureFlags
from engine.signals.base import SignalProducer
from engine.types import FlowOutput, MarketData, SignalOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable thresholds (module-level so they're easy to tweak / pin in tests)
# ---------------------------------------------------------------------------

#: How many recent candles to look back when classifying short-term price drift.
PRICE_LOOKBACK_CANDLES = 12

#: % close-to-close change over ``PRICE_LOOKBACK_CANDLES`` to call price "up".
PRICE_UP_THRESHOLD_PCT = 1.0

#: % close-to-close change to call price "down". Negative.
PRICE_DOWN_THRESHOLD_PCT = -1.0

#: Funding rate (in % per 8h period) above which we treat as a *crowded long*
#: contrarian bearish signal. Matches the 0.10% extreme threshold from the
#: spec — orders of magnitude above ``CryptoFlowProvider``'s normal
#: ``CROWDED_LONG`` classification at 0.01% so the two rules don't collide.
FUNDING_EXTREME_CROWDED_LONG = 0.10

#: Funding rate below which we treat as a *crowded short* contrarian bullish
#: signal. Symmetric with ``FUNDING_EXTREME_CROWDED_LONG``.
FUNDING_EXTREME_CROWDED_SHORT = -0.10

#: Funding rate above which the divergence rule treats funding as "non-bearish".
#: Used in the BEARISH-divergence rule: funding must be at or below this for
#: the rule to fire. 0.0 means "any negative or zero".
FUNDING_BEARISH_MAX = 0.0

#: Funding rate above which the accumulation rule treats funding as
#: "non-bullish enough". Used in the BULLISH-accumulation rule: funding
#: must be at or below this for the rule to fire.
FUNDING_NON_POSITIVE_MAX = 0.0

#: % OI change over the lookback window required to call OI "dropping".
#: Matches the BEARISH-divergence "smart money exiting" reading.
OI_DROP_THRESHOLD_PCT = -2.0

#: % OI change required to call OI "building / increasing".
OI_BUILD_THRESHOLD_PCT = 2.0

#: Confidence assigned to BEARISH/BULLISH divergence signals (high — these
#: are the highest-quality flow reads in the rule set).
DIVERGENCE_CONFIDENCE = 0.70

#: Confidence assigned to extreme-funding contrarian signals. Lower than
#: divergence — funding extremes can persist for hours before snapping.
EXTREME_FUNDING_CONFIDENCE = 0.55

#: Confidence assigned to NEUTRAL output. Low so the FlowAgent doesn't
#: drown out the other agents during normal market conditions.
NEUTRAL_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------


class FlowSignalAgent(SignalProducer):
    """Code-only SignalProducer that interprets order flow as a directional read.

    Implements the :class:`SignalProducer` ABC so it slots into the
    ``SignalRegistry`` next to the three LLM agents. ``signal_type``
    returns the new ``"flow"`` value (alongside ``"llm"`` / ``"ml"``)
    so consumers that key on type can distinguish flow signals.
    """

    def __init__(self, feature_flags: FeatureFlags | None = None) -> None:
        self._flags = feature_flags

    # -- SignalProducer interface ----------------------------------------

    def name(self) -> str:
        return "flow_signal_agent"

    def signal_type(self) -> str:
        # New third type alongside "llm" / "ml". The signal_type field is
        # an open string in the ABC, but the existing convention only had
        # those two values — see the docstring updates on
        # ``engine/signals/base.py`` and ``engine/types.py``.
        return "flow"

    def is_enabled(self) -> bool:
        if self._flags is None:
            return True
        return self._flags.is_enabled("flow_signal_agent")

    def requires_vision(self) -> bool:
        return False

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        """Apply the rule set and return a directional ``SignalOutput``.

        Returns ``None`` only on truly unrecoverable input shape errors —
        the SignalRegistry treats that as "agent skipped" and continues
        with the other producers. For all normal-but-quiet cases (no
        flow data, missing OI history, etc.) the agent returns a
        low-confidence NEUTRAL so its voice still appears in the
        conviction prompt with an explicit "no signal" reasoning.
        """
        try:
            return self._evaluate(data)
        except Exception:
            logger.exception("FlowSignalAgent: analyze() crashed; returning None")
            return None

    # -- Rule pipeline ---------------------------------------------------

    def _evaluate(self, data: MarketData) -> SignalOutput:
        flow = data.flow
        candles = data.candles

        # No flow data at all → NEUTRAL with explicit reasoning. We never
        # return None for "data missing" — that would silently drop the
        # FlowAgent's voice in ConvictionAgent's prompt and hide the
        # situation from the data moat.
        if flow is None:
            return self._neutral(
                reasoning="No flow data available — exchange adapter did not "
                "return funding rate or OI. FlowSignalAgent abstains.",
                richness="minimal",
            )

        funding = flow.funding_rate
        oi_change = flow.oi_change_4h
        price_change_pct = self._recent_price_change_pct(candles)

        # Rule 1: Bearish divergence — price up + funding bearish + OI dropping.
        if (
            funding is not None
            and oi_change is not None
            and price_change_pct is not None
            and price_change_pct > PRICE_UP_THRESHOLD_PCT
            and funding <= FUNDING_BEARISH_MAX
            and oi_change <= OI_DROP_THRESHOLD_PCT
        ):
            reasoning = (
                f"BEARISH divergence: price +{price_change_pct:.2f}% over the "
                f"last {PRICE_LOOKBACK_CANDLES} candles while funding flipped to "
                f"{funding:+.4f}% and OI dropped {oi_change:+.1f}% over 4h. "
                "Smart money is exiting into retail buying — classic "
                "distribution pattern."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=DIVERGENCE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Rule 2: Bullish accumulation — price down + OI building + funding ≤ 0.
        if (
            funding is not None
            and oi_change is not None
            and price_change_pct is not None
            and price_change_pct < PRICE_DOWN_THRESHOLD_PCT
            and funding <= FUNDING_NON_POSITIVE_MAX
            and oi_change >= OI_BUILD_THRESHOLD_PCT
        ):
            reasoning = (
                f"BULLISH accumulation: price {price_change_pct:+.2f}% over the "
                f"last {PRICE_LOOKBACK_CANDLES} candles while OI built "
                f"{oi_change:+.1f}% and funding stayed at {funding:+.4f}%. "
                "Smart money is adding into the pullback — shorts are paying "
                "longs to absorb the dip."
            )
            return self._build_signal(
                direction="BULLISH",
                confidence=DIVERGENCE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Rule 3: Extreme crowded long → contrarian BEARISH lean.
        if funding is not None and funding > FUNDING_EXTREME_CROWDED_LONG:
            reasoning = (
                f"Extreme crowded long: funding rate {funding:+.4f}% is above "
                f"the {FUNDING_EXTREME_CROWDED_LONG:+.2f}% extreme threshold. "
                "Crowded one-sided positioning historically resolves with a "
                "long squeeze — contrarian BEARISH lean."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=EXTREME_FUNDING_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Rule 4: Extreme crowded short → contrarian BULLISH lean.
        if funding is not None and funding < FUNDING_EXTREME_CROWDED_SHORT:
            reasoning = (
                f"Extreme crowded short: funding rate {funding:+.4f}% is below "
                f"the {FUNDING_EXTREME_CROWDED_SHORT:+.2f}% extreme threshold. "
                "Crowded one-sided positioning historically resolves with a "
                "short squeeze — contrarian BULLISH lean."
            )
            return self._build_signal(
                direction="BULLISH",
                confidence=EXTREME_FUNDING_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Rule 5: Default NEUTRAL with explicit reasoning so the data moat
        # captures *why* nothing fired.
        reason_parts: list[str] = []
        if funding is None:
            reason_parts.append("funding rate unavailable")
        else:
            reason_parts.append(f"funding {funding:+.4f}% (within normal band)")
        if oi_change is None:
            reason_parts.append(
                "OI delta unavailable — provider does not yet track 4h OI history"
            )
        else:
            reason_parts.append(f"OI {oi_change:+.1f}% over 4h (no clear bias)")
        if price_change_pct is None:
            reason_parts.append("insufficient candles for price-drift check")
        else:
            reason_parts.append(
                f"price {price_change_pct:+.2f}% over last "
                f"{PRICE_LOOKBACK_CANDLES} candles"
            )

        return self._neutral(
            reasoning="No flow signal: " + "; ".join(reason_parts) + ".",
            richness=_normalise_richness(flow.data_richness),
        )

    # -- Helpers ---------------------------------------------------------

    @staticmethod
    def _recent_price_change_pct(candles: Iterable[Mapping]) -> float | None:
        """Compute the close-to-close % change over the last N candles.

        Returns ``None`` when the candle stream is too short or malformed.
        """
        rows = list(candles)
        if len(rows) < PRICE_LOOKBACK_CANDLES + 1:
            return None
        try:
            anchor_close = float(rows[-(PRICE_LOOKBACK_CANDLES + 1)]["close"])
            current_close = float(rows[-1]["close"])
        except (KeyError, TypeError, ValueError):
            return None
        if anchor_close == 0:
            return None
        return ((current_close - anchor_close) / anchor_close) * 100.0

    def _build_signal(
        self,
        *,
        direction: str,
        confidence: float,
        reasoning: str,
        richness: str,
    ) -> SignalOutput:
        return SignalOutput(
            agent_name=self.name(),
            signal_type=self.signal_type(),
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            signal_category="directional",
            data_richness=richness,
            contradictions="",
            key_levels={},
            pattern_detected=None,
            raw_output=reasoning,
        )

    def _neutral(self, *, reasoning: str, richness: str) -> SignalOutput:
        return self._build_signal(
            direction="NEUTRAL",
            confidence=NEUTRAL_CONFIDENCE,
            reasoning=reasoning,
            richness=richness,
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _normalise_richness(value: str | None) -> str:
    """Map FlowOutput's UPPER richness to SignalOutput's lower-case convention."""
    if not value:
        return "minimal"
    return value.lower()
