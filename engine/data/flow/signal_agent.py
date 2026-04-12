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

Rule priority (first match wins):
    1–5. Options rules (BTC / ETH only)
    6–9. COT rules (GOLD / SILVER / WTIOIL / BRENTOIL only)
    10–13. RegSHO equity rules (TSLA / NVDA / GOOGL only)
    14–17. Funding / OI rules (all symbols, crypto or otherwise)
    18. Default NEUTRAL

Each block's symbol gating is enforced by ``None`` guards on its
derived fields — the relevant FlowProvider returns empty dicts for
out-of-scope symbols, leaving those fields ``None``, so any rule that
reads them short-circuits and the agent falls through to the next
block cleanly.

Options rules (BTC / ETH only — other symbols have ``put_call_ratio``,
``dvol``, ``skew_25d``, ``gex_regime`` = None from OptionsEnrichment
and fall through to the funding/OI rules unchanged) run FIRST because
options data is more informative than funding alone:

1. **Options bearish hedging** (``put_call_ratio > 1.2``) → heavy
   downside hedging from institutions. BEARISH @ 0.60.
2. **Options complacent longs** (``put_call_ratio < 0.5``) → call
   dominance. Contrarian BEARISH @ 0.55.
3. **Extreme skew** (``skew_25d > 10``) → options market paying up for
   downside protection. BEARISH @ 0.55.
4. **Volatility spike** (``dvol_change_24h > 20``) → uncertainty too
   high, wait for regime to settle. NEUTRAL @ 0.55.
5. **Positive gamma pinning** (``gex_regime == POSITIVE_GAMMA``) →
   market makers dampen moves, expect range-bound price action.
   NEUTRAL @ 0.50. Note: the divergence / accumulation rules below
   get a +0.10 confidence boost when ``gex_regime == NEGATIVE_GAMMA``
   (capped at 0.80) because negative gamma amplifies moves.

COT rules (commodity-only — CommodityFlowProvider leaves every COT
field None for non-commodity symbols so the guards short-circuit):

6. **Extreme speculator long** (``cot_speculator_percentile > 90``) →
   managed money positioning at the 90th percentile. Contrarian
   BEARISH @ 0.55.
7. **Extreme speculator short** (``cot_speculator_percentile < 10``) →
   managed money positioning at the 10th percentile. Contrarian
   BULLISH @ 0.55.
8. **Commercial divergence bullish** (``cot_divergence > 0`` AND
   ``cot_divergence_abs_percentile > 80`` AND
   ``cot_speculator_percentile < 30``) → commercials accumulating
   while speculators are short. BULLISH @ 0.60.
9. **Commercial divergence bearish** (``cot_divergence < 0`` AND
   ``cot_divergence_abs_percentile > 80`` AND
   ``cot_speculator_percentile > 70``) → commercials reducing while
   speculators are long. BEARISH @ 0.60.

RegSHO equity rules (TSLA / NVDA / GOOGL only — EquityFlowProvider
leaves every ``svr_*`` and ``market_open`` field None for non-equity
symbols so these guards short-circuit):

10. **Extreme short volume** (``svr_zscore > 2.0``) → unusual
    institutional short activity. BEARISH @ 0.55.
11. **Short squeeze setup** (``svr_zscore > 2.0`` AND price up > 1%
    over 12 candles AND ``svr_trend == "RISING"``) → shorts are
    being overrun by a rising tape. BULLISH @ 0.60.
12. **Short volume collapse** (``svr_zscore < -1.5``) → bears
    stepping away. BULLISH @ 0.50.
13. **Outside market hours caution** (``market_open is False``) →
    HIP-3 oracle tracking is weaker, spreads are wider, so flow
    alone shouldn't drive a directional trade. NEUTRAL @ 0.40.

Funding / OI rules (run for every symbol after the options + COT +
RegSHO blocks fall through, or immediately for symbols where all
three upstream blocks' fields are None):

6. **Bearish divergence** (price up + funding bearish + OI dropping) →
   smart money exiting while retail is still buying. Drop conviction on
   any LONG setup.

7. **Bullish accumulation** (price down + funding ≤ 0 + OI building) →
   smart money accumulating during a pullback. Often the highest-quality
   contrarian LONG signal.

8. **Extreme funding crowded long** (funding > +0.10%) → contrarian
   BEARISH lean, regardless of price action. The market is paying through
   the nose to be long; squeezes happen at extremes.

9. **Extreme funding crowded short** (funding < -0.10%) → contrarian
   BULLISH lean.

10. **Default NEUTRAL** — no clear signal. Confidence kept low so this
    doesn't drown out the other agents during normal conditions.

OI history
==========

``CryptoFlowProvider`` maintains a per-symbol rolling OI history buffer
(``deque(maxlen=480)``) and computes ``oi_change_4h`` after a 4-hour
warmup window. The BEARISH-divergence and BULLISH-accumulation rules
fire automatically once the buffer is warm — no change required to this
agent. During the cold-start warmup, ``oi_change_4h`` is ``None`` and
these rules fall through to NEUTRAL (safe default).
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
# Options thresholds (BTC / ETH only — OptionsEnrichment returns None for
# other symbols and the rules short-circuit on the None guard)
# ---------------------------------------------------------------------------

#: Put/call OI ratio above which institutions are heavily hedging
#: downside. BEARISH lean. Higher than 1.0 so we only fire when the
#: book is *materially* put-heavy, not just balanced.
OPTIONS_BEARISH_PCR_THRESHOLD = 1.2

#: Put/call OI ratio below which call dominance reads as complacent
#: longs. Contrarian BEARISH lean — very low PCR historically resolves
#: with a pullback as one-sided positioning unwinds.
OPTIONS_COMPLACENT_PCR_THRESHOLD = 0.5

#: 25-delta skew (put IV − call IV, in IV points) above which the
#: options market is pricing significant downside risk. BEARISH.
OPTIONS_EXTREME_SKEW_THRESHOLD = 10.0

#: 24-hour DVOL change (%) above which the market is in a vol-spike
#: regime — uncertainty too high for a confident directional read.
#: NEUTRAL with a short "wait for regime to settle" reasoning.
OPTIONS_VOL_SPIKE_THRESHOLD_PCT = 20.0

#: Confidence for the options-hedging / skew rules. Higher than NEUTRAL
#: but lower than the divergence rules because options data is leading
#: but still only one voice.
OPTIONS_BEARISH_CONFIDENCE = 0.60

#: Confidence for the contrarian complacent-longs rule and the extreme
#: skew rule. Slightly below the hedging rule — both are contrarian
#: reads that can take longer to resolve than direct hedging signals.
OPTIONS_CONTRARIAN_CONFIDENCE = 0.55

#: Confidence for volatility-spike NEUTRAL output. Higher than the
#: default NEUTRAL because we actively want to *discourage* trading
#: during vol spikes, not just stay quiet.
OPTIONS_VOL_SPIKE_CONFIDENCE = 0.55

#: Confidence for the positive-gamma "expect ranging" NEUTRAL output.
OPTIONS_POSITIVE_GAMMA_CONFIDENCE = 0.50

#: Confidence boost for divergence / accumulation rules when the
#: options market is in a NEGATIVE_GAMMA regime — market makers are
#: amplifying moves, so a directional read is more likely to continue.
#: Capped at 0.80 so the combined confidence never exceeds the
#: FlowSignalAgent's hard ceiling.
OPTIONS_NEGATIVE_GAMMA_BOOST = 0.10
OPTIONS_CONFIDENCE_CEILING = 0.80

# ---------------------------------------------------------------------------
# COT thresholds (GOLD / SILVER / WTIOIL / BRENTOIL — CommodityFlowProvider
# leaves the fields None for other symbols so the rules short-circuit).
# ---------------------------------------------------------------------------

#: Speculator percentile above which managed money is "extreme crowded
#: long" — contrarian BEARISH.
COT_SPEC_EXTREME_LONG_PCT = 90.0

#: Speculator percentile below which managed money is "extreme crowded
#: short" — contrarian BULLISH.
COT_SPEC_EXTREME_SHORT_PCT = 10.0

#: Percentile cut-off for the abs(divergence) "in the top 20%" clause
#: on the commercial-divergence rules. Percentile of 80 means the
#: current divergence is bigger than 80% of its own 52-week history.
COT_DIVERGENCE_EXTREME_PCT = 80.0

#: Speculator percentile ceiling on the "commercials bullish" rule —
#: the rule only fires when speculators are also net short enough to
#: make the commercial accumulation a real contrarian signal.
COT_BULLISH_SPEC_CEILING = 30.0

#: Speculator percentile floor on the "commercials bearish" rule —
#: only fires when speculators are net long enough for commercials'
#: exit to signal smart money stepping away from a crowded trade.
COT_BEARISH_SPEC_FLOOR = 70.0

#: Confidence for the contrarian extreme-speculator rules.
COT_EXTREME_SPEC_CONFIDENCE = 0.55

#: Confidence for the two commercial-divergence rules. Highest in the
#: COT block — when commercials and speculators disagree sharply in a
#: historically extreme way, commercials are usually right.
COT_DIVERGENCE_CONFIDENCE = 0.60

# ---------------------------------------------------------------------------
# RegSHO thresholds (TSLA / NVDA / GOOGL — EquityFlowProvider leaves every
# svr_* / market_open field None for non-equity symbols so the rules
# short-circuit).
# ---------------------------------------------------------------------------

#: Short-volume Z-score threshold for "extreme short activity".
#: Symmetrical high side — unusual ≈ 2σ above the 20-day mean.
REGSHO_EXTREME_ZSCORE = 2.0

#: Short-volume Z-score threshold for "bears stepping away". Asymmetric
#: — we care more about extreme highs than extreme lows, so the collapse
#: rule trips one step sooner than the high-side rule.
REGSHO_COLLAPSE_ZSCORE = -1.5

#: Minimum % price change over the 12-candle lookback for the short
#: squeeze rule to treat price as "rising" into the short activity.
REGSHO_SQUEEZE_PRICE_UP_PCT = 1.0

#: Confidence for the RegSHO rules.
REGSHO_EXTREME_SHORT_CONFIDENCE = 0.55
REGSHO_SQUEEZE_CONFIDENCE = 0.60
REGSHO_COLLAPSE_CONFIDENCE = 0.50
REGSHO_CLOSED_MARKET_CONFIDENCE = 0.40


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

        # ── Options rules (BTC / ETH only — every other symbol has
        # these fields = None from OptionsEnrichment and falls through
        # to the funding / OI rules unchanged). Run BEFORE funding /
        # OI because options data is more informative than funding
        # alone.

        pcr = flow.put_call_ratio
        skew = flow.skew_25d
        dvol_change = flow.dvol_change_24h
        gex = flow.gex_regime

        # Options rule 1: heavy put hedging.
        if pcr is not None and pcr > OPTIONS_BEARISH_PCR_THRESHOLD:
            reasoning = (
                f"Options bearish hedging: put/call OI ratio {pcr:.2f} above "
                f"the {OPTIONS_BEARISH_PCR_THRESHOLD:.2f} threshold — "
                "institutions are paying for downside protection."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=OPTIONS_BEARISH_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Options rule 2: extreme call dominance → contrarian BEARISH.
        if pcr is not None and pcr < OPTIONS_COMPLACENT_PCR_THRESHOLD:
            reasoning = (
                f"Options complacent longs: put/call OI ratio {pcr:.2f} "
                f"below the {OPTIONS_COMPLACENT_PCR_THRESHOLD:.2f} "
                "threshold — extreme call dominance historically resolves "
                "with a pullback as one-sided positioning unwinds."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=OPTIONS_CONTRARIAN_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Options rule 3: extreme 25-delta skew → BEARISH.
        if skew is not None and skew > OPTIONS_EXTREME_SKEW_THRESHOLD:
            reasoning = (
                f"Options extreme skew: 25-delta put IV − call IV = "
                f"{skew:+.1f} IV points, above the "
                f"{OPTIONS_EXTREME_SKEW_THRESHOLD:.1f} threshold — "
                "market is pricing significant downside risk."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=OPTIONS_CONTRARIAN_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Options rule 4: vol spike → NEUTRAL (wait out the regime).
        if (
            dvol_change is not None
            and dvol_change > OPTIONS_VOL_SPIKE_THRESHOLD_PCT
        ):
            reasoning = (
                f"Volatility spike: DVOL up {dvol_change:+.1f}% over 24h, "
                f"above the {OPTIONS_VOL_SPIKE_THRESHOLD_PCT:.0f}% "
                "threshold — uncertainty too high, wait for regime "
                "to settle."
            )
            return self._build_signal(
                direction="NEUTRAL",
                confidence=OPTIONS_VOL_SPIKE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Options rule 5: positive gamma → NEUTRAL (expect ranging).
        # Only fires when no directional divergence / accumulation is
        # about to fire below (it's checked first because positive
        # gamma actively suppresses directional moves).
        if gex == "POSITIVE_GAMMA":
            reasoning = (
                "Options positive gamma regime: market makers dampen "
                "moves, expect range-bound price action — no directional "
                "trade from flow."
            )
            return self._build_signal(
                direction="NEUTRAL",
                confidence=OPTIONS_POSITIVE_GAMMA_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # Negative-gamma confidence boost: amplifies the divergence /
        # accumulation rules below because market makers are
        # amplifying moves. Capped at the ceiling to keep the combined
        # confidence under 0.80.
        divergence_confidence = DIVERGENCE_CONFIDENCE
        if gex == "NEGATIVE_GAMMA":
            divergence_confidence = min(
                DIVERGENCE_CONFIDENCE + OPTIONS_NEGATIVE_GAMMA_BOOST,
                OPTIONS_CONFIDENCE_CEILING,
            )

        # ── COT rules (commodity only — CommodityFlowProvider leaves
        # every cot_* field None for non-commodity symbols so these
        # guards short-circuit on crypto / FX / equity symbols).

        spec_pct = flow.cot_speculator_percentile
        divergence = flow.cot_divergence
        divergence_abs_pct = flow.cot_divergence_abs_percentile

        # COT rule 1: extreme speculator long → contrarian BEARISH.
        if spec_pct is not None and spec_pct > COT_SPEC_EXTREME_LONG_PCT:
            reasoning = (
                f"COT extreme speculator long: managed money net "
                f"positioning at the {spec_pct:.0f}th percentile of the "
                f"last 52 weeks (> {COT_SPEC_EXTREME_LONG_PCT:.0f}th) — "
                "crowded positioning historically resolves with a "
                "pullback as the speculative book unwinds."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=COT_EXTREME_SPEC_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # COT rule 2: extreme speculator short → contrarian BULLISH.
        if spec_pct is not None and spec_pct < COT_SPEC_EXTREME_SHORT_PCT:
            reasoning = (
                f"COT extreme speculator short: managed money net "
                f"positioning at the {spec_pct:.0f}th percentile of the "
                f"last 52 weeks (< {COT_SPEC_EXTREME_SHORT_PCT:.0f}th) — "
                "extreme pessimism historically marks contrarian bottoms."
            )
            return self._build_signal(
                direction="BULLISH",
                confidence=COT_EXTREME_SPEC_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # COT rule 3: commercial divergence BULLISH — commercials are
        # more long than speculators, divergence is historically extreme,
        # and speculators are still net short enough to make the
        # commercial accumulation a real contrarian signal.
        if (
            divergence is not None
            and divergence_abs_pct is not None
            and spec_pct is not None
            and divergence > 0
            and divergence_abs_pct > COT_DIVERGENCE_EXTREME_PCT
            and spec_pct < COT_BULLISH_SPEC_CEILING
        ):
            reasoning = (
                f"COT commercial divergence BULLISH: commercial net "
                f"{flow.cot_commercial_net:+.0f} is {divergence:+.0f} "
                f"contracts above managed money net "
                f"{flow.cot_managed_money_net:+.0f} (top "
                f"{100 - divergence_abs_pct:.0f}% historically) while "
                f"speculators sit at the {spec_pct:.0f}th percentile. "
                "Commercials accumulating into speculative shorts — "
                "follow the hedgers."
            )
            return self._build_signal(
                direction="BULLISH",
                confidence=COT_DIVERGENCE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # COT rule 4: commercial divergence BEARISH — commercials are
        # more short than speculators, divergence is historically
        # extreme, and speculators are still net long enough to make
        # the commercial exit a real smart-money-leaving signal.
        if (
            divergence is not None
            and divergence_abs_pct is not None
            and spec_pct is not None
            and divergence < 0
            and divergence_abs_pct > COT_DIVERGENCE_EXTREME_PCT
            and spec_pct > COT_BEARISH_SPEC_FLOOR
        ):
            reasoning = (
                f"COT commercial divergence BEARISH: commercial net "
                f"{flow.cot_commercial_net:+.0f} is {divergence:+.0f} "
                f"contracts below managed money net "
                f"{flow.cot_managed_money_net:+.0f} (top "
                f"{100 - divergence_abs_pct:.0f}% historically) while "
                f"speculators sit at the {spec_pct:.0f}th percentile. "
                "Commercials reducing while speculators crowd long — "
                "smart money exiting ahead of a top."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=COT_DIVERGENCE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # ── RegSHO equity rules (TSLA / NVDA / GOOGL only —
        # EquityFlowProvider leaves svr_* and market_open None for
        # non-equity symbols so these guards short-circuit).

        svr_z = flow.svr_zscore
        svr_trend = flow.svr_trend
        market_open_flag = flow.market_open

        # RegSHO rule 1: extreme short volume → BEARISH.
        # Priority-split: if both rule 1 AND rule 2 would fire, rule 2
        # (squeeze setup) wins because it's more actionable — check
        # the squeeze condition FIRST.
        squeeze_setup = (
            svr_z is not None
            and svr_z > REGSHO_EXTREME_ZSCORE
            and svr_trend == "RISING"
            and price_change_pct is not None
            and price_change_pct > REGSHO_SQUEEZE_PRICE_UP_PCT
        )
        if squeeze_setup:
            reasoning = (
                f"RegSHO short squeeze setup: SVR Z-score {svr_z:+.1f} "
                f"above {REGSHO_EXTREME_ZSCORE:.1f} with SVR trend "
                f"RISING and price up {price_change_pct:+.2f}% over "
                f"{PRICE_LOOKBACK_CANDLES} candles — shorts being "
                "overrun by a rising tape."
            )
            return self._build_signal(
                direction="BULLISH",
                confidence=REGSHO_SQUEEZE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        if svr_z is not None and svr_z > REGSHO_EXTREME_ZSCORE:
            reasoning = (
                f"RegSHO extreme short volume: SVR Z-score {svr_z:+.1f} "
                f"above {REGSHO_EXTREME_ZSCORE:.1f} — unusual "
                "institutional short activity."
            )
            return self._build_signal(
                direction="BEARISH",
                confidence=REGSHO_EXTREME_SHORT_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # RegSHO rule 3: short volume collapse → BULLISH.
        if svr_z is not None and svr_z < REGSHO_COLLAPSE_ZSCORE:
            reasoning = (
                f"RegSHO short volume collapse: SVR Z-score {svr_z:+.1f} "
                f"below {REGSHO_COLLAPSE_ZSCORE:.1f} — bears stepping "
                "away from the name."
            )
            return self._build_signal(
                direction="BULLISH",
                confidence=REGSHO_COLLAPSE_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

        # RegSHO rule 4: market closed → NEUTRAL caution.
        # Only fires for equity symbols (market_open is None for every
        # non-equity symbol) so crypto / FX / commodity bots never
        # short-circuit on this rule.
        if market_open_flag is False:
            reasoning = (
                "RegSHO outside market hours: US cash equities are "
                "closed — HIP-3 oracle tracking weaker, spreads wider, "
                "no directional trade from flow."
            )
            return self._build_signal(
                direction="NEUTRAL",
                confidence=REGSHO_CLOSED_MARKET_CONFIDENCE,
                reasoning=reasoning,
                richness=_normalise_richness(flow.data_richness),
            )

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
                confidence=divergence_confidence,
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
                confidence=divergence_confidence,
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
