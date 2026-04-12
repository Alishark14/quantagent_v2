"""All shared types: MarketData, SignalOutput, ConvictionOutput, etc."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class ParentTFContext:
    """Parent timeframe trend context for higher-TF confluence."""

    timeframe: str
    trend_direction: str  # "BULLISH" | "BEARISH" | "NEUTRAL"
    ma_position: str  # "ABOVE_50MA" | "BELOW_50MA"
    adx_value: float
    adx_classification: str  # "TRENDING" | "RANGING" | "WEAK"
    bb_width_percentile: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowOutput:
    """Order flow and derivatives data from FlowProviders."""

    funding_rate: float | None
    funding_signal: str  # "CROWDED_LONG" | "CROWDED_SHORT" | "NEUTRAL"
    oi_change_4h: float | None
    oi_trend: str  # "BUILDING" | "DECLINING" | "STABLE"
    nearest_liquidation_above: dict | None  # {price, size}
    nearest_liquidation_below: dict | None  # {price, size}
    gex_regime: str | None  # "POSITIVE_GAMMA" | "NEGATIVE_GAMMA" | None
    gex_flip_level: float | None
    data_richness: str  # "FULL" | "PARTIAL" | "MINIMAL"
    # Options-derived fields (OptionsEnrichment, BTC/ETH only — default
    # None so live / non-crypto paths that don't run the provider keep
    # working unchanged). All four follow the same "None when data
    # unavailable" convention FlowSignalAgent already handles.
    put_call_ratio: float | None = None    # total put OI / total call OI; >1 = put-heavy
    dvol: float | None = None              # Deribit DVOL implied-vol index, current
    dvol_change_24h: float | None = None   # % change in DVOL over 24h
    skew_25d: float | None = None          # 25-delta put IV − 25-delta call IV

    # COT positioning fields (CommodityFlowProvider — GOLD / SILVER /
    # WTIOIL / BRENTOIL only). Weekly CFTC Commitment of Traders data.
    # All default None so crypto / forex / equity paths stay unchanged
    # and FlowSignalAgent's None guards short-circuit the COT rules.
    cot_speculator_percentile: float | None = None  # 0-100; >90 = extreme long, <10 = extreme short
    cot_commercial_net: float | None = None         # raw commercial (Prod_Merc) net position
    cot_managed_money_net: float | None = None      # raw managed-money net position
    cot_weekly_change_pct: float | None = None      # week-over-week % change in MM net
    cot_divergence: float | None = None             # commercial_net − managed_money_net
    cot_divergence_abs_percentile: float | None = None  # 0-100; top-20% == > 80

    # RegSHO equity-flow fields (EquityFlowProvider — TSLA / NVDA /
    # GOOGL only). Daily FINRA off-exchange short-volume file.
    # ``market_open`` is populated for US equity symbols regardless of
    # whether the file fetch succeeded, so the "outside market hours"
    # rule can fire even without a FINRA row.
    short_volume_ratio: float | None = None   # 0.52 = 52% of daily volume was short
    svr_zscore: float | None = None           # 20-day rolling Z-score
    svr_trend: str | None = None              # "RISING" | "FALLING" | "STABLE"
    market_open: bool | None = None           # True when US cash equities are in session

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketData:
    """Complete market data package passed through the analysis pipeline."""

    symbol: str
    timeframe: str
    candles: list[dict]  # OHLCV dicts: {timestamp, open, high, low, close, volume}
    num_candles: int
    lookback_description: str  # e.g., "~6 days"
    forecast_candles: int
    forecast_description: str  # e.g., "~3 hours"
    indicators: dict  # computed indicator values (RSI, MACD, etc.)
    swing_highs: list[float]
    swing_lows: list[float]
    parent_tf: ParentTFContext | None = None
    flow: FlowOutput | None = None
    external_signals: dict = field(default_factory=dict)  # provider_name -> signal

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SignalOutput:
    """Output from a SignalProducer (LLM agent or ML model)."""

    agent_name: str
    signal_type: str  # "llm" | "ml" | "flow"
    direction: str | None  # "BULLISH" | "BEARISH" | "NEUTRAL" | None
    confidence: float  # 0.0 to 1.0
    reasoning: str
    signal_category: str  # "directional" | "regime" | "anomaly"
    data_richness: str  # "full" | "partial" | "minimal"
    contradictions: str  # noted conflicts with grounding data
    key_levels: dict  # {"resistance": float, "support": float}
    pattern_detected: str | None  # e.g., "ascending_triangle"
    raw_output: str  # full LLM response for data moat

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConvictionOutput:
    """Output from the ConvictionAgent meta-evaluator.

    The macro_* fields are populated by ConvictionAgent when a non-expired
    `macro_regime.json` is loaded at cycle start (ARCHITECTURE §13.2.4).
    They default to the no-overlay state so existing call sites that
    don't care about the macro layer continue to work unchanged.
    """

    conviction_score: float  # 0.0 to 1.0
    direction: str  # "LONG" | "SHORT" | "SKIP"
    regime: str  # "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "HIGH_VOLATILITY" | "BREAKOUT"
    regime_confidence: float
    signal_quality: str  # "HIGH" | "MEDIUM" | "LOW" | "CONFLICTING"
    contradictions: list[str]
    reasoning: str
    factual_weight: float
    subjective_weight: float
    raw_output: str  # full LLM response for data moat
    macro_regime: str = "NEUTRAL"  # "RISK_ON" | "RISK_OFF" | "NEUTRAL"
    macro_threshold_boost: float = 0.0  # added to downstream conviction threshold
    macro_position_size_multiplier: float = 1.0  # passed to DecisionAgent for sizing
    macro_blackout_reason: str | None = None  # set when conviction is forced to 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeAction:
    """Decision output from the DecisionAgent.

    DecisionAgent outputs trade INTENT only — direction, SL/TP levels, and a
    deterministic ``risk_weight`` derived from conviction. Dollar sizing is
    owned downstream by ``PortfolioRiskManager``; ``position_size`` is left
    here as ``None`` by DecisionAgent and populated by PRM in the pipeline
    after sizing rules run (Sprint Portfolio-Risk-Manager Task 1).
    """

    action: str  # "LONG" | "SHORT" | "ADD_LONG" | "ADD_SHORT" | "CLOSE_ALL" | "HOLD" | "SKIP"
    conviction_score: float
    position_size: float | None
    sl_price: float | None
    tp1_price: float | None
    tp2_price: float | None
    rr_ratio: float | None
    atr_multiplier: float | None
    reasoning: str
    raw_output: str
    risk_weight: float | None = None  # 0.75 / 1.0 / 1.15 / 1.3, None for non-entry actions

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderResult:
    """Result of an exchange order execution."""

    success: bool
    order_id: str | None
    fill_price: float | None
    fill_size: float | None
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    """Current exchange position snapshot."""

    symbol: str
    direction: str  # "long" | "short"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PriceUpdate:
    """Real-time price tick from a PriceFeed (trades channel).

    Payload for `PriceUpdated` events emitted by the WebSocket PriceFeed
    system (Sprint Week 7 — Event-Driven Refactor Phase 1). Consumers
    like `SLTPMonitor` read `price` on every tick to check open trades
    against SL/TP levels without REST polling.
    """

    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    size: float | None = None  # trade size if from trades channel
    exchange: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandleClose:
    """Completed OHLCV candle from a PriceFeed.

    Payload for `CandleClosed` events. Emitted when a PriceFeed detects
    that the previous candle has finalised (the candle channel pushes a
    new candle with a newer timestamp). Sentinel subscribes to this so
    readiness computation triggers on candle close instead of a 30s
    sleep-poll loop.
    """

    symbol: str
    timeframe: str  # "1h", "4h", etc.
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # candle close time

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FundingUpdate:
    """Funding rate snapshot from a PriceFeed.

    Payload for `FundingUpdated` events. Populated from Hyperliquid's
    `activeAssetCtx` channel so FlowSignalAgent/CryptoFlowProvider read
    the current funding rate from local memory instead of REST.
    """

    symbol: str
    funding_rate: float
    next_funding_time: datetime | None = None
    exchange: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OIUpdate:
    """Open interest snapshot from a PriceFeed.

    Payload for `OpenInterestUpdated` events. `oi_change_pct` is the
    delta vs the previous snapshot captured by the PriceFeed and is
    None on the first observation.
    """

    symbol: str
    open_interest: float
    oi_change_pct: float | None = None  # vs previous snapshot
    exchange: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdapterCapabilities:
    """Declares what an exchange adapter supports."""

    native_sl_tp: bool
    supports_short: bool
    market_hours: dict | None  # None = 24/7
    asset_types: list[str]
    margin_type: str
    has_funding_rate: bool
    has_oi_data: bool
    max_leverage: float
    order_types: list[str]
    supports_partial_close: bool

    def to_dict(self) -> dict:
        return asdict(self)
