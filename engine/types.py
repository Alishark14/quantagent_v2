"""All shared types: MarketData, SignalOutput, ConvictionOutput, etc."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


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
    signal_type: str  # "llm" | "ml"
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
    """Output from the ConvictionAgent meta-evaluator."""

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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeAction:
    """Decision output from the DecisionAgent."""

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
