"""Pydantic request/response models for all API endpoints."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bot schemas
# ---------------------------------------------------------------------------


class BotCreateRequest(BaseModel):
    """Request to create a new bot."""

    symbol: str = Field(..., examples=["BTC-USDC"])
    timeframe: str = Field(..., examples=["1h"])
    exchange: str = Field(default="hyperliquid", examples=["hyperliquid"])
    account_balance: float = Field(default=0, description="0 = fetch from exchange")
    conviction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_position_pct: float = Field(default=1.0, ge=0.0, le=1.0)


class BotResponse(BaseModel):
    """Bot detail response."""

    id: str
    user_id: str
    symbol: str
    timeframe: str
    exchange: str
    status: str
    config: dict
    created_at: str
    last_health: dict | None = None
    last_cycle: dict | None = None


class BotListResponse(BaseModel):
    """List of bots."""

    bots: list[BotResponse]
    count: int


class AnalyzeResponse(BaseModel):
    """Result of a manual analysis cycle."""

    bot_id: str
    status: str
    action: str
    conviction_score: float
    reasoning: str
    signals_summary: list[dict] = Field(default_factory=list)
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Trade schemas
# ---------------------------------------------------------------------------


class TradeResponse(BaseModel):
    """Full trade detail."""

    id: str
    user_id: str
    bot_id: str
    symbol: str
    timeframe: str
    direction: str
    entry_price: float | None = None
    exit_price: float | None = None
    size: float | None = None
    pnl: float | None = None
    r_multiple: float | None = None
    entry_time: str | None = None
    exit_time: str | None = None
    exit_reason: str | None = None
    conviction_score: float | None = None
    engine_version: str | None = None
    status: str


class TradeListResponse(BaseModel):
    """List of trades."""

    trades: list[TradeResponse]
    count: int


# ---------------------------------------------------------------------------
# Position schemas
# ---------------------------------------------------------------------------


class PositionResponse(BaseModel):
    """Open position snapshot."""

    symbol: str
    direction: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float | None = None
    bot_id: str | None = None


class PositionListResponse(BaseModel):
    """All open positions."""

    positions: list[PositionResponse]
    count: int


# ---------------------------------------------------------------------------
# Health schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """System health snapshot."""

    status: str
    uptime_seconds: float
    total_events: int
    event_counts: dict[str, int]
    error_count: int
    recent_errors: list[dict]
    active_bots: int
    db_status: str


# ---------------------------------------------------------------------------
# Rule schemas
# ---------------------------------------------------------------------------


class RuleResponse(BaseModel):
    """Active reflection rule."""

    id: str
    symbol: str
    timeframe: str
    rule_text: str
    score: int
    active: bool
    created_at: str


class RuleListResponse(BaseModel):
    """List of rules."""

    rules: list[RuleResponse]
    count: int


# ---------------------------------------------------------------------------
# Error schemas
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
