"""Pydantic schema for QuantAgent eval scenarios.

A scenario is a frozen moment in market history paired with the
expected behaviour the engine *should* exhibit on it. Scenarios are
JSON files on disk so they can be hand-edited, version-controlled,
and crowdsourced. The schema is enforced via Pydantic at load time
so a malformed scenario fails loudly instead of silently producing
garbage scores.

See ARCHITECTURE.md §31.4.3 for the full design.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScenarioInput(BaseModel):
    """The market state the engine sees at decision time."""

    model_config = ConfigDict(extra="allow")

    symbol: str
    timeframe: str
    ohlcv: list[dict] = Field(..., description="50-100 historical candles")
    indicators: dict = Field(
        default_factory=dict,
        description="Pre-computed indicator values (rsi, macd, atr, ...)",
    )
    flow_data: dict | None = Field(
        default=None,
        description="Funding rate, OI, GEX, etc. None if unavailable",
    )
    regime_context: str | None = Field(
        default=None,
        description="Coarse regime label: trending / ranging / volatile / quiet",
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp at decision time")


class ExpectedBehavior(BaseModel):
    """Graded expectations the engine output is compared against."""

    model_config = ConfigDict(extra="allow")

    signal_direction: str | None = Field(
        default=None,
        description="BULLISH / BEARISH / NEUTRAL / None (any acceptable)",
    )
    signal_confidence_min: float | None = None
    signal_confidence_max: float | None = None
    conviction_min: float | None = None
    conviction_max: float | None = None
    expected_action: str = Field(
        ...,
        description="LONG / SHORT / SKIP — the canonical action for this scenario",
    )
    key_features_to_mention: list[str] = Field(
        default_factory=list,
        description="Phrases the reasoning trace should reference (e.g. 'overhead resistance')",
    )
    notes: str | None = Field(
        default=None,
        description="Free-text explanation for the labeler / future reviewer",
    )


class Scenario(BaseModel):
    """One self-contained eval scenario."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    category: str = Field(..., description="e.g. clear_setups / clear_avoids / ...")
    version: int = 1
    created_at: str
    last_validated: str
    inputs: ScenarioInput
    expected: ExpectedBehavior
    reference_output: dict | None = Field(
        default=None,
        description="Canonical output from the teacher model — populated after first run",
    )
    metadata: dict | None = None
