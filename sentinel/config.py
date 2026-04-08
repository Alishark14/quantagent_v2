"""Sentinel thresholds, cooldown periods, and budget configuration.

Cooldowns and daily budgets are timeframe-dependent: a 15m bot can
trigger more often than a 4h bot. Defaults are sensible for cost control.
"""

from __future__ import annotations

# Cooldown between triggers (seconds). One candle period per timeframe.
SENTINEL_COOLDOWNS: dict[str, int] = {
    "15m": 900,       # 15 minutes
    "30m": 1800,      # 30 minutes
    "1h": 3600,       # 60 minutes
    "4h": 14400,      # 4 hours
    "1d": 86400,      # 24 hours
}

# Max full-pipeline triggers per day per symbol.
SENTINEL_DAILY_BUDGETS: dict[str, int] = {
    "15m": 16,
    "30m": 12,
    "1h": 8,
    "4h": 4,
    "1d": 2,
}

# ---------------------------------------------------------------------------
# Readiness escalation (post-SKIP threshold ratchet)
# ---------------------------------------------------------------------------
#
# After a SetupDetected fires and the pipeline returns SKIP, the Sentinel
# raises its readiness threshold for that symbol by ESCALATION_STEP and
# applies a SHORTER cooldown (SKIP_COOLDOWN_SECONDS) instead of the full
# candle period. The threshold is capped at BASE + MAX_ESCALATION so it
# never gates the symbol completely. On a successful TRADE, the threshold
# resets to BASE_READINESS_THRESHOLD and the cooldown returns to the full
# candle period. On every new candle close, ALL escalations are reset.
#
# Why escalate after SKIP rather than just back off?
#   1. A SKIP means the readiness conditions fired but the analysis
#      pipeline didn't see a tradeable setup. The conditions were
#      probably noise — re-firing them at the same threshold within
#      the same candle wastes the LLM budget.
#   2. The escalation creates a self-tuning hysteresis: if the market
#      is genuinely setting up, readiness will climb past the higher
#      threshold and try again; if it was noise, the threshold stays
#      raised and the bot stays quiet.
#   3. Reset-on-candle-close keeps the system from getting stuck — every
#      new candle is a fresh chance.
#
# All four constants are module-level so tests can pin the literal
# numbers and so future tuning is one diff. They are also exposed
# through `get_sentinel_escalation_config()` for callers that want a
# single bag of escalation tunables (used by `SentinelMonitor.__init__`).
#
BASE_READINESS_THRESHOLD: float = 0.30
ESCALATION_STEP: float = 0.10
MAX_ESCALATION: float = 0.25  # threshold ceiling = BASE + MAX_ESCALATION = 0.55
SKIP_COOLDOWN_SECONDS: int = 900  # 15 minutes after SKIP outcome


def get_sentinel_cooldown(timeframe: str) -> int:
    """Get cooldown in seconds for a timeframe. Default: 3600 (1h)."""
    return SENTINEL_COOLDOWNS.get(timeframe, 3600)


def get_sentinel_daily_budget(timeframe: str) -> int:
    """Get max daily triggers for a timeframe. Default: 8."""
    return SENTINEL_DAILY_BUDGETS.get(timeframe, 8)


def get_sentinel_escalation_config() -> dict:
    """Return the current escalation tunables as a dict.

    Convenience accessor used by `SentinelMonitor.__init__` so the
    monitor doesn't have to import each constant individually. The
    returned dict is a *copy* of the module-level state — mutating it
    has no effect on subsequent calls.
    """
    return {
        "base_threshold": BASE_READINESS_THRESHOLD,
        "escalation_step": ESCALATION_STEP,
        "max_escalation": MAX_ESCALATION,
        "skip_cooldown_seconds": SKIP_COOLDOWN_SECONDS,
    }
