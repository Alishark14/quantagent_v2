"""QuantAgent v2 version metadata and cost tracking."""

ENGINE_VERSION = "2026.04.2.0.0-alpha.1"
API_VERSION = "v1"

PROMPT_VERSIONS: dict[str, str] = {
    "indicator_agent": "1.0",
    "pattern_agent": "1.0",
    "trend_agent": "1.0",
    "conviction_agent": "1.0",
    "decision_agent": "1.0",
    "reflection_agent": "1.0",
}

ML_MODEL_VERSIONS: dict[str, str | None] = {
    "direction_model": None,
    "regime_model": None,
    "anomaly_detector": None,
}
