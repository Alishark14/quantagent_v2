"""Single source of truth for all version strings.

Versioning: YYYY.MM.MAJOR.MINOR.PATCH[-prerelease][+build]
  - YYYY.MM: auto-updates each calendar month
  - MAJOR: breaking changes to engine API
  - MINOR: new features, agents, adapters
  - PATCH: bug fixes, prompt improvements
  - Pre-release: -alpha.N, -beta.N, -rc.N

Every trade record stores engine_version + prompt_versions at decision time.
"""

ENGINE_VERSION = "2026.04.3.9.0-alpha.1"

API_VERSION = "v1"

PROMPT_VERSIONS: dict[str, str] = {
    "indicator_agent": "3.2",
    "pattern_agent": "2.1",
    "trend_agent": "2.1",
    "conviction_agent": "1.2",
    "decision_agent": "2.1",
    "reflection_agent": "1.0",
    "flow_signal_agent": "1.0",
    "quant_data_scientist": "1.0",
    "macro_regime_manager": "1.0",
}

ML_MODEL_VERSIONS: dict[str, str] = {
    "direction_model": "0.0",
    "regime_model": "0.0",
    "anomaly_detector": "0.0",
}
