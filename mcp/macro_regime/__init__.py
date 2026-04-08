"""Macro Regime Manager — offline MCP agent.

Reads macro-economic conditions from public APIs and produces
``macro_regime.json`` consumed by ConvictionAgent + Sentinel for
global risk-parameter overlay and blackout-window enforcement.

Public API:

- :class:`MacroDataFetcher` — fetches VIX/DXY/DVOL/F&G/etc.
- :class:`MacroSnapshot` — one fetch result (with `available_sources`)
- :class:`LightweightCheck` — hourly delta gate, no LLM
- :class:`CheckResult` — output of LightweightCheck.run
- :class:`MacroRegimeManager` — deep-analysis agent (LLM-powered)
- :class:`MacroRegime` / :class:`MacroAdjustments` / :class:`BlackoutWindow`
- :class:`EconomicEvent` — calendar entry

See ARCHITECTURE.md §13.2.
"""

from mcp.macro_regime.agent import (
    BlackoutWindow,
    MacroAdjustments,
    MacroRegime,
    MacroRegimeManager,
)
from mcp.macro_regime.data_fetcher import (
    EconomicEvent,
    MacroDataFetcher,
    MacroSnapshot,
)
from mcp.macro_regime.lightweight_check import (
    CheckResult,
    LightweightCheck,
)

__all__ = [
    "BlackoutWindow",
    "CheckResult",
    "EconomicEvent",
    "LightweightCheck",
    "MacroAdjustments",
    "MacroDataFetcher",
    "MacroRegime",
    "MacroRegimeManager",
    "MacroSnapshot",
]
