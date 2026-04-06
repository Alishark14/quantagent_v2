"""Memory system: 4 feedback loops for the analysis pipeline.

Loop 1: CycleMemory — recent cycle decisions (short-term)
Loop 2: ReflectionRules — learned rules with self-correcting counters (medium-term)
Loop 3: CrossBotSignals — cross-bot signal sharing, user_id scoped (real-time)
Loop 4: RegimeHistory — regime ring buffer for transition detection (medium-term)
"""

from __future__ import annotations

from engine.memory.cross_bot import CrossBotSignals
from engine.memory.cycle_memory import CycleMemory
from engine.memory.reflection_rules import ReflectionRules
from engine.memory.regime_history import RegimeHistory


async def build_memory_context(
    cycle_mem: CycleMemory,
    rules: ReflectionRules,
    cross_bot: CrossBotSignals,
    regime: RegimeHistory,
    bot_id: str,
    symbol: str,
    timeframe: str,
    user_id: str,
) -> str:
    """Assemble full memory context string for ConvictionAgent/DecisionAgent.

    Gathers data from all 4 memory loops and formats them into a single
    string that gets injected into agent prompts.
    """
    recent_cycles = await cycle_mem.get_recent(bot_id)
    active_rules = await rules.get_active_rules(symbol, timeframe)
    other_signals = await cross_bot.get_other_bot_signals(symbol, user_id)

    parts = [
        cycle_mem.format_for_prompt(recent_cycles),
        rules.format_for_prompt(active_rules),
        cross_bot.format_for_prompt(other_signals),
        regime.format_for_prompt(),
    ]
    return "\n\n".join(parts)
