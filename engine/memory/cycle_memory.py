"""Loop 1: recent cycle memory and position state tracking.

Stores and retrieves the last N analysis cycles per bot.
Injected into ConvictionAgent and DecisionAgent for short-term context.
"""

from __future__ import annotations

import logging

from storage.repositories.base import CycleRepository

logger = logging.getLogger(__name__)


class CycleMemory:
    """Short-term memory: last N cycle decisions per bot."""

    def __init__(self, cycle_repo: CycleRepository) -> None:
        self._repo = cycle_repo

    async def save_cycle(self, bot_id: str, cycle_data: dict) -> None:
        """Save a cycle record with the bot_id injected."""
        await self._repo.save_cycle({**cycle_data, "bot_id": bot_id})

    async def get_recent(self, bot_id: str, limit: int = 5) -> list[dict]:
        """Get the most recent cycles for a bot, ordered by timestamp descending."""
        return await self._repo.get_recent_cycles(bot_id, limit)

    def format_for_prompt(self, cycles: list[dict]) -> str:
        """Format recent cycles as context string for agent prompts."""
        if not cycles:
            return "No prior cycles."
        lines = []
        for c in cycles:
            lines.append(
                f"- {c.get('timestamp', '?')}: {c.get('action', '?')} "
                f"(conviction={c.get('conviction_score', '?')})"
            )
        return f"Recent cycles:\n" + "\n".join(lines)
