"""Loop 3: cross-bot signal sharing (user_id scoped).

Multiple bots may run on different timeframes for the same asset.
This module lets bots see each other's recent decisions — scoped by user_id
for multi-tenant isolation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from storage.repositories.base import CrossBotRepository

logger = logging.getLogger(__name__)


class CrossBotSignals:
    """Real-time cross-bot intelligence sharing (user_id scoped)."""

    def __init__(self, cross_bot_repo: CrossBotRepository) -> None:
        self._repo = cross_bot_repo

    async def publish_signal(
        self,
        user_id: str,
        bot_id: str,
        symbol: str,
        direction: str,
        conviction: float,
    ) -> None:
        """Publish this bot's latest signal for other bots to read."""
        await self._repo.save_signal({
            "user_id": user_id,
            "bot_id": bot_id,
            "symbol": symbol,
            "direction": direction,
            "conviction": conviction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def get_other_bot_signals(
        self, symbol: str, user_id: str, limit: int = 10
    ) -> list[dict]:
        """Get recent signals from other bots for the same symbol/user."""
        return await self._repo.get_recent_signals(symbol, user_id, limit)

    def format_for_prompt(self, signals: list[dict]) -> str:
        """Format cross-bot signals as context string for agent prompts."""
        if not signals:
            return "No signals from other bots."
        lines = []
        for s in signals:
            lines.append(
                f"- Bot {s.get('bot_id', '?')}: {s.get('direction', '?')} "
                f"(conviction={s.get('conviction', '?')}) at {s.get('timestamp', '?')}"
            )
        return f"Cross-bot signals:\n" + "\n".join(lines)
