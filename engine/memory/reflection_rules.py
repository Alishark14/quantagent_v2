"""Loop 2: learned rules with self-correcting success/fail counters.

Rules are per-asset, per-timeframe. Each rule has a score counter:
  +1 when the rule correctly prevented a loss
  -1 when the rule incorrectly prevented a winner
When score drops below -2, the rule auto-deactivates (handled by repository).
"""

from __future__ import annotations

import logging

from storage.repositories.base import RuleRepository

logger = logging.getLogger(__name__)


class ReflectionRules:
    """Medium-term memory: learned trading rules from ReflectionAgent."""

    def __init__(self, rule_repo: RuleRepository) -> None:
        self._repo = rule_repo

    async def get_active_rules(self, symbol: str, timeframe: str) -> list[dict]:
        """Get active rules for a symbol+timeframe. Already filtered by repo."""
        return await self._repo.get_rules(symbol, timeframe)

    async def save_rule(self, rule: dict) -> str:
        """Save a new reflection rule. Returns the rule ID."""
        return await self._repo.save_rule(rule)

    async def increment_score(self, rule_id: str) -> None:
        """Rule correctly prevented a loss: +1."""
        await self._repo.update_rule_score(rule_id, +1)

    async def decrement_score(self, rule_id: str) -> None:
        """Rule incorrectly prevented a winner: -1.

        Auto-deactivation at score < -2 is handled by the repository layer.
        """
        await self._repo.update_rule_score(rule_id, -1)

    async def deactivate_rule(self, rule_id: str) -> None:
        """Manually deactivate a rule."""
        await self._repo.deactivate_rule(rule_id)

    def format_for_prompt(self, rules: list[dict]) -> str:
        """Format active rules as context string for agent prompts."""
        if not rules:
            return "No learned rules for this asset."
        lines = []
        for r in rules:
            lines.append(f"- [score={r.get('score', 0)}] {r.get('rule_text', '?')}")
        return f"Learned rules:\n" + "\n".join(lines)
