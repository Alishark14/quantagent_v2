"""Prompt caching management for Anthropic cache TTL tracking."""

from __future__ import annotations

from datetime import datetime, timezone

# Anthropic prompt cache TTL is ~5 minutes
_CACHE_TTL_SECONDS = 300


class PromptCache:
    """Track which system prompts are warm in Anthropic's cache."""

    def __init__(self, ttl_seconds: int = _CACHE_TTL_SECONDS) -> None:
        self._warm_prompts: dict[str, datetime] = {}
        self._ttl = ttl_seconds

    def mark_warm(self, prompt_hash: str) -> None:
        """Record that a prompt was just cached."""
        self._warm_prompts[prompt_hash] = datetime.now(timezone.utc)

    def is_warm(self, prompt_hash: str) -> bool:
        """Check if a prompt is likely still in Anthropic's cache."""
        ts = self._warm_prompts.get(prompt_hash)
        if ts is None:
            return False
        elapsed = (datetime.now(timezone.utc) - ts).total_seconds()
        if elapsed > self._ttl:
            del self._warm_prompts[prompt_hash]
            return False
        return True

    def clear(self) -> None:
        """Clear all tracked prompts."""
        self._warm_prompts.clear()
