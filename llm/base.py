"""Abstract LLMProvider for text and vision generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str  # raw text response
    input_tokens: int
    output_tokens: int
    cost: float  # estimated cost in USD
    model: str
    latency_ms: float
    cached_input_tokens: int  # tokens served from cache


class LLMProvider(ABC):
    """Abstract base for LLM providers (Claude, Groq, etc.)."""

    @abstractmethod
    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse: ...

    @abstractmethod
    async def generate_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: bytes,
        image_media_type: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse: ...

    # ------------------------------------------------------------------
    # Per-cycle LLM usage tracking (concrete — shared by all providers)
    # ------------------------------------------------------------------

    def reset_usage(self) -> None:
        """Reset per-cycle token/cost accumulators. Called at cycle start."""
        self._cycle_input_tokens = 0
        self._cycle_output_tokens = 0
        self._cycle_cost = 0.0

    def get_usage(self) -> dict:
        """Return accumulated usage since last reset."""
        return {
            "input_tokens": getattr(self, "_cycle_input_tokens", 0),
            "output_tokens": getattr(self, "_cycle_output_tokens", 0),
            "cost_usd": round(getattr(self, "_cycle_cost", 0.0), 6),
        }

    def _accumulate_usage(self, response: LLMResponse) -> None:
        """Add a response's token counts to the cycle accumulators."""
        self._cycle_input_tokens = getattr(self, "_cycle_input_tokens", 0) + response.input_tokens
        self._cycle_output_tokens = getattr(self, "_cycle_output_tokens", 0) + response.output_tokens
        self._cycle_cost = getattr(self, "_cycle_cost", 0.0) + response.cost
