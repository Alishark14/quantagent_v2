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
