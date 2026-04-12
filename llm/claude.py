"""ClaudeProvider: primary LLM provider using Anthropic SDK.

Includes optional LangSmith tracing when LANGCHAIN_TRACING_V2=true.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time

import anthropic

from llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Cost per million tokens (USD)
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "input": 3.0,
        "output": 15.0,
        "cached_input": 0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input": 0.80,
        "output": 4.0,
        "cached_input": 0.08,
    },
}

_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1, 2, 4]

# ---------------------------------------------------------------------------
# LangSmith tracing — conditional, never crashes
# ---------------------------------------------------------------------------

_TRACING_ENABLED = False
_traceable = None

try:
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes"):
        from langsmith import traceable as _traceable_fn
        _traceable = _traceable_fn
        _TRACING_ENABLED = True
        logger.info("LangSmith tracing enabled")
except ImportError:
    logger.debug("langsmith not installed — tracing disabled")
except Exception as e:
    logger.debug(f"LangSmith tracing init failed: {e}")


def _calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> float:
    """Calculate estimated cost in USD."""
    costs = MODEL_COSTS.get(model, MODEL_COSTS["claude-sonnet-4-20250514"])
    billable_input = input_tokens - cached_input_tokens
    return (
        (billable_input / 1_000_000) * costs["input"]
        + (output_tokens / 1_000_000) * costs["output"]
        + (cached_input_tokens / 1_000_000) * costs["cached_input"]
    )


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider with prompt caching and retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        llm_call_repo=None,
    ) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self._llm_call_repo = llm_call_repo

    def _build_system(self, system_prompt: str, cache: bool) -> list[dict]:
        block: dict = {"type": "text", "text": system_prompt}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": user_prompt}]
        return await self._call(
            system=self._build_system(system_prompt, cache_system_prompt),
            messages=messages,
            agent_name=agent_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt_text=system_prompt,
            user_prompt_text=user_prompt,
        )

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
    ) -> LLMResponse:
        b64 = base64.standard_b64encode(image_data).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        return await self._call(
            system=self._build_system(system_prompt, cache_system_prompt),
            messages=messages,
            agent_name=agent_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt_text=system_prompt,
            user_prompt_text=user_prompt,
            has_image=True,
        )

    async def _call(
        self,
        system: list[dict],
        messages: list[dict],
        agent_name: str,
        max_tokens: int,
        temperature: float,
        system_prompt_text: str = "",
        user_prompt_text: str = "",
        has_image: bool = False,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                start = time.perf_counter()
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cached = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                content = response.content[0].text if response.content else ""
                cost = _calculate_cost(self.model, input_tokens, output_tokens, cached)

                logger.info(
                    f"LLM call: agent={agent_name}, "
                    f"tokens={input_tokens}/{output_tokens}, "
                    f"cost=${cost:.4f}, latency={elapsed_ms:.0f}ms"
                )

                llm_response = LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    model=self.model,
                    latency_ms=elapsed_ms,
                    cached_input_tokens=cached,
                )

                # LangSmith tracing (fire-and-forget, never blocks)
                self._trace_call(
                    agent_name=agent_name,
                    system_prompt_text=system_prompt_text,
                    user_prompt_text=user_prompt_text,
                    has_image=has_image,
                    response=llm_response,
                    temperature=temperature,
                )

                self._accumulate_usage(llm_response)

                # Fire-and-forget: persist to llm_calls table for analytics.
                if self._llm_call_repo is not None:
                    try:
                        from uuid import uuid4
                        from datetime import datetime, timezone
                        await self._llm_call_repo.insert_call({
                            "id": str(uuid4()),
                            "agent_name": agent_name,
                            "model": self.model,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost_usd": cost,
                            "latency_ms": int(elapsed_ms),
                            "cache_hit": cached > 0,
                        })
                    except Exception:
                        logger.debug("LLM call record insert failed", exc_info=True)

                return llm_response

            except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_SECONDS[attempt]
                    logger.warning(
                        f"LLM retry {attempt + 1}/{_MAX_RETRIES} for {agent_name}: {e}. "
                        f"Waiting {wait}s."
                    )
                    await asyncio.sleep(wait)

        raise last_error  # type: ignore[misc]

    def _trace_call(
        self,
        agent_name: str,
        system_prompt_text: str,
        user_prompt_text: str,
        has_image: bool,
        response: LLMResponse,
        temperature: float,
    ) -> None:
        """Send trace to LangSmith if tracing is enabled. Never raises."""
        if not _TRACING_ENABLED or _traceable is None:
            return

        try:
            from langsmith import Client

            client = Client()
            client.create_run(
                name=agent_name,
                run_type="llm",
                inputs={
                    "system_prompt": system_prompt_text[:500],
                    "user_prompt": user_prompt_text[:500],
                    "has_image": has_image,
                },
                outputs={
                    "content": response.content[:1000],
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost,
                },
                extra={
                    "metadata": {
                        "model": response.model,
                        "temperature": temperature,
                        "cached_input_tokens": response.cached_input_tokens,
                        "latency_ms": response.latency_ms,
                        "engine_version": self._get_engine_version(),
                        "prompt_version": self._get_prompt_version(agent_name),
                    },
                    "tags": ["quantagent", "v2", agent_name],
                },
                # Per-call read of LANGCHAIN_PROJECT — the CLI dispatcher
                # in `quantagent/main.py` sets this env var to
                # `quantagent-shadow` / `quantagent-paper` / leaves the
                # operator's `.env` value (typically `quantagent-live`)
                # depending on which mode flag was passed. Reading per
                # call instead of caching at module load means a future
                # mixed-mode process that wants per-bot routing can
                # mutate the env var around individual LLM calls without
                # restarting. Default fallback is `quantagent-live` so
                # an operator who runs the bare `python -m quantagent run`
                # without setting LANGCHAIN_PROJECT in `.env` lands in
                # the live project, NOT in some legacy `quantagent-v2`
                # bucket — matches the convention the CLI dispatcher
                # set up for the other two modes.
                project_name=os.environ.get("LANGCHAIN_PROJECT", "quantagent-live"),
                end_time=None,  # auto-completed
            )
        except Exception as e:
            logger.debug(f"LangSmith trace failed for {agent_name}: {e}")

    @staticmethod
    def _get_engine_version() -> str:
        try:
            from quantagent.version import ENGINE_VERSION
            return ENGINE_VERSION
        except ImportError:
            return "unknown"

    @staticmethod
    def _get_prompt_version(agent_name: str) -> str:
        try:
            from quantagent.version import PROMPT_VERSIONS
            return PROMPT_VERSIONS.get(agent_name, "unknown")
        except ImportError:
            return "unknown"
