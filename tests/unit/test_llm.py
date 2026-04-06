"""Unit tests for LLM provider abstraction layer."""

from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import anthropic
from llm.base import LLMProvider, LLMResponse
from llm.claude import ClaudeProvider, _calculate_cost, MODEL_COSTS
from llm.cache import PromptCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    text: str = "test response",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
) -> SimpleNamespace:
    """Build a fake anthropic message response."""
    content_block = SimpleNamespace(text=text)
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
    )
    return SimpleNamespace(content=[content_block], usage=usage)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_construction(self) -> None:
        r = LLMResponse(
            content="hello",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            model="claude-sonnet-4-20250514",
            latency_ms=250.0,
            cached_input_tokens=0,
        )
        assert r.content == "hello"
        assert r.cost == 0.001


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_sonnet_no_cache(self) -> None:
        cost = _calculate_cost("claude-sonnet-4-20250514", 1000, 500, 0)
        expected = (1000 / 1e6) * 3.0 + (500 / 1e6) * 15.0
        assert cost == pytest.approx(expected)

    def test_sonnet_with_cache(self) -> None:
        cost = _calculate_cost("claude-sonnet-4-20250514", 1000, 500, 800)
        # 200 billable input + 800 cached + 500 output
        expected = (200 / 1e6) * 3.0 + (500 / 1e6) * 15.0 + (800 / 1e6) * 0.30
        assert cost == pytest.approx(expected)

    def test_haiku(self) -> None:
        cost = _calculate_cost("claude-haiku-4-5-20251001", 1000, 500, 0)
        expected = (1000 / 1e6) * 0.80 + (500 / 1e6) * 4.0
        assert cost == pytest.approx(expected)

    def test_unknown_model_uses_sonnet_rates(self) -> None:
        cost = _calculate_cost("some-future-model", 1000, 500, 0)
        expected = (1000 / 1e6) * 3.0 + (500 / 1e6) * 15.0
        assert cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ClaudeProvider — generate_text
# ---------------------------------------------------------------------------


class TestGenerateText:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(
            return_value=_mock_response("result text", 200, 80, 50)
        )

        resp = await provider.generate_text(
            system_prompt="You are a trader.",
            user_prompt="Analyze BTC.",
            agent_name="indicator_agent",
        )

        assert isinstance(resp, LLMResponse)
        assert resp.content == "result text"
        assert resp.input_tokens == 200
        assert resp.output_tokens == 80
        assert resp.cached_input_tokens == 50
        assert resp.model == "claude-sonnet-4-20250514"
        assert resp.latency_ms > 0
        assert resp.cost > 0

    @pytest.mark.asyncio
    async def test_cache_control_set_when_enabled(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(
            return_value=_mock_response()
        )

        await provider.generate_text(
            system_prompt="system",
            user_prompt="user",
            agent_name="test",
            cache_system_prompt=True,
        )

        call_kwargs = provider.client.messages.create.call_args.kwargs
        system_block = call_kwargs["system"][0]
        assert system_block["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_cache_control_not_set_when_disabled(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(
            return_value=_mock_response()
        )

        await provider.generate_text(
            system_prompt="system",
            user_prompt="user",
            agent_name="test",
            cache_system_prompt=False,
        )

        call_kwargs = provider.client.messages.create.call_args.kwargs
        system_block = call_kwargs["system"][0]
        assert "cache_control" not in system_block


# ---------------------------------------------------------------------------
# ClaudeProvider — generate_vision
# ---------------------------------------------------------------------------


class TestGenerateVision:
    @pytest.mark.asyncio
    async def test_image_content_block_included(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()
        provider.client.messages.create = AsyncMock(
            return_value=_mock_response("pattern found", 300, 100, 0)
        )

        image_bytes = b"\x89PNG\r\n\x1a\nfake_image_data"
        resp = await provider.generate_vision(
            system_prompt="Analyze chart",
            user_prompt="What patterns?",
            image_data=image_bytes,
            image_media_type="image/png",
            agent_name="pattern_agent",
        )

        assert resp.content == "pattern found"

        call_kwargs = provider.client.messages.create.call_args.kwargs
        msg = call_kwargs["messages"][0]
        assert msg["role"] == "user"

        # Content should have image block + text block
        content_blocks = msg["content"]
        assert len(content_blocks) == 2

        image_block = content_blocks[0]
        assert image_block["type"] == "image"
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"
        expected_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        assert image_block["source"]["data"] == expected_b64

        text_block = content_blocks[1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "What patterns?"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_api_error(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()

        # First call raises, second succeeds
        provider.client.messages.create = AsyncMock(
            side_effect=[
                anthropic.APIConnectionError(request=MagicMock()),
                _mock_response("success after retry"),
            ]
        )

        with patch("llm.claude.asyncio.sleep", new_callable=AsyncMock):
            resp = await provider.generate_text(
                system_prompt="sys",
                user_prompt="usr",
                agent_name="test",
            )

        assert resp.content == "success after retry"
        assert provider.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()

        provider.client.messages.create = AsyncMock(
            side_effect=anthropic.APIConnectionError(request=MagicMock())
        )

        with patch("llm.claude.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(anthropic.APIConnectionError):
                await provider.generate_text(
                    system_prompt="sys",
                    user_prompt="usr",
                    agent_name="test",
                )

        assert provider.client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self) -> None:
        provider = ClaudeProvider(api_key="test-key")
        provider.client = MagicMock()
        provider.client.messages = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}

        provider.client.messages.create = AsyncMock(
            side_effect=[
                anthropic.RateLimitError(
                    message="rate limited",
                    response=mock_resp,
                    body=None,
                ),
                _mock_response("ok after rate limit"),
            ]
        )

        with patch("llm.claude.asyncio.sleep", new_callable=AsyncMock):
            resp = await provider.generate_text(
                system_prompt="sys",
                user_prompt="usr",
                agent_name="test",
            )

        assert resp.content == "ok after rate limit"


# ---------------------------------------------------------------------------
# PromptCache
# ---------------------------------------------------------------------------


class TestPromptCache:
    def test_mark_and_check_warm(self) -> None:
        cache = PromptCache()
        cache.mark_warm("abc123")
        assert cache.is_warm("abc123") is True

    def test_unknown_hash_is_cold(self) -> None:
        cache = PromptCache()
        assert cache.is_warm("nonexistent") is False

    def test_expired_entry_is_cold(self) -> None:
        cache = PromptCache(ttl_seconds=300)
        cache.mark_warm("abc123")
        # Manually backdate the timestamp
        cache._warm_prompts["abc123"] = datetime.now(timezone.utc) - timedelta(seconds=301)
        assert cache.is_warm("abc123") is False
        # Entry should be cleaned up
        assert "abc123" not in cache._warm_prompts

    def test_clear(self) -> None:
        cache = PromptCache()
        cache.mark_warm("a")
        cache.mark_warm("b")
        cache.clear()
        assert cache.is_warm("a") is False
        assert cache.is_warm("b") is False
