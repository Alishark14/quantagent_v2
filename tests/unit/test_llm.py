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


# ---------------------------------------------------------------------------
# LangSmith project routing — Paper Trading Task 4
# ---------------------------------------------------------------------------


class TestLangSmithProjectRouting:
    """Pin the contract that ``ClaudeProvider._trace_call`` reads
    ``LANGCHAIN_PROJECT`` PER CALL from the process environment.

    This is the load-bearing property that lets the CLI dispatcher in
    ``quantagent/main.py`` route shadow / paper traces to separate
    LangSmith projects without any per-call API surface change. If
    ``_trace_call`` cached the project name at module load (or at
    ``ClaudeProvider.__init__`` time), the CLI flag would have no
    effect on traces and shadow / paper observability would silently
    leak into the live dashboard.

    These tests force-enable tracing so the trace path actually
    executes, then mock ``langsmith.Client`` so we can read back the
    ``project_name`` kwarg that ``create_run`` was called with —
    without making a real network request.
    """

    @pytest.fixture(autouse=True)
    def restore_langchain_project_env(self):
        """Snapshot LANGCHAIN_PROJECT, clear at start, restore after.

        Two-phase isolation so a leaked value from another test file
        cannot poison our snapshot. Mirrors the autouse fixture in
        ``tests/unit/test_main_cli.py`` for the same reason.
        """
        import os
        saved = os.environ.get("LANGCHAIN_PROJECT")
        os.environ.pop("LANGCHAIN_PROJECT", None)
        yield
        if saved is None:
            os.environ.pop("LANGCHAIN_PROJECT", None)
        else:
            os.environ["LANGCHAIN_PROJECT"] = saved

    def _make_traced_provider_with_capture(self):
        """Build a ClaudeProvider whose _trace_call will execute.

        Returns ``(provider, captured_kwargs_list)`` where the list
        accumulates every kwarg dict that ``client.create_run`` was
        called with. Mocks `langsmith.Client` so no network goes out.
        """
        provider = ClaudeProvider(api_key="test-key")
        captured: list[dict] = []

        # Force-enable the tracing path. The module-level
        # `_TRACING_ENABLED` is set at IMPORT time from
        # `LANGCHAIN_TRACING_V2`, so flipping the env var here would
        # be too late — patch the module-level globals directly.
        from llm import claude as claude_module

        # `_traceable` only needs to be truthy for the early-return
        # guard at the top of `_trace_call` (`if not _TRACING_ENABLED
        # or _traceable is None: return`). The actual trace API uses
        # `Client.create_run`, which is what we mock.
        mock_client_class = MagicMock()
        mock_client_instance = MagicMock()

        def _capture_create_run(**kwargs):
            captured.append(kwargs)

        mock_client_instance.create_run = MagicMock(side_effect=_capture_create_run)
        mock_client_class.return_value = mock_client_instance

        # `_trace_call` does `from langsmith import Client` lazily
        # inside its try block, so we patch the import target. The
        # patch context lives as long as the returned object — pass
        # back the patcher so the caller can stop it after the call.
        return provider, captured, mock_client_class

    def test_trace_call_reads_langchain_project_from_env(self):
        """When LANGCHAIN_PROJECT=quantagent-shadow is set in env,
        the create_run call must receive project_name='quantagent-shadow'."""
        import os
        from llm import claude as claude_module

        provider, captured, mock_client_class = (
            self._make_traced_provider_with_capture()
        )
        os.environ["LANGCHAIN_PROJECT"] = "quantagent-shadow"

        with patch.object(claude_module, "_TRACING_ENABLED", True), \
             patch.object(claude_module, "_traceable", lambda f: f), \
             patch("langsmith.Client", mock_client_class):
            response = LLMResponse(
                content="x", input_tokens=1, output_tokens=1, cost=0.0,
                model="claude-sonnet-4-20250514", latency_ms=10.0,
                cached_input_tokens=0,
            )
            provider._trace_call(
                agent_name="indicator_agent",
                system_prompt_text="sys",
                user_prompt_text="usr",
                has_image=False,
                response=response,
                temperature=0.3,
            )

        assert len(captured) == 1
        assert captured[0]["project_name"] == "quantagent-shadow"

    def test_trace_call_reads_paper_project_when_paper_env_set(self):
        """Symmetric coverage for paper mode — different env value,
        different captured project_name. Proves the read happens
        per-call, not from a cached value at provider construction."""
        import os
        from llm import claude as claude_module

        provider, captured, mock_client_class = (
            self._make_traced_provider_with_capture()
        )
        os.environ["LANGCHAIN_PROJECT"] = "quantagent-paper"

        with patch.object(claude_module, "_TRACING_ENABLED", True), \
             patch.object(claude_module, "_traceable", lambda f: f), \
             patch("langsmith.Client", mock_client_class):
            response = LLMResponse(
                content="x", input_tokens=1, output_tokens=1, cost=0.0,
                model="claude-sonnet-4-20250514", latency_ms=10.0,
                cached_input_tokens=0,
            )
            provider._trace_call(
                agent_name="conviction_agent",
                system_prompt_text="sys",
                user_prompt_text="usr",
                has_image=False,
                response=response,
                temperature=0.2,
            )

        assert captured[0]["project_name"] == "quantagent-paper"

    def test_trace_call_default_fallback_is_quantagent_live(self):
        """When LANGCHAIN_PROJECT is NOT set in the env, the default
        fallback should be `quantagent-live` (NOT the legacy
        `quantagent-v2`). This is the bare `python -m quantagent run`
        case where the operator never set LANGCHAIN_PROJECT in `.env`
        — they should land in the live project, not in some legacy
        bucket."""
        import os
        from llm import claude as claude_module

        provider, captured, mock_client_class = (
            self._make_traced_provider_with_capture()
        )
        # Make sure no env var is set — the autouse fixture cleared it
        # at start, but defense in depth.
        assert "LANGCHAIN_PROJECT" not in os.environ

        with patch.object(claude_module, "_TRACING_ENABLED", True), \
             patch.object(claude_module, "_traceable", lambda f: f), \
             patch("langsmith.Client", mock_client_class):
            response = LLMResponse(
                content="x", input_tokens=1, output_tokens=1, cost=0.0,
                model="claude-sonnet-4-20250514", latency_ms=10.0,
                cached_input_tokens=0,
            )
            provider._trace_call(
                agent_name="decision_agent",
                system_prompt_text="sys",
                user_prompt_text="usr",
                has_image=False,
                response=response,
                temperature=0.2,
            )

        assert captured[0]["project_name"] == "quantagent-live"
        # And specifically NOT the stale legacy default
        assert captured[0]["project_name"] != "quantagent-v2"

    def test_trace_call_reads_env_per_call_not_cached(self):
        """The same provider instance should route to different
        projects across two consecutive _trace_call invocations if
        LANGCHAIN_PROJECT is mutated in between. Proves the env read
        is genuinely per-call — load-bearing for any future mixed-mode
        process that wants to swap projects between LLM calls."""
        import os
        from llm import claude as claude_module

        provider, captured, mock_client_class = (
            self._make_traced_provider_with_capture()
        )

        with patch.object(claude_module, "_TRACING_ENABLED", True), \
             patch.object(claude_module, "_traceable", lambda f: f), \
             patch("langsmith.Client", mock_client_class):
            response = LLMResponse(
                content="x", input_tokens=1, output_tokens=1, cost=0.0,
                model="claude-sonnet-4-20250514", latency_ms=10.0,
                cached_input_tokens=0,
            )

            # Call 1: shadow project
            os.environ["LANGCHAIN_PROJECT"] = "quantagent-shadow"
            provider._trace_call(
                agent_name="indicator_agent",
                system_prompt_text="sys", user_prompt_text="usr",
                has_image=False, response=response, temperature=0.3,
            )

            # Call 2: paper project (same provider instance)
            os.environ["LANGCHAIN_PROJECT"] = "quantagent-paper"
            provider._trace_call(
                agent_name="indicator_agent",
                system_prompt_text="sys", user_prompt_text="usr",
                has_image=False, response=response, temperature=0.3,
            )

        assert len(captured) == 2
        assert captured[0]["project_name"] == "quantagent-shadow"
        assert captured[1]["project_name"] == "quantagent-paper"

    def test_trace_call_silent_when_tracing_disabled(self):
        """If `_TRACING_ENABLED` is False (the default — no
        LANGCHAIN_TRACING_V2 env var at module load), `_trace_call`
        must early-return without ever touching langsmith.Client.
        Pins the no-op contract so production deployments without
        a LangSmith API key keep working."""
        from llm import claude as claude_module

        provider, captured, mock_client_class = (
            self._make_traced_provider_with_capture()
        )

        with patch.object(claude_module, "_TRACING_ENABLED", False), \
             patch.object(claude_module, "_traceable", None), \
             patch("langsmith.Client", mock_client_class):
            response = LLMResponse(
                content="x", input_tokens=1, output_tokens=1, cost=0.0,
                model="claude-sonnet-4-20250514", latency_ms=10.0,
                cached_input_tokens=0,
            )
            provider._trace_call(
                agent_name="indicator_agent",
                system_prompt_text="sys", user_prompt_text="usr",
                has_image=False, response=response, temperature=0.3,
            )

        # Client.create_run was never called
        assert captured == []
        mock_client_class.assert_not_called()


class TestShadowFlagSetsLangSmithProject:
    """Symmetric to the paper-flag tests in test_main_cli.py — verify
    that the `--shadow` CLI dispatcher branch ALSO sets
    `LANGCHAIN_PROJECT=quantagent-shadow` so shadow traces don't leak
    into the live observability dashboard.

    Lives in test_llm.py (not test_main_cli.py) because it's the
    LangSmith routing contract that this task adds, not a general
    CLI flag test. The CLI dispatch mechanics are already covered
    in test_main_cli.py.
    """

    @pytest.fixture(autouse=True)
    def restore_env(self):
        import os
        keys = ("QUANTAGENT_SHADOW", "LANGCHAIN_PROJECT")
        saved = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)
        yield
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_shadow_flag_sets_langchain_project_quantagent_shadow(self):
        """Pin the parity: paper sets LANGCHAIN_PROJECT=quantagent-paper,
        so shadow MUST set LANGCHAIN_PROJECT=quantagent-shadow.
        Otherwise shadow LLM traces inherit the operator's `.env`
        value (typically `quantagent-live`) and silently leak into
        the live dashboard."""
        from quantagent.main import main

        with patch("sys.argv", ["quantagent.main", "run", "--shadow"]), \
             patch("quantagent.main.run"), \
             patch("quantagent.main.migrate"), \
             patch("quantagent.main.seed"):
            main()

        import os
        assert os.environ.get("QUANTAGENT_SHADOW") == "1"
        assert os.environ.get("LANGCHAIN_PROJECT") == "quantagent-shadow"
