"""Tests for LLM base classes and MockLLM behavior."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent_harness.core.message import Message
from agent_harness.llm.types import FinishReason

from tests.conftest import MockLLM


class TestMockLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_default_response(self) -> None:
        """MockLLM with no queued responses returns the default."""
        llm = MockLLM()
        messages = [Message.user("hello")]
        response = await llm.generate(messages)

        assert response.message.content == "Default mock response"
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_returns_enqueued_text(self) -> None:
        """add_text_response queues a response that generate() returns."""
        llm = MockLLM()
        llm.add_text_response("custom reply")

        response = await llm.generate([Message.user("hi")])

        assert response.message.content == "custom reply"
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_returns_tool_call(self) -> None:
        """add_tool_call_response queues a tool-call response."""
        llm = MockLLM()
        llm.add_tool_call_response("search", {"query": "test"})

        response = await llm.generate([Message.user("search for test")])

        assert response.has_tool_calls
        assert response.finish_reason == FinishReason.TOOL_CALLS
        assert response.message.tool_calls is not None
        assert response.message.tool_calls[0].name == "search"
        assert response.message.tool_calls[0].arguments == {"query": "test"}


class TestMockLLMCallHistory:
    @pytest.mark.asyncio
    async def test_call_history_is_tracked(self) -> None:
        """Each generate() call is recorded in call_history."""
        llm = MockLLM()
        msg1 = [Message.user("first")]
        msg2 = [Message.user("second")]

        await llm.generate(msg1)
        await llm.generate(msg2)

        assert len(llm.call_history) == 2
        assert llm.call_history[0][0].content == "first"
        assert llm.call_history[1][0].content == "second"

    @pytest.mark.asyncio
    async def test_empty_call_history_initially(self) -> None:
        """Fresh MockLLM has no call history."""
        llm = MockLLM()
        assert llm.call_history == []


class TestMockLLMMultipleResponses:
    @pytest.mark.asyncio
    async def test_responses_consumed_in_order(self) -> None:
        """Multiple enqueued responses are returned in FIFO order."""
        llm = MockLLM()
        llm.add_text_response("first")
        llm.add_text_response("second")
        llm.add_text_response("third")

        r1 = await llm.generate([Message.user("a")])
        r2 = await llm.generate([Message.user("b")])
        r3 = await llm.generate([Message.user("c")])

        assert r1.message.content == "first"
        assert r2.message.content == "second"
        assert r3.message.content == "third"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_after_queue_exhausted(self) -> None:
        """After all queued responses are consumed, returns default."""
        llm = MockLLM()
        llm.add_text_response("only one")

        await llm.generate([Message.user("a")])
        fallback = await llm.generate([Message.user("b")])

        assert fallback.message.content == "Default mock response"


class TestWithRetryTransientErrors:
    """_with_retry should handle ConnectionError and TimeoutError in addition to LLMRateLimitError."""

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        """_with_retry retries on ConnectionError then succeeds."""
        llm = MockLLM()
        call_count = 0

        async def flaky_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("connection reset")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await llm._with_retry(flaky_call)

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self) -> None:
        """_with_retry retries on TimeoutError then succeeds."""
        llm = MockLLM()
        call_count = 0

        async def flaky_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timed out")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await llm._with_retry(flaky_call)

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_non_transient_error(self) -> None:
        """_with_retry does not retry on ValueError."""
        llm = MockLLM()

        async def bad_call() -> str:
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await llm._with_retry(bad_call)


class TestBaseLLMClose:
    @pytest.mark.asyncio
    async def test_aclose_closes_sdk_client(self) -> None:
        """aclose() awaits the provider's SDK client close()."""
        from types import SimpleNamespace

        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
        closed = AsyncMock()
        provider._client = SimpleNamespace(close=closed)

        await provider.aclose()

        closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_swallows_close_errors(self) -> None:
        """A failing client close() must not propagate out of aclose()."""
        from types import SimpleNamespace

        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(LLMConfig(provider="anthropic", model="claude-3"))
        provider._client = SimpleNamespace(
            close=AsyncMock(side_effect=RuntimeError("loop closing")),
        )

        await provider.aclose()  # no raise

    @pytest.mark.asyncio
    async def test_aclose_noop_without_client(self) -> None:
        """aclose() on a provider without a closable client is a no-op."""
        llm = MockLLM()
        await llm.aclose()  # MockLLM has no _client; must not raise

    @pytest.mark.asyncio
    async def test_async_context_manager_closes(self) -> None:
        """`async with provider` closes the client on exit."""
        from types import SimpleNamespace

        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
        closed = AsyncMock()
        provider._client = SimpleNamespace(close=closed)

        async with provider as p:
            assert p is provider
        closed.assert_awaited_once()
