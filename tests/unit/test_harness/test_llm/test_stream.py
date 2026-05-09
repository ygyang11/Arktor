"""Tests for streaming tool call accumulation in LLM providers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_harness.core.message import Message
from agent_harness.llm.types import FinishReason, StreamDelta, Usage


def _make_openai_chunk(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_openai_usage_chunk(usage: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=usage)


def _make_usage(
    *,
    prompt: int = 12,
    completion: int = 3,
    total: int = 15,
    cached: int = 4,
    reasoning: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


class TestOpenAIStream:

    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        chunks = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(content=" world"),
            _make_openai_chunk(finish_reason="stop"),
        ]

        provider = _make_provider(chunks)

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        texts = [d.chunk.delta_content for d in deltas if d.chunk.delta_content]
        assert texts == ["Hello", " world"]
        assert deltas[-1].finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_usage_on_official_empty_choices_chunk_is_preserved(self) -> None:
        chunks = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(finish_reason="stop"),
            _make_openai_usage_chunk(_make_usage()),
        ]

        provider = _make_provider(chunks)

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        assert _usages(deltas) == [
            Usage(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                cache_read_tokens=4,
                reasoning_tokens=1,
            )
        ]

    @pytest.mark.asyncio
    async def test_usage_on_choice_chunk_is_preserved(self) -> None:
        chunks = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(finish_reason="stop", usage=_make_usage()),
        ]

        provider = _make_provider(chunks)

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        assert _usages(deltas) == [
            Usage(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                cache_read_tokens=4,
                reasoning_tokens=1,
            )
        ]

    @pytest.mark.asyncio
    async def test_multiple_usage_chunks_keep_last_snapshot(self) -> None:
        chunks = [
            _make_openai_chunk(
                content="Hello",
                usage=_make_usage(prompt=10, completion=2, total=12),
            ),
            _make_openai_chunk(finish_reason="stop"),
            _make_openai_usage_chunk(
                _make_usage(prompt=20, completion=5, total=25, cached=6, reasoning=2)
            ),
        ]

        provider = _make_provider(chunks)

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        assert _usages(deltas) == [
            Usage(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=25,
                cache_read_tokens=6,
                reasoning_tokens=2,
            )
        ]

    @pytest.mark.asyncio
    async def test_stream_response_usage_does_not_sum_total_snapshots(self) -> None:
        chunks = [
            _make_openai_chunk(
                content="Hello",
                usage=_make_usage(prompt=10, completion=2, total=12),
            ),
            _make_openai_chunk(content=" world", finish_reason="stop"),
            _make_openai_usage_chunk(
                _make_usage(prompt=20, completion=5, total=25, cached=6, reasoning=2)
            ),
        ]

        provider = _make_provider(chunks)

        response = await provider.stream_with_events([Message.user("hi")])

        assert response.message.content == "Hello world"
        assert response.usage == Usage(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
            cache_read_tokens=6,
            reasoning_tokens=2,
        )

    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self) -> None:
        tc_chunk_1 = SimpleNamespace(
            index=0,
            id="call_abc",
            function=SimpleNamespace(name="search", arguments='{"q":'),
        )
        tc_chunk_2 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='"hello"}'),
        )

        chunks = [
            _make_openai_chunk(tool_calls=[tc_chunk_1]),
            _make_openai_chunk(tool_calls=[tc_chunk_2]),
            _make_openai_chunk(finish_reason="tool_calls"),
        ]

        provider = _make_provider(chunks)

        deltas: list[StreamDelta] = []
        async for d in provider.stream([Message.user("hi")]):
            deltas.append(d)

        final = deltas[-1]
        assert final.finish_reason == FinishReason.TOOL_CALLS
        assert final.chunk.delta_tool_calls is not None
        assert len(final.chunk.delta_tool_calls) == 1
        tc = final.chunk.delta_tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}


class _AsyncIter:
    """Helper to simulate an async iterator from a list."""
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _make_provider(chunks: list[SimpleNamespace]) -> object:
    from agent_harness.llm.openai_provider import OpenAIProvider

    async def fake_create(**kwargs: object) -> _AsyncIter:
        return _AsyncIter(chunks)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = SimpleNamespace(
        model="test", temperature=0.7, max_tokens=100,
        reasoning_effort=None, max_retries=0, retry_delay=1.0,
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )
    provider._rate_limiter = None
    return provider


def _usages(deltas: list[StreamDelta]) -> list[Usage]:
    return [d.usage for d in deltas if d.usage is not None]
