"""Tests for streaming tool call accumulation in LLM providers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_harness.core.message import Message, Role, ToolCall
from agent_harness.llm.types import FinishReason, StreamDelta, Usage


def _make_openai_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[dict[str, object]] | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
        tool_calls=tool_calls,
    )
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


class TestProviderMetadataRoundTrip:

    @pytest.mark.asyncio
    async def test_streamed_reasoning_content_accumulates(self) -> None:
        chunks = [
            _make_openai_chunk(reasoning_content="Let me "),
            _make_openai_chunk(reasoning_content="think..."),
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(finish_reason="stop"),
        ]
        provider = _make_provider(chunks)
        response = await provider.stream_with_events([Message.user("hi")])

        assert response.message.content == "Hello"
        assert response.message.provider_metadata == {
            "openai_chat": {"reasoning_content": "Let me think..."},
        }

    @pytest.mark.asyncio
    async def test_streamed_reasoning_details_last_write_wins(self) -> None:
        chunks = [
            _make_openai_chunk(reasoning_details=[{"text": "step 1"}]),
            _make_openai_chunk(reasoning_details=[{"text": "step 1"}, {"text": "step 2"}]),
            _make_openai_chunk(content="ok"),
            _make_openai_chunk(finish_reason="stop"),
        ]
        provider = _make_provider(chunks)
        response = await provider.stream_with_events([Message.user("hi")])

        assert response.message.provider_metadata == {
            "openai_chat": {"reasoning_details": [{"text": "step 1"}, {"text": "step 2"}]},
        }

    def test_format_message_replays_captured_reasoning_content(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        msg = Message(
            role=Role.ASSISTANT,
            content="answer",
            tool_calls=[ToolCall(id="x", name="Glob", arguments={})],
            provider_metadata={"openai_chat": {"reasoning_content": "thinking..."}},
        )
        wire = OpenAIProvider._format_message(msg)
        assert wire["reasoning_content"] == "thinking..."
        assert "reasoning_details" not in wire

    def test_format_message_replays_captured_reasoning_details(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        msg = Message(
            role=Role.ASSISTANT,
            content="answer",
            provider_metadata={"openai_chat": {"reasoning_details": [{"text": "x"}]}},
        )
        wire = OpenAIProvider._format_message(msg)
        assert wire["reasoning_details"] == [{"text": "x"}]
        assert "reasoning_content" not in wire

    def test_format_message_preserves_empty_string(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        msg = Message(
            role=Role.ASSISTANT,
            content="answer",
            provider_metadata={"openai_chat": {"reasoning_content": ""}},
        )
        wire = OpenAIProvider._format_message(msg)
        assert wire["reasoning_content"] == ""

    def test_format_message_omits_field_when_no_sidecar(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        msg = Message(role=Role.ASSISTANT, content="hi")
        wire = OpenAIProvider._format_message(msg)
        assert "reasoning_content" not in wire
        assert "reasoning_details" not in wire

    def test_format_message_ignores_foreign_namespace(self) -> None:
        from agent_harness.llm.openai_provider import OpenAIProvider

        msg = Message(
            role=Role.ASSISTANT,
            content="hi",
            provider_metadata={"anthropic": {"thinking_blocks": [{"type": "thinking", "thinking": "x", "signature": "s"}]}},
        )
        wire = OpenAIProvider._format_message(msg)
        assert "thinking_blocks" not in wire
        assert "reasoning_content" not in wire


class TestSyntheticTurnSidecar:
    """`synthetic_turn_sidecar` is the placeholder provider_metadata stamped
    on harness-synthesized assistant turns (e.g. `@file` mention expansion).
    Without it, reasoning-capable backends 400 on the next request."""

    def test_base_default_is_empty(self) -> None:
        # BaseLLM is abstract — instantiate via OpenAI and force-call BaseLLM's
        # implementation to check the default.
        from agent_harness.llm.base import BaseLLM

        assert BaseLLM.synthetic_turn_sidecar.__qualname__.startswith("BaseLLM.")
        # Call the unbound method with a stub `self` — confirms default = {}
        assert BaseLLM.synthetic_turn_sidecar(object()) == {}  # type: ignore[arg-type]

    def test_openai_stamps_both_reasoning_fields(self) -> None:
        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
        sidecar = provider.synthetic_turn_sidecar()
        assert sidecar == {
            "openai_chat": {"reasoning_content": "", "reasoning_details": []},
        }

    def test_openai_stamps_unconditionally(self) -> None:
        # Reasoning-capable models (deepseek-reasoner, OpenRouter thinking)
        # require the field even when reasoning_effort is not configured.
        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.openai_provider import OpenAIProvider

        no_effort = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
        with_effort = OpenAIProvider(LLMConfig(
            provider="openai", model="gpt-4o", reasoning_effort="medium",
        ))
        assert no_effort.synthetic_turn_sidecar() == with_effort.synthetic_turn_sidecar()

    def test_openai_returns_fresh_dict_per_call(self) -> None:
        # Callers may stamp the dict onto a Message and provider format
        # paths may mutate sub-dicts — must not alias the class-level constant.
        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))
        a = provider.synthetic_turn_sidecar()
        b = provider.synthetic_turn_sidecar()
        assert a == b
        assert a is not b
        assert a["openai_chat"] is not b["openai_chat"]

    def test_anthropic_inherits_empty_default(self) -> None:
        # Anthropic doesn't override — extended thinking + synthesized turn
        # has no fabricable placeholder (signatures), so we accept the
        # documented limitation and inherit base.
        from agent_harness.core.config import LLMConfig
        from agent_harness.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(LLMConfig(
            provider="anthropic", model="claude-opus-4-5", api_key="dummy",
        ))
        assert provider.synthetic_turn_sidecar() == {}
