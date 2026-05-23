"""Tests for AnthropicProvider thinking-block round-trip."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness.core.message import Message, Role, ToolCall
from agent_harness.llm.types import Usage


class _AsyncStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self._index = 0

    async def __aenter__(self) -> _AsyncStream:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def __aiter__(self) -> _AsyncStream:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


def _make_provider() -> Any:
    from agent_harness.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.config = SimpleNamespace(
        model="mimo-v2.5-pro",
        temperature=0.7,
        max_tokens=100,
        max_retries=0,
        retry_delay=1.0,
        reasoning_effort=None,
    )
    provider._rate_limiter = None
    return provider


class TestThinkingBlockRoundTrip:

    def test_parse_response_captures_thinking_block(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="reasoning...", signature="sig123"),
                SimpleNamespace(type="text", text="Hello"),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-6",
        )
        result = _make_provider()._parse_response(response)

        assert result.message.content == "Hello"
        assert result.message.provider_metadata == {
            "anthropic": {
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "reasoning...", "signature": "sig123"},
                ],
            },
        }

    def test_split_system_message_replays_thinking_blocks_first(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content="Answer",
            tool_calls=[ToolCall(id="t1", name="Glob", arguments={"q": "a"})],
            provider_metadata={
                "anthropic": {
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "...", "signature": "sig"},
                    ],
                },
            },
        )
        _, api_msgs = _make_provider()._split_system_message([msg])
        assert len(api_msgs) == 1
        blocks = api_msgs[0]["content"]
        assert blocks[0]["type"] == "thinking"
        assert blocks[1]["type"] == "text"
        assert blocks[2]["type"] == "tool_use"

    def test_multiple_consecutive_thinking_blocks_preserve_order(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="step A", signature="s1"),
                SimpleNamespace(type="thinking", thinking="step B", signature="s2"),
                SimpleNamespace(type="text", text="done"),
            ],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-6",
        )
        result = _make_provider()._parse_response(response)
        blocks = result.message.provider_metadata["anthropic"]["thinking_blocks"]
        assert blocks[0]["thinking"] == "step A"
        assert blocks[1]["thinking"] == "step B"

    def test_omitted_mode_empty_thinking_preserved(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="", signature="sig-encrypted"),
                SimpleNamespace(type="text", text="answer"),
            ],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
            model="claude-opus-4-7",
        )
        result = _make_provider()._parse_response(response)
        blocks = result.message.provider_metadata["anthropic"]["thinking_blocks"]
        assert blocks == [{"type": "thinking", "thinking": "", "signature": "sig-encrypted"}]

    def test_redacted_thinking_block_round_trips(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="redacted_thinking", data="enc-blob"),
                SimpleNamespace(type="text", text="ok"),
            ],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
            model="claude-sonnet-3-7",
        )
        result = _make_provider()._parse_response(response)
        blocks = result.message.provider_metadata["anthropic"]["thinking_blocks"]
        assert blocks == [{"type": "redacted_thinking", "data": "enc-blob"}]

    def test_no_thinking_blocks_means_no_sidecar(self) -> None:
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-6",
        )
        result = _make_provider()._parse_response(response)
        assert result.message.provider_metadata == {}

    def test_split_system_ignores_foreign_namespace(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content="hi",
            provider_metadata={"openai_chat": {"reasoning_content": "ignored"}},
        )
        _, api_msgs = _make_provider()._split_system_message([msg])
        blocks = api_msgs[0]["content"]
        for block in blocks:
            assert block["type"] not in ("thinking", "redacted_thinking")


class TestStreamUsageAggregation:
    """Stream-event → response.usage aggregation under the diff-emit design."""

    @pytest.mark.asyncio
    async def test_native_anthropic_prompt_at_start_output_at_delta(self) -> None:
        # Anthropic-native shape: message_start carries the full input snapshot,
        # message_delta carries only the cumulative output_tokens (input_tokens
        # is Optional and typically absent / 0 in delta usage).
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10000,
                        output_tokens=0,
                        cache_read_input_tokens=500,
                        cache_creation_input_tokens=600,
                    )
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=124),
            ),
        ]

        provider = _make_provider()
        provider._client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: _AsyncStream(events))
        )

        response = await provider.stream_with_events([Message.user("hi")])

        assert response.message.content == "Hello"
        assert response.usage == Usage(
            prompt_tokens=11100,
            completion_tokens=124,
            total_tokens=11224,
            cache_read_tokens=500,
            cache_creation_tokens=600,
        )

    @pytest.mark.asyncio
    async def test_full_usage_delivered_only_at_message_delta(self) -> None:
        # Some Anthropic-compatible providers (e.g. mimo) send no usage at
        # message_start and put the entire snapshot in message_delta. The
        # diff-emit logic must backfill prompt + cache here, not just output.
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=None),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(
                    input_tokens=10600,
                    output_tokens=124,
                    cache_read_input_tokens=500,
                    cache_creation_input_tokens=0,
                ),
            ),
        ]

        provider = _make_provider()
        provider._client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: _AsyncStream(events))
        )

        response = await provider.stream_with_events([Message.user("hi")])

        assert response.message.content == "Hello"
        assert response.usage == Usage(
            prompt_tokens=11100,
            completion_tokens=124,
            total_tokens=11224,
            cache_read_tokens=500,
            cache_creation_tokens=0,
        )

    @pytest.mark.asyncio
    async def test_cumulative_output_across_multiple_message_delta_events(self) -> None:
        # MessageDeltaUsage.output_tokens is documented as cumulative.
        # Multiple events (e.g. with server_tool_use) must not double-count.
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    )
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason=None),
                usage=SimpleNamespace(output_tokens=50),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=120),
            ),
        ]

        provider = _make_provider()
        provider._client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: _AsyncStream(events))
        )

        response = await provider.stream_with_events([Message.user("hi")])

        assert response.usage == Usage(
            prompt_tokens=100,
            completion_tokens=120,
            total_tokens=220,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
