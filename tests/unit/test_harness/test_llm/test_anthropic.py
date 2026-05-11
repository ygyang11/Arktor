"""Tests for AnthropicProvider thinking-block round-trip."""
from __future__ import annotations

from types import SimpleNamespace

from agent_harness.core.message import Message, Role, ToolCall


class TestThinkingBlockRoundTrip:

    def test_parse_response_captures_thinking_block(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

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
        result = AnthropicProvider._parse_response(response)

        assert result.message.content == "Hello"
        assert result.message.provider_metadata == {
            "anthropic": {
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "reasoning...", "signature": "sig123"},
                ],
            },
        }

    def test_split_system_message_replays_thinking_blocks_first(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

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
        _, api_msgs = AnthropicProvider._split_system_message([msg])
        assert len(api_msgs) == 1
        blocks = api_msgs[0]["content"]
        assert blocks[0]["type"] == "thinking"
        assert blocks[1]["type"] == "text"
        assert blocks[2]["type"] == "tool_use"

    def test_multiple_consecutive_thinking_blocks_preserve_order(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

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
        result = AnthropicProvider._parse_response(response)
        blocks = result.message.provider_metadata["anthropic"]["thinking_blocks"]
        assert blocks[0]["thinking"] == "step A"
        assert blocks[1]["thinking"] == "step B"

    def test_omitted_mode_empty_thinking_preserved(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

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
        result = AnthropicProvider._parse_response(response)
        blocks = result.message.provider_metadata["anthropic"]["thinking_blocks"]
        assert blocks == [{"type": "thinking", "thinking": "", "signature": "sig-encrypted"}]

    def test_redacted_thinking_block_round_trips(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

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
        result = AnthropicProvider._parse_response(response)
        blocks = result.message.provider_metadata["anthropic"]["thinking_blocks"]
        assert blocks == [{"type": "redacted_thinking", "data": "enc-blob"}]

    def test_no_thinking_blocks_means_no_sidecar(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

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
        result = AnthropicProvider._parse_response(response)
        assert result.message.provider_metadata == {}

    def test_split_system_ignores_foreign_namespace(self) -> None:
        from agent_harness.llm.anthropic_provider import AnthropicProvider

        msg = Message(
            role=Role.ASSISTANT,
            content="hi",
            provider_metadata={"openai_chat": {"reasoning_content": "ignored"}},
        )
        _, api_msgs = AnthropicProvider._split_system_message([msg])
        blocks = api_msgs[0]["content"]
        for block in blocks:
            assert block["type"] not in ("thinking", "redacted_thinking")
