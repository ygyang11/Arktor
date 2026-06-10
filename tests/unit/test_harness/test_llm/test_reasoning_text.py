"""Tests for BaseLLM.reasoning_text — provider sidecar decoding."""
from __future__ import annotations

from agent_harness.core.config import LLMConfig
from agent_harness.core.message import Message, Role
from agent_harness.llm.anthropic_provider import AnthropicProvider
from agent_harness.llm.openai_provider import OpenAIProvider
from tests.conftest import MockLLM


def _msg(provider_metadata: dict | None = None) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="visible text",
        provider_metadata=provider_metadata or {},
    )


def _openai() -> OpenAIProvider:
    return OpenAIProvider(LLMConfig(provider="openai", model="gpt-4o"))


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(LLMConfig(provider="anthropic", model="claude-3"))


class TestBaseDefault:
    def test_default_returns_none(self) -> None:
        msg = _msg({"openai_chat": {"reasoning_content": "thinking hard"}})
        assert MockLLM().reasoning_text(msg) is None


class TestOpenAIReasoningText:
    def test_reasoning_content_preferred(self) -> None:
        msg = _msg({"openai_chat": {
            "reasoning_content": "step by step",
            "reasoning_details": [{"text": "ignored"}],
        }})
        assert _openai().reasoning_text(msg) == "step by step"

    def test_reasoning_details_fallback_joins_text_and_summary(self) -> None:
        msg = _msg({"openai_chat": {"reasoning_details": [
            {"text": "part one. "},
            {"summary": "part two."},
            {"type": "encrypted"},  # no text/summary — skipped
            "not-a-dict",
        ]}})
        assert _openai().reasoning_text(msg) == "part one. part two."

    def test_empty_sidecar_returns_none(self) -> None:
        assert _openai().reasoning_text(_msg()) is None
        assert _openai().reasoning_text(_msg({"openai_chat": {}})) is None
        assert _openai().reasoning_text(
            _msg({"openai_chat": {"reasoning_content": ""}})
        ) is None
        assert _openai().reasoning_text(
            _msg({"openai_chat": {"reasoning_details": []}})
        ) is None

    def test_foreign_namespace_returns_none(self) -> None:
        msg = _msg({"anthropic": {"thinking_blocks": [
            {"type": "thinking", "thinking": "other provider"},
        ]}})
        assert _openai().reasoning_text(msg) is None


class TestAnthropicReasoningText:
    def test_joins_thinking_blocks(self) -> None:
        msg = _msg({"anthropic": {"thinking_blocks": [
            {"type": "thinking", "thinking": "first", "signature": "s1"},
            {"type": "thinking", "thinking": "second", "signature": "s2"},
        ]}})
        assert _anthropic().reasoning_text(msg) == "first\nsecond"

    def test_skips_redacted_blocks(self) -> None:
        msg = _msg({"anthropic": {"thinking_blocks": [
            {"type": "redacted_thinking", "data": "EqMBCkgIBBABGAIiQ=="},
            {"type": "thinking", "thinking": "readable", "signature": "s"},
        ]}})
        assert _anthropic().reasoning_text(msg) == "readable"

    def test_only_redacted_returns_none(self) -> None:
        msg = _msg({"anthropic": {"thinking_blocks": [
            {"type": "redacted_thinking", "data": "opaque"},
        ]}})
        assert _anthropic().reasoning_text(msg) is None

    def test_empty_sidecar_returns_none(self) -> None:
        assert _anthropic().reasoning_text(_msg()) is None
        assert _anthropic().reasoning_text(
            _msg({"anthropic": {"thinking_blocks": []}})
        ) is None
