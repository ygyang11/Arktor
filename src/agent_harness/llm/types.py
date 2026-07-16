"""LLM request and response types for agent_harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_harness.core.message import Message, MessageChunk


class FinishReason(str, Enum):
    """Reason the LLM stopped generating."""
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


UsageSource = Literal["main", "compressor", "subagent", "background", "goal_eval"]


class Usage(BaseModel):
    """Token usage statistics from an LLM call.

    prompt_tokens is the TOTAL input across all categories:
      - Anthropic: input_tokens + cache_creation_input_tokens + cache_read_input_tokens
      - OpenAI:    response.usage.prompt_tokens (already total)
    cache_read_tokens / cache_creation_tokens are sub-breakdowns of
    prompt_tokens, not additive on top.
    OpenAI never emits cache_creation_tokens (no concept); it stays 0.

    reasoning_tokens is the ephemeral reasoning portion of completion_tokens
    that is NOT persisted as message content (OpenAI). Anthropicextended
    thinking is returned as a persistent thinking content block, so its tokens
    remain in completion_tokens and reasoning_tokens stays 0.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class _Bucket:
    usage: Usage = field(default_factory=Usage)
    calls: int = 0


@dataclass
class ProcessUsageMeter:
    """Cumulative LLM usage across all components in this REPL process.

    Not persisted: a fresh process / session resume starts from zero.
    Shared by reference via AgentContext.fork() — compressor, sub-agents,
    background tasks all record into the same instance.
    """
    total: Usage = field(default_factory=Usage)
    by_model: dict[str, _Bucket] = field(default_factory=dict)
    by_source: dict[str, _Bucket] = field(default_factory=dict)
    call_count: int = 0

    def record(self, usage: Usage, *, model: str, source: UsageSource) -> None:
        if usage.total_tokens == 0 and usage.prompt_tokens == 0:
            return
        self.total = self.total + usage
        self._add(self.by_model, model, usage)
        self._add(self.by_source, source, usage)
        self.call_count += 1

    @staticmethod
    def _add(bucket_map: dict[str, _Bucket], key: str, usage: Usage) -> None:
        b = bucket_map.setdefault(key, _Bucket())
        b.usage = b.usage + usage
        b.calls += 1

    def reset(self) -> None:
        self.total = Usage()
        self.by_model.clear()
        self.by_source.clear()
        self.call_count = 0


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    message: Message
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    model: str | None = None
    raw_response: Any | None = None

    @property
    def has_tool_calls(self) -> bool:
        return self.message.has_tool_calls


class StreamDelta(BaseModel):
    """A single chunk in a streaming LLM response."""
    chunk: MessageChunk
    usage: Usage | None = None
    finish_reason: FinishReason | None = None


@dataclass(frozen=True)
class LLMRetryInfo:
    """Carried to ``on_llm_retry`` hooks before each retry sleep."""
    kind: Literal["stream", "generate"]
    attempt: int
    max_retries: int
    wait: float
    error: Exception
