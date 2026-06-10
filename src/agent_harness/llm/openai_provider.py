"""OpenAI LLM provider for agent_harness."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from agent_harness.core.config import HarnessConfig, LLMConfig, resolve_llm_config
from agent_harness.core.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnsupportedContentError,
)
from agent_harness.core.message import Attachment, Message, MessageChunk, Role, ToolCall
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import ToolSchema
from agent_harness.utils.media import is_media_rejection

logger = logging.getLogger(__name__)

_PROTOCOL_KEY = "openai_chat"
# Single source of truth for the OpenAI-compatible reasoning sidecar.
_REASONING_SIDECAR: dict[str, Any] = {
    "reasoning_content": "",  # models native + OpenRouter alias
    "reasoning_details": [],  # OpenRouter's structured form (list of blocks)
}


class OpenAIProvider(BaseLLM):
    """OpenAI API provider (GPT-4o, o1, etc.).

    Handles:
    - Message format conversion (Message -> OpenAI dict)
    - Function/tool calling
    - Streaming
    - Error mapping to framework exceptions
    """

    def __init__(self, config: HarnessConfig | LLMConfig | None = None) -> None:
        llm_config = resolve_llm_config(config)
        super().__init__(llm_config)
        self._client = AsyncOpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=llm_config.timeout,
        )
        self._additive_semantics: bool = False
        self._strip_reasoning_details: bool = False

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request_kwargs = self._build_request(
            messages, tools, tool_choice, temperature, max_tokens, **kwargs
        )
        if self._strip_reasoning_details:
            _normalize_reasoning_for_strict_input(request_kwargs)

        retried = False
        while True:
            try:
                response = await self._client.chat.completions.create(**request_kwargs)
                break
            except openai.RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except openai.AuthenticationError as e:
                raise LLMAuthenticationError(str(e)) from e
            except openai.APIConnectionError as e:
                raise LLMConnectionError(str(e)) from e
            except openai.APIStatusError as e:
                s = str(e)
                if is_media_rejection(s):
                    raise LLMUnsupportedContentError(s) from e
                if isinstance(e, openai.BadRequestError):
                    if not retried and _is_reasoning_details_rejection(e):
                        self._strip_reasoning_details = True
                        _normalize_reasoning_for_strict_input(request_kwargs)
                        retried = True
                        continue
                    if "context_length" in s.lower() or "maximum context" in s.lower():
                        raise LLMContextLengthError(s) from e
                raise LLMError(s) from e
            except openai.APIError as e:
                raise LLMError(str(e)) from e

        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        request_kwargs = self._build_request(
            messages, tools, tool_choice, temperature, max_tokens, stream=True, **kwargs
        )
        if self._strip_reasoning_details:
            _normalize_reasoning_for_strict_input(request_kwargs)

        tc_buffer: dict[int, dict[str, str]] = {}
        # OpenAI stream usage is a request-total snapshot, not a chunk delta.
        # Keep the last snapshot and emit it once so upstream adders stay safe.
        final_usage: Usage | None = None

        retried = False
        while True:
            try:
                stream = await self._client.chat.completions.create(**request_kwargs)
                break
            except openai.RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except openai.AuthenticationError as e:
                raise LLMAuthenticationError(str(e)) from e
            except openai.APIConnectionError as e:
                raise LLMConnectionError(str(e)) from e
            except openai.APIStatusError as e:
                s = str(e)
                if is_media_rejection(s):
                    raise LLMUnsupportedContentError(s) from e
                if isinstance(e, openai.BadRequestError):
                    if not retried and _is_reasoning_details_rejection(e):
                        self._strip_reasoning_details = True
                        _normalize_reasoning_for_strict_input(request_kwargs)
                        retried = True
                        continue
                raise LLMError(s) from e
            except openai.APIError as e:
                raise LLMError(str(e)) from e

        try:
            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    final_usage = self._parse_usage(chunk.usage)

                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tc_buffer:
                            tc_buffer[idx] = {"id": "", "name": "", "args": ""}
                        if tc.id:
                            tc_buffer[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tc_buffer[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tc_buffer[idx]["args"] += tc.function.arguments

                finish_reason = None
                delta_tool_calls = None
                if choice.finish_reason:
                    finish_reason = _map_finish_reason(choice.finish_reason)
                    if tc_buffer and finish_reason == FinishReason.TOOL_CALLS:
                        delta_tool_calls = []
                        for idx in sorted(tc_buffer):
                            buf = tc_buffer[idx]
                            try:
                                args = json.loads(buf["args"]) if buf["args"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            delta_tool_calls.append(
                                ToolCall(id=buf["id"], name=buf["name"], arguments=args)
                            )

                delta_sidecar: dict[str, Any] = {}
                for fname in _REASONING_SIDECAR:
                    val = getattr(delta, fname, None)
                    if val is not None:
                        delta_sidecar[fname] = val
                delta_pm: dict[str, dict[str, Any]] | None = (
                    {_PROTOCOL_KEY: delta_sidecar} if delta_sidecar else None
                )

                yield StreamDelta(
                    chunk=MessageChunk(
                        delta_content=delta.content,
                        delta_tool_calls=delta_tool_calls,
                        delta_provider_metadata=delta_pm,
                        finish_reason=choice.finish_reason,
                    ),
                    finish_reason=finish_reason,
                )

            if final_usage is not None:
                yield StreamDelta(chunk=MessageChunk(), usage=final_usage)

        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(str(e)) from e

    def reasoning_text(self, message: Message) -> str | None:
        """Decode the OpenAI-compatible reasoning sidecar into plain text."""
        sidecar = message.provider_metadata.get(_PROTOCOL_KEY, {})
        rc = sidecar.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc
        rd = sidecar.get("reasoning_details")
        if isinstance(rd, list):
            return _flatten_reasoning_details(rd) or None
        return None

    def _build_request(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._expand_tool_result_media(messages)
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._format_message(m) for m in messages],
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
            "stream": stream,
        }

        if stream:
            request["stream_options"] = {"include_usage": True}

        if tools:
            request["tools"] = [t.to_openai_format() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "required", "none"):
                    request["tool_choice"] = tool_choice
                else:
                    request["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        if self.config.reasoning_effort:
            request["reasoning_effort"] = self.config.reasoning_effort

        request.update(kwargs)
        return request

    def _format_message(self, msg: Message) -> dict[str, Any]:
        """Convert a Message to OpenAI API format."""
        result: dict[str, Any] = {"role": msg.role.value}

        if msg.role == Role.TOOL and msg.tool_result:
            result["tool_call_id"] = msg.tool_result.tool_call_id
            result["content"] = msg.tool_result.content
            return result

        if msg.role == Role.USER and msg.attachments:
            result["content"] = self._render_user_content(msg)
        elif msg.content is not None:
            result["content"] = msg.content

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.role == Role.ASSISTANT:
            sidecar = msg.provider_metadata.get(_PROTOCOL_KEY, {})
            for fname in _REASONING_SIDECAR:
                if fname in sidecar:
                    result[fname] = sidecar[fname]

        if msg.name:
            result["name"] = msg.name

        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Convert OpenAI response to LLMResponse."""
        if not response.choices:
            raise LLMResponseError(
                "OpenAI API returned empty choices",
                details={"model": response.model},
            )
        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                )
                for tc in msg.tool_calls
            ]

        sidecar: dict[str, Any] = {}
        for fname in _REASONING_SIDECAR:
            val = getattr(msg, fname, None)
            if val is not None:
                sidecar[fname] = val
        provider_metadata: dict[str, dict[str, Any]] = {_PROTOCOL_KEY: sidecar} if sidecar else {}

        message = Message(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
            provider_metadata=provider_metadata,
        )

        usage = self._parse_usage(response.usage)

        return LLMResponse(
            message=message,
            usage=usage,
            finish_reason=_map_finish_reason(choice.finish_reason),
            model=response.model,
            raw_response=response,
        )

    def _parse_usage(self, raw_usage: Any) -> Usage:
        if not raw_usage:
            return Usage()
        prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
        cached = getattr(prompt_details, "cached_tokens", 0) or 0
        completion_details = getattr(raw_usage, "completion_tokens_details", None)
        reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0
        prompt = raw_usage.prompt_tokens or 0
        completion = raw_usage.completion_tokens or 0
        total = raw_usage.total_tokens or 0

        # OpenAI invariant: cached_tokens is a subset of prompt_tokens.
        # Some OpenAI-compatible relays (e.g. opencode-zen → minimax) leak
        # Anthropic-style additive semantics through OpenAI field names —
        # cached is reported *outside* prompt. Detect via the invariant
        # violation and normalize to inclusive form. Sticky per instance
        # because additive providers can also return cached <= prompt on
        # cache-miss-heavy turns.
        if not self._additive_semantics and cached > prompt:
            self._additive_semantics = True
            logger.info(
                "Detected additive cache semantics on model=%s base_url=%s; "
                "normalizing prompt_tokens to inclusive form.",
                self.config.model, self.config.base_url,
            )
        if self._additive_semantics:
            prompt = prompt + cached
            total = prompt + completion

        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cache_read_tokens=cached,
            cache_creation_tokens=0,
            reasoning_tokens=reasoning,
        )

    def _image_block(self, att: Attachment, b64: str) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{att.mime};base64,{b64}"},
        }

    def _pdf_block(self, att: Attachment, b64: str) -> dict[str, Any]:
        return {
            "type": "file",
            "file": {
                "filename": att.filename or "document.pdf",
                "file_data": f"data:application/pdf;base64,{b64}",
            },
        }


def _is_reasoning_details_rejection(e: openai.BadRequestError) -> bool:
    s = str(e)
    return "Extra inputs are not permitted" in s and "reasoning_details" in s


def _flatten_reasoning_details(rd: list[Any]) -> str:
    """Flatten OpenRouter reasoning_details blocks into plain text."""
    parts = [
        b.get("text") or b.get("summary") or ""
        for b in rd
        if isinstance(b, dict)
    ]
    return "".join(p for p in parts if p)


def _normalize_reasoning_for_strict_input(request_kwargs: dict[str, Any]) -> None:
    """Sticky workaround for relays that emit reasoning_details but reject it
    as input (opencode-zen → Moonshot). Flatten its block text into
    reasoning_content (keeps multi-step thinking continuity) and drop
    reasoning_details. Never runs for spec-conformant providers since they do
    not raise the rejection that sets the sticky flag."""
    messages = request_kwargs.get("messages")
    if not isinstance(messages, list):
        return
    for m in messages:
        rd = m.get("reasoning_details")
        if rd is None:
            continue
        if not m.get("reasoning_content") and isinstance(rd, list):
            joined = _flatten_reasoning_details(rd)
            if joined:
                m["reasoning_content"] = joined
        m.pop("reasoning_details", None)


def _map_finish_reason(reason: str | None) -> FinishReason:
    """Map OpenAI finish reason to FinishReason enum."""
    mapping = {
        "stop": FinishReason.STOP,
        "tool_calls": FinishReason.TOOL_CALLS,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(reason or "stop", FinishReason.STOP)
