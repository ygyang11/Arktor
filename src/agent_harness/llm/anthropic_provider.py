"""Anthropic LLM provider for agent_harness (Claude models)."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from agent_harness.core.config import HarnessConfig, LLMConfig, resolve_llm_config
from agent_harness.core.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
)
from agent_harness.core.message import Message, MessageChunk, Role, ToolCall
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)

_PROTOCOL_KEY = "anthropic"


class AnthropicProvider(BaseLLM):
    """Anthropic API provider (Claude 3.5, Claude 4, etc.).

    Handles Anthropic-specific differences:
    - System message is a top-level parameter, not in messages
    - Tool results use content blocks, not separate message role
    - Streaming uses server-sent events
    """

    def __init__(self, config: HarnessConfig | LLMConfig | None = None) -> None:
        llm_config = resolve_llm_config(config)
        super().__init__(llm_config)
        self._client = AsyncAnthropic(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            timeout=llm_config.timeout,
        )

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

        try:
            response = await self._client.messages.create(**request_kwargs)
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except anthropic.BadRequestError as e:
            if "context" in str(e).lower() or "too long" in str(e).lower():
                raise LLMContextLengthError(str(e)) from e
            raise LLMError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except anthropic.APIError as e:
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
            messages, tools, tool_choice, temperature, max_tokens, **kwargs
        )

        tc_buffer: dict[int, dict[str, str]] = {}
        thinking_buffer: dict[int, dict[str, Any]] = {}
        completed_thinking: list[dict[str, Any]] = []
        last_cum_output = 0

        try:
            async with self._client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        btype = getattr(block, "type", None) if block else None
                        idx_start: int = getattr(event, "index", 0)
                        if btype == "tool_use":
                            tc_buffer[idx_start] = {
                                "id": getattr(block, "id", ""),
                                "name": getattr(block, "name", ""),
                                "input_json": "",
                            }
                        elif btype == "thinking":
                            thinking_buffer[idx_start] = {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "",
                            }
                        elif btype == "redacted_thinking":
                            completed_thinking.append({
                                "type": "redacted_thinking",
                                "data": getattr(block, "data", ""),
                            })
                            yield StreamDelta(
                                chunk=MessageChunk(
                                    delta_provider_metadata={
                                        _PROTOCOL_KEY: {"thinking_blocks": list(completed_thinking)},
                                    },
                                ),
                            )
                        continue

                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        dtype = getattr(delta, "type", None) if delta else None
                        idx_d = getattr(event, "index", 0)
                        if dtype == "input_json_delta":
                            if idx_d in tc_buffer:
                                tc_buffer[idx_d]["input_json"] += getattr(
                                    delta, "partial_json", ""
                                )
                            continue
                        if dtype == "text_delta":
                            yield StreamDelta(
                                chunk=MessageChunk(delta_content=delta.text),
                            )
                            continue
                        if dtype == "thinking_delta":
                            if idx_d in thinking_buffer:
                                thinking_buffer[idx_d]["thinking"] += getattr(
                                    delta, "thinking", ""
                                )
                            continue
                        if dtype == "signature_delta":
                            if idx_d in thinking_buffer:
                                thinking_buffer[idx_d]["signature"] += getattr(
                                    delta, "signature", ""
                                )
                            continue

                    if event_type == "content_block_stop":
                        idx_stop = getattr(event, "index", 0)
                        if idx_stop in thinking_buffer:
                            completed_thinking.append(thinking_buffer.pop(idx_stop))
                            yield StreamDelta(
                                chunk=MessageChunk(
                                    delta_provider_metadata={
                                        _PROTOCOL_KEY: {"thinking_blocks": list(completed_thinking)},
                                    },
                                ),
                            )
                        continue

                    if event_type == "message_start":
                        msg = getattr(event, "message", None)
                        msg_usage = getattr(msg, "usage", None)
                        if msg_usage:
                            input_uncached = getattr(msg_usage, "input_tokens", 0) or 0
                            cache_read = getattr(msg_usage, "cache_read_input_tokens", 0) or 0
                            cache_creation = getattr(msg_usage, "cache_creation_input_tokens", 0) or 0
                            total_input = input_uncached + cache_read + cache_creation
                            yield StreamDelta(
                                chunk=MessageChunk(),
                                usage=Usage(
                                    prompt_tokens=total_input,
                                    completion_tokens=0,
                                    total_tokens=total_input,
                                    cache_read_tokens=cache_read,
                                    cache_creation_tokens=cache_creation,
                                ),
                            )
                        continue

                    if event_type == "message_delta":
                        stop_reason = getattr(
                            getattr(event, "delta", None), "stop_reason", None
                        )
                        finish = FinishReason.STOP
                        delta_tool_calls = None
                        if stop_reason == "tool_use":
                            finish = FinishReason.TOOL_CALLS
                            delta_tool_calls = []
                            for buf_idx in sorted(tc_buffer):
                                buf = tc_buffer[buf_idx]
                                try:
                                    args = json.loads(buf["input_json"]) if buf["input_json"] else {}
                                except json.JSONDecodeError:
                                    args = {}
                                delta_tool_calls.append(
                                    ToolCall(id=buf["id"], name=buf["name"], arguments=args)
                                )
                        elif stop_reason == "max_tokens":
                            finish = FinishReason.LENGTH

                        delta_usage: Usage | None = None
                        evt_usage = getattr(event, "usage", None)
                        if evt_usage:
                            current_cum = getattr(evt_usage, "output_tokens", 0) or 0
                            delta_out = current_cum - last_cum_output
                            last_cum_output = current_cum
                            if delta_out:
                                delta_usage = Usage(
                                    completion_tokens=delta_out,
                                    total_tokens=delta_out,
                                )

                        yield StreamDelta(
                            chunk=MessageChunk(
                                delta_tool_calls=delta_tool_calls,
                                finish_reason=stop_reason,
                            ),
                            finish_reason=finish,
                            usage=delta_usage,
                        )

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(str(e)) from e

    def _build_request(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        tool_choice: str | None,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Extract system message (Anthropic takes it as top-level param)
        system_content, api_messages = self._split_system_message(messages)

        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }

        if system_content:
            request["system"] = system_content

        if tools:
            request["tools"] = [t.to_anthropic_format() for t in tools]
            if tool_choice:
                if tool_choice == "auto":
                    request["tool_choice"] = {"type": "auto"}
                elif tool_choice == "required":
                    request["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    pass  # Don't send tools
                else:
                    request["tool_choice"] = {"type": "tool", "name": tool_choice}

        if self.config.reasoning_effort:
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
            request["output_config"] = {"effort": self.config.reasoning_effort}

        request.update(kwargs)
        return request

    @staticmethod
    def _split_system_message(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separate system messages and format remaining for Anthropic API."""
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
                continue

            if msg.role == Role.TOOL and msg.tool_result:
                # Anthropic: tool results are user messages with tool_result content blocks
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_result.tool_call_id,
                            "content": msg.tool_result.content,
                            "is_error": msg.tool_result.is_error,
                        }
                    ],
                })
                continue

            if msg.role == Role.ASSISTANT:
                sidecar = msg.provider_metadata.get(_PROTOCOL_KEY, {})
                content_blocks: list[dict[str, Any]] = []
                for tb in sidecar.get("thinking_blocks", []):
                    content_blocks.append(tb)
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in (msg.tool_calls or []):
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": ""})
                api_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
                continue

            api_messages.append({
                "role": msg.role.value,
                "content": msg.content or "",
            })

        system_content = "\n\n".join(system_parts) if system_parts else None
        if len(system_parts) > 1:
            logger.debug(
                "Multiple system messages (%d) merged for Anthropic API",
                len(system_parts),
            )
        return system_content, api_messages

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        content_text: str | None = None
        tool_calls: list[ToolCall] = []
        thinking_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
            elif block.type == "thinking":
                thinking_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
            elif block.type == "redacted_thinking":
                thinking_blocks.append({
                    "type": "redacted_thinking",
                    "data": block.data,
                })

        provider_metadata: dict[str, dict[str, Any]] = {}
        if thinking_blocks:
            provider_metadata[_PROTOCOL_KEY] = {"thinking_blocks": thinking_blocks}

        message = Message(
            role=Role.ASSISTANT,
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
            provider_metadata=provider_metadata,
        )

        if response.usage:
            input_uncached = response.usage.input_tokens or 0
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            total_input = input_uncached + cache_read + cache_creation
            output_tokens = response.usage.output_tokens or 0
            usage = Usage(
                prompt_tokens=total_input,
                completion_tokens=output_tokens,
                total_tokens=total_input + output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )
        else:
            usage = Usage()

        finish_reason = FinishReason.STOP
        if response.stop_reason == "tool_use":
            finish_reason = FinishReason.TOOL_CALLS
        elif response.stop_reason == "max_tokens":
            finish_reason = FinishReason.LENGTH

        return LLMResponse(
            message=message,
            usage=usage,
            finish_reason=finish_reason,
            model=response.model,
            raw_response=response,
        )

    @staticmethod
    def _parse_stream_event(event: Any) -> StreamDelta | None:
        """Parse an Anthropic stream event into a StreamDelta."""
        # Handle different event types from Anthropic's streaming API
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                return StreamDelta(
                    chunk=MessageChunk(delta_content=delta.text),
                )
            if delta and getattr(delta, "type", None) == "input_json_delta":
                # Partial tool input JSON
                return None  # Accumulate externally

        if event_type == "message_delta":
            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            if stop_reason:
                finish = FinishReason.STOP
                if stop_reason == "tool_use":
                    finish = FinishReason.TOOL_CALLS
                elif stop_reason == "max_tokens":
                    finish = FinishReason.LENGTH
                return StreamDelta(
                    chunk=MessageChunk(finish_reason=stop_reason),
                    finish_reason=finish,
                )

        return None
