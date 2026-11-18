"""Tests for provider request and stream retries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from agent_harness import HarnessConfig, ReActAgent, RecordingHooks
from agent_harness.core.config import LLMConfig
from agent_harness.core.message import Message
from agent_harness.llm.anthropic_provider import AnthropicProvider
from agent_harness.llm.openai_provider import OpenAIProvider


class _FailingAsyncIterator:
    def __init__(self, error: Exception | None, items: list[Any]) -> None:
        self._error = error
        self._items = items
        self._failed = False
        self._index = 0

    def __aiter__(self) -> _FailingAsyncIterator:
        return self

    async def __anext__(self) -> object:
        if self._error is not None and not self._failed:
            self._failed = True
            raise self._error
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class _AsyncStream(_FailingAsyncIterator):
    async def __aenter__(self) -> _AsyncStream:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model="test",
        temperature=0.7,
        max_tokens=100,
        reasoning_effort=None,
        max_retries=1,
        retry_delay=0.0,
    )


def _openai_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="ok",
                    reasoning_content=None,
                    reasoning_details=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("incomplete chunked read"),
        httpx.ReadTimeout("stream read timed out"),
    ],
)
async def test_openai_stream_retries_httpx_transport_error(error: httpx.TransportError) -> None:
    calls = 0

    async def create(**kwargs: Any) -> _FailingAsyncIterator:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _FailingAsyncIterator(error, [])
        return _FailingAsyncIterator(None, [_openai_chunk()])

    provider: Any = OpenAIProvider.__new__(OpenAIProvider)
    provider.config = _config()
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider._rate_limiter = None
    provider._additive_semantics = False
    provider._strip_reasoning_details = False

    response = await provider.stream_with_events([Message.user("hi")])

    assert response.message.content == "ok"
    assert calls == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_openai_http_header_retry_and_record(
    stream: bool, enabled: bool, tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"message": "retry", "type": "rate_limit"}})

        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        if stream:
            data = {
                "id": "test", "object": "chat.completion.chunk", "created": 0, "model": "mock",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": usage,
            }
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n",
            )

        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": usage,
        })

    config = HarnessConfig(llm=LLMConfig(
        model="mock", api_key="test", max_retries=2,
        retry_delay=0, send_request_context=enabled,
    ))
    provider = OpenAIProvider(config.llm)
    await provider._client.close()
    provider._client = AsyncOpenAI(
        api_key="test", base_url="http://test/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    recorder = RecordingHooks(tmp_path / "trace.jsonl")
    agent = ReActAgent(name="worker", llm=provider, config=config, hooks=recorder, stream=stream)
    try:
        result = await agent.run("respond")
    finally:
        await agent.aclose()

    assert result.output == "ok" and len(requests) == 2
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["call"]["seq"] == 1
    if enabled:
        headers = [request.headers["x-arktor-context"] for request in requests]
        assert headers[0] == headers[1]
        assert json.loads(headers[1])["call"] == records[0]["call"]
    else:
        assert all("x-arktor-context" not in request.headers for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("incomplete chunked read"),
        httpx.ReadTimeout("stream read timed out"),
    ],
)
async def test_anthropic_stream_retries_httpx_transport_error(
    error: httpx.TransportError,
) -> None:
    calls = 0
    events = [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="ok"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=None,
        ),
    ]

    def stream(**kwargs: Any) -> _AsyncStream:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _AsyncStream(error, [])
        return _AsyncStream(None, events)

    provider: Any = AnthropicProvider.__new__(AnthropicProvider)
    provider.config = _config()
    provider._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    provider._rate_limiter = None

    response = await provider.stream_with_events([Message.user("hi")])

    assert response.message.content == "ok"
    assert calls == 2
