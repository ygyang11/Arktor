from __future__ import annotations

import os
from typing import Any
from collections.abc import AsyncIterator

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def _ensure_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            monkeypatch.setenv(key, "sk-test-dummy")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-e2e", action="store_true", default=False, help="Run e2e tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-e2e", default=False):
        return
    skip_e2e = pytest.mark.skip(reason="e2e test — pass --run-e2e to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)

from agent_harness.core.config import HarnessConfig, LLMConfig
from agent_harness.core.message import Message, MessageChunk, Role, ToolCall
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import FinishReason, LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import BaseTool, ToolSchema


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """A deterministic LLM for testing that returns pre-configured responses."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        super().__init__(config or LLMConfig(provider="mock", model="mock-model", api_key="fake"))
        self.responses: list[LLMResponse] = list(responses or [])
        self.call_history: list[list[Message]] = []
        self._call_index = 0
        self._default_response = LLMResponse(
            message=Message.assistant("Default mock response"),
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason=FinishReason.STOP,
        )

    # -- Helper methods to enqueue responses --------------------------------

    def add_text_response(self, text: str) -> None:
        """Enqueue a simple text assistant response."""
        self.responses.append(
            LLMResponse(
                message=Message.assistant(text),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason=FinishReason.STOP,
            )
        )

    def add_tool_call_response(self, name: str, args: dict[str, Any]) -> None:
        """Enqueue a response that triggers a tool call."""
        tool_call = ToolCall(name=name, arguments=args)
        self.responses.append(
            LLMResponse(
                message=Message.assistant(content=None, tool_calls=[tool_call]),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )

    # -- Abstract method implementations ------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_history.append(list(messages))
        if self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            self._call_index += 1
            return response
        return self._default_response

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        response = await self.generate(
            messages, tools=tools, tool_choice=tool_choice,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        yield StreamDelta(
            chunk=MessageChunk(
                delta_content=response.message.content,
                delta_tool_calls=response.message.tool_calls,
                finish_reason=response.finish_reason.value,
            ),
            usage=response.usage,
            finish_reason=response.finish_reason,
        )


# ---------------------------------------------------------------------------
# Mock Tool
# ---------------------------------------------------------------------------

class MockTool(BaseTool):
    """A simple tool for testing that returns a configurable response."""

    def __init__(self, response: str = "mock tool result") -> None:
        super().__init__(name="mock_tool", description="A mock tool for testing")
        self.response = response
        self.call_history: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> str:
        self.call_history.append(kwargs)
        return self.response

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string",
                    },
                },
                "required": ["query"],
            },
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm() -> MockLLM:
    """Provide a fresh :class:`MockLLM` instance."""
    return MockLLM()


@pytest.fixture()
def mock_tool() -> MockTool:
    """Provide a fresh :class:`MockTool` instance."""
    return MockTool()


@pytest.fixture()
def config() -> HarnessConfig:
    """Provide a default :class:`HarnessConfig` suitable for tests."""
    return HarnessConfig(
        llm=LLMConfig(provider="mock", model="mock-model", api_key="fake"),
        verbose=False,
    )
