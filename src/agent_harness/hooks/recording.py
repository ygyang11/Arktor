"""Opt-in JSONL recording of completed model and tool interactions."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness.core.message import Message, ToolResult
from agent_harness.hooks.base import DefaultHooks
from agent_harness.llm.call_context import ModelCallContext
from agent_harness.llm.types import LLMResponse
from agent_harness.tool.base import ToolSchema

logger = logging.getLogger(__name__)


class RecordingHooks(DefaultHooks):
    """Record completed interactions without keeping an open file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = asyncio.Lock()
        self._recording_error: Exception | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8"):
            pass

    @property
    def error(self) -> Exception | None:
        """Return the first recording failure, if recording has stopped."""
        return self._recording_error

    async def on_llm_response(
        self,
        agent_name: str,
        messages: list[Message],
        response: LLMResponse,
        *,
        call_context: ModelCallContext,
        tools: list[ToolSchema] | None = None,
    ) -> None:
        async with self._lock:
            if self._recording_error is not None:
                return
            try:
                self._append({
                    "type": "llm_call",
                    "created_at": datetime.now().isoformat(),
                    "agent_name": call_context.agent.name,
                    "parent_agent_name": call_context.agent.parent,
                    "agent_type": call_context.agent.type,
                    "call": call_context.to_dict()["call"],
                    "request": {
                        "messages": [
                            message.model_dump(mode="json") for message in messages
                        ],
                        "tools": [schema.model_dump(mode="json") for schema in tools or []],
                    },
                    "response": response.model_dump(
                        mode="json", exclude={"raw_response"},
                    ),
                })
            except Exception as error:
                self._disable(error)

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        async with self._lock:
            if self._recording_error is not None:
                return
            try:
                if not isinstance(result, ToolResult):
                    raise TypeError("Expected a ToolResult.")
                self._append({
                    "type": "tool_result",
                    "created_at": datetime.now().isoformat(),
                    "agent_name": agent_name,
                    "result": result.model_dump(mode="json"),
                })
            except Exception as error:
                self._disable(error)

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")

    def _disable(self, error: Exception) -> None:
        self._recording_error = error
        logger.error(
            "Interaction recording stopped; %s is incomplete: %s",
            self.path, error,
        )
