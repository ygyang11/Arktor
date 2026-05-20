"""Base hook interface for agent lifecycle events."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness.approval.types import ApprovalRequest, ApprovalResult
    from agent_harness.core.message import ToolCall
    from agent_harness.llm.types import LLMRetryInfo, StreamDelta


class DefaultHooks:
    """No-op base hooks. Subclass to add custom behavior."""

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        pass

    async def on_step_start(self, agent_name: str, step: int) -> None:
        pass

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        pass

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        pass

    async def on_llm_retry(self, agent_name: str, info: LLMRetryInfo) -> None:
        pass

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        pass

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        pass

    async def on_step_end(self, agent_name: str, step: int) -> None:
        pass

    async def on_run_end(self, agent_name: str, output: str) -> None:
        pass

    async def on_error(self, agent_name: str, error: Exception) -> None:
        pass

    async def on_self_heal(self, agent_name: str, summary: str) -> None:
        pass

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        pass

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        pass

    async def on_dag_start(self, dag_name: str) -> None:
        pass

    async def on_dag_end(self, dag_name: str) -> None:
        pass

    async def on_dag_node_start(self, node_id: str) -> None:
        pass

    async def on_dag_node_end(self, node_id: str) -> None:
        pass

    async def on_approval_request(
        self, agent_name: str, request: ApprovalRequest
    ) -> None:
        pass

    async def on_approval_result(
        self, agent_name: str, result: ApprovalResult
    ) -> None:
        pass

    async def on_compression_start(self, agent_name: str) -> None:
        pass

    async def on_compression_end(
        self,
        agent_name: str,
        original_count: int,
        compressed_count: int,
        summary_tokens: int,
    ) -> None:
        pass

    async def on_todo_update(
        self,
        agent_name: str,
        todos: list[Any],
        stats: dict[str, int],
    ) -> None:
        pass

    async def on_team_start(self, team_name: str, mode: str) -> None:
        pass

    async def on_team_end(self, team_name: str, mode: str) -> None:
        pass

    async def on_subagent_start(
        self,
        parent_name: str,
        subagent_name: str,
        agent_type: str,
        description: str,
        prompt: str,
    ) -> None:
        pass

    async def on_subagent_end(
        self,
        parent_name: str,
        subagent_name: str,
        agent_type: str,
        description: str,
        steps: int,
        tool_calls: int,
        duration_ms: float,
    ) -> None:
        pass
