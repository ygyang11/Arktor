"""CompositeHooks — run multiple hooks in sequence."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_harness.hooks.base import DefaultHooks

if TYPE_CHECKING:
    from agent_harness.approval.types import ApprovalRequest, ApprovalResult
    from agent_harness.core.message import ToolCall
    from agent_harness.llm.types import LLMRetryInfo, StreamDelta


class CompositeHooks(DefaultHooks):
    """Runs multiple hooks implementations in sequence."""

    def __init__(self, *hooks: DefaultHooks) -> None:
        self._hooks = [h for h in hooks if h is not None]

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        for h in self._hooks:
            await h.on_run_start(agent_name, input_text)

    async def on_step_start(self, agent_name: str, step: int) -> None:
        for h in self._hooks:
            await h.on_step_start(agent_name, step)

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        for h in self._hooks:
            await h.on_llm_call(agent_name, messages)

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        for h in self._hooks:
            await h.on_llm_stream_delta(agent_name, delta)

    async def on_llm_retry(self, agent_name: str, info: LLMRetryInfo) -> None:
        for h in self._hooks:
            await h.on_llm_retry(agent_name, info)

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        for h in self._hooks:
            await h.on_tool_call(agent_name, tool_call)

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        for h in self._hooks:
            await h.on_tool_result(agent_name, result)

    async def on_step_end(self, agent_name: str, step: int) -> None:
        for h in self._hooks:
            await h.on_step_end(agent_name, step)

    async def on_run_end(self, agent_name: str, output: str) -> None:
        for h in self._hooks:
            await h.on_run_end(agent_name, output)

    async def on_error(self, agent_name: str, error: Exception) -> None:
        for h in self._hooks:
            await h.on_error(agent_name, error)

    async def on_self_heal(self, agent_name: str, summary: str) -> None:
        for h in self._hooks:
            await h.on_self_heal(agent_name, summary)

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        for h in self._hooks:
            await h.on_pipeline_start(pipeline_name)

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        for h in self._hooks:
            await h.on_pipeline_end(pipeline_name)

    async def on_dag_start(self, dag_name: str) -> None:
        for h in self._hooks:
            await h.on_dag_start(dag_name)

    async def on_dag_end(self, dag_name: str) -> None:
        for h in self._hooks:
            await h.on_dag_end(dag_name)

    async def on_dag_node_start(self, node_id: str) -> None:
        for h in self._hooks:
            await h.on_dag_node_start(node_id)

    async def on_dag_node_end(self, node_id: str) -> None:
        for h in self._hooks:
            await h.on_dag_node_end(node_id)

    async def on_compression_start(self, agent_name: str) -> None:
        for h in self._hooks:
            await h.on_compression_start(agent_name)

    async def on_compression_end(
        self,
        agent_name: str,
        original_count: int,
        compressed_count: int,
        summary_tokens: int,
    ) -> None:
        for h in self._hooks:
            await h.on_compression_end(
                agent_name, original_count, compressed_count, summary_tokens
            )

    async def on_approval_request(
        self, agent_name: str, request: ApprovalRequest
    ) -> None:
        for h in self._hooks:
            await h.on_approval_request(agent_name, request)

    async def on_approval_result(
        self, agent_name: str, result: ApprovalResult
    ) -> None:
        for h in self._hooks:
            await h.on_approval_result(agent_name, result)

    async def on_todo_update(
        self,
        agent_name: str,
        todos: list[Any],
        stats: dict[str, int],
    ) -> None:
        for h in self._hooks:
            await h.on_todo_update(agent_name, todos, stats)

    async def on_team_start(self, team_name: str, mode: str) -> None:
        for h in self._hooks:
            await h.on_team_start(team_name, mode)

    async def on_team_end(self, team_name: str, mode: str) -> None:
        for h in self._hooks:
            await h.on_team_end(team_name, mode)

    async def on_subagent_start(
        self, parent_name: str, subagent_name: str,
        agent_type: str, description: str, prompt: str,
    ) -> None:
        for h in self._hooks:
            await h.on_subagent_start(
                parent_name, subagent_name, agent_type, description, prompt,
            )

    async def on_subagent_end(
        self, parent_name: str, subagent_name: str,
        agent_type: str, description: str,
        steps: int, tool_calls: int, duration_ms: float,
        error: str | None = None,
    ) -> None:
        for h in self._hooks:
            await h.on_subagent_end(
                parent_name, subagent_name, agent_type, description,
                steps, tool_calls, duration_ms, error,
            )
