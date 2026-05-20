"""CliHooks — translate framework events to CliAdapter."""
from __future__ import annotations

import traceback
from typing import Any

from rich.markup import escape as rich_escape

from agent_cli.adapter import CliAdapter
from agent_cli.approval_handler import CliApprovalHandler
from agent_cli.render.status_lines import fmt_duration
from agent_cli.runtime.session import _TurnContext
from agent_cli.theme import COMPRESSION, SUBAGENT, SUBAGENT_DONE
from agent_harness.approval.types import ApprovalRequest, ApprovalResult
from agent_harness.core.errors import LLMUnsupportedContentError
from agent_harness.core.message import ToolCall
from agent_harness.hooks.base import DefaultHooks
from agent_harness.hooks.progress import _subagent_active
from agent_harness.llm.types import LLMRetryInfo, StreamDelta
from agent_harness.tool.registry import _tool_state_restoring
from agent_harness.utils.logging_config import setup_logging

_debug_enabled: list[bool] = [False]


def is_debug_enabled() -> bool:
    return _debug_enabled[0]


def toggle_debug() -> bool:
    _debug_enabled[0] = not _debug_enabled[0]
    setup_logging("DEBUG" if _debug_enabled[0] else "WARNING")
    return _debug_enabled[0]


class CliHooks(DefaultHooks):
    def __init__(
        self,
        adapter: CliAdapter,
        approval_handler: CliApprovalHandler | None = None,
    ) -> None:
        self.adapter = adapter
        self._approval_handler = approval_handler
        self._active_fg_subagents: set[str] = set()
        self._turn: _TurnContext | None = None

    def begin_turn(self, ctx: _TurnContext) -> None:
        self._turn = ctx

    def end_turn(self) -> None:
        self._turn = None

    @property
    def turn(self) -> _TurnContext | None:
        return self._turn

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        pass

    async def on_step_start(self, agent_name: str, step: int) -> None:
        pass

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        # subagent_start is handled in there because
        # we can get _subagent_active for fg/bg mode when they call llm
        # while on_subagent_start only fires at spawn — before fg/bg mode is set
        if _subagent_active.get(False):
            if not self._is_background():
                if agent_name not in self._active_fg_subagents:
                    self._active_fg_subagents.add(agent_name)
                    await self.adapter.start_subagent()
            return
        await self.adapter.on_llm_call()

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        if _subagent_active.get(False):
            return
        if delta.chunk.delta_content:
            if self._turn is not None:
                self._turn.committed = True
            await self.adapter.on_stream_delta(delta.chunk.delta_content)

    async def on_llm_retry(self, agent_name: str, info: LLMRetryInfo) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.on_retry(info)

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        if _subagent_active.get(False):
            if not self._is_background():
                self.adapter.tick_subagent_tool()
            return
        if self._turn is not None:
            self._turn.committed = True
        await self.adapter.on_tool_call(tool_call)

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.on_tool_result(result)

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if _subagent_active.get(False):
            if not self._is_background():
                self.adapter.tick_subagent_step()
            return
        await self.adapter.end_step()

    async def on_run_end(self, agent_name: str, output: str) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.end_run()

    async def on_error(self, agent_name: str, error: Exception) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.end_step()
        if isinstance(error, LLMUnsupportedContentError):
            return  # _run owns rendering for this class
        await self.adapter.print_inline(f"[error]! Error: {error}[/error]")
        if _debug_enabled[0]:
            tb = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            await self.adapter.print_inline(f"[dim]{rich_escape(tb)}[/dim]")

    async def on_self_heal(self, agent_name: str, summary: str) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.print_inline(f"[muted]{COMPRESSION} {summary}[/muted]")

    async def on_approval_request(
        self, agent_name: str, request: ApprovalRequest
    ) -> None:
        # pause_for_stdin is handled atomically inside
        # CliApprovalHandler._prompt_user under the shared _console_lock,
        # so the hook stays IO-free (pause + panel + prompt must be in
        # the same critical section to avoid terminal ownership race).
        if _subagent_active.get(False):
            return
        if self._is_background():
            return
        if self._turn is not None:
            self._turn.committed = True

    async def on_approval_result(
        self, agent_name: str, result: ApprovalResult
    ) -> None:
        if _subagent_active.get(False):
            return
        if self._is_background():
            return
        from agent_harness.approval.types import ApprovalDecision
        if result.decision == ApprovalDecision.DENY:
            if self._turn is not None:
                self._turn.committed = True
            await self.adapter.on_tool_denied(result)

    def _is_background(self) -> bool:
        return (
            self._approval_handler is not None
            and self._approval_handler.is_in_background_task()
        )

    async def on_compression_start(self, agent_name: str) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.print_inline(
            f"[info]{COMPRESSION} Compressing context...[/info]"
        )

    async def on_compression_end(
        self,
        agent_name: str,
        original_count: int,
        compressed_count: int,
        summary_tokens: int,
    ) -> None:
        if _subagent_active.get(False):
            return
        await self.adapter.print_inline(
            f"[info]{COMPRESSION} Compressed: {original_count} → "
            f"{compressed_count} msgs (~{summary_tokens} tokens)[/info]"
        )

    async def on_todo_update(
        self,
        agent_name: str,
        todos: list[Any],
        stats: dict[str, int],
    ) -> None:
        # Buffer only: rendering here would collide with the active general tool_display
        # Live table. end_step flushes the buffer after Live exits.
        if _subagent_active.get(False):
            return
        if _tool_state_restoring.get(False):
            # Replayed during session restore — historical, not "just happened".
            return
        self.adapter.queue_todo(todos, stats)

    async def on_subagent_start(
        self,
        parent_name: str,
        subagent_name: str,
        agent_type: str,
        description: str,
        prompt: str,
    ) -> None:
        short = description if len(description) <= 60 else description[:59] + "…"
        safe_type = rich_escape(f"[{agent_type}]")
        safe_desc = rich_escape(short)
        await self.adapter.print_inline(
            f"[accent]╭─ {SUBAGENT} SubAgent {safe_type}[/accent] "
            f'[dim]"{safe_desc}"[/dim]'
        )

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
        if self._is_background():
            return
        if subagent_name in self._active_fg_subagents:
            self._active_fg_subagents.discard(subagent_name)
            await self.adapter.stop_subagent()
        short = description if len(description) <= 60 else description[:59] + "…"
        safe_type = rich_escape(f"[{agent_type}]")
        safe_desc = rich_escape(short)
        await self.adapter.print_inline(
            f"[accent]╰─ {SUBAGENT_DONE} Done · SubAgent {safe_type}[/accent] "
            f'[dim]"{safe_desc}" '
            f"({steps} steps, {tool_calls} tools, {fmt_duration(int(duration_ms / 1000))})[/dim]"
        )

    async def on_pipeline_start(self, pipeline_name: str) -> None: pass
    async def on_pipeline_end(self, pipeline_name: str) -> None: pass
    async def on_dag_start(self, dag_name: str) -> None: pass
    async def on_dag_end(self, dag_name: str) -> None: pass
    async def on_dag_node_start(self, node_id: str) -> None: pass
    async def on_dag_node_end(self, node_id: str) -> None: pass
    async def on_team_start(self, team_name: str, mode: str) -> None: pass
    async def on_team_end(self, team_name: str, mode: str) -> None: pass
