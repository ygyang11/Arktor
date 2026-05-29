"""CliAdapter — phase state machine coordinating streaming output."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from rich.console import Console, RenderableType
from rich.markup import escape as rich_escape

from agent_cli.config import current_effort
from agent_cli.render.markdown_stream import MarkdownStream
from agent_cli.render.status_lines import SubagentLine, ThinkingLine, fmt_duration
from agent_cli.render.tool_display import SUPPRESSED_IN_ROW, ToolDisplay
from agent_cli.theme import RUN_DONE, CliTheme
from agent_harness.approval.types import ApprovalResult
from agent_harness.core.message import ToolCall, ToolResult
from agent_harness.llm.types import LLMRetryInfo

Phase = Literal["markdown", "tools", "none"]

_RUN_SUMMARY_MIN_S = 60


class CliAdapter:
    def __init__(
        self,
        console: Console,
        theme: CliTheme,
    ) -> None:
        self.console = console
        self.theme = theme
        self.markdown = MarkdownStream(console, theme)
        self.tool_display = ToolDisplay(console)
        self._phase: Phase = "none"
        self._denied_ids: set[str] = set()
        self._pending_todo: tuple[list[dict[str, str]], dict[str, int]] | None = None
        self._console_lock: asyncio.Lock = asyncio.Lock()
        self._run_started: float | None = None
        self._has_prior_step: bool = False
        self._thinking_line = ThinkingLine(
            console,
            self._console_lock,
            theme,
            effort_provider=current_effort,
            run_elapsed_provider=lambda: (
                int(time.monotonic() - self._run_started)
                if self._run_started is not None and self._has_prior_step
                else None
            ),
        )
        self._subagent_line = SubagentLine(console, self._console_lock, theme)

    def lock(self) -> asyncio.Lock:
        return self._console_lock

    def begin_run(self) -> None:
        self._run_started = time.monotonic()
        self._has_prior_step = False

    async def _enter_markdown(self) -> None:
        if self._phase == "markdown":
            return
        await self._thinking_line.stop()
        if self._phase == "tools":
            self.tool_display.end()
        self._phase = "markdown"

    async def _enter_tools(self) -> None:
        if self._phase == "tools":
            return
        await self._thinking_line.stop()
        if self._phase == "markdown":
            self.markdown.finalize()
        self._phase = "tools"

    async def _enter_none(self) -> None:
        await self._thinking_line.stop()
        if self._phase == "markdown":
            self.markdown.abort()
        elif self._phase == "tools":
            self.tool_display.end()
        self._phase = "none"

    async def _finalize_to_none(self) -> None:
        await self._thinking_line.stop()
        if self._phase == "markdown":
            self.markdown.finalize()
        elif self._phase == "tools":
            self.tool_display.end()
        self._phase = "none"

    async def on_llm_call(self) -> None:
        await self._thinking_line.start()

    async def on_stream_delta(self, text: str) -> None:
        if not text:
            return
        await self._enter_markdown()
        self.markdown.update(text)

    async def on_retry(self, info: LLMRetryInfo) -> None:
        await self._enter_none()
        err_name = rich_escape(type(info.error).__name__)
        line = f"[muted]── Retrying LLM ({info.attempt}/{info.max_retries}) · {err_name} ──[/muted]"
        await self.print_inline(line)

    async def on_tool_call(self, tool_call: ToolCall) -> None:
        if tool_call.name in SUPPRESSED_IN_ROW:
            await self._finalize_to_none()
            return
        await self._enter_tools()
        self.tool_display.add_call(tool_call)

    async def on_tool_result(self, result: ToolResult) -> None:
        if result.tool_call_id in self._denied_ids:
            return
        self.tool_display.mark_result(result)

    async def on_tool_denied(self, approval_result: ApprovalResult) -> None:
        await self._enter_tools()
        self._denied_ids.add(approval_result.tool_call_id)
        self.tool_display.add_denied(approval_result)

    def queue_todo(self, todos: list[dict[str, str]], stats: dict[str, int]) -> None:
        self._pending_todo = (todos, stats)

    async def end_step(self) -> None:
        await self._thinking_line.stop()
        await self._subagent_line.force_stop()
        if self._phase == "tools":
            self.tool_display.end()
        elif self._phase == "markdown":
            self.markdown.finalize()
        self._phase = "none"
        self._denied_ids.clear()
        if self._pending_todo is not None:
            todos, stats = self._pending_todo
            self.tool_display.show_todos(todos, stats)
            self._pending_todo = None
        self._has_prior_step = True

    async def end_run(self) -> None:
        await self.end_step()
        if self._run_started is not None:
            elapsed = int(time.monotonic() - self._run_started)
            if elapsed >= _RUN_SUMMARY_MIN_S:
                await self.print_inline(
                    f"[dim]{RUN_DONE} Worked for {fmt_duration(elapsed)}[/dim]"
                )
        self._run_started = None

    async def pause_for_stdin(self) -> None:
        # Caller holds _console_lock; use no-lock variants.
        await self._thinking_line.stop_no_lock()
        self._subagent_line.clear_no_lock()
        if self._phase == "markdown":
            self.markdown.pause()
        elif self._phase == "tools":
            self.tool_display.pause()

    async def print_inline(self, renderable: RenderableType) -> None:
        async with self._console_lock:
            # Prevent collision with an active live line mid-tick
            self._thinking_line.clear_no_lock()
            self._subagent_line.clear_no_lock()
            self.console.print(renderable)
            self.console.print()

    async def render_attachments(
        self,
        summaries: list[dict[str, Any]],
    ) -> None:
        if not summaries:
            return
        from agent_cli.render.tool_display import format_attachments  # noqa: PLC0415

        async with self._console_lock:
            self._thinking_line.clear_no_lock()
            self._subagent_line.clear_no_lock()
            for r in format_attachments(summaries):
                self.console.print(r)
            self.console.print()

    async def on_shell_run(
        self,
        command: str,
        exit_code: int,
        output: str,
    ) -> None:
        from agent_cli.render.tool_display import format_shell_run  # noqa: PLC0415

        await self._enter_none()
        async with self._console_lock:
            self._thinking_line.clear_no_lock()
            self._subagent_line.clear_no_lock()
            for r in format_shell_run(command, exit_code, output):
                self.console.print(r)
            self.console.print()

    async def start_subagent(self) -> None:
        # Handoff from parent thinking: sub_agent is SUPPRESSED_IN_ROW so
        # _enter_tools was never called to stop thinking on the parent's
        # dispatch path. Stop explicitly to avoid two live lines on the
        # same terminal row. stop() on an already-stopped line is a no-op.
        await self._thinking_line.stop()
        await self._subagent_line.start()

    async def stop_subagent(self) -> None:
        await self._subagent_line.stop()

    def tick_subagent_step(self) -> None:
        self._subagent_line.tick_step()

    def tick_subagent_tool(self) -> None:
        self._subagent_line.tick_tool()
