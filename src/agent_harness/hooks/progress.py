"""ProgressHooks — user-facing progress output."""
from __future__ import annotations

import contextvars
import sys
from typing import TYPE_CHECKING, Any

from agent_harness.hooks.base import DefaultHooks
from agent_harness.utils.theme import COLORS, ICONS

_subagent_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_subagent_active", default=False
)

if TYPE_CHECKING:
    from agent_harness.approval.types import ApprovalRequest, ApprovalResult
    from agent_harness.core.message import ToolCall
    from agent_harness.llm.types import LLMRetryInfo, StreamDelta


class ProgressHooks(DefaultHooks):
    """User-facing progress output: tool calls, streaming content, errors."""

    _MAX_VISIBLE_TOOLS = 3

    def __init__(self, output: Any = None, color: bool = True) -> None:
        self._output = output or sys.stdout
        self._color = color and hasattr(self._output, "isatty") and self._output.isatty()
        self._streaming = False
        self._tool_call_count = 0
        self._tool_error_count = 0
        self._denied_tool_ids: set[str] = set()

    def _c(self, name: str) -> str:
        return COLORS.get(name, "") if self._color else ""

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        if _subagent_active.get(False):
            return
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        self._tool_call_count += 1
        if self._tool_call_count <= self._MAX_VISIBLE_TOOLS:
            bold, reset = self._c("bold"), self._c("reset")
            args_preview = ", ".join(
                f'{k}="{v}"' for k, v in tool_call.arguments.items()
            )
            yellow, reset2 = self._c("yellow"), self._c("reset")
            prefix = f"{yellow}⏺{reset2} " if self._tool_call_count == 1 else "  "
            self._write(f"{prefix}⚡ {bold}{tool_call.name}{reset}({args_preview})\n")

    async def on_self_heal(self, agent_name: str, summary: str) -> None:
        if _subagent_active.get(False):
            return
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        dim, reset = self._c("dim"), self._c("reset")
        self._write(f"  {ICONS['heal']} {dim}{summary}{reset}\n")

    async def on_compression_start(self, agent_name: str) -> None:
        if _subagent_active.get(False):
            return
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        dim, reset = self._c("dim"), self._c("reset")
        self._write(
            f"  {ICONS['summary']} {dim}Compressing context...{reset}\n"
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
        dim, reset = self._c("dim"), self._c("reset")
        self._write(
            f"  {ICONS['summary']} {dim}Context compressed: "
            f"{original_count} → {compressed_count} messages "
            f"(~{summary_tokens} tokens){reset}\n"
        )

    async def on_approval_request(
        self, agent_name: str, request: ApprovalRequest
    ) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False

    async def on_approval_result(self, agent_name: str, result: ApprovalResult) -> None:
        from agent_harness.approval.types import ApprovalDecision

        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        if result.decision == ApprovalDecision.DENY:
            self._denied_tool_ids.add(result.tool_call_id)
            red, reset = self._c("red"), self._c("reset")
            denied = ICONS.get("denied", "")
            label = result.tool_name or result.tool_call_id
            reason = f" — {result.reason}" if result.reason else ""
            self._write(f"  {red}{denied} Denied: {label}{reason}{reset}\n")

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        if _subagent_active.get(False):
            return
        tool_call_id = getattr(result, "tool_call_id", None)
        if tool_call_id and tool_call_id in self._denied_tool_ids:
            return
        if getattr(result, "is_error", False):
            self._tool_error_count += 1

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        if _subagent_active.get(False):
            return
        if delta.chunk.delta_content:
            if not self._streaming:
                self._streaming = True
                self._write("⏺ ")
            self._output.write(delta.chunk.delta_content)
            self._output.flush()

    async def on_llm_retry(self, agent_name: str, info: LLMRetryInfo) -> None:
        pass

    async def on_todo_update(
        self,
        agent_name: str,
        todos: list[Any],
        stats: dict[str, int],
    ) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False

        total = stats.get("total", 0)
        completed = stats.get("completed", 0)

        green, yellow, dim, bold, reset = (
            self._c("green"),
            self._c("yellow"),
            self._c("dim"),
            self._c("bold"),
            self._c("reset"),
        )

        icon = ICONS.get("todo_active", "")
        self._write(f"  {bold}{icon} Tasks [{completed}/{total}]{reset}\n")

        for t in todos:
            status = t.get("status", "pending")
            content = t.get("content", "")
            if status == "completed":
                self._write(
                    f"    {green}{ICONS['todo_done']}{reset} {dim}{content}{reset}\n"
                )
            elif status == "in_progress":
                self._write(
                    f"    {yellow}{ICONS['todo_active']}{reset} {content}\n"
                )
            else:
                self._write(
                    f"    {dim}{ICONS['todo_pending']} {content}{reset}\n"
                )

        if completed == total and total > 0:
            self._write(f"    {green}All tasks completed.{reset}\n")

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if _subagent_active.get(False):
            return
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        if self._tool_call_count > 0:
            self._print_tool_summary()

    async def on_error(self, agent_name: str, error: Exception) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        if self._tool_call_count > 0:
            self._print_tool_summary()
        red, reset = self._c("red"), self._c("reset")
        self._write(f"  {red}❌ Error: {error}{reset}\n")

    def _print_tool_summary(self) -> None:
        green, red, reset = self._c("green"), self._c("red"), self._c("reset")
        total = self._tool_call_count
        if total > self._MAX_VISIBLE_TOOLS:
            overflow = total - self._MAX_VISIBLE_TOOLS
            self._write(f"  ... and {overflow} more tools\n")
        errors = self._tool_error_count
        if errors:
            self._write(
                f"  ⎿ {green}✓ {total - errors}/{total} completed{reset}, "
                f"{red}✗ {errors} failed{reset}\n"
            )
        else:
            self._write(f"  ⎿ {green}✓ {total}/{total} completed{reset}\n")
        self._tool_call_count = 0
        self._tool_error_count = 0
        self._denied_tool_ids.clear()

    async def on_subagent_start(
        self, parent_name: str, subagent_name: str,
        agent_type: str, description: str, prompt: str,
    ) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        bold, dim, reset = self._c("bold"), self._c("dim"), self._c("reset")
        icon = ICONS.get("subagent", "")
        preview = description[:60] + "..." if len(description) > 60 else description
        self._write(
            f"  {icon} {bold}SubAgent{reset} [{agent_type}] "
            f"{dim}\"{preview}\"{reset}\n"
        )

    async def on_subagent_end(
        self, parent_name: str, subagent_name: str,
        agent_type: str, description: str,
        steps: int, tool_calls: int, duration_ms: float,
        error: str | None = None,
    ) -> None:
        if self._streaming:
            self._output.write("\n")
            self._output.flush()
            self._streaming = False
        green, dim, reset = self._c("green"), self._c("dim"), self._c("reset")
        icon = ICONS.get("summary", "")
        preview = description[:60] + "..." if len(description) > 60 else description
        duration_s = duration_ms / 1000
        if error is not None:
            red = self._c("red")
            first = error.splitlines()[0] if error.strip() else error
            err_short = first if len(first) <= 60 else first[:59] + "…"
            self._write(
                f"  {icon} {red}✗{reset} [{agent_type}] "
                f"\"{preview}\" "
                f"{dim}(failed after {duration_s:.1f}s: {err_short}){reset}\n"
            )
            return
        self._write(
            f"  {icon} {green}✓{reset} [{agent_type}] "
            f"\"{preview}\" "
            f"{dim}({steps} steps, {tool_calls} tools, {duration_s:.1f}s){reset}\n"
        )

    def _write(self, text: str) -> None:
        self._output.write(text)
        self._output.flush()
