"""TracingHooks — span-based observability with console and JSON export."""
from __future__ import annotations

import contextvars
import sys
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_harness.hooks.base import DefaultHooks
from agent_harness.tracing.exporters.console import ConsoleExporter
from agent_harness.tracing.exporters.json_file import JsonFileExporter
from agent_harness.tracing.tracer import Span
from agent_harness.utils.theme import ICONS
from agent_harness.utils.token_counter import truncate_text_by_tokens

if TYPE_CHECKING:
    from agent_harness.approval.types import ApprovalRequest, ApprovalResult
    from agent_harness.core.message import ToolCall
    from agent_harness.llm.types import LLMRetryInfo, StreamDelta

_active_orchestration_parent: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_active_orchestration_parent", default=None
)
_active_step_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_active_step_span", default=None
)
_active_run_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_active_run_span", default=None
)
_streaming_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_streaming_active", default=False
)


_TRACE_TEXT_MAX_TOKENS = 64
_TRACE_TOOL_ARG_MAX_TOKENS = 20
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "token",
    "authorization",
    "password",
    "secret",
    "key",
}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _SENSITIVE_KEYS or lowered.endswith("_key")


def _truncate_tool_arg(value: object) -> str:
    return truncate_text_by_tokens(
        str(value),
        max_tokens=_TRACE_TOOL_ARG_MAX_TOKENS,
        suffix="...",
    )


def _summarize_tool_arguments(arguments: dict[str, object]) -> dict[str, str]:
    summarized: dict[str, str] = {}
    for key, value in arguments.items():
        summarized[key] = "[REDACTED]" if _is_sensitive_key(key) else _truncate_tool_arg(value)
    return summarized


def _summarize_tool_call(tool_call: ToolCall) -> dict[str, object]:
    return {
        "tool": tool_call.name,
        "args": _summarize_tool_arguments(tool_call.arguments),
    }


class TracingHooks(DefaultHooks):
    """Hooks that automatically create and manage trace spans."""

    def __init__(
        self,
        *,
        trace_dir: str = "./traces",
        exporter: str = "both",
        console_color: bool = True,
    ) -> None:
        self._trace_dir = trace_dir
        self._console_enabled = exporter in ("console", "both")
        self._json_enabled = exporter in ("json_file", "both")
        self._console = (
            ConsoleExporter(stream=sys.stderr, color=console_color)
            if self._console_enabled else None
        )

        self._trace_id: str = ""
        self._root_span: Span | None = None
        self._span_stack: list[Span] = []
        self._run_span_stack: list[Span] = []
        self._pipeline_span_stack: list[Span] = []
        self._dag_span_stack: list[Span] = []
        self._team_span_stack: list[Span] = []
        self._dag_node_span_stack: list[Span] = []
        self._subagent_spans: dict[str, Span] = {}
        self._all_spans: list[Span] = []
        self._span_depth_map: dict[str, int] = {}
        self._depth = 0

    def _cwrite(self, text: str) -> None:
        if self._console is not None:
            self._console.write_inline(text)

    def _cprint_start(self, **kwargs: Any) -> None:
        if self._console is not None:
            self._console.print_start(**kwargs)

    def _cexport_one(self, span: Span, *, indent: int) -> None:
        if self._console is not None:
            self._console.export_one(span, indent=indent)

    def _new_span(
        self,
        name: str,
        kind: str = "internal",
        *,
        parent_span: Span | None = None,
        **attrs: Any,
    ) -> Span:
        parent_id: str | None
        if parent_span is not None:
            parent_id = parent_span.span_id
        else:
            parent_id = self._span_stack[-1].span_id if self._span_stack else None

        span = Span(
            trace_id=self._trace_id,
            parent_span_id=parent_id,
            name=name,
            kind=kind,
            attributes=attrs,
        )

        if parent_id and parent_id in self._span_depth_map:
            self._span_depth_map[span.span_id] = self._span_depth_map[parent_id] + 1
        else:
            self._span_depth_map[span.span_id] = 0

        self._span_stack.append(span)
        return span

    def _finish_span(self, span: Span) -> None:
        indent = self._span_depth_map.get(span.span_id, 0)
        span.finish()
        self._all_spans.append(span)
        if span in self._span_stack:
            self._span_stack.remove(span)
        self._cexport_one(span, indent=indent)

    def _export_all(self) -> None:
        if not self._json_enabled or not self._all_spans:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exporter = JsonFileExporter(path=f"{self._trace_dir}/{ts}.jsonl")
        exporter.export(self._all_spans)

    def _finish_until(self, target: Span) -> None:
        if target not in self._span_stack:
            return
        self._finish_span(target)

    def _begin_execution(self) -> None:
        self._depth += 1
        if self._depth == 1:
            self._trace_id = uuid.uuid4().hex
            self._root_span = None
            self._span_stack = []
            self._run_span_stack = []
            self._pipeline_span_stack = []
            self._dag_span_stack = []
            self._team_span_stack = []
            self._dag_node_span_stack = []
            self._subagent_spans = {}
            self._all_spans = []
            self._span_depth_map = {}

    def _end_execution(self) -> None:
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0:
            self._export_all()

    async def on_run_start(self, agent_name: str, input_text: str) -> None:
        self._begin_execution()
        truncated_input = truncate_text_by_tokens(
            input_text, max_tokens=_TRACE_TEXT_MAX_TOKENS, suffix="",
        )
        orchestration_parent = _active_orchestration_parent.get(None)
        run_span = self._new_span(
            f"agent.{agent_name}", kind="agent",
            parent_span=orchestration_parent, input=truncated_input,
        )
        self._run_span_stack.append(run_span)
        _active_run_span.set(run_span)
        self._root_span = run_span
        self._cprint_start(
            kind=run_span.kind, name=run_span.name,
            indent=self._span_depth_map.get(run_span.span_id, 0),
            attributes={"input": truncated_input},
        )

    async def on_step_start(self, agent_name: str, step: int) -> None:
        run_span = _active_run_span.get(None)
        step_span = self._new_span(
            f"step.{step}", kind="internal", parent_span=run_span, agent=agent_name,
        )
        _active_step_span.set(step_span)
        indent = self._span_depth_map.get(step_span.span_id, 0)
        self._cwrite(f"{'  ' * indent}⟳ step.{step}\n")

    async def on_step_end(self, agent_name: str, step: int) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        step_span = _active_step_span.get(None)
        if step_span:
            step_span.finish()
            self._all_spans.append(step_span)
            if step_span in self._span_stack:
                self._span_stack.remove(step_span)
            indent = self._span_depth_map.get(step_span.span_id, 0)
            duration = (
                f"{step_span.duration_ms:.1f}ms"
                if step_span.duration_ms is not None else "..."
            )
            self._cwrite(f"{'  ' * indent}✓ {step_span.name} ({duration})\n")
            _active_step_span.set(None)

    async def on_llm_call(self, agent_name: str, messages: list[Any]) -> None:
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            active.add_event("llm_call", agent=agent_name, message_count=len(messages))
            indent = self._span_depth_map.get(active.span_id, 0) + 1
            self._cwrite(
                f"{'  ' * indent}• llm_call "
                f"{{agent={agent_name}, message_count={len(messages)}}}\n"
            )

    async def on_llm_stream_delta(self, agent_name: str, delta: StreamDelta) -> None:
        if delta.chunk.delta_content:
            if not _streaming_active.get(False):
                _streaming_active.set(True)
                step_span = _active_step_span.get(None)
                depth = (
                    self._span_depth_map.get(step_span.span_id, 0) + 1
                    if step_span else 0
                )
                self._cwrite("  " * depth + "▸ ")
            self._cwrite(delta.chunk.delta_content)

    async def on_llm_retry(self, agent_name: str, info: LLMRetryInfo) -> None:
        pass

    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            details = _summarize_tool_call(tool_call)
            active.add_event("tool_call", agent=agent_name, **details)
            indent = self._span_depth_map.get(active.span_id, 0) + 1
            self._cwrite(
                f"{'  ' * indent}• tool_call "
                f"{{agent={agent_name}, tool={details['tool']}, args={details['args']}}}\n"
            )

    async def on_tool_result(self, agent_name: str, result: Any) -> None:
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            output = result.content if hasattr(result, "content") else str(result)
            truncated = truncate_text_by_tokens(
                output, max_tokens=_TRACE_TEXT_MAX_TOKENS, suffix="",
            )
            active.add_event("tool_result", agent=agent_name, output=truncated)
            indent = self._span_depth_map.get(active.span_id, 0) + 1
            self._cwrite(
                f"{'  ' * indent}• tool_result {{agent={agent_name}, output={truncated}}}\n"
            )

    async def on_compression_start(self, agent_name: str) -> None:
        if _streaming_active.get(False):
            print("", file=sys.stderr)
            _streaming_active.set(False)
        indent = self._depth
        self._cwrite(
            f"{'  ' * indent}{ICONS['summary']} Compressing context...\n"
        )

    async def on_self_heal(self, agent_name: str, summary: str) -> None:
        if _streaming_active.get(False):
            print("", file=sys.stderr)
            _streaming_active.set(False)
        indent = self._depth
        self._cwrite(f"{'  ' * indent}{ICONS['heal']} {summary}\n")

    async def on_compression_end(
        self,
        agent_name: str,
        original_count: int,
        compressed_count: int,
        summary_tokens: int,
    ) -> None:
        indent = self._depth
        self._cwrite(
            f"{'  ' * indent}{ICONS['summary']} Context compressed: "
            f"{original_count} messages → {compressed_count} "
            f"(summary: ~{summary_tokens} tokens)\n"
        )

    async def on_approval_request(
        self, agent_name: str, request: ApprovalRequest
    ) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        active = _active_step_span.get(None) or _active_run_span.get(None)
        indent = self._span_depth_map.get(active.span_id, 0) + 1 if active else 1
        icon = ICONS.get("pending", "")
        self._cwrite(
            f"{'  ' * indent}{icon} approval_request tool={request.tool_call.name}\n"
        )

    async def on_approval_result(
        self, agent_name: str, result: ApprovalResult
    ) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        active = _active_step_span.get(None) or _active_run_span.get(None)
        indent = self._span_depth_map.get(active.span_id, 0) + 1 if active else 1
        icon_map = {
            "allow_once": ICONS.get("check", ""),
            "allow_session": ICONS.get("double_check", ""),
            "deny": ICONS.get("cross", ""),
        }
        icon = icon_map.get(result.decision, "?")
        self._cwrite(
            f"{'  ' * indent}{icon} approval={result.decision}"
            f" tool={result.tool_name or result.tool_call_id}\n"
        )

    async def on_error(self, agent_name: str, error: Exception) -> None:
        if _streaming_active.get(False):
            self._cwrite("\n")
            _streaming_active.set(False)
        step_span = _active_step_span.get(None)
        if step_span:
            step_span.finish()
            self._all_spans.append(step_span)
            if step_span in self._span_stack:
                self._span_stack.remove(step_span)
            indent = self._span_depth_map.get(step_span.span_id, 0)
            duration = (
                f"{step_span.duration_ms:.1f}ms"
                if step_span.duration_ms is not None else "..."
            )
            self._cwrite(f"{'  ' * indent}✗ {step_span.name} ({duration})\n")
            _active_step_span.set(None)
        run_span = _active_run_span.get(None)
        _active_run_span.set(None)
        if run_span:
            run_span.set_error(error)
            self._finish_span(run_span)
            if run_span in self._run_span_stack:
                self._run_span_stack.remove(run_span)
        self._root_span = self._run_span_stack[-1] if self._run_span_stack else None
        self._end_execution()

    async def on_run_end(self, agent_name: str, output: str) -> None:
        run_span = _active_run_span.get(None)
        _active_run_span.set(None)
        if run_span:
            run_span.attributes["output"] = truncate_text_by_tokens(
                output, max_tokens=_TRACE_TEXT_MAX_TOKENS, suffix="",
            )
            self._finish_span(run_span)
            if run_span in self._run_span_stack:
                self._run_span_stack.remove(run_span)
        self._root_span = self._run_span_stack[-1] if self._run_span_stack else None
        self._end_execution()

    async def on_pipeline_start(self, pipeline_name: str) -> None:
        self._begin_execution()
        span = self._new_span(f"pipeline.{pipeline_name}", kind="orchestration")
        span.attributes["orchestration"] = "pipeline"
        self._pipeline_span_stack.append(span)
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind, name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"orchestration": "pipeline"},
        )

    async def on_pipeline_end(self, pipeline_name: str) -> None:
        _active_orchestration_parent.set(None)
        span = self._pipeline_span_stack.pop() if self._pipeline_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_dag_start(self, dag_name: str) -> None:
        self._begin_execution()
        span = self._new_span(f"dag.{dag_name}", kind="orchestration")
        span.attributes["orchestration"] = "dag"
        self._dag_span_stack.append(span)
        self._cprint_start(
            kind=span.kind, name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"orchestration": "dag"},
        )

    async def on_dag_end(self, dag_name: str) -> None:
        span = self._dag_span_stack.pop() if self._dag_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_dag_node_start(self, node_id: str) -> None:
        dag_span = self._dag_span_stack[-1] if self._dag_span_stack else None
        span = self._new_span(
            f"dag_node.{node_id}", kind="orchestration",
            parent_span=dag_span, node=node_id,
        )
        self._dag_node_span_stack.append(span)
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind, name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"node": node_id},
        )

    async def on_dag_node_end(self, node_id: str) -> None:
        _active_orchestration_parent.set(None)
        span = self._dag_node_span_stack.pop() if self._dag_node_span_stack else None
        if span:
            self._finish_span(span)

    async def on_team_start(self, team_name: str, mode: str) -> None:
        self._begin_execution()
        span = self._new_span(f"team.{team_name}", kind="orchestration")
        span.attributes["orchestration"] = "team"
        span.attributes["mode"] = mode
        self._team_span_stack.append(span)
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind, name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"orchestration": "team", "mode": mode},
        )

    async def on_todo_update(
        self,
        agent_name: str,
        todos: list[Any],
        stats: dict[str, int],
    ) -> None:
        active = _active_step_span.get(None) or _active_run_span.get(None)
        if active:
            active.add_event(
                "todo_update",
                agent=agent_name,
                total=stats.get("total", 0),
                completed=stats.get("completed", 0),
                in_progress=stats.get("in_progress", 0),
            )
        indent = (
            self._span_depth_map.get(active.span_id, 0) + 1
            if active
            else 1
        )
        total = stats.get("total", 0)
        completed = stats.get("completed", 0)
        current = next(
            (
                t.get("content", "")
                for t in todos
                if t.get("status") == "in_progress"
            ),
            "",
        )
        label = f"progress={completed}/{total}"
        if current:
            label += f", current={current}"
        self._cwrite(f"{'  ' * indent}* todo_update {{{label}}}\n")

    async def on_team_end(self, team_name: str, mode: str) -> None:
        _active_orchestration_parent.set(None)
        span = self._team_span_stack.pop() if self._team_span_stack else None
        if span:
            self._finish_until(span)
        self._end_execution()

    async def on_subagent_start(
        self, parent_name: str, subagent_name: str,
        agent_type: str, description: str, prompt: str,
    ) -> None:
        truncated_prompt = truncate_text_by_tokens(
            prompt, max_tokens=_TRACE_TEXT_MAX_TOKENS, suffix="",
        )
        step_span = _active_step_span.get(None)
        span = self._new_span(
            f"subagent.{subagent_name}", kind="subagent",
            parent_span=step_span,
            agent_type=agent_type, description=description,
            prompt=truncated_prompt,
        )
        self._subagent_spans[subagent_name] = span
        _active_orchestration_parent.set(span)
        self._cprint_start(
            kind=span.kind, name=span.name,
            indent=self._span_depth_map.get(span.span_id, 0),
            attributes={"agent_type": agent_type, "description": description},
        )

    async def on_subagent_end(
        self, parent_name: str, subagent_name: str,
        agent_type: str, description: str,
        steps: int, tool_calls: int, duration_ms: float,
    ) -> None:
        _active_orchestration_parent.set(None)
        span = self._subagent_spans.pop(subagent_name, None)
        if span:
            span.attributes["steps"] = steps
            span.attributes["tool_calls"] = tool_calls
            span.attributes["duration_ms"] = duration_ms
            self._finish_span(span)
