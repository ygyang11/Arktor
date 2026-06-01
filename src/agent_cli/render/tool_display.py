"""ToolDisplay — Rich.Live multi-row status table + post-batch expansion."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape as rich_escape
from rich.padding import Padding
from rich.text import Text

from agent_cli.theme import (
    CONTINUATION,
    DENIED,
    LEFT_BAR,
    SEP_DOT,
    SEP_ELLIPSIS,
    TASKS_HEADER,
    TODO_IN_PROGRESS,
    TODO_PENDING,
    TOOL_DONE,
)
from agent_harness.approval.types import ApprovalResult
from agent_harness.core.message import Attachment, ToolCall, ToolResult
from agent_harness.utils.media import media_safe_filename

RowStatus = Literal["running", "done", "error", "denied"]


@dataclass
class _ToolRow:
    id: str
    name: str
    args_preview: str
    status: RowStatus = "running"
    tool_call: ToolCall | None = None
    result: ToolResult | None = None
    reason: str = ""


ResultFormatter = Callable[[ToolCall, ToolResult], str]
Expander = Callable[[Console, _ToolRow], None]

_RESULT_FORMATTERS: dict[str, ResultFormatter] = {}
_EXPANDERS: dict[str, Expander] = {}

SUPPRESSED_IN_ROW: frozenset[str] = frozenset({"sub_agent", "todo_write"})


def register_result_formatter(tool_name: str) -> Callable[[ResultFormatter], ResultFormatter]:
    def wrap(fn: ResultFormatter) -> ResultFormatter:
        _RESULT_FORMATTERS[tool_name] = fn
        return fn

    return wrap


def register_expander(tool_name: str) -> Callable[[Expander], Expander]:
    def wrap(fn: Expander) -> Expander:
        _EXPANDERS[tool_name] = fn
        return fn

    return wrap


def _truncate(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 1] + SEP_ELLIPSIS


def _is_error_result(r: ToolResult | None) -> bool:
    if r is None:
        return False
    if r.is_error:
        return True
    return bool(r.content) and r.content.startswith("Error:")


_PRIMARY_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "read_file": ("file_path",),
    "write_file": ("file_path",),
    "edit_file": ("file_path",),
    "document_parser": ("target",),
    "list_dir": ("path",),
    "glob_files": ("pattern", "path"),
    "grep_files": ("pattern", "path", "include"),
    "terminal_tool": ("command",),
    "web_fetch": ("url",),
    "web_search": ("query",),
    "paper_search": ("query",),
    "paper_fetch": ("paper_id",),
    "memory_tool": ("action", "name"),
    "skill_tool": ("skill_name",),
    "background_task": ("action", "task_id"),
}

_DISPLAY_NAMES: dict[str, str] = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Update",
    "list_dir": "List",
    "glob_files": "Glob",
    "grep_files": "Grep",
    "terminal_tool": "Bash",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
    "paper_search": "PaperSearch",
    "paper_fetch": "PaperFetch",
    "memory_tool": "Memory",
    "skill_tool": "Skill",
    "background_task": "Task",
    "document_parser": "Document",
}


def _display_name(raw: str) -> str:
    """Human verb for ``raw``; unregistered tools fall back to the raw name."""
    return _DISPLAY_NAMES.get(raw, raw)


def args_repr(arguments: dict[str, object]) -> str:
    """Full JSON repr of tool arguments — for transcript/export, no truncation."""
    return json.dumps(arguments, ensure_ascii=False)


def _args_preview(name: str, arguments: dict[str, object]) -> str:
    keys = _PRIMARY_ARG_KEYS.get(name)
    if keys is not None:
        values: list[object] = [
            arguments[k] for k in keys if k in arguments and arguments[k] not in (None, "")
        ]
    else:
        values = [v for v in arguments.values() if v not in (None, "")]
    parts = []
    for v in values:
        flat = str(v).replace("\n", " ").replace("\r", " ").replace("\t", " ")
        if name == "terminal_tool":
            flat = _truncate(flat, 60)
        parts.append(flat)
    return ", ".join(parts)


def _generic_result(tc: ToolCall, r: ToolResult) -> str:
    if not r.content:
        return ""
    return "Done"


class ToolDisplay:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: Live | None = None
        self._rows: list[_ToolRow] = []
        self._suspended: bool = False

    def _open_live(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._render(),
            console=self._console,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        self._live.refresh()

    def _refresh(self) -> None:
        if self._suspended:
            return
        if self._live is None:
            self._open_live()
        else:
            self._live.update(self._render(), refresh=True)

    def _close_live(self) -> None:
        if self._live is None:
            return
        self._live.__exit__(None, None, None)
        self._live = None

    def add_call(self, tool_call: ToolCall) -> None:
        row = _ToolRow(
            id=tool_call.id,
            name=tool_call.name,
            args_preview=_args_preview(tool_call.name, tool_call.arguments),
            tool_call=tool_call,
        )
        self._rows.append(row)
        self._refresh()

    def add_denied(self, approval_result: ApprovalResult) -> None:
        row = _ToolRow(
            id=approval_result.tool_call_id,
            name=approval_result.tool_name or "?",
            args_preview="",
            status="denied",
            reason=(approval_result.reason or "denied by user"),
        )
        self._rows.append(row)
        self._refresh()

    def mark_result(self, result: ToolResult) -> None:
        for row in self._rows:
            if row.id == result.tool_call_id:
                row.status = "error" if _is_error_result(result) else "done"
                row.result = result
                self._refresh()
                return
        # No matching row (e.g., suppressed tool like sub_agent/todo_write
        # whose add_call was skipped). Skip refresh to avoid opening an
        # empty Live that would collide with the next markdown phase.

    def pause(self) -> None:
        self._close_live()

    def suspend(self) -> None:
        """Vacate the live region for a side channel (subagent line / inline
        print) WITHOUT committing rows. Rows stay pending and keep updating
        their status silently; end() flushes them — resolved — at end_step.

        Tradeoff: a sibling row still running when suspended shows no live
        spinner until end_step (rare: a normal tool co-batched with a sub_agent).
        """
        self._close_live()
        self._suspended = True

    def end(self) -> None:
        self._close_live()
        self._suspended = False
        if not self._rows:
            return
        for i, row in enumerate(self._rows):
            if i > 0:
                self._console.print()
            self._console.print(_format_call_line(row))
            result_line = _format_result_line(row)
            if result_line is not None:
                self._console.print(result_line)
            if row.status == "done" and row.name in _EXPANDERS:
                _EXPANDERS[row.name](self._console, row)
        self._console.print()
        self._rows.clear()

    def show_todos(self, todos: list[dict[str, str]], stats: dict[str, int]) -> None:
        print_todos_panel(self._console, todos, stats)

    def _render(self) -> RenderableType:
        rows: list[RenderableType] = []
        for i, r in enumerate(self._rows):
            if i > 0:
                rows.append(Text(""))
            rows.append(_format_call_line(r))
            line = _format_result_line(r)
            if line is not None:
                rows.append(line)
        return Group(*rows)


def _format_call_line(row: _ToolRow) -> RenderableType:
    if row.status == "running":
        glyph, style = TOOL_DONE, "muted"
    elif row.status == "done":
        glyph, style = TOOL_DONE, "success"
    elif row.status == "error":
        glyph, style = TOOL_DONE, "error"
    else:
        glyph, style = DENIED, "error"

    t = Text()
    t.append(f"{glyph} ", style=style)
    t.append(_display_name(row.name), style="bold")
    if row.args_preview:
        t.append(f"({row.args_preview})", style="muted")
    return t


def _format_result_line(row: _ToolRow) -> Text | None:
    if row.status == "running":
        return None
    if row.name in SUPPRESSED_IN_ROW:
        return None

    if row.status == "denied":
        summary = f"Denied {SEP_DOT} Do not run this tool."
        value_style = "error"
    elif _is_error_result(row.result):
        content = row.result.content if row.result is not None else ""
        first_line = content.split("\n", 1)[0].strip()
        if first_line.startswith("Error:"):
            summary = _truncate(first_line, 72)
        else:
            summary = f"Error: {_truncate(first_line, 56)}"
        value_style = "error"
    else:
        fmt = _RESULT_FORMATTERS.get(row.name, _generic_result)
        if row.tool_call is not None and row.result is not None:
            summary = fmt(row.tool_call, row.result)
        else:
            summary = ""
        value_style = "muted"

    if not summary:
        return None
    line = Text()
    line.append(f"{CONTINUATION}  ", style="muted")
    line.append(summary, style=value_style)
    return line


def _summarize_result(tc: ToolCall, tr: ToolResult) -> str:
    if tc.name == "read_file" and tr.content:
        if tr.content.startswith(("Empty file", "(empty")):
            return tr.content.split(chr(10), 1)[0]
        from agent_cli.render.tool_formatters import _READ_HEADER_RE  # noqa: PLC0415

        m = _READ_HEADER_RE.match(tr.content)
        if m:
            start, end, total = int(m[2]), int(m[3]), int(m[4])
            shown = end - start + 1
            if shown == total:
                return f"{total} lines"
            return f"lines {start}-{end} of {total}"
        return ""
    if tc.name == "list_dir" and tr.content:
        from agent_cli.render.tool_formatters import _LIST_HEADER_RE  # noqa: PLC0415

        m = _LIST_HEADER_RE.search(tr.content)
        if m:
            return f"{m[2]} entries"
    return ""


def format_attachments(items: list[dict[str, Any]]) -> list[RenderableType]:
    """Renderables for an attachment row: a 'Loaded into context' header
    followed by one continuation-glyph row per attachment. Handles both the
    text-tool shape (:func:`attachment_summary`) and the media shape
    (:func:`media_attachment_summary`)."""
    from agent_harness.utils.media import human_size  # noqa: PLC0415

    rows: list[RenderableType] = []
    header = Text()
    header.append(f"{TOOL_DONE}  ", style="primary")
    header.append("Loaded into context", style="muted")
    rows.append(header)
    for it in items:
        line = Text()
        line.append(f"{CONTINUATION}  ", style="muted")
        if it.get("kind") == "media":
            label = "PDF" if it.get("mime") == "application/pdf" else "Image"
            target = str(it.get("filename") or it.get("mime") or "")
            line.append(f"{label} ", style="muted")
            line.append(target, style="muted bold")
            size = it.get("size")
            if isinstance(size, int) and size > 0:
                line.append(f" ({human_size(size)})", style="muted")
            rows.append(line)
            continue
        label = _DISPLAY_NAMES.get(it["tool_name"], it["tool_name"])
        args = it.get("arguments", {})
        target = str(args.get("file_path") or args.get("path") or "")
        if it["is_error"]:
            line.append(f"{label} ", style="error")
            line.append(target, style="error bold")
            snip = it.get("error_snippet") or ""
            if snip:
                line.append(f" {SEP_DOT} ", style="error")
                line.append(snip, style="error")
        else:
            line.append(f"{label} ", style="muted")
            line.append(target, style="muted bold")
            summary = it.get("summary") or ""
            if summary:
                line.append(f" ({summary})", style="muted")
        rows.append(line)
    return rows


def media_attachment_summary(att: Attachment) -> dict[str, Any]:
    return {
        "kind": "media",
        "mime": att.mime,
        "filename": media_safe_filename(att.filename, att.mime),
        "size": att.size,
        "is_error": False,
    }


def attachment_summary(tc: ToolCall, tr: ToolResult) -> dict[str, Any]:
    """Single encode point. Persisted to ``message.metadata['attachments']``;
    consumed by :func:`format_attachments` for both the live path (built
    fresh) and replay (read back from metadata) — same dict shape."""
    err = _is_error_result(tr)
    return {
        "tool_name": tc.name,
        "arguments": dict(tc.arguments),
        "summary": _summarize_result(tc, tr),
        "is_error": err,
        "error_snippet": (
            _truncate((tr.content or "").replace("\n", " "), 60) if err else ""
        ),
    }


def format_shell_run(
    command: str,
    exit_code: int,
    output: str,
) -> list[RenderableType]:
    glyph_style = "success" if exit_code == 0 else "error"

    call = Text()
    call.append(f"{TOOL_DONE} ", style=glyph_style)
    call.append("Run", style="bold")
    call.append(f"({command})", style="muted")

    body = Text()
    body.append(f"{CONTINUATION}  ", style="muted")
    if output.strip():
        lines = output.splitlines() or [output]
        body.append(lines[0], style="muted")
        for extra in lines[1:]:
            body.append("\n")
            body.append("   " + extra, style="muted")
    else:
        body.append("(Completed with no output)", style="muted")

    return [call, body]


def print_completed_call(
    console: Console,
    tc: ToolCall,
    tr: ToolResult | None,
    *,
    force_status: RowStatus | None = None,
) -> None:
    status: RowStatus
    if force_status is not None:
        status = force_status
    elif tr is None:
        status = "running"
    elif _is_error_result(tr):
        status = "error"
    else:
        status = "done"
    row = _ToolRow(
        id=tc.id,
        name=tc.name,
        args_preview=_args_preview(tc.name, tc.arguments),
        status=status,
        tool_call=tc,
        result=tr,
    )
    console.print(_format_call_line(row))
    line = _format_result_line(row)
    if line is not None:
        console.print(line)
    if row.status == "done" and row.name in _EXPANDERS:
        _EXPANDERS[row.name](console, row)


def print_todos_panel(
    console: Console,
    todos: list[dict[str, str]],
    stats: dict[str, int],
) -> None:
    total = stats.get("total", 0)
    done = stats.get("completed", 0)

    def _bar(body: Text) -> Text:
        row = Text(f"{LEFT_BAR} ", style="primary")
        row.append_text(body)
        return row

    def _bar_only() -> Text:
        return Text(LEFT_BAR, style="primary")

    rows: list[RenderableType] = [_bar_only()]
    rows.append(
        _bar(
            Text.from_markup(
                f"[primary]{TASKS_HEADER}[/primary] [bold]Tasks [{done}/{total}][/bold]"
            )
        )
    )
    for t in todos:
        status = t.get("status", "pending")
        content = rich_escape(t.get("content", ""))
        if status == "completed":
            body = Text.from_markup(
                f"  [success]{TOOL_DONE}[/success] [muted]{content}[/muted]"
            )
        elif status == "in_progress":
            body = Text.from_markup(f"  [accent]{TODO_IN_PROGRESS}[/accent] {content}")
        else:
            body = Text.from_markup(f"  [muted]{TODO_PENDING} {content}[/muted]")
        rows.append(_bar(body))
    rows.append(_bar_only())
    console.print()
    console.print(Padding(Group(*rows), (0, 1, 0, 0), style="section"))
    console.print()


def _todo_stats(todos: list[dict[str, object]]) -> dict[str, int]:
    """Reconstruct ``{total, pending, in_progress, completed}`` from raw todos."""
    counts = {"pending": 0, "in_progress": 0, "completed": 0}
    for t in todos:
        s = str(t.get("status", "pending"))
        if s in counts:
            counts[s] += 1
    return {"total": len(todos), **counts}

# Trigger formatter/expander registration via import side-effect.
from agent_cli.render import tool_formatters as _tool_formatters  # noqa: F401, E402
