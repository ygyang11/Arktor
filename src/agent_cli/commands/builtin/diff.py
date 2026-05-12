"""/diff — show uncommitted changes (staged + unstaged)."""
from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.builtin._git import run as _run_git
from agent_cli.commands.ui import bar_heading, err, soft


def _render_diff(status_out: str, diff_out: str) -> RenderableType:
    rows: list[RenderableType] = []
    status_lines = [line for line in status_out.splitlines() if line]
    if status_lines:
        rows.append(bar_heading("Files"))
        rows.append(Text(""))
        for line in status_lines:
            rows.append(Text(f"  {line}", style="muted"))
    if diff_out:
        if status_lines:
            rows.append(Text(""))
        rows.append(bar_heading("Diff"))
        rows.append(Text(""))
        rows.append(Syntax(
            diff_out, "diff", theme="ansi_dark",
            background_color="default", word_wrap=False,
        ))
    return Panel(
        Group(*rows),
        title="Uncommitted changes",
        title_align="left",
        border_style="muted",
        padding=(1, 1),
        expand=False,
    )


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    rc, diff_out, diff_err = await _run_git("diff", "--color=never")
    if rc != 0:
        return CommandResult(output=err(
            ("git diff failed: ", ""),
            (diff_err.strip() or f"exit {rc}", "warning"),
        ))
    rc, staged_out, staged_err = await _run_git("diff", "--cached", "--color=never")
    if rc != 0:
        return CommandResult(output=err(
            ("git diff --cached failed: ", ""),
            (staged_err.strip() or f"exit {rc}", "warning"),
        ))
    rc, status_out, status_err = await _run_git("status", "--short", timeout=5.0)
    if rc != 0:
        return CommandResult(output=err(
            ("git status failed: ", ""),
            (status_err.strip(), "warning"),
        ))
    combined = "\n".join(filter(None, (staged_out.rstrip(), diff_out.rstrip())))
    if not combined and not status_out:
        return CommandResult(output=soft(("Working tree clean", "")))
    return CommandResult(output=_render_diff(status_out, combined))


CMD = Command(
    name="/diff",
    description="Show uncommitted changes",
    handler=_handler,
)
