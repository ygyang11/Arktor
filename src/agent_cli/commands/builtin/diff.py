"""/diff — show uncommitted changes (staged + unstaged)."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.builtin._git import run as _run_git
from agent_cli.commands.ui import err, render_diff, soft


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
    return CommandResult(output=render_diff(status_out, combined))


CMD = Command(
    name="/diff",
    description="Show uncommitted changes",
    handler=_handler,
)
