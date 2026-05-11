"""builtin command registration."""
from __future__ import annotations

from agent_cli.commands.builtin import (
    clear,
    compact,
    context,
    copy,
    debug,
    exit,
    help,
    model,
    new,
    permissions,
    resume,
    status,
    tasks,
    theme,
    usage,
)
from agent_cli.commands.registry import CommandRegistry


def register_builtin(registry: CommandRegistry) -> None:
    for module in (
        help, exit, clear, compact, context, copy, debug, model, permissions,
        resume, new, status, tasks, theme, usage,
    ):
        registry.register_command(module.CMD)
