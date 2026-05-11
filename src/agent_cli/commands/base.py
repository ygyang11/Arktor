"""Command types and context for the CLI command system."""
from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import RenderableType

from agent_cli.runtime.session import SaveSession

if TYPE_CHECKING:
    from agent_cli.approval_handler import CliApprovalHandler
    from agent_cli.commands.registry import CommandRegistry
    from agent_harness.agent.base import BaseAgent
    from agent_harness.session.base import BaseSession


@dataclass
class CommandContext:
    agent: BaseAgent
    session_id: str
    registry: CommandRegistry
    save_session: SaveSession
    approval_handler: CliApprovalHandler
    session_backend: BaseSession


@dataclass
class CommandResult:
    output: RenderableType | None = None
    agent_input: str | None = None
    should_exit: bool = False
    new_session_id: str | None = None


Handler = Callable[[CommandContext, str], Coroutine[Any, Any, CommandResult]]


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    description: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    hidden: bool = False
