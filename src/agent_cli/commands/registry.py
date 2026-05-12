"""CLI command registry with slash/plain-alias dispatch rules."""
from __future__ import annotations

import os
import re

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_harness.core.registry import Registry

SLASH_CMD_RE = re.compile(r"^/[a-zA-Z][\w-]*$")


class CommandRegistry(Registry[Command]):
    """Registry of ``Command`` objects keyed by canonical name and aliases."""

    def register_command(self, cmd: Command) -> None:
        self.register(cmd.name.lower(), cmd)
        for alias in cmd.aliases:
            self.register(alias.lower(), cmd)

    async def dispatch(self, line: str, ctx: CommandContext) -> CommandResult | None:
        name, _, args = line.partition(" ")
        key = name.lower()

        if self.has(key):
            if not name.startswith("/") and args:
                return None
            return await self.get(key).handler(ctx, args.strip())

        if name.startswith("/"):
            if not SLASH_CMD_RE.match(name):
                return None
            if os.path.exists(name):
                return None
            return CommandResult(output=f"Unknown command: {name}. Type /help.")
        return None

    def get_completions(self) -> list[tuple[str, str]]:
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for name in self.list_names():
            cmd = self.get(name)
            if cmd.hidden or cmd.name in seen:
                continue
            seen.add(cmd.name)
            out.append((cmd.name, cmd.description))
        return out

    def unregister_skills(self) -> None:
        for key in [k for k, c in list(self._items.items()) if c.is_skill]:
            del self._items[key]

    def list_skill_commands(self) -> list[Command]:
        seen: set[str] = set()
        out: list[Command] = []
        for cmd in self._items.values():
            if cmd.is_skill and cmd.name not in seen:
                seen.add(cmd.name)
                out.append(cmd)
        return out
