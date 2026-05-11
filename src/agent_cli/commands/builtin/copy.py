"""/copy — copy last (or n-th-from-last) assistant reply to the clipboard."""
from __future__ import annotations

import base64
import subprocess
import sys

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok
from agent_harness.core.message import Role

_PLATFORM_TOOLS: dict[str, list[list[str]]] = {
    "darwin": [["pbcopy"]],
    "linux":  [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "-ib"],
        ["clip.exe"],
    ],
    "win32":  [["clip"]],
}

_INSTALL_HINTS: dict[str, str] = {
    "darwin": "pbcopy should be preinstalled on macOS",
    "linux": "install wl-copy (Wayland) or xclip / xsel (X11)",
    "win32": "clip should be preinstalled on Windows",
}


def _install_hint() -> str:
    return _INSTALL_HINTS.get(sys.platform, "no native clipboard tool detected")


def _native_copy(text: str) -> bool:
    for cmd in _PLATFORM_TOOLS.get(sys.platform, []):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    return False


def _osc52_copy(text: str) -> bool:
    try:
        b64 = base64.b64encode(text.encode()).decode()
        sys.stdout.write(f"\x1b]52;c;{b64}\x07")
        sys.stdout.flush()
        return True
    except Exception:
        return False


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    raw = args.strip()
    n = int(raw) if raw.isdigit() and int(raw) >= 1 else 1
    msgs = await ctx.agent.context.short_term_memory.get_context_messages()
    replies = [m for m in msgs if m.role == Role.ASSISTANT and m.content]
    if len(replies) < n:
        return CommandResult(output=err((f"Message #{n} not found", "")))
    target = replies[-n].content or ""

    if _native_copy(target) or _osc52_copy(target):
        label = (
            "Copied last message to clipboard"
            if n == 1
            else f"Copied message #{n} to clipboard"
        )
        return CommandResult(output=ok((label, "")))

    return CommandResult(output=err((
        f"Could not access clipboard — {_install_hint()}, "
        "or enable OSC 52 in your terminal", "",
    )))


CMD = Command(
    name="/copy",
    description="Copy the last assistant reply to the clipboard (/copy N for the N-th most recent)",
    handler=_handler,
)
