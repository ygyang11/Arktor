"""/feedback — open a prefilled GitHub issue for maintainer feedback."""
from __future__ import annotations

import platform
import webbrowser
from urllib.parse import quote

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import ok
from agent_harness import __version__ as _HARNESS_VERSION

_REPO = "ygyang11/Agent-Harness"


def _build_body(ctx: CommandContext) -> str:
    return (
        "<!-- Describe your feedback above this line -->\n\n\n\n\n"
        "---\n\n"
        f"- Version: {_HARNESS_VERSION}\n"
        f"- Platform: {platform.system()} {platform.release()}\n"
        f"- Python: {platform.python_version()}\n"
        f"- Model: {ctx.agent.llm.model_name}\n"
    )


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    title = args.strip() or "Problem-Feedback"
    body = _build_body(ctx)
    url = (
        f"https://github.com/{_REPO}/issues/new"
        f"?title={quote(title)}&body={quote(body)}"
    )
    webbrowser.open(url)
    return CommandResult(output=ok(
        ("Feedback form ", ""), (url, "primary"),
    ))


CMD = Command(
    name="/feedback",
    description="Submit feedback to issues",
    handler=_handler,
)
