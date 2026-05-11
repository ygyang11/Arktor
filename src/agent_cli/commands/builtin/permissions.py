"""/permissions — view or switch approval mode (ask / auto / yolo)."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import MODE_INFO, err, ok, render_permissions_panel
from agent_cli.runtime import session as sess_rt

_INPUT_TO_INTERNAL: dict[str, str] = {
    "ask": "ask",
    "auto": "auto",
    "yolo": "never",
    "never": "never",
}


async def _handler(ctx: CommandContext, args: str) -> CommandResult:
    policy = sess_rt.get_policy(ctx.agent)
    raw = args.strip().lower()
    if not raw:
        return CommandResult(output=render_permissions_panel(policy))
    if raw not in _INPUT_TO_INTERNAL:
        return CommandResult(output=err(
            ("Unknown mode: ", ""), (raw, "warning"),
            (". Use one of: ask / auto / yolo", ""),
        ))
    sess_rt.apply_mode(ctx.agent, _INPUT_TO_INTERNAL[raw])
    await ctx.save_session()
    return CommandResult(output=ok(
        ("Permission mode → ", ""),
        (MODE_INFO[policy.mode].label, "primary"),
    ))


CMD = Command(
    name="/permissions",
    description="View or switch approval mode (ask / auto / yolo)",
    handler=_handler,
)
