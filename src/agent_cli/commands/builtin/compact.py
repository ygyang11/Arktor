"""/compact — manually trigger conversation compression."""
from __future__ import annotations

from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import home_relative_path, ok, soft
from agent_cli.render.status_lines import make_command_status_line
from agent_cli.runtime import session as sess


def _compacted_output(res: object) -> Text:
    """Render ``Compacted: N → M msgs · K archived to ~/...``."""
    archived = res.original_count - res.compressed_count
    detail = ""
    if archived > 0 and res.archive_path:
        detail = f" · {archived} archived to {home_relative_path(res.archive_path)}"
    return ok(
        "Compacted: ",
        (str(res.original_count), "bold"),
        " → ",
        (str(res.compressed_count), "bold"),
        " msgs",
        (detail, "muted"),
    )


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    agent = ctx.agent
    compressor = agent.context.short_term_memory.compressor
    if compressor is None:
        return CommandResult(output=soft(
            "Compression not enabled (memory.strategy != 'summarize')",
        ))

    extra = args.strip() or None
    messages = sess.get_messages(agent)

    status = make_command_status_line(
        ctx.adapter.console, ctx.adapter.lock(), ctx.adapter.theme,
        label="Compacting", color="info",
    )
    await status.start()
    try:
        new_messages = await compressor.compress(messages, extra_instructions=extra)
    finally:
        await status.stop()

    sess.set_messages(agent, new_messages)

    res = compressor.take_last_result()
    if res is not None and res.llm_usage and res.llm_usage.total_tokens:
        agent.context.usage_meter.record(
            res.llm_usage,
            model=compressor.model_name,
            source="compressor",
        )

    await ctx.save_session()

    if res is None:
        return CommandResult(output=soft(
            "Nothing to compact yet — conversation is still too short",
        ))
    return CommandResult(output=_compacted_output(res))


CMD = Command(
    name="/compact",
    description=(
        "Compact conversation history and keep a summary in context"
        " (accepts optional extra instructions)"
    ),
    handler=handle,
)
