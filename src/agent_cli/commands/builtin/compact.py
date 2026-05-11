"""/compact — manually trigger conversation compression."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import ok, soft
from agent_cli.runtime import session as sess


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    agent = ctx.agent
    compressor = agent.context.short_term_memory.compressor
    if compressor is None:
        return CommandResult(output=soft(
            "Compression not enabled (memory.strategy != 'summarize')",
        ))

    extra = args.strip() or None
    messages = sess.get_messages(agent)
    new_messages = await compressor.compress(messages, extra_instructions=extra)
    sess.set_messages(agent, new_messages)

    res = compressor.take_last_result()
    if res is not None and res.llm_usage and res.llm_usage.total_tokens:
        agent.context.usage_meter.record(
            res.llm_usage,
            model=compressor.model_name,
            source="compressor",
        )

    await ctx.save_session()

    if res is not None:
        return CommandResult(output=ok(
            "Compacted: ",
            (str(res.original_count), "bold"),
            " → ",
            (str(res.compressed_count), "bold"),
            " msgs",
            (f" (~{res.summary_tokens} tokens)", "muted"),
        ))
    return CommandResult(output=soft(
        "Nothing to compact yet — conversation is still too short",
    ))


CMD = Command(
    name="/compact",
    description=(
        "Compact conversation history and keep a summary in context"
        " (accepts optional extra instructions)"
    ),
    handler=handle,
)
