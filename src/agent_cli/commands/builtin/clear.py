"""/clear — reset the session runtime in-place (session_id unchanged)."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, ok
from agent_cli.runtime import background
from agent_cli.runtime import session as sess


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    agent = ctx.agent

    # Phase A — cancellable external cleanup. A Ctrl+C here aborts before
    # any in-memory state is touched.
    await background.shutdown(agent)
    ctx.approval_handler.cancel_pending()
    sandbox_err: str | None = None
    try:
        await sess.stop_sandbox(agent)
    except Exception as e:
        sandbox_err = f"{type(e).__name__}: {e}"

    # Phase B — fast commit. Only save_session is real I/O
    background.clear_tasks(agent)
    await agent.context.short_term_memory.clear()
    await agent.context.working_memory.clear()
    agent.context.variables._agent_store.clear()
    agent.context.variables._global_store.clear()

    sess.reset_stateful_tools(agent)

    sess.reset_approval(agent)
    # Ctrl+C can leave state stuck in non-terminal; next run()'s
    # transition(THINKING) would raise StateTransitionError.
    agent.context.state.reset()

    compressor = agent.context.short_term_memory.compressor
    if compressor is not None:
        compressor.restore_runtime_state([])
    await ctx.save_session()
    
    if sandbox_err is not None:
        return CommandResult(output=err(
            "Context cleared",
            (f" · sandbox stop failed: {sandbox_err}", "muted"),
        ))
    return CommandResult(output=ok(
        "Context cleared",
        (" · session reset", "muted"),
    ))


CMD = Command(
    name="/clear",
    description="Clear context and reset session state",
    handler=handle,
)
