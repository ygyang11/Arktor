"""Short-term memory manipulation outside the normal agent.run flow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from agent_cli.render.notices import format_shell_run
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from agent_cli.runtime.session import SaveSession

_pending_writes: ContextVar[list[asyncio.Future[Any]] | None] = ContextVar(
    "_pending_writes", default=None,
)


def use_pending_tracker(tracker: list[asyncio.Future[Any]] | None) -> Token[Any]:
    return _pending_writes.set(tracker)


def reset_pending_tracker(token: Token[Any]) -> None:
    _pending_writes.reset(token)


async def drain_pending(pending: list[asyncio.Future[Any]]) -> None:
    """Await every queued shielded write and clear the list."""
    if not pending:
        return
    tasks = list(pending)
    pending.clear()
    await asyncio.gather(*tasks, return_exceptions=True)


async def append_tool_turn(
    agent: BaseAgent,
    pairs: list[tuple[ToolCall, ToolResult]],
    *,
    render: Callable[[list[tuple[ToolCall, ToolResult]]], Awaitable[None]] | None = None,
) -> None:
    """Append ``assistant(tool_calls) + tool×N`` in declaration order.

    Render runs first; if it raises, no memory write. Memory write is
    ``asyncio.shield``-ed: once started, completes even if the outer
    task is cancelled (so no orphan tool_calls survive). Cancel before
    the write phase leaves only the user message — also schema-valid.

    The inner write Task is registered into the current pending_writes
    tracker (if bound by the CLI _run via use_pending_tracker) so cancel-
    rollback can drain it before deciding commit state.
    """
    if not pairs:
        return

    if render is not None:
        await render(pairs)

    write_task = asyncio.ensure_future(_write(agent, tool_pairs=pairs))
    tracker = _pending_writes.get()
    if tracker is not None:
        tracker.append(write_task)

    await asyncio.shield(write_task)


async def append_shell_run(
    agent: BaseAgent,
    *,
    command: str,
    exit_code: int,
    output: str,
    save: SaveSession,
    post_notices: list[str] | None = None,
) -> None:
    """Inject a user-initiated shell run into short-term memory and persist.

    Wraps the run in a ``<user-shell-run>`` tag whose shape is described in
    the ``user_actions`` system-prompt section. ``post_notices`` carries
    harness-side messages emitted between the shell exit and this call
    (cwd reject, chdir failure) so the appended body matches what the
    user saw.

    The inner write+save Task is registered into the current
    ``_pending_writes`` tracker (if bound by `_handle_line`'s `!` branch)
    so a Ctrl+C between exec_shell and save completion lets the REPL
    drain this work before accepting another turn — otherwise a late
    save could overwrite the next turn's session snapshot.
    """
    content = format_shell_run(command, exit_code, output, post_notices)
    write_task = asyncio.ensure_future(
        _write(agent, user_content=content, save=save),
    )
    tracker = _pending_writes.get()
    if tracker is not None:
        tracker.append(write_task)
    await asyncio.shield(write_task)


def refresh_system_prompt(agent: BaseAgent) -> None:
    agent.system_prompt = agent._prompt_builder.build(agent._make_builder_context())


async def _write(
    agent: BaseAgent,
    *,
    user_content: str | None = None,
    tool_pairs: list[tuple[ToolCall, ToolResult]] | None = None,
    save: SaveSession | None = None,
) -> None:
    """Convert lane-specific input into Messages, append, optionally persist.

    Exactly one of ``user_content`` / ``tool_pairs`` should be provided.
    Order within ``tool_pairs`` is: 1 assistant message holding all
    ``ToolCall``s, then one ``Message.tool`` per pair in declaration order.
    """
    if tool_pairs is not None:
        tcs = [tc for tc, _ in tool_pairs]
        # Stamp the synthesized assistant turn with provider-specific
        # placeholder sidecar fields.
        await agent.context.short_term_memory.add_message(
            Message.assistant(
                content="",
                tool_calls=tcs,
                provider_metadata=agent.llm.synthetic_turn_sidecar(),
            )
        )
        for _, tr in tool_pairs:
            await agent.context.short_term_memory.add_message(
                Message.tool(
                    tool_call_id=tr.tool_call_id,
                    content=tr.content,
                    is_error=tr.is_error,
                )
            )
    elif user_content is not None:
        await agent.context.short_term_memory.add_message(Message.user(user_content))

    if save is not None:
        await save()
