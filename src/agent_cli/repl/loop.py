"""REPL main loop with 4-way race: bg approval / bg result / user input / bg wake."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout.processors import Processor
from rich.console import Console

from agent_cli.adapter import CliAdapter
from agent_cli.approval_handler import CliApprovalHandler, _PendingApproval
from agent_cli.commands.base import CommandContext
from agent_cli.commands.registry import CommandRegistry
from agent_cli.hooks import CliHooks
from agent_cli.render.notices import format_expired_notice
from agent_cli.render.ui import make_status_bar_text
from agent_cli.repl.completer import build_input_completer, refresh_input_completer
from agent_cli.repl.fill_block import (
    PROMPT_WIDTH,
    FillBlockProcessor,
    make_continuation_prompt,
    make_input_prompt,
)
from agent_cli.repl.keybindings import build_keybindings, reset_ctrl_c_state
from agent_cli.repl.mentions import expand_mentions
from agent_cli.repl.paste import PastePlaceholderProcessor, PasteStore
from agent_cli.runtime import background, file_observer
from agent_cli.runtime.conversation import (
    drain_pending,
    reset_pending_tracker,
    use_pending_tracker,
)
from agent_cli.runtime.session import (
    SaveSession,
    get_messages,
    make_save_session,
    rollback,
    should_rollback,
    switch_session,
    take_snapshot,
)
from agent_cli.runtime.shell import ShellState
from agent_cli.runtime.sigint import bind_work
from agent_cli.render.replay import render_post_switch
from agent_cli.theme import APPROVAL, COMPRESSION, CliTheme
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Attachment, Message, Role
from agent_harness.session.base import BaseSession

_ToolbarText = Callable[[], HTML]


@dataclass
class _InputOutcome:
    line: str | None = None
    eof: bool = False
    interrupted: bool = False
    cancelled: bool = False
    buffered: Document | None = None


@dataclass
class _LoopState:
    pending_input: str | Document = ""
    pending_from_race: _PendingApproval | None = None
    deferred_line: str | None = None


async def _cancel_and_collect(
    task: asyncio.Task[str],
    pt_session: PromptSession[str],
) -> _InputOutcome:
    if task.done():
        buffered = None
    else:
        buffered = pt_session.default_buffer.document
        task.cancel()
    try:
        line = await task
        return _InputOutcome(line=line)
    except EOFError:
        return _InputOutcome(eof=True)
    except KeyboardInterrupt:
        return _InputOutcome(interrupted=True)
    except asyncio.CancelledError:
        return _InputOutcome(cancelled=True, buffered=buffered)


async def _cancel_loser(task: asyncio.Task[Any] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _prompt_with_lock(
    pt_session: PromptSession[str],
    adapter: CliAdapter,
    default: str | Document,
    bottom_toolbar: _ToolbarText,
    input_processors: list[Processor] | None = None,
) -> str:
    async with adapter.lock():
        prev_processors = pt_session.input_processors
        try:
            return await pt_session.prompt_async(
                make_input_prompt(pt_session),
                prompt_continuation=make_continuation_prompt(pt_session),
                default=default,
                set_exception_handler=False,
                bottom_toolbar=bottom_toolbar,
                # prompt_toolkit's handle_sigint=True installs its own
                # SIGINT handler via add_signal_handler and removes it
                # on exit without restoring ours.
                handle_sigint=False,
                complete_while_typing=True,
                # prompt_async persists input_processors back onto the shared
                # PromptSession; snapshot/restore so approval prompts (which never
                # pass the kwarg) don't inherit the REPL placeholder highlighter.
                input_processors=input_processors,
            )
        finally:
            pt_session.input_processors = prev_processors
            reset_ctrl_c_state()


async def run_repl(
    agent: BaseAgent,
    console: Console,
    session_id: str,
    registry: CommandRegistry,
    session_backend: BaseSession,
    adapter: CliAdapter,
    handler: CliApprovalHandler,
    pt_session: PromptSession[str],
    shell_state: ShellState,
    cli_hooks: CliHooks,
    theme: CliTheme,
) -> None:
    paste_store = PasteStore()
    input_processors: list[Processor] = [
        PastePlaceholderProcessor(),
        FillBlockProcessor(offset=PROMPT_WIDTH),
    ]

    pt_session.completer = build_input_completer(registry)
    pt_session.key_bindings = build_keybindings(paste_store=paste_store, agent=agent)
    save = make_save_session(agent, session_backend)
    file_observer.enable(agent)

    state = _LoopState()

    try:
        while True:
            # Priority ladder (1→4). Step 4's race doesn't dispatch inline — winners
            # snapshot into _LoopState and are drained by N+1's 1-3 in fixed order,
            # so race-arrival order never leaks into handling order.

            # 1 bg approval drain — bg task is blocked on a future, highest priority
            pending = state.pending_from_race
            state.pending_from_race = None
            if pending is None:
                try:
                    pending = handler.pending_queue().get_nowait()
                except asyncio.QueueEmpty:
                    pass
            if pending is not None:
                if pending.future.done():
                    # Orphan from a cancelled bg task; skip without prompting.
                    continue
                async with adapter.lock():
                    console.print(
                        f"\n[accent]{APPROVAL} Background approval requested[/accent]",
                    )
                try:
                    await handler.resolve_pending(pending)
                except Exception as e:
                    # Future already carries the exception for the bg task;
                    # surface to user and keep REPL alive
                    console.print(f"[error]Approval handler failed: {e}[/error]")
                continue

            # 2 bg result collect — auto-trigger agent.run on completion
            if await background.collect_results(agent):
                async with adapter.lock():
                    console.print(
                        f"\n[info]{COMPRESSION} Background task completed[/info]",
                    )
                await _run(
                    agent,
                    Message.system(
                        "[Background Task Notification] Process the completed "
                        "background task results.",
                        metadata={"is_background_result": True},
                    ),
                    session_id,
                    console,
                    adapter,
                    cli_hooks,
                    session_backend,
                    save,
                )
                continue

            # 3 deferred input from last race — honour user's Enter before new prompt
            if state.deferred_line is not None:
                raw = state.deferred_line
                state.deferred_line = None
                # Expand paste placeholders before strip / dispatch so slash
                # commands receive real argv and whitespace-only pastes are
                # caught by the empty-check.
                raw, expired, pending_atts = paste_store.resolve(raw)
                if expired:
                    console.print()
                    await adapter.print_inline(format_expired_notice(expired))
                # strip() only for the empty-check; pass raw to preserve
                # indentation / multiline structure for pasted code blocks.
                # Attachments-only turns (empty raw + pending media) still
                # dispatch so the user message carries the media.
                if raw.strip() or pending_atts:
                    if await _handle_line(
                        raw, agent, console, registry, session_id, save, adapter, handler,
                        shell_state, pt_session, cli_hooks, session_backend, theme,
                        pending_atts,
                    ):
                        break
                continue

            # 4 prompt + race against bg events
            input_task: asyncio.Task[str] = asyncio.create_task(
                _prompt_with_lock(
                    pt_session,
                    adapter,
                    state.pending_input,
                    make_status_bar_text(agent),
                    input_processors=input_processors,
                ),
            )
            state.pending_input = ""

            wait_set: set[asyncio.Task[Any]] = {input_task}
            bg_wait: asyncio.Task[Any] | None = None
            if background.has_running(agent):
                bg_wait = asyncio.create_task(background.wait_next(agent))
                wait_set.add(bg_wait)
            approval_wait: asyncio.Task[Any] = asyncio.create_task(
                handler.pending_queue().get(),
            )
            wait_set.add(approval_wait)

            await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            # Post-snapshot completion still counts — `.done()` over `in done`.
            if approval_wait.done():
                state.pending_from_race = approval_wait.result()
            else:
                await _cancel_loser(approval_wait)

            # bg_wait is only a wake signal; drop payload, cancel if still pending.
            if bg_wait is not None:
                if bg_wait.done():
                    bg_wait.result()
                else:
                    await _cancel_loser(bg_wait)

            outcome = await _cancel_and_collect(input_task, pt_session)
            if outcome.eof or outcome.interrupted:
                break
            if outcome.line is not None:
                state.deferred_line = outcome.line
            elif outcome.buffered is not None and outcome.buffered.text:
                state.pending_input = outcome.buffered

    finally:
        if background.has_running(agent):
            count = background.cancel_all(agent)
            if count:
                console.print(f"[dim]Cancelled {count} background task(s).[/dim]")
        await background.shutdown(agent)
        handler.cancel_pending()
        await adapter.end_step()


async def _handle_line(
    line: str,
    agent: BaseAgent,
    console: Console,
    registry: CommandRegistry,
    session_id: str,
    save: Callable[[], Awaitable[None]],
    adapter: CliAdapter,
    handler: CliApprovalHandler,
    shell_state: ShellState,
    pt_session: PromptSession[str],
    cli_hooks: CliHooks,
    session_backend: BaseSession,
    theme: CliTheme,
    pending_atts: list[Attachment] | None = None,
) -> bool:
    console.print()

    if line.startswith("!"):
        cmd = line[1:].lstrip()
        if not cmd:
            return False
        from agent_cli.runtime.shell import exec_shell  # noqa: PLC0415

        completer = pt_session.completer
        assert completer is not None
        pending_writes: list[asyncio.Future[Any]] = []
        token = use_pending_tracker(pending_writes)
        try:
            task = asyncio.create_task(
                exec_shell(shell_state, cmd, agent, completer, adapter, save),
            )
            try:
                with bind_work(task):
                    await task
            except asyncio.CancelledError:
                await drain_pending(pending_writes)
                console.print("[dim]Command cancelled.[/dim]")
                console.print()
            except Exception as e:
                console.print(f"[error]Unexpected shell error: {e}[/error]")
                console.print()
        finally:
            reset_pending_tracker(token)
        return False

    ctx = CommandContext(
        agent=agent,
        session_id=session_backend.session_id,
        registry=registry,
        save_session=save,
        approval_handler=handler,
        session_backend=session_backend,
        refresh_completer=lambda: refresh_input_completer(pt_session, registry),
        adapter=adapter,
    )
    task = asyncio.create_task(registry.dispatch(line, ctx))
    try:
        with bind_work(task):
            result = await task
    except asyncio.CancelledError:
        # Keep REPL alive on Ctrl+C during a command
        console.print("[dim]Command cancelled.[/dim]")
        console.print()
        return False
    except Exception as e:
        console.print(f"[error]Command failed: {e}[/error]")
        console.print()
        return False
    if result is not None:
        if result.output is not None:
            console.print(result.output)
            console.print()
        if result.new_session_id is not None:
            await switch_session(
                agent, session_backend, handler, save, result.new_session_id,
            )
            render_post_switch(
                agent, console, theme, session_backend.session_id,
            )
        if result.should_exit:
            return True
        if result.agent_input:
            await _run(
                agent, result.agent_input, session_backend.session_id,
                console, adapter, cli_hooks, session_backend, save,
                pending_atts,
            )
        return False

    await _run(
        agent, line, session_id, console, adapter,
        cli_hooks, session_backend, save,
        pending_atts,
    )
    return False


async def _run(
    agent: BaseAgent,
    text: str | Message,
    session_id: str,
    console: Console,
    adapter: CliAdapter,
    cli_hooks: CliHooks,
    session_backend: BaseSession,
    save: SaveSession,
    pending_atts: list[Attachment] | None = None,
) -> None:
    cancelled = False
    adapter.begin_run()

    cb = None
    if isinstance(text, str):

        async def cb(a: BaseAgent, msg: Message, t: str) -> None:
            if msg.role != Role.USER:
                return
            await expand_mentions(a, adapter, t)
            if pending_atts:
                msg.attachments = list(msg.attachments or []) + pending_atts

    turn_ctx = take_snapshot(agent)
    cli_hooks.begin_turn(turn_ctx)

    task = asyncio.create_task(
        agent.run(text, session=session_backend, after_input_appended=cb),
    )
    try:
        with bind_work(task):
            try:
                await task
            except asyncio.CancelledError:
                # CancelledError bypasses base.run()'s `except Exception`; state
                # may be stuck in non-terminal, next run()'s transition(THINKING)
                # would raise.
                agent.context.state.reset()
                cancelled = True
                if should_rollback(turn_ctx, get_messages(agent)):
                    await rollback(agent, turn_ctx, save)
            except Exception:
                # hooks.on_error already rendered; just keep REPL alive.
                pass
    finally:
        cli_hooks.end_turn()
        # end_step flushes any pending stream / Live buffers; print cancel
        # banner AFTER so it doesn't get overwritten by late-arriving chunks.
        await adapter.end_step()
    if cancelled:
        console.print("[dim]⎯⎯ Interrupted · what should the agent do differently? ⎯⎯[/dim]")
        console.print()
