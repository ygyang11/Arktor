"""REPL loop helpers + race + shutdown."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.repl.loop import (
    _cancel_and_collect,
    _cancel_loser,
    _handle_line,
    _InputOutcome,
    _LoopState,
    _run,
    run_repl,
)

# ---- helpers --------------------------------------------------------------


def _done_future(result_spec: object) -> asyncio.Future[str]:
    """Build an already-done Future resolving to ``result_spec`` — a return
    value or an exception class/instance. Future duck-types as the awaitable
    ``_cancel_and_collect`` needs (``done``/``cancel``/``await``)."""
    fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    if isinstance(result_spec, type) and issubclass(result_spec, BaseException):
        fut.set_exception(result_spec())
    elif isinstance(result_spec, BaseException):
        fut.set_exception(result_spec)
    else:
        assert isinstance(result_spec, str)
        fut.set_result(result_spec)
    return fut


async def test_cancel_and_collect_returns_line_when_done() -> None:
    task = _done_future("hello")
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome == _InputOutcome(line="hello")


async def test_cancel_and_collect_classifies_eof_when_done() -> None:
    task = _done_future(EOFError)
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome.eof is True
    assert outcome.line is None


async def test_cancel_and_collect_classifies_keyboard_interrupt_when_done() -> None:
    task = _done_future(KeyboardInterrupt)
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome.interrupted is True


async def test_cancel_and_collect_classifies_cancelled_when_done() -> None:
    task = _done_future(asyncio.CancelledError)
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome.cancelled is True
    assert outcome.buffered is None


async def test_cancel_and_collect_captures_buffer_on_cancel() -> None:
    from prompt_toolkit.document import Document

    pt_session = MagicMock()
    pt_session.default_buffer.document = Document("partial typing", cursor_position=7)

    hold = asyncio.Event()

    async def _wait_forever() -> str:
        await hold.wait()
        return ""

    task: asyncio.Task[str] = asyncio.create_task(_wait_forever())
    await asyncio.sleep(0)

    outcome = await _cancel_and_collect(task, pt_session)
    assert outcome.cancelled is True
    assert outcome.buffered is not None
    assert outcome.buffered.text == "partial typing"
    assert outcome.buffered.cursor_position == 7
    assert outcome.line is None


async def test_cancel_and_collect_returns_real_line_if_race_completed() -> None:
    async def _instant() -> str:
        return "real input"

    task: asyncio.Task[str] = asyncio.create_task(_instant())
    await task
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome == _InputOutcome(line="real input")


async def test_cancel_loser_handles_none() -> None:
    await _cancel_loser(None)


async def test_cancel_loser_skips_done_task() -> None:
    async def _quick() -> str:
        return "x"

    done_task: asyncio.Task[str] = asyncio.create_task(_quick())
    await done_task
    await _cancel_loser(done_task)
    assert done_task.done()
    assert done_task.result() == "x"


async def test_cancel_loser_cancels_pending_task_and_awaits() -> None:
    async def _forever() -> str:
        await asyncio.sleep(3600)
        return ""

    task: asyncio.Task[str] = asyncio.create_task(_forever())
    await asyncio.sleep(0)
    await _cancel_loser(task)
    assert task.cancelled()


# ---- race (via _run) ------------------------------------------------------


def _run_extras() -> tuple[MagicMock, MagicMock, AsyncMock, _LoopState]:
    """Build shared dependencies for _run tests."""
    cli_hooks = MagicMock()
    cli_hooks.begin_turn = MagicMock()
    cli_hooks.end_turn = MagicMock()
    return cli_hooks, MagicMock(), AsyncMock(), _LoopState()


def _agent_with_stm(messages: list | None = None) -> MagicMock:
    agent = MagicMock()
    agent.context.short_term_memory._messages = messages or []
    agent.context.short_term_memory.compressor = None
    agent.system_prompt = ""
    return agent


async def test_run_cancelled_error_resets_state_and_ends_step() -> None:
    agent = _agent_with_stm()
    agent.context.state.reset = MagicMock()
    agent.run = AsyncMock(side_effect=asyncio.CancelledError())
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()
    console = MagicMock()

    await _run(
        agent, "text", "session-1", console, adapter, *_run_extras(),
        run_gate=asyncio.Lock(),
    )

    agent.context.state.reset.assert_called_once()
    adapter.end_step.assert_awaited_once()


async def test_run_swallows_generic_exception_to_keep_repl_alive() -> None:
    agent = _agent_with_stm()
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()
    console = MagicMock()

    await _run(
        agent, "text", "session-1", console, adapter, *_run_extras(),
        run_gate=asyncio.Lock(),
    )

    adapter.end_step.assert_awaited_once()
    # hooks.on_error owns error rendering now — _run must not print.
    console.print.assert_not_called()


async def test_run_ends_step_on_success() -> None:
    agent = _agent_with_stm()
    agent.run = AsyncMock(return_value=None)
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()

    await _run(
        agent, "text", "session-1", MagicMock(), adapter, *_run_extras(),
        run_gate=asyncio.Lock(),
    )

    agent.run.assert_awaited_once()
    adapter.end_step.assert_awaited_once()


# ---- _handle_line ---------------------------------------------------------


async def test_handle_line_cancelled_keeps_repl_alive() -> None:
    from agent_cli.runtime.shell import ShellState

    registry = MagicMock()
    registry.dispatch = AsyncMock(side_effect=asyncio.CancelledError())
    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    should_exit = await _handle_line(
        "/slow",
        MagicMock(),
        console,
        registry,
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert should_exit is False
    assert any(
        c.args and "cancelled" in str(c.args[0]).lower() for c in console.print.call_args_list
    )


async def test_handle_line_shell_lane_dispatches_to_exec_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.runtime.shell import ShellState

    registry = MagicMock()
    registry.dispatch = AsyncMock(return_value=None)
    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()
    agent = MagicMock()

    captured: list[tuple[object, ...]] = []

    async def fake_exec_shell(state, command, ag, comp, ad, save):
        captured.append((command, ag, comp, save))

    monkeypatch.setattr(
        "agent_cli.runtime.shell.exec_shell",
        fake_exec_shell,
    )

    should_exit = await _handle_line(
        "!ls -la",
        agent,
        console,
        registry,
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert should_exit is False
    assert len(captured) == 1
    assert captured[0][0] == "ls -la"
    registry.dispatch.assert_not_called()


async def test_handle_line_bare_bang_is_noop() -> None:
    from agent_cli.runtime.shell import ShellState

    registry = MagicMock()
    registry.dispatch = AsyncMock(return_value=None)
    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    should_exit = await _handle_line(
        "!",
        MagicMock(),
        console,
        registry,
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert should_exit is False
    registry.dispatch.assert_not_called()


async def test_handle_line_shell_cancellation_renders_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.runtime.shell import ShellState

    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    async def cancelling_exec(state, command, ag, comp, ad, save):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "agent_cli.runtime.shell.exec_shell",
        cancelling_exec,
    )

    await _handle_line(
        "!sleep 30",
        MagicMock(),
        console,
        MagicMock(),
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert any(
        c.args and "cancelled" in str(c.args[0]).lower() for c in console.print.call_args_list
    )


async def test_handle_line_shell_cancel_drains_pending_writes_before_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.runtime.conversation import _pending_writes
    from agent_cli.runtime.shell import ShellState

    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    write_done = asyncio.Event()
    write_started = asyncio.Event()
    cancelled_seen_during_write: list[bool] = []

    async def slow_write_then_record() -> None:
        try:
            await asyncio.sleep(0.05)
        finally:
            write_done.set()

    async def faux_exec(state, command, ag, comp, ad, save):
        tracker = _pending_writes.get()
        assert tracker is not None, "tracker must be bound by !-lane"
        write_task = asyncio.ensure_future(slow_write_then_record())
        tracker.append(write_task)
        write_started.set()
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            cancelled_seen_during_write.append(True)
            raise

    monkeypatch.setattr("agent_cli.runtime.shell.exec_shell", faux_exec)

    cancelled_at_print: list[bool] = []

    def record_print(*args: object, **kwargs: object) -> None:
        if args and "cancelled" in str(args[0]).lower():
            cancelled_at_print.append(write_done.is_set())

    console.print = MagicMock(side_effect=record_print)

    async def runner() -> None:
        await _handle_line(
            "!sleep 30",
            MagicMock(),
            console,
            MagicMock(),
            "sid",
            AsyncMock(),
            MagicMock(),
            MagicMock(),
            ShellState(),
            pt_session,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            _LoopState(),
            run_gate=asyncio.Lock(),
        )

    task = asyncio.create_task(runner())
    await write_started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert cancelled_seen_during_write == [True]
    assert write_done.is_set()
    assert cancelled_at_print == [True], (
        "cancel banner must print AFTER pending writes drain"
    )


# ---- run_repl: paste expand integration ----------------------------------


def _make_run_repl_mocks(prompt_results: list[object]):
    """Build the dependency mocks; pt_session.prompt_async runs through
    ``prompt_results`` (str values returned, exception classes/instances raised).
    """
    pt_session = MagicMock()
    pt_session.completer = None
    pt_session.key_bindings = None

    side_effects: list[object] = []
    for r in prompt_results:
        if isinstance(r, type) and issubclass(r, BaseException):
            side_effects.append(r())
        else:
            side_effects.append(r)
    pt_session.prompt_async = AsyncMock(side_effect=side_effects)

    @asynccontextmanager
    async def _lock():
        yield

    adapter = MagicMock()
    adapter.lock = _lock
    adapter.print_inline = AsyncMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()

    handler = MagicMock()
    empty_q: asyncio.Queue[object] = asyncio.Queue()
    handler.pending_queue = MagicMock(return_value=empty_q)
    handler.cancel_pending = MagicMock()

    agent = MagicMock()
    agent.run = AsyncMock(return_value=None)
    agent.llm.model_name = "test-model"
    agent.context.config.approval.mode = "auto"
    agent.context.short_term_memory.displayed_input_tokens = 0
    agent.context.short_term_memory.max_tokens = 1000
    agent.context.short_term_memory._messages = []
    agent.tools = []
    agent.tool_registry.has = MagicMock(return_value=False)
    agent._bg_manager.get_all = MagicMock(return_value=[])

    registry = MagicMock()
    registry.dispatch = AsyncMock(return_value=None)

    backend = MagicMock()
    console = MagicMock()

    return {
        "agent": agent,
        "console": console,
        "registry": registry,
        "backend": backend,
        "adapter": adapter,
        "handler": handler,
        "pt_session": pt_session,
    }


async def _drive_run_repl(mocks: dict[str, object]) -> None:
    from agent_cli.runtime.shell import ShellState

    cli_hooks = MagicMock()
    cli_hooks.begin_turn = MagicMock()
    cli_hooks.end_turn = MagicMock()
    input_window = MagicMock()
    input_window.style = ""
    with patch(
        "agent_cli.repl.loop.configure_input_window_layout",
        return_value=input_window,
    ):
        await run_repl(
            mocks["agent"],
            mocks["console"],
            "sid",
            mocks["registry"],
            mocks["backend"],
            mocks["adapter"],
            mocks["handler"],
            mocks["pt_session"],
            ShellState(),
            cli_hooks,
            MagicMock(),
        )


async def test_run_repl_expired_placeholder_prints_notice_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _make_run_repl_mocks(["[Pasted text #99]", EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    captured: list[str] = []

    async def fake_handle_line(line, *args, **kwargs):
        captured.append(line)
        return False

    monkeypatch.setattr("agent_cli.repl.loop._handle_line", fake_handle_line)

    await _drive_run_repl(mocks)

    assert captured == ["[Pasted text unavailable]"]
    mocks["adapter"].print_inline.assert_awaited_once()


async def test_run_repl_whitespace_only_expansion_skips_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A short whitespace-only line never triggers the placeholder path; this
    # test pins that empty/whitespace input goes through the strip() guard
    # without invoking _handle_line, regardless of paste store state.
    mocks = _make_run_repl_mocks(["   \n  ", EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    called = MagicMock()

    async def fake_handle_line(*args, **kwargs):
        called()
        return False

    monkeypatch.setattr("agent_cli.repl.loop._handle_line", fake_handle_line)

    await _drive_run_repl(mocks)

    called.assert_not_called()
    mocks["adapter"].print_inline.assert_not_called()


async def test_run_repl_plain_input_no_notice_no_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _make_run_repl_mocks(["hello world", EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    captured: list[str] = []

    async def fake_handle_line(line, *args, **kwargs):
        captured.append(line)
        return False

    monkeypatch.setattr("agent_cli.repl.loop._handle_line", fake_handle_line)

    await _drive_run_repl(mocks)

    assert captured == ["hello world"]
    mocks["adapter"].print_inline.assert_not_called()


async def test_run_repl_passes_paste_processor_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.repl.paste import PastePlaceholderProcessor

    mocks = _make_run_repl_mocks([EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    await _drive_run_repl(mocks)

    pt_session = mocks["pt_session"]
    call_kwargs = pt_session.prompt_async.call_args.kwargs
    procs = call_kwargs.get("input_processors")
    assert procs is not None
    assert any(isinstance(p, PastePlaceholderProcessor) for p in procs)


async def test_prompt_with_lock_restores_shared_prompt_state() -> None:
    """Approval prompts must not inherit main-REPL rendering state."""
    from contextlib import asynccontextmanager

    from agent_cli.repl.loop import _prompt_with_lock
    from agent_cli.repl.paste import PastePlaceholderProcessor

    processors = object()
    continuation = object()
    style = object()
    pt_session = MagicMock()
    pt_session.input_processors = processors
    pt_session.prompt_continuation = continuation
    pt_session.default_buffer.text = "hello"
    input_window = MagicMock()
    input_window.style = style

    async def _prompt_async(*args, **kwargs):
        assert input_window.style() == "class:input-block"
        pt_session.default_buffer.text = "!pwd"
        assert input_window.style() == "class:shell-line"
        return "x"

    pt_session.prompt_async = AsyncMock(side_effect=_prompt_async)

    @asynccontextmanager
    async def _lock():
        yield

    adapter = MagicMock()
    adapter.lock = _lock

    await _prompt_with_lock(
        pt_session,
        adapter,
        input_window,
        default="",
        bottom_toolbar=lambda: None,
        input_processors=[PastePlaceholderProcessor()],
    )

    assert input_window.style is style
    assert pt_session.input_processors is processors
    assert pt_session.prompt_continuation is continuation


# ---- _run cancel-rollback (Phase 4) ---------------------------------------


from agent_cli.hooks import CliHooks
from agent_cli.runtime.session import _TurnContext
from agent_harness.core.message import Message, Role


def _make_real_cli_hooks() -> CliHooks:
    a = MagicMock()
    a.on_stream_delta = AsyncMock()
    a.on_tool_call = AsyncMock()
    a.on_tool_denied = AsyncMock()
    a.on_llm_call = AsyncMock()
    a.end_step = AsyncMock()
    a.print_inline = AsyncMock()
    a.start_subagent = AsyncMock()
    a.stop_subagent = AsyncMock()
    a.tick_subagent_step = MagicMock()
    a.tick_subagent_tool = MagicMock()
    return CliHooks(a)


def _real_agent_with_messages(initial: list[Message]) -> MagicMock:
    from agent_harness.memory.short_term import ShortTermMemory

    agent = MagicMock()
    agent.system_prompt = ""
    agent._total_usage = MagicMock()
    stm = ShortTermMemory()
    stm._messages = list(initial)
    stm.compressor = None
    agent.context.short_term_memory = stm
    agent.context.state.reset = MagicMock()
    return agent


async def test_run_passes_session_backend_instance_not_string() -> None:
    initial = [Message.user("hi")]
    agent = _real_agent_with_messages(initial)
    captured: dict[str, object] = {}

    async def fake_run(text: object, session: object = None, **_: object) -> None:
        captured["session"] = session

    agent.run = fake_run
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    backend = MagicMock()
    cli_hooks = _make_real_cli_hooks()

    await _run(
        agent, "go", "sid", MagicMock(), adapter,
        cli_hooks, backend, AsyncMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert captured["session"] is backend


async def test_run_pre_commit_cancel_rolls_back_messages_and_calls_save() -> None:
    initial = [Message.user("u1")]
    agent = _real_agent_with_messages(initial)

    async def fake_run(text: object, session: object = None, **_: object) -> None:
        agent.context.short_term_memory._messages.append(Message.user("text"))
        raise asyncio.CancelledError()

    agent.run = fake_run
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    save = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    await _run(
        agent, "text", "sid", MagicMock(), adapter,
        cli_hooks, MagicMock(), save,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    msgs = agent.context.short_term_memory._messages
    assert len(msgs) == 1
    assert msgs[0].content == "u1"
    save.assert_awaited_once()


async def test_run_post_commit_cancel_does_not_rollback() -> None:
    initial = [Message.user("u1")]
    agent = _real_agent_with_messages(initial)
    cli_hooks = _make_real_cli_hooks()

    async def fake_run(text: object, session: object = None, **_: object) -> None:
        if cli_hooks.turn is not None:
            cli_hooks.turn.committed = True
        agent.context.short_term_memory._messages.append(Message.assistant("partial"))
        raise asyncio.CancelledError()

    agent.run = fake_run
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    save = AsyncMock()

    await _run(
        agent, "text", "sid", MagicMock(), adapter,
        cli_hooks, MagicMock(), save,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    msgs = agent.context.short_term_memory._messages
    assert len(msgs) == 2
    save.assert_not_awaited()


async def test_run_preserves_bg_message_during_rollback() -> None:
    initial = [Message.user("u1")]
    agent = _real_agent_with_messages(initial)

    async def fake_run(text: object, session: object = None, **_: object) -> None:
        bg = Message.system("[bg done]", metadata={"is_background_result": True})
        agent.context.short_term_memory._messages.append(Message.user("text"))
        agent.context.short_term_memory._messages.append(bg)
        raise asyncio.CancelledError()

    agent.run = fake_run
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    save = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    await _run(
        agent, "text", "sid", MagicMock(), adapter,
        cli_hooks, MagicMock(), save,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    msgs = agent.context.short_term_memory._messages
    assert len(msgs) == 2
    assert msgs[0].content == "u1"
    assert msgs[1].metadata.get("is_background_result") is True


async def test_run_end_turn_called_in_finally_after_normal_completion() -> None:
    initial: list[Message] = []
    agent = _real_agent_with_messages(initial)
    agent.run = AsyncMock(return_value=None)
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    await _run(
        agent, "text", "sid", MagicMock(), adapter,
        cli_hooks, MagicMock(), AsyncMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert cli_hooks.turn is None


async def test_run_end_turn_called_in_finally_after_cancel() -> None:
    initial: list[Message] = []
    agent = _real_agent_with_messages(initial)
    agent.run = AsyncMock(side_effect=asyncio.CancelledError())
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    await _run(
        agent, "text", "sid", MagicMock(), adapter,
        cli_hooks, MagicMock(), AsyncMock(),
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    assert cli_hooks.turn is None


# ---- LLMUnsupportedContentError handling ---------------------------------

from agent_harness.core.errors import LLMUnsupportedContentError  # noqa: E402


async def test_run_media_rejection_with_rollback() -> None:
    """User-side rejection: nothing committed → rollback + reverted notice."""
    agent = _agent_with_stm()
    agent.context.state.reset = MagicMock()
    agent.run = AsyncMock(side_effect=LLMUnsupportedContentError("invalid part type: file"))
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()
    console = MagicMock()
    save = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    await _run(
        agent, "text", "sid", console, adapter,
        cli_hooks, MagicMock(), save,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    agent.context.state.reset.assert_called_once()
    printed = "".join(
        str(call.args[0]) if call.args else "" for call in console.print.call_args_list
    )
    assert "reverted" in printed
    assert "/clear" not in printed


async def test_run_media_rejection_after_auto_compression_still_rolls_back() -> None:
    """Regression: auto-compression mid-turn must NOT block rollback when the
    failure is user-side media rejection (compression is invisible/clean-undo)."""
    agent = _agent_with_stm()
    agent.context.state.reset = MagicMock()

    captured_hooks: list = []

    async def fake_run(*args: object, **kwargs: object) -> None:
        # Simulate the compressor firing its hook before the main LLM call.
        await captured_hooks[0].on_compression_start("test")
        raise LLMUnsupportedContentError("invalid part type: file")

    agent.run = fake_run

    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    adapter.print_inline = AsyncMock()
    adapter.begin_run = MagicMock()
    console = MagicMock()
    save = AsyncMock()
    cli_hooks = _make_real_cli_hooks()
    captured_hooks.append(cli_hooks)

    await _run(
        agent, "text", "sid", console, adapter,
        cli_hooks, MagicMock(), save,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    agent.context.state.reset.assert_called_once()
    printed = "".join(
        str(call.args[0]) if call.args else "" for call in console.print.call_args_list
    )
    # Rollback path (not partial-recovery) must be taken
    assert "reverted" in printed
    assert "/clear" not in printed


async def test_run_media_rejection_without_rollback() -> None:
    """Committed state → no rollback; partial-recovery notice with /clear."""
    agent = _agent_with_stm()
    agent.context.state.reset = MagicMock()
    agent.run = AsyncMock(side_effect=LLMUnsupportedContentError("invalid part type: file"))
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()
    console = MagicMock()
    save = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    real_begin = cli_hooks.begin_turn

    def begin(ctx: _TurnContext) -> None:
        real_begin(ctx)
        ctx.committed = True

    cli_hooks.begin_turn = begin  # type: ignore[method-assign]

    await _run(
        agent, "text", "sid", console, adapter,
        cli_hooks, MagicMock(), save,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )

    agent.context.state.reset.assert_called_once()
    printed = "".join(
        str(call.args[0]) if call.args else "" for call in console.print.call_args_list
    )
    assert "partial recovery" in printed
    assert "/clear" in printed


# ---- run_gate single-flight ----------------------------------------------


async def test_run_gate_serializes_concurrent_runs() -> None:
    """Two concurrent _run on one agent serialize through run_gate: their
    agent.run executions never overlap, and neither dispatch is dropped."""
    agent = _real_agent_with_messages([])
    gate = asyncio.Lock()
    active = 0
    max_concurrent = 0
    completed = 0

    async def fake_run(text: object, session: object = None, **_: object) -> None:
        nonlocal active, max_concurrent, completed
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.01)
        active -= 1
        completed += 1

    agent.run = fake_run
    adapter = MagicMock()
    adapter.begin_run = MagicMock()
    adapter.end_step = AsyncMock()
    cli_hooks = _make_real_cli_hooks()

    await asyncio.gather(
        _run(agent, "a", "sid", MagicMock(), adapter,
             cli_hooks, MagicMock(), AsyncMock(), _LoopState(), run_gate=gate),
        _run(agent, "b", "sid", MagicMock(), adapter,
             cli_hooks, MagicMock(), AsyncMock(), _LoopState(), run_gate=gate),
    )

    assert completed == 2
    assert max_concurrent == 1


# ---- persistent goal integration -----------------------------------------


def _goal_agent() -> MagicMock:
    from agent_cli.runtime.goal import mode as goal_mode
    from agent_harness.llm.types import ProcessUsageMeter

    agent = _real_agent_with_messages([])
    agent.context.usage_meter = ProcessUsageMeter()
    agent._session_metadata_extras = {}
    goal_mode.begin(agent, "finish the work")
    return agent


async def test_run_completed_goal_turn_records_and_schedules_evaluation() -> None:
    from agent_cli.runtime.goal import mode as goal_mode

    agent = _goal_agent()
    agent.run = AsyncMock(return_value=None)
    adapter = MagicMock(begin_run=MagicMock(), end_step=AsyncMock())
    save = AsyncMock()
    state = _LoopState()

    await _run(
        agent,
        "work",
        "sid",
        MagicMock(),
        adapter,
        _make_real_cli_hooks(),
        MagicMock(),
        save,
        state,
        run_gate=asyncio.Lock(),
    )

    goal = goal_mode.get_state(agent)
    assert goal is not None and goal.turns == 1
    assert state.goal_eval_pending is True
    save.assert_awaited_once()
    goal_mode.clear(agent)


async def test_run_goal_save_failure_pauses_and_clears_pending() -> None:
    from agent_cli.runtime.goal import mode as goal_mode

    agent = _goal_agent()
    agent.run = AsyncMock(return_value=None)
    adapter = MagicMock(begin_run=MagicMock(), end_step=AsyncMock())
    console = MagicMock()
    state = _LoopState()

    await _run(
        agent,
        "work",
        "sid",
        console,
        adapter,
        _make_real_cli_hooks(),
        MagicMock(),
        AsyncMock(side_effect=OSError("disk full\nextra detail")),
        state,
        run_gate=asyncio.Lock(),
    )

    goal = goal_mode.get_state(agent)
    assert goal is not None and goal.status == "paused"
    assert goal.reason == "session save failed (OSError): disk full"
    assert goal.turns == 1
    assert state.goal_eval_pending is False
    rendered = "".join(
        str(call.args[0])
        for call in console.print.call_args_list
        if call.args
    )
    assert "◎ goal paused" in rendered
    assert "extra detail" not in rendered
    goal_mode.clear(agent)


async def test_run_failed_goal_turn_pauses_and_clears_pending() -> None:
    from agent_cli.runtime.goal import mode as goal_mode

    agent = _goal_agent()
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    adapter = MagicMock(begin_run=MagicMock(), end_step=AsyncMock())
    state = _LoopState(goal_eval_pending=True)
    save = AsyncMock()

    await _run(
        agent,
        "work",
        "sid",
        MagicMock(),
        adapter,
        _make_real_cli_hooks(),
        MagicMock(),
        save,
        state,
        run_gate=asyncio.Lock(),
    )

    goal = goal_mode.get_state(agent)
    assert goal is not None and goal.status == "paused"
    assert goal.reason == "turn did not complete"
    assert goal.turns == 0
    assert state.goal_eval_pending is False
    save.assert_awaited_once()
    goal_mode.clear(agent)


async def test_run_non_goal_clears_stale_evaluation_flag() -> None:
    agent = _real_agent_with_messages([])
    agent.run = AsyncMock(return_value=None)
    adapter = MagicMock(begin_run=MagicMock(), end_step=AsyncMock())
    state = _LoopState(goal_eval_pending=True)
    save = AsyncMock()

    await _run(
        agent,
        "work",
        "sid",
        MagicMock(),
        adapter,
        _make_real_cli_hooks(),
        MagicMock(),
        save,
        state,
        run_gate=asyncio.Lock(),
    )
    assert state.goal_eval_pending is False
    save.assert_not_awaited()


async def test_run_expands_mentions_only_for_string_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expand = AsyncMock()
    monkeypatch.setattr("agent_cli.repl.loop.expand_mentions", expand)
    agent = _real_agent_with_messages([])

    async def run(inp, *, after_input_appended, **kwargs):
        message = inp if isinstance(inp, Message) else Message.user(inp)
        await after_input_appended(agent, message, message.content or "")

    agent.run = run
    adapter = MagicMock(begin_run=MagicMock(), end_step=AsyncMock())
    extras = (_make_real_cli_hooks(), MagicMock(), AsyncMock())

    await _run(
        agent,
        "@SPEC.md",
        "sid",
        MagicMock(),
        adapter,
        *extras,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )
    await _run(
        agent,
        Message.user("@SPEC.md", metadata={"is_goal_continuation": True}),
        "sid",
        MagicMock(),
        adapter,
        *extras,
        _LoopState(),
        run_gate=asyncio.Lock(),
    )
    expand.assert_awaited_once()


async def test_repl_goal_continues_then_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.commands.base import CommandResult
    from agent_cli.runtime.goal import mode as goal_mode
    from agent_cli.runtime.goal.driver import GoalDecision
    from agent_harness.llm.types import ProcessUsageMeter

    mocks = _make_run_repl_mocks(["/goal x", EOFError])
    agent = mocks["agent"]
    agent._session_metadata_extras = {}
    agent.context.usage_meter = ProcessUsageMeter()
    goal = goal_mode.begin(agent, "x")
    monkeypatch.setattr(
        "agent_cli.repl.loop.make_save_session",
        lambda *args, **kwargs: AsyncMock(),
    )
    mocks["registry"].dispatch = AsyncMock(return_value=CommandResult(
        agent_input=goal_mode.make_start_input("x")
    ))
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.cancel_all_with_note",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )
    status = MagicMock(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(
        "agent_cli.repl.loop.make_command_status_line",
        lambda *args, **kwargs: status,
    )
    decisions = 0

    async def decide(current) -> GoalDecision:
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            continuation = goal_mode.make_continuation_message(
                current, "gap", "finish verification"
            )
            assert continuation is not None
            return GoalDecision("continue", "gap", continuation)
        goal_mode.finish(current, "complete", "verified")
        return GoalDecision("complete", "verified")

    monkeypatch.setattr("agent_cli.repl.loop.goal_driver.decide", decide)
    await _drive_run_repl(mocks)

    assert agent.run.await_count == 2
    assert decisions == 2
    assert goal.status == "complete"
    assert goal.turns == 2
    assert status.start.await_count == 2
    assert status.stop.await_count == 2
    goal_mode.clear(agent)


async def test_repl_goal_evaluator_failure_pauses_and_keeps_loop_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.commands.base import CommandResult
    from agent_cli.runtime.goal import evaluator, mode as goal_mode
    from agent_harness.llm.types import ProcessUsageMeter

    mocks = _make_run_repl_mocks(["/goal x", EOFError])
    agent = mocks["agent"]
    agent._session_metadata_extras = {}
    agent.context.usage_meter = ProcessUsageMeter()
    goal = goal_mode.begin(agent, "x")
    save = AsyncMock(side_effect=[None, OSError("disk full")])
    monkeypatch.setattr(
        "agent_cli.repl.loop.make_save_session",
        lambda *args, **kwargs: save,
    )
    mocks["registry"].dispatch = AsyncMock(return_value=CommandResult(
        agent_input=goal_mode.make_start_input("x")
    ))
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.cancel_all_with_note",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr("agent_cli.repl.loop.background.shutdown", AsyncMock())
    status = MagicMock(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(
        "agent_cli.repl.loop.make_command_status_line",
        lambda *args, **kwargs: status,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.goal_driver.decide",
        AsyncMock(side_effect=evaluator.GoalEvaluationError(
            "input exceeds context limit; run /compact before resuming"
        )),
    )

    await _drive_run_repl(mocks)
    assert goal.status == "paused"
    assert "/compact" in goal.reason
    assert save.await_count == 2
    assert status.stop.await_count == 1
    rendered = "".join(
        str(call.args[0])
        for call in mocks["console"].print.call_args_list
        if call.args
    )
    assert "/compact" in rendered
    assert "session save failed (OSError): disk full" in rendered
    goal_mode.clear(agent)


async def test_repl_goal_save_failure_stops_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.commands.base import CommandResult
    from agent_cli.runtime.goal import mode as goal_mode
    from agent_cli.runtime.goal.driver import GoalDecision
    from agent_harness.llm.types import ProcessUsageMeter

    mocks = _make_run_repl_mocks(["/goal x", EOFError])
    agent = mocks["agent"]
    agent._session_metadata_extras = {}
    agent.context.usage_meter = ProcessUsageMeter()
    goal = goal_mode.begin(agent, "x")
    save = AsyncMock(side_effect=[None, OSError("disk full")])
    monkeypatch.setattr(
        "agent_cli.repl.loop.make_save_session",
        lambda *args, **kwargs: save,
    )
    mocks["registry"].dispatch = AsyncMock(return_value=CommandResult(
        agent_input=goal_mode.make_start_input("x")
    ))
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.cancel_all_with_note",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr("agent_cli.repl.loop.background.shutdown", AsyncMock())
    status = MagicMock(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(
        "agent_cli.repl.loop.make_command_status_line",
        lambda *args, **kwargs: status,
    )

    async def decide(current) -> GoalDecision:
        continuation = goal_mode.make_continuation_message(
            current, "work remains", "finish verification"
        )
        assert continuation is not None
        return GoalDecision("continue", "work remains", continuation)

    monkeypatch.setattr("agent_cli.repl.loop.goal_driver.decide", decide)

    await _drive_run_repl(mocks)

    assert agent.run.await_count == 1
    assert save.await_count == 2
    assert goal.status == "paused"
    assert goal.reason == "session save failed (OSError): disk full"
    rendered = "".join(
        str(call.args[0])
        for call in mocks["console"].print.call_args_list
        if call.args
    )
    assert "◎ goal · continue · work remains" not in rendered
    assert "◎ goal paused · session save failed" in rendered
    goal_mode.clear(agent)
