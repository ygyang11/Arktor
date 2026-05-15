from unittest.mock import AsyncMock, MagicMock

from agent_cli.hooks import CliHooks
from agent_harness.core.errors import LLMConnectionError
from agent_harness.hooks.progress import _subagent_active
from agent_harness.llm.types import LLMRetryInfo


def _mock_adapter() -> MagicMock:
    m = MagicMock()
    m.on_stream_delta = AsyncMock()
    m.on_tool_call = AsyncMock()
    m.on_tool_result = AsyncMock()
    m.on_tool_denied = AsyncMock()
    m.on_llm_call = AsyncMock()
    m.end_step = AsyncMock()
    m.end_run = AsyncMock()
    m.pause_for_stdin = AsyncMock()
    m.queue_todo = MagicMock()
    m.print_inline = AsyncMock()
    m.start_subagent = AsyncMock()
    m.stop_subagent = AsyncMock()
    m.tick_subagent_step = MagicMock()
    m.tick_subagent_tool = MagicMock()
    return m


async def test_stream_delta_routes() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    delta = MagicMock()
    delta.chunk.delta_content = "hello"
    await hooks.on_llm_stream_delta("cli", delta)
    a.on_stream_delta.assert_awaited_once_with("hello")


async def test_approval_request_hook_is_noop() -> None:
    # pause_for_stdin is called atomically inside handler._prompt_user under
    # the console lock; the hook stays a no-op so pause + panel + prompt_async
    # all run in the same critical section.
    a = _mock_adapter()
    hooks = CliHooks(a)
    await hooks.on_approval_request("cli", MagicMock())
    a.pause_for_stdin.assert_not_called()


async def test_approval_result_deny_calls_on_tool_denied() -> None:
    from agent_harness.approval.types import ApprovalDecision
    a = _mock_adapter()
    hooks = CliHooks(a)
    result = MagicMock(decision=ApprovalDecision.DENY)
    await hooks.on_approval_result("cli", result)
    a.on_tool_denied.assert_awaited_once_with(result)


async def test_approval_result_allow_is_noop() -> None:
    from agent_harness.approval.types import ApprovalDecision
    a = _mock_adapter()
    hooks = CliHooks(a)
    result = MagicMock(decision=ApprovalDecision.ALLOW_ONCE)
    await hooks.on_approval_result("cli", result)
    a.on_tool_denied.assert_not_awaited()


async def test_approval_hooks_suppressed_only_in_background_task() -> None:
    from agent_harness.approval.types import ApprovalDecision
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=True)
    hooks = CliHooks(a, approval_handler=handler)

    await hooks.on_approval_request("child", MagicMock())
    deny_result = MagicMock(decision=ApprovalDecision.DENY)
    await hooks.on_approval_result("child", deny_result)

    a.pause_for_stdin.assert_not_called()
    a.on_tool_denied.assert_not_awaited()


async def test_approval_result_is_suppressed_in_foreground_subagent_context() -> None:
    from agent_harness.approval.types import ApprovalDecision
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)

    token = _subagent_active.set(True)
    try:
        await hooks.on_approval_request("child", MagicMock())
        deny_result = MagicMock(decision=ApprovalDecision.DENY)
        await hooks.on_approval_result("child", deny_result)
    finally:
        _subagent_active.reset(token)

    # Foreground subagent approvals are intentionally not mirrored into the
    # parent's ToolDisplay; only the subagent heartbeat should remain visible.
    a.pause_for_stdin.assert_not_called()
    a.on_tool_denied.assert_not_awaited()


async def test_todo_update_is_queued_not_printed() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    await hooks.on_todo_update("cli", [{"content": "x"}], {"total": 1, "completed": 0})
    a.queue_todo.assert_called_once()


async def test_todo_update_suppressed_during_tool_state_restore() -> None:
    """`on_todo_update` skips `queue_todo` while `_tool_state_restoring` is True
    so a session restore's historical TODO broadcast doesn't surface as if
    the user had just updated their list."""
    from agent_harness.tool.registry import _tool_state_restoring

    a = _mock_adapter()
    hooks = CliHooks(a)
    token = _tool_state_restoring.set(True)
    try:
        await hooks.on_todo_update(
            "cli", [{"content": "x"}], {"total": 1, "completed": 0},
        )
    finally:
        _tool_state_restoring.reset(token)
    a.queue_todo.assert_not_called()


async def test_on_error_calls_end_step_first_then_prints() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    await hooks.on_error("cli", RuntimeError("boom"))
    a.end_step.assert_awaited_once()
    a.print_inline.assert_called_once()


async def test_on_error_appends_traceback_when_debug_enabled() -> None:
    from agent_cli import hooks as hooks_mod

    a = _mock_adapter()
    cli_hooks = CliHooks(a)
    hooks_mod._debug_enabled[0] = True
    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            await cli_hooks.on_error("cli", e)
    finally:
        hooks_mod._debug_enabled[0] = False

    assert a.print_inline.await_count == 2
    calls = [c.args[0] for c in a.print_inline.await_args_list]
    assert "boom" in calls[0]
    assert "Traceback" in calls[1] or "RuntimeError" in calls[1]


async def test_on_error_debug_disabled_prints_one_line_only() -> None:
    from agent_cli import hooks as hooks_mod

    a = _mock_adapter()
    cli_hooks = CliHooks(a)
    hooks_mod._debug_enabled[0] = False
    await cli_hooks.on_error("cli", RuntimeError("boom"))
    assert a.print_inline.await_count == 1


async def test_step_end_calls_end_step_run_end_calls_end_run() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    await hooks.on_step_end("cli", 1)
    await hooks.on_run_end("cli", "output")
    assert a.end_step.await_count == 1
    assert a.end_run.await_count == 1


async def test_llm_call_hook_routes_to_adapter() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    await hooks.on_llm_call("cli", [])
    a.on_llm_call.assert_awaited_once()


async def test_llm_call_hook_suppressed_in_subagent() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    hooks = CliHooks(a)
    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_call("child", [])
    finally:
        _subagent_active.reset(token)
    a.on_llm_call.assert_not_awaited()


async def test_noop_orchestration_hooks() -> None:
    hooks = CliHooks(_mock_adapter())
    await hooks.on_pipeline_start("p")
    await hooks.on_dag_node_end("n")
    await hooks.on_team_end("t", "supervisor")


async def test_subagent_active_suppresses_stream_delta() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    hooks = CliHooks(a)
    delta = MagicMock()
    delta.chunk.delta_content = "child text"
    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_stream_delta("child", delta)
    finally:
        _subagent_active.reset(token)
    a.on_stream_delta.assert_not_awaited()


async def test_subagent_active_suppresses_tool_events_and_step_end() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    hooks = CliHooks(a)
    token = _subagent_active.set(True)
    try:
        await hooks.on_tool_call("child", MagicMock())
        await hooks.on_tool_result("child", MagicMock())
        await hooks.on_step_end("child", 1)
        await hooks.on_todo_update("child", [], {"total": 0, "completed": 0})
        await hooks.on_compression_start("child")
        await hooks.on_error("child", RuntimeError("boom"))
    finally:
        _subagent_active.reset(token)
    a.on_tool_call.assert_not_awaited()
    a.on_tool_result.assert_not_awaited()
    a.end_step.assert_not_awaited()
    a.queue_todo.assert_not_called()
    a.print_inline.assert_not_called()


async def test_on_llm_call_starts_subagent_when_fg_active() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)
    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_call("subA", [])
    finally:
        _subagent_active.reset(token)
    a.start_subagent.assert_awaited_once()
    assert "subA" in hooks._active_fg_subagents


async def test_on_llm_call_skips_subagent_when_bg_active() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=True)
    hooks = CliHooks(a, approval_handler=handler)
    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_call("subB", [])
    finally:
        _subagent_active.reset(token)
    a.start_subagent.assert_not_awaited()
    assert "subB" not in hooks._active_fg_subagents


async def test_on_llm_call_dedups_same_subagent_across_steps() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)
    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_call("subA", [])
        await hooks.on_llm_call("subA", [])
        await hooks.on_llm_call("subA", [])
    finally:
        _subagent_active.reset(token)
    a.start_subagent.assert_awaited_once()


async def test_suppressed_tool_and_step_tick_only_for_fg() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    handler = MagicMock()
    hooks = CliHooks(a, approval_handler=handler)

    handler.is_in_background_task = MagicMock(return_value=False)
    token = _subagent_active.set(True)
    try:
        await hooks.on_tool_call("subA", MagicMock())
        await hooks.on_step_end("subA", 1)
    finally:
        _subagent_active.reset(token)
    a.tick_subagent_tool.assert_called_once()
    a.tick_subagent_step.assert_called_once()

    a.tick_subagent_tool.reset_mock()
    a.tick_subagent_step.reset_mock()
    handler.is_in_background_task = MagicMock(return_value=True)
    token = _subagent_active.set(True)
    try:
        await hooks.on_tool_call("subB", MagicMock())
        await hooks.on_step_end("subB", 1)
    finally:
        _subagent_active.reset(token)
    a.tick_subagent_tool.assert_not_called()
    a.tick_subagent_step.assert_not_called()


async def test_on_subagent_end_stops_only_if_started() -> None:
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)
    # Never called on_llm_call, set is empty — stop should not be invoked.
    await hooks.on_subagent_end("parent", "subA", "generic", "desc", 0, 0, 0.0)
    a.stop_subagent.assert_not_awaited()


async def test_on_subagent_end_stops_when_previously_started() -> None:
    from agent_harness.hooks.progress import _subagent_active
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)
    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_call("subA", [])
    finally:
        _subagent_active.reset(token)
    assert "subA" in hooks._active_fg_subagents

    await hooks.on_subagent_end("parent", "subA", "generic", "desc", 3, 5, 1234.5)
    a.stop_subagent.assert_awaited_once()
    assert "subA" not in hooks._active_fg_subagents


async def test_subagent_start_always_fires() -> None:
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)
    await hooks.on_subagent_start("parent", "sub", "generic", "do X", "prompt")
    a.print_inline.assert_called_once()


async def test_subagent_end_suppressed_in_background_context() -> None:
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=True)
    hooks = CliHooks(a, approval_handler=handler)
    await hooks.on_subagent_end("parent", "sub", "generic", "do X", 3, 2, 1234.5)
    a.print_inline.assert_not_called()


async def test_subagent_end_fires_in_foreground_context() -> None:
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=False)
    hooks = CliHooks(a, approval_handler=handler)
    await hooks.on_subagent_end("parent", "sub", "generic", "do X", 3, 2, 1234.5)
    a.print_inline.assert_called_once()


async def test_on_llm_retry_skipped_in_subagent() -> None:
    a = _mock_adapter()
    a.on_retry = AsyncMock()
    hooks = CliHooks(a)
    info = LLMRetryInfo(
        kind="stream", attempt=1, max_retries=3, wait=1.0,
        error=LLMConnectionError("x"),
    )

    token = _subagent_active.set(True)
    try:
        await hooks.on_llm_retry("subagent", info)
    finally:
        _subagent_active.reset(token)

    a.on_retry.assert_not_awaited()


async def test_on_llm_retry_dispatched_for_main_agent() -> None:
    a = _mock_adapter()
    a.on_retry = AsyncMock()
    hooks = CliHooks(a)
    info = LLMRetryInfo(
        kind="generate", attempt=2, max_retries=3, wait=2.0,
        error=TimeoutError(),
    )

    await hooks.on_llm_retry("main", info)

    a.on_retry.assert_awaited_once_with(info)


# ── Turn lifecycle + commit flag flips ──


from agent_harness.approval.types import ApprovalDecision
from agent_harness.core.message import Message

from agent_cli.runtime.session import _TurnContext


def _empty_turn() -> _TurnContext:
    a = Message.system("sys")
    return _TurnContext(
        snapshot_messages=[a.model_copy(deep=True)],
        snapshot_compressor_state=None,
        snapshot_ids=frozenset({id(a)}),
        main_system_id=id(a),
    )


def test_begin_end_turn_lifecycle() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    assert hooks.turn is None

    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    assert hooks.turn is ctx

    hooks.end_turn()
    assert hooks.turn is None


def test_end_turn_idempotent() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    hooks.end_turn()
    hooks.end_turn()
    assert hooks.turn is None


async def test_compression_start_flips_commit_when_turn_bound() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)

    await hooks.on_compression_start("main")

    assert ctx.committed is True
    a.print_inline.assert_awaited_once()


async def test_compression_start_no_turn_does_not_raise() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)

    await hooks.on_compression_start("main")

    a.print_inline.assert_awaited_once()


async def test_stream_delta_with_content_flips_commit() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    delta = MagicMock()
    delta.chunk.delta_content = "hello"

    await hooks.on_llm_stream_delta("main", delta)

    assert ctx.committed is True
    a.on_stream_delta.assert_awaited_once_with("hello")


async def test_stream_delta_without_content_does_not_flip() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    delta = MagicMock()
    delta.chunk.delta_content = ""

    await hooks.on_llm_stream_delta("main", delta)

    assert ctx.committed is False
    a.on_stream_delta.assert_not_awaited()


async def test_tool_call_flips_commit() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    tc = MagicMock()

    await hooks.on_tool_call("main", tc)

    assert ctx.committed is True
    a.on_tool_call.assert_awaited_once_with(tc)


async def test_tool_call_in_subagent_does_not_flip_top_level_commit() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    tc = MagicMock()
    token = _subagent_active.set(True)
    try:
        await hooks.on_tool_call("subagent", tc)
    finally:
        _subagent_active.reset(token)

    assert ctx.committed is False
    a.on_tool_call.assert_not_awaited()


async def test_approval_request_flips_commit() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)

    await hooks.on_approval_request("main", MagicMock())

    assert ctx.committed is True
    a.pause_for_stdin.assert_not_called()


async def test_approval_request_in_subagent_does_not_flip() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    token = _subagent_active.set(True)
    try:
        await hooks.on_approval_request("sub", MagicMock())
    finally:
        _subagent_active.reset(token)

    assert ctx.committed is False


async def test_approval_request_in_background_does_not_flip() -> None:
    a = _mock_adapter()
    handler = MagicMock()
    handler.is_in_background_task = MagicMock(return_value=True)
    hooks = CliHooks(a, approval_handler=handler)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)

    await hooks.on_approval_request("main", MagicMock())

    assert ctx.committed is False


async def test_approval_result_deny_flips_and_calls_on_tool_denied() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    result = MagicMock(decision=ApprovalDecision.DENY)

    await hooks.on_approval_result("main", result)

    assert ctx.committed is True
    a.on_tool_denied.assert_awaited_once_with(result)


async def test_approval_result_allow_does_not_flip() -> None:
    a = _mock_adapter()
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)
    result = MagicMock(decision=ApprovalDecision.ALLOW_ONCE)

    await hooks.on_approval_result("main", result)

    assert ctx.committed is False
    a.on_tool_denied.assert_not_awaited()


async def test_flag_flips_before_adapter_call() -> None:
    a = _mock_adapter()
    seen_committed: list[bool] = []

    async def capture(*args: object, **kwargs: object) -> None:
        seen_committed.append(ctx.committed)

    a.on_tool_call = AsyncMock(side_effect=capture)
    hooks = CliHooks(a)
    ctx = _empty_turn()
    hooks.begin_turn(ctx)

    await hooks.on_tool_call("main", MagicMock())

    assert seen_committed == [True]
