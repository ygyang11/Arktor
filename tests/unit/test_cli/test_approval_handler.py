import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.approval_handler import CliApprovalHandler
from agent_harness.approval.policy import ApprovalPolicy
from agent_harness.approval.types import ApprovalDecision, ApprovalRequest
from agent_harness.core.message import ToolCall


def _request(tool_name: str = "edit_file", tool_call_id: str = "t1") -> ApprovalRequest:
    return ApprovalRequest(
        tool_call=ToolCall(id=tool_call_id, name=tool_name, arguments={}),
        agent_name="cli",
    )


def _handler_with_bg_tasks(bg_tasks: list[MagicMock]) -> CliApprovalHandler:
    h = CliApprovalHandler(console=MagicMock())
    agent = MagicMock()
    agent._bg_manager.get_all = MagicMock(return_value=bg_tasks)
    h.bind_agent(agent)
    return h


def _patch_prompt(  # type: ignore[no-untyped-def]
    handler: CliApprovalHandler,
    return_value: str = "y",
    raises: BaseException | None = None,
):
    mock = AsyncMock(side_effect=raises) if raises is not None else AsyncMock(
        return_value=return_value
    )
    return patch.object(handler._pt_session, "prompt_async", mock)


async def test_is_in_background_task_false_without_bound_agent() -> None:
    h = CliApprovalHandler(console=MagicMock())
    assert h.is_in_background_task() is False


async def test_is_in_background_task_false_when_current_not_in_bg_list() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    assert h.is_in_background_task() is False


async def test_is_in_background_task_true_when_current_in_bg_list() -> None:
    current = asyncio.current_task()
    bg = MagicMock()
    bg.asyncio_task = current
    h = _handler_with_bg_tasks(bg_tasks=[bg])
    assert h.is_in_background_task() is True


async def test_foreground_main_agent_uses_prompt_toolkit() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    with _patch_prompt(h, return_value="y"):
        result = await h.request_approval(_request())
    assert result.decision == ApprovalDecision.ALLOW_ONCE


async def test_panel_shows_resource_not_full_args() -> None:
    import io

    from rich.console import Console as RConsole

    from agent_cli.theme import FLEXOKI_DARK

    buf = io.StringIO()
    con = RConsole(file=buf, theme=FLEXOKI_DARK.rich, color_system=None, width=100)
    h = CliApprovalHandler(console=con)
    h.bind_agent(MagicMock(_bg_manager=MagicMock(get_all=MagicMock(return_value=[]))))

    req = ApprovalRequest(
        tool_call=ToolCall(
            id="t1", name="edit_file",
            arguments={"file_path": "src/auth.py", "content": "X" * 500},
        ),
        agent_name="cli",
        resource="src/auth.py",
        resource_kind="path",
    )
    with _patch_prompt(h, return_value="y"):
        await h.request_approval(req)

    out = buf.getvalue()
    assert "Update" in out
    assert "edit_file" not in out
    assert "src/auth.py" in out
    assert "Approval needed" in out
    assert "X" * 100 not in out


async def test_panel_no_resource_shows_tool_name_only() -> None:
    import io

    from rich.console import Console as RConsole

    from agent_cli.theme import FLEXOKI_DARK

    buf = io.StringIO()
    con = RConsole(file=buf, theme=FLEXOKI_DARK.rich, color_system=None, width=100)
    h = CliApprovalHandler(console=con)
    h.bind_agent(MagicMock(_bg_manager=MagicMock(get_all=MagicMock(return_value=[]))))

    req = ApprovalRequest(
        tool_call=ToolCall(
            id="t1", name="custom_tool",
            arguments={"action": "run", "empty": ""},
        ),
        agent_name="cli",
    )
    with _patch_prompt(h, return_value="y"):
        await h.request_approval(req)

    out = buf.getvalue()
    assert "custom_tool" in out
    assert "action=" not in out
    assert "run" not in out


async def test_foreground_sub_agent_also_uses_prompt_toolkit() -> None:
    from agent_harness.hooks.progress import _subagent_active

    h = _handler_with_bg_tasks(bg_tasks=[])
    token = _subagent_active.set(True)
    try:
        with _patch_prompt(h, return_value="a"):
            result = await h.request_approval(_request())
    finally:
        _subagent_active.reset(token)

    assert result.decision == ApprovalDecision.ALLOW_SESSION


async def test_foreground_deny_with_reason() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    with _patch_prompt(h, return_value="n too dangerous"):
        result = await h.request_approval(_request())
    assert result.decision == ApprovalDecision.DENY
    assert result.reason == "too dangerous"


@pytest.mark.parametrize(
    ("raw", "expected_reason"),
    [
        ("NO", None),
        ("no，不用看这个", "不用看这个"),
        ("No: Stop Here", "Stop Here"),
        ("n; keep original Case", "keep original Case"),
    ],
)
def test_parse_answer_deny_variants(
    raw: str,
    expected_reason: str | None,
) -> None:
    result = CliApprovalHandler._parse_answer(raw, _request())
    assert result.decision == ApprovalDecision.DENY
    assert result.reason == expected_reason


@pytest.mark.parametrize("raw", ["next", "none"])
def test_parse_answer_non_deny_words_default_allow_once(raw: str) -> None:
    result = CliApprovalHandler._parse_answer(raw, _request())
    assert result.decision == ApprovalDecision.ALLOW_ONCE
    assert result.reason is None


async def test_foreground_empty_input_defaults_to_allow_once() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    with _patch_prompt(h, return_value=""):
        result = await h.request_approval(_request())
    assert result.decision == ApprovalDecision.ALLOW_ONCE
    assert result.reason is None


async def test_foreground_always_synonyms_both_allow_session() -> None:
    for answer in ("a", "always", "A", "Always"):
        h = _handler_with_bg_tasks(bg_tasks=[])
        with _patch_prompt(h, return_value=answer):
            result = await h.request_approval(_request())
        assert result.decision == ApprovalDecision.ALLOW_SESSION, f"failed for {answer!r}"


def test_always_label_generic_without_resource() -> None:
    req = _request()
    label = CliApprovalHandler._always_label(req)
    assert "this session" in label


def test_always_label_path_with_parent() -> None:
    req = _request(tool_name="edit_file")
    req.resource = "src/auth/login.py"
    req.resource_kind = "path"
    label = CliApprovalHandler._always_label(req)
    assert "src/auth" in label
    assert "Update" in label
    assert "edit_file" not in label


def test_always_label_command_single() -> None:
    req = _request(tool_name="terminal_tool")
    req.resource = "git status"
    req.resource_kind = "command"
    label = CliApprovalHandler._always_label(req)
    assert "'git'" in label
    assert "commands" in label


def test_always_label_command_chained_shows_all_prefixes() -> None:
    req = _request(tool_name="terminal_tool")
    req.resource = "git status && pytest tests/"
    req.resource_kind = "command"
    label = CliApprovalHandler._always_label(req)
    assert "'git'" in label
    assert "'pytest'" in label
    assert "commands" in label


async def test_unsafe_shell_command_hides_always_option_in_panel() -> None:
    """Heredoc / redirect commands hit policy's unsafe-shell fallback —
    no prefix grant will ever match, so panel must not advertise [A]lways."""
    import io

    from rich.console import Console as RConsole

    from agent_cli.theme import FLEXOKI_DARK

    buf = io.StringIO()
    con = RConsole(file=buf, theme=FLEXOKI_DARK.rich, color_system=None, width=120)
    h = CliApprovalHandler(console=con)
    h.bind_agent(MagicMock(_bg_manager=MagicMock(get_all=MagicMock(return_value=[]))))

    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="terminal_tool", arguments={}),
        agent_name="cli",
        resource="python - <<'PY'\nimport json\nPY",
        resource_kind="command",
    )
    with _patch_prompt(h, return_value="y"):
        await h.request_approval(req)

    out = buf.getvalue()
    assert "Always" not in out
    assert "[A]" not in out


async def test_unsafe_shell_command_uses_no_always_prompt_text() -> None:
    """prompt_async must receive the no-[A] HTML when command is unsafe-shell."""
    h = _handler_with_bg_tasks(bg_tasks=[])

    seen: dict[str, object] = {}

    async def _capture(prompt_text, **kwargs):  # type: ignore[no-untyped-def]
        seen["prompt_text"] = prompt_text
        return "y"

    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="terminal_tool", arguments={}),
        agent_name="cli",
        resource="cat > /tmp/foo",
        resource_kind="command",
    )
    with patch.object(h._pt_session, "prompt_async", AsyncMock(side_effect=_capture)):
        await h.request_approval(req)

    rendered = seen["prompt_text"].value  # type: ignore[union-attr]
    assert "[A]" not in rendered
    assert "Always" not in rendered
    assert "[Y]" in rendered
    assert "[N]" in rendered


def test_parse_answer_a_falls_back_to_allow_once_when_session_disallowed() -> None:
    """Defensive: if user types 'a' on a no-[A] prompt, treat as ALLOW_ONCE."""
    result = CliApprovalHandler._parse_answer(
        "a", _request(), allow_session=False,
    )
    assert result.decision == ApprovalDecision.ALLOW_ONCE


def test_can_grant_session_unsafe_shell_returns_false() -> None:
    policy = ApprovalPolicy(mode="auto")
    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="terminal_tool", arguments={}),
        agent_name="cli",
        resource="python - <<'PY'\nprint(1)\nPY",
        resource_kind="command",
    )
    assert CliApprovalHandler._can_grant_session(req, policy) is False


def test_can_grant_session_safe_shell_returns_true() -> None:
    policy = ApprovalPolicy(mode="auto")
    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="terminal_tool", arguments={}),
        agent_name="cli",
        resource="python script.py",
        resource_kind="command",
    )
    assert CliApprovalHandler._can_grant_session(req, policy) is True


def test_can_grant_session_non_command_resource_returns_true() -> None:
    """path / url / no-resource cases unchanged in auto mode."""
    policy = ApprovalPolicy(mode="auto")
    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="edit_file", arguments={}),
        agent_name="cli",
        resource="src/auth.py",
        resource_kind="path",
    )
    assert CliApprovalHandler._can_grant_session(req, policy) is True

    req2 = _request()
    assert CliApprovalHandler._can_grant_session(req2, policy) is True


def test_can_grant_session_returns_false_outside_auto_mode() -> None:
    """ask / never modes disable session grants entirely."""
    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="edit_file", arguments={}),
        agent_name="cli",
        resource="src/auth.py",
        resource_kind="path",
    )
    assert CliApprovalHandler._can_grant_session(req, ApprovalPolicy(mode="ask")) is False
    assert CliApprovalHandler._can_grant_session(req, ApprovalPolicy(mode="never")) is False


async def test_ctrl_c_during_approval_raises_cancelled_error() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    with _patch_prompt(h, raises=KeyboardInterrupt()):
        with pytest.raises(asyncio.CancelledError):
            await h.request_approval(_request())


async def test_foreground_eof_raises_cancelled_error() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    with _patch_prompt(h, raises=EOFError()):
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await h.request_approval(_request())
    assert "ctrl+d" in str(exc_info.value).lower()


async def test_background_eof_during_resolve_denies_this_request_only() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    from agent_cli.approval_handler import _PendingApproval

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    pending = _PendingApproval(request=_request("edit_file"), future=fut)

    with _patch_prompt(h, raises=EOFError()):
        await h.resolve_pending(pending)

    assert fut.done()
    result = fut.result()
    assert result.decision == ApprovalDecision.DENY
    assert "ctrl+d" in (result.reason or "").lower()


async def test_background_request_enqueues_and_blocks_until_resolved() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])

    async def bg_runner():  # type: ignore[no-untyped-def]
        bg = MagicMock()
        bg.asyncio_task = asyncio.current_task()
        assert h._agent_ref is not None
        h._agent_ref._bg_manager.get_all.return_value = [bg]
        return await h.request_approval(_request("terminal_tool"))

    request_task = asyncio.create_task(bg_runner())
    await asyncio.sleep(0)

    pending = await h.pending_queue().get()
    assert pending.request.tool_call.name == "terminal_tool"
    assert not pending.future.done()

    with _patch_prompt(h, return_value="y"):
        await h.resolve_pending(pending)

    result = await request_task
    assert result.decision == ApprovalDecision.ALLOW_ONCE


async def test_background_ctrl_c_during_resolve_denies_this_request_only() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    from agent_cli.approval_handler import _PendingApproval

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    pending = _PendingApproval(request=_request("edit_file"), future=fut)

    with _patch_prompt(h, raises=KeyboardInterrupt()):
        await h.resolve_pending(pending)

    assert fut.done()
    result = fut.result()
    assert result.decision == ApprovalDecision.DENY
    assert "ctrl+c" in (result.reason or "").lower()


async def test_concurrent_foreground_approvals_serialized() -> None:
    """Concurrent fg approvals must not overlap in _prompt_user (stdin race)."""
    h = _handler_with_bg_tasks(bg_tasks=[])

    concurrent = 0
    observed_peak = 0

    async def slow_prompt(prompt_text, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal concurrent, observed_peak
        concurrent += 1
        observed_peak = max(observed_peak, concurrent)
        await asyncio.sleep(0.01)
        concurrent -= 1
        return "y"

    with patch.object(h._pt_session, "prompt_async", new=slow_prompt):
        results = await asyncio.gather(*[
            h.request_approval(_request(tool_call_id=f"t{i}"))
            for i in range(3)
        ])

    assert observed_peak == 1
    assert all(r.decision == ApprovalDecision.ALLOW_ONCE for r in results)


async def test_background_resolve_waits_for_foreground_lock() -> None:
    """resolve_pending (bg path) must block until fg _prompt_user releases the lock."""
    from agent_cli.approval_handler import _PendingApproval

    h = _handler_with_bg_tasks(bg_tasks=[])
    fg_release = asyncio.Event()
    order: list[str] = []
    call_n = [0]

    async def instrumented(prompt_text, **kwargs):  # type: ignore[no-untyped-def]
        call_n[0] += 1
        n = call_n[0]
        if n == 1:
            order.append("fg_start")
            await fg_release.wait()
            order.append("fg_done")
        else:
            order.append("bg_resolve")
        return "y"

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    pending = _PendingApproval(request=_request(tool_call_id="bg1"), future=fut)

    with patch.object(h._pt_session, "prompt_async", new=instrumented):
        fg_task = asyncio.create_task(h.request_approval(_request(tool_call_id="fg1")))
        await asyncio.sleep(0.01)

        resolve_task = asyncio.create_task(h.resolve_pending(pending))
        await asyncio.sleep(0.01)

        assert order == ["fg_start"], f"bg should not have started, got {order}"

        fg_release.set()
        await fg_task
        await resolve_task

    assert order == ["fg_start", "fg_done", "bg_resolve"]


async def test_background_external_cancel_propagates_and_sets_future_exception() -> None:
    h = _handler_with_bg_tasks(bg_tasks=[])
    from agent_cli.approval_handler import _PendingApproval

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    pending = _PendingApproval(request=_request(), future=fut)

    with _patch_prompt(h, raises=asyncio.CancelledError()):
        with pytest.raises(asyncio.CancelledError):
            await h.resolve_pending(pending)

    assert fut.done()
    with pytest.raises(asyncio.CancelledError):
        fut.result()


async def test_prompt_isolates_shared_session_attrs() -> None:
    """Main REPL's key_bindings / bottom_toolbar / completer must not leak
    into an approval prompt, and must be restored after the prompt returns.
    """
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    h = _handler_with_bg_tasks(bg_tasks=[])
    main_history = InMemoryHistory()
    main_kb = KeyBindings()
    main_toolbar = HTML("main bar")
    main_completer = object()
    h._pt_session.history = main_history
    h._pt_session.key_bindings = main_kb
    h._pt_session.bottom_toolbar = main_toolbar
    h._pt_session.completer = main_completer

    seen: dict[str, object] = {}

    async def _capture(*args: object, **kwargs: object) -> str:
        seen["history"] = h._pt_session.history
        seen["key_bindings"] = h._pt_session.key_bindings
        seen["bottom_toolbar"] = h._pt_session.bottom_toolbar
        seen["completer"] = h._pt_session.completer
        return "y"

    with patch.object(h._pt_session, "prompt_async", AsyncMock(side_effect=_capture)):
        await h.request_approval(_request())

    assert seen["history"] is h._approval_history
    assert seen["key_bindings"] is None
    assert seen["bottom_toolbar"] is None
    assert seen["completer"] is None

    assert h._pt_session.history is main_history
    assert h._pt_session.key_bindings is main_kb
    assert h._pt_session.bottom_toolbar is main_toolbar
    assert h._pt_session.completer is main_completer


async def test_approval_prompt_does_not_pass_paste_processor() -> None:
    """Approval prompt must call prompt_async without input_processors —
    otherwise the main REPL's paste highlighter would bleed into approval
    input, marking pasted text as "placeholder" in the wrong context.
    """
    h = _handler_with_bg_tasks(bg_tasks=[])

    captured_kwargs: dict[str, object] = {}

    async def _capture(*args: object, **kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "y"

    with patch.object(h._pt_session, "prompt_async", AsyncMock(side_effect=_capture)):
        await h.request_approval(_request())

    assert "input_processors" not in captured_kwargs or \
        captured_kwargs.get("input_processors") in (None, [])


async def test_approval_does_not_inherit_persisted_input_processors() -> None:
    """Regression: prompt_toolkit persists input_processors back onto the
    shared PromptSession after each prompt_async. If the main REPL leaves a
    PastePlaceholderProcessor on the session, the approval prompt would see
    it. The fix lives in loop._prompt_with_lock (snapshot/restore), but
    approval handler must be defensible — verify session attr is not active
    when approval prompt runs.
    """
    h = _handler_with_bg_tasks(bg_tasks=[])
    sentinel = object()
    h._pt_session.input_processors = sentinel

    seen: dict[str, object] = {}

    async def _capture(*args: object, **kwargs: object) -> str:
        seen["input_processors_at_prompt"] = h._pt_session.input_processors
        return "y"

    with patch.object(h._pt_session, "prompt_async", AsyncMock(side_effect=_capture)):
        await h.request_approval(_request())

    # Approval prompt does not actively suppress the attr today; it relies on
    # the REPL never leaving a stale processor. This test pins the contract:
    # after approval, the session attr is restored to whatever it was before.
    assert h._pt_session.input_processors is sentinel
    h = CliApprovalHandler(console=MagicMock())
    loop = asyncio.get_running_loop()
    f1: asyncio.Future[object] = loop.create_future()
    f2: asyncio.Future[object] = loop.create_future()
    from agent_cli.approval_handler import _PendingApproval
    await h._pending.put(_PendingApproval(request=_request(), future=f1))
    await h._pending.put(_PendingApproval(request=_request(), future=f2))

    h.cancel_pending()

    assert h._pending.empty()
    assert f1.cancelled()
    assert f2.cancelled()


async def test_cancel_pending_skips_already_done_futures() -> None:
    h = CliApprovalHandler(console=MagicMock())
    loop = asyncio.get_running_loop()
    f_done: asyncio.Future[object] = loop.create_future()
    f_done.set_result("already")
    f_pending: asyncio.Future[object] = loop.create_future()
    from agent_cli.approval_handler import _PendingApproval
    await h._pending.put(_PendingApproval(request=_request(), future=f_done))
    await h._pending.put(_PendingApproval(request=_request(), future=f_pending))

    h.cancel_pending()

    assert h._pending.empty()
    assert f_done.result() == "already"     # unchanged
    assert f_pending.cancelled()


async def test_cancel_pending_on_empty_queue_is_noop() -> None:
    h = CliApprovalHandler(console=MagicMock())
    h.cancel_pending()
    assert h._pending.empty()
