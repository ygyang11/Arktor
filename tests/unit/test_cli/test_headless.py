"""Tests for headless run mode — `arktor -p / --prompt`."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_cli.app import _build_parser
from agent_cli.headless import _parse_task, run_headless
from agent_cli.runtime import plan_mode
from agent_cli.runtime.goal import mode as goal_mode
from agent_cli.runtime.goal.driver import GoalDecision
from agent_harness import AgentResult
from agent_harness.agent.base import StepResult
from agent_harness.approval.handler import AutoApproveHandler
from agent_harness.approval.types import ApprovalDecision, ApprovalRequest
from agent_harness.core.message import Message, ToolCall
from agent_harness.hooks.base import DefaultHooks
from agent_harness.llm.types import ProcessUsageMeter, Usage


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "agent_harness.session.file_session._DEFAULT_SESSION_DIR", tmp_path,
    )
    return tmp_path


def _args(*argv: str) -> argparse.Namespace:
    return _build_parser().parse_args(list(argv))


class _FakeApproval:
    def __init__(self, mode: str) -> None:
        self._mode = mode
        self.history = [mode]

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, m: str) -> None:
        self._mode = m
        self.history.append(m)


class _FakeAgent:
    def __init__(
        self, *, output: str = "", error: Exception | None = None,
        mode: str = "auto", steps: list[StepResult] | None = None,
    ) -> None:
        self._approval = _FakeApproval(mode)
        self._session_metadata_extras: dict[str, Any] = {}
        self.run_mode_seen: str | None = None
        self.llm = SimpleNamespace(model_name="test-model")
        self.context = SimpleNamespace(
            short_term_memory=SimpleNamespace(
                last_call=None, max_tokens=200_000, displayed_input_tokens=1234,
            ),
            usage_meter=ProcessUsageMeter(),
        )
        if error is not None:
            self.run = AsyncMock(side_effect=error)
        else:
            res = AgentResult(output=output, steps=steps or [], usage=Usage())

            async def _run(task: str, *, session: Any = None, **kw: Any) -> AgentResult:
                self.run_mode_seen = self._approval.mode
                return res

            self.run = _run  # type: ignore[assignment]

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def clear_runtime_modes():  # type: ignore[no-untyped-def]
    goal_mode._goals.clear()
    plan_mode._active.clear()
    yield
    goal_mode._goals.clear()
    plan_mode._active.clear()


@pytest.fixture
def patch_runtime(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {}

    def _install(agent: _FakeAgent) -> dict[str, Any]:
        def fake_create(*, hooks: Any = None, approval_handler: Any = None, **kw: Any) -> Any:
            captured["hooks"] = hooks
            captured["approval_handler"] = approval_handler
            return agent

        monkeypatch.setattr("agent_cli.agent_factory.create_cli_agent", fake_create)
        monkeypatch.setattr("agent_cli.config.load_config", lambda: None)
        save = AsyncMock()
        captured["save"] = save
        monkeypatch.setattr(
            "agent_cli.runtime.session.make_save_session", lambda a, b: save,
        )
        monkeypatch.setattr("agent_cli.runtime.session.stop_sandbox", AsyncMock())
        monkeypatch.setattr("agent_cli.runtime.background.has_running", lambda a: False)
        monkeypatch.setattr(
            "agent_cli.runtime.background.collect_results",
            AsyncMock(return_value=False),
        )
        monkeypatch.setattr(
            "agent_cli.runtime.background.cancel_all_with_note",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr("agent_cli.runtime.background.shutdown", AsyncMock())
        return captured

    return _install


async def test_empty_prompt_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = await run_headless(_args("-p", "   "))
    assert rc == 2
    assert "non-empty task" in capsys.readouterr().err


async def test_runs_and_prints_only_output(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    patch_runtime(_FakeAgent(output="final answer"))
    rc = await run_headless(_args("-p", "do the thing"))
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "final answer\n"
    assert out.err == ""


async def test_unrestricted_then_restores_mode(
    session_dir: Path, patch_runtime: Any,
) -> None:
    agent = _FakeAgent(output="x", mode="auto")
    patch_runtime(agent)
    await run_headless(_args("-p", "go"))
    assert agent.run_mode_seen == "never"
    assert agent._approval.mode == "auto"
    assert "never" in agent._approval.history


async def test_uses_silent_hooks_and_auto_approve(
    session_dir: Path, patch_runtime: Any,
) -> None:
    captured = patch_runtime(_FakeAgent(output="x"))
    await run_headless(_args("-p", "go"))
    assert isinstance(captured["hooks"], DefaultHooks)
    assert isinstance(captured["approval_handler"], AutoApproveHandler)


async def test_run_error_returns_1_and_restores_mode(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent(error=RuntimeError("boom"), mode="auto")
    patch_runtime(agent)
    rc = await run_headless(_args("-p", "go"))
    out = capsys.readouterr()
    assert rc == 1
    assert "boom" in out.err
    assert out.out == ""
    assert agent._approval.mode == "auto"


async def test_cleanup_failure_preserves_output(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patch_runtime(_FakeAgent(output="result"))
    monkeypatch.setattr(
        "agent_cli.runtime.session.stop_sandbox",
        AsyncMock(side_effect=RuntimeError("docker boom")),
    )
    rc = await run_headless(_args("-p", "go"))
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == "result\n"
    assert "stop sandbox" in out.err
    assert "docker boom" in out.err


async def test_json_output_emits_ndjson_steps_and_result(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    steps = [StepResult(thought="thinking"), StepResult(response="done")]
    patch_runtime(_FakeAgent(output="done", steps=steps))
    rc = await run_headless(_args("-p", "go", "-s", "sess1", "--output-format", "json"))
    out = capsys.readouterr()
    assert rc == 0
    assert out.err == ""
    lines = [json.loads(x) for x in out.out.splitlines()]
    assert [line["type"] for line in lines] == ["step", "step", "result"]
    assert lines[0]["index"] == 0
    assert lines[0]["thought"] == "thinking"
    res = lines[-1]
    assert res["is_error"] is False
    assert res["session_id"] == "sess1"
    assert res["output"] == "done"
    assert res["num_steps"] == 2
    assert "usage" in res
    assert res["context"] == {"input_tokens": 1234, "max_tokens": 200_000}


async def test_json_output_error_is_single_result_line(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    patch_runtime(_FakeAgent(error=RuntimeError("kaboom")))
    rc = await run_headless(_args("-p", "go", "-s", "sess2", "--output-format", "json"))
    out = capsys.readouterr()
    assert rc == 1
    assert out.err == ""  # json mode: error lives in the result line, not stderr
    lines = [json.loads(x) for x in out.out.splitlines()]
    assert lines == [{
        "type": "result", "is_error": True, "session_id": "sess2",
        "output": None, "error": "kaboom",
    }]


async def test_corrupted_resume_returns_2(
    session_dir: Path, patch_runtime: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    (session_dir / "wrecked.json").write_text("garbage", encoding="utf-8")
    patch_runtime(_FakeAgent(output="never reached"))
    rc = await run_headless(_args("-p", "go", "-r", "wrecked"))
    assert rc == 2
    assert "corrupted" in capsys.readouterr().err


async def test_auto_approve_handler_allows_once() -> None:
    handler = AutoApproveHandler()
    req = ApprovalRequest(
        tool_call=ToolCall(id="t1", name="terminal_tool", arguments={}),
        agent_name="cli",
    )
    res = await handler.request_approval(req)
    assert res.decision == ApprovalDecision.ALLOW_ONCE
    assert res.tool_call_id == "t1"
    assert res.tool_name == "terminal_tool"


def _result(output: str, tokens: int, label: str) -> AgentResult:
    return AgentResult(
        output=output,
        steps=[StepResult(response=label)],
        usage=Usage(
            prompt_tokens=tokens - 1,
            completion_tokens=1,
            total_tokens=tokens,
        ),
    )


def _install_runs(
    agent: _FakeAgent,
    results: list[AgentResult | BaseException],
) -> list[object]:
    inputs: list[object] = []
    pending = iter(results)

    async def run(inp: object, *, session: Any = None, **kwargs: Any) -> AgentResult:
        inputs.append(inp)
        item = next(pending)
        if isinstance(item, BaseException):
            raise item
        agent.context.usage_meter.record(
            item.usage,
            model=agent.llm.model_name,
            source="main",
        )
        return item

    agent.run = run  # type: ignore[assignment]
    return inputs


async def test_goal_continues_then_completes_with_aggregate_result(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    inputs = _install_runs(agent, [
        _result("first", 10, "step-a"),
        _result("SMOKE-DONE", 20, "step-b"),
    ])
    patch_runtime(agent)
    calls = 0

    async def decide(current: _FakeAgent) -> GoalDecision:
        nonlocal calls
        calls += 1
        if calls == 1:
            continuation = goal_mode.make_continuation_message(
                current, "verification remains", "run final verification"
            )
            assert continuation is not None
            return GoalDecision("continue", "verification remains", continuation)
        assert goal_mode.finish(current, "complete", "all verified") is not None
        return GoalDecision("complete", "all verified")

    monkeypatch.setattr("agent_cli.runtime.goal.driver.decide", decide)
    rc = await run_headless(_args(
        "-p", "/goal ship it", "-s", "goal-ok", "--output-format", "json"
    ))

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert [event["type"] for event in events] == [
        "step", "goal", "step", "goal", "result"
    ]
    assert [event["index"] for event in events if event["type"] == "step"] == [0, 1]
    goals = [event for event in events if event["type"] == "goal"]
    assert [event["status"] for event in goals] == ["continue", "complete"]
    assert goals[-1]["turns"] == 2
    result = events[-1]
    assert result["is_error"] is False
    assert result["output"] == "SMOKE-DONE"
    assert result["num_steps"] == 2
    assert result["usage"]["total_tokens"] == goals[-1]["tokens"] == 30
    assert isinstance(inputs[0], str)
    assert goal_mode.is_goal_continuation_message(inputs[1])


async def test_goal_blocked_returns_3_and_success_result(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    _install_runs(agent, [_result("checked dependency", 5, "check")])
    patch_runtime(agent)

    async def decide(current: _FakeAgent) -> GoalDecision:
        goal_mode.finish(current, "blocked", "external signing key is missing")
        return GoalDecision("blocked", "external signing key is missing")

    monkeypatch.setattr("agent_cli.runtime.goal.driver.decide", decide)
    rc = await run_headless(_args(
        "-p", "/goal sign release", "--output-format", "json"
    ))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 3
    assert [event["status"] for event in events if event["type"] == "goal"] == [
        "blocked"
    ]
    assert events[-1]["type"] == "result"
    assert events[-1]["is_error"] is False


async def test_max_turns_evaluates_then_blocks_without_continuation_turn(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    inputs = _install_runs(agent, [_result("partial", 5, "partial")])
    patch_runtime(agent)
    decide = AsyncMock()

    async def continuing(current: _FakeAgent) -> GoalDecision:
        continuation = goal_mode.make_continuation_message(
            current, "work remains", "continue work"
        )
        assert continuation is not None
        return GoalDecision("continue", "work remains", continuation)

    decide.side_effect = continuing
    monkeypatch.setattr("agent_cli.runtime.goal.driver.decide", decide)
    rc = await run_headless(_args(
        "-p", "/goal multi turn", "--max-turns", "1", "--output-format", "json"
    ))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 3
    decide.assert_awaited_once()
    assert len(inputs) == 1
    goal_events = [event for event in events if event["type"] == "goal"]
    assert len(goal_events) == 1
    assert goal_events[0]["status"] == "blocked"
    assert "max-turns (1)" in goal_events[0]["reason"]


async def test_goal_error_emits_goal_then_error_result_and_pauses(
    session_dir: Path,
    patch_runtime: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    _install_runs(agent, [RuntimeError("boom")])
    captured = patch_runtime(agent)
    rc = await run_headless(_args(
        "-p", "/goal fail", "--output-format", "json"
    ))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 1
    assert [event["type"] for event in events] == ["goal", "result"]
    assert events[0]["status"] == "error"
    assert events[1]["is_error"] is True
    goal = goal_mode.get_state(agent)
    assert goal is not None and goal.status == "paused"
    captured["save"].assert_awaited()


async def test_goal_cancel_propagates_after_pausing_and_saving(
    session_dir: Path,
    patch_runtime: Any,
) -> None:
    agent = _FakeAgent()
    _install_runs(agent, [asyncio.CancelledError()])
    captured = patch_runtime(agent)
    with pytest.raises(asyncio.CancelledError):
        await run_headless(_args("-p", "/goal interrupt"))
    goal = goal_mode.get_state(agent)
    assert goal is not None and goal.status == "paused"
    assert goal.reason == "interrupted"
    captured["save"].assert_awaited()


async def test_normal_task_delivers_background_and_aggregates_json(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    inputs = _install_runs(agent, [
        _result("TASK-BG-STARTED", 5, "start"),
        _result("TASK-BG-DONE", 7, "finish"),
    ])
    patch_runtime(agent)
    monkeypatch.setattr(
        "agent_cli.runtime.background.collect_results",
        AsyncMock(side_effect=[True, False, False, False]),
    )

    rc = await run_headless(_args(
        "-p", "background task", "--output-format", "json"
    ))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert [event["type"] for event in events] == ["step", "step", "result"]
    assert [event["index"] for event in events[:-1]] == [0, 1]
    assert events[-1]["output"] == "TASK-BG-DONE"
    assert events[-1]["num_steps"] == 2
    assert events[-1]["usage"]["total_tokens"] == 12
    assert isinstance(inputs[1], Message)
    assert inputs[1].metadata["is_background_result"] is True
    assert not any(event["type"] == "goal" for event in events)


async def test_goal_background_notification_counts_as_turn_before_evaluation(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    inputs = _install_runs(agent, [
        _result("started", 5, "start"),
        _result("processed", 7, "notification"),
    ])
    patch_runtime(agent)
    monkeypatch.setattr(
        "agent_cli.runtime.background.collect_results",
        AsyncMock(side_effect=[True, False, False, False]),
    )

    async def decide(current: _FakeAgent) -> GoalDecision:
        goal = goal_mode.get_state(current)
        assert goal is not None and goal.turns == 2
        goal_mode.finish(current, "complete", "background evidence verified")
        return GoalDecision("complete", "background evidence verified")

    monkeypatch.setattr("agent_cli.runtime.goal.driver.decide", decide)
    rc = await run_headless(_args(
        "-p", "/goal process background", "--output-format", "json"
    ))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert len(inputs) == 2
    assert isinstance(inputs[1], Message)
    assert [event["index"] for event in events if event["type"] == "step"] == [0, 1]
    assert next(event for event in events if event["type"] == "goal")["turns"] == 2


@pytest.mark.parametrize("argv,error", [
    (("-p", "/goal"), "non-empty objective"),
    (("-p", "/goal pause"), "only accepts a new objective"),
    (("-p", "task", "--max-turns", "2"), "requires a /goal prompt"),
    (("-p", "/goal x", "--max-turns", "0"), "must be positive"),
])
async def test_goal_argument_errors_return_2(
    argv: tuple[str, ...],
    error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await run_headless(_args(*argv)) == 2
    assert error in capsys.readouterr().err


def test_parse_goal_command_token_boundaries() -> None:
    parsed = _parse_task(_args("-p", "/GoAl\tobjective"))
    assert parsed == ("/GoAl\tobjective", "objective")
    parsed = _parse_task(_args("-p", "/goal\nobjective"))
    assert parsed == ("/goal\nobjective", "objective")
    assert _parse_task(_args("-p", "/goalx objective")) == (
        "/goalx objective", None
    )


async def test_oversized_goal_returns_2_with_file_hint(
    session_dir: Path,
    patch_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _FakeAgent()
    captured = patch_runtime(agent)
    monkeypatch.setattr(
        "agent_harness.utils.token_counter.count_tokens",
        lambda *args, **kwargs: goal_mode.MAX_OBJECTIVE_TOKENS + 1,
    )
    assert await run_headless(_args("-p", "/goal huge")) == 2
    output = capsys.readouterr().err
    assert "too large" in output
    assert "file" in output.lower()
    captured["save"].assert_not_awaited()
    assert agent._approval.history == ["auto"]
