from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agent_cli.commands.builtin import goal, register_builtin
from agent_cli.commands.registry import CommandRegistry
from agent_cli.runtime import plan_mode
from agent_cli.runtime.goal import mode as goal_mode
from agent_harness.context.context import AgentContext

from ..conftest import render_output


def _ctx() -> MagicMock:
    agent = MagicMock()
    agent.context = AgentContext()
    agent._session_metadata_extras = {}
    agent.llm.model_name = "test-model"
    return MagicMock(agent=agent, save_session=AsyncMock())


def teardown_function() -> None:
    goal_mode._goals.clear()
    plan_mode._active.clear()


async def test_view_renders_live_tokens() -> None:
    ctx = _ctx()
    ctx.agent.context.usage_meter.total.total_tokens = 10
    goal_mode.begin(ctx.agent, "objective")
    ctx.agent.context.usage_meter.total.total_tokens = 35
    result = await goal.CMD.handler(ctx, "")
    output = render_output(result.output)
    assert "Goal" in output
    assert "objective" in output
    assert "25" in output


async def test_set_goal_saves_and_returns_start_string() -> None:
    ctx = _ctx()
    result = await goal.CMD.handler(ctx, "ship the release")
    state = goal_mode.get_state(ctx.agent)
    assert state is not None and state.objective == "ship the release"
    assert isinstance(result.agent_input, str)
    assert result.agent_input == goal_mode.make_start_input("ship the release")
    ctx.save_session.assert_awaited_once()


async def test_set_rejects_plan_and_live_goal() -> None:
    ctx = _ctx()
    plan_mode.enter(ctx.agent)
    result = await goal.CMD.handler(ctx, "x")
    assert "plan mode" in render_output(result.output)
    plan_mode.exit(ctx.agent)

    goal_mode.begin(ctx.agent, "existing")
    result = await goal.CMD.handler(ctx, "replacement")
    output = render_output(result.output)
    assert "already exists" in output
    assert "/goal clear" in output
    goal_mode.pause(ctx.agent)
    result = await goal.CMD.handler(ctx, "replacement")
    assert "already exists" in render_output(result.output)


async def test_terminal_goal_can_be_replaced() -> None:
    ctx = _ctx()
    goal_mode.begin(ctx.agent, "old")
    goal_mode.finish(ctx.agent, "complete", "done")
    await goal.CMD.handler(ctx, "new")
    state = goal_mode.get_state(ctx.agent)
    assert state is not None and state.objective == "new"


async def test_large_objective_rejected_with_file_hint(monkeypatch) -> None:
    ctx = _ctx()
    monkeypatch.setattr(goal, "count_tokens", lambda *args, **kwargs: 2_001)
    result = await goal.CMD.handler(ctx, "large")
    output = render_output(result.output)
    assert "too large" in output
    assert "file" in output
    assert goal_mode.get_state(ctx.agent) is None


async def test_pause_resume_clear_controls() -> None:
    ctx = _ctx()
    goal_mode.begin(ctx.agent, "x")
    paused = await goal.CMD.handler(ctx, "pause")
    assert "paused" in render_output(paused.output).lower()
    assert goal_mode.get_state(ctx.agent).status == "paused"

    resumed = await goal.CMD.handler(ctx, "resume")
    assert goal_mode.get_state(ctx.agent).status == "active"
    assert resumed.agent_input is not None
    assert goal_mode.is_goal_continuation_message(resumed.agent_input)

    cleared = await goal.CMD.handler(ctx, "clear")
    assert "cleared" in render_output(cleared.output).lower()
    assert goal_mode.get_state(ctx.agent) is None
    assert ctx.save_session.await_count == 3


async def test_control_state_and_plan_errors() -> None:
    ctx = _ctx()
    assert "No active" in render_output((await goal.CMD.handler(ctx, "pause")).output)
    assert "No paused" in render_output((await goal.CMD.handler(ctx, "resume")).output)

    goal_mode.begin(ctx.agent, "x")
    goal_mode.pause(ctx.agent)
    plan_mode.enter(ctx.agent)
    result = await goal.CMD.handler(ctx, "resume")
    assert "plan mode" in render_output(result.output)
    assert goal_mode.get_state(ctx.agent).status == "paused"


async def test_control_word_with_extra_text_is_objective() -> None:
    ctx = _ctx()
    await goal.CMD.handler(ctx, "pause deploy after tests")
    state = goal_mode.get_state(ctx.agent)
    assert state is not None and state.objective == "pause deploy after tests"


def test_builtin_registration_exposes_goal() -> None:
    registry = CommandRegistry()
    register_builtin(registry)
    assert registry.has("/goal")
