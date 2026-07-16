from unittest.mock import AsyncMock, MagicMock

from agent_cli.commands.builtin.plan import CMD
from agent_cli.runtime import plan_mode
from agent_cli.runtime.goal import mode as goal_mode
from agent_harness.context.context import AgentContext

from ..conftest import render_output


def _ctx(approval_mode: str = "auto") -> MagicMock:
    agent = MagicMock()
    agent.context = AgentContext()
    agent._approval.mode = approval_mode
    return MagicMock(agent=agent, save_session=AsyncMock())


def teardown_function() -> None:
    plan_mode._active.clear()
    goal_mode._goals.clear()


async def test_plan_enter_without_args_returns_entered_text() -> None:
    ctx = _ctx()
    result = await CMD.handler(ctx, "")
    assert plan_mode.is_active(ctx.agent) is True
    out = render_output(result.output)
    assert "Plan mode entered" in out
    assert result.agent_input is None
    ctx.save_session.assert_awaited_once()


async def test_plan_enter_with_arg_passes_agent_input() -> None:
    ctx = _ctx()
    result = await CMD.handler(ctx, "design /diff panel")
    assert plan_mode.is_active(ctx.agent) is True
    assert result.agent_input == "design /diff panel"


async def test_plan_exit_when_active() -> None:
    ctx = _ctx()
    plan_mode.enter(ctx.agent)
    result = await CMD.handler(ctx, "")
    assert plan_mode.is_active(ctx.agent) is False
    assert "Plan mode exited" in render_output(result.output)
    ctx.save_session.assert_awaited_once()


async def test_plan_entered_output_carries_mode_label() -> None:
    ctx = _ctx("ask")
    out = render_output((await CMD.handler(ctx, "")).output)
    assert "Ask Approval" in out


async def test_plan_rejects_active_goal_but_allows_paused_goal() -> None:
    ctx = _ctx()
    ctx.agent._session_metadata_extras = {}
    goal_mode.begin(ctx.agent, "x")
    result = await CMD.handler(ctx, "")
    assert "goal is active" in render_output(result.output)
    assert not plan_mode.is_active(ctx.agent)

    goal_mode.pause(ctx.agent)
    result = await CMD.handler(ctx, "")
    assert "Plan mode entered" in render_output(result.output)
    assert plan_mode.is_active(ctx.agent)
