from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime.goal import driver, evaluator, mode
from agent_harness.context.context import AgentContext


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.context = AgentContext()
    agent._session_metadata_extras = {}
    return agent


def teardown_function() -> None:
    mode._goals.clear()


async def test_inactive_goal_skips_evaluator(monkeypatch) -> None:
    evaluate = AsyncMock()
    monkeypatch.setattr(driver, "evaluate", evaluate)
    assert await driver.decide(_agent()) is None
    evaluate.assert_not_awaited()


@pytest.mark.parametrize("status", ["complete", "blocked"])
async def test_terminal_verdict_finishes_goal(monkeypatch, status: str) -> None:
    agent = _agent()
    goal = mode.begin(agent, "x")
    monkeypatch.setattr(
        driver,
        "evaluate",
        AsyncMock(return_value=evaluator.GoalVerdict(status, "terminal", "")),
    )
    decision = await driver.decide(agent)
    assert decision is not None
    assert decision.status == status
    assert decision.continuation is None
    assert goal.status == status
    assert goal.turns == 0


async def test_continue_returns_persisted_continuation(monkeypatch) -> None:
    agent = _agent()
    goal = mode.begin(agent, "x")
    monkeypatch.setattr(
        driver,
        "evaluate",
        AsyncMock(return_value=evaluator.GoalVerdict(
            "continue", "gap", "perform missing work"
        )),
    )
    decision = await driver.decide(agent)
    assert decision is not None
    assert decision.status == "continue"
    assert decision.continuation is not None
    assert mode.is_goal_continuation_message(decision.continuation)
    assert goal.reason == "gap"


@pytest.mark.parametrize("change", ["pause", "clear", "replace"])
async def test_stale_evaluator_result_is_ignored(monkeypatch, change: str) -> None:
    agent = _agent()
    original = mode.begin(agent, "old")

    async def evaluate(*args, **kwargs):
        if change == "pause":
            mode.pause(agent)
        elif change == "clear":
            mode.clear(agent)
        else:
            mode.begin(agent, "new")
        return evaluator.GoalVerdict("complete", "done", "")

    monkeypatch.setattr(driver, "evaluate", evaluate)
    assert await driver.decide(agent) is None
    current = mode.get_state(agent)
    if change == "replace":
        assert current is not original
        assert current is not None and current.objective == "new"
    elif change == "pause":
        assert original.status == "paused"
    else:
        assert current is None


async def test_evaluation_error_propagates(monkeypatch) -> None:
    agent = _agent()
    mode.begin(agent, "x")
    monkeypatch.setattr(
        driver,
        "evaluate",
        AsyncMock(side_effect=evaluator.GoalEvaluationError("bad")),
    )
    with pytest.raises(evaluator.GoalEvaluationError):
        await driver.decide(agent)
    assert mode.is_active(agent)


async def test_transition_race_returns_none(monkeypatch) -> None:
    agent = _agent()
    mode.begin(agent, "x")
    monkeypatch.setattr(
        driver,
        "evaluate",
        AsyncMock(return_value=evaluator.GoalVerdict("complete", "done", "")),
    )
    monkeypatch.setattr(mode, "finish", MagicMock(return_value=None))
    assert await driver.decide(agent) is None

    monkeypatch.setattr(
        driver,
        "evaluate",
        AsyncMock(return_value=evaluator.GoalVerdict("continue", "gap", "next")),
    )
    monkeypatch.setattr(mode, "make_continuation_message", MagicMock(return_value=None))
    assert await driver.decide(agent) is None
