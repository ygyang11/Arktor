from __future__ import annotations

from unittest.mock import MagicMock

from agent_cli.runtime.goal import mode
from agent_harness.context.context import AgentContext
from agent_harness.core.message import Message


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.context = AgentContext()
    agent._session_metadata_extras = {}
    return agent


def teardown_function() -> None:
    mode._goals.clear()


def test_begin_and_live_ledger(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr(mode.time, "monotonic", lambda: now[0])
    agent = _agent()
    agent.context.usage_meter.total.total_tokens = 100

    goal = mode.begin(agent, "ship it")
    agent.context.usage_meter.total.total_tokens = 145
    now[0] = 14.8

    assert mode.get_state(agent) is goal
    assert mode.is_active(agent)
    assert mode.has_live_goal(agent)
    assert goal.elapsed_s() == 4
    assert goal.tokens_used(145) == 45
    assert goal.accumulated_tokens == 0


def test_pause_resume_and_finish_fold_live_usage(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr(mode.time, "monotonic", lambda: now[0])
    agent = _agent()
    agent.context.usage_meter.total.total_tokens = 10
    goal = mode.begin(agent, "objective")

    agent.context.usage_meter.total.total_tokens = 30
    now[0] = 13.5
    assert mode.pause(agent, "hold") is goal
    assert goal.status == "paused"
    assert goal.accumulated_s == 3.5
    assert goal.accumulated_tokens == 20
    assert goal.tokens_used(100) == 20

    agent.context.usage_meter.total.total_tokens = 50
    now[0] = 20.0
    assert mode.resume(agent) is goal
    assert goal._process_token_baseline == 50

    agent.context.usage_meter.total.total_tokens = 65
    now[0] = 22.0
    assert mode.finish(agent, "complete", "done") is goal
    assert goal.status == "complete"
    assert goal.elapsed_s() == 5
    assert goal.accumulated_tokens == 35
    assert not mode.has_live_goal(agent)


def test_invalid_transitions_return_none() -> None:
    agent = _agent()
    assert mode.pause(agent) is None
    assert mode.resume(agent) is None
    assert mode.finish(agent, "blocked", "x") is None

    goal = mode.begin(agent, "x")
    assert mode.resume(agent) is None
    assert mode.pause(agent) is goal
    assert mode.pause(agent) is None
    assert mode.finish(agent, "complete", "x") is None


def test_record_turn_persists_live_snapshot_without_folding() -> None:
    agent = _agent()
    agent.context.usage_meter.total.total_tokens = 10
    goal = mode.begin(agent, "x")
    baseline = goal._process_token_baseline
    agent.context.usage_meter.total.total_tokens = 35

    assert mode.record_completed_turn(agent) is goal
    assert goal.turns == 1
    assert goal.accumulated_tokens == 0
    assert goal._process_token_baseline == baseline
    assert agent._session_metadata_extras["_goal"]["accumulated_tokens"] == 25


def test_restore_live_goal_as_paused_and_clear_invalid_payloads() -> None:
    agent = _agent()
    restored = mode.restore(agent, {
        "objective": "restore me",
        "status": "active",
        "reason": "latest",
        "turns": 3,
        "accumulated_s": 4.5,
        "accumulated_tokens": 90,
    })
    assert restored is not None
    assert restored.status == "paused"
    assert restored.turns == 3
    assert restored.accumulated_s == 4.5
    assert restored.accumulated_tokens == 90
    assert restored._start is None
    assert restored._process_token_baseline == 0

    for raw in (
        None,
        {"objective": "", "status": "active"},
        {"objective": "x", "status": "complete"},
    ):
        assert mode.restore(agent, raw) is None
        assert mode.get_state(agent) is None
        assert "_goal" not in agent._session_metadata_extras


def test_restore_invalid_ledger_fields_default_to_zero() -> None:
    agent = _agent()
    goal = mode.restore(agent, {
        "objective": "x",
        "status": "paused",
        "turns": True,
        "accumulated_s": True,
        "accumulated_tokens": -1,
    })
    assert goal is not None
    assert goal.turns == 0
    assert goal.accumulated_s == 0
    assert goal.accumulated_tokens == 0


def test_clear_removes_runtime_and_metadata() -> None:
    agent = _agent()
    mode.begin(agent, "x")
    mode.clear(agent)
    assert mode.get_state(agent) is None
    assert "_goal" not in agent._session_metadata_extras


def test_start_resume_and_continuation_prompts() -> None:
    agent = _agent()
    goal = mode.begin(agent, "full objective")

    start = mode.make_start_input(goal.objective)
    resume = mode.make_resume_message(goal.objective)
    continuation = mode.make_continuation_message(
        agent,
        "more work",
        "run the missing verification",
    )

    assert isinstance(start, str)
    assert "full objective" in start
    assert "initial worker turn" in start
    assert "previous review" not in start.lower()
    assert resume.content != start
    assert "[Goal continuation]" in (resume.content or "")
    assert "full objective" in (resume.content or "")
    assert mode.is_goal_continuation_message(resume)
    assert continuation is not None
    assert "[Goal continuation]" in (continuation.content or "")
    assert "run the missing verification" in (continuation.content or "")
    assert mode.is_goal_continuation_message(continuation)
    assert goal.reason == "more work"

    assert not mode.is_goal_continuation_message(Message.user("plain"))
    mode.pause(agent)
    assert mode.make_continuation_message(agent, "x", "y") is None
