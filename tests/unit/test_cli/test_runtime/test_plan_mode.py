from unittest.mock import MagicMock

from agent_cli.runtime import plan_mode
from agent_harness.context.context import AgentContext


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.context = AgentContext()
    agent._session_metadata_extras = {}
    return agent


def teardown_function() -> None:
    plan_mode._active.clear()


def test_is_active_default_false() -> None:
    agent = _agent()
    assert plan_mode.is_active(agent) is False


def test_enter_marks_active_and_appends_patch() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    assert plan_mode.is_active(agent) is True
    assert len(agent.context.context_patches) == 1
    patch = agent.context.context_patches[0]
    assert patch.at == "tail"


def test_enter_is_idempotent() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    plan_mode.enter(agent)
    assert len(agent.context.context_patches) == 1


def test_exit_marks_inactive_without_removing_patch() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    plan_mode.exit(agent)
    assert plan_mode.is_active(agent) is False
    # patch is left in list; build() returns None when not active
    assert len(agent.context.context_patches) == 1
    assert agent.context.context_patches[0].build() is None


def test_patch_build_returns_reminder_message_when_active() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    msg = agent.context.context_patches[0].build()
    assert msg is not None
    assert msg.role.value == "user"
    assert msg.content is not None
    assert "<system-reminder>" in msg.content
    assert "Plan mode is active" in msg.content


def test_patch_build_returns_none_after_exit() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    plan_mode.exit(agent)
    assert agent.context.context_patches[0].build() is None


def test_research_agent_max_substituted_in_reminder() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    msg = agent.context.context_patches[0].build()
    assert msg is not None
    assert f"up to {plan_mode.research_agent_max} in parallel" in (msg.content or "")


def test_enter_syncs_metadata_extras() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    assert agent._session_metadata_extras["_plan_mode"] is True


def test_exit_syncs_metadata_extras() -> None:
    agent = _agent()
    plan_mode.enter(agent)
    plan_mode.exit(agent)
    assert agent._session_metadata_extras["_plan_mode"] is False
