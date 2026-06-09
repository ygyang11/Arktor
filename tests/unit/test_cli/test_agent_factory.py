import sys

import pytest

from agent_app.tools import BUILTIN_TOOLS
from agent_cli.agent_factory import create_cli_agent
from agent_cli.prompt import USER_ACTIONS_NAME
from agent_harness.agent.react import ReActAgent
from agent_harness.hooks.progress import ProgressHooks


def test_cli_agent_is_react_with_expected_defaults() -> None:
    agent = create_cli_agent()
    assert isinstance(agent, ReActAgent)
    assert agent.name == "cli"
    assert agent._stream is True
    assert agent.max_steps == sys.maxsize
    assert isinstance(agent.hooks, ProgressHooks)


def test_cli_agent_exposes_all_builtin_tools() -> None:
    agent = create_cli_agent()
    actual = {t.name for t in agent.tools}
    expected = {t.name for t in BUILTIN_TOOLS}
    assert actual == expected


def test_cli_agent_system_prompt_contains_user_actions_section() -> None:
    agent = create_cli_agent()
    assert "# User Side-Channel Actions" in agent.system_prompt
    assert "<user-shell-run>" in agent.system_prompt


def test_cli_agent_user_actions_after_guidelines_before_tools() -> None:
    agent = create_cli_agent()
    sp = agent.system_prompt
    actions_pos = sp.find("# User Side-Channel Actions")
    assert actions_pos >= 0
    builder = agent._prompt_builder
    names = builder.section_names
    assert USER_ACTIONS_NAME in names
    assert names.index("guidelines") < names.index(USER_ACTIONS_NAME)
    assert names.index(USER_ACTIONS_NAME) < names.index("tools")


def test_passing_handler_skips_cli_approval_handler_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_cli.approval_handler as ah_mod
    from agent_harness.approval.handler import AutoApproveHandler

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("CliApprovalHandler must not be constructed")

    monkeypatch.setattr(ah_mod.CliApprovalHandler, "__init__", _boom)
    agent = create_cli_agent(approval_handler=AutoApproveHandler())
    assert isinstance(agent, ReActAgent)
    assert isinstance(agent._approval_handler, AutoApproveHandler)


def test_sub_agent_fork_drops_user_actions_section() -> None:
    agent = create_cli_agent()
    child = agent._prompt_builder.fork()
    assert child.has(USER_ACTIONS_NAME) is False
    assert "User Side-Channel Actions" not in child.build(
        agent._make_builder_context(),
    )
    assert agent._prompt_builder.has(USER_ACTIONS_NAME) is True
