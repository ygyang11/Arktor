"""Tests for SubAgentTool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_app.tools.sub_agent.sub_agent import SubAgentTool
from agent_app.tools.sub_agent.types import (
    _ALWAYS_EXCLUDE,
    _BUILTIN_TYPES,
    _SUBAGENT_INTRO,
)
from agent_harness.agent.base import AgentResult, StepResult
from agent_harness.core.config import HarnessConfig, SubAgentConfig, SubAgentTypeSpec
from agent_harness.core.errors import ToolExecutionError, ToolValidationError
from agent_harness.core.message import ToolCall
from agent_harness.hooks.base import DefaultHooks
from agent_harness.prompt.sections import _TOOL_SUPPLEMENTS


# ── Helpers ──


class _MockTool:
    """Minimal mock tool that does NOT match AgentAware protocol."""

    def __init__(self, name: str) -> None:
        self.name = name


def _make_mock_tools(names: list[str]) -> list[_MockTool]:
    return [_MockTool(n) for n in names]


def _make_bound_tool(
    tool_names: list[str] | None = None,
    config: HarnessConfig | None = None,
) -> SubAgentTool:
    tool = SubAgentTool()
    agent = MagicMock()
    agent.tools = _make_mock_tools(
        tool_names
        or [
            "read_file", "write_file", "terminal_tool",
            "web_fetch", "sub_agent", "todo_write", "skill_tool",
        ]
    )
    agent.context.config = config or HarnessConfig()
    agent.name = "test-agent"
    agent.llm = MagicMock(name="main_llm")
    agent.sub_llm = MagicMock(name="sub_llm")
    agent._prompt_builder = MagicMock()
    agent._prompt_builder.fork.return_value = MagicMock()
    tool.bind_agent(agent)
    return tool


# ── Schema ──


class TestSchema:
    def test_schema_name(self) -> None:
        tool = SubAgentTool()
        schema = tool.get_schema()
        assert schema.name == "sub_agent"

    def test_schema_has_required_params(self) -> None:
        tool = SubAgentTool()
        schema = tool.get_schema()
        assert set(schema.parameters["required"]) == {
            "description", "prompt", "agent_type", "model",
        }

    def test_schema_model_contract(self) -> None:
        schema = SubAgentTool().get_schema()
        model = schema.parameters["properties"]["model"]
        assert model["enum"] == ["main", "sub"]
        assert model["default"] == "main"
        assert "including when unsure" in model["description"]

    def test_schema_agent_type_enum_builtin(self) -> None:
        tool = SubAgentTool()
        schema = tool.get_schema()
        enum = schema.parameters["properties"]["agent_type"]["enum"]
        assert "research" in enum
        assert "plan" in enum
        assert "general" in enum

    def test_schema_agent_type_enum_includes_custom(self) -> None:
        tool = _make_bound_tool(
            config=HarnessConfig(
                sub_agent=SubAgentConfig(
                    types={"security": SubAgentTypeSpec(
                        tools=["read_file"],
                        intro="You are a security auditor.",
                    )}
                )
            ),
        )
        schema = tool.get_schema()
        enum = schema.parameters["properties"]["agent_type"]["enum"]
        assert "security" in enum
        assert "research" in enum

    def test_schema_description_contains_types(self) -> None:
        tool = SubAgentTool()
        schema = tool.get_schema()
        assert "research" in schema.description
        assert "plan" in schema.description
        assert "general" in schema.description

    def test_schema_description_dynamic_with_custom_type(self) -> None:
        tool = _make_bound_tool(
            config=HarnessConfig(
                sub_agent=SubAgentConfig(
                    types={"security": SubAgentTypeSpec(
                        tools=["read_file"],
                        intro="You are a security auditor.",
                    )}
                )
            ),
        )
        schema = tool.get_schema()
        assert "security" in schema.description
        assert "security auditor" in schema.description

    def test_description_and_system_section_define_model_selection(self) -> None:
        description = SubAgentTool().get_schema().description
        supplement = _TOOL_SUPPLEMENTS["sub_agent"]

        assert "Choose model independently" in description
        assert 'model="sub"' in description
        assert 'model="main"' in description
        assert "### Model Selection" in supplement
        assert 'Use model="sub"' in supplement
        assert "Use main when unsure" in supplement


# ── Validation ──


class TestValidation:
    async def test_unbound_raises(self) -> None:
        tool = SubAgentTool()
        with pytest.raises(ToolExecutionError, match="not bound"):
            await tool.execute(
                description="test", prompt="do something", agent_type="research",
                model="main",
            )

    async def test_empty_description_raises(self) -> None:
        tool = _make_bound_tool()
        with pytest.raises(ToolValidationError, match="description"):
            await tool.execute(
                description="", prompt="do something", agent_type="research",
                model="main",
            )

    async def test_empty_prompt_raises(self) -> None:
        tool = _make_bound_tool()
        with pytest.raises(ToolValidationError, match="prompt"):
            await tool.execute(
                description="test", prompt="", agent_type="research", model="main",
            )

    async def test_empty_agent_type_raises(self) -> None:
        tool = _make_bound_tool()
        with pytest.raises(ToolValidationError, match="agent_type"):
            await tool.execute(
                description="test", prompt="do something", agent_type="", model="main",
            )

    async def test_invalid_type_raises(self) -> None:
        tool = _make_bound_tool()
        with pytest.raises(ToolValidationError, match="Unknown agent_type"):
            await tool.execute(
                description="test", prompt="do something", agent_type="invalid", model="main",
            )

    @pytest.mark.parametrize("model", [None, "", "other"])
    async def test_invalid_model_raises(self, model: object) -> None:
        tool = _make_bound_tool()
        with pytest.raises(ToolValidationError, match="model"):
            await tool.execute(
                description="test",
                prompt="do something",
                agent_type="research",
                model=model,
            )


# ── Tool resolution ──


class TestToolResolution:
    def test_research_filters_to_whitelist(self) -> None:
        tool = _make_bound_tool()
        resolved = tool._resolve_tools("research")
        names = {t.name for t in resolved}
        assert "read_file" in names
        assert "web_fetch" in names
        assert "write_file" not in names
        assert "terminal_tool" not in names
        assert "sub_agent" not in names
        assert "todo_write" not in names

    def test_general_inherits_minus_excluded(self) -> None:
        tool = _make_bound_tool()
        resolved = tool._resolve_tools("general")
        names = {t.name for t in resolved}
        assert "read_file" in names
        assert "write_file" in names
        assert "terminal_tool" in names
        assert "sub_agent" not in names
        assert "todo_write" not in names
        assert "skill_tool" not in names

    def test_plan_has_readonly_plus_web(self) -> None:
        tool = _make_bound_tool(
            tool_names=[
                "read_file", "list_dir", "glob_files", "grep_files",
                "web_fetch", "write_file", "terminal_tool",
            ],
        )
        resolved = tool._resolve_tools("plan")
        names = {t.name for t in resolved}
        assert "read_file" in names
        assert "web_fetch" in names
        assert "write_file" not in names
        assert "terminal_tool" not in names

    def test_custom_type_from_config(self) -> None:
        tool = _make_bound_tool(
            tool_names=["read_file", "grep_files", "terminal_tool"],
            config=HarnessConfig(
                sub_agent=SubAgentConfig(
                    types={"security": SubAgentTypeSpec(
                        tools=["read_file", "grep_files", "terminal_tool"],
                    )}
                )
            ),
        )
        resolved = tool._resolve_tools("security")
        names = {t.name for t in resolved}
        assert names == {"read_file", "grep_files", "terminal_tool"}

    def test_empty_tools_returns_empty(self) -> None:
        tool = _make_bound_tool(
            config=HarnessConfig(
                sub_agent=SubAgentConfig(
                    types={"empty": SubAgentTypeSpec(tools=[])}
                )
            ),
        )
        resolved = tool._resolve_tools("empty")
        assert resolved == []

    def test_resolve_tools_empty_intersection(self) -> None:
        tool = _make_bound_tool(tool_names=["terminal_tool", "write_file"])
        resolved = tool._resolve_tools("research")
        assert resolved == []


# ── Prompt builder ──


class TestPromptBuilder:
    def test_intro_replaced_by_type(self) -> None:
        tool = _make_bound_tool()
        tool._build_subagent_prompt_builder("research")
        forked = tool._agent._prompt_builder.fork.return_value
        forked.register.assert_called_once()
        section = forked.register.call_args[0][0]
        assert "research sub-agent" in section.content.lower()

    def test_general_intro(self) -> None:
        tool = _make_bound_tool()
        tool._build_subagent_prompt_builder("general")
        forked = tool._agent._prompt_builder.fork.return_value
        section = forked.register.call_args[0][0]
        assert "general sub-agent" in section.content.lower()

    def test_custom_type_uses_config_intro(self) -> None:
        tool = _make_bound_tool(
            config=HarnessConfig(
                sub_agent=SubAgentConfig(
                    types={"security": SubAgentTypeSpec(
                        tools=["read_file"],
                        intro="You are a security auditor.",
                    )}
                )
            ),
        )
        tool._build_subagent_prompt_builder("security")
        forked = tool._agent._prompt_builder.fork.return_value
        section = forked.register.call_args[0][0]
        assert "security auditor" in section.content.lower()

    def test_custom_type_without_intro_falls_back(self) -> None:
        tool = _make_bound_tool(
            config=HarnessConfig(
                sub_agent=SubAgentConfig(
                    types={"custom": SubAgentTypeSpec(tools=["read_file"])}
                )
            ),
        )
        tool._build_subagent_prompt_builder("custom")
        forked = tool._agent._prompt_builder.fork.return_value
        section = forked.register.call_args[0][0]
        assert "sub-agent assisting" in section.content.lower()

    def test_bg_constraint_injected_for_all_types(self) -> None:
        """The background-mode ban must reach EVERY sub-agent prompt —
        built-in, custom-with-intro, and the no-intro fallback — since
        it's the sole (prompt-only) guard against backgrounded results
        being lost in a sub-agent."""
        cfg = HarnessConfig(
            sub_agent=SubAgentConfig(
                types={
                    "security": SubAgentTypeSpec(
                        tools=["read_file"], intro="You are a security auditor.",
                    ),
                    "nointro": SubAgentTypeSpec(tools=["read_file"]),
                }
            ),
        )
        for agent_type in ("research", "plan", "general", "security", "nointro"):
            tool = _make_bound_tool(config=cfg)
            tool._build_subagent_prompt_builder(agent_type)
            forked = tool._agent._prompt_builder.fork.return_value
            section = forked.register.call_args[0][0]
            assert "background mode (background=true)" in section.content, (
                f"bg constraint missing for type {agent_type!r}"
            )
            assert "GUARANTEED LOST" in section.content


# ── Result formatting ──


class TestResultFormat:
    def test_format_with_tools(self) -> None:
        tool = SubAgentTool()
        result = tool._format_result(
            output="Found 3 endpoints.",
            steps=5,
            tool_usage={"grep_files": 2, "read_file": 3},
            duration_ms=4200.0,
        )
        assert "Found 3 endpoints." in result
        assert "Steps: 5" in result
        assert "grep_files x2" in result
        assert "read_file x3" in result
        assert "4.2s" in result

    def test_format_single_tool_shows_x1(self) -> None:
        tool = SubAgentTool()
        result = tool._format_result(
            output="Done.",
            steps=1,
            tool_usage={"read_file": 1},
            duration_ms=500.0,
        )
        assert "read_file x1" in result

    def test_format_no_tools(self) -> None:
        tool = SubAgentTool()
        result = tool._format_result(
            output="Analysis complete.", steps=1, tool_usage={}, duration_ms=500.0,
        )
        assert "Steps: 1" in result
        assert "Tools:" not in result


# ── bind_agent ──


class TestBindAgent:
    def test_bind_stores_reference(self) -> None:
        tool = SubAgentTool()
        agent = MagicMock()
        tool.bind_agent(agent)
        assert tool._agent is agent

    def test_isinstance_agent_aware(self) -> None:
        from agent_harness.tool.base import AgentAware

        tool = SubAgentTool()
        assert isinstance(tool, AgentAware)


class TestBindSession:
    def test_isinstance_session_aware(self) -> None:
        from agent_harness.tool.base import SessionAware

        tool = SubAgentTool()
        assert isinstance(tool, SessionAware)

    def test_bind_stores_session_id(self) -> None:
        tool = SubAgentTool()
        assert tool._session_id is None
        tool.bind_session("sess_X")
        assert tool._session_id == "sess_X"
        tool.bind_session(None)
        assert tool._session_id is None

    async def test_execute_with_session_passes_inmemory_session_to_child(
        self,
    ) -> None:
        """parent_sid → child.run(session=InMemorySession(parent_sid)).
        InMemorySession scopes child's output paths but is ephemeral
        (load_state returns None, save_state writes only to its own dict)
        so parent's session file is never touched."""
        from agent_harness.session.memory_session import InMemorySession

        tool = _make_bound_tool()
        tool.bind_session("sess_X")
        tool._agent.hooks = DefaultHooks()
        mock_result = AgentResult(
            output="done", steps=[StepResult(response="done")],
        )

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            await tool.execute(
                description="x", prompt="y", agent_type="research", model="main",
            )

        passed = mock_instance.run.call_args.kwargs.get("session")
        assert isinstance(passed, InMemorySession)
        assert passed.session_id == "sess_X"

    async def test_execute_without_session_passes_none_to_child(
        self,
    ) -> None:
        tool = _make_bound_tool()
        tool._agent.hooks = DefaultHooks()
        mock_result = AgentResult(
            output="done", steps=[StepResult(response="done")],
        )

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            await tool.execute(
                description="x", prompt="y", agent_type="research", model="main",
            )

        assert mock_instance.run.call_args.kwargs.get("session") is None


# ── Type sequencing ──


class TestTypeSequencing:
    def test_type_seq_increments_per_type(self) -> None:
        tool = SubAgentTool()
        assert tool._type_seq == {}
        tool._type_seq["research"] = tool._type_seq.get("research", 0) + 1
        tool._type_seq["plan"] = tool._type_seq.get("plan", 0) + 1
        tool._type_seq["research"] = tool._type_seq.get("research", 0) + 1
        assert tool._type_seq == {"research": 2, "plan": 1}


# ── Executor timeout ──


class TestExecutorTimeout:
    def test_executor_timeout_large(self) -> None:
        tool = SubAgentTool()
        assert tool.executor_timeout == 600.0


# ── Execution (mock LLM) ──


class TestExecution:
    @pytest.mark.parametrize(
        ("selection", "expected_attr"),
        [("main", "llm"), ("sub", "sub_llm")],
    )
    async def test_model_selection_routes_child_llm(
        self, selection: str, expected_attr: str,
    ) -> None:
        tool = _make_bound_tool()
        tool._agent.hooks = DefaultHooks()
        mock_result = AgentResult(output="done", steps=[])

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            await tool.execute(
                description="test",
                prompt="do something",
                agent_type="research",
                model=selection,
            )

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["llm"] is getattr(tool._agent, expected_attr)
        assert kwargs["sub_llm"] is tool._agent.sub_llm

    async def test_sub_selection_aliases_main_when_no_separate_sub(self) -> None:
        tool = _make_bound_tool()
        tool._agent.sub_llm = tool._agent.llm
        tool._agent.hooks = DefaultHooks()
        mock_result = AgentResult(output="done", steps=[])

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            await tool.execute(
                description="test",
                prompt="do something",
                agent_type="research",
                model="sub",
            )

        assert MockAgent.call_args.kwargs["llm"] is tool._agent.llm

    async def test_execute_returns_result_with_summary(self) -> None:
        tool = _make_bound_tool()
        mock_result = AgentResult(
            output="Found auth handler in src/auth.py",
            steps=[
                StepResult(thought="searching", action=[
                    ToolCall(id="1", name="grep_files", arguments={}),
                ]),
                StepResult(thought="reading", action=[
                    ToolCall(id="2", name="read_file", arguments={}),
                ]),
                StepResult(response="Found auth handler in src/auth.py"),
            ],
        )
        tool._agent.hooks = DefaultHooks()

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            result = await tool.execute(
                description="Find auth handler",
                prompt="Search for authentication handler",
                agent_type="research",
                model="main",
            )

        assert "Found auth handler" in result
        assert "[Execution:" in result
        assert "Steps: 3" in result
        assert "grep_files x1" in result
        assert "read_file x1" in result

    async def test_execute_error_wraps_in_tool_execution_error(self) -> None:
        tool = _make_bound_tool()
        tool._agent.hooks = DefaultHooks()

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(side_effect=RuntimeError("LLM failed"))
            MockAgent.return_value = mock_instance

            with pytest.raises(ToolExecutionError, match="failed"):
                await tool.execute(
                    description="test", prompt="do something", agent_type="research",
                    model="main",
                )

    async def test_hooks_called_on_success(self) -> None:
        tool = _make_bound_tool()
        hooks = AsyncMock(spec=DefaultHooks)
        tool._agent.hooks = hooks

        mock_result = AgentResult(output="done", steps=[])

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            await tool.execute(
                description="test task", prompt="do something", agent_type="research",
                model="main",
            )

        hooks.on_subagent_start.assert_called_once()
        start_args = hooks.on_subagent_start.call_args
        assert start_args[0][2] == "research"
        assert start_args[0][3] == "test task"

        hooks.on_subagent_end.assert_called_once()
        end_args = hooks.on_subagent_end.call_args
        assert end_args[0][2] == "research"
        assert end_args[0][3] == "test task"

    async def test_hooks_called_on_error(self) -> None:
        tool = _make_bound_tool()
        hooks = AsyncMock(spec=DefaultHooks)
        tool._agent.hooks = hooks

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
            MockAgent.return_value = mock_instance

            with pytest.raises(ToolExecutionError):
                await tool.execute(
                    description="test", prompt="do something", agent_type="research",
                    model="main",
                )

        hooks.on_subagent_start.assert_called_once()
        hooks.on_subagent_end.assert_called_once()
        end_args = hooks.on_subagent_end.call_args
        assert end_args[0][7] is not None          # error passed as 8th positional
        assert "boom" in end_args[0][7]

    async def test_subagent_name_includes_type_and_seq(self) -> None:
        tool = _make_bound_tool()
        hooks = AsyncMock(spec=DefaultHooks)
        tool._agent.hooks = hooks

        mock_result = AgentResult(output="done", steps=[])

        with patch("agent_harness.agent.react.ReActAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_instance

            await tool.execute(
                description="first", prompt="task 1", agent_type="research", model="main",
            )
            await tool.execute(
                description="second", prompt="task 2", agent_type="research", model="main",
            )

        calls = hooks.on_subagent_start.call_args_list
        assert "research.1" in calls[0][0][1]
        assert "research.2" in calls[1][0][1]
