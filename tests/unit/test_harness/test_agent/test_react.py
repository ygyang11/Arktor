"""Tests for agent_harness.agent.react — ReActAgent with MockLLM."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_app.tools.skill import skill_tool
from agent_harness.agent.react import ReActAgent
from agent_harness.core.config import HarnessConfig, SkillConfig
from agent_harness.core.errors import MaxStepsExceededError
from agent_harness.core.message import Message, Role, ToolCall
from agent_harness.llm.types import FinishReason, LLMResponse, Usage
from agent_harness.prompt.sections import DEFAULT_INTRO
from agent_harness.session.base import SessionState
from agent_harness.session.memory_session import InMemorySession
from tests.conftest import MockLLM, MockTool


class TestReActSimpleResponse:
    @pytest.mark.asyncio
    async def test_single_step_text_response(self) -> None:
        """LLM returns text immediately → 1 step, no tool calls."""
        llm = MockLLM()
        llm.add_text_response("The answer is 42.")

        agent = ReActAgent(name="test", llm=llm, max_steps=5)
        result = await agent.run("What is the answer?")

        assert result.output == "The answer is 42."
        assert result.step_count == 1
        assert result.steps[0].response == "The answer is 42."
        assert result.steps[0].action is None

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_user_messages(self) -> None:
        llm = MockLLM()
        llm.add_text_response("Done")

        agent = ReActAgent(name="test", llm=llm, max_steps=5)
        await agent.run("Hello")

        # call_history[0] should include system prompt + user message
        messages = llm.call_history[0]
        roles = [m.role.value for m in messages]
        assert "system" in roles
        assert "user" in roles


class TestReActToolUsage:
    @pytest.mark.asyncio
    async def test_tool_call_then_response(self) -> None:
        """LLM calls a tool, then provides final answer → 2 steps."""
        llm = MockLLM()
        llm.add_tool_call_response("mock_tool", {"query": "search this"})
        llm.add_text_response("Based on the search: the answer is X.")

        mock_tool = MockTool(response="search result: X")
        agent = ReActAgent(name="test", llm=llm, tools=[mock_tool], max_steps=5)
        result = await agent.run("Find X")

        assert result.step_count == 2
        # Step 1: tool call
        assert result.steps[0].action is not None
        assert len(result.steps[0].action) == 1
        assert result.steps[0].action[0].name == "mock_tool"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation[0].content == "search result: X"
        assert result.steps[0].response is None

        # Step 2: final answer
        assert result.steps[1].response is not None
        assert "answer is X" in result.steps[1].response
        assert result.output == "Based on the search: the answer is X."

        # Tool was actually called (check cloned instance in registry)
        registered = agent.tool_registry.get("mock_tool")
        assert len(registered.call_history) == 1
        assert registered.call_history[0] == {"query": "search this"}


class _ReasoningMockLLM(MockLLM):
    """MockLLM that decodes reasoning from a 'mock' sidecar namespace."""

    def reasoning_text(self, message: Message) -> str | None:
        return message.provider_metadata.get("mock", {}).get("reasoning")

    def add_response(
        self,
        content: str | None,
        reasoning: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        pm = {"mock": {"reasoning": reasoning}} if reasoning else {}
        self.responses.append(
            LLMResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls,
                    provider_metadata=pm,
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason=(
                    FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP
                ),
            )
        )


class TestReActStepSemantics:
    @pytest.mark.asyncio
    async def test_thought_is_reasoning_and_response_is_visible_text(self) -> None:
        """thought carries provider reasoning; response carries the visible
        assistant text on tool steps too — without ending the run."""
        llm = _ReasoningMockLLM()
        llm.add_response(
            "Let me check.",
            reasoning="user wants X, search first",
            tool_calls=[ToolCall(name="mock_tool", arguments={"query": "x"})],
        )
        llm.add_response("The answer is X.", reasoning="results confirm X")

        agent = ReActAgent(
            name="test", llm=llm, tools=[MockTool(response="found")], max_steps=5,
        )
        result = await agent.run("Find X")

        # Tool step: visible text in response must NOT terminate the loop.
        assert result.step_count == 2
        assert result.steps[0].thought == "user wants X, search first"
        assert result.steps[0].response == "Let me check."
        assert result.steps[0].action is not None

        # Final step: reasoning captured alongside the answer.
        assert result.steps[1].thought == "results confirm X"
        assert result.steps[1].response == "The answer is X."
        assert result.output == "The answer is X."

    @pytest.mark.asyncio
    async def test_thought_none_when_provider_has_no_sidecar(self) -> None:
        llm = MockLLM()  # base reasoning_text → None
        llm.add_text_response("done")

        agent = ReActAgent(name="test", llm=llm, max_steps=5)
        result = await agent.run("go")

        assert result.steps[0].thought is None
        assert result.steps[0].response == "done"


class TestReActMaxSteps:
    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self) -> None:
        """LLM always calls tools → exceeds max_steps."""
        llm = MockLLM()
        # Queue more tool calls than max_steps
        for _ in range(10):
            llm.add_tool_call_response("mock_tool", {"query": "again"})

        mock_tool = MockTool(response="still searching")
        agent = ReActAgent(name="test", llm=llm, tools=[mock_tool], max_steps=3)

        with pytest.raises(MaxStepsExceededError):
            await agent.run("Never-ending search")

    @pytest.mark.asyncio
    async def test_finishes_at_exactly_max_steps(self) -> None:
        """LLM uses tools for (max_steps - 1) times, then responds at the limit."""
        llm = MockLLM()
        llm.add_tool_call_response("mock_tool", {"query": "q1"})
        llm.add_tool_call_response("mock_tool", {"query": "q2"})
        llm.add_text_response("Final answer at step 3")

        mock_tool = MockTool(response="result")
        agent = ReActAgent(name="test", llm=llm, tools=[mock_tool], max_steps=3)
        result = await agent.run("Complex question")

        assert result.step_count == 3
        assert result.output == "Final answer at step 3"


@pytest.fixture
def restore_harness_config() -> Iterator[None]:
    original = HarnessConfig._instance
    try:
        yield
    finally:
        HarnessConfig._instance = original


@pytest.fixture(autouse=True)
def reset_skill_tool_cache() -> Iterator[None]:
    skill_tool._loader = None
    skill_tool._loader_dirs_key = None
    skill_tool._loader_state_key = None
    skill_tool._state_key_checked_at = 0.0
    try:
        yield
    finally:
        skill_tool._loader = None
        skill_tool._loader_dirs_key = None
        skill_tool._loader_state_key = None
        skill_tool._state_key_checked_at = 0.0


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    sd = root / "demo"
    sd.mkdir(parents=True)
    (sd / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill.\n---\n\nDemo body.\n")
    return root


class TestSkillPromptSupplement:
    def test_supplement_appended_when_skill_tool_present(
        self,
        skills_dir: Path,
        restore_harness_config: None,
    ) -> None:
        HarnessConfig._instance = HarnessConfig(skill=SkillConfig(dirs=[str(skills_dir)]))
        agent = ReActAgent(name="test", tools=[skill_tool])
        assert "# Skills" in agent.system_prompt
        assert "<available_skills>" in agent.system_prompt

    def test_no_supplement_without_skill_tool(self) -> None:
        agent = ReActAgent(name="test", tools=[])
        assert "# Skills" not in agent.system_prompt

    def test_custom_prompt_gets_supplement(
        self,
        skills_dir: Path,
        restore_harness_config: None,
    ) -> None:
        HarnessConfig._instance = HarnessConfig(skill=SkillConfig(dirs=[str(skills_dir)]))
        agent = ReActAgent(
            name="test",
            system_prompt="Custom prompt.",
            tools=[skill_tool],
        )
        assert "Custom prompt." in agent.system_prompt
        assert "# Skills" in agent.system_prompt

    def test_custom_prompt_no_supplement_without_skill_tool(self) -> None:
        agent = ReActAgent(name="test", system_prompt="Custom prompt.", tools=[])
        assert "Custom prompt." in agent.system_prompt
        assert "# Skills" not in agent.system_prompt

    async def test_sync_system_prompt_is_idempotent(
        self,
        skills_dir: Path,
        restore_harness_config: None,
    ) -> None:
        HarnessConfig._instance = HarnessConfig(skill=SkillConfig(dirs=[str(skills_dir)]))
        agent = ReActAgent(name="test", tools=[skill_tool])
        msgs = agent.context.short_term_memory._messages

        await agent._sync_system_prompt()
        assert len(msgs) == 1
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == agent.system_prompt

        await agent._sync_system_prompt()
        assert len(msgs) == 1
        assert msgs[0].content == agent.system_prompt

    def test_default_intro_when_no_system_prompt(self) -> None:
        agent = ReActAgent(name="test")
        assert DEFAULT_INTRO.split("\n")[0] in agent.system_prompt


class TestStreamDefault:
    def test_stream_defaults_to_true(self) -> None:
        agent = ReActAgent(name="test", llm=MockLLM())
        assert agent._stream is True

    def test_stream_explicit_false(self) -> None:
        agent = ReActAgent(name="test", llm=MockLLM(), stream=False)
        assert agent._stream is False


class TestSessionIntegration:
    async def test_run_restores_from_session(self) -> None:
        llm = MockLLM()
        llm.add_text_response("I remember")

        session = InMemorySession("s1")
        state = SessionState(
            session_id="s1",
            messages=[
                Message.system("You are helpful."),
                Message.user("hello"),
                Message.assistant("hi there"),
            ],
        )
        await session.save_state(state)

        agent = ReActAgent(name="test", llm=llm)
        result = await agent.run("what did I say?", session=session)

        messages = llm.call_history[0]
        assert any(m.content == "hello" for m in messages)
        assert any(m.content == "what did I say?" for m in messages)

    async def test_run_skips_load_when_context_has_data(self) -> None:
        llm = MockLLM()
        llm.add_text_response("first")
        llm.add_text_response("second")

        session = InMemorySession("s1")
        agent = ReActAgent(name="test", llm=llm)

        await agent.run("hello", session=session)
        await agent.run("follow up", session=session)

        messages = llm.call_history[1]
        user_messages = [m for m in messages if m.role.value == "user"]
        assert len(user_messages) == 2

    async def test_run_saves_session_on_error(self) -> None:
        llm = MockLLM()
        for _ in range(5):
            llm.add_tool_call_response("mock_tool", {"q": "x"})

        tool = MockTool(response="result")
        session = InMemorySession("s1")
        agent = ReActAgent(name="test", llm=llm, tools=[tool], max_steps=2)

        with pytest.raises(MaxStepsExceededError):
            await agent.run("go", session=session)

        saved = await session.load_state()
        assert saved is not None
        assert saved.agent_state == "error"

    async def test_restore_replaces_mismatched_system_prompt(self) -> None:
        llm = MockLLM()
        llm.add_text_response("ok")

        session = InMemorySession("s1")
        state = SessionState(
            session_id="s1",
            messages=[
                Message.system("Old prompt."),
                Message.user("hello"),
                Message.assistant("hi"),
            ],
        )
        await session.save_state(state)

        agent = ReActAgent(name="test", llm=llm)
        agent.system_prompt = "New prompt."
        await agent.run("continue", session=session)

        messages = llm.call_history[0]
        system_msgs = [m for m in messages if m.role.value == "system"]
        assert system_msgs[0].content == "New prompt."

    async def test_restore_working_memory_and_variables(self) -> None:
        session = InMemorySession("s1")
        state = SessionState(
            session_id="s1",
            messages=[Message.system("sys"), Message.user("hi"), Message.assistant("hello")],
            working_memory_scratchpad={"goal": "test research"},
            variables_agent={"topic": "5G"},
            variables_global={"user": "Alice"},
        )
        await session.save_state(state)

        agent = ReActAgent(name="test", llm=MockLLM())
        s = await session.load_state()
        assert s is not None
        await agent.context.restore_from_state(s, agent.system_prompt)

        assert agent.context.working_memory.get("goal") == "test research"
        assert agent.context.variables.get("topic") == "5G"
        assert agent.context.variables.get("user") == "Alice"

    async def test_run_without_session_unchanged(self) -> None:
        llm = MockLLM()
        llm.add_text_response("ok")
        agent = ReActAgent(name="test", llm=llm)
        result = await agent.run("hello")
        assert result.output == "ok"

    async def test_run_with_string_session(self, tmp_path: Path) -> None:
        import agent_harness.session.file_session as fs_mod

        original = fs_mod._DEFAULT_SESSION_DIR
        fs_mod._DEFAULT_SESSION_DIR = tmp_path
        try:
            llm = MockLLM()
            llm.add_text_response("ok")
            agent = ReActAgent(name="test", llm=llm)
            await agent.run("hello", session="test-session")
            assert (tmp_path / "test-session.json").exists()
        finally:
            fs_mod._DEFAULT_SESSION_DIR = original

    async def test_chat_saves_via_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = MockLLM()
        llm.add_text_response("hi")

        inputs = iter(["hello", "exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        session = InMemorySession("s1")
        agent = ReActAgent(name="test", llm=llm)
        await agent.chat(session=session)

        saved = await session.load_state()
        assert saved is not None
        assert any(m.content == "hello" for m in saved.messages)

    async def test_chat_continues_after_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = MockLLM()
        for _ in range(5):
            llm.add_tool_call_response("mock_tool", {"q": "x"})
        llm.add_text_response("recovered")

        tool = MockTool(response="r")
        inputs = iter(["trigger error", "hello", "exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        agent = ReActAgent(name="test", llm=llm, tools=[tool], max_steps=1)
        await agent.chat()
