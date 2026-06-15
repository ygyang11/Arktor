"""Tests for agent_harness.agent.planner — PlanAndExecuteAgent with MockLLM."""
from __future__ import annotations

import json

import pytest

from agent_harness.agent.planner import Plan, PlanAndExecuteAgent, PlanStep
from agent_harness.context.context import AgentContext
from agent_harness.core.message import Message
from agent_harness.llm.types import FinishReason, LLMResponse, Usage
from agent_harness.session.base import SessionState
from agent_harness.session.memory_session import InMemorySession
from agent_harness.utils.token_counter import count_tokens

from tests.conftest import MockLLM, MockTool


def _make_json_response(data: dict[str, object]) -> LLMResponse:
    """Build an LLMResponse containing a JSON string."""
    return LLMResponse(
        message=Message.assistant(json.dumps(data)),
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        finish_reason=FinishReason.STOP,
    )

def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message.assistant(text),
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason=FinishReason.STOP,
    )


class TestPlanProgressSummary:
    def test_result_summary_uses_token_truncation(self) -> None:
        long_result = "analysis " * 400
        plan = Plan(
            goal="Goal",
            steps=[
                PlanStep(
                    id="1",
                    description="Run analysis",
                    status="done",
                    result=long_result,
                )
            ],
        )

        summary = plan.progress_summary
        line = summary.splitlines()[1]
        result_text = line.split(" -> ", 1)[1]
        assert result_text.endswith("...")
        assert count_tokens(result_text) <= 26


class TestPlanAndExecuteAgent:
    @pytest.mark.asyncio
    async def test_simple_plan_execution(self) -> None:
        """Test a simple 1-step plan that succeeds."""
        llm = MockLLM()

        # 1. Planner generates plan
        llm.responses.append(_make_json_response({
            "goal": "Test Goal",
            "steps": [{"id": "1", "description": "Step 1"}]
        }))

        # 2. Executor executes Step 1
        llm.responses.append(_make_text_response("Step 1 execution result"))

        # 3. Replanner evaluates Step 1 -> Goal Achieved
        llm.responses.append(_make_json_response({
            "goal_achieved": True,
            "final_answer": "Task Complete"
        }))

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=5)
        result = await agent.run("Do it")

        assert result.output == "Task Complete"

    @pytest.mark.asyncio
    async def test_plan_run_records_root_input_and_final_output(self) -> None:
        llm = MockLLM()
        llm.responses.append(
            _make_json_response(
                {
                    "goal": "Test Goal",
                    "steps": [{"id": "1", "description": "Step 1"}],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 1 execution result"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": True,
                    "final_answer": "Task Complete",
                }
            )
        )

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=5)
        result = await agent.run("Do it")

        messages = await agent.context.short_term_memory.get_context_messages()
        assert any(
            msg.role.value == "user" and msg.content == "Do it"
            for msg in messages
        )
        assert any(
            msg.role.value == "assistant" and msg.content == result.output
            for msg in messages
        )

    @pytest.mark.asyncio
    async def test_plan_run_restores_session_before_forking_children(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = MockLLM()
        llm.responses.append(
            _make_json_response(
                {
                    "goal": "Test Goal",
                    "steps": [{"id": "1", "description": "Step 1"}],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 1 execution result"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": True,
                    "final_answer": "Task Complete",
                }
            )
        )

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=5)
        session = InMemorySession("s1")
        await session.save_state(
            SessionState(
                session_id="s1",
                messages=[
                    Message.system(
                        "summary",
                        metadata={
                            "is_compression_summary": True,
                            "compression_round": 2,
                            "archive_paths": ["/tmp/r1.md", "/tmp/r2.md"],
                        },
                    )
                ],
            )
        )

        observed_rounds: list[int | None] = []
        observed_session_ids: list[str | None] = []
        original_fork = agent.context.fork

        def recording_fork(name: str | None = None) -> AgentContext:
            compressor = agent.context.short_term_memory.compressor
            observed_rounds.append(
                compressor._compression_count if compressor is not None else None
            )
            observed_session_ids.append(
                compressor._session_id if compressor is not None else None
            )
            return original_fork(name)

        monkeypatch.setattr(agent.context, "fork", recording_fork)

        await agent.run("Do it", session=session)

        assert observed_rounds == [2, 2, 2]
        assert observed_session_ids == ["s1", "s1", "s1"]

    @pytest.mark.asyncio
    async def test_plan_run_saves_non_empty_root_messages(self) -> None:
        llm = MockLLM()
        llm.responses.append(
            _make_json_response(
                {
                    "goal": "Test Goal",
                    "steps": [{"id": "1", "description": "Step 1"}],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 1 execution result"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": True,
                    "final_answer": "Task Complete",
                }
            )
        )

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=5)
        session = InMemorySession("s1")

        await agent.run("Do it", session=session)

        state = await session.load_state()
        assert state is not None
        assert len(state.messages) >= 2

    @pytest.mark.asyncio
    async def test_multi_step_execution(self) -> None:
        """Test a 2-step plan."""
        llm = MockLLM()

        # 1. Planner: 2 steps
        llm.responses.append(_make_json_response({
            "goal": "Test Goal",
            "steps": [
                {"id": "1", "description": "Step 1"},
                {"id": "2", "description": "Step 2"}
            ]
        }))

        # 2. Executor Step 1
        llm.responses.append(_make_text_response("Step 1 done"))

        # 3. Replanner Step 1 -> Continue (not replan, not done)
        llm.responses.append(_make_json_response({
            "goal_achieved": False,
            "should_replan": False
        }))

        # 4. Executor Step 2
        llm.responses.append(_make_text_response("Step 2 done"))

        # 5. Replanner Step 2 -> Done
        llm.responses.append(_make_json_response({
            "goal_achieved": True,
            "final_answer": "All Done"
        }))

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=10)
        result = await agent.run("Do it")

        assert result.output == "All Done"

    @pytest.mark.asyncio
    async def test_replanning(self) -> None:
        """Test replanning flow."""
        llm = MockLLM()

        # 1. Planner: 1 step initially
        llm.responses.append(_make_json_response({
            "goal": "Test Goal",
            "steps": [{"id": "1", "description": "Step 1"}]
        }))

        # 2. Executor Step 1
        llm.responses.append(_make_text_response("Step 1 result (partial)"))

        # 3. Replanner -> Replan! Add Step 2
        llm.responses.append(_make_json_response({
            "goal_achieved": False,
            "should_replan": True,
            "reason": "Need more steps",
            "updated_steps": [{"id": "2", "description": "Step 2"}]
        }))

        # 4. Executor Step 2 (assuming Step 1 is done and kept)
        llm.responses.append(_make_text_response("Step 2 result"))

        # 5. Replanner Step 2 -> Done
        llm.responses.append(_make_json_response({
            "goal_achieved": True,
            "final_answer": "Finally Done"
        }))

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=10)
        result = await agent.run("Do it")

        assert result.output == "Finally Done"

    @pytest.mark.asyncio
    async def test_replanner_uses_fixed_instruction_input(self) -> None:
        llm = MockLLM()
        llm.responses.append(
            _make_json_response(
                {
                    "goal": "Test Goal",
                    "steps": [{"id": "1", "description": "Step 1"}],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 1 done"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": True,
                    "final_answer": "Done",
                }
            )
        )

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=5)
        await agent.run("Do it")

        replanner_call = next(
            call
            for call in llm.call_history
            if any(
                msg.role.value == "system"
                and (msg.content or "").startswith("You are a replanning agent")
                for msg in call
            )
        )
        user_inputs = [
            msg.content or ""
            for msg in replanner_call
            if msg.role.value == "user"
        ]
        assert user_inputs == [
            "Please provide your replanning decision in valid JSON."
        ]

    @pytest.mark.asyncio
    async def test_replanning_injects_brief_notice_and_working_fields(
        self,
    ) -> None:
        llm = MockLLM()

        llm.responses.append(
            _make_json_response(
                {
                    "goal": "Test Goal",
                    "steps": [{"id": "1", "description": "Step 1"}],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 1 result (partial)"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": False,
                    "should_replan": True,
                    "reason": "Need more steps",
                    "updated_steps": [{"id": "2", "description": "Step 2"}],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 2 result"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": True,
                    "final_answer": "Finally Done",
                }
            )
        )

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=10)
        await agent.run("Do it")

        second_executor_call = next(
            call
            for call in llm.call_history
            if any(
                msg.role.value == "user" and msg.content == "Step 2"
                for msg in call
            )
        )
        system_texts = [
            msg.content or ""
            for msg in second_executor_call
            if msg.role.value == "system"
        ]

        assert any("Plan has been updated." in text for text in system_texts)
        assert any("replan_reason" in text for text in system_texts)
        assert any("plan_change" in text for text in system_texts)
        assert any(
            "done or failed steps are retained from the previous plan."
            in text
            for text in system_texts
        )

    @pytest.mark.asyncio
    async def test_no_replan_does_not_inject_brief_notice(self) -> None:
        llm = MockLLM()

        llm.responses.append(
            _make_json_response(
                {
                    "goal": "Test Goal",
                    "steps": [
                        {"id": "1", "description": "Step 1"},
                        {"id": "2", "description": "Step 2"},
                    ],
                }
            )
        )
        llm.responses.append(_make_text_response("Step 1 done"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": False,
                    "should_replan": False,
                }
            )
        )
        llm.responses.append(_make_text_response("Step 2 done"))
        llm.responses.append(
            _make_json_response(
                {
                    "goal_achieved": True,
                    "final_answer": "All Done",
                }
            )
        )

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=10)
        await agent.run("Do it")

        second_executor_call = next(
            call
            for call in llm.call_history
            if any(
                msg.role.value == "user" and msg.content == "Step 2"
                for msg in call
            )
        )
        system_texts = [
            msg.content or ""
            for msg in second_executor_call
            if msg.role.value == "system"
        ]

        assert not any("Plan has been updated." in text for text in system_texts)

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self) -> None:
        """Test loop termination."""
        llm = MockLLM()

        # 1. Planner: 3 steps
        llm.responses.append(_make_json_response({
            "goal": "Test Goal",
            "steps": [
                {"id": "1", "description": "Step 1"},
                {"id": "2", "description": "Step 2"},
                {"id": "3", "description": "Step 3"}
            ]
        }))

        # Iteration 1: Step 1 executes
        llm.responses.append(_make_text_response("Step 1 done"))
        llm.responses.append(_make_json_response({"goal_achieved": False, "should_replan": False}))

        # Iteration 2: Step 2 executes
        llm.responses.append(_make_text_response("Step 2 done"))
        llm.responses.append(_make_json_response({"goal_achieved": False, "should_replan": False}))

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, max_steps=2) # Only allow 2 iterations

        result = await agent.run("Do it")
        assert "Max iterations reached" in result.output

    @pytest.mark.asyncio
    async def test_executor_tool_usage(self) -> None:
        """Test Executor using a tool."""
        llm = MockLLM()
        mock_tool = MockTool(response="found it")

        # 1. Planner
        llm.responses.append(_make_json_response({
            "goal": "Test",
            "steps": [{"id": "1", "description": "Search"}]
        }))

        # 2. Executor: Calls tool
        # Executor.run() -> Executor.step() -> call_llm
        llm.add_tool_call_response("mock_tool", {"query": "foo"})
        # Executor.step() returns StepResult with observation.
        # Executor.run() loop continues.
        # Executor.step() called again -> call_llm
        llm.responses.append(_make_text_response("I found 'found it'"))

        # 3. Replanner
        llm.responses.append(_make_json_response({
            "goal_achieved": True,
            "final_answer": "Found"
        }))

        agent = PlanAndExecuteAgent(name="test_agent", llm=llm, tools=[mock_tool], max_steps=5)
        result = await agent.run("Search foo")

        assert "Found" in result.output


class TestPlanDetailedProgress:
    def test_detailed_progress_includes_full_results(self) -> None:
        """detailed_progress preserves complete step results while
        progress_summary truncates them."""
        long_result = "A" * 2000
        plan = Plan(
            goal="Test goal",
            steps=[
                PlanStep(
                    id="1",
                    description="Step 1",
                    status="done",
                    result=long_result,
                ),
            ],
        )

        detailed = plan.detailed_progress
        summary = plan.progress_summary

        assert long_result in detailed
        assert long_result not in summary
        assert "..." in summary


class TestReplannerContext:
    @pytest.mark.asyncio
    async def test_replanner_with_full_context_achieves_goal(self) -> None:
        """When all plan steps are done the replanner should set goal_achieved
        and return the final answer instead of falling back to progress_summary."""
        llm = MockLLM()
        llm.responses.append(_make_json_response({
            "goal": "Write report",
            "steps": [
                {"id": "1", "description": "Research"},
                {"id": "2", "description": "Write"},
            ],
        }))
        llm.responses.append(_make_text_response("Research findings here"))
        llm.responses.append(_make_json_response({
            "goal_achieved": False,
            "should_replan": False,
        }))
        llm.responses.append(_make_text_response("Final report content here"))
        llm.responses.append(_make_json_response({
            "goal_achieved": True,
            "final_answer": "Final report content here",
        }))

        agent = PlanAndExecuteAgent(name="test", llm=llm, max_steps=10)
        result = await agent.run("Write a report")

        assert result.output == "Final report content here"
        assert "Goal:" not in result.output

    @pytest.mark.asyncio
    async def test_replanner_receives_remaining_steps(self) -> None:
        """The replanner working memory should contain a remaining_steps
        field indicating how many steps are left or that all are completed."""
        llm = MockLLM()
        llm.responses.append(_make_json_response({
            "goal": "G",
            "steps": [{"id": "1", "description": "Only step"}],
        }))
        llm.responses.append(_make_text_response("Done"))
        llm.responses.append(_make_json_response({
            "goal_achieved": True,
            "final_answer": "Done",
        }))

        agent = PlanAndExecuteAgent(name="test", llm=llm, max_steps=5)
        result = await agent.run("Simple task")
        assert result.output == "Done"

        replanner_call = next(
            call
            for call in llm.call_history
            if any(
                msg.role.value == "system"
                and (msg.content or "").startswith("You are a replanning agent")
                for msg in call
            )
        )
        system_texts = [
            msg.content or ""
            for msg in replanner_call
            if msg.role.value == "system"
        ]
        assert any("remaining_steps" in text for text in system_texts)
