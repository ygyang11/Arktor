"""Regression tests for BaseAgent bug fixes.

Covers:
- Agent reuse (second run() after FINISHED state)
- Usage accumulation across steps
- PlanAgent state reset on re-run
- use_long_term_memory instance flag
- Approval integration
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_harness.agent.base import AgentResult, BaseAgent, StepResult
from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
)
from agent_harness.context.context import AgentContext
from agent_harness.context.state import AgentState
from agent_harness.core.config import HarnessConfig, LLMConfig, MemoryConfig, TracingConfig
from agent_harness.core.errors import MaxStepsExceededError
from agent_harness.core.message import Message, ToolCall, ToolResult
from agent_harness.hooks import DefaultHooks, TracingHooks
from agent_harness.llm.types import Usage
from agent_harness.session import InMemorySession
from agent_harness.tool.base import BaseTool
from tests.conftest import MockLLM, MockTool


class _FailsOnceAgent(ConversationalAgent):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._should_fail = True

    async def step(self) -> StepResult:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("intentional failure")
        return await super().step()


class _CountingAgent(BaseAgent):
    def __init__(self, turns: list[int], **kwargs: Any) -> None:
        super().__init__(name="counter", llm=MockLLM(), hooks=DefaultHooks(), **kwargs)
        self.turns = turns
        self.count = 0
        self.turn = 0
        self.ready = asyncio.Event()
        self.released = False
        self.spawn_on_finish: Callable[[], None] | None = None

    async def step(self) -> StepResult:
        self.count += 1
        self._run_usage = self._run_usage + Usage(total_tokens=1)
        if self.count == self.turns[self.turn]:
            self.count = 0
            self.turn += 1
            if self.spawn_on_finish is not None:
                self.spawn_on_finish()
            return StepResult(response=f"turn-{self.turn}")
        return StepResult(action=[ToolCall(name="count", arguments={"step": self.count})])


class TestRun:
    @pytest.fixture(autouse=True)
    def _isolate_background_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agent_harness.background.manager._HARNESS_DIR", tmp_path)

    async def test_budget_shared_across_background_turns(self) -> None:
        agent = _CountingAgent([70, 30], max_steps=100)
        steps: list[int] = []

        async def background() -> tuple[str, str]:
            await agent.ready.wait()
            return "evidence", "completed"

        def launch() -> None:
            if agent.turn == 1:
                agent._bg_manager.spawn("tool", "evidence", background())

        async def after_turn(result: AgentResult) -> None:
            steps.append(len(result.steps))
            agent.ready.set()

        agent.spawn_on_finish = launch
        result = await agent.run("task", on_turn_result=after_turn)

        assert steps == [70, 30]
        assert len(result.steps) == 100 and result.usage.total_tokens == 100
        assert result.output == "turn-2"
        assert agent.max_steps == 100
        assert not agent._bg_manager.has_running()

    async def test_outer_budget_error_preserves_turn_state_and_notifies_once(self) -> None:
        agent = _CountingAgent([1], max_steps=1)
        session = InMemorySession("budget")
        agent.hooks.on_error = AsyncMock()

        async def pending() -> tuple[str, str]:
            try:
                agent.ready.set()
                await asyncio.Event().wait()
            finally:
                agent.released = True
            return "unused", "unused"

        async def after_turn(result: AgentResult) -> None:
            await agent.ready.wait()

        agent.spawn_on_finish = lambda: agent._bg_manager.spawn("tool", "pending", pending())
        with pytest.raises(MaxStepsExceededError, match="exhausted the step budget"):
            await agent.run("task", session=session, on_turn_result=after_turn)

        assert agent.released
        assert agent.context.state.current == AgentState.FINISHED
        state = await session.load_state()
        assert state is not None and state.agent_state == "finished"
        assert any("Background Task Cancelled" in (msg.content or "") for msg in state.messages)
        agent.hooks.on_error.assert_not_awaited()

    @pytest.mark.parametrize(
        ("max_steps", "cleanup_error"), [(1, False), (2, False), (2, True)],
    )
    async def test_run_returns_after_cancelled_background_cleanup(
        self, max_steps: int, cleanup_error: bool,
    ) -> None:
        agent = _CountingAgent([1, 1], max_steps=max_steps)
        cleaning = asyncio.Event()
        release = asyncio.Event()

        async def background() -> tuple[str, str]:
            try:
                agent.ready.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                if cleanup_error:
                    raise RuntimeError("cleanup failure")
                return "partial output", "partial result"
            return "unused", "unused"

        task_id = agent._bg_manager.spawn("tool", "pending", background())

        async def after_turn(result: AgentResult) -> None:
            if agent.turn == 1:
                await agent.ready.wait()
                assert agent._bg_manager.cancel(task_id)

        run = asyncio.create_task(agent.run("task", on_turn_result=after_turn))
        try:
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            assert not run.done()
        finally:
            release.set()

        result = await asyncio.wait_for(run, timeout=2)
        assert result.output == "turn-1" and len(result.steps) == 1
        task = agent._bg_manager.get_task(task_id)
        assert task is not None and task.status == "cancelled"
        assert task.asyncio_task is not None and task.asyncio_task.done()
        assert agent._bg_manager.collect_completed() == []

    async def test_run_turn_does_not_drain_background(self) -> None:
        agent = _CountingAgent([1])

        async def pending() -> tuple[str, str]:
            await agent.ready.wait()
            return "done", "done"

        agent.spawn_on_finish = lambda: agent._bg_manager.spawn("tool", "pending", pending())
        try:
            result = await asyncio.wait_for(agent.run_turn("task"), timeout=2)
            assert result.output == "turn-1" and agent._bg_manager.has_running()
            agent.ready.set()
            await agent._bg_manager.wait_next()
        finally:
            await agent._bg_manager.shutdown()

    @pytest.mark.parametrize("entry", ["run", "run_turn"])
    async def test_invalid_budget_does_not_mutate_memory(self, entry: str) -> None:
        agent = _CountingAgent([1])
        before = list(await agent.context.short_term_memory.get_context_messages())

        with pytest.raises(ValueError, match="positive"):
            await getattr(agent, entry)("task", max_steps=0)

        assert before == await agent.context.short_term_memory.get_context_messages()

    async def test_task_budget_resets_and_override_does_not_change_default(self) -> None:
        agent = _CountingAgent([1, 1], max_steps=3)
        assert (await agent.run("first", max_steps=1)).output == "turn-1"
        assert (await agent.run("second", max_steps=1)).output == "turn-2"
        assert agent.max_steps == 3

    async def test_callback_error_preserved_and_cleanup_waited(self) -> None:
        agent = _CountingAgent([1])
        cleaning = asyncio.Event()
        release = asyncio.Event()
        failure = RuntimeError("callback")

        async def pending() -> tuple[str, str]:
            try:
                agent.ready.set()
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                agent.released = True
            return "unused", "unused"

        async def after_turn(result: AgentResult) -> None:
            await agent.ready.wait()
            raise failure

        agent.spawn_on_finish = lambda: agent._bg_manager.spawn("tool", "pending", pending())
        task = asyncio.create_task(agent.run("task", on_turn_result=after_turn))
        try:
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()

        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(task, timeout=2)
        assert caught.value is failure and agent.released

    async def test_restore_failure_does_not_save_current_context(self) -> None:
        agent = _CountingAgent([1])
        session = InMemorySession("broken")
        failure = OSError("read failed")
        session.load_state = AsyncMock(side_effect=failure)
        session.save_state = AsyncMock()

        with pytest.raises(OSError) as caught:
            await agent.run("task", session=session)

        assert caught.value is failure
        session.save_state.assert_not_awaited()

    @pytest.mark.parametrize("fail_turn", [False, True])
    async def test_save_failure_only_replaces_success(self, fail_turn: bool) -> None:
        agent = _CountingAgent([2 if fail_turn else 1], max_steps=1)
        session = InMemorySession("save")
        session.save_state = AsyncMock(side_effect=OSError("save failed"))

        with pytest.raises(MaxStepsExceededError if fail_turn else OSError):
            await agent.run("task", session=session)

    async def test_multiple_background_batches_and_followup_spawn(self) -> None:
        agent = _CountingAgent([1, 1, 1, 1], max_steps=4)
        releases = [asyncio.Event() for _ in range(3)]
        task_ids: list[str] = []
        observed: list[str] = []

        async def work(index: int) -> tuple[str, str]:
            await releases[index].wait()
            return f"output-{index}", f"evidence-{index}"

        def launch() -> None:
            if agent.turn == 1:
                task_ids.extend(
                    agent._bg_manager.spawn("tool", f"batch-{index}", work(index))
                    for index in (0, 1)
                )
            elif agent.turn == 2:
                task_ids.append(agent._bg_manager.spawn("tool", "batch-2", work(2)))

        async def after_turn(result: AgentResult) -> None:
            observed.append(result.output)
            texts = [message.content or "" for message in result.messages]
            if agent.turn > 1:
                assert any(f"evidence-{agent.turn - 2}" in text for text in texts)
            if agent.turn <= 3:
                releases[agent.turn - 1].set()

        agent.spawn_on_finish = launch
        result = await asyncio.wait_for(agent.run("task", on_turn_result=after_turn), timeout=2)

        assert observed == ["turn-1", "turn-2", "turn-3", "turn-4"]
        assert result.output == "turn-4" and len(result.steps) == 4
        assert result.usage.total_tokens == 4
        messages = [message.content or "" for message in result.messages]
        for index, task_id in enumerate(task_ids):
            assert sum(f"[Background Task Completed] {task_id}" in text for text in messages) == 1
            task = agent._bg_manager.get_task(task_id)
            assert task.status == "completed" and task.asyncio_task.done()
            assert Path(task.result.output_path).read_text() == f"output-{index}"
        assert not agent._bg_manager.has_running()

    async def test_failed_background_result_is_delivered_once(self) -> None:
        agent = _CountingAgent([1, 1], max_steps=2)

        async def fail() -> tuple[str, str]:
            await agent.ready.wait()
            raise RuntimeError("background evidence unavailable")

        def launch() -> None:
            if agent.turn == 1:
                agent._bg_manager.spawn("tool", "failure", fail())

        async def after_turn(result: AgentResult) -> None:
            agent.ready.set()

        agent.spawn_on_finish = launch
        result = await agent.run("task", on_turn_result=after_turn)

        assert result.output == "turn-2"
        failures = [
            message for message in result.messages
            if "[Background Task Failed]" in (message.content or "")
        ]
        assert len(failures) == 1
        assert "background evidence unavailable" in failures[0].content
        assert failures[0].metadata["is_background_result"]
        assert agent._bg_manager.collect_completed() == []

    async def test_outer_budget_failure_preserves_parent_pipeline_trace(self) -> None:
        from agent_harness.orchestration.pipeline import Pipeline, PipelineStep

        hooks = TracingHooks(exporter="none")
        hooks.on_error = AsyncMock(wraps=hooks.on_error)
        child = _CountingAgent([1], max_steps=1)
        child.hooks = hooks

        async def pending() -> tuple[str, str]:
            await asyncio.Event().wait()
            return "unused", "unused"

        child.spawn_on_finish = lambda: child._bg_manager.spawn("tool", "pending", pending())
        pipeline = Pipeline([PipelineStep(agent=child)], hooks=hooks)
        try:
            with pytest.raises(MaxStepsExceededError):
                await pipeline.run("task")
            assert hooks._depth == 1
            assert len(hooks._pipeline_span_stack) == 1
            parent_span = hooks._pipeline_span_stack[0]
            trace_id = hooks._trace_id
            assert parent_span in hooks._span_stack and parent_span.end_time is None
            hooks.on_error.assert_not_awaited()

            following = _CountingAgent([1])
            following.hooks = hooks
            await following.run("continue within parent")

            assert hooks._trace_id == trace_id and hooks._depth == 1
            assert hooks._pipeline_span_stack == [parent_span]
            agent_spans = [span for span in hooks._all_spans if span.kind == "agent"]
            assert len(agent_spans) == 2
            assert all(span.parent_span_id == parent_span.span_id for span in agent_spans)
        finally:
            await hooks.on_pipeline_end("pipeline")
        assert hooks._depth == 0 and not hooks._span_stack


class TestAgentReuse:
    """Issue #1: Second run() on a finished agent must not crash."""

    @pytest.mark.asyncio
    async def test_second_run_succeeds(self) -> None:
        llm = MockLLM()
        llm.add_text_response("first answer")
        llm.add_text_response("second answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")

        r1 = await agent.run("hello")
        assert r1.output == "first answer"
        assert agent.context.state.current == AgentState.FINISHED

        r2 = await agent.run("hello again")
        assert r2.output == "second answer"

    @pytest.mark.asyncio
    async def test_run_after_real_error_succeeds(self) -> None:
        llm = MockLLM()
        llm.add_text_response("recovered")

        agent = _FailsOnceAgent(name="test", llm=llm, system_prompt="")

        with pytest.raises(RuntimeError, match="intentional failure"):
            await agent.run("first")
        assert agent.context.state.current == AgentState.ERROR

        r2 = await agent.run("second")
        assert r2.output == "recovered"
        assert agent.context.state.current == AgentState.FINISHED

    @pytest.mark.asyncio
    async def test_system_prompt_not_duplicated_on_rerun(self) -> None:
        llm = MockLLM()
        llm.add_text_response("first")
        llm.add_text_response("second")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="SYS")
        await agent.run("hello")
        await agent.run("hello again")

        messages = await agent.context.short_term_memory.get_context_messages()
        system_messages = [
            msg
            for msg in messages
            if msg.role.value == "system" and (msg.content or "") == "SYS"
        ]
        assert len(system_messages) == 1

    @pytest.mark.asyncio
    async def test_system_prompt_injected_when_first_system_differs(self) -> None:
        llm = MockLLM()
        llm.add_text_response("ok")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="SYS_B")
        await agent.context.short_term_memory.add_message(Message.system("SYS_A"))

        await agent.run("hello")
        messages = await agent.context.short_term_memory.get_context_messages()
        assert any(
            msg.role.value == "system" and (msg.content or "") == "SYS_B"
            for msg in messages
        )


class TestUsageAccumulation:
    """Issue #2: AgentResult.usage must reflect actual token consumption."""

    @pytest.mark.asyncio
    async def test_usage_is_nonzero(self) -> None:
        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")
        result = await agent.run("question")

        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_steps(self) -> None:
        """For agents with multiple steps, usage should sum up."""
        from agent_harness.agent.react import ReActAgent
        from tests.conftest import MockTool

        llm = MockLLM()
        tool = MockTool(response="tool result")

        # Step 1: tool call → Step 2: observe + respond
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("final answer")

        agent = ReActAgent(name="test", llm=llm, tools=[tool], system_prompt="")
        result = await agent.run("do something")

        # Two LLM calls → usage should be at least 2 × (10 prompt + 5 completion)
        assert result.usage.prompt_tokens >= 20
        assert result.usage.completion_tokens >= 10

    @pytest.mark.asyncio
    async def test_tool_call_response_records_snapshot_and_meter(self) -> None:
        from agent_harness.agent.react import ReActAgent
        from tests.conftest import MockTool

        llm = MockLLM()
        tool = MockTool(response="tool result")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("final")

        agent = ReActAgent(name="test", llm=llm, tools=[tool], system_prompt="")
        await agent.run("do something")

        snap = agent.context.short_term_memory.last_call
        assert snap is not None
        assert snap.input_tokens > 0
        assert snap.completion_tokens > 0
        assert snap.total_tokens == snap.input_tokens + snap.completion_tokens

        meter = agent.context.usage_meter
        assert meter.call_count == 2
        assert meter.by_source["main"].calls == 2


class TestUseLongTermMemory:
    """use_long_term_memory flag on BaseAgent controls call_llm default."""

    @pytest.mark.asyncio
    async def test_long_term_memory_flag_defaults_false(self) -> None:
        """Default: use_long_term_memory is False, so long-term is not used."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")
        assert agent.use_long_term_memory is False

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            # Bypass the rest of run(); call call_llm directly
            await agent.call_llm()

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is False

    @pytest.mark.asyncio
    async def test_long_term_memory_flag_true_propagates(self) -> None:
        """use_long_term_memory=True makes call_llm() use long-term by default."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(
            name="test", llm=llm, system_prompt="", use_long_term_memory=True,
        )
        assert agent.use_long_term_memory is True

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            await agent.call_llm()

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is True


class TestAgentConfigPropagation:
    @pytest.mark.asyncio
    async def test_agent_uses_explicit_config_when_context_not_passed(self) -> None:
        original_instance = HarnessConfig._instance
        HarnessConfig._instance = HarnessConfig(tracing=TracingConfig(enabled=True))

        llm = MockLLM()
        llm.add_text_response("answer")
        explicit_config = HarnessConfig(tracing=TracingConfig(enabled=False))

        try:
            agent = ConversationalAgent(
                name="test",
                llm=llm,
                system_prompt="",
                config=explicit_config,
            )
            assert agent.context.config is explicit_config
            assert isinstance(agent.hooks, DefaultHooks)
            assert not isinstance(agent.hooks, TracingHooks)
            assert HarnessConfig.get().tracing.enabled is True
        finally:
            HarnessConfig._instance = original_instance

    @pytest.mark.asyncio
    async def test_explicit_context_still_initializes_compressor_under_summarize_strategy(
        self,
    ) -> None:
        cfg = HarnessConfig(memory=MemoryConfig(strategy="summarize"))
        llm = MockLLM()
        llm.add_text_response("answer")
        ctx = AgentContext.create(config=cfg)

        agent = ConversationalAgent(
            name="test",
            llm=llm,
            system_prompt="",
            context=ctx,
        )

        assert agent.context.short_term_memory.compressor is not None

    @pytest.mark.asyncio
    async def test_agent_silently_reuses_active_global_config(self) -> None:
        original_instance = HarnessConfig._instance
        active_config = HarnessConfig(tracing=TracingConfig(enabled=False))
        HarnessConfig._instance = active_config

        first_llm = MockLLM()
        first_llm.add_text_response("first")
        second_llm = MockLLM()
        second_llm.add_text_response("second")

        try:
            first_agent = ConversationalAgent(
                name="first",
                llm=first_llm,
                system_prompt="",
            )
            second_agent = ConversationalAgent(
                name="second",
                llm=second_llm,
                system_prompt="",
            )

            assert first_agent.context.config is active_config
            assert second_agent.context.config is active_config
            assert isinstance(first_agent.hooks, DefaultHooks)
            assert not isinstance(first_agent.hooks, TracingHooks)
        finally:
            HarnessConfig._instance = original_instance

    @pytest.mark.asyncio
    async def test_explicit_false_overrides_instance_flag(self) -> None:
        """Explicit use_long_term=False in call_llm() overrides the instance setting."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(
            name="test", llm=llm, system_prompt="", use_long_term_memory=True,
        )

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            await agent.call_llm(use_long_term=False)

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is False

    @pytest.mark.asyncio
    async def test_explicit_true_overrides_instance_default(self) -> None:
        """Explicit use_long_term=True in call_llm() works even when instance flag is False."""
        from unittest.mock import AsyncMock, patch

        llm = MockLLM()
        llm.add_text_response("answer")

        agent = ConversationalAgent(name="test", llm=llm, system_prompt="")

        with patch.object(
            agent.context, "build_llm_messages", new_callable=AsyncMock
        ) as mock_build:
            mock_build.return_value = []
            await agent.call_llm(use_long_term=True)

            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["include_long_term"] is True

    @pytest.mark.asyncio
    async def test_agent_auto_creates_llm_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created_llm = MockLLM()
        created_llm.add_text_response("auto-created")

        def _fake_create_llm(config: HarnessConfig | LLMConfig | None = None) -> MockLLM:
            assert isinstance(config, HarnessConfig)
            assert config.llm.model == "auto-model"
            return created_llm

        monkeypatch.setattr("agent_harness.agent.base.create_llm", _fake_create_llm)

        config = HarnessConfig(
            llm=LLMConfig(provider="openai", model="auto-model", api_key="fake")
        )
        agent = ConversationalAgent(
            name="auto",
            llm=None,
            system_prompt="",
            config=config,
        )
        result = await agent.run("hello")

        assert agent.llm is created_llm
        assert result.output == "auto-created"

    @pytest.mark.asyncio
    async def test_explicit_llm_skips_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        explicit_llm = MockLLM()
        explicit_llm.add_text_response("explicit")
        called = {"value": False}

        def _fake_create_llm(config: HarnessConfig | LLMConfig | None = None) -> MockLLM:
            called["value"] = True
            return MockLLM()

        monkeypatch.setattr("agent_harness.agent.base.create_llm", _fake_create_llm)

        config = HarnessConfig(
            llm=LLMConfig(provider="openai", model="unused-model", api_key="fake")
        )
        agent = ConversationalAgent(
            name="explicit",
            llm=explicit_llm,
            system_prompt="",
            config=config,
        )
        result = await agent.run("hello")

        assert agent.llm is explicit_llm
        assert result.output == "explicit"
        assert called["value"] is False


class _MockApprovalHandler(ApprovalHandler):
    def __init__(self, decisions: dict[str, ApprovalDecision]) -> None:
        self._decisions = decisions
        self.call_count = 0

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        self.call_count += 1
        decision = self._decisions.get(
            request.tool_call.name, ApprovalDecision.ALLOW_ONCE
        )
        return ApprovalResult(tool_call_id=request.tool_call.id, decision=decision)


class TestApprovalIntegration:
    @pytest.mark.asyncio
    async def test_no_approval_passthrough(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("done")

        agent = ReActAgent(name="test", llm=llm, tools=[tool], system_prompt="")
        result = await agent.run("go")
        assert result.output == "done"
        registered = agent.tool_registry.get("mock_tool")
        assert len(registered.call_history) == 1

    @pytest.mark.asyncio
    async def test_deny_returns_error_result(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({"mock_tool": ApprovalDecision.DENY})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("tool was denied")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(), approval_handler=handler,
        )
        result = await agent.run("go")

        assert result.output == "tool was denied"
        assert len(tool.call_history) == 0
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_allow_session_remembered(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({"mock_tool": ApprovalDecision.ALLOW_SESSION})

        llm.add_tool_call_response("mock_tool", {"query": "a"})
        llm.add_tool_call_response("mock_tool", {"query": "b"})
        llm.add_text_response("done")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(), approval_handler=handler,
        )
        result = await agent.run("go")

        assert result.output == "done"
        registered = agent.tool_registry.get("mock_tool")
        assert len(registered.call_history) == 2
        assert handler.call_count == 1  # Second call was auto-approved

    @pytest.mark.asyncio
    async def test_always_allow_skips_handler(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("done")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(always_allow={"mock_tool"}),
            approval_handler=handler,
        )
        await agent.run("go")

        assert handler.call_count == 0
        registered = agent.tool_registry.get("mock_tool")
        assert len(registered.call_history) == 1

    @pytest.mark.asyncio
    async def test_always_deny_blocks_without_handler(self) -> None:
        from agent_harness.agent.react import ReActAgent

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("denied")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(always_deny={"mock_tool"}),
            approval_handler=handler,
        )
        result = await agent.run("go")

        assert result.output == "denied"
        assert handler.call_count == 0
        assert len(tool.call_history) == 0

    @pytest.mark.asyncio
    async def test_handler_failure_degrades_to_deny(self) -> None:
        from agent_harness.agent.react import ReActAgent

        class _FailingHandler(ApprovalHandler):
            async def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
                raise RuntimeError("stdin closed")

        llm = MockLLM()
        tool = MockTool(response="ok")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("handler failed")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            approval=ApprovalPolicy(), approval_handler=_FailingHandler(),
        )
        result = await agent.run("go")

        assert result.output == "handler failed"
        assert len(tool.call_history) == 0

    @pytest.mark.asyncio
    async def test_config_driven_approval(self) -> None:
        from agent_harness.core.config import ApprovalConfig

        config = HarnessConfig(
            approval=ApprovalConfig(mode="auto", always_allow=["mock_tool"]),
            tracing=TracingConfig(enabled=False),
        )
        llm = MockLLM()
        tool = MockTool(response="ok")
        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("done")

        from agent_harness.agent.react import ReActAgent

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            config=config,
        )
        # mock_tool in always_allow → no handler needed → no prompt
        assert agent._approval is not None
        result = await agent.run("go")
        assert result.output == "done"
        registered = agent.tool_registry.get("mock_tool")
        assert len(registered.call_history) == 1


class TestExecuteToolsTwoPhase:
    """Verify two-phase execute_tools: hooks in completion order, memory in call order."""

    @pytest.mark.asyncio
    async def test_hooks_fire_for_denied_results(self) -> None:
        from agent_harness.agent.react import ReActAgent

        hook_results: list[str] = []

        class _Tracker(DefaultHooks):
            async def on_tool_result(self, agent_name: str, result: Any) -> None:
                hook_results.append(result.tool_call_id)

        llm = MockLLM()
        tool = MockTool(response="ok")
        handler = _MockApprovalHandler({"mock_tool": ApprovalDecision.DENY})

        llm.add_tool_call_response("mock_tool", {"query": "test"})
        llm.add_text_response("denied")

        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="",
            hooks=_Tracker(),
            approval=ApprovalPolicy(), approval_handler=handler,
        )
        await agent.run("go")

        # denied result must trigger on_tool_result
        assert len(hook_results) >= 1

    @pytest.mark.asyncio
    async def test_memory_order_matches_call_order(self) -> None:
        from agent_harness.agent.react import ReActAgent
        from agent_harness.core.message import Role

        llm = MockLLM()
        tool = MockTool(response="ok")

        # Two sequential tool-call steps
        llm.add_tool_call_response("mock_tool", {"query": "first"})
        llm.add_tool_call_response("mock_tool", {"query": "second"})
        llm.add_text_response("done")

        agent = ReActAgent(name="test", llm=llm, tools=[tool], system_prompt="")
        await agent.run("go")

        tool_msgs = [m for m in agent.context.short_term_memory._messages
                     if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].tool_result is not None
        assert tool_msgs[1].tool_result is not None

    @pytest.mark.parametrize("hook_error", [False, True])
    async def test_run_turn_waits_for_tool_cleanup(self, hook_error: bool) -> None:
        from agent_harness.agent.react import ReActAgent
        from agent_harness.core.config import ToolConfig
        from agent_harness.llm.types import LLMResponse

        started = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()
        released = asyncio.Event()

        class _GatedTool(BaseTool):
            def __init__(self) -> None:
                super().__init__(name="gated", description="Test resource lifetime")

            async def execute(self, **kwargs: Any) -> str:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cleaning.set()
                    await release.wait()
                    released.set()
                    if hook_error:
                        raise RuntimeError("cleanup failure")
                    raise
                raise AssertionError("tool should be cancelled")

        class _QuickTool(BaseTool):
            def __init__(self) -> None:
                super().__init__(name="quick", description="Wait until sibling starts")

            async def execute(self, **kwargs: Any) -> str:
                await started.wait()
                return "quick result"

        class _HookFailure(Exception):
            pass

        class _FailingHook(DefaultHooks):
            async def on_tool_result(self, agent_name: str, result: ToolResult) -> None:
                raise _HookFailure("original hook")

        tools = [_GatedTool(), _QuickTool()] if hook_error else [_GatedTool()]
        calls = [ToolCall(name=tool.name, arguments={}) for tool in tools]
        llm = MockLLM(responses=[LLMResponse(message=Message.assistant(None, tool_calls=calls))])
        agent = ReActAgent(
            name="test", llm=llm, tools=tools,
            hooks=_FailingHook() if hook_error else DefaultHooks(),
            config=HarnessConfig(tool=ToolConfig(default_timeout=20)),
        )
        task = asyncio.create_task(agent.run_turn("test"))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            if not hook_error:
                task.cancel()
            await asyncio.wait_for(cleaning.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()

        with pytest.raises(_HookFailure if hook_error else asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert released.is_set()


class _SelfHealTracker(DefaultHooks):
    def __init__(self) -> None:
        self.heals: list[str] = []
        self.errors: list[Exception] = []

    async def on_self_heal(self, agent_name: str, summary: str) -> None:
        self.heals.append(summary)

    async def on_error(self, agent_name: str, error: Exception) -> None:
        self.errors.append(error)


class _RejectOnNthCallLLM(MockLLM):
    """Raises LLMUnsupportedContentError on the N-th generate() invocation;
    other calls fall through to the parent's pre-queued responses."""

    def __init__(self, reject_at: int) -> None:
        super().__init__()
        self._reject_at = reject_at
        self._calls = 0

    async def generate(self, messages: list[Message], **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls == self._reject_at:
            from agent_harness.core.errors import LLMUnsupportedContentError
            raise LLMUnsupportedContentError("invalid part type: file")
        return await super().generate(messages, **kwargs)


class _MediaReturningTool(MockTool):
    """Mock tool that returns a ToolOutput carrying a PDF attachment."""

    def __init__(self) -> None:
        super().__init__(response="downloaded")

    async def execute(self, **kwargs: Any) -> Any:
        from agent_harness.core.message import Attachment, ToolOutput
        att = Attachment(
            digest="a" * 64, mime="application/pdf", size=1, filename="x.pdf",
        )
        return ToolOutput(content="Downloaded PDF (108KB)", attachments=[att])


class TestMediaRejectionSelfHeal:
    """Agent retries step once after stripping tool-side media attachments."""

    @pytest.mark.asyncio
    async def test_tool_side_strip_and_retry(self) -> None:
        from agent_harness.agent.react import ReActAgent
        from agent_harness.core.message import Role

        # Step 1: tool_call (call 1) → tool runs, adds TOOL with attachments
        # Step 2: LLM call rejects (call 2) → strip + retry
        # Step 2 retry: text response (call 3)
        llm = _RejectOnNthCallLLM(reject_at=2)
        llm.add_tool_call_response("mock_tool", {"query": "fetch"})
        llm.add_text_response("answered after recovery")

        tracker = _SelfHealTracker()
        tool = _MediaReturningTool()
        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="", hooks=tracker,
        )

        result = await agent.run("fetch and describe")

        assert result.output == "answered after recovery"
        assert len(tracker.heals) == 1
        assert "1 unsupported media attachment" in tracker.heals[0]
        # Tool attachments cleared, note appended
        tool_msgs = [
            m for m in agent.context.short_term_memory._messages
            if m.role == Role.TOOL
        ]
        assert tool_msgs[-1].tool_result.attachments is None
        assert "<system-reminder>" in tool_msgs[-1].tool_result.content

    @pytest.mark.asyncio
    async def test_user_side_propagates_no_strip(self) -> None:
        """First LLM call rejects with no prior tool messages — propagate."""
        from agent_harness.core.errors import LLMUnsupportedContentError

        llm = _RejectOnNthCallLLM(reject_at=1)
        tracker = _SelfHealTracker()
        agent = ConversationalAgent(
            name="test", llm=llm, system_prompt="", hooks=tracker,
        )

        with pytest.raises(LLMUnsupportedContentError):
            await agent.run("hello")

        assert tracker.heals == []
        assert len(tracker.errors) == 1
        assert isinstance(tracker.errors[0], LLMUnsupportedContentError)

    @pytest.mark.asyncio
    async def test_persistent_rejection_after_strip_propagates(self) -> None:
        """Second LLM call after strip also rejects → single retry only."""
        from agent_harness.agent.react import ReActAgent
        from agent_harness.core.errors import LLMUnsupportedContentError
        from agent_harness.core.message import Role

        class _AlwaysRejectAfterToolCall(MockLLM):
            def __init__(self) -> None:
                super().__init__()
                self._calls = 0

            async def generate(self, messages: list[Message], **kwargs: Any) -> Any:
                self._calls += 1
                if self._calls == 1:
                    return await super().generate(messages, **kwargs)
                raise LLMUnsupportedContentError("invalid part type: file")

        llm = _AlwaysRejectAfterToolCall()
        llm.add_tool_call_response("mock_tool", {"query": "fetch"})

        tracker = _SelfHealTracker()
        tool = _MediaReturningTool()
        agent = ReActAgent(
            name="test", llm=llm, tools=[tool], system_prompt="", hooks=tracker,
        )

        with pytest.raises(LLMUnsupportedContentError):
            await agent.run("go")

        assert len(tracker.heals) == 1
        tool_msgs = [
            m for m in agent.context.short_term_memory._messages
            if m.role == Role.TOOL
        ]
        assert tool_msgs[-1].tool_result.attachments is None


class _SessionRecorderTool(BaseTool):
    """Records every session id pushed via bind_session — used to verify
    SessionAware lifecycle (binds at run start, including the sessionless
    path that clears a previously-bound id back to None).
    """

    def __init__(self) -> None:
        super().__init__(name="session_recorder", description="record session binds")
        self.binds: list[str | None] = []

    def bind_session(self, session_id: str | None) -> None:
        self.binds.append(session_id)

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class TestSessionAwareLifecycle:
    """Issue: a sessionless run after a sessionful run must clear the
    previously-bound session id from SessionAware tools — otherwise tools
    keep writing into the prior session's storage."""

    @pytest.mark.asyncio
    async def test_sessionless_run_unbinds_prior_session(self) -> None:
        from agent_harness.session.memory_session import InMemorySession

        llm = MockLLM()
        llm.add_text_response("a")
        llm.add_text_response("b")

        tool = _SessionRecorderTool()
        agent = ConversationalAgent(
            name="t", llm=llm, tools=[tool], system_prompt="",
        )

        # First run with a session — binds "S1"
        await agent.run("first", session=InMemorySession(session_id="S1"))
        bound_tool = agent.tool_registry.get("session_recorder")
        assert isinstance(bound_tool, _SessionRecorderTool)
        assert "S1" in bound_tool.binds

        # Second run without a session — must rebind to None,
        # otherwise tools keep using S1's session storage.
        await agent.run("second")
        assert bound_tool.binds[-1] is None


class TestAgentClose:
    """aclose() releases the agent's LLM pools, deduping a compressor that
    shares the main provider and closing a distinct one."""

    def _agent(self) -> Any:
        from agent_harness.agent.react import ReActAgent
        return ReActAgent(name="t", llm=MockLLM(), system_prompt="")

    @pytest.mark.asyncio
    async def test_closes_main_llm(self) -> None:
        from unittest.mock import AsyncMock

        agent = self._agent()
        agent.llm.aclose = AsyncMock()  # type: ignore[method-assign]
        await agent.aclose()
        agent.llm.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shared_compressor_llm_closed_once(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        agent = self._agent()
        agent.llm.aclose = AsyncMock()  # type: ignore[method-assign]
        agent.context.short_term_memory.compressor = SimpleNamespace(_llm=agent.llm)

        await agent.aclose()

        agent.llm.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_distinct_compressor_llm_closed_too(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        agent = self._agent()
        agent.llm.aclose = AsyncMock()  # type: ignore[method-assign]
        summary_llm = MockLLM()
        summary_llm.aclose = AsyncMock()  # type: ignore[method-assign]
        agent.context.short_term_memory.compressor = SimpleNamespace(_llm=summary_llm)

        await agent.aclose()

        agent.llm.aclose.assert_awaited_once()
        summary_llm.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_error_does_not_propagate(self) -> None:
        from unittest.mock import AsyncMock

        agent = self._agent()
        agent.llm.aclose = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        await agent.aclose()  # no raise

    @pytest.mark.asyncio
    async def test_closes_current_sub_compressor_and_retired_once(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        main = MockLLM()
        sub = MockLLM()
        retired = MockLLM()
        for runtime_llm in (main, sub, retired):
            runtime_llm.aclose = AsyncMock()  # type: ignore[method-assign]

        agent = ConversationalAgent(
            name="test",
            llm=main,
            sub_llm=sub,
            system_prompt="",
        )
        agent._retired_llms[id(retired)] = retired
        agent.context.short_term_memory.compressor = SimpleNamespace(_llm=sub)

        await agent.aclose()

        main.aclose.assert_awaited_once()
        sub.aclose.assert_awaited_once()
        retired.aclose.assert_awaited_once()


class TestSubLLMLifecycle:
    def test_sub_llm_is_keyword_only_and_positional_tools_remain_valid(self) -> None:
        llm = MockLLM()
        tool = MockTool()

        agent = ConversationalAgent("test", llm, [tool], system_prompt="")

        assert agent.llm is llm
        assert agent.sub_llm is llm
        assert agent.tool_registry.has("mock_tool")

    def test_auto_creates_configured_sub_llm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        main = MockLLM(config=LLMConfig(model="main"))
        sub = MockLLM(config=LLMConfig(model="sub"))
        config = HarnessConfig(
            llm=LLMConfig(model="main", sub_model={"model": "sub"}),
        )
        monkeypatch.setattr("agent_harness.agent.base.create_llm", lambda _cfg: main)
        monkeypatch.setattr("agent_harness.agent.base.create_sub_llm", lambda _cfg: sub)

        agent = ConversationalAgent(name="test", config=config, system_prompt="")

        assert agent.llm is main
        assert agent.sub_llm is sub

    def test_unconfigured_sub_aliases_created_main(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        main = MockLLM(config=LLMConfig(model="main"))
        monkeypatch.setattr("agent_harness.agent.base.create_llm", lambda _cfg: main)
        monkeypatch.setattr("agent_harness.agent.base.create_sub_llm", lambda _cfg: None)

        agent = ConversationalAgent(
            name="test",
            config=HarnessConfig(llm=LLMConfig(model="main")),
            system_prompt="",
        )

        assert agent.sub_llm is main

    def test_explicit_main_skips_both_factories(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from unittest.mock import MagicMock

        create_main = MagicMock()
        create_sub = MagicMock()
        monkeypatch.setattr("agent_harness.agent.base.create_llm", create_main)
        monkeypatch.setattr("agent_harness.agent.base.create_sub_llm", create_sub)
        main = MockLLM(config=LLMConfig(model="main"))

        agent = ConversationalAgent(name="test", llm=main, system_prompt="")

        assert agent.sub_llm is main
        create_main.assert_not_called()
        create_sub.assert_not_called()

    def test_explicit_main_and_sub_are_preserved(self) -> None:
        main = MockLLM(config=LLMConfig(model="main"))
        sub = MockLLM(config=LLMConfig(model="sub"))

        agent = ConversationalAgent(
            name="test",
            llm=main,
            sub_llm=sub,
            system_prompt="",
        )

        assert agent.llm is main
        assert agent.sub_llm is sub
        assert main._event_bus is agent.context.event_bus
        assert sub._event_bus is agent.context.event_bus

    def test_explicit_main_syncs_memory_and_compressor(self) -> None:
        from agent_harness.memory.compressor import ContextCompressor

        old = MockLLM(config=LLMConfig(model="old"))
        main = MockLLM(config=LLMConfig(model="main"))
        sub = MockLLM(config=LLMConfig(model="sub"))
        compressor = ContextCompressor(llm=old, model="old")
        context = AgentContext(compressor=compressor)
        context.short_term_memory.last_call = object()  # type: ignore[assignment]

        agent = ConversationalAgent(
            name="test",
            llm=main,
            sub_llm=sub,
            context=context,
            system_prompt="",
        )

        assert agent.context.short_term_memory.model == "main"
        assert agent.context.short_term_memory.last_call is None
        assert compressor._llm is sub
        assert compressor._model == "main"

    @pytest.mark.parametrize(
        ("old_distinct", "new_distinct", "retired_count"),
        [
            (False, False, 1),
            (False, True, 1),
            (True, False, 2),
            (True, True, 2),
        ],
    )
    def test_replace_llms_handles_alias_combinations(
        self,
        old_distinct: bool,
        new_distinct: bool,
        retired_count: int,
    ) -> None:
        old_main = MockLLM(config=LLMConfig(model="old-main"))
        old_sub = (
            MockLLM(config=LLMConfig(model="old-sub"))
            if old_distinct
            else old_main
        )
        agent = ConversationalAgent(
            name="test",
            llm=old_main,
            sub_llm=old_sub,
            system_prompt="",
        )
        new_main = MockLLM(config=LLMConfig(model="new-main"))
        new_sub = (
            MockLLM(config=LLMConfig(model="new-sub"))
            if new_distinct
            else new_main
        )

        agent.replace_llms(new_main, new_sub)

        assert agent.llm is new_main
        assert agent.sub_llm is new_sub
        assert len(agent._retired_llms) == retired_count
        assert new_main._event_bus is agent.context.event_bus
        assert new_sub._event_bus is agent.context.event_bus

    def test_replace_same_model_name_still_clears_snapshot(self) -> None:
        old = MockLLM(config=LLMConfig(model="same"))
        agent = ConversationalAgent(name="test", llm=old, system_prompt="")
        agent.context.short_term_memory.last_call = object()  # type: ignore[assignment]
        new = MockLLM(config=LLMConfig(model="same"))

        agent.replace_llms(new, new)

        assert agent.context.short_term_memory.last_call is None

    def test_unchanged_sub_is_not_retired_and_compressor_is_rebound(self) -> None:
        main = MockLLM(config=LLMConfig(model="main"))
        sub = MockLLM(config=LLMConfig(model="sub"))
        agent = ConversationalAgent(
            name="test",
            llm=main,
            sub_llm=sub,
            system_prompt="",
        )
        new_main = MockLLM(config=LLMConfig(model="new-main"))

        agent.replace_llms(new_main, sub)

        assert id(main) in agent._retired_llms
        assert id(sub) not in agent._retired_llms
        compressor = agent.context.short_term_memory.compressor
        assert compressor is not None
        assert compressor._llm is sub
        assert compressor._model == "new-main"
