"""Regression tests for BaseAgent bug fixes.

Covers:
- Agent reuse (second run() after FINISHED state)
- Usage accumulation across steps
- PlanAgent state reset on re-run
- use_long_term_memory instance flag
- Approval integration
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_harness.agent.base import StepResult
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
from agent_harness.core.message import Message
from agent_harness.hooks import DefaultHooks, TracingHooks
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
