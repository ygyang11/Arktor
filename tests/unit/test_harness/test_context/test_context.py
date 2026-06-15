"""Tests for agent_harness.context.context — AgentContext create, fork, isolation."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_harness.context.context import AgentContext
from agent_harness.context.state import AgentState
from agent_harness.context.variables import ContextVariables, Scope
from agent_harness.core.config import HarnessConfig, LLMConfig, MemoryConfig
from agent_harness.core.event import EventBus
from agent_harness.memory.compressor import ContextCompressor
from agent_harness.memory.short_term import ShortTermMemory
from agent_harness.memory.working_term import WorkingMemory


class TestAgentContextCreate:
    def test_create_with_defaults(self) -> None:
        ctx = AgentContext.create()
        assert isinstance(ctx.config, HarnessConfig)
        assert isinstance(ctx.short_term_memory, ShortTermMemory)
        assert isinstance(ctx.working_memory, WorkingMemory)
        assert isinstance(ctx.variables, ContextVariables)
        assert isinstance(ctx.event_bus, EventBus)
        assert ctx.long_term_memory is None
        assert ctx.tracer is None
        assert ctx.state.current == AgentState.IDLE

    def test_create_with_custom_config(self) -> None:
        cfg = HarnessConfig(memory=MemoryConfig(max_tokens=50000))
        ctx = AgentContext.create(config=cfg)
        assert ctx.config.memory.max_tokens == 50000
        assert ctx.short_term_memory.max_tokens == 50000

    def test_create_propagates_llm_model_to_short_term_memory(self) -> None:
        cfg = HarnessConfig(llm=LLMConfig(model="gpt-5-mini"))
        ctx = AgentContext.create(config=cfg)
        assert ctx.short_term_memory.model == "gpt-5-mini"

    def test_create_uses_active_harness_config_instance(self) -> None:
        original_instance = HarnessConfig._instance
        cfg = HarnessConfig(memory=MemoryConfig(max_tokens=50000))
        HarnessConfig._instance = cfg
        try:
            ctx = AgentContext.create()
            assert ctx.config is cfg
            assert ctx.short_term_memory.max_tokens == 50000
        finally:
            HarnessConfig._instance = original_instance

    def test_create_with_explicit_components(self) -> None:
        bus = EventBus()
        stm = ShortTermMemory(max_tokens=5000)
        ctx = AgentContext(short_term_memory=stm, event_bus=bus)
        assert ctx.short_term_memory is stm
        assert ctx.event_bus is bus


class TestAgentContextFork:
    def test_fork_shares_event_bus(self) -> None:
        parent = AgentContext.create()
        child = parent.fork("child")
        assert child.event_bus is parent.event_bus

    def test_fork_shares_config(self) -> None:
        parent = AgentContext.create()
        child = parent.fork()
        assert child.config is parent.config

    def test_fork_propagates_llm_model_to_child_short_term_memory(self) -> None:
        cfg = HarnessConfig(llm=LLMConfig(model="claude-4-sonnet"))
        parent = AgentContext.create(config=cfg)
        child = parent.fork()
        assert child.short_term_memory.model == "claude-4-sonnet"

    def test_fork_shares_long_term_memory(self) -> None:
        parent = AgentContext.create()
        child = parent.fork()
        assert child.long_term_memory is parent.long_term_memory

    def test_fork_independent_short_term_memory(self) -> None:
        parent = AgentContext.create()
        child = parent.fork("child")
        assert child.short_term_memory is not parent.short_term_memory

    def test_fork_clones_compressor_instead_of_sharing(self) -> None:
        compressor = ContextCompressor(
            llm=AsyncMock(),
            threshold=0.75,
            retain_count=4,
            model="gpt-4o",
        )
        parent = AgentContext(compressor=compressor)

        child = parent.fork("executor")

        assert child.short_term_memory.compressor is not None
        assert child.short_term_memory.compressor is not compressor

    def test_fork_uses_name_as_scope_when_provided(self) -> None:
        compressor = ContextCompressor(
            llm=AsyncMock(),
            threshold=0.75,
            retain_count=4,
            model="gpt-4o",
        )
        parent = AgentContext(compressor=compressor)

        child = parent.fork("executor")

        assert child._compressor is not None
        assert child._compressor.scope == "executor"

    def test_fork_assigns_numbered_scope_for_unnamed_child(self) -> None:
        compressor = ContextCompressor(
            llm=AsyncMock(),
            threshold=0.75,
            retain_count=4,
            model="gpt-4o",
        )
        parent = AgentContext(compressor=compressor)

        child1 = parent.fork()
        child2 = parent.fork()

        assert child1._compressor is not None
        assert child2._compressor is not None
        assert child1._compressor.scope == "child_01"
        assert child2._compressor.scope == "child_02"

    @pytest.mark.asyncio
    async def test_fork_short_term_isolation(self) -> None:
        parent = AgentContext.create()
        from agent_harness.core.message import Message
        await parent.short_term_memory.add_message(Message.user("parent msg"))

        child = parent.fork()
        # Child should have empty short-term memory
        assert await child.short_term_memory.size() == 0
        # Parent still has its message
        assert await parent.short_term_memory.size() == 1

    def test_fork_independent_working_memory(self) -> None:
        parent = AgentContext.create()
        child = parent.fork()
        assert child.working_memory is not parent.working_memory

    def test_fork_independent_state(self) -> None:
        parent = AgentContext.create()
        child = parent.fork()
        assert child.state is not parent.state
        assert child.state.current == AgentState.IDLE

    def test_fork_shared_global_variables(self) -> None:
        parent = AgentContext.create()
        parent.variables.set("shared_key", "shared_val", scope=Scope.GLOBAL)

        child = parent.fork()
        assert child.variables.get("shared_key") == "shared_val"

        # Modification in child global scope is visible in parent
        child.variables.set("child_global", 42, scope=Scope.GLOBAL)
        assert parent.variables.get("child_global") == 42

    def test_fork_independent_agent_variables(self) -> None:
        parent = AgentContext.create()
        parent.variables.set("agent_key", "parent_val", scope=Scope.AGENT)

        child = parent.fork()
        # Child should NOT see parent's agent-scoped variable
        assert child.variables.get("agent_key") is None

        child.variables.set("agent_key", "child_val", scope=Scope.AGENT)
        # Parent's variable unchanged
        assert parent.variables.get("agent_key") == "parent_val"

    def test_repr(self) -> None:
        ctx = AgentContext.create()
        r = repr(ctx)
        assert "AgentContext" in r


class TestBuildLLMMessages:
    @pytest.mark.asyncio
    async def test_basic_build_from_short_term(self) -> None:
        from agent_harness.core.message import Message
        ctx = AgentContext.create()
        await ctx.short_term_memory.add_message(Message.system("You are helpful."))
        await ctx.short_term_memory.add_message(Message.user("Hello"))

        msgs = await ctx.build_llm_messages()
        assert len(msgs) == 2
        assert msgs[0].role.value == "system"
        assert msgs[1].role.value == "user"

    @pytest.mark.asyncio
    async def test_build_injects_working_memory(self) -> None:
        from agent_harness.core.message import Message
        ctx = AgentContext.create()
        await ctx.short_term_memory.add_message(Message.system("System"))
        await ctx.short_term_memory.add_message(Message.user("Hi"))
        ctx.working_memory.set("goal", "test goal")

        msgs = await ctx.build_llm_messages(include_working=True)
        # Working memory injected after system message
        assert len(msgs) == 3
        assert msgs[0].role.value == "system"
        assert "goal" in msgs[1].content
        assert msgs[2].role.value == "user"

    @pytest.mark.asyncio
    async def test_build_skips_working_when_disabled(self) -> None:
        from agent_harness.core.message import Message
        ctx = AgentContext.create()
        await ctx.short_term_memory.add_message(Message.user("Hi"))
        ctx.working_memory.set("goal", "ignored")

        msgs = await ctx.build_llm_messages(include_working=False)
        assert len(msgs) == 1

    @pytest.mark.asyncio
    async def test_build_with_base_messages_override(self) -> None:
        from agent_harness.core.message import Message
        ctx = AgentContext.create()
        await ctx.short_term_memory.add_message(Message.user("ignored"))

        custom = [Message.system("Custom"), Message.user("Direct")]
        msgs = await ctx.build_llm_messages(base_messages=custom, include_working=False)
        assert len(msgs) == 2
        assert msgs[0].content == "Custom"


class TestStateManager:
    """Tests for state machine transitions."""

    def test_initial_state_is_idle(self) -> None:
        from agent_harness.context.state import StateManager
        sm = StateManager()
        assert sm.current == AgentState.IDLE

    def test_valid_transition_sequence(self) -> None:
        from agent_harness.context.state import StateManager
        sm = StateManager()
        sm.transition(AgentState.THINKING)
        assert sm.current == AgentState.THINKING
        sm.transition(AgentState.ACTING)
        assert sm.current == AgentState.ACTING
        sm.transition(AgentState.OBSERVING)
        assert sm.current == AgentState.OBSERVING
        sm.transition(AgentState.FINISHED)
        assert sm.current == AgentState.FINISHED

    def test_invalid_transition_raises_error(self) -> None:
        from agent_harness.context.state import StateManager
        from agent_harness.core.errors import StateTransitionError
        sm = StateManager()
        with pytest.raises(StateTransitionError, match="Invalid transition"):
            sm.transition(AgentState.ACTING)

    def test_is_terminal(self) -> None:
        from agent_harness.context.state import StateManager
        sm = StateManager()
        assert not sm.is_terminal
        sm.transition(AgentState.THINKING)
        sm.transition(AgentState.FINISHED)
        assert sm.is_terminal

    def test_reset_returns_to_idle(self) -> None:
        from agent_harness.context.state import StateManager
        sm = StateManager()
        sm.transition(AgentState.THINKING)
        sm.transition(AgentState.FINISHED)
        sm.reset()
        assert sm.current == AgentState.IDLE


class TestContextVariables:
    """Tests for scoped variable storage."""

    def test_set_get_default_agent_scope(self) -> None:
        cv = ContextVariables()
        cv.set("key", "value")
        assert cv.get("key") == "value"

    def test_get_returns_default_when_missing(self) -> None:
        cv = ContextVariables()
        assert cv.get("nope") is None
        assert cv.get("nope", 42) == 42

    def test_global_scope_shared_via_fork(self) -> None:
        cv = ContextVariables()
        cv.set("shared", "data", scope=Scope.GLOBAL)
        cv.set("private", "secret", scope=Scope.AGENT)
        child = cv.fork()
        assert child.get("shared") == "data"
        assert child.get("private") is None

    def test_agent_scope_overrides_global(self) -> None:
        cv = ContextVariables()
        cv.set("key", "global-val", scope=Scope.GLOBAL)
        cv.set("key", "agent-val", scope=Scope.AGENT)
        assert cv.get("key") == "agent-val"

    def test_has_checks_both_scopes(self) -> None:
        cv = ContextVariables()
        cv.set("g", 1, scope=Scope.GLOBAL)
        cv.set("a", 2, scope=Scope.AGENT)
        assert cv.has("g")
        assert cv.has("a")
        assert not cv.has("missing")
