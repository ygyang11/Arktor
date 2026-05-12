"""Tests for BaseAgent._session_metadata_extras dict round-trip via finally save."""
from __future__ import annotations

import pytest

from agent_harness.agent.react import ReActAgent
from agent_harness.session.memory_session import InMemorySession
from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_extras_written_to_metadata_on_save() -> None:
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)
    agent._session_metadata_extras["_plan_mode"] = True
    agent._session_metadata_extras["_custom_key"] = "x"

    session = InMemorySession("s1")
    await agent.run("hi", session=session)

    state = await session.load_state()
    assert state is not None
    assert state.metadata["_plan_mode"] is True
    assert state.metadata["_custom_key"] == "x"


@pytest.mark.asyncio
async def test_extras_empty_does_not_write_keys() -> None:
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)

    session = InMemorySession("s2")
    await agent.run("hi", session=session)

    state = await session.load_state()
    assert state is not None
    assert "_plan_mode" not in state.metadata


@pytest.mark.asyncio
async def test_owned_keys_win_over_extras_on_save() -> None:
    # extras is updated first; owned keys write after, so live owned values win
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)
    agent._session_metadata_extras["_approval_mode"] = "stale"

    session = InMemorySession("s3")
    await agent.run("hi", session=session)

    state = await session.load_state()
    assert state is not None
    assert state.metadata["_approval_mode"] != "stale"
    assert state.metadata["_approval_mode"] == agent._approval.mode


@pytest.mark.asyncio
async def test_reset_session_state_clears_extras() -> None:
    llm = MockLLM()
    agent = ReActAgent(name="t", llm=llm, max_steps=1)
    agent._session_metadata_extras["_plan_mode"] = True
    agent._session_metadata_extras["_custom"] = "x"

    await agent.reset_session_state("new-id")
    assert agent._session_metadata_extras == {}


@pytest.mark.asyncio
async def test_reset_session_state_clears_context_patches() -> None:
    from agent_harness.core.message import Message
    from agent_harness.prompt.patch import ContextPatch

    llm = MockLLM()
    agent = ReActAgent(name="t", llm=llm, max_steps=1)
    agent.context.context_patches.append(
        ContextPatch(at="tail", build=lambda: Message.user("x")),
    )

    await agent.reset_session_state("new-id")
    assert agent.context.context_patches == []


@pytest.mark.asyncio
async def test_apply_session_state_clears_context_patches() -> None:
    from agent_harness.core.message import Message
    from agent_harness.prompt.patch import ContextPatch
    from agent_harness.session.memory_session import InMemorySession
    from agent_harness.session.base import SessionState

    llm = MockLLM()
    agent = ReActAgent(name="t", llm=llm, max_steps=1)
    agent.context.context_patches.append(
        ContextPatch(at="tail", build=lambda: Message.user("stale")),
    )

    session = InMemorySession("s")
    await session.save_state(SessionState(session_id="s", messages=[]))
    state = await session.load_state()
    assert state is not None
    await agent.apply_session_state(state)
    assert agent.context.context_patches == []
