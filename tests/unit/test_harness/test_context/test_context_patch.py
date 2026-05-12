"""Tests for ContextPatch dispatch in BaseAgent.call_llm."""
from __future__ import annotations

import pytest

from agent_harness.agent.react import ReActAgent
from agent_harness.core.message import Message, Role
from agent_harness.prompt.patch import ContextPatch
from tests.conftest import MockLLM


@pytest.mark.asyncio
async def test_system_patch_prepended_to_extra_sys() -> None:
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)

    agent.context.context_patches.append(
        ContextPatch(at="system", build=lambda: Message.system("INJECT-SYS")),
    )

    await agent.run("hi")
    first_call = llm.call_history[0]
    contents = [m.content for m in first_call if m.role == Role.SYSTEM]
    assert any("INJECT-SYS" in (c or "") for c in contents)


@pytest.mark.asyncio
async def test_tail_patch_appended_after_messages() -> None:
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)

    agent.context.context_patches.append(
        ContextPatch(at="tail", build=lambda: Message.user("TAIL-NOTE")),
    )

    await agent.run("hi")
    first_call = llm.call_history[0]
    assert first_call[-1].content == "TAIL-NOTE"


@pytest.mark.asyncio
async def test_patch_returning_none_is_skipped() -> None:
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)

    agent.context.context_patches.append(
        ContextPatch(at="tail", build=lambda: None),
    )

    await agent.run("hi")
    first_call = llm.call_history[0]
    assert all("TAIL-NOTE" not in (m.content or "") for m in first_call)


@pytest.mark.asyncio
async def test_system_and_tail_patches_coexist() -> None:
    llm = MockLLM()
    llm.add_text_response("done")
    agent = ReActAgent(name="t", llm=llm, max_steps=1)

    agent.context.context_patches.extend([
        ContextPatch(at="system", build=lambda: Message.system("S1")),
        ContextPatch(at="tail", build=lambda: Message.user("T1")),
    ])

    await agent.run("hi")
    first_call = llm.call_history[0]
    sys_contents = [m.content for m in first_call if m.role == Role.SYSTEM]
    assert any("S1" in (c or "") for c in sys_contents)
    assert first_call[-1].content == "T1"
