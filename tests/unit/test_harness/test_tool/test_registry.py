"""Tests for agent_harness.tool.registry — ToolRegistry CRUD and schema generation."""
from __future__ import annotations

from typing import Any

import pytest

from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.registry import ToolRegistry


class _DummyTool(BaseTool):
    """Minimal concrete tool for registry tests."""

    async def execute(self, **kwargs: Any) -> str:
        return "dummy"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)


class TestToolRegistry:
    def _make_tool(self, name: str = "my_tool", desc: str = "A tool") -> _DummyTool:
        return _DummyTool(name=name, description=desc)

    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        t = self._make_tool()
        reg.register(t)
        assert reg.get("my_tool") is t

    def test_get_unknown_raises(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_registry_membership_contract(self) -> None:
        reg = ToolRegistry()
        t = self._make_tool()
        assert reg.has("my_tool") is False
        assert "my_tool" not in reg
        assert len(reg) == 0
        reg.register(t)
        assert reg.has("my_tool") is True
        assert "my_tool" in reg
        assert len(reg) == 1

    def test_list_tools(self) -> None:
        reg = ToolRegistry()
        t1 = self._make_tool("alpha", "First")
        t2 = self._make_tool("beta", "Second")
        reg.register(t1)
        reg.register(t2)
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"alpha", "beta"}

    def test_get_schemas(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("x", "Desc X"))
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "x"
        assert schemas[0].description == "Desc X"

    def test_unregister(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("removable"))
        assert reg.has("removable")
        reg.unregister("removable")
        assert not reg.has("removable")

    def test_get_openai_schemas(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("fn1", "Func one"))
        schemas = reg.get_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "fn1"

    def test_get_anthropic_schemas(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("fn1", "Func one"))
        schemas = reg.get_anthropic_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "fn1"
        assert "input_schema" in schemas[0]

    def test_repr(self) -> None:
        reg = ToolRegistry()
        reg.register(self._make_tool("t1"))
        r = repr(reg)
        assert "ToolRegistry" in r
        assert "t1" in r


# ── _tool_state_restoring contextvar ─────────────────────────────────

class _StatefulTool(_DummyTool):
    """Tool whose notify_state records the contextvar value seen at call time."""

    def __init__(self, name: str = "stateful") -> None:
        super().__init__(name=name, description="stateful")
        self._state: dict[str, Any] = {}
        self.notify_calls: list[bool] = []

    def get_state(self) -> dict[str, Any] | None:
        return self._state or None

    def restore_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)

    async def notify_state(self, hooks: Any, agent_name: str) -> None:
        from agent_harness.tool.registry import _tool_state_restoring
        self.notify_calls.append(_tool_state_restoring.get(False))


@pytest.mark.asyncio
async def test_restore_states_sets_contextvar_for_notify_state() -> None:
    """During `restore_states`, the `_tool_state_restoring` contextvar reads
    True inside `notify_state`; outside that window it stays False."""
    from agent_harness.hooks.base import DefaultHooks
    from agent_harness.tool.registry import _tool_state_restoring

    reg = ToolRegistry()
    tool = _StatefulTool()
    reg.register(tool)

    assert _tool_state_restoring.get(False) is False
    await reg.restore_states({"stateful": {"k": "v"}}, DefaultHooks(), "agent")
    assert tool.notify_calls == [True]
    assert _tool_state_restoring.get(False) is False  # reset on exit


@pytest.mark.asyncio
async def test_restore_states_resets_contextvar_on_exception() -> None:
    from agent_harness.hooks.base import DefaultHooks
    from agent_harness.tool.registry import _tool_state_restoring

    class _Boom(_StatefulTool):
        async def notify_state(self, hooks: Any, agent_name: str) -> None:
            raise RuntimeError("boom")

    reg = ToolRegistry()
    reg.register(_Boom())
    with pytest.raises(RuntimeError, match="boom"):
        await reg.restore_states({"stateful": {}}, DefaultHooks(), "agent")
    assert _tool_state_restoring.get(False) is False


@pytest.mark.asyncio
async def test_restore_states_empty_skips_contextvar_block() -> None:
    """If there's nothing to restore, the contextvar shouldn't even toggle."""
    from agent_harness.hooks.base import DefaultHooks

    reg = ToolRegistry()
    await reg.restore_states({}, DefaultHooks(), "agent")  # no-op, no error
