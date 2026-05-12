"""Tool registry for managing available tools."""
from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from agent_harness.core.registry import Registry
from agent_harness.tool.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from agent_harness.hooks.base import DefaultHooks


_tool_state_restoring: ContextVar[bool] = ContextVar(
    "_tool_state_restoring", default=False,
)


class ToolRegistry:
    """Registry for tool instances.

    Provides tool discovery and schema generation for LLM function calling.

    Example:
        registry = ToolRegistry()
        registry.register(my_tool)
        schemas = registry.get_schemas()  # Pass to LLM
    """

    def __init__(self) -> None:
        self._registry: Registry[BaseTool] = Registry()

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name."""
        self._registry.register(tool.name, tool)

    def get(self, name: str) -> BaseTool:
        """Get a tool by name. Raises KeyError if not found."""
        return self._registry.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return self._registry.has(name)

    def unregister(self, name: str) -> None:
        """Remove a tool."""
        self._registry.unregister(name)

    def list_tools(self) -> list[BaseTool]:
        """List all registered tools."""
        return list(self._registry.list_all().values())

    def get_schemas(self) -> list[ToolSchema]:
        """Get schemas for all registered tools (for LLM function calling)."""
        return [tool.get_schema() for tool in self.list_tools()]

    def get_openai_schemas(self) -> list[dict]:
        """Get all tool schemas in OpenAI format."""
        return [schema.to_openai_format() for schema in self.get_schemas()]

    def get_anthropic_schemas(self) -> list[dict]:
        """Get all tool schemas in Anthropic format."""
        return [schema.to_anthropic_format() for schema in self.get_schemas()]

    def __len__(self) -> int:
        return len(self._registry)

    # ── Stateful tool support ──

    def save_states(self) -> dict[str, dict[str, Any]]:
        """Collect states from all stateful tools for session persistence."""
        states: dict[str, dict[str, Any]] = {}
        for tool in self._registry.list_all().values():
            s = tool.get_state()
            if s is not None:
                states[tool.name] = s
        return states

    async def restore_states(
        self,
        states: dict[str, dict[str, Any]],
        hooks: DefaultHooks,
        agent_name: str,
    ) -> None:
        """Restore tool states from session and notify hooks."""
        if not states:
            return
        token = _tool_state_restoring.set(True)
        try:
            for tool in self._registry.list_all().values():
                if tool.name in states:
                    tool.restore_state(states[tool.name])
                    await tool.notify_state(hooks, agent_name)
        finally:
            _tool_state_restoring.reset(token)

    def __contains__(self, name: str) -> bool:
        return self._registry.has(name)

    def __repr__(self) -> str:
        tools = self._registry.list_names()
        return f"ToolRegistry(tools={tools})"
