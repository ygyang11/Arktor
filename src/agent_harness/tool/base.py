"""Base tool interface and schema for agent_harness.

Tools are the primary way agents interact with the external world.
Every tool exposes a JSON Schema description for LLM function calling.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_harness.core.message import Message
    from agent_harness.hooks.base import DefaultHooks

from pydantic import BaseModel, Field

from agent_harness.core.message import ToolOutput


@runtime_checkable
class AgentAware(Protocol):
    """Protocol for tools that need access to the parent agent.

    Tools implementing this protocol will receive a reference to
    the parent agent during registration via bind_agent().
    """

    def bind_agent(self, agent: Any) -> None: ...


@runtime_checkable
class SessionAware(Protocol):
    """Protocol for tools that need the current session id.

    Tools implementing this protocol will receive the session id
    whenever it is bound, restored, or switched, via bind_session().
    """

    def bind_session(self, session_id: str | None) -> None: ...


class ToolParameter(BaseModel):
    """Description of a single tool parameter."""
    name: str
    type: str  # JSON Schema type: "string", "integer", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


class ToolSchema(BaseModel):
    """JSON Schema-compatible tool description for LLM function calling.

    This is the format passed to LLM providers to describe available tools.
    """
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclass this to create custom tools, or use the @tool decorator
    for simpler function-based tools.

    Attributes:
        name: Unique tool name (used by LLM to invoke).
        description: Human-readable description (included in LLM prompt).
    """

    name: str
    description: str

    def __init__(
        self,
        name: str,
        description: str,
        *,
        executor_timeout: float | None = None,
        approval_resource_key: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.executor_timeout = executor_timeout
        self.approval_resource_key = approval_resource_key
        self.context_order: int = 0

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | ToolOutput:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments matching the schema.

        Returns:
            String result to be passed back to the LLM, or a ToolOutput
            carrying text plus optional media attachments.
        """
        ...

    def get_schema(self) -> ToolSchema:
        """Get the JSON Schema description of this tool.

        Default implementation returns a basic schema.
        Override in subclasses or use @tool decorator for auto-generation.
        """
        return ToolSchema(name=self.name, description=self.description)

    # ── Stateful tool protocol ──

    def get_state(self) -> dict[str, Any] | None:
        """Return intermediate state for session persistence.

        Stateless tools return None (default). Stateful tools override
        to return a serializable state dict.
        """
        return None

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore intermediate state from session data."""

    def reset_state(self) -> None:
        """Clear intermediate state. Stateful tools override."""

    async def notify_state(self, hooks: DefaultHooks, agent_name: str) -> None:
        """Notify hooks after state change. Stateful tools override."""

    def build_context_message(self) -> Message | None:
        """Build ephemeral context for LLM injection. Stateful tools override."""
        return None

    def clone(self) -> BaseTool:
        """Create an independent copy for per-agent isolation.

        Called by BaseAgent during tool registration. Each agent gets
        its own tool instance to prevent bind_agent/state pollution.

        Default uses deepcopy with _agent temporarily cleared to avoid
        copying the entire agent object graph. Override if the tool holds
        non-copyable resources (connections, locks) or should be shared.
        """
        saved = getattr(self, "_agent", None)
        try:
            if saved is not None:
                self._agent = None
            return copy.deepcopy(self)
        finally:
            if saved is not None:
                self._agent = saved

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
