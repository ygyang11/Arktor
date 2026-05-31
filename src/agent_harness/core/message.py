"""
Core message model for the agent framework.

This module defines the fundamental message types and structures used for
communication between all components in the agent framework (LLM, Agent, Tool).
Messages serve as the universal protocol for inter-component communication.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Message roles defining the sender type."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Attachment(BaseModel):
    """Immutable reference to a media payload in the blob store."""

    model_config = ConfigDict(frozen=True)

    digest: str
    mime: str
    filename: str | None = None
    size: int


class ToolCall(BaseModel):
    """
    Represents a tool/function call made by the assistant.
    
    Attributes:
        id: Unique identifier for the tool call, auto-generated if not provided
        name: The name of the tool/function to call
        arguments: Arguments to pass to the tool, as a dictionary
    """
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    """Rich tool return: text content plus optional media attachments and metadata."""

    content: str
    attachments: list[Attachment] | None = None
    tool_metadata: dict[str, Any] | None = None


class ToolResult(BaseModel):
    """
    Represents the result from executing a tool call.

    Attributes:
        tool_call_id: Reference to the ToolCall that produced this result
        content: The output/result from the tool execution
        is_error: Whether the tool execution resulted in an error
        attachments: Optional media attachments produced by the tool
        tool_metadata: Optional metadata associated with the tool.
    """
    tool_call_id: str
    content: str
    attachments: list[Attachment] | None = None
    tool_metadata: dict[str, Any] | None = None
    is_error: bool = False


class Message(BaseModel):
    """
    Universal message type - the fundamental communication unit in the framework.
    
    All components (LLM, Agent, Tool) communicate through Message objects.
    A message can represent system instructions, user inputs, assistant responses,
    or tool execution results.
    
    Attributes:
        role: The role/type of message sender
        content: Text content of the message
        name: Optional name of the message sender
        tool_calls: List of tool calls made by the assistant (if role is assistant)
        tool_result: Tool execution result (if role is tool)
        metadata: Additional arbitrary metadata associated with the message
        created_at: Timestamp when the message was created (UTC)
    """
    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    attachments: list[Attachment] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now())

    # Convenience factory methods
    @classmethod
    def system(cls, content: str, **kwargs: Any) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs: Any) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content, **kwargs)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        **kwargs: Any,
    ) -> Message:
        """Create an assistant message, optionally with tool calls."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls, **kwargs)

    @classmethod
    def tool(
        cls,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
        attachments: list[Attachment] | None = None,
        tool_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Message:
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            tool_result=ToolResult(
                tool_call_id=tool_call_id,
                content=content,
                attachments=attachments,
                tool_metadata=tool_metadata,
                is_error=is_error,
            ),
            content=content,
            **kwargs,
        )

    @property
    def has_tool_calls(self) -> bool:
        """Check if this message contains tool calls."""
        return bool(self.tool_calls)


class MessageChunk(BaseModel):
    """
    Incremental chunk for streaming responses.
    
    Used when streaming message content from LLMs or other components
    to efficiently handle partial/incremental updates.
    
    Attributes:
        delta_content: Incremental content update
        delta_tool_calls: Incremental tool call updates
        finish_reason: Reason for stream completion (e.g., "stop", "tool_calls")
    """
    delta_content: str | None = None
    delta_tool_calls: list[ToolCall] | None = None
    delta_provider_metadata: dict[str, dict[str, Any]] | None = None
    finish_reason: str | None = None

    def merge_provider_metadata_into(self, target: dict[str, dict[str, Any]]) -> None:
        if not self.delta_provider_metadata:
            return
        for proto_key, fields in self.delta_provider_metadata.items():
            bucket = target.setdefault(proto_key, {})
            for fname, value in fields.items():
                if isinstance(value, str) and isinstance(bucket.get(fname), str):
                    bucket[fname] = bucket[fname] + value
                else:
                    bucket[fname] = value
