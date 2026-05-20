"""Agent Harness exception hierarchy.

All framework exceptions inherit from HarnessError, enabling broad catch
while allowing fine-grained handling of specific error categories.
"""
from __future__ import annotations

from typing import Any


class HarnessError(Exception):
    """Base exception for all agent_harness errors."""

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# --- LLM Errors ---

class LLMError(HarnessError):
    """Error during LLM interaction."""

    def __init__(
        self, message: str = "", *, details: dict[str, Any] | None = None,
    ) -> None:
        from agent_harness.utils.blob import scrub_base64  # noqa: PLC0415

        super().__init__(scrub_base64(message), details=details)

class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded. Retryable."""
    def __init__(self, message: str = "Rate limit exceeded", *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

class LLMConnectionError(LLMError):
    """Provider connection/transport failure. Retryable."""

class LLMAuthenticationError(LLMError):
    """Invalid or missing API credentials."""

class LLMContextLengthError(LLMError):
    """Input exceeds model context window."""
    def __init__(
        self,
        message: str = "Context length exceeded",
        *,
        actual_tokens: int | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.actual_tokens = actual_tokens
        self.max_tokens = max_tokens

class LLMResponseError(LLMError):
    """Failed to parse LLM response."""


# --- Tool Errors ---

class ToolError(HarnessError):
    """Error during tool execution."""

class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""

class ToolTimeoutError(ToolError):
    """Tool execution exceeded timeout."""
    def __init__(
        self,
        message: str = "Tool timed out",
        *,
        tool_name: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds

class ToolValidationError(ToolError):
    """Tool arguments failed validation."""

class ToolExecutionError(ToolError):
    """Tool execution failed at runtime."""


# --- Config Errors ---

class ConfigError(HarnessError):
    """Configuration error."""

class ConfigMissingError(ConfigError):
    """Required configuration value is missing."""


# --- Context Errors ---

class ContextError(HarnessError):
    """Error related to agent context."""

class StateTransitionError(ContextError):
    """Invalid state transition."""


# --- Memory Errors ---

class MemoryError(HarnessError):
    """Error related to memory operations."""


# --- Orchestration Errors ---

class OrchestrationError(HarnessError):
    """Error during agent orchestration."""

class CyclicDependencyError(OrchestrationError):
    """Detected cyclic dependency in DAG orchestration."""
    def __init__(
        self,
        message: str = "Cyclic dependency detected",
        *,
        cycle_path: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.cycle_path = cycle_path


# --- Agent Errors ---

class AgentError(HarnessError):
    """Error during agent execution."""

class ApprovalError(HarnessError):
    """Error during the approval process (e.g., handler failure)."""


class MaxStepsExceededError(AgentError):
    """Agent exceeded maximum allowed steps."""
    def __init__(
        self,
        message: str = "Max steps exceeded",
        *,
        max_steps: int | None = None,
        actual_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.max_steps = max_steps
        self.actual_steps = actual_steps


class LoopDetectedError(AgentError):
    """Agent stuck in a repetitive loop."""
    def __init__(
        self,
        message: str = "Loop detected",
        *,
        streak: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.streak = streak


# --- HTTP Errors ---

class HttpResponseTooLargeError(HarnessError):
    """HTTP response body exceeded the configured byte cap."""
    def __init__(
        self,
        message: str = "Response exceeds byte cap",
        *,
        limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.limit = limit
