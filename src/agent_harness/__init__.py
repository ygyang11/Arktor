"""Arktor: A complete, extensible agent framework."""
from agent_harness.utils.logging_config import setup_logging

# Auto-configure logging on import
setup_logging()

from agent_harness.agent.base import AgentResult, BaseAgent
from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.agent.planner import PlanAgent, PlanAndExecuteAgent
from agent_harness.agent.react import ReActAgent
from agent_harness.approval import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    StdinApprovalHandler,
)
from agent_harness.context.context import AgentContext
from agent_harness.core.config import (
    ApprovalConfig,
    DockerConfig,
    HarnessConfig,
    LLMConfig,
    MemoryConfig,
    PaperConfig,
    SandboxConfig,
    SearchConfig,
    SkillConfig,
    SubAgentConfig,
    SubAgentTypeSpec,
    ToolConfig,
    TracingConfig,
)
from agent_harness.core.errors import HarnessError, LLMConnectionError
from agent_harness.core.event import Event, EventBus
from agent_harness.core.message import Attachment, Message, Role, ToolCall, ToolOutput, ToolResult
from agent_harness.llm.base import BaseLLM
from agent_harness.sandbox import ExecuteResult, LocalBackend, SandboxBackend, SandboxManager
from agent_harness.session import BaseSession, FileSession, InMemorySession, SessionState
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.decorator import tool

__version__ = "0.5.2"

__all__ = [
    # Logging
    "setup_logging",
    # Approval
    "ApprovalAction",
    "ApprovalConfig",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalResult",
    "StdinApprovalHandler",
    # Core
    "Attachment",
    "Message",
    "Role",
    "ToolCall",
    "ToolOutput",
    "ToolResult",
    "DockerConfig",
    "HarnessConfig",
    "LLMConfig",
    "MemoryConfig",
    "SandboxConfig",
    "TracingConfig",
    "ToolConfig",
    "SearchConfig",
    "PaperConfig",
    "SkillConfig",
    "SubAgentConfig",
    "SubAgentTypeSpec",
    "Event",
    "EventBus",
    "HarnessError",
    "LLMConnectionError",
    # LLM
    "BaseLLM",
    # Tool
    "BaseTool",
    "ToolSchema",
    "tool",
    # Agent
    "BaseAgent",
    "AgentResult",
    "ReActAgent",
    "PlanAgent",
    "PlanAndExecuteAgent",
    "ConversationalAgent",
    # Context
    "AgentContext",
    # Sandbox
    "ExecuteResult",
    "LocalBackend",
    "SandboxBackend",
    "SandboxManager",
    # Session
    "BaseSession",
    "SessionState",
    "FileSession",
    "InMemorySession",
]
