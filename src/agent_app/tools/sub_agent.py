"""SubAgent — runtime sub-agent delegation tool."""
from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

from agent_harness.core.errors import ToolExecutionError, ToolValidationError
from agent_harness.session.memory_session import InMemorySession
from agent_harness.tool.base import BaseTool, ToolSchema

if TYPE_CHECKING:
    from agent_harness.agent.base import AgentResult
    from agent_harness.prompt.system_builder import SystemPromptBuilder

logger = logging.getLogger(__name__)

# ── Fallback intro (used when config custom types have no intro) ──

_SUBAGENT_INTRO = """\
You are a sub-agent assisting the primary agent with a focused task. \
Given the task description, use your available tools to complete it fully.

When you complete the task, respond with a concise report covering what \
was done and any key findings.

Rules:
- Complete the task efficiently — don't over-engineer, but don't leave it half-done
- Do not ask clarifying questions — interpret the request directly
- If the first approach doesn't work, try alternative strategies — \
exhaust reasonable options before reporting failure"""

# Injected into EVERY sub-agent prompt (built-in, custom, and fallback), 
# so the constraint can't be bypassed by a custom type that supplies its own intro.
_SUBAGENT_BG_CONSTRAINT = (
    "You MUST NOT run tools in background mode (background=true): the "
    "result is GUARANTEED LOST — continuing means it never returns to "
    "you, and ANY attempt to wait exits you immediately."
)

# ── Built-in type definitions ──

_BUILTIN_TYPES: dict[str, dict[str, Any]] = {
    "research": {
        "tools": [
            "read_file", "list_dir", "glob_files", "grep_files",
            "web_fetch", "web_search",
            "paper_search", "paper_fetch", "document_parser",
            "memory_tool",
        ],
        "intro": (
            "You are a research sub-agent specialized in exploring codebases, "
            "searching the web, and gathering information.\n\n"
            "=== CRITICAL: READ-ONLY MODE — NO FILE MODIFICATIONS ===\n"
            "This is a READ-ONLY research task. You are STRICTLY PROHIBITED from:\n"
            "- Creating, modifying, or deleting files\n"
            "- Running commands that change system state\n"
            "- Saving or deleting memories, but reading is ok\n"
            "Your role is EXCLUSIVELY to search, read, and analyze.\n\n"
            "Your strengths:\n"
            "- Searching, reading and analyzing file contents across large codebases\n"
            "- Web research, academic paper lookup, and document parsing\n\n"
            "When you complete the task, respond with a structured "
            "report of your findings. Be factual and specific — include "
            "file paths, function names, code snippets, URLs, or references "
            "as appropriate."
        ),
    },
    "plan": {
        "tools": [
            "read_file", "list_dir", "glob_files", "grep_files",
            "web_fetch", "web_search",
            "paper_search", "paper_fetch", "document_parser",
            "memory_tool",
        ],
        "intro": (
            "You are a planning sub-agent specialized in analyzing content and "
            "designing implementation strategies.\n\n"
            "=== CRITICAL: READ-ONLY MODE — NO FILE MODIFICATIONS ===\n"
            "This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:\n"
            "- Creating, modifying, or deleting files\n"
            "- Running commands that change system state\n"
            "- Saving or deleting memories, but reading is ok\n"
            "Your role is EXCLUSIVELY to explore existing content and design plans.\n\n"
            "Your strengths:\n"
            "- Analyzing multiple files to understand system architecture\n"
            "- Identifying critical files, dependencies, and potential impacts\n"
            "- Producing clear, actionable implementation plans\n\n"
            "When you complete the task, produce a clear, actionable plan. "
            "Include specific file paths, concrete steps, and design decisions "
            "with rationale. The plan should be detailed enough to implement "
            "without further clarification."
        ),
    },
    "general": {
        "tools": "__inherit__",
        "intro": (
            "You are an general sub-agent with full tool access. Given the "
            "task description, use your available tools to complete it fully — "
            "don't over-engineer, but don't leave it half-done.\n\n"
            "Your strengths:\n"
            "- Completing multi-step implementation tasks end-to-end\n"
            "- Making coordinated changes across multiple files\n"
            "- Building, testing, and verifying changes\n"
            "- Handling any substantial, self-contained task that benefits from isolation\n\n"
            "Guidelines:\n"
            "- Don't add features or make improvements beyond what was asked\n"
            "- Be careful not to introduce security vulnerabilities\n"
            "- Stay focused on the task scope — do not expand beyond the stated objective\n\n"
            "When you complete the task, respond with a concise report covering "
            "what was done and any key findings."
        ),
    },
}

_ALWAYS_EXCLUDE = frozenset({"sub_agent", "todo_write", "skill_tool", "background_task"})

# ── Tool Description Template ──

_TOOL_DESCRIPTION_TEMPLATE = """\
Launch an ephemeral sub-agent to handle complex, multi-step tasks \
with isolated context. Each sub-agent runs autonomously and returns \
its final output with an execution summary (steps, tools used, duration).

## Available Agent Types
{agent_types}

## Usage Notes
1. Each sub-agent invocation is stateless. Your prompt should contain \
all necessary context for the sub-agent to perform the task autonomously. \
Specify exactly what information the sub-agent should return.
2. You can launch multiple sub-agents concurrently to maximize performance by calling sub_agent \
multiple times in a single turn.
3. The sub-agent's outputs should generally be trusted.
4. Set background=true when your workflow can proceed without waiting \
for this result.

## Examples

<example>
User: "I want to refactor the error handling. Compare how errors \
are handled across the auth, billing, and notification modules."
Assistant: *Launches three research sub-agents in parallel, one \
per module*
<commentary>
The comparison requires analyzing three independent modules, each \
spanning multiple files. Each module's analysis is self-contained \
and does not depend on the others. Parallel sub-agents complete \
faster and each dives deep into one module's patterns, absorbing \
the context cost of scanning many files. The main agent receives \
three focused reports and synthesizes the comparison, rather than \
carrying the full exploration history of all three modules in its \
own context.
</commentary>
</example>

<example>
User: "Study this open-source project's architecture and design \
patterns — I want to learn from their approach before designing \
our own implementation."
Assistant: *Launches a single research sub-agent to analyze the \
codebase*
<commentary>
The sub-agent explores a large external codebase end-to-end — \
reading source files, tracing patterns, and synthesizing findings \
into a structured analysis. This keeps the main thread clean of \
the heavy exploration overhead. The main agent receives a concise \
architectural summary to reference when designing the implementation, \
rather than carrying hundreds of file reads in its own context.
</commentary>
</example>

<example>
User: "I need to add WebSocket support to this REST API. Design \
an implementation plan."
Assistant: *Launches a plan sub-agent to analyze the existing API \
architecture and design the WebSocket integration strategy*
<commentary>
Designing a plan requires deep analysis of the existing architecture \
across many files. The sub-agent reads the codebase deeply, evaluates \
trade-offs, and produces a structured plan. This is analysis-heavy \
work that benefits from context isolation — the main agent receives \
the plan and can then proceed based on its recommendations.
</commentary>
</example>

<example>
User: "The utils/ module has no tests. Create a comprehensive test \
suite and make sure everything passes."
Assistant: *Launches a general sub-agent to create the test suite*
<commentary>
This is a self-contained implementation task: read each source file, \
write corresponding tests, and run them to verify — many steps where \
only the outcome matters. Use general when the task requires file \
modifications or command execution, the scope is well-defined, and \
you don't need the intermediate details in your context. If the task is small or you need the details in your own context \
for subsequent work, do it yourself instead.
</commentary>
</example>

<example>
User: "Analyze our competitor's open-source project and then refactor \
our codebase following their best practices."
Assistant: *Launches a research sub-agent in background to analyze \
the competitor's codebase, then starts reading our own codebase to \
understand the current structure*
<commentary>
The analysis will take many steps exploring an external codebase. \
Meanwhile the agent can begin reading our own code — work that \
doesn't depend on the analysis result. When the background research \
completes, the agent combines both to plan the refactoring.
</commentary>
</example>

<example>
User: "Check if the login function handles rate limiting."
Assistant: *Uses grep_files to find the login function, then reads \
the relevant file directly*
<commentary>
The assistant did not use sub_agent because this is a focused question \
with a narrow target. Even if it takes a few attempts to locate, the \
exploration is lightweight and the findings may inform immediate next \
steps.
</commentary>
</example>

### Example with custom agent types:

<example_agent_descriptions>
"security": use this agent to perform security-focused code review
</example_agent_descriptions>

<example>
User: "Review the code I just wrote for security issues."
Assistant: *Launches a security sub-agent to review the changes*
<commentary>
A custom "security" type has been configured for this project with \
specialized review instructions. The task falls within this agent's specialized focus area, \
so it can apply its tailored instructions and capabilities more effectively than general. \
Custom types appear in the agent_type enum — use them when the task matches their description.
</commentary>
</example>
"""


class SubAgentTool(BaseTool):
    """Runtime sub-agent delegation tool.

    Spawns an isolated ReActAgent to execute a focused task.
    The sub-agent runs to completion and returns its final output.
    """

    def __init__(self) -> None:
        super().__init__(
            name="sub_agent",
            description="",
            executor_timeout=600.0,
        )
        self._agent: Any = None
        self._type_seq: dict[str, int] = {}
        self._session_id: str | None = None

    def bind_agent(self, agent: Any) -> None:
        self._agent = agent

    def bind_session(self, session_id: str | None) -> None:
        self._session_id = session_id

    def get_schema(self) -> ToolSchema:
        available_types = self._get_available_types()
        type_enum = list(available_types.keys())

        type_lines = []
        for name, spec in available_types.items():
            intro = spec.get("intro", "")
            short = intro.split("\n")[0].strip() if intro else name
            type_lines.append(f"- {name}: {short}")
        agent_types = "\n".join(type_lines)
        description = _TOOL_DESCRIPTION_TEMPLATE.format(agent_types=agent_types)

        return ToolSchema(
            name=self.name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short (3-10 word) description of the task.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The task for the sub-agent to perform. Brief it "
                            "like a colleague who just walked in — explain "
                            "what to accomplish, what you've already learned, "
                            "and give enough context for judgment calls. The "
                            "sub-agent has no access to your conversation."
                        ),
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": type_enum,
                        "description": "Select one of the available agent types listed above.",
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "Run in background. Returns a task ID immediately; "
                            "results are delivered automatically when complete."
                        ),
                        "default": False,
                    },
                },
                "required": ["description", "prompt", "agent_type"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        from agent_harness.hooks.progress import _subagent_active

        if self._agent is None:
            raise ToolExecutionError(
                "SubAgentTool is not bound to a parent agent. "
                "Register it via BaseAgent(tools=[sub_agent, ...])."
            )

        description = kwargs.get("description", "")
        prompt = kwargs.get("prompt", "")
        agent_type = kwargs.get("agent_type", "")
        background = kwargs.get("background", False)

        if not description.strip():
            raise ToolValidationError("'description' is required and cannot be empty.")
        if not prompt.strip():
            raise ToolValidationError("'prompt' is required and cannot be empty.")
        if not agent_type.strip():
            raise ToolValidationError("'agent_type' is required and cannot be empty.")

        available_types = self._get_available_types()
        if agent_type not in available_types:
            raise ToolValidationError(
                f"Unknown agent_type '{agent_type}'. "
                f"Available: {list(available_types.keys())}"
            )

        tools = self._resolve_tools(agent_type)
        prompt_builder = self._build_subagent_prompt_builder(agent_type)

        parent_name = self._agent.name
        self._type_seq[agent_type] = self._type_seq.get(agent_type, 0) + 1
        seq = self._type_seq[agent_type]
        subagent_name = f"{parent_name}.sub.{agent_type}.{seq}"

        max_steps = self._agent.context.config.sub_agent.max_steps

        from agent_harness.agent.react import ReActAgent

        child = ReActAgent(
            name=subagent_name,
            llm=self._agent.llm,
            tools=tools,
            context=self._agent.context.fork(f"sub.{agent_type}.{seq}"),
            hooks=self._agent.hooks,
            max_steps=max_steps,
            stream=False if background else self._agent._stream,
            approval=self._agent._approval,
            approval_handler=self._agent._approval_handler,
            prompt_builder=prompt_builder,
        )
        child._usage_source = "background" if background else "subagent"
        # Share parent's sandbox (child reuses the same container / backend)
        child._sandbox = self._agent._sandbox

        await self._agent.hooks.on_subagent_start(
            parent_name, subagent_name, agent_type, description, prompt,
        )

        sub_session = (
            InMemorySession(self._session_id) if self._session_id else None
        )

        if background:
            return self._start_background(
                child, description, prompt, agent_type, subagent_name,
                sub_session,
            )

        # Synchronous path
        try:
            result, tool_usage, elapsed_ms = await self._run_child(
                child, prompt, parent_name, subagent_name,
                agent_type, description, sub_session,
            )
            return self._format_result(
                output=result.output,
                steps=result.step_count,
                tool_usage=tool_usage,
                duration_ms=elapsed_ms,
            )
        except Exception as e:
            raise ToolExecutionError(
                f"Sub-agent '{subagent_name}' failed: {e}"
            ) from e

    def _start_background(
        self,
        child: Any,
        description: str,
        prompt: str,
        agent_type: str,
        subagent_name: str,
        sub_session: InMemorySession | None,
    ) -> str:
        from agent_harness.utils.token_counter import truncate_text_by_tokens

        parent_name = self._agent.name

        async def work() -> tuple[str, str]:
            result, tool_usage, _ = await self._run_child(
                child, prompt, parent_name, subagent_name,
                agent_type, description, sub_session,
            )
            result_preview = truncate_text_by_tokens(
                result.output, max_tokens=100, suffix="..."
            )
            summary = (
                f"Completed in {result.step_count} steps, "
                f"{sum(tool_usage.values())} tool calls.\n"
                f"Result: {result_preview}"
            )
            return result.output, summary

        desc = truncate_text_by_tokens(description, max_tokens=12, suffix="...")
        task_id = self._agent._bg_manager.spawn(
            tool_name="sub_agent",
            description=desc,
            coro=work(),
        )
        return f"Background sub-agent {task_id} started ({agent_type}): {description}"

    async def _run_child(
        self,
        child: Any,
        prompt: str,
        parent_name: str,
        subagent_name: str,
        agent_type: str,
        description: str,
        sub_session: InMemorySession | None,
    ) -> tuple[Any, dict[str, int], float]:
        """Run child agent with hooks and progress tracking."""
        from agent_harness.hooks.progress import _subagent_active

        start_time = time.monotonic()
        _steps = 0
        _tool_calls = 0
        token = _subagent_active.set(True)
        try:
            result = await child.run(prompt, session=sub_session)
            _steps = result.step_count
            tool_usage = self._extract_tool_usage(result)
            _tool_calls = sum(tool_usage.values())
            return result, tool_usage, (time.monotonic() - start_time) * 1000
        finally:
            _subagent_active.reset(token)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            try:
                await self._agent.hooks.on_subagent_end(
                    parent_name, subagent_name, agent_type, description,
                    _steps, _tool_calls, elapsed_ms,
                )
            except Exception:
                pass

    def _get_available_types(self) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = dict(_BUILTIN_TYPES)
        if self._agent is not None:
            config_types = self._agent.context.config.sub_agent.types
            for name, spec in config_types.items():
                merged[name] = {
                    "tools": spec.tools if spec.tools else merged.get(name, {}).get("tools", []),
                    "intro": spec.intro or merged.get(name, {}).get("intro", ""),
                }
        return merged

    def _resolve_tools(self, agent_type: str) -> list[BaseTool]:
        assert self._agent is not None
        type_spec = self._get_available_types()[agent_type]
        parent_tools: list[BaseTool] = self._agent.tools
        tool_names = type_spec.get("tools", [])

        if tool_names == "__inherit__":
            filtered = [t for t in parent_tools if t.name not in _ALWAYS_EXCLUDE]
        elif not tool_names:
            logger.warning(
                "Sub-agent type '%s' has no tools configured. "
                "The sub-agent will run without any tools. "
                "If this is unintended, add tools to the type config.",
                agent_type,
            )
            return []
        else:
            allowed = set(tool_names) - _ALWAYS_EXCLUDE
            filtered = [t for t in parent_tools if t.name in allowed]

        return self._isolate_agent_aware(filtered)

    @staticmethod
    def _isolate_agent_aware(tools: list[BaseTool]) -> list[BaseTool]:
        """Create fresh instances for AgentAware tools to prevent rebind."""
        from agent_harness.tool.base import AgentAware

        result: list[BaseTool] = []
        for t in tools:
            if isinstance(t, AgentAware):
                result.append(t.__class__())
            else:
                result.append(t)
        return result

    def _build_subagent_prompt_builder(self, agent_type: str) -> SystemPromptBuilder:
        from agent_harness.prompt.sections import make_intro_section

        assert self._agent is not None
        builder = self._agent._prompt_builder.fork()

        type_spec = self._get_available_types()[agent_type]
        intro = type_spec.get("intro") or _SUBAGENT_INTRO
        intro = f"{intro}\n\n{_SUBAGENT_BG_CONSTRAINT}"
        builder.register(make_intro_section(intro))

        return builder

    def _format_result(
        self,
        output: str,
        steps: int,
        tool_usage: dict[str, int],
        duration_ms: float,
    ) -> str:
        parts = [output.strip()]

        duration_s = duration_ms / 1000
        tool_summary = ", ".join(
            f"{name} x{count}" for name, count in tool_usage.items()
        )
        summary = f"Steps: {steps}"
        if tool_summary:
            summary += f" | Tools: {tool_summary}"
        summary += f" | Duration: {duration_s:.1f}s"
        parts.append(f"\n[Execution: {summary}]")
        return "\n".join(parts)

    @staticmethod
    def _extract_tool_usage(result: AgentResult) -> dict[str, int]:
        usage: dict[str, int] = {}
        for step in result.steps:
            if step.action:
                for tc in step.action:
                    usage[tc.name] = usage.get(tc.name, 0) + 1
        return usage


sub_agent = SubAgentTool()

SUB_AGENT_TOOLS: list[BaseTool] = [sub_agent]
