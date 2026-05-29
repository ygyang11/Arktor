"""Terminal tool for shell command execution.

Security model — three layers:
- Layer 1 (Permission): ApprovalPolicy decides if execution is allowed
- Layer 2 (Sandbox): Pluggable backend for execution isolation
- Layer 3 (Tool): Pure execution — no command filtering

Commands run from the workspace root. The sandbox backend determines
the execution environment (host subprocess or container).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_harness.core.errors import ToolExecutionError, ToolValidationError
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.utils.token_counter import truncate_text_by_tokens

_MAX_OUTPUT_TOKENS: int = 15_000
_DEFAULT_TIMEOUT: int = 120  # seconds
_MAX_TIMEOUT: int = 600  # seconds (10 minutes)
_MAX_BG_TIMEOUT: int = 86400  # seconds (24 hours) — background tasks can run much longer

TERMINAL_TOOL_DESCRIPTION = (
    "Execute a shell command in a bash subprocess and return its output.\n\n"
    "The working directory defaults to the workspace root. "
    "Shell state (variables, aliases) does not persist between calls — "
    "each invocation starts a fresh subprocess.\n\n"
    "Use this tool for operations without a dedicated equivalent: "
    "git, pytest, pip, npm, make, docker, curl, python/node/bash scripts, etc.\n\n"
    "Set background=true for commands that take a long time where blocking "
    "would be wasteful (e.g. model training, large builds, long data processing). "
    "Results are delivered automatically when complete.\n\n"
    "Examples:\n"
    "  Good:\n"
    "    terminal_tool(command='pytest tests/ -v')\n"
    "    terminal_tool(command='python scripts/migrate.py')\n"
    '    terminal_tool(command=\'git add -A && git commit -m "fix bug"\')\n'
    "    terminal_tool(command='cd src && python -m mymodule', timeout=300)\n"
    "    terminal_tool(command='python train.py --epochs=50', background=true)\n"
    "  Bad:\n"
    "    terminal_tool(command='cat file.txt')       # use dedicated file reading tool\n"
    "    terminal_tool(command='grep -r pattern .')   # use dedicated search tool\n"
    "    terminal_tool(command='find . -name *.py')   # use dedicated file search tool\n"
    "    terminal_tool(command='python scripts/check_env.py', background=true)  # need result before proceeding\n\n"
    "Guidelines:\n"
    "- Always quote file paths containing spaces with double quotes\n"
    "- If a command creates files or directories, verify the parent exists first\n"
    "- Chain dependent commands with && (e.g. 'cd src && pytest')\n"
    "- Use ; only when you don't care if earlier commands fail\n"
    "- Set appropriate timeout for long-running commands (default 120s, max 600s; "
    "background mode allows up to 24h)"
)


def _workspace_root() -> Path:
    return Path.cwd().resolve()


class TerminalTool(BaseTool):

    def __init__(self) -> None:
        super().__init__(
            name="terminal_tool",
            description=TERMINAL_TOOL_DESCRIPTION,
            executor_timeout=_MAX_TIMEOUT + 10,
            approval_resource_key="command",
        )
        self._agent: Any = None

    def bind_agent(self, agent: Any) -> None:
        self._agent = agent

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to execute. Supports full bash syntax "
                            "including pipes, redirects, chaining, and variable expansion."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Maximum execution time in seconds "
                            f"(default {_DEFAULT_TIMEOUT}, max {_MAX_TIMEOUT}; "
                            f"background mode max {_MAX_BG_TIMEOUT})."
                        ),
                        "default": _DEFAULT_TIMEOUT,
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
                "required": ["command"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        background = kwargs.get("background", False)

        if not command.strip():
            raise ToolValidationError("command cannot be empty")
        if timeout <= 0:
            raise ToolValidationError("timeout must be greater than 0")
        if self._agent is None:
            raise ToolExecutionError(
                "terminal_tool is not bound to a parent agent. "
                "Register it via BaseAgent(tools=[terminal_tool, ...])."
            )

        if background:
            timeout = min(timeout, _MAX_BG_TIMEOUT)
            return self._start_background(command, timeout)

        timeout = min(timeout, _MAX_TIMEOUT)

        result = await self._agent._sandbox.execute(
            command, timeout=timeout, workdir=str(_workspace_root()),
        )
        exit_code = result.exit_code

        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}".strip() if output else result.stderr
        if output:
            output = truncate_text_by_tokens(
                output, max_tokens=_MAX_OUTPUT_TOKENS, suffix="\n... (truncated)",
            )

        if exit_code is None:
            return f"[exit code N/A]\n{output}".rstrip()
        if exit_code != 0:
            return f"[exit code {exit_code}]\n{output}".rstrip()
        return output or "(no output)"

    def _start_background(self, command: str, timeout: int) -> str:
        sandbox = self._agent._sandbox

        async def work() -> tuple[str, str]:
            result = await sandbox.execute(
                command, timeout=timeout, workdir=str(_workspace_root()),
            )
            if result.exit_code is None:
                raise RuntimeError(result.stdout)
            output = result.stdout
            if result.stderr:
                output = f"{output}\n{result.stderr}".strip() if output else result.stderr
            summary = _build_terminal_summary(result.exit_code, output)
            return output, summary

        desc = truncate_text_by_tokens(command, max_tokens=12, suffix="...")
        task_id = self._agent._bg_manager.spawn(
            tool_name="terminal_tool",
            description=desc,
            coro=work(),
        )
        return f"Background command {task_id} started: {command}"


def _build_terminal_summary(exit_code: int | None, output: str) -> str:
    lines = output.splitlines()
    head = "\n".join(lines[:3]) if len(lines) > 3 else ""
    tail = "\n".join(lines[-5:]) if len(lines) > 5 else output
    parts: list[str] = []
    if exit_code is not None:
        parts.append(f"Exit code: {exit_code}")
    else:
        parts.append("Exit code: N/A (timeout or error)")
    parts.append(f"Output: {len(lines)} lines")
    if head and head != tail:
        parts.append(f"Head:\n{head}")
    if tail:
        parts.append(f"Tail:\n{tail}")
    return "\n".join(parts)


terminal_tool = TerminalTool()
