"""ExecutorAgent — executes a single plan step using available tools."""
from __future__ import annotations

from typing import Any

from agent_harness.agent.base import BaseAgent, StepResult

_EXECUTOR_PROMPT = (
    "You are an execution agent responsible for completing a specific step of a plan.\n"
    "\n"
    "## Role\n"
    "You receive a single step to execute along with context from previously completed steps.\n"
    "Use the available tools to gather information, perform actions, and accomplish the step.\n"
    "\n"
    "## Execution Rules\n"
    "1. Focus ONLY on the current step — do not attempt other steps\n"
    "2. Use tools when you need external information or must perform actions\n"
    "3. You may call multiple tools in sequence if the step requires it\n"
    "4. Analyze tool results before deciding next actions\n"
    "5. When the step is complete, provide a clear, concise summary of the result\n"
    "\n"
    "## Output Guidelines\n"
    "- Your final response should summarize what was accomplished\n"
    "- Include key data, findings, or outcomes from tool usage\n"
    "- Report failures clearly if the step could not be completed\n"
    "- Keep the result focused and actionable for subsequent steps\n"
    "\n"
    "## Constraints\n"
    "- Do not attempt steps outside your current assignment\n"
    "- Do not modify the plan — only execute the assigned step\n"
    "- If a step cannot be completed with available tools, explain why\n"
    "- Provide your result as plain text (not JSON)"
)


class ExecutorAgent(BaseAgent):
    """Executes a single plan step using available tools (ReAct-style).

    Supports multi-step tool calling via the standard BaseAgent.run() loop.
    Each call to step() either invokes tools (loop continues) or returns
    a final text result (loop ends).
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _EXECUTOR_PROMPT)
        super().__init__(**kwargs)

    async def step(self) -> StepResult:
        response = await self.call_llm()
        message = response.message
        thought = self.llm.reasoning_text(message)
        if response.has_tool_calls:
            tool_calls = message.tool_calls or []
            results = await self.execute_tools(tool_calls)
            return StepResult(
                thought=thought,
                action=tool_calls,
                observation=results,
                response=message.content,
            )
        return StepResult(thought=thought, response=message.content or "")
