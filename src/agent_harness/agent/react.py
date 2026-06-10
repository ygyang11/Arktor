"""ReAct agent: Reasoning + Acting loop."""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.agent.base import BaseAgent, StepResult

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """ReAct agent implementing the Reasoning + Acting paradigm.

    Execution loop:
    1. THINK: LLM reasons about the current state and decides what to do
    2. ACT: If LLM calls tools, execute them
    3. OBSERVE: Feed tool results back to LLM
    4. Repeat until LLM provides a final answer (no tool calls)

    Supports:
    - Parallel tool calls (when LLM returns multiple tool_calls)
    - Automatic thought chain tracking
    - Configurable system prompt (uses DEFAULT_INTRO when not provided)
    - Force tool usage via tool_choice

    Example:
        agent = ReActAgent(
            name="researcher",
            llm=openai_provider,
            tools=[search_tool, calculator_tool],
        )
        result = await agent.run("What is the population of France?")
    """

    def __init__(self, system_prompt: str | None = None, **kwargs: Any) -> None:
        super().__init__(system_prompt=system_prompt or "", **kwargs)

    async def step(self) -> StepResult:
        """Execute one ReAct cycle: Think -> (Act -> Observe)? -> Response?"""
        # THINK: Call LLM with current context and available tools
        response = await self.call_llm()
        message = response.message
        thought = self.llm.reasoning_text(message)

        # Check if LLM wants to call tools
        if response.has_tool_calls:
            tool_calls = message.tool_calls or []

            logger.debug(
                "Agent '%s' calling %d tool(s): %s",
                self.name,
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )

            # ACT: Execute tools
            results = await self.execute_tools(tool_calls)

            # OBSERVE: Results are now in short-term memory
            # Return step with action — loop continues
            return StepResult(
                thought=thought,
                action=tool_calls,
                observation=results,
                response=message.content,
            )

        # No tool calls — LLM is providing a final answer
        return StepResult(
            thought=thought,
            response=message.content or "",
        )
