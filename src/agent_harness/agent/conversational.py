"""Conversational agent: simple chat without tool usage."""
from __future__ import annotations

from agent_harness.agent.base import BaseAgent, StepResult


class ConversationalAgent(BaseAgent):
    """Simple conversational agent that generates responses without tool calling.

    Use for: chatbots, summarization, translation, and other tasks
    that don't require external tool interaction.

    Always completes in a single step.
    """

    async def step(self) -> StepResult:
        """Generate a response from the LLM (no tools)."""
        response = await self.call_llm(tools=None)

        return StepResult(
            thought=self.llm.reasoning_text(response.message),
            response=response.message.content or "",
        )
