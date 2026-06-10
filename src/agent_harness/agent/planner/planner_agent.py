"""PlannerAgent — generates a structured plan from user request."""
from __future__ import annotations

from typing import Any

from agent_harness.agent.base import BaseAgent, StepResult

_PLANNER_PROMPT = (
    "You are a planning agent specialized in task decomposition and strategic planning.\n"
    "\n"
    "## Role\n"
    "Analyze the user's request and break it down into a structured, actionable plan.\n"
    "Think carefully about dependencies between steps and the optimal execution order.\n"
    "\n"
    "## Output Format\n"
    "Respond with ONLY a valid JSON object. Do not include any text, markdown, or explanation\n"
    "outside the JSON structure.\n"
    "\n"
    "Required schema:\n"
    "{\n"
    '    "goal": "A clear, refined statement of the overall objective",\n'
    '    "steps": [\n'
    '        {\n'
    '            "id": "1",\n'
    '            "description": "Clear description of what this step accomplishes"\n'
    '        }\n'
    "    ]\n"
    "}\n"
    "\n"
    "## Planning Rules\n"
    "1. Keep each step atomic — one clear action per step\n"
    "2. Steps must be independently executable given prior step results\n"
    "3. Order steps by dependency: prerequisites before dependents\n"
    "4. Aim for 2-6 steps; avoid over-decomposition for simple tasks\n"
    "5. Each step should produce a verifiable outcome\n"
    "\n"
    "## Constraints\n"
    "- Output ONLY valid JSON — no markdown fences, no preamble, no commentary\n"
    "- Every step MUST have 'id' and 'description' fields\n"
    "- The 'id' field must be a unique string (e.g., '1', '2', '3')\n"
    "- Do not reference tools or implementation details — focus on WHAT, not HOW\n"
    "- If the task is trivial, a single-step plan is acceptable"
)


class PlannerAgent(BaseAgent):
    """Generates a structured plan from the user's request.

    Pure LLM reasoning — no tools. Produces a JSON plan matching the
    Plan / PlanStep schema. Always completes in a single step.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _PLANNER_PROMPT)
        super().__init__(**kwargs)

    async def step(self) -> StepResult:
        response = await self.call_llm(tools=None)
        return StepResult(
            thought=self.llm.reasoning_text(response.message),
            response=response.message.content,
        )
