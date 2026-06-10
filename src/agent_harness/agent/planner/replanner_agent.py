"""ReplannerAgent — evaluates step results and decides next action."""
from __future__ import annotations

from typing import Any

from agent_harness.agent.base import BaseAgent, StepResult

_REPLANNER_PROMPT = (
    "You are a replanning agent that evaluates execution progress and decides the next action.\n"
    "\n"
    "## Role\n"
    "After each step is executed, you receive the original goal, the current plan status,\n"
    "and the result of the latest step. Your job is to evaluate whether the goal has been\n"
    "achieved, whether the remaining plan is still valid, or whether replanning is needed.\n"
    "\n"
    "## Output Format\n"
    "Respond with ONLY a valid JSON object in one of these three formats:\n"
    "\n"
    "### Goal Achieved — task is complete\n"
    '{"goal_achieved": true, "final_answer": "Comprehensive answer addressing the original task"}\n'
    "\n"
    "### Continue — proceed with the next step as planned\n"
    '{"goal_achieved": false, "should_replan": false}\n'
    "\n"
    "### Replan — modify the remaining steps\n"
    "{\n"
    '    "goal_achieved": false,\n'
    '    "should_replan": true,\n'
    '    "reason": "Explanation of why replanning is needed",\n'
    '    "updated_steps": [\n'
    '        {"id": "N", "description": "New or revised step"}\n'
    "    ]\n"
    "}\n"
    "\n"
    "## Evaluation Rules\n"
    "1. Set goal_achieved=true ONLY when you have enough information for a complete,\n"
    "   comprehensive answer to the original task\n"
    "2. The final_answer must directly and fully address the original task\n"
    "3. Use should_replan=true when step results reveal the plan is insufficient,\n"
    "   incorrect, or needs adjustment\n"
    "4. updated_steps replaces ALL remaining pending steps (completed steps are preserved)\n"
    "5. If the step failed but remaining steps can still achieve the goal, continue\n"
    "\n"
    "## Constraints\n"
    "- Output ONLY valid JSON — no markdown fences, no preamble, no commentary\n"
    "- Do not add steps unnecessarily — only replan when genuinely needed\n"
    "- The final_answer should be self-contained and comprehensive\n"
    "- When replanning, ensure updated_steps are actionable and ordered correctly"
)


class ReplannerAgent(BaseAgent):
    """Evaluates step results and decides whether to continue, replan, or finish.

    Pure LLM reasoning — no tools. Returns a JSON decision that the
    orchestrator parses into a ReplanDecision. Single-step execution.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("system_prompt", _REPLANNER_PROMPT)
        super().__init__(**kwargs)

    async def step(self) -> StepResult:
        response = await self.call_llm(tools=None)
        return StepResult(
            thought=self.llm.reasoning_text(response.message),
            response=response.message.content,
        )
