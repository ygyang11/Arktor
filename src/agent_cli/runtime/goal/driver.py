from __future__ import annotations

from dataclasses import dataclass

from agent_cli.runtime.goal import mode as goal_mode
from agent_cli.runtime.goal.evaluator import GoalVerdictStatus, evaluate
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message


@dataclass(frozen=True, slots=True)
class GoalDecision:
    status: GoalVerdictStatus
    reason: str
    continuation: Message | None = None


async def decide(agent: BaseAgent) -> GoalDecision | None:
    goal = goal_mode.get_state(agent)
    if goal is None or goal.status != "active":
        return None
    verdict = await evaluate(
        agent,
        goal.objective,
        turns=goal.turns,
        elapsed_s=goal.elapsed_s(),
        tokens=goal.tokens_used(
            agent.context.usage_meter.total.total_tokens
        ),
    )
    current = goal_mode.get_state(agent)
    if current is not goal or current.status != "active":
        return None

    if verdict.status != "continue":
        if goal_mode.finish(agent, verdict.status, verdict.reason) is None:
            return None
        return GoalDecision(verdict.status, verdict.reason)

    continuation = goal_mode.make_continuation_message(
        agent,
        verdict.reason,
        verdict.directive,
    )
    if continuation is None:
        return None
    return GoalDecision("continue", verdict.reason, continuation)
