"""Persistent goal state, transitions, ledger, and worker prompts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message, Role

GoalStatus = Literal["active", "paused", "complete", "blocked"]
TerminalStatus = Literal["complete", "blocked"]

MAX_OBJECTIVE_TOKENS = 2_000

_GOAL_START = (
    "[Goal] Begin working toward the persistent objective below. This is the "
    "initial worker turn for this goal. The goal spans multiple turns, so treat "
    "this as continuing work toward the real requested end state rather than a "
    "fresh one-shot task."
)

_GOAL_CONTINUATION = (
    "[Goal continuation] Continue the same persistent objective below from the "
    "current state and prior progress. Determine what remains toward the real "
    "requested end state and pursue it, correcting prior work where necessary. "
    "Do not repeat work already established complete unless current evidence "
    "requires revalidation."
)

_GOAL_OBJECTIVE = "<objective>\n{objective}\n</objective>"

_GOAL_DIRECTIVE = (
    "The previous review identified this outstanding work:\n\n"
    "<directive>\n{directive}\n</directive>\n\n"
    "- Use the review directive to focus the next concrete work, but verify it "
    "against the current state and full objective. It is guidance, not a replacement "
    "for the objective. Do not merely restate progress while meaningful work remains."
)

_GOAL_BEHAVIOR = (
    "- Preserve the full objective across turns. MUST NOT redefine success as "
    "something smaller, easier, narrower, or merely compatible, or easier-to-test "
    "results because it is more likely to pass current requirements.\n"
    "- Work from the real current state: treat the actual files, command output, "
    "and external state as authoritative over anything earlier in the conversation "
    "context - inspect necessary current state before relying on it.\n"
    "- Before reporting completion, derive a concrete checklist from the full "
    "objective and any referenced files, plans, specifications, issues, or "
    "instructions. For every explicit requirement, named artifact, command, test, "
    "gate, invariant, and deliverable, inspect authoritative current-state evidence "
    "that proves it.\n"
    "- Report completion only when that audit leaves no unmet requirement or "
    "missing or weak evidence. Intent, partial progress, prior memory, and a "
    "plausible final answer are not proof; otherwise continue working and state the "
    "remaining gap."
)


@dataclass
class GoalState:
    objective: str
    status: GoalStatus = "active"
    reason: str = ""
    turns: int = 0
    accumulated_s: float = 0.0
    accumulated_tokens: int = 0
    _start: float | None = None
    _process_token_baseline: int = 0

    def elapsed_s(self) -> int:
        live = (time.monotonic() - self._start) if self._start is not None else 0.0
        return int(self.accumulated_s + live)

    def tokens_used(self, process_total_tokens: int) -> int:
        if self.status != "active":
            return self.accumulated_tokens
        live = max(0, process_total_tokens - self._process_token_baseline)
        return self.accumulated_tokens + live


_goals: dict[int, GoalState] = {}


def get_state(agent: BaseAgent) -> GoalState | None:
    return _goals.get(id(agent))


def is_active(agent: BaseAgent) -> bool:
    g = get_state(agent)
    return g is not None and g.status == "active"


def has_live_goal(agent: BaseAgent) -> bool:
    """Return whether an active or paused goal exists."""
    g = get_state(agent)
    return g is not None and g.status in ("active", "paused")


def begin(agent: BaseAgent, objective: str) -> GoalState:
    g = GoalState(
        objective=objective,
        status="active",
        _start=time.monotonic(),
        _process_token_baseline=agent.context.usage_meter.total.total_tokens,
    )
    _goals[id(agent)] = g
    _persist(agent, g)
    return g


def pause(agent: BaseAgent, reason: str = "") -> GoalState | None:
    g = get_state(agent)
    if g is None or g.status != "active":
        return None
    _fold_elapsed(g)
    _fold_tokens(agent, g)
    g.status = "paused"
    if reason:
        g.reason = reason
    _persist(agent, g)
    return g


def resume(agent: BaseAgent) -> GoalState | None:
    g = get_state(agent)
    if g is None or g.status != "paused":
        return None
    g.status = "active"
    g._start = time.monotonic()
    g._process_token_baseline = agent.context.usage_meter.total.total_tokens
    _persist(agent, g)
    return g


def finish(
    agent: BaseAgent,
    status: TerminalStatus,
    reason: str,
) -> GoalState | None:
    g = get_state(agent)
    if g is None or g.status != "active":
        return None
    _fold_elapsed(g)
    _fold_tokens(agent, g)
    g.status = status
    g.reason = reason
    _persist(agent, g)
    return g


def clear(agent: BaseAgent) -> None:
    _goals.pop(id(agent), None)
    agent._session_metadata_extras.pop("_goal", None)


def record_completed_turn(agent: BaseAgent) -> GoalState | None:
    g = get_state(agent)
    if g is None or g.status != "active":
        return None
    g.turns += 1
    _persist(agent, g)
    return g


def is_goal_continuation_message(message: Message) -> bool:
    return (
        message.role == Role.USER
        and bool((message.metadata or {}).get("is_goal_continuation"))
    )


def _render_goal_prompt(
    objective: str,
    *,
    continuation: bool,
    directive: str | None = None,
) -> str:
    parts = [
        _GOAL_CONTINUATION if continuation else _GOAL_START,
        _GOAL_OBJECTIVE.format(objective=objective),
    ]
    if directive is not None:
        parts.append(_GOAL_DIRECTIVE.format(directive=directive))
    parts.append(_GOAL_BEHAVIOR)
    return "\n\n".join(parts)


def make_start_input(objective: str) -> str:
    return _render_goal_prompt(objective, continuation=False)


def make_resume_message(objective: str) -> Message:
    return Message.user(
        _render_goal_prompt(objective, continuation=True),
        metadata={"is_goal_continuation": True},
    )


def make_continuation_message(
    agent: BaseAgent,
    reason: str,
    directive: str,
) -> Message | None:
    g = get_state(agent)
    if g is None or g.status != "active":
        return None
    g.reason = reason
    _persist(agent, g)
    return Message.user(
        _render_goal_prompt(
            g.objective,
            continuation=True,
            directive=directive,
        ),
        metadata={"is_goal_continuation": True},
    )


def restore(agent: BaseAgent, raw: object) -> GoalState | None:
    clear(agent)
    if not isinstance(raw, dict):
        return None

    objective = raw.get("objective")
    status = raw.get("status")
    if (
        not isinstance(objective, str)
        or not objective.strip()
        or status not in ("active", "paused")
    ):
        return None

    reason = raw.get("reason")
    turns = raw.get("turns")
    accumulated_s = raw.get("accumulated_s")
    accumulated_tokens = raw.get("accumulated_tokens")
    g = GoalState(
        objective=objective,
        status="paused",
        reason=reason if isinstance(reason, str) else "",
        turns=turns if type(turns) is int and turns >= 0 else 0,
        accumulated_s=(
            float(accumulated_s)
            if isinstance(accumulated_s, (int, float))
            and not isinstance(accumulated_s, bool)
            and accumulated_s >= 0
            else 0.0
        ),
        accumulated_tokens=(
            accumulated_tokens
            if type(accumulated_tokens) is int and accumulated_tokens >= 0
            else 0
        ),
    )
    _goals[id(agent)] = g
    _persist(agent, g)
    return g


def _fold_elapsed(g: GoalState) -> None:
    if g._start is not None:
        g.accumulated_s += time.monotonic() - g._start
        g._start = None


def _fold_tokens(agent: BaseAgent, g: GoalState) -> None:
    total = agent.context.usage_meter.total.total_tokens
    g.accumulated_tokens = g.tokens_used(total)
    g._process_token_baseline = total


def _persist(agent: BaseAgent, g: GoalState) -> None:
    agent._session_metadata_extras["_goal"] = {
        "objective": g.objective,
        "status": g.status,
        "reason": g.reason,
        "turns": g.turns,
        "accumulated_s": g.elapsed_s(),
        "accumulated_tokens": g.tokens_used(
            agent.context.usage_meter.total.total_tokens
        ),
    }
