"""Runtime-layer frozen snapshots for UI consumption.

All UI surfaces (/status, /context, /usage, status bar) read frozen
views from this module instead of reaching into agent.context internals.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_cli.runtime import background
from agent_harness.agent.base import BaseAgent
from agent_harness.llm.types import Usage
from agent_harness.memory.short_term import CallSnapshot


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    model: str
    approval_mode: str
    tool_count: int
    skill_count: int
    message_count: int
    input_tokens: int | None
    max_tokens: int
    todo_count: int
    bg_running: int
    bg_total: int


def collect(agent: BaseAgent) -> StatusSnapshot:
    stm = agent.context.short_term_memory
    bg_all = background.get_all(agent)
    return StatusSnapshot(
        model=agent.llm.model_name,
        approval_mode=agent._approval.mode,
        tool_count=len(agent.tools),
        skill_count=_skill_count(agent),
        message_count=len(stm._messages),
        input_tokens=stm.displayed_input_tokens,
        max_tokens=stm.max_tokens,
        todo_count=_todo_count(agent),
        bg_running=sum(1 for t in bg_all if t.status == "running"),
        bg_total=len(bg_all),
    )


@dataclass(frozen=True, slots=True)
class WindowView:
    last_call: CallSnapshot | None
    max_tokens: int
    displayed_input_tokens: int | None


def collect_window(agent: BaseAgent) -> WindowView:
    stm = agent.context.short_term_memory
    return WindowView(
        last_call=stm.last_call,
        max_tokens=stm.max_tokens,
        displayed_input_tokens=stm.displayed_input_tokens,
    )


@dataclass(frozen=True)
class BucketView:
    usage: Usage
    calls: int


@dataclass(frozen=True)
class UsageView:
    total: Usage
    by_model: dict[str, BucketView]
    by_source: dict[str, BucketView]
    call_count: int


def collect_usage(agent: BaseAgent) -> UsageView:
    m = agent.context.usage_meter
    return UsageView(
        total=m.total,
        by_model={k: BucketView(b.usage, b.calls) for k, b in m.by_model.items()},
        by_source={k: BucketView(b.usage, b.calls) for k, b in m.by_source.items()},
        call_count=m.call_count,
    )


def _skill_count(agent: BaseAgent) -> int:
    for tool in agent.tools:
        if tool.name == "skill_tool":
            loader = getattr(tool, "loader", None)
            if loader is None:
                return 0
            try:
                return len(loader.list_names())
            except Exception:
                return 0
    return 0


def _todo_count(agent: BaseAgent) -> int:
    if not agent.tool_registry.has("todo_write"):
        return 0
    tool = agent.tool_registry.get("todo_write")
    todos = getattr(tool, "_todos", None) or []
    return len(todos)
