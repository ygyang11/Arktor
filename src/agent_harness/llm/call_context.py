"""Per-call identity for transport metadata and interaction recording."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

CallKind = Literal["agent", "compact", "goal_eval"]
ModelSource = Literal["main", "sub"]
_REQUEST_HEADER = "X-Arktor-Context"


@dataclass(frozen=True)
class AgentIdentity:
    name: str
    parent: str | None
    type: str


@dataclass(frozen=True)
class ModelCallContext:
    agent: AgentIdentity
    seq: int
    kind: CallKind
    llm: ModelSource

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": {
                "name": self.agent.name,
                "parent": self.agent.parent,
                "type": self.agent.type,
            },
            "call": {
                "seq": self.seq,
                "kind": self.kind,
                "llm": self.llm,
            },
        }


def with_request_context(
    kwargs: dict[str, Any],
    context: ModelCallContext,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Copy request kwargs and optionally add the context header."""
    request = dict(kwargs)
    if not enabled:
        return request

    headers = dict(request.get("extra_headers") or {})
    for name in list(headers):
        if name.lower() == _REQUEST_HEADER.lower():
            del headers[name]
    headers[_REQUEST_HEADER] = json.dumps(
        context.to_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    request["extra_headers"] = headers
    return request
