"""Optimistic concurrency control for filesystem tools."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agent_harness.agent.base import BaseAgent
from agent_harness.context.variables import Scope

_KEY_PREFIX = "_fs.sig:"
_MAX_TRACKED = 100


@dataclass(frozen=True, slots=True)
class FileSignature:
    mtime_ns: int
    size: int


class Verdict(str, Enum):
    UNKNOWN = "unknown"
    FRESH = "fresh"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class Drift:
    path: str
    recorded: FileSignature
    current: FileSignature | None


def _key(path: str | Path) -> str:
    return f"{_KEY_PREFIX}{Path(path).resolve()}"


def _stat_signature(path: str | Path) -> FileSignature | None:
    try:
        st = os.stat(path)
        return FileSignature(mtime_ns=st.st_mtime_ns, size=st.st_size)
    except OSError:
        return None


def _from_raw(raw: object) -> FileSignature | None:
    if not isinstance(raw, dict):
        return None
    try:
        return FileSignature(mtime_ns=int(raw["mtime_ns"]), size=int(raw["size"]))
    except (KeyError, TypeError, ValueError):
        return None


def record_signature(agent: BaseAgent, path: str | Path) -> None:
    key = _key(path)
    sig = _stat_signature(path)
    if sig is None:
        agent.context.variables.delete(key)
        return
    agent.context.variables.delete(key)
    agent.context.variables.set(
        key, {"mtime_ns": sig.mtime_ns, "size": sig.size}, scope=Scope.AGENT,
    )
    _evict_overflow(agent)


def _evict_overflow(agent: BaseAgent) -> None:
    fs_keys = [
        k for k in agent.context.variables.get_all(Scope.AGENT)
        if k.startswith(_KEY_PREFIX)
    ]
    overflow = len(fs_keys) - _MAX_TRACKED
    if overflow <= 0:
        return
    for k in fs_keys[:overflow]:
        agent.context.variables.delete(k)


def check_freshness(agent: BaseAgent, path: str | Path) -> Verdict:
    recorded = _from_raw(agent.context.variables.get(_key(path)))
    if recorded is None:
        return Verdict.UNKNOWN
    current = _stat_signature(path)
    if current is None:
        return Verdict.STALE
    return Verdict.FRESH if current == recorded else Verdict.STALE


def poll_dirty(agent: BaseAgent) -> list[Drift]:
    drifts: list[Drift] = []
    for key, raw in agent.context.variables.get_all(Scope.AGENT).items():
        if not key.startswith(_KEY_PREFIX):
            continue
        recorded = _from_raw(raw)
        if recorded is None:
            continue
        path = key[len(_KEY_PREFIX):]
        current = _stat_signature(path)
        if current == recorded:
            continue
        drifts.append(Drift(path=path, recorded=recorded, current=current))
    return drifts
