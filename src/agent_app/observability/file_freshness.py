"""Optimistic concurrency control + external-drift tracking for filesystem tools."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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
    recorded: FileSignature | None
    current: FileSignature | None


def _key(path: str | Path) -> str:
    return f"{_KEY_PREFIX}{Path(path).resolve()}"


def _stat_signature(path: str | Path) -> FileSignature | None:
    try:
        st = os.stat(path)
        return FileSignature(mtime_ns=st.st_mtime_ns, size=st.st_size)
    except OSError:
        return None


def _sig_to_raw(sig: FileSignature | None) -> dict[str, int] | None:
    if sig is None:
        return None
    return {"mtime_ns": sig.mtime_ns, "size": sig.size}


def _sig_from_raw(raw: object) -> FileSignature | None:
    if not isinstance(raw, dict):
        return None
    try:
        return FileSignature(mtime_ns=int(raw["mtime_ns"]), size=int(raw["size"]))
    except (KeyError, TypeError, ValueError):
        return None


def _entry_from_raw(raw: object) -> tuple[FileSignature | None, FileSignature | None]:
    if not isinstance(raw, dict):
        return None, None
    if "read" in raw or "seen" in raw:
        return _sig_from_raw(raw.get("read")), _sig_from_raw(raw.get("seen"))
    legacy = _sig_from_raw(raw)
    return legacy, legacy


def _write_entry(
    agent: BaseAgent,
    path: str | Path,
    read: FileSignature | None,
    seen: FileSignature | None,
) -> None:
    key = _key(path)
    agent.context.variables.delete(key)
    agent.context.variables.set(
        key, {"read": _sig_to_raw(read), "seen": _sig_to_raw(seen)}, scope=Scope.AGENT,
    )


def mark_read(agent: BaseAgent, path: str | Path) -> None:
    sig = _stat_signature(path)
    if sig is None:
        agent.context.variables.delete(_key(path))
        return
    _write_entry(agent, path, sig, sig)
    _evict_overflow(agent)


def mark_seen(agent: BaseAgent, path: str | Path) -> None:
    raw = agent.context.variables.get(_key(path))
    if raw is None:
        return
    read, _ = _entry_from_raw(raw)
    _write_entry(agent, path, read, _stat_signature(path))


def stale_guard(agent: BaseAgent, path: str | Path) -> Verdict:
    read, _ = _entry_from_raw(agent.context.variables.get(_key(path)))
    if read is None:
        return Verdict.UNKNOWN
    current = _stat_signature(path)
    if current is None:
        return Verdict.STALE
    return Verdict.FRESH if current == read else Verdict.STALE


def poll_drift(agent: BaseAgent) -> list[Drift]:
    drifts: list[Drift] = []
    for key, raw in agent.context.variables.get_all(Scope.AGENT).items():
        if not key.startswith(_KEY_PREFIX):
            continue
        _, seen = _entry_from_raw(raw)
        path = key[len(_KEY_PREFIX):]
        current = _stat_signature(path)
        if current == seen:
            continue
        drifts.append(Drift(path=path, recorded=seen, current=current))
    return drifts


def snapshot_state(agent: BaseAgent) -> dict[str, Any]:
    return {
        k: v
        for k, v in agent.context.variables.get_all(Scope.AGENT).items()
        if k.startswith(_KEY_PREFIX)
    }


def restore_state(agent: BaseAgent, state: dict[str, Any]) -> None:
    for k in list(agent.context.variables.get_all(Scope.AGENT)):
        if k.startswith(_KEY_PREFIX):
            agent.context.variables.delete(k)
    for k, v in state.items():
        agent.context.variables.set(k, v, scope=Scope.AGENT)


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
