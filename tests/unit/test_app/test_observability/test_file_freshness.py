"""Tests for the file_freshness OCC + drift-tracking primitive."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

from agent_app.observability import file_freshness as ff
from agent_app.observability.file_freshness import (
    FileSignature,
    Verdict,
    _entry_from_raw,
    _key,
    _sig_from_raw,
    mark_read,
    mark_seen,
    poll_drift,
    restore_state,
    snapshot_state,
    stale_guard,
)
from agent_harness.agent.base import BaseAgent
from agent_harness.context.variables import ContextVariables, Scope
from agent_harness.session.base import SessionState


class _StubContext:
    def __init__(self) -> None:
        self.variables = ContextVariables()


class _StubAgent:
    def __init__(self) -> None:
        self.context = _StubContext()


def _make_agent() -> BaseAgent:
    return cast(BaseAgent, _StubAgent())


# ── Edit stale-guard (read signature) ──


def test_mark_read_then_guard_returns_fresh(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    assert stale_guard(agent, f) is Verdict.FRESH


def test_guard_returns_unknown_without_mark(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert stale_guard(agent, f) is Verdict.UNKNOWN


def test_guard_returns_stale_after_external_modify(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    future = time.time() + 10
    os.utime(f, (future, future))
    assert stale_guard(agent, f) is Verdict.STALE


def test_guard_returns_stale_after_size_change(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    f.write_text("hello world")
    assert stale_guard(agent, f) is Verdict.STALE


def test_guard_returns_stale_when_file_deleted(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    os.unlink(f)
    assert stale_guard(agent, f) is Verdict.STALE


def test_mark_read_on_missing_file_clears_entry(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    assert agent.context.variables.get(_key(f)) is not None
    os.unlink(f)
    mark_read(agent, f)
    assert agent.context.variables.get(_key(f)) is None


# ── Drift reminder (seen signature) ──


def test_poll_drift_returns_only_changed_paths(tmp_path: Path) -> None:
    agent = _make_agent()
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for p in (a, b, c):
        p.write_text("x")
        mark_read(agent, p)
    future = time.time() + 10
    os.utime(b, (future, future))
    drifts = poll_drift(agent)
    assert len(drifts) == 1
    assert drifts[0].path == str(b.resolve())


def test_mark_read_clears_drift(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    assert poll_drift(agent) == []


# ── The core split: mark_seen updates seen, not read ──


def test_mark_seen_clears_drift_but_keeps_guard_stale(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    f.write_text("changed on disk")
    assert len(poll_drift(agent)) == 1
    assert stale_guard(agent, f) is Verdict.STALE

    mark_seen(agent, f)
    assert poll_drift(agent) == []
    assert stale_guard(agent, f) is Verdict.STALE


def test_mark_seen_on_delete_clears_drift(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    os.unlink(f)
    assert len(poll_drift(agent)) == 1
    mark_seen(agent, f)
    assert poll_drift(agent) == []


def test_mark_seen_untracked_is_noop(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_seen(agent, f)
    assert agent.context.variables.get(_key(f)) is None


def test_drift_reports_again_after_new_change(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    f.write_text("v1")
    assert len(poll_drift(agent)) == 1
    mark_seen(agent, f)
    assert poll_drift(agent) == []
    f.write_text("v2-longer")
    assert len(poll_drift(agent)) == 1


# ── Snapshot / restore (turn rollback) ──


def test_restore_reverts_mark_seen(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    f.write_text("changed")
    snap = snapshot_state(agent)

    mark_seen(agent, f)
    assert poll_drift(agent) == []

    restore_state(agent, snap)
    assert len(poll_drift(agent)) == 1
    assert stale_guard(agent, f) is Verdict.STALE


def test_restore_removes_entries_added_after_snapshot(tmp_path: Path) -> None:
    agent = _make_agent()
    a = tmp_path / "a.txt"
    a.write_text("x")
    mark_read(agent, a)
    snap = snapshot_state(agent)

    b = tmp_path / "b.txt"
    b.write_text("x")
    mark_read(agent, b)

    restore_state(agent, snap)
    assert agent.context.variables.get(_key(b)) is None
    assert agent.context.variables.get(_key(a)) is not None


# ── Storage shape + legacy migration ──


def test_entry_stored_with_read_and_seen(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    raw = agent.context.variables.get(_key(f))
    assert isinstance(raw, dict)
    assert set(raw.keys()) == {"read", "seen"}
    assert set(raw["read"].keys()) == {"mtime_ns", "size"}
    assert raw["read"] == raw["seen"]


def test_signature_uses_mtime_ns(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("abc")
    mark_read(agent, f)
    raw = agent.context.variables.get(_key(f))
    assert isinstance(raw, dict)
    base_ns = int(raw["read"]["mtime_ns"])
    os.utime(f, ns=(base_ns + 1_000_000, base_ns + 1_000_000))
    assert stale_guard(agent, f) is Verdict.STALE


def test_sig_from_raw_tolerates_malformed_dict() -> None:
    assert _sig_from_raw(None) is None
    assert _sig_from_raw("not a dict") is None
    assert _sig_from_raw({}) is None
    assert _sig_from_raw({"mtime_ns": 1}) is None
    assert _sig_from_raw({"size": 1}) is None
    assert _sig_from_raw({"mtime_ns": "bad", "size": 1}) is None
    assert _sig_from_raw({"mtime_ns": 1, "size": 2}) == FileSignature(mtime_ns=1, size=2)


def test_entry_from_raw_handles_both_shapes() -> None:
    assert _entry_from_raw("nope") == (None, None)
    legacy = {"mtime_ns": 1, "size": 2}
    read, seen = _entry_from_raw(legacy)
    assert read == FileSignature(1, 2)
    assert seen == FileSignature(1, 2)
    new = {"read": {"mtime_ns": 1, "size": 2}, "seen": None}
    read, seen = _entry_from_raw(new)
    assert read == FileSignature(1, 2)
    assert seen is None


def test_legacy_flat_entry_is_honored(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    sig = ff._stat_signature(f)
    assert sig is not None
    agent.context.variables.set(
        _key(f), {"mtime_ns": sig.mtime_ns, "size": sig.size}, scope=Scope.AGENT,
    )
    assert stale_guard(agent, f) is Verdict.FRESH
    assert poll_drift(agent) == []
    f.write_text("hello world")
    assert stale_guard(agent, f) is Verdict.STALE
    assert len(poll_drift(agent)) == 1


def test_signature_survives_session_round_trip(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    mark_read(agent, f)
    state = SessionState(
        session_id="t",
        variables_agent=agent.context.variables.get_all(Scope.AGENT),
    )
    json_str = state.model_dump_json()
    restored = SessionState.model_validate_json(json_str)

    new_agent = _make_agent()
    for k, v in restored.variables_agent.items():
        new_agent.context.variables.set(k, v, scope=Scope.AGENT)
    assert stale_guard(new_agent, f) is Verdict.FRESH


# ── LRU eviction ──


def test_mark_read_evicts_lru_when_over_max_tracked(
    tmp_path: Path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(ff, "_MAX_TRACKED", 3)
    agent = _make_agent()
    paths = [tmp_path / f"f{i}.txt" for i in range(5)]
    for p in paths:
        p.write_text("x")
        mark_read(agent, p)
    kept = [
        k for k in agent.context.variables.get_all(Scope.AGENT)
        if k.startswith("_fs.sig:")
    ]
    assert len(kept) == 3
    for old in paths[:2]:
        assert agent.context.variables.get(_key(old)) is None
    for fresh in paths[2:]:
        assert agent.context.variables.get(_key(fresh)) is not None


def test_mark_read_refreshes_lru_order_on_re_record(
    tmp_path: Path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(ff, "_MAX_TRACKED", 2)
    agent = _make_agent()
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for p in (a, b, c):
        p.write_text("x")

    mark_read(agent, a)
    mark_read(agent, b)
    mark_read(agent, a)
    mark_read(agent, c)

    assert agent.context.variables.get(_key(a)) is not None
    assert agent.context.variables.get(_key(c)) is not None
    assert agent.context.variables.get(_key(b)) is None


def test_eviction_order_survives_session_round_trip(
    tmp_path: Path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(ff, "_MAX_TRACKED", 5)
    agent = _make_agent()
    paths = [tmp_path / f"f{i}.txt" for i in range(5)]
    for p in paths:
        p.write_text("x")
        mark_read(agent, p)

    state = SessionState(
        session_id="t",
        variables_agent=agent.context.variables.get_all(Scope.AGENT),
    )
    restored = SessionState.model_validate_json(state.model_dump_json())

    new_agent = _make_agent()
    for k, v in restored.variables_agent.items():
        new_agent.context.variables.set(k, v, scope=Scope.AGENT)

    extra = tmp_path / "extra.txt"
    extra.write_text("x")
    mark_read(new_agent, extra)

    assert new_agent.context.variables.get(_key(paths[0])) is None
    for p in paths[1:]:
        assert new_agent.context.variables.get(_key(p)) is not None
    assert new_agent.context.variables.get(_key(extra)) is not None
