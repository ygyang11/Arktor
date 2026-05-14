"""Tests for the file_freshness OCC primitive."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

from agent_app.observability import file_freshness as ff
from agent_app.observability.file_freshness import (
    FileSignature,
    Verdict,
    _from_raw,
    _key,
    check_freshness,
    poll_dirty,
    record_signature,
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


def test_record_then_check_returns_fresh(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    assert check_freshness(agent, f) is Verdict.FRESH


def test_check_returns_unknown_without_record(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert check_freshness(agent, f) is Verdict.UNKNOWN


def test_check_returns_stale_after_external_modify(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    future = time.time() + 10
    os.utime(f, (future, future))
    assert check_freshness(agent, f) is Verdict.STALE


def test_check_returns_stale_after_size_change(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    f.write_text("hello world")
    assert check_freshness(agent, f) is Verdict.STALE


def test_check_returns_stale_when_file_deleted(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    os.unlink(f)
    assert check_freshness(agent, f) is Verdict.STALE


def test_record_on_missing_file_clears_sig(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    assert agent.context.variables.get(_key(f)) is not None
    os.unlink(f)
    record_signature(agent, f)
    assert agent.context.variables.get(_key(f)) is None


def test_poll_dirty_returns_only_changed_paths(tmp_path: Path) -> None:
    agent = _make_agent()
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for p in (a, b, c):
        p.write_text("x")
        record_signature(agent, p)
    future = time.time() + 10
    os.utime(b, (future, future))
    drifts = poll_dirty(agent)
    assert len(drifts) == 1
    assert drifts[0].path == str(b.resolve())


def test_signature_uses_mtime_ns(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("abc")
    record_signature(agent, f)
    raw = agent.context.variables.get(_key(f))
    assert isinstance(raw, dict)
    base_ns = int(raw["mtime_ns"])
    os.utime(f, ns=(base_ns + 1_000_000, base_ns + 1_000_000))
    assert check_freshness(agent, f) is Verdict.STALE


def test_signature_stored_as_plain_dict(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    raw = agent.context.variables.get(_key(f))
    assert isinstance(raw, dict)
    assert not isinstance(raw, FileSignature)
    assert set(raw.keys()) == {"mtime_ns", "size"}


def test_signature_survives_session_round_trip(tmp_path: Path) -> None:
    agent = _make_agent()
    f = tmp_path / "a.txt"
    f.write_text("hello")
    record_signature(agent, f)
    state = SessionState(
        session_id="t",
        variables_agent=agent.context.variables.get_all(Scope.AGENT),
    )
    json_str = state.model_dump_json()
    restored = SessionState.model_validate_json(json_str)

    new_agent = _make_agent()
    for k, v in restored.variables_agent.items():
        new_agent.context.variables.set(k, v, scope=Scope.AGENT)
    assert check_freshness(new_agent, f) is Verdict.FRESH


def test_from_raw_tolerates_malformed_dict() -> None:
    assert _from_raw(None) is None
    assert _from_raw("not a dict") is None
    assert _from_raw({}) is None
    assert _from_raw({"mtime_ns": 1}) is None
    assert _from_raw({"size": 1}) is None
    assert _from_raw({"mtime_ns": "bad", "size": 1}) is None
    assert _from_raw({"mtime_ns": 1, "size": "bad"}) is None
    assert _from_raw({"mtime_ns": 1, "size": 2}) == FileSignature(mtime_ns=1, size=2)


def test_record_evicts_lru_when_over_max_tracked(
    tmp_path: Path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(ff, "_MAX_TRACKED", 3)
    agent = _make_agent()
    paths = [tmp_path / f"f{i}.txt" for i in range(5)]
    for p in paths:
        p.write_text("x")
        record_signature(agent, p)
    kept = [
        k for k in agent.context.variables.get_all(Scope.AGENT)
        if k.startswith("_fs.sig:")
    ]
    assert len(kept) == 3
    for old in paths[:2]:
        assert agent.context.variables.get(_key(old)) is None
    for fresh in paths[2:]:
        assert agent.context.variables.get(_key(fresh)) is not None


def test_record_refreshes_lru_order_on_re_record(
    tmp_path: Path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(ff, "_MAX_TRACKED", 2)
    agent = _make_agent()
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for p in (a, b, c):
        p.write_text("x")

    record_signature(agent, a)
    record_signature(agent, b)
    record_signature(agent, a)
    record_signature(agent, c)

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
        record_signature(agent, p)

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
    record_signature(new_agent, extra)

    assert new_agent.context.variables.get(_key(paths[0])) is None
    for p in paths[1:]:
        assert new_agent.context.variables.get(_key(p)) is not None
    assert new_agent.context.variables.get(_key(extra)) is not None
