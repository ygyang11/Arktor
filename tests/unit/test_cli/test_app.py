"""Tests for app.py — argument parsing, session resolution, --version."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent_cli import __version__
from agent_cli.app import _build_parser, main
from agent_cli.render.ui import print_exit_reminder
from agent_cli.runtime.session import resolve_session_id
from agent_harness.session.base import SessionState
from agent_harness.session.file_session import FileSession


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "agent_harness.session.file_session._DEFAULT_SESSION_DIR", tmp_path,
    )
    return tmp_path


def _args(*argv: str) -> argparse.Namespace:
    return _build_parser().parse_args(list(argv))


async def _save(dir_: Path, sid: str, updated_at: datetime) -> None:
    backend = FileSession(sid, path=dir_)
    state = SessionState(session_id=sid)
    state.created_at = updated_at
    state.updated_at = updated_at
    await backend.save_state(state)


async def test_continue_picks_latest_by_updated_at(session_dir: Path) -> None:
    now = datetime.now()
    await _save(session_dir, "old-id", now - timedelta(hours=1))
    await _save(session_dir, "new-id", now)
    probe = FileSession("_probe")

    sid = await resolve_session_id(_args("-c"), probe)
    assert sid == "new-id"


async def test_continue_empty_returns_none(
    session_dir: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    probe = FileSession("_probe")
    assert await resolve_session_id(_args("-c"), probe) is None
    assert "no prior session found" in capsys.readouterr().err


async def test_resume_missing_id_returns_none(
    session_dir: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    probe = FileSession("_probe")
    assert await resolve_session_id(_args("-r", "ghost"), probe) is None
    assert "session not found" in capsys.readouterr().err


async def test_resume_existing_id_returns_id(session_dir: Path) -> None:
    await _save(session_dir, "kept", datetime.now())
    probe = FileSession("_probe")

    sid = await resolve_session_id(_args("-r", "kept"), probe)
    assert sid == "kept"


async def test_session_id_invalid_returns_none(
    session_dir: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    probe = FileSession("_probe")
    # argparse accepts anything; only the resolver validates the shape.
    args = _build_parser().parse_args(["-s", "ok-id"])
    args.session_id = "bad/id"
    assert await resolve_session_id(args, probe) is None
    assert "invalid session id" in capsys.readouterr().err


async def test_session_id_existing_returns_none(
    session_dir: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    await _save(session_dir, "taken", datetime.now())
    probe = FileSession("_probe")
    assert await resolve_session_id(_args("-s", "taken"), probe) is None
    assert "already exists" in capsys.readouterr().err


async def test_session_id_new_returns_id(session_dir: Path) -> None:
    probe = FileSession("_probe")
    sid = await resolve_session_id(_args("-s", "brand-new"), probe)
    assert sid == "brand-new"


async def test_no_flags_returns_uuid(session_dir: Path) -> None:
    probe = FileSession("_probe")
    sid = await resolve_session_id(_args(), probe)
    # UUID4 with dashes: 36 chars
    assert len(sid) == 36
    assert sid.count("-") == 4


async def test_continue_ignores_older_corrupted_when_newest_is_healthy(
    session_dir: Path,
) -> None:
    # An older corrupted file must not interfere when the newest is readable.
    import os

    now = datetime.now()
    await _save(session_dir, "newest", now)
    corrupted = session_dir / "older-corrupted.json"
    corrupted.write_text("garbage", encoding="utf-8")
    earlier = (now - timedelta(hours=1)).timestamp()
    os.utime(corrupted, (earlier, earlier))

    probe = FileSession("_probe")
    sid = await resolve_session_id(_args("-c"), probe)
    assert sid == "newest"


async def test_resume_explicit_corrupted_id_passes_resolver(
    session_dir: Path,
) -> None:
    # -r checks file existence only; corruption is caught later by restore_session.
    # The resolver must accept it (return the id), so the caller can attempt
    # to load and surface the corruption error itself.
    (session_dir / "wrecked.json").write_text("garbage", encoding="utf-8")
    probe = FileSession("_probe")

    sid = await resolve_session_id(_args("-r", "wrecked"), probe)
    assert sid == "wrecked"


async def test_continue_uses_state_updated_at_over_mtime(
    session_dir: Path,
) -> None:
    # Diverge mtime from state.updated_at: file A is touched most recently
    # on disk but its JSON says it was updated earlier than file B. The
    # resolver must use state.updated_at, not mtime, to rank "latest".
    import os

    base = datetime.now()
    await _save(session_dir, "older-state", base + timedelta(hours=1))
    await _save(session_dir, "newer-state", base + timedelta(hours=2))
    # Now flip mtimes: make 'older-state' the mtime-newest.
    older_path = session_dir / "older-state.json"
    newer_path = session_dir / "newer-state.json"
    now_ts = datetime.now().timestamp()
    os.utime(older_path, (now_ts + 10, now_ts + 10))
    os.utime(newer_path, (now_ts - 100, now_ts - 100))

    probe = FileSession("_probe")
    sid = await resolve_session_id(_args("-c"), probe)
    assert sid == "newer-state"


async def test_continue_returns_id_from_state_not_filename(
    session_dir: Path,
) -> None:
    # Resolver should parse the file and return state.session_id, so that a
    # renamed file (stem != state.session_id) still resolves to the canonical id.
    now = datetime.now()
    await _save(session_dir, "real-id", now)
    # Rename the file on disk to simulate user-renamed file (or just sanity-check
    # the resolver reads from JSON, not filename).
    (session_dir / "real-id.json").rename(session_dir / "renamed.json")

    probe = FileSession("_probe")
    sid = await resolve_session_id(_args("-c"), probe)
    assert sid == "real-id"


async def test_resume_then_restore_signals_corruption(session_dir: Path) -> None:
    # The resolver+restore_session contract: resolver passes through if file
    # exists, restore_session returns None on unreadable JSON.
    from unittest.mock import MagicMock

    from agent_cli.runtime.session import restore_session

    (session_dir / "wrecked.json").write_text("garbage", encoding="utf-8")
    backend = FileSession("wrecked")
    agent = MagicMock()
    agent.apply_session_state = MagicMock()

    result = await restore_session(agent, backend)
    assert result is None
    agent.apply_session_state.assert_not_called()


async def test_exit_reminder_uses_current_backend_id(session_dir: Path) -> None:
    # Simulate /new or /resume mid-session: startup id was "start-id", but the
    # backend got switched to "switched-id" later. The reminder must reflect
    # the current backend.session_id, not the startup one.
    from io import StringIO

    from rich.console import Console

    await _save(session_dir, "switched-id", datetime.now())
    backend = FileSession("switched-id")  # simulates state after switch_session
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=120)

    await print_exit_reminder(console, backend)
    out = buf.getvalue()
    assert "arktor --resume switched-id" in out
    assert "start-id" not in out  # the (hypothetical) startup id never appears


async def test_exit_reminder_skips_when_no_persisted_file(
    session_dir: Path,
) -> None:
    # Opened REPL, never ran a turn → no .json on disk → reminder suppressed.
    from io import StringIO

    from rich.console import Console

    backend = FileSession("never-saved")  # nothing on disk
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True)

    await print_exit_reminder(console, backend)
    assert buf.getvalue() == ""


async def test_restore_session_loads_and_applies(session_dir: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from agent_cli.runtime.session import restore_session

    sid = "loadable"
    await _save(session_dir, sid, datetime.now())
    raw = (session_dir / f"{sid}.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    data["metadata"] = {"_plan_mode": True}
    (session_dir / f"{sid}.json").write_text(
        json.dumps(data), encoding="utf-8",
    )

    backend = FileSession(sid)
    agent = MagicMock()
    agent.apply_session_state = AsyncMock()
    agent._session_metadata_extras = {}
    agent.context = MagicMock()
    agent.context.context_patches = []

    result = await restore_session(agent, backend)
    assert result is not None
    assert result.session_id == sid
    agent.apply_session_state.assert_awaited_once()


# ── argument parsing (_build_parser) ─────────────────────────────────


def test_mutual_exclusion_continue_and_resume() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-c", "-r", "abc"])


def test_mutual_exclusion_continue_and_session_id() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-c", "-s", "abc"])


def test_mutual_exclusion_resume_and_session_id() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-r", "abc", "-s", "def"])


def test_continue_short_form() -> None:
    assert _build_parser().parse_args(["-c"]).resume_latest is True


def test_continue_long_form() -> None:
    assert _build_parser().parse_args(["--continue"]).resume_latest is True


def test_resume_requires_id() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-r"])


def test_resume_with_id() -> None:
    args = _build_parser().parse_args(["-r", "abc-123"])
    assert args.resume == "abc-123"
    assert args.resume_latest is False
    assert args.session_id is None


def test_session_id_short_alias() -> None:
    assert _build_parser().parse_args(["-s", "abc-123"]).session_id == "abc-123"


def test_session_id_long_alias() -> None:
    assert _build_parser().parse_args(["--session-id", "abc"]).session_id == "abc"


def test_no_flags_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.resume_latest is False
    assert args.resume is None
    assert args.session_id is None
    assert args.prompt is None


# ── -p / --prompt (headless) ─────────────────────────────────────────


def test_prompt_short_form() -> None:
    assert _build_parser().parse_args(["-p", "do x"]).prompt == "do x"


def test_prompt_long_form() -> None:
    assert _build_parser().parse_args(["--prompt", "do x"]).prompt == "do x"


def test_prompt_combines_with_session_id() -> None:
    args = _build_parser().parse_args(["-p", "task", "-s", "sid"])
    assert args.prompt == "task"
    assert args.session_id == "sid"


def test_prompt_combines_with_continue() -> None:
    args = _build_parser().parse_args(["-p", "task", "-c"])
    assert args.prompt == "task"
    assert args.resume_latest is True


def test_prompt_combines_with_resume() -> None:
    args = _build_parser().parse_args(["-p", "task", "-r", "sid"])
    assert args.prompt == "task"
    assert args.resume == "sid"


# ── --version ────────────────────────────────────────────────────────


def test_version_flag_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "arktor" in out
    assert __version__ in out


def test_main_dispatches_headless_empty_prompt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # main() routes -p to run_headless; empty task short-circuits to 2 before
    # any session/agent setup, exercising the dispatch end-to-end.
    rc = main(["-p", "   "])
    assert rc == 2
    assert "non-empty task" in capsys.readouterr().err
