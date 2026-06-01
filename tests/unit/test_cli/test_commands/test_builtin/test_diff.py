from unittest.mock import MagicMock

from agent_cli.commands.builtin import diff as diff_mod
from agent_cli.commands.builtin.diff import CMD

from ..conftest import render_output


def _ctx() -> MagicMock:
    return MagicMock()


def _stub_git(monkeypatch, *, status: str, diff: str, staged: str = "", rc: int = 0) -> None:
    async def fake(*args: str, timeout: float = 10.0) -> tuple[int, str, str]:
        if args[:1] == ("diff",) and "--cached" in args:
            return (rc, staged, "")
        if args[:1] == ("diff",):
            return (rc, diff, "")
        if args[:1] == ("status",):
            return (rc, status, "")
        return (rc, "", "")

    monkeypatch.setattr(diff_mod, "_run_git", fake)


async def test_diff_clean_tree(monkeypatch) -> None:
    _stub_git(monkeypatch, status="", diff="")
    result = await CMD.handler(_ctx(), "")
    assert "Working tree clean" in render_output(result.output)


async def test_diff_renders_panel_with_status_and_diff(monkeypatch) -> None:
    _stub_git(
        monkeypatch,
        status=" M src/a.py\n?? new.py",
        diff="diff --git a/src/a.py b/src/a.py\n+added line",
    )
    out = render_output((await CMD.handler(_ctx(), "")).output)
    assert "Uncommitted changes" in out
    assert "src/a.py" in out
    assert "added line" in out


async def test_diff_includes_staged_changes(monkeypatch) -> None:
    _stub_git(
        monkeypatch,
        status="M  src/b.py",
        diff="",
        staged="diff --git a/src/b.py b/src/b.py\n+staged change",
    )
    out = render_output((await CMD.handler(_ctx(), "")).output)
    assert "staged change" in out


async def test_diff_git_failure_returns_err(monkeypatch) -> None:
    async def fake(*args: str, timeout: float = 10.0) -> tuple[int, str, str]:
        return (128, "", "not a git repo")
    monkeypatch.setattr(diff_mod, "_run_git", fake)
    out = render_output((await CMD.handler(_ctx(), "")).output)
    assert "git diff failed" in out
