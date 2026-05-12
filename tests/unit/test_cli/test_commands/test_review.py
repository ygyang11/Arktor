from unittest.mock import MagicMock

from agent_cli.commands.builtin import review as review_mod
from agent_cli.commands.builtin.review import CMD

from .conftest import render_output


def _ctx() -> MagicMock:
    return MagicMock()


def _stub_git(monkeypatch, *, status: str, rc: int = 0) -> None:
    async def fake(*args: str, timeout: float = 10.0) -> tuple[int, str, str]:
        return (rc, status, "")
    monkeypatch.setattr(review_mod, "_run_git", fake)


async def test_review_with_explicit_target_skips_git(monkeypatch) -> None:
    called: list[str] = []

    async def fake(*args: str, timeout: float = 10.0) -> tuple[int, str, str]:
        called.append(args[0])
        return (0, "", "")
    monkeypatch.setattr(review_mod, "_run_git", fake)

    result = await CMD.handler(_ctx(), "src/agent_cli/commands")
    assert called == []
    assert result.agent_input is not None
    assert "Review focus: src/agent_cli/commands" in result.agent_input


async def test_review_no_args_uses_default_target_when_dirty(monkeypatch) -> None:
    _stub_git(monkeypatch, status=" M foo.py")
    result = await CMD.handler(_ctx(), "")
    assert result.agent_input is not None
    assert "Review focus: the uncommitted changes" in result.agent_input


async def test_review_no_args_clean_tree_returns_soft(monkeypatch) -> None:
    _stub_git(monkeypatch, status="")
    out = render_output((await CMD.handler(_ctx(), "")).output)
    assert "Working tree clean" in out


async def test_review_no_args_not_a_git_repo_returns_err(monkeypatch) -> None:
    _stub_git(monkeypatch, status="", rc=128)
    out = render_output((await CMD.handler(_ctx(), "")).output)
    assert "Not a git repo" in out
