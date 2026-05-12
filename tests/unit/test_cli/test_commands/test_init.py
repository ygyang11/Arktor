from unittest.mock import MagicMock

from agent_cli.commands.builtin.init import CMD


def _ctx() -> MagicMock:
    return MagicMock()


async def test_init_new_when_agents_md_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    result = await CMD.handler(_ctx(), "")
    assert result.agent_input is not None
    assert "Generate a file named AGENTS.md" in result.agent_input
    assert "Focus:" not in result.agent_input


async def test_init_update_when_agents_md_exists(monkeypatch, tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Repository Guidelines\n")
    monkeypatch.chdir(tmp_path)
    result = await CMD.handler(_ctx(), "")
    assert result.agent_input is not None
    assert "AGENTS.md already exists" in result.agent_input


async def test_init_with_focus_argument(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    result = await CMD.handler(_ctx(), "testing strategy")
    assert result.agent_input is not None
    assert "Focus: testing strategy" in result.agent_input
