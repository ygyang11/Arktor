import io
from unittest.mock import MagicMock

from prompt_toolkit.formatted_text import HTML
from rich.console import Console

from agent_cli.render.ui import (
    _BANNER_LINES,
    _BANNER_WIDTH,
    _fmt,
    make_status_bar_text,
    render_welcome,
)


def test_banner_lines_all_width_58() -> None:
    for line in _BANNER_LINES:
        assert len(line) == _BANNER_WIDTH, f"len={len(line)}: {line!r}"


def test_render_welcome_emits_tagline_and_meta() -> None:
    buf = io.StringIO()
    console = Console(file=buf, color_system=None, width=100)
    render_welcome(
        console,
        version="0.4.10",
        model="gpt-5",
        cwd="/home/user/proj",
        config_source="/home/user/.agent-harness/config.yaml",
    )
    out = buf.getvalue()
    assert "Agents, harnessed." in out
    assert "v0.4.10" in out
    assert "model" in out and "gpt-5" in out
    assert "cwd" in out and "/home/user/proj" in out
    assert "config" in out and ".agent-harness/config.yaml" in out
    assert "session" in out
    assert "fresh" in out
    assert "/resume to restore" in out
    assert "commands" in out and "files" in out and "shell" in out


def test_fmt_thresholds() -> None:
    assert _fmt(500) == "500"
    assert _fmt(1500) == "1k"
    assert _fmt(42_000) == "42k"
    assert _fmt(1_500_000) == "1.5M"


def _stub_agent(model: str, input_tokens: int | None, max_tokens: int) -> MagicMock:
    agent = MagicMock()
    agent.llm.model_name = model
    agent._approval.mode = "auto"
    stm = MagicMock()
    stm._messages = []
    stm.displayed_input_tokens = input_tokens
    stm.max_tokens = max_tokens
    agent.context.short_term_memory = stm
    agent.tools = []
    agent.tool_registry.has = MagicMock(return_value=False)
    agent._bg_manager.get_all = MagicMock(return_value=[])
    return agent


def test_status_bar_text_contains_model_and_tokens() -> None:
    from unittest.mock import patch

    agent = _stub_agent("gpt-5", 12_345, 100_000)

    renderer = make_status_bar_text(agent)
    fake_app = MagicMock()
    fake_app.output.get_size.return_value = MagicMock(columns=80)
    with patch("agent_cli.render.ui.get_app", return_value=fake_app):
        out = renderer()
    assert isinstance(out, HTML)
    html = out.value
    assert "gpt-5" in html
    assert "12k/100k" in html
    assert "session" not in html.lower()


def test_status_bar_text_returns_callable() -> None:
    agent = _stub_agent("gpt-5", 500, 100_000)
    renderer = make_status_bar_text(agent)
    assert callable(renderer)


def test_status_bar_text_right_aligns_to_terminal_width() -> None:
    from unittest.mock import patch

    agent = _stub_agent("m", 1, 1000)

    renderer = make_status_bar_text(agent)
    fake_app = MagicMock()
    fake_app.output.get_size.return_value = MagicMock(columns=40)
    with patch("agent_cli.render.ui.get_app", return_value=fake_app):
        out = renderer()
    assert out.value.startswith(" ")
    assert out.value.rstrip().endswith("1/1k")


def test_status_bar_text_dash_when_no_call_yet() -> None:
    from unittest.mock import patch

    agent = _stub_agent("gpt-5", None, 100_000)

    renderer = make_status_bar_text(agent)
    fake_app = MagicMock()
    fake_app.output.get_size.return_value = MagicMock(columns=80)
    with patch("agent_cli.render.ui.get_app", return_value=fake_app):
        out = renderer()
    assert "—/100k" in out.value
