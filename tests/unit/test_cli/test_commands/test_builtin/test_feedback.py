"""Tests for /feedback — GitHub issue prefill URL + browser open."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_cli.commands.builtin import feedback as feedback_mod
from agent_cli.commands.builtin.feedback import CMD

from ..conftest import render_output


def _ctx() -> MagicMock:
    agent = MagicMock()
    agent.llm.model_name = "test-model"
    return MagicMock(agent=agent)


@pytest.fixture(autouse=True)
def _stub_webbrowser(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    captured: dict[str, str] = {}

    def fake_open(url: str) -> bool:
        captured["url"] = url
        return True

    monkeypatch.setattr(feedback_mod.webbrowser, "open", fake_open)
    return captured


async def test_feedback_opens_url_and_returns_link(_stub_webbrowser: dict[str, str]) -> None:
    result = await CMD.handler(_ctx(), "broken plan mode")
    url = _stub_webbrowser["url"]
    assert url.startswith("https://github.com/ygyang11/Arktor/issues/new")
    assert "title=broken%20plan%20mode" in url
    rendered = render_output(result.output)
    assert "Feedback form" in rendered
    assert "github.com" in rendered


async def test_feedback_default_title_when_no_args(_stub_webbrowser: dict[str, str]) -> None:
    await CMD.handler(_ctx(), "")
    assert "title=Problem-Feedback" in _stub_webbrowser["url"]


async def test_feedback_body_includes_metadata(_stub_webbrowser: dict[str, str]) -> None:
    from urllib.parse import parse_qs, urlparse
    await CMD.handler(_ctx(), "x")
    qs = parse_qs(urlparse(_stub_webbrowser["url"]).query)
    body = qs["body"][0]
    assert "Version:" in body
    assert "Platform:" in body
    assert "Python:" in body
    assert "Model: test-model" in body
