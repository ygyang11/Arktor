"""Tests for repl/mentions.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.repl.mentions import (
    _build_calls,
    _within,
    embed_attachments_into_last_user,
    expand_mentions,
    find_at_token,
    parse_mentions,
)
from agent_harness.core.message import Message, Role, ToolCall, ToolResult


class TestFindAtToken:
    @pytest.mark.parametrize("text,expected", [
        ("", None),
        ("@", (0, "")),
        ("@src", (0, "src")),
        ("hello @src/", (6, "src/")),
        ("email@example.com", None),
        ("@src ", None),
        ("see @a and @b", (11, "b")),
        ("\t@src", (1, "src")),
        ("@src/agent_cli/app.py", (0, "src/agent_cli/app.py")),
        ("\n@x", (1, "x")),
    ])
    def test_table(self, text: str, expected: tuple[int, str] | None) -> None:
        assert find_at_token(text) == expected


class TestParseMentions:
    @pytest.mark.parametrize("text,expected", [
        ("", []),
        ("no at", []),
        ("@foo.py", ["foo.py"]),
        ("@a.py and @b.py", ["a.py", "b.py"]),
        ("email@x.com @real.py", ["real.py"]),
        ("@/etc/hosts", ["/etc/hosts"]),
        ("@~/foo", ["~/foo"]),
        ("@my docs/file", ["my"]),
        ("\n@multi\n@line", ["multi", "line"]),
    ])
    def test_table(self, text: str, expected: list[str]) -> None:
        assert parse_mentions(text) == expected


class TestWithin:
    def test_inside(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        assert _within(sub, tmp_path) is True

    def test_outside(self, tmp_path: Path) -> None:
        other = tmp_path.parent
        assert _within(other, tmp_path) is False


class TestBuildCalls:
    def _agent_with_tools(self, *names: str) -> Any:
        agent = MagicMock()
        registered = set(names)
        agent.tool_registry.has = lambda n: n in registered
        return agent

    def test_skips_home_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        out = _build_calls(agent, ["~/anything"])

        assert out == []

    def test_skips_nonexistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        out = _build_calls(agent, ["nope.py"])

        assert out == []

    def test_skips_outside_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        outside = tmp_path.parent / "outside_xyz.py"
        outside.write_text("x")
        try:
            monkeypatch.chdir(tmp_path)
            agent = self._agent_with_tools("read_file", "list_dir")

            out = _build_calls(agent, [str(outside)])

            assert out == []
        finally:
            outside.unlink(missing_ok=True)

    def test_file_routes_to_read_file_with_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        out = _build_calls(agent, ["foo.py"])

        assert out == [("read_file", {"file_path": "foo.py", "limit": 500})]

    def test_dir_routes_to_list_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        out = _build_calls(agent, ["src"])

        assert out == [("list_dir", {"path": "src"})]

    def test_dedupe_canonicalizes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        out = _build_calls(agent, ["foo.py", "./foo.py"])

        assert len(out) == 1

    def test_skips_unknown_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools()

        out = _build_calls(agent, ["foo.py"])

        assert out == []

    def test_mixed_file_and_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        out = _build_calls(agent, ["foo.py", "src"])

        assert len(out) == 2
        assert out[0][0] == "read_file"
        assert out[1][0] == "list_dir"

    def test_completer_suggestion_resolves_to_same_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agent_cli.runtime.file_index import list_project_files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x")
        (tmp_path / "top.py").write_text("y")
        monkeypatch.chdir(tmp_path)

        cache = list_project_files(tmp_path)
        agent = self._agent_with_tools("read_file", "list_dir")

        for entry in cache:
            if entry.endswith("/"):
                continue
            out = _build_calls(agent, [entry])
            assert len(out) == 1, f"completer entry {entry!r} not resolvable"
            _, args = out[0]
            raw = args.get("file_path") or args.get("path")
            assert raw is not None
            tool_resolved = (
                Path(raw) if Path(raw).is_absolute()
                else tmp_path / raw
            ).resolve()
            assert tool_resolved == (tmp_path / entry).resolve()


def _agent_with_stm(last_content: str | None) -> Any:
    agent = MagicMock()
    agent.tool_registry.has = lambda n: True
    msgs: list[Message] = []
    if last_content is not None:
        msgs.append(Message.user(last_content))
    agent.context.short_term_memory._messages = msgs
    return agent


def _stream_factory(content_by_name: dict[str, str] | None = None,
                     is_error: bool = False) -> Any:
    async def fake_stream(tcs: list[ToolCall]) -> Any:
        for tc in tcs:
            body = (content_by_name or {}).get(tc.name, "ok")
            yield ToolResult(
                tool_call_id=tc.id, content=body, is_error=is_error,
            )
    return fake_stream


class TestExpandMentions:
    @pytest.mark.asyncio
    async def test_no_mentions_skips_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        agent = MagicMock()
        agent.tool_executor.execute_stream = MagicMock()
        adapter = MagicMock()

        await expand_mentions(agent, adapter, "no mentions here")

        agent.tool_executor.execute_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_skipped_paths_no_executor_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        agent = MagicMock()
        agent.tool_registry.has = lambda n: True
        agent.tool_executor.execute_stream = MagicMock()
        adapter = MagicMock()

        await expand_mentions(agent, adapter, "@~/foo @/etc/hosts @nope")

        agent.tool_executor.execute_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_file_mention_embeds_call_and_result_reminders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)
        agent = _agent_with_stm("look at @foo.py")
        agent.tool_executor.execute_stream = _stream_factory(
            {"read_file": "[foo.py] lines 1-1 of 1\n1\tx"},
        )
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "look at @foo.py")

        adapter.render_attachments.assert_awaited_once()
        last = agent.context.short_term_memory._messages[-1]
        assert last.role == Role.USER
        assert last.content.startswith(
            "<system-reminder>\nCalled the read_file tool with the following input:"
        )
        assert "Result of calling the read_file tool:" in last.content
        assert last.content.endswith("look at @foo.py")
        atts = last.metadata["attachments"]
        assert len(atts) == 1
        assert atts[0]["tool_name"] == "read_file"
        assert atts[0]["is_error"] is False

    @pytest.mark.asyncio
    async def test_directory_mention_uses_list_dir_tool_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        agent = _agent_with_stm("@src")
        agent.tool_executor.execute_stream = _stream_factory(
            {"list_dir": "[src] (0 entries):"},
        )
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "@src")

        last = agent.context.short_term_memory._messages[-1]
        assert "Called the list_dir tool" in last.content
        assert last.metadata["attachments"][0]["tool_name"] == "list_dir"

    @pytest.mark.asyncio
    async def test_error_attachment_keeps_error_content_in_result_reminder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)
        agent = _agent_with_stm("@foo.py")
        agent.tool_executor.execute_stream = _stream_factory(
            {"read_file": "Permission denied"}, is_error=True,
        )
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "@foo.py")

        last = agent.context.short_term_memory._messages[-1]
        assert "Permission denied" in last.content
        assert last.metadata["attachments"][0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_multiple_mentions_preserve_declaration_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        agent = _agent_with_stm("@a.py @src compare")
        agent.tool_executor.execute_stream = _stream_factory()
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "@a.py @src compare")

        last = agent.context.short_term_memory._messages[-1]
        i_read = last.content.index("Called the read_file tool")
        i_list = last.content.index("Called the list_dir tool")
        assert i_read < i_list
        names = [a["tool_name"] for a in last.metadata["attachments"]]
        assert names == ["read_file", "list_dir"]

    @pytest.mark.asyncio
    async def test_pure_at_mention_no_text_keeps_at_in_trailing_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)
        agent = _agent_with_stm("@foo.py")
        agent.tool_executor.execute_stream = _stream_factory()
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "@foo.py")

        last = agent.context.short_term_memory._messages[-1]
        assert last.content.endswith("@foo.py")
        assert last.content.count("<system-reminder>") == 2

    @pytest.mark.asyncio
    async def test_no_mention_no_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        agent = _agent_with_stm("plain message")
        agent.tool_executor.execute_stream = _stream_factory()
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "plain message")

        last = agent.context.short_term_memory._messages[-1]
        assert last.content == "plain message"
        assert "attachments" not in last.metadata


class TestEmbedAttachmentsIntoLastUser:
    @pytest.mark.asyncio
    async def test_noop_when_last_not_user(self) -> None:
        agent = MagicMock()
        agent.context.short_term_memory._messages = [
            Message.assistant(content="hi"),
        ]
        tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "f"})
        tr = ToolResult(tool_call_id="c1", content="data")

        await embed_attachments_into_last_user(agent, [(tc, tr)])

        last = agent.context.short_term_memory._messages[-1]
        assert last.content == "hi"

    @pytest.mark.asyncio
    async def test_noop_when_no_messages(self) -> None:
        agent = MagicMock()
        agent.context.short_term_memory._messages = []
        tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "f"})
        tr = ToolResult(tool_call_id="c1", content="data")

        await embed_attachments_into_last_user(agent, [(tc, tr)])

        assert agent.context.short_term_memory._messages == []
