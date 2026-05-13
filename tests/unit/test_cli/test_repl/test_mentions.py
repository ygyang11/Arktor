"""Tests for repl/mentions.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.repl.mentions import (
    _build_calls,
    _within,
    expand_mentions,
    find_at_token,
    is_attachment_turn,
    parse_mentions,
)
from agent_harness.core.message import Message, ToolCall, ToolResult


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
    async def test_file_mention_invokes_executor_and_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)

        agent = MagicMock()
        agent.tool_registry.has = lambda n: True
        agent.llm.synthetic_turn_sidecar = MagicMock(return_value={})

        async def fake_stream(tcs: list[ToolCall]):
            for tc in tcs:
                yield ToolResult(
                    tool_call_id=tc.id, content="ok", is_error=False,
                )

        agent.tool_executor.execute_stream = fake_stream
        recorded: list[Any] = []

        async def add_message(msg: Any) -> None:
            recorded.append(msg)

        agent.context.short_term_memory.add_message = add_message
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "look at @foo.py")

        adapter.render_attachments.assert_awaited_once()
        assert len(recorded) == 2  # assistant + 1 tool

    def test_completer_suggestion_resolves_to_same_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Contract: whatever the completer cache surfaces, _build_calls
        resolves to the same physical path. Pins completer/executor
        symmetry so the menu can't show X while the tool loads Y."""
        from agent_cli.runtime.file_index import list_project_files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x")
        (tmp_path / "top.py").write_text("y")
        monkeypatch.chdir(tmp_path)

        cache = list_project_files(tmp_path)
        agent = MagicMock()
        agent.tool_registry.has = lambda n: True

        for entry in cache:
            if entry.endswith("/"):
                continue
            out = _build_calls(agent, [entry])
            assert len(out) == 1, f"completer entry {entry!r} not resolvable"
            name, args = out[0]
            raw = args.get("file_path") or args.get("path")
            assert raw is not None
            tool_resolved = (
                Path(raw) if Path(raw).is_absolute()
                else tmp_path / raw
            ).resolve()
            cache_resolved = (tmp_path / entry).resolve()
            assert tool_resolved == cache_resolved

    @pytest.mark.asyncio
    async def test_dedupe_canonicalizes_to_one_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "foo.py").write_text("x")
        monkeypatch.chdir(tmp_path)

        agent = MagicMock()
        agent.tool_registry.has = lambda n: True
        agent.llm.synthetic_turn_sidecar = MagicMock(return_value={})

        captured: list[list[ToolCall]] = []

        async def fake_stream(tcs: list[ToolCall]):
            captured.append(list(tcs))
            for tc in tcs:
                yield ToolResult(tool_call_id=tc.id, content="ok")

        agent.tool_executor.execute_stream = fake_stream
        agent.context.short_term_memory.add_message = AsyncMock()
        adapter = MagicMock()
        adapter.render_attachments = AsyncMock()

        await expand_mentions(agent, adapter, "@foo.py @./foo.py")

        assert len(captured[0]) == 1


# ── is_attachment_turn ───────────────────────────────────────────────


class TestIsAttachmentTurn:
    """Validate the detector contract: a tool_call counts as part of an
    attachment turn iff one of its string arguments equals one of the
    user's @-mentions."""

    def test_true_when_path_matches_mention(self) -> None:
        u = Message.user("look at @foo.py")
        a = Message.assistant(
            content="",
            tool_calls=[ToolCall(
                id="c1", name="read_file",
                arguments={"file_path": "foo.py", "limit": 500},
            )],
        )
        assert is_attachment_turn(u, a) is True

    def test_true_for_multiple_mentions_and_tools(self) -> None:
        u = Message.user("compare @a.py and @src")
        a = Message.assistant(content="", tool_calls=[
            ToolCall(id="c1", name="read_file", arguments={"file_path": "a.py"}),
            ToolCall(id="c2", name="list_dir", arguments={"path": "src"}),
        ])
        assert is_attachment_turn(u, a) is True

    def test_true_for_future_tool_name(self) -> None:
        """Detection is tool-name agnostic — any tool that stores the raw
        mention text in a string arg should be recognised."""
        u = Message.user("@foo")
        a = Message.assistant(
            content="",
            tool_calls=[ToolCall(
                id="c1", name="future_tool",
                arguments={"target": "foo"},
            )],
        )
        assert is_attachment_turn(u, a) is True

    def test_false_when_agent_calls_unrelated_path(self) -> None:
        u = Message.user("search for bar")
        a = Message.assistant(
            content="",
            tool_calls=[ToolCall(
                id="c1", name="read_file",
                arguments={"file_path": "bar.py"},
            )],
        )
        assert is_attachment_turn(u, a) is False

    def test_false_when_user_has_no_mention(self) -> None:
        u = Message.user("just chat")
        a = Message.assistant(
            content="",
            tool_calls=[ToolCall(
                id="c1", name="read_file",
                arguments={"file_path": "foo.py"},
            )],
        )
        assert is_attachment_turn(u, a) is False

    def test_false_when_assistant_has_content(self) -> None:
        u = Message.user("@foo.py")
        a = Message.assistant(
            content="thinking...",
            tool_calls=[ToolCall(
                id="c1", name="read_file",
                arguments={"file_path": "foo.py"},
            )],
        )
        assert is_attachment_turn(u, a) is False

    def test_false_when_tool_arg_value_is_not_string(self) -> None:
        """numeric-only arg values can't match a mention path, so a tool
        with only numeric args is rejected."""
        u = Message.user("@500 please")
        a = Message.assistant(
            content="",
            tool_calls=[ToolCall(
                id="c1", name="t",
                arguments={"limit": 500},
            )],
        )
        assert is_attachment_turn(u, a) is False

    def test_false_when_some_tool_does_not_match(self) -> None:
        """All tool_calls must correspond to a mention; a mixed turn is
        not the @-expansion shape (`_build_calls` never produces it)."""
        u = Message.user("@foo.py")
        a = Message.assistant(content="", tool_calls=[
            ToolCall(id="c1", name="read_file", arguments={"file_path": "foo.py"}),
            ToolCall(id="c2", name="read_file", arguments={"file_path": "unrelated.py"}),
        ])
        assert is_attachment_turn(u, a) is False

    def test_false_for_empty_tool_calls(self) -> None:
        u = Message.user("@foo.py")
        a = Message.assistant(content="", tool_calls=[])
        assert is_attachment_turn(u, a) is False

    def test_false_when_user_content_empty(self) -> None:
        u = Message.user("")
        a = Message.assistant(
            content="",
            tool_calls=[ToolCall(
                id="c1", name="read_file",
                arguments={"file_path": "foo.py"},
            )],
        )
        assert is_attachment_turn(u, a) is False
