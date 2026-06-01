import io
from unittest.mock import MagicMock

from rich.console import Console

from agent_cli.render.tool_display import (
    ToolDisplay,
    _format_call_line,
    _format_result_line,
    _ToolRow,
)
from agent_cli.theme import SPINNER_STATIC


def _mock_live_display() -> ToolDisplay:
    d = ToolDisplay(MagicMock())
    d._open_live = MagicMock(side_effect=lambda: setattr(d, "_live", MagicMock()))
    d._close_live = MagicMock(side_effect=lambda: setattr(d, "_live", None))
    return d


def _read_result(content: str) -> MagicMock:
    return MagicMock(
        tool_call_id="t1", content=content, is_error=False, attachments=None,
    )


def test_add_call_opens_live_and_appends_row() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"file_path": "a.py"})
    tc.name = "read_file"
    d.add_call(tc)
    assert len(d._rows) == 1
    assert d._rows[0].status == "running"
    d._open_live.assert_called_once()


def test_read_pdf_attachment_branch_shows_attached_pdf() -> None:
    from agent_harness.core.message import Attachment, ToolCall, ToolResult

    d = _mock_live_display()
    tc = ToolCall(
        id="rdf-1", name="read_file", arguments={"file_path": "doc.pdf"},
    )
    d.add_call(tc)
    att = Attachment(
        digest="d" * 64, mime="application/pdf", filename="doc.pdf", size=2_000_000,
    )
    tr = ToolResult(
        tool_call_id="rdf-1", content="Read PDF", attachments=[att],
    )
    d.mark_result(tr)
    row = d._rows[0]
    line = _format_result_line(row)
    assert line is not None
    assert "Attached PDF" in line.plain


def test_mark_result_done_status_and_summary_formats() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"file_path": "a.py"})
    tc.name = "read_file"
    d.add_call(tc)
    d.mark_result(_read_result("[a.py] lines 1-3 of 3\n1\tfoo\n2\tbar\n3\tbaz"))
    row = d._rows[0]
    assert row.status == "done"
    line = _format_result_line(row)
    assert line is not None and "Read 3 lines" in line.plain


def test_document_parser_background_shows_backgrounded_not_parsed() -> None:
    """In background mode the tool returns immediately with a 'Background
    document_parser bg_NNNN started: ...' line, not a parse result. The row
    must show 'Backgrounded: bg_NNNN', not the fallback 'Parsed'."""
    d = _mock_live_display()
    tc = MagicMock(
        id="t1", arguments={"target": "https://x/a.pdf", "background": True},
    )
    tc.name = "document_parser"
    d.add_call(tc)
    d.mark_result(_read_result(
        "Background document_parser bg_0042 started: https://x/a.pdf"
    ))
    row = d._rows[0]
    line = _format_result_line(row)
    assert line is not None
    assert "Backgrounded: bg_0042" in line.plain
    assert "Parsed" not in line.plain


def test_document_parser_foreground_still_shows_parsed() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"target": "https://x/a.pdf"})
    tc.name = "document_parser"
    d.add_call(tc)
    d.mark_result(_read_result(
        "Document parsed and saved.\nformat: pdf (10 pages)\n"
    ))
    row = d._rows[0]
    line = _format_result_line(row)
    assert line is not None and "Parsed 10p pdf" in line.plain


def test_mark_result_error_summary_is_error_prefixed() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "terminal_tool"
    d.add_call(tc)
    d.mark_result(MagicMock(
        tool_call_id="t1", content="command failed: permission", is_error=True,
    ))
    row = d._rows[0]
    assert row.status == "error"
    line = _format_result_line(row)
    assert line is not None and line.plain.lstrip().startswith("⎿  Error:")


def test_multiline_error_collapses_to_first_line() -> None:
    """`document_parser` and similar multi-section errors should render their
    headline (first line) instead of being squashed onto one line with
    everything jammed together and then truncated mid-detail."""
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"target": "https://x/a.pdf"})
    tc.name = "document_parser"
    d.add_call(tc)
    multiline = (
        "Error: document parsing failed.\n\n"
        "Tried:\n"
        "  1. paddleocr-vl-1.5  url   TIMEOUT (request timeout during PaddleOCR POST)\n"
        "\nSkipped (preflight):\n"
        "  - mineru-lightweight size>10MB(url)\n"
    )
    d.mark_result(MagicMock(
        tool_call_id="t1", content=multiline, is_error=True, attachments=None,
    ))
    row = d._rows[0]
    line = _format_result_line(row)
    assert line is not None
    plain = line.plain
    assert "Error: document parsing failed." in plain
    # detail lines should NOT bleed into the inline summary
    assert "Tried" not in plain
    assert "paddleocr-vl-1.5" not in plain


def test_error_content_string_without_flag_is_detected() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"file_path": "x"})
    tc.name = "read_file"
    d.add_call(tc)
    d.mark_result(MagicMock(
        tool_call_id="t1",
        content="Error: Permission denied: /etc/shadow",
        is_error=False,
    ))
    row = d._rows[0]
    assert row.status == "error"
    line = _format_result_line(row)
    assert line is not None
    assert "Error:" in line.plain and "Permission denied" in line.plain


def test_args_preview_strips_newlines() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"pattern": "foo\nbar"})
    tc.name = "grep_files"
    d.add_call(tc)
    assert "\n" not in d._rows[0].args_preview


def test_args_preview_filters_to_primary_keys() -> None:
    d = _mock_live_display()
    tc = MagicMock(
        id="t1",
        arguments={"file_path": "a.py", "limit": 260, "offset": 0},
    )
    tc.name = "read_file"
    d.add_call(tc)
    preview = d._rows[0].args_preview
    assert preview == "a.py"
    assert "file_path" not in preview
    assert "limit" not in preview
    assert "offset" not in preview


def test_args_preview_unknown_tool_shows_bare_values() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"foo": "1", "bar": "2"})
    tc.name = "unregistered_tool"
    d.add_call(tc)
    preview = d._rows[0].args_preview
    assert preview == "1, 2"
    assert "foo" not in preview and "bar" not in preview
    assert '"' not in preview


def test_add_denied_appends_denied_row_with_reason() -> None:
    d = _mock_live_display()
    ar = MagicMock(tool_call_id="t1", tool_name="edit_file", reason="policy")
    d.add_denied(ar)
    row = d._rows[0]
    assert row.status == "denied"
    assert row.reason == "policy"
    line = _format_result_line(row)
    assert line is not None and "Denied" in line.plain


def test_suppressed_tools_have_no_result_line() -> None:
    row = _ToolRow(id="t1", name="sub_agent", args_preview="", status="done")
    assert _format_result_line(row) is None
    row = _ToolRow(id="t2", name="todo_write", args_preview="", status="done")
    assert _format_result_line(row) is None


def test_end_closes_live_prints_call_and_summary_and_diff_expander() -> None:
    buf = io.StringIO()
    con = Console(file=buf, color_system=None, width=120)
    d = ToolDisplay(con)
    d._live = MagicMock()
    tc = MagicMock(id="t1", arguments={"file_path": "a.py"})
    tc.name = "edit_file"
    d._rows = [_ToolRow(
        id="t1", name="edit_file", args_preview="a.py",
        status="done", tool_call=tc,
        result=MagicMock(
            content="Edited a.py (1 replacement)",
            is_error=False,
            tool_metadata={"diff": "--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n"},
        ),
    )]
    d.end()
    out = buf.getvalue()
    assert "Edit" in out
    assert "edit_file" not in out
    assert "Edited file (1 replacement)" in out
    assert "-x" in out and "+y" in out
    assert d._rows == []
    assert d._live is None


def test_format_call_line_running_uses_static_glyph() -> None:
    row = _ToolRow(id="t1", name="x", args_preview="", status="running")
    result = _format_call_line(row)
    assert SPINNER_STATIC in result.plain


def test_pause_closes_live_but_keeps_rows() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "x"
    d.add_call(tc)
    d.pause()
    assert d._live is None
    assert len(d._rows) == 1


def test_suspend_closes_live_keeps_rows_and_sets_flag() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "x"
    d.add_call(tc)
    d.suspend()
    assert d._live is None
    assert d._suspended is True
    assert len(d._rows) == 1


def test_mark_result_while_suspended_records_status_without_reopen() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "x"
    d.add_call(tc)
    d.suspend()
    d._open_live.reset_mock()
    d.mark_result(_read_result("done"))
    assert d._rows[0].status == "done"  # status recorded off-screen
    d._open_live.assert_not_called()    # but the Live is not reopened
    assert d._live is None


def test_end_after_suspend_clears_flag_and_rows() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "x"
    d.add_call(tc)
    d.suspend()
    d.mark_result(_read_result("done"))
    d.end()
    assert d._suspended is False
    assert d._rows == []


def test_mark_result_without_matching_row_skips_refresh() -> None:
    # Regression: suppressed tools (sub_agent/todo_write) skip add_call but
    # mark_result used to refresh anyway, opening an empty Live that collided
    # with the next markdown phase.
    d = _mock_live_display()
    d.mark_result(_read_result("anything"))
    assert d._live is None
    d._open_live.assert_not_called()


def _tc(name: str, **args: object) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.arguments = args
    return m


def _tr(content: str, *, is_error: bool = False) -> MagicMock:
    return MagicMock(tool_call_id="t1", content=content, is_error=is_error)


def _render_rows(rows: object) -> str:
    from agent_cli.theme import FLEXOKI_DARK
    buf = io.StringIO()
    con = Console(
        file=buf, color_system=None, width=200, theme=FLEXOKI_DARK.rich,
    )
    for r in rows:  # type: ignore[union-attr]
        con.print(r)
    return buf.getvalue()


class TestAttachmentSummaryAndFormat:
    def _render_one(self, tc: object, tr: object) -> str:
        from agent_cli.render.tool_display import (
            attachment_summary,
            format_attachments,
        )
        return _render_rows(format_attachments([attachment_summary(tc, tr)]))

    def test_read_normal_includes_label_target_lines(self) -> None:
        tc = _tc("read_file", file_path="src/foo.py")
        tr = _tr("[src/foo.py] lines 1-3 of 3\nline1\nline2\nline3")
        out = self._render_one(tc, tr)
        assert "Loaded into context" in out
        assert "Read" in out
        assert "src/foo.py" in out
        assert "(3 lines)" in out

    def test_read_truncated_shows_range(self) -> None:
        tc = _tc("read_file", file_path="big.py")
        tr = _tr("[big.py] lines 1-500 of 1200\n" + "\n".join(f"l{i}" for i in range(500)))
        assert "(lines 1-500 of 1200)" in self._render_one(tc, tr)


    def test_read_empty_branch(self) -> None:
        tc = _tc("read_file", file_path="empty.txt")
        tr = _tr("Empty file")
        assert "Empty file" in self._render_one(tc, tr)

    def test_list_dir_normal_includes_entries(self) -> None:
        tc = _tc("list_dir", path="src/")
        tr = _tr("src/  (8 entries)\n  app.py\n  util.py")
        out = self._render_one(tc, tr)
        assert "List" in out
        assert "src/" in out
        assert "(8 entries)" in out

    def test_unknown_tool_falls_back_to_raw_name(self) -> None:
        tc = _tc("custom_tool", path="x")
        tr = _tr("ok")
        out = self._render_one(tc, tr)
        assert "custom_tool" in out
        assert "x" in out

    def test_path_with_brackets_does_not_break(self) -> None:
        tc = _tc("read_file", file_path="src/[generated]/foo.py")
        tr = _tr("x\ny")
        assert "[generated]" in self._render_one(tc, tr)

    def test_attachment_summary_includes_error_snippet_on_error(self) -> None:
        from agent_cli.render.tool_display import attachment_summary
        tc = _tc("read_file", file_path="missing.py")
        tr = _tr("Error: file not found", is_error=True)
        s = attachment_summary(tc, tr)
        assert s["is_error"] is True
        assert "file not found" in s["error_snippet"]
        assert s["summary"] == ""

    def test_format_attachments_renders_from_summary_dict(self) -> None:
        from agent_cli.render.tool_display import (
            attachment_summary,
            format_attachments,
        )
        tc = _tc("read_file", file_path="a.py")
        tr = _tr("[a.py] lines 1-2 of 2\nx\ny")
        out = _render_rows(format_attachments([attachment_summary(tc, tr)]))
        assert "Read" in out
        assert "a.py" in out
        assert "(2 lines)" in out

    def test_format_attachments_renders_error_row(self) -> None:
        from agent_cli.render.tool_display import (
            attachment_summary,
            format_attachments,
        )
        tc = _tc("read_file", file_path="missing.py")
        tr = _tr("Error: nope", is_error=True)
        out = _render_rows(format_attachments([attachment_summary(tc, tr)]))
        assert "missing.py" in out
        assert "nope" in out

    def test_live_and_replay_same_dict_render_identical(self) -> None:
        from agent_cli.render.tool_display import (
            attachment_summary,
            format_attachments,
        )
        tc = _tc("read_file", file_path="a.py")
        tr = _tr("[a.py] lines 1-1 of 1\nx")
        live_dict = attachment_summary(tc, tr)
        # replay path reads the same dict back from persisted metadata
        replay_dict = dict(live_dict)
        assert (
            _render_rows(format_attachments([live_dict]))
            == _render_rows(format_attachments([replay_dict]))
        )


def test_show_todos_escapes_markup_in_content() -> None:
    # Regression: content like "Fix [linux] path" used to be parsed as Rich
    # markup and lose text.
    from agent_cli.theme import FLEXOKI_DARK
    buf = io.StringIO()
    con = Console(file=buf, color_system=None, width=120, theme=FLEXOKI_DARK.rich)
    d = ToolDisplay(con)
    d.show_todos(
        [
            {"status": "pending", "content": "Fix [linux] path handling"},
            {"status": "completed", "content": "[done] wrap it up"},
            {"status": "in_progress", "content": "refactor [core]"},
        ],
        {"total": 3, "completed": 1},
    )
    out = buf.getvalue()
    assert "Fix [linux] path handling" in out
    assert "[done] wrap it up" in out
    assert "refactor [core]" in out
