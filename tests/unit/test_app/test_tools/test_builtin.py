"""Tests for builtin tool exports."""
from __future__ import annotations

from agent_app.tools import BUILTIN_TOOLS


class TestBuiltinTools:
    def test_builtin_tools_include_web_search(self) -> None:
        names = [t.name for t in BUILTIN_TOOLS]
        assert "web_search" in names

    def test_builtin_tools_include_all_core_tools(self) -> None:
        names = [t.name for t in BUILTIN_TOOLS]
        assert "terminal_tool" in names
        assert "web_fetch" in names
        assert "document_parser" in names
        assert "skill_tool" in names
        assert "memory_tool" in names

    def test_pdf_parser_is_removed(self) -> None:
        names = [t.name for t in BUILTIN_TOOLS]
        assert "pdf_parser" not in names

    def test_all_tools_have_schema(self) -> None:
        for t in BUILTIN_TOOLS:
            schema = t.get_schema()
            assert schema.name == t.name
