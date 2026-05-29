"""Tests for web_search tool output formatting."""
from __future__ import annotations

import pytest

from agent_app.tools.web.web_search import (
    _format_result,
    _format_search_results,
    web_search,
)
from agent_harness.core.errors import ToolValidationError


class TestWebSearchValidation:
    async def test_empty_query_rejected(self) -> None:
        with pytest.raises(ToolValidationError):
            await web_search.execute(query="   ")


class TestFormatResult:
    def test_includes_snippet_label(self) -> None:
        result = _format_result(1, "Title", "summary text", "https://example.com")
        assert "[snippet]" in result
        assert "Title" in result
        assert "https://example.com" in result

    def test_index_is_included(self) -> None:
        result = _format_result(3, "Third", "content", "https://example.com")
        assert result.startswith("3.")

    def test_url_on_separate_line(self) -> None:
        result = _format_result(1, "T", "S", "https://x.com")
        lines = result.strip().splitlines()
        assert any("URL: https://x.com" in line for line in lines)


class TestFormatSearchResults:
    def test_includes_web_fetch_hint(self) -> None:
        results = [("Title", "summary", "https://example.com")]
        output = _format_search_results(results)
        assert "web_fetch" in output
        assert "full page content" in output

    def test_multiple_results_numbered(self) -> None:
        results = [
            ("First", "s1", "https://a.com"),
            ("Second", "s2", "https://b.com"),
        ]
        output = _format_search_results(results)
        assert "1. First" in output
        assert "2. Second" in output

    def test_empty_results_still_has_hint(self) -> None:
        output = _format_search_results([])
        assert "web_fetch" in output
