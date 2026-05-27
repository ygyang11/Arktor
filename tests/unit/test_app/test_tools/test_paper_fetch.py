"""Tests for the paper_fetch builtin tool (class-based + session-aware)."""
from __future__ import annotations

import sys

import pytest

from agent_app.tools.paper.paper_fetch import (
    PaperFetchTool,
    _fetch_arxiv_metadata,
    _format_metadata,
    paper_fetch,
)

paper_fetch_mod = sys.modules["agent_app.tools.paper.paper_fetch"]


class TestFormatMetadata:
    def test_full_arxiv_metadata(self) -> None:
        paper = {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "authors": ["Vaswani", "Shazeer", "Parmar"],
            "published": "2017-06-12",
            "abstract": "The dominant sequence transduction models...",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "abs_url": "https://arxiv.org/abs/1706.03762",
        }
        result = _format_metadata(paper)
        assert "# Attention Is All You Need" in result
        assert "1706.03762" in result
        assert "Vaswani" in result
        assert "## Abstract" in result
        assert "paper_fetch" in result

    def test_full_s2_metadata(self) -> None:
        paper = {
            "title": "Test Paper",
            "s2_id": "abc123",
            "doi": "10.1234/xxx",
            "authors": ["Alice"],
            "year": 2023,
            "venue": "NeurIPS",
            "citation_count": 50,
            "reference_count": 30,
            "fields_of_study": ["Computer Science"],
            "publication_types": ["Conference"],
            "abstract": "We propose...",
            "tldr": "A short summary.",
            "pdf_url": "https://example.com/paper.pdf",
        }
        result = _format_metadata(paper)
        assert "# Test Paper" in result
        assert "NeurIPS" in result
        assert "Citations" in result
        assert "References" in result
        assert "Fields of Study" in result
        assert "## TL;DR" in result

    def test_metadata_footer_wording_unchanged(self) -> None:
        paper = {"title": "X", "arxiv_id": "2301.07041"}
        result = _format_metadata(paper)
        assert 'mode="full"' in result
        assert "To get the full paper content, use:" in result


class TestPaperFetchTool:
    def test_class_based(self) -> None:
        assert isinstance(paper_fetch, PaperFetchTool)
        assert paper_fetch.name == "paper_fetch"

    def test_session_aware_structural(self) -> None:
        from agent_harness.tool.base import SessionAware
        t = PaperFetchTool()
        assert isinstance(t, SessionAware)
        t.bind_session("S-x")
        assert t._session_id == "S-x"

    def test_schema_params(self) -> None:
        schema = paper_fetch.get_schema()
        assert schema.name == "paper_fetch"
        props = schema.parameters["properties"]
        assert "paper_id" in props
        assert "mode" in props
        assert "source" in props
        assert props["mode"]["enum"] == ["metadata", "full"]
        assert props["source"]["enum"] == ["arxiv", "semantic_scholar"]

    async def test_empty_id(self) -> None:
        result = await paper_fetch.execute(paper_id="")
        assert "Error" in result

    async def test_unknown_mode(self) -> None:
        result = await paper_fetch.execute(paper_id="test", mode="unknown")
        assert "Error" in result

    async def test_unknown_source(self) -> None:
        result = await paper_fetch.execute(
            paper_id="test", mode="metadata", source="unknown",
        )
        assert "Error" in result


class TestFullModeRoutesThroughParseDocument:
    async def test_arxiv_calls_parse_document_with_session_and_slug_hint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorded: dict[str, object] = {}

        async def _fake_parse(
            *, target: str, session_id: str | None, slug_hint: str | None,
        ) -> str:
            recorded["target"] = target
            recorded["session_id"] = session_id
            recorded["slug_hint"] = slug_hint
            return "Document parsed and saved."

        async def _fake_check(clean_id: str) -> paper_fetch_mod._ArxivIdCheck:
            return paper_fetch_mod._ArxivIdCheck.PRESENT

        monkeypatch.setattr(paper_fetch_mod, "_arxiv_id_check", _fake_check)
        monkeypatch.setattr(paper_fetch_mod, "parse_document", _fake_parse)

        tool = PaperFetchTool()
        tool.bind_session("S1")
        out = await tool.execute(
            paper_id="2401.14200", mode="full", source="arxiv",
        )
        assert recorded["target"] == "https://arxiv.org/pdf/2401.14200"
        assert recorded["session_id"] == "S1"
        assert recorded["slug_hint"] == "arxiv-2401.14200"
        assert "Document parsed and saved." in out

    async def test_arxiv_removed_legacy_paths_stay_gone(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # HTML fast path + legacy pdf_parser glue are gone for good.
        # (existence probe `_arxiv_id_check` is re-introduced; see dedicated tests below)
        assert not hasattr(paper_fetch_mod, "_try_arxiv_html")
        assert not hasattr(paper_fetch_mod, "_fetch_via_pdf_parser")
        assert not hasattr(paper_fetch_mod, "_clean_pdf_failure_reason")

        called: list[str] = []

        async def _fake_parse(
            *, target: str, session_id: str | None, slug_hint: str | None,
        ) -> str:
            called.append("parse")
            return "ok"

        async def _fake_check(clean_id: str) -> paper_fetch_mod._ArxivIdCheck:
            return paper_fetch_mod._ArxivIdCheck.PRESENT

        monkeypatch.setattr(paper_fetch_mod, "parse_document", _fake_parse)
        monkeypatch.setattr(paper_fetch_mod, "_arxiv_id_check", _fake_check)

        tool = PaperFetchTool()
        out = await tool.execute(paper_id="2401.14200", mode="full", source="arxiv")
        assert called == ["parse"]
        assert out == "ok"

    async def test_arxiv_missing_id_fast_fails_before_parse(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called: list[str] = []

        async def _fake_check(clean_id: str) -> paper_fetch_mod._ArxivIdCheck:
            return paper_fetch_mod._ArxivIdCheck.MISSING

        async def _fake_parse(
            *, target: str, session_id: str | None, slug_hint: str | None,
        ) -> str:
            called.append("parse")
            return "should not run"

        monkeypatch.setattr(paper_fetch_mod, "_arxiv_id_check", _fake_check)
        monkeypatch.setattr(paper_fetch_mod, "parse_document", _fake_parse)

        tool = PaperFetchTool()
        out = await tool.execute(paper_id="9999.99999", mode="full", source="arxiv")
        assert out == "Error: no arXiv paper found for ID: 9999.99999"
        assert called == []

    async def test_arxiv_unknown_probe_falls_through_to_parse(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called: list[str] = []

        async def _fake_check(clean_id: str) -> paper_fetch_mod._ArxivIdCheck:
            return paper_fetch_mod._ArxivIdCheck.UNKNOWN

        async def _fake_parse(
            *, target: str, session_id: str | None, slug_hint: str | None,
        ) -> str:
            called.append("parse")
            return "ok"

        monkeypatch.setattr(paper_fetch_mod, "_arxiv_id_check", _fake_check)
        monkeypatch.setattr(paper_fetch_mod, "parse_document", _fake_parse)

        tool = PaperFetchTool()
        out = await tool.execute(paper_id="2401.14200", mode="full", source="arxiv")
        assert out == "ok"
        assert called == ["parse"]

    async def test_s2_pdf_resolve_failure_propagates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fake_resolve(
            paper_id: str, source: str, api_key: str | None,
        ) -> str:
            return "Error: paper not found: x"

        async def _fake_parse(
            *, target: str, session_id: str | None, slug_hint: str | None,
        ) -> str:
            raise AssertionError("parse_document must not be called")

        monkeypatch.setattr(paper_fetch_mod, "_resolve_pdf_url", _fake_resolve)
        monkeypatch.setattr(paper_fetch_mod, "parse_document", _fake_parse)

        tool = PaperFetchTool()
        out = await tool.execute(
            paper_id="DOI:10.1234/test", mode="full", source="semantic_scholar",
        )
        assert out == "Error: paper not found: x"

    async def test_no_slug_hint_instance_state(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Confirm the tool has no _slug_hint module-global / instance attribute
        assert not hasattr(paper_fetch, "_slug_hint")

        captured_hints: list[str | None] = []

        async def _fake_check(clean_id: str) -> paper_fetch_mod._ArxivIdCheck:
            return paper_fetch_mod._ArxivIdCheck.PRESENT

        async def _fake_parse(
            *, target: str, session_id: str | None, slug_hint: str | None,
        ) -> str:
            captured_hints.append(slug_hint)
            return "ok"

        monkeypatch.setattr(paper_fetch_mod, "_arxiv_id_check", _fake_check)
        monkeypatch.setattr(paper_fetch_mod, "parse_document", _fake_parse)

        a = PaperFetchTool()
        a.bind_session("SA")
        b = PaperFetchTool()
        b.bind_session("SB")

        await a.execute(paper_id="2401.14200", mode="full", source="arxiv")
        await b.execute(paper_id="2401.99999", mode="full", source="arxiv")
        assert captured_hints == ["arxiv-2401.14200", "arxiv-2401.99999"]


class TestArxivMetadata:
    async def test_fetch_xml_exception_returns_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_search_module = sys.modules["agent_app.tools.paper.paper_search"]

        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)
        result = await _fetch_arxiv_metadata("2301.07041")
        assert result.startswith("Error: arXiv request failed:")
