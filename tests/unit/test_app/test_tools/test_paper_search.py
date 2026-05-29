"""Tests for the paper_search builtin tool."""
from __future__ import annotations

import json
import sys
from xml.etree.ElementTree import fromstring

import pytest

from agent_harness.core.errors import ToolValidationError
from agent_app.tools.paper.paper_search import (
    _build_arxiv_query_url,
    _fetch_xml,
    _format_paper_results,
    _looks_like_arxiv_id,
    _normalize_arxiv_id,
    _parse_arxiv_entry,
    _parse_s2_paper,
    paper_search,
)


class TestArxivIdParsing:
    def test_new_style_id(self) -> None:
        assert _looks_like_arxiv_id("2301.07041")

    def test_new_style_with_version(self) -> None:
        assert _looks_like_arxiv_id("2301.07041v2")

    def test_old_style_id(self) -> None:
        assert _looks_like_arxiv_id("cs/0601001")

    def test_url_stripped(self) -> None:
        assert _looks_like_arxiv_id("https://arxiv.org/abs/2301.07041")

    def test_pdf_url_stripped(self) -> None:
        assert _looks_like_arxiv_id("https://arxiv.org/pdf/2301.07041.pdf")

    def test_not_arxiv_id(self) -> None:
        assert not _looks_like_arxiv_id("transformer attention")

    def test_normalize_strips_version(self) -> None:
        assert _normalize_arxiv_id("2301.07041v3") == "2301.07041"

    def test_normalize_strips_url(self) -> None:
        assert _normalize_arxiv_id("https://arxiv.org/abs/2301.07041") == "2301.07041"

    def test_normalize_strips_pdf_url(self) -> None:
        result = _normalize_arxiv_id("https://arxiv.org/pdf/2301.07041.pdf")
        assert result == "2301.07041"


class TestArxivQueryUrl:
    def test_keyword_search(self) -> None:
        url = _build_arxiv_query_url("transformer attention", 10)
        assert "search_query=" in url
        assert "sortBy=relevance" in url

    def test_id_search(self) -> None:
        url = _build_arxiv_query_url("2301.07041", 1)
        assert "id_list=2301.07041" in url
        assert "search_query" not in url

    def test_explicit_id_prefix(self) -> None:
        url = _build_arxiv_query_url("id:2301.07041", 1)
        assert "id_list=2301.07041" in url

    def test_field_prefix_passes_through(self) -> None:
        url = _build_arxiv_query_url("au:bengio", 10)
        assert "search_query=au%3Abengio" in url
        assert "all%3A" not in url

    def test_boolean_passes_through(self) -> None:
        url = _build_arxiv_query_url("foo AND bar", 10)
        assert "all%3A" not in url
        assert "AND" in url

    def test_standalone_submitted_date_passes_through(self) -> None:
        url = _build_arxiv_query_url(
            "submittedDate:[202601010000 TO 202605152359]", 10
        )
        assert "all%3A" not in url
        assert "submittedDate" in url

    def test_plain_title_with_colon_not_misdetected(self) -> None:
        url = _build_arxiv_query_url("BERT: pre-training", 10)
        assert "search_query=all%3A" in url

    def test_plain_keyword_wrapped_with_all(self) -> None:
        url = _build_arxiv_query_url("transformer", 10)
        assert "search_query=all%3Atransformer" in url


class TestPaperSearchDescription:
    def test_schema_description_is_structured_constant(self) -> None:
        from agent_app.tools.paper.paper_search import PAPER_SEARCH_DESCRIPTION

        schema = paper_search.get_schema()
        assert schema.description == PAPER_SEARCH_DESCRIPTION
        assert "## source='arxiv'" in schema.description
        assert "submittedDate" in schema.description
        assert "lastUpdatedDate" not in schema.description


class TestArxivParsing:
    def test_parse_entry(self) -> None:
        xml = (
            '<entry xmlns="http://www.w3.org/2005/Atom"'
            '       xmlns:arxiv="http://arxiv.org/schemas/atom">'
            "<id>http://arxiv.org/abs/2301.07041v1</id>"
            "<title>Test Paper Title</title>"
            "<summary>This is the abstract.</summary>"
            "<author><name>Alice</name></author>"
            "<author><name>Bob</name></author>"
            "<published>2023-01-17T00:00:00Z</published>"
            "<updated>2023-01-18T00:00:00Z</updated>"
            '<arxiv:primary_category term="cs.AI"/>'
            '<category term="cs.AI"/>'
            '<category term="cs.CL"/>'
            "</entry>"
        )
        entry = fromstring(xml)
        result = _parse_arxiv_entry(entry)
        assert result["arxiv_id"] == "2301.07041"
        assert result["title"] == "Test Paper Title"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["abstract"] == "This is the abstract."
        assert result["published"] == "2023-01-17"
        assert result["pdf_url"] == "https://arxiv.org/pdf/2301.07041.pdf"

    def test_parse_entry_no_authors(self) -> None:
        xml = (
            '<entry xmlns="http://www.w3.org/2005/Atom">'
            "<id>http://arxiv.org/abs/2301.00001v1</id>"
            "<title>No Authors</title>"
            "<summary>Abstract.</summary>"
            "<published>2023-01-01T00:00:00Z</published>"
            "<updated>2023-01-01T00:00:00Z</updated>"
            "</entry>"
        )
        entry = fromstring(xml)
        result = _parse_arxiv_entry(entry)
        assert result["authors"] == []


class TestS2Parsing:
    def test_parse_s2_paper(self) -> None:
        raw = {
            "paperId": "abc123",
            "title": "A Great Paper",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "abstract": "We propose ...",
            "year": 2023,
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.1234/xxx", "ArXiv": "2301.07041"},
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            "citationCount": 42,
            "publicationTypes": ["Conference"],
        }
        result = _parse_s2_paper(raw)
        assert result["s2_id"] == "abc123"
        assert result["title"] == "A Great Paper"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["doi"] == "10.1234/xxx"
        assert result["arxiv_id"] == "2301.07041"
        assert result["citation_count"] == 42
        assert result["pdf_url"] == "https://example.com/paper.pdf"

    def test_parse_s2_paper_missing_fields(self) -> None:
        raw = {"paperId": "abc", "title": "Minimal"}
        result = _parse_s2_paper(raw)
        assert result["s2_id"] == "abc"
        assert result["authors"] == []
        assert result["abstract"] == ""
        assert result["doi"] == ""


class TestFormatResults:
    def test_arxiv_format_includes_all_fields(self) -> None:
        papers = [
            {
                "arxiv_id": "2301.07041",
                "title": "Test",
                "authors": ["Alice"],
                "abstract": "Abstract text",
                "published": "2023-01-17",
                "categories": ["cs.AI"],
                "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
                "abs_url": "https://arxiv.org/abs/2301.07041",
            }
        ]
        result = _format_paper_results(papers, source="arxiv")
        assert "2301.07041" in result
        assert "Alice" in result
        assert "Abstract text" in result
        assert "paper_fetch" in result

    def test_s2_format_includes_venue_and_citations(self) -> None:
        papers = [
            {
                "s2_id": "abc",
                "title": "Test",
                "authors": ["Bob"],
                "abstract": "",
                "year": 2023,
                "venue": "ICML",
                "doi": "10.1234/xxx",
                "citation_count": 100,
                "pdf_url": "",
            }
        ]
        result = _format_paper_results(papers, source="semantic_scholar")
        assert "ICML" in result
        assert "100" in result
        assert "DOI" in result

    def test_arxiv_footer_suggests_semantic_scholar(self) -> None:
        papers = [{"title": "X", "authors": [], "abstract": ""}]
        result = _format_paper_results(papers, source="arxiv")
        assert "semantic_scholar" in result

    def test_many_authors_truncated(self) -> None:
        papers = [
            {
                "title": "X",
                "authors": ["A", "B", "C", "D", "E", "F", "G"],
                "abstract": "",
            }
        ]
        result = _format_paper_results(papers, source="arxiv")
        assert "et al." in result
        assert "7 authors" in result

    def test_many_categories_truncated(self) -> None:
        papers = [
            {
                "title": "X",
                "authors": [],
                "abstract": "",
                "categories": ["cs.AI", "cs.CL", "cs.LG", "cs.CV", "stat.ML"],
            }
        ]
        result = _format_paper_results(papers, source="arxiv")
        assert "Categories: cs.AI, cs.CL, cs.LG et al. (5 categories)" in result

    def test_footer_uses_single_paper_fetch_hint(self) -> None:
        papers = [{"title": "X", "authors": [], "abstract": ""}]
        result = _format_paper_results(papers, source="arxiv")
        assert 'mode="<metadata|full>"' in result

    def test_footer_full_mode_wording_on_disk_artifacts(self) -> None:
        papers = [{"title": "X", "authors": [], "abstract": ""}]
        result = _format_paper_results(papers, source="arxiv")
        assert "content.md / images / layout, on-disk artifacts" in result
        assert "full paper body text" not in result


class TestPaperSearchTool:
    async def test_empty_query(self) -> None:
        with pytest.raises(ToolValidationError):
            await paper_search.execute(query="")

    async def test_unknown_source(self) -> None:
        with pytest.raises(ToolValidationError, match="semantic_scholar"):
            await paper_search.execute(query="test", source="unknown")

    def test_schema_has_correct_params(self) -> None:
        schema = paper_search.get_schema()
        assert schema.name == "paper_search"
        props = schema.parameters["properties"]
        assert "query" in props
        assert "source" in props
        assert "max_results" in props
        assert props["source"]["enum"] == ["arxiv", "semantic_scholar"]

    @pytest.mark.asyncio
    async def test_arxiv_failure_returns_error_string(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("arXiv API returned HTTP 503")

        paper_search_module = sys.modules[_fetch_xml.__module__]
        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)

        result = await paper_search.execute(query="2301.07041", source="arxiv")
        assert result == "Error: arXiv API returned HTTP 503"

    @pytest.mark.asyncio
    async def test_arxiv_failure_does_not_duplicate_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("arXiv request failed: simulated network failure")

        paper_search_module = sys.modules[_fetch_xml.__module__]
        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)

        result = await paper_search.execute(query="2301.07041", source="arxiv")
        assert result == "Error: arXiv request failed: simulated network failure"


class _FakeTimeout:
    def __init__(self, total: int) -> None:
        self.total = total


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body
        self.charset = "utf-8"

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def text(self) -> str:
        return self._body

    async def read(self) -> bytes:
        return self._body.encode("utf-8")


class _FakeSession:
    def __init__(
        self,
        statuses: list[int],
        calls: list[int],
        bodies: list[str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._calls = calls
        self._bodies = bodies or []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _FakeResponse:
        _ = (method, url, kwargs)
        self._calls.append(1)
        status = self._statuses[min(len(self._calls) - 1, len(self._statuses) - 1)]
        if self._bodies:
            body = self._bodies[min(len(self._calls) - 1, len(self._bodies) - 1)]
        else:
            body = f"status={status}"
        return _FakeResponse(status, body)


class _FakeAiohttpModule:
    class ClientError(Exception):
        pass

    ClientTimeout = _FakeTimeout

    def __init__(
        self,
        statuses: list[int],
        calls: list[int],
        bodies: list[str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._calls = calls
        self._bodies = bodies

    def ClientSession(self) -> _FakeSession:  # noqa: N802
        return _FakeSession(self._statuses, self._calls, self._bodies)


class TestFetchXmlWithRetry:
    @pytest.mark.asyncio
    async def test_max_retries_means_total_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_search_module = sys.modules[_fetch_xml.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule([429, 429, 429, 429], call_markers)
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_search_module,
            "_CFG",
            paper_search_module.PaperSearchConfig(
                max_retries=3,
                retry_base_delay=0.0,
            ),
        )

        with pytest.raises(RuntimeError, match="HTTP 429"):
            await _fetch_xml("https://example.com")
        assert len(call_markers) == 3

    @pytest.mark.asyncio
    async def test_minimum_one_attempt_when_config_is_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_search_module = sys.modules[_fetch_xml.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule(
            [200],
            call_markers,
            bodies=["<feed xmlns='http://www.w3.org/2005/Atom'></feed>"],
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_search_module,
            "_CFG",
            paper_search_module.PaperSearchConfig(
                max_retries=0,
                retry_base_delay=0.0,
            ),
        )

        root = await _fetch_xml("https://example.com")

        assert root.tag.endswith("feed")
        assert len(call_markers) == 1

    @pytest.mark.asyncio
    async def test_retries_on_http_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        paper_search_module = sys.modules[_fetch_xml.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule(
            [500, 502, 200],
            call_markers,
            bodies=["e1", "e2", "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"],
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_search_module,
            "_CFG",
            paper_search_module.PaperSearchConfig(max_retries=3, retry_base_delay=0.0),
        )

        root = await _fetch_xml("https://example.com")

        assert root.tag.endswith("feed")
        assert len(call_markers) == 3


class TestSemanticScholarParsing:
    @pytest.mark.asyncio
    async def test_semantic_scholar_invalid_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_search_module = sys.modules[_fetch_xml.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule([200], call_markers, bodies=["not-json"])
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_search_module,
            "_CFG",
            paper_search_module.PaperSearchConfig(max_retries=1, retry_base_delay=0.0),
        )

        result = await paper_search.execute(query="federated learning", source="semantic_scholar")
        assert result == "Error: Semantic Scholar returned an unreadable response; retry shortly."

    @pytest.mark.asyncio
    async def test_semantic_scholar_non_object_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_search_module = sys.modules[_fetch_xml.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule([200], call_markers, bodies=[json.dumps([1, 2, 3])])
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_search_module,
            "_CFG",
            paper_search_module.PaperSearchConfig(max_retries=1, retry_base_delay=0.0),
        )

        result = await paper_search.execute(query="federated learning", source="semantic_scholar")
        assert result == "Error: Semantic Scholar returned an unreadable response; retry shortly."


class TestPaperSearchDefaults:
    def test_default_max_results_is_10(self) -> None:
        schema = paper_search.get_schema()
        props = schema.parameters["properties"]
        assert props["max_results"]["default"] == 10

    @pytest.mark.asyncio
    async def test_max_results_above_30_is_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, int] = {}
        paper_search_module = sys.modules[_fetch_xml.__module__]

        async def _fake_search_arxiv(query: str, max_results: int) -> str:
            captured["value"] = max_results
            _ = query
            return "ok"

        monkeypatch.setattr(paper_search_module, "_search_arxiv", _fake_search_arxiv)
        result = await paper_search.execute(query="transformer", source="arxiv", max_results=999)

        assert result == "ok"
        assert captured["value"] == 30
