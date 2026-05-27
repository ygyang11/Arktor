"""Academic paper search tool supporting arXiv and Semantic Scholar."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode
from xml.etree.ElementTree import Element, fromstring

from agent_harness.core.config import resolve_paper_config
from agent_harness.tool.decorator import tool
from agent_harness.utils.http_retry import HttpRetryConfig, http_get_with_retry
from agent_harness.utils.json_utils import parse_json_lenient

logger = logging.getLogger(__name__)


PAPER_SEARCH_DESCRIPTION = (
    "Searches academic paper databases and returns structured metadata "
    "(title, authors, abstract, etc.).\n\n"
    "## source='arxiv'\n"
    "Plain keywords (matched across all fields) OR arXiv structured "
    "query syntax:\n"
    "- Field prefixes: au: (author), ti: (title), abs: (abstract), "
    "cat: (category), co: (comment), jr: (journal ref), "
    "rn: (report number), all: (any field)\n"
    "- Boolean operators: AND, OR, ANDNOT\n"
    "- Date range: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM] "
    "(12-digit timestamp, GMT)\n\n"
    "Examples:\n"
    "- paper_search(query='diffusion transformers')\n"
    "- paper_search(query='au:bengio AND ti:attention')\n"
    "- paper_search(query='cat:cs.AI AND "
    "submittedDate:[202601010000 TO 202605152359]')\n"
    "- paper_search(query='abs:diffusion ANDNOT cat:cs.CL')\n\n"
    "## source='semantic_scholar'\n"
    "Keyword-only. Field prefixes and boolean operators are NOT "
    "supported — pass natural-language terms "
    "(e.g. 'graph neural network robustness')."
)


@dataclass(frozen=True)
class PaperSearchConfig:
    max_retries: int = 3
    retry_base_delay: float = 2.0
    request_timeout: float = 20.0


_CFG = PaperSearchConfig()
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_ARXIV_API = "http://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1"

_NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_ID_RE = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)

_S2_SEARCH_FIELDS = (
    "paperId,title,authors,abstract,year,venue,"
    "externalIds,openAccessPdf,citationCount,publicationTypes"
)


# ---------------------------------------------------------------------------
# arXiv helpers
# ---------------------------------------------------------------------------


_ARXIV_FIELD_RE = re.compile(r"\b(au|ti|abs|cat|co|jr|rn|id|all|submittedDate):")
_ARXIV_BOOL_RE = re.compile(r"\b(AND|OR|ANDNOT)\b")

def _looks_like_arxiv_id(value: str) -> bool:
    cleaned = re.sub(r"^(https?://)?arxiv\.org/(abs|pdf)/", "", value)
    cleaned = re.sub(r"(\.pdf)?$", "", cleaned)
    return bool(_NEW_ID_RE.match(cleaned) or _OLD_ID_RE.match(cleaned))


def _normalize_arxiv_id(arxiv_id: str) -> str:
    cleaned = re.sub(r"^(https?://)?arxiv\.org/(abs|pdf)/", "", arxiv_id)
    cleaned = re.sub(r"(\.pdf)?$", "", cleaned)
    cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned


def _is_structured_arxiv_query(query: str) -> bool:
    return bool(_ARXIV_FIELD_RE.search(query) or _ARXIV_BOOL_RE.search(query))


def _build_arxiv_query_url(query: str, max_results: int, start: int = 0) -> str:
    params: dict[str, str | int] = {"max_results": max_results, "start": start}
    if _looks_like_arxiv_id(query) or query.startswith("id:"):
        raw_id = query.removeprefix("id:")
        params["id_list"] = _normalize_arxiv_id(raw_id)
    elif _is_structured_arxiv_query(query):
        params["search_query"] = query
        params["sortBy"] = "relevance"
        params["sortOrder"] = "descending"
    else:
        params["search_query"] = f"all:{query}"
        params["sortBy"] = "relevance"
        params["sortOrder"] = "descending"
    return f"{_ARXIV_API}?{urlencode(params)}"


async def _fetch_xml(url: str) -> Element:
    retry = HttpRetryConfig(
        max_attempts=max(1, _CFG.max_retries),
        base_delay=_CFG.retry_base_delay,
    )
    headers = {"User-Agent": _USER_AGENT}
    try:
        status, text = await http_get_with_retry(
            url,
            headers=headers,
            timeout=int(_CFG.request_timeout),
            retry=retry,
        )
    except Exception as exc:
        raise RuntimeError(f"arXiv request failed: {exc}") from exc
    if status == 429:
        raise RuntimeError("arXiv API returned HTTP 429")
    if status != 200:
        raise RuntimeError(f"arXiv API returned HTTP {status}")
    try:
        return fromstring(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse arXiv Atom response: {exc}") from exc


def _parse_arxiv_entry(entry: Element) -> dict[str, Any]:
    def _text(tag: str) -> str:
        el = entry.find(f"{_ATOM_NS}{tag}")
        return " ".join((el.text or "").split()) if el is not None else ""

    raw_id = _text("id")
    arxiv_id = _normalize_arxiv_id(raw_id)

    authors: list[str] = []
    for a in entry.findall(f"{_ATOM_NS}author"):
        name_el = a.find(f"{_ATOM_NS}name")
        if name_el is not None:
            authors.append(" ".join((name_el.text or "").split()))

    categories: list[str] = [
        c.get("term", "") for c in entry.findall(f"{_ARXIV_NS}primary_category")
    ]
    categories.extend(
        c.get("term", "")
        for c in entry.findall(f"{_ATOM_NS}category")
        if c.get("term", "") not in categories
    )

    return {
        "arxiv_id": arxiv_id,
        "title": _text("title"),
        "authors": authors,
        "abstract": _text("summary"),
        "published": _text("published")[:10],
        "updated": _text("updated")[:10],
        "categories": categories,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }


async def _search_arxiv(query: str, max_results: int) -> str:
    url = _build_arxiv_query_url(query, max_results)
    try:
        root = await _fetch_xml(url)
    except Exception as exc:
        err = str(exc)
        if err.startswith("arXiv request failed:"):
            return f"Error: {err}"
        return f"Error: arXiv request failed: {err}"

    entries = root.findall(f"{_ATOM_NS}entry")
    if not entries:
        return f"No arXiv results found for: {query}"

    papers = [_parse_arxiv_entry(e) for e in entries]
    return _format_paper_results(papers, source="arxiv")


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------


async def _search_semantic_scholar(
    query: str, max_results: int, api_key: str | None
) -> str:
    params = urlencode({
        "query": query,
        "limit": min(max_results, 30),
        "fields": _S2_SEARCH_FIELDS,
    })
    url = f"{_S2_API}/paper/search?{params}"

    extra_headers: dict[str, str] = {}
    if api_key:
        extra_headers["x-api-key"] = api_key

    headers = {"User-Agent": _USER_AGENT}
    headers.update(extra_headers)
    retry = HttpRetryConfig(
        max_attempts=max(1, _CFG.max_retries),
        base_delay=_CFG.retry_base_delay,
    )
    try:
        status, body = await http_get_with_retry(
            url,
            headers=headers,
            timeout=int(_CFG.request_timeout),
            retry=retry,
        )
    except Exception as exc:
        return f"Error: Semantic Scholar request failed: {exc}"

    if status == 429:
        return (
            "Error: Semantic Scholar rate limit exceeded after retries. "
            "Try again later, reduce max_results, or use source='arxiv' instead."
        )
    if status != 200:
        return (
            f"Error: Semantic Scholar API returned HTTP {status}. "
            f"Detail: {body[:200]}"
        )

    try:
        data_raw = parse_json_lenient(body)
    except ValueError as exc:
        return f"Error: failed to parse Semantic Scholar response: {exc}"
    if not isinstance(data_raw, dict):
        return "Error: unexpected Semantic Scholar response format"
    data = data_raw

    total = data.get("total", 0)
    papers_raw = data.get("data", [])
    if not papers_raw:
        return f"No results found for: {query}"

    papers = [_parse_s2_paper(p) for p in papers_raw]
    result = _format_paper_results(papers, source="semantic_scholar")

    if total > len(papers_raw):
        result += f"\n\n(Showing {len(papers_raw)} of {total} total results)"

    return result


def _parse_s2_paper(paper: dict[str, Any]) -> dict[str, Any]:
    external_ids = paper.get("externalIds") or {}
    authors = [a.get("name", "") for a in (paper.get("authors") or [])]
    pdf_info = paper.get("openAccessPdf") or {}

    return {
        "s2_id": paper.get("paperId", ""),
        "title": paper.get("title", ""),
        "authors": authors,
        "abstract": paper.get("abstract") or "",
        "year": paper.get("year"),
        "venue": paper.get("venue", ""),
        "doi": external_ids.get("DOI", ""),
        "arxiv_id": external_ids.get("ArXiv", ""),
        "citation_count": paper.get("citationCount", 0),
        "pdf_url": pdf_info.get("url", ""),
        "publication_types": paper.get("publicationTypes") or [],
        "source": "semantic_scholar",
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_paper_results(papers: list[dict[str, Any]], source: str) -> str:
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        parts = [f"{i}. {p.get('title', 'Untitled')}"]

        if p.get("arxiv_id"):
            parts.append(f"   arXiv ID: {p['arxiv_id']}")
        if p.get("s2_id"):
            parts.append(f"   S2 ID: {p['s2_id']}")
        if p.get("doi"):
            parts.append(f"   DOI: {p['doi']}")

        authors = p.get("authors", [])
        if authors:
            display = ", ".join(authors[:3])
            if len(authors) > 3:
                display += f" et al. ({len(authors)} authors)"
            parts.append(f"   Authors: {display}")

        if p.get("published"):
            parts.append(f"   Published: {p['published']}")
        elif p.get("year"):
            parts.append(f"   Year: {p['year']}")

        if p.get("venue"):
            parts.append(f"   Venue: {p['venue']}")
        if p.get("categories"):
            categories = p["categories"]
            display_categories = ", ".join(categories[:3])
            if len(categories) > 3:
                display_categories += f" et al. ({len(categories)} categories)"
            parts.append(f"   Categories: {display_categories}")
        if p.get("citation_count") is not None and p["citation_count"] > 0:
            parts.append(f"   Citations: {p['citation_count']}")
        if p.get("publication_types"):
            parts.append(f"   Type: {', '.join(p['publication_types'])}")

        abstract = p.get("abstract", "")
        if abstract:
            parts.append(f"   Abstract: {abstract}")

        if p.get("pdf_url"):
            parts.append(f"   PDF: {p['pdf_url']}")
        if p.get("abs_url"):
            parts.append(f"   Page: {p['abs_url']}")

        lines.append("\n".join(parts))

    footer_lines = ["\n---"]
    footer_lines.append(
        '- Use `paper_fetch(paper_id="<arXiv ID or DOI>", mode="<metadata|full>")` '
        'only when needed: search results already include core metadata, so avoid '
        'calling `mode="metadata"` again unless specific missing fields are '
        'required; use `mode="full"` when full paper (content.md / images / '
        'layout, on-disk artifacts) is needed.'
    )
    if source == "arxiv":
        footer_lines.append(
            "- For non-arXiv papers (IEEE, ACM, ScienceDirect), "
            'use `paper_search(query="...", source="semantic_scholar")`.'
        )
    lines.append("\n".join(footer_lines))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


@tool(executor_timeout=130.0, description=PAPER_SEARCH_DESCRIPTION)
async def paper_search(
    query: str,
    source: Literal["arxiv", "semantic_scholar"] = "arxiv",
    max_results: int = 10,
) -> str:
    """Search for academic papers and return structured metadata.

    Searches academic paper databases and returns structured results
    including title, authors, abstract, and links.

    For arXiv papers, use source="arxiv" (default and recommended for arXiv IDs,
    preprints, and fast CS/AI literature scans). For papers from
    IEEE, ACM Digital Library, ScienceDirect, PubMed, or other
    publishers, use source="semantic_scholar" which indexes 200M+
    papers across all major academic databases.

    Args:
        query: A plain-text search query. For source='arxiv', alternatively an arXiv ID (e.g. "2301.07041") or arXiv structured syntax (e.g. 'au:bengio AND ti:attention').
        source: "arxiv" by default (recommended for preprints and CS/AI) or "semantic_scholar" for broader cross-publisher search (e.g., IEEE, ACM, ScienceDirect, PubMed, etc.).
        max_results: Number of results to return (1-30, default 10).
    """
    if not query.strip():
        return "Error: query cannot be empty"

    max_results = max(1, min(max_results, 30))

    if source == "arxiv":
        return await _search_arxiv(query, max_results)
    elif source == "semantic_scholar":
        cfg = resolve_paper_config(None)
        api_key = cfg.semantic_scholar_api_key
        return await _search_semantic_scholar(query, max_results, api_key)
    else:
        return (
            f"Error: unknown source {source!r}. "
            f"Use 'arxiv' for arXiv papers, or 'semantic_scholar' "
            f"for IEEE/ACM/ScienceDirect and other publishers."
        )
