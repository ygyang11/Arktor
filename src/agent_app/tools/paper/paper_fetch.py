"""Academic paper fetching tool for metadata and full content retrieval."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from urllib.parse import urlencode

from agent_app.tools.document_parser import parse_document
from agent_harness.core.config import resolve_paper_config
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.utils.http_retry import HttpRetryConfig, http_get_with_retry
from agent_harness.utils.json_utils import parse_json_lenient

logger = logging.getLogger(__name__)


PAPER_FETCH_DESCRIPTION = (
    "Fetch a specific paper by ID and source, returning either structured "
    "metadata or parsed content of the full paper.\n\n"
    "## Modes\n"
    "- `metadata` (default): return detailed structured info — title, authors, "
    "full abstract, citations, fields of study, etc.\n"
    "- `full`: return the paper's full text, parsed from its open-access PDF "
    "(routes through `document_parser` tool) into on-disk artifacts — `content.md` (body), "
    "`images/` (figures), `layout.json`/`manifest.json` (rarely need to be read); the response "
    "lists the paths. On failure the response explains why — e.g. no open-access PDF "
    "or a parsing failure (per-tier trail)."
)

_FULL_EXECUTOR_TIMEOUT = 720.0 + 30.0

_S2_DETAIL_FIELDS = (
    "paperId,title,authors,abstract,year,venue,"
    "externalIds,openAccessPdf,citationCount,referenceCount,"
    "fieldsOfStudy,publicationDate,publicationTypes,tldr,journal"
)

_OA_LOOKUP_FAILED = (
    "Error: couldn't look up this paper's open-access availability "
    "(temporary failure); retry shortly."
)
_NO_OA_PDF = (
    "Error: no open-access PDF is available for this paper; it appears "
    "paywalled or access-restricted. Full text can't be retrieved"
)


# ---------------------------------------------------------------------------
# Metadata formatting (shared by arXiv and S2)
# ---------------------------------------------------------------------------


def _format_metadata(paper: dict[str, Any]) -> str:
    lines = [f"# {paper.get('title', 'Untitled')}"]

    if paper.get("arxiv_id"):
        lines.append(f"**arXiv ID**: {paper['arxiv_id']}")
    if paper.get("s2_id"):
        lines.append(f"**S2 ID**: {paper['s2_id']}")
    if paper.get("doi"):
        lines.append(f"**DOI**: {paper['doi']}")

    authors = paper.get("authors", [])
    if authors:
        lines.append(f"**Authors**: {', '.join(authors)}")

    if paper.get("publication_date"):
        lines.append(f"**Publication Date**: {paper['publication_date']}")
    elif paper.get("published"):
        lines.append(f"**Published**: {paper['published']}")
    elif paper.get("year"):
        lines.append(f"**Year**: {paper['year']}")

    if paper.get("venue"):
        lines.append(f"**Venue**: {paper['venue']}")
    if paper.get("categories"):
        lines.append(f"**Categories**: {', '.join(paper['categories'])}")
    if paper.get("fields_of_study"):
        lines.append(f"**Fields of Study**: {', '.join(paper['fields_of_study'])}")
    if paper.get("publication_types"):
        lines.append(f"**Type**: {', '.join(paper['publication_types'])}")
    if paper.get("citation_count") is not None:
        lines.append(f"**Citations**: {paper['citation_count']}")
    if paper.get("reference_count") is not None:
        lines.append(f"**References**: {paper['reference_count']}")

    if paper.get("abstract"):
        lines.append(f"\n## Abstract\n\n{paper['abstract']}")

    if paper.get("tldr"):
        lines.append(f"\n## TL;DR\n\n{paper['tldr']}")

    if paper.get("pdf_url"):
        lines.append(f"\n**PDF**: {paper['pdf_url']}")
    if paper.get("abs_url"):
        lines.append(f"**Page**: {paper['abs_url']}")

    identifier = paper.get("arxiv_id") or paper.get("doi") or paper.get("s2_id") or ""
    if identifier:
        source_hint = "arxiv" if paper.get("arxiv_id") else "semantic_scholar"
        lines.append(
            f"\n---\nTo get the full paper content, use: "
            f'`paper_fetch(paper_id="{identifier}", mode="full", source="{source_hint}")`'
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# arXiv metadata
# ---------------------------------------------------------------------------


async def _fetch_arxiv_metadata(arxiv_id: str) -> str:
    from agent_app.tools.paper.paper_search import (
        _fetch_xml,
        _normalize_arxiv_id,
        _parse_arxiv_entry,
    )

    clean_id = _normalize_arxiv_id(arxiv_id)
    ns = "{http://www.w3.org/2005/Atom}"
    url = f"http://export.arxiv.org/api/query?{urlencode({'id_list': clean_id})}"
    try:
        root = await _fetch_xml(url)
    except Exception as exc:
        return f"Error: {exc}"

    entries = root.findall(f"{ns}entry")
    if not entries:
        return f"Error: no arXiv paper found for ID: {clean_id}"

    paper = _parse_arxiv_entry(entries[0])
    return _format_metadata(paper)


# ---------------------------------------------------------------------------
# Semantic Scholar metadata
# ---------------------------------------------------------------------------


async def _fetch_s2_metadata(paper_id: str, api_key: str | None) -> str:
    from agent_app.tools.paper.paper_search import _parse_s2_paper

    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/"
        f"{paper_id}?fields={_S2_DETAIL_FIELDS}"
    )

    extra_headers: dict[str, str] = {}
    if api_key:
        extra_headers["x-api-key"] = api_key

    try:
        status, body = await http_get_with_retry(
            url,
            headers=extra_headers,
            retry=HttpRetryConfig(max_attempts=3, base_delay=1.0),
        )
    except Exception as exc:
        return f"Error: Semantic Scholar request failed: {exc}"

    if status == 404:
        return (
            f"Error: paper not found: {paper_id}. "
            f"Supported ID formats: DOI:10.xxx, ARXIV:2301.xxxxx, "
            f"ACM:xxx, PMID:xxx, CorpusId:xxx, or S2 paper ID."
        )
    if status == 429:
        return (
            "Error: Semantic Scholar rate limit exceeded after retries. "
            "Try again later, or use source='arxiv' if the paper is on arXiv."
        )
    if status != 200:
        return f"Error: Semantic Scholar API returned HTTP {status}."

    try:
        data_raw = parse_json_lenient(body)
    except ValueError:
        return "Error: Semantic Scholar returned an unreadable response; retry shortly."
    if not isinstance(data_raw, dict):
        return "Error: Semantic Scholar returned an unreadable response; retry shortly."
    data = data_raw

    paper = _parse_s2_paper(data)
    paper["publication_date"] = data.get("publicationDate", "")
    paper["reference_count"] = data.get("referenceCount", 0)
    paper["fields_of_study"] = data.get("fieldsOfStudy") or []
    paper["publication_types"] = data.get("publicationTypes") or []
    tldr = data.get("tldr") or {}
    paper["tldr"] = tldr.get("text", "")
    journal = data.get("journal") or {}
    if journal.get("name"):
        paper["venue"] = paper.get("venue") or journal["name"]

    return _format_metadata(paper)


# ---------------------------------------------------------------------------
# Full content retrieval
# ---------------------------------------------------------------------------


class _ArxivIdCheck(Enum):
    MISSING = "missing"
    PRESENT = "present"
    UNKNOWN = "unknown"


async def _arxiv_id_check(clean_id: str) -> _ArxivIdCheck:
    """Cheap existence probe. Only MISSING is a safe fast-fail signal —
    UNKNOWN (network / API failure) must fall through to the parser so a
    valid paper is never blocked on a probe failure.
    """
    from agent_app.tools.paper.paper_search import _fetch_xml

    ns = "{http://www.w3.org/2005/Atom}"
    url = f"http://export.arxiv.org/api/query?{urlencode({'id_list': clean_id})}"
    try:
        root = await _fetch_xml(url)
    except Exception:
        return _ArxivIdCheck.UNKNOWN
    return (
        _ArxivIdCheck.PRESENT
        if root.findall(f"{ns}entry")
        else _ArxivIdCheck.MISSING
    )


async def _fetch_full_content(
    paper_id: str, source: str, api_key: str | None,
    *, session_id: str | None,
) -> str:
    if source == "arxiv":
        from agent_app.tools.paper.paper_search import _normalize_arxiv_id
        clean_id = _normalize_arxiv_id(paper_id)

        if await _arxiv_id_check(clean_id) is _ArxivIdCheck.MISSING:
            return f"Error: no arXiv paper found for ID: {clean_id}"

        pdf_url = f"https://arxiv.org/pdf/{clean_id}"
        return await parse_document(
            target=pdf_url,
            session_id=session_id,
            slug_hint=f"{source}-{clean_id}",
        )

    pdf_url = await _resolve_pdf_url(paper_id, source, api_key)
    if pdf_url.startswith("Error:"):
        return pdf_url
    return await parse_document(
        target=pdf_url,
        session_id=session_id,
        slug_hint=f"{source}-{paper_id}",
    )


async def _resolve_pdf_url(
    paper_id: str, source: str, api_key: str | None,
) -> str:
    if source not in ("arxiv", "semantic_scholar"):
        return f"Error: unknown source {source!r}. Use 'arxiv' or 'semantic_scholar'."

    fields = "openAccessPdf,externalIds"
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/"
        f"{paper_id}?fields={fields}"
    )

    extra_headers: dict[str, str] = {}
    if api_key:
        extra_headers["x-api-key"] = api_key

    try:
        status, body = await http_get_with_retry(
            url,
            headers=extra_headers,
            retry=HttpRetryConfig(max_attempts=3, base_delay=1.0),
        )
    except Exception:
        return _OA_LOOKUP_FAILED

    if status == 404:
        return f"Error: paper not found: {paper_id}."
    if status != 200:
        return _OA_LOOKUP_FAILED

    try:
        data_raw = parse_json_lenient(body)
    except ValueError:
        return _OA_LOOKUP_FAILED
    if not isinstance(data_raw, dict):
        return _OA_LOOKUP_FAILED
    data = data_raw

    pdf_info: dict[str, Any] = data.get("openAccessPdf") or {}
    oa_pdf_url: str = pdf_info.get("url") or ""
    if oa_pdf_url:
        return oa_pdf_url

    external: dict[str, Any] = data.get("externalIds") or {}
    arxiv_id: str = external.get("ArXiv") or ""
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"

    doi: str = external.get("DOI") or ""
    if doi:
        return await _resolve_oa_via_openalex(doi)

    return _NO_OA_PDF


async def _resolve_oa_via_openalex(doi: str) -> str:
    """Resolve a DOI to an open-access PDF URL via OpenAlex.
    OpenAlex's single-entity lookup is free, unlimited, and keyless
    """
    url = f"https://api.openalex.org/works/doi:{doi}"
    try:
        status, body = await http_get_with_retry(
            url,
            timeout=15,
            retry=HttpRetryConfig(max_attempts=3, base_delay=1.0),
        )
    except Exception:
        return _OA_LOOKUP_FAILED
    if status == 404:
        return _NO_OA_PDF
    if status != 200:
        return _OA_LOOKUP_FAILED

    try:
        data_raw = parse_json_lenient(body)
    except ValueError:
        return _OA_LOOKUP_FAILED
    if not isinstance(data_raw, dict):
        return _OA_LOOKUP_FAILED
    data = data_raw

    best: dict[str, Any] = data.get("best_oa_location") or {}
    best_pdf: str = best.get("pdf_url") or ""
    if best_pdf:
        return best_pdf

    for loc in data.get("locations") or []:
        if isinstance(loc, dict) and loc.get("is_oa") and loc.get("pdf_url"):
            return str(loc["pdf_url"])

    return _NO_OA_PDF


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


class PaperFetchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="paper_fetch",
            description=PAPER_FETCH_DESCRIPTION,
            executor_timeout=_FULL_EXECUTOR_TIMEOUT,
        )
        self._session_id: str | None = None

    def bind_session(self, session_id: str | None) -> None:
        self._session_id = session_id

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": (
                            "arXiv ID like '2301.07041', or "
                            "DOI:/ARXIV:/ACM:-prefixed ID for S2."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["metadata", "full"],
                        "description": (
                            "'full' for parsed paper content (on-disk artifact "
                            "paths); 'metadata' for concise paper info (title, "
                            "authors, abstract, IDs, citations, etc.) (default)."
                        ),
                        "default": "metadata",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["arxiv", "semantic_scholar"],
                        "description": (
                            "'arxiv' (default) or 'semantic_scholar' for "
                            "IEEE/ACM/ScienceDirect etc."
                        ),
                        "default": "arxiv",
                    },
                },
                "required": ["paper_id"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        paper_id: str = kwargs.get("paper_id") or ""
        mode: str = kwargs.get("mode") or "metadata"
        source: str = kwargs.get("source") or "arxiv"

        if not paper_id.strip():
            raise ToolValidationError("paper_id cannot be empty")
        if mode not in ("metadata", "full"):
            raise ToolValidationError(
                f"unknown mode {mode!r}. "
                f"Use 'metadata' for structured info or 'full' for complete content."
            )
        if source not in ("arxiv", "semantic_scholar"):
            raise ToolValidationError(
                f"unknown source {source!r}. Use 'arxiv' or 'semantic_scholar'."
            )

        cfg = resolve_paper_config(None)

        if mode == "metadata":
            if source == "arxiv":
                return await _fetch_arxiv_metadata(paper_id)
            return await _fetch_s2_metadata(paper_id, cfg.semantic_scholar_api_key)

        return await _fetch_full_content(
            paper_id, source, cfg.semantic_scholar_api_key,
            session_id=self._session_id,
        )


paper_fetch = PaperFetchTool()
