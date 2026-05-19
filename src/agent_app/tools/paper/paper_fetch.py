"""Academic paper fetching tool for metadata and full content retrieval."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlencode

from agent_harness.core.config import resolve_paper_config
from agent_harness.tool.decorator import tool
from agent_harness.utils.http_retry import HttpRetryConfig, http_get_with_retry
from agent_harness.utils.json_utils import parse_json_lenient
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperFetchConfig:
    max_full_tokens: int = 15_000
    html_fetch_timeout: int = 30


_CFG = PaperFetchConfig()

# full mode delegates to pdf_parser (URL path: no local upload). The
# executor ceiling must cover pdf_parser's poll budget + submission +
# the cheap pre-PDF probes
_FULL_EXECUTOR_TIMEOUT = 300.0 + 30.0 + 30.0


class _ArxivIdCheck(Enum):
    MISSING = "missing"    # arXiv returned 0 entries -> confirmed absent
    PRESENT = "present"    # >=1 entry -> exists
    UNKNOWN = "unknown"    # _fetch_xml raised -> transient, undecidable

_S2_DETAIL_FIELDS = (
    "paperId,title,authors,abstract,year,venue,"
    "externalIds,openAccessPdf,citationCount,referenceCount,"
    "fieldsOfStudy,publicationDate,publicationTypes,tldr,journal"
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
        err = str(exc)
        if err.startswith("arXiv request failed:"):
            return f"Error: {err}"
        return f"Error: arXiv request failed: {err}"

    entries = root.findall(f"{ns}entry")
    if not entries:
        return f"Error: no arXiv paper found for ID: {clean_id}"

    paper = _parse_arxiv_entry(entries[0])
    return _format_metadata(paper)


# ---------------------------------------------------------------------------
# Semantic Scholar metadata
# ---------------------------------------------------------------------------


async def _fetch_s2_metadata(paper_id: str, api_key: str | None) -> str:
    from agent_app.tools.paper.paper_search import (
        _parse_s2_paper,
    )

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
        return (
            f"Error: Semantic Scholar API returned HTTP {status}. Detail: {body[:200]}"
        )

    try:
        data_raw = parse_json_lenient(body)
    except ValueError as exc:
        return f"Error: failed to parse Semantic Scholar response: {exc}"
    if not isinstance(data_raw, dict):
        return "Error: unexpected Semantic Scholar response format"
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


async def _arxiv_id_check(clean_id: str) -> _ArxivIdCheck:
    """Pure existence probe: distinguishes MISSING vs transient UNKNOWN.

    Reuses paper_search._fetch_xml; does not format metadata. Only
    MISSING is a safe fast-fail signal -- UNKNOWN must still try the PDF.
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
    paper_id: str, source: str, api_key: str | None
) -> str:
    if source == "arxiv":
        from agent_app.tools.paper.paper_search import _normalize_arxiv_id

        clean_id = _normalize_arxiv_id(paper_id)

        # 1) HTML first -- a hit costs no precheck (the common modern path).
        html_result = await _try_arxiv_html(clean_id)
        if html_result is not None:
            return truncate_text_by_tokens(
                html_result,
                max_tokens=_CFG.max_full_tokens,
                suffix="\n... (truncated)",
            )

        # 2) HTML miss -- only a confirmed-missing ID fast-fails (skips the
        #    expensive PDF cloud parser). UNKNOWN/PRESENT fall through.
        if await _arxiv_id_check(clean_id) is _ArxivIdCheck.MISSING:
            return f"Error: no arXiv paper found for ID: {clean_id}"

        # 3) PDF; pdf_parser already carries a metadata hint on failure,
        #    so its result is returned verbatim (no soft-landing).
        #    No `.pdf` suffix: arxiv.org/pdf/<id>.pdf 301-redirects with a
        #    text/html body that PaddleOCR's URL fetcher rejects as
        #    "unsupported format"; the bare URL serves application/pdf.
        pdf_url = f"https://arxiv.org/pdf/{clean_id}"
        return await _fetch_via_pdf_parser(pdf_url, paper_id=clean_id)

    pdf_url = await _resolve_pdf_url(paper_id, source, api_key)
    if pdf_url.startswith("Error:"):
        return pdf_url
    return await _fetch_via_pdf_parser(pdf_url)


async def _try_arxiv_html(arxiv_id: str) -> str | None:
    from agent_app.tools.paper.paper_search import _USER_AGENT as _ARXIV_UA

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        status, body = await http_get_with_retry(
            html_url,
            headers={"User-Agent": _ARXIV_UA},
            timeout=_CFG.html_fetch_timeout,
            retry=HttpRetryConfig(max_attempts=3, base_delay=1.0),
        )
    except Exception:
        return None

    if status != 200:
        return None

    lowered = body.lower()
    if "<html" not in lowered and "<!doctype html" not in lowered:
        return None

    from agent_app.tools.web.web_fetch import _extract_text_from_html

    text = _extract_text_from_html(body)
    if len(text.strip()) < 500:
        return None
    return text


async def _fetch_via_pdf_parser(
    pdf_url: str, paper_id: str = ""
) -> str:
    from agent_app.tools.pdf_parser import pdf_parser as _pdf_parser

    result = await _pdf_parser.execute(url=pdf_url)

    if result.startswith("Error:"):
        reason = result.removeprefix("Error:").strip()
        msg = (
            f"Error: failed to extract full content from PDF ({pdf_url}). \n"
            f"Reason: {reason}"
        )
        if paper_id:
            msg += (
                "\nMetadata (title, authors, abstract) is still available "
                'via paper_fetch(mode="metadata").'
            )
        return msg

    return truncate_text_by_tokens(
        result, max_tokens=_CFG.max_full_tokens, suffix="\n... (truncated)"
    )


async def _resolve_pdf_url(
    paper_id: str, source: str, api_key: str | None
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
    except Exception as exc:
        return f"Error: cannot resolve PDF URL: {exc}"

    if status == 404:
        return f"Error: paper not found: {paper_id}"
    if status != 200:
        return f"Error: cannot resolve PDF URL (HTTP {status})"

    try:
        data_raw = parse_json_lenient(body)
    except ValueError as exc:
        return f"Error: failed to parse Semantic Scholar response: {exc}"
    if not isinstance(data_raw, dict):
        return "Error: unexpected Semantic Scholar response format"
    data = data_raw

    pdf_info: dict[str, str] = data.get("openAccessPdf") or {}
    oa_pdf_url: str = pdf_info.get("url", "")
    if oa_pdf_url:
        return oa_pdf_url

    external: dict[str, str] = data.get("externalIds") or {}
    arxiv_id = external.get("ArXiv")
    if arxiv_id:
        # Bare URL (no `.pdf`): the suffixed form 301-redirects to a
        # text/html body that PaddleOCR's URL fetcher rejects.
        return f"https://arxiv.org/pdf/{arxiv_id}"

    doi = external.get("DOI")
    if doi:
        return await _try_unpaywall(doi)

    return (
        "Error: no open access PDF available for this paper. "
        "The paper may require institutional access or subscription."
    )


async def _try_unpaywall(doi: str) -> str:
    url = f"https://api.unpaywall.org/v2/{doi}?email=agent-harness@example.com"
    try:
        status, body = await http_get_with_retry(
            url,
            timeout=15,
            retry=HttpRetryConfig(max_attempts=3, base_delay=1.0),
        )
    except Exception:
        return f"Error: failed to query Unpaywall for DOI {doi}"
    if status != 200:
        return f"Error: no open access PDF found for DOI {doi}"

    try:
        data_raw = parse_json_lenient(body)
    except ValueError as exc:
        return f"Error: failed to parse Unpaywall response for DOI {doi}: {exc}"
    if not isinstance(data_raw, dict):
        return f"Error: unexpected Unpaywall response format for DOI {doi}"
    data = data_raw

    best: dict[str, str] = data.get("best_oa_location") or {}
    oa_url: str = best.get("url_for_pdf", "")
    if oa_url:
        return oa_url

    return f"Error: no open access PDF found for DOI {doi}"


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


@tool(executor_timeout=_FULL_EXECUTOR_TIMEOUT)
async def paper_fetch(
    paper_id: str,
    mode: Literal["metadata", "full"] = "metadata",
    source: Literal["arxiv", "semantic_scholar"] = "arxiv",
) -> str:
    """Fetch a specific paper by ID, returning full text or metadata.

    Two modes: "metadata" returns detailed structured info (title,
    authors, full abstract, citations, fields of study); "full"
    returns the complete paper content from HTML or PDF.

    Args:
        paper_id: arXiv ID like "2301.07041", or DOI:/ARXIV:/ACM:-prefixed ID for S2.
        mode: "full" returns raw paper body text for downstream analysis/summarization, "metadata" returns concise paper info (e.g., title, authors, abstract, year/venue, IDs, citation/reference stats) (default).
        source: "arxiv" (default) or "semantic_scholar" for IEEE/ACM/ScienceDirect etc.
    """
    if not paper_id.strip():
        return "Error: paper_id cannot be empty"

    if mode not in ("metadata", "full"):
        return (
            f"Error: unknown mode {mode!r}. "
            f"Use 'metadata' for structured info or 'full' for complete content."
        )

    if source not in ("arxiv", "semantic_scholar"):
        return (
            f"Error: unknown source {source!r}. "
            f"Use 'arxiv' or 'semantic_scholar'."
        )

    cfg = resolve_paper_config(None)

    if mode == "metadata":
        if source == "arxiv":
            return await _fetch_arxiv_metadata(paper_id)
        return await _fetch_s2_metadata(paper_id, cfg.semantic_scholar_api_key)

    return await _fetch_full_content(paper_id, source, cfg.semantic_scholar_api_key)
