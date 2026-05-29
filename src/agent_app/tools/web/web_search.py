"""Web search tool for discovering information via search engines."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agent_harness.core.config import resolve_search_config
from agent_harness.core.errors import ToolValidationError
from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchConfig:
    """Configuration for the web_search tool."""

    max_snippet_tokens: int = 500
    executor_timeout: float = 60.0


_CFG = WebSearchConfig()


def _format_result(index: int, title: str, snippet: str, url: str) -> str:
    return f"{index}. {title}\n   [snippet] {snippet}\n   URL: {url}"


def _format_search_results(results: list[tuple[str, str, str]]) -> str:
    lines = [
        _format_result(i, title, snippet, url)
        for i, (title, snippet, url) in enumerate(results, 1)
    ]
    lines.append(
        "\n---\n"
        "Use `web_fetch` tool to read the full page content "
        "if needed and the tool is available."
    )
    return "\n\n".join(lines)


async def _search_tavily(query: str, max_results: int, api_key: str) -> str:
    """Search using Tavily API."""
    try:
        from tavily import AsyncTavilyClient  # noqa: PLC0415
    except ImportError:
        return "Error: tavily-python is not installed. Run `pip install tavily-python`."

    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(query=query, max_results=max_results)

    raw_results = response.get("results", [])
    if not raw_results:
        return f"No results found for: {query}"

    results: list[tuple[str, str, str]] = []
    for r in raw_results:
        title = r.get("title", "No title")
        snippet = truncate_text_by_tokens(
            r.get("content", ""),
            max_tokens=_CFG.max_snippet_tokens,
            suffix="...",
        )
        url = r.get("url", "")
        results.append((title, snippet, url))
    return _format_search_results(results)


async def _search_serpapi(query: str, max_results: int, api_key: str) -> str:
    """Search using SerpAPI."""
    try:
        from serpapi import GoogleSearch  # noqa: PLC0415
    except ImportError:
        return "Error: google-search-results is not installed. Run `pip install google-search-results`."

    def _do_search() -> dict[str, list[dict[str, str]]]:
        search = GoogleSearch({"q": query, "num": max_results, "api_key": api_key})
        result: dict[str, list[dict[str, str]]] = search.get_dict()
        return result

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_search)
    organic = result.get("organic_results", [])
    if not organic:
        return f"No results found for: {query}"

    results: list[tuple[str, str, str]] = []
    for r in organic[:max_results]:
        title = r.get("title", "No title")
        snippet = truncate_text_by_tokens(
            r.get("snippet", ""),
            max_tokens=_CFG.max_snippet_tokens,
            suffix="...",
        )
        url = r.get("link", "")
        results.append((title, snippet, url))
    return _format_search_results(results)


@tool(executor_timeout=_CFG.executor_timeout)
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return result snippets with URLs.

    Returns ranked search result snippets (not full page content).
    Use web_fetch to retrieve the full content of a specific result.

    Args:
        query: The search query string.
        max_results: Number of results to return (1-20, default 5).
    """
    if not query.strip():
        raise ToolValidationError("query cannot be empty")
    max_results = max(1, min(max_results, 20))

    cfg = resolve_search_config(None)
    provider = cfg.provider

    if provider == "tavily":
        api_key = cfg.tavily_api_key or ""
        if not api_key:
            return "Error: web search is not configured (no provider API key set)."
        return await _search_tavily(query, max_results, api_key)
    elif provider == "serpapi":
        api_key = cfg.serpapi_api_key or ""
        if not api_key:
            return "Error: web search is not configured (no provider API key set)."
        return await _search_serpapi(query, max_results, api_key)
    else:
        return "Error: web search provider is misconfigured."
