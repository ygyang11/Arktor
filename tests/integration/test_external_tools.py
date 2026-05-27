"""Integration tests for external service tools with mocked APIs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_web_search_missing_api_key() -> None:
    """web_search returns error when API key is missing."""
    from agent_app.tools.web import web_search

    with patch("agent_app.tools.web.web_search.resolve_search_config") as mock_cfg:
        cfg = MagicMock()
        cfg.provider = "tavily"
        cfg.tavily_api_key = None
        cfg.serpapi_api_key = None
        mock_cfg.return_value = cfg

        result = await web_search.execute(query="test query")
        assert "Error" in result or "api" in result.lower()


@pytest.mark.asyncio
async def test_document_parser_empty_target() -> None:
    """document_parser returns error on empty target."""
    from agent_app.tools.document_parser import document_parser

    result = await document_parser.execute(target="")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_document_parser_whitespace_target() -> None:
    """document_parser returns error on whitespace-only target."""
    from agent_app.tools.document_parser import document_parser

    result = await document_parser.execute(target="   ")
    assert result.startswith("Error:")
