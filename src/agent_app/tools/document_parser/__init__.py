from agent_app.tools.document_parser.document_parser import (
    DocumentParserTool,
    document_parser,
    parse_document,
)
from agent_harness.tool.base import BaseTool

DOCUMENT_TOOLS: list[BaseTool] = [document_parser]

__all__ = [
    "DOCUMENT_TOOLS",
    "DocumentParserTool",
    "document_parser",
    "parse_document",
]
