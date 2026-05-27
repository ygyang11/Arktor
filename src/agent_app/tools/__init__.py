"""Built-in tools for the agent application."""

from agent_app.tools.background import BACKGROUND_TOOLS, background_task
from agent_app.tools.document_parser import DOCUMENT_TOOLS, document_parser
from agent_app.tools.filesystem import (
    FILESYSTEM_TOOLS,
    READONLY_FILESYSTEM_TOOLS,
    WRITABLE_FILESYSTEM_TOOLS,
    edit_file,
    glob_files,
    grep_files,
    list_dir,
    read_file,
    write_file,
)
from agent_app.tools.memory import MEMORY_TOOLS, memory_tool
from agent_app.tools.paper import PAPER_TOOLS, paper_fetch, paper_search
from agent_app.tools.skill import SKILL_TOOLS, skill_tool
from agent_app.tools.sub_agent import SUB_AGENT_TOOLS, sub_agent
from agent_app.tools.terminal import TERMINAL_TOOLS, terminal_tool
from agent_app.tools.todo_write import TODO_TOOLS, todo_write
from agent_app.tools.web import WEB_TOOLS, web_fetch, web_search
from agent_harness.tool.base import BaseTool

BUILTIN_TOOLS: list[BaseTool] = [
    *TERMINAL_TOOLS,
    *WEB_TOOLS,
    *DOCUMENT_TOOLS,
    *PAPER_TOOLS,
    *MEMORY_TOOLS,
    *SKILL_TOOLS,
    *TODO_TOOLS,
    *SUB_AGENT_TOOLS,
    *BACKGROUND_TOOLS,
    *FILESYSTEM_TOOLS,
]

__all__ = [
    "BUILTIN_TOOLS",
    "FILESYSTEM_TOOLS",
    "READONLY_FILESYSTEM_TOOLS",
    "WRITABLE_FILESYSTEM_TOOLS",
    "TERMINAL_TOOLS",
    "WEB_TOOLS",
    "DOCUMENT_TOOLS",
    "PAPER_TOOLS",
    "SKILL_TOOLS",
    "TODO_TOOLS",
    "SUB_AGENT_TOOLS",
    "MEMORY_TOOLS",
    "BACKGROUND_TOOLS",
    "terminal_tool",
    "web_fetch",
    "web_search",
    "document_parser",
    "paper_search",
    "paper_fetch",
    "background_task",
    "memory_tool",
    "skill_tool",
    "todo_write",
    "sub_agent",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
]
