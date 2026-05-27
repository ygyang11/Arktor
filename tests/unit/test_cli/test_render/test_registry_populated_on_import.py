from agent_cli.render.tool_display import (
    _DISPLAY_NAMES,
    _EXPANDERS,
    _RESULT_FORMATTERS,
    SUPPRESSED_IN_ROW,
)

_EXPECTED_FORMATTERS = {
    "read_file", "write_file", "edit_file",
    "glob_files", "grep_files", "list_dir",
    "terminal_tool",
    "web_search", "web_fetch",
    "document_parser",
    "paper_search", "paper_fetch",
    "memory_tool", "skill_tool",
    "background_task",
}
_EXPECTED_EXPANDERS = {
    "write_file", "edit_file",
    "glob_files", "grep_files", "list_dir",
    "terminal_tool",
    "web_search",
    "paper_search", "paper_fetch",
    "memory_tool",
    "background_task",
}


def test_formatters_registered() -> None:
    assert _EXPECTED_FORMATTERS.issubset(_RESULT_FORMATTERS.keys())


def test_expanders_registered() -> None:
    assert _EXPECTED_EXPANDERS.issubset(_EXPANDERS.keys())


def test_every_registered_tool_has_display_name_or_is_suppressed() -> None:
    # Guard against silently showing raw snake_case IDs when a tool is added
    # to the framework but its display mapping is forgotten.
    missing = {
        name for name in _RESULT_FORMATTERS
        if name not in _DISPLAY_NAMES and name not in SUPPRESSED_IN_ROW
    }
    assert not missing, f"tools missing display-name mapping: {sorted(missing)}"


def test_no_orphan_display_names() -> None:
    # Every display-name entry must correspond to a real registered tool.
    orphans = {
        name for name in _DISPLAY_NAMES
        if name not in _RESULT_FORMATTERS
    }
    assert not orphans, f"display-name entries without formatter: {sorted(orphans)}"
