"""Per-tool formatter and expander implementations.

Registered into ``tool_display._RESULT_FORMATTERS`` / ``_EXPANDERS`` at import
time via the ``@register_result_formatter`` / ``@register_expander`` decorators.
``tool_display.py`` imports this module at its bottom so registration fires
whenever the display module is loaded.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlparse

from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from agent_cli.render._code_highlight import (
    detect_lexer_name,
    highlight_code,
    make_highlighter,
)
from agent_cli.render.tool_display import (
    _is_error_result,
    _ToolRow,
    _truncate,
    register_expander,
    register_result_formatter,
)
from agent_cli.theme import SEP_DOT, SEP_ELLIPSIS
from agent_harness.core.message import ToolCall, ToolResult
from agent_harness.utils.media import human_size as _human_size

_PREVIEW_CAP = 10
_INDENT = "    "
_WRITE_DIFF_MAX_LINES = 100

_READ_HEADER_RE = re.compile(r"^\[(.+?)\] lines (\d+)-(\d+) of (\d+)")
_WRITE_HEADER_RE = re.compile(r"^Created (.+?) \((\d+) lines\)\s*$")
_EDIT_HEADER_RE = re.compile(r"^Edited (.+?) \((\d+) replacements?\)")
_GLOB_HEADER_RE = re.compile(r"^(\d+) files matching '([^']+)'")
_GREP_HEADER_RE = re.compile(r"^(\d+) matches in (\d+) files")
_LIST_HEADER_RE = re.compile(r"^(.+)/  \((\d+) entries\)")
_TERM_EXIT_RE = re.compile(r"^\[exit code (\d+|N/A)\]")
_TERM_BG_RE = re.compile(r"^Background command (\S+) started:")
_DOC_BG_RE = re.compile(r"^Background document_parser (\S+) started:")
_PAPER_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_DOC_FORMAT_RE = re.compile(r"format\s*:\s*(\w+)(?:\s*\((\d+)\s*pages?)?")
_DOC_IMAGES_RE = re.compile(r"\((\d+)\s*figures?\)")
_SKILL_LOADED_RE = re.compile(
    r'<skill-loaded name="[^"]+">\n(.*)\n</skill-loaded>', re.DOTALL
)
_DIFF_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

_DIFF_GUTTER_DIGITS = 4
_DIFF_GUTTER_SEP = " │ "
_DIFF_RIGHT_PAD = 4


def _plural(n: int, s: str, p: str | None = None) -> str:
    return s if n == 1 else (p or s + "s")


def _cell_ljust(s: str, width: int) -> str:
    return s + " " * max(width - cell_len(s), 0)


def _print_body_line(console: Console, line: str) -> None:
    console.print(_INDENT + line, markup=False, highlight=False, no_wrap=True, overflow="ellipsis")


def _print_body_lines(
    console: Console,
    lines: Iterable[str],
    cap: int = _PREVIEW_CAP,
    more_label: str = "lines",
) -> None:
    ls = list(lines)
    for line in ls[:cap]:
        _print_body_line(console, line)
    if len(ls) > cap:
        console.print(
            f"{_INDENT}[muted]{SEP_ELLIPSIS} {len(ls) - cap} more {more_label}[/muted]"
        )


def _print_body_head_tail(
    console: Console,
    lines: Iterable[str],
    head: int = 3,
    tail: int = 5,
    more_label: str = "lines",
) -> None:
    ls = list(lines)
    if len(ls) <= head + tail:
        for line in ls:
            _print_body_line(console, line)
        return
    for line in ls[:head]:
        _print_body_line(console, line)
    middle = len(ls) - head - tail
    console.print(
        f"{_INDENT}[muted]{SEP_ELLIPSIS} {middle} more {more_label}[/muted]"
    )
    for line in ls[-tail:]:
        _print_body_line(console, line)


def _render_diff_lines(
    console: Console,
    diff: str,
    lexer_name: str | None = None,
    indent: int = 3,
) -> None:
    pad = " " * indent
    gutter_w = _DIFF_GUTTER_DIGITS + len(_DIFF_GUTTER_SEP)
    right_pad_s = " " * _DIFF_RIGHT_PAD
    content_w = max(
        console.size.width - indent - gutter_w - _DIFF_RIGHT_PAD, 1,
    )
    full_bar_w = content_w + gutter_w

    highlighter = make_highlighter(lexer_name)

    old_ln: int | None = None
    new_ln: int | None = None

    def _emit_meta(body: str) -> None:
        line = Text()
        line.append(pad)
        line.append(body, style="diff_meta")
        console.print(line, soft_wrap=True, overflow="ellipsis")

    for raw in diff.splitlines():
        if not raw:
            continue

        if raw.startswith(("+++", "---")):
            continue

        m = _DIFF_HUNK_RE.match(raw)
        if m:
            if new_ln is not None:
                console.print()
            old_ln, new_ln = int(m[1]), int(m[2])
            line = Text()
            line.append(pad)
            line.append(_cell_ljust(raw, full_bar_w), style="diff_hunk")
            line.append(right_pad_s)
            console.print(line, soft_wrap=True, overflow="ellipsis")
            continue

        # write_file pseudo-diff: no @@ header, synthesize counters from 1
        if new_ln is None:
            old_ln, new_ln = 1, 1

        first = raw[0]
        if first == "+":
            gutter = f"{new_ln:>{_DIFF_GUTTER_DIGITS}}{_DIFF_GUTTER_SEP}"
            style = "diff_add"
            gutter_style = "success"
            new_ln += 1
        elif first == "-":
            gutter = f"{old_ln:>{_DIFF_GUTTER_DIGITS}}{_DIFF_GUTTER_SEP}"
            style = "diff_remove"
            gutter_style = "error"
            old_ln += 1
        elif first == " ":
            gutter = f"{new_ln:>{_DIFF_GUTTER_DIGITS}}{_DIFF_GUTTER_SEP}"
            style = ""
            gutter_style = "muted"
            old_ln += 1
            new_ln += 1
        else:
            # "\ No newline at end of file", truncation marker, etc.
            _emit_meta(raw)
            continue

        content = Text(first, style=style)
        content.append_text(highlight_code(highlighter, raw[1:]))
        pad_n = max(content_w - cell_len(content.plain), 0)
        if pad_n:
            content.append(" " * pad_n, style=style)

        line = Text()
        line.append(pad)
        line.append(gutter, style=gutter_style)
        line.append_text(content)
        line.append(right_pad_s)
        console.print(line, soft_wrap=True, overflow="ellipsis")


@register_result_formatter("read_file")
def _fmt_read(tc: ToolCall, r: ToolResult) -> str:
    if r.attachments:
        att = r.attachments[0]
        kind = "PDF" if att.mime == "application/pdf" else "image"
        return f"Attached {kind} ({_human_size(att.size)})"
    c = r.content
    if c.startswith("(empty"):
        return "Empty file"
    m = _READ_HEADER_RE.match(c)
    if not m:
        return "Read file"
    start, end, total = int(m[2]), int(m[3]), int(m[4])
    shown = end - start + 1
    return (
        f"Read {shown} lines totally"
        if shown == total
        else f"Read lines {start}-{end} of {total}"
    )


@register_result_formatter("write_file")
def _fmt_write(tc: ToolCall, r: ToolResult) -> str:
    m = _WRITE_HEADER_RE.match(r.content)
    if not m:
        return "Wrote file"
    n = int(m[2])
    return f"Wrote {n} {_plural(n, 'line')}"


@register_result_formatter("edit_file")
def _fmt_edit(tc: ToolCall, r: ToolResult) -> str:
    m = _EDIT_HEADER_RE.match(r.content)
    if not m:
        return "Edited file"
    n = int(m[2])
    return f"Edited file ({n} {_plural(n, 'replacement')})"


@register_result_formatter("glob_files")
def _fmt_glob(tc: ToolCall, r: ToolResult) -> str:
    c = r.content
    if c.startswith("No files matching"):
        return "No matches"
    m = _GLOB_HEADER_RE.match(c)
    if not m:
        return "Globbed"
    n = int(m[1])
    return f"Found {n} {_plural(n, 'file')} matching"


@register_result_formatter("grep_files")
def _fmt_grep(tc: ToolCall, r: ToolResult) -> str:
    c = r.content
    if c.startswith("No matches"):
        return "No matches"
    m = _GREP_HEADER_RE.match(c)
    if not m:
        return "Grepped"
    n, f = int(m[1]), int(m[2])
    return f"Found {n} {_plural(n, 'match', 'matches')} across {f} {_plural(f, 'file')}"


@register_result_formatter("list_dir")
def _fmt_ls(tc: ToolCall, r: ToolResult) -> str:
    c = r.content
    if "(empty directory)" in c:
        return "Empty directory"
    m = _LIST_HEADER_RE.match(c)
    return f"{m[2]} entries in {m[1]}/" if m else "Listed directory"


@register_result_formatter("terminal_tool")
def _fmt_term(tc: ToolCall, r: ToolResult) -> str:
    c = r.content
    if tc.arguments.get("background"):
        m = _TERM_BG_RE.match(c)
        return f"Backgrounded: {m[1]}" if m else "Backgrounded"
    m = _TERM_EXIT_RE.match(c)
    code = m[1] if m else "0"
    body = c[m.end():].lstrip() if m else c
    # Prefer last non-empty line: pytest "N passed", npm "added N packages",
    lines = [s for s in body.splitlines() if s.strip()]
    def sep_only(s: str) -> bool:
        return bool(s) and all(ch in "=-_*#" or ch.isspace() for ch in s)
    lines = [s for s in lines if not sep_only(s)]
    picked = lines[-1].strip() if lines else "(no output)"
    return f"exit {code} {SEP_DOT} {_truncate(picked, 50)}"


@register_result_formatter("web_search")
def _fmt_websearch(tc: ToolCall, r: ToolResult) -> str:
    c = r.content
    if c.startswith("No results"):
        return "No results"
    n = len(re.findall(r"^\d+\. ", c, re.MULTILINE))
    return f'Found {n} {_plural(n, "result")}'


@register_result_formatter("web_fetch")
def _fmt_webfetch(tc: ToolCall, r: ToolResult) -> str:
    host = urlparse(str(tc.arguments.get("url", ""))).hostname or "?"
    if r.attachments:
        att = r.attachments[0]
        kind = "PDF" if att.mime == "application/pdf" else "image"
        return f"Fetched {kind} ({_human_size(att.size)}) from {host}"
    size = _human_size(len(r.content.encode("utf-8", "ignore")))
    return f"Fetched {size} from {host}"


@register_result_formatter("document_parser")
def _fmt_doc(tc: ToolCall, r: ToolResult) -> str:
    if tc.arguments.get("background"):
        m = _DOC_BG_RE.match(r.content)
        return f"Backgrounded: {m[1]}" if m else "Backgrounded"
    fm = _DOC_FORMAT_RE.search(r.content)
    if not fm:
        return "Parsed"
    fmt, pages = fm.group(1), fm.group(2)
    im = _DOC_IMAGES_RE.search(r.content)
    n_imgs = int(im.group(1)) if im else 0
    head = f"Parsed {pages}p {fmt}" if pages else f"Parsed {fmt}"
    return f"{head} (contains {n_imgs} figures)" if n_imgs > 0 else head


@register_result_formatter("paper_search")
def _fmt_papersearch(tc: ToolCall, r: ToolResult) -> str:
    c = r.content
    if c.startswith("No"):
        return "No papers"
    n = len(re.findall(r"^\d+\. ", c, re.MULTILINE))
    src = tc.arguments.get("source", "arxiv")
    return f'Found {n} {_plural(n, "paper")} (via {src})'


@register_result_formatter("paper_fetch")
def _fmt_paperfetch(tc: ToolCall, r: ToolResult) -> str:
    if tc.arguments.get("mode") == "full":
        fm = _DOC_FORMAT_RE.search(r.content)
        if fm:
            fmt, pages = fm.group(1), fm.group(2)
            return f"Parsed paper: {pages}p {fmt}" if pages else f"Parsed paper: {fmt}"
        return "Parsed paper"
    m = _PAPER_TITLE_RE.search(r.content)
    return f'Fetched metadata: "{_truncate(m[1], 60)}"' if m else "Fetched metadata"


@register_result_formatter("memory_tool")
def _fmt_memory(tc: ToolCall, r: ToolResult) -> str:
    action = tc.arguments.get("action", "")
    name = tc.arguments.get("name", "")
    type_ = tc.arguments.get("type", "")
    key = f"{type_}/{name}" if type_ else name
    if action == "save":
        verb = "Updated" if r.content.startswith("Memory updated") else "Saved"
        return f"{verb} memory: {key}"
    if action == "read":
        return f"Read memory: {key}"
    if action == "delete":
        return f"Deleted memory: {key}"
    return "Memory op"


@register_result_formatter("skill_tool")
def _fmt_skill(tc: ToolCall, r: ToolResult) -> str:
    return f"Loaded skill: {tc.arguments.get('skill_name', '?')}"


@register_result_formatter("background_task")
def _fmt_bgtask(tc: ToolCall, r: ToolResult) -> str:
    action = tc.arguments.get("action", "")
    c = r.content
    if action == "list":
        m = re.match(r"^(\d+) background task", c)
        if m:
            n = int(m[1])
            return f"{n} Background {_plural(n, 'Task')}"
        return "No tasks"
    if action == "status":
        m = re.search(r"^Status: (\w+)", c, re.MULTILINE)
        tid = tc.arguments.get("task_id", "")
        return f"{tid} is {m[1]}" if m else tid
    if action == "cancel":
        return f"Cancelled {tc.arguments.get('task_id', '')}"
    return "Background op"


@register_expander("write_file")
def _expand_write(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.tool_call is None:
        return
    content = str(row.tool_call.arguments.get("content", ""))
    lines = content.splitlines() or [""]
    shown = lines[:_WRITE_DIFF_MAX_LINES]
    diff = "\n".join(f"+{line}" for line in shown)
    if len(lines) > _WRITE_DIFF_MAX_LINES:
        diff += f"\n... ({len(lines) - _WRITE_DIFF_MAX_LINES} more lines)"
    file_path = str(row.tool_call.arguments.get("file_path", ""))
    _render_diff_lines(console, diff, lexer_name=detect_lexer_name(file_path))


@register_expander("edit_file")
def _expand_edit(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None or row.tool_call is None:
        return
    diff = str((row.result.tool_metadata or {}).get("diff", ""))
    if not diff.strip():
        return
    file_path = str(row.tool_call.arguments.get("file_path", ""))
    _render_diff_lines(console, diff, lexer_name=detect_lexer_name(file_path))


@register_expander("glob_files")
def _expand_glob(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None:
        return
    if row.result.content.startswith("No files matching"):
        return
    _, _, body = row.result.content.partition("\n")
    paths = [p for p in body.splitlines() if p.strip()]
    _print_body_lines(console, paths, more_label="files")


@register_expander("grep_files")
def _expand_grep(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None:
        return
    if row.result.content.startswith("No matches"):
        return
    _, _, body = row.result.content.partition("\n")
    matches = [
        line for line in body.splitlines()
        if line.strip() and not line.startswith("--")
    ]
    _print_body_lines(console, matches, more_label="matches")


@register_expander("list_dir")
def _expand_ls(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None:
        return
    if "(empty directory)" in row.result.content:
        return
    _, _, body = row.result.content.partition("\n")
    entries = [line.strip() for line in body.splitlines() if line.strip()]
    _print_body_lines(console, entries, more_label="entries")


@register_expander("terminal_tool")
def _expand_term(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None or row.tool_call is None:
        return
    if row.tool_call.arguments.get("background"):
        return
    content = row.result.content
    body = content
    if _TERM_EXIT_RE.match(content):
        body = content.partition("\n")[2]
    _print_body_head_tail(console, body.splitlines(), more_label="lines")


@register_expander("web_search")
def _expand_websearch(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None:
        return
    if row.result.content.startswith("No results"):
        return
    blocks = [
        b for b in row.result.content.split("\n\n")
        if re.match(r"^\d+\. ", b)
    ]
    cap = 5
    for b in blocks[:cap]:
        lines = b.splitlines()
        title = lines[0] if lines else ""
        url = next(
            (line.strip().replace("URL: ", "") for line in lines
             if line.strip().startswith("URL:")),
            "",
        )
        console.print(f"{_INDENT}[bold]{title}[/bold]")
        if url:
            console.print(f"{_INDENT}  [muted]{url}[/muted]")
    if len(blocks) > cap:
        console.print(
            f"{_INDENT}[muted]{SEP_ELLIPSIS} {len(blocks) - cap} more results[/muted]"
        )


# @register_expander("web_fetch")
# def _expand_webfetch(console: Console, row: _ToolRow) -> None:
#     if _is_error_result(row.result) or row.result is None:
#         return
#     _print_body_lines(console, row.result.content.splitlines(), more_label="lines")




@register_expander("paper_search")
def _expand_papersearch(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None:
        return
    if row.result.content.startswith("No"):
        return
    blocks = [
        b for b in row.result.content.split("\n\n")
        if re.match(r"^\d+\. ", b)
    ]
    cap = 5
    for b in blocks[:cap]:
        lines = b.splitlines()
        title = lines[0] if lines else ""
        authors = next(
            (line.strip().replace("Authors: ", "") for line in lines
             if line.strip().startswith("Authors:")),
            "",
        )
        year = next(
            (line.split(":", 1)[1].strip() for line in lines
             if line.strip().startswith(("Published:", "Year:"))),
            "",
        )
        console.print(f"{_INDENT}[bold]{title}[/bold]")
        meta = f" {SEP_DOT} ".join(x for x in (authors, year) if x)
        if meta:
            console.print(f"{_INDENT}  [muted]{meta}[/muted]")
    if len(blocks) > cap:
        console.print(
            f"{_INDENT}[muted]{SEP_ELLIPSIS} {len(blocks) - cap} more papers[/muted]"
        )


@register_expander("paper_fetch")
def _expand_paperfetch(console: Console, row: _ToolRow) -> None:
    if (
        _is_error_result(row.result)
        or row.result is None
        or row.tool_call is None
        or row.tool_call.arguments.get("mode") == "full"
    ):
        return
    _print_body_lines(console, row.result.content.splitlines(), more_label="lines")


@register_expander("memory_tool")
def _expand_memory(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None or row.tool_call is None:
        return
    action = row.tool_call.arguments.get("action")
    if action == "read":
        _print_body_lines(console, row.result.content.splitlines(), more_label="lines")
    elif action == "save":
        content = str(row.tool_call.arguments.get("content", ""))
        _print_body_lines(console, content.splitlines(), cap=5, more_label="lines")


# @register_expander("skill_tool")
# def _expand_skill(console: Console, row: _ToolRow) -> None:
#     if _is_error_result(row.result) or row.result is None:
#         return
#     content = row.result.content
#     if content.startswith("Skill '") and "not found" in content:
#         return
#     m = _SKILL_LOADED_RE.search(content)
#     if m:
#         _print_body_lines(console, m.group(1).splitlines(), cap=5, more_label="lines")


@register_expander("background_task")
def _expand_bgtask(console: Console, row: _ToolRow) -> None:
    if _is_error_result(row.result) or row.result is None or row.tool_call is None:
        return
    action = row.tool_call.arguments.get("action")
    content = row.result.content
    if action == "list" and not content.startswith("No background"):
        _, _, body = content.partition("\n")
        _print_body_lines(console, body.splitlines(), more_label="tasks")
    elif action == "status":
        _print_body_lines(console, content.splitlines(), more_label="lines")
