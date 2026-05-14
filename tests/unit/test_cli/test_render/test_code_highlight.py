"""Tests for diff-line syntax highlighting helper."""
from __future__ import annotations

from rich.style import Style

from agent_cli.render._code_highlight import (
    detect_lexer_name,
    highlight_code,
    make_highlighter,
)


def test_detect_lexer_name_python() -> None:
    assert detect_lexer_name("foo.py") == "python"


def test_detect_lexer_name_unknown_extension() -> None:
    assert detect_lexer_name("foo.this-ext-does-not-exist-12345") is None


def test_detect_lexer_name_empty_path() -> None:
    assert detect_lexer_name("") is None


def test_make_highlighter_none_for_empty_lexer_name() -> None:
    assert make_highlighter(None) is None
    assert make_highlighter("") is None


def test_highlight_code_no_highlighter_returns_plain() -> None:
    text = highlight_code(None, "def foo():")
    assert text.plain == "def foo():"
    assert text.spans == []


def test_highlight_code_empty_string_returns_plain() -> None:
    hl = make_highlighter("python")
    text = highlight_code(hl, "")
    assert text.plain == ""


def test_highlight_code_python_emits_token_color_spans() -> None:
    hl = make_highlighter("python")
    text = highlight_code(hl, "def foo():")
    # At least one span must carry a fg color (token coloring active).
    assert any(
        not isinstance(span.style, str) and span.style.color is not None
        for span in text.spans
    )


def test_highlight_code_preserves_trailing_spaces() -> None:
    hl = make_highlighter("python")
    text = highlight_code(hl, "x = 1  ")
    assert text.plain == "x = 1  "


def test_highlight_code_preserves_trailing_tab() -> None:
    hl = make_highlighter("python")
    # Pygments expands tabs to spaces; what matters is the trailing whitespace
    # is not stripped to nothing.
    text = highlight_code(hl, "x = 1\t")
    assert text.plain.startswith("x = 1") and text.plain != "x = 1"


def test_highlight_code_preserves_whitespace_only_line() -> None:
    hl = make_highlighter("python")
    text = highlight_code(hl, "    ")
    assert text.plain == "    "


def test_highlight_code_strips_trailing_newline_only() -> None:
    hl = make_highlighter("python")
    text = highlight_code(hl, "x = 1")
    assert not text.plain.endswith("\n")
    assert text.plain == "x = 1"


def test_highlight_code_carries_no_background_info() -> None:
    # The whole contract: outer diff_add/diff_remove base style must compose
    # cleanly with the highlighted Text. Any bg info inside the highlighted
    # Text (token bg, `on default` reset span, or hi.style with a bgcolor)
    # would clobber that compose. Validate all three.
    hl = make_highlighter("python")
    text = highlight_code(hl, "def foo():")

    # 1. Text-level base style must not carry a bg.
    if not isinstance(text.style, str):
        assert text.style.bgcolor is None
    else:
        assert "on " not in text.style

    # 2. No span may carry a bg (no per-token bg, no `on default` reset span).
    for span in text.spans:
        if isinstance(span.style, str):
            assert "on " not in span.style
        else:
            assert span.style.bgcolor is None


def test_highlight_code_preserves_literal_tabs() -> None:
    # Pygments' default tab_size expands `\t` to spaces, which corrupts
    # tab-sensitive files (Makefiles, heredocs). tab_size=0 must keep tabs.
    hl = make_highlighter("make")
    text = highlight_code(hl, "\tcmd")
    assert text.plain == "\tcmd"

    text = highlight_code(hl, "a\tb")
    assert text.plain == "a\tb"


def test_highlight_code_falls_back_on_lex_error() -> None:
    # Rich's Syntax constructor defers lexer resolution, so a bad name doesn't
    # throw at make_highlighter time — it throws inside highlight(). The
    # helper's try/except must swallow it and return plain text.
    hl = make_highlighter("definitely-not-a-real-lexer-xyz")
    text = highlight_code(hl, "def foo():")
    assert text.plain == "def foo():"


def test_diff_bg_overlay_composes_cleanly() -> None:
    # Build the same composition pattern _render_diff_lines uses, then check
    # the per-character effective style via Text.get_style_at_offset. This
    # validates "diff bg preserved under token fg" without depending on the
    # terminal/NO_COLOR-sensitive ANSI render path.
    from rich.console import Console
    from rich.text import Text

    hl = make_highlighter("python")
    content = Text("+", style=Style(color="white", bgcolor="#1F3A2D"))
    content.append_text(highlight_code(hl, "def foo():"))

    con = Console()
    # Walk every character; the merged style at each offset must have our bg.
    for i in range(len(content)):
        style = content.get_style_at_offset(con, i)
        assert style.bgcolor is not None, f"char {i}={content.plain[i]!r} lost bg"
        assert style.bgcolor.triplet is not None
        assert style.bgcolor.triplet.hex.lower() == "#1f3a2d"
