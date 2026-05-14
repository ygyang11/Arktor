"""Syntax highlighting for diff content lines."""
from __future__ import annotations

from pathlib import Path

from pygments.lexers import get_lexer_for_filename  # type: ignore[import-untyped]
from pygments.util import ClassNotFound  # type: ignore[import-untyped]
from rich.syntax import Syntax
from rich.text import Text

_THEME = "ansi_dark"


def detect_lexer_name(path: str | Path) -> str | None:
    path_str = str(path)
    if not path_str:
        return None
    try:
        alias: str = get_lexer_for_filename(path_str).aliases[0]
        return alias
    except (ClassNotFound, IndexError):
        return None


def make_highlighter(lexer_name: str | None) -> Syntax | None:
    if not lexer_name:
        return None
    try:
        return Syntax(
            "", lexer_name, theme=_THEME, background_color=None, tab_size=0,
        )
    except Exception:
        return None


def highlight_code(highlighter: Syntax | None, code: str) -> Text:
    if highlighter is None or not code:
        return Text(code)
    try:
        hi = highlighter.highlight(code)
    except Exception:
        return Text(code)
    if hi.plain.endswith("\n"):
        hi.right_crop(1)
    return hi
