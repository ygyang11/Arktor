"""Processor that styles each visual input line as a full-width block"""
from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.layout.processors import (
    Processor,
    Transformation,
    TransformationInput,
)
from prompt_toolkit.utils import get_cwidth

from agent_cli.theme import PROMPT

PROMPT_TEXT = f"{PROMPT} "
PROMPT_WIDTH = sum(get_cwidth(c) for c in PROMPT_TEXT)


def pick_block_class(text: str) -> str:
    return "class:shell-line" if text.startswith("!") else "class:input-block"


def make_input_prompt(pt_session: PromptSession[str]) -> Callable[[], FormattedText]:
    def _render() -> FormattedText:
        klass = pick_block_class(pt_session.default_buffer.text)
        return FormattedText([(klass, PROMPT_TEXT)])
    return _render


def make_continuation_prompt(
    pt_session: PromptSession[str],
) -> Callable[[int, int, int], FormattedText]:
    def _render(width: int, line_number: int, wrap_count: int) -> FormattedText:
        klass = pick_block_class(pt_session.default_buffer.text)
        return FormattedText([(klass, " " * width)])
    return _render


class FillBlockProcessor(Processor):
    def __init__(self, offset: int = 0) -> None:
        self._offset = offset

    def apply_transformation(
        self, transformation_input: TransformationInput,
    ) -> Transformation:
        try:
            base = pick_block_class(transformation_input.document.text)
            fragments: StyleAndTextTuples = []
            cell_width = 0
            for fragment in transformation_input.fragments:
                style, text = fragment[0], fragment[1]
                new_style = f"{base} {style}".strip() if style else base
                fragments.append((new_style, text, *fragment[2:]))
                cell_width += sum(get_cwidth(c) for c in text)
            effective_width = max(1, transformation_input.width - self._offset)
            if cell_width == 0:
                pad = effective_width - 1
            else:
                remainder = cell_width % effective_width
                pad = effective_width - remainder - 1 if remainder else 0
            pad = max(0, pad)
            if pad:
                fragments.append((base, " " * pad))
            return Transformation(fragments)
        except Exception:
            try:
                return Transformation(list(transformation_input.fragments))
            except Exception:
                return Transformation([])
