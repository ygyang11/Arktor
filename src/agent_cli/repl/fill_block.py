"""Full-width input block layout and dynamic prompt styling."""
from __future__ import annotations

from collections.abc import Callable, Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout import (
    BufferControl,
    Container,
    Dimension,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)

from agent_cli.theme import PROMPT

PROMPT_TEXT = f"{PROMPT} "


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


def configure_input_window_layout(
    pt_session: PromptSession[str],
) -> Window:
    """Keep the styled input at content height without shrinking completions."""
    input_window = next(
        window
        for window in pt_session.layout.find_all_windows()
        if isinstance(window.content, BufferControl)
        and window.content.buffer is pt_session.default_buffer
    )
    completion_host = next(
        container
        for container in _walk_containers(pt_session.layout.container)
        if isinstance(container, FloatContainer)
        and any(
            child is input_window
            for child in _walk_containers(container.content)
        )
    )

    reserve_height = input_window.height
    natural_content = completion_host.content
    input_window.height = None
    input_window.dont_extend_height = to_filter(True)
    minimum_host = VSplit([
        natural_content,
        Window(width=0, height=reserve_height),
    ])
    completion_host.content = HSplit([
        minimum_host,
        Window(height=Dimension(preferred=0)),
    ])
    return input_window


def _walk_containers(root: Container) -> Iterator[Container]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        container = pending.pop()
        identity = id(container)
        if identity in seen:
            continue
        seen.add(identity)
        yield container
        pending.extend(reversed(container.get_children()))
