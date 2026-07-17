import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.output import DummyOutput

from agent_cli.repl.fill_block import (
    configure_input_window_layout,
    make_continuation_prompt,
    make_input_prompt,
)


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("prompt did not reach the expected render state")
        await asyncio.sleep(0.005)


@asynccontextmanager
async def _running_prompt(
    *,
    default: str = "",
    completer: WordCompleter | None = None,
) -> AsyncIterator[tuple[PromptSession[str], Window, PipeInput, asyncio.Task[str]]]:
    with create_pipe_input() as pipe_input:
        output = DummyOutput()
        with create_app_session(input=pipe_input, output=output):
            session: PromptSession[str] = PromptSession(
                input=pipe_input,
                output=output,
                completer=completer,
                complete_while_typing=True,
                bottom_toolbar="status",
            )
            input_window = configure_input_window_layout(session)
            input_window.style = "class:input-block"
            task = asyncio.create_task(session.prompt_async(
                make_input_prompt(session),
                prompt_continuation=make_continuation_prompt(session),
                default=default,
                handle_sigint=False,
            ))
            await _wait_until(lambda: input_window.render_info is not None)
            try:
                yield session, input_window, pipe_input, task
            finally:
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task


async def test_input_window_uses_one_styled_row_without_shrinking_completions(
) -> None:
    words = [f"/item{i}" for i in range(25)]
    async with _running_prompt(completer=WordCompleter(words)) as (
        session, input_window, pipe_input, _task,
    ):
        pipe_input.send_text("/")
        await _wait_until(lambda: any(
            isinstance(window.content, CompletionsMenuControl)
            for window in session.layout.visible_windows
        ))

        assert input_window.render_info is not None
        assert input_window.render_info.window_height == 1
        menu = next(
            window
            for window in session.layout.visible_windows
            if isinstance(window.content, CompletionsMenuControl)
        )
        assert menu.render_info is not None
        assert menu.render_info.window_height == 16

        toolbar = next(
            window
            for window in session.layout.visible_windows
            if window.style == "class:bottom-toolbar"
        )
        screen = session.app.renderer._last_screen
        assert screen is not None
        assert screen.visible_windows_to_write_positions[toolbar].ypos == 39
        styled_rows = {
            row_index
            for row_index, row in screen.data_buffer.items()
            if any("class:input-block" in char.style for char in row.values())
        }
        assert styled_rows == {0}


@pytest.mark.parametrize(
    ("text", "expected_height"),
    [
        ("x" * 1000, 13),
        ("\n".join(str(i) for i in range(12)), 12),
    ],
)
async def test_input_window_grows_past_completion_reserve(
    text: str,
    expected_height: int,
) -> None:
    async with _running_prompt(
        default=text,
        completer=WordCompleter(["/item"]),
    ) as (_session, input_window, _pipe_input, _task):
        assert input_window.render_info is not None
        assert input_window.render_info.window_height == expected_height
        assert input_window.vertical_scroll == 0


async def test_completed_and_approval_prompts_do_not_keep_completion_reserve(
) -> None:
    async with _running_prompt(
        default="x",
        completer=WordCompleter(["/item"]),
    ) as (session, input_window, pipe_input, task):
        pipe_input.send_text("\r")
        assert await asyncio.wait_for(task, timeout=1) == "x"
        assert input_window.render_info is not None
        assert input_window.render_info.window_height == 1

        session.completer = None
        previous_render = session.app.render_counter
        approval_task = asyncio.create_task(session.prompt_async(
            default="y",
            handle_sigint=False,
        ))
        await _wait_until(lambda: session.app.render_counter > previous_render)
        assert input_window.render_info is not None
        assert input_window.render_info.window_height == 1
        pipe_input.send_text("\r")
        assert await asyncio.wait_for(approval_task, timeout=1) == "y"
