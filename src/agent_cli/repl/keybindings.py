"""Main REPL key bindings: Ctrl+C (single/double), Ctrl+D, Alt+Enter,
BracketedPaste, Backspace."""
from __future__ import annotations

import time

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import has_selection
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys

from agent_cli.repl.paste import PasteStore, trailing_placeholder_length
from agent_cli.runtime import plan_mode, session as sess_rt
from agent_harness.agent.base import BaseAgent

_CTRL_C_DOUBLE_WINDOW_S = 2.0

# Module-level so it can be reset between prompts; otherwise a stale first-click
# timestamp can trigger a false double-click exit across prompt boundaries.
_ctrl_c_state: list[float] = [0.0]

_HINT = "\x1b[2m  ⎋ Ctrl+C again or /exit\x1b[0m"


def reset_ctrl_c_state() -> None:
    _ctrl_c_state[0] = 0.0


def build_keybindings(
    *, paste_store: PasteStore, agent: BaseAgent,
) -> KeyBindings:
    kb = KeyBindings()

    @kb.add(Keys.BracketedPaste)
    def _(event: KeyPressEvent) -> None:
        # Preserve prompt_toolkit default \r\n/\r → \n normalization before
        # register, so threshold check and stored content use unified \n.
        data = event.data or ""
        if not data:
            return
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        placeholder = paste_store.register(data)
        event.current_buffer.insert_text(placeholder if placeholder else data)

    def _backspace_atomic_placeholder(event: KeyPressEvent) -> None:
        # Filter excludes selection; default has_selection binding still runs.
        # Store entry is intentionally NOT freed — buffer delete is a pure UI
        # op; keeping the entry preserves undo / cross-turn recall semantics.
        buf = event.current_buffer
        n = trailing_placeholder_length(buf.document.text_before_cursor)
        if n is not None:
            buf.delete_before_cursor(count=n)
            return
        buf.delete_before_cursor(count=1)

    kb.add(Keys.Backspace, filter=~has_selection)(_backspace_atomic_placeholder)
    kb.add("c-h", filter=~has_selection)(_backspace_atomic_placeholder)


    @kb.add("tab")
    def _(event: KeyPressEvent) -> None:
        buf = event.current_buffer
        state = buf.complete_state
        if state is not None:
            completion = state.current_completion
            if completion is None:
                buf.complete_next()
                state = buf.complete_state
                completion = state.current_completion if state is not None else None
            if completion is not None:
                buf.apply_completion(completion)
                return
        buf.start_completion(select_first=True)

    @kb.add("c-c")
    def _(event: KeyPressEvent) -> None:
        buf = event.current_buffer
        if buf.text:
            buf.reset()
            _ctrl_c_state[0] = 0.0
            return
        now = time.monotonic()
        if now - _ctrl_c_state[0] < _CTRL_C_DOUBLE_WINDOW_S:
            # Use EOFError (not KeyboardInterrupt) so asyncio.Task doesn't
            # re-raise a BaseException and abort the event loop.
            event.app.exit(exception=EOFError)
            return
        _ctrl_c_state[0] = now

        async def _show_hint() -> None:
            await run_in_terminal(lambda: print(_HINT))

        event.app.create_background_task(_show_hint())

    @kb.add("c-d")
    def _(event: KeyPressEvent) -> None:
        if not event.current_buffer.text:
            event.app.exit(exception=EOFError)

    @kb.add("escape", "enter")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("s-tab")
    def _cycle_mode(event: KeyPressEvent) -> None:
        cur = sess_rt.current_mode_key(agent)
        next_mode = sess_rt.cycle_next_mode(cur)
        if cur == "plan" and next_mode != "plan":
            plan_mode.exit(agent)
        sess_rt.apply_mode(agent, next_mode)
        event.app.invalidate()

    return kb
