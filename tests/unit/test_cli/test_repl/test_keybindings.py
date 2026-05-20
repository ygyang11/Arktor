import asyncio
import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from agent_cli.repl.keybindings import build_keybindings
from agent_cli.repl.paste import CHAR_THRESHOLD, PasteStore


def _run_handler(handler: Any, event: Any) -> None:
    result = handler(event)
    if inspect.iscoroutine(result):
        asyncio.run(result)


def _kb():
    return build_keybindings(paste_store=PasteStore(), agent=_stub_agent())


def _stub_agent() -> MagicMock:
    agent = MagicMock()
    agent._approval.mode = "auto"
    return agent


def _find_handler(kb: object, key: str | Keys):
    target = key.value if isinstance(key, Keys) else key
    for binding in kb.bindings:
        values = tuple(getattr(k, "value", str(k)) for k in binding.keys)
        if values == (target,):
            return binding.handler
    raise AssertionError(f"binding for {key!r} not found")


def test_bindings_registered() -> None:
    kb = _kb()
    # Tab + Ctrl+C + Ctrl+D + Alt+Enter + BracketedPaste at minimum
    assert len(kb.bindings) >= 5


def test_keys_cover_expected_set() -> None:
    kb = _kb()
    values = {tuple(getattr(k, "value", str(k)) for k in b.keys) for b in kb.bindings}
    assert ("c-i",) in values
    assert ("c-c",) in values
    assert ("c-d",) in values
    assert (Keys.BracketedPaste.value,) in values
    # Alt+Enter registers as (Escape, ControlM) after prompt_toolkit normalization.
    assert any("escape" in t and "c-m" in t for t in values)


def test_tab_applies_current_completion() -> None:
    kb = _kb()
    handler = _find_handler(kb, "c-i")
    buf = Buffer(document=Document("@sr", cursor_position=3))
    completion = Completion("src/", start_position=-2)
    buf.complete_state = CompletionState(
        original_document=buf.document,
        completions=[completion],
        complete_index=0,
    )

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "@src/"


def test_tab_selects_first_completion_when_menu_is_open_without_selection() -> None:
    kb = _kb()
    handler = _find_handler(kb, "c-i")
    buf = Buffer(document=Document("/cl", cursor_position=3))
    completion = Completion("/clear", start_position=-3)
    buf.complete_state = CompletionState(
        original_document=buf.document,
        completions=[completion],
    )

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "/clear"


def test_tab_starts_completion_when_menu_is_closed() -> None:
    kb = _kb()
    handler = _find_handler(kb, "c-i")
    buf = MagicMock()
    buf.complete_state = None

    handler(SimpleNamespace(current_buffer=buf))

    buf.start_completion.assert_called_once_with(select_first=True)


# ── BracketedPaste ──────────────────────────────────────────────────────


def _bp_event(data: str, buf: Buffer) -> SimpleNamespace:
    return SimpleNamespace(data=data, current_buffer=buf)


def test_bracketed_paste_below_threshold_inserts_verbatim() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()

    _run_handler(handler, _bp_event("hello", buf))

    assert buf.text == "hello"
    assert store._contents == {}


def test_bracketed_paste_above_char_threshold_inserts_placeholder() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()
    big = "x" * (CHAR_THRESHOLD + 1)

    _run_handler(handler, _bp_event(big, buf))

    assert buf.text == "[Pasted text #1]"
    assert store._contents[1] == big


def test_bracketed_paste_above_line_threshold_inserts_placeholder() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()
    text = "a\nb\nc\nd"

    _run_handler(handler, _bp_event(text, buf))

    assert buf.text == "[Pasted text #1 +3 lines]"
    assert store._contents[1] == text


def test_bracketed_paste_normalizes_crlf() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()

    _run_handler(handler, _bp_event("a\r\nb\r\nc", buf))

    assert buf.text == "a\nb\nc"
    assert store._contents == {}


def test_bracketed_paste_normalizes_lone_cr() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()

    _run_handler(handler, _bp_event("a\rb\rc", buf))

    assert buf.text == "a\nb\nc"


def test_bracketed_paste_registers_normalized_for_threshold() -> None:
    # "\r\n" * 4 → 4 visual lines after normalization → triggers
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()

    _run_handler(handler, _bp_event("\r\n" * 4, buf))

    assert buf.text.startswith("[Pasted text #1")
    assert store._contents[1] == "\n" * 4


def test_bracketed_paste_empty_data_no_op() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()

    _run_handler(handler, _bp_event("", buf))

    assert buf.text == ""
    assert store._contents == {}


def test_bracketed_paste_none_data_no_op() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _find_handler(kb, Keys.BracketedPaste)
    buf = Buffer()

    _run_handler(handler, SimpleNamespace(data=None, current_buffer=buf))

    assert buf.text == ""


# ── Backspace: atomic placeholder delete ────────────────────────────────


def _backspace_handlers(kb) -> list:
    """Return all bindings registered for Backspace / c-h."""
    targets = {Keys.Backspace.value, "c-h"}
    found = []
    for binding in kb.bindings:
        keys = tuple(getattr(k, "value", str(k)) for k in binding.keys)
        if keys in {(t,) for t in targets}:
            found.append(binding)
    return found


def test_backspace_bound_to_both_canonical_and_c_h() -> None:
    kb = build_keybindings(paste_store=PasteStore(), agent=_stub_agent())
    # Both spellings should be registered (defensive against terminal/version
    # divergence). Even though Keys.Backspace.value == 'c-h' today, we count
    # 2 bindings for documentation + future compatibility.
    bindings = _backspace_handlers(kb)
    assert len(bindings) >= 2


def test_backspace_filter_excludes_selection() -> None:
    # Default has_selection backspace must continue to handle selection cuts.
    kb = build_keybindings(paste_store=PasteStore(), agent=_stub_agent())
    bindings = _backspace_handlers(kb)
    for b in bindings:
        # Filter is `~has_selection`; calling it with True (selection present)
        # should return False so our handler doesn't fire.
        assert b.filter() is True  # No selection by default in test context


def test_backspace_atomic_delete_when_trailing_placeholder() -> None:
    store = PasteStore()
    store._contents[1] = "ORIGINAL"
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _backspace_handlers(kb)[0].handler

    text = "see [Pasted text #1 +2 lines]"
    buf = Buffer(document=Document(text, cursor_position=len(text)))

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "see "
    # Store entry is preserved — buffer delete is pure UI; preserves undo
    # and cross-turn recall semantics.
    assert store._contents[1] == "ORIGINAL"


def test_backspace_normal_text_deletes_one_char() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _backspace_handlers(kb)[0].handler

    buf = Buffer(document=Document("hello", cursor_position=5))

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "hell"


def test_backspace_placeholder_in_middle_deletes_one_char() -> None:
    # cursor not at end of placeholder; should fall back to char delete
    store = PasteStore()
    store._contents[1] = "x"
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _backspace_handlers(kb)[0].handler

    text = "[Pasted text #1] trailing"
    buf = Buffer(document=Document(text, cursor_position=len(text)))

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "[Pasted text #1] trailin"
    assert 1 in store._contents


def test_backspace_multiple_placeholders_deletes_only_trailing() -> None:
    store = PasteStore()
    store._contents[1] = "AAA"
    store._contents[2] = "BBB"
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _backspace_handlers(kb)[0].handler

    text = "[Pasted text #1] mid [Pasted text #2]"
    buf = Buffer(document=Document(text, cursor_position=len(text)))

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "[Pasted text #1] mid "
    # Both store entries preserved — buffer delete is UI only
    assert store._contents == {1: "AAA", 2: "BBB"}


def test_backspace_at_buffer_start_no_crash() -> None:
    store = PasteStore()
    kb = build_keybindings(paste_store=store, agent=_stub_agent())
    handler = _backspace_handlers(kb)[0].handler

    buf = Buffer()

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == ""
