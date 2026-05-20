"""Tests for repl/paste.py — PasteStore + PastePlaceholderProcessor."""
from __future__ import annotations

from unittest.mock import MagicMock

from prompt_toolkit.document import Document
from prompt_toolkit.layout.processors import (
    Transformation,
    TransformationInput,
)

from agent_cli.repl.paste import (
    CHAR_THRESHOLD,
    LINE_THRESHOLD,
    PastePlaceholderProcessor,
    PasteStore,
)

# ── PasteStore.register ─────────────────────────────────────────────────


class TestRegisterThresholds:
    def test_empty_returns_none(self) -> None:
        assert PasteStore().register("") is None

    def test_short_returns_none(self) -> None:
        store = PasteStore()
        assert store.register("hello\nworld") is None
        assert store._contents == {}

    def test_char_boundary_700_below(self) -> None:
        store = PasteStore()
        text = "a" * CHAR_THRESHOLD
        assert store.register(text) is None

    def test_char_boundary_701_above(self) -> None:
        store = PasteStore()
        text = "a" * (CHAR_THRESHOLD + 1)
        assert store.register(text) == "[Pasted text #1]"

    def test_line_boundary_three_lines(self) -> None:
        store = PasteStore()
        text = "a\nb\nc"  # 3 visual lines, no trailing newline
        assert store.register(text) is None

    def test_line_boundary_four_lines(self) -> None:
        store = PasteStore()
        text = "a\nb\nc\nd"  # 4 visual lines
        assert store.register(text) == "[Pasted text #1 +3 lines]"

    def test_trailing_newline_not_inflated(self) -> None:
        store = PasteStore()
        text = "a\nb\nc\n"  # 3 lines + trailing newline = 3 visual lines
        assert store.register(text) is None

    def test_newline_only_paste_triggers(self) -> None:
        store = PasteStore()
        text = "\n" * LINE_THRESHOLD  # 4 newlines = 4 visual lines
        result = store.register(text)
        assert result == "[Pasted text #1 +4 lines]"

    def test_id_increments(self) -> None:
        store = PasteStore()
        big = "x" * (CHAR_THRESHOLD + 1)
        assert store.register(big) == "[Pasted text #1]"
        assert store.register(big) == "[Pasted text #2]"
        assert store.register(big) == "[Pasted text #3]"

    def test_no_newline_no_lines_suffix(self) -> None:
        store = PasteStore()
        assert store.register("z" * (CHAR_THRESHOLD + 1)) == "[Pasted text #1]"


# ── PasteStore.expand ───────────────────────────────────────────────────


class TestExpand:
    def test_empty_string_returns_empty_and_no_missing(self) -> None:
        text, missing, _ = PasteStore().resolve("")
        assert text == ""
        assert missing == []

    def test_no_placeholder(self) -> None:
        text, missing, _ = PasteStore().resolve("hello world")
        assert text == "hello world"
        assert missing == []

    def test_single_known(self) -> None:
        store = PasteStore()
        store._contents[1] = "ORIGINAL"
        text, missing, _ = store.resolve("see [Pasted text #1 +2 lines] here")
        assert text == "see ORIGINAL here"
        assert missing == []

    def test_multiple_known_preserves_offsets(self) -> None:
        store = PasteStore()
        store._contents[1] = "AAA"
        store._contents[2] = "BBB"
        text, missing, _ = store.resolve(
            "first [Pasted text #1] mid [Pasted text #2] end",
        )
        assert text == "first AAA mid BBB end"
        assert missing == []

    def test_unknown_id_rewrites_unavailable(self) -> None:
        text, missing, _ = PasteStore().resolve("x [Pasted text #5] y")
        assert text == "x [Pasted text unavailable] y"
        assert missing == [5]

    def test_dedupes_missing_ids(self) -> None:
        text, missing, _ = PasteStore().resolve(
            "[Pasted text #7] and [Pasted text #7]",
        )
        assert missing == [7]
        assert text == "[Pasted text unavailable] and [Pasted text unavailable]"

    def test_mixed_known_unknown_preserves_source_order(self) -> None:
        store = PasteStore()
        store._contents[1] = "K"
        text, missing, _ = store.resolve(
            "[Pasted text #1] [Pasted text #9] [Pasted text #7]",
        )
        assert text == "K [Pasted text unavailable] [Pasted text unavailable]"
        assert missing == [9, 7]

    def test_interleaved_missing_preserves_source_order(self) -> None:
        # Regression: 单遍反向 + seen + reverse() 会得到 [7, 9]，源顺序应是 [9, 7]
        text, missing, _ = PasteStore().resolve(
            "[Pasted text #9] x [Pasted text #7] y [Pasted text #9]",
        )
        assert missing == [9, 7]


class TestTrailingPlaceholderLength:
    def test_plain_text_returns_none(self) -> None:
        from agent_cli.repl.paste import trailing_placeholder_length

        assert trailing_placeholder_length("hello world") is None

    def test_trailing_placeholder_returns_length(self) -> None:
        from agent_cli.repl.paste import trailing_placeholder_length

        assert trailing_placeholder_length(
            "see [Pasted text #1 +2 lines]",
        ) == len("[Pasted text #1 +2 lines]")

    def test_no_lines_suffix_variant(self) -> None:
        from agent_cli.repl.paste import trailing_placeholder_length

        assert trailing_placeholder_length("[Pasted text #3]") == len(
            "[Pasted text #3]",
        )

    def test_placeholder_in_middle_returns_none(self) -> None:
        from agent_cli.repl.paste import trailing_placeholder_length

        assert trailing_placeholder_length(
            "[Pasted text #1] trailing text",
        ) is None

    def test_unknown_id_still_returns_length(self) -> None:
        # Pure measurement; doesn't consult any store.
        from agent_cli.repl.paste import trailing_placeholder_length

        assert trailing_placeholder_length("[Pasted text #99]") == len(
            "[Pasted text #99]",
        )

    def test_does_not_mutate_store(self) -> None:
        # Regression: Backspace must not free store entries (preserves undo /
        # cross-turn recall).
        from agent_cli.repl.paste import trailing_placeholder_length

        store = PasteStore()
        store._contents[1] = "ORIGINAL"
        trailing_placeholder_length("[Pasted text #1]")
        assert store._contents == {1: "ORIGINAL"}


# ── PastePlaceholderProcessor ───────────────────────────────────────────


def _make_ti(plain: str) -> TransformationInput:
    return TransformationInput(
        buffer_control=MagicMock(),
        document=Document(text=plain),
        lineno=0,
        source_to_display=lambda i: i,
        fragments=[("", plain)],
        width=80,
        height=24,
    )


def _styles_for_text(
    fragments: list[tuple[str, str]],
    target: str,
) -> set[str]:
    out: set[str] = set()
    pos = 0
    full = "".join(t for _, t in fragments)
    start = full.find(target)
    end = start + len(target)
    for style, t in fragments:
        seg_end = pos + len(t)
        if pos < end and seg_end > start:
            out.add(style)
        pos = seg_end
    return out


class TestProcessor:
    def test_no_match_returns_identity(self) -> None:
        proc = PastePlaceholderProcessor()
        ti = _make_ti("hello world")
        result = proc.apply_transformation(ti)
        assert isinstance(result, Transformation)
        assert "".join(t for _, t in result.fragments) == "hello world"

    def test_single_match_restyles_span(self) -> None:
        proc = PastePlaceholderProcessor()
        ti = _make_ti("see [Pasted text #1 +2 lines] here")
        result = proc.apply_transformation(ti)
        styles = _styles_for_text(result.fragments, "[Pasted text #1 +2 lines]")
        assert any("class:paste-placeholder" in s for s in styles)

    def test_multiple_matches_restyles_each(self) -> None:
        proc = PastePlaceholderProcessor()
        ti = _make_ti("a [Pasted text #1] b [Pasted text #2] c")
        result = proc.apply_transformation(ti)
        for token in ("[Pasted text #1]", "[Pasted text #2]"):
            styles = _styles_for_text(result.fragments, token)
            assert any("class:paste-placeholder" in s for s in styles)

    def test_layers_on_existing_style(self) -> None:
        # Regression: prompt_toolkit's selection / search processors run
        # before user processors. When the user selects a placeholder token,
        # the fragment carries class:selected — replacing it with our class
        # would erase the selection visual. Verify we layer instead.
        proc = PastePlaceholderProcessor()
        plain = "[Pasted text #1]"
        ti = TransformationInput(
            buffer_control=MagicMock(),
            document=Document(text=plain),
            lineno=0,
            source_to_display=lambda i: i,
            fragments=[("class:selected", plain)],
            width=80,
            height=24,
        )
        result = proc.apply_transformation(ti)
        styles = _styles_for_text(result.fragments, plain)
        assert any(
            "class:selected" in s and "class:paste-placeholder" in s
            for s in styles
        )

    def test_preserves_non_match_fragments(self) -> None:
        proc = PastePlaceholderProcessor()
        ti = _make_ti("plain text only")
        result = proc.apply_transformation(ti)
        full = "".join(t for _, t in result.fragments)
        assert full == "plain text only"
        for style, _ in result.fragments:
            assert style == ""

    def test_returns_identity_on_exception(self) -> None:
        proc = PastePlaceholderProcessor()
        broken = MagicMock()
        broken.fragments = None
        result = proc.apply_transformation(broken)
        assert isinstance(result, Transformation)

    def test_handles_empty_fragments(self) -> None:
        proc = PastePlaceholderProcessor()
        ti = TransformationInput(
            buffer_control=MagicMock(),
            document=Document(text=""),
            lineno=0,
            source_to_display=lambda i: i,
            fragments=[],
            width=80,
            height=24,
        )
        result = proc.apply_transformation(ti)
        assert isinstance(result, Transformation)
        assert list(result.fragments) == []
