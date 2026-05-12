from agent_cli.render.markdown_stream import _transform_plan_blocks


def test_closed_block_becomes_blockquote() -> None:
    text = "<proposed_plan>\n## Title\n\nBullet one.\n</proposed_plan>"
    out = _transform_plan_blocks(text)
    assert "> **Proposed plan**" in out
    assert "> ## Title" in out
    assert "> Bullet one." in out
    assert "<proposed_plan>" not in out
    assert "</proposed_plan>" not in out


def test_empty_block_keeps_header_only() -> None:
    text = "<proposed_plan>\n</proposed_plan>"
    out = _transform_plan_blocks(text)
    assert "> **Proposed plan**" in out
    assert "<proposed_plan>" not in out


def test_streaming_partial_uses_eof_anchor() -> None:
    text = "<proposed_plan>\n## Title\n\nWriting..."
    out = _transform_plan_blocks(text)
    assert "> ## Title" in out
    assert "> Writing..." in out
    assert "<proposed_plan>" not in out


def test_inline_open_tag_not_matched() -> None:
    text = "The text mentions <proposed_plan> inline like this."
    assert _transform_plan_blocks(text) == text


def test_lone_close_tag_not_matched() -> None:
    text = "Some prose then </proposed_plan> sentence."
    assert _transform_plan_blocks(text) == text


def test_blank_body_lines_quoted_as_lone_gt() -> None:
    text = "<proposed_plan>\nfirst\n\nsecond\n</proposed_plan>"
    out = _transform_plan_blocks(text)
    assert "> first" in out
    assert "\n>\n" in out
    assert "> second" in out


def test_close_tag_with_trailing_chars_does_not_swallow_postfix() -> None:
    """Regression: close-tag-line with trailing chars used to fail the close
    branch and the lazy body would expand to end-of-input, eating any prose
    after the plan into the blockquote."""
    text = "<proposed_plan>\nbody\n</proposed_plan> done.\nLet me also note X."
    out = _transform_plan_blocks(text)
    assert "> body" in out
    assert "Let me also note X." in out
    # post-tag prose must NOT be inside the blockquote
    assert "> Let me also note" not in out
    # close tag literal must not survive in output (its line is consumed)
    assert "</proposed_plan>" not in out


def test_close_tag_at_eof_without_trailing_newline() -> None:
    """Streaming mid-frame where buffer ends right at close tag."""
    text = "<proposed_plan>\nbody\n</proposed_plan>"
    out = _transform_plan_blocks(text)
    assert "> body" in out
    assert "</proposed_plan>" not in out
