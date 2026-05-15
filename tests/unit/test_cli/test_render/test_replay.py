"""Tests for static replay path (no adapter / no Live)."""
from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from agent_cli.render.replay import (
    _hard_clear,
    _index_results,
    render_post_switch,
    replay,
    slice_last_turns,
)
from agent_cli.theme import DEFAULT_THEME, PROMPT
from agent_harness.core.message import Message, ToolCall, ToolResult


def _u(text: str) -> Message:
    return Message.user(text)


def _a(text: str | None = None, calls: list[ToolCall] | None = None) -> Message:
    return Message.assistant(content=text, tool_calls=calls)


def _t(call_id: str, content: str, is_error: bool = False) -> Message:
    return Message.tool(tool_call_id=call_id, content=content, is_error=is_error)


def _render(*messages: Message) -> str:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=False, color_system=None,
        width=200, theme=DEFAULT_THEME.rich,
    )
    replay(console, DEFAULT_THEME, list(messages))
    return buf.getvalue()


# ── slice_last_turns ─────────────────────────────────────────────────


def test_slice_last_turns_takes_last_n_user_msgs() -> None:
    msgs = [_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3"), _a("r3")]
    sliced = slice_last_turns(msgs, 2)
    assert sliced == msgs[2:]


def test_slice_last_turns_falls_back_to_all_when_fewer_users() -> None:
    msgs = [_u("q1"), _a("r1"), _u("q2"), _a("r2")]
    assert slice_last_turns(msgs, 5) == msgs


def test_slice_last_turns_filters_system_messages() -> None:
    sys = Message.system("system prompt")
    msgs = [sys, _u("q1"), _a("r1")]
    assert sys not in slice_last_turns(msgs, 5)


def test_slice_last_turns_first_after_slice_is_user() -> None:
    msgs = [_u("q1"), _a("r1"), _u("q2"), _a("r2"), _u("q3"), _a("r3")]
    sliced = slice_last_turns(msgs, 2)
    assert sliced[0].role.value == "user"


def test_slice_last_turns_empty_input() -> None:
    assert slice_last_turns([], 5) == []


def test_slice_last_turns_no_user_returns_all_non_system() -> None:
    msgs = [_a("r1"), _a("r2")]
    assert slice_last_turns(msgs, 3) == msgs


# ── _index_results ───────────────────────────────────────────────────


def test_index_results_picks_tool_messages_only() -> None:
    tr = _t("call_1", "done")
    msgs = [_u("q"), _a("r", calls=[ToolCall(id="call_1", name="x")]), tr]
    idx = _index_results(msgs)
    assert "call_1" in idx
    assert idx["call_1"].content == "done"


def test_index_results_empty_when_no_tool_messages() -> None:
    assert _index_results([_u("q"), _a("r")]) == {}


# ── replay rendering ─────────────────────────────────────────────────


def test_replay_empty_messages_renders_nothing() -> None:
    assert _render() == ""


def test_replay_user_message_uses_prompt_glyph() -> None:
    out = _render(_u("hello"))
    assert f"{PROMPT} hello" in out


def test_replay_user_line_renders_as_section_bg_block() -> None:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=True, color_system="truecolor",
        width=200, theme=DEFAULT_THEME.rich, no_color=False,
    )
    replay(console, DEFAULT_THEME, [Message.user("hello world content")])
    out = buf.getvalue()
    # section_bg in DEFAULT_THEME (flexoki-dark) is #2E2D2B → truecolor escape.
    # The ❯ glyph + content + padding must all live under one bg span (single
    # full-width block); ❯ must NOT carry primary fg anymore (was the old look).
    section_bg_open = "\x1b[48;2;46;45;43m"
    primary_open = "\x1b[38;2;218;112;44m"
    assert section_bg_open in out, "user line should carry section_bg block style"
    assert primary_open not in out, "❯ must not be primary-colored anymore"
    # Block must extend past the typed text — i.e. trailing padding before reset.
    after_bg = out.split(section_bg_open, 1)[1]
    reset_pos = after_bg.find("\x1b[0m")
    content = "❯ hello world content"
    content_pos = after_bg.find(content)
    assert content_pos != -1 and content_pos < reset_pos, \
        "❯ and content must live inside the section_bg span"
    trailing = after_bg[content_pos + len(content):reset_pos]
    assert trailing and set(trailing) == {" "}, \
        "block should pad with spaces to fill terminal width before reset"


def test_replay_user_skips_empty_content() -> None:
    out = _render(Message.user(""))
    assert out == ""


def test_replay_assistant_text_includes_tool_done_glyph() -> None:
    out = _render(_a("hi there"))
    assert "● " in out
    assert "hi there" in out


def test_replay_assistant_skips_empty_content() -> None:
    out = _render(_a(None))
    assert out == ""


def test_replay_assistant_tool_call_renders_name_and_args() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "/a/b.py"})
    tr = _t("c1", "lines 1-10")
    out = _render(_a(calls=[tc]), tr)
    assert "Read" in out
    assert "/a/b.py" in out


def test_replay_running_when_result_missing() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "/a"})
    out = _render(_a(calls=[tc]))
    assert "Read" in out


def test_replay_error_result_renders_error_branch() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "/x"})
    tr = _t("c1", "Error: file not found", is_error=True)
    out = _render(_a(calls=[tc]), tr)
    assert "Error" in out


def test_replay_multiple_tool_calls_have_blank_line_between() -> None:
    tc1 = ToolCall(id="c1", name="read_file", arguments={"file_path": "/a"})
    tc2 = ToolCall(id="c2", name="read_file", arguments={"file_path": "/b"})
    tr1 = _t("c1", "ok")
    tr2 = _t("c2", "ok")
    out = _render(_a(calls=[tc1, tc2]), tr1, tr2)
    # blank line between the two call rows; both file paths present
    assert "/a" in out and "/b" in out


def test_replay_tool_message_alone_not_rendered() -> None:
    out = _render(_t("orphan", "stray content"))
    assert "stray content" not in out


def test_replay_write_file_emits_expander_diff() -> None:
    tc = ToolCall(
        id="c1",
        name="write_file",
        arguments={"file_path": "/a.py", "content": "print('hi')\nprint('bye')\n"},
    )
    tr = _t("c1", "Created /a.py (2 lines)")
    out = _render(_a(calls=[tc]), tr)
    assert "+print('hi')" in out
    assert "+print('bye')" in out


def test_replay_edit_file_emits_expander_diff() -> None:
    tc = ToolCall(
        id="c1",
        name="edit_file",
        arguments={"file_path": "/a.py", "old_string": "x", "new_string": "y"},
    )
    body = "Edited /a.py (1 replacement)\n@@ -1,1 +1,1 @@\n-x\n+y"
    tr = _t("c1", body)
    out = _render(_a(calls=[tc]), tr)
    assert "-x" in out
    assert "+y" in out


def test_replay_paper_search_emits_expander_list() -> None:
    tc = ToolCall(
        id="c1",
        name="paper_search",
        arguments={"query": "llm iov", "source": "arxiv"},
    )
    body = (
        "1. Paper Title One\n"
        "   Authors: Alice, Bob\n"
        "   Year: 2024\n"
        "   URL: http://x\n\n"
        "2. Paper Title Two\n"
        "   Authors: Carol\n"
        "   Year: 2025\n"
        "   URL: http://y"
    )
    tr = _t("c1", body)
    out = _render(_a(calls=[tc]), tr)
    assert "Paper Title One" in out
    assert "Paper Title Two" in out
    assert "Alice, Bob" in out


def test_replay_web_search_emits_expander_list() -> None:
    tc = ToolCall(id="c1", name="web_search", arguments={"query": "claude"})
    body = (
        "1. First Result Title\n"
        "   URL: https://example.com/a\n\n"
        "2. Second Result Title\n"
        "   URL: https://example.com/b"
    )
    tr = _t("c1", body)
    out = _render(_a(calls=[tc]), tr)
    assert "First Result Title" in out
    assert "Second Result Title" in out
    assert "https://example.com/a" in out


def test_replay_grep_emits_expander_matches() -> None:
    tc = ToolCall(id="c1", name="grep_files", arguments={"pattern": "TODO"})
    body = "2 matches in 1 files\n/a.py:3:TODO fix\n/a.py:7:TODO refactor"
    tr = _t("c1", body)
    out = _render(_a(calls=[tc]), tr)
    assert "TODO fix" in out
    assert "TODO refactor" in out


def test_replay_skips_expander_when_result_is_error() -> None:
    tc = ToolCall(
        id="c1",
        name="write_file",
        arguments={"file_path": "/a.py", "content": "print('hi')\n"},
    )
    tr = _t("c1", "Error: disk full", is_error=True)
    out = _render(_a(calls=[tc]), tr)
    assert "+print('hi')" not in out


# ── user-shell-run rendering ─────────────────────────────────────────


def test_replay_user_shell_run_uses_bang_form() -> None:
    from agent_cli.render.notices import format_shell_run
    raw = format_shell_run("cd Agent-Harness", 0, "")
    out = _render(_u(raw))
    # `❯ !cd Agent-Harness` echo line
    assert "!cd Agent-Harness" in out
    # live shell-run renderables: `● Run(cmd)` call line + `⎿  ...` body
    assert "Run(cd Agent-Harness)" in out
    assert "(Completed with no output)" in out
    # envelope tags must not leak through
    assert "<user-shell-run>" not in out
    assert "</user-shell-run>" not in out
    assert "```sh" not in out


def test_replay_user_shell_run_with_output_renders_body() -> None:
    from agent_cli.render.notices import format_shell_run
    raw = format_shell_run("pwd", 0, "/Users/ygyang/Project")
    out = _render(_u(raw))
    assert "!pwd" in out
    assert "Run(pwd)" in out
    assert "/Users/ygyang/Project" in out


def test_replay_user_shell_run_failure_strips_exit_code_prefix() -> None:
    from agent_cli.render.notices import format_shell_run
    raw = format_shell_run("nope", 127, "bash: nope: not found")
    out = _render(_u(raw))
    assert "!nope" in out
    assert "Run(nope)" in out
    assert "bash: nope: not found" in out
    # exit_code is encoded into the live glyph color, not text body
    assert "[exit code 127]" not in out


def test_replay_plain_user_message_unchanged_by_shell_run_path() -> None:
    out = _render(_u("hello world"))
    assert "hello world" in out
    assert "!" not in out.split("hello world")[0]


def test_split_exit_code_handles_prefix_and_plain_body() -> None:
    from agent_cli.render.replay import _split_exit_code
    assert _split_exit_code("ok output") == (0, "ok output")
    assert _split_exit_code("[exit code 1]\nboom") == (1, "boom")
    assert _split_exit_code("[exit code 127]") == (127, "")
    assert _split_exit_code("[exit code 2]\n(Completed with no output)") == (
        2, "(Completed with no output)",
    )


def test_split_exit_code_handles_signal_kill_negative() -> None:
    """Subprocess killed by signal returns negative code (e.g. -9 for SIGKILL)."""
    from agent_cli.render.replay import _split_exit_code
    assert _split_exit_code("[exit code -9]\nkilled") == (-9, "killed")
    assert _split_exit_code("[exit code -15]") == (-15, "")


# ── command-invocation matchers ──────────────────────────────────────


def test_match_init_new_without_focus() -> None:
    from agent_cli.commands.builtin.init import _INIT_NEW
    from agent_cli.render.replay import _match_init
    content = _INIT_NEW.format(focus="")
    assert _match_init(content) == ""


def test_match_init_new_with_focus() -> None:
    from agent_cli.commands.builtin.init import _INIT_NEW
    from agent_cli.render.replay import _match_init
    content = _INIT_NEW.format(focus="\n\nFocus: testing strategy")
    assert _match_init(content) == "testing strategy"


def test_match_init_update_without_focus() -> None:
    from agent_cli.commands.builtin.init import _INIT_UPDATE
    from agent_cli.render.replay import _match_init
    content = _INIT_UPDATE.format(target="AGENTS.md", focus="")
    assert _match_init(content) == ""


def test_match_init_update_with_focus() -> None:
    from agent_cli.commands.builtin.init import _INIT_UPDATE
    from agent_cli.render.replay import _match_init
    content = _INIT_UPDATE.format(target="AGENTS.md", focus="\n\nFocus: env vars")
    assert _match_init(content) == "env vars"


def test_match_init_returns_none_for_plain_text() -> None:
    from agent_cli.render.replay import _match_init
    assert _match_init("hello world") is None
    assert _match_init("") is None


def test_match_review_default_target_returns_empty_args() -> None:
    from agent_cli.commands.builtin.review import _DEFAULT_REVIEW_TARGET, _REVIEW_PROMPT
    from agent_cli.render.replay import _match_review
    content = _REVIEW_PROMPT.format(target=_DEFAULT_REVIEW_TARGET)
    assert _match_review(content) == ""


def test_match_review_custom_target_returns_args() -> None:
    from agent_cli.commands.builtin.review import _REVIEW_PROMPT
    from agent_cli.render.replay import _match_review
    content = _REVIEW_PROMPT.format(target="src/agent_cli/commands")
    assert _match_review(content) == "src/agent_cli/commands"


def test_match_review_returns_none_for_plain_text() -> None:
    from agent_cli.render.replay import _match_review
    assert _match_review("hello world") is None
    assert _match_review("") is None


def test_match_skill_with_args() -> None:
    from agent_cli.render.replay import _match_skill
    content = (
        'find docs\n\n'
        '<system-reminder>The user has explicitly requested the web-search '
        'skill. Apply the skill instructions below to address their '
        'request.</system-reminder>\n\n'
        '<skill-loaded name="web-search">\nbody\n</skill-loaded>'
    )
    assert _match_skill(content) == ("web-search", "find docs")


def test_match_skill_without_args() -> None:
    from agent_cli.render.replay import _match_skill
    content = (
        '<system-reminder>The user has explicitly requested the humanizer '
        'skill. Apply the skill instructions below to address their '
        'request.</system-reminder>\n\n'
        '<skill-loaded name="humanizer">\nbody\n</skill-loaded>'
    )
    assert _match_skill(content) == ("humanizer", "")


def test_match_skill_returns_none_for_plain_text() -> None:
    from agent_cli.render.replay import _match_skill
    assert _match_skill("hello world") is None
    assert _match_skill("") is None
    # partial envelope (missing skill-loaded) → no match
    assert _match_skill(
        "args\n\n<system-reminder>The user has explicitly requested the foo "
        "skill.</system-reminder>"
    ) is None


# ── peel_user_command — canonical "what user typed" string ─────────────


def test_peel_user_command_shell_run_returns_bang_form() -> None:
    from agent_cli.render.notices import format_shell_run
    from agent_cli.render.replay import peel_user_command

    content = format_shell_run("ls -la", 0, "out")
    assert peel_user_command(content) == "! ls -la"


def test_peel_user_command_init_with_focus() -> None:
    from agent_cli.commands.builtin.init import _INIT_NEW
    from agent_cli.render.replay import peel_user_command

    content = _INIT_NEW.format(focus="\n\nFocus: testing strategy")
    assert peel_user_command(content) == "/init testing strategy"


def test_peel_user_command_init_without_focus() -> None:
    from agent_cli.commands.builtin.init import _INIT_NEW
    from agent_cli.render.replay import peel_user_command

    content = _INIT_NEW.format(focus="")
    assert peel_user_command(content) == "/init"


def test_peel_user_command_review_default_target_drops_args() -> None:
    from agent_cli.commands.builtin.review import _DEFAULT_REVIEW_TARGET, _REVIEW_PROMPT
    from agent_cli.render.replay import peel_user_command

    content = _REVIEW_PROMPT.format(target=_DEFAULT_REVIEW_TARGET)
    assert peel_user_command(content) == "/review"


def test_peel_user_command_review_custom_target() -> None:
    from agent_cli.commands.builtin.review import _REVIEW_PROMPT
    from agent_cli.render.replay import peel_user_command

    content = _REVIEW_PROMPT.format(target="src/agent_cli")
    assert peel_user_command(content) == "/review src/agent_cli"


def test_peel_user_command_skill_with_args() -> None:
    from agent_cli.render.replay import peel_user_command

    content = (
        "find docs\n\n"
        "<system-reminder>The user has explicitly requested the web-search "
        "skill. Apply the skill instructions below to address their "
        "request.</system-reminder>\n\n"
        '<skill-loaded name="web-search">\nbody\n</skill-loaded>'
    )
    assert peel_user_command(content) == "/web-search find docs"


def test_peel_user_command_skill_without_args() -> None:
    from agent_cli.render.replay import peel_user_command

    content = (
        "<system-reminder>The user has explicitly requested the humanizer "
        "skill. Apply the skill instructions below to address their "
        "request.</system-reminder>\n\n"
        '<skill-loaded name="humanizer">\nbody\n</skill-loaded>'
    )
    assert peel_user_command(content) == "/humanizer"


def test_peel_user_command_returns_none_for_plain_text() -> None:
    from agent_cli.render.replay import peel_user_command

    assert peel_user_command("hello world") is None
    assert peel_user_command("") is None


# ── replay integration: command invocations render as ❯ /<cmd> args ──


def test_replay_init_invocation_renders_as_slash_command() -> None:
    from agent_cli.commands.builtin.init import _INIT_NEW
    content = _INIT_NEW.format(focus="\n\nFocus: testing")
    out = _render(_u(content))
    assert "/init testing" in out
    assert "Generate a file named AGENTS.md" not in out


def test_replay_review_default_target_omits_args() -> None:
    from agent_cli.commands.builtin.review import _DEFAULT_REVIEW_TARGET, _REVIEW_PROMPT
    content = _REVIEW_PROMPT.format(target=_DEFAULT_REVIEW_TARGET)
    out = _render(_u(content))
    assert "/review" in out
    # default target should not appear in the rendered output
    assert _DEFAULT_REVIEW_TARGET not in out
    # template prefix should be gone too
    assert "reviewing a code change" not in out


def test_replay_review_custom_target_shows_args() -> None:
    from agent_cli.commands.builtin.review import _REVIEW_PROMPT
    content = _REVIEW_PROMPT.format(target="src/foo.py")
    out = _render(_u(content))
    assert "/review src/foo.py" in out


def test_replay_skill_invocation_renders_compactly() -> None:
    content = (
        'find docs\n\n'
        '<system-reminder>The user has explicitly requested the web-search '
        'skill. Apply the skill instructions below to address their '
        'request.</system-reminder>\n\n'
        '<skill-loaded name="web-search">\nthe full skill body here\n'
        '</skill-loaded>'
    )
    out = _render(_u(content))
    assert "/web-search find docs" in out
    # the verbose body and system-reminder must not leak
    assert "the full skill body here" not in out
    assert "<system-reminder>" not in out
    assert "<skill-loaded" not in out


def test_replay_skill_invocation_without_args() -> None:
    content = (
        '<system-reminder>The user has explicitly requested the humanizer '
        'skill. Apply the skill instructions below to address their '
        'request.</system-reminder>\n\n'
        '<skill-loaded name="humanizer">\nbody\n</skill-loaded>'
    )
    out = _render(_u(content))
    assert "/humanizer" in out
    assert "<skill-loaded" not in out


# ── @file mention attachment turn ────────────────────────────────────


def test_replay_at_mention_renders_attachment_block() -> None:
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "foo.py"})
    msgs = [
        _u("@foo.py what does this?"),
        _a("", [tc]),
        _t("c1", "content here"),
        _a("It is the foo module."),
    ]
    out = _render(*msgs)
    assert "Loaded into context" in out
    assert "foo.py" in out
    # the regular `Read(foo.py)` style must NOT appear in attachment turn
    assert "Read(foo.py)" not in out
    # subsequent assistant message still rendered
    assert "It is the foo module." in out


def test_replay_at_mention_without_at_in_user_falls_back_to_regular() -> None:
    """User msg without '@' but assistant uses empty content + read_file — not
    treated as attachment turn (could be an agent's own tool call)."""
    tc = ToolCall(id="c1", name="read_file", arguments={"file_path": "foo.py"})
    msgs = [
        _u("show me foo.py"),
        _a("", [tc]),
        _t("c1", "content"),
    ]
    out = _render(*msgs)
    assert "Loaded into context" not in out
    assert "Read(foo.py)" in out


def test_replay_at_mention_with_non_attachment_tool_falls_back() -> None:
    """User msg has '@' but the assistant's tool isn't read_file/list_dir — not
    an attachment turn (an agent's own use of the @ char doesn't qualify)."""
    tc = ToolCall(id="c1", name="terminal_tool", arguments={"command": "ls"})
    msgs = [
        _u("@example.com please email"),
        _a("", [tc]),
        _t("c1", "ok"),
    ]
    out = _render(*msgs)
    assert "Loaded into context" not in out


# ── compaction marker via render_post_switch ─────────────────────────


def test_render_post_switch_emits_marker_when_summary_present() -> None:
    from io import StringIO
    from unittest.mock import MagicMock
    from rich.console import Console
    from agent_cli.render.replay import render_post_switch
    from agent_cli.theme import DEFAULT_THEME
    from agent_harness.core.message import Message

    summary = Message.system(
        "(prior history summary)",
        metadata={"is_compression_summary": True, "compression_round": 1},
    )
    user_msg = Message.user("continue")
    agent = MagicMock()
    agent.context.short_term_memory._messages = [summary, user_msg]

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120, theme=DEFAULT_THEME.rich)
    render_post_switch(agent, console, DEFAULT_THEME, "session-id")
    out = buf.getvalue()
    assert "── Earlier messages compacted ──" in out


def test_render_post_switch_emits_round_count_when_above_one() -> None:
    from io import StringIO
    from unittest.mock import MagicMock
    from rich.console import Console
    from agent_cli.render.replay import render_post_switch
    from agent_cli.theme import DEFAULT_THEME
    from agent_harness.core.message import Message

    summary = Message.system(
        "(merged summary)",
        metadata={"is_compression_summary": True, "compression_round": 3},
    )
    user_msg = Message.user("ok")
    agent = MagicMock()
    agent.context.short_term_memory._messages = [summary, user_msg]

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120, theme=DEFAULT_THEME.rich)
    render_post_switch(agent, console, DEFAULT_THEME, "session-id")
    out = buf.getvalue()
    assert "── Earlier messages compacted ×3 ──" in out


def test_render_post_switch_omits_marker_without_summary() -> None:
    from io import StringIO
    from unittest.mock import MagicMock
    from rich.console import Console
    from agent_cli.render.replay import render_post_switch
    from agent_cli.theme import DEFAULT_THEME
    from agent_harness.core.message import Message

    agent = MagicMock()
    agent.context.short_term_memory._messages = [Message.user("hi")]

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120, theme=DEFAULT_THEME.rich)
    render_post_switch(agent, console, DEFAULT_THEME, "session-id")
    out = buf.getvalue()
    assert "Earlier messages compacted" not in out


def test_replay_at_mention_multiple_files_in_one_block() -> None:
    tcs = [
        ToolCall(id="c1", name="read_file", arguments={"file_path": "a.py"}),
        ToolCall(id="c2", name="list_dir", arguments={"path": "src"}),
    ]
    msgs = [
        _u("compare @a.py and @src"),
        _a("", tcs),
        _t("c1", "a"),
        _t("c2", "listing"),
    ]
    out = _render(*msgs)
    # one shared "Loaded into context" header, not two
    assert out.count("Loaded into context") == 1
    assert "a.py" in out
    assert "src" in out


# ── render_post_switch ───────────────────────────────────────────────


def _post_switch(messages: list[Message]) -> str:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=False, color_system=None,
        width=200, theme=DEFAULT_THEME.rich,
    )
    agent = MagicMock()
    agent.context.short_term_memory._messages = messages
    render_post_switch(agent, console, DEFAULT_THEME, "abc123")
    return buf.getvalue()


def test_hard_clear_writes_viewport_and_scrollback_sequences() -> None:
    buf = StringIO()
    console = Console(
        file=buf, force_terminal=False, color_system=None,
        width=200, theme=DEFAULT_THEME.rich,
    )
    _hard_clear(console)
    raw = buf.getvalue()
    assert "\x1b[2J" in raw
    assert "\x1b[3J" in raw
    assert "\x1b[H" in raw


def test_render_post_switch_empty_session_shows_new_marker() -> None:
    out = _post_switch([])
    assert "New session" in out
    assert "abc123" in out


def test_render_post_switch_with_messages_replays_and_shows_resumed_marker() -> None:
    out = _post_switch([_u("hi"), _a("hello")])
    assert "hi" in out
    assert "hello" in out
    assert "Resumed" in out
    assert "abc123" in out


def test_render_post_switch_resumed_marker_appears_after_replay() -> None:
    out = _post_switch([_u("greet"), _a("response")])
    assert out.index("response") < out.index("Resumed")


# ── todo_write panel replay ──────────────────────────────────────────


def _todo_call(call_id: str, todos: list[dict[str, str]]) -> ToolCall:
    return ToolCall(id=call_id, name="todo_write", arguments={"todos": todos})


def test_replay_todo_write_renders_tasks_panel() -> None:
    todos = [
        {"id": "1", "status": "in_progress", "content": "wire up handler"},
        {"id": "2", "status": "pending", "content": "write tests"},
    ]
    tc = _todo_call("c1", todos)
    tr = _t("c1", "[0/2] In progress: wire up handler | Pending: write tests")
    out = _render(_a(calls=[tc]), tr)
    assert "Tasks [0/2]" in out
    assert "wire up handler" in out
    assert "write tests" in out
    # the suppressed call row must NOT appear
    assert "todo_write(" not in out


def test_replay_todo_write_alongside_other_tool_panel_comes_last() -> None:
    read_tc = ToolCall(
        id="c0", name="read_file", arguments={"file_path": "/a.py"},
    )
    todo_tc = _todo_call(
        "c1", [{"id": "1", "status": "in_progress", "content": "task A"}],
    )
    read_tr = _t("c0", "lines 1-5")
    todo_tr = _t("c1", "[0/1] In progress: task A")
    out = _render(_a(calls=[read_tc, todo_tc]), read_tr, todo_tr)
    # both rendered
    assert "/a.py" in out
    assert "Tasks [0/1]" in out
    # panel comes after the read_file row
    assert out.index("/a.py") < out.index("Tasks [0/1]")


def test_replay_multiple_todo_writes_in_one_msg_keeps_last() -> None:
    first = _todo_call("c1", [{"id": "1", "status": "in_progress", "content": "first phase"}])
    second = _todo_call(
        "c2",
        [
            {"id": "1", "status": "completed", "content": "first phase"},
            {"id": "2", "status": "in_progress", "content": "second phase"},
        ],
    )
    tr1 = _t("c1", "ok")
    tr2 = _t("c2", "ok")
    out = _render(_a(calls=[first, second]), tr1, tr2)
    # only one panel, showing the second call's state
    assert out.count("Tasks [") == 1
    assert "Tasks [1/2]" in out
    assert "second phase" in out


def test_replay_todo_write_failed_then_successful_uses_successful() -> None:
    bad = _todo_call("c1", [{"id": "1", "status": "in_progress", "content": "bad phase"}])
    good = _todo_call("c2", [{"id": "2", "status": "in_progress", "content": "good phase"}])
    bad_tr = _t("c1", "Error: validation failed", is_error=True)
    good_tr = _t("c2", "ok")
    out = _render(_a(calls=[bad, good]), bad_tr, good_tr)
    assert "Tasks [" in out
    assert "good phase" in out
    assert "bad phase" not in out


def test_replay_todo_write_successful_then_failed_uses_successful() -> None:
    # Failure on the later call must not overwrite the earlier successful one.
    good = _todo_call("c1", [{"id": "1", "status": "in_progress", "content": "good phase"}])
    bad = _todo_call("c2", [{"id": "2", "status": "in_progress", "content": "bad phase"}])
    good_tr = _t("c1", "ok")
    bad_tr = _t("c2", "Error: too many tasks", is_error=True)
    out = _render(_a(calls=[good, bad]), good_tr, bad_tr)
    assert "Tasks [" in out
    assert "good phase" in out
    assert "bad phase" not in out


def test_replay_todo_write_only_failed_renders_nothing() -> None:
    bad = _todo_call("c1", [{"id": "1", "status": "in_progress", "content": "broken"}])
    bad_tr = _t("c1", "Error: validation failed", is_error=True)
    out = _render(_a(calls=[bad]), bad_tr)
    assert "Tasks [" not in out
    assert "todo_write(" not in out
    assert "broken" not in out


def test_replay_todo_write_without_result_renders_nothing() -> None:
    # Interrupted mid-flight — tool_call exists, result missing.
    tc = _todo_call("c1", [{"id": "1", "status": "pending", "content": "incomplete"}])
    out = _render(_a(calls=[tc]))
    assert "Tasks [" not in out
    assert "todo_write(" not in out


def test_replay_todo_write_across_two_messages_renders_two_panels() -> None:
    tc1 = _todo_call("c1", [{"id": "1", "status": "in_progress", "content": "step one"}])
    tc2 = _todo_call(
        "c2",
        [
            {"id": "1", "status": "completed", "content": "step one"},
            {"id": "2", "status": "in_progress", "content": "step two"},
        ],
    )
    out = _render(
        _a(calls=[tc1]), _t("c1", "ok"),
        _a(calls=[tc2]), _t("c2", "ok"),
    )
    # two distinct panels rendered
    assert out.count("Tasks [") == 2
    assert "Tasks [0/1]" in out
    assert "Tasks [1/2]" in out
    assert out.index("Tasks [0/1]") < out.index("Tasks [1/2]")
