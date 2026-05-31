"""Tests for resource-aware ApprovalPolicy."""
from pathlib import Path

from agent_harness.approval.policy import ApprovalPolicy, derive_session_prefix
from agent_harness.approval.rules import _canonicalize
from agent_harness.approval.types import ApprovalAction
from agent_harness.core.message import ToolCall

EXECUTE = ApprovalAction.EXECUTE
ASK = ApprovalAction.ASK
DENY = ApprovalAction.DENY


# ── backward compatibility (original tests) ──────────────────────────────────

class TestBackwardCompat:
    def test_mode_never_always_executes(self) -> None:
        p = ApprovalPolicy(mode="never")
        tc = ToolCall(name="dangerous_tool", arguments={})
        assert p.check(tc) == EXECUTE

    def test_mode_never_ignores_deny_list(self) -> None:
        p = ApprovalPolicy(mode="never", always_deny={"tool_x"})
        tc = ToolCall(name="tool_x", arguments={})
        assert p.check(tc) == EXECUTE

    def test_always_allow_executes(self) -> None:
        p = ApprovalPolicy(always_allow={"safe_tool"})
        tc = ToolCall(name="safe_tool", arguments={})
        assert p.check(tc) == EXECUTE

    def test_always_deny_denies(self) -> None:
        p = ApprovalPolicy(always_deny={"banned_tool"})
        tc = ToolCall(name="banned_tool", arguments={})
        assert p.check(tc) == DENY

    def test_deny_overrides_allow(self) -> None:
        p = ApprovalPolicy(always_allow={"tool_x"}, always_deny={"tool_x"})
        tc = ToolCall(name="tool_x", arguments={})
        assert p.check(tc) == DENY

    def test_unknown_tool_asks_in_auto(self) -> None:
        p = ApprovalPolicy(mode="auto")
        tc = ToolCall(name="unknown_tool", arguments={})
        assert p.check(tc) == ASK

    def test_unknown_tool_asks_in_always(self) -> None:
        p = ApprovalPolicy(mode="always")
        tc = ToolCall(name="any_tool", arguments={})
        assert p.check(tc) == ASK

    def test_session_allow_remembered(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("my_tool")
        tc = ToolCall(name="my_tool", arguments={})
        assert p.check(tc) == EXECUTE

    def test_reset_session_clears_grants(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("my_tool")
        p.reset_session()
        tc = ToolCall(name="my_tool", arguments={})
        assert p.check(tc) == ASK

    def test_session_allow_does_not_override_deny(self) -> None:
        p = ApprovalPolicy(mode="auto", always_deny={"tool_x"})
        p.grant_session("tool_x")
        tc = ToolCall(name="tool_x", arguments={})
        assert p.check(tc) == DENY

    def test_default_mode_is_auto(self) -> None:
        p = ApprovalPolicy()
        tc = ToolCall(name="any_tool", arguments={})
        assert p.check(tc) == ASK

    def test_no_resource_falls_to_tool_level(self) -> None:
        p = ApprovalPolicy(always_allow={"read_file"})
        tc = ToolCall(name="read_file", arguments={})
        assert p.check(tc) == EXECUTE


# ── deny > allow > session priority ──────────────────────────────────────────

class TestPriority:
    def test_deny_beats_allow_resource_level(self) -> None:
        p = ApprovalPolicy(
            always_deny={"write_file(**/.env*)"},
            always_allow={"write_file"},
        )
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource=".env.local", kind="path") == DENY

    def test_allow_beats_session(self) -> None:
        p = ApprovalPolicy(always_allow={"read_file(src/**)"})
        tc = ToolCall(name="read_file", arguments={})
        assert p.check(tc, resource="src/main.py", kind="path") == EXECUTE

    def test_no_match_returns_ask(self) -> None:
        p = ApprovalPolicy(mode="auto")
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource="src/main.py", kind="path") == ASK

    def test_resource_allow_no_match(self) -> None:
        p = ApprovalPolicy(always_allow={"write_file(src/**)"})
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource="other/file.py", kind="path") == ASK

    def test_resource_deny_specific(self) -> None:
        p = ApprovalPolicy(
            always_deny={"write_file(**/*.pem)"},
            always_allow={"write_file(certs/**)"},
        )
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource="certs/server.pem", kind="path") == DENY


# ── command segment-aware ────────────────────────────────────────────────────

class TestCommandSegment:
    def test_chain_all_allowed(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git add && git push", kind="command") == EXECUTE

    def test_chain_partial_asks(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git status && rm -rf .", kind="command") == ASK

    def test_chain_deny_any(self) -> None:
        p = ApprovalPolicy(always_deny={"terminal_tool(rm *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git status ; rm -rf .", kind="command") == DENY

    def test_allow_session_merged(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        p.grant_session("terminal_tool", resource="pytest tests/", kind="command")
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git add && pytest tests/", kind="command") == EXECUTE

    def test_pipe_splits_segments(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git log | head -5", kind="command") == ASK


# ── unsafe shell fallback ────────────────────────────────────────────────────

class TestUnsafeShell:
    def test_subshell_substitution(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git $(rm -rf /)", kind="command") == ASK

    def test_backtick(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git `rm -rf /`", kind="command") == ASK

    def test_ansi_c(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(echo *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="echo $'\\x41'", kind="command") == ASK

    def test_redirect(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git status > .env", kind="command") == ASK

    def test_newline(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git status\nrm -rf .", kind="command") == ASK

    def test_background(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(git *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git status &", kind="command") == ASK

    def test_parens_not_detected(self) -> None:
        """( in arguments must NOT trigger unsafe detection."""
        p = ApprovalPolicy(always_allow={"terminal_tool(python *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource='python -c "print(1)"', kind="command") == EXECUTE

    def test_unsafe_still_checks_deny(self) -> None:
        """Deny rules checked against full command even when unsafe detected."""
        p = ApprovalPolicy(always_deny={"terminal_tool(rm *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="rm -rf / > /dev/null", kind="command") == DENY


# ── session grant: command ───────────────────────────────────────────────────

class TestSessionCommand:
    def test_chain_grants_all_segments(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("terminal_tool", resource="git add && pytest tests/", kind="command")
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git push", kind="command") == EXECUTE
        assert p.check(tc, resource="pytest -v", kind="command") == EXECUTE
        assert p.check(tc, resource="rm file", kind="command") == ASK

    def test_prefix(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("terminal_tool", resource="git status", kind="command")
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="git push", kind="command") == EXECUTE

    def test_word_boundary(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("terminal_tool", resource="git status", kind="command")
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="gitevil", kind="command") == ASK


# ── session grant: path ──────────────────────────────────────────────────────

class TestSessionPath:
    def test_parent_dir(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file", resource="src/main.py", kind="path")
        tc = ToolCall(name="read_file", arguments={})
        assert p.check(tc, resource="src/utils/helper.py", kind="path") == EXECUTE

    def test_sep_boundary(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file", resource="src/main.py", kind="path")
        tc = ToolCall(name="read_file", arguments={})
        assert p.check(tc, resource="src_evil/file.py", kind="path") == ASK


# ── session grant: directory semantics ───────────────────────────────────────

class TestDirGrantSemantics:
    def test_dir_grant_matches_subpath_same_tool(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        p = ApprovalPolicy(mode="auto")
        p.grant_session("list_dir", resource=str(tmp_path), kind="path")
        tc = ToolCall(name="list_dir", arguments={})
        assert p.check(tc, resource=str(sub), kind="path") == EXECUTE

    def test_dir_grant_does_not_escalate_to_sibling(self, tmp_path: Path) -> None:
        sub_a = tmp_path / "a"
        sub_a.mkdir()
        sub_b = tmp_path / "b"
        sub_b.mkdir()
        p = ApprovalPolicy(mode="auto")
        p.grant_session("list_dir", resource=str(sub_a), kind="path")
        tc = ToolCall(name="list_dir", arguments={})
        assert p.check(tc, resource=str(sub_b), kind="path") == ASK

    def test_dir_grant_does_not_cross_tools(self, tmp_path: Path) -> None:
        sub_file = tmp_path / "x.md"
        sub_file.write_text("x")
        p = ApprovalPolicy(mode="auto")
        p.grant_session("list_dir", resource=str(tmp_path), kind="path")
        tc = ToolCall(name="read_file", arguments={})
        assert p.check(tc, resource=str(sub_file), kind="path") == ASK

    def test_top_level_file_grant_self_only(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file", resource="/foo.txt", kind="path")
        tc = ToolCall(name="read_file", arguments={})
        assert p.check(tc, resource="/foo.txt", kind="path") == EXECUTE
        assert p.check(tc, resource="/bar.txt", kind="path") == ASK

    def test_workspace_root_grant_covers_relative_subpaths(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("list_dir", resource=".", kind="path")
        tc = ToolCall(name="list_dir", arguments={})
        assert p.check(tc, resource=".", kind="path") == EXECUTE
        assert p.check(tc, resource="src", kind="path") == EXECUTE
        assert p.check(tc, resource="tests/x.py", kind="path") == EXECUTE
        # external (absolute) paths are not covered by a workspace-root grant
        assert p.check(tc, resource="/etc/passwd", kind="path") == ASK


# ── session grant: url ───────────────────────────────────────────────────────

class TestSessionUrl:
    def test_hostname(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("web_fetch", resource="https://github.com/repo", kind="url")
        tc = ToolCall(name="web_fetch", arguments={})
        assert p.check(tc, resource="https://github.com/other", kind="url") == EXECUTE
        assert p.check(tc, resource="https://evil.com/phish", kind="url") == ASK


# ── session: edge cases ──────────────────────────────────────────────────────

class TestSessionEdge:
    def test_tool_level_no_downgrade(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("terminal_tool")
        p.grant_session("terminal_tool", resource="git status", kind="command")
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="rm -rf /", kind="command") == EXECUTE

    def test_deny_overrides_session(self) -> None:
        p = ApprovalPolicy(always_deny={"terminal_tool(rm *)"})
        p.grant_session("terminal_tool", resource="rm -rf /", kind="command")
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="rm -rf /", kind="command") == DENY


# ── safety: path traversal ───────────────────────────────────────────────────

class TestSafety:
    def test_traversal_deny(self) -> None:
        resource = _canonicalize("src/../.env", "file_path")
        p = ApprovalPolicy(always_deny={"write_file(**/.env*)"})
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource=resource, kind="path") == DENY

    def test_deny_env_root_level(self) -> None:
        p = ApprovalPolicy(always_deny={"write_file(**/.env*)"})
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource=".env", kind="path") == DENY
        assert p.check(tc, resource=".env.local", kind="path") == DENY

    def test_traversal_not_allowed(self) -> None:
        resource = _canonicalize("src/../.env", "file_path")
        p = ApprovalPolicy(always_allow={"write_file(src/**)"})
        tc = ToolCall(name="write_file", arguments={})
        assert p.check(tc, resource=resource, kind="path") == ASK

    def test_subshell_blocked_by_first_word(self) -> None:
        p = ApprovalPolicy(always_allow={"terminal_tool(rm *)"})
        tc = ToolCall(name="terminal_tool", arguments={"command": "x"})
        assert p.check(tc, resource="(rm -rf /)", kind="command") == ASK

    def test_domain_suffix_attack(self) -> None:
        p = ApprovalPolicy(always_allow={"web_fetch(domain:github.com)"})
        tc = ToolCall(name="web_fetch", arguments={})
        assert p.check(tc, resource="https://github.com.evil.com/x", kind="url") == ASK


# ── derive_session_prefix ────────────────────────────────────────────────────

class TestDerivePrefix:
    def test_path_parent(self) -> None:
        assert derive_session_prefix("src/utils/helper.py", "path") == "src/utils"

    def test_path_root_file(self) -> None:
        assert derive_session_prefix("README.md", "path") == "README.md"

    def test_url_hostname(self) -> None:
        assert derive_session_prefix("https://github.com/repo", "url") == "github.com"

    def test_command_first_word(self) -> None:
        assert derive_session_prefix("git status", "command") == "git"

    def test_command_empty(self) -> None:
        assert derive_session_prefix("", "command") == ""


class TestSessionGrantsPersistence:
    def test_export_empty(self) -> None:
        p = ApprovalPolicy(mode="auto")
        assert p.export_session_grants() == {}

    def test_export_import_roundtrip_tool_level(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file")

        exported = p.export_session_grants()
        assert "read_file" in exported
        assert exported["read_file"] is None

        p2 = ApprovalPolicy(mode="auto")
        p2.import_session_grants(exported)
        assert p2._session_grants == p._session_grants

    def test_export_import_roundtrip_resource_level(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file", resource="src/main.py", kind="path")

        exported = p.export_session_grants()
        assert "read_file" in exported
        assert exported["read_file"] is not None

        p2 = ApprovalPolicy(mode="auto")
        p2.import_session_grants(exported)
        assert p2._session_grants == p._session_grants

    def test_import_empty_data_clears_existing(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file")
        assert "read_file" in p._session_grants

        p.import_session_grants({})
        assert p._session_grants == {}

    def test_import_overwrites_existing(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file")

        p.import_session_grants({"write_file": None})
        assert "write_file" in p._session_grants
        assert "read_file" not in p._session_grants

    def test_export_import_mixed_grants(self) -> None:
        p = ApprovalPolicy(mode="auto")
        p.grant_session("read_file")
        p.grant_session("terminal_tool", resource="git status", kind="command")

        exported = p.export_session_grants()
        p2 = ApprovalPolicy(mode="auto")
        p2.import_session_grants(exported)
        assert p2._session_grants == p._session_grants
