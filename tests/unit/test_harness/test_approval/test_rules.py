"""Tests for approval/rules.py — rule parsing, pattern matching, resource extraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.approval.rules import (
    PermissionRule,
    _canonicalize,
    any_rule_matches,
    extract_resource,
    has_tool_level_rule,
    match_pattern,
    parse_rules,
    rule_matches,
)

# ── parse_rules ──────────────────────────────────────────────────────────────

class TestParseRules:
    def test_tool_level(self) -> None:
        assert parse_rules({"read_file"}) == [PermissionRule("read_file", None)]

    def test_resource_level(self) -> None:
        assert parse_rules({"terminal_tool(git *)"}) == [
            PermissionRule("terminal_tool", "git *"),
        ]

    def test_domain_rule(self) -> None:
        assert parse_rules({"web_fetch(domain:*.github.com)"}) == [
            PermissionRule("web_fetch", "domain:*.github.com"),
        ]

    def test_starstar_pattern(self) -> None:
        assert parse_rules({"write_file(**/.env*)"}) == [
            PermissionRule("write_file", "**/.env*"),
        ]

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            parse_rules({"not valid!"})

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            parse_rules({""})

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            parse_rules({"   "})

    def test_accepts_list_input(self) -> None:
        result = parse_rules(["read_file", "write_file(src/**)"])
        assert len(result) == 2


# ── match_pattern: domain ────────────────────────────────────────────────────

class TestMatchDomain:
    def test_exact(self) -> None:
        assert match_pattern("domain:github.com", "https://github.com/org/repo") is True

    def test_wildcard(self) -> None:
        assert match_pattern(
            "domain:*.readthedocs.io", "https://flask.readthedocs.io/en/latest",
        ) is True

    def test_suffix_attack(self) -> None:
        assert match_pattern("domain:github.com", "https://github.com.evil.com/x") is False

    def test_invalid_url(self) -> None:
        assert match_pattern("domain:x.com", "not-a-url") is False

    def test_case_insensitive(self) -> None:
        assert match_pattern("domain:GitHub.COM", "https://github.com/repo") is True


# ── match_pattern: **/xxx (basename at any depth) ───────────────────────────

class TestMatchStarstarSlash:
    def test_root_level(self) -> None:
        assert match_pattern("**/.env*", ".env") is True
        assert match_pattern("**/.env*", ".env.local") is True

    def test_nested(self) -> None:
        assert match_pattern("**/.env*", "config/.env.production") is True
        assert match_pattern("**/*.pem", "certs/server.pem") is True

    def test_no_match(self) -> None:
        assert match_pattern("**/.env*", "README.md") is False
        assert match_pattern("**/*.pem", "config/server.key") is False

    def test_deeply_nested(self) -> None:
        assert match_pattern("**/.env*", "a/b/c/.env.staging") is True


# ── match_pattern: xxx/** (recursive directory prefix) ──────────────────────

class TestMatchRecursiveDir:
    def test_match(self) -> None:
        assert match_pattern("src/**", "src/sub/deep/file.py") is True

    def test_exact_prefix(self) -> None:
        assert match_pattern("src/**", "src") is True

    def test_sep_boundary(self) -> None:
        assert match_pattern("src/**", "src_evil/file.py") is False

    def test_direct_child(self) -> None:
        assert match_pattern("src/**", "src/main.py") is True


# ── match_pattern: PurePosixPath glob ───────────────────────────────────────

class TestMatchPosixGlob:
    def test_single_level(self) -> None:
        assert match_pattern("src/*.py", "src/main.py") is True

    def test_not_recursive(self) -> None:
        assert match_pattern("src/*.py", "src/sub/main.py") is False

    def test_nested_pattern(self) -> None:
        assert match_pattern("tests/unit/*.py", "tests/unit/test_foo.py") is True


# ── match_pattern: fnmatch fallback ─────────────────────────────────────────

class TestMatchFnmatch:
    def test_command(self) -> None:
        assert match_pattern("git *", "git status") is True

    def test_no_match_prefix(self) -> None:
        assert match_pattern("git *", "gitevil") is False

    def test_dotenv(self) -> None:
        assert match_pattern(".env*", ".env.local") is True

    def test_wildcard_all(self) -> None:
        assert match_pattern("*", "anything") is True


# ── rule_matches / any_rule_matches / has_tool_level_rule ────────────────────

class TestRuleMatching:
    def test_tool_level_matches_any_resource(self) -> None:
        rule = PermissionRule("read_file", None)
        assert rule_matches(rule, "read_file", "anything") is True
        assert rule_matches(rule, "read_file", None) is True

    def test_wrong_tool_name(self) -> None:
        rule = PermissionRule("read_file", None)
        assert rule_matches(rule, "write_file", "x") is False

    def test_resource_level_needs_resource(self) -> None:
        rule = PermissionRule("write_file", "src/**")
        assert rule_matches(rule, "write_file", None) is False
        assert rule_matches(rule, "write_file", "src/x.py") is True

    def test_any_rule_matches(self) -> None:
        rules = parse_rules({"read_file", "write_file(src/**)"})
        assert any_rule_matches(rules, "read_file", "anything") is True
        assert any_rule_matches(rules, "write_file", "src/x.py") is True
        assert any_rule_matches(rules, "write_file", "other/x.py") is False

    def test_has_tool_level_rule(self) -> None:
        rules = parse_rules({"read_file", "write_file(src/**)"})
        assert has_tool_level_rule(rules, "read_file") is True
        assert has_tool_level_rule(rules, "write_file") is False
        assert has_tool_level_rule(rules, "unknown") is False


# ── extract_resource ─────────────────────────────────────────────────────────

class TestExtractResource:
    def test_no_key(self) -> None:
        assert extract_resource("read_file", {"file_path": "x"}, None) == (None, None)

    def test_missing_arg(self) -> None:
        assert extract_resource("read_file", {}, "file_path") == (None, None)

    def test_path_kind(self) -> None:
        resource, kind = extract_resource("read_file", {"file_path": "src/main.py"}, "file_path")
        assert kind == "path"
        assert resource is not None

    def test_command_kind(self) -> None:
        resource, kind = extract_resource(
            "terminal_tool", {"command": "git status"}, "command",
        )
        assert kind == "command"
        assert resource == "git status"

    def test_url_kind(self) -> None:
        resource, kind = extract_resource(
            "web_fetch", {"url": "https://example.com"}, "url",
        )
        assert kind == "url"
        assert resource == "https://example.com"

    def test_unknown_key_returns_none_kind(self) -> None:
        resource, kind = extract_resource("my_tool", {"custom": "val"}, "custom")
        assert resource == "val"
        assert kind is None


# ── _canonicalize ────────────────────────────────────────────────────────────

class TestCanonicalize:
    def test_relative_path(self) -> None:
        result = _canonicalize("src/main.py", "file_path")
        assert result == "src/main.py"

    def test_traversal_normalized(self) -> None:
        result = _canonicalize("src/../README.md", "file_path")
        assert ".." not in result
        assert result == "README.md"

    def test_non_path_passthrough(self) -> None:
        assert _canonicalize("git status", "command") == "git status"
        assert _canonicalize("https://x.com", "url") == "https://x.com"

    def test_path_key_alias(self) -> None:
        result = _canonicalize("src/main.py", "path")
        assert result == "src/main.py"

    def test_home_expanded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _canonicalize("~/notes.md", "file_path")
        assert result == str((tmp_path / "notes.md").resolve())

    def test_outside_workspace_returns_absolute(self) -> None:
        result = _canonicalize("/etc/hosts", "file_path")
        assert result == str(Path("/etc/hosts").resolve())

    def test_outside_workspace_traversal_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = _canonicalize("../outside_dir", "file_path")
        assert result == str((tmp_path.parent / "outside_dir").resolve())
