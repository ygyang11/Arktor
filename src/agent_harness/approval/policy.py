"""Resource-aware approval policy engine."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from agent_harness.approval.rules import (
    any_rule_matches,
    has_tool_level_rule,
    parse_rules,
)

ApprovalMode = Literal["never", "ask", "auto"]
from agent_harness.approval.types import ApprovalAction
from agent_harness.core.message import ToolCall

_CHAIN_RE = re.compile(r"\s*(?:&&|\|\||[;|])\s*")
_UNSAFE_SHELL_RE = re.compile(
    r"`"
    r"|\$\("
    r"|\$'"
    r"|\$\{"
    r"|[<>]"
    r"|(?<!&)&(?!&)"
    r"|\n"
)


def derive_session_prefix(resource: str, kind: str) -> str:
    """Derive a session grant prefix from a concrete resource.

    path:    directories grant the directory itself; files grant the parent
             directory, collapsing to the file itself when the parent would be
             the filesystem root.
    url:     "https://github.com/repo" -> "github.com" (hostname)
    command: "git status" -> "git" (first word)
    """
    if kind == "path":
        normed = os.path.normpath(resource)
        try:
            if Path(normed).is_dir():
                return normed
        except OSError:
            pass
        parent = os.path.dirname(normed)
        if parent and parent != os.sep:
            return parent
        return normed
    if kind == "url":
        try:
            return urlparse(resource).hostname or resource
        except Exception:
            return resource
    if kind == "command":
        return resource.split()[0] if resource.strip() else resource
    return resource


class ApprovalPolicy:
    """Resource-aware approval policy engine.

    Combines deny/allow rule evaluation with session-level grants.
    Supports segment-aware checking for chained shell commands.
    """

    def __init__(
        self,
        *,
        mode: ApprovalMode = "auto",
        always_allow: set[str] | None = None,
        always_deny: set[str] | None = None,
    ) -> None:
        self._mode: ApprovalMode = mode
        self._deny_rules = parse_rules(always_deny or set())
        self._allow_rules = parse_rules(always_allow or set())
        self._session_grants: dict[str, None | set[tuple[str, str]]] = {}

    @property
    def mode(self) -> ApprovalMode:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("never", "ask", "auto"):
            raise ValueError(f"unknown approval mode: {mode}")
        self._mode = cast(ApprovalMode, mode)

    def allows_session_grants(self) -> bool:
        return self._mode == "auto"

    def check(
        self,
        tool_call: ToolCall,
        resource: str | None = None,
        kind: str | None = None,
    ) -> ApprovalAction:
        """Resource-aware approval check."""
        if self._mode == "never":
            return ApprovalAction.EXECUTE

        name = tool_call.name

        if self._mode == "ask":
            if self._deny_matches(name, resource, kind):
                return ApprovalAction.DENY
            return ApprovalAction.ASK

        if kind == "command" and resource is not None:
            return self._check_command(name, resource)

        return self._check_generic(name, resource)

    def grant_session(
        self,
        tool_name: str,
        resource: str | None = None,
        kind: str | None = None,
    ) -> None:
        """Grant session-level approval."""
        if resource is None or kind is None:
            self._session_grants[tool_name] = None
            return
        if tool_name in self._session_grants and self._session_grants[tool_name] is None:
            return

        bucket = self._session_grants.setdefault(tool_name, set())
        assert isinstance(bucket, set)

        if kind == "command":
            segments = [s.strip() for s in _CHAIN_RE.split(resource) if s.strip()]
            for seg in segments:
                prefix = derive_session_prefix(seg, kind)
                bucket.add((prefix, kind))
        else:
            prefix = derive_session_prefix(resource, kind)
            bucket.add((prefix, kind))

    def reset_session(self) -> None:
        """Clear all session-level grants."""
        self._session_grants.clear()

    def export_session_grants(self) -> dict[str, Any]:
        """Serialize session grants for persistence.

        Converts internal set[tuple] structure to JSON-safe list[list].
        """
        if not self._session_grants:
            return {}
        return {
            k: (sorted([list(t) for t in v]) if v is not None else None)
            for k, v in self._session_grants.items()
        }

    def import_session_grants(self, data: dict[str, Any]) -> None:
        """Restore session grants from persisted data (authoritative).

        Always clears existing grants first, then restores from data.
        Empty data = clear all grants (not a no-op).
        """
        self._session_grants.clear()
        for k, v in data.items():
            self._session_grants[k] = (
                set(tuple(t) for t in v) if v is not None else None
            )

    # ── internal ──

    def _deny_matches(
        self, name: str, resource: str | None, kind: str | None,
    ) -> bool:
        """Segment-aware deny check shared by ask / auto modes.

        Mirrors `_check_command`'s split: unsafe-shell commands match against
        the whole string; safe chained commands match each segment so that
        `terminal_tool(rm *)` still blocks `git status && rm -rf .`.
        """
        if kind == "command" and resource is not None:
            if _UNSAFE_SHELL_RE.search(resource):
                return any_rule_matches(self._deny_rules, name, resource)
            segments = [s.strip() for s in _CHAIN_RE.split(resource) if s.strip()]
            if not segments:
                return False
            return any(
                any_rule_matches(self._deny_rules, name, seg) for seg in segments
            )
        return any_rule_matches(self._deny_rules, name, resource)

    def _check_generic(self, name: str, resource: str | None) -> ApprovalAction:
        """deny > allow > session > ASK."""
        if self._deny_matches(name, resource, None):
            return ApprovalAction.DENY
        if any_rule_matches(self._allow_rules, name, resource):
            return ApprovalAction.EXECUTE
        if self._session_matches(name, resource):
            return ApprovalAction.EXECUTE
        return ApprovalAction.ASK

    def _check_command(self, name: str, command: str) -> ApprovalAction:
        """Segment-aware check for chained commands.

        Unsafe shell patterns fall back to tool-level, but deny rules are
        still checked against the full command string first.
        """
        if self._deny_matches(name, command, "command"):
            return ApprovalAction.DENY

        if _UNSAFE_SHELL_RE.search(command):
            return self._check_generic(name, None)

        segments = [s.strip() for s in _CHAIN_RE.split(command) if s.strip()]
        if not segments:
            return self._check_generic(name, None)

        if has_tool_level_rule(self._allow_rules, name):
            return ApprovalAction.EXECUTE

        if all(
            any_rule_matches(self._allow_rules, name, seg)
            or self._segment_in_session(name, seg)
            for seg in segments
        ):
            return ApprovalAction.EXECUTE

        return ApprovalAction.ASK

    def _session_matches(self, name: str, resource: str | None) -> bool:
        """Generic session grant matching (path / url / tool-level)."""
        if name not in self._session_grants:
            return False
        grants = self._session_grants[name]
        if grants is None:
            return True
        if resource is None:
            return False
        for prefix, kind in grants:
            if kind == "path":
                normed = os.path.normpath(resource)
                if normed == prefix or normed.startswith(prefix + os.sep):
                    return True
            elif kind == "url":
                try:
                    if (urlparse(resource).hostname or "") == prefix:
                        return True
                except Exception:
                    pass
        return False

    def _segment_in_session(self, name: str, segment: str) -> bool:
        """Check if a single command segment is covered by session grants."""
        if name not in self._session_grants:
            return False
        grants = self._session_grants[name]
        if grants is None:
            return True
        cmd_prefixes = {p for p, k in grants if k == "command"}
        first_word = segment.split()[0] if segment.strip() else ""
        return first_word in cmd_prefixes
