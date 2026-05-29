"""Permission rule parsing, pattern matching, and resource extraction."""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Rule representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule.

    Attributes:
        tool_name: Target tool name.
        pattern: Resource matching pattern (None = tool-level, matches all invocations).
    """

    tool_name: str
    pattern: str | None = None

    @property
    def is_tool_level(self) -> bool:
        return self.pattern is None


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

_RULE_RE = re.compile(r"^([a-zA-Z_]\w*)\((.+)\)$")
_NAME_RE = re.compile(r"^[a-zA-Z_]\w*$")


def parse_rules(raw: set[str] | list[str]) -> list[PermissionRule]:
    """Parse raw rule strings into PermissionRule objects.

    Valid syntax:
      "read_file"                -> PermissionRule("read_file", None)
      "terminal_tool(git *)"    -> PermissionRule("terminal_tool", "git *")
      "web_fetch(domain:x.com)" -> PermissionRule("web_fetch", "domain:x.com")

    Raises ValueError on malformed rules (fail-fast at config load time).
    """
    rules: list[PermissionRule] = []
    for s in raw:
        s = s.strip()
        if not s:
            raise ValueError("Empty approval rule")

        m = _RULE_RE.match(s)
        if m:
            rules.append(PermissionRule(m.group(1), m.group(2)))
        elif _NAME_RE.match(s):
            rules.append(PermissionRule(s, None))
        else:
            raise ValueError(
                f"Invalid approval rule: {s!r}. "
                f"Expected 'tool_name' or 'tool_name(pattern)'."
            )
    return rules


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def match_pattern(pattern: str, resource: str) -> bool:
    """Match a resource string against a pattern.

    Strategy selected by pattern syntax (checked in order):

    1. domain:xxx   -> URL hostname match via fnmatch
    2. **/xxx       -> basename match at any depth INCLUDING root
    3. xxx/**       -> recursive directory prefix with os.sep boundary
    4. contains /   -> PurePosixPath.match() glob semantics
    5. fallback     -> fnmatch.fnmatch() wildcard
    """
    # 1. domain:xxx -> URL hostname
    if pattern.startswith("domain:"):
        target_host = pattern[7:]
        try:
            actual_host = urlparse(resource).hostname or ""
        except Exception:
            return False
        return fnmatch.fnmatch(actual_host.lower(), target_host.lower())

    # 2. **/xxx -> basename match at any depth (including root)
    if pattern.startswith("**/") and not pattern.endswith("/**"):
        suffix = pattern[3:]
        return fnmatch.fnmatch(os.path.basename(resource), suffix)

    # 3. xxx/** -> recursive directory prefix
    if pattern.endswith("/**"):
        prefix = os.path.normpath(pattern[:-3])
        normed = os.path.normpath(resource)
        return normed == prefix or normed.startswith(prefix + os.sep)

    # 4. contains / -> PurePosixPath glob
    if "/" in pattern:
        return PurePosixPath(resource).match(pattern)

    # 5. fallback -> fnmatch
    return fnmatch.fnmatch(resource, pattern)


# ---------------------------------------------------------------------------
# Rule matching helpers
# ---------------------------------------------------------------------------

def rule_matches(rule: PermissionRule, tool_name: str, resource: str | None) -> bool:
    """Check if a single rule matches the given tool call."""
    if rule.tool_name != tool_name:
        return False
    if rule.is_tool_level:
        return True
    if resource is None:
        return False
    return match_pattern(rule.pattern, resource)  # type: ignore[arg-type]


def any_rule_matches(
    rules: list[PermissionRule], tool_name: str, resource: str | None,
) -> bool:
    """Check if any rule in the list matches."""
    return any(rule_matches(r, tool_name, resource) for r in rules)


def has_tool_level_rule(rules: list[PermissionRule], tool_name: str) -> bool:
    """Check if the rule list contains a tool-level rule for this tool."""
    return any(r.tool_name == tool_name and r.is_tool_level for r in rules)


# ---------------------------------------------------------------------------
# Resource extraction
# ---------------------------------------------------------------------------

_KIND_MAP: dict[str, str] = {
    "file_path": "path",
    "path": "path",
    "url": "url",
    "command": "command",
}


def _canonicalize(raw: str, key: str) -> str:
    """Normalize a resource identifier for stable rule and grant matching.

    Path keys: expanduser + resolve. Returns the workspace-relative form when
    inside the workspace, the absolute resolved form otherwise. Non-path keys
    pass through unchanged.
    """
    if key not in ("file_path", "path"):
        return raw
    try:
        expanded = os.path.expanduser(raw)
        ws = Path.cwd().resolve()
        p = Path(expanded)
        resolved = (p if p.is_absolute() else ws / p).resolve()
        try:
            return str(resolved.relative_to(ws))
        except ValueError:
            return str(resolved)
    except OSError:
        return os.path.normpath(raw)


def extract_resource(
    tool_name: str,
    arguments: dict[str, object],
    resource_key: str | None,
    *,
    default: str | None = None,
) -> tuple[str | None, str | None]:
    """Extract and canonicalize the resource from tool call arguments.

    Returns (resource, kind). Both are None if the tool has no resource key
    or the argument is missing (and no default provided).
    """
    if resource_key is None:
        return None, None
    raw = arguments.get(resource_key)
    if raw is None:
        raw = default
    if raw is None:
        return None, None
    resource = _canonicalize(str(raw), resource_key)
    kind = _KIND_MAP.get(resource_key)
    return resource, kind
