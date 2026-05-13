"""Memory tool — persistent knowledge management with Markdown + YAML frontmatter."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agent_harness.core.errors import ToolValidationError
from agent_harness.core.message import Message
from agent_harness.tool.base import BaseTool, ToolSchema

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "MEMORY.md"
_INDEX_SOFT_LIMIT = 200
_INDEX_HARD_LIMIT = 250
_VALID_TYPES = frozenset({"user", "feedback", "project", "reference", "knowledge"})
_VALID_ACTIONS = frozenset({"save", "read", "delete"})
_VALID_SCOPES = frozenset({"project", "global"})
_RESERVED_SLUGS = frozenset({"memory"})
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

_GLOBAL_MEMORY_DIR = Path.home() / ".agent-harness" / "memory"
_PROJECT_MEMORY_DIRNAME = ".agent-harness/memory"

MEMORY_TOOL_DESCRIPTION = """\
Manage persistent memories that survive across sessions. Memories are \
stored as Markdown files organized by type and scope. Your system context \
includes a memory index with one-line descriptions of all saved memories, \
updated in real time.

## Actions

- **save** (all 6 params): Create a new memory or update an existing one. \
The tool writes the file at `{scope}/{type}/{name}.md`, assembles YAML \
frontmatter with timestamps, and updates the memory index automatically. \
To update, read the existing memory first, merge new information in content, \
update the description to reflect it for later judgment if necessary, and save.
- **read** (action, scope, type, name): Retrieve the full content of a \
stored memory including its frontmatter. Use when the one-line index \
description is not enough and you need the complete information.
- **delete** (action, scope, type, name): Remove a memory file and its \
index entry.

## Usage Notes
1. For most types, save one specific fact per memory. Each `description` \
should be specific enough to judge relevance at a glance — \
good: "Integration tests must use a real database, not mocks"; \
bad: "Testing preferences".
2. Only for the knowledge type, use a focused sub-topic as `name` \
(e.g. `5g_ntn_satellite`, not `communication`) — build up and refine \
one memory over time as you research and learn more.
3. When saving is warranted, do it as your first action — before \
responding to the user.
4. Check the memory index before creating a new memory. If a related \
memory already exists, read and update it rather than creating a duplicate.

## Examples

<example>
User: "I've been writing Go for ten years but this is my first time \
touching the React side of this repo"
Assistant: *Saves a user memory about the user's skill profile*
memory_tool(action="save", scope="global", type="user", \
name="user_skill_profile", \
description="Deep Go expertise, new to React", \
content="User has 10 years of Go experience but is a beginner \
with React. Frame frontend explanations in terms of backend \
analogues they already know.")
<commentary>
The user revealed lasting skill information — deep Go expertise \
and a React knowledge gap. This is worth saving because it's a \
persistent trait, not a transient task. Scope is global because \
their skills don't change per project. The description is specific \
enough to surface when the user asks about React. The \
content goes beyond the description — it includes actionable \
guidance on how to adapt explanations (use backend analogues), \
which helps future sessions tailor responses without re-asking.
</commentary>
</example>

<example>
User: "don't mock the database in these tests — we got burned last \
quarter when mocked tests passed but the prod migration failed"
Assistant: *Saves a feedback memory capturing the correction and reason*
memory_tool(action="save", scope="project", type="feedback", \
name="no_mock_db", \
description="Integration tests must use a real database, not mocks", \
content="**Why:** Mock/prod divergence masked a broken migration \
last quarter — mocked tests passed but production failed.\\n\\n\
**How to apply:** For all tests under tests/integration/, always \
connect to a real test database. Never mock the ORM or query layer.")
<commentary>
The user gave explicit feedback with a clear reason — this is a \
correction worth persisting, not just a one-time discussion. Scope \
is project because the testing convention is specific to this codebase. \
Type is feedback because it's a direct reaction to how tests were \
written. The description captures the rule itself so the index entry \
alone is useful. The content adds the Why (past incident) and How \
to apply (which tests, what to use instead), so future sessions can \
apply the rule correctly and judge edge cases without re-asking.
</commentary>
</example>

<example>
After several turns of researching 5G NTN (Non-Terrestrial Networks) \
satellite communication — searching the web, reading papers, and \
comparing approaches — the agent proactively saves a knowledge memory.\
memory_tool(action="save", scope="global", type="knowledge", \
name="5g_ntn_satellite", \
description="5G NTN satellite communication — key challenges, \
architecture, and standardization status", \
content="5G NTN extends cellular coverage via LEO/GEO satellites, \
standardized in 3GPP Rel-17/18...\n\n\
## Key Challenges\n\
- Long propagation delay requires HARQ and timing advance adaptations\n\
- ...\n\n\
## Architecture\n\
Transparent (bent-pipe) vs regenerative payloads. ...\n\n\
## References\n\
- [paper] <title>, <doi/url>\n\
- [web] <title>, <url>\n\
- ...")
<commentary>
The agent saved proactively — multi-turn research \
across multiple sources has high re-research cost, making it worth \
persisting automatically. Scope is global because 5G NTN is domain \
expertise not tied to any specific codebase — use project scope only \
when findings are specific to the current project. The name is a \
focused sub-topic (`5g_ntn_satellite`, not `communication` or `5g`). \
The content is structured with sections and references(use [paper] \
and [web] tags to distinguish source types) — in practice it would be \
much more detailed and comprehensive than shown here. Beyond \
research findings, the content should also capture experiments \
you've run, configurations you've tried, lessons learned from \
failures, and practical tips that would save time in future sessions. \
As the agent learns more (e.g. Rel-19 updates, new experiments), \
it would read this memory, merge the new findings, and save the \
updated version.
</commentary>
</example>

<example>
User: "I'm debugging the auth module, there's a token expiration bug"
Assistant: *Does NOT save a memory*
<commentary>
This is a transient task detail about the current debugging session. \
It does not reveal lasting preferences, project decisions, or domain \
expertise. Memory is for what persists across sessions — not for what \
you're doing right now.
</commentary>
</example>

<example>
User: "I'm on macOS and I have Python 3.11 installed"
Assistant: *Does NOT save a memory*
<commentary>
This looks like user information, but ask: "in a new conversation, \
would this change how I work?" The OS and Python version are \
environment details the agent can detect at runtime — they don't \
need to be memorized. Only save information that the agent cannot \
discover on its own and that will remain relevant.
</commentary>
</example>"""


class MemoryTool(BaseTool):
    """Persistent knowledge tool with type-based directory organization."""

    def __init__(self) -> None:
        super().__init__(name="memory_tool", description="")
        self.context_order = 100

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=MEMORY_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(_VALID_ACTIONS),
                        "description": (
                            "The operation to perform: "
                            "'save' to create or update a memory "
                            "(requires description and content), "
                            "'read' to retrieve full content, "
                            "'delete' to remove a memory and its index entry."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": sorted(_VALID_SCOPES),
                        "description": (
                            "'global' for user-level memories across all projects "
                            "'project' for current-project-specific memories "
                        ),
                    },
                    "type": {
                        "type": "string",
                        "enum": sorted(_VALID_TYPES),
                        "description": (
                            "Memory category: "
                            "'user' (role, skill level, preferences), "
                            "'feedback' (corrections and preferences that carry forward), "
                            "'project' (context and decisions not visible in the codebase), "
                            "'reference' (URLs, links and pointers to external resources), "
                            "'knowledge' (domain expertise)."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Unique identifier. Lowercase letters, numbers, and underscores "
                            "only (e.g. 'no_mock_db'). For knowledge type, use a "
                            "focused sub-topic (e.g. '5g_ntn_satellite'). "
                            "Used as the filename."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One-line summary for the memory index (required for save). "
                            "Be specific — this determines whether the memory is "
                            "surfaced as relevant in future sessions."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The detailed memory body (required for save). "
                            "Provide context beyond the one-line description. "
                            "For knowledge type, this can be comprehensive — "
                            "structured sections, references, lessons learned."
                        ),
                    },
                },
                "required": ["action", "scope", "type", "name"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "")).strip()
        scope = str(kwargs.get("scope", "")).strip()
        type_ = str(kwargs.get("type", "")).strip()
        name = str(kwargs.get("name", "")).strip()
        description = str(kwargs.get("description", "")).strip()
        content = str(kwargs.get("content", "")).strip()

        if action not in _VALID_ACTIONS:
            raise ToolValidationError(f"Invalid action: {action!r}")
        if scope not in _VALID_SCOPES:
            raise ToolValidationError(f"Invalid scope: {scope!r}")
        if type_ not in _VALID_TYPES:
            raise ToolValidationError(f"Invalid type: {type_!r}")
        if not name:
            raise ToolValidationError("name is required")
        if not _SLUG_RE.match(name):
            raise ToolValidationError(
                f"Invalid name: {name!r}. Use lowercase letters, numbers, "
                f"and underscores only (e.g. 'no_mock_db')."
            )
        if name in _RESERVED_SLUGS:
            raise ToolValidationError(f"Name {name!r} is reserved.")

        if action == "save":
            if not description:
                raise ToolValidationError("description is required for save")
            if not content:
                raise ToolValidationError("content is required for save")
            return self._save(name, scope, type_, description, content)
        elif action == "read":
            return self._read(name, scope, type_)
        else:
            return self._delete(name, scope, type_)

    # ── Path Resolution ──

    def _resolve_memory_dir(self, scope: str) -> Path:
        if scope == "global":
            return _GLOBAL_MEMORY_DIR
        return Path.cwd() / _PROJECT_MEMORY_DIRNAME

    def _resolve_file(self, name: str, scope: str, type_: str) -> Path:
        return self._resolve_memory_dir(scope) / type_ / f"{name}.md"

    def _resolve_index(self, scope: str) -> Path:
        return self._resolve_memory_dir(scope) / _INDEX_FILENAME

    # ── Actions ──

    def _save(self, name: str, scope: str, type_: str, description: str, content: str) -> str:
        file_path = self._resolve_file(name, scope, type_)
        is_update = file_path.exists()

        now = datetime.now().isoformat(timespec="seconds")
        if is_update:
            old_fm, _ = _parse_frontmatter(file_path.read_text(encoding="utf-8"))
            created_at = old_fm.get("created_at", now)
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat(timespec="seconds")
        else:
            created_at = now

        frontmatter = {
            "type": type_,
            "description": description,
            "created_at": created_at,
            "updated_at": now,
        }
        file_content = _render_frontmatter(frontmatter) + content
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_content, encoding="utf-8")

        count = self._update_index(scope, name, description, type_)

        if count < 0:
            file_path.unlink(missing_ok=True)
            return (
                f"Error: memory index has reached the hard limit "
                f"({_INDEX_HARD_LIMIT} entries). Delete unused memories to free space."
            )

        verb = "updated" if is_update else "saved"
        result = f"Memory {verb}: [{name}]({type_}/{name}.md) — {description} (scope={scope})"

        if count > _INDEX_SOFT_LIMIT:
            result += (
                f"\nWarning: memory index has {count} entries "
                f"(soft limit: {_INDEX_SOFT_LIMIT}, hard limit: {_INDEX_HARD_LIMIT}). "
                f"At {_INDEX_HARD_LIMIT} entries, new saves will be rejected. "
                f"Delete unused memories to free space."
            )
        return result

    def _read(self, name: str, scope: str, type_: str) -> str:
        file_path = self._resolve_file(name, scope, type_)
        if not file_path.exists():
            return f"Error: memory '{name}' not found in {scope}/{type_}."
        return file_path.read_text(encoding="utf-8")

    def _delete(self, name: str, scope: str, type_: str) -> str:
        file_path = self._resolve_file(name, scope, type_)
        if not file_path.exists():
            return f"Error: memory '{name}' not found in {scope}/{type_}."
        file_path.unlink()
        self._remove_from_index(scope, name, type_)
        return f"Memory deleted: {name} (scope={scope}, type={type_})"

    # ── Index Management ──

    def _update_index(self, scope: str, name: str, description: str, type_: str) -> int:
        """Add or replace an entry in MEMORY.md. Returns total entry count."""
        index_path = self._resolve_index(scope)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        raw = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        sections = _parse_index_sections(raw)

        new_entry = f"- [{name}]({type_}/{name}.md) — {description}"
        pattern = f"]({type_}/{name}.md)"

        type_entries = sections.get(type_, [])
        replaced = False
        for i, line in enumerate(type_entries):
            if pattern in line:
                type_entries[i] = new_entry
                replaced = True
                break

        if not replaced:
            total = sum(len(e) for e in sections.values())
            if total >= _INDEX_HARD_LIMIT:
                return -1
            type_entries.append(new_entry)

        sections[type_] = type_entries
        index_path.write_text(_render_index_sections(sections), encoding="utf-8")
        return sum(len(e) for e in sections.values())

    def _remove_from_index(self, scope: str, name: str, type_: str) -> None:
        index_path = self._resolve_index(scope)
        if not index_path.exists():
            return
        raw = index_path.read_text(encoding="utf-8")
        sections = _parse_index_sections(raw)
        pattern = f"]({type_}/{name}.md)"
        if type_ in sections:
            sections[type_] = [line for line in sections[type_] if pattern not in line]
        index_path.write_text(_render_index_sections(sections), encoding="utf-8")

    # ── Context Injection ──

    def build_context_message(self) -> Message | None:
        global_index = self.load_index("global")
        project_index = self.load_index("project")
        parts = ["# Memory"]
        parts.append(
            "## Global Memory\n" + (global_index or "(no entries yet)")
        )
        parts.append(
            "## Project Memory\n" + (project_index or "(no entries yet)")
        )
        parts.append(
            "This index lists every saved memory — do not call `read` "
            "with names not listed. When a one-line description is not "
            "enough, read the full memory with "
            'memory_tool(action="read", scope="...", type="...", name="..."). '
            "Memories may be stale — verify before relying on them."
        )
        return Message.system("\n\n".join(parts))

    def load_index(self, scope: str) -> str:
        index_path = self._resolve_index(scope)
        if not index_path.exists():
            return ""
        return index_path.read_text(encoding="utf-8").strip()


# ── Helpers ──


def _parse_index_sections(content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_type: str | None = None
    for line in content.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("## "):
            header = line_stripped[3:].strip().lower()
            if header in _VALID_TYPES:
                current_type = header
                if current_type not in sections:
                    sections[current_type] = []
        elif current_type is not None and line_stripped.startswith("- ["):
            sections[current_type].append(line_stripped)
    return sections


def _render_index_sections(sections: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for type_ in ("user", "feedback", "project", "reference", "knowledge"):
        entries = sections.get(type_, [])
        if entries:
            header = type_.capitalize()
            parts.append(f"## {header}\n" + "\n".join(entries))
    return "\n\n".join(parts) + "\n" if parts else ""


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    text = text.strip()
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    yaml_block = text[3:end].strip()
    body = text[end + 3:].strip()
    try:
        frontmatter = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(frontmatter, dict):
        return {}, text
    return frontmatter, body


def _render_frontmatter(frontmatter: dict[str, Any]) -> str:
    yaml_block = yaml.dump(
        frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False,
    ).strip()
    return f"---\n{yaml_block}\n---\n\n"


memory_tool = MemoryTool()
