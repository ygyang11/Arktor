"""Tests for MemoryTool."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_app.tools.memory.memory_tool import (
    MemoryTool,
    _SLUG_RE,
    _INDEX_SOFT_LIMIT,
    _INDEX_HARD_LIMIT,
    _parse_frontmatter,
    _render_frontmatter,
    _parse_index_sections,
    _render_index_sections,
)
from agent_harness.core.errors import ToolValidationError


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    mod = sys.modules["agent_app.tools.memory.memory_tool"]
    monkeypatch.setattr(mod, "_GLOBAL_MEMORY_DIR", tmp_path / "global_mem")
    monkeypatch.setattr(mod, "_PROJECT_MEMORY_DIRNAME", str(tmp_path / "project_mem"))


def _tool() -> MemoryTool:
    return MemoryTool()


# ── Schema ──


class TestSchema:
    def test_schema_name(self) -> None:
        schema = _tool().get_schema()
        assert schema.name == "memory_tool"

    def test_required_params(self) -> None:
        schema = _tool().get_schema()
        assert set(schema.parameters["required"]) == {"action", "scope", "type", "name"}

    def test_action_enum(self) -> None:
        schema = _tool().get_schema()
        enum = schema.parameters["properties"]["action"]["enum"]
        assert "save" in enum
        assert "read" in enum
        assert "delete" in enum

    def test_scope_enum(self) -> None:
        schema = _tool().get_schema()
        enum = schema.parameters["properties"]["scope"]["enum"]
        assert "global" in enum
        assert "project" in enum

    def test_type_enum(self) -> None:
        schema = _tool().get_schema()
        enum = schema.parameters["properties"]["type"]["enum"]
        assert "user" in enum
        assert "feedback" in enum
        assert "project" in enum
        assert "reference" in enum
        assert "knowledge" in enum


# ── Validation ──


class TestValidation:
    async def test_invalid_action(self) -> None:
        with pytest.raises(ToolValidationError, match="Invalid action"):
            await _tool().execute(action="update", scope="project", type="user", name="x")

    async def test_invalid_scope(self) -> None:
        with pytest.raises(ToolValidationError, match="Invalid scope"):
            await _tool().execute(action="read", scope="team", type="user", name="x")

    async def test_invalid_type(self) -> None:
        with pytest.raises(ToolValidationError, match="Invalid type"):
            await _tool().execute(action="read", scope="project", type="misc", name="x")

    async def test_empty_name(self) -> None:
        with pytest.raises(ToolValidationError, match="name is required"):
            await _tool().execute(action="read", scope="project", type="user", name="")

    async def test_invalid_name_format(self) -> None:
        with pytest.raises(ToolValidationError, match="Invalid name"):
            await _tool().execute(action="read", scope="project", type="user", name="has-dash")

    async def test_reserved_name(self) -> None:
        with pytest.raises(ToolValidationError, match="reserved"):
            await _tool().execute(action="read", scope="project", type="user", name="memory")

    async def test_save_missing_description(self) -> None:
        with pytest.raises(ToolValidationError, match="description is required"):
            await _tool().execute(
                action="save", scope="project", type="user", name="x", content="body",
            )

    async def test_save_missing_content(self) -> None:
        with pytest.raises(ToolValidationError, match="content is required"):
            await _tool().execute(
                action="save", scope="project", type="user", name="x", description="desc",
            )


# ── Name Validation ──


class TestNameValidation:
    def test_valid_lowercase(self) -> None:
        assert _SLUG_RE.match("abc")

    def test_valid_with_underscore(self) -> None:
        assert _SLUG_RE.match("a_b_c")

    def test_valid_with_numbers(self) -> None:
        assert _SLUG_RE.match("test123")

    def test_invalid_uppercase(self) -> None:
        assert not _SLUG_RE.match("Auth_Rewrite")

    def test_invalid_dash(self) -> None:
        assert not _SLUG_RE.match("a-b")

    def test_invalid_space(self) -> None:
        assert not _SLUG_RE.match("a b")

    def test_invalid_leading_underscore(self) -> None:
        assert not _SLUG_RE.match("_abc")

    def test_invalid_chinese(self) -> None:
        assert not _SLUG_RE.match("中文")


# ── Save ──


class TestSave:
    async def test_save_creates_file(self, tmp_path: Path) -> None:
        tool = _tool()
        result = await tool.execute(
            action="save", scope="project", type="feedback",
            name="no_mock_db", description="Use real DB", content="Details here",
        )
        assert "saved" in result
        assert "no_mock_db" in result
        file_path = tmp_path / "project_mem" / "feedback" / "no_mock_db.md"
        assert file_path.exists()

    async def test_save_writes_frontmatter(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="knowledge",
            name="quantum_ec", description="Surface codes best", content="Body text",
        )
        file_path = tmp_path / "project_mem" / "knowledge" / "quantum_ec.md"
        text = file_path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        assert fm["type"] == "knowledge"
        assert fm["description"] == "Surface codes best"
        assert "created_at" in fm
        assert "updated_at" in fm
        assert body == "Body text"

    async def test_save_creates_index(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="user",
            name="role", description="Senior engineer", content="Details",
        )
        index = (tmp_path / "project_mem" / "MEMORY.md").read_text(encoding="utf-8")
        assert "role" in index
        assert "Senior engineer" in index
        assert "## User" in index

    async def test_save_global_scope(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="global", type="user",
            name="identity", description="Data scientist", content="Body",
        )
        file_path = tmp_path / "global_mem" / "user" / "identity.md"
        assert file_path.exists()


# ── Save Update ──


class TestSaveUpdate:
    async def test_update_preserves_created_at(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="feedback",
            name="test_update", description="v1", content="First",
        )
        file_path = tmp_path / "project_mem" / "feedback" / "test_update.md"
        fm1, _ = _parse_frontmatter(file_path.read_text(encoding="utf-8"))
        created1 = fm1["created_at"]

        result = await tool.execute(
            action="save", scope="project", type="feedback",
            name="test_update", description="v2", content="Second",
        )
        assert "updated" in result
        fm2, _ = _parse_frontmatter(file_path.read_text(encoding="utf-8"))
        assert fm2["created_at"] == created1
        assert fm2["description"] == "v2"

    async def test_update_replaces_index_entry(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="feedback",
            name="entry", description="old desc", content="Body",
        )
        await tool.execute(
            action="save", scope="project", type="feedback",
            name="entry", description="new desc", content="Body2",
        )
        index = (tmp_path / "project_mem" / "MEMORY.md").read_text(encoding="utf-8")
        assert index.count("entry") == 2  # name + link
        assert "new desc" in index
        assert "old desc" not in index


# ── Read ──


class TestRead:
    async def test_read_existing(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="knowledge",
            name="test_read", description="desc", content="Read me",
        )
        result = await tool.execute(
            action="read", scope="project", type="knowledge", name="test_read",
        )
        assert "Read me" in result
        assert "knowledge" in result

    async def test_read_not_found(self) -> None:
        result = await _tool().execute(
            action="read", scope="project", type="user", name="nonexistent",
        )
        assert result.startswith("Error:")


# ── Delete ──


class TestDelete:
    async def test_delete_removes_file_and_index(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="reference",
            name="linear", description="Bug tracker", content="Body",
        )
        file_path = tmp_path / "project_mem" / "reference" / "linear.md"
        assert file_path.exists()

        result = await tool.execute(
            action="delete", scope="project", type="reference", name="linear",
        )
        assert "deleted" in result.lower()
        assert not file_path.exists()
        index = (tmp_path / "project_mem" / "MEMORY.md").read_text(encoding="utf-8")
        assert "linear" not in index

    async def test_delete_not_found(self) -> None:
        result = await _tool().execute(
            action="delete", scope="project", type="user", name="ghost",
        )
        assert result.startswith("Error:")


# ── Index ──


class TestIndex:
    async def test_index_grouped_by_type(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="user",
            name="a", description="user a", content="body",
        )
        await tool.execute(
            action="save", scope="project", type="knowledge",
            name="b", description="knowledge b", content="body",
        )
        index = (tmp_path / "project_mem" / "MEMORY.md").read_text(encoding="utf-8")
        user_pos = index.index("## User")
        knowledge_pos = index.index("## Knowledge")
        assert user_pos < knowledge_pos

    async def test_soft_limit_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = sys.modules["agent_app.tools.memory.memory_tool"]

        monkeypatch.setattr(mod, "_INDEX_SOFT_LIMIT", 2)
        monkeypatch.setattr(mod, "_INDEX_HARD_LIMIT", 5)
        tool = _tool()
        for i in range(3):
            result = await tool.execute(
                action="save", scope="project", type="knowledge",
                name=f"item{i}", description=f"desc {i}", content="body",
            )
        assert "Warning" in result

    async def test_hard_limit_rejects_new(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = sys.modules["agent_app.tools.memory.memory_tool"]

        monkeypatch.setattr(mod, "_INDEX_SOFT_LIMIT", 2)
        monkeypatch.setattr(mod, "_INDEX_HARD_LIMIT", 3)
        tool = _tool()
        for i in range(3):
            await tool.execute(
                action="save", scope="project", type="knowledge",
                name=f"item{i}", description=f"desc {i}", content="body",
            )
        # 4th should still save file but index won't grow
        result = await tool.execute(
            action="save", scope="project", type="knowledge",
            name="overflow", description="overflow", content="body",
        )
        index = (tmp_path / "project_mem" / "MEMORY.md").read_text(encoding="utf-8")
        assert "overflow" not in index

    async def test_hard_limit_allows_replace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = sys.modules["agent_app.tools.memory.memory_tool"]

        monkeypatch.setattr(mod, "_INDEX_SOFT_LIMIT", 2)
        monkeypatch.setattr(mod, "_INDEX_HARD_LIMIT", 3)
        tool = _tool()
        for i in range(3):
            await tool.execute(
                action="save", scope="project", type="knowledge",
                name=f"item{i}", description=f"desc {i}", content="body",
            )
        # Replacing existing entry should work
        result = await tool.execute(
            action="save", scope="project", type="knowledge",
            name="item0", description="updated desc", content="new body",
        )
        assert "updated" in result
        index = (tmp_path / "project_mem" / "MEMORY.md").read_text(encoding="utf-8")
        assert "updated desc" in index


# ── build_context_message ──


class TestBuildContextMessage:
    async def test_empty_indexes_emit_placeholder_message(self) -> None:
        tool = _tool()
        msg = tool.build_context_message()
        assert msg is not None
        assert "# Memory" in msg.content
        assert "## Global Memory\n(no entries yet)" in msg.content
        assert "## Project Memory\n(no entries yet)" in msg.content
        assert "do not call `read` with names not listed" in msg.content

    async def test_returns_message_after_save(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="user",
            name="test_ctx", description="test desc", content="body",
        )
        msg = tool.build_context_message()
        assert msg is not None
        assert "# Memory" in msg.content
        assert "test_ctx" in msg.content
        # global still empty, project populated
        assert "## Global Memory\n(no entries yet)" in msg.content

    async def test_reflects_delete(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="user",
            name="ephemeral", description="will be deleted", content="body",
        )
        await tool.execute(
            action="delete", scope="project", type="user", name="ephemeral",
        )
        msg = tool.build_context_message()
        assert msg is not None
        assert "ephemeral" not in msg.content
        assert "## Project Memory\n(no entries yet)" in msg.content


# ── Scope Isolation ──


class TestScope:
    async def test_scopes_are_independent(self, tmp_path: Path) -> None:
        tool = _tool()
        await tool.execute(
            action="save", scope="global", type="user",
            name="shared", description="global version", content="global body",
        )
        await tool.execute(
            action="save", scope="project", type="user",
            name="shared", description="project version", content="project body",
        )
        global_result = await tool.execute(
            action="read", scope="global", type="user", name="shared",
        )
        project_result = await tool.execute(
            action="read", scope="project", type="user", name="shared",
        )
        assert "global body" in global_result
        assert "project body" in project_result


# ── Timestamp ──


class TestTimestamp:
    async def test_datetime_object_normalized(self, tmp_path: Path) -> None:
        from datetime import datetime

        tool = _tool()
        await tool.execute(
            action="save", scope="project", type="feedback",
            name="ts_test", description="timestamp test", content="body",
        )
        file_path = tmp_path / "project_mem" / "feedback" / "ts_test.md"
        fm, _ = _parse_frontmatter(file_path.read_text(encoding="utf-8"))
        assert isinstance(fm["created_at"], str)
        assert isinstance(fm["updated_at"], str)


# ── Helpers ──


class TestHelpers:
    def test_parse_frontmatter_valid(self) -> None:
        fm, body = _parse_frontmatter("---\ntype: user\n---\n\nBody text")
        assert fm["type"] == "user"
        assert body == "Body text"

    def test_parse_frontmatter_missing(self) -> None:
        fm, body = _parse_frontmatter("No frontmatter here")
        assert fm == {}
        assert body == "No frontmatter here"

    def test_parse_frontmatter_invalid_yaml(self) -> None:
        fm, body = _parse_frontmatter("---\n: invalid: yaml:\n---\n\nBody")
        assert fm == {} or isinstance(fm, dict)

    def test_render_frontmatter_roundtrip(self) -> None:
        original = {"type": "feedback", "description": "test", "created_at": "2026-01-01"}
        rendered = _render_frontmatter(original)
        assert rendered.startswith("---\n")
        assert rendered.endswith("---\n\n")
        fm, _ = _parse_frontmatter(rendered + "body")
        assert fm["type"] == "feedback"
        assert fm["description"] == "test"

    def test_parse_index_sections(self) -> None:
        content = "## User\n- [a](user/a.md) — desc a\n\n## Knowledge\n- [b](knowledge/b.md) — desc b\n"
        sections = _parse_index_sections(content)
        assert len(sections["user"]) == 1
        assert len(sections["knowledge"]) == 1

    def test_render_index_sections_order(self) -> None:
        sections = {
            "knowledge": ["- [b](knowledge/b.md) — b"],
            "user": ["- [a](user/a.md) — a"],
        }
        rendered = _render_index_sections(sections)
        assert rendered.index("## User") < rendered.index("## Knowledge")
