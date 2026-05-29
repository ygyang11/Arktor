"""Tests for write_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_app.observability.file_freshness import _key
from agent_app.tools.filesystem.write_file import write_file
from agent_harness.agent.base import BaseAgent
from agent_harness.core.errors import ToolValidationError


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_create_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "new.txt"
        result = await write_file.execute(file_path=str(target), content="hello\n")
        assert "Created" in result
        assert target.read_text() == "hello\n"

    @pytest.mark.asyncio
    async def test_empty_path_rejected(self) -> None:
        with pytest.raises(ToolValidationError):
            await write_file.execute(file_path="  ", content="x")

    @pytest.mark.asyncio
    async def test_create_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.txt"
        result = await write_file.execute(file_path=str(target), content="deep\n")
        assert "Created" in result
        assert target.read_text() == "deep\n"

    @pytest.mark.asyncio
    async def test_existing_file_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "exist.txt"
        target.write_text("original\n")
        result = await write_file.execute(file_path=str(target), content="overwrite\n")
        assert "already exists" in result
        assert "edit_file" in result
        assert target.read_text() == "original\n"

    @pytest.mark.asyncio
    async def test_sensitive_path_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / ".env"
        result = await write_file.execute(file_path=str(target), content="SECRET=bad")
        assert "Refusing" in result
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_sensitive_path_case_bypass(self, tmp_path: Path) -> None:
        target = tmp_path / ".ENV"
        result = await write_file.execute(file_path=str(target), content="SECRET=bad")
        assert "Refusing" in result

    @pytest.mark.asyncio
    async def test_directory_rejected(self, tmp_path: Path) -> None:
        result = await write_file.execute(file_path=str(tmp_path), content="bad")
        assert "directory" in result.lower()

    @pytest.mark.asyncio
    async def test_line_count(self, tmp_path: Path) -> None:
        target = tmp_path / "multi.txt"
        result = await write_file.execute(file_path=str(target), content="a\nb\nc\n")
        assert "3 lines" in result

    @pytest.mark.asyncio
    async def test_write_records_signature_after_create(
        self, tmp_path: Path, fs_agent: BaseAgent,
    ) -> None:
        target = tmp_path / "fresh.txt"
        await write_file.execute(file_path=str(target), content="created\n")
        recorded = fs_agent.context.variables.get(_key(target))
        assert isinstance(recorded, dict)
        assert recorded["size"] == target.stat().st_size
