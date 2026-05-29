"""Tests for glob_files tool."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_app.tools.filesystem.glob_files import glob_files
from agent_harness.core.errors import ToolValidationError


class TestGlobFiles:
    @pytest.mark.asyncio
    async def test_parent_dir_pattern_is_contained(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        (tmp_path / "outside.py").write_text("x")
        result = await glob_files.execute(pattern="../*.py", path=str(base))
        assert "outside.py" not in result

    @pytest.mark.asyncio
    async def test_invalid_glob_pattern_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="invalid glob pattern"):
            await glob_files.execute(pattern="a/**b/c", path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_absolute_pattern_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="invalid glob pattern"):
            await glob_files.execute(pattern="/abs/*", path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_recursive_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass")
        (tmp_path / "README.md").write_text("# hi")
        result = await glob_files.execute(pattern="**/*.py", path=str(tmp_path))
        assert "1 files" in result
        assert "main.py" in result
        assert "README.md" not in result

    @pytest.mark.asyncio
    async def test_sorted_by_mtime(self, tmp_path: Path) -> None:
        old = tmp_path / "old.py"
        old.write_text("old")
        time.sleep(0.05)
        new = tmp_path / "new.py"
        new.write_text("new")
        result = await glob_files.execute(pattern="*.py", path=str(tmp_path))
        assert result.index("new.py") < result.index("old.py")

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path: Path) -> None:
        result = await glob_files.execute(pattern="*.xyz", path=str(tmp_path))
        assert "No files" in result

    @pytest.mark.asyncio
    async def test_external_symlink_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("pass")
        link = tmp_path / "escape.py"
        link.symlink_to("/etc/passwd")
        result = await glob_files.execute(pattern="*.py", path=str(tmp_path))
        assert "1 files" in result
        assert "escape.py" not in result

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tmp_path: Path) -> None:
        result = await glob_files.execute(pattern="*.py", path=str(tmp_path / "nope"))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_direct_children_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "root.py").write_text("pass")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("pass")
        result = await glob_files.execute(pattern="*.py", path=str(tmp_path))
        assert "root.py" in result
        # *.py in Path.glob only matches direct children
        assert "nested.py" not in result
