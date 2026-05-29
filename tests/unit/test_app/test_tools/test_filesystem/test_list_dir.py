"""Tests for list_dir tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_app.tools.filesystem.list_dir import list_dir


class TestListDir:
    @pytest.mark.asyncio
    async def test_basic_listing(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "README.md").write_text("# hi")
        result = await list_dir.execute(path=str(tmp_path))
        assert "src/" in result
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_dirs_before_files(self, tmp_path: Path) -> None:
        (tmp_path / "z_file.txt").touch()
        (tmp_path / "a_dir").mkdir()
        result = await list_dir.execute(path=str(tmp_path))
        assert result.index("a_dir/") < result.index("z_file.txt")

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = await list_dir.execute(path=str(empty))
        assert "empty directory" in result

    @pytest.mark.asyncio
    async def test_external_symlink_redacted(self, tmp_path: Path) -> None:
        link = tmp_path / "extlink"
        link.symlink_to("/etc")
        result = await list_dir.execute(path=str(tmp_path))
        assert "extlink -> (external)" in result
        assert "/etc" not in result

    @pytest.mark.asyncio
    async def test_internal_symlink_shows_target(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        result = await list_dir.execute(path=str(tmp_path))
        assert "link.txt -> real.txt" in result

    @pytest.mark.asyncio
    async def test_broken_symlink_marked(self, tmp_path: Path) -> None:
        link = tmp_path / "broken"
        link.symlink_to(tmp_path / "nonexistent_target")
        result = await list_dir.execute(path=str(tmp_path))
        assert "broken -> (unresolved symlink)" in result

    @pytest.mark.asyncio
    async def test_file_path_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.touch()
        result = await list_dir.execute(path=str(f))
        assert "not a directory" in result

    @pytest.mark.asyncio
    async def test_file_sizes_shown(self, tmp_path: Path) -> None:
        (tmp_path / "small.txt").write_text("hi")
        result = await list_dir.execute(path=str(tmp_path))
        assert "B)" in result

    @pytest.mark.asyncio
    async def test_entry_count(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"file_{i}.txt").touch()
        result = await list_dir.execute(path=str(tmp_path))
        assert "5 entries" in result
