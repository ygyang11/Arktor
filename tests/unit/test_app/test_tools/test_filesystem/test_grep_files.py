"""Tests for grep_files tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_app.tools.filesystem.grep_files import grep_files
from agent_harness.core.errors import ToolValidationError


class TestGrepFiles:
    @pytest.mark.asyncio
    async def test_basic_search(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def hello():\n    print('hello')\n")
        result = await grep_files.execute(pattern="hello", path=str(tmp_path))
        assert "2 matches in 1 files" in result

    @pytest.mark.asyncio
    async def test_include_basename(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("TODO\n")
        (tmp_path / "notes.md").write_text("TODO\n")
        result = await grep_files.execute(pattern="TODO", path=str(tmp_path), include="*.py")
        assert "code.py" in result
        assert "notes.md" not in result

    @pytest.mark.asyncio
    async def test_excluded_dirs_surfaced_when_empty(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "x.js").write_text("TODO here\n")
        result = await grep_files.execute(pattern="TODO", path=str(tmp_path))
        assert "No matches" in result
        assert "node_modules" in result
        assert "excluded" in result

    @pytest.mark.asyncio
    async def test_excluded_dirs_surfaced_with_matches(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("TODO real\n")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "x.js").write_text("TODO dep\n")
        result = await grep_files.execute(pattern="TODO", path=str(tmp_path))
        assert "1 matches in 1 files" in result
        assert "excluded: node_modules" in result

    @pytest.mark.asyncio
    async def test_oversized_skip_surfaced_when_no_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys
        grep_mod = sys.modules["agent_app.tools.filesystem.grep_files"]
        monkeypatch.setattr(grep_mod, "_MAX_GREP_FILE_SIZE", 10)
        (tmp_path / "big.log").write_text("x" * 100)
        result = await grep_files.execute(pattern="NEEDLE", path=str(tmp_path))
        assert "No matches" in result
        assert "skipped" in result
        assert "big.log" in result

    @pytest.mark.asyncio
    async def test_include_relative_path(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("TODO\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_app.py").write_text("TODO\n")
        result = await grep_files.execute(pattern="TODO", path=str(tmp_path), include="src/*.py")
        assert "app.py" in result
        assert "test_app.py" not in result

    @pytest.mark.asyncio
    async def test_include_respects_depth(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("FIND\n")
        sub = src / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("FIND\n")
        result = await grep_files.execute(pattern="FIND", path=str(tmp_path), include="src/*.py")
        assert "1 matches in 1 files" in result

    @pytest.mark.asyncio
    async def test_context_lines(self, tmp_path: Path) -> None:
        content = "\n".join(f"line {i}" for i in range(10))
        (tmp_path / "data.txt").write_text(content)
        result = await grep_files.execute(pattern="line 5", path=str(tmp_path), context=1)
        assert "1 matches" in result
        assert "line 4" in result
        assert "line 6" in result

    @pytest.mark.asyncio
    async def test_match_count_excludes_context(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("a\nb\nTARGET\nd\ne\n")
        result = await grep_files.execute(pattern="TARGET", path=str(tmp_path), context=2)
        assert "1 matches" in result

    @pytest.mark.asyncio
    async def test_case_insensitive(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("TODO\ntodo\n")
        result = await grep_files.execute(pattern="todo", path=str(tmp_path), case_insensitive=True)
        assert "2 matches" in result

    @pytest.mark.asyncio
    async def test_pagination(self, tmp_path: Path) -> None:
        lines = [f"TODO item {i}" for i in range(100)]
        (tmp_path / "todos.py").write_text("\n".join(lines) + "\n")
        result = await grep_files.execute(
            pattern="TODO", path=str(tmp_path), max_results=10, offset=0
        )
        assert "offset=10 for more" in result

    @pytest.mark.asyncio
    async def test_pagination_offset(self, tmp_path: Path) -> None:
        lines = [f"TODO item {i}" for i in range(20)]
        (tmp_path / "todos.py").write_text("\n".join(lines) + "\n")
        result1 = await grep_files.execute(
            pattern="TODO", path=str(tmp_path), max_results=5, offset=0
        )
        result2 = await grep_files.execute(
            pattern="TODO", path=str(tmp_path), max_results=5, offset=5
        )
        assert "offset 0" not in result1
        assert "offset 5" in result2

    @pytest.mark.asyncio
    async def test_invalid_regex(self) -> None:
        with pytest.raises(ToolValidationError, match="invalid regex"):
            await grep_files.execute(pattern="[invalid")

    @pytest.mark.asyncio
    async def test_skip_git(self, tmp_path: Path) -> None:
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("match")
        result = await grep_files.execute(pattern="match", path=str(tmp_path))
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_external_symlink_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "real.py").write_text("target\n")
        link = tmp_path / "escape.py"
        link.symlink_to("/etc/passwd")
        result = await grep_files.execute(pattern="target", path=str(tmp_path))
        assert "1 matches in 1 files" in result

    @pytest.mark.asyncio
    async def test_symlink_dir_not_recursed(self, tmp_path: Path) -> None:
        real = tmp_path / "real_dir"
        real.mkdir()
        (real / "file.py").write_text("FINDME\n")
        link = tmp_path / "link_dir"
        link.symlink_to(real)
        result = await grep_files.execute(pattern="FINDME", path=str(tmp_path))
        assert "1 matches in 1 files" in result

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("hello\n")
        result = await grep_files.execute(pattern="zzzzz", path=str(tmp_path))
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_empty_pattern_rejected(self) -> None:
        with pytest.raises(ToolValidationError):
            await grep_files.execute(pattern="")
