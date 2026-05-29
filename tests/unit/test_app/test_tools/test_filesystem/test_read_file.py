"""Tests for read_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_app.observability.file_freshness import _key
from agent_app.tools.filesystem.read_file import read_file
from agent_harness.agent.base import BaseAgent
from agent_harness.core.errors import ToolValidationError


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_full_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("print('hello')\nprint('world')\n")
        result = await read_file.execute(file_path=str(f))
        assert "lines 1-2 of 2" in result
        assert "1\tprint('hello')" in result
        assert "2\tprint('world')" in result

    @pytest.mark.asyncio
    async def test_pagination(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
        result = await read_file.execute(file_path=str(f), offset=10, limit=5)
        assert "lines 11-15 of 100" in result
        assert "10 lines before" in result
        assert "85 lines after" in result

    @pytest.mark.asyncio
    async def test_pagination_from_start(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
        result = await read_file.execute(file_path=str(f), offset=0, limit=10)
        assert "lines 1-10 of 50" in result
        assert "before" not in result
        assert "40 lines after" in result

    @pytest.mark.asyncio
    async def test_binary_detection(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        result = await read_file.execute(file_path=str(f))
        assert isinstance(result, str)
        assert result.startswith("Error: binary file (not readable as text or media)")

    @pytest.mark.asyncio
    async def test_pdf_attachment(self, tmp_path: Path) -> None:
        from agent_harness.core.message import ToolOutput
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 minimal content")
        result = await read_file.execute(file_path=str(f))
        assert isinstance(result, ToolOutput)
        assert result.attachments and result.attachments[0].mime == "application/pdf"
        assert "Read PDF" in result.content
        assert "as an attachment" in result.content

    @pytest.mark.asyncio
    async def test_image_attachment(self, tmp_path: Path) -> None:
        from agent_harness.core.message import ToolOutput
        f = tmp_path / "cover.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        result = await read_file.execute(file_path=str(f))
        assert isinstance(result, ToolOutput)
        assert result.attachments and result.attachments[0].mime == "image/png"
        assert "Read image" in result.content

    @pytest.mark.asyncio
    async def test_default_limit_is_200(self) -> None:
        sch = read_file.get_schema()
        assert sch.parameters["properties"]["limit"]["default"] == 200

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = await read_file.execute(file_path=str(f))
        assert "empty file" in result

    @pytest.mark.asyncio
    async def test_path_escape_rejected(self) -> None:
        result = await read_file.execute(file_path="/etc/passwd")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_directory_rejected(self, tmp_path: Path) -> None:
        result = await read_file.execute(file_path=str(tmp_path))
        assert "directory" in result.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = await read_file.execute(file_path=str(tmp_path / "nope.txt"))
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ToolValidationError):
            await read_file.execute(file_path="any.txt", offset=-1)

    @pytest.mark.asyncio
    async def test_zero_limit_rejected(self) -> None:
        with pytest.raises(ToolValidationError):
            await read_file.execute(file_path="any.txt", limit=0)

    @pytest.mark.asyncio
    async def test_empty_path_rejected(self) -> None:
        with pytest.raises(ToolValidationError):
            await read_file.execute(file_path="  ")

    @pytest.mark.asyncio
    async def test_long_line_truncated(self, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        f.write_text("x" * 10000 + "\n")
        result = await read_file.execute(file_path=str(f))
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_utf8_bom(self, tmp_path: Path) -> None:
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfhello\n")
        result = await read_file.execute(file_path=str(f))
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_read_records_signature(
        self, tmp_path: Path, fs_agent: BaseAgent,
    ) -> None:
        f = tmp_path / "tracked.txt"
        f.write_text("hello\n")
        await read_file.execute(file_path=str(f))
        recorded = fs_agent.context.variables.get(_key(f))
        assert isinstance(recorded, dict)
        assert recorded["size"] == f.stat().st_size
