"""Tests for log_info — tail_lines and summarize_log."""
from __future__ import annotations

from pathlib import Path

from agent_harness.utils.log_info import summarize_log, tail_lines


class TestTailLines:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert tail_lines(tmp_path / "nope.txt", 5) == ("", False)

    def test_full_when_under_n(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("a\nb\nc\n")
        tail, truncated = tail_lines(p, 5)
        assert tail == "a\nb\nc"
        assert truncated is False

    def test_truncated_when_over_n(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("\n".join(str(i) for i in range(10)) + "\n")
        tail, truncated = tail_lines(p, 3)
        assert tail == "7\n8\n9"
        assert truncated is True

    def test_window_reads_only_tail(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("\n".join(str(i) for i in range(10000)) + "\n")
        tail, truncated = tail_lines(p, 2, window=256)
        assert tail.splitlines()[-1] == "9999"
        assert truncated is True

    def test_truncated_by_window_even_with_few_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("x" * 200_000 + "\nlast\n")
        tail, truncated = tail_lines(p, 30)
        assert truncated is True
        assert "last" in tail


class TestSummarizeLog:
    def test_with_exit_code(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("hello\n")
        out = summarize_log(p, 0)
        assert "Exit code: 0" in out
        assert "Output:\nhello" in out

    def test_without_exit_code(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("hello\n")
        out = summarize_log(p)
        assert "Exit code" not in out
        assert "hello" in out

    def test_truncated_label(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("\n".join(str(i) for i in range(50)) + "\n")
        out = summarize_log(p, 0, n=5)
        assert "Output (last 5 lines):" in out

    def test_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "log.txt"
        p.write_text("")
        out = summarize_log(p, 0)
        assert "Output: (none)" in out
