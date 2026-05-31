"""Tests for filesystem security utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_app.tools.filesystem._security import (
    check_traversal,
    detect_text_file,
    include_matches,
    normalize_path,
    walk_files,
)


class TestNormalizePath:
    def test_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        assert normalize_path("src", workspace=tmp_path) == tmp_path / "src"

    def test_dot_path(self, tmp_path: Path) -> None:
        assert normalize_path(".", workspace=tmp_path) == tmp_path

    def test_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            normalize_path("nonexistent.txt", workspace=tmp_path, must_exist=True)

    def test_absolute_outside_allowed(self, tmp_path: Path) -> None:
        result = normalize_path("/etc/hosts", workspace=tmp_path)
        assert result == Path("/etc/hosts").resolve()

    def test_traversal_allowed(self, tmp_path: Path) -> None:
        result = normalize_path("../", workspace=tmp_path)
        assert result == tmp_path.parent.resolve()

    def test_home_path_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "secret").write_text("ok")
        result = normalize_path("~/secret", workspace=tmp_path, must_exist=True)
        assert result == (tmp_path / "secret").resolve()

    def test_symlink_outside_followed(self, tmp_path: Path) -> None:
        external = tmp_path.parent / "outside_target.txt"
        external.write_text("hi")
        link = tmp_path / "outlink.txt"
        link.symlink_to(external)
        result = normalize_path("outlink.txt", workspace=tmp_path, must_exist=True)
        assert result == external.resolve()

    def test_symlink_within_workspace_allowed(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("ok")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        result = normalize_path("link.txt", workspace=tmp_path, must_exist=True)
        assert result == target.resolve()


class TestCheckTraversal:
    def test_symlink_to_outside_skipped(self, tmp_path: Path) -> None:
        link = tmp_path / "escape"
        link.symlink_to("/etc/passwd")
        assert check_traversal(link, workspace=tmp_path) is False

    def test_symlink_within_workspace_passes(self, tmp_path: Path) -> None:
        target = tmp_path / "real.py"
        target.touch()
        link = tmp_path / "link.py"
        link.symlink_to(target)
        assert check_traversal(link, workspace=tmp_path) is True

    def test_broken_symlink_skipped(self, tmp_path: Path) -> None:
        link = tmp_path / "broken"
        link.symlink_to("/nonexistent_xyz")
        assert check_traversal(link, workspace=tmp_path) is False

    def test_regular_file_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.py"
        f.touch()
        assert check_traversal(f, workspace=tmp_path) is True


class TestDetectTextFile:
    def test_utf8_lf(self, tmp_path: Path) -> None:
        f = tmp_path / "plain.py"
        f.write_text("hello\nworld\n", encoding="utf-8")
        info = detect_text_file(f)
        assert info.encoding == "utf-8"
        assert info.newline == "\n"
        assert info.has_bom is False

    def test_utf8_bom(self, tmp_path: Path) -> None:
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfhello\n")
        info = detect_text_file(f)
        assert info.encoding == "utf-8-sig"
        assert info.has_bom is True

    def test_crlf_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "win.txt"
        f.write_bytes(b"line1\r\nline2\r\n")
        info = detect_text_file(f)
        assert info.newline == "\r\n"

    def test_binary_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Binary file"):
            detect_text_file(f)

    def test_utf16_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "utf16.txt"
        f.write_bytes(b"\xff\xfeh\x00i\x00")
        with pytest.raises(ValueError, match="UTF-16"):
            detect_text_file(f)

    def test_non_utf8_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "gbk.txt"
        f.write_bytes(b"\xc4\xe3\xba\xc3")
        with pytest.raises(ValueError, match="not valid UTF-8"):
            detect_text_file(f)

    def test_size_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "huge.txt"
        f.write_bytes(b"x" * 100)
        with pytest.raises(ValueError, match="too large"):
            detect_text_file(f, max_size=50)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        info = detect_text_file(f)
        assert info.content == ""
        assert info.newline == "\n"


class TestWalkFiles:
    def test_skips_git_directory(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git" / "objects"
        git_dir.mkdir(parents=True)
        (git_dir / "abc123").write_text("blob")
        (tmp_path / "src.py").write_text("pass")
        files = walk_files(tmp_path, "", workspace=tmp_path)
        names = [f.name for f in files]
        assert "src.py" in names
        assert "abc123" not in names

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("{}")
        (tmp_path / "app.js").write_text("ok")
        files = walk_files(tmp_path, "", workspace=tmp_path)
        names = [f.name for f in files]
        assert "app.js" in names
        assert "index.js" not in names

    def test_skips_pytest_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / ".pytest_cache"
        cache.mkdir()
        (cache / "data.json").write_text("{}")
        (tmp_path / "test.py").write_text("pass")
        files = walk_files(tmp_path, "", workspace=tmp_path)
        names = [f.name for f in files]
        assert "test.py" in names
        assert "data.json" not in names

    def test_include_basename_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("pass")
        (tmp_path / "notes.md").write_text("# hi")
        files = walk_files(tmp_path, "*.py", workspace=tmp_path)
        assert all(f.suffix == ".py" for f in files)

    def test_include_respects_depth(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("pass")
        sub = src / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("pass")
        (tmp_path / "setup.py").write_text("pass")

        files = walk_files(tmp_path, "src/*.py", workspace=tmp_path)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "deep.py" not in names
        assert "setup.py" not in names

    def test_include_recursive_pattern(self, tmp_path: Path) -> None:
        """src/**/*.py matches .py anywhere under src/, including direct children."""
        src = tmp_path / "src"
        sub = src / "sub"
        sub.mkdir(parents=True)
        (src / "main.py").write_text("pass")
        (sub / "deep.py").write_text("pass")
        (tmp_path / "setup.py").write_text("pass")

        files = walk_files(tmp_path, "src/**/*.py", workspace=tmp_path)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "deep.py" in names
        assert "setup.py" not in names

    def test_external_symlink_file_skipped(self, tmp_path: Path) -> None:
        link = tmp_path / "escape.py"
        link.symlink_to("/etc/passwd")
        (tmp_path / "real.py").write_text("pass")
        files = walk_files(tmp_path, "", workspace=tmp_path)
        names = [f.name for f in files]
        assert "real.py" in names
        assert "escape.py" not in names

    def test_symlink_dir_not_recursed(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real_subdir"
        real_dir.mkdir()
        (real_dir / "inside.py").write_text("pass")
        link_dir = tmp_path / "link_subdir"
        link_dir.symlink_to(real_dir)

        files = walk_files(tmp_path, "*.py", workspace=tmp_path)
        paths_str = [str(f) for f in files]
        assert sum("inside.py" in p for p in paths_str) == 1
        assert all("link_subdir" not in p for p in paths_str)

    def test_external_symlink_dir_not_entered(self, tmp_path: Path) -> None:
        link_dir = tmp_path / "external_dir"
        link_dir.symlink_to("/etc")
        (tmp_path / "real.py").write_text("pass")
        files = walk_files(tmp_path, "", workspace=tmp_path)
        for f in files:
            assert "etc" not in str(f)

    def test_empty_include_returns_all(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.md").write_text("hi")
        files = walk_files(tmp_path, "", workspace=tmp_path)
        names = [f.name for f in files]
        assert "a.py" in names
        assert "b.md" in names


class TestIncludeMatch:
    def test_bare_name_matches_any_depth(self) -> None:
        assert include_matches("foo.py", "*.py")
        assert include_matches("a/foo.py", "*.py")
        assert include_matches("a/b/foo.py", "*.py")
        assert not include_matches("foo.txt", "*.py")

    def test_globstar_prefix_includes_root(self) -> None:
        assert include_matches("foo.py", "**/*.py")
        assert include_matches("a/foo.py", "**/*.py")
        assert include_matches("a/b/foo.py", "**/*.py")

    def test_slash_pattern_anchors_to_direct_children(self) -> None:
        assert include_matches("src/foo.py", "src/*.py")
        assert not include_matches("src/sub/foo.py", "src/*.py")
        assert not include_matches("foo.py", "src/*.py")
        assert not include_matches("other/foo.py", "src/*.py")

    def test_trailing_globstar_matches_subtree(self) -> None:
        assert include_matches("src/foo.py", "src/**")
        assert include_matches("src/a/b.py", "src/**")
        assert not include_matches("foo.py", "src/**")
        assert not include_matches("srcx/y.py", "src/**")

    def test_globstar_in_middle_includes_direct_children(self) -> None:
        assert include_matches("src/main.py", "src/**/*.py")
        assert include_matches("src/a/b.py", "src/**/*.py")
        assert not include_matches("main.py", "src/**/*.py")

    def test_char_class_delegated_to_fnmatch(self) -> None:
        assert include_matches("a.c", "*.[ch]")
        assert include_matches("x/b.h", "*.[ch]")
        assert not include_matches("a.py", "*.[ch]")

    def test_prefix_glob_matches_any_depth(self) -> None:
        assert include_matches("test_foo.py", "test_*.py")
        assert include_matches("pkg/test_foo.py", "test_*.py")
        assert not include_matches("foo_test.py", "test_*.py")
