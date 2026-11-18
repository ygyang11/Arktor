"""Tests for scripts/install.sh with isolated command stubs."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"


@pytest.mark.parametrize("version", [None, "0.6.0"])
def test_installer_existing_uv_and_repeat(tmp_path: Path, version: str | None) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "calls"
    uv = binary / "uv"
    uv.write_text("""#!/bin/sh
printf '%s\\n' "$*" >> "$TEST_LOG"
case "$1 $2" in
  'tool dir') printf '%s\\n' "$TEST_BIN" ;;
  'tool install'|'tool update-shell') exit 0 ;;
  *) exit 7 ;;
esac
""")
    uv.chmod(0o755)
    arktor = binary / "arktor"
    arktor.write_text('#!/bin/sh\nprintf "arktor 0.6.0\\n"\n')
    arktor.chmod(0o755)
    env = {"PATH": str(binary), "TEST_LOG": str(log), "TEST_BIN": str(binary)}
    args = ["/bin/sh", str(_SCRIPT)]
    if version is not None:
        args.extend(["--version", version])

    for _ in range(2):
        result = subprocess.run(args, env=env, text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert "Run: arktor" in result.stdout

    calls = log.read_text()
    assert calls.count("tool install --upgrade --python 3.11") == 2
    assert "arktor[cli,app,sandbox]" + ("" if version is None else f"=={version}") in calls
    assert calls.count("tool update-shell") == 2


@pytest.mark.parametrize("args", [["--version"], ["--wrong"], ["--version", ""]])
def test_installer_rejects_invalid_options(args: list[str]) -> None:
    result = subprocess.run(
        ["/bin/sh", str(_SCRIPT), *args], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2 and result.stderr


@pytest.mark.parametrize("failure", ["download", "install", "shell"])
def test_installer_failures(tmp_path: Path, failure: str) -> None:
    binary = tmp_path / "bin with spaces"
    binary.mkdir()
    env = {
        "PATH": str(binary), "TMPDIR": str(tmp_path),
        "UV_INSTALL_DIR": str(tmp_path / "new uv"),
        "TEST_BIN": str(binary), "FAILURE": failure,
    }
    if failure == "download":
        for utility in ("mktemp", "rm"):
            path = shutil.which(utility)
            assert path is not None
            (binary / utility).symlink_to(path)
        command = binary / "curl"
        command.write_text("#!/bin/sh\nexit 23\n")
    else:
        command = binary / "uv"
        command.write_text("""#!/bin/sh
case "$1 $2" in
  'tool install') [ "$FAILURE" != install ] || exit 24 ;;
  'tool dir') printf '%s\\n' "$TEST_BIN" ;;
  'tool update-shell') [ "$FAILURE" != shell ] || exit 25 ;;
esac
""")
        arktor = binary / "arktor"
        arktor.write_text('#!/bin/sh\nprintf "arktor 0.6.0\\n"\n')
        arktor.chmod(0o755)
    command.chmod(0o755)

    result = subprocess.run(
        ["/bin/sh", str(_SCRIPT)], env=env, text=True, capture_output=True, timeout=10,
    )

    assert result.returncode == {"download": 23, "install": 24, "shell": 0}[failure]
    if failure == "shell":
        assert "PATH setup was not confirmed" in result.stderr
    else:
        assert "Arktor installed at:" not in result.stdout
