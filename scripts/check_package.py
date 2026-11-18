"""Check that a built wheel installs and exposes the SDK and CLI."""
from __future__ import annotations

import argparse
import os
import subprocess
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

_TEMPLATE = Path(__file__).resolve().parents[1] / "arktor.example.yaml"

_SDK_CHECK = """\
import asyncio
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import agent_harness
from agent_app.tools import read_file
from agent_harness import HarnessConfig, ReActAgent, RecordingHooks

assert Path(agent_harness.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
Path("sample.txt").write_text("package check", encoding="utf-8")
assert "package check" in asyncio.run(read_file.execute(file_path="sample.txt"))
for package in ("agent_harness", "agent_app", "agent_cli"):
    assert files(package).joinpath("py.typed").is_file(), f"{package}: missing py.typed"
template = files("agent_harness").joinpath("resources", "arktor.example.yaml")
assert template.read_bytes() == Path(sys.argv[1]).read_bytes(), "Config template mismatch"
installed_version = version("arktor")
assert agent_harness.__version__ == installed_version, "SDK version mismatch"
print(installed_version)
"""


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(
        command, cwd=cwd, env=env, text=True, timeout=600,
    ).strip()


def check(wheel: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("PYTHON")}
    with TemporaryDirectory(prefix="arktor-package-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.create(environment, with_pip=True)
        binary = environment / ("Scripts" if os.name == "nt" else "bin")
        python = str(binary / ("python.exe" if os.name == "nt" else "python"))
        arktor = str(binary / ("arktor.exe" if os.name == "nt" else "arktor"))
        install = [python, "-I", "-m", "pip", "install", "--quiet", "--disable-pip-version-check"]

        _run([*install, str(wheel)], cwd=root, env=env)
        version = _run([python, "-I", "-c", _SDK_CHECK, str(_TEMPLATE)], cwd=root, env=env)

        _run([*install, f"arktor[cli,app,sandbox] @ {wheel.as_uri()}"], cwd=root, env=env)
        _run([python, "-I", "-m", "pip", "check"], cwd=root, env=env)
        cli_version = _run([arktor, "--version"], cwd=root, env=env)
        if cli_version != f"arktor {version}":
            raise RuntimeError(f"Expected arktor {version}, got {cli_version}")
        _run([python, "-I", "-m", "agent_cli", "--help"], cwd=root, env=env)
    print(f"Package checks passed: {wheel.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve()
    if wheel.suffix != ".whl" or not wheel.is_file():
        parser.error(f"expected a wheel file: {wheel}")
    check(wheel)
