"""Shell lane execution — fresh subprocess per command with state persistence."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent_harness.utils.token_counter import truncate_text_by_tokens

if TYPE_CHECKING:
    from prompt_toolkit.completion import Completer

    from agent_cli.adapter import CliAdapter
    from agent_cli.runtime.session import SaveSession
    from agent_harness.agent.base import BaseAgent

logger = logging.getLogger(__name__)

_SNAPSHOT_TIMEOUT = 5.0
_CANCEL_GRACE = 2.0

_SHELL_LANE_OUTPUT_TOKENS = 10_000
_SHELL_RUN_OPEN_TAG = "<user-shell-run>"
_SHELL_RUN_CLOSE_TAG = "</user-shell-run>"
_SHELL_RUN_CLOSE_TAG_ESCAPED = "</user-shell-run​>"

_SHELL_RUN_RE = re.compile(
    r"\A" + re.escape(_SHELL_RUN_OPEN_TAG) + r"\s*\n"
    r"```sh\n(?P<cmd>.*?)\n```\n"
    r"(?P<body>.*?)\n" + re.escape(_SHELL_RUN_CLOSE_TAG) + r"\s*\Z",
    re.DOTALL,
)

_TRANSIENT_ENV = frozenset({
    "HARNESS_CWD_F", "HARNESS_ENV_F", "HARNESS_SNAP_OUT", "HARNESS_CMD",
    "PWD", "OLDPWD", "SHLVL", "_",
})

_INTERACTIVE_DENYLIST = frozenset({
    "vim", "vi", "nvim", "emacs", "nano", "pico", "helix", "hx",
    "less", "more", "most",
    "top", "htop", "btop", "glances", "iotop", "nvtop",
    "fzf", "dialog", "whiptail", "gum", "peco",
    "tmux", "screen",
    "man", "info",
    "gdb", "lldb", "pdb",
})

_SH_COMPATIBLE = frozenset({"bash", "zsh", "sh", "dash", "ash"})

_SNAPSHOT_SCRIPTS: dict[str, str] = {
    "bash": (
        'declare -f  >"$HARNESS_SNAP_OUT" 2>/dev/null; '
        'alias      >>"$HARNESS_SNAP_OUT" 2>/dev/null; '
        'shopt -p   >>"$HARNESS_SNAP_OUT" 2>/dev/null || true'
    ),
    "zsh": (
        'typeset -f  >"$HARNESS_SNAP_OUT" 2>/dev/null; '
        'alias -L   >>"$HARNESS_SNAP_OUT" 2>/dev/null'
    ),
}


def _escape_envelope(s: str) -> str:
    return s.replace(_SHELL_RUN_CLOSE_TAG, _SHELL_RUN_CLOSE_TAG_ESCAPED)


def parse_shell_run_envelope(content: str) -> tuple[str, str] | None:
    """Reverse of :func:`format_shell_run`. Returns (cmd, body) or None."""
    m = _SHELL_RUN_RE.match(content)
    if m is None:
        return None
    return m.group("cmd"), m.group("body")


def format_shell_run(
    command: str,
    exit_code: int,
    output: str,
    post_notices: list[str] | None = None,
) -> str:
    """Format a `!`-lane shell run as the body of a ``Message.user``."""
    truncated = truncate_text_by_tokens(
        output,
        max_tokens=_SHELL_LANE_OUTPUT_TOKENS,
        suffix="\n... (truncated)",
    )
    has_output = bool(truncated.strip())

    if exit_code != 0 and has_output:
        body = f"[exit code {exit_code}]\n{truncated}"
    elif exit_code != 0:
        body = f"[exit code {exit_code}]\n(Completed with no output)"
    elif has_output:
        body = truncated
    else:
        body = "(Completed with no output)"

    if post_notices:
        body = body + "\n" + "\n".join(f"[Accident] {n}" for n in post_notices)

    safe_command = _escape_envelope(command)
    safe_body = _escape_envelope(body)
    return f"<user-shell-run>\n```sh\n{safe_command}\n```\n{safe_body}\n</user-shell-run>"


def _detect_shell() -> str:
    s = os.environ.get("SHELL", "")
    if s and Path(s).is_file() and Path(s).name in _SH_COMPATIBLE:
        return s
    return "/bin/sh"


@dataclass
class ShellState:
    cwd: str = field(default_factory=os.getcwd)
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    shell_bin: str = field(default_factory=_detect_shell)
    _startup_path: str | None = None
    _snapshot_tried: bool = False
    _snapshot_pending_notice: str | None = None

    def cleanup(self) -> None:
        if self._startup_path:
            try:
                os.unlink(self._startup_path)
            except OSError:
                pass
            self._startup_path = None


def _detect_interactive(command: str) -> str | None:
    parts = command.strip().split()
    if not parts:
        return None

    while parts and "=" in parts[0] and not parts[0].startswith(("/", "-")):
        parts = parts[1:]
    if not parts:
        return None

    name = parts[0].rsplit("/", 1)[-1]
    if name in _INTERACTIVE_DENYLIST:
        return name

    if name == "sudo" and len(parts) > 1:
        target = parts[1].rsplit("/", 1)[-1]
        if target in _INTERACTIVE_DENYLIST:
            return target

    return None


async def capture_startup_snapshot(state: ShellState) -> None:
    if state._snapshot_tried:
        return
    state._snapshot_tried = True

    base = Path(state.shell_bin).name
    script = _SNAPSHOT_SCRIPTS.get(base)
    if script is None:
        return

    fd, path = tempfile.mkstemp(prefix="harness-snapshot-", suffix=".sh")
    os.close(fd)

    env = {**state.env, "HARNESS_SNAP_OUT": path}
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            state.shell_bin,
            "-ic",
            script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        await asyncio.wait_for(proc.wait(), timeout=_SNAPSHOT_TIMEOUT)
        if proc.returncode != 0 or not Path(path).stat().st_size:
            os.unlink(path)
            return
        state._startup_path = path
    except TimeoutError:
        if proc is not None:
            _killpg_now(proc)
            await proc.wait()
        state._snapshot_pending_notice = "capture timed out"
        try:
            os.unlink(path)
        except OSError:
            pass
    except asyncio.CancelledError:
        if proc is not None:
            _killpg_now(proc)
            await proc.wait()
        try:
            os.unlink(path)
        except OSError:
            pass
        state._snapshot_tried = False
        raise
    except Exception:
        state._snapshot_pending_notice = "capture failed"
        try:
            os.unlink(path)
        except OSError:
            pass


async def exec_shell(
    state: ShellState,
    command: str,
    agent: BaseAgent,
    completer: Completer,
    adapter: CliAdapter,
    save: SaveSession,
) -> None:
    if sys.platform == "win32":
        await adapter.on_shell_run(
            command,
            1,
            "Shell lane is not supported on Windows.",
        )
        return

    blocked = _detect_interactive(command)
    if blocked is not None:
        from agent_cli.render.notices import format_warning  # noqa: PLC0415

        await adapter.print_inline(
            format_warning(
                f"Cannot run {blocked!r}: interactive programs need a real TTY "
                f"but shell lane doesn't provide."
            )
        )
        return

    await _ensure_cwd_valid(state, agent, completer, adapter)
    await capture_startup_snapshot(state)

    if state._snapshot_pending_notice is not None:
        from agent_cli.render.notices import format_warning  # noqa: PLC0415

        await adapter.print_inline(
            format_warning(
                f"Shell snapshot unavailable: {state._snapshot_pending_notice}"
                f" (aliases/functions not loaded)"
            )
        )
        state._snapshot_pending_notice = None

    cwd_f: tempfile._TemporaryFileWrapper[bytes] | None = None
    env_f: tempfile._TemporaryFileWrapper[bytes] | None = None
    proc: asyncio.subprocess.Process | None = None
    try:
        cwd_f = tempfile.NamedTemporaryFile(delete=False, prefix="harness-cwd-")
        env_f = tempfile.NamedTemporaryFile(delete=False, prefix="harness-env-")
        cwd_f.close()
        env_f.close()

        script = _build_script(state, cwd_f.name, env_f.name)
        sent_env = {
            **state.env,
            "HARNESS_CWD_F": cwd_f.name,
            "HARNESS_ENV_F": env_f.name,
            "HARNESS_CMD": command,
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                state.shell_bin,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=sent_env,
                cwd=state.cwd,
                start_new_session=True,
            )
        except FileNotFoundError:
            await adapter.on_shell_run(
                command,
                127,
                f"Shell binary not found: {state.shell_bin}",
            )
            return

        try:
            out_bytes, _ = await proc.communicate()
        except asyncio.CancelledError:
            await asyncio.shield(_kill_group(proc))
            raise

        exit_code = proc.returncode if proc.returncode is not None else 1
        output = out_bytes.decode(errors="replace").rstrip("\n")

        await adapter.on_shell_run(command, exit_code, output)

        new_cwd = _read_text(cwd_f.name).strip() or state.cwd
        parsed_env = _parse_env0(_read_bytes(env_f.name))
        old_cwd = state.cwd
        post_notices: list[str] = []

        if new_cwd != old_cwd:
            bg = getattr(agent, "_bg_manager", None)
            if bg is not None and bg.has_running():
                from agent_cli.render.notices import format_warning  # noqa: PLC0415

                msg = (
                    "Cannot change directory while background tasks are "
                    f"running; keeping {old_cwd}"
                )
                await adapter.print_inline(format_warning(msg))
                post_notices.append(msg)
                new_cwd = old_cwd
            else:
                try:
                    os.chdir(new_cwd)
                except OSError as e:
                    from agent_cli.render.notices import format_warning  # noqa: PLC0415

                    msg = f"Could not change directory to {new_cwd}: {e}"
                    await adapter.print_inline(format_warning(msg))
                    post_notices.append(msg)
                    new_cwd = old_cwd
                else:
                    if hasattr(completer, "invalidate_file_root"):
                        completer.invalidate_file_root(Path(new_cwd))
                    try:
                        from agent_cli.runtime.conversation import (  # noqa: PLC0415
                            refresh_system_prompt,
                        )

                        refresh_system_prompt(agent)
                    except Exception:
                        logger.exception("refresh_system_prompt failed after cwd change")

        state.cwd = new_cwd
        if "PATH" in parsed_env:
            state.env = _merge_env(sent_env, parsed_env)

        from agent_cli.runtime.conversation import append_shell_run  # noqa: PLC0415
        await append_shell_run(
            agent,
            command=command,
            exit_code=exit_code,
            output=output,
            save=save,
            post_notices=post_notices,
        )
    finally:
        for f in (cwd_f, env_f):
            if f is not None:
                try:
                    os.unlink(f.name)
                except OSError:
                    pass


def _killpg_now(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=_CANCEL_GRACE)
    except TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await proc.wait()


def _build_script(state: ShellState, cwd_f: str, env_f: str) -> str:
    head = ""
    if state._startup_path:
        if Path(state._startup_path).is_file():
            head = f"source {shlex.quote(state._startup_path)} 2>/dev/null\n"
        else:
            state._startup_path = None
            state._snapshot_tried = False
    head += (
        "shopt -s expand_aliases 2>/dev/null || true\n"
        'trap \'__s=$?; pwd -P > "$HARNESS_CWD_F"; '
        'env -0 > "$HARNESS_ENV_F"; exit $__s\' EXIT\n'
    )
    return head + 'eval "$HARNESS_CMD"\n'


async def _ensure_cwd_valid(
    state: ShellState,
    agent: BaseAgent,
    completer: Completer,
    adapter: CliAdapter,
) -> None:
    if Path(state.cwd).is_dir():
        return
    p = Path(state.cwd)
    while p != p.parent and not p.is_dir():
        p = p.parent
    fallback = str(p) if p.is_dir() else str(Path.home())
    state.cwd = fallback
    os.chdir(fallback)
    if hasattr(completer, "invalidate_file_root"):
        completer.invalidate_file_root(Path(fallback))
    from agent_cli.runtime.conversation import refresh_system_prompt  # noqa: PLC0415

    refresh_system_prompt(agent)
    from agent_cli.render.notices import format_warning  # noqa: PLC0415

    await adapter.print_inline(
        format_warning(f"Working directory was removed; falling back to {fallback}")
    )


def _merge_env(
    sent: dict[str, str],
    parsed: dict[str, str],
) -> dict[str, str]:
    result = dict(sent)
    for k in _TRANSIENT_ENV:
        result.pop(k, None)
    for k, v in parsed.items():
        if k in _TRANSIENT_ENV:
            continue
        if sent.get(k) != v:
            result[k] = v
    for k in list(sent):
        if k in _TRANSIENT_ENV:
            continue
        if k not in parsed:
            result.pop(k, None)
    return result


def _read_text(p: str) -> str:
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return ""


def _read_bytes(p: str) -> bytes:
    try:
        return Path(p).read_bytes()
    except OSError:
        return b""


def _parse_env0(blob: bytes) -> dict[str, str]:
    if not blob:
        return {}
    out: dict[str, str] = {}
    for item in blob.split(b"\0"):
        if not item:
            continue
        k, eq, v = item.partition(b"=")
        if not eq:
            continue
        try:
            out[k.decode()] = v.decode(errors="replace")
        except UnicodeDecodeError:
            continue
    return out
