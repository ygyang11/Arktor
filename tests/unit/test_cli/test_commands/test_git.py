from unittest.mock import MagicMock, patch

import subprocess

from agent_cli.commands.builtin._git import run


async def test_run_returns_rc_stdout_stderr() -> None:
    completed = MagicMock(returncode=0, stdout="hello\n", stderr="")
    with patch("subprocess.run", return_value=completed) as m:
        rc, out, err = await run("status")
    assert (rc, out, err) == (0, "hello\n", "")
    m.assert_called_once()
    args, _ = m.call_args
    assert args[0] == ["git", "status"]


async def test_run_timeout_returns_minus_one() -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 1.0)):
        rc, out, err = await run("status", timeout=1.0)
    assert rc == -1
    assert out == ""
    assert err


async def test_run_os_error_returns_minus_one() -> None:
    with patch("subprocess.run", side_effect=OSError("no git")):
        rc, out, err = await run("status")
    assert rc == -1
    assert "no git" in err


async def test_run_does_not_block_event_loop() -> None:
    """Concurrent run() calls should overlap when wrapped via to_thread."""
    import asyncio
    import time

    def slow(*args, **kwargs):
        time.sleep(0.05)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=slow):
        start = time.monotonic()
        await asyncio.gather(run("a"), run("b"), run("c"))
        elapsed = time.monotonic() - start
    # 3x 50ms sequential = 150ms; via to_thread they overlap → well under 150ms
    assert elapsed < 0.12
