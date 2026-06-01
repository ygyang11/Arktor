import asyncio
from unittest.mock import MagicMock

from agent_cli.render.live_line import CLEAR_LINE, LiveLine


def _make_line(
    writer=None, debounce_s: float = 0.0, tick_s: float = 0.005,
) -> LiveLine:
    console = MagicMock()
    console.file = MagicMock()
    lock = asyncio.Lock()
    if writer is None:
        def writer(frame: int, elapsed: int) -> str:
            return f"frame={frame} t={elapsed}"
    return LiveLine(console, lock, writer, debounce_s=debounce_s, tick_s=tick_s)


async def test_start_debounced_then_quick_stop_no_write() -> None:
    line = _make_line(debounce_s=0.5)
    await line.start()
    await line.stop()
    # Loop cancelled during debounce sleep — no write ever happened.
    assert line._console.file.write.call_count == 0
    assert line.visible is False
    assert line.task is None


async def test_start_clears_stale_line_before_new_loop() -> None:
    line = _make_line(debounce_s=0.0, tick_s=0.005)
    await line.start()
    await asyncio.sleep(0.02)
    assert line.visible is True
    writes_before = line._console.file.write.call_count

    await line.start()
    # The restart path must emit CLEAR_LINE (from the embedded stop() →
    # _clear()) before the new loop's first tick. Without that, a stale
    # visible line would linger during the new loop's debounce window.
    writes_after = line._console.file.write.call_count
    assert writes_after > writes_before, "restart must clear stale line"
    await line.stop()


async def test_tick_calls_writer_with_incrementing_frame() -> None:
    frames_seen: list[int] = []

    def writer(frame: int, elapsed: int) -> str:
        frames_seen.append(frame)
        return "x"

    line = _make_line(writer=writer, debounce_s=0.0, tick_s=0.005)
    await line.start()
    await asyncio.sleep(0.03)
    await line.stop()
    assert len(frames_seen) >= 2
    assert frames_seen[0] == 0
    assert all(b == a + 1 for a, b in zip(frames_seen, frames_seen[1:]))


async def test_stop_no_lock_does_not_acquire_lock() -> None:
    line = _make_line(debounce_s=0.0)
    async with line._lock:
        # If stop_no_lock tried to acquire the lock, we'd deadlock.
        await asyncio.wait_for(line.stop_no_lock(), timeout=0.5)


async def test_clear_no_lock_is_noop_when_invisible() -> None:
    line = _make_line()
    # Never started — visible is False. clear_no_lock must not write.
    line.clear_no_lock()
    assert line._console.file.write.call_count == 0


async def test_loop_writes_clear_line_prefix_each_tick() -> None:
    line = _make_line(debounce_s=0.0, tick_s=0.005)
    await line.start()
    await asyncio.sleep(0.02)
    await line.stop()
    writes = [c.args[0] for c in line._console.file.write.call_args_list]
    # At least one tick emitted the CLEAR_LINE prefix + writer output.
    assert any(w.startswith(CLEAR_LINE) for w in writes)
