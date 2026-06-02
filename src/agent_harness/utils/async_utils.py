"""Async utility functions for agent_harness."""
from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_sync_pool: concurrent.futures.ThreadPoolExecutor | None = None


def _get_sync_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _sync_pool
    if _sync_pool is None:
        _sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        atexit.register(_sync_pool.shutdown, wait=False)
    return _sync_pool


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    If an event loop is already running, creates a new thread to run the coroutine.
    Otherwise, uses asyncio.run().
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    pool = _get_sync_pool()
    future = pool.submit(asyncio.run, coro)
    return future.result()


async def gather_with_concurrency(
    n: int,
    *coros: Coroutine[Any, Any, T],
) -> list[T]:
    """Like asyncio.gather but with a concurrency limit.

    Args:
        n: Maximum number of coroutines to run concurrently.
        *coros: Coroutines to execute.

    Returns:
        Results in the same order as input coroutines.
    """
    semaphore = asyncio.Semaphore(n)
    results: list[Any] = [None] * len(coros)

    async def limited(index: int, coro: Coroutine[Any, Any, T]) -> None:
        async with semaphore:
            results[index] = await coro

    await asyncio.gather(*(limited(i, c) for i, c in enumerate(coros)))
    return results


async def async_retry(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        fn: Async callable to retry.
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for wait time between retries.
        retryable_exceptions: Tuple of exception types that trigger retry.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable_exceptions as e:
            last_error = e
            if attempt < max_retries:
                wait = backoff_factor ** attempt
                await asyncio.sleep(wait)
    raise last_error  # type: ignore[misc]


def ensure_async(fn: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Wrap a sync function to be async (runs in thread executor)."""
    if asyncio.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

    return wrapper
