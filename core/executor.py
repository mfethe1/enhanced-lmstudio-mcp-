from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeout
from typing import Any, Optional


class AsyncExecutor:
    """Centralized async execution with proper loop management.

    - Maintains a single background event loop on a daemon thread
    - Provides synchronous `run(coro, timeout=None)` to execute coroutines safely
    - Provides `schedule(coro)` to schedule without blocking and get a Future
    - Thread-safe and resilient to caller context (works inside/outside running loops)
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_loop_thread()

    # --- public API ---
    def run(self, coro: Any, timeout: Optional[float] = None) -> Any:
        """Run a coroutine on the executor loop and return result.

        If called from any thread (including one already running an event loop), this
        will submit the coroutine to the dedicated loop thread and wait for the result.
        """
        loop = self._ensure_loop()
        if not asyncio.iscoroutine(coro):
            raise TypeError("AsyncExecutor.run expects a coroutine")
        fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            # Attempt to cancel task on timeout
            try:
                fut.cancel()
            except Exception:
                pass
            raise TimeoutError(f"Async task timed out after {timeout}s")

    def schedule(self, coro: Any) -> Future:
        """Schedule a coroutine for execution and return a concurrent.futures.Future."""
        loop = self._ensure_loop()
        if not asyncio.iscoroutine(coro):
            raise TypeError("AsyncExecutor.schedule expects a coroutine")
        return asyncio.run_coroutine_threadsafe(coro, loop)

    # --- internals ---
    def _start_loop_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, name="AsyncExecutorLoop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._start_loop_thread()
        assert self._loop is not None, "Executor loop not initialized"
        return self._loop

    # Optional graceful shutdown if ever needed by tests
    def shutdown(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None
        self._ready.clear()


# Singleton instance for easy import
async_executor = AsyncExecutor()

