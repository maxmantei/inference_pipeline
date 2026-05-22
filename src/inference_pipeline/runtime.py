"""Threaded execution helpers for pipeline stages.

This module keeps threading concerns separate from stage processing logic while
providing ergonomic adapters for running long-lived stage loops in worker
threads.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self


@dataclass(slots=True)
class ThreadTask:
    """Context-managed wrapper around a worker thread.

    Instances are typically created via :meth:`from_runner`.

    Lifecycle methods are intentionally forgiving for orchestration code:
    ``start()`` is idempotent, and ``join()`` before ``start()`` is a no-op.
    """

    _runner: Callable[[], None]
    name: str
    daemon: bool = True
    join_timeout: float = 5.0
    _thread: threading.Thread | None = field(init=False, default=None)
    _error: BaseException | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Validate thread task configuration."""
        if self.join_timeout < 0:
            raise ValueError("join_timeout must be >= 0")
        if not self.name:
            raise ValueError("name must not be empty")

    @classmethod
    def from_runner(
        cls,
        runner: Callable[[], None],
        *,
        thread_name: str,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> Self:
        """Create a ``ThreadTask`` from a bound zero-argument runner."""
        return cls(
            _runner=runner,
            name=thread_name,
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @property
    def is_alive(self) -> bool:
        """Return whether the managed thread is currently alive."""
        return self._thread.is_alive() if self._thread is not None else False

    @property
    def error(self) -> BaseException | None:
        """Return the worker exception, if one has occurred."""
        return self._error

    def _run(self) -> None:
        """Invoke the user runner and capture unexpected errors."""
        try:
            self._runner()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def _raise_worker_error(self) -> None:
        """Raise a wrapped worker error if one has been captured."""
        if self._error is None:
            return
        raise RuntimeError(f"Worker thread {self.name!r} failed.") from self._error

    def start(self) -> None:
        """Start the managed worker thread once.

        Repeated calls are ignored, allowing defensive start calls from multiple
        code paths without raising lifecycle errors.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=self.name,
            daemon=self.daemon,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for worker completion and surface timeout or worker errors.

        ``timeout`` overrides the per-task ``join_timeout`` when provided.
        Calling ``join()`` before ``start()`` is a no-op, which makes teardown
        code safe to call unconditionally in ``finally`` blocks.
        """
        if self._thread is None:
            return
        effective = timeout if timeout is not None else self.join_timeout
        self._thread.join(timeout=effective)
        if self._thread.is_alive():
            message = (
                f"Worker thread {self.name!r} did not finish within {effective:.1f}s."
            )
            raise TimeoutError(message)
        self._raise_worker_error()

    def __enter__(self) -> Self:
        """Start the worker thread and return the task itself."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Join the worker and propagate worker failures when appropriate.

        If the ``with`` body raised an exception, that exception is preserved and
        worker timeout/failure signals are not re-raised from ``__exit__``.
        """
        if self._thread is None:
            return
        self._thread.join(timeout=self.join_timeout)
        if exc_type is not None:
            return
        if self._thread.is_alive():
            message = (
                f"Worker thread {self.name!r} did not finish within "
                f"{self.join_timeout:.1f}s."
            )
            raise TimeoutError(message)
        self._raise_worker_error()
