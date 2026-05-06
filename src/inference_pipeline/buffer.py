"""Bounded queue wrapper with close-aware iteration semantics."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from types import TracebackType
from typing import Final, Self, TypeGuard, TypeVar

T = TypeVar("T")


class _Sentinel:
    __slots__ = ()


_SENTINEL: Final[_Sentinel] = _Sentinel()


class BufferQ[T]:
    """A bounded producer/consumer queue with sentinel-based shutdown.

    The queue favors recent items on overflow by dropping the oldest payload when
    ``put`` is called on a full queue. Consumers should iterate via
    ``iter_until_closed`` or plain iteration on the instance.
    """

    def __init__(self, maxsize: int, default_timeout: float = 0.5):
        """Initialize the queue.

        Args:
            maxsize: Maximum number of queued items.
            default_timeout: Timeout used by ``iter_until_closed`` when none is passed.
        """
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0 for bounded buffer behavior.")
        self._queue: queue.Queue[T | _Sentinel] = queue.Queue(maxsize=maxsize)
        self._sentinel: Final[_Sentinel] = _SENTINEL
        self.default_timeout = default_timeout
        self._closed = False
        self._n_consumers = 0
        self._state_lock = threading.Lock()

    @property
    def closed(self) -> bool:
        """Return whether ``close`` has been called."""
        with self._state_lock:
            return self._closed

    def _is_payload(self, item: T | _Sentinel) -> TypeGuard[T]:
        """Narrow queue items to payload type for static type checkers."""
        return item is not self._sentinel

    def close(self) -> None:
        """Close the queue and unblock all active consumers.

        This call blocks until all active ``iter_until_closed`` consumers exit.
        Sentinels are never inserted by dropping queued payloads.
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        while True:
            with self._state_lock:
                if self._n_consumers == 0:
                    return
            try:
                self._queue.put(self._sentinel, timeout=self.default_timeout)
            except queue.Full:
                continue

    def put(self, item: T) -> None:
        """Put one payload item into the queue.

        When full, the oldest queued payload is dropped to make room.
        After closing, incoming items are ignored.
        """
        with self._state_lock:
            if self._closed:
                return
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(item)

    def iter_until_closed(
        self,
        timeout: float | None = None,
        stop: threading.Event | None = None,
    ) -> Iterator[T]:
        """Yield payload items until closed or an optional stop event is set.

        Args:
            timeout: Optional per-get timeout; falls back to ``default_timeout``.
            stop: Optional external cancellation event.
        """
        effective_timeout = self.default_timeout if timeout is None else timeout
        with self._state_lock:
            self._n_consumers += 1
        try:
            while True:
                try:
                    if stop is not None and stop.is_set():
                        return
                    item = self._queue.get(timeout=effective_timeout)
                except queue.Empty:
                    with self._state_lock:
                        if self._closed and self._queue.empty():
                            return
                    continue
                if self._is_payload(item):
                    yield item
                    continue
                return
        finally:
            with self._state_lock:
                self._n_consumers -= 1

    def __iter__(self) -> Iterator[T]:
        """Return a close-aware payload iterator."""
        return self.iter_until_closed()

    def __enter__(self) -> Self:
        """Return ``self`` for context-manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the queue when leaving a context."""
        self.close()
