"""Abstract sink-adapter interfaces for queue-based pipeline outputs."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from inference_pipeline.buffer import BufferQ
from inference_pipeline.protocols import BaseConfigI
from inference_pipeline.runtime import ThreadTask


class _OrStopEvent(threading.Event):
    """Read-only event that reports set when either source event is set."""

    def __init__(self, left: threading.Event, right: threading.Event) -> None:
        super().__init__()
        self._left = left
        self._right = right

    def is_set(self) -> bool:
        return self._left.is_set() or self._right.is_set()


class SinkAdapter[T, ConfT: BaseConfigI](ABC):
    """Interface for start/stop managed sinks that consume from a ``BufferQ``."""

    def __init__(self, config: ConfT):
        self.config = config

    @property
    @abstractmethod
    def running(self) -> bool:
        """Return whether the sink is currently running."""
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """Start sink lifecycle resources."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop sink lifecycle resources."""
        raise NotImplementedError

    @abstractmethod
    def consume(self, item: T) -> None:
        """Consume one item from an input queue."""
        raise NotImplementedError

    @abstractmethod
    def run(self, input_q: BufferQ[T], stop: threading.Event | None = None) -> None:
        """Run sink consumption in blocking mode until closure or stop signal."""
        raise NotImplementedError

    @abstractmethod
    def threaded(
        self,
        input_q: BufferQ[T],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this sink in a worker thread."""
        raise NotImplementedError

    def __enter__(self) -> Self:
        """Start the sink in context-manager usage."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the sink when leaving a context.

        If the context body already raised, sink shutdown errors are ignored so
        the original exception is preserved.
        """
        if exc_type is not None:
            try:
                self.stop()
            except BaseException:
                return
            return
        self.stop()


class ManagedSinkAdapter[T, ConfT: BaseConfigI](SinkAdapter[T, ConfT], ABC):
    """Reusable base for sink lifecycle and consumer-style orchestration.

    Lifecycle operations are serialized so concurrent ``start()`` and ``stop()``
    calls cannot overlap.
    """

    def __init__(self, *, config: ConfT) -> None:
        """Create a managed sink with thread-safe lifecycle state."""
        self.config = config
        self._state_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._running = False
        self._shutdown_event = threading.Event()

    @property
    def running(self) -> bool:
        """Return whether the sink has been started and not yet stopped."""
        with self._state_lock:
            return self._running

    def start(self) -> None:
        """Start the sink once; repeated calls while running are no-ops.

        Startup is lifecycle-serialized and rolls back running state if
        ``_start_impl`` raises.
        """
        with self._lifecycle_lock:
            with self._state_lock:
                if self._running:
                    return
                self._running = True
                self._shutdown_event.clear()
            try:
                self._start_impl()
            except BaseException:
                with self._state_lock:
                    self._running = False
                    self._shutdown_event.set()
                raise

    def stop(self) -> None:
        """Stop the sink once; repeated calls are no-ops."""
        with self._lifecycle_lock:
            with self._state_lock:
                if not self._running:
                    return
                self._running = False
                self._shutdown_event.set()
            self._stop_impl()

    def run(self, input_q: BufferQ[T], stop: threading.Event | None = None) -> None:
        """Run sink lifecycle until input closes or any stop signal is set.

        Consumption stops when either the optional external stop event is set or
        this sink is stopped via ``stop()``.
        """
        effective_stop = (
            self._shutdown_event
            if stop is None
            else _OrStopEvent(self._shutdown_event, stop)
        )

        with self:
            for item in input_q.iter_until_closed(stop=effective_stop):
                self.consume(item)

    def threaded(
        self,
        input_q: BufferQ[T],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this sink in a worker thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=name or f"{self.config.name or type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    def _start_impl(self) -> None:
        """Subclass hook containing concrete startup logic."""

    def _stop_impl(self) -> None:
        """Subclass hook containing concrete teardown logic."""
