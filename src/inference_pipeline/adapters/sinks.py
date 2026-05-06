"""Abstract sink-adapter interfaces for queue-based pipeline outputs."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from inference_pipeline.buffer import BufferQ
from inference_pipeline.runtime import ThreadTask


class SinkAdapter[T](ABC):
    """Interface for start/stop managed sinks that consume from a ``BufferQ``."""

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
        """Stop the sink when leaving a context."""
        self.stop()


class ManagedSinkAdapter[T](SinkAdapter[T], ABC):
    """Reusable base for sink lifecycle and consumer-style orchestration."""

    def __init__(self) -> None:
        """Create a managed sink with thread-safe lifecycle state."""
        self._state_lock = threading.RLock()
        self._running = False
        self._shutdown_event = threading.Event()

    @property
    def running(self) -> bool:
        """Return whether the sink has been started and not yet stopped."""
        with self._state_lock:
            return self._running

    def start(self) -> None:
        """Start the sink once; repeated calls are no-ops."""
        with self._state_lock:
            if self._running:
                return
            self._shutdown_event.clear()
        self._start_impl()
        with self._state_lock:
            self._running = True

    def stop(self) -> None:
        """Stop the sink once; repeated calls are no-ops."""
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            self._shutdown_event.set()
        self._stop_impl()

    def run(self, input_q: BufferQ[T], stop: threading.Event | None = None) -> None:
        """Run sink lifecycle until input closes or a stop signal is set."""
        effective_stop = stop if stop is not None else self._shutdown_event
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
            thread_name=name or f"{type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    def _start_impl(self) -> None:
        """Subclass hook containing concrete startup logic."""

    def _stop_impl(self) -> None:
        """Subclass hook containing concrete teardown logic."""
