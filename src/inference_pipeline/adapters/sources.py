"""Abstract source-adapter interfaces for queue-based pipeline ingestion."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from inference_pipeline.buffer import BufferQ
from inference_pipeline.protocols import BaseProducerConfigI
from inference_pipeline.runtime import ThreadTask


class SourceAdapter[T, ConfT: BaseProducerConfigI](ABC):
    """Interface for start/stop managed sources that publish to a BufferQ."""

    def __init__(self, *, config: ConfT):
        self.config = config

    @property
    @abstractmethod
    def output(self) -> BufferQ[T]:
        """Return the output queue that downstream stages consume from."""
        raise NotImplementedError

    @property
    @abstractmethod
    def running(self) -> bool:
        """Return whether the source is currently running."""
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """Start source acquisition/production."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop source acquisition/production.

        Implementations may treat ``stop()`` before ``start()`` as a no-op.
        Managed implementations close output when stopping a running source.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, stop: threading.Event | None = None) -> None:
        """Run source lifecycle in blocking mode until an optional stop signal."""
        raise NotImplementedError

    @abstractmethod
    def threaded(
        self,
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this source in a worker thread."""
        raise NotImplementedError

    def __enter__(self) -> Self:
        """Start the source in context-manager usage."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the source when leaving a context.

        If the context body already raised, source shutdown errors are ignored
        so the original exception is preserved.
        """
        if exc_type is not None:
            try:
                self.stop()
            except BaseException:
                return
            return
        self.stop()


class ManagedSourceAdapter[T, ConfT: BaseProducerConfigI](SourceAdapter[T, ConfT], ABC):
    """Reusable base for source lifecycle and producer-style orchestration.

    A managed source is single-use: once stopped, its output queue is closed and
    restarting is forbidden. Create a new adapter instance to start again.
    """

    def __init__(self, *, config: ConfT) -> None:
        """Create a managed source with a bounded output queue."""
        self.config = config
        self._output = BufferQ[T](
            maxsize=config.out_maxsize, default_timeout=config.out_timeout
        )
        self._state_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._running = False
        self._shutdown_event = threading.Event()

    @property
    def output(self) -> BufferQ[T]:
        """Return this source's output queue.

        The queue is closed permanently on stop; restarting is not supported.
        """
        return self._output

    @property
    def running(self) -> bool:
        """Return whether the source has been started and not yet stopped."""
        with self._state_lock:
            return self._running

    def start(self) -> None:
        """Start the source once; repeated calls while running are no-ops.

        Startup is lifecycle-serialized so concurrent ``start()`` and ``stop()``
        calls cannot overlap. Restart after stop is forbidden.
        """
        with self._lifecycle_lock:
            with self._state_lock:
                if self._running:
                    return
                if self._output.closed:
                    raise RuntimeError(
                        "Cannot restart a stopped source adapter; "
                        "create a new adapter instance."
                    )
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
        """Stop the source once; repeated calls are no-ops.

        Calling ``stop()`` before ``start()`` is also a no-op so callers can
        run cleanup code unconditionally in shutdown paths.
        """
        with self._lifecycle_lock:
            with self._state_lock:
                if not self._running:
                    return
                self._running = False
                self._shutdown_event.set()
            try:
                self._stop_impl()
            finally:
                self._output.close()

    def run(self, stop: threading.Event | None = None) -> None:
        """Run source lifecycle in blocking mode until stop is requested.

        The method enters source context management and blocks until either the
        external ``stop`` event is set or this adapter begins shutting down.
        """
        with self:
            while True:
                if self._shutdown_event.wait(timeout=0.1):
                    return
                if stop is not None and stop.is_set():
                    return

    def threaded(
        self,
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this source in a worker thread."""
        return ThreadTask.from_runner(
            lambda: self.run(stop=stop),
            thread_name=name or f"{type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def _start_impl(self) -> None:
        """Subclass hook containing concrete startup logic."""
        raise NotImplementedError

    @abstractmethod
    def _stop_impl(self) -> None:
        """Subclass hook containing concrete teardown logic."""
        raise NotImplementedError
