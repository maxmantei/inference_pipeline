"""Structural contracts for pipeline producers, consumers, and adapters."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol

from inference_pipeline.buffer import BufferQ

if TYPE_CHECKING:
    from inference_pipeline.runtime import ThreadTask


class ProducerLike[T](Protocol):
    """Structural interface for producer-style pipeline components."""

    @property
    def output(self) -> BufferQ[T]:
        """Return the output queue consumed by downstream stages."""
        ...

    def run(self, stop: threading.Event | None = None) -> None:
        """Run in blocking mode until completion or optional stop signal."""
        ...

    def threaded(
        self,
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this producer in a worker thread."""
        ...


class SourceLike[T](ProducerLike[T], Protocol):
    """Structural interface for lifecycle-managed sources with producer behavior."""

    @property
    def running(self) -> bool:
        """Return whether the source is currently running."""
        ...

    def start(self) -> None:
        """Start source acquisition/production."""
        ...

    def stop(self) -> None:
        """Stop source acquisition/production and close output delivery."""
        ...


class ConsumerLike[T](Protocol):
    """Structural interface for consumer-style pipeline components."""

    def run(self, input_q: BufferQ[T], stop: threading.Event | None = None) -> None:
        """Run in blocking mode until input closes or an optional stop signal."""
        ...

    def threaded(
        self,
        input_q: BufferQ[T],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this consumer in a worker thread."""
        ...


class SinkLike[T](ConsumerLike[T], Protocol):
    """Structural interface for lifecycle-managed sinks with consumer behavior."""

    @property
    def running(self) -> bool:
        """Return whether the sink is currently running."""
        ...

    def start(self) -> None:
        """Start sink lifecycle resources."""
        ...

    def stop(self) -> None:
        """Stop sink lifecycle resources."""
        ...
