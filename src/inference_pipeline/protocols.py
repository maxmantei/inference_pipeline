"""Structural contracts for pipeline producers, consumers, and adapters."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from inference_pipeline.buffer import BufferQ

if TYPE_CHECKING:
    from inference_pipeline.runtime import ThreadTask


class BaseConfigI(Protocol):
    """Structural interface for base stage configuration objects."""

    @property
    def name(self) -> str | None:
        """Return the optional name for this stage."""
        ...


class BaseStageConfigI(BaseConfigI, Protocol):
    """Structural interface for base stage configuration objects."""

    @property
    def out_maxsize(self) -> int:
        """Return the output queue capacity for this stage."""
        ...

    @property
    def out_timeout(self) -> float:
        """Return the default timeout for this stage's output queue."""
        ...


class BaseProducerConfigI(BaseStageConfigI, Protocol):
    """Structural interface for producer configuration objects."""

    @property
    def out_maxsize(self) -> int:
        """Return the output queue capacity for this producer."""
        ...

    @property
    def out_timeout(self) -> float:
        """Return the default timeout for this producer's output queue."""
        ...


class BaseProcessorConfigI(BaseStageConfigI, Protocol):
    """Structural interface for processor configuration objects."""

    @property
    def out_maxsize(self) -> int:
        """Return the output queue capacity for this processor."""
        ...

    @property
    def out_timeout(self) -> float:
        """Return the default timeout for this processor's output queue."""
        ...


class BaseConsumerConfigI(BaseConfigI, Protocol):
    """Structural interface for consumer configuration objects."""

    pass


class BaseBroadcastStageConfigI[T](BaseStageConfigI, Protocol):
    """Structural interface for broadcast stage configuration objects."""

    @property
    def n_outputs(self) -> int:
        """Return the number of output queues to expose for this stage."""
        ...

    @property
    def copy_fn(self) -> Callable[[T], T] | None:
        """Return the optional per-branch payload copy function for this stage."""
        ...


class BaseSplitStageConfigI(BaseConfigI, Protocol):
    @property
    def out_a_maxsize(self) -> int:
        """Return the capacity for output queue A."""
        ...

    @property
    def out_b_maxsize(self) -> int:
        """Return the capacity for output queue B."""
        ...

    @property
    def out_a_timeout(self) -> float:
        """Return the timeout for output queue A."""
        ...

    @property
    def out_b_timeout(self) -> float:
        """Return the timeout for output queue B."""
        ...


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


class ProcessorLike[InT, OutT](Protocol):
    def run(self, input_q: BufferQ[InT], stop: threading.Event | None = None) -> None:
        """Run in blocking mode until input closes or an optional stop signal."""
        ...

    def threaded(
        self,
        input_q: BufferQ[InT],
        *,
        stop: threading.Event | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this processor in a worker thread."""
        ...

    def process(self, item: InT) -> OutT | None:
        """Process a single item and return the result or None."""
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
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this consumer in a worker thread."""
        ...

    def consume(self, item: T) -> None:
        """Consume a single item from the input queue."""
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


class SplitLike[InT, OutAT, OutBT](Protocol):
    """Structural interface for split stages that route one input to two outputs."""

    @property
    def output_a(self) -> BufferQ[OutAT]:
        """Return output queue A."""
        ...

    @property
    def output_b(self) -> BufferQ[OutBT]:
        """Return output queue B."""
        ...

    def run(self, input_q: BufferQ[InT], stop: threading.Event | None = None) -> None:
        """Run in blocking mode until input closes or an optional stop signal."""
        ...

    def threaded(
        self,
        input_q: BufferQ[InT],
        *,
        stop: threading.Event | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this split stage in a worker thread."""
        ...

    def split(self, item: InT) -> tuple[OutAT | None, OutBT | None]:
        """Map one input item to optional outputs for branch A and B."""
        ...


class BroadcastLike[T](Protocol):
    """Structural interface for broadcast stages that route one input to N outputs."""

    @property
    def outputs(self) -> tuple[BufferQ[T], ...]:
        """Return all output queues as an immutable tuple."""
        ...

    def output(self, index: int) -> BufferQ[T]:
        """Return one output queue by index."""
        ...

    def run(self, input_q: BufferQ[T], stop: threading.Event | None = None) -> None:
        """Broadcast each incoming item to all outputs until input closes."""
        ...

    def threaded(
        self,
        input_q: BufferQ[T],
        *,
        stop: threading.Event | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this broadcast stage in a worker thread."""
        ...
