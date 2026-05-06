"""Abstract staged-processing primitives built on BufferQ queues."""

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Self, TypeVar

from inference_pipeline.buffer import BufferQ
from inference_pipeline.runtime import ThreadTask

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class Stage[InT, OutT](ABC):
    """Base context-managed stage with an optional output queue."""

    def __init__(
        self,
        *,
        out_maxsize: int | None = None,
        out_timeout: float = 0.5,
    ) -> None:
        """Create a stage.

        Args:
            out_maxsize: Output queue capacity, or ``None`` for sink-only stages.
            out_timeout: Default timeout for the output queue.
        """
        self._out: BufferQ[OutT] | None = None
        if out_maxsize is not None:
            self._out = BufferQ[OutT](maxsize=out_maxsize, default_timeout=out_timeout)

    @property
    def output(self) -> BufferQ[OutT]:
        """Return the stage output queue.

        Raises:
            RuntimeError: If this stage has no output queue.
        """
        if self._out is None:
            raise RuntimeError(f"{type(self).__name__} has no output queue.")
        return self._out

    @property
    def has_output(self) -> bool:
        """Return whether the stage exposes an output queue."""
        return self._out is not None

    def __enter__(self) -> Self:
        """Return ``self`` for context-managed stage usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the stage output queue, if present."""
        if self._out is not None:
            self._out.close()


class ProducerStage[OutT](Stage[None, OutT], ABC):
    """Base class for producer stages that emit items without an input queue."""

    def __init__(
        self,
        *,
        out_maxsize: int,
        out_timeout: float = 0.5,
    ) -> None:
        """Initialize a producer stage with a mandatory output queue."""
        super().__init__(
            out_maxsize=out_maxsize,
            out_timeout=out_timeout,
        )

    def run(self, stop: threading.Event | None = None) -> None:
        """Run the producer loop and forward produced items to output."""
        with self:
            for item in self.produce(stop=stop):
                self.output.put(item)

    def threaded(
        self,
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this producer in a worker thread."""
        return ThreadTask.from_runner(
            lambda: self.run(stop=stop),
            thread_name=name or f"{type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def produce(self, stop: threading.Event | None = None) -> Iterator[OutT]:
        """Yield produced items until completion or stop is requested."""
        raise NotImplementedError


class ProcessorStage[InT, OutT](Stage[InT, OutT], ABC):
    """Base class for stages transforming items from input queue to output queue."""

    def __init__(
        self,
        *,
        out_maxsize: int,
        out_timeout: float = 0.5,
    ) -> None:
        """Initialize a processor stage with a mandatory output queue."""
        super().__init__(
            out_maxsize=out_maxsize,
            out_timeout=out_timeout,
        )

    def run(self, input_q: BufferQ[InT], stop: threading.Event | None = None) -> None:
        """Consume input items, process them, and emit non-None outputs."""
        with self:
            for item in input_q.iter_until_closed(stop=stop):
                out_item = self.process(item)
                if out_item is not None:
                    self.output.put(out_item)

    def threaded(
        self,
        input_q: BufferQ[InT],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this processor in a worker thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=name or f"{type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def process(self, item: InT) -> OutT | None:
        """Transform one input item into output, or return ``None`` to skip."""
        raise NotImplementedError


class ConsumerStage[InT](ABC):
    """Base class for terminal stages that consume items without producing output."""

    def run(self, input_q: BufferQ[InT], stop: threading.Event | None = None) -> None:
        """Consume items from an input queue until closure or stop event."""
        for item in input_q.iter_until_closed(stop=stop):
            self.consume(item)

    def threaded(
        self,
        input_q: BufferQ[InT],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this consumer in a worker thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=name or f"{type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def consume(self, item: InT) -> None:
        """Handle a single input item."""
        raise NotImplementedError
