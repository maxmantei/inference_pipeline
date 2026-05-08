"""Abstract staged-processing primitives built on BufferQ queues."""

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Self

from inference_pipeline.buffer import BufferQ
from inference_pipeline.protocols import (
    BaseConsumerConfigI,
    BaseProcessorConfigI,
    BaseProducerConfigI,
    BaseStageConfigI,
)
from inference_pipeline.runtime import ThreadTask


class Stage[InT, OutT, ConfT: BaseStageConfigI](ABC):
    """Base context-managed stage with an output queue."""

    def __init__(self, *, config: ConfT) -> None:
        """Create a stage.

        Args:
            config: Configuration object for this stage.
        """
        self.config = config
        self._out = BufferQ[OutT](
            maxsize=config.out_maxsize, default_timeout=config.out_timeout
        )

    @property
    def output(self) -> BufferQ[OutT]:
        """Return the stage output queue."""
        return self._out

    def __enter__(self) -> Self:
        """Return ``self`` for context-managed stage usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the stage output queue."""
        self._out.close()


class ProducerStage[OutT, ConfT: BaseProducerConfigI](Stage[None, OutT, ConfT], ABC):
    """Base class for producer stages that emit items without an input queue."""

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
            thread_name=name or f"{self.config.name or type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def produce(self, stop: threading.Event | None = None) -> Iterator[OutT]:
        """Yield produced items until completion or stop is requested."""
        raise NotImplementedError


class ProcessorStage[InT, OutT, ConfT: BaseProcessorConfigI](
    Stage[InT, OutT, ConfT], ABC
):
    """Base class for stages transforming items from input queue to output queue."""

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
            thread_name=name or f"{self.config.name or type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def process(self, item: InT) -> OutT | None:
        """Transform one input item into output, or return ``None`` to skip."""
        raise NotImplementedError


class ConsumerStage[InT, ConfT: BaseConsumerConfigI](ABC):
    """Base class for terminal stages that consume items without producing output."""

    def __init__(self, *, config: ConfT) -> None:
        """Create a consumer stage.

        Args:
            config: Configuration object for this stage.
        """
        self.config = config

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
            thread_name=name or f"{self.config.name or type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def consume(self, item: InT) -> None:
        """Handle a single input item."""
        raise NotImplementedError
