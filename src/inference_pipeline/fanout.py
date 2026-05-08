"""Fan-out stage primitives for branching queue-based pipelines.

These stages provide best-effort branch independence: each branch enqueues
independently and one slow branch does not block delivery to the others.
"""

import threading
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from inference_pipeline.buffer import BufferQ
from inference_pipeline.protocols import (
    BaseBroadcastStageConfigI,
    BaseSplitStageConfigI,
)
from inference_pipeline.runtime import ThreadTask


class BroadcastStage[T](ABC):
    """Broadcast one input stream to ``N`` homogeneous output queues."""

    def __init__(self, *, config: BaseBroadcastStageConfigI[T]) -> None:
        """Create a broadcast stage.

        Args:
            config: Broadcast stage configuration object.

        Raises:
            ValueError: If ``n_outputs`` is less than 1.
        """
        self.config = config
        if config.n_outputs < 1:
            raise ValueError("n_outputs must be >= 1")
        self._outputs: tuple[BufferQ[T], ...] = tuple(
            BufferQ[T](maxsize=config.out_maxsize, default_timeout=config.out_timeout)
            for _ in range(config.n_outputs)
        )

    @property
    def outputs(self) -> tuple[BufferQ[T], ...]:
        """Return all output queues as an immutable tuple."""
        return self._outputs

    def output(self, index: int) -> BufferQ[T]:
        """Return one output queue by index."""
        return self._outputs[index]

    def run(self, input_q: BufferQ[T], stop: threading.Event | None = None) -> None:
        """Broadcast each incoming item to all outputs until input closes."""
        with self:
            for item in input_q.iter_until_closed(stop=stop):
                for out_q in self._outputs:
                    out_q.put(
                        item
                        if self.config.copy_fn is None
                        else self.config.copy_fn(item)
                    )

    def threaded(
        self,
        input_q: BufferQ[T],
        *,
        stop: threading.Event | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this broadcast stage in a thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=f"{self.config.name or type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    def __enter__(self) -> Self:
        """Return ``self`` for context-managed stage usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close every output queue on context exit."""
        for out_q in self._outputs:
            out_q.close()


class SplitStage[InT, OutAT, OutBT, SConfT: BaseSplitStageConfigI](ABC):
    """Split one input stream into two typed output streams."""

    def __init__(self, *, config: SConfT) -> None:
        """Create a two-branch split stage with independent output queues.

        Args:
            config: Split stage configuration object.
        """
        self.config = config
        self._output_a = BufferQ[OutAT](
            maxsize=config.out_a_maxsize, default_timeout=config.out_a_timeout
        )
        self._output_b = BufferQ[OutBT](
            maxsize=config.out_b_maxsize, default_timeout=config.out_b_timeout
        )

    @property
    def output_a(self) -> BufferQ[OutAT]:
        """Return output queue A."""
        return self._output_a

    @property
    def output_b(self) -> BufferQ[OutBT]:
        """Return output queue B."""
        return self._output_b

    def run(self, input_q: BufferQ[InT], stop: threading.Event | None = None) -> None:
        """Route each input item to zero, one, or both outputs."""
        with self:
            for item in input_q.iter_until_closed(stop=stop):
                out_a, out_b = self.split(item)
                if out_a is not None:
                    self._output_a.put(out_a)
                if out_b is not None:
                    self._output_b.put(out_b)

    def threaded(
        self,
        input_q: BufferQ[InT],
        *,
        stop: threading.Event | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this split stage in a thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=f"{self.config.name or type(self).__name__}-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )

    @abstractmethod
    def split(self, item: InT) -> tuple[OutAT | None, OutBT | None]:
        """Map one input item to optional outputs for branch A and B."""
        raise NotImplementedError

    def __enter__(self) -> Self:
        """Return ``self`` for context-managed stage usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close both output queues on context exit."""
        self._output_a.close()
        self._output_b.close()
