"""Fan-out stage primitives for branching queue-based pipelines.

These stages provide best-effort branch independence: each branch enqueues
independently and one slow branch does not block delivery to the others.
"""

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Self, TypeVar

from inference_pipeline.buffer import BufferQ
from inference_pipeline.runtime import ThreadTask

T = TypeVar("T")
InT = TypeVar("InT")
OutAT = TypeVar("OutAT")
OutBT = TypeVar("OutBT")


class BroadcastStage[T]:
    """Broadcast one input stream to ``N`` homogeneous output queues."""

    def __init__(
        self,
        n_outputs: int,
        *,
        out_maxsize: int,
        out_timeout: float = 0.5,
        copy_fn: Callable[[T], T] | None = None,
    ) -> None:
        """Create a broadcast stage.

        Args:
            n_outputs: Number of output queues to expose.
            out_maxsize: Capacity for each output queue.
            out_timeout: Default timeout for each output queue iterator.
            copy_fn: Optional per-branch payload copy function.

        Raises:
            ValueError: If ``n_outputs`` is less than 1.
        """
        if n_outputs < 1:
            raise ValueError("n_outputs must be >= 1")
        self._outputs: tuple[BufferQ[T], ...] = tuple(
            BufferQ[T](maxsize=out_maxsize, default_timeout=out_timeout)
            for _ in range(n_outputs)
        )
        self._copy_fn = copy_fn

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
                    out_q.put(item if self._copy_fn is None else self._copy_fn(item))

    def threaded(
        self,
        input_q: BufferQ[T],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this broadcast stage in a thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=name or f"{type(self).__name__}-worker",
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


class SplitStage[InT, OutAT, OutBT](ABC):
    """Split one input stream into two typed output streams."""

    def __init__(
        self,
        *,
        out_a_maxsize: int,
        out_b_maxsize: int,
        out_a_timeout: float = 0.5,
        out_b_timeout: float = 0.5,
    ) -> None:
        """Create a two-branch split stage with independent output queues."""
        self._output_a = BufferQ[OutAT](
            maxsize=out_a_maxsize, default_timeout=out_a_timeout
        )
        self._output_b = BufferQ[OutBT](
            maxsize=out_b_maxsize, default_timeout=out_b_timeout
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
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        """Return a ``ThreadTask`` that runs this split stage in a thread."""
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=name or f"{type(self).__name__}-worker",
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
