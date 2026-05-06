from __future__ import annotations

import threading

from inference_pipeline.buffer import BufferQ
from inference_pipeline.protocols import (
    ConsumerLike,
    ProducerLike,
    SinkLike,
    SourceLike,
)
from inference_pipeline.runtime import ThreadTask


class _ProducerImpl:
    def __init__(self) -> None:
        self._output: BufferQ[int] = BufferQ(maxsize=4, default_timeout=0.01)

    @property
    def output(self) -> BufferQ[int]:
        return self._output

    def run(self, stop: threading.Event | None = None) -> None:
        if stop is None or not stop.is_set():
            self._output.put(1)
        self._output.close()

    def threaded(
        self,
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        return ThreadTask.from_runner(
            lambda: self.run(stop=stop),
            thread_name=name or "producer-like-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )


class _ConsumerImpl:
    def __init__(self) -> None:
        self.items: list[int] = []

    def run(self, input_q: BufferQ[int], stop: threading.Event | None = None) -> None:
        for item in input_q.iter_until_closed(stop=stop):
            self.items.append(item)

    def threaded(
        self,
        input_q: BufferQ[int],
        *,
        stop: threading.Event | None = None,
        name: str | None = None,
        daemon: bool = True,
        join_timeout: float = 5.0,
    ) -> ThreadTask:
        return ThreadTask.from_runner(
            lambda: self.run(input_q, stop=stop),
            thread_name=name or "consumer-like-worker",
            daemon=daemon,
            join_timeout=join_timeout,
        )


class _SourceImpl(_ProducerImpl):
    def __init__(self) -> None:
        super().__init__()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.output.close()


class _SinkImpl(_ConsumerImpl):
    def __init__(self) -> None:
        super().__init__()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False


def test_producer_like_runtime_behavior() -> None:
    producer: ProducerLike[int] = _ProducerImpl()

    producer.run()

    assert list(producer.output.iter_until_closed()) == [1]


def test_consumer_like_runtime_behavior() -> None:
    consumer: ConsumerLike[int] = _ConsumerImpl()
    q: BufferQ[int] = BufferQ(maxsize=4, default_timeout=0.01)
    q.put(10)
    q.put(20)
    q.close()

    consumer.run(q)

    assert consumer.items == [10, 20]


def test_source_like_lifecycle_behavior() -> None:
    source: SourceLike[int] = _SourceImpl()

    assert source.running is False
    source.start()
    assert source.running is True
    source.stop()
    assert source.running is False


def test_sink_like_lifecycle_behavior() -> None:
    sink: SinkLike[int] = _SinkImpl()

    assert sink.running is False
    sink.start()
    assert sink.running is True
    sink.stop()
    assert sink.running is False
