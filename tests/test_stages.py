from __future__ import annotations

import threading
from collections.abc import Iterator

from inference_pipeline.buffer import BufferQ
from inference_pipeline.configs import (
    BaseConsumerConfig,
    BaseProcessorConfig,
    BaseProducerConfig,
)
from inference_pipeline.stages import (
    ConsumerStage,
    ProcessorStage,
    ProducerStage,
)


class _RangeProducer(ProducerStage[int, BaseProducerConfig]):
    def __init__(self, values: list[int]) -> None:
        super().__init__(config=BaseProducerConfig(out_maxsize=32, out_timeout=0.01))
        self._values = values

    def produce(self, stop: threading.Event | None = None) -> Iterator[int]:
        for value in self._values:
            if stop is not None and stop.is_set():
                return
            yield value


class _EvenDoubler(ProcessorStage[int, int, BaseProcessorConfig]):
    def __init__(self) -> None:
        super().__init__(config=BaseProcessorConfig(out_maxsize=32, out_timeout=0.01))

    def process(self, item: int) -> int | None:
        if item % 2:
            return None
        return item * 2


class _CollectingConsumer(ConsumerStage[int, BaseConsumerConfig]):
    def __init__(self) -> None:
        super().__init__(config=BaseConsumerConfig())
        self.items: list[int] = []

    def consume(self, item: int) -> None:
        self.items.append(item)


def test_producer_stage_run_emits_items_and_closes_output() -> None:
    producer = _RangeProducer([1, 2, 3, 4])

    producer.run()

    assert producer.output.closed is True
    assert list(producer.output.iter_until_closed()) == [1, 2, 3, 4]


def test_producer_stage_stop_event_stops_early() -> None:
    producer = _RangeProducer([10, 20, 30])
    stop = threading.Event()
    stop.set()

    producer.run(stop=stop)

    assert list(producer.output.iter_until_closed()) == []


def test_processor_stage_filters_none_and_closes_output() -> None:
    processor = _EvenDoubler()
    input_q: BufferQ[int] = BufferQ(maxsize=16, default_timeout=0.01)
    for value in [1, 2, 3, 4, 5, 6]:
        input_q.put(value)
    input_q.close()

    processor.run(input_q)

    assert processor.output.closed is True
    assert list(processor.output.iter_until_closed()) == [4, 8, 12]


def test_processor_stage_honors_pre_set_stop_event() -> None:
    processor = _EvenDoubler()
    stop = threading.Event()
    stop.set()
    input_q: BufferQ[int] = BufferQ(maxsize=8, default_timeout=0.01)
    for value in [2, 4, 6]:
        input_q.put(value)
    input_q.close()

    processor.run(input_q, stop=stop)

    assert list(processor.output.iter_until_closed()) == []


def test_consumer_stage_collects_until_input_closes() -> None:
    consumer = _CollectingConsumer()
    input_q: BufferQ[int] = BufferQ(maxsize=8, default_timeout=0.01)
    for value in [7, 8, 9]:
        input_q.put(value)
    input_q.close()

    consumer.run(input_q)

    assert consumer.items == [7, 8, 9]


def test_producer_stage_threaded_runs_and_closes_output() -> None:
    producer = _RangeProducer([11, 12, 13])
    task = producer.threaded(daemon=False, join_timeout=1.0)

    task.start()
    task.join()

    assert producer.output.closed is True
    assert list(producer.output.iter_until_closed()) == [11, 12, 13]


def test_processor_stage_threaded_consumes_and_emits_transforms() -> None:
    processor = _EvenDoubler()
    input_q: BufferQ[int] = BufferQ(maxsize=16, default_timeout=0.01)
    for value in [1, 2, 3, 4]:
        input_q.put(value)
    input_q.close()

    task = processor.threaded(input_q, daemon=False, join_timeout=1.0)
    task.start()
    task.join()

    assert processor.output.closed is True
    assert list(processor.output.iter_until_closed()) == [4, 8]


def test_consumer_stage_threaded_consumes_until_input_close() -> None:
    consumer = _CollectingConsumer()
    input_q: BufferQ[int] = BufferQ(maxsize=8, default_timeout=0.01)
    for value in [21, 22, 23]:
        input_q.put(value)
    input_q.close()

    task = consumer.threaded(input_q, daemon=False, join_timeout=1.0)
    task.start()
    task.join()

    assert consumer.items == [21, 22, 23]
