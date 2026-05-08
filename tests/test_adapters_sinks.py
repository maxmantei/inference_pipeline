from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from inference_pipeline.adapters.sinks import ManagedSinkAdapter
from inference_pipeline.buffer import BufferQ
from inference_pipeline.configs import BaseConsumerConfig


class _CollectingSink(ManagedSinkAdapter[int, BaseConsumerConfig]):
    def __init__(self, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.start_calls = 0
        self.stop_calls = 0
        self.items: list[int] = []

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1

    def consume(self, item: int) -> None:
        self.items.append(item)


@dataclass(frozen=True, kw_only=True, slots=True)
class _FailingSinkConfig(BaseConsumerConfig):
    fail_on: int


class _FailingSink(ManagedSinkAdapter[int, _FailingSinkConfig]):
    def __init__(self, config: _FailingSinkConfig) -> None:
        super().__init__(config=config)
        self.fail_on = config.fail_on
        self.start_calls = 0
        self.stop_calls = 0
        self.items: list[int] = []

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1

    def consume(self, item: int) -> None:
        if item == self.fail_on:
            raise RuntimeError("consume failed")
        self.items.append(item)


class _FailingStopSink(ManagedSinkAdapter[int, BaseConsumerConfig]):
    def __init__(self, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.start_calls = 0
        self.stop_calls = 0

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("stop failed")

    def consume(self, item: int) -> None:
        return


class _FailingStartSink(ManagedSinkAdapter[int, BaseConsumerConfig]):
    def __init__(self, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.start_calls = 0

    def _start_impl(self) -> None:
        self.start_calls += 1
        raise RuntimeError("start failed")

    def consume(self, item: int) -> None:
        return


class _BlockingStartSink(ManagedSinkAdapter[int, BaseConsumerConfig]):
    def __init__(self, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.start_calls = 0
        self.stop_calls = 0
        self.entered_start = threading.Event()
        self.release_start = threading.Event()

    def _start_impl(self) -> None:
        self.start_calls += 1
        self.entered_start.set()
        self.release_start.wait(timeout=1.0)

    def _stop_impl(self) -> None:
        self.stop_calls += 1

    def consume(self, item: int) -> None:
        return


def wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def _queue_with(values: list[int]) -> BufferQ[int]:
    q: BufferQ[int] = BufferQ(maxsize=max(len(values), 1), default_timeout=0.01)
    for value in values:
        q.put(value)
    q.close()
    return q


def test_managed_sink_initial_state() -> None:
    sink = _CollectingSink(BaseConsumerConfig())

    assert sink.running is False


def test_managed_sink_start_stop_are_idempotent() -> None:
    sink = _CollectingSink(BaseConsumerConfig())

    sink.start()
    sink.start()
    assert sink.running is True
    assert sink.start_calls == 1

    sink.stop()
    sink.stop()
    assert sink.running is False
    assert sink.stop_calls == 1


def test_managed_sink_concurrent_start_calls_start_once() -> None:
    sink = _BlockingStartSink(BaseConsumerConfig())

    first = threading.Thread(target=sink.start, daemon=True)
    second = threading.Thread(target=sink.start, daemon=True)

    first.start()
    assert sink.entered_start.wait(timeout=1.0)
    second.start()
    sink.release_start.set()

    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sink.start_calls == 1
    assert sink.running is True

    sink.stop()
    assert sink.stop_calls == 1


def test_managed_sink_stop_during_startup_is_not_lost() -> None:
    sink = _BlockingStartSink(BaseConsumerConfig())

    start_thread = threading.Thread(target=sink.start, daemon=True)
    stop_thread = threading.Thread(target=sink.stop, daemon=True)

    start_thread.start()
    assert sink.entered_start.wait(timeout=1.0)
    stop_thread.start()
    sink.release_start.set()

    start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert not start_thread.is_alive()
    assert not stop_thread.is_alive()
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_start_failure_rolls_back_running_state() -> None:
    sink = _FailingStartSink(BaseConsumerConfig())

    with pytest.raises(RuntimeError, match="start failed"):
        sink.start()

    assert sink.running is False


def test_managed_sink_run_consumes_until_input_closes() -> None:
    sink = _CollectingSink(BaseConsumerConfig())
    q = _queue_with([1, 2, 3])

    sink.run(q)

    assert sink.items == [1, 2, 3]
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_run_respects_external_stop_event_when_set() -> None:
    sink = _CollectingSink(BaseConsumerConfig())
    q = _queue_with([10, 20, 30])
    stop = threading.Event()
    stop.set()

    sink.run(q, stop=stop)

    assert sink.items == []
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_run_uses_internal_shutdown_when_no_stop_provided() -> None:
    sink = _CollectingSink(BaseConsumerConfig())
    q: BufferQ[int] = BufferQ(maxsize=4, default_timeout=0.01)

    thread = threading.Thread(target=lambda: sink.run(q), daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: sink.running)
        sink.stop()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        sink.stop()
        q.close()
        thread.join(timeout=1.0)

    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_run_stops_on_internal_shutdown_with_external_stop() -> None:
    sink = _CollectingSink(BaseConsumerConfig())
    external_stop = threading.Event()
    q: BufferQ[int] = BufferQ(maxsize=4, default_timeout=0.01)

    thread = threading.Thread(
        target=lambda: sink.run(q, stop=external_stop),
        daemon=True,
    )
    thread.start()
    try:
        assert wait_until(lambda: sink.running)
        sink.stop()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        sink.stop()
        q.close()
        thread.join(timeout=1.0)

    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_run_propagates_consume_error_and_stops() -> None:
    sink = _FailingSink(_FailingSinkConfig(fail_on=2))
    q = _queue_with([1, 2, 3])

    with pytest.raises(RuntimeError, match="consume failed"):
        sink.run(q)

    assert sink.items == [1]
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_sink_context_manager_preserves_body_exception_if_stop_fails() -> None:
    sink = _FailingStopSink(BaseConsumerConfig())

    with pytest.raises(ValueError, match="body failed"):
        with sink:
            raise ValueError("body failed")

    assert sink.stop_calls == 1


def test_managed_sink_threaded_runs_to_completion() -> None:
    sink = _CollectingSink(BaseConsumerConfig())
    q = _queue_with([4, 5, 6])
    task = sink.threaded(
        q,
        daemon=False,
        join_timeout=1.0,
    )

    task.start()
    task.join()

    assert sink.items == [4, 5, 6]
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False
