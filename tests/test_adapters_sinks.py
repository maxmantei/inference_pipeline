from __future__ import annotations

import threading

import pytest

from inference_pipeline.adapters.sinks import ManagedSinkAdapter
from inference_pipeline.buffer import BufferQ


class _CollectingSink(ManagedSinkAdapter[int]):
    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0
        self.items: list[int] = []

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1

    def consume(self, item: int) -> None:
        self.items.append(item)


class _FailingSink(ManagedSinkAdapter[int]):
    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.fail_on = fail_on
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


def _queue_with(values: list[int]) -> BufferQ[int]:
    q: BufferQ[int] = BufferQ(maxsize=max(len(values), 1), default_timeout=0.01)
    for value in values:
        q.put(value)
    q.close()
    return q


def test_managed_sink_initial_state() -> None:
    sink = _CollectingSink()

    assert sink.running is False


def test_managed_sink_start_stop_are_idempotent() -> None:
    sink = _CollectingSink()

    sink.start()
    sink.start()
    assert sink.running is True
    assert sink.start_calls == 1

    sink.stop()
    sink.stop()
    assert sink.running is False
    assert sink.stop_calls == 1


def test_managed_sink_run_consumes_until_input_closes() -> None:
    sink = _CollectingSink()
    q = _queue_with([1, 2, 3])

    sink.run(q)

    assert sink.items == [1, 2, 3]
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_run_respects_external_stop_event_when_set() -> None:
    sink = _CollectingSink()
    q = _queue_with([10, 20, 30])
    stop = threading.Event()
    stop.set()

    sink.run(q, stop=stop)

    assert sink.items == []
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_run_uses_internal_shutdown_when_no_stop_provided() -> None:
    sink = _CollectingSink()
    q: BufferQ[int] = BufferQ(maxsize=4, default_timeout=0.01)

    thread = threading.Thread(target=lambda: sink.run(q), daemon=True)
    thread.start()
    try:
        assert sink.running is True
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
    sink = _FailingSink(fail_on=2)
    q = _queue_with([1, 2, 3])

    with pytest.raises(RuntimeError, match="consume failed"):
        sink.run(q)

    assert sink.items == [1]
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False


def test_managed_sink_threaded_runs_to_completion() -> None:
    sink = _CollectingSink()
    q = _queue_with([4, 5, 6])
    task = sink.threaded(
        q,
        name="sink-adapter-worker",
        daemon=False,
        join_timeout=1.0,
    )

    task.start()
    task.join()

    assert sink.items == [4, 5, 6]
    assert sink.start_calls == 1
    assert sink.stop_calls == 1
    assert sink.running is False
