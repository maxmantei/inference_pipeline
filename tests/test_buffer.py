from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest

from inference_pipeline.buffer import BufferQ, _Sentinel  # type: ignore[import]


@pytest.fixture
def tiny_timeout() -> float:
    return 0.01


@pytest.fixture
def int_queue_factory(tiny_timeout: float) -> Callable[..., BufferQ[int]]:
    def _factory(
        maxsize: int = 8,
        *,
        default_timeout: float | None = None,
    ) -> BufferQ[int]:
        timeout = tiny_timeout if default_timeout is None else default_timeout
        return BufferQ[int](maxsize=maxsize, default_timeout=timeout)

    return _factory


def wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


@pytest.mark.parametrize("maxsize", [0, -1, -10])
def test_bufferq_init_rejects_non_positive_maxsize(maxsize: int) -> None:
    with pytest.raises(ValueError, match="maxsize must be > 0"):
        BufferQ[int](maxsize=maxsize)


def test_bufferq_fifo_without_overflow(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory(maxsize=3)
    for item in [1, 2, 3]:
        q.put(item)

    q.close()
    assert list(q.iter_until_closed()) == [1, 2, 3]


@pytest.mark.parametrize(
    ("maxsize", "values", "expected"),
    [
        (1, [1, 2, 3], [3]),
        (3, [1, 2, 3, 4, 5], [3, 4, 5]),
        (4, [1, 2, 3], [1, 2, 3]),
    ],
)
def test_bufferq_overflow_drops_oldest_keeps_newest_items(
    maxsize: int,
    values: list[int],
    expected: list[int],
    tiny_timeout: float,
) -> None:
    q = BufferQ[int](maxsize=maxsize, default_timeout=tiny_timeout)
    for item in values:
        q.put(item)

    q.close()
    assert list(q.iter_until_closed()) == expected


def test_bufferq_put_after_close_is_noop(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory(maxsize=4)
    q.put(1)
    q.close()
    q.put(2)

    assert list(q.iter_until_closed()) == [1]


def test_bufferq_iter_returns_when_closed_and_empty(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory(maxsize=2)
    q.close()

    assert list(q.iter_until_closed()) == []


def test_bufferq_iter_drains_buffered_payloads_after_close(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory(maxsize=4)
    for item in [10, 20, 30]:
        q.put(item)

    q.close()
    assert list(q) == [10, 20, 30]


def test_bufferq_close_is_idempotent(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory(maxsize=2)
    q.put(1)

    q.close()
    q.close()

    assert q.closed is True
    assert list(q.iter_until_closed()) == [1]


def test_bufferq_multiple_consumers_all_exit_after_close(tiny_timeout: float) -> None:
    q = BufferQ[int](maxsize=8, default_timeout=tiny_timeout)
    n_consumers = 3
    consumed: list[list[int]] = [[] for _ in range(n_consumers)]
    threads = [
        threading.Thread(
            target=lambda bucket=bucket: bucket.extend(  # type: ignore[misc]
                q.iter_until_closed(timeout=tiny_timeout)
            ),
            daemon=True,
        )
        for bucket in consumed
    ]

    for thread in threads:
        thread.start()

    try:
        assert wait_until(lambda: q._n_consumers == n_consumers)  # type: ignore[reportPrivateUsage]
        q.close()
        for thread in threads:
            thread.join(timeout=1.0)
            assert not thread.is_alive()
    finally:
        q.close()
        for thread in threads:
            thread.join(timeout=1.0)

    assert consumed == [[], [], []]


def test_bufferq_stop_event_terminates_iteration_without_close(
    tiny_timeout: float,
) -> None:
    q = BufferQ[int](maxsize=4, default_timeout=tiny_timeout)
    stop = threading.Event()
    done = threading.Event()
    collected: list[int] = []

    def _consume() -> None:
        collected.extend(q.iter_until_closed(stop=stop))
        done.set()

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: q._n_consumers == 1)  # type: ignore[reportPrivateUsage]
        stop.set()
        assert done.wait(timeout=1.0)
    finally:
        q.close()
        thread.join(timeout=1.0)

    assert collected == []
    assert q.closed is True


def test_bufferq_iter_uses_explicit_timeout_when_provided(tiny_timeout: float) -> None:
    q = BufferQ[int](maxsize=2, default_timeout=tiny_timeout)
    q.close()

    with patch.object(q._queue, "get", side_effect=queue.Empty) as mocked_get:  # type: ignore[reportPrivateUsage]
        assert list(q.iter_until_closed(timeout=0.123)) == []

    mocked_get.assert_called_once_with(timeout=0.123)


def test_bufferq_iter_uses_default_timeout_when_none(tiny_timeout: float) -> None:
    q = BufferQ[int](maxsize=2, default_timeout=tiny_timeout)
    q.close()

    with patch.object(q._queue, "get", side_effect=queue.Empty) as mocked_get:  # type: ignore[reportPrivateUsage]
        assert list(q.iter_until_closed()) == []

    mocked_get.assert_called_once_with(timeout=tiny_timeout)


def test_bufferq_context_manager_closes_queue(tiny_timeout: float) -> None:
    with BufferQ[int](maxsize=4, default_timeout=tiny_timeout) as q:
        q.put(1)

    q.put(2)
    assert q.closed is True
    assert list(q) == [1]


def test_bufferq_negative_default_timeout_raises_when_close_needs_sentinel() -> None:
    q = BufferQ[int](maxsize=1, default_timeout=-0.01)
    done = threading.Event()

    def _consume() -> None:
        try:
            for _ in q.iter_until_closed(timeout=0.01):
                pass
        finally:
            done.set()

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: q._n_consumers == 1)  # type: ignore[reportPrivateUsage]
        with pytest.raises(ValueError, match="timeout"):
            q.close()
        assert done.wait(timeout=1.0)
    finally:
        q.close()
        thread.join(timeout=1.0)

    assert q.closed is True


def test_bufferq_negative_iter_timeout_propagates_value_error(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory()
    iterator = q.iter_until_closed(timeout=-0.01)

    with pytest.raises(ValueError, match="timeout"):
        next(iterator)


def test_bufferq_high_contention_put_and_consume_shuts_down_cleanly(
    tiny_timeout: float,
) -> None:
    q = BufferQ[int](maxsize=16, default_timeout=tiny_timeout)
    consumed: list[int] = []

    def _producer() -> None:
        for item in range(500):
            q.put(item)
        q.close()

    def _consumer() -> None:
        consumed.extend(q.iter_until_closed(timeout=tiny_timeout))

    producer_thread = threading.Thread(target=_producer, daemon=True)
    consumer_thread = threading.Thread(target=_consumer, daemon=True)

    consumer_thread.start()
    producer_thread.start()

    producer_thread.join(timeout=2.0)
    consumer_thread.join(timeout=2.0)

    assert not producer_thread.is_alive()
    assert not consumer_thread.is_alive()
    assert consumed
    assert consumed == sorted(consumed)
    assert consumed[-1] == 499


def test_bufferq_close_retries_when_sentinel_enqueue_sees_full(
    tiny_timeout: float,
) -> None:
    q = BufferQ[int](maxsize=1, default_timeout=tiny_timeout)
    done = threading.Event()

    def _consume_and_set() -> None:
        list(q.iter_until_closed(timeout=tiny_timeout))
        done.set()

    consume_thread = threading.Thread(
        target=_consume_and_set,
        daemon=True,
    )
    consume_thread.start()
    assert wait_until(lambda: q._n_consumers == 1)  # type: ignore[reportPrivateUsage]

    original_put = q._queue.put  # type: ignore[reportPrivateUsage]
    calls = 0

    def _flaky_put(item: int | _Sentinel, timeout: float | None = None) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise queue.Full
        original_put(item, timeout=timeout)

    with patch.object(q._queue, "put", side_effect=_flaky_put):  # type: ignore[reportPrivateUsage]
        q.close()

    assert calls >= 2
    assert done.wait(timeout=1.0)
    consume_thread.join(timeout=1.0)
    assert not consume_thread.is_alive()


def test_bufferq_put_handles_full_then_empty_race_path(
    int_queue_factory: Callable[..., BufferQ[int]],
) -> None:
    q = int_queue_factory(maxsize=1)

    with (
        patch.object(
            q._queue,  # type: ignore[reportPrivateUsage]
            "put_nowait",
            side_effect=[queue.Full, None],
        ) as mocked_put,
        patch.object(q._queue, "get_nowait", side_effect=queue.Empty) as mocked_get,  # type: ignore[reportPrivateUsage]
    ):
        q.put(123)

    assert mocked_put.call_count == 2
    mocked_get.assert_called_once_with()
