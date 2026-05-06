from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from inference_pipeline.adapters.sources import ManagedSourceAdapter


class _CountingSource(ManagedSourceAdapter[int]):
    def __init__(self) -> None:
        super().__init__(out_maxsize=8, out_timeout=0.01)
        self.start_calls = 0
        self.stop_calls = 0

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1


class _FailingStopSource(ManagedSourceAdapter[int]):
    def __init__(self) -> None:
        super().__init__(out_maxsize=8, out_timeout=0.01)
        self.start_calls = 0
        self.stop_calls = 0

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("stop failed")


class _FailingStartSource(ManagedSourceAdapter[int]):
    def __init__(self) -> None:
        super().__init__(out_maxsize=8, out_timeout=0.01)
        self.start_calls = 0
        self.stop_calls = 0

    def _start_impl(self) -> None:
        self.start_calls += 1
        raise RuntimeError("start failed")

    def _stop_impl(self) -> None:
        self.stop_calls += 1


class _BlockingStartSource(ManagedSourceAdapter[int]):
    def __init__(self) -> None:
        super().__init__(out_maxsize=8, out_timeout=0.01)
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


def wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def test_managed_source_initial_state() -> None:
    source = _CountingSource()

    assert source.running is False
    assert source.output.closed is False


def test_managed_source_start_stop_are_idempotent() -> None:
    source = _CountingSource()

    source.start()
    source.start()
    assert source.running is True
    assert source.start_calls == 1

    source.stop()
    source.stop()
    assert source.running is False
    assert source.stop_calls == 1


def test_managed_source_concurrent_start_calls_start_once() -> None:
    source = _BlockingStartSource()

    first = threading.Thread(target=source.start, daemon=True)
    second = threading.Thread(target=source.start, daemon=True)

    first.start()
    assert source.entered_start.wait(timeout=1.0)
    second.start()
    source.release_start.set()

    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert source.start_calls == 1
    assert source.running is True

    source.stop()
    assert source.stop_calls == 1


def test_managed_source_stop_during_startup_is_not_lost() -> None:
    source = _BlockingStartSource()

    start_thread = threading.Thread(target=source.start, daemon=True)
    stop_thread = threading.Thread(target=source.stop, daemon=True)

    start_thread.start()
    assert source.entered_start.wait(timeout=1.0)
    stop_thread.start()
    source.release_start.set()

    start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert not start_thread.is_alive()
    assert not stop_thread.is_alive()
    assert source.start_calls == 1
    assert source.stop_calls == 1
    assert source.running is False
    assert source.output.closed is True


def test_managed_source_stop_before_start_is_noop() -> None:
    source = _CountingSource()

    source.stop()

    assert source.running is False
    assert source.start_calls == 0
    assert source.stop_calls == 0
    assert source.output.closed is False


def test_managed_source_start_failure_rolls_back_running_state() -> None:
    source = _FailingStartSource()

    with pytest.raises(RuntimeError, match="start failed"):
        source.start()

    assert source.running is False
    assert source.output.closed is False
    assert source.stop_calls == 0


def test_managed_source_stop_closes_output() -> None:
    source = _CountingSource()
    source.start()

    source.stop()

    assert source.output.closed is True


def test_managed_source_cannot_restart_after_stop() -> None:
    source = _CountingSource()
    source.start()
    source.stop()

    with pytest.raises(RuntimeError, match="Cannot restart"):
        source.start()


def test_managed_source_stop_closes_output_even_if_stop_impl_raises() -> None:
    source = _FailingStopSource()
    source.start()

    with pytest.raises(RuntimeError, match="stop failed"):
        source.stop()

    assert source.running is False
    assert source.output.closed is True
    assert source.stop_calls == 1


def test_source_context_manager_preserves_body_exception_if_stop_fails() -> None:
    source = _FailingStopSource()

    with pytest.raises(ValueError, match="body failed"):
        with source:
            raise ValueError("body failed")

    assert source.stop_calls == 1


def test_managed_source_run_exits_on_external_stop_event() -> None:
    source = _CountingSource()
    stop = threading.Event()

    thread = threading.Thread(target=lambda: source.run(stop=stop), daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: source.running)
        stop.set()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        source.stop()
        thread.join(timeout=1.0)

    assert source.start_calls == 1
    assert source.stop_calls == 1
    assert source.running is False
    assert source.output.closed is True


def test_managed_source_run_exits_when_stopped_internally() -> None:
    source = _CountingSource()

    thread = threading.Thread(target=source.run, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: source.running)
        source.stop()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        source.stop()
        thread.join(timeout=1.0)

    assert source.start_calls == 1
    assert source.stop_calls == 1
    assert source.running is False
    assert source.output.closed is True


def test_managed_source_threaded_runs_with_external_stop() -> None:
    source = _CountingSource()
    stop = threading.Event()
    task = source.threaded(
        stop=stop,
        name="source-adapter-worker",
        daemon=False,
        join_timeout=1.0,
    )

    task.start()
    stop.set()
    task.join()

    assert source.start_calls == 1
    assert source.stop_calls == 1
    assert source.running is False
    assert source.output.closed is True
