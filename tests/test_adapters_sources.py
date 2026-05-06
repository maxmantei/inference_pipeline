from __future__ import annotations

import threading

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


def test_managed_source_stop_before_start_is_noop() -> None:
    source = _CountingSource()

    source.stop()

    assert source.running is False
    assert source.start_calls == 0
    assert source.stop_calls == 0
    assert source.output.closed is False


def test_managed_source_stop_closes_output() -> None:
    source = _CountingSource()
    source.start()

    source.stop()

    assert source.output.closed is True


def test_managed_source_stop_closes_output_even_if_stop_impl_raises() -> None:
    source = _FailingStopSource()
    source.start()

    with pytest.raises(RuntimeError, match="stop failed"):
        source.stop()

    assert source.running is False
    assert source.output.closed is True
    assert source.stop_calls == 1


def test_managed_source_run_exits_on_external_stop_event() -> None:
    source = _CountingSource()
    stop = threading.Event()

    thread = threading.Thread(target=lambda: source.run(stop=stop), daemon=True)
    thread.start()
    try:
        assert source.running is True
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
        assert source.running is True
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
