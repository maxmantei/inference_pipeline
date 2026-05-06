from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from inference_pipeline.runtime import ThreadTask


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"join_timeout": -0.01, "name": "worker"}, "join_timeout must be >= 0"),
        ({"join_timeout": 0.1, "name": ""}, "name must not be empty"),
    ],
)
def test_threadtask_rejects_invalid_configuration(
    kwargs: dict[str, float | str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ThreadTask(_runner=lambda: None, **kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_threadtask_start_join_happy_path_and_state_transitions() -> None:
    ran = threading.Event()

    task = ThreadTask.from_runner(
        runner=lambda: ran.set(),
        thread_name="happy-worker",
        daemon=False,
        join_timeout=0.2,
    )

    assert task.name == "happy-worker"
    assert task.daemon is False
    assert task.join_timeout == 0.2
    assert task.is_alive is False
    assert task.error is None

    task.start()
    assert ran.wait(timeout=1.0)

    task.join()
    assert task.is_alive is False
    assert task.error is None


def test_threadtask_join_times_out_then_can_finish_cleanly() -> None:
    unblock = threading.Event()

    def _runner() -> None:
        unblock.wait(timeout=5.0)

    task = ThreadTask.from_runner(
        runner=_runner,
        thread_name="timeout-worker",
        join_timeout=0.01,
    )
    task.start()

    with pytest.raises(TimeoutError, match="did not finish"):
        task.join()

    unblock.set()
    task.join()
    assert task.error is None


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: ValueError("boom"),
        lambda: SystemExit("stop"),
    ],
)
def test_threadtask_join_wraps_worker_base_exceptions(
    exc_factory: Callable[[], BaseException],
) -> None:
    expected = exc_factory()

    def _runner() -> None:
        raise expected

    task = ThreadTask.from_runner(
        runner=_runner,
        thread_name="error-worker",
        join_timeout=0.2,
    )
    task.start()

    with pytest.raises(
        RuntimeError, match="Worker thread 'error-worker' failed"
    ) as exc_info:
        task.join()

    assert exc_info.value.__cause__ is expected
    assert task.error is expected


def test_threadtask_context_manager_raises_worker_failure_after_body() -> None:
    task = ThreadTask.from_runner(
        runner=lambda: (_ for _ in ()).throw(ValueError("worker fail")),
        thread_name="ctx-worker",
        join_timeout=0.2,
    )

    with pytest.raises(
        RuntimeError, match="Worker thread 'ctx-worker' failed"
    ) as exc_info:
        with task:
            pass

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_threadtask_context_manager_keeps_body_exception_on_timeout() -> None:
    unblock = threading.Event()

    def _runner() -> None:
        unblock.wait(timeout=5.0)

    task = ThreadTask.from_runner(
        runner=_runner,
        thread_name="ctx-precedence-worker",
        join_timeout=0.01,
    )

    with pytest.raises(KeyError, match="body error"):
        with task:
            raise KeyError("body error")

    unblock.set()
    task.join()


def test_threadtask_start_is_idempotent() -> None:
    started_count = 0
    lock = threading.Lock()
    release = threading.Event()

    def _runner() -> None:
        nonlocal started_count
        with lock:
            started_count += 1
        release.wait(timeout=1.0)

    task = ThreadTask.from_runner(
        runner=_runner,
        thread_name="idempotent-start-worker",
        join_timeout=0.5,
    )
    task.start()
    task.start()

    release.set()
    task.join()

    assert started_count == 1


def test_threadtask_join_before_start_is_noop() -> None:
    task = ThreadTask.from_runner(
        runner=lambda: None,
        thread_name="join-before-start-worker",
        join_timeout=0.1,
    )

    task.join()

    assert task.is_alive is False
    assert task.error is None
